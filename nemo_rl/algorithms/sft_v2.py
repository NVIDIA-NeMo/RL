# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-controller SFT with one colocated Energon loader per DP replica."""

from __future__ import annotations

import os
import statistics
import time
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import ray
import torch
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.loss.loss_functions import NLLLossFn
from nemo_rl.algorithms.sft import SFTConfig
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.energon.config import EnergonSourceConfig
from nemo_rl.data.energon.sft_types import StepEnvelope
from nemo_rl.data.energon.topology import (
    DataLoaderPlacementPlan,
    resolve_topology_mapper,
)
from nemo_rl.data_plane.interfaces import LocalDataPlaneConfig
from nemo_rl.distributed.named_sharding import REPLICATED_AXES
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
    prepare_segment_topology,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.telemetry.config import TelemetryConfig
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig
from nemo_rl.utils.timer import TimeoutChecker


class MasterConfig(BaseModel, extra="allow"):
    """Standalone SFTv2 configuration."""

    policy: PolicyConfig
    data: DataConfig
    sft: SFTConfig
    data_plane: LocalDataPlaneConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig
    telemetry: Optional[TelemetryConfig] = None


@dataclass
class SFTV2SaveState:
    """Controller state committed at an optimizer-step boundary."""

    total_steps: int
    consumed_samples: int
    total_valid_tokens: int
    placement_hash: str


@dataclass
class SFTV2ActorArgs:
    """Driver-built objects used by the SFTv2 controller actor."""

    trainer: TQPolicy
    loss_fn: NLLLossFn
    train_cluster: RayVirtualCluster
    placement_plan: DataLoaderPlacementPlan
    save_state: SFTV2SaveState
    loader_states: list[dict[str, Any]] | None


def _initial_save_state(placement_hash: str) -> SFTV2SaveState:
    return SFTV2SaveState(
        total_steps=0,
        consumed_samples=0,
        total_valid_tokens=0,
        placement_hash=placement_hash,
    )


def _restore_save_state(
    loaded: Optional[dict[str, Any]], *, placement_hash: str
) -> SFTV2SaveState:
    if loaded is None:
        return _initial_save_state(placement_hash)
    saved_hash = loaded.get("placement_hash")
    if saved_hash != placement_hash:
        raise ValueError(
            "SFTv2 loader placement changed since the checkpoint was written: "
            f"saved={saved_hash!r}, current={placement_hash!r}."
        )
    defaults = vars(_initial_save_state(placement_hash))
    known = {item.name for item in fields(SFTV2SaveState)}
    defaults.update({key: value for key, value in loaded.items() if key in known})
    return SFTV2SaveState(**defaults)


def _max_train_steps(master_config: MasterConfig) -> int:
    source = EnergonSourceConfig.model_validate(master_config.data["train"])
    # The loader rejects a non-positive value too, but only once the actor builds
    # it -- by then the cluster is up and megatron_cfg.train_iters has already
    # been set from this result. Fail on the driver, before any allocation.
    if source.virtual_epoch_length <= 0:
        raise ValueError(
            "Energon training requires data.train.virtual_epoch_length in batches "
            "(optimizer steps per virtual epoch); it drives both the epoch budget "
            "and megatron_cfg.train_iters."
        )
    return min(
        master_config.sft.max_num_steps,
        master_config.sft.max_num_epochs * source.virtual_epoch_length,
    )


@ray.remote(num_cpus=1, num_gpus=0)  # pragma: no cover
class SFTSingleControllerActor:
    """Drive colocated loaders and the existing TQPolicy from one actor."""

    def __init__(self, master_config: MasterConfig, actor_args: SFTV2ActorArgs) -> None:
        self._master_config = master_config
        self._trainer = actor_args.trainer
        self._loss_fn = actor_args.loss_fn
        self._train_cluster = actor_args.train_cluster
        self._placement_plan = actor_args.placement_plan
        self._save_state = actor_args.save_state
        self._max_steps = _max_train_steps(master_config)
        self._loader_states = actor_args.loader_states
        self._logger = Logger(master_config.logger)  # type: ignore[arg-type]
        self._logger.log_hyperparams(master_config.model_dump())
        self._checkpointer = CheckpointManager(master_config.checkpointing)
        # Also built here, not on the driver: TimeoutChecker starts its
        # wall clock in __init__, so a driver-built one would count the
        # cluster and policy setup against the training budget.
        self._timeout = TimeoutChecker(
            timeout=master_config.checkpointing.get("checkpoint_must_save_by"),
            fit_last_save_time=True,
        )
        self._timeout.start_iterations()
        self._setup_loaders()

    def run(self) -> dict[str, Any]:
        """Run SFT training."""
        try:
            self._trainer.prepare_for_training()
            while self._save_state.total_steps < self._max_steps:
                metrics = self._run_train_step()
                self._logger.log_metrics(metrics, self._save_state.total_steps)
                metric = self._checkpoint_metric(metrics)
                self._timeout.mark_iteration()
                save_by_timeout = self._timeout.check_save()
                if self._should_save(save_by_timeout=save_by_timeout):
                    self._save_checkpoint(metric)
                if save_by_timeout:
                    # check_save fires once and then latches, so continuing
                    # would train unsaved until the walltime kill.
                    print("Timeout has been reached, stopping training early")
                    break
            return vars(self._save_state).copy()
        finally:
            for cleanup, name in (
                (self._close_loaders, "loader close"),
                (self._logger.finish, "logger close"),
                (self._checkpointer.shutdown, "checkpoint close"),
            ):
                try:
                    cleanup()
                except Exception as error:  # teardown must preserve the run failure
                    warnings.warn(f"SFTv2 {name} failed: {error}", stacklevel=2)

    def _setup_loaders(self) -> None:
        config = self._master_config
        common_kwargs = {
            "data_config": config.data,
            "batch_size": config.policy["train_global_batch_size"]
            // self._placement_plan.logical_world_size,
            "max_sequence_length": config.data["max_input_seq_length"],
            "placement_fingerprint": self._placement_plan.placement_hash,
        }
        if self._loader_states is None:
            futures = self._trainer.worker_group.run_all_workers_single_data(
                "setup_sft_dataloader",
                run_rank_0_only_axes=list(REPLICATED_AXES),
                **common_kwargs,
            )
        else:
            if len(self._loader_states) != self._placement_plan.logical_world_size:
                raise ValueError(
                    "The checkpoint must contain one loader state per logical DP shard."
                )
            futures = self._trainer.worker_group.run_all_workers_multiple_data(
                "setup_sft_dataloader",
                restored_state=self._loader_states,
                run_rank_0_only_axes=list(REPLICATED_AXES),
                common_kwargs=common_kwargs,
            )
        results = ray.get(futures)
        if results != [True] * self._placement_plan.logical_world_size:
            raise RuntimeError(f"Unexpected SFT loader setup results: {results!r}.")

    def _load_envelopes(self) -> list[StepEnvelope]:
        futures = self._trainer.worker_group.run_all_workers_single_data(
            "load_next_sft_batch",
            run_rank_0_only_axes=list(REPLICATED_AXES),
            only_unmask_final=self._master_config.sft.only_unmask_final,
            make_sequence_length_divisible_by=self._master_config.policy[
                "make_sequence_length_divisible_by"
            ],
        )
        envelopes = ray.get(futures)
        logical_ranks = [envelope.logical_rank for envelope in envelopes]
        expected = list(range(self._placement_plan.logical_world_size))
        if logical_ranks != expected:
            raise RuntimeError(
                f"SFT loader envelopes arrived for ranks {logical_ranks}; expected {expected}."
            )
        return envelopes

    def _run_train_step(self) -> dict[str, Any]:
        started = time.monotonic()
        envelopes = self._load_envelopes()
        train_started = time.monotonic()
        step_open = False
        try:
            self._trainer.begin_train_step(self._loss_fn)
            step_open = True
            self._trainer.train_placed_microbatches(
                [envelope.meta for envelope in envelopes]
            )
            train_results = self._trainer.finish_train_step()
            step_open = False
            self._owner_call("commit_sft_batch")
        except Exception:
            if step_open:
                try:
                    self._trainer.abort_train_step()
                except Exception as error:  # preserve the policy-step failure
                    warnings.warn(f"SFTv2 policy abort failed: {error}", stacklevel=2)
            try:
                self._owner_call("abort_sft_batch")
            except Exception as error:  # preserve the policy-step failure
                warnings.warn(f"SFTv2 loader abort failed: {error}", stacklevel=2)
            raise

        policy_seconds = time.monotonic() - train_started
        valid_tokens = sum(envelope.valid_tokens for envelope in envelopes)
        self._save_state.total_steps += 1
        self._save_state.consumed_samples += sum(
            len(envelope.source_ids) for envelope in envelopes
        )
        self._save_state.total_valid_tokens += valid_tokens
        loader_seconds = [envelope.load_seconds for envelope in envelopes]
        loader_latency_max = max(loader_seconds)
        metrics: dict[str, Any] = {
            "loader_latency_max": loader_latency_max,
            "loader_latency_mean": statistics.fmean(loader_seconds),
            "loader_copy_imbalance": loader_latency_max - min(loader_seconds),
            "policy_time": policy_seconds,
            "total_step_time": time.monotonic() - started,
            "valid_tokens": valid_tokens,
            "source_samples": sum(len(envelope.source_ids) for envelope in envelopes),
            "physical_packs": sum(
                len(envelope.meta.sample_ids) for envelope in envelopes
            ),
            "valid_tokens_per_second": valid_tokens
            / max(time.monotonic() - started, 1e-12),
        }
        metrics.update(self._policy_metrics(train_results))
        return metrics

    @staticmethod
    def _policy_metrics(train_results: dict[str, Any]) -> dict[str, Any]:
        """Convert policy output into flat scalar logger metrics."""
        metrics: dict[str, Any] = {
            "loss": float(train_results["loss"]),
            "grad_norm": float(train_results["grad_norm"]),
        }
        for key, values in train_results.get("all_mb_metrics", {}).items():
            if key in {"lr", "wd", "global_valid_seqs", "global_valid_toks"}:
                metrics[key] = np.mean(values).item()
            else:
                metrics[key] = np.sum(values).item()
        for key, value in train_results.get("moe_metrics", {}).items():
            metrics[f"moe/{key}"] = value
        for key in ("total_flops", "num_ranks", "theoretical_tflops"):
            if key in train_results:
                metrics[key] = train_results[key]
        return metrics

    def _owner_call(self, method_name: str) -> list[Any]:
        futures = self._trainer.worker_group.run_all_workers_single_data(
            method_name,
            run_rank_0_only_axes=list(REPLICATED_AXES),
        )
        return ray.get(futures)

    def _loader_state_dicts(self) -> list[dict[str, Any]]:
        return self._owner_call("sft_dataloader_state_dict")

    def _should_save(self, *, save_by_timeout: bool) -> bool:
        config = self._master_config.checkpointing
        ft_save_period = config.get("ft_save_period")
        steps = self._save_state.total_steps
        return bool(config["enabled"]) and (
            save_by_timeout
            or steps == self._max_steps
            or steps % config["save_period"] == 0
            or (ft_save_period is not None and steps % ft_save_period == 0)
        )

    def _checkpoint_metric(self, metrics: dict[str, Any]) -> dict[str, float]:
        """Read checkpointing.metric_name out of one step's training metrics.

        Called every step rather than only on a save so a name that no step
        produces fails on step 1 instead of at the first checkpoint.
        """
        metric_name = self._master_config.checkpointing["metric_name"]
        if metric_name is None:
            return {}
        # setup_sft_v2 already rejected anything but a train: name.
        key = metric_name.split(":", 1)[1]
        if key not in metrics:
            raise ValueError(
                f"checkpointing.metric_name={metric_name!r} names a training "
                f"metric this step did not produce. Available: {sorted(metrics)}."
            )
        return {metric_name: float(metrics[key])}

    def _save_checkpoint(self, metric: dict[str, float]) -> None:
        step = self._save_state.total_steps
        loader_states = self._loader_state_dicts()
        if len(loader_states) != self._placement_plan.logical_world_size:
            raise RuntimeError("Refusing to save without every logical loader state.")
        # CheckpointManager ranks keep_top_k on metric_name read back out of
        # training_info, so the metric travels with the save state.
        training_info = vars(self._save_state) | metric
        checkpoint_path = self._checkpointer.init_tmp_checkpoint(
            step, training_info, self._master_config
        )
        self._trainer.save_checkpoint(
            weights_path=os.path.join(checkpoint_path, "policy", "weights"),
            optimizer_path=(
                os.path.join(checkpoint_path, "policy", "optimizer")
                if self._checkpointer.save_optimizer
                else None
            ),
            tokenizer_path=os.path.join(checkpoint_path, "policy", "tokenizer"),
            checkpointing_cfg=self._master_config.checkpointing,
        )
        torch.save(loader_states, os.path.join(checkpoint_path, "sft_v2_loaders.pt"))
        self._checkpointer.begin_finalization(
            checkpoint_path, wait_fn=self._trainer.finalize_async_save
        )

    def _close_loaders(self) -> None:
        self._owner_call("close_sft_dataloader")


def setup_sft_v2(
    master_config: MasterConfig, tokenizer_or_processor: Any
) -> SFTV2ActorArgs:
    """Build the V2 cluster, TQPolicy, placement, and resume state."""
    set_seed(master_config.sft.seed)
    if master_config.data.get("backend") != "energon":
        raise ValueError("SFTv2 requires data.backend=energon.")
    if not master_config.policy["megatron_cfg"]["enabled"]:
        raise ValueError("SFTv2 supports only the Megatron policy backend.")
    if (
        master_config.policy["sequence_packing"]["enabled"]
        or master_config.policy["dynamic_batching"]["enabled"]
    ):
        raise ValueError("SFTv2 requires fixed NeMo-RL batching.")
    # SFTConfig carries validation knobs that default to on (val_period=10,
    # val_at_start=True) and this loop has no validation path, so reject them
    # rather than accepting a config whose validation silently never runs.
    if (
        master_config.sft.val_period != 0
        or master_config.sft.val_at_start
        or master_config.sft.val_at_end
    ):
        raise ValueError(
            "SFTv2 has no validation loop. Set sft.val_period=0, "
            "sft.val_at_start=false and sft.val_at_end=false."
        )
    # sft_worker builds its loader with split_role="train" only, so a validation
    # source would be accepted and never read.
    if master_config.data.get("validation") is not None:
        raise ValueError("SFTv2 reads no validation source. Set data.validation=null.")
    # Same reason: only train metrics exist here, so a val: name would leave
    # keep_top_k ranking every checkpoint on a metric that is never written.
    metric_name = master_config.checkpointing["metric_name"]
    if metric_name is not None and not metric_name.startswith("train:"):
        raise ValueError(
            "SFTv2 can rank checkpoints on training metrics only. Set "
            "checkpointing.metric_name to null or 'train:<metric>'; got "
            f"{metric_name!r}."
        )
    max_sequence_length = master_config.data["max_input_seq_length"]
    if max_sequence_length is None:
        raise ValueError("SFTv2 requires data.max_input_seq_length.")

    processor = None
    tokenizer = tokenizer_or_processor
    if not isinstance(tokenizer_or_processor, PreTrainedTokenizerBase):
        processor = tokenizer_or_processor
        tokenizer = tokenizer_or_processor.tokenizer
    if processor is None:
        raise ValueError("SFTv2 requires a multimodal processor.")
    # Workers rebuild the processor in-process rather than receiving it as a
    # pickled constructor argument. A trust_remote_code processor's class lives
    # in ``transformers_modules``, which Ray's worker interpreters cannot import
    # while deserializing their arguments, so shipping the object fails there.
    master_config.policy["tokenizer"]["use_processor"] = True

    checkpoint_probe = CheckpointManager(master_config.checkpointing)
    latest = checkpoint_probe.get_latest_checkpoint_path()
    loaded_training_info = checkpoint_probe.load_training_info(latest)
    weights_path, optimizer_path = checkpoint_probe.get_resume_paths(latest)
    checkpoint_probe.shutdown()

    cluster_config = master_config.cluster
    num_nodes = cluster_config["num_nodes"]
    segment_size = cluster_config.get("segment_size")
    node_constraints, _, _ = prepare_segment_topology(segment_size, num_nodes)
    cluster = RayVirtualCluster(
        name="sft_v2_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]] * num_nodes,
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
        port_range_low=cluster_config.get("master_port_range_low"),
        port_range_high=cluster_config.get("master_port_range_high"),
        segment_size=segment_size,
        node_resource_constraints=node_constraints,
    )
    megatron_config = cast(dict[str, Any], master_config.policy["megatron_cfg"])
    megatron_config["train_iters"] = _max_train_steps(master_config)
    trainer = TQPolicy(
        cluster=cluster,
        config=master_config.policy,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=False,
        worker_extension_cls_fqn=(
            "nemo_rl.data.energon.sft_worker.SFTMegatronPolicyWorker"
        ),
        dp_cfg=master_config.data_plane,
    )
    mapper_name = master_config.data["energon"].topology_mapper
    placement_plan = resolve_topology_mapper(mapper_name).map(
        trainer.sharding_annotations
    )
    global_batch_size = master_config.policy["train_global_batch_size"]
    if global_batch_size % placement_plan.logical_world_size != 0:
        raise ValueError(
            "policy.train_global_batch_size must be divisible by the logical "
            f"DP size: {global_batch_size} % "
            f"{placement_plan.logical_world_size} != 0."
        )
    save_state = _restore_save_state(
        loaded_training_info, placement_hash=placement_plan.placement_hash
    )
    loader_states = None
    if latest is not None:
        loader_path = Path(latest) / "sft_v2_loaders.pt"
        if not loader_path.exists():
            raise ValueError(f"SFTv2 checkpoint is missing {loader_path.name}.")
        loader_states = torch.load(loader_path, weights_only=False)
    loss_fn = NLLLossFn(
        use_fused_linear_logprobs=megatron_config["use_fused_linear_logprobs"]
    )
    return SFTV2ActorArgs(
        trainer=trainer,
        loss_fn=loss_fn,
        train_cluster=cluster,
        placement_plan=placement_plan,
        save_state=save_state,
        loader_states=loader_states,
    )


__all__ = [
    "MasterConfig",
    "SFTSingleControllerActor",
    "SFTV2ActorArgs",
    "SFTV2SaveState",
    "setup_sft_v2",
]
