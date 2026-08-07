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

"""Offline GRPO training over fixed, rewarded teacher trajectories."""

import os
import warnings
from dataclasses import dataclass, fields
from typing import Any, Literal, cast

import numpy as np
import torch
from pydantic import BaseModel
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.advantage_estimator import OfflineGRPOAdvantageEstimator
from nemo_rl.algorithms.loss import OfflineGRPOLossConfig, OfflineGRPOLossFn
from nemo_rl.algorithms.utils import maybe_pad_last_batch, set_seed
from nemo_rl.data.offline_grpo import (
    OfflineGRPOBatchMetrics,
    OfflineGRPODataConfig,
    OfflineGRPODataset,
    OfflineGRPOGroup,
    offline_grpo_collate_fn,
    prepare_offline_grpo_batch,
)
from nemo_rl.data import DataConfig
from nemo_rl.data.utils import load_dataloader_state
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
    prepare_segment_topology,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import PolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer


class OfflineGRPOConfig(BaseModel, extra="allow"):
    """Training and group-advantage configuration for offline GRPO."""

    num_prompts_per_step: int = 4
    num_responses_per_prompt: int = 8
    response_selection: Literal["first", "random"] = "random"
    normalize_rewards: bool = False
    use_leave_one_out_baseline: bool = False
    all_positive_bias: float = 0.1
    positive_reward_threshold: float = 0.0
    max_num_steps: int = 60
    max_num_epochs: int = 3
    val_period: int = 10
    val_batches: int = 8
    val_num_prompts_per_step: int = 4
    val_micro_batch_size: int = 1
    val_at_start: bool = True
    val_at_end: bool = False
    seed: int = 42


class MasterConfig(BaseModel, extra="allow"):
    """Top-level offline GRPO configuration."""

    policy: PolicyConfig
    data: OfflineGRPODataConfig
    offline_grpo: OfflineGRPOConfig
    loss_fn: OfflineGRPOLossConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


@dataclass
class OfflineGRPOSaveState:
    """Checkpointed driver state for offline GRPO."""

    epoch: int
    step: int
    total_steps: int
    consumed_samples: int
    total_valid_tokens: int


def _initial_save_state() -> OfflineGRPOSaveState:
    return OfflineGRPOSaveState(
        epoch=0,
        step=0,
        total_steps=0,
        consumed_samples=0,
        total_valid_tokens=0,
    )


def _validate_batch_configuration(master_config: MasterConfig) -> None:
    positive_fields = {
        "num_prompts_per_step": master_config.offline_grpo.num_prompts_per_step,
        "max_num_steps": master_config.offline_grpo.max_num_steps,
        "max_num_epochs": master_config.offline_grpo.max_num_epochs,
        "val_num_prompts_per_step": (
            master_config.offline_grpo.val_num_prompts_per_step
        ),
        "val_micro_batch_size": master_config.offline_grpo.val_micro_batch_size,
    }
    for field_name, value in positive_fields.items():
        if value <= 0:
            raise ValueError(f"offline_grpo.{field_name} must be positive")
    if master_config.offline_grpo.num_responses_per_prompt <= 1:
        raise ValueError("offline_grpo.num_responses_per_prompt must be greater than 1")
    if master_config.offline_grpo.all_positive_bias < 0:
        raise ValueError("offline_grpo.all_positive_bias must be non-negative")

    trajectories_per_step = (
        master_config.offline_grpo.num_prompts_per_step
        * master_config.offline_grpo.num_responses_per_prompt
    )
    if master_config.policy["train_global_batch_size"] != trajectories_per_step:
        raise ValueError(
            "policy.train_global_batch_size must equal "
            "offline_grpo.num_prompts_per_step * "
            "offline_grpo.num_responses_per_prompt; got "
            f"{master_config.policy['train_global_batch_size']} and "
            f"{trajectories_per_step}"
        )


def setup(
    master_config: MasterConfig,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: OfflineGRPODataset,
    val_dataset: OfflineGRPODataset | None,
) -> tuple[
    Policy,
    RayVirtualCluster,
    StatefulDataLoader,
    StatefulDataLoader | None,
    OfflineGRPOLossFn,
    OfflineGRPOAdvantageEstimator,
    Logger,
    CheckpointManager,
    OfflineGRPOSaveState,
    MasterConfig,
]:
    """Initialize data loaders, policy, loss, logging, and checkpoint state."""
    _validate_batch_configuration(master_config)
    set_seed(master_config.offline_grpo.seed)

    policy_config = master_config.policy
    algorithm_config = master_config.offline_grpo
    data_config = master_config.data
    checkpointing_config = master_config.checkpointing

    checkpointing_pretrained = checkpointing_config.get("pretrained_checkpoint")
    if checkpointing_pretrained is not None:
        policy_config["pretrained_checkpoint"] = checkpointing_pretrained

    logger = Logger(master_config.logger)
    logger.log_hyperparams(master_config.model_dump())

    checkpointer = CheckpointManager(checkpointing_config)
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    loaded_state = checkpointer.load_training_info(last_checkpoint_path)
    if loaded_state is None:
        save_state = _initial_save_state()
    else:
        loaded_state.setdefault("total_valid_tokens", 0)
        known_fields = {field.name for field in fields(OfflineGRPOSaveState)}
        save_state = OfflineGRPOSaveState(
            **{key: value for key, value in loaded_state.items() if key in known_fields}
        )

    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=algorithm_config.num_prompts_per_step,
        shuffle=data_config.shuffle,
        collate_fn=offline_grpo_collate_fn,
        drop_last=True,
        num_workers=data_config.num_workers,
    )
    if last_checkpoint_path is not None:
        load_dataloader_state(
            train_dataloader,
            last_checkpoint_path,
            cast(DataConfig, data_config.model_dump()),
        )

    val_dataloader = (
        StatefulDataLoader(
            val_dataset,
            batch_size=algorithm_config.val_num_prompts_per_step,
            shuffle=False,
            collate_fn=offline_grpo_collate_fn,
            drop_last=False,
            num_workers=data_config.num_workers,
        )
        if val_dataset is not None
        else None
    )

    print("\n▶ Setting up compute cluster...")
    num_nodes = master_config.cluster["num_nodes"]
    gpus_per_node = master_config.cluster["gpus_per_node"]
    segment_size = master_config.cluster.get("segment_size")
    node_constraints, _, _ = prepare_segment_topology(segment_size, num_nodes)
    cluster = RayVirtualCluster(
        name="offline_grpo_cluster",
        bundle_ct_per_node_list=[gpus_per_node] * num_nodes,
        use_gpus=True,
        num_gpus_per_node=gpus_per_node,
        max_colocated_worker_groups=1,
        port_range_low=master_config.cluster.get("master_port_range_low"),
        port_range_high=master_config.cluster.get("master_port_range_high"),
        segment_size=segment_size,
        node_resource_constraints=node_constraints,
    )

    if policy_config.get("megatron_cfg", {}).get("enabled", False):
        policy_config["megatron_cfg"]["train_iters"] = min(
            algorithm_config.max_num_steps,
            algorithm_config.max_num_epochs * len(train_dataloader),
        )

    weights_path, optimizer_path = checkpointer.get_resume_paths(last_checkpoint_path)
    policy = Policy(
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=master_config.loss_fn.reference_policy_kl_penalty > 0,
    )
    policy.print_node_ip_and_gpu_id()

    use_fused_logprobs = bool(
        policy_config["megatron_cfg"]["enabled"]
        and policy_config["megatron_cfg"].get("use_fused_linear_logprobs", False)
    )
    loss_fn = OfflineGRPOLossFn(
        master_config.loss_fn,
        use_fused_linear_logprobs=use_fused_logprobs,
    )
    advantage_estimator = OfflineGRPOAdvantageEstimator(
        use_leave_one_out_baseline=algorithm_config.use_leave_one_out_baseline,
        normalize_rewards=algorithm_config.normalize_rewards,
        all_positive_bias=algorithm_config.all_positive_bias,
        positive_reward_threshold=algorithm_config.positive_reward_threshold,
    )

    print("  ✓ Offline GRPO setup complete (no rollout worker allocated)")
    return (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        advantage_estimator,
        logger,
        checkpointer,
        save_state,
        master_config,
    )


def _prepare_policy_batch(
    groups: list[OfflineGRPOGroup],
    *,
    dataset: OfflineGRPODataset,
    policy: PolicyInterface,
    tokenizer: PreTrainedTokenizerBase,
    advantage_estimator: OfflineGRPOAdvantageEstimator,
    master_config: MasterConfig,
    selection_step: int,
    micro_batch_size: int,
    pad_partial_batch: bool,
) -> tuple[BatchedDataDict[Any], OfflineGRPOBatchMetrics]:
    """Prepare advantages and reference logprobs for a prompt-group batch."""
    algorithm_config = master_config.offline_grpo
    prepared = prepare_offline_grpo_batch(
        groups,
        dataset,
        tokenizer,
        num_responses_per_prompt=algorithm_config.num_responses_per_prompt,
        response_selection=algorithm_config.response_selection,
        seed=algorithm_config.seed,
        step=selection_step,
        positive_reward_threshold=algorithm_config.positive_reward_threshold,
        make_sequence_length_divisible_by=master_config.policy[
            "make_sequence_length_divisible_by"
        ],
    )
    data = prepared.data
    mask = data["token_mask"] * data["sample_mask"].unsqueeze(-1)
    data["advantages"] = advantage_estimator.compute_advantage(
        prepared.prompt_ids,
        prepared.rewards,
        mask,
    )
    del data["rewards"]
    del data["prompt_ids"]

    if pad_partial_batch:
        old_size = data.size
        dp_size = policy.sharding_annotations.get_axis_size("data_parallel")
        data = cast(
            BatchedDataDict[Any],
            maybe_pad_last_batch(cast(dict, data), dp_size, micro_batch_size),
        )
        if data.size > old_size:
            padding_shape = (data.size - old_size, data["advantages"].shape[1])
            data["advantages"] = torch.cat(
                [data["advantages"], torch.zeros(padding_shape)], dim=0
            )

    if master_config.loss_fn.reference_policy_kl_penalty > 0:
        data["reference_policy_logprobs"] = policy.get_reference_policy_logprobs(
            cast(Any, data),
            micro_batch_size=micro_batch_size,
        )["reference_logprobs"]
    return data, prepared.metrics


def _reduce_training_metrics(
    train_results: dict[str, Any], batch_metrics: OfflineGRPOBatchMetrics
) -> dict[str, float]:
    metrics = {
        "loss": train_results["loss"].numpy(),
        "grad_norm": train_results["grad_norm"].numpy(),
    }
    if "moe_metrics" in train_results:
        metrics.update(
            {f"moe/{key}": value for key, value in train_results["moe_metrics"].items()}
        )
    metrics.update(train_results["all_mb_metrics"])
    reduced = {}
    for key, value in metrics.items():
        reduction = (
            np.mean
            if key in {"lr", "wd", "global_valid_seqs", "global_valid_toks"}
            else np.sum
        )
        reduced[key] = float(reduction(value))
    reduced.update(
        {
            "reward/mean": batch_metrics.mean_reward,
            "reward/all_positive_group_fraction": batch_metrics.all_positive_group_fraction,
            "reward/all_non_positive_group_fraction": batch_metrics.all_non_positive_group_fraction,
            "data/invalid_sequence_fraction": batch_metrics.invalid_sequence_fraction,
            "reward/num_prompt_groups": float(batch_metrics.num_prompt_groups),
        }
    )
    return reduced


def validate(
    policy: PolicyInterface,
    val_dataloader: StatefulDataLoader | None,
    tokenizer: PreTrainedTokenizerBase,
    loss_fn: OfflineGRPOLossFn,
    advantage_estimator: OfflineGRPOAdvantageEstimator,
    master_config: MasterConfig,
    *,
    step: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Evaluate the offline objective on held-out trajectory groups."""
    if val_dataloader is None:
        if master_config.offline_grpo.val_period > 0:
            raise ValueError(
                "data.validation is required when offline_grpo.val_period is positive"
            )
        return {}, {}

    timer = Timer()
    token_weighted_keys = {
        "loss",
        "actor_loss",
        "kl_penalty",
        "offline_probability",
        "advantage",
    }
    weighted_sums = {key: 0.0 for key in token_weighted_keys}
    total_valid_tokens = 0.0
    total_valid_samples = 0.0
    reward_sum = 0.0
    all_positive_sum = 0.0
    all_non_positive_sum = 0.0
    total_groups = 0

    with timer.time("total_validation_time"):
        for batch_idx, groups in enumerate(val_dataloader):
            data, batch_metrics = _prepare_policy_batch(
                groups,
                dataset=cast(OfflineGRPODataset, val_dataloader.dataset),
                policy=policy,
                tokenizer=tokenizer,
                advantage_estimator=advantage_estimator,
                master_config=master_config,
                selection_step=0,
                micro_batch_size=master_config.offline_grpo.val_micro_batch_size,
                pad_partial_batch=True,
            )
            results = policy.train(
                data,
                loss_fn,
                eval_mode=True,
                gbs=data.size,
                mbs=master_config.offline_grpo.val_micro_batch_size,
            )
            metrics = results["all_mb_metrics"]
            batch_valid_tokens = float(np.mean(metrics["global_valid_toks"]))
            total_valid_tokens += batch_valid_tokens
            total_valid_samples += float(np.sum(metrics["num_valid_samples"]))
            for key in token_weighted_keys:
                weighted_sums[key] += float(np.sum(metrics[key])) * batch_valid_tokens

            group_count = batch_metrics.num_prompt_groups
            reward_sum += batch_metrics.mean_reward * group_count
            all_positive_sum += batch_metrics.all_positive_group_fraction * group_count
            all_non_positive_sum += (
                batch_metrics.all_non_positive_group_fraction * group_count
            )
            total_groups += group_count
            if (
                master_config.offline_grpo.val_batches > 0
                and batch_idx + 1 >= master_config.offline_grpo.val_batches
            ):
                break

    denominator = max(total_valid_tokens, 1.0)
    val_metrics = {
        f"val_{key}": value / denominator for key, value in weighted_sums.items()
    }
    val_metrics.update(
        {
            "val_num_valid_samples": total_valid_samples,
            "val_global_valid_toks": total_valid_tokens,
            "val_reward_mean": reward_sum / max(total_groups, 1),
            "val_all_positive_group_fraction": all_positive_sum / max(total_groups, 1),
            "val_all_non_positive_group_fraction": all_non_positive_sum
            / max(total_groups, 1),
        }
    )
    policy.prepare_for_training()
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    logger_lines = ", ".join(
        f"{key}={value:.4f}" for key, value in sorted(val_metrics.items())
    )
    print(f"📊 Offline validation at step {step}: {logger_lines}")
    return val_metrics, timing_metrics


def _save_checkpoint(
    *,
    policy: PolicyInterface,
    train_dataloader: StatefulDataLoader,
    checkpointer: CheckpointManager,
    save_state: OfflineGRPOSaveState,
    master_config: MasterConfig,
    metrics: dict[str, float],
    val_metrics: dict[str, float] | None,
    total_steps: int,
) -> None:
    """Save model, optimizer, tokenizer, dataloader, and driver state."""
    full_metric_name = master_config.checkpointing["metric_name"]
    if full_metric_name is not None:
        if not full_metric_name.startswith(("train:", "val:")):
            raise ValueError(
                "checkpointing.metric_name must start with 'train:' or 'val:'"
            )
        prefix, metric_name = full_metric_name.split(":", 1)
        metric_source = metrics if prefix == "train" else val_metrics
        if metric_source and metric_name in metric_source:
            setattr(save_state, full_metric_name, metric_source[metric_name])
        else:
            warnings.warn(
                f"Checkpoint metric {full_metric_name!r} is unavailable for this step",
                stacklevel=2,
            )

    checkpoint_path = cast(
        Any,
        checkpointer.init_tmp_checkpoint(total_steps, vars(save_state), master_config),
    )
    policy.save_checkpoint(
        weights_path=os.path.join(checkpoint_path, "policy", "weights"),
        optimizer_path=(
            os.path.join(checkpoint_path, "policy", "optimizer")
            if checkpointer.save_optimizer
            else None
        ),
        tokenizer_path=os.path.join(checkpoint_path, "policy", "tokenizer"),
        checkpointing_cfg=master_config.checkpointing,
    )
    torch.save(
        train_dataloader.state_dict(),
        os.path.join(checkpoint_path, "train_dataloader.pt"),
    )
    checkpointer.begin_finalization(
        checkpoint_path,
        wait_fn=policy.finalize_async_save,
    )


def offline_grpo_train(
    policy: PolicyInterface,
    train_dataloader: StatefulDataLoader,
    val_dataloader: StatefulDataLoader | None,
    tokenizer: PreTrainedTokenizerBase,
    loss_fn: OfflineGRPOLossFn,
    advantage_estimator: OfflineGRPOAdvantageEstimator,
    master_config: MasterConfig,
    logger: Logger,
    checkpointer: CheckpointManager,
    save_state: OfflineGRPOSaveState,
) -> None:
    """Run offline GRPO without generation or environment workers."""
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config.checkpointing["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    current_epoch = save_state.epoch
    current_step = save_state.step
    total_steps = save_state.total_steps
    total_valid_tokens = save_state.total_valid_tokens
    algorithm_config = master_config.offline_grpo
    val_metrics = None

    if algorithm_config.val_at_start and total_steps == 0:
        val_metrics, validation_timings = validate(
            policy,
            val_dataloader,
            tokenizer,
            loss_fn,
            advantage_estimator,
            master_config,
            step=0,
        )
        logger.log_metrics(val_metrics, 0, prefix="validation")
        logger.log_metrics(validation_timings, 0, prefix="timing/validation")

    policy.prepare_for_training()
    while (
        current_epoch < algorithm_config.max_num_epochs
        and total_steps < algorithm_config.max_num_steps
    ):
        print(
            f"\n{'=' * 25} Epoch {current_epoch + 1}/"
            f"{algorithm_config.max_num_epochs} {'=' * 25}"
        )
        for groups in train_dataloader:
            maybe_gpu_profile_step(cast(Any, policy), total_steps + 1)
            val_metrics = None
            validation_timings = None
            with timer.time("total_step_time"):
                with timer.time("data_processing"):
                    data, batch_metrics = _prepare_policy_batch(
                        groups,
                        dataset=cast(OfflineGRPODataset, train_dataloader.dataset),
                        policy=policy,
                        tokenizer=tokenizer,
                        advantage_estimator=advantage_estimator,
                        master_config=master_config,
                        selection_step=total_steps,
                        micro_batch_size=master_config.policy["train_micro_batch_size"],
                        pad_partial_batch=False,
                    )

                with timer.time("policy_training"):
                    train_results = policy.train(
                        data,
                        loss_fn,
                        gbs=data.size,
                        mbs=master_config.policy["train_micro_batch_size"],
                        timer=timer,
                    )
                metrics = _reduce_training_metrics(train_results, batch_metrics)
                total_valid_tokens += int(metrics["global_valid_toks"])

                is_last_step = total_steps + 1 >= algorithm_config.max_num_steps or (
                    current_epoch + 1 == algorithm_config.max_num_epochs
                    and current_step + 1 == len(train_dataloader)
                )
                if (
                    algorithm_config.val_period > 0
                    and (total_steps + 1) % algorithm_config.val_period == 0
                ) or (algorithm_config.val_at_end and is_last_step):
                    val_metrics, validation_timings = validate(
                        policy,
                        val_dataloader,
                        tokenizer,
                        loss_fn,
                        advantage_estimator,
                        master_config,
                        step=total_steps + 1,
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )
                    logger.log_metrics(
                        validation_timings,
                        total_steps + 1,
                        prefix="timing/validation",
                    )

                save_state.consumed_samples += algorithm_config.num_prompts_per_step
                timeout.mark_iteration()
                should_save_by_timeout = timeout.check_save()
                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config.checkpointing["save_period"]
                    == 0
                )
                if master_config.checkpointing["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    save_state.epoch = current_epoch
                    save_state.step = (current_step + 1) % len(train_dataloader)
                    save_state.total_steps = total_steps + 1
                    save_state.total_valid_tokens = total_valid_tokens
                    _save_checkpoint(
                        policy=policy,
                        train_dataloader=train_dataloader,
                        checkpointer=checkpointer,
                        save_state=save_state,
                        master_config=master_config,
                        metrics=metrics,
                        val_metrics=val_metrics,
                        total_steps=total_steps + 1,
                    )

            timing_metrics = timer.get_timing_metrics(reduction_op="sum")
            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")
            print(
                f"📊 Step {total_steps + 1}: loss={metrics['loss']:.6f}, "
                f"reward={metrics['reward/mean']:.4f}, "
                f"all-positive={metrics['reward/all_positive_group_fraction']:.4f}, "
                f"invalid={metrics['data/invalid_sequence_fraction']:.4f}"
            )
            timer.reset()
            current_step += 1
            total_steps += 1

            if should_save_by_timeout:
                print("Timeout reached; stopped after saving a checkpoint")
                return
            if total_steps >= algorithm_config.max_num_steps:
                return

        current_epoch += 1
        current_step = 0
