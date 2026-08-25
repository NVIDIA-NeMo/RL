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

"""Non-colocated teacher worker group for MOPD async distillation.

Each TeacherWorkerGroup wraps a RayWorkerGroup running MegatronPolicyWorker
in inference-only mode for a single teacher model checkpoint.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.opd import TeacherPrecision, TeacherResourceConfig
from nemo_rl.algorithms.opd_packed import (
    OPD_TEACHER_TOPK_PACKED_KEY,
    resolve_packed_field,
)
from nemo_rl.distributed.batched_data_dict import (
    BatchedDataDict,
    SequencePackingArgs,
)
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.interfaces import GenerationDatumSpec
from nemo_rl.models.policy.interfaces import (
    ReferenceLogprobOutputSpec,
    TopkLogprobsOutputSpec,
)


@dataclass
class TeacherConfig:
    """Resolved config for a single non-colocated teacher (built in-process).

    Parallel-size fields contain the effective values after applying
    ``megatron_cfg_overrides`` so placement, sharding, and Megatron agree.
    """

    alias: str
    model_name: str  # checkpoint path
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    context_parallel_size: int
    expert_tensor_parallel_size: int
    expert_model_parallel_size: int
    num_nodes: int
    gpus_per_node: int
    precision: TeacherPrecision
    micro_batch_size: int
    megatron_cfg_overrides: dict[str, Any]
    segment_size: Optional[int] = None


def create_teacher_configs_from_opd_config(
    opd_cfg: dict[str, Any],
) -> list[TeacherConfig]:
    """Build per-teacher configs from on_policy_distillation config.

    Per-teacher fields and ``megatron_cfg_overrides`` are merged over the
    defaults. The precedence from least to most specific is default field,
    default Megatron override, alias field, then alias Megatron override.
    Aliases sharing a checkpoint are deduplicated only when their effective
    configs are identical.

    Raises:
        ValueError: If aliases sharing a checkpoint have conflicting effective
            configs while checkpoint deduplication is enabled.
    """
    teacher_model_by_agent_name: dict[str, str] = dict(
        opd_cfg.get("teacher_model_by_agent_name", {})
    )
    non_coloc_cfg = dict(opd_cfg.get("non_colocated_teachers", {}))
    default_cfg = dict(non_coloc_cfg.get("default_teacher_cfg", {}))
    overrides = dict(non_coloc_cfg.get("teacher_overrides", {}))
    deduplicate = bool(opd_cfg.get("deduplicate_shared_teacher_checkpoints", True))

    configs: list[TeacherConfig] = []
    primary_config_by_model: dict[str, TeacherConfig] = {}

    for alias, model_name in teacher_model_by_agent_name.items():
        # defaults <- per-alias override, then validated/typed by the schema.
        alias_override = dict(overrides.get(alias, {}))
        merged = {**default_cfg, **alias_override}
        alias_mco = dict(alias_override.get("megatron_cfg_overrides", {}))
        default_mco = {
            key: value
            for key, value in default_cfg.get("megatron_cfg_overrides", {}).items()
            if key not in alias_override
        }
        merged["megatron_cfg_overrides"] = {
            **default_mco,
            **alias_mco,
        }
        res = TeacherResourceConfig(**merged)

        # Unknown top-level keys (extra="allow") fold into megatron_cfg_overrides;
        # explicit megatron_cfg_overrides take precedence.
        all_overrides = {**(res.model_extra or {}), **res.megatron_cfg_overrides}
        tp = res.tensor_model_parallel_size
        pp = res.pipeline_model_parallel_size
        cp = res.context_parallel_size
        etp = res.expert_tensor_parallel_size
        ep = res.expert_model_parallel_size
        if "tensor_model_parallel_size" in all_overrides:
            tp = int(all_overrides["tensor_model_parallel_size"])
        if "pipeline_model_parallel_size" in all_overrides:
            pp = int(all_overrides["pipeline_model_parallel_size"])
        if "context_parallel_size" in all_overrides:
            cp = int(all_overrides["context_parallel_size"])
        if "expert_tensor_parallel_size" in all_overrides:
            etp = int(all_overrides["expert_tensor_parallel_size"])
        if "expert_model_parallel_size" in all_overrides:
            ep = int(all_overrides["expert_model_parallel_size"])

        config = TeacherConfig(
            alias=alias,
            model_name=model_name,
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            context_parallel_size=cp,
            expert_tensor_parallel_size=etp,
            expert_model_parallel_size=ep,
            num_nodes=res.num_nodes,
            gpus_per_node=res.gpus_per_node,
            precision=res.precision,
            micro_batch_size=res.micro_batch_size,
            megatron_cfg_overrides=all_overrides,
            segment_size=res.segment_size,
        )
        if deduplicate and model_name in primary_config_by_model:
            primary = primary_config_by_model[model_name]
            if replace(config, alias=primary.alias) != primary:
                raise ValueError(
                    "Aliases sharing a teacher checkpoint must have identical "
                    "effective resource configs when "
                    "deduplicate_shared_teacher_checkpoints=true; "
                    f"'{alias}' conflicts with primary alias '{primary.alias}' "
                    f"for checkpoint '{model_name}'. Align the aliases' overrides "
                    "or set deduplicate_shared_teacher_checkpoints=false to run "
                    "separate teacher groups."
                )
            continue
        primary_config_by_model[model_name] = config
        configs.append(config)

    return configs


def _apply_teacher_resource_config(
    cfg: dict[str, Any], teacher_cfg: TeacherConfig
) -> None:
    """Apply resolved teacher resources to a copied policy config."""
    policy_etp = cfg["megatron_cfg"].get("expert_tensor_parallel_size")
    teacher_etp = teacher_cfg.expert_tensor_parallel_size
    if policy_etp is not None and int(policy_etp) != teacher_etp:
        warnings.warn(
            f"Teacher '{teacher_cfg.alias}' uses expert_tensor_parallel_size="
            f"{teacher_etp}, independently of the policy value {policy_etp}. "
            "This may change per-rank expert memory compared with configurations "
            "that previously inherited the policy value.",
            stacklevel=2,
        )

    cfg["precision"] = teacher_cfg.precision
    cfg["megatron_cfg"]["enabled"] = True
    cfg["megatron_cfg"]["tensor_model_parallel_size"] = (
        teacher_cfg.tensor_model_parallel_size
    )
    cfg["megatron_cfg"]["pipeline_model_parallel_size"] = (
        teacher_cfg.pipeline_model_parallel_size
    )
    cfg["megatron_cfg"]["context_parallel_size"] = teacher_cfg.context_parallel_size
    cfg["megatron_cfg"]["expert_tensor_parallel_size"] = teacher_etp
    cfg["megatron_cfg"]["expert_model_parallel_size"] = (
        teacher_cfg.expert_model_parallel_size
    )


class TeacherWorkerGroup:
    """Inference-only mcore worker group for a single teacher model.

    Unlike the training policy, this group:
    - Never initializes an optimizer
    - Never initializes a reference model
    - Loads the checkpoint once at startup
    - Exposes sampled-token, caller-selected, and teacher-selected support inference
    """

    def __init__(
        self,
        teacher_cfg: TeacherConfig,
        cluster: RayVirtualCluster,
        policy_config: dict[str, Any],
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.alias = teacher_cfg.alias
        self.model_name = teacher_cfg.model_name
        self.teacher_cfg = teacher_cfg

        # Build a policy config for inference-only use.
        cfg = deepcopy(policy_config)
        cfg["model_name"] = self.model_name
        # Override parallelism from teacher config.
        if "megatron_cfg" not in cfg:
            cfg["megatron_cfg"] = {}
        _apply_teacher_resource_config(cfg, teacher_cfg)

        # Apply any additional megatron config overrides from teacher config.
        for key, value in teacher_cfg.megatron_cfg_overrides.items():
            cfg["megatron_cfg"][key] = value

        # Teachers run Megatron inference-only. Don't let the student's other
        # backend or parameter-adding features leak onto the frozen teacher.
        if cfg.get("dtensor_cfg", {}).get("enabled", False):
            raise ValueError(
                f"Teacher '{self.alias}': only the Megatron backend is supported "
                "for teachers, but the policy config has dtensor_cfg.enabled=True."
            )
        if "dtensor_cfg" in cfg:
            cfg["dtensor_cfg"]["enabled"] = False
        if "peft" in cfg["megatron_cfg"]:
            cfg["megatron_cfg"]["peft"]["enabled"] = False
        if "draft" in cfg:
            cfg["draft"]["enabled"] = False
        # The teacher uses the plain Megatron worker, so a student-side quant_cfg
        # would be silently ignored. Drop it explicitly and warn instead.
        if cfg.get("quant_cfg") is not None:
            warnings.warn(
                f"Teacher '{self.alias}': quantization is not supported for teachers; "
                "running the teacher unquantized (ignoring the policy's quant_cfg)."
            )
            cfg["quant_cfg"] = None

        tp = teacher_cfg.tensor_model_parallel_size
        pp = teacher_cfg.pipeline_model_parallel_size
        cp = teacher_cfg.context_parallel_size
        etp = teacher_cfg.expert_tensor_parallel_size
        ep = teacher_cfg.expert_model_parallel_size

        # Validate parallelism fits the cluster (matches lm_policy.py)
        world_size = cluster.world_size()
        model_parallel_size = tp * pp * cp
        if world_size < model_parallel_size:
            raise ValueError(
                f"Teacher '{self.alias}': world_size ({world_size}) < TP({tp}) * PP({pp}) * CP({cp}) = {model_parallel_size}"
            )
        if world_size % model_parallel_size != 0:
            raise ValueError(
                f"Teacher '{self.alias}': world_size ({world_size}) not divisible by TP({tp}) * PP({pp}) * CP({cp}) = {model_parallel_size}"
            )
        expert_parallel_size = etp * ep * pp
        if world_size < expert_parallel_size:
            raise ValueError(
                f"Teacher '{self.alias}': world_size ({world_size}) < "
                f"ETP({etp}) * EP({ep}) * PP({pp}) = {expert_parallel_size}"
            )
        if world_size % expert_parallel_size != 0:
            raise ValueError(
                f"Teacher '{self.alias}': world_size ({world_size}) not divisible "
                f"by ETP({etp}) * EP({ep}) * PP({pp}) = {expert_parallel_size}"
            )

        self.sharding_annotations = NamedSharding(
            layout=np.arange(world_size).reshape(pp, -1, cp, tp),
            names=[
                "pipeline_parallel",
                "data_parallel",
                "context_parallel",
                "tensor_parallel",
            ],
        )

        from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup

        worker_builder = RayWorkerBuilder(
            "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker",
            cfg,
            tokenizer=tokenizer,
            processor=None,
            init_optimizer=False,
            weights_path=None,
            optimizer_path=None,
            init_reference_model=False,
            worker_sharding_annotations=self.sharding_annotations,
        )

        env_vars = cfg["megatron_cfg"].get("env_vars", {})

        self.worker_group = RayWorkerGroup(
            cluster,
            worker_builder,
            name_prefix=f"teacher_{self.alias}",
            sharding_annotations=self.sharding_annotations,
            env_vars=env_vars or {},
        )

        self.cfg = cfg
        self._micro_batch_size = teacher_cfg.micro_batch_size

        # Set up sequence packing / dynamic batching (mirrors lm_policy.py)
        self.use_sequence_packing = cfg["sequence_packing"]["enabled"]
        self.use_dynamic_batches = cfg["dynamic_batching"]["enabled"]
        # SP-forward divisor; the collector reads it to pre-pad non-packed inputs.
        self.sequence_length_pad_multiple = cp * 2 * tp if cp > 1 else tp
        if self.use_sequence_packing:
            self.sequence_packing_args: SequencePackingArgs = {
                "algorithm": cfg["sequence_packing"]["algorithm"],
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_pad_multiple": self.sequence_length_pad_multiple,
            }
            microbatch_order = cfg["sequence_packing"].get("microbatch_order")
            if microbatch_order is not None:
                self.sequence_packing_args["microbatch_order"] = microbatch_order

    def get_logprobs(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Run forward pass on teacher and return logprobs."""
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        mbs = micro_batch_size or self._micro_batch_size

        if self.use_sequence_packing:
            self.sequence_packing_args["max_tokens_per_microbatch"] = self.cfg[
                "sequence_packing"
            ]["logprob_mb_tokens"]
            sharded_data, unsorted_data_indices = data.shard_by_batch_size(
                dp_size,
                batch_size=None,
                sequence_packing_args=self.sequence_packing_args,
            )
        else:
            sharded_data = data.shard_by_batch_size(dp_size, batch_size=None)
            unsorted_data_indices = None

        futures = self.worker_group.run_all_workers_sharded_data(
            "get_logprobs",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=[
                "context_parallel",
                "tensor_parallel",
                "pipeline_parallel",
            ],
            output_is_replicated=[
                "context_parallel",
                "tensor_parallel",
                "pipeline_parallel",
            ],
            common_kwargs={"micro_batch_size": mbs},
        )
        logprobs = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures)
        )

        result = BatchedDataDict[ReferenceLogprobOutputSpec](
            reference_logprobs=logprobs["logprobs"].cpu()
        )

        # Undo packing reorder if needed — must use inverse permutation
        # (argsort), matching lm_policy.py's reorder_data.
        if unsorted_data_indices is not None:
            result.reorder_data(unsorted_data_indices)

        return result

    def get_logprobs_on_support(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[Any]:
        """Evaluate teacher logprobs on caller-selected vocabulary indices.

        ``data['topk_indices']`` must have shape ``[B, S, K]`` and align with
        ``input_ids``. This first implementation intentionally excludes packing
        and context parallelism; those layouts require explicit support-index
        packing/sharding rather than implicit reshaping.
        """
        if self.use_sequence_packing:
            raise NotImplementedError(
                "Top-k teacher evaluation does not yet support sequence packing."
            )
        if self.cfg["megatron_cfg"]["context_parallel_size"] != 1:
            raise NotImplementedError(
                "Top-k teacher evaluation does not yet support context parallelism."
            )
        if "topk_indices" not in data:
            raise ValueError("get_logprobs_on_support requires data['topk_indices'].")

        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        mbs = micro_batch_size or self._micro_batch_size
        sharded_data = data.shard_by_batch_size(dp_size, batch_size=None)
        futures = self.worker_group.run_all_workers_sharded_data(
            "get_logprobs_on_support",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=[
                "context_parallel",
                "tensor_parallel",
                "pipeline_parallel",
            ],
            output_is_replicated=[
                "context_parallel",
                "tensor_parallel",
                "pipeline_parallel",
            ],
            common_kwargs={"micro_batch_size": mbs},
        )
        worker_batches = self.worker_group.get_all_worker_results(futures)
        return BatchedDataDict(
            {
                "support_logprobs": torch.cat(
                    [batch["support_logprobs"] for batch in worker_batches], dim=0
                ).cpu()
            }
        )

    def get_topk_logprobs(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        k: int,
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[TopkLogprobsOutputSpec]:
        """Return target logprobs and teacher-selected support in one forward.

        With sequence packing, the heavy support tensors remain in per-sample
        Ray objects emitted by the Megatron workers. Only the small target
        logprobs are reconstructed here.
        """
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        mbs = micro_batch_size or self._micro_batch_size
        if self.use_sequence_packing:
            self.sequence_packing_args["max_tokens_per_microbatch"] = self.cfg[
                "sequence_packing"
            ]["logprob_mb_tokens"]
            sharded_data, unsorted_data_indices = data.shard_by_batch_size(
                dp_size,
                batch_size=None,
                sequence_packing_args=self.sequence_packing_args,
            )
        else:
            sharded_data = data.shard_by_batch_size(dp_size, batch_size=None)
            unsorted_data_indices = None
        futures = self.worker_group.run_all_workers_sharded_data(
            "get_topk_logits",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=[
                "context_parallel",
                "tensor_parallel",
                "pipeline_parallel",
            ],
            output_is_replicated=[
                "context_parallel",
                "tensor_parallel",
                "pipeline_parallel",
            ],
            common_kwargs={
                "k": k,
                "micro_batch_size": mbs,
                "return_logprobs": True,
            },
        )
        worker_batches = self.worker_group.get_all_worker_results(futures)
        if self.use_sequence_packing:
            if not all(
                isinstance(batch, dict) and "per_sample_refs" in batch
                for batch in worker_batches
            ):
                raise RuntimeError(
                    "Packed teacher top-k workers must return per-sample refs."
                )
            unpacked_seq_length = int(data["input_ids"].shape[1])
            for worker_idx, batch in enumerate(worker_batches):
                worker_seq_length = int(batch["unpacked_seq_length"])
                if worker_seq_length != unpacked_seq_length:
                    raise ValueError(
                        "Packed teacher workers disagree on unpacked sequence "
                        f"length: worker {worker_idx} returned "
                        f"{worker_seq_length}, expected {unpacked_seq_length}."
                    )
            flat_refs = [
                entry for batch in worker_batches for entry in batch["per_sample_refs"]
            ]
            if len(flat_refs) != data.size:
                raise ValueError(
                    "Packed teacher top-k returned an unexpected number of rows: "
                    f"got {len(flat_refs)}, expected {data.size}."
                )

            target_tensors = resolve_packed_field(flat_refs, "target_logprobs")
            target_dtype = target_tensors[0].dtype if target_tensors else torch.float32
            reference_logprobs = torch.zeros((len(flat_refs), unpacked_seq_length), dtype=target_dtype)
            for sample_idx, (entry, target_logprobs) in enumerate(zip(flat_refs, target_tensors)):
                seq_len = int(entry["seq_len"])
                if tuple(target_logprobs.shape) != (seq_len,):
                    raise ValueError(
                        f"Packed target_logprobs sample {sample_idx} has shape "
                        f"{tuple(target_logprobs.shape)}, expected {(seq_len,)}."
                    )
                if seq_len > max(unpacked_seq_length - 1, 0):
                    raise ValueError(
                        f"Packed target_logprobs sample {sample_idx} has "
                        f"seq_len={seq_len}, larger than the available next-token "
                        f"width {max(unpacked_seq_length - 1, 0)}."
                    )
                if seq_len:
                    reference_logprobs[sample_idx, 1 : 1 + seq_len].copy_(target_logprobs)
                entry.pop("target_logprobs", None)
                entry.pop("target_logprobs_ref", None)

            result = BatchedDataDict[TopkLogprobsOutputSpec](
                {
                    "reference_logprobs": reference_logprobs,
                    OPD_TEACHER_TOPK_PACKED_KEY: flat_refs,
                }
            )
        else:
            result = BatchedDataDict[TopkLogprobsOutputSpec](
                {
                    "reference_logprobs": torch.cat([batch["logprobs"] for batch in worker_batches], dim=0).cpu(),
                    "topk_indices": torch.cat([batch["topk_indices"] for batch in worker_batches], dim=0).cpu(),
                    "topk_logprobs": torch.cat([batch["topk_logprobs"] for batch in worker_batches], dim=0).cpu(),
                }
            )
        if unsorted_data_indices is not None:
            result.reorder_data(unsorted_data_indices)
        return result

    def shutdown(self) -> bool:
        """Shut down all workers and clean up resources."""
        try:
            return self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            print(f"Error during teacher worker group shutdown: {e}")
            return False

    def __del__(self) -> None:
        """Safety net for cleanup."""
        if hasattr(self, "worker_group"):
            self.shutdown()
