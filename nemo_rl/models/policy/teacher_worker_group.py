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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.opd import TeacherResourceConfig
from nemo_rl.distributed.batched_data_dict import (
    BatchedDataDict,
    SequencePackingArgs,
)
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.interfaces import GenerationDatumSpec
from nemo_rl.models.policy.interfaces import ReferenceLogprobOutputSpec


@dataclass
class TeacherConfig:
    """Resolved config for a single non-colocated teacher (built in-process)."""

    alias: str
    model_name: str  # checkpoint path
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    context_parallel_size: int
    expert_model_parallel_size: int
    num_nodes: int
    gpus_per_node: int
    precision: str
    micro_batch_size: int
    megatron_cfg_overrides: dict[str, Any]


def create_teacher_configs_from_opd_config(
    opd_cfg: dict[str, Any],
) -> list[TeacherConfig]:
    """Build per-teacher configs from on_policy_distillation config.

    Handles deduplication (multiple aliases sharing one checkpoint produce
    one TeacherConfig) and per-teacher overrides on top of defaults.
    """
    teacher_model_by_agent_name: dict[str, str] = dict(
        opd_cfg.get("teacher_model_by_agent_name", {})
    )
    non_coloc_cfg = dict(opd_cfg.get("non_colocated_teachers", {}))
    default_cfg = dict(non_coloc_cfg.get("default_teacher_cfg", {}))
    overrides = dict(non_coloc_cfg.get("teacher_overrides", {}))
    deduplicate = bool(opd_cfg.get("deduplicate_shared_teacher_checkpoints", True))

    configs: list[TeacherConfig] = []
    seen_models: set[str] = set()

    for alias, model_name in teacher_model_by_agent_name.items():
        if deduplicate and model_name in seen_models:
            continue
        seen_models.add(model_name)

        # defaults <- per-alias override, then validated/typed by the schema.
        merged = {**default_cfg, **dict(overrides.get(alias, {}))}
        res = TeacherResourceConfig(**merged)

        # Unknown top-level keys (extra="allow") fold into megatron_cfg_overrides;
        # explicit megatron_cfg_overrides take precedence.
        all_overrides = {**(res.model_extra or {}), **res.megatron_cfg_overrides}

        configs.append(
            TeacherConfig(
                alias=alias,
                model_name=model_name,
                tensor_model_parallel_size=res.tensor_model_parallel_size,
                pipeline_model_parallel_size=res.pipeline_model_parallel_size,
                context_parallel_size=res.context_parallel_size,
                expert_model_parallel_size=res.expert_model_parallel_size,
                num_nodes=res.num_nodes,
                gpus_per_node=res.gpus_per_node,
                precision=res.precision,
                micro_batch_size=res.micro_batch_size,
                megatron_cfg_overrides=all_overrides,
            )
        )

    return configs


class TeacherWorkerGroup:
    """Inference-only mcore worker group for a single teacher model.

    Unlike the training policy, this group:
    - Never initializes an optimizer
    - Never initializes a reference model
    - Loads the checkpoint once at startup
    - Only exposes get_logprobs()
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
        cfg["megatron_cfg"]["enabled"] = True
        cfg["megatron_cfg"]["tensor_model_parallel_size"] = (
            teacher_cfg.tensor_model_parallel_size
        )
        cfg["megatron_cfg"]["pipeline_model_parallel_size"] = (
            teacher_cfg.pipeline_model_parallel_size
        )
        cfg["megatron_cfg"]["context_parallel_size"] = teacher_cfg.context_parallel_size
        cfg["megatron_cfg"]["expert_model_parallel_size"] = (
            teacher_cfg.expert_model_parallel_size
        )

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

        tp = teacher_cfg.tensor_model_parallel_size
        pp = teacher_cfg.pipeline_model_parallel_size
        cp = teacher_cfg.context_parallel_size

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
