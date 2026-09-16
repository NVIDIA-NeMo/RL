# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MCore-specific runtime binding for driver-planned context parallelism."""

from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from megatron.core import parallel_state
from megatron.core.transformer.attention import Attention

from nemo_rl.distributed.dynamic_context_parallel import CPRankPlan, CPRankStep


@dataclass(frozen=True)
class RuntimeCPContext:
    """Attention group for a single forward/backward, separate from DDP groups."""

    size: int
    rank: int
    group: Any


def initialize_dynamic_cp_runtime() -> None:
    """Initialize CP resources missing on CP=1 builds of the pinned MCore.

    This creates the same stream as TEDotProductAttention's CP constructor;
    it does not replace methods, edit dependency files, or change static groups.
    Newer MCore creates this lazily, so the initialization is idempotent.
    """
    # TE is an optional dependency outside the Megatron worker environment.
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    if TEDotProductAttention.cp_stream is None:
        TEDotProductAttention.cp_stream = torch.cuda.Stream()


@contextmanager
def preserve_attention_cp_groups(model: torch.nn.Module) -> Iterator[None]:
    """Isolate attention's runtime groups from shared model/DDP collections.

    Keep the active group through backward recomputation; restore it after the
    complete no-pipeline schedule. TP, DP and optimizer groups are unchanged.
    """
    saved = [
        (module, module.pg_collection)
        for module in model.modules()
        if isinstance(module, Attention)
    ]
    for module, collection in saved:
        module.pg_collection = copy(collection)
    try:
        yield
    finally:
        for module, collection in saved:
            module.pg_collection = collection


def bind_attention_cp_group(model: torch.nn.Module, packed_seq_params: Any) -> Any:
    """Bind RoPE as well as TE attention to the microbatch's active group.

    The pinned MCore forwards packed.cp_group to TE but its RoPE still reads
    Attention.pg_collection.cp. CP=1 needs a real singleton here because None
    means fall back to static CP in MCore's RoPE API. PP=1 supplies that singleton.
    Pass a copy to the model so loss/gather metadata retains CP=1's None group.
    """
    context = runtime_cp_from_packed(packed_seq_params)
    group = context.group
    if context.size == 1:
        group = parallel_state.get_pipeline_model_parallel_group()
        if group.size() != 1:
            raise ValueError("Dynamic CP attention requires PP=1")
    for module in model.modules():
        if isinstance(module, Attention):
            module.pg_collection.cp = group
    model_packed = copy(packed_seq_params)
    model_packed.cp_group = group
    return model_packed


def planned_microbatches(
    data: Any, plan: CPRankPlan, step: CPRankStep, straggler_timer: Any
) -> Iterator[Any]:
    """Yield the lane's uneven task list with explicit group boundaries."""
    # Avoid a cycle: data.py dispatches to this iterator.
    from nemo_rl.models.megatron.data import ProcessedMicrobatch, process_microbatch

    domain = parallel_state.get_data_parallel_group(with_context_parallel=True)
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    expected = [rank + tp_rank for rank in plan.lane_ranks]
    if (
        torch.distributed.get_process_group_ranks(domain) != expected
        or domain.rank() != plan.lane
    ):
        raise ValueError("Ray's DP*CP lane map disagrees with initialized MCore groups")
    for group_index, rank_group in enumerate(step.groups):
        if not rank_group.assignments:
            raise ValueError("Every CP synchronization group needs one local task")
        for task_index, assignment in enumerate(rank_group.assignments):
            size = assignment.cp_size
            group = (
                parallel_state.get_hybrid_data_context_parallel_groups(group_size=size)
                if size > 1
                else None
            )
            rank = plan.lane - assignment.lane_start
            if group is not None:
                members = expected[assignment.lane_start : assignment.lane_start + size]
                if (
                    group.size() != size
                    or group.rank() != rank
                    or torch.distributed.get_process_group_ranks(group) != members
                ):
                    raise ValueError(
                        "Active CP group disagrees with the driver's assignment"
                    )
            expert_group = parallel_state.get_expert_model_parallel_group()
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            task_ranks = {
                base + offset
                for base in plan.lane_ranks[
                    assignment.lane_start : assignment.lane_start + size
                ]
                for offset in range(tp_size)
            }
            if not set(
                torch.distributed.get_process_group_ranks(expert_group)
            ).issubset(task_ranks):
                raise ValueError(
                    "Expert communication group crosses dynamic CP task boundaries"
                )
            context = RuntimeCPContext(size=size, rank=rank, group=group)
            if assignment.sample_indices:
                batch = data.select_indices(list(assignment.sample_indices)).to("cuda")
            else:
                batch = data.select_indices([0]).to("cuda")
                # A real attention invocation keeps collective counts aligned, but
                # none of this placeholder's targets or metrics belong to the batch.
                for key, value in list(batch.items()):
                    if isinstance(value, torch.Tensor):
                        batch[key] = torch.zeros_like(value)
                batch["input_lengths"].fill_(2)
            inputs = process_microbatch(
                batch,
                seq_length_key="input_lengths",
                pack_sequences=True,
                pad_individual_seqs_to_multiple_of=assignment.pad_multiple,
                straggler_timer=straggler_timer,
                cp_context=context,
            )
            if inputs.input_ids_cp_sharded.shape[1] * size != assignment.padded_tokens:
                raise ValueError(
                    "Packed worker token count disagrees with the driver's plan"
                )
            payload_kind = "data" if assignment.sample_indices else "padding"
            # The generator stays paused inside this range while MCore consumes
            # the microbatch, making uneven task counts visible in Nsight.
            with torch.cuda.nvtx.range(
                f"dynamic_cp/group_{group_index}/task_{task_index}/cp_{size}/"
                f"lane_{plan.lane}/{payload_kind}"
            ):
                yield ProcessedMicrobatch(
                    data_dict=batch,
                    dynamic_cp_group_start=task_index == 0,
                    dynamic_cp_group_index=group_index,
                    dynamic_cp_task_index=task_index,
                    **vars(inputs),
                )


def runtime_cp_from_packed(packed_seq_params: Any) -> RuntimeCPContext:
    """Resolve explicit runtime metadata, falling back only for static CP."""
    if packed_seq_params is not None and packed_seq_params.local_cp_size is not None:
        size = packed_seq_params.local_cp_size
        group = packed_seq_params.cp_group
        if size == 1:
            if group is not None:
                raise ValueError("CP=1 must not carry an attention communication group")
            return RuntimeCPContext(size=1, rank=0, group=None)
        if group is None or group.size() != size:
            raise ValueError("Packed CP size and process group disagree")
        return RuntimeCPContext(size=size, rank=group.rank(), group=group)
    return RuntimeCPContext(
        size=parallel_state.get_context_parallel_world_size(),
        rank=parallel_state.get_context_parallel_rank(),
        group=parallel_state.get_context_parallel_group(),
    )
