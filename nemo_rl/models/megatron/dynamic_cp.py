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
from megatron.core.transformer.moe.router import Router

from nemo_rl.distributed.dynamic_context_parallel import CPRankPlan, CPRankStep


_DYNAMIC_TP_CP_GROUPS: dict[int, Any] = {}
_ROUTER_CONFIG_BASELINES: dict[int, tuple[Any, Any]] = {}


@dataclass(frozen=True)
class RuntimeCPContext:
    """Attention group for a single forward/backward, separate from DDP groups."""

    size: int
    rank: int
    group: Any


def initialize_dynamic_cp_runtime(*, max_cp_size: int) -> None:
    """Initialize CP resources missing on CP=1 builds of the pinned MCore.

    This creates the same stream as TEDotProductAttention's CP constructor;
    it also creates the active TP*CP groups used by MoE routers.  Groups must
    be created eagerly and in the same order on every rank; creating one from
    a microbatch forward would deadlock as different lanes select different CP
    sizes.  Static DP, EP and optimizer groups are left unchanged.
    """
    # TE is an optional dependency outside the Megatron worker environment.
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    if TEDotProductAttention.cp_stream is None:
        TEDotProductAttention.cp_stream = torch.cuda.Stream()

    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    lane_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
    lane_ranks = torch.distributed.get_process_group_ranks(lane_group)
    base_lane_ranks = tuple(rank - tp_rank for rank in lane_ranks)
    if any(base % tp_size for base in base_lane_ranks):
        raise ValueError("Dynamic CP requires contiguous TP ranks")
    if max_cp_size > len(base_lane_ranks):
        raise ValueError("Dynamic CP max_size exceeds the available DP*CP lanes")

    _DYNAMIC_TP_CP_GROUPS.clear()
    _DYNAMIC_TP_CP_GROUPS[1] = parallel_state.get_tensor_model_parallel_group()
    cp_size = 2
    while cp_size <= max_cp_size:
        if len(base_lane_ranks) % cp_size:
            raise ValueError("Every dynamic CP size must divide the DP*CP lane domain")
        local_group = None
        for start in range(0, len(base_lane_ranks), cp_size):
            ranks = [
                base + offset
                for base in base_lane_ranks[start : start + cp_size]
                for offset in range(tp_size)
            ]
            if tp_size == 1:
                group = parallel_state.get_hybrid_data_context_parallel_groups(
                    group_size=cp_size
                )
            else:
                group = torch.distributed.new_group(ranks=ranks)
            if torch.distributed.get_rank() in ranks:
                local_group = group
        if local_group is None:
            raise ValueError("Rank was not assigned to a dynamic TP*CP group")
        _DYNAMIC_TP_CP_GROUPS[cp_size] = local_group
        cp_size *= 2


def _is_mamba_mixer(module: torch.nn.Module) -> bool:
    cp = getattr(module, "cp", None)
    return cp is not None and all(
        hasattr(cp, name)
        for name in (
            "d_inner_local_tp",
            "nheads_local_tp",
            "ngroups_local_tp",
            "conv1d_weight_cp1",
            "conv1d_bias_cp1",
            "dt_bias_cp1",
            "A_log_cp1",
            "D_cp1",
        )
    )


def _is_gated_delta_net(module: torch.nn.Module) -> bool:
    return (
        ".ssm.gated_delta_net" in type(module).__module__
        and hasattr(module, "cp_size")
        and hasattr(module, "pg_collection")
    )


def _rebuild_mamba_cp(module: torch.nn.Module, group: Any) -> None:
    """Rebuild Mamba's cached CP helper for the active microbatch size."""
    cp = module.cp
    module.cp = type(cp)(
        cp_group=group,
        d_inner_local_tp=cp.d_inner_local_tp,
        nheads_local_tp=cp.nheads_local_tp,
        ngroups_local_tp=cp.ngroups_local_tp,
        d_state=cp.d_state,
        conv1d_weight_cp1=cp.conv1d_weight_cp1,
        conv1d_bias_cp1=cp.conv1d_bias_cp1,
        conv1d_padding=cp.conv1d_padding,
        dt_bias_cp1=cp.dt_bias_cp1,
        A_log_cp1=cp.A_log_cp1,
        D_cp1=cp.D_cp1,
        D_has_hdim=cp.D_has_hdim,
    )


@contextmanager
def preserve_attention_cp_groups(model: torch.nn.Module) -> Iterator[None]:
    """Isolate runtime attention, SSM and router state from fixed groups.

    Keep the active group through backward recomputation; restore it after the
    complete no-pipeline schedule. TP, DP and optimizer groups are unchanged.
    """
    saved_collections = [
        (module, module.pg_collection)
        for module in model.modules()
        if isinstance(module, Attention)
        or _is_mamba_mixer(module)
        or _is_gated_delta_net(module)
    ]
    saved_mamba = [
        (module, module.cp) for module in model.modules() if _is_mamba_mixer(module)
    ]
    saved_gdn = [
        (module, module.cp_size)
        for module in model.modules()
        if _is_gated_delta_net(module)
    ]
    saved_routers = [
        (module, module.tp_cp_group)
        for module in model.modules()
        if isinstance(module, Router)
    ]
    router_configs: dict[int, Any] = {}
    for module, collection in saved_collections:
        module.pg_collection = copy(collection)
    for module, _ in saved_routers:
        config = module.config
        router_configs[id(config)] = config
        _ROUTER_CONFIG_BASELINES[id(config)] = (
            config.moe_aux_loss_coeff,
            config.moe_z_loss_coeff,
        )
    try:
        yield
    finally:
        for module, collection in saved_collections:
            module.pg_collection = collection
        for module, cp in saved_mamba:
            module.cp = cp
        for module, cp_size in saved_gdn:
            module.cp_size = cp_size
        for module, group in saved_routers:
            module.tp_cp_group = group
        for config_id, config in router_configs.items():
            aux_coeff, z_coeff = _ROUTER_CONFIG_BASELINES.pop(config_id)
            config.moe_aux_loss_coeff = aux_coeff
            config.moe_z_loss_coeff = z_coeff


def _bind_router_config(router: Router, *, padding_only: bool) -> None:
    baseline = _ROUTER_CONFIG_BASELINES.get(id(router.config))
    if baseline is None:
        raise RuntimeError("Dynamic CP router binding escaped its preservation context")
    aux_coeff, z_coeff = baseline
    if padding_only:
        if isinstance(aux_coeff, tuple):
            aux_coeff = tuple(0.0 for _ in aux_coeff)
        elif isinstance(aux_coeff, list):
            aux_coeff = [0.0 for _ in aux_coeff]
        else:
            aux_coeff = 0.0
        z_coeff = None
    router.config.moe_aux_loss_coeff = aux_coeff
    router.config.moe_z_loss_coeff = z_coeff


def bind_attention_cp_group(model: torch.nn.Module, packed_seq_params: Any) -> Any:
    """Bind attention, MoE router and SSM modules to the active CP task.

    The pinned MCore forwards packed.cp_group to TE but its RoPE still reads
    ``Attention.pg_collection.cp``. Mamba and GatedDeltaNet also cache their CP
    helper/group, while MoE routers cache TP*CP for aux-loss token reductions.
    CP=1 needs a real singleton because None means static fallback in MCore.
    Pass a copy to the model so loss/gather metadata retains CP=1's None group.
    """
    context = runtime_cp_from_packed(packed_seq_params)
    group = context.group
    if context.size == 1:
        group = parallel_state.get_pipeline_model_parallel_group()
        if group.size() != 1:
            raise ValueError("Dynamic CP attention requires PP=1")
    tp_cp_group = _DYNAMIC_TP_CP_GROUPS.get(context.size)
    if tp_cp_group is None:
        raise ValueError(f"No dynamic TP*CP group was initialized for CP={context.size}")
    expected_tp_cp_size = (
        context.size * parallel_state.get_tensor_model_parallel_world_size()
    )
    if tp_cp_group.size() != expected_tp_cp_size:
        raise ValueError("Dynamic MoE TP*CP group has the wrong size")
    padding_only = bool(
        getattr(packed_seq_params, "dynamic_cp_padding_only", False)
    )
    for module in model.modules():
        if isinstance(module, Attention):
            module.pg_collection.cp = group
        if isinstance(module, Router):
            module.tp_cp_group = tp_cp_group
            _bind_router_config(module, padding_only=padding_only)
        if _is_mamba_mixer(module):
            module.pg_collection.cp = group
            _rebuild_mamba_cp(module, group)
        elif _is_gated_delta_net(module):
            module.pg_collection.cp = group
            module.cp_size = context.size
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
                create_packed_seq_padding_mask=True,
            )
            padding_only = not assignment.sample_indices
            inputs.packed_seq_params.dynamic_cp_padding_only = padding_only
            if padding_only:
                # MCore uses True to mean "padding".  Excluding every physical
                # placeholder token keeps expert-bias counters clean; router aux
                # coefficients are disabled for this forward to avoid a 0/0 loss.
                inputs.padding_mask = torch.ones_like(
                    inputs.input_ids_cp_sharded, dtype=torch.bool
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
