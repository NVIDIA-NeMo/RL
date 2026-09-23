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
from typing import Any, Callable, Iterator, Optional

import torch
from megatron.core import parallel_state
from megatron.core.transformer.moe.router import Router

from nemo_rl.distributed.dynamic_context_parallel import CPRankPlan, CPRankStep


_DYNAMIC_TP_CP_GROUPS: dict[int, Any] = {}
_ROUTER_CONFIG_BASELINES: dict[int, tuple[Any, Any]] = {}
_DYNAMIC_MTP_METRICS: dict[str, torch.Tensor] = {}
_ACTIVE_BIND_TARGETS: dict[int, "_BindTargets"] = {}
_ACTIVE_BIND_SIGNATURES: dict[int, tuple[int, int, int, bool]] = {}


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


def _is_gated_delta_product(module: torch.nn.Module) -> bool:
    """Return whether ``module`` owns a headwise GDP CP helper."""
    cp = getattr(module, "cp", None)
    return cp is not None and all(
        hasattr(cp, name)
        for name in (
            "d_inner_local_tp",
            "nheads_local_tp",
            "ngroups_local_tp",
            "num_householder",
            "headdim",
            "conv1d_cp1",
            "dt_bias_cp1",
            "A_log_cp1",
            "D_cp1",
        )
    )


def _uses_direct_cp_group(module: torch.nn.Module) -> bool:
    """Return whether pinned MCore caches CP outside ``pg_collection``."""
    module_name = type(module).__module__
    return module_name.startswith(
        "megatron.core.transformer.multi_token_prediction"
    ) or module_name.startswith("megatron.core.models.hybrid.hybrid_block")


def _is_hybrid_stack(module: torch.nn.Module) -> bool:
    return (
        type(module).__module__.startswith("megatron.core.models.hybrid.hybrid_block")
        and hasattr(module, "layer_config_list")
        and hasattr(module, "_cp_layout_manager")
    )


@dataclass(frozen=True)
class _BindTargets:
    """Stable module classifications reused by every task in one model call."""

    modules: tuple[torch.nn.Module, ...]
    pg_collections: tuple[torch.nn.Module, ...]
    direct_groups: tuple[torch.nn.Module, ...]
    hybrid_stacks: tuple[torch.nn.Module, ...]
    routers: tuple[Router, ...]
    mamba_mixers: tuple[torch.nn.Module, ...]
    gated_delta_products: tuple[torch.nn.Module, ...]
    gated_delta_nets: tuple[torch.nn.Module, ...]


def _classify_bind_targets(model: torch.nn.Module) -> _BindTargets:
    modules = tuple(model.modules())
    return _BindTargets(
        modules=modules,
        pg_collections=tuple(
            module
            for module in modules
            if getattr(module, "pg_collection", None) is not None
            and hasattr(module.pg_collection, "cp")
        ),
        direct_groups=tuple(
            module
            for module in modules
            if _uses_direct_cp_group(module) and hasattr(module, "cp_group")
        ),
        hybrid_stacks=tuple(module for module in modules if _is_hybrid_stack(module)),
        routers=tuple(module for module in modules if isinstance(module, Router)),
        mamba_mixers=tuple(module for module in modules if _is_mamba_mixer(module)),
        gated_delta_products=tuple(
            module for module in modules if _is_gated_delta_product(module)
        ),
        gated_delta_nets=tuple(
            module for module in modules if _is_gated_delta_net(module)
        ),
    )


def _bind_targets(model: torch.nn.Module) -> _BindTargets:
    """Reuse the outer schedule's classification, with a safe direct-call fallback."""
    return _ACTIVE_BIND_TARGETS.get(id(model)) or _classify_bind_targets(model)


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


def _rebuild_gdp_cp(module: torch.nn.Module, group: Any) -> None:
    """Rebuild GDP's cached headwise CP helper for one runtime task."""
    cp = module.cp
    module.cp = type(cp)(
        cp_group=group,
        d_inner_local_tp=cp.d_inner_local_tp,
        nheads_local_tp=cp.nheads_local_tp,
        ngroups_local_tp=cp.ngroups_local_tp,
        d_state=cp.d_state,
        num_householder=cp.num_householder,
        headdim=cp.headdim,
        conv1d_cp1=cp.conv1d_cp1,
        dt_bias_cp1=cp.dt_bias_cp1,
        A_log_cp1=cp.A_log_cp1,
        D_cp1=cp.D_cp1,
        D_has_hdim=cp.D_has_hdim,
        sequence_is_contiguous=cp.sequence_is_contiguous,
    )
    module.d_inner_local_cp = module.cp.d_inner_local_tpcp
    module.nheads_local_cp = module.cp.nheads_local_tpcp
    module.ngroups_local_cp = module.cp.ngroups_local_tpcp


def _bind_hybrid_stack_layout(
    module: torch.nn.Module, *, group: Any, tp_cp_group: Any
) -> None:
    """Build the CP layout converter omitted by a static CP=1 construction."""
    from megatron.core.context_parallel import ContextParallelLayoutManager
    from megatron.core.models.hybrid.layers import utils as layer_utils

    module.cp_group = group
    module.tp_cp_group = tp_cp_group
    module._has_linear_layer_with_chunkwise_cp = any(
        type(layer_config) is layer_utils.MambaLayerConfig
        and layer_config.linear_cp_mode == "chunkwise"
        for layer_config in module.layer_config_list
    )
    if group.size() == 1:
        module._cp_layout_manager = None
        return

    layer_layouts = tuple(
        (
            layer_config.attention_cp_layout
            if type(layer_config) in layer_utils.Symbols.ATTENTION_LAYER_CONFIGS
            else layer_config.linear_cp_layout
        )
        for layer_config in module.layer_config_list
    )
    boundary_layout = (
        module.config.attention_cp_layout
        if getattr(module, "is_mtp_layer", False)
        else module.config.linear_cp_layout
    )
    module._cp_layout_manager = ContextParallelLayoutManager(
        layer_layouts=layer_layouts,
        boundary_layout=boundary_layout,
        sequence_parallel=module.config.sequence_parallel,
        cp_group=group,
        tp_group=module.tp_group,
        tp_cp_group=tp_cp_group,
    )


def validate_dynamic_cp_model(model: torch.nn.Module) -> None:
    """Reject MCore model internals that cannot be safely rebound at runtime."""
    modules = list(model.modules())
    for module in modules:
        module_name = type(module).__module__
        class_name = type(module).__name__
        if (
            module_name.startswith("megatron.core.models.hybrid.hybrid_model")
            and getattr(module, "mtp", None) is not None
            and getattr(module.config, "linear_cp_layout", None)
            != getattr(module.config, "attention_cp_layout", None)
        ):
            raise ValueError(
                "Dynamic CP Hybrid MTP requires matching linear_cp_layout and "
                "attention_cp_layout because NeMo-RL supplies one packed layout"
            )
        if class_name in {"MLASelfAttention", "AbsorbedMLASelfAttention"} or (
            "multi_latent_attention" in module_name
            or "experimental_attention_variant.absorbed_mla" in module_name
        ):
            raise ValueError(
                "Dynamic CP with MLA requires the unmerged MCore runtime-group "
                "support; this pinned MCore still rejects dynamic packed metadata"
            )
        if (
            hasattr(module, "config")
            and getattr(module.config, "linear_cp_mode", None) == "chunkwise"
            and (
                _is_mamba_mixer(module)
                or _is_gated_delta_product(module)
                or ".ssm." in module_name
            )
        ):
            raise ValueError(
                "Dynamic CP does not support chunkwise linear CP with the pinned "
                "MCore; use headwise linear_cp_mode"
            )


def _save_dynamic_mtp_metrics(
    *,
    loss_sum: torch.Tensor,
    num_tokens: torch.Tensor,
    correct: torch.Tensor,
    total: torch.Tensor,
    layer_number: int,
    num_layers: int,
) -> None:
    """Accumulate MTP numerators/counts without a stale fixed CP average."""
    values = {
        "loss_sums": loss_sum,
        "loss_token_counts": num_tokens,
        "correct_values": correct,
        "total_values": total,
    }
    for name, value in values.items():
        if name not in _DYNAMIC_MTP_METRICS:
            _DYNAMIC_MTP_METRICS[name] = torch.zeros(
                num_layers, dtype=torch.float32, device=value.device
            )
        _DYNAMIC_MTP_METRICS[name][layer_number] += value.detach().float()


def get_dynamic_mtp_metrics(
    *, parallel_group: torch.distributed.ProcessGroup
) -> dict[str, float]:
    """Reduce token-weighted MTP metrics over the fixed DP*CP lane group."""
    # This branch is collective-safe: training gives every lane at least one
    # real or placeholder task, and even a zero-token placeholder records all
    # four tensors. Evaluation records MTP metrics on no lane, so all lanes exit.
    if "loss_sums" not in _DYNAMIC_MTP_METRICS:
        return {}
    try:
        totals = torch.stack(
            tuple(
                _DYNAMIC_MTP_METRICS[name]
                for name in (
                    "loss_sums",
                    "loss_token_counts",
                    "correct_values",
                    "total_values",
                )
            )
        )
        if parallel_group.size() > 1:
            torch.distributed.all_reduce(
                totals, op=torch.distributed.ReduceOp.SUM, group=parallel_group
            )
        losses = totals[0] / totals[1].clamp(min=1)
        acceptance = totals[2] / totals[3].clamp(min=1) * 100.0
        loss_values, acceptance_values = torch.stack((losses, acceptance)).tolist()
        metrics: dict[str, float] = {}
        for index, (loss, rate) in enumerate(
            zip(loss_values, acceptance_values, strict=True)
        ):
            metrics[f"mtp_{index + 1}_loss"] = float(loss)
            metrics[f"mtp_{index + 1}_acceptance_rate"] = float(rate)
        return metrics
    finally:
        _DYNAMIC_MTP_METRICS.clear()


def _runtime_mtp_token_counts(
    original_num_tokens: torch.Tensor,
    mtp_num_tokens: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return task-wide main/MTP counts, including empty CP shards."""
    counts = torch.stack((original_num_tokens, mtp_num_tokens))
    if cp_group is not None and cp_group.size() > 1:
        torch.distributed.all_reduce(
            counts, op=torch.distributed.ReduceOp.SUM, group=cp_group
        )
    return counts.unbind()


def _dynamic_process_mtp_loss(
    hidden_states: torch.Tensor,
    labels: Optional[torch.Tensor],
    loss_mask: Optional[torch.Tensor],
    output_layer: Callable[..., Any],
    output_weight: Optional[torch.Tensor],
    runtime_gather_output: Optional[bool],
    is_training: bool,
    compute_language_model_loss: Callable[..., torch.Tensor],
    config: Any,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    packed_seq_params: Optional[Any] = None,
    scale_logits_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    input_ids: Optional[torch.Tensor] = None,
    mtp_input_mask: Optional[torch.Tensor] = None,
    metric_avg_group: Optional[torch.distributed.ProcessGroup] = None,
    main_hidden_states: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pinned MCore MTP loss with runtime-CP normalization and raw metrics."""
    del metric_avg_group  # Dynamic metrics reduce once over the fixed lane group.
    from megatron.core.transformer.multi_token_prediction import (
        MTPLossAutoScaler,
        _compute_mtp_acceptance_counts,
        roll_tensor,
    )

    hidden_states_list = torch.chunk(hidden_states, 1 + config.mtp_num_layers, dim=0)
    hidden_states = (
        hidden_states_list[0] if main_hidden_states is None else main_hidden_states
    )

    derived_labels_from_input_ids = False
    if labels is None:
        if input_ids is None:
            return hidden_states
        labels, _ = roll_tensor(
            input_ids,
            shifts=-1,
            dims=-1,
            cp_group=cp_group,
            packed_seq_params=packed_seq_params,
            return_sum=False,
        )
        derived_labels_from_input_ids = True

    if config.mtp_detach_heads:
        output_weight = (
            output_layer.weight.detach()
            if output_weight is None
            else output_weight.detach()
        )

    mtp_labels = labels.clone()
    if loss_mask is None:
        loss_mask = torch.ones_like(mtp_labels)
    if derived_labels_from_input_ids:
        loss_mask, _ = roll_tensor(
            loss_mask,
            shifts=-1,
            dims=-1,
            cp_group=cp_group,
            packed_seq_params=packed_seq_params,
            return_sum=False,
        )

    original_num_tokens = loss_mask.sum()
    cumulative_mtp_input_mask = None
    rolled_num_tokens = original_num_tokens
    if mtp_input_mask is not None:
        if mtp_input_mask.shape != loss_mask.shape:
            raise ValueError(
                f"mtp_input_mask shape {mtp_input_mask.shape} must match "
                f"loss_mask shape {loss_mask.shape}"
            )
        mtp_input_mask = mtp_input_mask.to(dtype=torch.bool)

    for mtp_layer_number in range(config.mtp_num_layers):
        mtp_logits, _ = output_layer(
            hidden_states_list[mtp_layer_number + 1],
            weight=output_weight,
            runtime_gather_output=runtime_gather_output,
        )
        if scale_logits_fn is not None:
            mtp_logits = scale_logits_fn(mtp_logits)
        mtp_labels, _ = roll_tensor(
            mtp_labels,
            shifts=-1,
            dims=-1,
            cp_group=cp_group,
            packed_seq_params=packed_seq_params,
            return_sum=False,
        )

        if mtp_input_mask is not None:
            mask_metadata = torch.cat(
                (loss_mask, mtp_input_mask.to(dtype=loss_mask.dtype)), dim=0
            )
            mask_metadata, _ = roll_tensor(
                mask_metadata,
                shifts=-1,
                dims=-1,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
                return_sum=False,
            )
            loss_mask, mtp_input_mask = mask_metadata.chunk(2, dim=0)
            mtp_input_mask = mtp_input_mask.to(dtype=torch.bool)
            cumulative_mtp_input_mask = (
                mtp_input_mask
                if cumulative_mtp_input_mask is None
                else cumulative_mtp_input_mask & mtp_input_mask
            )
            layer_loss_mask = loss_mask * cumulative_mtp_input_mask
            num_tokens = layer_loss_mask.sum()
        else:
            loss_mask, rolled_num_tokens = roll_tensor(
                loss_mask,
                shifts=-1,
                dims=-1,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
            )
            layer_loss_mask = loss_mask
            num_tokens = rolled_num_tokens

        mtp_loss = layer_loss_mask * compute_language_model_loss(mtp_labels, mtp_logits)
        if is_training:
            correct, total = _compute_mtp_acceptance_counts(
                mtp_logits,
                mtp_labels,
                layer_loss_mask,
                output_layer,
                runtime_gather_output,
                tp_group,
            )
            _save_dynamic_mtp_metrics(
                loss_sum=mtp_loss.sum(),
                num_tokens=num_tokens,
                correct=correct,
                total=total,
                layer_number=mtp_layer_number,
                num_layers=config.mtp_num_layers,
            )

        mtp_loss_scale = config.mtp_loss_scaling_factor / config.mtp_num_layers
        if config.calculate_per_token_loss:
            main_num_tokens, task_mtp_num_tokens = _runtime_mtp_token_counts(
                original_num_tokens, num_tokens, cp_group
            )
            mtp_loss = (
                mtp_loss_scale
                * mtp_loss
                * (main_num_tokens / task_mtp_num_tokens.clamp(min=1))
            )
        else:
            mtp_loss = mtp_loss_scale * mtp_loss / num_tokens.clamp(min=1)
        hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss)

    return hidden_states


@contextmanager
def _patch_mtp_loss_for_dynamic_cp(enabled: bool) -> Iterator[None]:
    """Temporarily route GPT/HybridModel MTP through the NeMo-side fix.

    This module-level patch relies on the current worker contract: one training
    model executes at a time and generation does not share the worker process.
    """
    if not enabled:
        yield
        return

    from megatron.core.models.gpt import gpt_model

    targets = [gpt_model]
    try:
        from megatron.core.models.hybrid import hybrid_model

        targets.append(hybrid_model)
    except ImportError:
        pass
    originals = [(target, target.process_mtp_loss) for target in targets]
    for target, _ in originals:
        target.process_mtp_loss = _dynamic_process_mtp_loss
    try:
        yield
    finally:
        for target, original in originals:
            target.process_mtp_loss = original


def _dynamic_attach_and_log_load_balancing_loss(
    self: Router,
    activation: torch.Tensor,
    aux_loss_coeff: float,
    aux_loss: torch.Tensor,
    aux_loss_name: str,
    reduce_group: torch.distributed.ProcessGroup,
    needs_dp_avg: bool = True,
    valid_token_count: int | torch.Tensor | None = None,
) -> torch.Tensor:
    """Pinned router attachment with exact runtime-group token scaling.

    MCore's pinned implementation multiplies by ``local_tokens *
    tp_cp_group.size()``.  That is only equal to the task token count for
    equally populated fixed shards.  The task setup (or the MTP mask hook) has
    already reduced the current mask over the runtime TP*CP group, so attach
    that exact count. Logging intentionally retains the unscaled aux value.
    """
    from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    if (
        self.is_mtp_layer
        and self.config.mtp_use_repeated_layer
        and self.config.mtp_num_layers is not None
    ):
        aux_loss = aux_loss / self.config.mtp_num_layers

    num_layers = self.config.num_layers
    if self.config.mtp_num_layers is not None:
        num_layers += self.config.mtp_num_layers
    layer_number = (
        self.layer_number + self.config.num_layers
        if self.is_mtp_layer
        else self.layer_number
    )
    get_moe_metrics_tracker().record(
        aux_loss_name,
        aux_loss / aux_loss_coeff,
        layer_number,
        num_layers,
        reduce_group=reduce_group,
        needs_dp_avg=needs_dp_avg,
    )

    if self.calculate_per_token_loss:
        task_tokens = getattr(self, "_nemo_dynamic_aux_scale_tokens", None)
        if task_tokens is None:
            raise RuntimeError(
                "Dynamic CP MoE token scaling was not prepared before router forward"
            )
        return MoEAuxLossAutoScaler.apply(activation, aux_loss * task_tokens)
    return MoEAuxLossAutoScaler.apply(activation, aux_loss)


@contextmanager
def _patch_hybrid_mtp_padding_masks(
    model: torch.nn.Module, modules: tuple[torch.nn.Module, ...] | None = None
) -> Iterator[None]:
    """Carry correct router padding semantics through every dynamic MTP block.

    The pinned HybridModel accepts ``padding_mask`` and sends it through the
    backbone, while its nested MTP call accidentally omits the same keyword.
    A HybridModel pre-hook captures that missing value and an MTP pre-hook
    supplies it without changing static execution or the MCore submodule.

    MCore's MTP roll helper fills newly exposed positions with false.  That is
    correct for a validity mask, but ``padding_mask`` uses the opposite meaning
    (true means padding).  Transport validity through MTP and invert it only at
    descendant MoE routers.  This also guarantees that a padding-only Dynamic
    CP call stays empty at every MTP depth.

    Dynamic CP supports headwise linear CP only, where the hybrid backbone and
    MTP boundary both use the attention (zigzag) token layout.  Consequently
    the mask received by HybridModel already has the ordering needed by MTP.
    """
    padding_masks: dict[int, torch.Tensor | None] = {}
    mtp_token_counts: dict[int, tuple[torch.Tensor, Any, torch.Tensor]] = {}
    active_task_marker: object | None = None
    handles: list[Any] = []
    modules = modules or tuple(model.modules())
    mtp_blocks: list[torch.nn.Module] = []

    def capture_padding_mask(
        module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        del args
        padding_masks[id(module.mtp)] = kwargs.get("padding_mask")

    def prepare_mtp_validity_mask(
        module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        padding_mask = kwargs.get("padding_mask")
        if padding_mask is None and len(args) > 4:
            padding_mask = args[4]
        if padding_mask is None:
            padding_mask = padding_masks.get(id(module))
        if padding_mask is not None:
            validity_mask = ~padding_mask.to(dtype=torch.bool)
            if len(args) > 4:
                args = (*args[:4], validity_mask, *args[5:])
            else:
                kwargs["padding_mask"] = validity_mask
        return args, kwargs

    for module in modules:
        if not type(module).__module__.startswith(
            "megatron.core.models.hybrid.hybrid_model"
        ):
            continue
        mtp = getattr(module, "mtp", None)
        if mtp is None:
            continue
        handles.append(
            module.register_forward_pre_hook(capture_padding_mask, with_kwargs=True)
        )
        mtp_blocks.append(mtp)

    for module in modules:
        if (
            type(module).__module__.startswith(
                "megatron.core.transformer.multi_token_prediction"
            )
            and type(module).__name__ == "MultiTokenPredictionBlock"
            and module not in mtp_blocks
        ):
            mtp_blocks.append(module)

    mtp_router_ids = {
        id(nested)
        for mtp in mtp_blocks
        for nested in mtp.modules()
        if isinstance(nested, Router)
    }

    def prepare_router_padding_mask(
        module: Router, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        nonlocal active_task_marker
        padding_mask = kwargs.get("padding_mask")
        if padding_mask is None and len(args) > 1:
            padding_mask = args[1]
        validity_mask = padding_mask
        if padding_mask is not None:
            padding_mask = ~padding_mask.to(dtype=torch.bool)
            if len(args) > 1:
                args = (args[0], padding_mask, *args[2:])
            else:
                kwargs["padding_mask"] = padding_mask

        if (
            module.training
            and torch.is_grad_enabled()
            and getattr(module, "calculate_per_token_loss", False)
            and _has_positive_coefficient(module.config.moe_aux_loss_coeff)
        ):
            if validity_mask is None:
                raise ValueError(
                    "Dynamic CP MoE aux loss requires a packed padding mask"
                )
            task_marker = getattr(module, "_nemo_dynamic_moe_task_marker", None)
            if task_marker is None:
                raise RuntimeError(
                    "Dynamic CP MoE token scaling was not configured for this task"
                )
            if task_marker is not active_task_marker:
                mtp_token_counts.clear()
                active_task_marker = task_marker

            cache_key = id(validity_mask)
            cached = mtp_token_counts.get(cache_key)
            if (
                cached is not None
                and cached[0] is validity_mask
                and cached[1] is module.tp_cp_group
            ):
                group_tokens = cached[2]
            else:
                group_tokens = validity_mask.sum().detach()
                if module.tp_cp_group.size() > 1:
                    torch.distributed.all_reduce(
                        group_tokens,
                        op=torch.distributed.ReduceOp.SUM,
                        group=module.tp_cp_group,
                    )
                mtp_token_counts[cache_key] = (
                    validity_mask,
                    module.tp_cp_group,
                    group_tokens,
                )
            module._nemo_dynamic_aux_scale_tokens = group_tokens
        return args, kwargs

    patched_classes: list[tuple[type[Any], bool, Any]] = []
    # This class-level patch has the same single-model worker assumption as the
    # MTP patch above; the preservation context always restores it in finally.
    for router_class in {
        type(module) for module in modules if isinstance(module, Router)
    }:
        if not hasattr(router_class, "attach_and_log_load_balancing_loss"):
            continue
        had_direct_method = (
            "attach_and_log_load_balancing_loss" in router_class.__dict__
        )
        original_method = router_class.__dict__.get(
            "attach_and_log_load_balancing_loss"
        )
        patched_classes.append((router_class, had_direct_method, original_method))
        router_class.attach_and_log_load_balancing_loss = (
            _dynamic_attach_and_log_load_balancing_loss
        )

    for mtp in mtp_blocks:
        handles.append(
            mtp.register_forward_pre_hook(prepare_mtp_validity_mask, with_kwargs=True)
        )
    for module in modules:
        if isinstance(module, Router) and id(module) in mtp_router_ids:
            handles.append(
                module.register_forward_pre_hook(
                    prepare_router_padding_mask, with_kwargs=True
                )
            )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for module in modules:
            if isinstance(module, Router) and hasattr(
                module, "_nemo_dynamic_aux_scale_tokens"
            ):
                del module._nemo_dynamic_aux_scale_tokens
            if isinstance(module, Router) and hasattr(
                module, "_nemo_dynamic_moe_task_marker"
            ):
                del module._nemo_dynamic_moe_task_marker
        for router_class, had_direct_method, original_method in patched_classes:
            if had_direct_method:
                router_class.attach_and_log_load_balancing_loss = original_method
            else:
                delattr(router_class, "attach_and_log_load_balancing_loss")


@contextmanager
def preserve_attention_cp_groups(model: torch.nn.Module) -> Iterator[None]:
    """Isolate runtime attention, SSM and router state from fixed groups.

    Keep the active group through backward recomputation; restore it after the
    complete no-pipeline schedule. TP, DP and optimizer groups are unchanged.
    """
    targets = _classify_bind_targets(model)
    modules = targets.modules
    saved_collections = [
        (module, module.pg_collection) for module in targets.pg_collections
    ]
    saved_mamba = [(module, module.cp) for module in targets.mamba_mixers]
    saved_gdp = [
        (
            module,
            module.cp,
            module.d_inner_local_cp,
            module.nheads_local_cp,
            module.ngroups_local_cp,
        )
        for module in targets.gated_delta_products
    ]
    saved_gdn = [
        (module, module.cp_size, getattr(module, "feat_dim_split", None))
        for module in targets.gated_delta_nets
    ]
    saved_direct_groups = [
        (
            module,
            module.cp_group,
            getattr(module, "tp_cp_group", None),
            hasattr(module, "tp_cp_group"),
        )
        for module in targets.direct_groups
    ]
    saved_hybrid_stacks = [
        (
            module,
            module._cp_layout_manager,
            module._has_linear_layer_with_chunkwise_cp,
        )
        for module in targets.hybrid_stacks
    ]
    saved_routers = [
        (module, module.cp_group, module.tp_cp_group) for module in targets.routers
    ]
    router_configs: dict[int, Any] = {}
    mtp_enabled = any(
        bool(getattr(getattr(module, "config", None), "mtp_num_layers", 0))
        for module in modules
    )
    model_id = id(model)
    if model_id in _ACTIVE_BIND_TARGETS:
        raise RuntimeError("Dynamic CP model binding is not re-entrant")
    try:
        _ACTIVE_BIND_TARGETS[model_id] = targets
        for module, collection in saved_collections:
            module.pg_collection = copy(collection)
        for module, _, _ in saved_routers:
            config = module.config
            config_id = id(config)
            # Transformer layers commonly share one TransformerConfig.  Record
            # that shared object once, while still rejecting overlap with a
            # different active model binding.
            if config_id in router_configs:
                continue
            if config_id in _ROUTER_CONFIG_BASELINES:
                raise RuntimeError("Dynamic CP router binding is not re-entrant")
            baseline = (
                config.moe_aux_loss_coeff,
                config.moe_z_loss_coeff,
            )
            _ROUTER_CONFIG_BASELINES[config_id] = baseline
            router_configs[config_id] = config
        with (
            _patch_mtp_loss_for_dynamic_cp(mtp_enabled),
            _patch_hybrid_mtp_padding_masks(model, modules),
        ):
            try:
                yield
            except Exception:
                _DYNAMIC_MTP_METRICS.clear()
                raise
    finally:
        _ACTIVE_BIND_TARGETS.pop(model_id, None)
        _ACTIVE_BIND_SIGNATURES.pop(model_id, None)
        for module, collection in saved_collections:
            module.pg_collection = collection
        for module, cp in saved_mamba:
            module.cp = cp
        for module, cp_size, feat_dim_split in saved_gdn:
            module.cp_size = cp_size
            if feat_dim_split is not None:
                module.feat_dim_split = feat_dim_split
        for (
            module,
            cp,
            d_inner_local_cp,
            nheads_local_cp,
            ngroups_local_cp,
        ) in saved_gdp:
            module.cp = cp
            module.d_inner_local_cp = d_inner_local_cp
            module.nheads_local_cp = nheads_local_cp
            module.ngroups_local_cp = ngroups_local_cp
        for module, cp_group, tp_cp_group, had_tp_cp_group in saved_direct_groups:
            module.cp_group = cp_group
            if had_tp_cp_group:
                module.tp_cp_group = tp_cp_group
        for module, manager, has_chunkwise in saved_hybrid_stacks:
            module._cp_layout_manager = manager
            module._has_linear_layer_with_chunkwise_cp = has_chunkwise
        for module, cp_group, tp_cp_group in saved_routers:
            module.cp_group = cp_group
            module.tp_cp_group = tp_cp_group
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
        routing_type = getattr(router.config, "moe_router_load_balancing_type", None)
        if isinstance(routing_type, (list, tuple)) and isinstance(
            aux_coeff, (list, tuple)
        ):
            filtered = [
                coeff if kind == "global_aux_loss" else 0.0
                for kind, coeff in zip(routing_type, aux_coeff)
            ]
            aux_coeff = tuple(filtered) if isinstance(aux_coeff, tuple) else filtered
        elif routing_type != "global_aux_loss":
            aux_coeff = 0.0
        z_coeff = None
    router.config.moe_aux_loss_coeff = aux_coeff
    router.config.moe_z_loss_coeff = z_coeff


def _has_positive_coefficient(value: Any) -> bool:
    """Return whether a scalar or coefficient list enables an aux loss."""
    values = value if isinstance(value, (list, tuple)) else (value,)
    return any(isinstance(item, (int, float)) and item > 0 for item in values)


def configure_dynamic_moe_loss_scaling(
    model: torch.nn.Module, padding_mask: torch.Tensor | None
) -> None:
    """Reduce a task token count once and share it across all MoE routers."""
    routers = _bind_targets(model).routers
    if not routers:
        return

    if not model.training or not torch.is_grad_enabled():
        return
    if not any(
        getattr(router, "calculate_per_token_loss", False)
        and _has_positive_coefficient(router.config.moe_aux_loss_coeff)
        for router in routers
    ):
        return
    if padding_mask is None:
        raise ValueError("Dynamic CP MoE aux loss requires a packed padding mask")

    if any(router.tp_cp_group is not routers[0].tp_cp_group for router in routers[1:]):
        raise ValueError("Dynamic CP routers disagree on the active TP*CP group")

    task_marker = object()
    task_tokens = (~padding_mask.to(dtype=torch.bool)).sum().detach()
    tp_cp_group = routers[0].tp_cp_group
    if tp_cp_group.size() > 1:
        torch.distributed.all_reduce(
            task_tokens,
            op=torch.distributed.ReduceOp.SUM,
            group=tp_cp_group,
        )
    for router in routers:
        router._nemo_dynamic_moe_task_marker = task_marker
        router._nemo_dynamic_aux_scale_tokens = task_tokens


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
        raise ValueError(
            f"No dynamic TP*CP group was initialized for CP={context.size}"
        )
    expected_tp_cp_size = (
        context.size * parallel_state.get_tensor_model_parallel_world_size()
    )
    if tp_cp_group.size() != expected_tp_cp_size:
        raise ValueError("Dynamic MoE TP*CP group has the wrong size")
    padding_only = bool(getattr(packed_seq_params, "dynamic_cp_padding_only", False))
    targets = _bind_targets(model)
    model_id = id(model)
    binding_signature = (context.size, id(group), id(tp_cp_group), padding_only)
    binding_cache_active = model_id in _ACTIVE_BIND_TARGETS
    if (
        not binding_cache_active
        or _ACTIVE_BIND_SIGNATURES.get(model_id) != binding_signature
    ):
        for module in targets.pg_collections:
            module.pg_collection.cp = group
            if hasattr(module.pg_collection, "tp_cp"):
                module.pg_collection.tp_cp = tp_cp_group
        for module in targets.direct_groups:
            module.cp_group = group
            if hasattr(module, "tp_cp_group"):
                module.tp_cp_group = tp_cp_group
        for module in targets.hybrid_stacks:
            _bind_hybrid_stack_layout(module, group=group, tp_cp_group=tp_cp_group)
        for module in targets.routers:
            module.cp_group = group
            module.tp_cp_group = tp_cp_group
            _bind_router_config(module, padding_only=padding_only)
        for module in targets.mamba_mixers:
            _rebuild_mamba_cp(module, group)
        for module in targets.gated_delta_products:
            _rebuild_gdp_cp(module, group)
        for module in targets.gated_delta_nets:
            baseline_size = module.cp_size
            baseline_split = getattr(module, "feat_dim_split", None)
            module.cp_size = context.size
            if baseline_split is not None:
                scaled_split = []
                for value in baseline_split:
                    numerator = value * baseline_size
                    if numerator % context.size:
                        raise ValueError(
                            "GatedDeltaNet projection dimensions are not divisible "
                            f"by runtime CP={context.size}"
                        )
                    scaled_split.append(numerator // context.size)
                module.feat_dim_split = tuple(scaled_split)
        if binding_cache_active:
            _ACTIVE_BIND_SIGNATURES[model_id] = binding_signature
    model_packed = copy(packed_seq_params)
    model_packed.cp_group = group
    return model_packed


def planned_microbatches(
    data: Any,
    plan: CPRankPlan,
    step: CPRankStep,
    straggler_timer: Any,
    *,
    prepad_packed_seq_for_hybridep: bool = False,
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
    expert_group = parallel_state.get_expert_tensor_and_model_parallel_group()
    expert_ranks = set(torch.distributed.get_process_group_ranks(expert_group))
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    runtime_contexts: dict[tuple[int, int], RuntimeCPContext] = {}
    for group_index, rank_group in enumerate(step.groups):
        if not rank_group.assignments:
            raise ValueError("Every CP synchronization group needs one local task")
        for task_index, assignment in enumerate(rank_group.assignments):
            size = assignment.cp_size
            context_key = (assignment.lane_start, size)
            context = runtime_contexts.get(context_key)
            if context is None:
                group = (
                    parallel_state.get_hybrid_data_context_parallel_groups(
                        group_size=size
                    )
                    if size > 1
                    else None
                )
                rank = plan.lane - assignment.lane_start
                if group is not None:
                    members = expected[
                        assignment.lane_start : assignment.lane_start + size
                    ]
                    if (
                        group.size() != size
                        or group.rank() != rank
                        or torch.distributed.get_process_group_ranks(group) != members
                    ):
                        raise ValueError(
                            "Active CP group disagrees with the driver's assignment"
                        )
                task_ranks = {
                    base + offset
                    for base in plan.lane_ranks[
                        assignment.lane_start : assignment.lane_start + size
                    ]
                    for offset in range(tp_size)
                }
                if not expert_ranks.issubset(task_ranks):
                    raise ValueError(
                        "Joint expert TP*EP group crosses dynamic CP task boundaries"
                    )
                context = RuntimeCPContext(size=size, rank=rank, group=group)
                runtime_contexts[context_key] = context
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
                pad_packed_seq_to_multiple_of=assignment.pad_multiple,
                straggler_timer=straggler_timer,
                cp_context=context,
                create_packed_seq_padding_mask=True,
                prepad_packed_seq_for_hybridep=prepad_packed_seq_for_hybridep,
            )
            padding_only = not assignment.sample_indices
            inputs.packed_seq_params.dynamic_cp_padding_only = padding_only
            if padding_only:
                # MCore uses True to mean "padding".  Excluding every physical
                # placeholder token keeps expert-bias counters clean. Local aux
                # coefficients are disabled; an enabled global aux loss still
                # enters its aligned full-domain collective with zero tokens.
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
