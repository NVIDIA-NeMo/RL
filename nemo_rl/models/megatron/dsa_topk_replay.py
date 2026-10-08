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

"""Replay vLLM DSA top-k selections in Megatron-Core.

The pinned Megatron-Core version does not expose a public DSA replay API.  This
module therefore installs a deliberately small runtime patch at the two
index-selection boundaries used by :class:`DSAttention`.  The sparse-attention
kernel remains unchanged; only the indices supplied to it are replaced.

Replay payloads use vLLM layer numbering (zero based) and have shape
``[B, S, L, K]`` or, for packed batches, ``[T, L, K]``.  ``L`` contains the
selected DSA *computing* layers in ascending layer-id order.  DSA skip layers
continue to use Megatron-Core's normal index-sharing holder.
"""

from __future__ import annotations

import inspect
import os
import weakref
from collections.abc import Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Optional, cast

import torch

from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.policy import (
    DSATopKReplayConfig,
    PolicyConfig,
    coerce_dsa_topk_replay_config,
)

_STATE_ATTR = "_nrl_dsa_topk_replay_state"
_PATCH_ATTR = "_nrl_dsa_topk_replay_patch"
_VALIDATE_ENV = "NRL_DSA_TOPK_REPLAY_VALIDATE"
_MISSING_INDEX = -1
_REPLAY_MODULES: weakref.WeakSet[Any] = weakref.WeakSet()


class DSATopKReplayAction(Enum):
    """Current action for one model-owned DSA replay state."""

    REPLAY_FORWARD = "replay_forward"
    REPLAY_BACKWARD = "replay_backward"


@dataclass
class DSATopKReplayState:
    """Replay state attached to one top-k-computing ``DSAttention`` module."""

    layer_number: int
    action: Optional[DSATopKReplayAction] = None
    target_topk_indices: Optional[torch.Tensor] = None
    replay_backward_list: list[torch.Tensor] = field(default_factory=list)

    def clear(self) -> None:
        self.action = None
        self.target_topk_indices = None
        self.replay_backward_list.clear()


@dataclass
class _ActiveReplay:
    module: Any
    state: DSATopKReplayState
    target: torch.Tensor
    effective_indices: Optional[torch.Tensor] = None


_ACTIVE_REPLAY: ContextVar[Optional[_ActiveReplay]] = ContextVar(
    "nrl_active_dsa_topk_replay", default=None
)


def dsa_topk_replay_enabled(config: PolicyConfig) -> bool:
    """Return whether policy-side DSA top-k replay is enabled."""
    replay_config = coerce_dsa_topk_replay_config(config.get("dsa_topk_replay"))
    return isinstance(replay_config, DSATopKReplayConfig)


def should_use_dsa_topk_replay(
    *,
    enabled: bool,
    data: Mapping[str, Any],
    stage: str,
    require: bool,
) -> bool:
    """Require the captured rollout payload for replay-sensitive stages."""
    if not enabled or not require:
        return False
    if "dsa_topk_indices" in data:
        return True
    raise RuntimeError(
        "policy.dsa_topk_replay.enabled=true requires dsa_topk_indices for "
        f"{stage}, but the fetched batch does not contain that field. "
        "Reference-logprob intentionally skips replay; prev-logprob and train "
        "must carry the rollout capture."
    )


def _normalized_config_layer_ids(config: PolicyConfig) -> Optional[list[int]]:
    replay_config = coerce_dsa_topk_replay_config(config.get("dsa_topk_replay"))
    if not isinstance(replay_config, DSATopKReplayConfig):
        raise ValueError(
            "dsa_topk_replay.layer_ids is only available when replay is enabled."
        )
    layer_ids = replay_config.layer_ids
    if layer_ids is None:
        return None
    return sorted(layer_ids)


def configure_vllm_for_dsa_topk_replay(config: PolicyConfig) -> None:
    """Apply generation settings needed to transport vLLM DSA indices."""
    if not dsa_topk_replay_enabled(config):
        return

    layer_ids = _normalized_config_layer_ids(config)
    generation = cast(VllmConfig, config["generation"])
    generation["_dsa_topk_replay_enabled"] = True
    generation["_dsa_topk_replay_layer_ids"] = layer_ids
    vllm_kwargs = generation.setdefault("vllm_kwargs", {})
    # The vLLM source patch uses the routed-experts transport as a generic
    # token-aligned signed-int32 payload channel.  Router and DSA replay are therefore
    # intentionally mutually exclusive.
    vllm_kwargs["enable_return_routed_experts"] = True


def validate_dsa_topk_replay_static_config(config: PolicyConfig) -> None:
    """Validate replay topology without importing or patching Megatron-Core.

    This check is safe to run in the driver before policy-backend dispatch.  In
    particular, it prevents a non-Megatron policy from silently accepting and
    then ignoring the replay payload.
    """
    if not dsa_topk_replay_enabled(config):
        return

    _normalized_config_layer_ids(config)
    generation = config.get("generation") or {}
    megatron_cfg = config.get("megatron_cfg") or {}
    if generation.get("backend") != "vllm":
        raise ValueError("dsa_topk_replay.enabled requires vLLM generation.")
    if not megatron_cfg.get("enabled", False):
        raise ValueError(
            "dsa_topk_replay.enabled requires the Megatron policy backend."
        )
    if bool((config.get("router_replay") or {}).get("enabled", False)):
        raise ValueError(
            "dsa_topk_replay and router_replay cannot be enabled together because "
            "both use vLLM's routed-experts return channel."
        )
    if megatron_cfg.get("cuda_graph_impl", "none") != "none":
        raise ValueError(
            "dsa_topk_replay requires policy.megatron_cfg.cuda_graph_impl=none "
            "because replay indices are runtime side inputs that are not part of "
            "Megatron's captured CUDA-graph input contract."
        )

    vllm_cfg = generation.get("vllm_cfg") or {}
    if vllm_cfg.get("async_engine") is not False:
        raise ValueError(
            "dsa_topk_replay requires generation.vllm_cfg.async_engine=false."
        )
    if vllm_cfg.get("pipeline_parallel_size") != 1:
        raise ValueError(
            "dsa_topk_replay requires generation.vllm_cfg.pipeline_parallel_size=1."
        )
    if vllm_cfg.get("enforce_eager") is not True:
        raise ValueError(
            "dsa_topk_replay requires generation.vllm_cfg.enforce_eager=true."
        )

    vllm_kwargs = generation.get("vllm_kwargs") or {}
    speculative_config = vllm_kwargs.get("speculative_config")
    if speculative_config is not None and (
        not isinstance(speculative_config, dict)
        or speculative_config.get("num_speculative_tokens") != 0
    ):
        raise ValueError(
            "dsa_topk_replay requires speculative decoding to be disabled; "
            "generation.vllm_kwargs.speculative_config must be absent, null, or "
            "set num_speculative_tokens=0."
        )

    model_overrides = megatron_cfg.get("model_overrides") or {}
    loss_coeff = model_overrides.get("dsa_indexer_loss_coeff")
    if loss_coeff not in (0, 0.0):
        raise ValueError(
            "dsa_topk_replay requires dsa_indexer_loss_coeff=0 to be explicitly "
            "set in policy.megatron_cfg.model_overrides "
            "because the model provider may otherwise enable the auxiliary "
            "indexer loss, while replayed indices do not provide its scores."
        )


def validate_dsa_topk_replay_config(config: PolicyConfig) -> None:
    """Validate replay configuration and install the Megatron runtime patch."""
    validate_dsa_topk_replay_static_config(config)
    if not dsa_topk_replay_enabled(config):
        return

    _install_dsa_topk_replay_patch()


def _iter_model_modules_with_mtp_ancestry(
    model: Any, *, beneath_mtp: bool = False
) -> Iterable[tuple[Any, bool]]:
    if isinstance(model, (list, tuple)):
        for item in model:
            yield from _iter_model_modules_with_mtp_ancestry(
                item, beneath_mtp=beneath_mtp
            )
        return

    beneath_mtp = beneath_mtp or bool(getattr(model, "is_mtp_layer", False))
    yield model, beneath_mtp
    children = getattr(model, "children", None)
    if callable(children):
        child_modules = children()
        if not isinstance(child_modules, Iterable):
            raise TypeError(
                "Model children() must return an iterable, got "
                f"{type(child_modules).__name__}."
            )
        for child in child_modules:
            yield from _iter_model_modules_with_mtp_ancestry(
                child, beneath_mtp=beneath_mtp
            )


def _unwrap_model_config(model: Any) -> Optional[Any]:
    if isinstance(model, (list, tuple)):
        for item in model:
            config = _unwrap_model_config(item)
            if config is not None:
                return config
        return None

    current = model
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        config = getattr(current, "config", None)
        if config is not None:
            return config
        current = getattr(current, "module", None)
    return None


def _global_dsa_compute_layer_numbers(model_config: Any) -> list[int]:
    """Return all 1-based layers which compute rather than share DSA top-k."""
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        is_dsa_skip_topk_layer,
    )

    num_layers = int(getattr(model_config, "num_layers"))
    topk_freq = int(getattr(model_config, "dsa_indexer_topk_freq", None) or 1)
    skip_topk_offset = int(
        getattr(model_config, "dsa_indexer_skip_topk_offset", None) or 0
    )
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}.")
    return [
        layer_number
        for layer_number in range(1, num_layers + 1)
        if not is_dsa_skip_topk_layer(layer_number, skip_topk_offset, topk_freq)
    ]


def _selected_compute_layer_numbers(
    model_config: Any, layer_ids: Optional[Sequence[int]]
) -> list[int]:
    compute_layers = _global_dsa_compute_layer_numbers(model_config)
    if layer_ids is None:
        return compute_layers

    normalized_ids = sorted(layer_ids)
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("DSA replay layer_ids must not contain duplicates.")
    if any(
        isinstance(layer_id, bool) or not isinstance(layer_id, int)
        for layer_id in normalized_ids
    ):
        raise ValueError("DSA replay layer_ids entries must be integers.")
    if any(layer_id < 0 for layer_id in normalized_ids):
        raise ValueError("DSA replay layer_ids entries must be non-negative.")

    selected = [layer_id + 1 for layer_id in normalized_ids]
    invalid = sorted(set(selected).difference(compute_layers))
    if invalid:
        raise ValueError(
            "DSA replay layer_ids must identify top-k-computing DSA layers; "
            f"invalid 1-based layers={invalid}, compute_layers={compute_layers}."
        )
    return selected


def dsa_topk_replay_dimensions(
    model_config: Any, layer_ids: Optional[Sequence[int]] = None
) -> tuple[int, int]:
    """Return ``(selected_compute_layers, top_k)`` for a replay payload."""
    num_layers = len(_selected_compute_layer_numbers(model_config, layer_ids))
    top_k = int(getattr(model_config, "dsa_indexer_topk"))
    if num_layers <= 0 or top_k <= 0:
        raise ValueError(
            "DSA top-k replay requires positive payload dimensions, got "
            f"num_layers={num_layers}, top_k={top_k}."
        )
    return num_layers, top_k


def _dsa_compute_modules(model: Any) -> list[tuple[Any, int]]:
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAttention,
    )

    modules: list[tuple[Any, int]] = []
    seen: set[int] = set()
    for module, beneath_mtp in _iter_model_modules_with_mtp_ancestry(model):
        if beneath_mtp or not isinstance(module, DSAttention):
            continue
        if bool(getattr(module, "skip_topk", False)):
            continue
        layer_number = getattr(module, "layer_number", None)
        if layer_number is None or id(module) in seen:
            continue
        seen.add(id(module))
        modules.append((module, int(layer_number)))
    return modules


def _local_layer_numbers_for_model(model: Any) -> set[int]:
    """Return non-MTP layer numbers represented by this local model stage."""
    layer_numbers: set[int] = set()
    for module, beneath_mtp in _iter_model_modules_with_mtp_ancestry(model):
        if beneath_mtp:
            continue
        layer_number = getattr(module, "layer_number", None)
        if layer_number is not None:
            # A transformer-layer wrapper and its attention child commonly carry
            # the same global number.  The set deliberately collapses those
            # duplicates before comparing the stage topology with assignments.
            layer_numbers.add(int(layer_number))
    return layer_numbers


def _normalize_payload(payload: torch.Tensor) -> torch.Tensor:
    if not isinstance(payload, torch.Tensor):
        raise TypeError(
            f"dsa_topk_indices must be a torch.Tensor, got {type(payload).__name__}."
        )
    if payload.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(
            f"dsa_topk_indices must use an integer dtype, got {payload.dtype}."
        )
    if payload.dim() == 3:
        return payload.unsqueeze(0)
    if payload.dim() == 4:
        return payload
    raise ValueError(
        "dsa_topk_indices must have shape [T, L, K] or [B, S, L, K], "
        f"got {tuple(payload.shape)}."
    )


def _validation_enabled() -> bool:
    value = os.getenv(_VALIDATE_ENV, "0").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid {_VALIDATE_ENV}={value!r}; expected a boolean environment value."
    )


def _validate_replay_tensor(
    replay_tensor: torch.Tensor, *, layer_number: int, payload_idx: int
) -> None:
    """Optionally run value-level checks which are expensive for large payloads."""
    if replay_tensor.numel() == 0 or not _validation_enabled():
        return

    if bool(replay_tensor.lt(_MISSING_INDEX).any().item()):
        minimum = int(replay_tensor.min().item())
        raise ValueError(
            "dsa_topk_indices contains an index below the -1 sentinel: "
            f"min={minimum}, layer_number={layer_number}, payload_idx={payload_idx}."
        )

    valid = replay_tensor.ge(0)
    # Invalid entries describe a short causal prefix and must be a suffix.  This
    # is also the layout expected by cuDNN's topk_length contract.
    invalid_before_valid = (~valid).cummax(dim=-1).values & valid
    if bool(invalid_before_valid.any().item()):
        raise ValueError(
            "dsa_topk_indices -1 entries must form a suffix in each top-k row: "
            f"layer_number={layer_number}, payload_idx={payload_idx}."
        )

    sorted_indices = (
        replay_tensor.to(dtype=torch.int64)
        .masked_fill(~valid, torch.iinfo(torch.int64).max)
        .sort(dim=-1)
        .values
    )
    duplicate_valid = sorted_indices[..., 1:].eq(
        sorted_indices[..., :-1]
    ) & sorted_indices[..., 1:].ne(torch.iinfo(torch.int64).max)
    if bool(duplicate_valid.any().item()):
        raise ValueError(
            "dsa_topk_indices contains duplicate valid key ids within a row: "
            f"layer_number={layer_number}, payload_idx={payload_idx}."
        )


def build_dsa_topk_replay_assignments(
    model: Any,
    dsa_topk_indices: torch.Tensor,
    layer_ids: Optional[Sequence[int]] = None,
) -> list[tuple[Any, torch.Tensor]]:
    """Pair local computing ``DSAttention`` modules with ``[B, S, K]`` payloads."""
    payload = _normalize_payload(dsa_topk_indices)
    model_config = _unwrap_model_config(model)
    if model_config is None:
        raise ValueError("Could not locate Megatron model config for DSA top-k replay.")

    selected_layers = _selected_compute_layer_numbers(model_config, layer_ids)
    expected_topk = int(getattr(model_config, "dsa_indexer_topk"))
    if payload.shape[-1] != expected_topk:
        raise ValueError(
            "dsa_topk_indices top-k width does not match Megatron: "
            f"payload={payload.shape[-1]}, model={expected_topk}."
        )
    if payload.shape[2] != len(selected_layers):
        raise ValueError(
            "dsa_topk_indices layer axis must contain the selected computing layers "
            "in ascending order: "
            f"payload={payload.shape[2]}, selected_layers={selected_layers}."
        )

    layer_to_payload_idx = {
        layer_number: payload_idx
        for payload_idx, layer_number in enumerate(selected_layers)
    }
    assignments: list[tuple[Any, torch.Tensor]] = []
    assigned_layer_numbers: set[int] = set()
    for module, layer_number in _dsa_compute_modules(model):
        payload_idx = layer_to_payload_idx.get(layer_number)
        if payload_idx is None:
            continue
        replay_tensor = payload[:, :, payload_idx, :].contiguous()
        _validate_replay_tensor(
            replay_tensor, layer_number=layer_number, payload_idx=payload_idx
        )
        assignments.append((module, replay_tensor))
        assigned_layer_numbers.add(layer_number)

    local_selected_layers = set(selected_layers).intersection(
        _local_layer_numbers_for_model(model)
    )
    missing_local_layers = sorted(local_selected_layers - assigned_layer_numbers)
    if missing_local_layers:
        raise ValueError(
            "Could not find computing DSAttention modules for selected local DSA "
            f"layers {missing_local_layers}. Ensure the model uses the DSA module "
            "spec and that AbsorbedMLASelfAttention.core_attention is DSAttention."
        )
    return assignments


def build_dsa_topk_replay_tensors(
    model: Any,
    dsa_topk_indices: torch.Tensor,
    layer_ids: Optional[Sequence[int]] = None,
) -> list[torch.Tensor]:
    return [
        replay_tensor
        for _, replay_tensor in build_dsa_topk_replay_assignments(
            model, dsa_topk_indices, layer_ids
        )
    ]


def _state_for_module(module: Any) -> DSATopKReplayState:
    state = getattr(module, _STATE_ATTR, None)
    if state is None:
        state = DSATopKReplayState(layer_number=int(module.layer_number))
        setattr(module, _STATE_ATTR, state)
    _REPLAY_MODULES.add(module)
    return state


def set_dsa_topk_replay_forward(
    model: Any,
    dsa_topk_indices: torch.Tensor,
    layer_ids: Optional[Sequence[int]] = None,
) -> None:
    """Arm selected local DSA modules for the next forward microbatch."""
    _install_dsa_topk_replay_patch()
    selected_modules: set[int] = set()
    for module, replay_tensor in build_dsa_topk_replay_assignments(
        model, dsa_topk_indices, layer_ids
    ):
        state = _state_for_module(module)
        state.target_topk_indices = replay_tensor
        state.action = DSATopKReplayAction.REPLAY_FORWARD
        selected_modules.add(id(module))

    # A configured subset may change between calls in tests or debugging.  Do
    # not leave a previously selected module armed with stale indices.
    for module, _ in _dsa_compute_modules(model):
        if id(module) in selected_modules:
            continue
        state = getattr(module, _STATE_ATTR, None)
        if state is not None:
            # A deselected module cannot consume any entries recorded while it
            # was selected.  Retaining that FIFO would make a later re-selection
            # replay an older microbatch during activation recomputation.
            state.clear()


def set_dsa_topk_replay_backward(model: Any) -> None:
    """Use each module's FIFO during activation-checkpoint recomputation."""
    for module, _ in _dsa_compute_modules(model):
        state = getattr(module, _STATE_ATTR, None)
        if state is not None and state.target_topk_indices is not None:
            state.action = DSATopKReplayAction.REPLAY_BACKWARD


def clear_dsa_topk_replay(model: Optional[Any] = None) -> None:
    """Clear all model-scoped replay targets, actions, and recompute FIFOs."""
    modules = (
        list(_REPLAY_MODULES)
        if model is None
        else [module for module, _ in _dsa_compute_modules(model)]
    )
    for module in modules:
        state = getattr(module, _STATE_ATTR, None)
        if state is not None:
            state.clear()


def _tp_size_and_rank(module: Any) -> tuple[int, int]:
    tp_group = getattr(getattr(module, "pg_collection", None), "tp", None)
    if tp_group is None:
        return 1, 0
    size = int(tp_group.size())
    rank = int(tp_group.rank())
    return size, rank


def _indices_for_selector(
    active: _ActiveReplay, q: torch.Tensor, index_topk: int
) -> torch.Tensor:
    target = active.target
    if target.dim() != 3:
        raise RuntimeError(
            f"DSA replay target must have shape [B, S, K], got {tuple(target.shape)}."
        )
    query_rows, batch = q.shape[:2]
    if target.shape[0] != batch:
        raise RuntimeError(
            "DSA replay batch dimension does not match the indexer query: "
            f"target={target.shape[0]}, query={batch}, "
            f"layer={active.state.layer_number}."
        )
    if target.shape[-1] != index_topk:
        raise RuntimeError(
            "DSA replay top-k width does not match the selector: "
            f"target={target.shape[-1]}, selector={index_topk}, "
            f"layer={active.state.layer_number}."
        )

    if target.shape[1] != query_rows:
        tp_size, tp_rank = _tp_size_and_rank(active.module)
        if tp_size <= 1 or target.shape[1] != query_rows * tp_size:
            raise RuntimeError(
                "DSA replay sequence dimension does not match the indexer query or a "
                "contiguous sequence-parallel shard: "
                f"target={target.shape[1]}, query={query_rows}, tp_size={tp_size}, "
                f"layer={active.state.layer_number}."
            )
        row_start = tp_rank * query_rows
        target = target[:, row_start : row_start + query_rows, :]

    return target.to(device=q.device, dtype=torch.int32).contiguous()


def _prepare_fused_replay_indices(
    indices: torch.Tensor, *, key_sequence_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match MCore's compact-and-sort contract for fused sparse attention.

    ``run_fused_qk_topk`` normally returns a compact prefix of valid indices,
    sorted by key position, together with the prefix length.  FlashMLA trusts
    that representation when ``topk_length`` is supplied and does not prepare
    it again.  vLLM replay order is therefore canonicalized at this boundary;
    this changes only ordering, not the selected key set.
    """
    valid = indices.ge(0)
    topk_length = valid.sum(dim=-1, dtype=torch.int32)

    # First move valid slots left while preserving their vLLM order, matching
    # MCore's _compact_valid_topk_indices.
    positions = torch.arange(indices.size(-1), device=indices.device)
    positions = positions.view(*((1,) * (indices.dim() - 1)), -1)
    compact_order_key = torch.where(
        valid,
        positions.expand_as(indices),
        torch.full_like(indices, indices.size(-1)),
    )
    compact_order = compact_order_key.argsort(dim=-1)
    compacted = torch.gather(indices, dim=-1, index=compact_order)
    compacted_valid = torch.gather(valid, dim=-1, index=compact_order)
    compacted = compacted.masked_fill(~compacted_valid, _MISSING_INDEX)

    # Then sort only the consumed prefix by key id.  Invalid suffix slots use
    # sk as the sort sentinel, exactly like MCore's sort_topk_by_index helper.
    prefix_valid = positions < topk_length.unsqueeze(-1)
    sort_key = torch.where(
        prefix_valid,
        compacted,
        torch.full_like(compacted, key_sequence_length),
    )
    sort_order = sort_key.argsort(dim=-1)
    sorted_indices = torch.gather(compacted, dim=-1, index=sort_order)
    sorted_valid = torch.gather(prefix_valid, dim=-1, index=sort_order)
    sorted_indices = sorted_indices.masked_fill(
        ~sorted_valid, _MISSING_INDEX
    ).contiguous()
    return sorted_indices.to(dtype=torch.int32), topk_length.contiguous()


def _merge_native_fallback(
    active: _ActiveReplay,
    replay_indices: torch.Tensor,
    native_indices: torch.Tensor,
) -> torch.Tensor:
    fallback_rows = replay_indices.eq(_MISSING_INDEX).all(dim=-1)
    effective = replay_indices
    if bool(fallback_rows.any().item()):
        native_indices = native_indices.to(
            device=replay_indices.device, dtype=replay_indices.dtype
        )
        if (
            native_indices.shape[:-1] != replay_indices.shape[:-1]
            or native_indices.shape[-1] > replay_indices.shape[-1]
        ):
            raise RuntimeError(
                "Megatron fallback top-k shape does not match DSA replay: "
                f"native={tuple(native_indices.shape)}, "
                f"replay={tuple(replay_indices.shape)}."
            )
        if native_indices.shape[-1] < replay_indices.shape[-1]:
            # The naive selector returns min(index_topk, key_sequence_length)
            # columns.  Replay payloads retain their configured fixed K, so a
            # short sequence needs an invalid suffix before rows can be merged.
            padded_native_indices = torch.full_like(replay_indices, _MISSING_INDEX)
            padded_native_indices[..., : native_indices.shape[-1]] = native_indices
            native_indices = padded_native_indices
        effective = replay_indices.clone()
        effective[fallback_rows] = native_indices[fallback_rows]
    active.effective_indices = effective
    return effective


def _target_for_state(state: DSATopKReplayState) -> Optional[torch.Tensor]:
    if state.action == DSATopKReplayAction.REPLAY_FORWARD:
        return state.target_topk_indices
    if state.action == DSATopKReplayAction.REPLAY_BACKWARD:
        return state.replay_backward_list[0] if state.replay_backward_list else None
    return None


def _complete_active_replay(active: _ActiveReplay) -> None:
    if active.effective_indices is None:
        raise RuntimeError(
            "Megatron DSAttention completed without reaching a patched top-k "
            f"selector at layer {active.state.layer_number}."
        )
    if active.state.action == DSATopKReplayAction.REPLAY_FORWARD:
        active.state.replay_backward_list.append(
            active.effective_indices.detach().contiguous()
        )
    elif active.state.action == DSATopKReplayAction.REPLAY_BACKWARD:
        active.state.replay_backward_list.pop(0)


def _install_dsa_topk_replay_patch() -> None:
    """Patch the pinned MCore DSA selector boundaries exactly once."""
    from megatron.core.transformer.experimental_attention_variant import (
        dsa,
        dsa_kernels,
    )

    if getattr(dsa.DSAttention.forward, _PATCH_ATTR, False):
        return

    expected_forward_params = [
        "self",
        "query",
        "key",
        "value",
        "attention_mask",
        "x",
        "qr",
        "position_ids",
        "attn_mask_type",
        "attention_bias",
        "packed_seq_params",
        "up_v_weight",
    ]
    actual_forward_params = list(inspect.signature(dsa.DSAttention.forward).parameters)
    if actual_forward_params != expected_forward_params:
        raise RuntimeError(
            "Unsupported Megatron DSAttention.forward signature for DSA replay: "
            f"expected={expected_forward_params}, actual={actual_forward_params}."
        )

    original_forward = dsa.DSAttention.forward
    original_fused_attention = dsa_kernels.run_fused_dsa_attention
    original_fused_topk = dsa_kernels.run_fused_qk_topk
    original_naive_topk = dsa.fused_qk_topk_naive

    @wraps(original_forward)
    def wrapped_forward(module: Any, *args: Any, **kwargs: Any) -> Any:
        state = getattr(module, _STATE_ATTR, None)
        if state is None or state.action is None:
            return original_forward(module, *args, **kwargs)
        if float(getattr(module.config, "dsa_indexer_loss_coeff", None) or 0.0) > 0:
            raise RuntimeError(
                "DSA top-k replay cannot run with dsa_indexer_loss_coeff > 0."
            )

        target = _target_for_state(state)
        if target is None:
            raise RuntimeError(
                "DSA top-k replay has no target indices for "
                f"{state.action.value} at layer {state.layer_number}."
            )

        active = _ActiveReplay(module=module, state=state, target=target)
        token = _ACTIVE_REPLAY.set(active)
        try:
            output = original_forward(module, *args, **kwargs)
            _complete_active_replay(active)
            return output
        finally:
            _ACTIVE_REPLAY.reset(token)

    @wraps(original_fused_attention)
    def wrapped_fused_attention(*args: Any, **kwargs: Any) -> Any:
        # A full fused kernel owns both selection and attention and therefore
        # cannot accept externally selected indices.  Returning None selects
        # MCore's existing split selector + sparse-attention fallback; the
        # sparse attention itself can still use its fused cuDNN backend.
        if _ACTIVE_REPLAY.get() is not None:
            return None
        return original_fused_attention(*args, **kwargs)

    @wraps(original_fused_topk)
    def wrapped_fused_topk(*args: Any, **kwargs: Any) -> Any:
        active = _ACTIVE_REPLAY.get()
        if active is None:
            return original_fused_topk(*args, **kwargs)

        q = kwargs["q"] if "q" in kwargs else args[1]
        k = kwargs["k"] if "k" in kwargs else args[2]
        index_topk = kwargs["index_topk"] if "index_topk" in kwargs else args[4]
        if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor):
            raise TypeError(
                "DSA fused top-k expects tensor q and k inputs, got "
                f"q={type(q).__name__}, k={type(k).__name__}."
            )
        if not isinstance(index_topk, int):
            raise TypeError(
                "DSA fused top-k expects an integer index_topk, got "
                f"{type(index_topk).__name__}."
            )
        replay_indices = _indices_for_selector(active, q, index_topk)
        fallback_rows = replay_indices.eq(_MISSING_INDEX).all(dim=-1)
        if bool(fallback_rows.any().item()):
            native_result = original_fused_topk(*args, **kwargs)
            if native_result is None:
                # Let DSAttention reach the patched naive selector, which can
                # produce the fallback rows on every backend.
                return None
            native_indices, _ = native_result
            replay_indices = _merge_native_fallback(
                active, replay_indices, native_indices
            )
        else:
            active.effective_indices = replay_indices
        replay_indices, topk_length = _prepare_fused_replay_indices(
            replay_indices, key_sequence_length=int(k.size(0))
        )
        active.effective_indices = replay_indices
        return replay_indices, topk_length

    @wraps(original_naive_topk)
    def wrapped_naive_topk(
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        index_topk: int,
        mask: Optional[torch.Tensor] = None,
        varlen_starts: Optional[torch.Tensor] = None,
        varlen_ends: Optional[torch.Tensor] = None,
        key_positions: Optional[torch.Tensor] = None,
        use_relu: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active = _ACTIVE_REPLAY.get()
        if active is None:
            return original_naive_topk(
                q,
                k,
                weights,
                index_topk,
                mask,
                varlen_starts,
                varlen_ends,
                key_positions,
                use_relu,
            )

        replay_indices = _indices_for_selector(active, q, index_topk)
        fallback_rows = replay_indices.eq(_MISSING_INDEX).all(dim=-1)
        if bool(fallback_rows.any().item()):
            scores, native_indices = original_naive_topk(
                q,
                k,
                weights,
                index_topk,
                mask,
                varlen_starts,
                varlen_ends,
                key_positions,
                use_relu,
            )
            replay_indices = _merge_native_fallback(
                active, replay_indices, native_indices
            )
            return scores, replay_indices

        active.effective_indices = replay_indices
        # With indexer loss disabled the caller immediately discards scores.
        scores = torch.empty(0, dtype=torch.float32, device=q.device)
        return scores, replay_indices

    setattr(wrapped_forward, _PATCH_ATTR, True)
    setattr(wrapped_fused_attention, _PATCH_ATTR, True)
    setattr(wrapped_fused_topk, _PATCH_ATTR, True)
    setattr(wrapped_naive_topk, _PATCH_ATTR, True)
    dsa.DSAttention.forward = wrapped_forward
    dsa_kernels.run_fused_dsa_attention = wrapped_fused_attention
    dsa_kernels.run_fused_qk_topk = wrapped_fused_topk
    dsa.fused_qk_topk_naive = wrapped_naive_topk


__all__ = [
    "DSATopKReplayAction",
    "DSATopKReplayState",
    "build_dsa_topk_replay_assignments",
    "build_dsa_topk_replay_tensors",
    "clear_dsa_topk_replay",
    "configure_vllm_for_dsa_topk_replay",
    "dsa_topk_replay_dimensions",
    "dsa_topk_replay_enabled",
    "set_dsa_topk_replay_backward",
    "set_dsa_topk_replay_forward",
    "should_use_dsa_topk_replay",
    "validate_dsa_topk_replay_config",
    "validate_dsa_topk_replay_static_config",
]
