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



from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
    from megatron.core.inference.text_generation_controllers.text_generation_controller import (
        TextGenerationController,
    )

_NRL_LOGGER = logging.getLogger(__name__)

# One-shot guard so the unpermute-combine-path diagnostic surfaces which combine
# branch actually executes (fused vs fixed-order vs scatter_add) without flooding.
_NRL_UNPERMUTE_PATH_SEEN: set[str] = set()

_UNPERMUTE_ORIG: Optional[Callable[..., torch.Tensor]] = None
_TOKEN_DISPATCHER_UNPERMUTE_ORIG: Optional[Callable[..., torch.Tensor]] = None
_DYNAMIC_STEP_BOOKKEEPING_ORIG: Optional[Callable[..., Dict[str, Any]]] = None
_ASYNC_BOOKKEEP_ORIG: Optional[Callable[..., Any]] = None

_MOE_UNPERMUTE_PATCHED = False
_ROUTER_REPLAY_INFERENCE_PATCHED = False


def _nrl_log_unpermute_path(path: str) -> None:
    if path not in _NRL_UNPERMUTE_PATH_SEEN:
        _NRL_UNPERMUTE_PATH_SEEN.add(path)
        _NRL_LOGGER.warning("[moe-combine] unpermute executed via '%s'", path)


def _unpermute_fixed_order_combine(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
) -> torch.Tensor:
    """Sum expert outputs per token in stable (permute) order via [T, max_slots, H].sum(1).

    Avoids atomic ``scatter_add_`` / ``index_add_``. Uses the same accumulation dtype as
    ``scatter_add_`` (``permuted_tokens.dtype`` after gating weights).
    """
    num_tokens, hidden = restore_shape
    num_permuted = permuted_tokens.size(0)
    if num_permuted == 0:
        return torch.zeros(
            restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device
        )

    sort_perm = torch.argsort(sorted_indices, stable=True)
    dest = sorted_indices[sort_perm]
    vals = permuted_tokens[sort_perm]

    seq = torch.arange(num_permuted, device=permuted_tokens.device, dtype=torch.long)
    if num_permuted > 1:
        change = dest.new_ones(num_permuted, dtype=torch.bool)
        change[1:] = dest[1:] != dest[:-1]
    else:
        change = dest.new_ones(1, dtype=torch.bool)
    group_id = change.long().cumsum(0) - 1
    num_groups = int(group_id[-1].item()) + 1
    group_sizes = torch.bincount(group_id, minlength=num_groups)
    starts = torch.zeros(num_groups, dtype=torch.long, device=permuted_tokens.device)
    if num_groups > 1:
        starts[1:] = group_sizes.cumsum(0)[:-1]
    slot = seq - starts[group_id]
    max_slots = int(group_sizes.max().item())

    contrib = torch.zeros(
        num_tokens, max_slots, hidden, dtype=permuted_tokens.dtype, device=permuted_tokens.device
    )
    contrib[dest, slot] = vals
    return contrib.sum(dim=1)


def _patched_unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: Optional[torch.Tensor] = None,
    routing_map: Optional[torch.Tensor] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
    pad_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``moe_utils.unpermute`` with fixed-order combine."""
    import megatron.core.transformer.moe.moe_utils as moe_utils

    if fused:
        if not moe_utils.HAVE_TE or moe_utils.fused_unpermute is None:
            raise ValueError("fused_unpermute is not available. Please install TE >= 2.1.0.")
        _nrl_log_unpermute_path("fused_unpermute")
        extra_kwargs = {}
        if moe_utils.is_te_min_version("2.12.0"):
            extra_kwargs["pad_offsets"] = pad_offsets
        return moe_utils.fused_unpermute(
            permuted_tokens,
            sorted_indices,
            merging_probs=probs,
            restore_shape=restore_shape,
            **extra_kwargs,
        )

    input_dtype = permuted_tokens.dtype

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        if drop_and_pad:
            num_experts = routing_map.size(1)
            num_permuted_tokens = sorted_indices.size(0)
            capacity = num_permuted_tokens // num_experts
            num_unpermuted_tokens = probs.size(0)

            probs_T_1D = probs.T.contiguous().view(-1)
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_unpermuted_tokens + indices_dim1).view(-1)
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
        else:
            permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    _nrl_log_unpermute_path("fixed_order_combine")
    output_tokens = _unpermute_fixed_order_combine(
        permuted_tokens, sorted_indices, restore_shape
    )
    return output_tokens.to(dtype=input_dtype)


def _nrl_dynamic_step_context_bookkeeping(self: "TextGenerationController") -> Dict[str, Any]:
    """Early MoE routing reconstruction before KV blocks are released (a6e829b)."""
    from torch.cuda.nvtx import range_pop, range_push

    context = self.inference_wrapped_model.inference_context
    active_request_count = context.total_request_count - context.paused_request_count
    active_request_slice = slice(context.paused_request_count, context.total_request_count)

    range_push("transfer_samples_to_cpu")
    sampled_tokens_cpu, sampled_mtp_tokens_cpu = self._transfer_samples_to_cpu(
        active_request_count
    )
    range_pop()

    range_push("active_request_mask")
    active_request_ids = context.request_ids[active_request_slice].long()
    active_sequence_lengths = context.get_active_sequence_lengths()
    active_sequence_lengths += 1
    max_sequence_lengths = context.get_max_sequence_lengths()

    active_request_mask = (
        sampled_tokens_cpu
        != context.active_request_metadata["termination_id"][:active_request_count]
    ).byte() & torch.less(active_sequence_lengths, max_sequence_lengths).byte()

    if self._get_stop_word_finished_ids_callback is not None:
        request_ids_list = active_request_ids.tolist()
        stop_word_finished_ids = self._get_stop_word_finished_ids_callback(request_ids_list)
        if stop_word_finished_ids:
            for idx, request_id in enumerate(request_ids_list):
                if request_id in stop_word_finished_ids:
                    active_request_mask[idx] = 0

    finished_idxs = (
        torch.nonzero(active_request_mask == 0, as_tuple=True)[0] + context.paused_request_count
    )
    finished_request_ids = context.request_ids[finished_idxs]

    finished_routing_indices: Dict[int, Any] = {}
    if context.moe_enable_routing_replay and finished_idxs.numel() > 0:
        for fidx in finished_idxs.tolist():
            req_id = int(context.request_ids[fidx].item())
            blocks = context.request_to_kv_block_ids[fidx]
            valid = blocks[blocks >= 0].tolist()
            if not valid:
                continue
            total_tokens = int(
                active_sequence_lengths[fidx - context.paused_request_count].item()
            )
            routing = context.kv_block_allocator.reconstruct_routing_from_blocks(
                valid, total_tokens - 1
            )
            if routing is not None:
                finished_routing_indices[req_id] = routing

    new_sample_copy = sampled_tokens_cpu.clone()
    range_pop()

    range_push("update_requests")
    update_result = context.update_requests(
        active_request_mask, new_sample_copy, sampled_mtp_tokens_cpu
    )
    range_pop()

    return {
        "active_request_ids": active_request_ids,
        "finished_request_ids": finished_request_ids,
        "sample": sampled_tokens_cpu,
        "finished_routing_indices": finished_routing_indices,
        **(update_result or {}),
    }


async def _nrl_async_bookkeep(
    self: "DynamicInferenceEngine",
    step_result: Optional[Dict[str, Any]],
    context_state: Dict[str, Any],
    step_time: float,
):
    """Apply pre-reconstructed routing before upstream post_process (a6e829b)."""
    if step_result is not None:
        finished_routing_indices = step_result.get("finished_routing_indices")
        if finished_routing_indices:
            for request_id, routing in finished_routing_indices.items():
                if request_id in self.requests:
                    self.get_request(request_id).routing_indices = routing
        step_result = dict(step_result)
        step_result.pop("finished_routing_block_ids", None)
    assert _ASYNC_BOOKKEEP_ORIG is not None
    return await _ASYNC_BOOKKEEP_ORIG(self, step_result, context_state, step_time)


def apply_moe_unpermute_determinism_patch() -> None:
    """Patch MoE unpermute and the token dispatcher's cached import."""
    global _UNPERMUTE_ORIG, _TOKEN_DISPATCHER_UNPERMUTE_ORIG, _MOE_UNPERMUTE_PATCHED
    if _MOE_UNPERMUTE_PATCHED:
        return
    try:
        import megatron.core.transformer.moe.moe_utils as moe_utils
        import megatron.core.transformer.moe.token_dispatcher as token_dispatcher
    except ImportError:
        print(
            "moe_determinism_patches: Megatron MoE modules are not importable; "
            "skipping unpermute patch."
        )
        return

    _UNPERMUTE_ORIG = moe_utils.unpermute
    _TOKEN_DISPATCHER_UNPERMUTE_ORIG = token_dispatcher.unpermute
    moe_utils.unpermute = _patched_unpermute
    # token_dispatcher imports unpermute by value, so changing only the source
    # module leaves its already-bound call site on the original implementation.
    token_dispatcher.unpermute = _patched_unpermute
    _MOE_UNPERMUTE_PATCHED = True
    print(
        "[moe_determinism_patches] patched moe_utils.unpermute and "
        "token_dispatcher.unpermute with fixed-order combine."
    )


def apply_router_replay_inference_patches() -> None:
    """Patch dynamic inference for early router-replay reconstruction (a6e829b)."""
    global _DYNAMIC_STEP_BOOKKEEPING_ORIG, _ASYNC_BOOKKEEP_ORIG, _ROUTER_REPLAY_INFERENCE_PATCHED
    if _ROUTER_REPLAY_INFERENCE_PATCHED:
        return
    try:
        from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
        from megatron.core.inference.text_generation_controllers.text_generation_controller import (
            TextGenerationController,
        )
    except ImportError:
        print(
            "moe_determinism_patches: Megatron inference modules are not importable; "
            "skipping router-replay inference patches."
        )
        return

    _DYNAMIC_STEP_BOOKKEEPING_ORIG = TextGenerationController._dynamic_step_context_bookkeeping
    TextGenerationController._dynamic_step_context_bookkeeping = (
        _nrl_dynamic_step_context_bookkeeping
    )

    _ASYNC_BOOKKEEP_ORIG = DynamicInferenceEngine.async_bookkeep
    DynamicInferenceEngine.async_bookkeep = _nrl_async_bookkeep

    _ROUTER_REPLAY_INFERENCE_PATCHED = True
    print(
        "[moe_determinism_patches] patched TextGenerationController._dynamic_step_context_bookkeeping "
        "and DynamicInferenceEngine.async_bookkeep for early router-replay routing reconstruction."
    )


def apply_moe_determinism_patches(*, router_replay: bool = False) -> None:
    """Apply all requested MoE determinism / router-replay runtime patches."""
    apply_moe_unpermute_determinism_patch()
    if router_replay:
        apply_router_replay_inference_patches()


def restore_moe_determinism_patches() -> None:
    """Restore Megatron entry points patched by this module (for tests)."""
    global _MOE_UNPERMUTE_PATCHED, _ROUTER_REPLAY_INFERENCE_PATCHED

    if _MOE_UNPERMUTE_PATCHED and _UNPERMUTE_ORIG is not None:
        import megatron.core.transformer.moe.moe_utils as moe_utils
        import megatron.core.transformer.moe.token_dispatcher as token_dispatcher

        moe_utils.unpermute = _UNPERMUTE_ORIG
        if _TOKEN_DISPATCHER_UNPERMUTE_ORIG is not None:
            token_dispatcher.unpermute = _TOKEN_DISPATCHER_UNPERMUTE_ORIG
        _MOE_UNPERMUTE_PATCHED = False

    if _ROUTER_REPLAY_INFERENCE_PATCHED:
        from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
        from megatron.core.inference.text_generation_controllers.text_generation_controller import (
            TextGenerationController,
        )

        if _DYNAMIC_STEP_BOOKKEEPING_ORIG is not None:
            TextGenerationController._dynamic_step_context_bookkeeping = (
                _DYNAMIC_STEP_BOOKKEEPING_ORIG
            )
        if _ASYNC_BOOKKEEP_ORIG is not None:
            DynamicInferenceEngine.async_bookkeep = _ASYNC_BOOKKEEP_ORIG
        _ROUTER_REPLAY_INFERENCE_PATCHED = False

    _NRL_UNPERMUTE_PATH_SEEN.clear()
