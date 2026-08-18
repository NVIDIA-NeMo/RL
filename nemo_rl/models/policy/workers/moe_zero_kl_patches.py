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

"""Runtime patches for zero train/gen KL on colocated Megatron MoE inference.

Wired from ``setup._apply_zero_train_gen_mismatch`` for recipes such as
Qwen3-30B-A3B DAPO with ``zero_train_gen_mismatch: true``:

1. **MoE routing + combine** — ``topk_routing_with_score_function`` builds sparse
   routing maps with ``index_put_`` (not ``scatter``); ``unpermute`` uses gather on
   train and gather+droppad on CUDA-graphed decode. Fused TE router/unpermute paths
   are rejected (``moe_permute_fusion=false`` required).
2. **Router replay** — reconstruct finished-request routing before KV blocks release.
3. **CUDA graphs** — 64-token graph bucket floor, runtime decode padding alignment,
   suppress spurious graph-bucket match when inference graphs are off (via lifecycle
   wrappers on ``prepare_for_generation`` / ``finish_generation``), and colocated pad
   reset after inference sleep (before scoring).

TP=1 log-softmax lives in ``patches.py`` (generic zero-KL, not MoE-specific).

Public API::

    apply_moe_determinism_patches()
    apply_cuda_graph_inference_determinism_patches()  # when cuda_graph_impl != none
    restore_moe_determinism_patches()
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Tuple

import torch

if TYPE_CHECKING:
    from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
    from megatron.core.inference.text_generation_controllers.text_generation_controller import (
        TextGenerationController,
    )

_NRL_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_ZERO_KL_EAGER_MOE_MSG = (
    "zero_train_gen_mismatch requires eager MoE routing/combine; set "
    "policy.megatron_cfg.moe_permute_fusion=false and disable fused router paths."
)

# MoE routing + combine (single moe_utils monkeypatch entry)
_NRL_UNPERMUTE_PATH_SEEN: set[str] = set()
_UNPERMUTE_ORIG: Optional[Callable[..., torch.Tensor]] = None
_TOKEN_DISPATCHER_UNPERMUTE_ORIG: Optional[Callable[..., torch.Tensor]] = None
_TOPK_ROUTING_ORIG: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]] = None
_ROUTER_TOPK_ROUTING_ORIG: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]] = None
_MOE_UNPERMUTE_PATCHED = False
_NRL_DET_COMBINE_BANNER = False
_COMBINE_IMPL_OVERRIDE: Optional[str] = None
_COMBINE_GATHER_DROPPAD_OVERRIDE: Optional[bool] = None

# Router replay inference
_DYNAMIC_STEP_BOOKKEEPING_ORIG: Optional[Callable[..., Dict[str, Any]]] = None
_ASYNC_BOOKKEEP_ORIG: Optional[Callable[..., Any]] = None
_ROUTER_REPLAY_INFERENCE_PATCHED = False

# CUDA graph inference (colocated decode)
_CUDA_GRAPH_BUCKET_FLOOR_PATCHED = False
_CG_DIMS_GEN_ORIG: Optional[Callable[..., Tuple[Sequence, Optional[Sequence]]]] = None
_MIN_TOKEN_PAD_PATCHED = False
_DIC_SETATTR_ORIG: Optional[Callable[..., None]] = None
_FORCE_EAGER_DIMS_PATCHED = False
_IAS_ORIG: Optional[Callable[..., Any]] = None
_MATCH_GRAPH_CONFIG_ORIG: Optional[Callable[..., Any]] = None
_IN_GRAPH_CAPTURE = False
_INFERENCE_CUDA_GRAPHS_ENABLED = False
_FORCE_EAGER_DIMS_BANNER = False
_FORCE_EAGER_LIFECYCLE_PATCHED = False
_PREPARE_FOR_GEN_ORIG: Optional[Callable[..., None]] = None
_FINISH_GEN_ORIG: Optional[Callable[..., None]] = None

# Colocated inference sleep — clear graph decode expert padding before scoring
_SLEEP_ORIG: Optional[Callable[..., None]] = None
_PAD_RESET_PATCHED = False

# ---------------------------------------------------------------------------
# MoE combine — configuration
# ---------------------------------------------------------------------------


def configure_moe_combine_for_zero_kl(*, gather_droppad: bool = False) -> None:
    """Select gather-based MoE combine for zero train/gen KL (rl-fp4 ground truth).

    Train and logprob scoring use ``gather_combine``; CUDA-graphed decode with
    ``drop_and_pad`` uses ``gather_combine_droppad`` when ``gather_droppad=True``.
    """
    global _COMBINE_IMPL_OVERRIDE, _COMBINE_GATHER_DROPPAD_OVERRIDE
    _COMBINE_IMPL_OVERRIDE = "gather"
    _COMBINE_GATHER_DROPPAD_OVERRIDE = gather_droppad


def configure_moe_combine_for_cuda_graph_inference() -> None:
    """Gather on train, gather+droppad on CUDA-graphed MoE decode."""
    configure_moe_combine_for_zero_kl(gather_droppad=True)


def _combine_impl() -> str:
    if _COMBINE_IMPL_OVERRIDE is not None:
        impl = _COMBINE_IMPL_OVERRIDE
    else:
        impl = os.environ.get("NRL_COMBINE_IMPL", "gather").strip().lower()
    return impl if impl in ("segmented", "gather") else "gather"


def _combine_gather_droppad() -> bool:
    if _COMBINE_GATHER_DROPPAD_OVERRIDE is not None:
        return _COMBINE_GATHER_DROPPAD_OVERRIDE
    return os.environ.get("NRL_COMBINE_GATHER_DROPPAD", "0") == "1"


def _nrl_log_unpermute_path(path: str) -> None:
    if path not in _NRL_UNPERMUTE_PATH_SEEN:
        _NRL_UNPERMUTE_PATH_SEEN.add(path)
        _NRL_LOGGER.warning("[moe-combine] unpermute executed via '%s'", path)


def _build_sparse_routing_with_index_put(
    logits: torch.Tensor,
    probs: torch.Tensor,
    top_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build ``[T, E]`` routing tensors with deterministic ``index_put_`` (rl-fp4 path)."""
    num_tokens = logits.size(0)
    rows = torch.arange(num_tokens, device=logits.device).unsqueeze(1)
    routing_probs = torch.zeros_like(logits)
    routing_probs.index_put_((rows, top_indices), probs, accumulate=False)
    routing_map = torch.zeros_like(logits, dtype=logits.dtype)
    routing_map.index_put_(
        (rows, top_indices),
        torch.ones_like(probs, dtype=routing_map.dtype),
        accumulate=False,
    )
    return routing_probs, routing_map.bool()


def _patched_topk_routing_with_score_function(
    *args: Any, **kwargs: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``topk_routing_with_score_function`` with ``index_put_`` sparse routing maps."""
    if kwargs.get("fused", False):
        raise ValueError(_ZERO_KL_EAGER_MOE_MSG)
    if kwargs.get("dense_output", False):
        assert _TOPK_ROUTING_ORIG is not None
        return _TOPK_ROUTING_ORIG(*args, **kwargs)

    assert _TOPK_ROUTING_ORIG is not None
    eager_kwargs = dict(kwargs)
    eager_kwargs["dense_output"] = True
    eager_kwargs["fused"] = False
    probs, top_indices = _TOPK_ROUTING_ORIG(*args, **eager_kwargs)
    logits = args[0] if args else kwargs["logits"]
    return _build_sparse_routing_with_index_put(logits, probs, top_indices)


# ---------------------------------------------------------------------------
# MoE combine — deterministic kernels
# ---------------------------------------------------------------------------


def _segment_sum(vals: torch.Tensor, group_sizes: torch.Tensor) -> torch.Tensor:
    """Deterministic per-segment sum of contiguous row groups (fp32 accumulate)."""
    vf = vals.float()
    seg_fn = getattr(torch, "segment_reduce", None)
    if seg_fn is not None:
        try:
            return seg_fn(vf, "sum", lengths=group_sizes, axis=0, unsafe=True).to(vals.dtype)
        except Exception:
            pass
    n, h = vals.shape
    csum = torch.zeros(n + 1, h, dtype=torch.float32, device=vals.device)
    torch.cumsum(vf, dim=0, out=csum[1:])
    ends = group_sizes.cumsum(0)
    starts = ends - group_sizes
    return (csum[ends] - csum[starts]).to(vals.dtype)


def _unpermute_segmented_combine(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
) -> torch.Tensor:
    """Deterministic combine via segmented sum (no dense [T, max_slots, H] buffer)."""
    num_tokens, hidden = restore_shape
    num_permuted = permuted_tokens.size(0)
    out = torch.zeros(num_tokens, hidden, dtype=permuted_tokens.dtype, device=permuted_tokens.device)
    if num_permuted == 0:
        return out
    sort_perm = torch.argsort(sorted_indices, stable=True)
    dest = sorted_indices[sort_perm]
    vals = permuted_tokens[sort_perm]
    if num_permuted > 1:
        change = dest.new_ones(num_permuted, dtype=torch.bool)
        change[1:] = dest[1:] != dest[:-1]
    else:
        change = dest.new_ones(1, dtype=torch.bool)
    group_id = change.long().cumsum(0) - 1
    num_groups = int(group_id[-1].item()) + 1
    group_sizes = torch.bincount(group_id, minlength=num_groups)
    unique_dest = dest[change]
    seg = _segment_sum(vals, group_sizes)
    out[unique_dest] = seg.to(out.dtype)
    return out


def _unpermute_gather_combine(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    routing_map: torch.Tensor,
) -> torch.Tensor:
    """Packed top-k combine for train/logprob (matches droppad expert-id order)."""
    num_tokens, hidden = restore_shape
    rmap = routing_map.bool()
    within = rmap.long().cumsum(dim=0) - 1
    expert_counts = rmap.long().sum(dim=0)
    expert_offset = torch.zeros_like(expert_counts)
    expert_offset[1:] = expert_counts.cumsum(0)[:-1]
    pos = expert_offset.unsqueeze(0) + within
    pos_sel = pos[rmap]
    if num_tokens == 0 or pos_sel.numel() % num_tokens != 0:
        return _unpermute_segmented_combine(permuted_tokens, sorted_indices, restore_shape)
    gathered = permuted_tokens.index_select(0, pos_sel)
    return gathered.view(num_tokens, -1, hidden).sum(dim=1)


def _unpermute_gather_combine_droppad(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    routing_map: torch.Tensor,
) -> torch.Tensor:
    """Gather combine for ``drop_and_pad`` CUDA-graphed decode (capture-safe, no ``nonzero()``)."""
    num_tokens, hidden = restore_shape
    rmap = routing_map.bool()
    num_experts = rmap.size(1)
    total_slots = permuted_tokens.size(0)
    if num_experts == 0 or total_slots % num_experts != 0:
        return _unpermute_segmented_combine(permuted_tokens, sorted_indices, restore_shape)
    capacity = total_slots // num_experts
    rank = rmap.long().cumsum(dim=0) - 1
    expert_base = torch.arange(num_experts, device=rmap.device) * capacity
    pos = expert_base.unsqueeze(0) + rank
    sel_key = torch.where(
        rmap,
        torch.arange(num_experts, device=rmap.device).unsqueeze(0).expand_as(rmap),
        num_experts + torch.arange(num_experts, device=rmap.device).unsqueeze(0).expand_as(rmap),
    )
    expert_idx_sorted = sel_key.sort(dim=1).values
    _tk = getattr(_unpermute_gather_combine_droppad, "_nrl_static_topk", None)
    if _tk is None:
        _tk = int(rmap[0].sum().item()) if num_tokens > 0 else 0
        _unpermute_gather_combine_droppad._nrl_static_topk = _tk
    if _tk == 0:
        return _unpermute_segmented_combine(permuted_tokens, sorted_indices, restore_shape)
    expert_idx = expert_idx_sorted[:, :_tk]
    pos_sel = torch.gather(pos, 1, expert_idx)
    keep_sel = torch.gather(rank, 1, expert_idx) < capacity
    pos_safe = torch.where(keep_sel, pos_sel, torch.zeros_like(pos_sel))
    gathered = permuted_tokens.index_select(0, pos_safe.reshape(-1))
    gathered = gathered * keep_sel.reshape(-1).unsqueeze(-1).to(gathered.dtype)
    return gathered.view(num_tokens, -1, hidden).sum(dim=1)


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
    """``moe_utils.unpermute`` with deterministic combine routing."""
    global _NRL_DET_COMBINE_BANNER
    if fused:
        raise ValueError(_ZERO_KL_EAGER_MOE_MSG)

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

    if not _NRL_DET_COMBINE_BANNER:
        _NRL_DET_COMBINE_BANNER = True
        print(
            f"[NRL_DET_COMBINE] deterministic combine ACTIVE impl={_combine_impl()} "
            f"droppad={_combine_gather_droppad()}",
            flush=True,
        )

    impl = _combine_impl()
    if impl == "gather" and routing_map is not None and not drop_and_pad:
        _nrl_log_unpermute_path("gather_combine")
        output_tokens = _unpermute_gather_combine(
            permuted_tokens, sorted_indices, restore_shape, routing_map
        )
    elif (
        impl == "gather"
        and drop_and_pad
        and routing_map is not None
        and _combine_gather_droppad()
    ):
        _nrl_log_unpermute_path("gather_combine_droppad")
        output_tokens = _unpermute_gather_combine_droppad(
            permuted_tokens, sorted_indices, restore_shape, routing_map
        )
    else:
        _nrl_log_unpermute_path("segmented_combine")
        output_tokens = _unpermute_segmented_combine(
            permuted_tokens, sorted_indices, restore_shape
        )
    return output_tokens.to(dtype=input_dtype)


def apply_moe_unpermute_determinism_patch() -> None:
    """Patch MoE routing/combine in ``moe_utils`` (+ cached import sites)."""
    global _UNPERMUTE_ORIG, _TOKEN_DISPATCHER_UNPERMUTE_ORIG, _MOE_UNPERMUTE_PATCHED
    global _TOPK_ROUTING_ORIG, _ROUTER_TOPK_ROUTING_ORIG
    if _MOE_UNPERMUTE_PATCHED:
        return
    try:
        import megatron.core.transformer.moe.moe_utils as moe_utils
        import megatron.core.transformer.moe.token_dispatcher as token_dispatcher
    except ImportError:
        print(
            "moe_zero_kl_patches: Megatron MoE modules are not importable; "
            "skipping routing/combine patch."
        )
        return

    _UNPERMUTE_ORIG = moe_utils.unpermute
    _TOKEN_DISPATCHER_UNPERMUTE_ORIG = token_dispatcher.unpermute
    moe_utils.unpermute = _patched_unpermute
    # token_dispatcher binds unpermute at import time; patch both call sites.
    token_dispatcher.unpermute = _patched_unpermute

    if hasattr(moe_utils, "topk_routing_with_score_function"):
        _TOPK_ROUTING_ORIG = moe_utils.topk_routing_with_score_function
        moe_utils.topk_routing_with_score_function = _patched_topk_routing_with_score_function
    try:
        import megatron.core.transformer.moe.router as moe_router

        if hasattr(moe_router, "topk_routing_with_score_function"):
            _ROUTER_TOPK_ROUTING_ORIG = moe_router.topk_routing_with_score_function
            moe_router.topk_routing_with_score_function = (
                _patched_topk_routing_with_score_function
            )
    except ImportError:
        pass

    _MOE_UNPERMUTE_PATCHED = True
    print(
        "[moe_zero_kl_patches] patched moe_utils topk_routing_with_score_function "
        "(index_put routing maps), unpermute/token_dispatcher.unpermute "
        "(gather combine), and rejected fused MoE paths."
    )


# ---------------------------------------------------------------------------
# Router replay inference
# ---------------------------------------------------------------------------


def _nrl_dynamic_step_context_bookkeeping(self: "TextGenerationController") -> Dict[str, Any]:
    """Reconstruct MoE routing for finished requests before KV blocks are released."""
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
    """Apply pre-reconstructed routing before upstream post_process."""
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


def apply_router_replay_inference_patches() -> None:
    """Patch dynamic inference for early router-replay reconstruction."""
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


# ---------------------------------------------------------------------------
# CUDA graph inference — bucket floor + runtime decode padding
# ---------------------------------------------------------------------------


def floor_cuda_graph_batch_dimensions(
    ladder: Sequence,
    *,
    max_tokens: int,
    max_requests: int,
) -> Tuple[list, list[int]]:
    """Keep only 64-token-multiple CUDA graph decode buckets."""
    from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions

    if not ladder:
        return [], []

    seen: set[int] = set()
    new_list: list[InferenceBatchDimensions] = []
    cap = min(max_tokens, (max_requests // 64) * 64)
    for dim in ladder:
        token_count = ((max(dim.token_count, 64) + 63) // 64) * 64
        if token_count > cap:
            token_count = cap
        if token_count < 64 or token_count in seen:
            continue
        seen.add(token_count)
        new_list.append(
            InferenceBatchDimensions(
                token_count=token_count,
                prefill_req_count=dim.prefill_req_count,
                decode_req_count=(
                    token_count if dim.prefill_req_count == 0 else dim.decode_req_count
                ),
            )
        )
    token_counts = [dim.token_count for dim in new_list]
    return new_list, token_counts


def apply_cuda_graph_bucket_floor_patch() -> None:
    """Patch ``CUDAGraphBatchDimensionBuilder`` to 64-align graph capture buckets."""
    global _CUDA_GRAPH_BUCKET_FLOOR_PATCHED, _CG_DIMS_GEN_ORIG
    if _CUDA_GRAPH_BUCKET_FLOOR_PATCHED:
        return

    from megatron.core.inference.batch_dimensions_utils import (
        CUDAGraphBatchDimensionBuilder,
    )

    _CG_DIMS_GEN_ORIG = (
        CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list
    )

    @classmethod  # type: ignore[misc]
    def _patched_generate(cls, *args, **kwargs):
        assert _CG_DIMS_GEN_ORIG is not None
        ladder, token_counts = _CG_DIMS_GEN_ORIG(*args, **kwargs)
        max_tokens = kwargs.get("max_tokens")
        max_requests = kwargs.get("max_requests")
        if max_tokens is None or max_requests is None:
            if len(args) >= 6:
                max_requests = args[5]
            if len(args) >= 7:
                max_tokens = args[6]
        if max_tokens is None or max_requests is None:
            return ladder, token_counts

        old_len = len(ladder)
        new_ladder, new_counts = floor_cuda_graph_batch_dimensions(
            ladder,
            max_tokens=int(max_tokens),
            max_requests=int(max_requests),
        )
        if new_ladder:
            print(
                f"[cuda_graph_bucket_floor] ladder {old_len} -> "
                f"{[dim.token_count for dim in new_ladder]} (64-multiples)",
                flush=True,
            )
            return new_ladder, new_counts
        return ladder, token_counts

    CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list = (
        _patched_generate
    )
    _CUDA_GRAPH_BUCKET_FLOOR_PATCHED = True


def restore_cuda_graph_bucket_floor_patch() -> None:
    """Restore the original CUDA graph bucket builder (for tests)."""
    global _CUDA_GRAPH_BUCKET_FLOOR_PATCHED, _CG_DIMS_GEN_ORIG
    if not _CUDA_GRAPH_BUCKET_FLOOR_PATCHED or _CG_DIMS_GEN_ORIG is None:
        return
    from megatron.core.inference.batch_dimensions_utils import (
        CUDAGraphBatchDimensionBuilder,
    )

    CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list = (
        _CG_DIMS_GEN_ORIG
    )
    _CG_DIMS_GEN_ORIG = None
    _CUDA_GRAPH_BUCKET_FLOOR_PATCHED = False


def align_runtime_decode_padded_dimensions(
    dims: Any,
    *,
    max_tokens: int,
    max_requests: int,
    round_up_tokens: Callable[[int], int],
    is_creating_cuda_graphs: bool = False,
) -> Any:
    """Round pure-decode padded dims up to the 64-token quantum."""
    if is_creating_cuda_graphs:
        return dims
    if dims is None or dims.token_count <= 0:
        return dims
    if dims.prefill_req_count != 0:
        return dims
    if dims.token_count != dims.decode_req_count:
        return dims

    token_count = dims.token_count
    new_token_count = min(
        round_up_tokens(token_count), max_tokens, max_requests
    )
    if new_token_count <= token_count:
        return dims

    from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions

    return InferenceBatchDimensions(
        token_count=new_token_count,
        prefill_req_count=0,
        decode_req_count=new_token_count,
    )


def _maybe_align_runtime_decode_padding(context: Any) -> None:
    if getattr(context, "is_creating_cuda_graphs", False):
        return

    aligned = align_runtime_decode_padded_dimensions(
        context.padded_batch_dimensions,
        max_tokens=int(context.max_tokens),
        max_requests=int(context.max_requests),
        round_up_tokens=context.round_up_tokens,
        is_creating_cuda_graphs=False,
    )
    if aligned is context.padded_batch_dimensions:
        return

    if not getattr(type(context), "_nrl_pad_banner", False):
        type(context)._nrl_pad_banner = True
        old_tc = context.padded_batch_dimensions.token_count
        print(
            f"[cuda_graph_min_token_pad] decode step M {old_tc} -> "
            f"{aligned.token_count} (tokens+requests, 64-quantum)",
            flush=True,
        )
    object.__setattr__(context, "padded_batch_dimensions", aligned)


def apply_min_token_pad_patch() -> None:
    """Patch ``DynamicInferenceContext`` to 64-align runtime decode padding."""
    global _MIN_TOKEN_PAD_PATCHED, _DIC_SETATTR_ORIG
    if _MIN_TOKEN_PAD_PATCHED:
        return

    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

    _DIC_SETATTR_ORIG = DynamicInferenceContext.__setattr__

    def _patched_setattr(self: Any, name: str, value: Any) -> None:
        assert _DIC_SETATTR_ORIG is not None
        _DIC_SETATTR_ORIG(self, name, value)
        if name == "padded_batch_dimensions":
            _maybe_align_runtime_decode_padding(self)

    DynamicInferenceContext.__setattr__ = _patched_setattr  # type: ignore[method-assign]
    _MIN_TOKEN_PAD_PATCHED = True


def restore_min_token_pad_patch() -> None:
    """Restore the original ``DynamicInferenceContext.__setattr__`` (for tests)."""
    global _MIN_TOKEN_PAD_PATCHED, _DIC_SETATTR_ORIG
    if not _MIN_TOKEN_PAD_PATCHED or _DIC_SETATTR_ORIG is None:
        return
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

    DynamicInferenceContext.__setattr__ = _DIC_SETATTR_ORIG  # type: ignore[method-assign]
    _DIC_SETATTR_ORIG = None
    _MIN_TOKEN_PAD_PATCHED = False


def set_inference_cuda_graphs_enabled(enabled: bool) -> None:
    """Track whether colocated inference has CUDA graphs toggled on."""
    global _INFERENCE_CUDA_GRAPHS_ENABLED
    _INFERENCE_CUDA_GRAPHS_ENABLED = enabled


def _should_suppress_spurious_cuda_graph_match() -> bool:
    """Return True when eager steps must not match the graph capture ladder."""
    if _IN_GRAPH_CAPTURE:
        return False
    force_eager = os.environ.get("NRL_FORCE_EAGER_DIMS", "auto")
    if force_eager == "1":
        return True
    if force_eager == "0":
        return False
    # Auto: only suppress on colocated train/score steps where lifecycle tracking
    # explicitly cleared inference graphs. Non-colocated (or pre-prepare) defaults
    # to graphs-on so decode replay is never accidentally disabled.
    if not _FORCE_EAGER_LIFECYCLE_PATCHED:
        return False
    return not _INFERENCE_CUDA_GRAPHS_ENABLED


def apply_force_eager_dims_patch() -> None:
    """Suppress graph-bucket matching when inference CUDA graphs are disabled."""
    global _FORCE_EAGER_DIMS_PATCHED, _IAS_ORIG, _MATCH_GRAPH_CONFIG_ORIG
    if _FORCE_EAGER_DIMS_PATCHED:
        return

    from megatron.core.inference.batch_dimensions_utils import (
        CUDAGraphBatchDimensionBuilder,
    )
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

    _IAS_ORIG = DynamicInferenceContext.initialize_attention_state
    _MATCH_GRAPH_CONFIG_ORIG = CUDAGraphBatchDimensionBuilder.match_graph_config

    def _patched_initialize_attention_state(self: Any, **kwargs: Any) -> Any:
        global _IN_GRAPH_CAPTURE
        assert _IAS_ORIG is not None
        in_capture = kwargs.get("construct_graph_dimensions") is not None
        _IN_GRAPH_CAPTURE = in_capture
        try:
            return _IAS_ORIG(self, **kwargs)
        finally:
            _IN_GRAPH_CAPTURE = False

    @classmethod  # type: ignore[misc]
    def _patched_match_graph_config(cls, *args: Any, **kwargs: Any) -> Any:
        assert _MATCH_GRAPH_CONFIG_ORIG is not None
        if _should_suppress_spurious_cuda_graph_match():
            global _FORCE_EAGER_DIMS_BANNER
            if not _FORCE_EAGER_DIMS_BANNER:
                _FORCE_EAGER_DIMS_BANNER = True
                print(
                    "[cuda_graph_force_eager_dims] suppressing spurious bucket match "
                    "(inference graphs off)",
                    flush=True,
                )
            return None
        return _MATCH_GRAPH_CONFIG_ORIG(*args, **kwargs)

    DynamicInferenceContext.initialize_attention_state = (  # type: ignore[method-assign]
        _patched_initialize_attention_state
    )
    CUDAGraphBatchDimensionBuilder.match_graph_config = _patched_match_graph_config
    _FORCE_EAGER_DIMS_PATCHED = True


def restore_force_eager_dims_patch() -> None:
    """Restore graph-bucket matching hooks (for tests)."""
    restore_colocated_force_eager_lifecycle_patch()
    global _FORCE_EAGER_DIMS_PATCHED, _IAS_ORIG, _MATCH_GRAPH_CONFIG_ORIG
    global _IN_GRAPH_CAPTURE, _INFERENCE_CUDA_GRAPHS_ENABLED, _FORCE_EAGER_DIMS_BANNER
    if not _FORCE_EAGER_DIMS_PATCHED:
        return
    from megatron.core.inference.batch_dimensions_utils import (
        CUDAGraphBatchDimensionBuilder,
    )
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

    if _IAS_ORIG is not None:
        DynamicInferenceContext.initialize_attention_state = _IAS_ORIG  # type: ignore[method-assign]
    if _MATCH_GRAPH_CONFIG_ORIG is not None:
        CUDAGraphBatchDimensionBuilder.match_graph_config = _MATCH_GRAPH_CONFIG_ORIG
    _IAS_ORIG = None
    _MATCH_GRAPH_CONFIG_ORIG = None
    _IN_GRAPH_CAPTURE = False
    _INFERENCE_CUDA_GRAPHS_ENABLED = False
    _FORCE_EAGER_DIMS_BANNER = False
    _FORCE_EAGER_DIMS_PATCHED = False


def _generation_cuda_graph_impl(cfg: dict[str, Any]) -> str:
    mcore_gen = (cfg.get("generation") or {}).get("mcore_generation_config") or {}
    return mcore_gen.get("cuda_graph_impl", "none") or "none"


def _should_track_inference_cuda_graph_lifecycle(cfg: dict[str, Any]) -> bool:
    """True for colocated zero-KL runs that toggle inference CUDA graphs on/off."""
    megatron_cfg = cfg.get("megatron_cfg") or {}
    if not megatron_cfg.get("zero_train_gen_mismatch"):
        return False
    generation = cfg.get("generation") or {}
    if not (generation.get("colocated") or {}).get("enabled", False):
        return False
    return _generation_cuda_graph_impl(cfg) not in ("none",)


def _sync_inference_cuda_graphs_enabled_for_prepare(cfg: dict[str, Any]) -> None:
    if not _should_track_inference_cuda_graph_lifecycle(cfg):
        return
    enabled = _generation_cuda_graph_impl(cfg) not in ("none",)
    set_inference_cuda_graphs_enabled(enabled)


def apply_colocated_force_eager_lifecycle_patch() -> None:
    """Wrap gen entry/exit so force-eager knows when inference graphs are toggled."""
    global _FORCE_EAGER_LIFECYCLE_PATCHED, _PREPARE_FOR_GEN_ORIG, _FINISH_GEN_ORIG
    global _INFERENCE_CUDA_GRAPHS_ENABLED
    if _FORCE_EAGER_LIFECYCLE_PATCHED:
        return

    from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin

    _PREPARE_FOR_GEN_ORIG = MegatronGenerationMixin.prepare_for_generation
    _FINISH_GEN_ORIG = MegatronGenerationMixin.finish_generation
    # Default graphs-on until colocated finish_generation clears the flag after
    # toggle_cuda_graphs(none); avoids suppressing decode before the first prepare.
    _INFERENCE_CUDA_GRAPHS_ENABLED = True

    def _patched_prepare_for_generation(self: Any, tags: Any = None, **kwargs: Any) -> None:
        _sync_inference_cuda_graphs_enabled_for_prepare(self.cfg)
        assert _PREPARE_FOR_GEN_ORIG is not None
        _PREPARE_FOR_GEN_ORIG(self, tags=tags, **kwargs)

    def _patched_finish_generation(self: Any) -> None:
        assert _FINISH_GEN_ORIG is not None
        _FINISH_GEN_ORIG(self)
        if _should_track_inference_cuda_graph_lifecycle(self.cfg):
            set_inference_cuda_graphs_enabled(False)

    MegatronGenerationMixin.prepare_for_generation = (  # type: ignore[method-assign]
        _patched_prepare_for_generation
    )
    MegatronGenerationMixin.finish_generation = _patched_finish_generation  # type: ignore[method-assign]
    _FORCE_EAGER_LIFECYCLE_PATCHED = True


def restore_colocated_force_eager_lifecycle_patch() -> None:
    global _FORCE_EAGER_LIFECYCLE_PATCHED, _PREPARE_FOR_GEN_ORIG, _FINISH_GEN_ORIG
    if not _FORCE_EAGER_LIFECYCLE_PATCHED:
        return
    from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin

    if _PREPARE_FOR_GEN_ORIG is not None:
        MegatronGenerationMixin.prepare_for_generation = _PREPARE_FOR_GEN_ORIG  # type: ignore[method-assign]
    if _FINISH_GEN_ORIG is not None:
        MegatronGenerationMixin.finish_generation = _FINISH_GEN_ORIG  # type: ignore[method-assign]
    _PREPARE_FOR_GEN_ORIG = None
    _FINISH_GEN_ORIG = None
    _FORCE_EAGER_LIFECYCLE_PATCHED = False


def apply_cuda_graph_inference_determinism_patches() -> None:
    """CUDA-graph zero-KL path: gather+droppad MoE combine, bucket floor, decode padding."""
    configure_moe_combine_for_cuda_graph_inference()
    apply_cuda_graph_bucket_floor_patch()
    apply_min_token_pad_patch()
    apply_force_eager_dims_patch()
    apply_colocated_force_eager_lifecycle_patch()
    apply_colocated_pad_reset_patch()


def _should_reset_pad_before_scoring(cfg: dict[str, Any]) -> bool:
    megatron_cfg = cfg.get("megatron_cfg") or {}
    if not megatron_cfg.get("zero_train_gen_mismatch"):
        return False
    if not megatron_cfg.get("moe_pad_experts_for_cuda_graph_inference"):
        return False
    generation = cfg.get("generation") or {}
    if not (generation.get("colocated") or {}).get("enabled", False):
        return False
    mcore_gen = generation.get("mcore_generation_config") or {}
    return mcore_gen.get("cuda_graph_impl", "none") not in (None, "none")


def _reset_decode_expert_padding(model: Any) -> None:
    from megatron.core.inference.utils import set_decode_expert_padding
    from megatron.core.utils import unwrap_model

    set_decode_expert_padding(unwrap_model(model), False)
    print("[NRL_PAD_RESET] expert padding cleared before scoring", flush=True)


def apply_colocated_pad_reset_patch() -> None:
    """Wrap colocated ``_sleep`` so graph decode padding does not leak into scoring."""
    global _SLEEP_ORIG, _PAD_RESET_PATCHED
    if _PAD_RESET_PATCHED:
        return

    from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin

    _SLEEP_ORIG = MegatronGenerationMixin._sleep

    def _patched_sleep(self: Any) -> None:
        assert _SLEEP_ORIG is not None
        _SLEEP_ORIG(self)
        if _should_reset_pad_before_scoring(self.cfg):
            _reset_decode_expert_padding(self.model)

    MegatronGenerationMixin._sleep = _patched_sleep  # type: ignore[method-assign]
    _PAD_RESET_PATCHED = True


def restore_colocated_pad_reset_patch() -> None:
    global _SLEEP_ORIG, _PAD_RESET_PATCHED
    if not _PAD_RESET_PATCHED or _SLEEP_ORIG is None:
        return
    from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin

    MegatronGenerationMixin._sleep = _SLEEP_ORIG  # type: ignore[method-assign]
    _SLEEP_ORIG = None
    _PAD_RESET_PATCHED = False


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def apply_moe_determinism_patches() -> None:
    """Apply MoE combine and router replay patches."""
    configure_moe_combine_for_zero_kl(gather_droppad=False)
    apply_moe_unpermute_determinism_patch()
    apply_router_replay_inference_patches()


def restore_moe_determinism_patches() -> None:
    """Restore all Megatron entry points patched by this module (for tests)."""
    global _MOE_UNPERMUTE_PATCHED
    global _ROUTER_REPLAY_INFERENCE_PATCHED
    global _NRL_DET_COMBINE_BANNER
    global _COMBINE_IMPL_OVERRIDE, _COMBINE_GATHER_DROPPAD_OVERRIDE
    global _TOPK_ROUTING_ORIG, _ROUTER_TOPK_ROUTING_ORIG

    if _MOE_UNPERMUTE_PATCHED and _UNPERMUTE_ORIG is not None:
        import megatron.core.transformer.moe.moe_utils as moe_utils
        import megatron.core.transformer.moe.token_dispatcher as token_dispatcher

        moe_utils.unpermute = _UNPERMUTE_ORIG
        if _TOKEN_DISPATCHER_UNPERMUTE_ORIG is not None:
            token_dispatcher.unpermute = _TOKEN_DISPATCHER_UNPERMUTE_ORIG
        if _TOPK_ROUTING_ORIG is not None:
            moe_utils.topk_routing_with_score_function = _TOPK_ROUTING_ORIG
        try:
            import megatron.core.transformer.moe.router as moe_router

            if _ROUTER_TOPK_ROUTING_ORIG is not None:
                moe_router.topk_routing_with_score_function = _ROUTER_TOPK_ROUTING_ORIG
        except ImportError:
            pass
        _TOPK_ROUTING_ORIG = None
        _ROUTER_TOPK_ROUTING_ORIG = None
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

    restore_cuda_graph_bucket_floor_patch()
    restore_min_token_pad_patch()
    restore_force_eager_dims_patch()
    restore_colocated_pad_reset_patch()

    _NRL_UNPERMUTE_PATH_SEEN.clear()
    if hasattr(_unpermute_gather_combine_droppad, "_nrl_static_topk"):
        del _unpermute_gather_combine_droppad._nrl_static_topk
    _NRL_DET_COMBINE_BANNER = False
    _COMBINE_IMPL_OVERRIDE = None
    _COMBINE_GATHER_DROPPAD_OVERRIDE = None
