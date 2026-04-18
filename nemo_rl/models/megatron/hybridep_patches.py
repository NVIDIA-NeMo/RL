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
"""Monkey-patches for Megatron-LM fused_a2a.py to fix HybridEP issues.

These patches address bugs in the upstream HybridEP implementation that cause
NCCL hangs and NaN gradients. They are applied at import time so that Megatron's
MoE token_dispatcher picks up the patched behavior transparently.

Fixes applied:
1. `enable_custom_allgather=True` in HybridEPBuffer constructor to avoid NCCL
   allgather size-mismatch hang with variable per-rank token counts.
2. Buffer reinitialization when seq_len exceeds the previous max to avoid
   silent buffer overflow (deep_ep's update_template_config grows
   max_num_of_tokens_per_rank in Python but does not reallocate C++ buffers).
3. needs_reset flag set after every backward pass to force a fresh buffer on
   the next forward, ensuring clean IMEX registration and sync-flag state.
4. HybridEPDispatch.backward returns 9 grads matching the 9 forward inputs.
5. HybridEPCombine.backward returns 4 grads matching the 4 forward inputs.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

_patched = False
# Set NRL_HYBRIDEP_DEBUG=1 to print per-call dispatch/combine shapes and routing.
_debug = os.environ.get("NRL_HYBRIDEP_DEBUG", "0") == "1"


def apply_hybridep_patches() -> None:
    """Apply monkey-patches to megatron.core.transformer.moe.fused_a2a.

    Idempotent — safe to call multiple times. No-op if deep_ep HybridEP is not
    installed.
    """
    global _patched
    if _patched:
        return

    try:
        from megatron.core.transformer.moe import fused_a2a
    except ImportError:
        return

    if not getattr(fused_a2a, "HAVE_HYBRIDEP", False):
        return

    try:
        from deep_ep import HybridEPBuffer
    except ImportError:
        return

    # Module-level state tracking buffer size and post-backward dirty flag.
    fused_a2a._hybrid_ep_buffer_max_seq_len = 0
    fused_a2a._hybrid_ep_buffer_needs_reset = False

    def init_hybrid_ep_buffer(
        group,
        hidden_dim,
        seq_len,
        num_local_experts,
        num_sms_dispatch_api,
        num_sms_combine_api,
        fp8_dispatch,
    ):
        """Patched init: force enable_custom_allgather=True, track seq_len."""
        assert not fp8_dispatch, "HybridEP dispatcher does not support fp8 dispatch"
        fused_a2a._hybrid_ep_buffer = HybridEPBuffer(
            group=group,
            hidden_dim=hidden_dim,
            max_num_of_tokens_per_rank=seq_len,
            num_local_experts=num_local_experts,
            use_fp8=fp8_dispatch,
            num_sms_dispatch_api=num_sms_dispatch_api,
            num_sms_combine_api=num_sms_combine_api,
            enable_custom_allgather=True,
        )
        fused_a2a._hybrid_ep_buffer_max_seq_len = seq_len
        fused_a2a._hybrid_ep_buffer_needs_reset = False

    fused_a2a.init_hybrid_ep_buffer = init_hybrid_ep_buffer

    # Patch HybridEPDispatch: reinit buffer on growth/dirty, return 9 grads.
    class PatchedHybridEPDispatch(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            x,
            routing_map,
            probs,
            group,
            num_local_experts,
            num_sms_dispatch_api=24,
            num_sms_combine_api=24,
            num_permuted_tokens=None,
            pad_multiple=None,
        ):
            seq_len, hidden_dim = x.shape[-2:]
            if (
                fused_a2a._hybrid_ep_buffer is None
                or seq_len > fused_a2a._hybrid_ep_buffer_max_seq_len
                or fused_a2a._hybrid_ep_buffer_needs_reset
            ):
                fused_a2a.init_hybrid_ep_buffer(
                    group,
                    hidden_dim,
                    seq_len,
                    num_local_experts,
                    num_sms_dispatch_api,
                    num_sms_combine_api,
                    False,  # fp8_dispatch
                )
            non_blocking = num_permuted_tokens is not None
            (
                dispatched_hidden,
                dispatched_probs,
                dispatched_scaling_factor,
                tokens_per_expert,
                handle,
            ) = fused_a2a._hybrid_ep_buffer.dispatch_with_permute(
                hidden=x,
                routing_map=routing_map,
                probs=probs,
                scaling_factor=None,
                num_of_experts_per_rank=num_local_experts,
                pad_multiple=pad_multiple,
                num_permuted_tokens=num_permuted_tokens,
                non_blocking=non_blocking,
            )
            ctx.handle = handle
            ctx.pad_multiple = pad_multiple
            if _debug:
                rank = (
                    torch.distributed.get_rank()
                    if torch.distributed.is_initialized()
                    else -1
                )
                ep_rank = torch.distributed.get_rank(group) if group is not None else -1
                ep_size = (
                    torch.distributed.get_world_size(group) if group is not None else -1
                )
                print(
                    f"[HEDBG dispatch rank={rank} ep={ep_rank}/{ep_size}] "
                    f"x={tuple(x.shape)} routing_map={tuple(routing_map.shape)} "
                    f"dispatched={tuple(dispatched_hidden.shape)} "
                    f"tokens_per_expert={tokens_per_expert.tolist()}",
                    flush=True,
                )
            return (
                dispatched_hidden,
                dispatched_probs,
                dispatched_scaling_factor,
                tokens_per_expert,
                handle,
            )

        @staticmethod
        def backward(
            ctx,
            grad_x,
            grad_probs,
            grad_scaling_factor,
            grad_tokens_per_expert,
            grad_handle,
        ):
            handle = ctx.handle
            combined_hidden, combined_probs = (
                fused_a2a._hybrid_ep_buffer.combine_with_unpermute(
                    hidden=grad_x,
                    probs=grad_probs,
                    handle=handle,
                    pad_multiple=ctx.pad_multiple,
                )
            )
            fused_a2a._hybrid_ep_buffer_needs_reset = True
            if _debug:
                rank = (
                    torch.distributed.get_rank()
                    if torch.distributed.is_initialized()
                    else -1
                )
                print(
                    f"[HEDBG dispatch.backward rank={rank}] "
                    f"grad_x={tuple(grad_x.shape)} "
                    f"combined={tuple(combined_hidden.shape)} needs_reset=True",
                    flush=True,
                )
            # 9 grads for 9 forward inputs
            # (x, routing_map, probs, group, num_local_experts,
            #  num_sms_dispatch_api, num_sms_combine_api,
            #  num_permuted_tokens, pad_multiple)
            return (
                combined_hidden,
                None,
                combined_probs,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    PatchedHybridEPDispatch.__name__ = "HybridEPDispatch"
    PatchedHybridEPDispatch.__qualname__ = "HybridEPDispatch"
    fused_a2a.HybridEPDispatch = PatchedHybridEPDispatch

    # Patch HybridEPCombine: return 4 grads matching 4 forward inputs.
    class PatchedHybridEPCombine(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, handle, num_permuted_tokens=None, pad_multiple=None):
            combined_hidden, _ = fused_a2a._hybrid_ep_buffer.combine_with_unpermute(
                hidden=x, handle=handle, pad_multiple=pad_multiple
            )
            ctx.handle = handle
            ctx.pad_multiple = pad_multiple
            ctx.num_permuted_tokens = num_permuted_tokens
            return combined_hidden

        @staticmethod
        def backward(ctx, grad_x):
            handle = ctx.handle
            dispatched_hidden, _, _, _, _ = (
                fused_a2a._hybrid_ep_buffer.dispatch_with_permute(
                    hidden=grad_x,
                    scaling_factor=None,
                    handle=handle,
                    pad_multiple=ctx.pad_multiple,
                    num_permuted_tokens=ctx.num_permuted_tokens,
                )
            )
            fused_a2a._hybrid_ep_buffer_needs_reset = True
            # 4 grads for 4 forward inputs (x, handle, num_permuted_tokens, pad_multiple)
            return dispatched_hidden, None, None, None

    PatchedHybridEPCombine.__name__ = "HybridEPCombine"
    PatchedHybridEPCombine.__qualname__ = "HybridEPCombine"
    fused_a2a.HybridEPCombine = PatchedHybridEPCombine

    # Also rebind the helper functions that call the classes via .apply().
    if hasattr(fused_a2a, "hybrid_ep_dispatch"):

        def hybrid_ep_dispatch(
            x,
            routing_map,
            probs,
            group,
            num_local_experts,
            num_sms_dispatch_api=24,
            num_sms_combine_api=24,
            num_permuted_tokens=None,
            pad_multiple=None,
        ):
            return PatchedHybridEPDispatch.apply(
                x,
                routing_map,
                probs,
                group,
                num_local_experts,
                num_sms_dispatch_api,
                num_sms_combine_api,
                num_permuted_tokens,
                pad_multiple,
            )

        fused_a2a.hybrid_ep_dispatch = hybrid_ep_dispatch

    if hasattr(fused_a2a, "hybrid_ep_combine"):

        def hybrid_ep_combine(x, handle, num_permuted_tokens, pad_multiple):
            return PatchedHybridEPCombine.apply(
                x, handle, num_permuted_tokens, pad_multiple
            )

        fused_a2a.hybrid_ep_combine = hybrid_ep_combine

    _patched = True
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    if rank <= 0:
        print(
            f"[hybridep_patches] Applied 5 monkey-patches to fused_a2a "
            f"(custom_allgather, buffer_realloc, needs_reset, backward_count) "
            f"debug={_debug}",
            flush=True,
        )
    logger.info("Applied HybridEP monkey-patches to fused_a2a")
