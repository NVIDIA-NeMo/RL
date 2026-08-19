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
from typing import Callable, Optional

import torch

_NRL_LOGGER = logging.getLogger(__name__)

_ZERO_KL_EAGER_MOE_MSG = (
    "zero_train_gen_mismatch requires eager MoE routing/combine; set "
    "policy.megatron_cfg.moe_permute_fusion=false and disable fused router paths."
)

_NRL_UNPERMUTE_PATH_SEEN: set[str] = set()
_UNPERMUTE_ORIG: Optional[Callable[..., torch.Tensor]] = None
_TOKEN_DISPATCHER_UNPERMUTE_ORIG: Optional[Callable[..., torch.Tensor]] = None
_MOE_UNPERMUTE_PATCHED = False
_NRL_DET_COMBINE_BANNER = False


def _nrl_log_unpermute_path(path: str) -> None:
    if path not in _NRL_UNPERMUTE_PATH_SEEN:
        _NRL_UNPERMUTE_PATH_SEEN.add(path)
        _NRL_LOGGER.warning("[moe-combine] unpermute executed via '%s'", path)


def _unpermute_fixed_order_combine(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
) -> torch.Tensor:
    """Sum expert outputs per token in stable (permute) order via ``[T, max_slots, H].sum(1)``.

    Avoids atomic ``scatter_add_`` / ``index_add_``. Works for standard top-k routing
    and decode ``drop_and_pad`` (variable rows per token).
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
    """``moe_utils.unpermute`` with fixed-order deterministic combine."""
    global _NRL_DET_COMBINE_BANNER
    del pad_offsets  # unused; same combine for train and generation
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
            "[NRL_DET_COMBINE] fixed-order combine ACTIVE (train + generation)",
            flush=True,
        )

    _nrl_log_unpermute_path("fixed_order_combine")
    output_tokens = _unpermute_fixed_order_combine(
        permuted_tokens, sorted_indices, restore_shape
    )
    return output_tokens.to(dtype=input_dtype)


def apply_moe_unpermute_determinism_patch() -> None:
    """Patch ``moe_utils.unpermute`` (+ cached token_dispatcher import site)."""
    global _UNPERMUTE_ORIG, _TOKEN_DISPATCHER_UNPERMUTE_ORIG, _MOE_UNPERMUTE_PATCHED
    if _MOE_UNPERMUTE_PATCHED:
        return
    try:
        import megatron.core.transformer.moe.moe_utils as moe_utils
        import megatron.core.transformer.moe.token_dispatcher as token_dispatcher
    except ImportError:
        print(
            "moe_zero_kl_patches: Megatron MoE modules are not importable; "
            "skipping deterministic combine patch."
        )
        return

    _UNPERMUTE_ORIG = moe_utils.unpermute
    _TOKEN_DISPATCHER_UNPERMUTE_ORIG = token_dispatcher.unpermute
    moe_utils.unpermute = _patched_unpermute
    token_dispatcher.unpermute = _patched_unpermute
    _MOE_UNPERMUTE_PATCHED = True
    print(
        "[moe_zero_kl_patches] patched moe_utils.unpermute and "
        "token_dispatcher.unpermute (fixed-order combine)."
    )


def apply_moe_determinism_patches() -> None:
    """Apply the deterministic MoE combine patch."""
    apply_moe_unpermute_determinism_patch()


def restore_moe_determinism_patches() -> None:
    """Restore Megatron unpermute entry points patched by this module (for tests)."""
    global _MOE_UNPERMUTE_PATCHED, _NRL_DET_COMBINE_BANNER
    global _UNPERMUTE_ORIG, _TOKEN_DISPATCHER_UNPERMUTE_ORIG

    if _MOE_UNPERMUTE_PATCHED and _UNPERMUTE_ORIG is not None:
        import megatron.core.transformer.moe.moe_utils as moe_utils
        import megatron.core.transformer.moe.token_dispatcher as token_dispatcher

        moe_utils.unpermute = _UNPERMUTE_ORIG
        if _TOKEN_DISPATCHER_UNPERMUTE_ORIG is not None:
            token_dispatcher.unpermute = _TOKEN_DISPATCHER_UNPERMUTE_ORIG
        _UNPERMUTE_ORIG = None
        _TOKEN_DISPATCHER_UNPERMUTE_ORIG = None
        _MOE_UNPERMUTE_PATCHED = False

    _NRL_UNPERMUTE_PATH_SEEN.clear()
    _NRL_DET_COMBINE_BANNER = False
