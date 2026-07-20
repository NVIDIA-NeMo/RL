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
"""Torch-only layout helpers for the refit-idempotent BF16 TRTLLM MoE path.

Split from bf16_trtllm_moe.py (which imports vllm and ray at module scope)
following the fp8_train_utils.py pattern, so the permutation-composition
pipeline is importable and unit-testable on CPU without vLLM, flashinfer,
or CUDA.
"""

import torch

BLOCK_K = 128
EPILOGUE_TILE_M = 128

# Shared gather destinations keyed by (tag, shape, device). They persist across
# refits so the batched shuffle allocates nothing after the first pass; their
# contents are rewritten on every call, so a sleep-mode discard is harmless.
bf16_trtllm_scratch_buffers: dict[
    tuple[str, tuple[int, ...], torch.device], torch.Tensor
] = {}


def _scratch(tag: str, shape: torch.Size, device: torch.device) -> torch.Tensor:
    key = (tag, tuple(shape), device)
    buf = bf16_trtllm_scratch_buffers.get(key)
    if buf is None:
        buf = torch.empty(shape, dtype=torch.uint8, device=device)
        bf16_trtllm_scratch_buffers[key] = buf
    return buf


def swap_w13_to_w31_row_indices(num_rows: int) -> torch.Tensor:
    half = num_rows // 2
    return torch.cat((torch.arange(half, num_rows), torch.arange(half)))


def block_layout_scratch_view(x_u8: torch.Tensor) -> torch.Tensor:
    """(E, M, K_bytes) row-gathered uint8 -> (E, K_bytes/BLOCK_K, M, BLOCK_K).

    Pure view/permute; batched equivalent of flashinfer's per-expert
    convert_to_block_layout.
    """
    num_experts, num_rows, k_bytes = x_u8.shape
    assert k_bytes % BLOCK_K == 0, "K bytes must be divisible by BLOCK_K"
    return x_u8.view(num_experts, num_rows, k_bytes // BLOCK_K, BLOCK_K).permute(
        0, 2, 1, 3
    )


def block_layout_alias(param: torch.nn.Parameter) -> torch.Tensor:
    """4D block-layout bf16 view sharing the checkpoint param's storage.

    The checkpoint-layout parameter stays the refit load target; the kernel
    consumes this alias. Sharing storage keeps memory flat and the data_ptr
    stable for CUDA graphs; each processing pass first gathers the freshly
    loaded checkpoint bytes into scratch, then overwrites the storage with the
    block layout through this view.
    """
    num_experts, num_rows, cols = param.shape
    k_bytes = cols * param.element_size()
    flat = param.data.view(torch.uint8).reshape(num_experts, num_rows * k_bytes)
    return flat.view(num_experts, k_bytes // BLOCK_K, num_rows, BLOCK_K).view(
        torch.bfloat16
    )


def gather_rows_to_block_layout(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    perm_w13: torch.Tensor,
    perm_w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-permute both stacked expert weights as one gather per tensor.

    Returns block-layout uint8 views over the shared scratch buffers; callers
    copy_ them into the persistent destinations.
    """
    w13_u8 = w13_weight.view(torch.uint8)
    w2_u8 = w2_weight.view(torch.uint8)
    w13_gathered = torch.index_select(
        w13_u8, 1, perm_w13, out=_scratch("w13", w13_u8.shape, w13_u8.device)
    )
    w2_gathered = torch.index_select(
        w2_u8, 1, perm_w2, out=_scratch("w2", w2_u8.shape, w2_u8.device)
    )
    return block_layout_scratch_view(w13_gathered), block_layout_scratch_view(
        w2_gathered
    )
