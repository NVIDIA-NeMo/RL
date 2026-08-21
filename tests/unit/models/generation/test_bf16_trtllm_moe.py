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
"""CPU bit-equality tests for the refit-idempotent BF16 TRTLLM MoE shuffle.

The flashinfer permutation-index helpers are reimplemented here in pure torch
(copied from flashinfer utils.py and fused_moe/core.py, Apache-2.0), so the
tests run without CUDA, vLLM, or flashinfer. The code under test is the
torch-only permutation composition in bf16_trtllm_moe_utils.py: the composed
batched gather must match vLLM's per-expert
swap_w13_to_w31 -> permute -> convert_to_block_layout pipeline bit-exactly,
and reprocessing the same parameter storage across refit cycles must be
idempotent.
"""

import pytest
import torch

from nemo_rl.models.generation.vllm.quantization.bf16_trtllm_moe_utils import (
    BLOCK_K,
    EPILOGUE_TILE_M,
    block_layout_alias,
    gather_rows_to_block_layout,
    swap_w13_to_w31_row_indices,
)

# fmt: off
_SRC_TO_DST_BLK16_ROW_MAP = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
_SRC_TO_DST_BLK32_ROW_MAP = [
    0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27,
    4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31,
]
# fmt: on

# (is_gated, num_experts, intermediate_size, hidden_size)
_SHUFFLE_CASES = [
    # Qwen3-30B-A3B-like TP1 shard (silu gated).
    (True, 4, 768, 2048),
    # Small gated.
    (True, 3, 128, 512),
    # Non-gated (relu2-style).
    (False, 3, 256, 512),
]


def _shuffle_matrix_a_row_indices(
    input_tensor: torch.Tensor, epilogue_tile_m: int
) -> torch.Tensor:
    assert input_tensor.dim() == 2
    num_rows = input_tensor.shape[0]
    shuffle_block_size = 32 if epilogue_tile_m % 128 == 0 else 16
    row_map = (
        _SRC_TO_DST_BLK16_ROW_MAP
        if shuffle_block_size == 16
        else _SRC_TO_DST_BLK32_ROW_MAP
    )
    assert num_rows % shuffle_block_size == 0
    old_rows = torch.arange(num_rows, dtype=torch.long)
    mapped_rows = torch.tensor(row_map, dtype=torch.long)[old_rows % shuffle_block_size]
    new_rows = (old_rows // shuffle_block_size) * shuffle_block_size + mapped_rows
    row_indices = torch.empty(num_rows, dtype=torch.long)
    row_indices[new_rows] = old_rows
    return row_indices


def _reorder_rows_for_gated_act_gemm_row_indices(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2
    num_rows = x.shape[0]
    assert num_rows % 2 == 0
    row_indices = torch.arange(num_rows, dtype=torch.long)
    permuted = torch.empty_like(row_indices)
    permuted[0::2] = row_indices[: num_rows // 2]
    permuted[1::2] = row_indices[num_rows // 2 :]
    return permuted


def _w3_w1_permute_indices(
    dst: torch.Tensor, epilogue_tile_m: int, is_gated_act_gemm: bool
) -> torch.Tensor:
    """flashinfer _maybe_get_cached_w3_w1_permute_indices without the cache."""
    if is_gated_act_gemm:
        p0 = _reorder_rows_for_gated_act_gemm_row_indices(dst)
    else:
        p0 = torch.arange(dst.shape[0], dtype=torch.long)
    p1 = _shuffle_matrix_a_row_indices(dst, epilogue_tile_m)
    return p0[p1]


def _w2_permute_indices(dst: torch.Tensor, epilogue_tile_m: int) -> torch.Tensor:
    """flashinfer get_w2_permute_indices_with_cache without the cache."""
    return _shuffle_matrix_a_row_indices(dst, epilogue_tile_m)


def _convert_to_block_layout(input_tensor: torch.Tensor, block_k: int) -> torch.Tensor:
    num_rows, k = input_tensor.shape
    assert k % block_k == 0
    return (
        input_tensor.view(num_rows, k // block_k, block_k).permute(1, 0, 2).contiguous()
    )


def _swap_w13_to_w31(x: torch.Tensor) -> torch.Tensor:
    return (
        x.reshape(-1, 2, x.shape[-2] // 2, x.shape[-1]).flip(dims=[1]).reshape(x.shape)
    )


def _reference_pipeline(
    w13: torch.Tensor, w2: torch.Tensor, is_gated: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM convert_to_unquantized_kernel_format, FLASHINFER_TRTLLM branch."""
    if is_gated:
        w13 = _swap_w13_to_w31(w13)
    w13_out, w2_out = [], []
    for expert in range(w13.shape[0]):
        p13 = _w3_w1_permute_indices(
            w13[expert].view(torch.uint8), EPILOGUE_TILE_M, is_gated_act_gemm=is_gated
        )
        t13 = w13[expert].clone().view(torch.uint8)[p13].contiguous()
        p2 = _w2_permute_indices(w2[expert].view(torch.uint8), EPILOGUE_TILE_M)
        t2 = w2[expert].clone().view(torch.uint8)[p2].contiguous()
        w13_out.append(_convert_to_block_layout(t13, BLOCK_K).view(torch.bfloat16))
        w2_out.append(_convert_to_block_layout(t2, BLOCK_K).view(torch.bfloat16))
    return torch.stack(w13_out).contiguous(), torch.stack(w2_out).contiguous()


def _composed_permutations(
    w13: torch.Tensor, w2: torch.Tensor, is_gated: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """The composed row permutations of _bf16_trtllm_row_permutations."""
    perm_w13 = _w3_w1_permute_indices(
        w13[0].view(torch.uint8), EPILOGUE_TILE_M, is_gated_act_gemm=is_gated
    )
    if is_gated:
        # vLLM applies the flashinfer permutation to the already-swapped
        # [w3; w1] tensor; fold the swap into the composed gather.
        perm_w13 = swap_w13_to_w31_row_indices(w13.shape[1])[perm_w13]
    perm_w2 = _w2_permute_indices(w2[0].view(torch.uint8), EPILOGUE_TILE_M)
    return perm_w13, perm_w2


def _random_moe_weights(
    is_gated: bool, num_experts: int, intermediate_size: int, hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    w13_rows = (2 if is_gated else 1) * intermediate_size
    w13 = torch.randn(num_experts, w13_rows, hidden_size, dtype=torch.bfloat16)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16)
    return w13, w2


@pytest.mark.parametrize(
    "is_gated,num_experts,intermediate_size,hidden_size", _SHUFFLE_CASES
)
def test_batched_shuffle_matches_per_expert(
    is_gated, num_experts, intermediate_size, hidden_size
):
    """Bitwise parity of the composed batched gather with vLLM's per-expert pipeline."""
    torch.manual_seed(0)
    w13, w2 = _random_moe_weights(is_gated, num_experts, intermediate_size, hidden_size)

    ref13, ref2 = _reference_pipeline(w13, w2, is_gated)
    perm_w13, perm_w2 = _composed_permutations(w13, w2, is_gated)
    got13_u8, got2_u8 = gather_rows_to_block_layout(w13, w2, perm_w13, perm_w2)

    assert got13_u8.shape == ref13.view(torch.uint8).shape
    assert got2_u8.shape == ref2.view(torch.uint8).shape
    assert torch.equal(got13_u8, ref13.view(torch.uint8)), f"w13 gated={is_gated}"
    assert torch.equal(got2_u8, ref2.view(torch.uint8)), f"w2 gated={is_gated}"


@pytest.mark.parametrize("is_gated", [True, False])
def test_block_layout_alias_shares_storage(is_gated):
    """The 4D apply alias must write through to the checkpoint param's bytes."""
    torch.manual_seed(0)
    w13, w2 = _random_moe_weights(is_gated, 3, 128, 512)
    param = torch.nn.Parameter(w13.clone(), requires_grad=False)

    alias = block_layout_alias(param)
    assert alias.data_ptr() == param.data_ptr()

    ref13, _ = _reference_pipeline(w13, w2, is_gated)
    alias.view(torch.uint8).copy_(ref13.view(torch.uint8))
    assert torch.equal(alias.view(torch.uint8), ref13.view(torch.uint8))
    assert torch.equal(
        param.data.view(torch.uint8).flatten(), ref13.view(torch.uint8).flatten()
    )


@pytest.mark.parametrize(
    "is_gated,num_experts,intermediate_size,hidden_size",
    [(True, 4, 768, 2048), (False, 3, 256, 512)],
)
def test_two_cycle_refit_idempotency(
    is_gated, num_experts, intermediate_size, hidden_size
):
    """Load -> process cycles on the same param storage stay bit-exact.

    Simulates the exact op sequence of process_weights_after_loading
    (gather from checkpoint bytes into scratch, overwrite the aliased storage
    with the block layout) across an initial load, a refit with new weights,
    and a refit that reloads the same weights, checking against the
    from-scratch reference each time (a double-shuffle would diverge).
    """
    torch.manual_seed(1)
    ckpt1_w13, ckpt1_w2 = _random_moe_weights(
        is_gated, num_experts, intermediate_size, hidden_size
    )
    ckpt2_w13, ckpt2_w2 = _random_moe_weights(
        is_gated, num_experts, intermediate_size, hidden_size
    )

    # Persistent param storage (initial load).
    w13_param = torch.nn.Parameter(ckpt1_w13.clone(), requires_grad=False)
    w2_param = torch.nn.Parameter(ckpt1_w2.clone(), requires_grad=False)
    ptr13, ptr2 = w13_param.data_ptr(), w2_param.data_ptr()

    # Permutations computed once at first load, exactly as the module does.
    perm_w13, perm_w2 = _composed_permutations(w13_param.data, w2_param.data, is_gated)

    aliases = {}

    def process(first_load: bool) -> None:
        w13_blocked, w2_blocked = gather_rows_to_block_layout(
            w13_param.data, w2_param.data, perm_w13, perm_w2
        )
        if first_load:
            aliases["w13"] = block_layout_alias(w13_param)
            aliases["w2"] = block_layout_alias(w2_param)
        aliases["w13"].view(torch.uint8).copy_(w13_blocked)
        aliases["w2"].view(torch.uint8).copy_(w2_blocked)

    def check(ref13: torch.Tensor, ref2: torch.Tensor, cycle: str) -> None:
        assert torch.equal(aliases["w13"].view(torch.uint8), ref13.view(torch.uint8)), (
            f"{cycle} w13"
        )
        assert torch.equal(aliases["w2"].view(torch.uint8), ref2.view(torch.uint8)), (
            f"{cycle} w2"
        )
        assert aliases["w13"].data_ptr() == ptr13
        assert aliases["w2"].data_ptr() == ptr2

    # Cycle 1: initial load + process.
    process(first_load=True)
    ref13, ref2 = _reference_pipeline(ckpt1_w13, ckpt1_w2, is_gated)
    check(ref13, ref2, "cycle1")

    # Cycle 2: refit loads new checkpoint-layout bytes in place, then process.
    w13_param.data.copy_(ckpt2_w13)
    w2_param.data.copy_(ckpt2_w2)
    assert w13_param.data_ptr() == ptr13 and w2_param.data_ptr() == ptr2
    process(first_load=False)
    ref13, ref2 = _reference_pipeline(ckpt2_w13, ckpt2_w2, is_gated)
    check(ref13, ref2, "cycle2 (double-shuffle?)")

    # Cycle 3: reloading the same checkpoint must reproduce identical bytes.
    w13_param.data.copy_(ckpt2_w13)
    w2_param.data.copy_(ckpt2_w2)
    process(first_load=False)
    check(ref13, ref2, "cycle3")
