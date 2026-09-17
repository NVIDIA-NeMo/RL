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

import sys

import pytest
import torch

from nemo_rl.models.generation.vllm.quantization.fp8_train_utils import (
    MXFP8_BLOCK_SIZE,
    _mxfp8_e4m3_quantize_torch,
    mxfp8_e4m3_quantize_for_refit,
)

pytestmark = pytest.mark.vllm


def _dequantize(x_fp8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    num_blocks = x_fp8.shape[-1] // MXFP8_BLOCK_SIZE
    x_blocked = x_fp8.to(torch.float32).view(
        *x_fp8.shape[:-1], num_blocks, MXFP8_BLOCK_SIZE
    )
    descale = torch.exp2(scales.to(torch.float32) - 127.0)
    return (x_blocked * descale.unsqueeze(-1)).view(*x_fp8.shape)


@pytest.mark.parametrize("shape", [(64, 128), (7, 96), (4, 16, 64)])
def test_torch_reference_shapes_and_roundtrip(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.bfloat16)

    x_fp8, scales = _mxfp8_e4m3_quantize_torch(x)

    assert x_fp8.shape == x.shape
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.uint8
    expected_scale_shape = (*shape[:-1], shape[-1] // MXFP8_BLOCK_SIZE)
    assert tuple(scales.shape) == expected_scale_shape

    x_dq = _dequantize(x_fp8, scales)
    x32 = x.to(torch.float32)
    abs_err = (x_dq - x32).abs()
    block_amax = (
        x32.abs()
        .reshape(*shape[:-1], shape[-1] // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
        .amax(dim=-1, keepdim=True)
        .expand(*shape[:-1], shape[-1] // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
        .reshape(shape)
    )
    # e4m3 has 3 mantissa bits: elements within the block's representable range
    # must round-trip to ~12.5% relative error; elements far below the block
    # amax may legitimately quantize to zero, so bound them by absolute error
    # instead of a ratio (keeps the test deterministic across torch versions).
    representable = x32.abs() >= block_amax / 64
    rel_err = (abs_err / x32.abs().clamp(min=1e-6))[representable]
    assert rel_err.median() < 0.05
    assert rel_err.max() < 0.25
    assert (abs_err[~representable] <= block_amax[~representable] / 32).all()


def test_last_dim_not_divisible_raises():
    x = torch.randn(8, MXFP8_BLOCK_SIZE + 1, dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        _mxfp8_e4m3_quantize_torch(x)


def test_refit_quantize_preserves_single_scale_block_dimension():
    x = torch.randn(8, MXFP8_BLOCK_SIZE, dtype=torch.bfloat16)

    _, scales = mxfp8_e4m3_quantize_for_refit(x)

    assert scales.shape == (8, 1)


def test_blackwell_refit_prequantization_requires_flashinfer(monkeypatch):
    class FakeBlackwellTensor:
        is_cuda = True
        device = "cuda"

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 0))
    monkeypatch.setitem(sys.modules, "flashinfer", None)

    with pytest.raises(RuntimeError, match=r"sm100\+ requires FlashInfer"):
        mxfp8_e4m3_quantize_for_refit(FakeBlackwellTensor())


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason=(
        "requires sm100+; below it both sides fall back to the shared torch "
        "reference and the comparison is vacuous"
    ),
)
def test_refit_quantize_matches_receiver_path():
    """Bitwise parity with the vLLM receiver path (mxfp8_e4m3_quantize + squeeze)."""
    vllm_mxfp8 = pytest.importorskip(
        "vllm.model_executor.layers.quantization.utils.mxfp8_utils"
    )

    torch.manual_seed(0)
    x = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
    x[0].zero_()

    ref_lp, ref_scale = vllm_mxfp8.mxfp8_e4m3_quantize(x)
    ref_scale = torch.squeeze(ref_scale, dim=-1)
    assert torch.any(ref_scale == 0)
    ref_scale = torch.where(ref_scale == 0, torch.ones_like(ref_scale), ref_scale)

    got_lp, got_scale = mxfp8_e4m3_quantize_for_refit(x)

    assert got_lp.dtype == ref_lp.dtype
    assert torch.equal(got_lp.view(torch.uint8), ref_lp.view(torch.uint8))
    assert got_scale.dtype == ref_scale.dtype
    assert got_scale.shape == ref_scale.shape
    assert torch.equal(got_scale.reshape(-1), ref_scale.reshape(-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_refit_quantize_matches_receiver_quantize_mxfp8_weight():
    """Sender prequantization and the receiver helper must agree bit-for-bit.

    The trainer streams E4M3 data + *_scale_from_checkpoint produced by
    mxfp8_e4m3_quantize_for_refit; weights the receiver quantizes itself go
    through quantize_mxfp8_weight. Refit correctness relies on the two
    implementations producing identical bits for the same input.
    """
    from nemo_rl.models.generation.vllm.quantization.fp8 import quantize_mxfp8_weight

    torch.manual_seed(0)
    x = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
    x[0].zero_()

    recv_lp, recv_scale = quantize_mxfp8_weight(x)
    sent_lp, sent_scale = mxfp8_e4m3_quantize_for_refit(x)

    assert sent_lp.dtype == recv_lp.dtype
    assert torch.equal(sent_lp.view(torch.uint8), recv_lp.view(torch.uint8))
    assert sent_scale.dtype == recv_scale.dtype
    assert sent_scale.shape == recv_scale.shape
    assert torch.equal(sent_scale, recv_scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "is_gated,intermediate_size,hidden_size",
    [
        # Aligned: both scale K dims (hidden/32=8, intermediate/32=4) are %4.
        (True, 128, 256),
        # w2 scale K = 192/32 = 6, so pad_flashinfer_scale_k pads it to 8.
        (True, 192, 128),
        # Non-gated (single w13 shard), aligned.
        (False, 128, 256),
    ],
)
def test_batched_moe_shuffle_matches_per_expert(
    is_gated, intermediate_size, hidden_size
):
    """Bitwise parity of the batched TRTLLM MoE shuffle with the per-expert loop."""
    pytest.importorskip("flashinfer")
    fp8 = pytest.importorskip("nemo_rl.models.generation.vllm.quantization.fp8")

    from types import SimpleNamespace

    torch.manual_seed(0)
    num_experts = 4
    w13_rows = (2 if is_gated else 1) * intermediate_size

    def rand_bytes(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")

    w13_weight = rand_bytes(num_experts, w13_rows, hidden_size).view(
        torch.float8_e4m3fn
    )
    w2_weight = rand_bytes(num_experts, hidden_size, intermediate_size).view(
        torch.float8_e4m3fn
    )
    w13_scale = rand_bytes(num_experts, w13_rows, hidden_size // MXFP8_BLOCK_SIZE)
    w2_scale = rand_bytes(
        num_experts, hidden_size, intermediate_size // MXFP8_BLOCK_SIZE
    )

    layer = SimpleNamespace()  # holds the cached row permutations
    epilogue_tile_m = 128
    batched = fp8._shuffle_mxfp8_moe_batched(
        layer, w13_weight, w2_weight, w13_scale, w2_scale, is_gated, epilogue_tile_m
    )
    reference = fp8._shuffle_mxfp8_moe_per_expert(
        w13_weight, w2_weight, w13_scale, w2_scale, is_gated, epilogue_tile_m
    )

    for got, want, name in zip(
        batched, reference, ("w13_weight", "w2_weight", "w13_scale", "w2_scale")
    ):
        assert got.shape == want.shape, name
        assert got.dtype == want.dtype, name
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8)), name
