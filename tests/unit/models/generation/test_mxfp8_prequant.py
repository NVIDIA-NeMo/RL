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

import pytest
import torch

from nemo_rl.models.generation.vllm.quantization.fp8_train_utils import (
    MXFP8_BLOCK_SIZE,
    _mxfp8_e4m3_quantize_torch,
    mxfp8_e4m3_quantize_for_refit,
)


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
    rel_err = (x_dq - x.to(torch.float32)).abs() / x.to(torch.float32).abs().clamp(
        min=1e-3
    )
    # e4m3 has 3 mantissa bits; block-pow2 scaling keeps values in normal range.
    assert rel_err.median() < 0.05
    assert rel_err.max() < 0.25


def test_last_dim_not_divisible_raises():
    x = torch.randn(8, MXFP8_BLOCK_SIZE + 1, dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        _mxfp8_e4m3_quantize_torch(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_refit_quantize_matches_receiver_path():
    """Bitwise parity with the vLLM receiver path (mxfp8_e4m3_quantize + squeeze)."""
    vllm_mxfp8 = pytest.importorskip(
        "vllm.model_executor.layers.quantization.utils.mxfp8_utils"
    )

    torch.manual_seed(0)
    x = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")

    ref_lp, ref_scale = vllm_mxfp8.mxfp8_e4m3_quantize(x)
    ref_scale = torch.squeeze(ref_scale, dim=-1)

    got_lp, got_scale = mxfp8_e4m3_quantize_for_refit(x)

    assert got_lp.dtype == ref_lp.dtype
    assert torch.equal(got_lp.view(torch.uint8), ref_lp.view(torch.uint8))
    assert got_scale.dtype == ref_scale.dtype
    assert got_scale.shape == ref_scale.shape
    assert torch.equal(got_scale.reshape(-1), ref_scale.reshape(-1))
