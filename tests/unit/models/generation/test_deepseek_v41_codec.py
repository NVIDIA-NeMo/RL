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

"""Native checkpoint encodings; independent E2M1 decoding and block invariants."""

import importlib.util
from pathlib import Path

import pytest
import torch

# Load the Torch-only codec without importing the Ray/vLLM generation package.
_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "nemo_rl/models/generation/vllm/quantization/deepseek_v41_codec.py"
)
_SPEC = importlib.util.spec_from_file_location("deepseek_v41_codec", _SOURCE)
_CODEC = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CODEC)
quantize_native_weight = _CODEC.quantize_native_weight


def test_fp4_exact_nibbles_ties_and_signed_zero():
    values = torch.tensor(
        [
            [
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ]
            * 2
        ]
    )
    packed, scales = quantize_native_weight(values, fp4=True)
    assert packed.tolist() == [[0x20, 0x42, 0x64, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2]
    assert scales.tolist() == [[127]]


@pytest.mark.parametrize("fp4", [False, True])
def test_zero_blocks_have_finite_nonzero_scale_bytes(fp4):
    encoded, scales = quantize_native_weight(torch.zeros(32, 64), fp4=fp4)
    assert torch.count_nonzero(encoded.float()) == 0
    assert torch.all(scales == 1)


def test_dense_scales_share_32_rows_but_expert_scales_do_not():
    values = torch.zeros(64, 64)
    values[0, 0], values[32, 32] = 448.0, 896.0
    encoded, scales = quantize_native_weight(values, fp4=False)
    assert scales.tolist() == [[127, 1], [1, 128]]
    assert encoded[0, 0].item() == 448.0
    assert encoded[32, 32].item() == 448.0
    values[0, 0], values[32, 32] = 6.0, 12.0
    _, scales = quantize_native_weight(values, fp4=True)
    assert scales[0, 0] == 127 and scales[1, 0] == 1 and scales[32, 1] == 128


@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_weights_are_rejected(fp4, bad):
    values = torch.zeros(32, 32)
    values[0, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        quantize_native_weight(values, fp4=fp4)


@pytest.mark.parametrize(
    "shape,fp4", [((32, 33), True), ((31, 32), False), ((2, 32, 32), True)]
)
def test_incomplete_blocks_are_rejected(shape, fp4):
    with pytest.raises(ValueError, match="aligned"):
        quantize_native_weight(torch.zeros(shape), fp4=fp4)
