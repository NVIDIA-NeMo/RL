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

import importlib.util
from pathlib import Path

import pytest


def _load_mxfp8_utils():
    module_path = (
        Path(__file__).parents[1]
        / "nemo_rl/models/generation/vllm/quantization/mxfp8_utils.py"
    )
    spec = importlib.util.spec_from_file_location("mxfp8_utils", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("k", "expected_pad_width"),
    [
        (0, 0),
        (1, 3),
        (4, 0),
        (6, 2),
        (8, 0),
    ],
)
def test_flashinfer_scale_k_pad_width_aligns_to_multiple_of_four(
    k: int, expected_pad_width: int
) -> None:
    flashinfer_scale_k_pad_width = _load_mxfp8_utils().flashinfer_scale_k_pad_width
    assert flashinfer_scale_k_pad_width(k) == expected_pad_width


def test_flashinfer_scale_k_pad_width_rejects_negative_k() -> None:
    flashinfer_scale_k_pad_width = _load_mxfp8_utils().flashinfer_scale_k_pad_width
    with pytest.raises(ValueError, match="non-negative"):
        flashinfer_scale_k_pad_width(-1)
