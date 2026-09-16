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

from typing import Any


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def flashinfer_scale_k_pad_width(k: int) -> int:
    if k < 0:
        raise ValueError("MXFP8 scale K dimension must be non-negative")
    return (-k) % 4


def pad_flashinfer_scale_k(input_tensor: Any) -> Any:
    pad_width = flashinfer_scale_k_pad_width(input_tensor.shape[-1])
    if pad_width == 0:
        return input_tensor

    padded_shape = (*input_tensor.shape[:-1], input_tensor.shape[-1] + pad_width)
    padded = input_tensor.new_zeros(padded_shape)
    padded[..., : input_tensor.shape[-1]] = input_tensor
    return padded


def flashinfer_mxfp8_moe_padding_plan(
    hidden_size: int, intermediate_size: int
) -> tuple[int, int]:
    if hidden_size <= 0:
        raise ValueError("MXFP8 MoE hidden_size must be positive")
    if hidden_size % 32 != 0:
        raise ValueError(
            "FlashInfer TRTLLM MXFP8 MoE requires hidden_size divisible by 32, "
            f"got {hidden_size}."
        )
    if intermediate_size <= 0:
        raise ValueError("MXFP8 MoE intermediate_size must be positive")
    if intermediate_size % 32 != 0:
        raise ValueError(
            "FlashInfer TRTLLM MXFP8 MoE requires intermediate_size divisible "
            f"by 32, got {intermediate_size}."
        )

    return _round_up(hidden_size, 512), _round_up(intermediate_size, 128)
