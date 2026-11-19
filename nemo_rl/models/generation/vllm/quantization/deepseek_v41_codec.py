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

"""Checkpoint-layout codecs for native V4.1 refit (before TP/EP packing)."""

import torch


def quantize_native_weight(
    weight: torch.Tensor, *, fp4: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one complete HF projection, independently of runtime kernel layouts.

    Args:
        weight: Finite floating tensor [out_features, in_features]. Both axes
            must be divisible by 32 for dense FP8; FP4 needs 32 input channels.
            Quantization first rounds to BF16, matching the training compute copy.
        fp4: Routed experts use E2M1, two adjacent columns per byte, low nibble
            first, with one E8M0 scale per row/32. Dense uses E4M3 and 32x32 scales.

    Returns:
        Values [out_features, in_features / 2] (uint8) for FP4, or the original
        shape (float8_e4m3fn) for FP8; scales [out_features / row_block,
        in_features / 32] as E8M0 bytes, where row_block is 1 or 32. Outputs own
        storage and do not mutate weight. Backend swizzling is the loader's job.
    """
    row_block = 1 if fp4 else 32
    if weight.ndim != 2 or weight.shape[1] % 32 or weight.shape[0] % row_block:
        raise ValueError(
            "Native weights require a 2D matrix aligned to the quantization blocks"
        )
    if not weight.is_floating_point():
        raise ValueError("Native refit expects floating trainer weights")
    values = weight.to(torch.bfloat16).float()
    if not torch.isfinite(values).all():
        raise ValueError("Native refit cannot encode non-finite weights")
    rows, cols = values.shape
    blocks = values.reshape(rows // row_block, row_block, cols // 32, 32)
    maximum = blocks.abs().amax(dim=(1, 3))
    limit = 6.0 if fp4 else 448.0
    # Exact IEEE754 ceiling avoids log2 rounding just above a power of two.
    scaled = (maximum / limit).clamp_min(torch.finfo(torch.float32).tiny)
    bits = scaled.contiguous().view(torch.int32)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    if (exponent > 254).any():
        raise ValueError("Native weight scale exceeds the finite E8M0 range")
    scale = (exponent << 23).view(torch.float32)
    normalized = (blocks / scale[:, None, :, None]).reshape(rows, cols)
    if not fp4:
        return normalized.clamp(-448, 448).to(torch.float8_e4m3fn), exponent.to(
            torch.uint8
        )
    magnitude = normalized.abs().contiguous()
    midpoints = magnitude.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    lower = torch.bucketize(magnitude, midpoints)
    tie = magnitude == midpoints[lower.clamp_max(6)]
    code = (lower + (tie & (lower % 2 == 1))).to(torch.uint8)
    code = code | (torch.signbit(normalized).to(torch.uint8) << 3)
    return code[:, 0::2] | (code[:, 1::2] << 4), exponent.to(torch.uint8)
