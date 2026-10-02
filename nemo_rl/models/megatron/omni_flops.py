# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen2.5-Omni vision work not represented by Bridge's generic ViT estimate."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import Qwen2_5OmniVisionEncoderConfig


def qwen25_omni_vision_flops(
    config: "Qwen2_5OmniVisionEncoderConfig", grid_thw: torch.Tensor
) -> int:
    """Count trainable encoder matrix/convolution FLOPs for raw image inputs.

    A multiply-add counts as two FLOPs. Count forward and backward, excluding
    elementwise operations, padding outside attention sequences, and activation
    recomputation. Raw pixels do not require input gradients. All encoder and
    merger weights must be trainable; the adapter checks freeze/PEFT settings.
    """
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
        raise ValueError("Omni vision grids must have shape [media, 3]")
    if grid_thw.dtype not in (torch.int32, torch.int64):
        raise ValueError("Omni vision grids must contain integers")
    merge = config.spatial_merge_size
    patch = config.patch_size
    temporal_patch = config.temporal_patch_size
    if not isinstance(patch, int) or not isinstance(temporal_patch, int):
        raise NotImplementedError("Omni FLOPs require scalar patch dimensions")
    if min(merge, patch, temporal_patch, config.window_size) <= 0:
        raise ValueError("Omni patch, merge, and window sizes must be positive")
    # HF partitions the spatially merged grid into windows, then expands each
    # window back into patches for attention. Edge windows contain fewer patches.
    window = config.window_size // merge // patch * merge
    if window == 0:
        raise ValueError("Omni attention window must contain a merged patch")
    full_layers = set(config.fullatt_block_indexes)
    if any(index < 0 or index >= config.depth for index in full_layers):
        raise ValueError("Omni full-attention layer index is outside the encoder")
    patches = full_pairs = window_pairs = 0
    for frames, height, width in grid_thw.tolist():
        if min(frames, height, width) <= 0 or height % merge or width % merge:
            raise ValueError("Omni grids must be positive and spatially mergeable")
        patches += frames * height * width
        full_pairs += frames * (height * width) ** 2
        h_windows, h_tail = divmod(height, window)
        w_windows, w_tail = divmod(width, window)
        window_pairs += (
            frames
            * (h_windows * window**2 + h_tail**2)
            * (w_windows * window**2 + w_tail**2)
        )

    hidden = config.hidden_size
    projections = (
        config.depth * patches * (8 * hidden**2 + 6 * hidden * config.intermediate_size)
    )
    attention = (
        4
        * hidden
        * (
            len(full_layers) * full_pairs
            + (config.depth - len(full_layers)) * window_pairs
        )
    )
    merged_hidden = hidden * merge**2
    merger = (
        patches
        // merge**2
        * (2 * merged_hidden**2 + 2 * merged_hidden * config.out_hidden_size)
    )
    embedding = 2 * patches * config.in_channels * temporal_patch * patch**2 * hidden
    # Internal matmuls need both input and weight gradients. The patch convolution
    # only needs weight gradients because its inputs are raw pixels.
    return 3 * (projections + attention + merger) + 2 * embedding
