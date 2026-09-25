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

"""Bridge FLOPs estimates for a worker's unsharded, local data batch."""

import math
from typing import TYPE_CHECKING, Any

import torch

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer


def compute_bridge_batch_flops(
    config: "ConfigContainer", data: BatchedDataDict[Any]
) -> float:
    """Estimate full-model training work for one DP shard, before TP/CP slicing.

    Count real input tokens, including prompts, irrespective of the loss mask.
    Packing does not join independent sequences for attention accounting.
    NotImplementedError means the caller may use its explicit fallback;
    malformed inputs and unexpected calculator errors must propagate.
    """
    # Bridge is installed only in Megatron worker environments.
    from megatron.bridge.training.utils import flop_utils

    model = config.model
    if any(
        getattr(model, name, False)
        for name in (
            "freeze_language_model",
            "freeze_vision_model",
            "freeze_vision_projection",
        )
    ):
        raise NotImplementedError("Bridge FLOPs for partially frozen models")
    if config.peft is not None:
        raise NotImplementedError("Bridge FLOPs for NeMo-RL PEFT batches")
    if hasattr(model, "_get_num_floating_point_operations") and not hasattr(
        model, "_get_num_floating_point_operations_with_runtime_stats"
    ):
        raise NotImplementedError(
            "Bridge model estimator does not accept runtime lengths"
        )
    if any(key in data for key in ("input_features", "audio_features", "audio_signal")):
        raise NotImplementedError("Bridge FLOPs for audio batches")

    lengths = data["input_lengths"].to(dtype=torch.int64)
    if lengths.ndim != 1 or len(lengths) != data.size or bool((lengths <= 0).any()):
        raise ValueError(
            "FLOPs input_lengths must contain one positive length per sample"
        )
    total = flop_utils.num_floating_point_operations(
        config,
        batch_size=data.size,
        seqlen_sum=int(lengths.sum().item()),
        seqlen_squared_sum=int(lengths.square().sum().item()),
        num_vision_patches=0,
    )
    for pixels_key, grid_key in (
        ("pixel_values", "image_grid_thw"),
        ("pixel_values_videos", "video_grid_thw"),
    ):
        grid = data.get(grid_key)
        if isinstance(grid, PackedTensor):
            grid = grid.as_tensor()
        if grid is None:
            pixels = data.get(pixels_key)
            tensors = pixels.tensors if isinstance(pixels, PackedTensor) else [pixels]
            if any(tensor is not None and tensor.numel() > 0 for tensor in tensors):
                raise NotImplementedError(f"Bridge vision FLOPs require {grid_key}")
            continue
        if grid.numel() == 0:
            continue
        vision_config = getattr(model, "vision_config", None) or getattr(
            getattr(model, "thinker_config", None), "vision_config", None
        )
        if vision_config is None:
            raise NotImplementedError("Bridge vision FLOPs require a vision config")
        total += flop_utils.vit_flops_from_grid_thw(config, grid)
    total = float(total)
    if not math.isfinite(total) or total < 0:
        raise ValueError(f"Bridge returned invalid training FLOPs: {total}")
    return total
