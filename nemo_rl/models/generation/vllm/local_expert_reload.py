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

"""Loading already-resharded BF16 experts into active checkpoint storage."""

from typing import Literal

import torch

ExpertProjection = Literal["gate_proj", "up_proj", "down_proj"]


def copy_local_bf16_expert(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    *,
    projection: ExpertProjection,
    gated: bool,
) -> None:
    """Copy one grouped local projection; leave other projections untouched.

    Both tensors use checkpoint orientation, never TRTLLM packed orientation.
    Reshard has already selected TP and EP regions. Zeroing and one payload
    copy are deliberately inside this loader so both run on real replay after
    vLLM's meta accounting pass, without counting padding as extra copies.
    """
    if param.dtype != torch.bfloat16 or loaded_weight.dtype != torch.bfloat16:
        raise ValueError("Local BF16 expert loading requires BF16 tensors")
    if param.ndim != 3 or loaded_weight.ndim != 3:
        raise ValueError("Local BF16 expert loading requires 3D checkpoint tensors")
    if projection not in ("gate_proj", "up_proj", "down_proj"):
        raise ValueError(f"Unsupported expert projection: {projection!r}")
    if projection == "gate_proj" and not gated:
        raise ValueError("A non-gated expert has no gate projection")
    if param.shape[0] != loaded_weight.shape[0]:
        raise ValueError("Local expert count does not match checkpoint storage")
    destination = param.data
    if gated and projection != "down_proj":
        if param.shape[1] % 2:
            raise ValueError("Gated w13 checkpoint width must have two equal halves")
        half = param.shape[1] // 2
        start = half if projection == "up_proj" else 0
        destination = destination[:, start : start + half, :]
    if any(size <= 0 for size in loaded_weight.shape):
        raise ValueError("Local expert payload dimensions must be positive")
    if any(
        received > available
        for received, available in zip(
            loaded_weight.shape, destination.shape, strict=True
        )
    ):
        raise ValueError(
            f"Local expert payload {tuple(loaded_weight.shape)} exceeds "
            f"checkpoint block {tuple(destination.shape)}"
        )
    destination.zero_()
    destination[:, : loaded_weight.shape[1], : loaded_weight.shape[2]].copy_(
        loaded_weight
    )
