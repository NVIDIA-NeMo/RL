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

"""Opt-in Super vision activation parity with vLLM RADIO."""

from typing import Any

import torch.nn as nn
import torch.nn.functional as F


def set_super_vision_exact_gelu(
    models: nn.Module | list[nn.Module] | tuple[nn.Module, ...],
) -> None:
    """Match exact GELU without hooks, parameter changes, or detached gradients."""
    models = models if isinstance(models, (list, tuple)) else [models]
    mlps: list[Any] = [
        module
        for model in models
        for name, module in model.named_modules()
        if "vision_model." in name and name.endswith(".mlp")
    ]
    if len(mlps) != 32:
        raise ValueError(f"Super exact GELU requires 32 vision MLPs, got {len(mlps)}")
    for mlp in mlps:
        if mlp.config.use_te_activation_func or mlp.config.bias_activation_fusion:
            raise ValueError("Super exact GELU requires unfused vision activations")
        if getattr(mlp.activation_func, "__name__", "") not in ("fast_gelu", "gelu"):
            raise ValueError(f"Unexpected vision activation: {mlp.activation_func!r}")
    for mlp in mlps:
        mlp.activation_func = F.gelu
    print(
        "Super vision: exact GELU enabled on 32 MLPs; gradients preserved", flush=True
    )
