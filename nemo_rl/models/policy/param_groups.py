# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import torch


def get_param_groups_with_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict]:
    """Create parameter groups with proper weight decay exclusion.

    Matches Megatron's default behavior: excludes biases, 1D parameters (e.g.,
    LayerNorm weights), and parameters with _no_weight_decay=True attribute.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if getattr(param, "_no_weight_decay", False):
            no_decay_params.append(param)
        elif name.endswith(".bias") or len(param.shape) == 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    return param_groups

