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

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from nemo_rl.models.megatron.flextron import FrozenFlextronRouter


def attach_mamba_hooks(
    layer: torch.nn.Module,
    global_idx: int,
    flextron: "FrozenFlextronRouter",
) -> None:
    mamba_mixer = None
    for module in layer.modules():
        if "MambaMixer" == module.__class__.__name__:
            mamba_mixer = module
            break
    if mamba_mixer is None:
        return

    in_proj = getattr(mamba_mixer, "in_proj", None)
    if in_proj is None:
        return

    def input_mask_hook(
        module: torch.nn.Module, inputs: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        del module
        emb_int = flextron._active_emb_int(global_idx)
        if emb_int is None or not inputs:
            return inputs
        return flextron._mask_first_tensor(inputs, emb_int)

    def in_proj_pre_hook(
        module: torch.nn.Module, inputs: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        emb_int = flextron._active_emb_int(global_idx)
        if emb_int is None:
            return inputs

        layernorm_epsilon = getattr(flextron.model_cfg, "layernorm_epsilon", None)
        if layernorm_epsilon is not None and hasattr(module, "eps"):
            emb_effective_per = emb_int / flextron.model_cfg.hidden_size
            module.eps = layernorm_epsilon * emb_effective_per
        return inputs

    def in_proj_post_hook(
        module: torch.nn.Module, inputs: tuple[Any, ...], output: Any
    ) -> Any:
        del inputs
        emb_int = flextron._active_emb_int(global_idx)
        if emb_int is None:
            return output

        layernorm_epsilon = getattr(flextron.model_cfg, "layernorm_epsilon", None)
        if layernorm_epsilon is not None and hasattr(module, "eps"):
            module.eps = layernorm_epsilon
        emb_effective_per = emb_int / flextron.model_cfg.hidden_size
        return flextron._scale_output(output, emb_effective_per**0.5)

    def output_mask_hook(
        module: torch.nn.Module, inputs: tuple[Any, ...], output: Any
    ) -> Any:
        del module, inputs
        emb_int = flextron._active_emb_int(global_idx)
        if emb_int is None:
            return output
        return flextron._mask_output(output, emb_int)

    flextron._handles.append(mamba_mixer.register_forward_pre_hook(input_mask_hook))
    flextron._handles.append(in_proj.register_forward_pre_hook(in_proj_pre_hook))
    flextron._handles.append(in_proj.register_forward_hook(in_proj_post_hook))
    flextron._handles.append(mamba_mixer.register_forward_hook(output_mask_hook))
