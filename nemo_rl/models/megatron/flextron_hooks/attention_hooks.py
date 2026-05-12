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


def attach_attention_hooks(
    layer: torch.nn.Module,
    global_idx: int,
    flextron: "FrozenFlextronRouter",
) -> None:
    attention_module = None
    for name, module in layer.named_modules():
        if "SelfAttention" == module.__class__.__name__:
            attention_module = module
            break
    if attention_module is None:
        return

    linear_qkv = getattr(attention_module, "linear_qkv", None)
    if linear_qkv is None:
        return

    def input_mask_hook(
        module: torch.nn.Module, inputs: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        del module
        emb_int = flextron._active_emb_int(global_idx)
        if emb_int is None or not inputs:
            return inputs
        return flextron._mask_first_tensor(inputs, emb_int)

    def linear_qkv_pre_hook(
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

    def linear_qkv_post_hook(
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

    def output_mask_hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any):
        del module, inputs
        emb_int = flextron._active_emb_int(global_idx)
        if emb_int is None:
            return output
        return flextron._mask_output(output, emb_int)

    main_handle = attention_module.register_forward_pre_hook(input_mask_hook)
    flextron._handles.append(main_handle)

    qkv_pre_handle = linear_qkv.register_forward_pre_hook(linear_qkv_pre_hook)
    qkv_post_handle = linear_qkv.register_forward_hook(linear_qkv_post_hook)
    flextron._handles.append(qkv_pre_handle)
    flextron._handles.append(qkv_post_handle)

    output_handle = attention_module.register_forward_hook(output_mask_hook)
    flextron._handles.append(output_handle)
