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


def _find_module_by_class_name(
    layer: torch.nn.Module, class_name: str
) -> torch.nn.Module | None:
    for module in layer.modules():
        if class_name == module.__class__.__name__:
            return module
    return None


def attach_moe_hooks(
    layer: torch.nn.Module,
    global_idx: int,
    flextron: "FrozenFlextronRouter",
) -> None:
    pre_mlp_layernorm = getattr(layer, "pre_mlp_layernorm", None)
    if pre_mlp_layernorm is not None:

        def pre_mlp_layernorm_pre_hook(
            module: torch.nn.Module, inputs: tuple[Any, ...]
        ) -> tuple[Any, ...]:
            emb_int = flextron._active_emb_int(global_idx)
            if emb_int is None or not inputs:
                return inputs

            layernorm_epsilon = getattr(flextron.model_cfg, "layernorm_epsilon", None)
            if layernorm_epsilon is not None and hasattr(module, "eps"):
                emb_effective_per = emb_int / flextron.model_cfg.hidden_size
                module.eps = layernorm_epsilon * emb_effective_per
            return flextron._mask_first_tensor(inputs, emb_int)

        def pre_mlp_layernorm_post_hook(
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

        flextron._handles.append(
            pre_mlp_layernorm.register_forward_pre_hook(pre_mlp_layernorm_pre_hook)
        )
        flextron._handles.append(
            pre_mlp_layernorm.register_forward_hook(pre_mlp_layernorm_post_hook)
        )

    moe_layer = _find_module_by_class_name(layer, "MoELayer")
    if moe_layer is not None:

        def moe_output_hook(
            module: torch.nn.Module, inputs: tuple[Any, ...], output: Any
        ) -> Any:
            del module, inputs
            emb_int = flextron._active_emb_int(global_idx)
            if emb_int is None:
                return output
            return flextron._mask_output(output, emb_int)

        flextron._handles.append(moe_layer.register_forward_hook(moe_output_hook))

    grouped_mlp = _find_module_by_class_name(layer, "TEGroupedMLP")
    if grouped_mlp is None:
        return

    def grouped_mlp_input_hook(
        module: torch.nn.Module, inputs: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        del module
        emb_int = flextron._active_emb_int(global_idx)
        if emb_int is None or not inputs:
            return inputs
        return flextron._mask_first_tensor(inputs, emb_int)

    def grouped_mlp_fc1_post_hook(
        module: torch.nn.Module, inputs: tuple[Any, ...], output: Any
    ) -> Any:
        del module, inputs
        mlp_int = flextron._active_mlp_int(global_idx)
        if mlp_int is None:
            return output
        return flextron._mask_output(
            output,
            mlp_int,
            full_dim=flextron.model_cfg.ffn_hidden_size,
        )

    def grouped_mlp_output_hook(
        module: torch.nn.Module, inputs: tuple[Any, ...], output: Any
    ) -> Any:
        del module, inputs
        emb_int = flextron._active_emb_int(global_idx)
        if emb_int is None:
            return output
        return flextron._mask_output(output, emb_int)

    flextron._handles.append(grouped_mlp.register_forward_pre_hook(grouped_mlp_input_hook))

    linear_fc1 = getattr(grouped_mlp, "linear_fc1", None)
    if linear_fc1 is not None:
        flextron._handles.append(linear_fc1.register_forward_hook(grouped_mlp_fc1_post_hook))

    flextron._handles.append(grouped_mlp.register_forward_hook(grouped_mlp_output_hook))
