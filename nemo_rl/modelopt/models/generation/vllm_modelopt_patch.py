# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""vLLM ModelOpt NVFP4 patches for dense rollout weight reloads."""

import torch
from torch.nn import Parameter

_DENSE_HF_PARAMS = ("weight", "weight_scale", "weight_scale_2")


def _unwrap_vllm_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.model if hasattr(model, "model") else model


def _canonicalize_nvfp4_weight_scale(layer: torch.nn.Module) -> None:
    weight_scale = layer.weight_scale
    scale = weight_scale.data.to(torch.float32).abs().to(weight_scale.dtype)
    weight_scale.data.copy_(scale)


def _convert_nvfp4_linear_kernel_format(quant_method, layer: torch.nn.Module) -> None:
    kernel = getattr(quant_method, "kernel", None)
    if kernel is not None:
        kernel.process_weights_after_loading(layer)
        return

    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        convert_to_nvfp4_linear_kernel_format,
    )

    convert_to_nvfp4_linear_kernel_format(quant_method.backend, layer)


def _modelopt_dense_process_weights(self, layer: torch.nn.Module) -> None:
    """Convert dense ModelOpt NVFP4 weights after initial load or refit."""
    if not hasattr(layer, "_nrl_modelopt_param_meta"):
        layer._nrl_modelopt_param_meta = {}
        layer._nrl_modelopt_weight_loaders = {}
        for param_name in _DENSE_HF_PARAMS:
            param = getattr(layer, param_name)
            meta = {
                "shape": tuple(param.shape),
                "dtype": param.dtype,
                "device": str(param.device),
                "param_class": type(param),
            }
            if hasattr(param, "_input_dim"):
                meta["input_dim"] = param._input_dim
            if hasattr(param, "_output_dim"):
                meta["output_dim"] = param._output_dim
            layer._nrl_modelopt_param_meta[param_name] = meta
            if hasattr(param, "weight_loader"):
                layer._nrl_modelopt_weight_loaders[param_name] = param.weight_loader

    input_global_scale = torch.ones(
        (),
        dtype=torch.float32,
        device=layer.weight.device,
    )
    layer.input_global_scale = Parameter(input_global_scale, requires_grad=False)

    weight_global_scale = layer.weight_scale_2.max().to(torch.float32)
    layer.weight_global_scale = Parameter(weight_global_scale, requires_grad=False)
    delattr(layer, "weight_scale_2")

    layer.alpha = Parameter(
        layer.input_global_scale * layer.weight_global_scale,
        requires_grad=False,
    )
    layer.input_global_scale_inv = Parameter(
        (1.0 / layer.input_global_scale).to(torch.float32),
        requires_grad=False,
    )

    _canonicalize_nvfp4_weight_scale(layer)
    _convert_nvfp4_linear_kernel_format(self, layer)


def prepare_modelopt_for_weight_reload(model, device=None) -> None:
    """Prepare a dense ModelOpt-vLLM model for one weight reload cycle."""
    inner_model = _unwrap_vllm_model(model)
    for module in inner_model.modules():
        layer_meta = getattr(module, "_nrl_modelopt_param_meta", None)
        if layer_meta is None:
            continue
        for param_name, meta in layer_meta.items():
            param = getattr(module, param_name, None)
            weight_loader = module._nrl_modelopt_weight_loaders.get(param_name)
            param_class = meta["param_class"]
            if (
                param is None
                or tuple(param.shape) != tuple(meta["shape"])
                or param.dtype != meta["dtype"]
                or (
                    weight_loader is not None
                    and (
                        not isinstance(param, param_class)
                        or not hasattr(param, "weight_loader")
                    )
                )
            ):
                data = torch.empty(
                    meta["shape"],
                    dtype=meta["dtype"],
                    device=device or meta["device"],
                )
                if param_class is not Parameter and weight_loader is not None:
                    kwargs = {"data": data, "weight_loader": weight_loader}
                    if "input_dim" in meta:
                        kwargs["input_dim"] = meta["input_dim"]
                    if "output_dim" in meta:
                        kwargs["output_dim"] = meta["output_dim"]
                    replacement = param_class(**kwargs)
                else:
                    replacement = Parameter(data, requires_grad=False)
                    if weight_loader is not None:
                        replacement.weight_loader = weight_loader
                setattr(module, param_name, replacement)


def modelopt_process_weights_after_loading(model) -> None:
    """Run vLLM ModelOpt post-load processing for dense quantized layers."""
    actual_model = _unwrap_vllm_model(model)

    for module in actual_model.modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method.__class__.__name__ == "ModelOptNvFp4LinearMethod":
            quant_method.process_weights_after_loading(module)


_patched = False


def apply_modelopt_nvfp4_patches() -> None:
    """Patch vLLM's dense ModelOpt NVFP4 method for rollout refits."""
    global _patched

    if _patched:
        return

    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4LinearMethod,
    )

    ModelOptNvFp4LinearMethod.process_weights_after_loading = (
        _modelopt_dense_process_weights
    )

    _patched = True
