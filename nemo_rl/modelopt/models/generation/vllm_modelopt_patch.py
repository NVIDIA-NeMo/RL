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

"""vLLM ModelOpt NVFP4 patches for rollout weight reloads.

These patches target vLLM 0.20.0 internals. Revalidate the imported APIs,
method signatures, and tensor layouts whenever vLLM is upgraded.
"""

import inspect
from typing import Any

import torch
from torch.nn import Parameter

_DENSE_W4A16_HF_PARAMS = ("weight", "weight_scale", "weight_scale_2")
_DENSE_W4A4_HF_PARAMS = (*_DENSE_W4A16_HF_PARAMS, "input_scale")
_DENSE_KERNEL_PARAMS = (
    "weight",
    "weight_scale",
    "weight_global_scale",
    "input_global_scale",
    "alpha",
    "input_global_scale_inv",
)
_MOE_HF_PARAMS = (
    "w13_weight",
    "w13_weight_scale",
    "w13_weight_scale_2",
    "w13_input_scale",
    "w2_weight",
    "w2_weight_scale",
    "w2_weight_scale_2",
    "w2_input_scale",
)
_MOE_INPUT_SCALE_PARAMS = ("w13_input_scale", "w2_input_scale")
_MOE_MARLIN_TENSOR_PARAMS = _MOE_HF_PARAMS[:3] + _MOE_HF_PARAMS[4:7]
_MODELOPT_W4A16_QUANT_MODES = frozenset({"w4a16_nvfp4", "nvfp4_w4a16"})
_MODELOPT_W4A16_ATTR = "_nrl_weight_only_w4a16"
_MODELOPT_W4A16_MOE_MARLIN_TILE_N = 64
_MODELOPT_PARAM_META_ATTR = "_nrl_modelopt_param_meta"
_MODELOPT_WEIGHT_LOADERS_ATTR = "_nrl_modelopt_weight_loaders"
_MODELOPT_RELOAD_PARAM_ATTRS = ("quant_method",)
_ORIGINAL_NVFP4_CONFIG_FROM_CONFIG_ATTR = "_nrl_original_from_config"
_ORIGINAL_LINEAR_APPLY_ATTR = "_nrl_original_apply"
_ORIGINAL_FUSED_MOE_INIT_ATTR = "_nrl_original_init"
_ORIGINAL_FUSED_MOE_ROUNDUP_SIZES_ATTR = "_nrl_original_maybe_roundup_sizes"
_ORIGINAL_FUSED_MOE_PROCESS_WEIGHTS_ATTR = "_nrl_original_process_weights_after_loading"
_ORIGINAL_KV_CACHE_PROCESS_WEIGHTS_ATTR = (
    "_nrl_original_kv_cache_process_weights_after_loading"
)
_MODELOPT_PROCESS_WEIGHTS_CALL_COUNT_ATTR = "_nrl_process_weights_call_count"
_MODELOPT_PROCESSED_TENSOR_REFS_ATTR = "_nrl_modelopt_processed_tensor_refs"
_MODELOPT_MOE_QUANT_CONFIG_SCALE_ATTRS = (
    "g1_alphas",
    "g2_alphas",
    "a1_gscale",
    "a2_gscale",
    "w1_scale",
    "w2_scale",
)


def _unwrap_vllm_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.model if hasattr(model, "model") else model


def _canonicalize_nvfp4_scale_parameter(
    layer: torch.nn.Module,
    param_name: str,
) -> None:
    param = getattr(layer, param_name)
    scale = param.data.to(torch.float32).abs().to(param.dtype)
    param.data.copy_(scale)


def _canonicalize_dense_nvfp4_weight_scale(layer: torch.nn.Module) -> None:
    _canonicalize_nvfp4_scale_parameter(layer, "weight_scale")


def _canonicalize_moe_nvfp4_weight_scales(layer: torch.nn.Module) -> None:
    _canonicalize_nvfp4_scale_parameter(layer, "w13_weight_scale")
    _canonicalize_nvfp4_scale_parameter(layer, "w2_weight_scale")


def _requests_w4a16_modelopt_config(config: dict[str, Any]) -> bool:
    quant_mode = config.get("quant_mode")
    if (
        isinstance(quant_mode, str)
        and quant_mode.lower() in _MODELOPT_W4A16_QUANT_MODES
    ):
        return True
    if config.get("weight_only") is True:
        return True

    nested = config.get("quantization")
    return isinstance(nested, dict) and _requests_w4a16_modelopt_config(nested)


def _is_w4a16_modelopt_quant_config(quant_config: object) -> bool:
    return bool(getattr(quant_config, _MODELOPT_W4A16_ATTR, False))


def _is_marlin_backend(quant_method: object) -> bool:
    for candidate in (
        getattr(quant_method, "nvfp4_backend", None),
        getattr(quant_method, "backend", None),
        getattr(quant_method, "kernel", None),
    ):
        if isinstance(candidate, str) and candidate.upper() == "MARLIN":
            return True
        name = getattr(candidate, "value", None) or getattr(candidate, "name", None)
        if isinstance(name, str) and name.upper() == "MARLIN":
            return True
        if candidate is not None and "marlin" in candidate.__class__.__name__.lower():
            return True
    return False


def _require_valid_scale(
    value: torch.Tensor,
    *,
    name: str,
    strictly_positive: bool,
) -> None:
    value = value.detach().float()
    if not torch.isfinite(value).all():
        raise RuntimeError(f"ModelOpt NVFP4 {name} must contain only finite values")
    if strictly_positive and not torch.all(value > 0):
        raise RuntimeError(f"ModelOpt NVFP4 {name} must be strictly positive")
    if not strictly_positive and not torch.all(value >= 0):
        raise RuntimeError(f"ModelOpt NVFP4 {name} must be non-negative")


def _sanitize_dummy_scale(
    value: torch.Tensor,
    *,
    strictly_positive: bool,
) -> None:
    """Make vLLM dummy-loader scale placeholders safe for initial kernel setup."""
    with torch.no_grad():
        sanitized = value.detach().float().abs()
        invalid = ~torch.isfinite(sanitized)
        if strictly_positive:
            invalid |= sanitized == 0
        sanitized.masked_fill_(invalid, 1.0)
        value.data.copy_(sanitized.to(dtype=value.dtype, device=value.device))


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _is_w4a16_marlin_moe_quant_method(quant_method: object) -> bool:
    backend = getattr(quant_method, "nvfp4_backend", None)
    return (
        quant_method.__class__.__name__ == "ModelOptNvFp4FusedMoE"
        and _is_w4a16_modelopt_quant_config(getattr(quant_method, "quant_config", None))
        and getattr(backend, "value", backend) == "MARLIN"
    )


def _stash_original(cls: type, attr: str, value: object) -> None:
    if not hasattr(cls, attr):
        setattr(cls, attr, value)


def _require_fp4_marlin_supported() -> None:
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        is_fp4_marlin_supported,
    )

    if not is_fp4_marlin_supported():
        raise RuntimeError(
            "ModelOpt NVFP4 W4A16 rollout requires vLLM FP4 Marlin support."
        )


def _modelopt_nvfp4_config_from_config(cls, *args: Any, **kwargs: Any) -> object:
    original_from_config = getattr(cls, _ORIGINAL_NVFP4_CONFIG_FROM_CONFIG_ATTR)
    quant_config = original_from_config(*args, **kwargs)

    original_config = kwargs.get("original_config")
    if isinstance(original_config, dict) and _requests_w4a16_modelopt_config(
        original_config
    ):
        setattr(quant_config, _MODELOPT_W4A16_ATTR, True)

    return quant_config


def _convert_nvfp4_linear_kernel_format(
    quant_method: object,
    layer: torch.nn.Module,
) -> None:
    kernel = getattr(quant_method, "kernel", None)
    if kernel is not None:
        kernel.process_weights_after_loading(layer)
        return

    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        convert_to_nvfp4_linear_kernel_format,
    )

    convert_to_nvfp4_linear_kernel_format(quant_method.backend, layer)


def _capture_modelopt_param_reload_meta(
    layer: torch.nn.Module,
    param_names: tuple[str, ...],
) -> None:
    param_meta = getattr(layer, _MODELOPT_PARAM_META_ATTR, None)
    if param_meta is None:
        param_meta = {}
        setattr(layer, _MODELOPT_PARAM_META_ATTR, param_meta)
    weight_loaders = getattr(layer, _MODELOPT_WEIGHT_LOADERS_ATTR, None)
    if weight_loaders is None:
        weight_loaders = {}
        setattr(layer, _MODELOPT_WEIGHT_LOADERS_ATTR, weight_loaders)

    for param_name in param_names:
        if param_name in param_meta:
            continue
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
        attrs = {
            attr: getattr(param, attr)
            for attr in _MODELOPT_RELOAD_PARAM_ATTRS
            if hasattr(param, attr)
        }
        if attrs:
            meta["attrs"] = attrs
        param_meta[param_name] = meta
        if hasattr(param, "weight_loader"):
            weight_loaders[param_name] = param.weight_loader


def _is_first_modelopt_process_weights_call(layer: torch.nn.Module) -> bool:
    count = getattr(layer, _MODELOPT_PROCESS_WEIGHTS_CALL_COUNT_ATTR, 0)
    setattr(layer, _MODELOPT_PROCESS_WEIGHTS_CALL_COUNT_ATTR, count + 1)
    return count == 0


def _set_or_update_processed_tensor_ref(
    layer: torch.nn.Module,
    param_name: str,
    data: torch.Tensor,
    is_first_call: bool,
) -> None:
    refs = getattr(layer, _MODELOPT_PROCESSED_TENSOR_REFS_ATTR, None)
    if refs is None:
        refs = {}
        setattr(layer, _MODELOPT_PROCESSED_TENSOR_REFS_ATTR, refs)

    ref = refs.get(param_name)
    if (
        is_first_call
        or ref is None
        or ref.shape != data.shape
        or ref.dtype != data.dtype
        or ref.device != data.device
    ):
        # vLLM may represent repeated per-expert scales with expand(), which
        # produces a zero-stride view. Such a view is valid for kernel reads,
        # but cannot be the destination of the in-place copy used by the next
        # refit. Materialize only those expanded dimensions; cloning all
        # processed tensors here would transiently duplicate the FP4 weights.
        if any(
            size > 1 and stride == 0
            for size, stride in zip(data.shape, data.stride(), strict=True)
        ):
            data = data.clone(memory_format=torch.preserve_format)
        setattr(layer, param_name, Parameter(data, requires_grad=False))
        refs[param_name] = getattr(layer, param_name).data
        return

    ref.copy_(data)
    setattr(layer, param_name, Parameter(ref, requires_grad=False))


def _modelopt_dense_process_w4a16_weights(self, layer: torch.nn.Module) -> None:
    """Convert dense ModelOpt NVFP4 W4A16 weights for Marlin weight-only GEMM."""
    _require_fp4_marlin_supported()
    _capture_modelopt_param_reload_meta(layer, _DENSE_W4A16_HF_PARAMS)
    is_first_call = _is_first_modelopt_process_weights_call(layer)

    if not is_first_call:
        weight_scale_2 = getattr(layer, "weight_scale_2", None)
        if not isinstance(weight_scale_2, torch.Tensor):
            raise RuntimeError(
                "ModelOpt NVFP4 W4A16 dense refit requires weight_scale_2"
            )
        _require_valid_scale(
            weight_scale_2,
            name="dense weight_scale_2",
            strictly_positive=True,
        )

    weight_global_scale = layer.weight_scale_2.max().to(torch.float32)
    layer.weight_global_scale = Parameter(weight_global_scale, requires_grad=False)
    delattr(layer, "weight_scale_2")

    for attr in (
        "input_scale",
        "input_global_scale",
        "alpha",
        "input_global_scale_inv",
    ):
        if hasattr(layer, attr):
            delattr(layer, attr)

    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        prepare_fp4_layer_for_marlin,
    )

    _canonicalize_dense_nvfp4_weight_scale(layer)
    prepare_fp4_layer_for_marlin(layer)
    for param_name in ("weight", "weight_scale", "weight_global_scale"):
        _set_or_update_processed_tensor_ref(
            layer,
            param_name,
            getattr(layer, param_name).data,
            is_first_call=is_first_call,
        )


def _modelopt_dense_process_weights(self, layer: torch.nn.Module) -> None:
    """Convert dense ModelOpt NVFP4 weights after initial load or refit."""
    if _is_w4a16_modelopt_quant_config(getattr(self, "quant_config", None)):
        _modelopt_dense_process_w4a16_weights(self, layer)
        return

    if _is_marlin_backend(self):
        raise RuntimeError(
            "ModelOpt NVFP4 W4A4 requires an activation-quantizing vLLM backend; "
            "Marlin is weight-only"
        )

    if not isinstance(getattr(layer, "input_scale", None), torch.Tensor):
        raise RuntimeError(
            "ModelOpt NVFP4 W4A4 dense refit requires input_scale; check the "
            "training quantizer and refit export"
        )

    _capture_modelopt_param_reload_meta(layer, _DENSE_W4A4_HF_PARAMS)
    is_first_call = _is_first_modelopt_process_weights_call(layer)

    scale_specs = (
        ("input_scale", True),
        ("weight_scale", True),
        ("weight_scale_2", True),
    )
    for param_name, strictly_positive in scale_specs:
        scale = getattr(layer, param_name)
        if is_first_call:
            _sanitize_dummy_scale(scale, strictly_positive=strictly_positive)
        else:
            _require_valid_scale(
                scale,
                name=f"dense {param_name}",
                strictly_positive=strictly_positive,
            )

    input_global_scale = layer.input_scale.max().to(torch.float32)
    layer.input_global_scale = Parameter(input_global_scale, requires_grad=False)

    weight_global_scale = layer.weight_scale_2.max().to(torch.float32)
    layer.weight_global_scale = Parameter(weight_global_scale, requires_grad=False)
    delattr(layer, "input_scale")
    delattr(layer, "weight_scale_2")

    layer.alpha = Parameter(
        layer.input_global_scale * layer.weight_global_scale,
        requires_grad=False,
    )
    layer.input_global_scale_inv = Parameter(
        (1.0 / layer.input_global_scale).to(torch.float32),
        requires_grad=False,
    )

    _convert_nvfp4_linear_kernel_format(self, layer)
    for param_name in _DENSE_KERNEL_PARAMS:
        if hasattr(layer, param_name):
            _set_or_update_processed_tensor_ref(
                layer,
                param_name,
                getattr(layer, param_name).data,
                is_first_call=is_first_call,
            )


def _modelopt_dense_apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if _is_w4a16_modelopt_quant_config(getattr(self, "quant_config", None)):
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
        )

        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )

    original_apply = getattr(type(self), _ORIGINAL_LINEAR_APPLY_ATTR, None)
    if original_apply is not None:
        return original_apply(self, layer, x, bias)

    return self.kernel.apply_weights(layer=layer, x=x, bias=bias)


def _modelopt_moe_init(self, quant_config: object, moe_config: object) -> None:
    if not _is_w4a16_modelopt_quant_config(quant_config):
        original_init = getattr(type(self), _ORIGINAL_FUSED_MOE_INIT_ATTR)
        original_init(self, quant_config, moe_config)
        if _is_marlin_backend(self):
            raise RuntimeError(
                "ModelOpt NVFP4 W4A4 MoE requires an activation-quantizing "
                "vLLM backend; Marlin is weight-only"
            )
        return

    try:
        from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
            FusedMoEMethodBase,
        )
    except ImportError:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoEMethodBase

    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import MarlinExperts
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend

    _require_fp4_marlin_supported()
    FusedMoEMethodBase.__init__(self, moe_config)
    self.__dict__.update(
        quant_config=quant_config,
        allow_flashinfer=False,
        cutlass_nvfp4_supported=False,
        flashinfer_moe_backend=None,
        use_marlin=True,
        backend="marlin",
        nvfp4_backend=NvFp4MoeBackend.MARLIN,
        experts_cls=MarlinExperts,
        use_global_sf=False,
    )


def _modelopt_moe_maybe_roundup_sizes(
    self,
    hidden_size: int,
    intermediate_size_per_partition: int,
    act_dtype: torch.dtype,
    moe_parallel_config: object,
) -> tuple[int, int]:
    """Round W4A16 MoE intermediate shards to Marlin's FP4 tile size."""
    original_roundup = getattr(type(self), _ORIGINAL_FUSED_MOE_ROUNDUP_SIZES_ATTR)
    hidden_size, intermediate_size_per_partition = original_roundup(
        self,
        hidden_size,
        intermediate_size_per_partition,
        act_dtype,
        moe_parallel_config,
    )

    if _is_w4a16_marlin_moe_quant_method(self):
        tile = _MODELOPT_W4A16_MOE_MARLIN_TILE_N
        intermediate_size_per_partition = (
            _ceil_div(
                intermediate_size_per_partition,
                tile,
            )
            * tile
        )

    return hidden_size, intermediate_size_per_partition


def _zero_modelopt_moe_padding(layer: torch.nn.Module) -> None:
    moe_config = getattr(layer, "moe_config", None)
    padded_size = getattr(moe_config, "intermediate_size_per_partition", None)
    unpadded_size = getattr(
        moe_config,
        "intermediate_size_per_partition_unpadded",
        padded_size,
    )
    if (
        not isinstance(padded_size, int)
        or not isinstance(unpadded_size, int)
        or padded_size <= unpadded_size
    ):
        return

    tp_rank = getattr(layer, "tp_rank", None)
    tp_size = getattr(layer, "tp_size", None)
    if isinstance(tp_rank, int) and isinstance(tp_size, int) and tp_size > 0:
        full_unpadded_size = unpadded_size * tp_size
        valid_size = max(
            0,
            min(padded_size, full_unpadded_size - padded_size * tp_rank),
        )
    else:
        valid_size = unpadded_size
    if valid_size >= padded_size:
        return

    quant_config = getattr(layer, "quant_config", None)
    group_size = getattr(quant_config, "group_size", None)
    if not isinstance(group_size, int) or group_size <= 0:
        raise RuntimeError(
            "Missing or invalid ModelOpt NVFP4 group_size for padded MoE reload"
        )
    with torch.no_grad():
        for param_name in ("w13_weight", "w13_weight_scale"):
            tensor = getattr(layer, param_name).data
            if tensor.ndim >= 2:
                for shard_start in range(0, tensor.shape[1], padded_size):
                    start = shard_start + valid_size
                    end = min(shard_start + padded_size, tensor.shape[1])
                    if start < end:
                        tensor.narrow(1, start, end - start).zero_()
        for tensor, start, end in (
            (
                layer.w2_weight.data,
                _ceil_div(valid_size, 2),
                _ceil_div(padded_size, 2),
            ),
            (
                layer.w2_weight_scale.data,
                _ceil_div(valid_size, group_size),
                _ceil_div(padded_size, group_size),
            ),
        ):
            if start < end and tensor.ndim > 2 and start < tensor.shape[2]:
                tensor.narrow(2, start, min(end, tensor.shape[2]) - start).zero_()


def _modelopt_moe_quant_config_inplace_update(
    self,
    layer: torch.nn.Module,
) -> bool:
    """Refresh scale values without changing CUDA-graph-visible addresses."""
    old_config = getattr(self, "moe_quant_config", None)
    old_kernel = getattr(self, "moe_kernel", None) or getattr(self, "kernel", None)
    if old_config is None or old_kernel is None:
        return False

    new_config = self.get_fused_moe_quant_config(layer)
    if new_config is None:
        return False

    updates: list[tuple[torch.Tensor, torch.Tensor]] = []
    for attr in _MODELOPT_MOE_QUANT_CONFIG_SCALE_ATTRS:
        old_value = getattr(old_config, attr, None)
        new_value = getattr(new_config, attr, None)
        if old_value is new_value or (old_value is None and new_value is None):
            continue
        if not isinstance(old_value, torch.Tensor) or not isinstance(
            new_value, torch.Tensor
        ):
            return False
        if (
            old_value.shape != new_value.shape
            or old_value.dtype != new_value.dtype
            or old_value.device != new_value.device
        ):
            return False
        updates.append((old_value, new_value))

    with torch.no_grad():
        for old_value, new_value in updates:
            old_value.copy_(new_value)

    self.moe_kernel = old_kernel
    self.kernel = old_kernel
    fused_experts = getattr(old_kernel, "fused_experts", None)
    if fused_experts is not None and hasattr(fused_experts, "quant_config"):
        fused_experts.quant_config = old_config
    return True


def _init_modelopt_moe_kernel(self, layer: torch.nn.Module) -> None:
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        make_nvfp4_moe_kernel,
    )

    moe_quant_config = self.get_fused_moe_quant_config(layer)
    if moe_quant_config is None:
        raise RuntimeError("ModelOpt NVFP4 MoE quant config is missing")
    if self.experts_cls is None:
        raise RuntimeError("ModelOpt NVFP4 MoE experts class is missing")

    kwargs: dict[str, object] = {
        "moe_quant_config": moe_quant_config,
        "moe_config": self.moe,
        "experts_cls": self.experts_cls,
    }
    parameters = inspect.signature(make_nvfp4_moe_kernel).parameters
    if "nvfp4_backend" in parameters:
        kwargs["nvfp4_backend"] = self.nvfp4_backend
    if "shared_experts" in parameters:
        kwargs["shared_experts"] = getattr(layer, "shared_experts", None)
    if "routing_tables" in parameters:
        kwargs["routing_tables"] = (
            layer._maybe_init_expert_routing_tables()
            if hasattr(layer, "_maybe_init_expert_routing_tables")
            else None
        )

    result = make_nvfp4_moe_kernel(**kwargs)
    if isinstance(result, tuple):
        kernel = result[0]
        if len(result) > 1:
            self.use_inplace = result[1]
    else:
        kernel = result
    self.moe_quant_config = moe_quant_config
    self.moe_kernel = kernel
    self.kernel = kernel


def _run_modelopt_moe_kernel_postprocess(
    self,
    layer: torch.nn.Module,
    *,
    is_first_call: bool,
) -> None:
    fused_experts = getattr(
        getattr(self, "moe_kernel", None),
        "fused_experts",
        None,
    )
    process_weights = getattr(fused_experts, "process_weights_after_loading", None)
    if process_weights is None:
        raise RuntimeError("ModelOpt NVFP4 MoE kernel post-load hook is missing")
    process_weights(layer)

    # vLLM's TRTLLM NVFP4 expert hook derives ``g1_scale_c`` from the freshly
    # loaded global scales and registers a new Parameter on every call. CUDA
    # graphs captured after the dummy load retain the original address, so a
    # later real-weight refit must update that storage in place just like the
    # other processed ModelOpt tensors. Other backends do not expose this
    # derived tensor and need no special handling.
    g1_scale_c = getattr(layer, "g1_scale_c", None)
    if isinstance(g1_scale_c, torch.Tensor):
        _set_or_update_processed_tensor_ref(
            layer,
            "g1_scale_c",
            g1_scale_c.data,
            is_first_call=is_first_call,
        )
        if hasattr(fused_experts, "g1_scale_c"):
            fused_experts.g1_scale_c = layer.g1_scale_c


def _modelopt_moe_process_w4a16_marlin_weights(
    self,
    layer: torch.nn.Module,
    is_first_call: bool,
) -> None:
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        convert_to_nvfp4_moe_kernel_format,
    )

    if not is_first_call:
        for param_name in (
            "w13_weight_scale_2",
            "w2_weight_scale_2",
        ):
            value = getattr(layer, param_name, None)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"ModelOpt NVFP4 W4A16 MoE requires {param_name}; "
                    "check the training quantizer and refit export"
                )
            _require_valid_scale(
                value,
                name=f"MoE {param_name}",
                strictly_positive=True,
            )

    w13_weight_scale_2 = layer.w13_weight_scale_2.data
    if w13_weight_scale_2.dim() == 2:
        w13_weight_scale_2 = w13_weight_scale_2[:, 0]
    layer.w13_weight_scale_2 = Parameter(w13_weight_scale_2, requires_grad=False)

    _canonicalize_moe_nvfp4_weight_scales(layer)

    (
        w13,
        w13_scale,
        w13_scale_2,
        _a13_scale,
        w2,
        w2_scale,
        w2_scale_2,
        _a2_scale,
    ) = convert_to_nvfp4_moe_kernel_format(
        nvfp4_backend=self.nvfp4_backend,
        layer=layer,
        w13=layer.w13_weight.data,
        w13_scale=layer.w13_weight_scale.data,
        w13_scale_2=layer.w13_weight_scale_2.data,
        a13_scale=None,
        w2=layer.w2_weight.data,
        w2_scale=layer.w2_weight_scale.data,
        w2_scale_2=layer.w2_weight_scale_2.data,
        a2_scale=None,
        is_act_and_mul=self.moe.is_act_and_mul,
    )

    for param_name, data in zip(
        _MOE_MARLIN_TENSOR_PARAMS,
        (w13, w13_scale, w13_scale_2, w2, w2_scale, w2_scale_2),
        strict=True,
    ):
        _set_or_update_processed_tensor_ref(
            layer,
            param_name,
            data,
            is_first_call=is_first_call,
        )

    layer.w13_input_scale = None
    layer.w2_input_scale = None
    _init_modelopt_moe_kernel(self, layer)
    _run_modelopt_moe_kernel_postprocess(
        self,
        layer,
        is_first_call=is_first_call,
    )


def _modelopt_moe_process_w4a4_weights(
    self,
    layer: torch.nn.Module,
    is_first_call: bool,
) -> None:
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        convert_to_nvfp4_moe_kernel_format,
    )

    if _is_marlin_backend(self):
        raise RuntimeError(
            "ModelOpt NVFP4 W4A4 MoE requires an activation-quantizing "
            "vLLM backend; Marlin is weight-only"
        )

    for param_name in (
        "w13_weight_scale",
        "w2_weight_scale",
    ):
        value = getattr(layer, param_name)
        if is_first_call:
            _sanitize_dummy_scale(value, strictly_positive=True)
        else:
            _require_valid_scale(
                value,
                name=f"MoE {param_name}",
                strictly_positive=True,
            )
    for param_name in (
        "w13_weight_scale_2",
        "w2_weight_scale_2",
        "w13_input_scale",
        "w2_input_scale",
    ):
        value = getattr(layer, param_name, None)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"ModelOpt NVFP4 W4A4 MoE requires {param_name}; "
                "check the training quantizer and refit export"
            )
        if is_first_call:
            _sanitize_dummy_scale(value, strictly_positive=True)
        else:
            _require_valid_scale(
                value,
                name=f"MoE {param_name}",
                strictly_positive=True,
            )

    local_num_experts = getattr(layer, "local_num_experts", layer.w13_weight.shape[0])
    global_num_experts = getattr(layer, "global_num_experts", local_num_experts)
    expert_map = getattr(layer, "expert_map", getattr(layer, "_expert_map", None))
    if local_num_experts != global_num_experts or expert_map is not None:
        raise RuntimeError(
            "ModelOpt NVFP4 W4A4 MoE refits require every vLLM rank to own "
            "the full expert set"
        )
    num_experts = global_num_experts
    w13_num_shards = 2 if self.moe.is_act_and_mul else 1
    if tuple(layer.w13_input_scale.shape) != (num_experts, w13_num_shards):
        raise RuntimeError(
            "ModelOpt NVFP4 W4A4 w13_input_scale must have shape "
            f"({num_experts}, {w13_num_shards}), got "
            f"{tuple(layer.w13_input_scale.shape)}"
        )
    if tuple(layer.w2_input_scale.shape) not in {
        (num_experts,),
        (num_experts, 1),
    }:
        raise RuntimeError(
            "ModelOpt NVFP4 W4A4 w2_input_scale must provide one scale per "
            f"expert, got {tuple(layer.w2_input_scale.shape)}"
        )

    w13_weight_scale_2 = layer.w13_weight_scale_2.data
    if w13_weight_scale_2.ndim == 2:
        if w13_weight_scale_2.shape[1] != w13_num_shards:
            raise RuntimeError(
                "ModelOpt NVFP4 W4A4 w13_weight_scale_2 must have "
                f"{w13_num_shards} projection column(s)"
            )
        if (
            w13_num_shards == 2
            and not is_first_call
            and not torch.allclose(
                w13_weight_scale_2[:, 0],
                w13_weight_scale_2[:, 1],
            )
        ):
            raise RuntimeError(
                "vLLM's fused W4A4 MoE kernel requires gate and up projections "
                "to share weight_scale_2"
            )
        w13_weight_scale_2 = w13_weight_scale_2[:, 0].contiguous()

    _canonicalize_moe_nvfp4_weight_scales(layer)
    converted = convert_to_nvfp4_moe_kernel_format(
        nvfp4_backend=self.nvfp4_backend,
        layer=layer,
        w13=layer.w13_weight.data,
        w13_scale=layer.w13_weight_scale.data,
        w13_scale_2=w13_weight_scale_2,
        a13_scale=layer.w13_input_scale.data,
        w2=layer.w2_weight.data,
        w2_scale=layer.w2_weight_scale.data,
        w2_scale_2=layer.w2_weight_scale_2.data,
        a2_scale=layer.w2_input_scale.data,
        is_act_and_mul=self.moe.is_act_and_mul,
    )
    for param_name, data in zip(_MOE_HF_PARAMS, converted, strict=True):
        _set_or_update_processed_tensor_ref(
            layer,
            param_name,
            data.data if isinstance(data, Parameter) else data,
            is_first_call=is_first_call,
        )

    if is_first_call or not _modelopt_moe_quant_config_inplace_update(self, layer):
        _init_modelopt_moe_kernel(self, layer)
    _run_modelopt_moe_kernel_postprocess(
        self,
        layer,
        is_first_call=is_first_call,
    )


def _modelopt_kv_cache_process_weights(self, layer: torch.nn.Module) -> None:
    """Update KV-cache quantization scales without deleting reload parameters."""
    from vllm.platforms import current_platform

    def copy_scalar(dst: torch.Tensor, value: float | torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            dst.copy_(value.to(device=dst.device, dtype=dst.dtype))
        else:
            dst.fill_(value)

    if layer.kv_cache_dtype != "auto" and not layer.calculate_kv_scales:
        if layer.k_scale > 0.0 and layer.v_scale > 0.0:
            k_scale = layer.k_scale.to("cpu").tolist()
            v_scale = layer.v_scale.to("cpu").tolist()
            if current_platform.is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2
        elif layer.k_scale < 0.0 and layer.v_scale < 0.0:
            k_scale = 1.0
            v_scale = 1.0
        else:
            scale_to_duplicate = max(layer.k_scale, layer.v_scale)
            k_scale = scale_to_duplicate.to("cpu").tolist()
            v_scale = scale_to_duplicate.to("cpu").tolist()
            if current_platform.is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2

        if not isinstance(k_scale, float) or not isinstance(v_scale, float):
            raise ValueError("Only support per-tensor scaling factor for fp8 KV cache")

        if layer.q_scale < 0.0:
            copy_scalar(layer._q_scale, k_scale)
            layer._q_scale_float = k_scale

        copy_scalar(layer._k_scale, k_scale)
        copy_scalar(layer._v_scale, v_scale)
        layer._k_scale_float = k_scale
        layer._v_scale_float = v_scale

    if layer.q_scale > 0.0:
        q_scale = layer.q_scale
        if current_platform.is_fp8_fnuz():
            q_scale *= 2
        layer.calculate_kv_scales = False
    else:
        q_scale = 1.0
    if layer.prob_scale > 0.0:
        prob_scale = layer.prob_scale
        if current_platform.is_fp8_fnuz():
            prob_scale *= 2
    else:
        prob_scale = 1.0

    def is_singleton_float(value: float | torch.Tensor) -> bool:
        return isinstance(value, float) or (
            isinstance(value, torch.Tensor)
            and value.numel() == 1
            and value.is_floating_point()
        )

    if not is_singleton_float(q_scale) or not is_singleton_float(prob_scale):
        raise ValueError("Only support per-tensor scaling factor for fp8 Q/prob")

    copy_scalar(layer._q_scale, q_scale)
    layer._q_scale_float = (
        q_scale.item() if isinstance(q_scale, torch.Tensor) else q_scale
    )
    copy_scalar(layer._prob_scale, prob_scale)


def _modelopt_moe_process_weights(self, layer: torch.nn.Module) -> None:
    """Convert MoE ModelOpt NVFP4 weights after initial load or refit."""
    if not _is_w4a16_marlin_moe_quant_method(self):
        missing = [
            name
            for name in _MOE_INPUT_SCALE_PARAMS
            if not isinstance(getattr(layer, name, None), torch.Tensor)
        ]
        if missing:
            raise RuntimeError(
                "ModelOpt NVFP4 W4A4 MoE refit requires activation scales: "
                + ", ".join(missing)
            )
    _capture_modelopt_param_reload_meta(layer, _MOE_HF_PARAMS)
    is_first_call = _is_first_modelopt_process_weights_call(layer)
    if _is_w4a16_marlin_moe_quant_method(self):
        _zero_modelopt_moe_padding(layer)
        _modelopt_moe_process_w4a16_marlin_weights(
            self,
            layer,
            is_first_call=is_first_call,
        )
        return

    _modelopt_moe_process_w4a4_weights(
        self,
        layer,
        is_first_call=is_first_call,
    )


def prepare_modelopt_for_weight_reload(
    model: torch.nn.Module,
    device: torch.device | str | None = None,
) -> None:
    """Prepare a ModelOpt-vLLM model for one weight reload cycle."""
    inner_model = _unwrap_vllm_model(model)
    for module in inner_model.modules():
        layer_meta = getattr(module, _MODELOPT_PARAM_META_ATTR, None)
        if layer_meta is None:
            continue
        weight_loaders = getattr(module, _MODELOPT_WEIGHT_LOADERS_ATTR, {})
        for param_name, meta in layer_meta.items():
            param = getattr(module, param_name, None)
            weight_loader = weight_loaders.get(param_name)
            param_class = meta["param_class"]
            is_sentinel_scale = param_name.endswith("scale_2") or param_name in (
                "input_scale",
                *_MOE_INPUT_SCALE_PARAMS,
            )
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
                if is_sentinel_scale:
                    # Missing small scales must fail post-load validation.
                    data.fill_(torch.nan)
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
                for attr, value in meta.get("attrs", {}).items():
                    setattr(replacement, attr, value)
                setattr(module, param_name, replacement)
            elif is_sentinel_scale:
                # Poison only small global/input scales. Exact key accounting
                # proves the large block scales arrived without adding a
                # multi-gigabyte sentinel write to every refit.
                param.data.fill_(torch.nan)


def modelopt_process_weights_after_loading(model: torch.nn.Module) -> None:
    """Run vLLM ModelOpt post-load processing for quantized layers."""
    actual_model = _unwrap_vllm_model(model)

    for module in actual_model.modules():
        scheme = getattr(module, "scheme", None)
        if (
            scheme is not None
            and getattr(type(scheme), "process_weights_after_loading", None)
            is _modelopt_kv_cache_process_weights
        ):
            scheme.process_weights_after_loading(module)

        quant_method = getattr(module, "quant_method", None)
        actual_quant_method = getattr(quant_method, "old_quant_method", quant_method)
        if actual_quant_method.__class__.__name__ in (
            "ModelOptNvFp4LinearMethod",
            "ModelOptNvFp4FusedMoE",
        ):
            actual_quant_method.process_weights_after_loading(module)
            if actual_quant_method is not quant_method and hasattr(
                actual_quant_method, "moe_quant_config"
            ):
                quant_method.moe_quant_config = actual_quant_method.moe_quant_config
                fused_experts = getattr(quant_method, "fused_experts", None)
                nested_experts = getattr(fused_experts, "fused_experts", None)
                if nested_experts is not None and hasattr(
                    nested_experts, "quant_config"
                ):
                    nested_experts.quant_config = actual_quant_method.moe_quant_config


_patched = False


def apply_modelopt_nvfp4_patches() -> None:
    """Patch vLLM's ModelOpt NVFP4 methods for rollout refits."""
    global _patched

    if _patched:
        return

    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4LinearMethod,
    )

    try:
        from vllm.model_executor.layers.quantization.kv_cache import (
            BaseKVCacheMethod,
        )
    except ImportError:
        BaseKVCacheMethod = None
    try:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4FusedMoE,
        )
    except ImportError:
        ModelOptNvFp4FusedMoE = None

    _stash_original(
        ModelOptNvFp4Config,
        _ORIGINAL_NVFP4_CONFIG_FROM_CONFIG_ATTR,
        ModelOptNvFp4Config._from_config,
    )
    ModelOptNvFp4Config._from_config = classmethod(_modelopt_nvfp4_config_from_config)

    _stash_original(
        ModelOptNvFp4LinearMethod,
        _ORIGINAL_LINEAR_APPLY_ATTR,
        ModelOptNvFp4LinearMethod.apply,
    )
    ModelOptNvFp4LinearMethod.process_weights_after_loading = (
        _modelopt_dense_process_weights
    )
    ModelOptNvFp4LinearMethod.apply = _modelopt_dense_apply
    if ModelOptNvFp4FusedMoE is not None:
        moe_patches = {
            "__init__": (_ORIGINAL_FUSED_MOE_INIT_ATTR, _modelopt_moe_init),
            "process_weights_after_loading": (
                _ORIGINAL_FUSED_MOE_PROCESS_WEIGHTS_ATTR,
                _modelopt_moe_process_weights,
            ),
        }
        if hasattr(ModelOptNvFp4FusedMoE, "maybe_roundup_sizes"):
            moe_patches["maybe_roundup_sizes"] = (
                _ORIGINAL_FUSED_MOE_ROUNDUP_SIZES_ATTR,
                _modelopt_moe_maybe_roundup_sizes,
            )
        for method_name, (original_attr, replacement) in moe_patches.items():
            _stash_original(
                ModelOptNvFp4FusedMoE,
                original_attr,
                getattr(ModelOptNvFp4FusedMoE, method_name),
            )
            setattr(ModelOptNvFp4FusedMoE, method_name, replacement)
    if BaseKVCacheMethod is not None:
        _stash_original(
            BaseKVCacheMethod,
            _ORIGINAL_KV_CACHE_PROCESS_WEIGHTS_ATTR,
            BaseKVCacheMethod.process_weights_after_loading,
        )
        BaseKVCacheMethod.process_weights_after_loading = (
            _modelopt_kv_cache_process_weights
        )

    _patched = True
