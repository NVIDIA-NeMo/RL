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

"""Narrow vLLM extensions for ModelOpt NVFP4 rollout checkpoints.

vLLM owns checkpoint-layout restoration, layerwise post-load processing,
CUDA-graph-stable tensor placement, and KV-cache scale reload.  This module
only supplies the vLLM 0.20 gaps needed here: ModelOpt W4A16 NVFP4 methods,
rank-local Marlin padding, per-projection ModelOpt MoE input-scale loading,
materialization of FlashInfer's global-scale views, and retention of
method-owned MoE kernel references across layerwise reload.
"""

import copy
from types import MethodType
from typing import Any

import torch
from torch.nn import Parameter

NEMO_MODELOPT_W4A4 = "nemo_modelopt_nvfp4"
NEMO_MODELOPT_W4A16 = "nemo_modelopt_w4a16_nvfp4"

_W4A4_ALGO = "NVFP4"
_W4A16_ALGO = "W4A16_NVFP4"
_registered = False


def quantization_method_for_mode(mode: str) -> str:
    """Return the registered vLLM quantization method for a rollout mode."""
    if mode == "w4a4":
        return NEMO_MODELOPT_W4A4
    if mode == "w4a16":
        return NEMO_MODELOPT_W4A16
    raise ValueError(f"Unsupported ModelOpt NVFP4 rollout mode: {mode!r}")


def _load_modelopt_moe_input_scale(
    moe_layer: Any,
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    return_success: bool = False,
) -> bool | None:
    """Load a ModelOpt input scale without losing the gate/up shard."""
    del weight_name
    global_expert_id = expert_id
    local_expert_id = moe_layer._map_global_expert_id_to_local_expert_id(
        global_expert_id
    )
    use_global_scale = bool(getattr(moe_layer.quant_method, "use_global_sf", False))
    if local_expert_id == -1 and not use_global_scale:
        return False if return_success else None

    target_expert_id = global_expert_id if use_global_scale else local_expert_id
    if shard_id == "w2":
        target = param.data[target_expert_id]
    elif shard_id in ("w1", "w3"):
        shard_index = 0 if shard_id == "w1" else min(1, param.shape[-1] - 1)
        target = param.data[target_expert_id, shard_index]
    else:
        raise ValueError(f"Unexpected ModelOpt MoE shard: {shard_id!r}")

    source = loaded_weight.to(device=target.device, dtype=target.dtype)
    target.copy_(source.reshape_as(target))
    return True if return_success else None


def _normalized_w4a16_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    quantization = normalized.get("quantization")
    target = quantization if isinstance(quantization, dict) else normalized
    if str(target.get("quant_algo", "")).upper() != _W4A16_ALGO:
        raise ValueError(f"{NEMO_MODELOPT_W4A16} requires quant_algo={_W4A16_ALGO!r}")
    # vLLM 0.20 validates known ModelOpt algorithms before dispatching to a
    # custom subclass. Normalize only for its parser; class identity selects
    # the W4A16 methods below.
    target["quant_algo"] = _W4A4_ALGO
    return normalized


def _canonicalize_nvfp4_scale_(scale: torch.Tensor) -> None:
    """Remove the E4M3 sign bit before Marlin's unsigned scale conversion."""
    with torch.no_grad():
        scale.copy_(scale.to(torch.float32).abs().to(scale.dtype))


def _pad_nvfp4_moe_for_marlin(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    is_act_and_mul: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Apply rank-local post-load padding required by the Marlin MoE kernel."""
    num_experts = w13.shape[0]
    num_shards = 2 if is_act_and_mul else 1
    intermediate_size = w13.shape[1] // num_shards
    hidden_size = w13.shape[2] * 2
    if hidden_size % 128 == 0:
        tile_size = 64
    elif hidden_size % 64 == 0:
        tile_size = 128
    else:
        raise ValueError(
            f"W4A16 Marlin MoE requires hidden_size divisible by 64, got {hidden_size}"
        )
    padded_size = (intermediate_size + tile_size - 1) // tile_size * tile_size
    if padded_size == intermediate_size:
        return w13, w13_scale, w2, w2_scale, intermediate_size

    def pad_w13(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.view(
            num_experts,
            num_shards,
            intermediate_size,
            tensor.shape[-1],
        )
        tensor = torch.nn.functional.pad(
            tensor,
            (0, 0, 0, padded_size - intermediate_size),
        )
        return tensor.reshape(num_experts, num_shards * padded_size, -1)

    w13 = pad_w13(w13)
    w13_scale = pad_w13(w13_scale)
    w2 = torch.nn.functional.pad(w2, (0, (padded_size - intermediate_size) // 2))
    w2_scale = torch.nn.functional.pad(
        w2_scale,
        (0, (padded_size - intermediate_size) // 16),
    )
    return w13, w13_scale, w2, w2_scale, padded_size


def register_nemo_modelopt_nvfp4() -> None:
    """Register NeMo's two ModelOpt NVFP4 configs through vLLM's public API."""
    global _registered
    if _registered:
        return

    from vllm.model_executor.kernels.linear import (
        MarlinNvFp4LinearKernel,
        NvFp4LinearLayerConfig,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        is_global_sf_supported_for_nvfp4_backend,
        select_nvfp4_moe_backend,
    )
    from vllm.model_executor.layers.linear import (
        register_weight_loader_v2_supported_method,
    )
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4FusedMoE,
        ModelOptNvFp4LinearMethod,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static
    from vllm.model_executor.utils import replace_parameter

    class NemoModelOptNvFp4FusedMoE(ModelOptNvFp4FusedMoE):
        """Native W4A4 MoE plus the vLLM 0.20 input-scale loader fix."""

        moe_kernel: Any
        moe_quant_config: Any

        def create_weights(self, layer: Any, *args: Any, **kwargs: Any) -> None:
            super().create_weights(layer, *args, **kwargs)
            # Bind to the layer so vLLM's reload metadata sanitizer can remove
            # and restore this reference. A partial would retain the whole model.
            loader = MethodType(_load_modelopt_moe_input_scale, layer)
            layer.w13_input_scale.weight_loader = loader
            layer.w2_input_scale.weight_loader = loader

        def process_weights_after_loading(self, layer: Any) -> None:
            reload_kernel = self.moe_kernel
            reload_quant_config = self.moe_quant_config
            super().process_weights_after_loading(layer)
            processed_quant_config = self.moe_quant_config
            if reload_kernel is None:
                # FlashInfer NVFP4 backends in vLLM 0.20 return global activation
                # scales as stride-zero expanded views. Materialize them before
                # native reload records these Parameters as future copy targets.
                layer.w13_input_scale.data = layer.w13_input_scale.data.contiguous()
                layer.w2_input_scale.data = layer.w2_input_scale.data.contiguous()
                return

            # Native reload copies registered layer tensors into their original
            # CUDA-graph-stable storage. The two reciprocal activation scales are
            # stored only in FusedMoEQuantConfig, so refresh them explicitly while
            # preserving the original config tensor addresses and kernel object.
            if reload_quant_config is None or processed_quant_config is None:
                raise RuntimeError("W4A4 MoE reload is missing its quant config")
            reload_a1_gscale = reload_quant_config.a1_gscale
            processed_a1_gscale = processed_quant_config.a1_gscale
            reload_a2_gscale = reload_quant_config.a2_gscale
            processed_a2_gscale = processed_quant_config.a2_gscale
            if (
                reload_a1_gscale is None
                or processed_a1_gscale is None
                or reload_a2_gscale is None
                or processed_a2_gscale is None
            ):
                raise RuntimeError("W4A4 MoE reload is missing activation scales")
            if (
                reload_a1_gscale.shape != processed_a1_gscale.shape
                or reload_a2_gscale.shape != processed_a2_gscale.shape
            ):
                raise RuntimeError("W4A4 MoE activation-scale shape changed on reload")
            reload_a1_gscale.copy_(processed_a1_gscale)
            reload_a2_gscale.copy_(processed_a2_gscale)
            self.moe_kernel = reload_kernel
            self.moe_quant_config = reload_quant_config

    class NemoModelOptNvFp4Config(ModelOptNvFp4Config):
        FusedMoEMethodCls = NemoModelOptNvFp4FusedMoE

        def get_name(self) -> str:
            return NEMO_MODELOPT_W4A4

        @classmethod
        def override_quantization_method(
            cls,
            hf_quant_cfg: dict[str, Any],
            user_quant: str | None,
            hf_config: Any = None,
        ) -> str | None:
            del hf_config
            if (
                user_quant == NEMO_MODELOPT_W4A4
                and cls._extract_modelopt_quant_algo(hf_quant_cfg) == _W4A4_ALGO
            ):
                return NEMO_MODELOPT_W4A4
            return None

    @register_weight_loader_v2_supported_method
    class NemoModelOptW4A16LinearMethod(ModelOptNvFp4LinearMethod):
        """ModelOpt NVFP4 weights with BF16/FP16 Marlin activations."""

        def __init__(self, quant_config: object) -> None:
            self.quant_config = quant_config
            self.marlin_input_dtype = None
            self.kernel = MarlinNvFp4LinearKernel(NvFp4LinearLayerConfig())

        def create_weights(self, layer: Any, *args: Any, **kwargs: Any) -> None:
            super().create_weights(layer, *args, **kwargs)
            del layer.input_scale

        def process_weights_after_loading(self, layer: Any) -> None:
            layer.weight_global_scale = Parameter(
                layer.weight_scale_2.max().to(torch.float32),
                requires_grad=False,
            )
            del layer.weight_scale_2
            _canonicalize_nvfp4_scale_(layer.weight_scale)
            self.kernel.process_weights_after_loading(layer)

        def apply(
            self,
            layer: Any,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return self.kernel.apply_weights(layer=layer, x=x, bias=bias)

    class NemoModelOptW4A16FusedMoE(ModelOptNvFp4FusedMoE):
        """ModelOpt W4A16 MoE using vLLM's NVFP4 Marlin implementation."""

        moe_kernel: Any
        moe_quant_config: Any

        def __init__(self, quant_config: object, moe_config: object) -> None:
            FusedMoEMethodBase.__init__(self, moe_config)
            self.quant_config = quant_config
            self.nvfp4_backend, self.experts_cls = select_nvfp4_moe_backend(
                config=self.moe,
                weight_key=kNvfp4Static,
                activation_key=None,
            )
            self.use_global_sf = is_global_sf_supported_for_nvfp4_backend(
                self.nvfp4_backend
            )

        def create_weights(
            self,
            layer: Any,
            num_experts: int,
            hidden_size: int,
            intermediate_size_per_partition: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs: Any,
        ) -> None:
            super().create_weights(
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                params_dtype,
                **extra_weight_attrs,
            )
            del layer.w13_input_scale
            del layer.w2_input_scale

        def process_weights_after_loading(self, layer: Any) -> None:
            reload_kernel = self.moe_kernel
            reload_quant_config = self.moe_quant_config
            original_intermediate_size = (
                layer.moe_config.intermediate_size_per_partition
            )
            if self.nvfp4_backend == NvFp4MoeBackend.MARLIN:
                w13, w13_scale, w2, w2_scale, padded_size = _pad_nvfp4_moe_for_marlin(
                    layer.w13_weight,
                    layer.w13_weight_scale,
                    layer.w2_weight,
                    layer.w2_weight_scale,
                    is_act_and_mul=self.moe.is_act_and_mul,
                )
                replace_parameter(layer, "w13_weight", w13)
                replace_parameter(layer, "w13_weight_scale", w13_scale)
                replace_parameter(layer, "w2_weight", w2)
                replace_parameter(layer, "w2_weight_scale", w2_scale)
                _canonicalize_nvfp4_scale_(layer.w13_weight_scale)
                _canonicalize_nvfp4_scale_(layer.w2_weight_scale)
                layer.moe_config.intermediate_size_per_partition = padded_size
            # W4A16 checkpoint metadata deliberately omits activation scales so
            # layerwise reload never waits for tensors that do not exist. The
            # native Marlin converter accepts None and removes these attributes.
            layer.w13_input_scale = None
            layer.w2_input_scale = None
            try:
                super().process_weights_after_loading(layer)
            finally:
                layer.moe_config.intermediate_size_per_partition = (
                    original_intermediate_size
                )
            if reload_kernel is not None:
                self.moe_kernel = reload_kernel
                self.moe_quant_config = reload_quant_config

    class NemoModelOptW4A16Config(ModelOptNvFp4Config):
        LinearMethodCls = NemoModelOptW4A16LinearMethod
        FusedMoEMethodCls = NemoModelOptW4A16FusedMoE

        def get_name(self) -> str:
            return NEMO_MODELOPT_W4A16

        @classmethod
        def override_quantization_method(
            cls,
            hf_quant_cfg: dict[str, Any],
            user_quant: str | None,
            hf_config: Any = None,
        ) -> str | None:
            del hf_config
            if (
                user_quant == NEMO_MODELOPT_W4A16
                and cls._extract_modelopt_quant_algo(hf_quant_cfg) == _W4A16_ALGO
            ):
                return NEMO_MODELOPT_W4A16
            return None

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> Any:
            return super().from_config(_normalized_w4a16_config(config))

    register_quantization_config(NEMO_MODELOPT_W4A4)(NemoModelOptNvFp4Config)
    register_quantization_config(NEMO_MODELOPT_W4A16)(NemoModelOptW4A16Config)
    _registered = True
