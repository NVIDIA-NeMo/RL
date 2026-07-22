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
    """Load a ModelOpt input scale without losing the gate/up shard.

    Replaces the ``input_scale`` branch of vLLM v0.20.0's
    ``FusedMoE.weight_loader``, whose ``_load_single_value`` writes
    ``param.data[expert_id]`` and drops the gate/up (w1/w3) shard index:
    https://github.com/vllm-project/vllm/blob/v0.20.0/vllm/model_executor/layers/fused_moe/layer.py#L1025-L1031
    Delete once upstream loads per-projection ModelOpt MoE input scales
    correctly.
    """
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


def _validated_w4a16_config(config: dict[str, Any]) -> dict[str, Any]:
    quantization = config.get("quantization")
    target = quantization if isinstance(quantization, dict) else config
    if str(target.get("quant_algo", "")).upper() != _W4A16_ALGO:
        raise ValueError(f"{NEMO_MODELOPT_W4A16} requires quant_algo={_W4A16_ALGO!r}")
    # vLLM 0.25 understands W4A16_NVFP4 natively, so the algo passes through
    # unchanged (the base __init__ keys use_a16/LinearMethodCls off it).
    return config


def _canonicalize_nvfp4_scale_(scale: torch.Tensor) -> None:
    """Remove the E4M3 sign bit before Marlin's unsigned scale conversion."""
    with torch.no_grad():
        scale.copy_(scale.to(torch.float32).abs().to(scale.dtype))


def register_nemo_modelopt_nvfp4() -> None:
    """Register NeMo's two ModelOpt NVFP4 configs through vLLM's public API."""
    global _registered
    if _registered:
        return

    from vllm.model_executor.kernels.linear import (
        MarlinNvFp4LinearKernel,
        NvFp4LinearLayerConfig,
    )
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
    from vllm.model_executor.layers.linear import (
        register_weight_loader_v2_supported_method,
    )
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4Config,
        ModelOptNvFp4FusedMoE,
        ModelOptNvFp4LinearMethod,
    )

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

        # Adapted from vLLM v0.20.0 ModelOptNvFp4LinearMethod
        # .process_weights_after_loading/.apply with input-scale/alpha handling
        # removed for weight-only W4A16:
        # https://github.com/vllm-project/vllm/blob/v0.20.0/vllm/model_executor/layers/quantization/modelopt.py#L1169-L1208
        # Re-sync on vLLM bumps; delete if upstream gains a native W4A16
        # NVFP4 method.
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
        """ModelOpt W4A16 MoE using vLLM's NVFP4 Marlin implementation.

        vLLM 0.25's base __init__ keys weight-only mode off
        quant_config.quant_method == "W4A16_NVFP4" (activation_key=None), so
        the 0.20-era duplicated __init__ is gone; the algo passes through
        from_config unchanged.
        """

        moe_kernel: Any
        moe_quant_config: Any

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
            if self.nvfp4_backend == NvFp4MoeBackend.MARLIN:
                # vLLM 0.25's prepare_nvfp4_moe_layer_for_marlin pads the
                # rank-local intermediate tiles itself (and asserts on the
                # unpadded checkpoint shapes), so no NeMo-side pre-padding.
                # Only the E4M3 sign-bit canonicalization of the ModelOpt
                # export remains our concern.
                _canonicalize_nvfp4_scale_(layer.w13_weight_scale)
                _canonicalize_nvfp4_scale_(layer.w2_weight_scale)
            # W4A16 checkpoint metadata deliberately omits activation scales so
            # layerwise reload never waits for tensors that do not exist. The
            # native Marlin converter accepts None and removes these attributes.
            layer.w13_input_scale = None
            layer.w2_input_scale = None
            super().process_weights_after_loading(layer)
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
            instance = super().from_config(_validated_w4a16_config(config))
            # vLLM 0.25's ModelOptNvFp4Config.__init__ selects LinearMethodCls
            # from the quant algo as an *instance* attribute, which shadows
            # the class attribute above; rebind the NeMo method explicitly so
            # W4A16 linears keep the refit-friendly Marlin implementation.
            instance.LinearMethodCls = NemoModelOptW4A16LinearMethod
            return instance

    register_quantization_config(NEMO_MODELOPT_W4A4)(NemoModelOptNvFp4Config)
    register_quantization_config(NEMO_MODELOPT_W4A16)(NemoModelOptW4A16Config)
    _registered = True
