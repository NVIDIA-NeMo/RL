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

import os
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import ray
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.linear import LinearBase
from vllm.triton_utils import tl, triton
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.utils import CoreEngineProcManager

from nemo_rl.models.generation.vllm.quantization.mxfp8_utils import (
    pad_flashinfer_scale_k,
)

logger = init_logger(__name__)

FP8_BLOCK_QUANT_KWARGS = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}

MXFP8_BLOCK_QUANT_KWARGS = {
    "quant_method": "modelopt",
    "quant_algo": "MXFP8",
}

DEFAULT_QUANTIZATION_IGNORED_LAYERS = ("lm_head",)


@dataclass(frozen=True)
class FP8Config:
    use_weight_pow2_scale: bool = False
    use_activation_pow2_scale: bool = False
    num_first_layers_in_bf16: int = 0
    num_last_layers_in_bf16: int = 0
    model_parallel_size: int = None
    kv_cache_dtype: str = "auto"
    use_fp8_weights: bool = True  # Whether model weights are quantized to FP8
    is_mx: bool = False


@dataclass()
class FP8State:
    # A cache of fp8 parameter names, we can check this cache to see if a
    # param name corresponds to a fp8 weight
    seen_params: set = field(default_factory=lambda: set())
    fp8_param_names: set = field(default_factory=lambda: set())
    vllm_patches: list = field(default_factory=lambda: [])


# Global FP8 config that can be accessed by patched vLLM functions
# initialized by 'init_fp8_cfg()'
global_fp8_config: FP8Config = None
# Global FP8 state that holds runtime fp8 objects
fp8_state: FP8State = FP8State()

fp8_patches_applied = False


original_run_engine_core = EngineCoreProc.run_engine_core
original_init = CoreEngineProcManager.__init__


def my_init(*args, **kwargs):
    kwargs["vllm_config"].nrl_fp8_cfg = global_fp8_config
    return original_init(*args, **kwargs)


def my_run_engine_core(*args, **kwargs):
    fp8_cfg = kwargs["vllm_config"].nrl_fp8_cfg
    del kwargs["vllm_config"].nrl_fp8_cfg
    monkey_patch_vllm_ray_executor(fp8_cfg)
    return original_run_engine_core(*args, **kwargs)


def monkey_patch_vllm_ray_executor(fp8_config):
    if fp8_config.model_parallel_size > 1:
        if envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND:
            from vllm.v1.executor.ray_executor_v2 import RayWorkerProc

            original_initialize_worker = RayWorkerProc.initialize_worker

            def patched_initialize_worker(self, *args, **kwargs):
                # Resolve state in the worker's module because cloudpickle snapshots
                # globals referenced by nested functions.
                from nemo_rl.models.generation.vllm.quantization import fp8

                if not fp8.fp8_patches_applied:
                    fp8.apply_fp8_patches(None, fp8_config)

                return original_initialize_worker(self, *args, **kwargs)

            # RayExecutorV2 creates ray.remote(RayWorkerProc) after this hook. Ray
            # copies inherited methods onto its generated actor subclass and
            # serializes it by value, so actors receive this driver-side replacement.
            RayWorkerProc.initialize_worker = patched_initialize_worker
            return

        # we patch vllm's collective_rpc so that before vllm initalizes the model on each rank, we execute
        # a ray remote that patches each worker with the required fp8 vllm patches
        from vllm.v1.executor.ray_executor import RayDistributedExecutor

        original_run_workers = RayDistributedExecutor.collective_rpc

        def patched_run_workers(self, *args, **kwargs):
            global fp8_patches_applied
            if not fp8_patches_applied:
                futures = [
                    worker.execute_method.remote(apply_fp8_patches, fp8_config)
                    for worker in self.workers
                ]
                [ray.get(future) for future in futures]
                fp8_patches_applied = True

            return original_run_workers(self, *args, **kwargs)

        RayDistributedExecutor.collective_rpc = patched_run_workers
    else:
        # for single gpu there is no ray, so just call the patches
        apply_fp8_patches(None, fp8_config)

        global fp8_patches_applied
        fp8_patches_applied = True


def apply_fp8_patches(self, fp8_config):
    global global_fp8_config, fp8_patches_applied
    assert not fp8_patches_applied

    global_fp8_config = fp8_config

    # Apply patches conditionally based on configuration
    # Only apply weight patches if using FP8 weights
    # Only apply KV cache patches if using FP8 KV cache

    # Apply weight-related patches only when using FP8 weights (precision=fp8)
    if global_fp8_config.use_fp8_weights:
        # This patch is used to support torch.compile with vllm parameter subclasses, such as
        # PerTensorScaleParameter. Because we need weight loaders to update fp8 weights each
        # refit, we patch fp8 parameters to have a reference to their weight loader. Eventually
        # with pytorch 2.8, parameter subclassing with torch.compile will be natively supported, in
        # which this patch can be removed.
        func1_path = "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading"
        patcher1 = patch(func1_path, process_weights_after_loading)
        fp8_state.vllm_patches.append(patcher1)
        func2_path = "vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod.process_weights_after_loading"
        patcher2 = patch(func2_path, process_weights_after_loading_moe)
        fp8_state.vllm_patches.append(patcher2)
        if global_fp8_config.is_mx:
            fp8_state.vllm_patches.append(
                patch(
                    "vllm.model_executor.layers.quantization.modelopt.ModelOptMxFp8LinearMethod.process_weights_after_loading",
                    process_weights_after_loading_mxfp8_linear,
                )
            )
            fp8_state.vllm_patches.append(
                patch(
                    "vllm.model_executor.layers.quantization.modelopt.ModelOptMxFp8FusedMoE.create_weights",
                    create_weights_mxfp8_moe,
                )
            )
            fp8_state.vllm_patches.append(
                patch(
                    "vllm.model_executor.layers.quantization.modelopt.ModelOptMxFp8FusedMoE.process_weights_after_loading",
                    process_weights_after_loading_mxfp8_moe,
                )
            )

        # These patches add support for pow2, e8 dynamic activation scalings factors which are believed to have higher
        # SNR compared to plain fp32 scaling factors. This feature is still under active research.
        if global_fp8_config.use_activation_pow2_scale:
            func2_path = "vllm.model_executor.layers.quantization.utils.fp8_utils.per_token_group_quant_fp8"
            func3_path = "vllm.model_executor.layers.quantization.utils.fp8_utils._per_token_group_quant_fp8"
            func4_path = "vllm.model_executor.layers.quantization.utils.fp8_utils._per_token_group_quant_fp8_colmajor"
            patcher2 = patch(func2_path, per_token_group_quant_fp8)
            patcher3 = patch(func3_path, _per_token_group_quant_fp8)
            patcher4 = patch(func4_path, _per_token_group_quant_fp8_colmajor)
            fp8_state.vllm_patches.extend([patcher2, patcher3, patcher4])

        # Static scales mode: preserve k_scale/v_scale for manual updates.
        # DeepSeek V4's MLA cache stores scales inline, so it has no
        # Parameter-style KV scales for this patch to preserve or refit.
        if global_fp8_config.kv_cache_dtype != "fp8_ds_mla":
            func5_path = "vllm.model_executor.layers.quantization.kv_cache.BaseKVCacheMethod.process_weights_after_loading"
            patcher5 = patch(func5_path, process_weights_after_loading_kv)
            fp8_state.vllm_patches.append(patcher5)

    for p in fp8_state.vllm_patches:
        p.start()

    fp8_patches_applied = True


def init_fp8(vllm_cfg, model_name, model_parallel_size):
    global global_fp8_config
    # Determine if we're using FP8 weights based on precision setting
    use_fp8_weights = vllm_cfg.get("precision") == "fp8"
    if vllm_cfg.get("is_mx") and not use_fp8_weights:
        raise ValueError("is_mx=True requires precision='fp8'")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    kv_cache_dtype = vllm_cfg["kv_cache_dtype"]

    # Validate configuration: kv_cache_dtype
    supported_kv_cache_dtypes = ["auto", "fp8", "fp8_e4m3", "fp8_ds_mla"]
    if kv_cache_dtype not in supported_kv_cache_dtypes:
        raise ValueError(
            f"kv_cache_dtype must be one of {supported_kv_cache_dtypes}, but got {kv_cache_dtype}"
        )

    # Validate configuration: kv_cache_dtype=fp8 requires precision=fp8
    if kv_cache_dtype.startswith("fp8") and not use_fp8_weights:
        raise ValueError(
            f"kv_cache_dtype='{kv_cache_dtype}' requires precision='fp8'. "
            "FP8 KV cache can only be used together with FP8 model weights."
        )

    if use_fp8_weights:
        is_mx = bool(vllm_cfg.get("is_mx"))
    else:
        is_mx = False
    quantization_ignore_patterns = vllm_cfg.get("quantization_ignore_patterns")
    if quantization_ignore_patterns is not None:
        if not is_mx:
            raise ValueError("quantization_ignore_patterns requires is_mx=True")
        if isinstance(quantization_ignore_patterns, (str, bytes)) or not isinstance(
            quantization_ignore_patterns, Sequence
        ):
            raise ValueError("quantization_ignore_patterns must be a list of strings")
        if any(
            not isinstance(pattern, str) or not pattern.strip()
            for pattern in quantization_ignore_patterns
        ):
            raise ValueError(
                "quantization_ignore_patterns must contain non-empty strings"
            )
        quantization_ignore_patterns = [
            pattern.strip() for pattern in quantization_ignore_patterns
        ]
    fp8_config_kwargs = {
        "num_first_layers_in_bf16": vllm_cfg.get("num_first_layers_in_bf16", 0),
        "num_last_layers_in_bf16": vllm_cfg.get("num_last_layers_in_bf16", 0),
        "model_parallel_size": model_parallel_size,
        "kv_cache_dtype": kv_cache_dtype,
        "use_fp8_weights": use_fp8_weights,
    }
    if is_mx:
        fp8_config_kwargs["is_mx"] = True
        if vllm_cfg.get("pow2_weight_scaling_factors") is False:
            raise ValueError("only pow2 weight scaling factors are supported for MXFP8")
        if vllm_cfg.get("pow2_activation_scaling_factors") is False:
            raise ValueError(
                "only pow2 activation scaling factors are supported for MXFP8"
            )
    else:
        fp8_config_kwargs["is_mx"] = False
        fp8_config_kwargs["use_weight_pow2_scale"] = vllm_cfg.get(
            "pow2_weight_scaling_factors", False
        )
        fp8_config_kwargs["use_activation_pow2_scale"] = vllm_cfg.get(
            "pow2_activation_scaling_factors", False
        )
    global_fp8_config = FP8Config(**fp8_config_kwargs)

    if vllm_cfg.get("use_deep_gemm", False) and not is_mx:
        os.environ["VLLM_USE_DEEP_GEMM"] = "1"
        # Leave VLLM_USE_DEEP_GEMM_E8M0 unset so vLLM can select the scale
        # format required by the GPU architecture. Blackwell requires UE8M0;
        # Hopper can continue using vLLM's non-E8M0 path.

    if vllm_cfg["async_engine"]:
        # for async engine, vllm spawns a process for each DP, so we patch
        # vllm so that upon spawning the thread it applies our FP8 patches
        EngineCoreProc.run_engine_core = my_run_engine_core
        CoreEngineProcManager.__init__ = my_init
    else:
        # if not async, just directly monkey patch the ray executor
        monkey_patch_vllm_ray_executor(global_fp8_config)

    # create fp8 kwargs for vllm's LLM(...)
    num_first_layers_in_bf16 = vllm_cfg.get("num_first_layers_in_bf16", 0)
    num_last_layers_in_bf16 = vllm_cfg.get("num_last_layers_in_bf16", 0)
    if global_fp8_config.is_mx:
        fp8_block_quant_kwargs = dict(MXFP8_BLOCK_QUANT_KWARGS)
    else:
        # Preserve model-specific checkpoint metadata such as DeepSeek V4's
        # scale_fmt="ue8m0" while keeping NeMo-RL's shared FP8 settings
        # authoritative for keys that overlap.
        disk_quantization_config = getattr(config, "quantization_config", None) or {}
        fp8_block_quant_kwargs = {
            **disk_quantization_config,
            **FP8_BLOCK_QUANT_KWARGS,
        }

    if num_first_layers_in_bf16 > 0 or num_last_layers_in_bf16 > 0:
        with init_empty_weights():
            model = AutoModel.from_config(config)
        param_names = [name for name, _ in model.named_parameters()]

        bf16_params = []
        if num_first_layers_in_bf16 > 0:
            layers = [l for l in range(num_first_layers_in_bf16)]
            bf16_params.extend(_get_params_in_layers(param_names, layers))

        if num_last_layers_in_bf16 > 0:
            layers = [
                l
                for l in range(
                    config.num_hidden_layers - num_last_layers_in_bf16,
                    config.num_hidden_layers,
                )
            ]
            bf16_params.extend(_get_params_in_layers(param_names, layers))

        fp8_block_quant_kwargs["ignored_layers"] = bf16_params
    quantization_ignored_layer_kws = vllm_cfg.get("quantization_ignored_layer_kws")
    if "quantization_ignored_layer_kws" in vllm_cfg:
        warnings.warn(
            "quantization_ignored_layer_kws is deprecated in NeMo RL 0.8; "
            "use quantization_ignore_patterns instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if quantization_ignored_layer_kws:
        with init_empty_weights():
            model = AutoModel.from_config(config)
        param_names = [
            f"model.{name}".removesuffix(".weight").replace(
                "model.backbone.", "backbone."
            )
            for name, _ in model.named_parameters()
        ]
        ignored_layers = [
            n
            for n in param_names
            if any(p in n for p in quantization_ignored_layer_kws)
        ]
        if "ignored_layers" not in fp8_block_quant_kwargs:
            fp8_block_quant_kwargs["ignored_layers"] = ignored_layers
        else:
            fp8_block_quant_kwargs["ignored_layers"].extend(ignored_layers)
        print("ignored_layers", fp8_block_quant_kwargs["ignored_layers"])

    ignored_layers = fp8_block_quant_kwargs.setdefault("ignored_layers", [])
    ignored_layers.extend(DEFAULT_QUANTIZATION_IGNORED_LAYERS)
    fp8_block_quant_kwargs["ignored_layers"] = list(dict.fromkeys(ignored_layers))
    if quantization_ignore_patterns:
        fp8_block_quant_kwargs.setdefault("ignore", []).extend(
            quantization_ignore_patterns
        )

    if "ignored_layers" in fp8_block_quant_kwargs:
        fp8_block_quant_kwargs.setdefault("ignore", []).extend(
            fp8_block_quant_kwargs["ignored_layers"]
        )
    if "ignore" in fp8_block_quant_kwargs:
        fp8_block_quant_kwargs["ignore"] = list(
            dict.fromkeys(fp8_block_quant_kwargs["ignore"])
        )

    # Return FP8 kwargs (precision=fp8 is required at this point)
    vllm_kwargs = {
        "quantization": "fp8",
        "kv_cache_dtype": kv_cache_dtype,
        "hf_overrides": {"quantization_config": fp8_block_quant_kwargs},
    }

    return vllm_kwargs


def is_fp8_model(vllm_config):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    try:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptMxFp8Config,
        )
    except ImportError:
        quant_config_types = (Fp8Config,)
    else:
        quant_config_types = (Fp8Config, ModelOptMxFp8Config)

    if hasattr(vllm_config, "quant_config") and isinstance(
        vllm_config.quant_config, quant_config_types
    ):
        if isinstance(vllm_config.quant_config, Fp8Config):
            assert vllm_config.quant_config.weight_block_size is not None, (
                "Only block scaling is currently supported in NeMo-RL!"
            )
        return True

    return False


def _get_params_in_layers(param_names, layers):
    layer_templates = []
    for i in layers:
        # Prefixes used by huggingface model transformer layers.
        # We'll use these to match against the parameter names to determine
        # which layer the parameter is in.
        layer_templates.extend(
            [
                f"transformer.h.{i}.",
                f"layers.{i}.",
                f"layer.{i}.",
            ]
        )
    prefixes = [p for p in layer_templates if any(p in n for n in param_names)]
    if len(prefixes) == 0:
        raise ValueError(f"Could not identify layers {layers} for model.")

    params = []
    for name in param_names:
        if (
            any(p in name for p in prefixes)
            and "bias" not in name
            and "layernorm" not in name
        ):
            # Convert the param name into vllm's module name
            # Vllm wraps the model with an extra 'model'
            params.append(
                f"model.{name}".removesuffix(".weight").replace(
                    "model.backbone.", "backbone."
                )
            )
    return params


def _get_packed_modules_mapping(model: torch.nn.Module) -> dict[str, list[str]]:
    packed_modules_mapping = dict(getattr(model, "packed_modules_mapping", {}) or {})
    if type(model).__name__.lower() == "deepseekv4forcausallm":
        # vLLM's DeepSeek V4 loader fuses these checkpoint modules without
        # exposing a packed_modules_mapping for refit's FP8 module lookup.
        packed_modules_mapping.update(
            {
                "attn.fused_wqa_wkv": ["attn.wq_a", "attn.wkv"],
                "compressor.fused_wkv_wgate": [
                    "compressor.wkv",
                    "compressor.wgate",
                ],
                "shared_experts.gate_up_proj": [
                    "shared_experts.w1",
                    "shared_experts.w3",
                ],
                "shared_experts.down_proj": ["shared_experts.w2"],
            }
        )
    return packed_modules_mapping


def _replace_packed_module_suffix(
    module_path: list[str], packed_modules_mapping: dict[str, list[str]]
) -> list[str]:
    for fused_name, original_names in packed_modules_mapping.items():
        fused_parts = fused_name.split(".")
        for original_name in original_names:
            original_parts = original_name.split(".")
            if module_path[-len(original_parts) :] == original_parts:
                return module_path[: -len(original_parts)] + fused_parts
    return module_path


def _apply_hf_to_vllm_mapping(
    model: torch.nn.Module, module_path: list[str]
) -> list[str]:
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is None:
        return module_path

    orig_to_new_prefix = getattr(mapper, "orig_to_new_prefix", {}) or {}
    if module_path and module_path[0] in orig_to_new_prefix:
        module_path[0] = orig_to_new_prefix[module_path[0]]

    orig_to_new_substr = getattr(mapper, "orig_to_new_substr", {}) or {}
    return [orig_to_new_substr.get(part, part) for part in module_path]


def _candidate_module_paths(model: torch.nn.Module, name: str) -> list[list[str]]:
    candidate_names = [name]
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    map_name = getattr(mapper, "_map_name", None)
    if map_name is not None:
        mapped_name = map_name(name)
        if mapped_name != name:
            candidate_names.append(mapped_name)

    packed_modules_mapping = _get_packed_modules_mapping(model)
    module_paths = []
    for candidate_name in candidate_names:
        path_parts = candidate_name.split(".")
        if len(path_parts) < 2:
            continue
        module_path = _replace_packed_module_suffix(
            path_parts[:-1], packed_modules_mapping
        )
        module_path = _apply_hf_to_vllm_mapping(model, module_path)
        module_paths.append(module_path)
        if module_path[0] != "model" and hasattr(model, "model"):
            module_paths.append(["model", *module_path])
    return module_paths


def _is_routed_experts_weight_owner(module: object) -> bool:
    """Recognize the module that owns vLLM's fused expert parameters.

    Pluggable MoE backends can replace ``RoutedExperts`` with a generated
    backend-specific module while retaining the same parameter interface.
    Refit operates on that interface, so an exact class check is too narrow.
    """
    return isinstance(module, RoutedExperts) or (
        module is not None
        and all(
            isinstance(getattr(module, name, None), torch.Tensor)
            for name in ("w13_weight", "w2_weight")
        )
    )


def _resolve_module_path(
    model: torch.nn.Module, module_path: list[str]
) -> torch.nn.Module | None:
    current_module = model
    try:
        # Traverse the model hierarchy
        for part in module_path:
            # vLLM 0.25 split the old FusedMoE module into a MoERunner that
            # delegates to a RoutedExperts submodule owning the expert weights
            # (w13_weight/w2_weight), so stop at either and return the
            # weight-owning module.
            routed_experts = getattr(current_module, "routed_experts", None)
            if isinstance(current_module, MoERunner) or (
                routed_experts is not None
                and _is_routed_experts_weight_owner(routed_experts)
            ):
                return routed_experts
            if _is_routed_experts_weight_owner(current_module):
                return current_module
            if part == "model" and not hasattr(current_module, part):
                # Some HF/vLLM model classes expose the decoder directly (for
                # example ``language_model``) while parameter names still carry
                # vLLM's synthetic ``model.`` prefix.
                continue
            if part == "layers" and not hasattr(current_module, part):
                # Qwen3.5-MoE VL exposes ``language_model`` as a CausalLM
                # wrapper; its decoder stack lives under ``language_model.model``.
                wrapped_model = getattr(current_module, "model", None)
                if wrapped_model is not None and hasattr(wrapped_model, part):
                    current_module = wrapped_model
            if isinstance(current_module, torch.nn.ModuleList):
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)
    except (AttributeError, IndexError, ValueError):
        return None
    # Fused param names (e.g. "...experts.w13_weight") end the traversal on the
    # MoERunner itself; normalize to the weight-owning RoutedExperts submodule.
    routed_experts = getattr(current_module, "routed_experts", None)
    if isinstance(current_module, MoERunner) or (
        routed_experts is not None
        and _is_routed_experts_weight_owner(routed_experts)
    ):
        return routed_experts
    return current_module


def _get_module_from_param_name(
    model: torch.nn.Module, name: str
) -> torch.nn.Module | None:
    for module_path in _candidate_module_paths(model, name):
        module = _resolve_module_path(model, module_path)
        if module is not None:
            return module
    print(f"Warning: Could not find module for parameter '{name}'.")
    return None


def _is_fp8_weight(name, model):
    if name not in fp8_state.seen_params:
        fp8_state.seen_params.add(name)
        # Filter out bias params
        if name.endswith("weight"):
            module = _get_module_from_param_name(model, name)
            # We currently only quantize linear layers
            if (
                isinstance(module, LinearBase)
                and module.weight.dtype == torch.float8_e4m3fn
                or (
                    _is_routed_experts_weight_owner(module)
                    and module.w13_weight.dtype == torch.float8_e4m3fn
                    and module.w2_weight.dtype == torch.float8_e4m3fn
                )
            ):
                fp8_state.fp8_param_names.add(name)
    return name in fp8_state.fp8_param_names


def quantize_mxfp8_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a checkpoint-layout weight for MXFP8 weight loading and refit.

    FlashInfer represents all-zero blocks with E8M0 scale byte 0. Replace those
    scale entries with byte 1 because the TRTLLM kernel does not accept byte 0;
    the represented values remain zero because every quantized value in each
    affected block is zero.
    """
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    value, scale = mxfp8_e4m3_quantize(weight)
    value = value.reshape(weight.shape)
    scale = scale.reshape(*weight.shape[:-1], weight.shape[-1] // 32)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return value, scale


def _is_deepseek_v4_model(model: torch.nn.Module) -> bool:
    if type(model).__name__.lower() == "deepseekv4forcausallm":
        return True
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = getattr(config, "architectures", []) or []
    return "deepseek_v4" in model_type or any(
        "deepseekv4" in str(architecture).lower() for architecture in architectures
    )


def _format_tensor_for_weight_load_debug(tensor: object) -> str:
    if not isinstance(tensor, torch.Tensor):
        return repr(type(tensor))
    return f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"


def _get_param_module_name(param_name: str) -> str | None:
    for suffix in (".weight_scale_inv", ".weight_scale", ".weight"):
        if param_name.endswith(suffix):
            return param_name[: -len(suffix)]
    return None


def _get_deepseek_v4_bmm_tp_rank(*objects: object) -> int:
    for obj in objects:
        tp_rank = getattr(obj, "tp_rank", None)
        if tp_rank is not None:
            return int(tp_rank)
    # Import lazily because distributed state is unavailable in CPU-only tests.
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:
        return 0


def _copy_deepseek_v4_bmm_weight(
    layer: torch.nn.Module,
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
) -> bool:
    if param.data.ndim != 3 or loaded_weight.ndim != 2:
        return False

    num_groups, rows_per_group, hidden_size = param.data.shape
    local_rows = num_groups * rows_per_group
    tp_rank = _get_deepseek_v4_bmm_tp_rank(param, layer.weight, layer)
    start = tp_rank * local_rows
    if loaded_weight.shape[0] < start + local_rows:
        return False
    if loaded_weight.shape[1] != hidden_size:
        return False

    local_weight = loaded_weight.narrow(0, start, local_rows)
    param.data.copy_(local_weight.view(num_groups, rows_per_group, hidden_size))
    return True


def _copy_deepseek_v4_bmm_scale(
    layer: torch.nn.Module,
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
) -> bool:
    if loaded_weight.ndim != 2:
        return False
    weight = getattr(layer, "weight", None)
    if weight is None or weight.data.ndim != 3:
        return False

    num_groups, rows_per_group, hidden_size = weight.data.shape
    block_m, block_n = tuple(layer.weight_block_size)
    local_scale_rows = (num_groups * rows_per_group + block_m - 1) // block_m
    tp_rank = _get_deepseek_v4_bmm_tp_rank(param, weight, layer)
    start = tp_rank * local_scale_rows
    if loaded_weight.shape[0] < start + local_scale_rows:
        return False

    local_scale = loaded_weight.narrow(0, start, local_scale_rows)
    if local_scale.shape == param.data.shape:
        param.data.copy_(local_scale)
        return True
    if hidden_size % block_n != 0 or rows_per_group % block_m != 0:
        return False

    scale_groups = local_scale.view(
        num_groups,
        rows_per_group // block_m,
        hidden_size // block_n,
    )
    # Import lazily because DeepGEMM is an optional vLLM runtime dependency.
    from vllm.utils.deep_gemm import transform_sf_into_required_layout

    scale_groups = transform_sf_into_required_layout(
        sf=scale_groups,
        mn=rows_per_group,
        k=hidden_size,
        recipe=(1, block_m, block_n),
        num_groups=num_groups,
        is_sfa=False,
    )
    if scale_groups.shape != param.data.shape:
        return False
    param.data.copy_(scale_groups)
    return True


def _try_load_deepseek_v4_bmm_param(
    module_map: dict[str, torch.nn.Module],
    param_name: str,
    param: torch.nn.Parameter,
    loaded_weight: object,
) -> bool:
    if not isinstance(loaded_weight, torch.Tensor):
        return False
    module_name = _get_param_module_name(param_name)
    if module_name is None:
        return False
    layer = module_map.get(module_name)
    if not getattr(layer, "is_bmm", False):
        return False
    if param_name.endswith(".weight"):
        return _copy_deepseek_v4_bmm_weight(layer, param, loaded_weight)
    if param_name.endswith((".weight_scale", ".weight_scale_inv")):
        return _copy_deepseek_v4_bmm_scale(layer, param, loaded_weight)
    return False


def _get_moe_module_name(param_name: str) -> str | None:
    for suffix in (
        ".w13_weight_scale_inv",
        ".w2_weight_scale_inv",
        ".w13_weight_scale",
        ".w2_weight_scale",
    ):
        if param_name.endswith(suffix):
            return param_name[: -len(suffix)]
    return None


def _get_routed_experts_module(
    module_map: dict[str, torch.nn.Module], module_name: str
) -> torch.nn.Module | None:
    layer = module_map.get(module_name)
    routed_experts = getattr(layer, "routed_experts", None)
    if isinstance(layer, MoERunner) or (
        routed_experts is not None
        and _is_routed_experts_weight_owner(routed_experts)
    ):
        layer = routed_experts
    return layer if _is_routed_experts_weight_owner(layer) else None


def _get_moe_tp_size_and_rank(layer: RoutedExperts) -> tuple[int, int]:
    moe_config = layer.moe_config
    parallel_config = getattr(moe_config, "moe_parallel_config", None)
    tp_size = getattr(moe_config, "tp_size", None)
    tp_rank = getattr(moe_config, "tp_rank", None)
    if tp_size is None:
        tp_size = getattr(parallel_config, "tp_size", 1)
    if tp_rank is None:
        tp_rank = getattr(parallel_config, "tp_rank", 0)
    return int(tp_size), int(tp_rank)


def _try_load_deepseek_v4_moe_block_scale(
    module_map: dict[str, torch.nn.Module],
    param_name: str,
    param: torch.nn.Parameter,
    loaded_weight: object,
    weight_name: str | None,
    shard_id: str | None,
    expert_id: int | None,
) -> bool | None:
    if not isinstance(loaded_weight, torch.Tensor):
        return None
    if loaded_weight.ndim != 2 or param.data.ndim != 3:
        return None
    if weight_name is None or shard_id is None or expert_id is None:
        return None

    module_name = _get_moe_module_name(param_name)
    if module_name is None:
        return None
    layer = _get_routed_experts_module(module_map, module_name)
    if layer is None:
        return None

    local_expert_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
    if local_expert_id == -1:
        return False

    expert_data = param.data[local_expert_id]
    loaded_weight = loaded_weight.to(device=expert_data.device)
    tp_size, tp_rank = _get_moe_tp_size_and_rank(layer)

    def tp_shard(scale: torch.Tensor, dim: int) -> torch.Tensor:
        if tp_size <= 1:
            return scale
        full_size = scale.shape[dim]
        if full_size % tp_size != 0:
            raise ValueError(
                "DeepSeek V4 MoE scale refit cannot TP-shard "
                f"dim={dim} size={full_size} across tp_size={tp_size} "
                f"for {param_name}"
            )
        shard_size = full_size // tp_size
        return scale.narrow(dim, tp_rank * shard_size, shard_size)

    if "w13_weight_scale" in param_name:
        if shard_id not in ("w1", "w3"):
            return None
        per_projection_rows = expert_data.shape[0] // 2
        if loaded_weight.shape[0] != per_projection_rows:
            loaded_weight = tp_shard(loaded_weight, dim=0)
        if loaded_weight.shape != (per_projection_rows, expert_data.shape[1]):
            raise ValueError(
                "DeepSeek V4 MoE w13 scale refit shape mismatch: "
                f"param={tuple(expert_data.shape)}, "
                f"loaded={tuple(loaded_weight.shape)}, shard_id={shard_id}"
            )
        row_offset = 0 if shard_id == "w1" else per_projection_rows
        expert_data.narrow(0, row_offset, per_projection_rows).copy_(loaded_weight)
        return True

    if "w2_weight_scale" in param_name:
        if shard_id != "w2":
            return None
        if loaded_weight.shape != expert_data.shape:
            loaded_weight = tp_shard(loaded_weight, dim=1)
        if loaded_weight.shape != expert_data.shape:
            raise ValueError(
                "DeepSeek V4 MoE w2 scale refit shape mismatch: "
                f"param={tuple(expert_data.shape)}, "
                f"loaded={tuple(loaded_weight.shape)}"
            )
        expert_data.copy_(loaded_weight)
        return True
    return None


def _wrap_deepseek_v4_weight_loaders_for_refit(
    model: torch.nn.Module,
) -> list[tuple[torch.nn.Parameter, Any]]:
    original_weight_loaders: list[tuple[torch.nn.Parameter, Any]] = []
    module_map = dict(model.named_modules())
    for param_name, param in model.named_parameters():
        weight_loader = getattr(param, "weight_loader", None)
        if weight_loader is None:
            continue

        def refit_weight_loader(
            *args: Any,
            _param_name: str = param_name,
            _param: torch.nn.Parameter = param,
            _weight_loader: Any = weight_loader,
            **kwargs: Any,
        ) -> Any:
            loaded_weight = args[1] if len(args) > 1 else kwargs.get("loaded_weight")
            weight_name = args[2] if len(args) > 2 else kwargs.get("weight_name")
            shard_id = args[3] if len(args) > 3 else kwargs.get("shard_id")
            expert_id = args[4] if len(args) > 4 else kwargs.get("expert_id")
            return_success = (
                args[5] if len(args) > 5 else kwargs.get("return_success", False)
            )
            try:
                moe_scale_result = _try_load_deepseek_v4_moe_block_scale(
                    module_map=module_map,
                    param_name=_param_name,
                    param=_param,
                    loaded_weight=loaded_weight,
                    weight_name=weight_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                if moe_scale_result is not None:
                    return moe_scale_result if return_success else None
                if _try_load_deepseek_v4_bmm_param(
                    module_map, _param_name, _param, loaded_weight
                ):
                    return True if return_success else None
                return _weight_loader(*args, **kwargs)
            except Exception:
                print(
                    "NeMo-RL FP8 refit failed while loading vLLM parameter "
                    f"{_param_name}: "
                    f"param {_format_tensor_for_weight_load_debug(_param)}, "
                    f"loaded {_format_tensor_for_weight_load_debug(loaded_weight)}"
                )
                raise

        original_weight_loaders.append((param, weight_loader))
        setattr(param, "weight_loader", refit_weight_loader)
    return original_weight_loaders


def _restore_weight_loaders(
    original_weight_loaders: list[tuple[torch.nn.Parameter, Any]],
) -> None:
    for param, weight_loader in original_weight_loaders:
        setattr(param, "weight_loader", weight_loader)


def load_weights(weights, model_runner):
    fp8_config = global_fp8_config
    weights_quantized = []
    model = model_runner.model
    for k, v in weights:
        # Grouped MoE experts arrive as fused slabs without a ``.weight`` suffix
        # (so `_is_fp8_weight` would skip them) and vLLM's grouped loader cannot
        # load their per-block scales. Expand them into the per-expert FP8 (w13, w2 -> w1, w2, and w3)
        # layout, then reshape to 2D [num_experts, out_features, in_features] -> [num_experts*out_features, in_features]
        # so the block scales can be quantized and routed correctly.
        if k.endswith("mlp.experts.gate_up_proj") or k.endswith(
            "mlp.experts.down_proj"
        ):
            # Quantize only if vLLM built this layer's experts as FP8. Experts
            # covered by ``ignored_layers`` (num_{first,last}_layers_in_bf16 /
            # quantization_ignored_layer_kws) are built unquantized, with bf16
            # w13/w2 and no ``*_weight_scale_inv`` params, so the per-expert
            # FP8 + scale entries would have nowhere to load. Pass the grouped
            # bf16 slab through instead; vLLM's fused expert mapping loads it
            # directly, same as a bf16 refit.
            experts_module = _get_module_from_param_name(model, k)
            if (
                isinstance(experts_module, RoutedExperts)
                and experts_module.w13_weight.dtype == torch.float8_e4m3fn
                and experts_module.w2_weight.dtype == torch.float8_e4m3fn
            ):
                if global_fp8_config.is_mx:
                    raise NotImplementedError(
                        "MXFP8 refit does not support grouped MoE expert weights."
                    )
                weights_quantized.extend(_expand_grouped_moe_expert_to_fp8(k, v))
            else:
                weights_quantized.append((k, v))
            continue
        if not _is_fp8_weight(k, model):
            weights_quantized.append((k, v))
            continue
        # Cast the weight into fp8 and its scale factor
        if global_fp8_config.is_mx:
            param_lp, param_scale = quantize_mxfp8_weight(v)
        else:
            param_lp, param_scale = cast_tensor_to_fp8_blockwise(
                v.to(torch.float),
                weight_block_size=FP8_BLOCK_QUANT_KWARGS["weight_block_size"],
                fp8_config=fp8_config,
            )
        param_scale = torch.squeeze(param_scale, dim=-1)
        if fp8_config.is_mx:
            weights_quantized.append([k, param_lp])
            weights_quantized.append([k + "_scale_from_checkpoint", param_scale])
        else:
            weights_quantized.append([k, param_lp])
            weights_quantized.append([k + "_scale_inv", param_scale])
    if not _is_deepseek_v4_model(model):
        model.load_weights(weights_quantized)
        return

    original_weight_loaders = _wrap_deepseek_v4_weight_loaders_for_refit(model)
    try:
        model.load_weights(weights_quantized)
    finally:
        _restore_weight_loaders(original_weight_loaders)


def cast_tensor_to_fp8_blockwise(
    data_hp,
    weight_block_size,
    fp8_config: FP8Config | None = None,
):
    assert len(data_hp.shape) == 2, "Only 2d input tensor is supported"

    block_size1 = weight_block_size[1]
    block_size0 = weight_block_size[0]
    shape_before_padding = data_hp.shape
    # pad data_hp to make its shape a multiple of weight_block_size with the last element of data_hp
    if data_hp.shape[1] % block_size1 != 0 or data_hp.shape[0] % block_size0 != 0:
        pad1 = (
            0
            if data_hp.shape[1] % block_size1 == 0
            else block_size1 - data_hp.shape[1] % block_size1
        )
        pad0 = (
            0
            if data_hp.shape[0] % block_size0 == 0
            else block_size0 - data_hp.shape[0] % block_size0
        )
        print(
            f"Padding data_hp from {data_hp.shape} to {(data_hp.shape[0] + pad0, data_hp.shape[1] + pad1)}"
        )
        data_hp = torch.nn.functional.pad(
            data_hp, (0, pad1, 0, pad0), mode="constant", value=data_hp[-1, -1]
        )

    # FP8
    max_dtype = torch.finfo(torch.float8_e4m3fn).max

    original_shape = data_hp.shape
    blk_m, blk_n = data_hp.shape[0] // block_size0, data_hp.shape[1] // block_size1

    assert block_size1 == block_size0
    data_hp = data_hp.reshape(blk_m, block_size0, blk_n, block_size1)

    # Permute to (BLK_M, BLK_N, BLOCK_SIZE_M, BLOCK_SIZE_N)
    data_hp = data_hp.permute(0, 2, 1, 3)
    # Flatten to (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    data_hp = data_hp.to(torch.float32).contiguous().flatten(start_dim=2)

    # Calculate max absolute value per block
    max_abs = torch.amax(torch.abs(data_hp), dim=-1, keepdim=True)
    # Calculate descale factor
    descale = max_abs / max_dtype

    if fp8_config is None:
        fp8_config = global_fp8_config
    if fp8_config.use_weight_pow2_scale:
        exponent = torch.ceil(torch.log2(descale))
        # Post process exponent to be in range of -127 to 127 and to be E8M0 biased
        exponent = torch.clamp(exponent, min=-127, max=127) + 127
        # Convert to uint8 container
        exponent = exponent.to(torch.uint8)
        # Calculate descale_fp to apply to data_hp
        scale_fp = torch.where(
            # If exponent is 0, descale_fp is 1.0 rather than 2^127
            exponent == 0,
            1.0,
            torch.exp2(127 - exponent.to(torch.float32)),
        )
        descale_fp = torch.reciprocal(scale_fp)
    else:
        scale_fp = max_dtype / max_abs
        scale_fp = torch.where(max_abs == 0, 1.0, scale_fp)
        # preserve the behavior for 0 amax case
        scale_fp = torch.where(max_abs == torch.inf, 1.0, scale_fp)

        descale_fp = torch.reciprocal(scale_fp)

    # Scale and saturate cast the data elements to max of target dtype
    data_lp = torch.clamp(data_hp * scale_fp, min=-1 * max_dtype, max=max_dtype)

    fp_data = data_lp.to(torch.float8_e4m3fn)

    # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
    fp_data = (
        fp_data.reshape(blk_m, blk_n, block_size0, block_size1)
        .permute(0, 2, 1, 3)
        .reshape(original_shape)
    )

    # remove the padding
    if data_hp.shape != shape_before_padding:
        fp_data = fp_data[: shape_before_padding[0], : shape_before_padding[1]]

    # Convert to target format, but still in original precision container
    return fp_data, descale_fp


def _quantize_grouped_experts_blockwise(grouped_moe_expert):
    """Block-FP8 quantize a grouped MoE expert slab expert-by-expert.

    Args:
        grouped_moe_expert: A bf16 grouped expert weight of shape
            ``[num_experts, out_features, in_features]`` (one unfused
            projection, e.g. all experts' ``gate_proj``).

    Returns:
        A tuple ``(weight_fp8, scale_inv)`` where ``weight_fp8`` matches
        ``grouped_moe_expert`` in shape with dtype ``float8_e4m3fn`` and ``scale_inv`` has
        shape ``[num_experts, out_features // block0, in_features // block1]``.
    """
    block0, block1 = FP8_BLOCK_QUANT_KWARGS["weight_block_size"]
    num_experts, out_features, in_features = grouped_moe_expert.shape
    assert out_features % block0 == 0 and in_features % block1 == 0, (
        f"Grouped expert shape {tuple(grouped_moe_expert.shape)} is not aligned to FP8 "
        f"block size {(block0, block1)}; per-expert block quantization would "
        "pad across expert boundaries."
    )

    # Quantize expert-by-expert rather than as one flat [E*out, in] tensor:
    # the fp32 upcast and cast-internal copies then peak at 1/num_experts the
    # size (4.75 GiB -> 2.25 GiB on the 35B-A3B gate_up slab), and this runs
    # during refit next to a live vLLM allocation. Bitwise-identical to the
    # flat path: the divisibility assert above means no 128-row block ever
    # straddles an expert boundary, so per-block amax (and hence scales) see
    # the same elements either way.
    weight_fp8 = torch.empty_like(grouped_moe_expert, dtype=torch.float8_e4m3fn)
    scale_inv = torch.empty(
        (num_experts, out_features // block0, in_features // block1),
        dtype=torch.float32,
        device=grouped_moe_expert.device,
    )
    for expert_id in range(num_experts):
        expert_fp8, expert_scale_inv = cast_tensor_to_fp8_blockwise(
            grouped_moe_expert[expert_id].to(torch.float),
            weight_block_size=FP8_BLOCK_QUANT_KWARGS["weight_block_size"],
        )
        weight_fp8[expert_id].copy_(expert_fp8)
        scale_inv[expert_id].copy_(torch.squeeze(expert_scale_inv, dim=-1))
    return weight_fp8, scale_inv


def _expand_grouped_moe_expert_to_fp8(key, weight):
    """Expand a grouped Qwen3.5 MoE expert slab into per-expert FP8 weights.

    NeMo-RL's Megatron export streams the experts of ``Qwen3_5MoeFor*`` models
    as two grouped slabs per layer (``mlp.experts.gate_up_proj`` and
    ``mlp.experts.down_proj``). vLLM's grouped FusedMoE loader for these models
    maps only the fused weight and has no path to load a per-block
    ``weight_scale_inv``, so block-FP8 experts would silently run with an
    identity (scale=1) block scale. Re-emitting the experts in the per-expert
    layout that other FusedMoE checkpoints use
    (``experts.{id}.{gate_proj,up_proj,down_proj}``) routes through vLLM's
    standard expert mapping, which loads both the FP8 weight and its block scale
    correctly.

    Args:
        key: The grouped expert parameter name, ending in
            ``mlp.experts.gate_up_proj`` or ``mlp.experts.down_proj``.
        weight: The bf16 grouped expert tensor. ``gate_up_proj`` is
            ``[num_experts, 2 * intermediate, hidden]`` (gate then up along
            dim 1); ``down_proj`` is ``[num_experts, hidden, intermediate]``.

    Returns:
        A list of ``(name, tensor)`` pairs: for every expert, the FP8 weight and
        its ``_scale_inv`` for each unfused projection.
    """
    base, proj = key.rsplit(".", 1)
    if proj == "gate_up_proj":
        intermediate = weight.shape[1] // 2
        shards = (
            ("gate_proj", weight[:, :intermediate, :]),
            ("up_proj", weight[:, intermediate:, :]),
        )
    else:
        shards = (("down_proj", weight),)

    entries = []
    # gate/up are dim-1 slices; feed the views directly — per-expert rows stay
    # consecutive because the export side hands over a contiguous slab, and a
    # .contiguous() here would materialize a 0.5 GiB copy at refit peak.
    for shard_name, grouped_moe_expert in shards:
        weight_fp8, scale_inv = _quantize_grouped_experts_blockwise(grouped_moe_expert)
        for expert_id in range(weight_fp8.shape[0]):
            name = f"{base}.{expert_id}.{shard_name}.weight"
            entries.append((name, weight_fp8[expert_id]))
            entries.append((name + "_scale_inv", scale_inv[expert_id]))
    return entries


def _assign_param_data(param, value: torch.Tensor) -> None:
    """Update tensor data while preserving the Parameter and its weight loader."""
    if param.data.shape == value.shape and param.data.dtype == value.dtype:
        if param.data.data_ptr() != value.data_ptr():
            param.data.copy_(value)
    else:
        param.data = value


def _iter_refit_e8m0_fp8_linears(
    model: torch.nn.Module,
) -> Iterator[torch.nn.Module]:
    # Import lazily because DeepGEMM is an optional vLLM runtime dependency.
    from vllm.utils.deep_gemm import should_use_deepgemm_for_fp8_linear

    for layer in model.modules():
        if not isinstance(layer, LinearBase) or getattr(layer, "is_bmm", False):
            continue
        weight = getattr(layer, "weight", None)
        orig_dtype = getattr(layer, "orig_dtype", None)
        if (
            getattr(layer, "weight_block_size", None) is None
            or weight is None
            or weight.data.ndim != 2
            or orig_dtype is None
        ):
            continue
        if should_use_deepgemm_for_fp8_linear(orig_dtype, tuple(weight.shape)):
            yield layer


def _reset_e8m0_fp8_linear_scales_for_refit(model: torch.nn.Module) -> None:
    for layer in _iter_refit_e8m0_fp8_linears(model):
        block_m, block_n = tuple(layer.weight_block_size)
        rows = (layer.weight.shape[0] + block_m - 1) // block_m
        cols = (layer.weight.shape[1] + block_n - 1) // block_n
        for scale_name in ("weight_scale_inv", "weight_scale"):
            scale = getattr(layer, scale_name, None)
            if scale is None:
                continue
            raw_shape = (rows, cols)
            if scale.data.shape == raw_shape and scale.data.dtype == torch.float32:
                continue
            scale.data = torch.empty(
                raw_shape, dtype=torch.float32, device=scale.data.device
            )


def _reset_fp8_moe_params_for_refit(model: torch.nn.Module) -> None:
    """Restore RoutedExperts weights and scales to their pre-kernel layout."""
    for layer in model.modules():
        if not _is_routed_experts_weight_owner(layer):
            continue
        weight_block_size = getattr(layer, "weight_block_size", None)
        if weight_block_size is None:
            continue

        moe_config = getattr(layer, "moe_config", None)

        def dimension(name: str, fallback: int | None = None) -> int:
            value = getattr(layer, name, None)
            if value is None:
                value = getattr(moe_config, name, None)
            if value is None:
                value = fallback
            if value is None:
                raise ValueError(
                    f"DeepSeek V4 MoE refit cannot determine {name} for "
                    f"{type(layer).__name__}"
                )
            return int(value)

        num_experts = dimension("num_experts", layer.w13_weight.shape[0])
        hidden_size = dimension("hidden_size")
        intermediate_size = dimension("intermediate_size_per_partition")
        raw_weight_shapes = {
            "w13": (num_experts, 2 * intermediate_size, hidden_size),
            "w2": (num_experts, hidden_size, intermediate_size),
        }
        for prefix, raw_shape in raw_weight_shapes.items():
            weight = getattr(layer, f"{prefix}_weight", None)
            if weight is None:
                continue
            if weight.data.shape != raw_shape or weight.data.dtype != torch.float8_e4m3fn:
                weight.data = torch.empty(
                    raw_shape,
                    dtype=torch.float8_e4m3fn,
                    device=weight.data.device,
                )

        block_m, block_n = (int(size) for size in weight_block_size)
        for prefix, weight in (
            ("w13", getattr(layer, "w13_weight", None)),
            ("w2", getattr(layer, "w2_weight", None)),
        ):
            if weight is None or weight.data.ndim != 3:
                continue
            num_experts, output_dim, input_dim = weight.data.shape
            raw_shape = (
                num_experts,
                (output_dim + block_m - 1) // block_m,
                (input_dim + block_n - 1) // block_n,
            )
            for scale_suffix in ("weight_scale_inv", "weight_scale"):
                scale = getattr(layer, f"{prefix}_{scale_suffix}", None)
                if scale is None:
                    continue
                if scale.data.shape == raw_shape and scale.data.dtype == torch.float32:
                    continue
                scale.data = torch.empty(
                    raw_shape,
                    dtype=torch.float32,
                    device=scale.data.device,
                )


def prepare_fp8_model_for_refit(model_runner: Any) -> None:
    """Restore refit-loadable parameters before a streamed FP8 weight update."""
    # Import lazily because DeepGEMM is an optional vLLM runtime dependency.
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

    model = model_runner.model
    is_deepseek_v4 = _is_deepseek_v4_model(model)
    if is_deep_gemm_e8m0_used():
        _reset_e8m0_fp8_linear_scales_for_refit(model)
    if is_deepseek_v4:
        # MoE backends may pack expert weights even when the dense-linear E8M0
        # scale path is disabled, so this restoration must not share its gate.
        _reset_fp8_moe_params_for_refit(model)


# Ref: https://github.com/vllm-project/vllm/blob/275de34170654274616082721348b7edd9741d32/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L1175
# Patches this method to not create new torch.nn.Parameter for layer weights
# to maintain weight loaders.
def maybe_post_process_fp8_weight_block(layer: torch.nn.Module):
    assert layer.weight_block_size is not None

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )
    from vllm.utils.deep_gemm import (
        is_deep_gemm_e8m0_used,
        should_use_deepgemm_for_fp8_linear,
    )

    is_bmm = getattr(layer, "is_bmm", False)
    if is_bmm and layer.weight.data.ndim == 3:
        # The DSv4 refit loader writes BMM weights and scales directly into
        # DeepGEMM's runtime layout.
        return

    # On Blackwell or Hopper, if E8M0 for DeepGemm is used, we need to
    # requantize the weight and input to the specific scale
    # at the same time.
    should_use_deepgemm = should_use_deepgemm_for_fp8_linear(
        layer.orig_dtype, tuple(layer.weight.shape)
    )
    if should_use_deepgemm:
        bmm_batch_size = getattr(layer, "bmm_batch_size", 0)
        scale_param = getattr(layer, "weight_scale", None)
        if scale_param is None:
            scale_param = layer.weight_scale_inv
        dg_weight, dg_weight_scale = deepgemm_post_process_fp8_weight_block(
            wq=layer.weight.data,
            ws=scale_param.data,
            quant_block_shape=tuple(layer.weight_block_size),
            use_e8m0=is_deep_gemm_e8m0_used(),
            is_bmm=is_bmm,
            bmm_batch_size=bmm_batch_size,
        )
        _assign_param_data(layer.weight, dg_weight)
        _assign_param_data(scale_param, dg_weight_scale)
        for scale_name in ("weight_scale", "weight_scale_inv"):
            mirrored_scale = getattr(layer, scale_name, None)
            if mirrored_scale is not None and mirrored_scale is not scale_param:
                _assign_param_data(mirrored_scale, dg_weight_scale)


def process_weights_after_loading(self, layer) -> None:
    """This function is used to process the weights after loading for a Linear layer.

    Compared to the original process_weights_after_loading in vllm, we avoid re-registering
    Parameter objects so their identity (and weight_loader attribute) survives for refit.

    Updated for vLLM 0.25, which moved the block-quant weight processing into the
    fp8_linear kernel classes (Fp8BlockScaledMMLinearKernel and subclasses) and keeps
    the block scale under the ``weight_scale_inv`` name (the forward path reads
    ``weight_scale_inv``, not ``weight_scale``).
    """
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        process_fp8_weight_block_strategy,
    )

    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"

    weight, weight_scale = process_fp8_weight_block_strategy(
        layer.weight, layer.weight_scale_inv
    )
    # Preserve Parameter identity (and weight_loader) for refit, and prefer
    # in-place copies over .data rebinding once the processed layout is
    # stable: this runs on every refit, and rebinding every linear layer's
    # weight/scale to fresh allocations each step slowly fragments GPU
    # memory until CuMemAllocator wake_up OOMs (observed as
    # "CUDA Error: out of memory at csrc/cumem_allocator.cpp" ~75 steps
    # into fp8-rollouts runs). The first call may change shapes (layout
    # transforms), so fall back to rebinding then.
    if (
        layer.weight.data.shape == weight.shape
        and layer.weight.data.dtype == weight.dtype
    ):
        if layer.weight.data.data_ptr() != weight.data_ptr():
            layer.weight.data.copy_(weight)
    else:
        layer.weight.data = weight.data
    if (
        layer.weight_scale_inv.data.shape == weight_scale.shape
        and layer.weight_scale_inv.data.dtype == weight_scale.dtype
    ):
        if layer.weight_scale_inv.data.data_ptr() != weight_scale.data_ptr():
            layer.weight_scale_inv.data.copy_(weight_scale)
    else:
        layer.weight_scale_inv.data = weight_scale.data

    maybe_post_process_fp8_weight_block(layer)

    # vLLM's apply() forward pass accesses layer.input_scale when block_quant=True.
    # The original process_weights_after_loading sets input_scale = None for dynamic activation
    # with block quantization. We must do the same to avoid AttributeError.
    if not hasattr(layer, "input_scale"):
        layer.input_scale = None


def process_weights_after_loading_mxfp8_linear(self, layer) -> None:
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        swizzle_mxfp8_scale,
    )
    from vllm.model_executor.parameter import ModelWeightParameter

    if layer.weight.ndim != 2:
        raise ValueError(
            f"MXFP8 linear layer weight must be 2D, but got {layer.weight.ndim}D"
        )

    backend = getattr(self, "backend", None)
    if backend is not None:
        try:
            from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
                Mxfp8LinearBackend,
            )
        except ImportError:
            if "FLASHINFER_CUTLASS" not in str(backend):
                raise AssertionError(
                    f"Unsupported MXFP8 linear backend for refit: {backend}"
                )
        else:
            assert backend == Mxfp8LinearBackend.FLASHINFER_CUTLASS
    else:
        kernel = getattr(self, "kernel", None)
        kernel_name = type(kernel).__name__ if kernel is not None else None
        if kernel_name == "FlashInferCutedslMxfp8LinearKernel":
            # vLLM 0.25 prefers the CuTe-DSL kernel, but it stores the weight
            # column-major [K, N] while this refit-friendly override (and the
            # MXFP8 refit loader) keeps the canonical [N, K] layout. The
            # CUTLASS kernel consumes [N, K] and is supported wherever
            # CuTe-DSL is (both require SM100), so swap it in.
            from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
                FlashInferCutlassMxfp8LinearKernel,
            )

            kernel = FlashInferCutlassMxfp8LinearKernel(kernel.config)
            self.kernel = kernel
            kernel_name = type(kernel).__name__
            # Record it: this demotes vLLM's first-choice MXFP8 linear kernel
            # on every such layer, so anyone comparing NeMo-RL rollout
            # throughput against a plain vLLM MXFP8 serve has an explanation
            # in the log rather than only in this comment.
            logger.warning_once(
                "NeMo-RL MXFP8 refit requires the [N, K] weight layout; "
                "replacing vLLM's preferred FlashInferCutedslMxfp8LinearKernel "
                "with FlashInferCutlassMxfp8LinearKernel. Expect a rollout "
                "throughput difference vs. plain vLLM serving."
            )
        if kernel_name != "FlashInferCutlassMxfp8LinearKernel":
            raise AssertionError(
                f"Unsupported MXFP8 linear kernel for refit: {kernel_name}"
            )

    weight = layer.weight.data  # [N, K]
    N, K = weight.shape

    if not hasattr(layer, "weight_scale_from_checkpoint"):
        layer.weight_scale_from_checkpoint = ModelWeightParameter(
            data=layer.weight_scale.data,
            input_dim=1,
            output_dim=0,
            weight_loader=layer.weight_scale.weight_loader,
        )
        layer.register_parameter(
            "weight_scale_from_checkpoint", layer.weight_scale_from_checkpoint
        )
        weight_scale = layer.weight_scale.data
        # Swizzle the weight scales
        scale_k = K // 32
        weight_scale_2d = weight_scale[:N, :scale_k].contiguous()
        weight_scale_swizzled = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)
        layer.weight_scale = torch.nn.Parameter(
            weight_scale_swizzled.contiguous(), requires_grad=False
        )
    else:
        weight_scale = layer.weight_scale_from_checkpoint.data
        # Swizzle the weight scales
        scale_k = K // 32
        weight_scale_2d = weight_scale[:N, :scale_k].contiguous()
        weight_scale_swizzled = swizzle_mxfp8_scale(weight_scale_2d, M=N, K=K)
        layer.weight_scale.copy_(weight_scale_swizzled.contiguous())


def create_weights_mxfp8_moe(
    self,
    layer: torch.nn.Module,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype: torch.dtype,
    **extra_weight_attrs,
):
    """Create ModelOpt MXFP8 MoE weights without assigning read-only vLLM attrs.

    vLLM 0.20.0's ModelOpt MXFP8 path writes hidden/intermediate sizes onto
    FusedMoE, but those are read-only properties backed by moe_config. Keep the
    upstream allocation behavior while relying on those existing properties.
    """
    # vLLM 0.25 moved FusedMoeWeightScaleSupported out of fused_moe.layer;
    # it is re-exported from the fused_moe package.
    from vllm.model_executor.layers.fused_moe import (
        FusedMoeWeightScaleSupported,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        MXFP8_BLOCK_SIZE,
        MXFP8_SCALE_DTYPE,
        MXFP8_VALUE_DTYPE,
    )
    from vllm.model_executor.parameter import ModelWeightParameter
    from vllm.model_executor.utils import set_weight_attrs

    layer.orig_dtype = params_dtype
    if hidden_size % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP8 MoE requires hidden_size divisible by {MXFP8_BLOCK_SIZE}, "
            f"got {hidden_size}."
        )
    if intermediate_size_per_partition % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            "MXFP8 MoE requires intermediate_size_per_partition divisible by "
            f"{MXFP8_BLOCK_SIZE}, got {intermediate_size_per_partition}."
        )

    layer.num_experts = num_experts
    weight_loader = extra_weight_attrs.get("weight_loader")
    w13_num_shards = 2 if self.moe.is_act_and_mul else 1

    w13_weight = ModelWeightParameter(
        data=torch.empty(
            num_experts,
            w13_num_shards * intermediate_size_per_partition,
            hidden_size,
            dtype=MXFP8_VALUE_DTYPE,
        ),
        input_dim=2,
        output_dim=1,
        weight_loader=weight_loader,
    )
    layer.register_parameter("w13_weight", w13_weight)

    w2_weight = ModelWeightParameter(
        data=torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=MXFP8_VALUE_DTYPE,
        ),
        input_dim=2,
        output_dim=1,
        weight_loader=weight_loader,
    )
    layer.register_parameter("w2_weight", w2_weight)

    w13_weight_scale = ModelWeightParameter(
        data=torch.empty(
            num_experts,
            w13_num_shards * intermediate_size_per_partition,
            hidden_size // MXFP8_BLOCK_SIZE,
            dtype=MXFP8_SCALE_DTYPE,
        ),
        input_dim=2,
        output_dim=1,
        weight_loader=weight_loader,
    )
    layer.register_parameter("w13_weight_scale", w13_weight_scale)

    w2_weight_scale = ModelWeightParameter(
        data=torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition // MXFP8_BLOCK_SIZE,
            dtype=MXFP8_SCALE_DTYPE,
        ),
        input_dim=2,
        output_dim=1,
        weight_loader=weight_loader,
    )
    layer.register_parameter("w2_weight_scale", w2_weight_scale)

    set_weight_attrs(
        layer.w13_weight_scale,
        {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value},
    )
    set_weight_attrs(
        layer.w2_weight_scale,
        {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value},
    )


def process_weights_after_loading_moe(self, layer) -> None:
    """This function is used to process the weights after loading for a FusedMoE layer.

    Compared to the original process_weights_after_loading in vllm, we use .copy_() instead of
    replace_parameter() to avoid creating new torch.nn.Parameter objects, because that removes
    the weight_loader attribute which we need for refit.

    Updated for vLLM 0.25 which passes a RoutedExperts module as `layer` and
    sets up the MoE kernel via make_fp8_moe_kernel(routing_tables=..., layer=...).
    """
    from vllm.model_executor.layers.quantization.fp8 import (
        convert_to_fp8_moe_kernel_format,
    )

    w13 = layer.w13_weight.data
    w2 = layer.w2_weight.data
    w13_scale = getattr(layer, f"w13_{self.weight_scale_name}").data
    w2_scale = getattr(layer, f"w2_{self.weight_scale_name}").data
    w13_input_scale = layer.w13_input_scale
    w2_input_scale = layer.w2_input_scale

    # Use vLLM's backend-specific weight conversion (handles deepgemm,
    # flashinfer, triton, etc. based on self.fp8_backend).
    w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
        fp8_backend=self.fp8_backend,
        layer=layer,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_input_scale=w13_input_scale,
        w2_input_scale=w2_input_scale,
    )

    _assign_param_data(layer.w13_weight, w13)
    _assign_param_data(layer.w2_weight, w2)
    _assign_param_data(getattr(layer, f"w13_{self.weight_scale_name}"), w13_scale)
    _assign_param_data(getattr(layer, f"w2_{self.weight_scale_name}"), w2_scale)

    # Set up the MoE kernel on initial load only (same as upstream _setup_kernel
    # but without replace_parameter). Gate on is None, not hasattr, because
    # FusedMoEMethodBase.__init__ always sets moe_kernel=None. Also skips refit
    # calls (finalize_layerwise_reload) which lack set_current_vllm_config context.
    self.moe_quant_config = self.get_fused_moe_quant_config(layer)
    if self.moe_quant_config and self.moe_kernel is None:
        from vllm.model_executor.layers.quantization.fp8 import make_fp8_moe_kernel

        assert self.experts_cls is not None
        self.moe_kernel = make_fp8_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            fp8_backend=self.fp8_backend,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
            layer=layer,
        )


def _mxfp8_moe_row_permutations(
    layer,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    is_gated: bool,
    epilogue_tile_m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row permutations equivalent to the FlashInfer shuffle calls."""
    perm_w13 = getattr(layer, "_mxfp8_shuffle_perm_w13", None)
    perm_w2 = getattr(layer, "_mxfp8_shuffle_perm_w2", None)
    if perm_w13 is None or perm_w2 is None:
        from flashinfer.fused_moe.core import (
            get_reorder_rows_for_gated_act_gemm_row_indices,
        )
        from flashinfer.utils import get_shuffle_matrix_a_row_indices

        perm_w13 = get_shuffle_matrix_a_row_indices(w13_weight[0], epilogue_tile_m)
        if is_gated:
            reorder = get_reorder_rows_for_gated_act_gemm_row_indices(w13_weight[0])
            perm_w13 = reorder[perm_w13]
        perm_w2 = get_shuffle_matrix_a_row_indices(w2_weight[0], epilogue_tile_m)
        layer._mxfp8_shuffle_perm_w13 = perm_w13
        layer._mxfp8_shuffle_perm_w2 = perm_w2
    device = w13_weight.device
    return perm_w13.to(device), perm_w2.to(device)


def _shuffle_mxfp8_moe_batched(
    layer,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    is_gated: bool,
    epilogue_tile_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shuffle stacked expert values and scales with four batched gathers."""
    from flashinfer import block_scale_interleave
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        MXFP8_SCALE_DTYPE,
        MXFP8_VALUE_DTYPE,
    )

    perm_w13, perm_w2 = _mxfp8_moe_row_permutations(
        layer, w13_weight, w2_weight, is_gated, epilogue_tile_m
    )
    num_experts = w13_weight.shape[0]
    w13_u8 = w13_weight.view(torch.uint8)
    w2_u8 = w2_weight.view(torch.uint8)
    w13_shuffled = torch.index_select(w13_u8, 1, perm_w13)
    w2_shuffled = torch.index_select(w2_u8, 1, perm_w2)

    w13_scale_u8 = pad_flashinfer_scale_k(w13_scale.view(torch.uint8))
    w2_scale_u8 = pad_flashinfer_scale_k(w2_scale.view(torch.uint8))
    assert w13_scale_u8.shape[1] % 128 == 0
    assert w2_scale_u8.shape[1] % 128 == 0
    w13_scale_gathered = torch.index_select(w13_scale_u8, 1, perm_w13)
    w2_scale_gathered = torch.index_select(w2_scale_u8, 1, perm_w2)
    w13_scale_shuffled = (
        block_scale_interleave(w13_scale_gathered)
        .view(MXFP8_SCALE_DTYPE)
        .view(num_experts, -1)
    )
    w2_scale_shuffled = (
        block_scale_interleave(w2_scale_gathered)
        .view(MXFP8_SCALE_DTYPE)
        .view(num_experts, -1)
    )
    return (
        w13_shuffled.view(MXFP8_VALUE_DTYPE),
        w2_shuffled.view(MXFP8_VALUE_DTYPE),
        w13_scale_shuffled,
        w2_scale_shuffled,
    )


def _shuffle_mxfp8_moe_per_expert(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    is_gated: bool,
    epilogue_tile_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the original per-expert FlashInfer shuffle as a reference path."""
    from flashinfer import (
        reorder_rows_for_gated_act_gemm,
        shuffle_matrix_a,
        shuffle_matrix_sf_a,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        MXFP8_SCALE_DTYPE,
        MXFP8_VALUE_DTYPE,
    )

    num_experts = w13_weight.shape[0]
    w13_rows = w13_weight.shape[1]
    w2_rows = w2_weight.shape[1]
    w13_weight_shuffled = []
    w2_weight_shuffled = []
    w13_scale_shuffled = []
    w2_scale_shuffled = []
    for i in range(num_experts):
        w13_i = w13_weight[i].reshape(w13_rows, -1)
        w13_sf_i = w13_scale[i].reshape(w13_rows, -1)
        if is_gated:
            # Reorder rows for gated activation layout expected by TRTLLM.
            w13_i = reorder_rows_for_gated_act_gemm(w13_i.clone())
            w13_sf_i = reorder_rows_for_gated_act_gemm(w13_sf_i.clone())

        w13_shuffled_i = shuffle_matrix_a(w13_i.view(torch.uint8), epilogue_tile_m)
        w2_shuffled_i = shuffle_matrix_a(
            w2_weight[i].view(torch.uint8), epilogue_tile_m
        )
        w13_weight_shuffled.append(w13_shuffled_i.contiguous().view(MXFP8_VALUE_DTYPE))
        w2_weight_shuffled.append(w2_shuffled_i.contiguous().view(MXFP8_VALUE_DTYPE))
        w13_sf_shuffled_i = shuffle_matrix_sf_a(
            pad_flashinfer_scale_k(w13_sf_i.view(torch.uint8).reshape(w13_rows, -1)),
            epilogue_tile_m,
        )
        w2_sf_shuffled_i = shuffle_matrix_sf_a(
            pad_flashinfer_scale_k(w2_scale[i].view(torch.uint8).reshape(w2_rows, -1)),
            epilogue_tile_m,
        )
        w13_scale_shuffled.append(
            w13_sf_shuffled_i.contiguous().view(MXFP8_SCALE_DTYPE)
        )
        w2_scale_shuffled.append(w2_sf_shuffled_i.contiguous().view(MXFP8_SCALE_DTYPE))

    return (
        torch.stack(w13_weight_shuffled).contiguous(),
        torch.stack(w2_weight_shuffled).contiguous(),
        torch.stack(w13_scale_shuffled).contiguous(),
        torch.stack(w2_scale_shuffled).contiguous(),
    )


def process_weights_after_loading_mxfp8_moe(self, layer) -> None:
    """Shuffle weights and scales into FlashInfer TRTLLM MXFP8 layout."""
    from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        swap_w13_to_w31,
    )
    from vllm.model_executor.parameter import ModelWeightParameter
    from vllm.model_executor.utils import set_weight_attrs

    if self.mxfp8_backend != Fp8MoeBackend.FLASHINFER_TRTLLM:
        raise NotImplementedError(
            "MXFP8 MoE refit layout conversion only supports FLASHINFER_TRTLLM; "
            f"got {self.mxfp8_backend}."
        )

    epilogue_tile_m = 128
    is_gated = self.moe.is_act_and_mul
    w13_weight = layer.w13_weight.data
    if not hasattr(layer, "w13_weight_scale_from_checkpoint"):
        w13_scale = layer.w13_weight_scale.data
    else:
        w13_scale = layer.w13_weight_scale_from_checkpoint.data
    if is_gated:
        # FI TRTLLM gated kernels use W31 ordering. Model checkpoints store
        # gated projection as W13, so convert once before shuffling.
        w13_weight = swap_w13_to_w31(w13_weight)
        w13_scale = swap_w13_to_w31(w13_scale)
    w2_weight = layer.w2_weight.data
    if not hasattr(layer, "w2_weight_scale_from_checkpoint"):
        w2_scale = layer.w2_weight_scale.data
    else:
        w2_scale = layer.w2_weight_scale_from_checkpoint.data

    shuffled = _shuffle_mxfp8_moe_batched(
        layer,
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        is_gated,
        epilogue_tile_m,
    )
    (
        w13_weight_shuffled,
        w2_weight_shuffled,
        w13_scale_shuffled,
        w2_scale_shuffled,
    ) = shuffled

    if not hasattr(layer, "w13_weight_scale_from_checkpoint"):
        layer.w13_weight_scale_from_checkpoint = ModelWeightParameter(
            data=layer.w13_weight_scale.data,
            input_dim=2,
            output_dim=1,
            weight_loader=layer.w13_weight_scale.weight_loader,
        )
        layer.w2_weight_scale_from_checkpoint = ModelWeightParameter(
            data=layer.w2_weight_scale.data,
            input_dim=2,
            output_dim=1,
            weight_loader=layer.w2_weight_scale.weight_loader,
        )
        layer.register_parameter(
            "w13_weight_scale_from_checkpoint", layer.w13_weight_scale_from_checkpoint
        )
        layer.register_parameter(
            "w2_weight_scale_from_checkpoint", layer.w2_weight_scale_from_checkpoint
        )
        print(
            f"layer.w13_weight_scale_from_checkpoint shape: {layer.w13_weight_scale_from_checkpoint.data.shape}"
        )
        print(
            f"layer.w2_weight_scale_from_checkpoint shape: {layer.w2_weight_scale_from_checkpoint.data.shape}"
        )
        set_weight_attrs(
            layer.w13_weight_scale_from_checkpoint,
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value},
        )
        set_weight_attrs(
            layer.w2_weight_scale_from_checkpoint,
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value},
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            w13_scale_shuffled, requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            w2_scale_shuffled, requires_grad=False
        )
    else:
        layer.w13_weight_scale.copy_(w13_scale_shuffled)
        layer.w2_weight_scale.copy_(w2_scale_shuffled)
    layer.w13_weight.copy_(w13_weight_shuffled)
    layer.w2_weight.copy_(w2_weight_shuffled)

    if self.moe_kernel is None:
        from vllm.model_executor.layers.quantization.fp8 import make_fp8_moe_kernel

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None
        assert self.experts_cls is not None
        self.moe_kernel = make_fp8_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            fp8_backend=self.mxfp8_backend,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
            layer=layer,
        )


def process_weights_after_loading_kv(self, layer) -> None:
    """Modified version of BaseKVCacheMethod.process_weights_after_loading.

    Doesn't delete k_scale, v_scale, q_scale, and prob_scale parameters to allow
    for dynamic updates during refit.
    """
    # If the kv-cache dtype is auto, we enforce the k/v_scale to be 1.0
    # regardless whether the kv-scale is available in the checkpoint.
    # No need to process kv scales after loading if we are going to
    # calculate them on the fly.
    from vllm.platforms import current_platform

    if layer.kv_cache_dtype != "auto" and not layer.calculate_kv_scales:
        if layer.k_scale > 0.0 and layer.v_scale > 0.0:
            # We prefer to use separate k_scale and v_scale if present
            k_scale = layer.k_scale.to("cpu").tolist()
            v_scale = layer.v_scale.to("cpu").tolist()
            if current_platform.is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2
        elif layer.k_scale < 0.0 and layer.v_scale < 0.0:
            # If no scales were loaded (both scales are invalid negative
            # values), use the default value of 1.0
            k_scale = 1.0
            v_scale = 1.0
        else:
            # If we find a single kv_scale in the checkpoint, we remap
            # kv_scale to k_scale during weight loading, and duplicate
            # k_scale to v_scale here
            assert layer.k_scale > 0.0
            scale_to_duplicate = max(layer.k_scale, layer.v_scale)
            k_scale = scale_to_duplicate.to("cpu").tolist()
            v_scale = scale_to_duplicate.to("cpu").tolist()
            if current_platform.is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2

        if not isinstance(k_scale, float) or not isinstance(v_scale, float):
            raise ValueError("Only support per-tensor scaling factor for fp8 KV cache")

        if layer.q_scale < 0.0:
            layer._q_scale.copy_(k_scale)
            layer._q_scale_float = k_scale

        # These are used in the final Attention.forward()
        layer._k_scale.copy_(k_scale)
        layer._v_scale.copy_(v_scale)
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

    is_singleton_float = lambda x: (
        isinstance(x, float)
        or isinstance(x, torch.Tensor)
        and x.numel() == 1
        and x.is_floating_point()
    )
    if not is_singleton_float(q_scale) or not is_singleton_float(prob_scale):
        raise ValueError(
            "Only support per-tensor scaling factorfor fp8-quantized Q/prob"
        )

    # These are used in the final Attention.forward()
    layer._q_scale.copy_(q_scale)
    layer._q_scale_float = (
        q_scale.item() if isinstance(q_scale, torch.Tensor) else q_scale
    )

    layer._prob_scale.copy_(prob_scale)

    # IMPORTANT: We DON'T delete the parameters here to allow for dynamic updates
    # Original code deleted: layer.k_scale, layer.v_scale, layer.q_scale, layer.prob_scale


@triton.jit
def _per_token_group_quant_fp8(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr += (row * y_row_stride) + (row_g_id * group_size)
    y_q_ptr += g_id * group_size
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)

    # pow2_scale
    inv_scale = fp8_max / _absmax
    exponent = tl.floor(tl.log2(inv_scale))
    # exponent is an integer
    exponent = tl.minimum(exponent, 126.0)

    # after rounding to exponent, round back to floating
    inv_scale_pow2 = tl.exp2(exponent)

    is_nan = inv_scale_pow2 != inv_scale_pow2
    is_inf = (inv_scale_pow2 == 1.0 / 0.0) | (inv_scale_pow2 == -1.0 / 0.0)

    # If the value is NaN or infinity, default it to 1.0,
    # otherwise keep its original value.
    inv_scale_pow2 = tl.where(is_nan | is_inf, 1.0, inv_scale_pow2)
    # finally uninverse
    y_s = 1.0 / inv_scale_pow2

    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@triton.jit
def _per_token_group_quant_fp8_colmajor(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    y_row_stride,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    fp8_min,
    fp8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
):
    groups_per_row = y_num_columns // group_size

    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    row = g_id // groups_per_row
    row_g_id = g_id % groups_per_row

    y_ptr += (row * y_row_stride) + (row_g_id * group_size)
    y_q_ptr += g_id * group_size

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    y_s_ptr += scale_col * y_s_col_stride + scale_row

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)

    # Quant pow2_scale:
    inv_scale = fp8_max / _absmax
    # calculate the nearest pow2 integer
    exponent = tl.floor(tl.log2(inv_scale))
    exponent = tl.minimum(exponent, 126.0)
    # round inv_scale to the nearest pow2 with the exp we just calculated
    inv_scale_pow2 = tl.exp2(exponent)
    # If the value is NaN or infinity, default it to 1.0,
    # otherwise keep its original value.
    is_nan = inv_scale_pow2 != inv_scale_pow2
    is_inf = (inv_scale_pow2 == float("inf")) | (inv_scale_pow2 == float("-inf"))
    inv_scale_pow2 = tl.where(is_nan | is_inf, 1.0, inv_scale_pow2)
    # finally uninverse
    y_s = 1.0 / inv_scale_pow2

    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8(
    *args,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert global_fp8_config.use_activation_pow2_scale
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8 as vllm_per_token_group_quant_fp8,
    )

    return vllm_per_token_group_quant_fp8(*args, **kwargs)
