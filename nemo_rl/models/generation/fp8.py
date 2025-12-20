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
from dataclasses import dataclass, field
from unittest.mock import patch

import ray
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import LinearBase
from vllm.triton_utils import tl, triton
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.utils import CoreEngineProcManager

FP8_BLOCK_QUANT_KWARGS = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}


@dataclass(frozen=True)
class FP8Config:
    use_weight_pow2_scale: bool = False
    use_activation_pow2_scale: bool = False
    num_first_layers_in_bf16: int = 0
    num_last_layers_in_bf16: int = 0
    model_parallel_size: int = None
    kv_cache_dtype: str = "auto"
    use_fp8_weights: bool = True  # Whether model weights are quantized to FP8


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
        # we patch vllm's collective_rpc so that before vllm initalizes the model on each rank, we execute
        # a ray remote that patches each worker with the required fp8 vllm patches
        from vllm.v1.executor.ray_distributed_executor import RayDistributedExecutor

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


def get_vllm_qkv_scale_names(layer_idx: int) -> dict[str, str]:
    """Get vLLM-compatible parameter names for Q/K/V FP8 scales.

    This function centralizes the naming convention for Q/K/V scale parameters
    that vLLM expects. These names must match vLLM's internal parameter structure.

    Args:
        layer_idx: The transformer layer index (0-based)

    Returns:
        Dictionary mapping scale types to vLLM parameter names:
        - 'q_scale': Q activation scale name
        - 'k_scale': K activation scale name
        - 'v_scale': V activation scale name

    Note:
        The q_scale has an extra '.attn.' component compared to k_scale/v_scale.
        This matches vLLM's parameter remapping logic in:
        vllm.model_executor.model_loader.weight_utils.maybe_remap_kv_scale_name

    Example:
        >>> get_vllm_qkv_scale_names(0)
        {
            'q_scale': 'model.layers.0.self_attn.attn.q_scale',
            'k_scale': 'model.layers.0.self_attn.k_scale',
            'v_scale': 'model.layers.0.self_attn.v_scale'
        }
    """
    return {
        "q_scale": f"model.layers.{layer_idx}.self_attn.attn.q_scale",
        "k_scale": f"model.layers.{layer_idx}.self_attn.k_scale",
        "v_scale": f"model.layers.{layer_idx}.self_attn.v_scale",
    }


def convert_calibration_to_vllm_format(
    calibration_results: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Convert NeMo-RL calibration results to vLLM parameter format.

    Currently only used by megatron policy worker.
    After FP8 KV cache is supported by DTensor path, this function can be reused.

    This function transforms the calibration output format (with layer_N keys)
    into the flat dictionary format that vLLM expects for parameter loading.

    Args:
        calibration_results: Dict with keys like "layer_0", "layer_1", etc.
            Each value is a dict with keys: "q_scale", "k_scale", "v_scale"
            and corresponding float scale values.

    Returns:
        Flat dictionary mapping vLLM parameter names to scale values.
        Keys follow vLLM's naming convention as defined in get_vllm_qkv_scale_names.

    Example:
        >>> calib = {
        ...     "layer_0": {"q_scale": 1.0, "k_scale": 2.0, "v_scale": 3.0},
        ...     "layer_1": {"q_scale": 1.5, "k_scale": 2.5, "v_scale": 3.5}
        ... }
        >>> convert_calibration_to_vllm_format(calib)
        {
            'model.layers.0.self_attn.attn.q_scale': 1.0,
            'model.layers.0.self_attn.k_scale': 2.0,
            'model.layers.0.self_attn.v_scale': 3.0,
            'model.layers.1.self_attn.attn.q_scale': 1.5,
            'model.layers.1.self_attn.k_scale': 2.5,
            'model.layers.1.self_attn.v_scale': 3.5
        }
    """
    vllm_scales = {}
    for layer_key, scales in calibration_results.items():
        # Extract layer index from "layer_N" format
        layer_idx = int(layer_key.split("_")[1])
        param_names = get_vllm_qkv_scale_names(layer_idx)

        vllm_scales[param_names["q_scale"]] = scales["q_scale"]
        vllm_scales[param_names["k_scale"]] = scales["k_scale"]
        vllm_scales[param_names["v_scale"]] = scales["v_scale"]

    return vllm_scales


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

        # These patches add support for pow2, e8 dynamic activation scalings factors which are believed to have higher
        # SNR compared to plain fp32 scaling factors. This feature is still under active research.
        if global_fp8_config.use_activation_pow2_scale:
            func2_path = "vllm.model_executor.layers.quantization.utils.fp8_utils.per_token_group_quant_fp8"
            func3_path = "vllm.model_executor.layers.quantization.utils.fp8_utils._per_token_group_quant_fp8"
            func4_path = "vllm.model_executor.layers.quantization.utils.fp8_utils._per_token_group_quant_fp8_colmajor"
            patcher2 = patch(func2_path, per_token_group_quant_fp8)
            patcher3 = patch(func3_path, _per_token_group_quant_fp8)
            patcher4 = patch(func4_path, _per_token_group_quant_fp8_colmajor)
            fp8_state.vllm_patches.append(patcher2, patcher3, patcher4)

        # Static scales mode: patch process_weights_after_loading to preserve k_scale/v_scale for manual updates
        func5_path = "vllm.model_executor.layers.quantization.kv_cache.BaseKVCacheMethod.process_weights_after_loading"
        patcher5 = patch(func5_path, process_weights_after_loading_kv)
        fp8_state.vllm_patches.append(patcher5)

    for p in fp8_state.vllm_patches:
        p.start()

    fp8_patches_applied = True


def init_fp8(vllm_cfg, model_name, model_parallel_size):
    config = AutoConfig.from_pretrained(model_name)
    global global_fp8_config
    # Determine if we're using FP8 weights based on precision setting
    use_fp8_weights = vllm_cfg.get("precision") == "fp8"
    kv_cache_dtype = vllm_cfg["kv_cache_dtype"]

    # Validate configuration: kv_cache_dtype
    if kv_cache_dtype not in ["auto", "fp8", "fp8_e4m3"]:
        raise ValueError(
            f"kv_cache_dtype must be one of ['auto', 'fp8', 'fp8_e4m3'], but got {kv_cache_dtype}"
        )

    # Validate configuration: kv_cache_dtype=fp8 requires precision=fp8
    if kv_cache_dtype.startswith("fp8") and not use_fp8_weights:
        raise ValueError(
            f"kv_cache_dtype='{kv_cache_dtype}' requires precision='fp8'. "
            "FP8 KV cache can only be used together with FP8 model weights."
        )

    global_fp8_config = FP8Config(
        use_weight_pow2_scale=vllm_cfg.get("pow2_weight_scaling_factors", False),
        use_activation_pow2_scale=vllm_cfg.get(
            "pow2_activation_scaling_factors", False
        ),
        num_first_layers_in_bf16=vllm_cfg.get("num_first_layers_in_bf16", 0),
        num_last_layers_in_bf16=vllm_cfg.get("num_last_layers_in_bf16", 0),
        model_parallel_size=model_parallel_size,
        kv_cache_dtype=kv_cache_dtype,
        use_fp8_weights=use_fp8_weights,
    )

    if vllm_cfg.get("use_deep_gemm", False):
        os.environ["VLLM_USE_DEEP_GEMM"] = "1"
        os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "0"

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
    fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)

    if num_first_layers_in_bf16 > 0 or num_last_layers_in_bf16 > 0:
        with init_empty_weights():
            model = AutoModel.from_config(config)
        param_names = [name for name, _ in model.named_parameters()]

        bf16_params = []
        if num_first_layers_in_bf16 > 0:
            layers = [l for l in range(num_first_layers_in_bf16)]
            bf16_params.append(_get_params_in_layers(param_names, layers))

        if num_last_layers_in_bf16 > 0:
            layers = [
                l
                for l in range(
                    config.num_hidden_layers - num_last_layers_in_bf16,
                    config.num_hidden_layers,
                )
            ]
            bf16_params.append(_get_params_in_layers(param_names, layers))

        fp8_block_quant_kwargs["ignored_layers"] = bf16_params
    quantization_ignored_layer_kws = vllm_cfg.get("quantization_ignored_layer_kws", [])
    if len(quantization_ignored_layer_kws):
        with init_empty_weights():
            model = AutoModel.from_config(config)
        param_names = [
            f"model.{name}".removesuffix(".weight")
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

    # Return FP8 kwargs (precision=fp8 is required at this point)
    vllm_kwargs = {
        "quantization": "fp8",
        "kv_cache_dtype": kv_cache_dtype,
        "hf_overrides": {"quantization_config": fp8_block_quant_kwargs},
    }

    return vllm_kwargs


def is_fp8_model(vllm_config):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if hasattr(vllm_config, "quant_config") and isinstance(
        vllm_config.quant_config, Fp8Config
    ):
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
            params.append(f"model.{name}".removesuffix(".weight"))
    return params


def _get_module_from_param_name(model, name: str):
    # Split the name into parts (e.g., 'layers', '0', 'self_attn', 'q_proj', 'weight')
    # The module path is all but the last part (the parameter's own name)
    path_parts = name.split(".")
    module_path = path_parts[:-1]
    # Replace with the fused model name
    packed_modules_mapping = model.packed_modules_mapping
    reversed_mapping = {
        original_name: fused_name
        for fused_name, original_names_list in packed_modules_mapping.items()
        for original_name in original_names_list
    }
    if module_path[-1] in reversed_mapping.keys():
        module_path[-1] = reversed_mapping[module_path[-1]]

    current_module = model
    try:
        # Traverse the model hierarchy
        for part in module_path:
            if isinstance(current_module, FusedMoE):
                return current_module
            if isinstance(current_module, torch.nn.ModuleList):
                current_module = current_module[int(part)]
            else:
                current_module = getattr(current_module, part)
    except (AttributeError, IndexError, ValueError) as e:
        print(f"Warning: Could not find module for parameter '{name}'. Error: {e}")
    return current_module


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
                    isinstance(module, FusedMoE)
                    and module.w13_weight.dtype == torch.float8_e4m3fn
                    and module.w2_weight.dtype == torch.float8_e4m3fn
                )
            ):
                fp8_state.fp8_param_names.add(name)
    return name in fp8_state.fp8_param_names


def load_weights(weights, model_runner):
    weights_quantized = []
    model = model_runner.model

    for k, v in weights:
        if not _is_fp8_weight(k, model):
            weights_quantized.append((k, v))
            continue
        # Cast the weight into fp8 and its scale factor
        param_lp, param_scale = cast_tensor_to_fp8_blockwise(
            v.to(torch.float),
            weight_block_size=FP8_BLOCK_QUANT_KWARGS["weight_block_size"],
        )
        param_scale = torch.squeeze(param_scale, dim=-1)
        weights_quantized.append([k, param_lp])
        weights_quantized.append([k + "_scale_inv", param_scale])
    # Finally load the weights into vllm
    model.load_weights(weights_quantized)


def cast_tensor_to_fp8_blockwise(
    data_hp,
    weight_block_size,
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

    global global_fp8_config
    if global_fp8_config.use_weight_pow2_scale:
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

    # On Blackwell or Hopper, if E8M0 for DeepGemm is used, we need to
    # requantize the weight and input to the specific scale
    # at the same time.
    should_use_deepgemm = should_use_deepgemm_for_fp8_linear(
        layer.orig_dtype, layer.weight
    )
    if should_use_deepgemm:
        dg_weight, dg_weight_scale = deepgemm_post_process_fp8_weight_block(
            wq=layer.weight.data,
            ws=layer.weight_scale.data,
            quant_block_shape=tuple(layer.weight_block_size),
            use_e8m0=is_deep_gemm_e8m0_used(),
        )
        # This is the only part we change from the original function (https://github.com/vllm-project/vllm/blob/275de34170654274616082721348b7edd9741d32/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L1196-L1197)
        # Instead of creating new torch.nn.Parameter, we update the data in place.
        layer.weight.data.copy_(dg_weight)
        layer.weight_scale.data.copy_(dg_weight_scale)


def process_weights_after_loading(self, layer) -> None:
    """This function is used to process the weights after loading for a Linear layer.

    Compared to the original process_weights_after_loading in vllm, we just avoid creation of
    new torch.nn.Parameter objects, because that removes the weight_loader attribute which we need for refit.
    """
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        process_fp8_weight_block_strategy,
    )

    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"

    weight_scale = layer.weight_scale_inv
    weight, weight_scale = process_fp8_weight_block_strategy(layer.weight, weight_scale)
    layer.weight.data = weight.data
    if hasattr(layer, "weight_scale"):
        # Not the first time to call this function, just need to update the data
        layer.weight_scale.copy_(weight_scale.data)
    else:
        # The first time to call this function, create a new parameter and update the tp status
        layer.weight_scale = torch.nn.Parameter(weight_scale.data, requires_grad=False)
        layer.update_param_tp_status()

    maybe_post_process_fp8_weight_block(layer)


def process_weights_after_loading_moe(self, layer) -> None:
    """This function is used to process the weights after loading for a FusedMoE layer.

    Compared to the original process_weights_after_loading in vllm, we just avoid creation of
    new torch.nn.Parameter objects, because that removes the weight_loader attribute which we need for refit.
    """
    # Lazy import to avoid importing triton too early.
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        swap_w13_to_w31,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )
    from vllm.utils.deep_gemm import (
        is_deep_gemm_e8m0_used,
    )

    self.rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()

    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"

    if self.flashinfer_moe_backend is not None:
        w13_weight = swap_w13_to_w31(layer.w13_weight.data)
        w13_weight_scale_inv = swap_w13_to_w31(layer.w13_weight_scale_inv.data)
    else:
        w13_weight = layer.w13_weight.data
        w13_weight_scale_inv = layer.w13_weight_scale_inv.data
    w2_weight = layer.w2_weight.data
    w2_weight_scale_inv = layer.w2_weight_scale_inv.data

    # DeepGemm scales need to be transposed and aligned. We try to do
    # it ahead of time for performance reasons.
    if self.allow_deep_gemm:
        w13_weight, w13_weight_scale_inv = deepgemm_post_process_fp8_weight_block(
            wq=w13_weight,
            ws=w13_weight_scale_inv,
            quant_block_shape=tuple(layer.weight_block_size),
            use_e8m0=is_deep_gemm_e8m0_used(),
        )
        w2_weight, w2_weight_scale_inv = deepgemm_post_process_fp8_weight_block(
            wq=w2_weight,
            ws=w2_weight_scale_inv,
            quant_block_shape=tuple(layer.weight_block_size),
            use_e8m0=is_deep_gemm_e8m0_used(),
        )
    layer.w13_weight.copy_(w13_weight)
    layer.w13_weight_scale_inv.copy_(w13_weight_scale_inv)
    layer.w2_weight.copy_(w2_weight)
    layer.w2_weight_scale_inv.copy_(w2_weight_scale_inv)


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

    is_singleton_float = (
        lambda x: isinstance(x, float)
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
