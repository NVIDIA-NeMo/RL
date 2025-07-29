import torch, os, ray
from accelerate import init_empty_weights
from dataclasses import dataclass, field
from transformers import AutoConfig, AutoModel
from vllm.model_executor.layers.linear import LinearBase
from vllm.triton_utils import tl, triton
from unittest.mock import patch

from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.utils import CoreEngineProcManager

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
        # we patch vllm's _run_workers so that before vllm initalizes the model on each rank, we execute
        # a ray remote that patches each worker with the required fp8 vllm patches
        from vllm.v1.executor.ray_distributed_executor import RayDistributedExecutor

        original_run_workers = RayDistributedExecutor._run_workers

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

        RayDistributedExecutor._run_workers = patched_run_workers
    else:
        # for single gpu there is no ray, so just call the patches
        apply_fp8_patches(None, fp8_config)

        global fp8_patches_applied
        fp8_patches_applied = True


def apply_fp8_patches(self, fp8_config):
    print(f"APPLYING FP8 PATCHES {fp8_config}")
    global global_fp8_config, fp8_patches_applied
    assert not fp8_patches_applied

    global_fp8_config = fp8_config

    # This patch is used to support torch.compile with vllm parameter subclasses, such as
    # PerTensorScaleParameter. Because we need weight loaders to update fp8 weights each
    # refit, we patch fp8 parameters to have a reference to their weight loader. Eventually
    # with pytorch 2.8, parameter subclassing with torch.compile will be natively supported, in
    # which this patch can be removed.
    func1_path = "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading"
    patcher1 = patch(func1_path, process_weights_after_loading)
    fp8_state.vllm_patches.append(patcher1)
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

    for p in fp8_state.vllm_patches:
        p.start()

    fp8_patches_applied = True


def init_fp8(vllm_cfg, model_name, model_parallel_size):
    global global_fp8_config
    global_fp8_config = FP8Config(
        use_weight_pow2_scale=vllm_cfg.get("pow2_weight_scaling_factors", False),
        use_activation_pow2_scale=vllm_cfg.get(
            "pow2_activation_scaling_factors", False
        ),
        num_first_layers_in_bf16=vllm_cfg.get("num_first_layers_in_bf16", 0),
        num_last_layers_in_bf16=vllm_cfg.get("num_last_layers_in_bf16", 0),
        model_parallel_size=model_parallel_size,
    )

    if vllm_cfg.get("use_deep_gemm", False):
        os.environ["VLLM_USE_DEEP_GEMM"] = "1"

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
        try:
            config = AutoConfig.from_pretrained(model_name)
            with init_empty_weights():
                model = AutoModel.from_config(config)
            param_names = [name for name, _ in model.named_parameters()]
        except OSError as e:
            raise ConnectionError(f"Cannot download model for '{model_name}'.") from e

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

    vllm_kwargs = {
        "quantization": "fp8",
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
        param_lp, param_scale = kitchen_block_scale(
            v.to(torch.float),
            weight_block_size=FP8_BLOCK_QUANT_KWARGS["weight_block_size"],
        )
        param_scale = torch.squeeze(param_scale)
        weights_quantized.append([k, param_lp])
        weights_quantized.append([k + "_scale_inv", param_scale])
    # Monkey patch the param class to their subclass, as certain models
    # will check the param type to call the proper weightloader
    for name, param in model.named_parameters():
        if hasattr(param, "subclass_type"):
            param.orig_type = param.__class__
            param.__class__ = param.subclass_type
    # Finally load the weights into vllm
    model.load_weights(weights_quantized)
    # Undo the type change above to the original type
    for name, param in model.named_parameters():
        if hasattr(param, "subclass_type"):
            param.__class__ = param.orig_type


def kitchen_block_scale(
    data_hp,
    weight_block_size,
):
    assert len(data_hp.shape) == 2, "Only 2d input tensor is supported"

    block_size1 = weight_block_size[1]
    block_size0 = weight_block_size[0]
    assert data_hp.shape[1] % block_size1 == 0, (
        f"data_hp.shape[1] {data_hp.shape[1]}  must be a multiple of block_size1: {block_size1}."
    )
    assert data_hp.shape[0] % block_size0 == 0, (
        f"data_hp.shape[0] {data_hp.shape[0]} must be a multiple of block_size0: {block_size0}."
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

    # Convert to target format, but still in original precision container
    return fp_data, descale_fp


def process_weights_after_loading(self, layer) -> None:
    from vllm.model_executor.parameter import (
        BlockQuantScaleParameter,
        ModelWeightParameter,
    )
    from torch.nn import Parameter

    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"

    def _create_param_from_subclass_attributes(custom_param):
        param = Parameter(custom_param.data, requires_grad=False)
        base_param_dir = dir(torch.nn.Parameter)
        custom_param_dir = dir(custom_param)
        # Find the attributes that are unique to the custom parameter
        custom_attributes = [
            attr
            for attr in custom_param_dir
            if attr not in base_param_dir and not attr.startswith("__")
        ]
        # Set the custom attributes into the base parameter object
        for attr in custom_attributes:
            setattr(param, attr, getattr(custom_param, attr))

        param.subclass_type = type(custom_param)
        return param

    weight = layer.weight.data
    weight_scale_inv = layer.weight_scale_inv.data
    weight = self._maybe_pad_weight(weight)

    layer.weight = _create_param_from_subclass_attributes(
        ModelWeightParameter(
            data=weight,
            output_dim=0,
            input_dim=1,
            weight_loader=layer.weight.weight_loader,
        )
    )
    layer.weight_scale_inv = _create_param_from_subclass_attributes(
        BlockQuantScaleParameter(
            data=weight_scale_inv,
            output_dim=0,
            input_dim=1,
            weight_loader=layer.weight_scale_inv.weight_loader,
        )
    )


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
