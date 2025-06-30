from typing import Any, Callable, Optional, Union

import torch, os
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from transformers import PretrainedConfig

USE_WEIGHT_POW2_SCALE = False
USE_ACTIVATION_POW2_SCALE = False
FIRST_LAST_LAYERS_IN_BF16 = False
NUM_LAYERS = None

def get_first_last_layer_param_names(model_name):
    def get_layer_params(layer_num):
        return [
            f'model.layers.{layer_num}.self_attn.q_proj',
            f'model.layers.{layer_num}.self_attn.k_proj',
            f'model.layers.{layer_num}.self_attn.v_proj',
            f'model.layers.{layer_num}.self_attn.o_proj',
            f'model.layers.{layer_num}.mlp.up_proj',
            f'model.layers.{layer_num}.mlp.down_proj',
            f'model.layers.{layer_num}.mlp.gate_proj',
        ]

    config_dict, _ = PretrainedConfig.get_config_dict(
        model_name, token=os.getenv('HF_TOKEN', None))
    
    global NUM_LAYERS
    NUM_LAYERS = config_dict['num_hidden_layers']
    layers = [0, NUM_LAYERS-1]

    return [param for l in layers for param in get_layer_params(l)]


def kitchen_block_scale(
    data_hp,
    weight_block_size,
):
    assert len(data_hp.shape) == 2, "Only 2d input tensor is supported"

    block_size1 = weight_block_size[1]
    block_size0 = weight_block_size[0]
    assert (
        data_hp.shape[1] % block_size1 == 0
    ), f"data_hp.shape[1] {data_hp.shape[1]}  must be a multiple of block_size1: {block_size1}."
    assert (
        data_hp.shape[0] % block_size0 == 0
    ), f"data_hp.shape[0] {data_hp.shape[0]} must be a multiple of block_size0: {block_size0}."

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

    if USE_WEIGHT_POW2_SCALE:
        # Calculate exponent HW instruction: cvt.rp.satfinite.ue8m0x2.f32
        exponent = torch.ceil(torch.log2(descale))
        # Post process exponent to be in range of -127 to 127 and to be E8M0 biased
        exponent = torch.clamp(exponent, min=-127, max=127) + 127
        # Convert to uint8 container
        exponent = exponent.to(torch.uint8)
        # Calculate descale_fp to apply to data_hp
        descale_fp = torch.where(
            # If exponent is 0, descale_fp is 1.0 rather than 2^127
            exponent == 0,
            1.0,
            torch.exp2(127 - exponent.to(torch.float32)),
        )
        descale_fp_inv = torch.reciprocal(descale_fp)
    else:
        descale_fp = max_dtype / max_abs
        descale_fp = torch.where(
            max_abs == 0, 1.0, descale_fp
        )  # preserve the behavior for 0 amax case
        descale_fp = torch.where(
            descale_fp == torch.inf, torch.finfo(data_hp.dtype).max, descale_fp
        )
        descale_fp_inv = torch.reciprocal(descale_fp)

    # Scale and saturate cast the data elements to max of target dtype
    data_lp = torch.clamp(data_hp * descale_fp, min=-1 * max_dtype, max=max_dtype)

    fp_data = data_lp.to(torch.float8_e4m3fn)

    # (BLK_M, BLK_N, BLOCK_SIZE_M * BLOCK_SIZE_N) to (M, N)
    fp_data = (
        fp_data.reshape(blk_m, blk_n, block_size0, block_size1)
        .permute(0, 2, 1, 3)
        .reshape(original_shape)
    )

    # Convert to target format, but still in original precision container
    return fp_data, descale_fp_inv 


def is_fp8_model(vllm_config):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if hasattr(vllm_config, "quant_config") and \
        isinstance(vllm_config.quant_config, Fp8Config):
        assert vllm_config.quant_config.weight_block_size is not None, \
            "Only block scaling is currently supported in NeMo-RL!"
        return True

    return False

def is_fp8_weight(name):
    fp8_params = [
        "q_proj.weight", 
        "k_proj.weight", 
        "v_proj.weight", 
        "o_proj.weight", 
        "down_proj.weight", 
        "up_proj.weight", 
        "gate_proj.weight",
    ]
    if FIRST_LAST_LAYERS_IN_BF16:
        if 'layers.0' in name or f'layers.{str(NUM_LAYERS-1)}' in name:
            return False
    return any([param in name for param in fp8_params])


def cast_weights_to_fp8(weights, quant_config):
    weights_quantized = []
    for k, v in weights:
        if not is_fp8_weight(k):
            weights_quantized.append((k,v))
            continue
        
        param_lp, param_scale = kitchen_block_scale(
            v.to(torch.float), 
            weight_block_size=quant_config.weight_block_size
        )
        param_scale = torch.squeeze(param_scale)
        weights_quantized.append([k, param_lp])
        weights_quantized.append([k + "_scale_inv", param_scale])

    return weights_quantized

def process_weights_after_loading(self, layer) -> None:
    from vllm.model_executor.parameter import (BlockQuantScaleParameter,
                                    ModelWeightParameter,
                                    PerTensorScaleParameter)
    from torch.nn import Parameter
    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"

    def _create_param_from_subclass_attributes(custom_param):
        param = Parameter(custom_param.data, requires_grad=False)
        base_param_dir = dir(torch.nn.Parameter)
        custom_param_dir = dir(custom_param)
        # Find the attributes that are unique to the custom parameter
        custom_attributes = [
            attr for attr in custom_param_dir 
            if attr not in base_param_dir and not attr.startswith('__')
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
        ModelWeightParameter(data=weight, output_dim=0, input_dim=1,
            weight_loader=layer.weight.weight_loader)
    )
    layer.weight_scale_inv = _create_param_from_subclass_attributes(
        BlockQuantScaleParameter(
            data=weight_scale_inv, output_dim=0, input_dim=1,
            weight_loader=layer.weight_scale_inv.weight_loader)
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
    pow2_scale: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
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

    if pow2_scale:
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

    else:
        # Original scaling logic
        y_s = _absmax / fp8_max

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
    pow2_scale: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
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

    # Quant
    if pow2_scale:
        inv_scale = fp8_max / _absmax
        # calculate the nearest pow2 integer
        exponent = tl.floor(tl.log2(inv_scale))
        exponent = tl.minimum(exponent, 126.0)
        # round inv_scale to the nearest pow2 with the exp we just calculated
        inv_scale_pow2 = tl.exp2(exponent)
        # If the value is NaN or infinity, default it to 1.0,
        # otherwise keep its original value.
        is_nan = inv_scale_pow2 != inv_scale_pow2
        is_inf = (inv_scale_pow2 == float('inf')) | (inv_scale_pow2 == float('-inf'))
        inv_scale_pow2 = tl.where(is_nan | is_inf, 1.0, inv_scale_pow2) 
        # finally uninverse        
        y_s = 1.0 / inv_scale_pow2
    else:
        y_s = _absmax / fp8_max

    y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    pow2_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function to perform per-token-group quantization on an input tensor `x`.
    It converts the tensor values into signed float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dype of output tensor. Note that only `torch.float8_e4m3fn`
        is supported for now.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
        scaling factor for quantization.
    """
    dtype = current_platform.fp8_dtype() if dtype is None else dtype
    assert (x.shape[-1] % group_size == 0), (
        f"the last dimension of `x` {x.shape[-1]} must be divisible "
        f"by `group_size` {group_size}")
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    M = x.numel() // group_size
    N = group_size
    if column_major_scales:
        shape = (x.shape[-1] // group_size, ) + x.shape[:-1]
        x_s = torch.empty(shape, device=x.device,
                          dtype=torch.float32).permute(-1, -2)
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size, )
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1

    if column_major_scales:
        _per_token_group_quant_fp8_colmajor[(M, )](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            x_s.stride(1),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            pow2_scale=USE_ACTIVATION_POW2_SCALE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _per_token_group_quant_fp8[(M, )](
            x,
            x_q,
            x_s,
            group_size,
            x.shape[1],
            x.stride(0),
            eps,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            BLOCK=BLOCK,
            pow2_scale=USE_ACTIVATION_POW2_SCALE,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return x_q, x_s
