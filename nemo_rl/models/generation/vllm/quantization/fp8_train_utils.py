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

import torch

FP8_WEIGHT_BLOCK_SIZE = [128, 128]


def should_quantize_to_fp8(name: str, tensor: torch.Tensor) -> bool:
    """Check whether a HuggingFace-named weight should be block-quantized to FP8.

    Matches the same set of parameters that vLLM quantizes (linear-layer
    weights only).  Embeddings, layernorms, biases, and lm_head are excluded.
    """
    if tensor.dim() != 2:
        return False
    if not name.endswith(".weight"):
        return False
    lower = name.lower()
    if any(kw in lower for kw in ("norm", "embed", "lm_head")):
        return False
    return True


def cast_tensor_to_fp8_blockwise(
    data_hp: torch.Tensor,
    weight_block_size: list[int],
    use_pow2_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise FP8 (E4M3) quantization — standalone, no vLLM dependencies.

    Args:
        data_hp: 2-D high-precision weight tensor (any float dtype).
        weight_block_size: [block_rows, block_cols], e.g. [128, 128].
        use_pow2_scale: If True, round scale factors to powers of two.

    Returns:
        (fp8_data, descale) where fp8_data has dtype float8_e4m3fn and
        descale is float32 with shape (blk_m, blk_n, 1).
    """
    assert len(data_hp.shape) == 2, "Only 2-D input tensor is supported"

    block_size0, block_size1 = weight_block_size
    shape_before_padding = data_hp.shape

    if data_hp.shape[0] % block_size0 != 0 or data_hp.shape[1] % block_size1 != 0:
        pad0 = (block_size0 - data_hp.shape[0] % block_size0) % block_size0
        pad1 = (block_size1 - data_hp.shape[1] % block_size1) % block_size1
        data_hp = torch.nn.functional.pad(
            data_hp, (0, pad1, 0, pad0), mode="constant", value=data_hp[-1, -1]
        )

    max_dtype = torch.finfo(torch.float8_e4m3fn).max
    original_shape = data_hp.shape
    blk_m = data_hp.shape[0] // block_size0
    blk_n = data_hp.shape[1] // block_size1

    assert block_size0 == block_size1
    data_hp = data_hp.reshape(blk_m, block_size0, blk_n, block_size1)
    data_hp = data_hp.permute(0, 2, 1, 3)
    data_hp = data_hp.to(torch.float32).contiguous().flatten(start_dim=2)

    max_abs = torch.amax(torch.abs(data_hp), dim=-1, keepdim=True)

    if use_pow2_scale:
        descale = max_abs / max_dtype
        exponent = torch.ceil(torch.log2(descale))
        exponent = torch.clamp(exponent, min=-127, max=127) + 127
        exponent = exponent.to(torch.uint8)
        scale_fp = torch.where(
            exponent == 0,
            1.0,
            torch.exp2(127 - exponent.to(torch.float32)),
        )
        descale_fp = torch.reciprocal(scale_fp)
    else:
        scale_fp = max_dtype / max_abs
        scale_fp = torch.where(max_abs == 0, 1.0, scale_fp)
        scale_fp = torch.where(max_abs == torch.inf, 1.0, scale_fp)
        descale_fp = torch.reciprocal(scale_fp)

    data_lp = torch.clamp(data_hp * scale_fp, min=-max_dtype, max=max_dtype)
    fp_data = data_lp.to(torch.float8_e4m3fn)

    fp_data = (
        fp_data.reshape(blk_m, blk_n, block_size0, block_size1)
        .permute(0, 2, 1, 3)
        .reshape(original_shape)
    )

    if original_shape != shape_before_padding:
        fp_data = fp_data[: shape_before_padding[0], : shape_before_padding[1]]

    return fp_data, descale_fp


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
