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

MXFP8_BLOCK_SIZE = 32
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn


def _mxfp8_e4m3_quantize_torch(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference MXFP8 quantization with row-major scales.

    Replicates vLLM's _mxfp8_e4m3_quantize_torch (Apache-2.0,
    vllm/model_executor/layers/quantization/utils/mxfp8_utils.py) for trainer
    processes without a vLLM install: for each block of 32 elements along the
    last dimension, a shared e8m0 scale (biased exponent of the block amax)
    and float8_e4m3fn values.
    """
    assert x.shape[-1] % MXFP8_BLOCK_SIZE == 0, (
        f"MXFP8 requires the last dim to be divisible by {MXFP8_BLOCK_SIZE}, got {x.shape}"
    )
    orig_shape = x.shape
    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE

    x_fp32 = x.to(torch.float32)
    x_blocked = x_fp32.view(*orig_shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)

    amax = x_blocked.abs().amax(dim=-1)
    amax = amax.clamp(min=torch.finfo(torch.float32).tiny)
    scale_biased = torch.floor(torch.log2(amax)) + 127.0
    scale_biased = scale_biased.clamp(0, 254)
    scales_uint8 = scale_biased.to(torch.uint8)

    descale = torch.exp2(scale_biased - 127.0)
    x_scaled = x_blocked / descale.unsqueeze(-1)

    x_fp8 = x_scaled.view(orig_shape).to(MXFP8_VALUE_DTYPE)

    if x.ndim == 2:
        scales_uint8 = scales_uint8.view(x.shape[0], -1)
    elif x.ndim == 3:
        scales_uint8 = scales_uint8.view(x.shape[0], x.shape[1], -1)

    return x_fp8, scales_uint8


def mxfp8_e4m3_quantize_for_refit(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight to MXFP8 on the trainer for pre-quantized refit.

    Mirrors the receiver path in quantization/fp8.py load_weights
    (mxfp8_e4m3_quantize + scale squeeze) so the streamed E4M3 data and
    *_scale_from_checkpoint scales load bit-identically without receiver-side
    re-quantization. Uses the same flashinfer kernel as vLLM on Blackwell and
    the torch reference elsewhere.
    """
    x_q = x_scales = None
    # Kernel dispatch keys off the TRAINER GPU while the receiver keys off the
    # inference GPU. On homogeneous clusters both take the same path; on mixed
    # Hopper/Blackwell clusters the flashinfer and torch paths may differ in
    # boundary rounding - validate with the parity test before relying on it.
    if x.is_cuda and torch.cuda.get_device_capability(x.device) >= (10, 0):
        try:
            from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize
        except ImportError:
            pass
        else:
            x_q, x_scales = flashinfer_mxfp8_quantize(
                x, is_sf_swizzled_layout=False, alignment=32
            )
            if x_scales.ndim == 1 and x.ndim == 2:
                x_scales = x_scales.view(x.size(0), -1)
    if x_q is None:
        x_q, x_scales = _mxfp8_e4m3_quantize_torch(x)
    x_scales = torch.squeeze(x_scales, dim=-1)
    # Match the receiver path's zero-scale clamp: an E8M0 byte of 0 (2^-127)
    # destabilizes the TRTLLM kernels, and pre-quantized tensors skip the
    # receiver-side quantize branch where the clamp normally runs.
    x_scales = torch.where(x_scales == 0, torch.ones_like(x_scales), x_scales)
    return x_q, x_scales


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
