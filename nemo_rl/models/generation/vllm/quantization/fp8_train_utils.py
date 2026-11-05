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


import functools
import importlib.util
import os
import re
import shutil
from typing import Any, Iterable

import torch

MXFP8_BLOCK_SIZE = 32
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_SCALE_SUFFIX = "_scale_from_checkpoint"
_ATTENTION_QKV_SUFFIX = ".linear_qkv.weight"
# Language-model (decoder or MTP) attention, optionally under a multimodal root;
# encoder attention (e.g. vision) also uses linear_qkv but has no KV cache.
_LANGUAGE_ATTENTION_PREFIX = re.compile(
    r"^(?:(?:\w+\.)*language_model\.)?(?:decoder|mtp)\.layers\."
)


def canonicalize_mxfp8_refit_output(
    weight_shape: torch.Size | tuple[int, ...],
    values: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and normalize the MXFP8 refit wire representation.

    Both supported trainer-side and vLLM receiver-side quantizers produce this
    format: E4M3 values retain the logical checkpoint shape, while E8M0 scale
    bytes use one entry per 32 values along the final dimension. A zero scale
    byte is only valid for an all-zero value block. The scale tensor is sent as
    the matching ``*_scale_from_checkpoint`` parameter.
    """
    shape = torch.Size(weight_shape)
    if not shape or shape[-1] % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP8 requires a non-empty shape with the last dim divisible by "
            f"{MXFP8_BLOCK_SIZE}, got {tuple(shape)}."
        )
    if values.numel() != shape.numel():
        raise ValueError(
            f"MXFP8 values must contain {shape.numel()} elements for weight "
            f"shape {tuple(shape)}, got {values.numel()}."
        )
    if not values.shape or values.shape[-1] != shape[-1]:
        raise ValueError(
            f"MXFP8 values must preserve the final dimension {shape[-1]} for "
            f"weight shape {tuple(shape)}, got {tuple(values.shape)}."
        )
    if values.dtype != MXFP8_VALUE_DTYPE:
        raise ValueError(
            f"MXFP8 values must use {MXFP8_VALUE_DTYPE}, got {values.dtype}."
        )
    if scales.dtype != MXFP8_SCALE_DTYPE:
        raise ValueError(
            f"MXFP8 scales must use {MXFP8_SCALE_DTYPE}, got {scales.dtype}."
        )

    if values.shape != shape:
        values = values.reshape(shape)
    scale_shape = torch.Size((*shape[:-1], shape[-1] // MXFP8_BLOCK_SIZE))
    expected_scales = scale_shape.numel()
    if scales.numel() != expected_scales:
        raise ValueError(
            f"MXFP8 scales must contain {expected_scales} values for weight "
            f"shape {tuple(shape)}, got {scales.numel()}."
        )
    scales = scales.reshape(scale_shape)
    # An E8M0 byte of zero represents 2^-127. FlashInfer emits it for all-zero
    # blocks, but TRTLLM does not accept it. Changing the scale to one preserves
    # the represented zeros because every value in the block is zero.
    scales = scales.masked_fill(scales == 0, 1)
    return values, scales


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
    fp8_max = torch.finfo(MXFP8_VALUE_DTYPE).max
    scale_biased = torch.ceil(torch.log2(amax / fp8_max)) + 127.0
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


@functools.cache
def _receiver_has_flashinfer() -> bool:
    """Mirror vLLM's FlashInfer availability check without importing vLLM in trainer workers."""
    if importlib.util.find_spec("flashinfer") is None:
        return False
    has_cubin = os.environ.get("VLLM_HAS_FLASHINFER_CUBIN") == "1" or (
        importlib.util.find_spec("flashinfer_cubin") is not None
    )
    return has_cubin or shutil.which("nvcc") is not None


def mxfp8_e4m3_quantize_for_refit(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight to MXFP8 on the trainer for pre-quantized refit.

    Mirrors the receiver path in quantization/fp8.py load_weights
    (mxfp8_e4m3_quantize + scale reshape) so the streamed E4M3 data and
    *_scale_from_checkpoint scales load bit-identically without receiver-side
    re-quantization. Uses the same FlashInfer CuTe-DSL backend as vLLM on
    Blackwell when vLLM can select it, and the torch reference otherwise.
    """
    x_q = x_scales = None
    if (
        x.is_cuda
        and torch.cuda.get_device_capability(x.device) >= (10, 0)
        and _receiver_has_flashinfer()
    ):
        try:
            from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize
        except ImportError as exc:
            raise RuntimeError(
                "Trainer-side MXFP8 refit prequantization on sm100+ requires "
                "FlashInfer so it matches the vLLM receiver quantization path."
            ) from exc
        else:
            x_q, x_scales = flashinfer_mxfp8_quantize(
                x,
                is_sf_swizzled_layout=False,
                alignment=32,
                backend="cute-dsl",
            )
            if x_scales.ndim == 1 and x.ndim == 2:
                x_scales = x_scales.view(x.size(0), -1)
    if x_q is None or x_scales is None:
        x_q, x_scales = _mxfp8_e4m3_quantize_torch(x)
    return canonicalize_mxfp8_refit_output(x.shape, x_q, x_scales)


def get_vllm_qkv_scale_names(attention_prefix: str) -> dict[str, str]:
    """Get vLLM-compatible parameter names for one attention layer's Q/K/V FP8 scales.

    Args:
        attention_prefix: HF name of the module that owns the layer's Q/K/V
            projections, as returned by ``get_hf_attention_prefixes``.

    Returns:
        Dictionary mapping 'q_scale', 'k_scale' and 'v_scale' to the scales of
        the vLLM ``Attention`` submodule (``attn``) under ``attention_prefix``.
        These are vLLM's own parameter names, so the cache-scale remapping in
        ``maybe_remap_kv_scale_name`` leaves them unchanged; the model's
        HF-to-vLLM mapper still rewrites the root prefix (e.g. ``backbone``).

    Example:
        >>> get_vllm_qkv_scale_names("backbone.layers.4.mixer")
        {
            'q_scale': 'backbone.layers.4.mixer.attn.q_scale',
            'k_scale': 'backbone.layers.4.mixer.attn.k_scale',
            'v_scale': 'backbone.layers.4.mixer.attn.v_scale'
        }
    """
    return {
        scale: f"{attention_prefix}.attn.{scale}"
        for scale in ("q_scale", "k_scale", "v_scale")
    }


def get_hf_attention_prefixes(conversion_tasks: Iterable[Any]) -> dict[str, str]:
    """Map each Megatron attention module to the HF name of the same module.

    The Megatron-Bridge mapping of ``<module>.linear_qkv.weight`` names the HF
    Q (or fused QKV) projection, whose parent is the HF attention module:
    ``decoder.layers.4.self_attention`` -> ``backbone.layers.4.mixer`` for
    NemotronH, ``model.layers.4.self_attn`` for Llama/Qwen, and MTP layers map
    to their own ``mtp.*`` names. Only language-model attention layers appear:
    hybrid models get no entries for Mamba or MoE layers, multimodal models none
    for their encoders, and MLA models (no ``linear_qkv``) none at all.

    Args:
        conversion_tasks: Megatron-Bridge ``WeightConversionTask`` objects;
            ``None`` entries (parameters owned by other ranks) are skipped.

    Returns:
        Dictionary from the unwrapped global Megatron module name to the HF
        attention module name.
    """
    prefixes = {}
    for task in conversion_tasks:
        if (
            task is None
            or not task.global_param_name.endswith(_ATTENTION_QKV_SUFFIX)
            or not _LANGUAGE_ATTENTION_PREFIX.match(task.global_param_name)
        ):
            continue
        hf_param = task.mapping.hf_param
        hf_name = hf_param["q"] if isinstance(hf_param, dict) else hf_param
        megatron_prefix = task.global_param_name.removesuffix(_ATTENTION_QKV_SUFFIX)
        prefixes[megatron_prefix] = hf_name.rsplit(".", 2)[0]
    return prefixes


def convert_calibration_to_vllm_format(
    calibration_results: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Convert NeMo-RL calibration results to vLLM parameter format.

    Args:
        calibration_results: Dict keyed by HF attention prefix (see
            ``get_hf_attention_prefixes``). Each value maps "k_scale",
            "v_scale" and optionally "q_scale" to its float scale.

    Returns:
        Flat dictionary mapping vLLM parameter names (see
        ``get_vllm_qkv_scale_names``) to scale values.

    Example:
        >>> convert_calibration_to_vllm_format(
        ...     {"backbone.layers.4.mixer": {"k_scale": 2.0, "v_scale": 3.0}}
        ... )
        {
            'backbone.layers.4.mixer.attn.k_scale': 2.0,
            'backbone.layers.4.mixer.attn.v_scale': 3.0
        }
    """
    vllm_scales = {}
    for attention_prefix, scales in calibration_results.items():
        param_names = get_vllm_qkv_scale_names(attention_prefix)
        for scale, value in scales.items():
            vllm_scales[param_names[scale]] = value
    return vllm_scales
