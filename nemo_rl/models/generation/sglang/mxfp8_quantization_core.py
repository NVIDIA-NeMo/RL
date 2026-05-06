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

"""Shared MXFP8 tensor quantization rules for SGLang rollout weight updates.

Offline conversion (``mxfp8_setup.py``) and online refit (the Megatron SGLang
weight iterator) must call into this module so they make the exact same
quantization decision for any given HF tensor name.
"""

from __future__ import annotations

from typing import Any

import torch

SKIP_WEIGHT_SUBSTRINGS: tuple[str, ...] = (
    "layernorm",
    "embed",
    "router",
    "mlp.gate.",
    "norm",
    "lm_head",
    "eh_proj",
    "weights_proj",
)
SOURCE_FP8_BLOCK_SIZE: list[int] = [128, 128]
TARGET_MXFP8_BLOCK_SIZE: list[int] = [1, 32]
SOURCE_FP8_SCALE_KEY_SUFFIX: str = ".weight_scale_inv"
SOURCE_FP8_DTYPES: tuple[torch.dtype, ...] = (torch.float8_e4m3fn,) + (
    (torch.float8_e4m3fnuz,) if hasattr(torch, "float8_e4m3fnuz") else ()
)

MXFP8_QUANTIZATION_CONFIG: dict[str, Any] = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "mxfp8",
    "weight_block_size": TARGET_MXFP8_BLOCK_SIZE,
    "scale_fmt": "ue8m0",
}


def strip_weight_suffix(weight_key: str) -> str:
    if not weight_key.endswith(".weight"):
        raise ValueError(f"Expected key ending with '.weight', got: {weight_key}")
    return weight_key[: -len(".weight")]


def is_mxfp8_quantization_config(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False
    return (
        config.get("quant_method") == "mxfp8"
        and list(config.get("weight_block_size", [])) == TARGET_MXFP8_BLOCK_SIZE
        and config.get("scale_fmt") == "ue8m0"
    )


def is_source_block_fp8_ue8m0_checkpoint(cfg: dict[str, Any]) -> bool:
    qcfg = cfg.get("quantization_config", {}) if isinstance(cfg, dict) else {}
    return (
        qcfg.get("quant_method") == "fp8"
        and list(qcfg.get("weight_block_size", [])) == SOURCE_FP8_BLOCK_SIZE
        and qcfg.get("scale_fmt") == "ue8m0"
    )


def is_bf16_source_checkpoint(cfg: dict[str, Any]) -> bool:
    qcfg = cfg.get("quantization_config", {}) if isinstance(cfg, dict) else {}
    if not isinstance(qcfg, dict) or not qcfg:
        return True
    return qcfg.get("quant_method") in (None, "", "bf16")


def should_quantize(
    name: str,
    weight: torch.Tensor,
    *,
    skip_weight_substrings: tuple[str, ...] = SKIP_WEIGHT_SUBSTRINGS,
    allow_source_fp8: bool = False,
) -> bool:
    allowed_dtypes: tuple[torch.dtype, ...] = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    )
    if allow_source_fp8:
        allowed_dtypes = allowed_dtypes + SOURCE_FP8_DTYPES
    if not name.endswith(".weight"):
        return False
    if any(substr in name for substr in skip_weight_substrings):
        return False
    if weight.dtype not in allowed_dtypes:
        return False
    if weight.dim() < 2:
        return False
    if weight.shape[-1] % 32 != 0:
        return False
    return True


def quantize_mxfp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(qweight, scale)`` in the SGLang MXFP8 layout.

    Uses flashinfer's swizzle-free MXFP8 kernel (``flashinfer.mxfp8_quantize``
    with ``is_sf_swizzled_layout=False``). flashinfer is a hard requirement
    here — both the SGLang and Megatron actor environments pin it via
    ``pyproject.toml``'s global ``flashinfer-python==0.6.4`` constraint, so a
    missing import means the env was built incorrectly.
    """
    try:
        from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize
    except ImportError as e:
        raise ImportError(
            "flashinfer is required for MXFP8 weight quantization but is not "
            "installed in the current actor environment. Install "
            "`flashinfer-python==0.6.4` (and `flashinfer-cubin==0.6.4`); "
            "in NeMo-RL this is normally provided by the `mcore` or `sglang` "
            "extras (see pyproject.toml constraint-dependencies)."
        ) from e

    weight = weight.contiguous()
    k = weight.shape[-1]
    if k % 32 != 0:
        raise ValueError(f"Last dim {k} must be divisible by 32 for MXFP8.")

    weight_flat = weight.view(-1, k).contiguous()
    qweight, scale = flashinfer_mxfp8_quantize(
        weight_flat, is_sf_swizzled_layout=False
    )
    qweight = qweight.view_as(weight)
    scale = scale.view(*weight.shape[:-1], k // 32).contiguous()
    return qweight, scale


def source_fp8_to_mxfp8_scale_u8(
    weight: torch.Tensor, source_scale_u8: torch.Tensor
) -> torch.Tensor:
    n, k = weight.shape[-2], weight.shape[-1]
    mxfp8_scale_u8 = source_scale_u8.repeat_interleave(
        SOURCE_FP8_BLOCK_SIZE[0], dim=-2
    ).repeat_interleave(
        SOURCE_FP8_BLOCK_SIZE[1] // TARGET_MXFP8_BLOCK_SIZE[1], dim=-1
    )
    return mxfp8_scale_u8[..., :n, : (k // TARGET_MXFP8_BLOCK_SIZE[1])].contiguous()


def build_dynamic_skip_substrings(
    *,
    quantization_config: dict[str, Any],
    num_hidden_layers: int,
) -> tuple[str, ...]:
    """Compute the dynamic skip substrings for one HF model.

    Combines the static ``SKIP_WEIGHT_SUBSTRINGS`` list with the user-provided
    ``extra_high_precision_layers_hf`` / ``modules_to_not_convert`` lists from
    the quantization config, plus per-layer prefixes for the ``head`` / ``tail``
    BF16-band layers.
    """
    extra_high_precision_layers_hf = tuple(
        quantization_config.get("extra_high_precision_layers_hf", ()) or ()
    )
    modules_to_not_convert = tuple(
        quantization_config.get("modules_to_not_convert", ()) or ()
    )
    num_layers_at_start_in_bf16 = int(
        quantization_config.get("num_layers_at_start_in_bf16", 0) or 0
    )
    num_layers_at_end_in_bf16 = int(
        quantization_config.get("num_layers_at_end_in_bf16", 0) or 0
    )

    head_end_idx = num_layers_at_start_in_bf16
    tail_start_idx = num_hidden_layers - num_layers_at_end_in_bf16
    dynamic_skip_layer_prefixes: set[str] = set()
    dynamic_skip_layer_prefixes.update(
        f"model.layers.{i}." for i in range(0, head_end_idx)
    )
    dynamic_skip_layer_prefixes.update(
        f"model.layers.{i}." for i in range(tail_start_idx, num_hidden_layers)
    )
    return (
        *SKIP_WEIGHT_SUBSTRINGS,
        *extra_high_precision_layers_hf,
        *modules_to_not_convert,
        *sorted(dynamic_skip_layer_prefixes),
    )


def maybe_quantize_hf_weight_mxfp8(
    name: str,
    tensor: torch.Tensor,
    *,
    quantization_config: dict[str, Any],
    num_hidden_layers: int,
) -> list[tuple[str, torch.Tensor]]:
    """Apply the HF-name MXFP8 policy to one finalized HF tensor.

    Returns a list of ``(name, tensor)`` pairs:

    - For an unquantized tensor: ``[(name, tensor)]``.
    - For a quantized tensor: ``[(name, qweight), (name_scale_inv, scale)]``.
    """
    skip_weight_substrings = build_dynamic_skip_substrings(
        quantization_config=quantization_config,
        num_hidden_layers=num_hidden_layers,
    )
    if not should_quantize(
        name,
        tensor,
        skip_weight_substrings=skip_weight_substrings,
        allow_source_fp8=False,
    ):
        return [(name, tensor)]

    qweight, scale = quantize_mxfp8(tensor)
    scale_name = strip_weight_suffix(name) + SOURCE_FP8_SCALE_KEY_SUFFIX
    return [(name, qweight), (scale_name, scale)]
