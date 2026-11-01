# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

Offline conversion (``mxfp8_setup.py``) and the Megatron SGLang online-refit
path must call into this module so they make the exact same
quantization decision for any given HF tensor name.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import torch

SKIP_WEIGHT_SUBSTRINGS: tuple[str, ...] = (
    "layernorm",
    "embed",
    "router",
    # A top-k router is a discrete argmax, so quantization error flips expert
    # selection outright -- the one failure an RL rollout cannot absorb. Every
    # MoE family spells it differently and only some contain "router":
    # ``mlp.gate.`` (Qwen3-MoE, DeepSeek), ``block_sparse_moe.gate.``
    # (Mixtral), ``mixer.gate.`` (NemotronH/nanov3) and the separate
    # ``shared_expert_gate`` (Qwen2-MoE). ``.gate.`` covers the first three as
    # a class rather than enumerating families we happen to have seen.
    # The dots are load-bearing: ``gate_proj``, ``gate_up_proj`` and
    # ``shared_expert.gate_proj`` are ordinary MLP weights and stay quantized.
    # The repo already draws this line -- see ``DEFAULT_NVFP4_IGNORE`` in
    # ``nemo_rl/modelopt/utils.py`` and the nanov3 recipe's
    # ``real_quant_ignore``.
    ".gate.",
    "shared_expert_gate.",
    "norm",
    "lm_head",
    "eh_proj",
    "weights_proj",
)
TARGET_MXFP8_BLOCK_SIZE: list[int] = [1, 32]
# Suffix of the block-scale tensor emitted alongside each quantized weight.
MXFP8_SCALE_KEY_SUFFIX: str = ".weight_scale_inv"

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
) -> bool:
    allowed_dtypes: tuple[torch.dtype, ...] = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    )
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

    Uses flashinfer's swizzle-free MXFP8 kernel
    (``is_sf_swizzled_layout=False``).
    """
    try:
        from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize
    except ImportError as e:
        raise ImportError(
            "flashinfer is required for MXFP8 weight quantization but is not "
            "installed in the current actor environment. In NeMo-RL this is "
            "normally provided by the pinned `mcore` or `sglang` extras."
        ) from e

    weight = weight.contiguous()
    k = weight.shape[-1]
    if k % 32 != 0:
        raise ValueError(f"Last dim {k} must be divisible by 32 for MXFP8.")

    weight_flat = weight.view(-1, k).contiguous()
    qweight, scale = flashinfer_mxfp8_quantize(weight_flat, is_sf_swizzled_layout=False)
    qweight = qweight.view_as(weight)
    scale = scale.view(*weight.shape[:-1], k // 32).contiguous()
    return qweight, scale


def iter_mxfp8_quantized_tensor_groups(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    skip_weight_substrings: tuple[str, ...],
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Yield each HF tensor and any MXFP8 scale companion as one group."""
    for name, tensor in named_tensors:
        if should_quantize(
            name,
            tensor,
            skip_weight_substrings=skip_weight_substrings,
        ):
            qweight, scale = quantize_mxfp8(tensor)
            scale_name = strip_weight_suffix(name) + MXFP8_SCALE_KEY_SUFFIX
            yield [(name, qweight), (scale_name, scale)]
        else:
            yield [(name, tensor)]
