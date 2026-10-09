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

"""Transformer Engine NVFP4 conversion rules for SGLang rollout weights."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import Any

import torch

from nemo_rl.models.generation.sglang.quantization_utils import (
    HF_MOE_EXPERT_NAME_MARKERS,
)

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448
NVFP4_GROUP_SIZE = 16
TE_NVFP4_ROW_ALIGNMENT = 16

NVFP4_QUANTIZATION_CONFIG: dict[str, Any] = {
    "group_size": NVFP4_GROUP_SIZE,
    "ignore": [],
    "quant_algo": "NVFP4",
    "quant_method": "modelopt",
}

EXPERT_WEIGHT_SUFFIXES = (
    ".w1.weight",
    ".w2.weight",
    ".w3.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
    ".gate_up_proj.weight",
)
GATED_PAIR_SUFFIXES = {
    ".gate_proj.weight": "gate",
    ".up_proj.weight": "up",
    ".w1.weight": "gate",
    ".w3.weight": "up",
}
GATED_PAIR_FAMILIES = (
    (".gate_proj.weight", ".up_proj.weight"),
    (".w1.weight", ".w3.weight"),
)


def strip_weight_suffix(weight_key: str) -> str:
    """Remove the trailing ``.weight`` from an HF parameter name."""
    if not weight_key.endswith(".weight"):
        raise ValueError(f"Expected key ending with '.weight', got: {weight_key}")
    return weight_key[: -len(".weight")]


def is_nvfp4_quantization_config(config: dict[str, Any] | None) -> bool:
    """Return whether ``config`` describes the ModelOpt NVFP4 layout."""
    if not isinstance(config, dict):
        return False
    return (
        config.get("quant_algo") == "NVFP4"
        and config.get("group_size") == NVFP4_GROUP_SIZE
    )


def is_bf16_source_checkpoint(config: dict[str, Any]) -> bool:
    """Return whether an HF checkpoint contains ordinary high-precision weights."""
    quantization_config = (
        config.get("quantization_config", {}) if isinstance(config, dict) else {}
    )
    if not isinstance(quantization_config, dict) or not quantization_config:
        return True
    return quantization_config.get("quant_method") in (None, "", "bf16")


def is_moe_expert_weight_name(name: str) -> bool:
    """Return whether an HF name denotes a quantizable MoE expert GEMM."""
    return (
        name.endswith(".weight")
        and any(marker in name for marker in HF_MOE_EXPERT_NAME_MARKERS)
        and any(name.endswith(suffix) for suffix in EXPERT_WEIGHT_SUFFIXES)
    )


def should_quantize_nvfp4(
    name: str,
    weight: torch.Tensor,
    *,
    skip_weight_substrings: tuple[str, ...] = (),
) -> bool:
    """Return whether an HF tensor should be emitted in NVFP4 format."""
    if any(substring in name for substring in skip_weight_substrings):
        return False
    if not is_moe_expert_weight_name(name):
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if weight.dim() != 2:
        return False
    if weight.shape[-1] % NVFP4_GROUP_SIZE != 0:
        raise ValueError(
            f"Last dim {weight.shape[-1]} must be divisible by "
            f"{NVFP4_GROUP_SIZE} for NVFP4 ({name})."
        )
    return True


def split_gated_pair_name(name: str) -> tuple[str | None, str | None]:
    """Return the shared base and role for an HF gate/up weight name."""
    for suffix, role in GATED_PAIR_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], role
    return None, None


def gated_pair_names(name: str) -> tuple[str, str] | None:
    """Return the gate/up names for the pair containing ``name``."""
    for gate_suffix, up_suffix in GATED_PAIR_FAMILIES:
        if name.endswith(gate_suffix):
            base = name[: -len(gate_suffix)]
            return base + gate_suffix, base + up_suffix
        if name.endswith(up_suffix):
            base = name[: -len(up_suffix)]
            return base + gate_suffix, base + up_suffix
    return None


def should_skip_nvfp4_gated_pair(
    name: str,
    *,
    skip_weight_substrings: tuple[str, ...],
) -> bool:
    """Keep both halves high precision when either gate/up name is excluded."""
    pair_names = gated_pair_names(name)
    if pair_names is None:
        return False
    return any(
        substring in pair_name
        for substring in skip_weight_substrings
        for pair_name in pair_names
    )


def nvfp4_weight_e4m3_max() -> int:
    """Return TE's configured global E4M3 bound for NVFP4 weights."""
    use_4over6 = os.getenv("NVTE_NVFP4_4OVER6", "").strip().lower()
    use_256 = os.getenv("NVTE_NVFP4_4OVER6_E4M3_USE_256", "all").strip().lower()
    if use_4over6 in ("weights", "all") and use_256 in ("weights", "all"):
        return 256
    return FP8_E4M3_MAX


def nvfp4_global_encode_scale_te(
    global_amax: torch.Tensor,
    *,
    nvfp4_e4m3_max: int,
) -> torch.Tensor:
    """Reproduce Transformer Engine's per-tensor NVFP4 encode scale."""
    fp4_max = torch.tensor(
        FP4_E2M1_MAX,
        device=global_amax.device,
        dtype=torch.float32,
    )
    fp8_max = torch.tensor(
        float(nvfp4_e4m3_max),
        device=global_amax.device,
        dtype=torch.float32,
    )
    global_encode_scale = torch.div(
        fp8_max * fp4_max,
        global_amax.to(torch.float32),
    )
    global_encode_scale = torch.minimum(
        global_encode_scale,
        torch.tensor(
            torch.finfo(torch.float32).max,
            device=global_amax.device,
            dtype=torch.float32,
        ),
    )
    return torch.where(
        global_encode_scale == 0.0,
        torch.ones_like(global_encode_scale),
        global_encode_scale,
    )


def nvfp4_global_decode_scale_te(
    global_amax: torch.Tensor,
    *,
    nvfp4_e4m3_max: int,
) -> torch.Tensor:
    """Return the FP32 scale stored as ``weight_scale_2`` by ModelOpt."""
    return torch.reciprocal(
        nvfp4_global_encode_scale_te(
            global_amax,
            nvfp4_e4m3_max=nvfp4_e4m3_max,
        )
    )


def _nvfp4_4over6_enabled() -> bool:
    return os.getenv("NVTE_NVFP4_4OVER6", "").strip().lower() in (
        "weights",
        "all",
    )


def _pad_rows_for_te_quantizer(weight: torch.Tensor) -> torch.Tensor:
    pad_rows = (-weight.shape[0]) % TE_NVFP4_ROW_ALIGNMENT
    if pad_rows == 0:
        return weight
    padding = torch.zeros(
        (pad_rows, weight.shape[1]),
        device=weight.device,
        dtype=weight.dtype,
    )
    return torch.cat((weight, padding), dim=0)


def _make_nvfp4_quantizer() -> Any:
    # Transformer Engine is heavy and optional outside GPU actor environments.
    try:
        from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
    except ImportError as error:
        raise ImportError(
            "Transformer Engine >=2.17 is required for NVFP4 weight "
            "quantization. Install NeMo-RL with the `mcore` extra."
        ) from error

    try:
        return NVFP4Quantizer(
            rowwise=True,
            columnwise=False,
            with_amax_reduction=False,
            with_rht=False,
            with_post_rht_amax=False,
            with_2d_quantization=False,
            stochastic_rounding=False,
            row_scaled_nvfp4=False,
            nvfp4_use_4over6=_nvfp4_4over6_enabled(),
            nvfp4_e4m3_max=nvfp4_weight_e4m3_max(),
            nvfp4_4over6_err_mode=os.getenv(
                "NVTE_NVFP4_4OVER6_ERR_MODE",
                "MAE",
            )
            .strip()
            .upper(),
            with_random_sign_mask=False,
        )
    except TypeError as error:
        raise RuntimeError(
            "The installed Transformer Engine does not expose the NVFP4 "
            "quantizer API required by NeMo-RL. Upgrade to version 2.17 or newer."
        ) from error


def nvfp4_quantize_2d(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize one 2D weight into ModelOpt's NVFP4 checkpoint layout."""
    if weight.dim() != 2:
        raise ValueError(f"nvfp4_quantize_2d expects a 2D tensor, got {weight.dim()}D.")
    if weight.shape[1] % NVFP4_GROUP_SIZE != 0:
        raise ValueError(
            f"NVFP4 requires K divisible by {NVFP4_GROUP_SIZE}, got {weight.shape[1]}."
        )

    weight = weight.contiguous()
    num_rows, num_cols = weight.shape
    nvfp4_e4m3_max = nvfp4_weight_e4m3_max()
    quantized = _make_nvfp4_quantizer().quantize(_pad_rows_for_te_quantizer(weight))

    rowwise_data = getattr(quantized, "_rowwise_data", None)
    rowwise_scale_inv = getattr(quantized, "_rowwise_scale_inv", None)
    amax_rowwise = getattr(quantized, "_amax_rowwise", None)
    if rowwise_data is None or rowwise_scale_inv is None or amax_rowwise is None:
        raise RuntimeError(
            "Transformer Engine returned an incomplete rowwise NVFP4 tensor."
        )

    qweight = rowwise_data[:num_rows, : num_cols // 2].contiguous()
    block_scale = rowwise_scale_inv[
        :num_rows,
        : num_cols // NVFP4_GROUP_SIZE,
    ].contiguous()
    weight_scale_2 = nvfp4_global_decode_scale_te(
        amax_rowwise.reshape(-1)[0],
        nvfp4_e4m3_max=nvfp4_e4m3_max,
    )
    return (
        qweight,
        block_scale.view(torch.float8_e4m3fn),
        weight_scale_2,
    )


def _contiguous_pair_view(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor | None:
    if not first.is_contiguous() or not second.is_contiguous():
        return None
    if first.stride() != (first.shape[1], 1):
        return None
    if (
        first.device != second.device
        or first.dtype != second.dtype
        or first.stride() != second.stride()
    ):
        return None
    if first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr():
        return None
    if first.storage_offset() + first.numel() != second.storage_offset():
        return None
    try:
        return first.as_strided(
            (first.shape[0] + second.shape[0], first.shape[1]),
            first.stride(),
            first.storage_offset(),
        )
    except RuntimeError:
        return None


def nvfp4_quantize_2d_pair(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Quantize a gate/up pair together so both weights share global scale."""
    if first.dim() != 2 or second.dim() != 2:
        raise ValueError("nvfp4_quantize_2d_pair expects two 2D tensors.")
    if first.shape[1] != second.shape[1]:
        raise ValueError(
            "NVFP4 paired quantization requires matching K dimensions, got "
            f"{first.shape[1]} and {second.shape[1]}."
        )
    if first.dtype != second.dtype:
        raise ValueError(
            "NVFP4 paired quantization requires matching dtypes, got "
            f"{first.dtype} and {second.dtype}."
        )
    if first.device != second.device:
        raise ValueError(
            "NVFP4 paired quantization requires weights on the same device, got "
            f"{first.device} and {second.device}."
        )

    first_rows = first.shape[0]
    combined = _contiguous_pair_view(first, second)
    if combined is None:
        combined = torch.cat((first.contiguous(), second.contiguous()), dim=0)
    combined_qweight, combined_block_scale, weight_scale_2 = nvfp4_quantize_2d(combined)
    first_result = (
        combined_qweight[:first_rows].contiguous(),
        combined_block_scale[:first_rows].contiguous(),
        weight_scale_2.clone(),
    )
    second_result = (
        combined_qweight[first_rows:].contiguous(),
        combined_block_scale[first_rows:].contiguous(),
        weight_scale_2.clone(),
    )
    return first_result, second_result


def quantize_nvfp4(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D expert weight."""
    if weight.dim() != 2:
        raise ValueError(
            f"Unsupported weight rank {weight.dim()} for NVFP4 quantization."
        )
    return nvfp4_quantize_2d(weight)


def quantize_nvfp4_pair(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Quantize a 2D gate/up pair under one shared global scale."""
    if first.dim() != 2 or second.dim() != 2:
        raise ValueError(
            "NVFP4 paired quantization requires 2D weights, got "
            f"{first.dim()}D and {second.dim()}D."
        )
    return nvfp4_quantize_2d_pair(first, second)


def nvfp4_quantized_entries(
    name: str,
    quantized: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    include_input_scale: bool,
) -> list[tuple[str, torch.Tensor]]:
    """Expand one quantized weight into the ModelOpt tensor names."""
    qweight, block_scale, weight_scale_2 = quantized
    base_name = strip_weight_suffix(name)
    result = [
        (name, qweight),
        (f"{base_name}.weight_scale", block_scale),
        (f"{base_name}.weight_scale_2", weight_scale_2),
    ]
    if include_input_scale:
        result.append(
            (
                f"{base_name}.input_scale",
                torch.ones_like(weight_scale_2, dtype=torch.float32),
            )
        )
    return result


def iter_nvfp4_quantized_tensor_groups(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    skip_weight_substrings: tuple[str, ...],
    require_gate_up_pairs: bool = True,
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Yield NVFP4 expert tensors, buffering complete gated pairs when required."""
    pending_pairs: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}

    for name, tensor in named_tensors:
        if require_gate_up_pairs and should_skip_nvfp4_gated_pair(
            name, skip_weight_substrings=skip_weight_substrings
        ):
            yield [(name, tensor)]
            continue

        if not should_quantize_nvfp4(
            name,
            tensor,
            skip_weight_substrings=skip_weight_substrings,
        ):
            yield [(name, tensor)]
            continue

        pair_base, pair_role = (
            split_gated_pair_name(name) if require_gate_up_pairs else (None, None)
        )
        if pair_base is None or pair_role is None:
            yield nvfp4_quantized_entries(
                name,
                quantize_nvfp4(tensor),
                include_input_scale=False,
            )
            continue

        pair = pending_pairs.setdefault(pair_base, {})
        if pair_role in pair:
            raise ValueError(
                "NVFP4 requires one complete gate/up pair per refit; "
                f"found duplicate {pair_role} tensor for {pair_base}."
            )
        pair[pair_role] = (name, tensor)
        if set(pair) != {"gate", "up"}:
            continue

        gate_name, gate_weight = pair["gate"]
        up_name, up_weight = pair["up"]
        gate_output, up_output = quantize_nvfp4_pair(gate_weight, up_weight)
        yield [
            *nvfp4_quantized_entries(
                gate_name,
                gate_output,
                include_input_scale=False,
            ),
            *nvfp4_quantized_entries(
                up_name,
                up_output,
                include_input_scale=False,
            ),
        ]
        del pending_pairs[pair_base]

    if pending_pairs:
        incomplete = {
            base: sorted(roles) for base, roles in sorted(pending_pairs.items())
        }
        raise ValueError(
            "NVFP4 gate/up weights must be quantized together; incomplete "
            f"pairs: {incomplete}."
        )
