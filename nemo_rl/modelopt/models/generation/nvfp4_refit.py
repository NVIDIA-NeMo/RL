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

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

import torch

NVFP4RefitMode = Literal["w4a16", "w4a4"]
NVFP4RoutedExpertFamily = Literal["gated", "non_gated"]

_BLOCK_SIZE = 16
_NVFP4_AMAX_DENOMINATOR = 6.0 * 448.0
_NVFP4_MAXBOUND = 6.0
_FP8_E4M3_MIN = 2**-9
_FP8_E4M3_MAX = 448.0
_EXPERT_PROJECTION = re.compile(
    r"^(?P<family_prefix>.+\.experts)\.(?P<expert>\d+)\."
    r"(?P<projection>gate|up|down)_proj\.weight$"
)
_ROUTED_EXPERT_PROJECTIONS: dict[NVFP4RoutedExpertFamily, frozenset[str]] = {
    "gated": frozenset({"gate", "up", "down"}),
    "non_gated": frozenset({"up", "down"}),
}


@dataclass(frozen=True)
class _QuantMeta:
    qformat: str
    block_size: int
    weight_amax: torch.Tensor
    input_amax: torch.Tensor | None


@dataclass(frozen=True)
class _Nvfp4InputQuantizerView:
    input_amax: torch.Tensor
    is_enabled: bool = True
    maxbound: float = _NVFP4_MAXBOUND

    def export_amax(self) -> torch.Tensor:
        return self.input_amax


_NVFP4Exporter = Callable[
    [str, torch.Tensor, _QuantMeta], Iterable[tuple[str, torch.Tensor]]
]


@dataclass(frozen=True)
class NVFP4Calibration:
    """Static input-activation amax values for W4A4 refit."""

    input_amax: Mapping[str, torch.Tensor]


def compute_nvfp4_input_scale(input_amax: torch.Tensor | None) -> torch.Tensor:
    """Compute the canonical static NVFP4 activation scale with ModelOpt."""
    if input_amax is None:
        raise RuntimeError("Missing ModelOpt input amax for NVFP4 W4A4 export")

    input_amax = input_amax.detach().float()
    if (
        input_amax.numel() == 0
        or not torch.isfinite(input_amax).all()
        or not torch.all(input_amax > 0)
    ):
        raise RuntimeError(
            f"Invalid ModelOpt input amax for NVFP4 W4A4 export: {input_amax}"
        )

    # ModelOpt is an optional runtime dependency loaded only for NVFP4 refit.
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import (  # pyrefly: ignore[import-error]
        NVFP4QTensor,
    )

    canonical_export = getattr(NVFP4QTensor, "get_activation_scaling_factor", None)
    if callable(canonical_export):
        input_scale = cast(
            torch.Tensor | None,
            canonical_export(_Nvfp4InputQuantizerView(input_amax)),
        )
    else:
        input_scale = input_amax / _NVFP4_AMAX_DENOMINATOR
    if (
        input_scale is None
        or input_scale.numel() == 0
        or not torch.isfinite(input_scale).all()
        or not torch.all(input_scale > 0)
    ):
        raise RuntimeError(
            f"Invalid ModelOpt input scale for NVFP4 W4A4 export: {input_scale}"
        )
    return input_scale.detach().float()


def get_modelopt_quant_exporter(quant_mode: str) -> tuple[str, object]:
    """Return the ModelOpt NVFP4 format and dependency-light exporter."""
    # ModelOpt is an optional runtime dependency loaded only for NVFP4 refit.
    from modelopt.torch.export import quant_utils  # pyrefly: ignore[import-error]

    normalized_mode = quant_mode.lower()
    if normalized_mode == "nvfp4":
        qformat = quant_utils.QUANTIZATION_NVFP4
    elif normalized_mode == "w4a16_nvfp4":
        qformat = getattr(quant_utils, "QUANTIZATION_W4A16_NVFP4", None)
        if qformat is None:
            raise RuntimeError(
                "The installed nvidia-modelopt version does not support W4A16 "
                "NVFP4 export; install a version that exposes "
                "QUANTIZATION_W4A16_NVFP4."
            )
    else:
        raise ValueError(f"Unsupported ModelOpt quant_mode: {quant_mode}")
    return qformat, _quantize_nvfp4_weight


def nvfp4_routed_expert_projections(
    projection_family: NVFP4RoutedExpertFamily,
) -> frozenset[str]:
    """Return the complete projection set for one routed-expert family."""
    try:
        return _ROUTED_EXPERT_PROJECTIONS[projection_family]
    except KeyError as error:
        raise ValueError(
            f"Unsupported NVFP4 routed-expert family: {projection_family!r}"
        ) from error


def resolve_nvfp4_routed_expert_families(
    names: Iterable[str],
) -> dict[str, str]:
    """Resolve and validate gated or non-gated experts for each MoE prefix."""
    projections_by_expert = _routed_expert_projections_by_expert(names)

    family_by_prefix: dict[str, str] = {}
    for (prefix, expert), projections in sorted(projections_by_expert.items()):
        if projections == _ROUTED_EXPERT_PROJECTIONS["gated"]:
            family: NVFP4RoutedExpertFamily = "gated"
        elif projections == _ROUTED_EXPERT_PROJECTIONS["non_gated"]:
            family = "non_gated"
        else:
            raise ValueError(
                "BF16-to-NVFP4 NCCL refit found an incomplete routed-expert "
                f"family for {prefix}.{expert}: got {sorted(projections)}"
            )

        existing = family_by_prefix.setdefault(prefix, family)
        if existing != family:
            raise ValueError(
                "BF16-to-NVFP4 NCCL refit mixes gated and non-gated experts "
                f"under {prefix!r}"
            )
    return family_by_prefix


def validate_nvfp4_routed_expert_selection(
    names: Iterable[str],
    *,
    available_names: Iterable[str],
) -> dict[str, str]:
    """Validate that quantization selects complete source expert families."""
    selected_names = frozenset(names)
    available_names = frozenset(available_names)
    unexpected = selected_names.difference(available_names)
    if unexpected:
        raise ValueError(
            f"NVFP4 routed-expert selection contains unknown weights: {sorted(unexpected)}"
        )

    available_families = resolve_nvfp4_routed_expert_families(available_names)
    selected_by_expert = _routed_expert_projections_by_expert(selected_names)
    selected_families: dict[str, str] = {}
    for (prefix, expert), projections in sorted(selected_by_expert.items()):
        family = cast(NVFP4RoutedExpertFamily, available_families[prefix])
        expected = nvfp4_routed_expert_projections(family)
        if projections != expected:
            raise ValueError(
                f"BF16-to-NVFP4 NCCL refit partially selects {family} routed "
                f"expert {prefix}.{expert}: expected {sorted(expected)}, "
                f"got {sorted(projections)}"
            )
        selected_families[prefix] = family
    return selected_families


def _routed_expert_projections_by_expert(
    names: Iterable[str],
) -> dict[tuple[str, int], set[str]]:
    projections_by_expert: dict[tuple[str, int], set[str]] = {}
    for name in names:
        match = _EXPERT_PROJECTION.fullmatch(name)
        if match is None:
            raise ValueError(f"Expected a routed-expert NVFP4 weight, got {name!r}")
        key = (match.group("family_prefix"), int(match.group("expert")))
        projections_by_expert.setdefault(key, set()).add(match.group("projection"))
    return projections_by_expert


def nvfp4_refit_group(
    name: str,
    *,
    projection_family: NVFP4RoutedExpertFamily = "gated",
) -> tuple[str, tuple[str, ...]]:
    """Return a staging key and complete member names for an HF weight.

    The returned ``w13`` and ``w2`` suffixes are completeness keys only. They
    are never emitted as checkpoint names; serialization keeps each original
    per-expert projection name and its separate canonical output family.
    """
    match = _EXPERT_PROJECTION.fullmatch(name)
    if match is None:
        raise ValueError(
            f"BF16 to NVFP4 refit supports routed-expert weights only, got {name!r}"
        )

    family_prefix = match.group("family_prefix")
    prefix = f"{family_prefix}.{match.group('expert')}"
    projection = match.group("projection")
    expected_projections = nvfp4_routed_expert_projections(projection_family)
    if projection not in expected_projections:
        raise ValueError(
            f"Projection {projection!r} does not belong to {projection_family!r} "
            f"NVFP4 routed experts"
        )
    if projection == "down":
        return f"{prefix}.w2", (name,)

    up_name = f"{prefix}.up_proj.weight"
    if projection_family == "non_gated":
        return f"{prefix}.w13", (up_name,)
    gate_name = f"{prefix}.gate_proj.weight"
    return f"{prefix}.w13", (gate_name, up_name)


def serialize_bf16_nvfp4_group(
    tensors: Mapping[str, torch.Tensor],
    *,
    mode: NVFP4RefitMode,
    calibration: NVFP4Calibration | None,
    projection_family: NVFP4RoutedExpertFamily = "gated",
) -> list[tuple[str, torch.Tensor]]:
    """Serialize one complete BF16 NVFP4 refit group with ModelOpt.

    Args:
        tensors: Logical BF16 HF weights belonging to one refit group.
        mode: ``w4a16`` for weight-only NVFP4 or ``w4a4`` for calibrated NVFP4.
        calibration: Named input amax values required by W4A4.
        projection_family: Gated ``gate/up/down`` or non-gated ``up/down`` MoE.

    Returns:
        ModelOpt checkpoint-layout tensors in exporter order.

    Raises:
        ValueError: If the mode, group, shape, amax, or calibration is invalid.
    """
    if mode not in {"w4a16", "w4a4"}:
        raise ValueError(f"Unsupported NVFP4 refit mode: {mode}")

    eligible_tensors = {
        name: tensor for name, tensor in tensors.items() if name.endswith(".weight")
    }
    if not eligible_tensors:
        return []

    _, resolved_names = _validate_group_members(
        eligible_tensors,
        projection_family=projection_family,
    )
    weights = [eligible_tensors[name] for name in resolved_names]
    _validate_weight_shapes(resolved_names, weights)
    shared_amax = _shared_weight_amax(resolved_names, weights)
    input_amaxes = {
        name: _input_amax_for_weight(name, mode, calibration) for name in resolved_names
    }

    quant_mode = "w4a16_nvfp4" if mode == "w4a16" else "nvfp4"
    qformat, exporter = get_modelopt_quant_exporter(quant_mode)
    canonical_exporter = _as_nvfp4_exporter(exporter)

    serialized: list[tuple[str, torch.Tensor]] = []
    for name, weight in zip(resolved_names, weights, strict=True):
        meta = _load_quant_meta()(
            qformat=qformat,
            block_size=_BLOCK_SIZE,
            weight_amax=shared_amax,
            input_amax=input_amaxes[name],
        )
        serialized.extend(canonical_exporter(name, weight, meta))
    return serialized


def _load_quant_meta() -> type[_QuantMeta]:
    return _QuantMeta


def _as_nvfp4_exporter(exporter: object) -> _NVFP4Exporter:
    if not callable(exporter):
        raise TypeError("ModelOpt returned a non-callable NVFP4 exporter")
    return cast(_NVFP4Exporter, exporter)


def _compute_nvfp4_weight_scale(
    weight: torch.Tensor,
    meta: _QuantMeta,
) -> tuple[torch.Tensor, torch.Tensor]:
    # ModelOpt is an optional runtime dependency loaded only for NVFP4 refit.
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import (  # pyrefly: ignore[import-error]
        NVFP4QTensor,
    )

    weight_scale_2 = (
        meta.weight_amax.to(weight.device).float().abs() / _NVFP4_AMAX_DENOMINATOR
    )
    weight_scale = NVFP4QTensor.get_weights_scaling_factor(
        weight,
        meta.block_size,
        weights_scaling_factor_2=weight_scale_2.reshape(()),
        keep_high_precision=False,
    )[0]
    weight_scale_float = weight_scale.float()
    if not torch.isfinite(weight_scale_float).all():
        raise RuntimeError(
            f"Invalid ModelOpt NVFP4 per-block weight scale: {weight_scale_float}"
        )
    weight_scale = (
        weight_scale_float.abs()
        .clamp(min=_FP8_E4M3_MIN, max=_FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
    )
    return weight_scale, weight_scale_2


def _quantize_nvfp4_weight(
    name: str,
    weight: torch.Tensor,
    meta: _QuantMeta,
) -> Iterable[tuple[str, torch.Tensor]]:
    # ModelOpt is an optional runtime dependency loaded only for NVFP4 refit.
    from modelopt.torch.export.quant_utils import (  # pyrefly: ignore[import-error]
        QUANTIZATION_NVFP4,
        to_quantized_weight,
    )

    if not name.endswith(".weight"):
        raise ValueError(f"Expected an NVFP4 HF weight name, got {name!r}")
    base_name = name.removesuffix(".weight")
    weight_scale, weight_scale_2 = _compute_nvfp4_weight_scale(weight, meta)
    quantized = to_quantized_weight(
        weight,
        weight_scale,
        meta.qformat,
        weight_scale_2.reshape(()),
        meta.block_size,
    )

    yield name, quantized.detach()
    yield f"{base_name}.weight_scale", weight_scale.detach()
    yield f"{base_name}.weight_scale_2", weight_scale_2.detach()
    if meta.qformat == QUANTIZATION_NVFP4:
        yield (
            f"{base_name}.input_scale",
            compute_nvfp4_input_scale(meta.input_amax).to(weight.device),
        )


def _validate_group_members(
    tensors: Mapping[str, torch.Tensor],
    *,
    projection_family: NVFP4RoutedExpertFamily,
) -> tuple[str, tuple[str, ...]]:
    groups = {
        nvfp4_refit_group(name, projection_family=projection_family)[0]
        for name in tensors
    }
    if len(groups) != 1:
        raise ValueError(f"Expected one complete NVFP4 group, got {sorted(groups)}")

    group_name = next(iter(groups))
    resolved_names = nvfp4_refit_group(
        next(iter(tensors)),
        projection_family=projection_family,
    )[1]
    if not resolved_names:
        raise ValueError(f"NVFP4 group {group_name} has no expected members")
    expected_groups = {
        nvfp4_refit_group(name, projection_family=projection_family)[0]
        for name in resolved_names
    }
    if expected_groups != {group_name}:
        raise ValueError(
            f"NVFP4 group {group_name} has members from different groups: "
            f"{sorted(expected_groups)}"
        )
    if set(tensors) != set(resolved_names):
        missing = sorted(set(resolved_names).difference(tensors))
        extra = sorted(set(tensors).difference(resolved_names))
        detail = f"missing {missing}" if missing else f"unexpected {extra}"
        raise ValueError(f"NVFP4 group {group_name} is not complete: {detail}")
    return group_name, resolved_names


def _validate_weight_shapes(
    names: tuple[str, ...], weights: list[torch.Tensor]
) -> None:
    for name, weight in zip(names, weights, strict=True):
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise ValueError(f"NVFP4 weight must be 2-D for {name}")
        if weight.shape[-1] % _BLOCK_SIZE != 0:
            raise ValueError(
                f"NVFP4 weight K dimension must be divisible by {_BLOCK_SIZE} for {name}; "
                f"got {tuple(weight.shape)}"
            )
    if len(weights) == 2 and weights[0].shape[-1] != weights[1].shape[-1]:
        raise ValueError(
            "NVFP4 gate/up group must use the same K dimension; "
            f"got {tuple(weights[0].shape)} and {tuple(weights[1].shape)}"
        )


def _shared_weight_amax(
    names: tuple[str, ...], weights: list[torch.Tensor]
) -> torch.Tensor:
    maxima = []
    for name, weight in zip(names, weights, strict=True):
        if weight.numel() == 0:
            raise ValueError(f"NVFP4 weight must be non-empty for {name}")
        maxima.append(weight.detach().float().abs().amax())
    shared_amax = torch.stack(maxima).amax().reshape(())
    if not torch.isfinite(shared_amax) or not torch.all(shared_amax > 0):
        raise ValueError(
            f"NVFP4 shared weight amax must be finite and positive: {shared_amax}"
        )
    return shared_amax


def _input_amax_for_weight(
    name: str,
    mode: NVFP4RefitMode,
    calibration: NVFP4Calibration | None,
) -> torch.Tensor | None:
    if mode == "w4a16":
        return None
    if calibration is None or name not in calibration.input_amax:
        raise ValueError(f"Missing input amax for NVFP4 W4A4 weight {name}")
    input_amax = calibration.input_amax[name]
    if not isinstance(input_amax, torch.Tensor):
        raise ValueError(
            f"Invalid input amax for NVFP4 W4A4 weight {name}: {input_amax}"
        )
    input_amax_float = input_amax.detach().float()
    if (
        input_amax_float.numel() != 1
        or not torch.isfinite(input_amax_float).all()
        or not torch.all(input_amax_float > 0)
    ):
        raise ValueError(
            f"Invalid scalar input amax for NVFP4 W4A4 weight {name}: {input_amax}"
        )
    try:
        compute_nvfp4_input_scale(input_amax)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid input amax for NVFP4 W4A4 weight {name}") from error
    return input_amax
