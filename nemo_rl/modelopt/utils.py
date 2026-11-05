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

"""Lightweight ModelOpt helpers shared by Megatron and vLLM workers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase
from typing import Any, Iterator, Literal

MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS = 600_000

_QUANT_IGNORE_NAME_SUFFIXES = (
    ".weight",
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
)

# Layers kept in native dtype by the real-quant vLLM rollout. Shared between the
# vLLM quantization_config and the Megatron export-side ignore patterns.
DEFAULT_NVFP4_IGNORE = [
    "lm_head",
    "*output_layer*",
    "*mlp.gate",
    "*router*",
    "*block_sparse_moe.gate*",
    "*self_attention*",
    "*self_attn*",
]

NVFP4RealQuantMode = Literal["w4a4", "w4a16"]
_NVFP4_REAL_QUANT_MODES = frozenset({"w4a4", "w4a16"})


def _iter_quant_ignore_suffix_variants(name: str) -> Iterator[str]:
    """Yield ``name`` and, if it ends in a known quant suffix, the stripped form."""
    yield name
    for suffix in _QUANT_IGNORE_NAME_SUFFIXES:
        if name.endswith(suffix):
            yield name[: -len(suffix)]
            break


def iter_quant_ignore_name_candidates(name: str) -> Iterator[str]:
    """Yield name variants matched by ModelOpt real-quant ignore patterns."""
    yield from _iter_quant_ignore_suffix_variants(name)

    alternate = (
        name.removeprefix("model.") if name.startswith("model.") else f"model.{name}"
    )
    if alternate == name:
        return

    yield from _iter_quant_ignore_suffix_variants(alternate)


def matches_quant_ignore_pattern(name: str, patterns: list[str]) -> bool:
    """Return whether ``name`` matches any ModelOpt real-quant ignore pattern."""
    return any(
        fnmatchcase(candidate, pattern)
        for candidate in iter_quant_ignore_name_candidates(name)
        for pattern in patterns
    )


def build_vllm_modelopt_nvfp4_config(
    *,
    mode: NVFP4RealQuantMode,
    ignore: list[str] | None = None,
) -> dict[str, Any]:
    """Build the HuggingFace quantization_config consumed by vLLM ModelOpt NVFP4.

    NeMo-RL's ``quant_cfg`` recipes are ModelOpt PTQ/QAT configs consumed by
    ``mtq.quantize``. vLLM expects the deployment/export-side
    ``quantization_config`` shape instead.
    """
    from modelopt.torch.export.convert_hf_config import (
        convert_hf_quant_config_format,
    )

    if mode not in _NVFP4_REAL_QUANT_MODES:
        raise ValueError(
            f"Unsupported NVFP4 real-quant mode {mode!r}; expected 'w4a4' or 'w4a16'."
        )
    return convert_hf_quant_config_format(
        {
            "producer": {"name": "modelopt"},
            "quantization": {
                "quant_algo": "NVFP4" if mode == "w4a4" else "W4A16_NVFP4",
                "group_size": 16,
                "exclude_modules": (
                    ignore if ignore is not None else list(DEFAULT_NVFP4_IGNORE)
                ),
            },
        }
    )


def _resolve_effective_quantizer_formats(
    quant_cfg: Sequence[Mapping[str, Any]],
    *,
    source: str,
) -> tuple[list[object], list[object]]:
    """Resolve enabled weight and input formats from ordered ModelOpt entries."""
    states: dict[str, dict[str, tuple[bool, object | None]]] = {
        "weight_quantizer": {},
        "input_quantizer": {},
    }

    def _updated_state(
        current: tuple[bool, object | None],
        entry: Mapping[str, Any],
    ) -> tuple[bool, object | None]:
        enabled, format_cfg = current
        if entry.get("cfg") is not None:
            format_cfg = entry["cfg"]
        enabled = entry["enable"]
        return enabled, format_cfg

    for entry in quant_cfg:
        pattern = entry["quantizer_name"]

        # Parent-scoped overrides describe exclusions, not a model-wide format.
        if entry.get("parent_class") is not None:
            continue

        if pattern == "*":
            if entry.get("cfg") is not None:
                raise ValueError(
                    f"Real quantization for {source!r} cannot infer weight and "
                    "activation formats from an enabled catch-all quantizer entry."
                )
            for kind_states in states.values():
                for existing_pattern, current in tuple(kind_states.items()):
                    kind_states[existing_pattern] = _updated_state(
                        current,
                        entry,
                    )
            continue

        matching_kinds = [kind for kind in states if kind in pattern]
        if not matching_kinds:
            continue
        if len(matching_kinds) != 1:
            raise ValueError(
                f"Quantization config {source!r} has ambiguous quantizer pattern "
                f"{pattern!r}."
            )

        kind = matching_kinds[0]
        kind_states = states[kind]
        if pattern == f"*{kind}":
            # A generic selector overrides every previously described subset.
            for existing_pattern, current in tuple(kind_states.items()):
                kind_states[existing_pattern] = _updated_state(
                    current,
                    entry,
                )
        kind_states[pattern] = _updated_state(
            kind_states.get(pattern, (False, None)),
            entry,
        )

    weight_formats = [
        format_cfg
        for enabled, format_cfg in states["weight_quantizer"].values()
        if enabled
    ]
    input_formats = [
        format_cfg
        for enabled, format_cfg in states["input_quantizer"].values()
        if enabled
    ]
    return weight_formats, input_formats


def _is_float_format(value: object, exponent_bits: int, mantissa_bits: int) -> bool:
    """Return whether a ModelOpt numeric-format value names the requested float."""
    if isinstance(value, str):
        return value.lower() == f"e{exponent_bits}m{mantissa_bits}"
    return value == (exponent_bits, mantissa_bits) or value == [
        exponent_bits,
        mantissa_bits,
    ]


def _validate_nvfp4_quantizer_format(
    format_cfg: object,
    *,
    quantizer_name: str,
    source: str,
) -> None:
    """Validate the block-16 E2M1 format supported by vLLM ModelOpt NVFP4."""
    if not isinstance(format_cfg, Mapping):
        raise ValueError(
            f"Real quantization for {source!r} requires a single NVFP4 "
            f"{quantizer_name} format; got {format_cfg!r}."
        )

    block_sizes = format_cfg.get("block_sizes")
    block_size = None
    block_type = None
    scale_bits = None
    if isinstance(block_sizes, Mapping):
        block_size = block_sizes.get(-1, block_sizes.get("-1"))
        block_type = block_sizes.get("type")
        scale_bits = block_sizes.get("scale_bits")

    if not (
        _is_float_format(format_cfg.get("num_bits"), 2, 1)
        and block_size == 16
        and block_type == "dynamic"
        and _is_float_format(scale_bits, 4, 3)
    ):
        raise ValueError(
            f"Real quantization for {source!r} supports only block-16 NVFP4 "
            f"(E2M1 with E4M3 dynamic scales) {quantizer_name}; got "
            f"{dict(format_cfg)!r}."
        )


def resolve_nvfp4_real_quant_mode(quant_cfg: str) -> NVFP4RealQuantMode:
    """Resolve a ModelOpt training config to its supported real-quant mode.

    The mode is derived from all effective weight and input quantizer entries,
    including model-specific selectors, rather than from a config name. NVFP4
    weights with no enabled input quantizer resolve to W4A16; block-16 NVFP4
    input quantization resolves to W4A4. FP8, W4A8, sequential, mixed, and other
    activation formats fail loudly because vLLM would otherwise run a kernel
    with incompatible semantics.
    """
    if not isinstance(quant_cfg, str) or not quant_cfg:
        raise ValueError("NVFP4 real quantization requires a non-empty quant_cfg.")

    from modelopt.torch.quantization.config import QuantizeConfig

    resolved = resolve_quant_cfg(quant_cfg)
    normalized = QuantizeConfig(**resolved)
    entries = [
        entry.model_dump(mode="python", exclude_none=True)
        for entry in normalized.quant_cfg
    ]

    weight_formats, input_formats = _resolve_effective_quantizer_formats(
        entries, source=quant_cfg
    )
    if not weight_formats:
        raise ValueError(
            f"Real quantization for {quant_cfg!r} requires enabled NVFP4 weights."
        )
    for weight_format in weight_formats:
        _validate_nvfp4_quantizer_format(
            weight_format,
            quantizer_name="weights",
            source=quant_cfg,
        )

    if not input_formats:
        return "w4a16"

    for input_format in input_formats:
        _validate_nvfp4_quantizer_format(
            input_format,
            quantizer_name="input activations",
            source=quant_cfg,
        )
    return "w4a4"


def resolve_quant_cfg(quant_cfg: str) -> dict[str, Any]:
    """Resolve a quantization config string into a dict consumable by ``mtq.quantize``.

    Resolution order:

    1. Built-in ModelOpt config constant exposed on ``modelopt.torch.quantization``
       (e.g. ``"NVFP4_DEFAULT_CFG"``, ``"FP8_DEFAULT_CFG"``).
    2. A ModelOpt PTQ recipe — either the name of a built-in recipe shipped under
       ``modelopt_recipes/`` (e.g. ``"general/ptq/nvfp4_default-fp8_kv"``; the
       ``.yml`` / ``.yaml`` suffix is optional) or the path to a user-authored
       YAML recipe. Resolution is performed by ``modelopt.recipe.load_config``,
       which searches the filesystem first and then the built-in recipe library.
       For Ray/container workers, use an absolute path for user-authored recipe
       files; NeMo-RL repo-relative recipe paths are not resolved here.

    YAML recipes are expected to follow the standard ModelOpt PTQ recipe layout
    with a top-level ``quantize:`` section in the ``{"quant_cfg": [...],
    "algorithm": ...}`` shape that ``mtq.quantize`` expects. A bare
    ``{"quant_cfg": [...], "algorithm": ...}`` document (without a wrapping
    ``quantize:`` key) is also accepted for convenience. If ``algorithm`` is
    omitted, it defaults to ``"max"`` so ModelOpt's calibration helpers see the
    same normalized config as ``mtq.quantize``. The extracted dict — not the full
    recipe — is returned.

    See ``modelopt_recipes/general/ptq/`` in the NVIDIA/Model-Optimizer repo
    (https://github.com/NVIDIA/Model-Optimizer) for the canonical format and
    ``examples/modelopt/quant_configs/`` for a user-authored example.
    """
    import modelopt.torch.quantization as mtq
    from modelopt.recipe import load_config

    def _normalize_mtq_cfg(config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(config, dict):
            raise ValueError(
                f"Quantization recipe '{quant_cfg}' must resolve to a dict."
            )
        mtq_cfg = config.get("quantize", config)
        if not isinstance(mtq_cfg, dict) or "quant_cfg" not in mtq_cfg:
            raise ValueError(
                f"Quantization recipe '{quant_cfg}' must contain a 'quant_cfg' "
                f"entry (optionally nested under a top-level 'quantize:' section)."
            )
        if "algorithm" not in mtq_cfg:
            mtq_cfg = {**mtq_cfg, "algorithm": "max"}
        return mtq_cfg

    builtin = getattr(mtq, quant_cfg, None)
    if builtin is not None:
        return _normalize_mtq_cfg(builtin)

    try:
        loaded = load_config(quant_cfg)
    except (ValueError, FileNotFoundError) as e:
        raise ValueError(
            f"Unknown quant_cfg '{quant_cfg}'. Must be either a built-in "
            f"ModelOpt config name (e.g. 'NVFP4_DEFAULT_CFG'), a built-in "
            f"ModelOpt PTQ recipe name (e.g. "
            f"'general/ptq/nvfp4_default-fp8_kv'), or an absolute path to a "
            f"YAML quantization recipe."
        ) from e

    return _normalize_mtq_cfg(loaded)
