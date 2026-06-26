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

"""Lightweight quantization config resolver usable by both Megatron and vLLM workers."""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any, Iterator

_QUANT_IGNORE_NAME_SUFFIXES = (
    ".weight",
    ".weight_scale",
    ".weight_scale_2",
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
    ignore: list[str] | None = None,
) -> dict[str, Any]:
    """Build the HuggingFace quantization_config consumed by vLLM ModelOpt NVFP4.

    NeMo-RL's ``quant_cfg`` recipes are ModelOpt PTQ/QAT configs consumed by
    ``mtq.quantize``. vLLM expects the deployment/export-side
    ``quantization_config`` shape instead.
    """
    return {
        "quant_method": "modelopt",
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "weights": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "targets": ["Linear"],
            }
        },
        "ignore": ignore if ignore is not None else list(DEFAULT_NVFP4_IGNORE),
        "quant_algo": "NVFP4",
        "quant_mode": "w4a16_nvfp4",
        "weight_only": True,
        "group_size": 16,
        "producer": {"name": "modelopt"},
    }


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
