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
from pathlib import Path
from typing import Any, Iterator

_QUANT_IGNORE_NAME_SUFFIXES = (
    ".weight",
    ".weight_scale",
    ".weight_scale_2",
)


def iter_quant_ignore_name_candidates(name: str) -> Iterator[str]:
    """Yield name variants matched by ModelOpt real-quant ignore patterns."""
    yield name
    for suffix in _QUANT_IGNORE_NAME_SUFFIXES:
        if name.endswith(suffix):
            yield name[: -len(suffix)]
            break

    alternate = (
        name.removeprefix("model.") if name.startswith("model.") else f"model.{name}"
    )
    if alternate == name:
        return

    yield alternate
    for suffix in _QUANT_IGNORE_NAME_SUFFIXES:
        if alternate.endswith(suffix):
            yield alternate[: -len(suffix)]
            break


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
    default_ignore = [
        "lm_head",
        "*output_layer*",
        "*mlp.gate",
        "*router*",
        "*block_sparse_moe.gate*",
        "*self_attention*",
        "*self_attn*",
    ]

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
        "ignore": ignore if ignore is not None else default_ignore,
        "quant_algo": "NVFP4",
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

    YAML recipes are expected to follow the standard ModelOpt PTQ recipe layout
    with a top-level ``quantize:`` section in the
    ``{"quant_cfg": [...], "algorithm": ...}`` shape that ``mtq.quantize``
    expects. A bare ``{"quant_cfg": [...], "algorithm": ...}`` document (without
    a wrapping ``quantize:`` key) is also accepted for convenience. The
    extracted dict — not the full recipe — is returned.

    See ``modelopt_recipes/general/ptq/`` in the TensorRT-Model-Optimizer repo
    for the canonical format and ``examples/modelopt/quant_configs/`` for a
    user-authored example.
    """
    import modelopt.torch.quantization as mtq
    from modelopt.recipe import load_config

    builtin = getattr(mtq, quant_cfg, None)
    if builtin is not None:
        return builtin

    config_path = Path(quant_cfg)
    if not config_path.is_absolute():
        repo_config_path = Path(__file__).resolve().parents[2] / config_path
        if repo_config_path.is_file():
            quant_cfg = str(repo_config_path)

    try:
        loaded = load_config(quant_cfg)
    except (ValueError, FileNotFoundError) as e:
        raise ValueError(
            f"Unknown quant_cfg '{quant_cfg}'. Must be either a built-in "
            f"ModelOpt config name (e.g. 'NVFP4_DEFAULT_CFG'), a built-in "
            f"ModelOpt PTQ recipe name (e.g. "
            f"'general/ptq/nvfp4_default-fp8_kv'), or a path to a YAML "
            f"quantization recipe."
        ) from e

    quantize = loaded.get("quantize", loaded)
    if not isinstance(quantize, dict) or "quant_cfg" not in quantize:
        raise ValueError(
            f"Quantization recipe '{quant_cfg}' must contain a 'quant_cfg' "
            f"entry (optionally nested under a top-level 'quantize:' section)."
        )
    return quantize
