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

import json
from pathlib import Path
from typing import Any

import pytest

from nemo_rl.models.generation.sglang.mxfp8_setup import (
    _hash_qualified_save_dir,
    ensure_mxfp8_checkpoint,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_checkpoint_metadata(
    checkpoint: Path,
    *,
    modules_to_not_convert: list[str],
    weight_names: list[str],
) -> None:
    checkpoint.mkdir(parents=True)
    _write_json(
        checkpoint / "config.json",
        {
            "num_hidden_layers": 2,
            "quantization_config": {
                "quant_method": "mxfp8",
                "weight_block_size": [1, 32],
                "scale_fmt": "ue8m0",
                "modules_to_not_convert": modules_to_not_convert,
            },
        },
    )
    _write_json(
        checkpoint / "model.safetensors.index.json",
        {"weight_map": {name: "model.safetensors" for name in weight_names}},
    )


def test_existing_mxfp8_checkpoint_preserves_supplied_qkv_skip(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "mxfp8"
    q = "model.layers.0.self_attn.q_proj"
    k = "model.layers.0.self_attn.k_proj"
    v = "model.layers.0.self_attn.v_proj"
    _write_checkpoint_metadata(
        checkpoint,
        modules_to_not_convert=[q, k, v],
        weight_names=[f"{q}.weight", f"{k}.weight", f"{v}.weight"],
    )
    config: dict[str, Any] = {
        "scheme": "mxfp8",
        "extra_high_precision_layers_hf": [q],
    }

    result = ensure_mxfp8_checkpoint(
        model_path=str(checkpoint),
        quantization_cfg=config,
    )

    assert result == str(checkpoint)
    assert config["modules_to_not_convert"] == [q, k, v]


def test_existing_mxfp8_checkpoint_rejects_new_partial_qkv_skip(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "mxfp8"
    q = "model.layers.0.self_attn.q_proj"
    k = "model.layers.0.self_attn.k_proj"
    v = "model.layers.0.self_attn.v_proj"
    weights = [
        f"{q}.weight",
        f"{q}.weight_scale_inv",
        f"{k}.weight",
        f"{k}.weight_scale_inv",
        f"{v}.weight",
        f"{v}.weight_scale_inv",
    ]
    _write_checkpoint_metadata(
        checkpoint,
        modules_to_not_convert=[],
        weight_names=weights,
    )
    config: dict[str, Any] = {
        "scheme": "mxfp8",
        "extra_high_precision_layers_hf": [q],
    }

    with pytest.raises(ValueError, match="Reconvert the original HF checkpoint"):
        ensure_mxfp8_checkpoint(
            model_path=str(checkpoint),
            quantization_cfg=config,
        )


def test_existing_mxfp8_checkpoint_rejects_missing_quantized_scale(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "mxfp8"
    weight = "model.layers.0.self_attn.q_proj.weight"
    _write_checkpoint_metadata(
        checkpoint,
        modules_to_not_convert=[],
        weight_names=[weight],
    )

    with pytest.raises(ValueError, match="no companion scale"):
        ensure_mxfp8_checkpoint(
            model_path=str(checkpoint),
            quantization_cfg={"scheme": "mxfp8"},
        )


def test_explicit_preconverted_checkpoint_supports_hf_source_id(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "mxfp8"
    module = "model.layers.0.self_attn.q_proj"
    _write_checkpoint_metadata(
        checkpoint,
        modules_to_not_convert=[],
        weight_names=[f"{module}.weight", f"{module}.weight_scale_inv"],
    )
    config: dict[str, Any] = {
        "scheme": "mxfp8",
        "converted_model_path": str(checkpoint),
    }

    result = ensure_mxfp8_checkpoint(
        model_path="NVIDIA/model-on-hf",
        quantization_cfg=config,
    )

    assert result == str(checkpoint.resolve())


def test_hf_source_id_requires_explicit_preconverted_checkpoint() -> None:
    with pytest.raises(ValueError, match="cannot auto-convert an HF repo id"):
        ensure_mxfp8_checkpoint(
            model_path="NVIDIA/model-on-hf",
            quantization_cfg={"scheme": "mxfp8"},
        )


def test_cache_path_changes_with_skip_policy_and_is_stable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_json(source / "config.json", {"num_hidden_layers": 2})
    cache_root = tmp_path / "cache"
    first_policy = {
        "scheme": "mxfp8",
        "extra_high_precision_layers_hf": ["model.layers.0.mlp"],
    }
    second_policy = {
        "scheme": "mxfp8",
        "extra_high_precision_layers_hf": ["model.layers.1.mlp"],
    }

    first = _hash_qualified_save_dir(
        model_dir=str(source),
        cache_root=str(cache_root),
        quantization_cfg=first_policy,
    )
    first_again = _hash_qualified_save_dir(
        model_dir=str(source),
        cache_root=str(cache_root),
        quantization_cfg=first_policy,
    )
    second = _hash_qualified_save_dir(
        model_dir=str(source),
        cache_root=str(cache_root),
        quantization_cfg=second_policy,
    )

    assert first == first_again
    assert first != second


def test_identical_policy_reuses_preconverted_cache(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_json(source / "config.json", {"num_hidden_layers": 2})
    cache_root = tmp_path / "cache"
    config: dict[str, Any] = {
        "scheme": "mxfp8",
        "cache_root": str(cache_root),
    }
    checkpoint = Path(
        _hash_qualified_save_dir(
            model_dir=str(source),
            cache_root=str(cache_root),
            quantization_cfg=config,
        )
    )
    module = "model.layers.0.self_attn.q_proj"
    _write_checkpoint_metadata(
        checkpoint,
        modules_to_not_convert=[],
        weight_names=[f"{module}.weight", f"{module}.weight_scale_inv"],
    )

    first = ensure_mxfp8_checkpoint(
        model_path=str(source),
        quantization_cfg=dict(config),
    )
    second = ensure_mxfp8_checkpoint(
        model_path=str(source),
        quantization_cfg=dict(config),
    )

    assert first == second == str(checkpoint)
