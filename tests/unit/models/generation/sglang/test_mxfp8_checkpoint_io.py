# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Checkpoint integrity at the offline-conversion/online-refit boundary."""

import json
from pathlib import Path

import pytest
import safetensors.torch
import torch

from nemo_rl.models.generation.sglang import mxfp8_setup
from nemo_rl.models.generation.sglang.mxfp8_quantization_core import (
    is_bf16_source_checkpoint,
    quantize_mxfp8,
)


@pytest.mark.parametrize(
    ("quantization", "expected"),
    [
        (None, True),
        ({}, True),
        ({"quant_method": "bf16"}, True),
        ({"quant_method": "mxfp8"}, False),
        ({"quant_method": "nvfp4"}, False),
    ],
)
def test_conversion_accepts_only_high_precision_sources(quantization, expected):
    assert is_bf16_source_checkpoint({"quantization_config": quantization}) is expected


def test_checkpoint_without_index_reads_real_safetensors_headers(tmp_path: Path):
    weights = {"model.layers.0.mlp.down_proj.weight": torch.ones((2, 32))}
    safetensors.torch.save_file(weights, tmp_path / "model.safetensors")

    assert mxfp8_setup._checkpoint_weight_names(str(tmp_path)) == tuple(weights)


def test_duplicate_tensor_across_shards_is_rejected(tmp_path: Path):
    weights = {"model.norm.weight": torch.ones(8)}
    for shard in ("part1.safetensors", "part2.safetensors"):
        safetensors.torch.save_file(weights, tmp_path / shard)

    with pytest.raises(ValueError, match="Duplicate tensor"):
        mxfp8_setup._checkpoint_weight_names(str(tmp_path))


@pytest.mark.parametrize(
    "index", [{}, {"weight_map": {}}, {"weight_map": {"weight": 7}}]
)
def test_corrupt_checkpoint_index_is_rejected(tmp_path: Path, index):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="weight_map"):
        mxfp8_setup._checkpoint_weight_names(str(tmp_path))


def test_checkpoint_without_weight_files_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="No safetensors"):
        mxfp8_setup._checkpoint_weight_names(str(tmp_path))


@pytest.mark.parametrize("layers", [None, True, 0, -1, "2"])
def test_invalid_layer_count_is_rejected(layers):
    with pytest.raises(ValueError, match="positive integer num_hidden_layers"):
        mxfp8_setup._get_num_hidden_layers(
            {"text_config": {"num_hidden_layers": layers}}, config_path="config.json"
        )


def test_nested_text_config_supplies_layer_count():
    assert (
        mxfp8_setup._get_num_hidden_layers(
            {"text_config": {"num_hidden_layers": 3}}, config_path="config.json"
        )
        == 3
    )


def test_file_conversion_preserves_unquantized_tensors_and_index_sizes(tmp_path: Path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    weights = {
        "model.embed_tokens.weight": torch.ones((4, 32), dtype=torch.bfloat16),
        "model.layers.0.mlp.gate.weight": torch.arange(64).reshape(2, 32).float(),
        "model.layers.0.mlp.experts.0.up_proj.weight": torch.ones((2, 48)),
        "model.norm.weight": torch.ones(32),
        "model.layers.0.self_attn.q_proj.bias": torch.arange(32).float(),
    }
    safetensors.torch.save_file(weights, source / "model.safetensors")
    result = mxfp8_setup._ConversionResult()

    mxfp8_setup._process_file(
        str(source),
        str(output),
        "model.safetensors",
        result_collector=result,
        device="cpu",
        skip_weight_substrings=mxfp8_setup.SKIP_WEIGHT_SUBSTRINGS,
    )

    actual = safetensors.torch.load_file(output / "model.safetensors")
    assert actual.keys() == weights.keys()
    for name, tensor in weights.items():
        torch.testing.assert_close(actual[name], tensor, rtol=0, atol=0)
    assert result.weight_map == {name: "model.safetensors" for name in weights}
    assert result.total_size == sum(
        t.numel() * t.element_size() for t in weights.values()
    )
    assert set(result.modules_to_not_convert) == {
        "model.embed_tokens",
        "model.layers.0.mlp.gate",
        "model.norm",
    }


@pytest.mark.parametrize("converted_path", ["", "   ", 7, "missing-directory"])
def test_startup_rejects_invalid_explicit_checkpoint(tmp_path: Path, converted_path):
    with pytest.raises(ValueError, match="converted_model_path"):
        mxfp8_setup.ensure_mxfp8_checkpoint(
            model_path=str(tmp_path),
            quantization_cfg={
                "scheme": "mxfp8",
                "converted_model_path": converted_path,
            },
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires real CUDA quantization"
)
@pytest.mark.parametrize("indexed", [False, True])
def test_offline_checkpoint_matches_online_quantization(tmp_path: Path, indexed: bool):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"num_hidden_layers": 2}))
    (source / "tokenizer_config.json").write_text('{"chat_template": "example"}')
    selected = "model.layers.1.mlp.down_proj.weight"
    preserved = "model.layers.0.mlp.down_proj.weight"
    weights = {
        selected: torch.linspace(-1, 1, 32 * 64).reshape(32, 64).bfloat16(),
        preserved: torch.full((32, 64), 0.5, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(64, dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(weights, source / "model.safetensors")
    if indexed:
        (source / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: "model.safetensors" for name in weights}})
        )

    mxfp8_setup.convert_mxfp8(
        str(source),
        str(output),
        num_layers_at_start_in_bf16=1,
    )

    actual = safetensors.torch.load_file(output / "model.safetensors", device="cuda")
    expected_weight, expected_scale = quantize_mxfp8(weights[selected].cuda())
    scale_key = selected.removesuffix(".weight") + ".weight_scale_inv"
    assert actual.keys() == weights.keys() | {scale_key}
    torch.testing.assert_close(
        actual[selected].float(), expected_weight.float(), rtol=0, atol=0
    )
    assert torch.equal(
        actual[scale_key].view(torch.uint8), expected_scale.view(torch.uint8)
    )
    assert actual[scale_key].shape == (32, 2)
    torch.testing.assert_close(
        actual[preserved].cpu(), weights[preserved], rtol=0, atol=0
    )
    assert (output / "tokenizer_config.json").read_bytes() == (
        source / "tokenizer_config.json"
    ).read_bytes()
    config = json.loads((output / "config.json").read_text())["quantization_config"]
    assert config["scale_fmt"] == "ue8m0"
    assert "layers.0." in config["modules_to_not_convert"]
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(actual)
    assert index["metadata"]["total_size"] == sum(
        t.numel() * t.element_size() for t in actual.values()
    )
