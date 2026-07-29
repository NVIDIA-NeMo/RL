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
import safetensors.torch
import torch

from nemo_rl.models.generation.sglang import nvfp4_setup


def _gate(layer: int, expert: int = 0) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"


def _up(layer: int, expert: int = 0) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"


def _down(layer: int, expert: int = 0) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"


def _weight(value: float, *, rows: int = 3, columns: int = 32) -> torch.Tensor:
    return torch.full((rows, columns), value, dtype=torch.bfloat16)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_source_checkpoint(
    model_dir: Path,
    shards: dict[str, dict[str, torch.Tensor]],
    *,
    num_hidden_layers: int,
    config_extra: dict[str, Any] | None = None,
    indexed: bool = True,
) -> dict[str, str]:
    model_dir.mkdir()
    config: dict[str, Any] = {"num_hidden_layers": num_hidden_layers}
    if config_extra is not None:
        config.update(config_extra)
    _write_json(model_dir / "config.json", config)

    weight_map: dict[str, str] = {}
    total_size = 0
    for filename, tensors in shards.items():
        safetensors.torch.save_file(
            tensors,
            str(model_dir / filename),
            metadata={"format": "pt"},
        )
        for key, tensor in tensors.items():
            assert key not in weight_map
            weight_map[key] = filename
            total_size += tensor.numel() * tensor.element_size()

    if indexed:
        _write_json(
            model_dir / "model.safetensors.index.json",
            {
                "weight_map": weight_map,
                "metadata": {"total_size": total_size},
            },
        )
    return weight_map


def _fake_quantized(
    weight: torch.Tensor,
    *,
    qvalue: int,
    global_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qweight = torch.full(
        (*weight.shape[:-1], weight.shape[-1] // 2),
        qvalue,
        dtype=torch.uint8,
    )
    block_scale = torch.zeros(
        (*weight.shape[:-1], weight.shape[-1] // 16),
        dtype=torch.uint8,
    ).view(torch.float8_e4m3fn)
    weight_scale_2 = torch.full(
        weight.shape[:-2],
        global_scale,
        dtype=torch.float32,
    )
    return qweight, block_scale, weight_scale_2


def _quantized_names(weight_name: str) -> set[str]:
    base = weight_name.removesuffix(".weight")
    return {
        weight_name,
        f"{base}.weight_scale",
        f"{base}.weight_scale_2",
        f"{base}.input_scale",
    }


def _load_shard(path: Path) -> dict[str, torch.Tensor]:
    return safetensors.torch.load_file(str(path), device="cpu")


def test_same_shard_pair_writes_modelopt_fields_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "source"
    save_dir = tmp_path / "converted"
    gate_name = _gate(0)
    up_name = _up(0)
    dense_name = "model.layers.0.self_attn.q_proj.weight"
    gate = _weight(1.0, rows=3)
    up = _weight(2.0, rows=5)
    dense = _weight(3.0)
    custom_kv_scheme = {"dynamic": True, "num_bits": 8, "type": "float"}
    shard_name = "model-00001-of-00001.safetensors"
    _write_source_checkpoint(
        model_dir,
        {shard_name: {gate_name: gate, up_name: up, dense_name: dense}},
        num_hidden_layers=1,
        config_extra={"quantization_config": {"kv_cache_scheme": custom_kv_scheme}},
    )
    _write_json(
        model_dir / "hf_quant_config.json",
        {
            "producer": {"name": "source"},
            "quantization": {"calibration": "preserved"},
        },
    )
    _write_json(model_dir / "tokenizer_config.json", {"tokenizer_class": "Test"})

    pair_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fake_pair(
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        pair_calls.append((gate_weight, up_weight))
        return (
            _fake_quantized(gate_weight, qvalue=11, global_scale=0.25),
            _fake_quantized(up_weight, qvalue=22, global_scale=0.25),
        )

    monkeypatch.setattr(nvfp4_setup, "quantize_nvfp4_pair", fake_pair)
    monkeypatch.setattr(
        nvfp4_setup,
        "quantize_nvfp4",
        lambda _weight: pytest.fail("gate/up must use paired quantization"),
    )

    ignore = nvfp4_setup.convert_nvfp4(
        str(model_dir),
        str(save_dir),
        device="cpu",
    )

    assert len(pair_calls) == 1
    torch.testing.assert_close(pair_calls[0][0], gate)
    torch.testing.assert_close(pair_calls[0][1], up)

    output = _load_shard(save_dir / shard_name)
    assert set(output) == _quantized_names(gate_name) | _quantized_names(up_name) | {
        dense_name
    }
    assert output[gate_name].dtype == torch.uint8
    assert output[up_name].dtype == torch.uint8
    torch.testing.assert_close(output[dense_name], dense)
    for name in (gate_name, up_name):
        input_scale = output[name.removesuffix(".weight") + ".input_scale"]
        assert input_scale.dtype == torch.float32
        assert input_scale.shape == ()
        assert input_scale.item() == 1.0

    output_config = _read_json(save_dir / "config.json")
    quantization_config = output_config["quantization_config"]
    assert quantization_config["quant_algo"] == "NVFP4"
    assert quantization_config["quant_method"] == "modelopt"
    assert quantization_config["group_size"] == 16
    assert quantization_config["kv_cache_scheme"] == custom_kv_scheme
    assert ignore == quantization_config["ignore"]
    assert dense_name.removesuffix(".weight") in ignore
    assert "model.layers.0.self_attn.qkv_proj" in ignore

    hf_quant_config = _read_json(save_dir / "hf_quant_config.json")
    assert hf_quant_config["producer"] == {"name": "source"}
    assert hf_quant_config["quantization"]["calibration"] == "preserved"
    assert hf_quant_config["quantization"]["exclude_modules"] == ignore
    assert hf_quant_config["quantization"]["quant_algo"] == "NVFP4"
    assert hf_quant_config["quantization"]["group_size"] == 16
    assert hf_quant_config["quantization"]["kv_cache_quant_algo"] == "FP8"
    assert _read_json(save_dir / "tokenizer_config.json") == {"tokenizer_class": "Test"}

    output_index = _read_json(save_dir / "model.safetensors.index.json")
    assert set(output_index["weight_map"]) == set(output)
    assert set(output_index["weight_map"].values()) == {shard_name}
    expected_total_size = sum(
        tensor.numel() * tensor.element_size() for tensor in output.values()
    )
    assert output_index["metadata"]["total_size"] == expected_total_size


def test_cross_shard_pair_is_quantized_once_and_written_to_source_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "source"
    save_dir = tmp_path / "converted"
    gate_name = _gate(0)
    up_name = _up(0)
    gate = _weight(1.0, rows=3)
    up = _weight(2.0, rows=5)
    up_shard = "a-up.safetensors"
    gate_shard = "z-gate.safetensors"
    _write_source_checkpoint(
        model_dir,
        {
            up_shard: {up_name: up},
            gate_shard: {gate_name: gate},
        },
        num_hidden_layers=1,
    )

    pair_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fake_pair(
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        pair_calls.append((gate_weight, up_weight))
        return (
            _fake_quantized(gate_weight, qvalue=31, global_scale=0.5),
            _fake_quantized(up_weight, qvalue=47, global_scale=0.5),
        )

    monkeypatch.setattr(nvfp4_setup, "quantize_nvfp4_pair", fake_pair)
    nvfp4_setup.convert_nvfp4(
        str(model_dir),
        str(save_dir),
        device="cpu",
    )

    assert len(pair_calls) == 1
    torch.testing.assert_close(pair_calls[0][0], gate)
    torch.testing.assert_close(pair_calls[0][1], up)

    gate_output = _load_shard(save_dir / gate_shard)
    up_output = _load_shard(save_dir / up_shard)
    assert set(gate_output) == _quantized_names(gate_name)
    assert set(up_output) == _quantized_names(up_name)
    assert torch.all(gate_output[gate_name] == 31)
    assert torch.all(up_output[up_name] == 47)

    output_index = _read_json(save_dir / "model.safetensors.index.json")
    for name in _quantized_names(gate_name):
        assert output_index["weight_map"][name] == gate_shard
    for name in _quantized_names(up_name):
        assert output_index["weight_map"][name] == up_shard


def test_single_side_extra_skip_keeps_entire_expert_container_in_bf16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "source"
    save_dir = tmp_path / "converted"
    skipped_names = (_gate(0), _up(0), _down(0))
    quantized_name = _down(1)
    source_weights = {
        skipped_names[0]: _weight(1.0),
        skipped_names[1]: _weight(2.0),
        skipped_names[2]: _weight(3.0),
        quantized_name: _weight(4.0),
    }
    shard_name = "model.safetensors"
    _write_source_checkpoint(
        model_dir,
        {shard_name: source_weights},
        num_hidden_layers=2,
    )

    quantized_calls: list[torch.Tensor] = []

    def fake_quantize(
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantized_calls.append(weight)
        return _fake_quantized(weight, qvalue=61, global_scale=0.75)

    monkeypatch.setattr(nvfp4_setup, "quantize_nvfp4", fake_quantize)
    monkeypatch.setattr(
        nvfp4_setup,
        "quantize_nvfp4_pair",
        lambda *_args: pytest.fail("a partially skipped pair must remain BF16"),
    )

    ignore = nvfp4_setup.convert_nvfp4(
        str(model_dir),
        str(save_dir),
        device="cpu",
        extra_high_precision_layers_hf=("model.layers.0.mlp.experts.0.gate_proj",),
    )

    assert len(quantized_calls) == 1
    output = _load_shard(save_dir / shard_name)
    for name in skipped_names:
        torch.testing.assert_close(output[name], source_weights[name])
        assert name.removesuffix(".weight") + ".weight_scale" not in output
    assert output[quantized_name].dtype == torch.uint8
    assert "model.layers.0.mlp.experts" in ignore


def test_first_and_last_layers_remain_bf16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "source"
    save_dir = tmp_path / "converted"
    source_weights = {_down(layer): _weight(float(layer + 1)) for layer in range(4)}
    shard_name = "model.safetensors"
    _write_source_checkpoint(
        model_dir,
        {shard_name: source_weights},
        num_hidden_layers=4,
    )

    quantized_calls: list[torch.Tensor] = []

    def fake_quantize(
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantized_calls.append(weight)
        return _fake_quantized(
            weight,
            qvalue=70 + len(quantized_calls),
            global_scale=1.0,
        )

    monkeypatch.setattr(nvfp4_setup, "quantize_nvfp4", fake_quantize)
    ignore = nvfp4_setup.convert_nvfp4(
        str(model_dir),
        str(save_dir),
        device="cpu",
        num_layers_at_start_in_bf16=1,
        num_layers_at_end_in_bf16=1,
    )

    assert len(quantized_calls) == 2
    output = _load_shard(save_dir / shard_name)
    for layer in (0, 3):
        name = _down(layer)
        torch.testing.assert_close(output[name], source_weights[name])
        assert name.removesuffix(".weight") + ".weight_scale" not in output
        assert f"model.layers.{layer}.mlp.experts" in ignore
    for layer in (1, 2):
        name = _down(layer)
        assert output[name].dtype == torch.uint8
        assert _quantized_names(name) <= set(output)


def test_incomplete_gate_up_pair_fails_before_writing_output(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "source"
    save_dir = tmp_path / "converted"
    _write_source_checkpoint(
        model_dir,
        {"model.safetensors": {_gate(0): _weight(1.0)}},
        num_hidden_layers=1,
    )

    with pytest.raises(ValueError, match="incomplete checkpoint pairs"):
        nvfp4_setup.convert_nvfp4(
            str(model_dir),
            str(save_dir),
            device="cpu",
        )
    assert not save_dir.exists()


def test_duplicate_indexed_tensor_across_shards_fails_loudly(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "source"
    save_dir = tmp_path / "converted"
    model_dir.mkdir()
    _write_json(model_dir / "config.json", {"num_hidden_layers": 1})
    gate_name = _gate(0)
    up_name = _up(0)
    dense_name = "model.embed_tokens.weight"
    shard_a = "model-00001-of-00002.safetensors"
    shard_b = "model-00002-of-00002.safetensors"
    safetensors.torch.save_file(
        {gate_name: _weight(1.0), dense_name: _weight(2.0)},
        str(model_dir / shard_a),
    )
    safetensors.torch.save_file(
        {gate_name: _weight(3.0), up_name: _weight(4.0)},
        str(model_dir / shard_b),
    )
    _write_json(
        model_dir / "model.safetensors.index.json",
        {
            "weight_map": {
                dense_name: shard_a,
                gate_name: shard_b,
                up_name: shard_b,
            }
        },
    )

    with pytest.raises(ValueError, match="Duplicate source tensor"):
        nvfp4_setup.convert_nvfp4(
            str(model_dir),
            str(save_dir),
            device="cpu",
        )
    assert not save_dir.exists()


def test_index_mismatch_fails_loudly(tmp_path: Path) -> None:
    model_dir = tmp_path / "source"
    save_dir = tmp_path / "converted"
    gate_name = _gate(0)
    up_name = _up(0)
    _write_source_checkpoint(
        model_dir,
        {"model.safetensors": {gate_name: _weight(1.0), up_name: _weight(2.0)}},
        num_hidden_layers=1,
    )
    index = _read_json(model_dir / "model.safetensors.index.json")
    del index["weight_map"][up_name]
    _write_json(model_dir / "model.safetensors.index.json", index)

    with pytest.raises(ValueError, match="index does not match"):
        nvfp4_setup.convert_nvfp4(
            str(model_dir),
            str(save_dir),
            device="cpu",
        )
    assert not save_dir.exists()


def test_existing_checkpoint_ignore_is_merged_back_into_refit_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_json(source_dir / "config.json", {"num_hidden_layers": 2})
    converted_dir = tmp_path / "converted"
    user_only = "model.layers.0.mlp.experts"
    checkpoint_only = "model.layers.1.mlp.experts"
    _write_source_checkpoint(
        converted_dir,
        {
            "model.safetensors": {
                _down(0): _weight(1.0),
                _down(1): _weight(2.0),
            }
        },
        num_hidden_layers=2,
        config_extra={
            "quantization_config": {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "ignore": [checkpoint_only],
            }
        },
    )
    quantization_cfg: dict[str, Any] = {
        "scheme": "nvfp4",
        "converted_model_path": str(converted_dir),
        "modules_to_not_convert": [user_only],
    }
    monkeypatch.setattr(
        nvfp4_setup,
        "convert_nvfp4",
        lambda *_args, **_kwargs: pytest.fail("existing checkpoint must be reused"),
    )

    result = nvfp4_setup.ensure_nvfp4_checkpoint(
        model_path=str(source_dir),
        quantization_cfg=quantization_cfg,
    )

    assert result == str(converted_dir)
    assert quantization_cfg["modules_to_not_convert"] == [
        user_only,
        checkpoint_only,
    ]


def test_existing_checkpoint_rejects_new_head_layer_bf16_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    _write_json(source_dir / "config.json", {"num_hidden_layers": 2})
    converted_dir = tmp_path / "converted"
    gate_name = _gate(0)
    gate = _weight(1.0)
    qweight, block_scale, global_scale = _fake_quantized(
        gate,
        qvalue=7,
        global_scale=0.5,
    )
    gate_base = gate_name.removesuffix(".weight")
    _write_source_checkpoint(
        converted_dir,
        {
            "model.safetensors": {
                gate_name: qweight,
                f"{gate_base}.weight_scale": block_scale,
                f"{gate_base}.weight_scale_2": global_scale,
                f"{gate_base}.input_scale": torch.ones((), dtype=torch.float32),
            }
        },
        num_hidden_layers=2,
        config_extra={
            "quantization_config": {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "ignore": [],
            }
        },
    )
    quantization_cfg: dict[str, Any] = {
        "scheme": "nvfp4",
        "converted_model_path": str(converted_dir),
        "num_layers_at_start_in_bf16": 1,
    }
    monkeypatch.setattr(
        nvfp4_setup,
        "convert_nvfp4",
        lambda *_args, **_kwargs: pytest.fail("existing checkpoint must be reused"),
    )

    with pytest.raises(ValueError, match="Reconvert the original HF checkpoint"):
        nvfp4_setup.ensure_nvfp4_checkpoint(
            model_path=str(source_dir),
            quantization_cfg=quantization_cfg,
        )
