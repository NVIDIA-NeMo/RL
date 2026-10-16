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

from collections.abc import Iterable
from typing import Any

import pytest
import torch

from nemo_rl.models.generation.sglang import mxfp8_quantization_core as mxfp8_core
from nemo_rl.models.generation.sglang import nvfp4_quantization_core as nvfp4_core
from nemo_rl.models.generation.sglang.quantization_utils import (
    build_dynamic_skip_substrings,
)


def _collect_entries(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    quantization_config: dict[str, Any] | None = None,
    num_hidden_layers: int = 4,
) -> list[tuple[str, torch.Tensor]]:
    skip_weight_substrings = build_dynamic_skip_substrings(
        quantization_config=dict(quantization_config or {}),
        num_hidden_layers=num_hidden_layers,
        static_skip_substrings=mxfp8_core.SKIP_WEIGHT_SUBSTRINGS,
    )
    groups = mxfp8_core.iter_mxfp8_quantized_tensor_groups(
        weights,
        skip_weight_substrings=skip_weight_substrings,
    )
    return [entry for group in groups for entry in group]


def _collect_nvfp4_entries(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    quantization_config: dict[str, Any] | None = None,
    num_hidden_layers: int = 4,
    require_gate_up_pairs: bool = True,
) -> list[tuple[str, torch.Tensor]]:
    skip_weight_substrings = build_dynamic_skip_substrings(
        quantization_config=dict(quantization_config or {}),
        num_hidden_layers=num_hidden_layers,
    )
    groups = nvfp4_core.iter_nvfp4_quantized_tensor_groups(
        weights,
        skip_weight_substrings=skip_weight_substrings,
        require_gate_up_pairs=require_gate_up_pairs,
    )
    return [entry for group in groups for entry in group]


def _fake_nvfp4_output(
    weight: torch.Tensor,
    global_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, columns = weight.shape
    if global_scale is None:
        global_scale = torch.ones((), dtype=torch.float32)
    return (
        torch.zeros((rows, columns // 2), dtype=torch.uint8),
        torch.zeros(
            (rows, columns // 16),
            dtype=torch.uint8,
        ).view(torch.float8_e4m3fn),
        global_scale,
    )


def test_mxfp8_iterator_respects_head_tail_and_extra_high_precision(
    monkeypatch,
) -> None:
    weights = [
        (
            f"model.layers.{layer}.mlp.down_proj.weight",
            torch.ones((2, 32), dtype=torch.bfloat16),
        )
        for layer in range(4)
    ]
    monkeypatch.setattr(
        mxfp8_core,
        "quantize_mxfp8",
        lambda tensor: (
            torch.zeros_like(tensor, dtype=torch.uint8),
            torch.zeros((tensor.shape[0], tensor.shape[1] // 32), dtype=torch.uint8),
        ),
    )

    entries = _collect_entries(
        weights,
        quantization_config={
            "num_layers_at_start_in_bf16": 1,
            "num_layers_at_end_in_bf16": 1,
            "extra_high_precision_layers_hf": ["model.layers.2."],
        },
    )

    scale_names = [name for name, _ in entries if name.endswith(".weight_scale_inv")]
    assert scale_names == ["model.layers.1.mlp.down_proj.weight_scale_inv"]


def test_mxfp8_iterator_keeps_synchronized_qkv_group_in_bf16(
    monkeypatch,
) -> None:
    base = "model.layers.1.self_attn"
    names = [
        f"{base}.{projection}.weight" for projection in ("q_proj", "k_proj", "v_proj")
    ]
    weights = [(name, torch.ones((2, 32), dtype=torch.bfloat16)) for name in names]

    def fail_quantize(_tensor: torch.Tensor):
        raise AssertionError("a synchronized high-precision QKV group must stay BF16")

    monkeypatch.setattr(mxfp8_core, "quantize_mxfp8", fail_quantize)
    entries = _collect_entries(
        weights,
        quantization_config={
            "extra_high_precision_layers_hf": [f"{base}.q_proj"],
            "modules_to_not_convert": [
                f"{base}.q_proj",
                f"{base}.k_proj",
                f"{base}.v_proj",
            ],
        },
    )

    assert [name for name, _ in entries] == names


def test_nvfp4_iterator_respects_head_tail_and_extra_high_precision(
    monkeypatch,
) -> None:
    weights = [
        (
            f"model.layers.{layer}.mlp.experts.0.down_proj.weight",
            torch.ones((2, 32), dtype=torch.bfloat16),
        )
        for layer in range(4)
    ]
    monkeypatch.setattr(
        nvfp4_core,
        "quantize_nvfp4",
        _fake_nvfp4_output,
    )
    entries = _collect_nvfp4_entries(
        weights,
        quantization_config={
            "num_layers_at_start_in_bf16": 1,
            "num_layers_at_end_in_bf16": 1,
            "extra_high_precision_layers_hf": ["model.layers.2."],
        },
    )

    names = [name for name, _ in entries]
    scale_names = [name for name in names if name.endswith(".weight_scale")]
    assert scale_names == ["model.layers.1.mlp.experts.0.down_proj.weight_scale"]


def test_nvfp4_live_pair_has_shared_scale_and_no_input_scale(monkeypatch) -> None:
    gate_name = "model.layers.1.mlp.experts.0.gate_proj.weight"
    up_name = "model.layers.1.mlp.experts.0.up_proj.weight"
    gate = torch.ones((3, 32), dtype=torch.bfloat16)
    up = torch.ones((5, 32), dtype=torch.bfloat16)
    pair_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fake_pair(
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ):
        pair_calls.append((gate_weight, up_weight))
        shared_scale = torch.tensor(0.25, dtype=torch.float32)
        return (
            _fake_nvfp4_output(gate_weight, shared_scale.clone()),
            _fake_nvfp4_output(up_weight, shared_scale.clone()),
        )

    monkeypatch.setattr(nvfp4_core, "quantize_nvfp4_pair", fake_pair)
    entries = _collect_nvfp4_entries(
        [(gate_name, gate), (up_name, up)],
    )
    tensors = dict(entries)

    assert pair_calls == [(gate, up)]
    assert not any(name.endswith(".input_scale") for name in tensors)
    torch.testing.assert_close(
        tensors[gate_name.replace(".weight", ".weight_scale_2")],
        tensors[up_name.replace(".weight", ".weight_scale_2")],
        rtol=0,
        atol=0,
    )


def test_nvfp4_live_non_gated_experts_quantize_without_gate_up_pairs(
    monkeypatch,
) -> None:
    up_name = "backbone.layers.1.mixer.experts.0.up_proj.weight"
    down_name = "backbone.layers.1.mixer.experts.0.down_proj.weight"
    up = torch.ones((3, 32), dtype=torch.bfloat16)
    down = torch.ones((5, 32), dtype=torch.bfloat16)

    monkeypatch.setattr(nvfp4_core, "quantize_nvfp4", _fake_nvfp4_output)
    monkeypatch.setattr(
        nvfp4_core,
        "quantize_nvfp4_pair",
        lambda *_args: pytest.fail("non-gated experts must not use pair quantization"),
    )

    entries = _collect_nvfp4_entries(
        [(up_name, up), (down_name, down)],
        require_gate_up_pairs=False,
    )
    names = {name for name, _ in entries}

    for weight_name in (up_name, down_name):
        base = weight_name.removesuffix(".weight")
        assert {
            weight_name,
            f"{base}.weight_scale",
            f"{base}.weight_scale_2",
        } <= names


def test_nvfp4_single_side_skip_keeps_whole_gate_up_pair_in_bf16(
    monkeypatch,
) -> None:
    gate_name = "model.layers.1.mlp.experts.0.gate_proj.weight"
    up_name = "model.layers.1.mlp.experts.0.up_proj.weight"
    gate = torch.ones((3, 32), dtype=torch.bfloat16)
    up = torch.ones((5, 32), dtype=torch.bfloat16)

    def fail_quantize_pair(*_args: Any, **_kwargs: Any):
        raise AssertionError("a partially skipped gate/up pair must not be quantized")

    monkeypatch.setattr(
        nvfp4_core,
        "quantize_nvfp4_pair",
        fail_quantize_pair,
    )
    entries = _collect_nvfp4_entries(
        [(gate_name, gate), (up_name, up)],
        quantization_config={
            "extra_high_precision_layers_hf": ["gate_proj"],
        },
    )
    tensors = dict(entries)

    assert set(tensors) == {gate_name, up_name}
    assert tensors[gate_name] is gate
    assert tensors[up_name] is up


@pytest.mark.parametrize(
    "weight_name",
    [
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.1.mlp.experts.0.up_proj.weight",
    ],
)
def test_nvfp4_iterator_rejects_incomplete_gate_up_pair(weight_name: str) -> None:
    weight = torch.ones((3, 32), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="incomplete pairs"):
        _collect_nvfp4_entries([(weight_name, weight)])


def test_nvfp4_iterator_rejects_duplicate_pair_role() -> None:
    gate_name = "model.layers.1.mlp.experts.0.gate_proj.weight"
    gate = torch.ones((3, 32), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="duplicate gate tensor"):
        _collect_nvfp4_entries([(gate_name, gate), (gate_name, gate.clone())])
