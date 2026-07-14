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

import torch

from nemo_rl.models.generation.sglang import mxfp8_quantization_core as mxfp8_core
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
