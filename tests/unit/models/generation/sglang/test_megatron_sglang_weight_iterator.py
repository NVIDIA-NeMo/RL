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

from nemo_rl.models.policy.workers import (
    megatron_sglang_weight_iterator as weight_iterator,
)


class _FakeBridge:
    def __init__(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        self._weights = list(weights)

    def export_hf_weights(self, *_args: Any, **_kwargs: Any):
        return iter(self._weights)


def _make_iterator(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    quantization_config: dict[str, Any] | None = None,
    num_hidden_layers: int = 4,
) -> weight_iterator.MegatronSGLangHfWeightIterator:
    return weight_iterator.MegatronSGLangHfWeightIterator(
        megatron_bridge=_FakeBridge(weights),
        models=[object()],
        conversion_tasks=object(),
        quantization_config=quantization_config,
        num_hidden_layers=num_hidden_layers,
    )


def _collect_entries(
    iterator: weight_iterator.MegatronSGLangHfWeightIterator,
    *,
    target_precision: str,
) -> list[tuple[str, torch.Tensor]]:
    buckets = iterator.iter_hf_weight_buckets(
        target_precision=target_precision,  # type: ignore[arg-type]
        buffer_size_bytes=1 << 30,
    )
    return [entry for bucket in buckets for entry in bucket]


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
        weight_iterator,
        "quantize_mxfp8",
        lambda tensor: (
            torch.zeros_like(tensor, dtype=torch.uint8),
            torch.zeros((tensor.shape[0], tensor.shape[1] // 32), dtype=torch.uint8),
        ),
    )
    iterator = _make_iterator(
        weights,
        quantization_config={
            "num_layers_at_start_in_bf16": 1,
            "num_layers_at_end_in_bf16": 1,
            "extra_high_precision_layers_hf": ["model.layers.2."],
        },
    )

    names = [name for name, _ in _collect_entries(iterator, target_precision="mxfp8")]
    scale_names = [name for name in names if name.endswith(".weight_scale_inv")]
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

    monkeypatch.setattr(weight_iterator, "quantize_mxfp8", fail_quantize)
    entries = _collect_entries(
        _make_iterator(
            weights,
            quantization_config={
                "extra_high_precision_layers_hf": [f"{base}.q_proj"],
                "modules_to_not_convert": [
                    f"{base}.q_proj",
                    f"{base}.k_proj",
                    f"{base}.v_proj",
                ],
            },
        ),
        target_precision="mxfp8",
    )

    assert [name for name, _ in entries] == names
