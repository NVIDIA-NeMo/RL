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

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor

from nemo_rl.models.megatron.draft.training import DSparkSpeculator
from nemo_rl.models.megatron.draft.utils import (
    _dflash_weight_layout,
    _load_normalized_hf_weights_to_dflash,
    export_dflash_weights_to_hf,
)
from nemo_rl.models.policy.draft_config import DSparkDraftConfig

pytestmark = pytest.mark.mcore


class _AsymmetricTP2DFlashBody:
    def __init__(self, state: dict[str, Tensor]) -> None:
        self.config = SimpleNamespace(
            hidden_size=2048,
            intermediate_size=6144,
            num_attention_heads=32,
            num_key_value_heads=4,
            head_dim=128,
            num_target_taps=5,
        )
        self._state = state
        self.loaded_state: dict[str, Tensor] | None = None

    def state_dict(self) -> dict[str, Tensor]:
        return self._state

    def load_state_dict(
        self, state: dict[str, Tensor], *, strict: bool
    ) -> SimpleNamespace:
        assert not strict
        self.loaded_state = state
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


def _tp2_states() -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    q_shape = (2048, 2048)
    o_shape = (2048, 2048)
    return (
        {
            "layers.0.self_attn.q_proj.weight": torch.zeros(q_shape),
            "layers.0.self_attn.o_proj.weight": torch.zeros(o_shape),
        },
        {
            "layers.0.self_attn.q_proj.weight": torch.ones(q_shape),
            "layers.0.self_attn.o_proj.weight": torch.ones(o_shape),
        },
    )


def test_qwen3_30b_a3b_asymmetric_tp2_layouts_are_exact() -> None:
    config = _AsymmetricTP2DFlashBody({}).config

    assert _dflash_weight_layout("layers.0.self_attn.q_proj.weight", config=config) == (
        (4096, 2048),
        0,
    )
    assert _dflash_weight_layout("layers.0.self_attn.o_proj.weight", config=config) == (
        (2048, 4096),
        1,
    )
    assert _dflash_weight_layout("layers.0.self_attn.k_proj.weight", config=config) == (
        (512, 2048),
        0,
    )
    assert _dflash_weight_layout("layers.0.self_attn.v_proj.weight", config=config) == (
        (512, 2048),
        0,
    )
    assert _dflash_weight_layout("layers.0.mlp.gate_proj.weight", config=config) == (
        (6144, 2048),
        0,
    )
    assert _dflash_weight_layout("layers.0.mlp.up_proj.weight", config=config) == (
        (6144, 2048),
        0,
    )
    assert _dflash_weight_layout("layers.0.mlp.down_proj.weight", config=config) == (
        (2048, 6144),
        1,
    )


def test_symmetric_qwen3_8b_layouts_remain_unchanged() -> None:
    config = SimpleNamespace(
        hidden_size=4096,
        intermediate_size=12288,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        num_target_taps=5,
    )

    assert _dflash_weight_layout("layers.0.self_attn.q_proj.weight", config=config) == (
        (4096, 4096),
        0,
    )
    assert _dflash_weight_layout("layers.0.self_attn.o_proj.weight", config=config) == (
        (4096, 4096),
        1,
    )
    assert _dflash_weight_layout("layers.0.self_attn.k_proj.weight", config=config) == (
        (1024, 4096),
        0,
    )
    assert _dflash_weight_layout("layers.0.mlp.up_proj.weight", config=config) == (
        (12288, 4096),
        0,
    )
    assert _dflash_weight_layout("layers.0.mlp.down_proj.weight", config=config) == (
        (4096, 12288),
        1,
    )


def _install_tp2_gather(
    monkeypatch: pytest.MonkeyPatch,
    rank_zero_state: dict[str, Tensor],
    rank_one_state: dict[str, Tensor],
) -> None:
    peers = {
        id(rank_zero_state["layers.0.self_attn.q_proj.weight"]): rank_one_state[
            "layers.0.self_attn.q_proj.weight"
        ],
        id(rank_zero_state["layers.0.self_attn.o_proj.weight"]): rank_one_state[
            "layers.0.self_attn.o_proj.weight"
        ],
    }

    def _gather(local_weight: Tensor) -> list[Tensor]:
        return [local_weight, peers[id(local_weight)]]

    monkeypatch.setattr(
        "nemo_rl.models.megatron.draft.utils._all_gather_tp_shards", _gather
    )


def _assert_tp2_export_and_refit(
    exported: dict[str, Tensor],
    rank_zero_state: dict[str, Tensor],
    rank_one_state: dict[str, Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q_name = "layers.0.self_attn.q_proj.weight"
    o_name = "layers.0.self_attn.o_proj.weight"
    assert tuple(exported[q_name].shape) == (4096, 2048)
    assert tuple(exported[o_name].shape) == (2048, 4096)
    torch.testing.assert_close(exported[q_name][:2048], rank_zero_state[q_name])
    torch.testing.assert_close(exported[q_name][2048:], rank_one_state[q_name])
    torch.testing.assert_close(exported[o_name][:, :2048], rank_zero_state[o_name])
    torch.testing.assert_close(exported[o_name][:, 2048:], rank_one_state[o_name])

    rank_one_model = _AsymmetricTP2DFlashBody(rank_one_state)
    monkeypatch.setattr("nemo_rl.models.megatron.draft.utils._get_tp_rank", lambda: 1)
    missing, unexpected = _load_normalized_hf_weights_to_dflash(
        rank_one_model, {q_name: exported[q_name], o_name: exported[o_name]}
    )
    assert missing == []
    assert unexpected == []
    assert rank_one_model.loaded_state is not None
    torch.testing.assert_close(
        rank_one_model.loaded_state[q_name], rank_one_state[q_name]
    )
    torch.testing.assert_close(
        rank_one_model.loaded_state[o_name], rank_one_state[o_name]
    )


def test_dflash_tp2_asymmetric_qwen_export_and_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank_zero_state, rank_one_state = _tp2_states()
    _install_tp2_gather(monkeypatch, rank_zero_state, rank_one_state)

    exported = dict(
        export_dflash_weights_to_hf(_AsymmetricTP2DFlashBody(rank_zero_state))
    )

    _assert_tp2_export_and_refit(exported, rank_zero_state, rank_one_state, monkeypatch)


def test_dspark_tp2_asymmetric_qwen_body_export_and_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank_zero_state, rank_one_state = _tp2_states()
    _install_tp2_gather(monkeypatch, rank_zero_state, rank_one_state)
    provider = DSparkSpeculator(
        DSparkDraftConfig(
            enabled=True,
            model_name=None,
            block_size=8,
            anchors_per_sample=2,
            mask_token_id=151669,
            target_hidden_state_layer_ids=[1, 12, 23, 34, 45],
            num_layers=5,
            markov_rank=8,
            confidence_enabled=False,
            confidence_with_markov=False,
        )
    )
    adapter: Any = SimpleNamespace(
        body=_AsymmetricTP2DFlashBody(rank_zero_state),
        markov_head=SimpleNamespace(
            markov_w1=SimpleNamespace(weight=torch.zeros(8, 2048)),
            markov_w2=SimpleNamespace(weight=torch.zeros(32, 8)),
            draft_vocab_size=32,
            markov_rank=8,
        ),
        confidence_head=None,
    )

    exported = dict(provider.export_weights(adapter))

    _assert_tp2_export_and_refit(exported, rank_zero_state, rank_one_state, monkeypatch)
