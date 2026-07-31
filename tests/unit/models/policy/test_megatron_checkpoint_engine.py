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

"""Tests for Megatron checkpoint-engine weight transfer."""

from types import SimpleNamespace

import torch

from nemo_rl.models.policy.workers.checkpoint_engine import (
    MegatronCheckpointEngineSendMixin,
)


class _Worker(MegatronCheckpointEngineSendMixin):
    pass


def test_checkpoint_engine_weight_iterator_keeps_weights_without_target_layout():
    worker = _Worker()
    weights = [("weight", torch.ones(2))]
    worker._iter_params_with_optional_kv_scales = lambda kv_scales=None: iter(weights)
    worker.checkpoint_engine = SimpleNamespace(get_target_weight_layout=lambda: None)

    selected = list(worker._checkpoint_engine_weight_iterator())

    assert selected == weights


def test_checkpoint_engine_weight_iterator_filters_for_vllm_layout():
    worker = _Worker()
    local_expert = torch.arange(16).reshape(4, 4)
    remote_expert = local_expert + 100
    dense_weight = torch.ones(4, 4)
    weights = [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", remote_expert),
        ("model.layers.0.mlp.experts.1.gate_proj.weight", local_expert),
        ("model.layers.0.self_attn.q_proj.weight", dense_weight),
        ("model.layers.1.self_attn.q_proj.weight", dense_weight),
    ]
    worker._iter_params_with_optional_kv_scales = lambda kv_scales=None: iter(weights)
    target_layout = {
        "expert_params": {
            "model.layers.0.mlp.experts.w13_weight": {
                "tp_rank": 0,
                "tp_size": 1,
                "local_expert_ids": [1],
            }
        },
        "missing_weight_prefixes": ["model.layers.1."],
    }
    worker.checkpoint_engine = SimpleNamespace(
        get_target_weight_layout=lambda: target_layout,
    )

    selected = list(worker._checkpoint_engine_weight_iterator())

    assert [name for name, _tensor in selected] == [
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ]
    assert selected[0][1] is local_expert
    assert selected[1][1] is dense_weight
