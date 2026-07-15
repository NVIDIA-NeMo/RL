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

"""Tests for GRPO checkpoint-engine refit routing."""

from pathlib import Path
from unittest.mock import MagicMock

from nemo_rl.utils.config import load_config


def test_nixl_example_is_an_enabled_non_colocated_overlay():
    repo_root = Path(__file__).parents[3]
    config = load_config(repo_root / "examples/configs/grpo_math_8B_megatron_nixl.yaml")

    generation = config.policy.generation
    assert generation.checkpoint_engine.enabled
    assert generation.checkpoint_engine.backend == "nixl"
    assert not generation.colocated.enabled
    assert config.cluster.num_nodes == 2


def test_refit_policy_generation_checkpoint_engine_uses_weight_sync(monkeypatch):
    from nemo_rl.algorithms import grpo as grpo_mod
    from nemo_rl.weight_sync import checkpoint_engine_weight_synchronizer

    policy = object()
    sync_weights = MagicMock()
    kv_scales = {"layer_0": 1.0}

    class DummyGeneration:
        cfg = {
            "backend": "vllm",
            "checkpoint_engine": {"enabled": True, "backend": "nixl"},
        }

    generation = DummyGeneration()
    monkeypatch.setattr(
        checkpoint_engine_weight_synchronizer,
        "sync_weights_with_checkpoint_engine",
        sync_weights,
    )

    grpo_mod.refit_policy_generation(
        policy=policy,
        policy_generation=generation,
        colocated_inference=False,
        _refit_buffer_size_gb=2,
        timer=None,
        kv_scales=kv_scales,
    )

    sync_weights.assert_called_once_with(
        policy,
        generation,
        timer=None,
        kv_scales=kv_scales,
    )
