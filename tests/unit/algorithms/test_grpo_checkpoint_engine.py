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
from typing import cast
from unittest.mock import MagicMock

from nemo_rl.utils.config import load_config


def test_nixl_example_is_an_enabled_non_colocated_overlay():
    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.models.generation.vllm.config import (
        VllmConfig,
        normalize_vllm_refit_config,
    )

    repo_root = Path(__file__).parents[3]
    raw_config = load_config(
        repo_root / "examples/configs/grpo_math_8B_megatron_nixl.yaml"
    )
    config = MasterConfig(**raw_config)

    generation = config.policy["generation"]
    normalize_vllm_refit_config(cast(VllmConfig, generation))
    assert generation["refit_transport"] == "nixl"
    assert generation["refit_cfg"].nixl.update_weights_bucket_memory_ratio == 0.05
    assert not generation["colocated"]["enabled"]
    assert config.cluster["num_nodes"] == 2


def test_refit_policy_generation_uses_attached_checkpoint_engine_synchronizer():
    from nemo_rl.algorithms import grpo as grpo_mod
    from nemo_rl.models.generation.vllm import VllmGeneration

    policy = object()
    kv_scales = {"layer_0": 1.0}

    generation = MagicMock(spec=VllmGeneration)
    generation.weight_synchronizer = MagicMock()
    generation.weight_synchronizer.sync_weights.return_value = {"transfer_s": 1.0}

    result = grpo_mod.refit_policy_generation(
        policy=policy,
        policy_generation=generation,
        colocated_inference=False,
        _refit_buffer_size_gb=2,
        timer=None,
        kv_scales=kv_scales,
    )

    generation.weight_synchronizer.sync_weights.assert_called_once_with(
        timer=None, kv_scales=kv_scales
    )
    assert result == {"transfer_s": 1.0}


def test_refit_policy_generation_sglang_uses_standard_refit(monkeypatch):
    from nemo_rl.algorithms import grpo as grpo_mod
    from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration

    policy = MagicMock()
    policy.stream_weights_via_http.return_value = [object()]

    generation = MagicMock(spec=SGLangGeneration)
    generation.get_rollout_engine_urls.return_value = ["http://rollout"]
    ray_get = MagicMock()
    monkeypatch.setattr(grpo_mod.ray, "get", ray_get)

    grpo_mod.refit_policy_generation(
        policy=policy,
        policy_generation=generation,
        colocated_inference=True,
        _refit_buffer_size_gb=2,
    )

    policy.stream_weights_via_http.assert_called_once_with(
        rollout_engine_urls=["http://rollout"],
        buffer_size_bytes=2 * 1024**3,
    )
    ray_get.assert_called_once_with(policy.stream_weights_via_http.return_value)
    assert generation.prepare_for_generation.call_count == 2
