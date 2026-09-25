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

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.models.generation.vllm.config import materialize_vllm_video_config
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

RECIPE = (
    Path(__file__).parents[2]
    / "examples/configs/recipes/vlm"
    / (
        "vlm_grpo-nemotron-omni-30ba3b-16n8g-megatron-tp4ep4-async-gym-"
        "spatialclaw.v1.yaml"
    )
)


def test_spatialclaw_recipe_resolves_to_async_multimodal_grpo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPATIALCLAW_DATA_PATH", "/tmp/spatialclaw.jsonl")
    register_omegaconf_resolvers()
    config = load_config(RECIPE)
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    MasterConfig.model_validate(resolved)

    grpo = resolved["grpo"]
    policy = resolved["policy"]
    data = resolved["data"]
    gym = resolved["env"]["nemo_gym"]
    materialize_vllm_video_config(policy, data)

    assert grpo["async_grpo"]["enabled"] is True
    assert grpo["async_grpo"]["in_flight_weight_updates"] is True
    assert grpo["deduplicate_multimodal_data"] is True
    assert policy["train_global_batch_size"] == (
        grpo["num_prompts_per_step"] * grpo["num_generations_per_prompt"]
    )
    assert policy["max_total_sequence_length"] == 16384
    assert policy["generation"]["max_new_tokens"] == 16384
    assert data["default"]["num_frames"] == 32
    assert policy["generation"]["vllm_cfg"]["video"] == {
        "sampling_style": "nemotron_vl",
        "num_frames": 32,
        "temporal_patch_size": 2,
    }
    assert policy["generation"]["vllm_kwargs"]["media_io_kwargs"] == {
        "video": {"num_frames": 32}
    }
    assert gym["config_paths"] == [
        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
        "resources_servers/spatialclaw/configs/spatialclaw.yaml",
        "responses_api_agents/spatialclaw_agent/configs/spatialclaw_agent.yaml",
    ]
    agent = gym["spatialclaw_agent"]["responses_api_agents"]["spatialclaw_agent"]
    assert agent["video_input_mode"] == "key-frame-aware"
    assert agent["main_enable_thinking"] is True
    assert agent["config_overrides"]["max_steps"] == 4
