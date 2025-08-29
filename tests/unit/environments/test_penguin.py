# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import time

from yaml import safe_load

import pytest
import ray

from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.environments.penguin import Penguin, PenguinConfig

# cluster and tokenizer are fixture imports
from tests.unit.models.generation.test_vllm_generation import cluster, tokenizer, configure_http_server_config


@pytest.fixture(scope="module")
def vllm_generation(cluster, tokenizer):
    vllm_config = configure_http_server_config(tokenizer)
    vllm_generation = VllmGeneration(cluster, vllm_config)

    yield vllm_generation

    vllm_generation.shutdown()


@pytest.fixture(scope="module")
def penguin(vllm_generation):
    """Create a Penguin actor for testing."""

    yaml_str = r"""multineedle_resources_server:
  resources_servers:
    multineedle:
      entrypoint: app.py
multineedle_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: multineedle_resources_server
      model_server:
        type: responses_api_models
        name: openai_model
openai_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: ${policy_base_url}
      api_key: ${policy_api_key}
      model: ${policy_model_name}
"""

    config = PenguinConfig(
        model_name=vllm_generation.cfg["model_name"],
        base_urls=vllm_generation.dp_openai_server_base_urls,
        initial_global_config_dict=safe_load(yaml_str),
    )
    env = Penguin.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.penguin.Penguin"
            ),
        }
    ).remote(config)

    yield env
    # Clean up the actor and wait for it to be killed
    env.shutdown.remote()
    ray.kill(env)
    # Give some time for cleanup
    time.sleep(0.1)


def test_math_env_step_basic(math_env, basic_test_data):
    """Test basic functionality of MathEnvironment step with simple messages."""
    result = ray.get(
        math_env.step.remote(
            basic_test_data["message_log_batch"], basic_test_data["metadata"]
        )
    )

    # Check observations using field access
    assert len(result.observations) == 3, (
        "Should return observations for all 3 messages"
    )
    assert all(obs["role"] == "environment" for obs in result.observations), (
        "All observations should be from environment"
    )
    assert all(
        obs["content"] == "Environment: correct" for obs in result.observations
    ), "All responses should be correct"

    # Check metadata
    assert len(result.metadata) == 3, "Should return metadata for all 3 messages"
    assert result.metadata == basic_test_data["metadata"], (
        "Metadata should be unchanged"
    )

    # Check rewards and done flags
    assert result.rewards.shape == (3,), "Rewards should be a tensor of shape (3,)"
    assert all(result.rewards == 1.0), "All rewards should be 1.0 for correct answers"
    assert result.terminateds.shape == (3,), (
        "Terminated flags should be a tensor of shape (3,)"
    )
    assert all(result.terminateds == 1.0), "All terminated flags should be 1.0"
