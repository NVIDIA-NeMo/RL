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

import json

from copy import deepcopy

from pathlib import Path

from yaml import safe_load

import pytest
import ray

from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.environments.penguin import Penguin, PenguinConfig

# cluster and tokenizer are fixture imports
from tests.unit.models.generation.test_vllm_generation import cluster, tokenizer as penguin_tokenizer, configure_http_server_config


@pytest.fixture(scope="function")
def penguin_vllm_generation(cluster, penguin_tokenizer):
    vllm_config = configure_http_server_config(penguin_tokenizer)
    vllm_config["vllm_cfg"]["max_model_len"] = 16_384
    vllm_config["vllm_cfg"]["http_server_serving_chat_kwargs"] = {
        "enable_auto_tools": True,
        "tool_parser": "hermes",
    }

    # For Qwen 3 models we need to disable thinking truncation over steps and turns. Here, we modify the chat template to do so.
    chat_template = penguin_tokenizer.chat_template
    to_replace = r"""        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}"""
    assert to_replace in chat_template
    chat_template = chat_template.replace(
        to_replace,
        r"""        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}""",
    )
    penguin_tokenizer.chat_template = chat_template
    vllm_config["vllm_cfg"]["http_server_serving_chat_kwargs"]["chat_template"] = penguin_tokenizer.chat_template

    vllm_generation = VllmGeneration(cluster, vllm_config)

    yield vllm_generation

    vllm_generation.shutdown()


@pytest.fixture(scope="function")
def penguin(penguin_vllm_generation):
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

    # A quick power nap
    print("Sleeping for 15s for Penguin to spin up...")
    time.sleep(15)

    yield env
    # Clean up the actor and wait for it to be killed
    env.shutdown.remote()
    ray.kill(env)
    # Give some time for cleanup
    time.sleep(0.1)


@pytest.fixture(scope="function")
def penguin_sanity_test_data():
    fpath = Path(__file__).parent / "penguin_test_data/test_penguin_sanity.json"
    with open(fpath) as f:
        data = json.load(f)
    return data


def test_penguin_sanity(penguin, penguin_sanity_test_data):
    """Test basic functionality of MathEnvironment step with simple messages."""
    actual_result = ray.get(penguin.run_rollouts.remote(penguin_sanity_test_data["input"]))
    expected_result = penguin_sanity_test_data["expected_output"]

    def _standardize_single_result(d: dict):
        d = deepcopy(d)
        d.pop("full_result", None)
        return d

    def _standardize(l: list[dict]):
        return list(map(_standardize_single_result, l))

    assert _standardize(expected_result) == _standardize(actual_result)
