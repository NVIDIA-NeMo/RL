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
import json
import time
from copy import deepcopy
from pathlib import Path

import pytest
import ray
from yaml import safe_load

from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.environments.penguin import Penguin, PenguinConfig, setup_penguin_config
from nemo_rl.models.generation.vllm import VllmGeneration

# cluster and tokenizer are fixture imports
from tests.unit.models.generation.test_vllm_generation import (
    basic_vllm_test_config,
    cluster,  # noqa: F401
)
from tests.unit.models.generation.test_vllm_generation import (
    tokenizer as penguin_tokenizer,  # noqa: F401
)

try:
    from penguin import config_types  # noqa: F401

    PENGUIN_INSTALLED = True
except ImportError:
    penguin = None
    PENGUIN_INSTALLED = False


@pytest.mark.skipif(
    not PENGUIN_INSTALLED,
    reason="Skipping Penguin test since Penguin is not installed!",
)
def test_penguin_stub_module():
    print(f"Penguin test successfully run! Penguin config_types module: {config_types}")


@pytest.fixture(scope="function")
def penguin_vllm_generation(cluster, penguin_tokenizer):  # noqa: F811
    generation_config = deepcopy(basic_vllm_test_config)
    master_config = {
        "policy": {
            "generation": generation_config,
        },
    }
    setup_penguin_config(master_config, penguin_tokenizer)

    generation_config["vllm_cfg"]["max_model_len"] = 16_384
    # This is the tool parser for Qwen/Qwen3-0.6B. This needs to be changed for other models.
    generation_config["vllm_cfg"]["http_server_serving_chat_kwargs"] = {
        "enable_auto_tools": True,
        "tool_parser": "hermes",
    }

    vllm_generation = VllmGeneration(cluster, generation_config)

    yield vllm_generation

    vllm_generation.shutdown()


@pytest.fixture(scope="function")
def penguin(penguin_vllm_generation):
    """Create a Penguin actor for testing."""

    yaml_str = r"""multineedle_resources_server:
  resources_servers:
    multineedle:
      entrypoint: app.py
      domain: instruction_following
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
      return_token_id_information: true
      uses_reasoning_parser: true
"""

    config = PenguinConfig(
        model_name=penguin_vllm_generation.cfg["model_name"],
        base_urls=penguin_vllm_generation.dp_openai_server_base_urls,
        initial_global_config_dict=safe_load(yaml_str),
    )
    env = Penguin.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.penguin.Penguin"
            ),
        }
    ).remote(config)

    # Blocking wait for penguin to spin up
    ray.get(env.health_check.remote())

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


@pytest.mark.skipif(
    not PENGUIN_INSTALLED,
    reason="Skipping Penguin test since Penguin is not installed!",
)
def test_penguin_sanity(
    penguin,
    penguin_sanity_test_data,
    penguin_vllm_generation,
    penguin_tokenizer,  # noqa: F811
):
    """Test basic functionality of MathEnvironment step with simple messages."""

    # We need to match NeMo RL generation config params before sending to Penguin
    generation_config = penguin_vllm_generation.cfg
    examples = penguin_sanity_test_data["input"]
    for example in examples:
        example["responses_create_params"]["temperature"] = generation_config[
            "temperature"
        ]
        example["responses_create_params"]["top_p"] = generation_config["top_p"]

    actual_result, _ = ray.get(
        penguin.run_rollouts.remote(
            penguin_sanity_test_data["input"], penguin_tokenizer, ""
        )
    )
    expected_result = penguin_sanity_test_data["expected_output"]

    # These are tensors originally and we swap them back to a list for comparison below
    for d in actual_result:
        for message in d["input_message_log"]:
            message["token_ids"] = message["token_ids"].tolist()
        # Right now, we don't need to swap the token ids in the message log since they pointto the same underlying dictionary as above.
        # for message in d["message_log"][:1]:
        #     message["token_ids"] = message["token_ids"].tolist()

    def _standardize_single_result(d: dict):
        d = deepcopy(d)
        d.pop("full_result", None)

        # We remove these fields and message from comparison since we cannot guarantee exact generation reproducibility
        d["message_log"] = d["message_log"][:2]
        for message in d["message_log"][1:]:
            if "token_ids" in message:
                message["token_ids"] = []
            if "generation_logprobs" in message:
                message["generation_logprobs"] = []
            if "prompt_str" in message:
                message["prompt_str"] = "dummy prompt_str"
            if "generation_str" in message:
                message["generation_str"] = "dummy generation_str"

        return d

    def _standardize(l: list[dict]):
        return list(map(_standardize_single_result, l))

    assert _standardize(expected_result) == _standardize(actual_result)
