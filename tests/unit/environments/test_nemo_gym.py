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
from types import SimpleNamespace

import pytest
import ray
import torch
from yaml import safe_load

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.environments.nemo_gym import (
    NemoGym,
    NemoGymConfig,
    _patch_nemotron_nano_v2_chat_template,
    setup_nemo_gym_config,
)
from nemo_rl.models.generation.vllm import VllmGeneration

# cluster and tokenizer are fixture imports
from tests.unit.models.generation.test_vllm_generation import (
    basic_vllm_test_config,
    cluster,  # noqa: F401
)


_NEMOTRON_NANO_V2_ASSISTANT_TEMPLATE = (
    "{%- elif message['role'] == 'assistant' -%}"
    "{%- if '</think>' in content -%}"
    "{%- set content = content.split('</think>')[1].strip() %}"
    "{%- endif -%}"
    "{{- '<SPECIAL_11>Assistant\n' + content.strip() }}"
)


def test_nemotron_nano_v2_chat_template_workaround_is_model_specific():
    original_template = f"before{_NEMOTRON_NANO_V2_ASSISTANT_TEMPLATE}after"
    tokenizer = SimpleNamespace(chat_template=original_template)
    generation_config = deepcopy(basic_vllm_test_config)
    master_config = MasterConfig.model_construct(
        policy={
            "model_name": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
            "generation": generation_config,
        }
    )

    setup_nemo_gym_config(master_config, tokenizer)

    patched_template = tokenizer.chat_template
    assert _NEMOTRON_NANO_V2_ASSISTANT_TEMPLATE not in patched_template
    assert "ns.enable_thinking is defined and ns.enable_thinking is false" in (
        patched_template
    )
    assert "{{- '<think></think>' -}}" in patched_template
    assert (
        generation_config["vllm_cfg"]["http_server_serving_chat_kwargs"][
            "chat_template"
        ]
        == patched_template
    )


def test_nemotron_nano_v2_chat_template_workaround_ignores_other_models():
    original_template = f"before{_NEMOTRON_NANO_V2_ASSISTANT_TEMPLATE}after"
    tokenizer = SimpleNamespace(chat_template=original_template)
    generation_config = deepcopy(basic_vllm_test_config)
    master_config = MasterConfig.model_construct(
        policy={
            "model_name": "Qwen/Qwen3-0.6B",
            "generation": generation_config,
        }
    )

    setup_nemo_gym_config(master_config, tokenizer)

    assert tokenizer.chat_template == original_template
    assert "chat_template" not in generation_config["vllm_cfg"].get(
        "http_server_serving_chat_kwargs", {}
    )


def test_nemotron_nano_v2_chat_template_workaround_is_idempotent():
    original_template = f"before{_NEMOTRON_NANO_V2_ASSISTANT_TEMPLATE}after"
    patched_template = _patch_nemotron_nano_v2_chat_template(original_template)

    assert _patch_nemotron_nano_v2_chat_template(patched_template) == patched_template


def test_nemotron_nano_v2_chat_template_workaround_rejects_upstream_drift():
    with pytest.raises(ValueError, match="no longer matches"):
        _patch_nemotron_nano_v2_chat_template("changed upstream template")


from tests.unit.models.generation.test_vllm_generation import (
    tokenizer as nemo_gym_tokenizer,  # noqa: F401
)


@pytest.mark.nemo_gym
def test_nemo_gym_stub_module():
    from nemo_gym import config_types

    print(
        f"NeMo-Gym test successfully run! NeMo-Gym config_types module: {config_types}"
    )


@pytest.fixture(scope="function")
def nemo_gym_vllm_generation(cluster, nemo_gym_tokenizer):  # noqa: F811
    generation_config = deepcopy(basic_vllm_test_config)
    master_config = MasterConfig.model_construct(
        policy={"generation": generation_config}
    )
    setup_nemo_gym_config(master_config, nemo_gym_tokenizer)

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
def nemo_gym(nemo_gym_vllm_generation):
    """Create a NeMo-Gym actor for testing."""

    yaml_str = r"""example_multi_step_resources_server:
  resources_servers:
    example_multi_step:
      entrypoint: app.py
      domain: instruction_following
example_multi_step_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: example_multi_step_resources_server
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
rollout_max_attempts_to_avoid_lp_nan: 1
"""

    config = NemoGymConfig(
        model_name=nemo_gym_vllm_generation.cfg["model_name"],
        base_urls=nemo_gym_vllm_generation.dp_openai_server_base_urls,
        initial_global_config_dict=safe_load(yaml_str),
    )
    env = NemoGym.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.nemo_gym.NemoGym"
            ),
        }
    ).remote(config)

    # Blocking wait for NeMo-Gym to spin up
    ray.get(env._spinup.remote())

    yield env
    # Clean up the actor and wait for it to be killed
    env.shutdown.remote()
    ray.kill(env)
    # Give some time for cleanup
    time.sleep(0.1)


@pytest.fixture(scope="function")
def nemo_gym_sanity_test_data():
    fpath = Path(__file__).parent / "nemo_gym_test_data/test_nemo_gym_sanity.json"
    with open(fpath) as f:
        data = json.load(f)
    return data


def _write_actual_test_data(original_input: list, actual_result: list):
    """Write actual rollout results to actual_test_nemo_gym_sanity.json.

    This makes it easy to update the expected output after a Gym commit bump:
        cp nemo_gym_test_data/actual_test_nemo_gym_sanity.json nemo_gym_test_data/test_nemo_gym_sanity.json
    """

    def _convert(obj):
        """Recursively convert torch tensors to Python lists for JSON serialization."""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    cleaned = deepcopy(actual_result)
    for r in cleaned:
        r.pop("full_result", None)
        for msg in r.get("message_log", [])[1:]:
            if "token_ids" in msg:
                msg["token_ids"] = []
            if "generation_logprobs" in msg:
                msg["generation_logprobs"] = []

    output_path = (
        Path(__file__).parent / "nemo_gym_test_data/actual_test_nemo_gym_sanity.json"
    )
    data = _convert({"input": original_input, "expected_output": cleaned})
    with open(output_path, "w") as f:
        json.dump(data, f)
        f.write("\n")
    print(f"Wrote updated test data to {output_path}")


def test_nemo_gym_postprocess_uses_batch_decode():
    class _Tokenizer:
        def __init__(self):
            self.batch_decode_calls = []

        def batch_decode(self, batch):
            self.batch_decode_calls.append([list(token_ids) for token_ids in batch])
            return [" ".join(map(str, token_ids)) for token_ids in batch]

    tokenizer = _Tokenizer()
    nemo_gym_result = {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                },
                {
                    "prompt_token_ids": [1, 2, 3, 4, 5],
                    "generation_token_ids": [6, 7],
                    "generation_log_probs": [-0.2, -0.3],
                },
            ]
        },
        "responses_create_params": {
            "input": [
                {
                    "role": "assistant",
                    "content": "prior",
                    "prompt_token_ids": [99],
                    "generation_token_ids": [100],
                    "generation_log_probs": [-1.0],
                }
            ]
        },
    }

    class _MockSelf:
        cfg = {}

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), nemo_gym_result, tokenizer
        )
    )

    assert tokenizer.batch_decode_calls == [
        [[1, 2], [1, 2, 3, 4, 5]],
        [[3], [6, 7]],
    ]
    assert result["message_log"][0]["token_ids"].tolist() == [1, 2]
    assert result["message_log"][1]["token_ids"].tolist() == [3]
    assert result["message_log"][2]["token_ids"].tolist() == [4, 5]
    assert result["message_log"][3]["token_ids"].tolist() == [6, 7]
    assert nemo_gym_result["response"]["output"][0]["prompt_str"] == "1 2"
    assert nemo_gym_result["response"]["output"][0]["generation_str"] == "3"
    assert nemo_gym_result["response"]["output"][1]["prompt_str"] == "1 2 3 4 5"
    assert nemo_gym_result["response"]["output"][1]["generation_str"] == "6 7"


def test_nemo_gym_postprocess_resets_non_contiguous_history():
    class _Tokenizer:
        def batch_decode(self, batch):
            return [" ".join(map(str, token_ids)) for token_ids in batch]

    nemo_gym_result = {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                },
                {
                    "prompt_token_ids": [8, 9],
                    "generation_token_ids": [10],
                    "generation_log_probs": [-0.2],
                },
            ]
        },
        "responses_create_params": {"input": []},
    }

    class _MockSelf:
        cfg = {}

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), nemo_gym_result, _Tokenizer()
        )
    )

    assert len(result["message_log"]) == 2
    assert result["message_log"][0]["token_ids"].tolist() == [8, 9]
    assert result["message_log"][1]["token_ids"].tolist() == [10]
    assert result["message_log"][1]["generation_logprobs"].tolist() == pytest.approx(
        [-0.2]
    )


def test_nemo_gym_token_only_response_compacts_terminal_result():
    class _Tokenizer:
        def batch_decode(self, batch):
            return [" ".join(map(str, token_ids)) for token_ids in batch]

        def apply_chat_template(self, messages, tokenize=True):
            return [1]

    nemo_gym_result = {
        "response": {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "answer"}],
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3, 4],
                }
            ]
        },
        "responses_create_params": {"input": []},
        "reward": 1.0,
    }

    class _MockSelf:
        cfg = {}

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), nemo_gym_result, _Tokenizer()
        )
    )
    assert result["message_log"][1]["generation_logprobs"].tolist() == [0.0, 0.0]

    compact = NemoGym.__ray_metadata__.modified_class._compact_direct_result(result)
    assert "message_log" not in compact
    assert "input_message_log" not in compact
    assert compact["total_tokens"] == 4
    assert compact["assistant_tokens"] == 2
    assert compact["full_result"]["reward"] == 1.0
    serialized = json.dumps(compact)
    assert "generation_token_ids" not in serialized
    assert "prompt_token_ids" not in serialized
    assert "generation_log_probs" not in serialized


@pytest.mark.nemo_gym
def test_nemo_gym_sanity(
    nemo_gym,
    nemo_gym_sanity_test_data,
    nemo_gym_vllm_generation,
    nemo_gym_tokenizer,  # noqa: F811
):
    """Test basic functionality of MathEnvironment step with simple messages."""

    # Save original input before mutation for writing the actual test data file
    original_input = deepcopy(nemo_gym_sanity_test_data["input"])

    # We need to match NeMo RL generation config params before sending to NeMo-Gym
    generation_config = nemo_gym_vllm_generation.cfg
    examples = nemo_gym_sanity_test_data["input"]
    for idx, example in enumerate(examples):
        example["responses_create_params"]["temperature"] = generation_config[
            "temperature"
        ]
        example["responses_create_params"]["top_p"] = generation_config["top_p"]
        example["_rowidx"] = idx

    actual_result, _ = ray.get(
        nemo_gym.run_rollouts.remote(
            nemo_gym_sanity_test_data["input"], nemo_gym_tokenizer, ""
        )
    )
    expected_result = nemo_gym_sanity_test_data["expected_output"]

    # These are tensors originally and we swap them back to a list for comparison below
    for d in actual_result:
        for message in d["input_message_log"]:
            message["token_ids"] = message["token_ids"].tolist()
        # Right now, we don't need to swap the token ids in the message log since they pointto the same underlying dictionary as above.
        # for message in d["message_log"][:1]:
        #     message["token_ids"] = message["token_ids"].tolist()

    # Write the actual result to a file so it can be used to update the expected output.
    # To update: cp actual_test_nemo_gym_sanity.json test_nemo_gym_sanity.json
    _write_actual_test_data(original_input, actual_result)

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
            message.setdefault("is_invalid_tool_call", False)
            message.setdefault("has_malformed_thinking", False)

        return d

    def _standardize(l: list[dict]):
        return list(map(_standardize_single_result, l))

    assert _standardize(expected_result) == _standardize(actual_result)


class TestBlackboxRolloutHelpers:
    """Row-level identity extraction for black-box register/seal."""

    def test_extracts_driver_minted_rollout_id(self):
        row = {
            "responses_create_params": {
                "metadata": {
                    "extra_body": json.dumps({"nemo_rl_rollout_id": "a" * 32})
                }
            }
        }
        assert NemoGym._blackbox_rollout_id(row) == "a" * 32

    @pytest.mark.parametrize(
        "row",
        [
            {},
            {"responses_create_params": {}},
            {"responses_create_params": {"metadata": {}}},
            {"responses_create_params": {"metadata": {"extra_body": "not json"}}},
            {"responses_create_params": {"metadata": {"extra_body": "[1]"}}},
            {"responses_create_params": {"metadata": {"extra_body": "{}"}}},
            {
                "responses_create_params": {
                    "metadata": {
                        "extra_body": json.dumps({"nemo_rl_rollout_id": ""})
                    }
                }
            },
        ],
    )
    def test_returns_none_without_a_valid_id(self, row):
        assert NemoGym._blackbox_rollout_id(row) is None
