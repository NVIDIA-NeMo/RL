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
import requests
import torch
from yaml import safe_load

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.environments.nemo_gym import (
    NemoGym,
    NemoGymConfig,
    setup_nemo_gym_config,
)
from nemo_rl.models.generation.vllm import VllmGeneration

# cluster and tokenizer are fixture imports
from tests.unit.models.generation.test_vllm_generation import (
    basic_vllm_test_config,
    cluster,  # noqa: F401
)
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
        "responses_create_params": {"input": []},
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


# Sentinel for omitting the top_logprobs field entirely, which is distinct from sending null.
_OMIT_TOP_LOGPROBS = object()


@pytest.mark.nemo_gym
def test_vllm_http_logprobs_contract(nemo_gym_vllm_generation):
    """Pin the vLLM OpenAI HTTP logprobs contract that NeMo-Gym capture depends on.

    NeMo-Gym's vllm_model sets logprobs=True and return_tokens_as_token_ids=True to extract
    per-token ids and logprobs for training (Gym omits top_logprobs on the capture path, so
    vLLM applies its default; Gym PR #1612 additionally pins top_logprobs=0, which is
    equivalent). vLLM computes `logprobs = top_logprobs if logprobs else None`, so omitting
    top_logprobs (default 0) or sending 0 returns logprobs, while an explicit null returns
    none and silently empties the captured token ids. This exercises the real HTTP path where
    that translation lives (the offline LLM API does not), so a vLLM bump that changes the
    contract fails here instead of silently freezing training.

    All three cases share the (expensive) vLLM fixture, so they run in a single test rather
    than as separate parametrized cases.
    """
    base_url = nemo_gym_vllm_generation.dp_openai_server_base_urls[0]
    gen_cfg = nemo_gym_vllm_generation.cfg

    def _chat(top_logprobs_field):
        body = {
            "model": gen_cfg["model_name"],
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 8,
            # The RL HTTP wrapper asserts these match the generation config exactly.
            "temperature": gen_cfg["temperature"],
            "top_p": gen_cfg["top_p"],
            # The fields NeMo-Gym sets to capture token ids.
            "logprobs": True,
            "return_tokens_as_token_ids": True,
        }
        if top_logprobs_field is not _OMIT_TOP_LOGPROBS:
            body["top_logprobs"] = top_logprobs_field

        # The base URL is known once the fixture is ready, but retry briefly to avoid racing
        # the very first connection to the server.
        last_exc = None
        for _ in range(30):
            try:
                return requests.post(
                    f"{base_url}/chat/completions", json=body, timeout=60
                )
            except requests.exceptions.ConnectionError as e:
                last_exc = e
                time.sleep(1)
        raise AssertionError(f"vLLM HTTP server never became reachable: {last_exc}")

    def _assert_has_token_ids(resp, label):
        resp.raise_for_status()
        content = resp.json()["choices"][0]["logprobs"]["content"]
        assert content, f"expected per-token logprobs for {label}"
        # return_tokens_as_token_ids makes each token a "token_id:<int>" string; capture
        # parses these into ints, so they must all parse.
        token_ids = [int(c["token"].removeprefix("token_id:")) for c in content]
        assert len(token_ids) == len(content)

    # Omitting top_logprobs (what Gym does on the capture path; vLLM default 0) and sending 0
    # (the equivalent explicit pin) must both yield per-token logprobs whose tokens decode to ints.
    _assert_has_token_ids(_chat(_OMIT_TOP_LOGPROBS), "omitted top_logprobs")
    _assert_has_token_ids(_chat(0), "top_logprobs=0")

    # Explicit null is the divergence that motivates the Gym fix: vLLM returns no logprobs
    # (200 with logprobs=None) or rejects the request outright. Both mean capture gets
    # nothing. If a future vLLM makes null behave like 0, this fails and signals the Gym
    # workaround can be relaxed.
    null_resp = _chat(None)
    if null_resp.status_code == 200:
        assert null_resp.json()["choices"][0].get("logprobs") is None
    else:
        # A rejection must be a client-side validation error, not an unrelated server failure
        # that would let this branch pass vacuously.
        assert 400 <= null_resp.status_code < 500, (
            f"expected null top_logprobs accepted-with-None or rejected as 4xx, "
            f"got {null_resp.status_code}: {null_resp.text}"
        )
