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

import json
import math
from copy import deepcopy
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest
import ray
import torch
from transformers import ByT5Tokenizer
from yaml import safe_load

from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets.response_datasets import NemoGymDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.processors import nemo_gym_data_processor
from nemo_rl.distributed.virtual_cluster import _get_node_ip_local
from nemo_rl.environments.nemo_gym import spinup_nemo_gym_actor
from nemo_rl.experience.rollouts import run_nemo_gym_rollout_sync

_REPO_ROOT = Path(__file__).parents[3]
_GYM_ROOT = _REPO_ROOT / "3rdparty/Gym-workspace/Gym"
_CASES_PATH = Path(__file__).parent / "nemo_gym_test_data/l0_rollout_acceptance.yaml"
_POLICY_MODEL_CONFIG = (
    "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"
)
_REQUIRED_L0_CASES = {
    "code_gen",
    "equivalence_llm_judge",
    "math_with_judge",
    "mcqa",
    "single_step_tool_use_with_argument_comparison",
    "structured_outputs_v4",
    "workplace_assistant",
}
_GENERATION_CONFIG = {
    "backend": "test",
    "max_new_tokens": 1024,
    # Tool-heavy environments such as workplace_assistant serialize more than
    # 64K byte-level tokens before a continuation generation.
    "max_total_sequence_length": 131072,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": None,
    "stop_token_ids": None,
    "stop_strings": None,
}
_CODE_GEN_COMPLETION = """```python
import sys

values = iter(map(int, sys.stdin.buffer.read().split()))
test_count = next(values)
answers = []
for _ in range(test_count):
    n = next(values)
    positions = [0] * (n + 1)
    for index in range(n):
        positions[next(values)] = index
    left = right = positions[1]
    bits = []
    for value in range(1, n + 1):
        left = min(left, positions[value])
        right = max(right, positions[value])
        bits.append("1" if right - left + 1 == value else "0")
    answers.append("".join(bits))
sys.stdout.write("\\n".join(answers))
```"""
_TOOL_CALL_ARGUMENTS = '{"event_id": "SHOW24", "section": "Medical Zone"}'
_TOOL_CALL_GENERATION = (
    '{"name":"check_seat_availability","arguments":'
    '{"event_id":"SHOW24","section":"Medical Zone"}}'
)
_MCQA_COMPLETION = r"\boxed{B}"
_EQUIVALENCE_COMPLETION = r"\boxed{Charles Darwin}"
_EQUIVALENCE_JUDGE_COMPLETION = (
    "The candidate identifies Darwin, so it matches the reference.\n\n"
    "[[A=B]] they are equivalent"
)
_STRUCTURED_OUTPUTS_ARGUMENTS = {
    "name": "Dizer Kola",
    "native_name": "ديزركلا",
    "romanized_name": "Dizer Kola",
    "settlement_type": "village",
    "country": "Iran",
    "province": "Mazandaran",
    "county": "Nowshahr",
    "bakhsh": "Central",
    "rural_district": "Baladeh Kojur",
    "coordinates": {"latitude": 36.55694, "longitude": 51.79389},
    "population_total": 250,
    "population_year": 2006,
    "number_of_families": 64,
    "timezone_standard": "UTC+3:30 (IRST)",
    "timezone_dst": "UTC+4:30 (IRDT)",
}
_WORKPLACE_ARGUMENTS = {
    "email_id": "00000057",
    "body": "Thanks for the update - I will get back to you tomorrow.",
}
_BYTE_TOKEN_OFFSET = 3
_CONTINUATION_MARKER = "\x02nemo-rl-continuation\x00"


def _tool_call_message(
    name: str, arguments: dict[str, Any], call_id: str
) -> tuple[dict[str, Any], str]:
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }
    generation = json.dumps(
        {"name": name, "arguments": arguments},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return message, generation


def _text_token_ids(text: str) -> list[int]:
    """Match ByT5's byte vocabulary without deriving goldens from the tokenizer."""
    return [byte + _BYTE_TOKEN_OFFSET for byte in text.encode()]


def _prompt_token_ids(body: dict[str, Any]) -> list[int]:
    serialized_request = json.dumps(body, sort_keys=True, separators=(",", ":"))
    prompt_token_ids = _text_token_ids(serialized_request)

    messages = body.get("messages", [])
    if not any(message.get("role") == "tool" for message in messages):
        return prompt_token_ids

    # The RL postprocessor requires each later prompt to start with every token
    # already observed. Reconstruct the first workplace request, then append the
    # first generation and a marked encoding of the continuation request.
    first_assistant_index = next(
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    )
    initial_body = deepcopy(body)
    initial_body["messages"] = messages[:first_assistant_index]
    initial_request = json.dumps(initial_body, sort_keys=True, separators=(",", ":"))
    _, first_generation = _tool_call_message(
        "email_reply_email",
        _WORKPLACE_ARGUMENTS,
        "call-l0-workplace",
    )
    return (
        _text_token_ids(initial_request)
        + _text_token_ids(first_generation)
        + _text_token_ids(_CONTINUATION_MARKER + serialized_request)
    )


class _ScriptedPolicyGeneration:
    """Minimum policy-generation surface consumed by the Gym rollout path."""

    cfg = _GENERATION_CONFIG


class _ScriptedOpenAIHandler(BaseHTTPRequestHandler):
    """Return deterministic policy outputs to real Gym model and agent services."""

    protocol_version = "HTTP/1.1"

    @staticmethod
    def _assert_request_contract(body: dict[str, Any]) -> None:
        serialized_body = json.dumps(body)
        is_judge_request = (
            "GOLD:" in serialized_body and "CANDIDATE:" in serialized_body
        )

        assert body["model"] == "scripted-model"
        assert body["logprobs"] is True
        assert body["top_logprobs"] == 0
        assert body["return_tokens_as_token_ids"] is True

        if is_judge_request:
            assert "temperature" not in body
            assert "top_p" not in body
            assert "max_tokens" not in body
        else:
            assert body["temperature"] == _GENERATION_CONFIG["temperature"]
            assert body["top_p"] == _GENERATION_CONFIG["top_p"]
            assert body["max_tokens"] == _GENERATION_CONFIG["max_new_tokens"]

        expected_tool_name = None
        if "SHOW24" in serialized_body and "Medical Zone" in serialized_body:
            expected_tool_name = "check_seat_availability"
        elif "Dizer Kola" in serialized_body:
            expected_tool_name = "response_tool_8"
        elif "Task Update on Develop prototype" in serialized_body:
            expected_tool_name = "email_reply_email"

        if expected_tool_name is not None:
            tool_names = {
                tool["function"]["name"]
                for tool in body.get("tools", [])
                if tool.get("type") == "function"
            }
            assert expected_tool_name in tool_names

    @staticmethod
    def _message_for(body: dict[str, Any]) -> tuple[dict[str, Any], str]:
        serialized_body = json.dumps(body)
        if "GOLD:" in serialized_body and "CANDIDATE:" in serialized_body:
            return {
                "role": "assistant",
                "content": _EQUIVALENCE_JUDGE_COMPLETION,
            }, _EQUIVALENCE_JUDGE_COMPLETION
        if "1000 digit numbers" in serialized_body:
            content = r"\boxed{32}"
            return {"role": "assistant", "content": content}, content
        if "python programming language only" in serialized_body:
            return {
                "role": "assistant",
                "content": _CODE_GEN_COMPLETION,
            }, _CODE_GEN_COMPLETION
        if "SHOW24" in serialized_body and "Medical Zone" in serialized_body:
            message, _ = _tool_call_message(
                "check_seat_availability",
                json.loads(_TOOL_CALL_ARGUMENTS),
                "call-l0-tool-use",
            )
            return message, _TOOL_CALL_GENERATION
        if "cystic fibrosis" in serialized_body:
            return {
                "role": "assistant",
                "content": _MCQA_COMPLETION,
            }, _MCQA_COMPLETION
        if "theory of evolution by natural selection" in serialized_body:
            return {
                "role": "assistant",
                "content": _EQUIVALENCE_COMPLETION,
            }, _EQUIVALENCE_COMPLETION
        if "Dizer Kola" in serialized_body:
            return _tool_call_message(
                "response_tool_8",
                {"extraction": _STRUCTURED_OUTPUTS_ARGUMENTS},
                "call-l0-structured-output",
            )
        if "Task Update on Develop prototype" in serialized_body:
            if any(
                message.get("role") == "tool" for message in body.get("messages", [])
            ):
                content = "Done"
                return {"role": "assistant", "content": content}, content
            return _tool_call_message(
                "email_reply_email",
                _WORKPLACE_ARGUMENTS,
                "call-l0-workplace",
            )
        raise AssertionError("L0 scripted policy received an unrecognized prompt")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(content_length))
        self._assert_request_contract(body)
        message, generation_text = self._message_for(body)
        prompt_token_ids = _prompt_token_ids(body)
        generation_token_ids = _text_token_ids(generation_text)
        assert (
            len(prompt_token_ids) + len(generation_token_ids)
            <= _GENERATION_CONFIG["max_total_sequence_length"]
        )
        message.update(
            {
                "prompt_token_ids": prompt_token_ids,
                "generation_token_ids": generation_token_ids,
                "generation_log_probs": [-0.1] * len(generation_token_ids),
            }
        )
        response = {
            "id": "chatcmpl-l0-acceptance",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "scripted-model"),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls"
                    if message.get("tool_calls")
                    else "stop",
                    "message": message,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(generation_token_ids),
                "total_tokens": len(prompt_token_ids) + len(generation_token_ids),
            },
        }
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        pass


def _load_acceptance_cases() -> list[dict[str, Any]]:
    with _CASES_PATH.open() as case_file:
        cases = safe_load(case_file)["cases"]

    assert cases, "L0 Gym rollout acceptance matrix must not be empty"
    required_fields = {
        "name",
        "config_path",
        "data_path",
        "example_index",
        "example_sha256",
        "agent_ref",
        "expected_generations",
        "expected_prompt_fragment",
        "expected_result",
        "expected_reward",
    }
    for case in cases:
        assert required_fields <= case.keys(), (
            f"acceptance case is missing fields: {required_fields - case.keys()}"
        )
        assert (_GYM_ROOT / case["config_path"]).is_file(), (
            f"{case['name']}: missing Gym config {case['config_path']}"
        )
        data_path = _GYM_ROOT / case["data_path"]
        assert data_path.is_file(), (
            f"{case['name']}: missing Gym data {case['data_path']}"
        )
        with data_path.open("rb") as data_file:
            examples = [line.rstrip(b"\r\n") for line in data_file if line.strip()]
        assert 0 <= case["example_index"] < len(examples), (
            f"{case['name']}: example_index {case['example_index']} is outside a {len(examples)}-row dataset"
        )
        actual_sha256 = sha256(examples[case["example_index"]]).hexdigest()
        assert actual_sha256 == case["example_sha256"], (
            f"{case['name']}: pinned example changed; review the row and update its golden values"
        )
        assert case["agent_ref"].keys() >= {"type", "name"}
        assert case["expected_generations"]
        assert all(
            isinstance(generation, str) for generation in case["expected_generations"]
        )
        assert math.isfinite(case["expected_reward"])

    names = [case["name"] for case in cases]
    assert len(names) == len(set(names)), "acceptance case names must be unique"
    assert set(names) == _REQUIRED_L0_CASES, (
        f"L0 matrix must contain exactly {_REQUIRED_L0_CASES}, got {set(names)}"
    )
    return cases


_L0_CASES = _load_acceptance_cases()


@pytest.mark.nemo_gym
def test_l0_scripted_multiturn_tokens_are_contiguous():
    initial_body = {
        "model": "scripted-model",
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": "Task Update on Develop prototype",
            },
        ],
    }
    continuation_body = deepcopy(initial_body)
    continuation_body["messages"].extend(
        [
            {"role": "assistant", "content": None, "tool_calls": []},
            {
                "role": "tool",
                "tool_call_id": "call-l0-workplace",
                "content": '{"output":"Email replied successfully."}',
            },
        ]
    )
    _, first_generation = _tool_call_message(
        "email_reply_email",
        _WORKPLACE_ARGUMENTS,
        "call-l0-workplace",
    )
    seen_token_ids = _prompt_token_ids(initial_body) + _text_token_ids(first_generation)

    continuation_prompt = _prompt_token_ids(continuation_body)

    assert continuation_prompt[: len(seen_token_ids)] == seen_token_ids
    decoded_continuation = ByT5Tokenizer().batch_decode([continuation_prompt])[0]
    assert _CONTINUATION_MARKER in decoded_continuation
    assert json.loads(decoded_continuation.rpartition(_CONTINUATION_MARKER)[2]) == (
        continuation_body
    )


def _load_case_datum(case: dict[str, Any]) -> DatumSpec:
    dataset = NemoGymDataset(str(_GYM_ROOT / case["data_path"]))
    datum = nemo_gym_data_processor(
        dataset.dataset[case["example_index"]], None, None, None, 0
    )
    datum["extra_env_info"]["agent_ref"] = deepcopy(case["agent_ref"])
    return datum


@pytest.fixture(scope="module")
def scripted_openai_base_url():
    server = ThreadingHTTPServer(("0.0.0.0", 0), _ScriptedOpenAIHandler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        yield f"http://{_get_node_ip_local()}:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


@pytest.fixture(scope="module")
def l0_nemo_gym(scripted_openai_base_url):
    """Start all acceptance environments in one real Gym actor."""
    config_paths = [_POLICY_MODEL_CONFIG] + [case["config_path"] for case in _L0_CASES]
    tokenizer = ByT5Tokenizer()
    env = spinup_nemo_gym_actor(
        {
            "nemo_gym": {
                "config_paths": config_paths,
                "skip_venv_if_present": True,
            }
        },
        base_urls=[scripted_openai_base_url],
        model_name="scripted-model",
        tokenizer=tokenizer,
        enable_router_replay=False,
        use_fastokens=False,
    )
    try:
        yield env
    finally:
        try:
            # The shared actor owns many Gym subprocesses, which are reaped
            # sequentially during graceful shutdown.
            ray.get(env.shutdown.remote(), timeout=60)
        finally:
            ray.kill(env)


@pytest.mark.nemo_gym
@pytest.mark.timeout(900)
@pytest.mark.parametrize("case", _L0_CASES, ids=[case["name"] for case in _L0_CASES])
def test_l0_gym_environments_roll_out_through_nemo_rl(l0_nemo_gym, case):
    """A pinned Gym example must preserve its contract across the NeMo RL boundary."""
    tokenizer = ByT5Tokenizer()
    result = run_nemo_gym_rollout_sync(
        policy_generation=_ScriptedPolicyGeneration(),
        input_batch=rl_collate_fn([_load_case_datum(case)]),
        tokenizer=tokenizer,
        task_to_env={"nemo_gym": l0_nemo_gym},
        max_seq_len=_GENERATION_CONFIG["max_total_sequence_length"],
        generation_config=deepcopy(_GENERATION_CONFIG),
        log_full_result_tables=True,
    )

    final_batch = result.final_batch
    assert final_batch.size == 1
    assert final_batch["agent_ref"] == [case["agent_ref"]]

    reward = final_batch["total_reward"].item()
    assert math.isfinite(reward)
    assert reward == pytest.approx(case["expected_reward"])
    assert final_batch["length"].item() > 0

    assistant_messages = [
        message
        for message in final_batch["message_log"][0]
        if message["role"] == "assistant"
    ]
    assert assistant_messages
    assert len(assistant_messages) == len(case["expected_generations"])
    for message, generation in zip(
        assistant_messages, case["expected_generations"], strict=True
    ):
        assert isinstance(message["token_ids"], torch.Tensor)
        assert isinstance(message["generation_logprobs"], torch.Tensor)
        expected_token_ids = torch.tensor(
            [byte + _BYTE_TOKEN_OFFSET for byte in generation.encode()]
        )
        torch.testing.assert_close(message["token_ids"], expected_token_ids)
        torch.testing.assert_close(
            message["generation_logprobs"],
            torch.full((len(expected_token_ids),), -0.1),
        )

    metric_prefix = f"{case['agent_ref']['name']}/reward/"
    assert any(key.startswith(metric_prefix) for key in result.rollout_metrics)

    full_result_key = f"{case['agent_ref']['name']}/full_result"
    assert full_result_key in result.rollout_metrics
    full_result_table = result.rollout_metrics[full_result_key]
    assert len(full_result_table.data) == 1
    full_result = json.loads(full_result_table.data[0][0])
    for field, expected_value in case["expected_result"].items():
        assert full_result[field] == expected_value
    assert full_result["response"]["output"]
    if "expected_tool_outputs" in case:
        tool_outputs = [
            item["output"]
            for item in full_result["response"]["output"]
            if item["type"] == "function_call_output"
        ]
        assert tool_outputs == case["expected_tool_outputs"]
    if "expected_judge_verdicts" in case:
        verdicts = [
            evaluation["verdict_label"]
            for evaluation in full_result["judge_evaluations"]
        ]
        assert verdicts == case["expected_judge_verdicts"]
    generation_strings = [
        item["generation_str"]
        for item in full_result["response"]["output"]
        if "generation_str" in item
    ]
    assert generation_strings == case["expected_generations"]
    prompt_strings = [
        item["prompt_str"]
        for item in full_result["response"]["output"]
        if "prompt_str" in item
    ]
    assert len(prompt_strings) == len(case["expected_generations"])
    assert all(case["expected_prompt_fragment"] in prompt for prompt in prompt_strings)
    if case["name"] == "workplace_assistant":
        assert _CONTINUATION_MARKER in prompt_strings[1]
        continuation_json = prompt_strings[1].rpartition(_CONTINUATION_MARKER)[2]
        continuation_messages = json.loads(continuation_json)["messages"]
        assert any(message.get("tool_calls") for message in continuation_messages)
        assert any(
            message.get("role") == "tool"
            and "Email replied successfully" in message.get("content", "")
            for message in continuation_messages
        )
