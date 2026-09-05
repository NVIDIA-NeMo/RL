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
from zlib import compress, decompress

import pytest
import ray
import torch
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
_CASES_PATH = Path(__file__).parent / "nemo_gym_test_data/p0_rollout_acceptance.yaml"
_POLICY_MODEL_CONFIG = (
    "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"
)
_REQUIRED_P0_CASES = {
    "code_gen",
    "math_with_judge",
    "single_step_tool_use_with_argument_comparison",
}
_GENERATION_CONFIG = {
    "backend": "test",
    "max_new_tokens": 1024,
    "max_total_sequence_length": 16384,
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


class _ScriptedPolicyGeneration:
    """Minimum policy-generation surface consumed by the Gym rollout path."""

    cfg = _GENERATION_CONFIG


class _ScriptedTokenizer:
    """Reversibly decode request and generation metadata without model weights."""

    pad_token_id = 256

    def batch_decode(self, batch: list[list[int]]) -> list[str]:
        decoded = []
        for token_ids in batch:
            payload = bytes(int(token_id) for token_id in token_ids)
            if payload.startswith(b"\x00"):
                decoded.append(decompress(payload[1:]).decode())
            elif payload.startswith(b"\x01"):
                decoded.append(payload[1:].decode())
            else:
                raise ValueError("Unrecognized scripted token sequence")
        return decoded


class _ScriptedOpenAIHandler(BaseHTTPRequestHandler):
    """Return deterministic policy outputs to real Gym model and agent services."""

    protocol_version = "HTTP/1.1"

    @staticmethod
    def _message_for(body: dict[str, Any]) -> tuple[dict[str, Any], str]:
        serialized_body = json.dumps(body)
        if "1000 digit numbers" in serialized_body:
            content = r"\boxed{32}"
            return {"role": "assistant", "content": content}, content
        if "python programming language only" in serialized_body:
            return {
                "role": "assistant",
                "content": _CODE_GEN_COMPLETION,
            }, _CODE_GEN_COMPLETION
        if "SHOW24" in serialized_body and "Medical Zone" in serialized_body:
            return (
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-p0-acceptance",
                            "type": "function",
                            "function": {
                                "name": "check_seat_availability",
                                "arguments": _TOOL_CALL_ARGUMENTS,
                            },
                        }
                    ],
                },
                _TOOL_CALL_GENERATION,
            )
        raise AssertionError("P0 scripted policy received an unrecognized prompt")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(content_length))
        message, generation_text = self._message_for(body)
        serialized_request = json.dumps(
            body, sort_keys=True, separators=(",", ":")
        ).encode()
        prompt_token_ids = list(b"\x00" + compress(serialized_request))
        generation_token_ids = list(b"\x01" + generation_text.encode())
        message.update(
            {
                "prompt_token_ids": prompt_token_ids,
                "generation_token_ids": generation_token_ids,
                "generation_log_probs": [-0.1] * len(generation_token_ids),
            }
        )
        response = {
            "id": "chatcmpl-p0-acceptance",
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

    assert cases, "P0 Gym rollout acceptance matrix must not be empty"
    required_fields = {
        "name",
        "config_path",
        "data_path",
        "example_index",
        "example_sha256",
        "agent_ref",
        "expected_generation",
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
        assert case["expected_result"]
        assert math.isfinite(case["expected_reward"])

    names = [case["name"] for case in cases]
    assert len(names) == len(set(names)), "acceptance case names must be unique"
    assert set(names) == _REQUIRED_P0_CASES, (
        f"P0 matrix must contain exactly {_REQUIRED_P0_CASES}, got {set(names)}"
    )
    return cases


_P0_CASES = _load_acceptance_cases()


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


@pytest.fixture
def p0_nemo_gym(scripted_openai_base_url, case):
    """Start the selected real Gym environment with case-local attribution."""
    config_paths = [_POLICY_MODEL_CONFIG, case["config_path"]]
    tokenizer = _ScriptedTokenizer()
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
            ray.get(env.shutdown.remote(), timeout=10)
        finally:
            ray.kill(env)


@pytest.mark.nemo_gym
@pytest.mark.timeout(900)
@pytest.mark.parametrize("case", _P0_CASES, ids=[case["name"] for case in _P0_CASES])
def test_p0_gym_environments_roll_out_through_nemo_rl(p0_nemo_gym, case):
    """A pinned Gym example must preserve its contract across the NeMo RL boundary."""
    tokenizer = _ScriptedTokenizer()
    result = run_nemo_gym_rollout_sync(
        policy_generation=_ScriptedPolicyGeneration(),
        input_batch=rl_collate_fn([_load_case_datum(case)]),
        tokenizer=tokenizer,
        task_to_env={"nemo_gym": p0_nemo_gym},
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
    for message in assistant_messages:
        assert isinstance(message["token_ids"], torch.Tensor)
        assert isinstance(message["generation_logprobs"], torch.Tensor)
        assert len(message["token_ids"]) > 0
        assert len(message["token_ids"]) == len(message["generation_logprobs"])
        assert torch.isfinite(message["generation_logprobs"]).all()

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
    generation_strings = [
        item["generation_str"]
        for item in full_result["response"]["output"]
        if "generation_str" in item
    ]
    assert generation_strings == [case["expected_generation"]]
    prompt_strings = [
        item["prompt_str"]
        for item in full_result["response"]["output"]
        if "prompt_str" in item
    ]
    assert len(prompt_strings) == 1
    assert case["expected_prompt_fragment"] in prompt_strings[0]
