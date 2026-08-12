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
"""vLLM chat-completions parity and golden generation.

Regenerate the shared golden from vLLM on one GPU with::

    NEMO_RL_GENERATE_PARITY_GOLDEN=1 uv run --extra vllm pytest \
        tests/unit/models/generation/test_vllm_http_server_parity.py \
        -k test_parity -p no:randomly --vllm-only
"""

import json
import os
import time
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import requests

from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from tests.unit.models.generation.chat_template_parity_common import (
    FOLLOWUP_USER_MSG,
    MODEL,
    MODEL_REVISION,
    PARSER_SCENARIOS,
    REASONING_PARSER_CONTRACT_CASES,
    TOOL_CALL_ID,
    TOOL_DEF,
    TOOL_PARSER_CONTRACT_CASES,
    TOOL_RESULT,
    USER_MSG,
    prompt_suffix_after_turn,
)

pytestmark = pytest.mark.vllm

GOLDEN_PATH = (
    Path(__file__).parent / "trtllm" / "fixtures" / "chat_template_parity_golden.json"
)

_BASE_VLLM_CFG: VllmConfig = {
    "backend": "vllm",
    "model_name": MODEL,
    "tokenizer": {"name": MODEL},
    "dtype": "bfloat16",
    "max_new_tokens": 512,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": None,
    "val_temperature": 0.0,
    "val_top_p": 1.0,
    "val_top_k": None,
    "stop_token_ids": None,
    "stop_strings": None,
    "vllm_cfg": {
        "precision": "bfloat16",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 1,
        "gpu_memory_utilization": 0.7,
        "max_model_len": 4096,
        "async_engine": True,
        "expose_http_server": True,
        "skip_tokenizer_init": False,
        "load_format": "auto",
        "enforce_eager": False,
        "kv_cache_dtype": "auto",
        "http_server_serving_chat_kwargs": {"tool_parser": "hermes"},
    },
    "colocated": {
        "enabled": True,
        "resources": {"gpus_per_node": None, "num_nodes": None},
    },
    "vllm_kwargs": {
        "revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
    },
}


def _wait_for_server(base_url: str, timeout: int = 180) -> None:
    health_url = base_url.rstrip("/").removesuffix("/v1") + "/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(health_url, timeout=3).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM server at {base_url} did not start within {timeout}s")


def _extract_fields(response_json: dict[str, Any]) -> dict[str, Any]:
    choice = response_json["choices"][0]
    msg = choice["message"]
    tool_calls = [
        {
            "function": {
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
            }
        }
        for tc in (msg.get("tool_calls") or [])
    ]
    return {
        "prompt_token_ids": msg["prompt_token_ids"],
        "generation_token_ids": msg["generation_token_ids"],
        "content": msg.get("content"),
        # Normalize vLLM and NeMo-Gym reasoning field names.
        "reasoning_content": msg.get("reasoning") or msg.get("reasoning_content") or "",
        "tool_calls": tool_calls,
        "finish_reason": choice["finish_reason"],
    }


def _run_scenario(base_url: str, enable_thinking: bool) -> list[dict[str, Any]]:
    template_kwargs = {"enable_thinking": enable_thinking}
    base = base_url.rstrip("/")

    body1 = {
        "model": MODEL,
        "messages": [{"role": "user", "content": USER_MSG}],
        "tools": [TOOL_DEF],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 512,
        "logprobs": True,
        "top_logprobs": 0,
        "chat_template_kwargs": template_kwargs,
    }
    r1 = requests.post(f"{base}/chat/completions", json=body1, timeout=180)
    r1.raise_for_status()
    resp1 = r1.json()
    turn1 = _extract_fields(resp1)
    assert turn1["finish_reason"] == "tool_calls", (
        f"Turn 1 expected finish_reason='tool_calls', got {turn1['finish_reason']!r}.\n"
        f"content: {resp1['choices'][0]['message'].get('content')}"
    )

    raw_msg1 = resp1["choices"][0]["message"]
    normalized_tool_calls = deepcopy(raw_msg1["tool_calls"])
    assert len(normalized_tool_calls) == 1
    normalized_tool_calls[0]["id"] = TOOL_CALL_ID
    asst_msg = {
        "role": "assistant",
        "content": raw_msg1.get("content"),
        "tool_calls": normalized_tool_calls,
        "prompt_token_ids": turn1["prompt_token_ids"],
        "generation_token_ids": turn1["generation_token_ids"],
        "generation_log_probs": raw_msg1["generation_log_probs"],
    }
    tool_result_msg = {
        "role": "tool",
        "tool_call_id": TOOL_CALL_ID,
        "content": TOOL_RESULT,
    }

    body2 = {
        "model": MODEL,
        "messages": [{"role": "user", "content": USER_MSG}, asst_msg, tool_result_msg],
        "tools": [TOOL_DEF],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 512,
        "logprobs": True,
        "top_logprobs": 0,
        "chat_template_kwargs": template_kwargs,
    }
    r2 = requests.post(f"{base}/chat/completions", json=body2, timeout=180)
    r2.raise_for_status()
    resp2 = r2.json()
    turn2 = _extract_fields(resp2)
    required_prefix = turn1["prompt_token_ids"] + turn1["generation_token_ids"]
    assert turn2["prompt_token_ids"][: len(required_prefix)] == required_prefix, (
        "vLLM turn-two engine prompt does not preserve the exact turn-one model prefix"
    )
    assert turn2["content"] is not None and not turn2["tool_calls"], (
        "Turn 2 must be a normal assistant answer before the user follow-up"
    )

    raw_msg2 = resp2["choices"][0]["message"]
    asst_msg2 = {
        "role": "assistant",
        "content": raw_msg2.get("content"),
        "prompt_token_ids": turn2["prompt_token_ids"],
        "generation_token_ids": turn2["generation_token_ids"],
        "generation_log_probs": raw_msg2["generation_log_probs"],
    }
    body3 = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": USER_MSG},
            asst_msg,
            tool_result_msg,
            asst_msg2,
            {"role": "user", "content": FOLLOWUP_USER_MSG},
        ],
        "tools": [TOOL_DEF],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 512,
        "logprobs": True,
        "top_logprobs": 0,
        "chat_template_kwargs": template_kwargs,
    }
    r3 = requests.post(f"{base}/chat/completions", json=body3, timeout=180)
    r3.raise_for_status()
    turn3 = _extract_fields(r3.json())
    required_prefix = turn2["prompt_token_ids"] + turn2["generation_token_ids"]
    assert turn3["prompt_token_ids"][: len(required_prefix)] == required_prefix, (
        "vLLM turn-three engine prompt does not preserve the latest assistant prefix"
    )
    return [turn1, turn2, turn3]


def _should_generate_golden() -> bool:
    return os.environ.get("NEMO_RL_GENERATE_PARITY_GOLDEN", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _load_golden(reasoning_parser: str) -> dict[str, Any]:
    assert GOLDEN_PATH.exists(), (
        "Golden missing; regenerate it with NEMO_RL_GENERATE_PARITY_GOLDEN=1 "
        "and the vLLM parity test"
    )
    golden = json.loads(GOLDEN_PATH.read_text())
    assert golden.get("source_backend") == "vllm", (
        "Golden was not generated by vLLM; regenerate it"
    )
    assert golden.get("model") == MODEL, (
        f"Golden is for {golden.get('model')!r}, not {MODEL!r}; regenerate it"
    )
    assert golden.get("model_revision") == MODEL_REVISION, (
        "Golden model revision does not match; regenerate it"
    )
    for scenario_name, _ in PARSER_SCENARIOS[reasoning_parser]:
        assert scenario_name in golden.get("scenarios", {}), (
            f"Scenario {scenario_name!r} missing from golden; regenerate it"
        )
    return golden


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=MODEL_REVISION, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _parse_with_vllm(
    tokenizer,
    *,
    raw_output: str,
    reasoning_parser: str | None,
    tool_parser: str | None,
    tools: list[dict[str, Any]] | None,
    enable_thinking: bool,
    reasoning_at_start: bool = False,
) -> dict[str, Any]:
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "parser contract"}],
        tools=tools,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )
    reasoning_content = ""
    content: str | None = raw_output

    if reasoning_parser is not None:
        parser_input = "<think>" + raw_output if reasoning_at_start else raw_output
        parser_type = ReasoningParserManager.get_reasoning_parser(reasoning_parser)
        parser = parser_type(tokenizer)
        reasoning_content, content = parser.extract_reasoning(parser_input, request)
        reasoning_content = reasoning_content or ""

    normalized_calls: list[dict[str, Any]] = []
    if tool_parser is not None and tools and content:
        parser_type = ToolParserManager.get_tool_parser(tool_parser)
        parser = parser_type(tokenizer, request.tools)
        parsed = parser.extract_tool_calls(content, request)
        if parsed.tools_called:
            content = parsed.content
            for call in parsed.tool_calls:
                arguments = call.function.arguments
                normalized_calls.append(
                    {
                        "name": call.function.name,
                        "arguments": (
                            json.loads(arguments)
                            if isinstance(arguments, str)
                            else arguments
                        ),
                    }
                )

    return {
        "reasoning_content": reasoning_content,
        "content": content,
        "tool_calls": normalized_calls,
    }


@pytest.mark.parametrize("reasoning_parser", tuple(PARSER_SCENARIOS))
def test_reasoning_parser_contracts(tokenizer, reasoning_parser: str) -> None:
    for case in REASONING_PARSER_CONTRACT_CASES:
        actual = _parse_with_vllm(
            tokenizer,
            raw_output=case["raw_output"],
            reasoning_parser=reasoning_parser,
            tool_parser="hermes",
            tools=[TOOL_DEF],
            enable_thinking=case["enable_thinking"],
            reasoning_at_start=case["reasoning_at_start"],
        )
        assert actual == case["expected"], "vLLM %s reasoning contract %r failed" % (
            reasoning_parser,
            case["name"],
        )


def test_tool_parser_contracts(tokenizer) -> None:
    for case in TOOL_PARSER_CONTRACT_CASES:
        actual = _parse_with_vllm(
            tokenizer,
            raw_output=case["raw_output"],
            reasoning_parser=None,
            tool_parser="hermes",
            tools=[TOOL_DEF],
            enable_thinking=False,
        )
        actual_norm = {**actual, "content": (actual["content"] or "").strip() or None}
        assert actual_norm == case["expected"], (
            "vLLM Hermes tool contract %r failed" % case["name"]
        )


@pytest.fixture(scope="module", params=tuple(PARSER_SCENARIOS))
def vllm_server(
    request: pytest.FixtureRequest,
    tokenizer,
) -> Iterator[tuple[str, str]]:
    reasoning_parser = request.param
    if not _should_generate_golden():
        _load_golden(reasoning_parser)

    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[1],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=1,
        name=f"vllm-parity-{reasoning_parser}-cluster",
    )
    raw_cfg = deepcopy(_BASE_VLLM_CFG)
    raw_cfg["vllm_cfg"]["http_server_serving_chat_kwargs"]["reasoning_parser"] = (
        reasoning_parser
    )
    cfg = configure_generation_config(raw_cfg, tokenizer, is_eval=True)
    gen = VllmGeneration(cluster, cfg)

    base_urls = gen.dp_openai_server_base_urls
    assert len(base_urls) == 1
    _wait_for_server(base_urls[0])
    yield reasoning_parser, base_urls[0]
    gen.shutdown()
    cluster.shutdown()


def test_parity(vllm_server: tuple[str, str]) -> None:
    reasoning_parser, base_url = vllm_server
    generate_golden = _should_generate_golden()
    golden = {} if generate_golden else _load_golden(reasoning_parser)

    suffix_mismatches = []
    for scenario_name, enable_thinking in PARSER_SCENARIOS[reasoning_parser]:
        actual_turns = _run_scenario(base_url, enable_thinking)

        if generate_golden:
            golden = json.loads(GOLDEN_PATH.read_text()) if GOLDEN_PATH.exists() else {}
            if (
                golden.get("model") != MODEL
                or golden.get("model_revision") != MODEL_REVISION
                or golden.get("source_backend") != "vllm"
            ):
                golden = {
                    "model": MODEL,
                    "model_revision": MODEL_REVISION,
                    "source_backend": "vllm",
                    "scenarios": {},
                }
            golden.setdefault("scenarios", {})
            golden["scenarios"][scenario_name] = {"turns": actual_turns}
            GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN_PATH.write_text(json.dumps(golden, indent=2) + "\n")
            continue

        expected_turns = golden["scenarios"][scenario_name]["turns"]
        assert len(actual_turns) == len(expected_turns)

        actual_turn_one, expected_turn_one = actual_turns[0], expected_turns[0]
        assert (
            actual_turn_one["prompt_token_ids"] == expected_turn_one["prompt_token_ids"]
        ), f"scenario={scenario_name!r}: turn-one engine prompt mismatch"

        for turn_index in range(len(actual_turns) - 1):
            actual_suffix = prompt_suffix_after_turn(actual_turns, turn_index)
            expected_suffix = prompt_suffix_after_turn(expected_turns, turn_index)
            if actual_suffix != expected_suffix:
                suffix_mismatches.append(
                    f"scenario={scenario_name!r}: transition {turn_index + 1}->"
                    f"{turn_index + 2} appended prompt suffix mismatch "
                    f"(vLLM={actual_suffix!r}, TRT-LLM={expected_suffix!r})"
                )

        if enable_thinking:
            assert actual_turn_one["reasoning_content"], (
                f"scenario={scenario_name!r}: reasoning parser was not exercised"
            )
            assert (
                not actual_turn_one["reasoning_content"].lstrip().startswith("<think>")
            ), f"scenario={scenario_name!r}: reasoning marker leaked into response"
        else:
            assert not actual_turn_one["reasoning_content"], (
                f"scenario={scenario_name!r}: reasoning leaked while thinking was disabled"
            )

    assert not suffix_mismatches, "\n".join(suffix_mismatches)
