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
"""vLLM chat-completions parity against the TRT-LLM golden."""

import json
import time
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import ray
import requests

from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from tests.unit.models.generation.chat_template_parity_common import (
    GENERATION_TOKEN_IDS_FIELD,
    FOLLOWUP_USER_MSG,
    MODEL,
    MODEL_REVISION,
    PARSER_SCENARIOS,
    REASONING_PARSER_CONTRACT_CASES,
    TOOL_CALL_ID,
    TOOL_DEF,
    TOOL_RESULT,
    TOOL_PARSER_CONTRACT_CASES,
    USER_MSG,
    inclusive_token_span,
    token_edit_similarity,
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
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(base_url, timeout=3)
            return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM server at {base_url} did not start within {timeout}s")


def _extract_generation_token_ids(choice: dict[str, Any]) -> list[int]:
    token_logprobs = (choice.get("logprobs") or {}).get("content") or []
    token_ids = []
    for token_logprob in token_logprobs:
        token = token_logprob.get("token", "")
        assert token.startswith("token_id:"), (
            f"vLLM did not return a token ID for generated token {token!r}"
        )
        token_ids.append(int(token.removeprefix("token_id:")))
    return token_ids


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
        "generation_token_ids": _extract_generation_token_ids(choice),
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
        "generation_log_probs": [-0.0] * len(turn1["generation_token_ids"]),
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
        "generation_log_probs": [-0.0] * len(turn2["generation_token_ids"]),
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


def _load_golden(reasoning_parser: str) -> dict[str, Any]:
    assert GOLDEN_PATH.exists(), (
        "Golden missing; run TRT-LLM test with NEMO_RL_GENERATE_PARITY_GOLDEN=1"
    )
    golden = json.loads(GOLDEN_PATH.read_text())
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


@pytest.fixture(scope="module", params=tuple(PARSER_SCENARIOS))
def vllm_server(
    request: pytest.FixtureRequest,
) -> Iterator[tuple[str, str, list[int], list[int], VllmGeneration]]:
    reasoning_parser = request.param
    _load_golden(reasoning_parser)
    from transformers import AutoTokenizer

    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[1],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=1,
        name=f"vllm-parity-{reasoning_parser}-cluster",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=MODEL_REVISION, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_cfg = deepcopy(_BASE_VLLM_CFG)
    raw_cfg["vllm_cfg"]["http_server_serving_chat_kwargs"]["reasoning_parser"] = (
        reasoning_parser
    )
    cfg = configure_generation_config(raw_cfg, tokenizer, is_eval=True)
    gen = VllmGeneration(cluster, cfg)

    base_urls = gen.dp_openai_server_base_urls
    assert len(base_urls) == 1
    _wait_for_server(base_urls[0])
    tool_call_start = tokenizer.encode("<tool_call>", add_special_tokens=False)
    tool_call_end = tokenizer.encode("</tool_call>", add_special_tokens=False)
    yield reasoning_parser, base_urls[0], tool_call_start, tool_call_end, gen
    gen.shutdown()
    cluster.shutdown()


def _parse_with_vllm(
    gen: VllmGeneration,
    *,
    raw_output: str,
    reasoning_parser: str | None,
    enable_thinking: bool,
) -> dict[str, Any]:
    worker_idx = gen.worker_group.get_dp_leader_worker_idx(0)
    result_ref = gen.worker_group.run_single_worker_single_data(
        "parse_chat_output",
        worker_idx=worker_idx,
        raw_output=raw_output,
        reasoning_parser=reasoning_parser,
        tool_parser="hermes",
        tools=[TOOL_DEF],
        enable_thinking=enable_thinking,
    )
    return ray.get(result_ref)


def test_parser_contracts(
    vllm_server: tuple[str, str, list[int], list[int], VllmGeneration],
) -> None:
    reasoning_parser, _, _, _, gen = vllm_server

    for case in REASONING_PARSER_CONTRACT_CASES:
        actual = _parse_with_vllm(
            gen,
            raw_output=case["raw_output"],
            reasoning_parser=reasoning_parser,
            enable_thinking=case["enable_thinking"],
        )
        assert actual == case["expected"], "vLLM %s reasoning contract %r failed" % (
            reasoning_parser,
            case["name"],
        )

    # Hermes cases need only one parser-parametrized server.
    if reasoning_parser == "qwen3":
        for case in TOOL_PARSER_CONTRACT_CASES:
            actual = _parse_with_vllm(
                gen,
                raw_output=case["raw_output"],
                reasoning_parser=None,
                enable_thinking=False,
            )
            assert actual == case["expected"], (
                "vLLM Hermes tool contract %r failed" % case["name"]
            )


def test_parity(
    vllm_server: tuple[str, str, list[int], list[int], VllmGeneration],
) -> None:
    reasoning_parser, base_url, tool_call_start, tool_call_end, _ = vllm_server
    golden = _load_golden(reasoning_parser)

    suffix_mismatches = []
    for scenario_name, enable_thinking in PARSER_SCENARIOS[reasoning_parser]:
        expected_turns = golden["scenarios"][scenario_name]["turns"]
        actual_turns = _run_scenario(base_url, enable_thinking)
        assert len(actual_turns) == len(expected_turns)

        actual_turn_one, expected_turn_one = actual_turns[0], expected_turns[0]
        assert (
            actual_turn_one["prompt_token_ids"] == expected_turn_one["prompt_token_ids"]
        ), f"scenario={scenario_name!r}: turn-one engine prompt mismatch"

        actual_tool_call_ids = inclusive_token_span(
            actual_turn_one[GENERATION_TOKEN_IDS_FIELD],
            tool_call_start,
            tool_call_end,
        )
        expected_tool_call_ids = inclusive_token_span(
            expected_turn_one[GENERATION_TOKEN_IDS_FIELD],
            tool_call_start,
            tool_call_end,
        )
        similarity = token_edit_similarity(actual_tool_call_ids, expected_tool_call_ids)
        assert similarity >= 0.9, (
            f"scenario={scenario_name!r}: turn-zero tool-call token similarity "
            f"{similarity:.3f} is below the 0.900 parity threshold"
        )
        print(
            f"[PARITY] scenario={scenario_name!r}: turn-zero tool-call token "
            f"similarity={similarity:.3f} "
            f"(vLLM tokens={len(actual_tool_call_ids)}, "
            f"TRT-LLM tokens={len(expected_tool_call_ids)})"
        )
        print(
            "[PARITY_OUTPUT] "
            + json.dumps(
                {
                    "scenario": scenario_name,
                    "tool_call_token_similarity": similarity,
                    "vllm_tool_call_token_ids": actual_tool_call_ids,
                    "trtllm_tool_call_token_ids": expected_tool_call_ids,
                    "vllm": actual_turn_one,
                    "trtllm": expected_turn_one,
                },
                indent=2,
                sort_keys=True,
            )
        )

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
