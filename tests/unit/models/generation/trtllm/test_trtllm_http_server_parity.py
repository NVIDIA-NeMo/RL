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
"""TRT-LLM chat-completions parity and golden generation.

Regenerate the golden on one GPU with::

    NEMO_RL_GENERATE_PARITY_GOLDEN=1 uv run --extra trtllm pytest \\
        tests/unit/models/generation/trtllm/test_trtllm_http_server_parity.py \\
        -k test_parity -p no:randomly --trtllm-only
"""

import json
import multiprocessing
import os
import queue
import sys
import time
import traceback
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest
import requests

from nemo_rl.models.generation.trtllm.trtllm_http_server import (
    _build_reasoning_parser,
    _build_tool_parser,
    _ends_with_token_suffix,
    _make_parse_tool_calls,
)
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

pytestmark = pytest.mark.trtllm

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "chat_template_parity_golden.json"

TRTLLM_REASONING_PARSERS = {
    "qwen3": "qwen3",
    "deepseek_r1": "deepseek-r1",
}

SERVER_PROCESS_START_TIMEOUT = 600
SERVER_PROCESS_STOP_TIMEOUT = 60


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
    raise TimeoutError(f"TRT-LLM server at {base_url} did not start within {timeout}s")


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
    generation_token_ids = msg["generation_token_ids"]
    generation_log_probs = msg["generation_log_probs"]
    return {
        "prompt_token_ids": msg["prompt_token_ids"],
        "generation_token_ids": generation_token_ids,
        "content": msg.get("content"),
        "reasoning_content": msg.get("reasoning_content") or "",
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
        "max_tokens": 512,
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
    normalized_tool_calls = json.loads(json.dumps(raw_msg1["tool_calls"]))
    assert len(normalized_tool_calls) == 1
    normalized_tool_calls[0]["id"] = TOOL_CALL_ID
    asst_msg = {
        "role": "assistant",
        "content": raw_msg1.get("content"),
        "tool_calls": normalized_tool_calls,
        "prompt_token_ids": raw_msg1["prompt_token_ids"],
        "generation_token_ids": raw_msg1["generation_token_ids"],
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
        "max_tokens": 512,
        "chat_template_kwargs": template_kwargs,
    }
    r2 = requests.post(f"{base}/chat/completions", json=body2, timeout=180)
    r2.raise_for_status()
    resp2 = r2.json()
    turn2 = _extract_fields(resp2)
    required_prefix = turn1["prompt_token_ids"] + turn1["generation_token_ids"]
    assert turn2["prompt_token_ids"][: len(required_prefix)] == required_prefix, (
        "TRT-LLM turn-two prompt does not preserve the exact turn-one model prefix"
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
        "max_tokens": 512,
        "chat_template_kwargs": template_kwargs,
    }
    r3 = requests.post(f"{base}/chat/completions", json=body3, timeout=180)
    assert r3.ok, f"TRT-LLM turn-three request failed: {r3.text}"
    turn3 = _extract_fields(r3.json())
    required_prefix = turn2["prompt_token_ids"] + turn2["generation_token_ids"]
    assert turn3["prompt_token_ids"][: len(required_prefix)] == required_prefix, (
        "TRT-LLM turn-three prompt does not preserve the latest assistant prefix"
    )
    return [turn1, turn2, turn3]


def _should_generate_golden() -> bool:
    return os.environ.get("NEMO_RL_GENERATE_PARITY_GOLDEN", "").lower() in (
        "1",
        "true",
    )


def _load_golden(reasoning_parser: str) -> dict[str, Any]:
    assert GOLDEN_PATH.exists(), (
        "Golden missing; run with NEMO_RL_GENERATE_PARITY_GOLDEN=1"
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


def _sanitize_trtllm_child_environment() -> None:
    """Remove scheduler launch state only inside the TRT-LLM server process."""
    for key in tuple(os.environ):
        if key.startswith(("PMIX_", "PMI_", "MPI_", "OMPI_", "SLURM_")):
            os.environ.pop(key, None)


def _import_trtllm_llm() -> Any:
    """Import TRT-LLM, recovering an editable install inside the child only."""
    try:
        from tensorrt_llm import LLM

        return LLM
    except (ImportError, ModuleNotFoundError):
        dist = metadata.distribution("tensorrt-llm")
        direct_url_text = dist.read_text("direct_url.json")
        source_path = (
            (json.loads(direct_url_text) if direct_url_text else {})
            .get("url", "")
            .removeprefix("file://")
        )
        if not source_path:
            raise
        sys.path.insert(0, source_path)
        for module_name in tuple(sys.modules):
            if module_name == "tensorrt_llm" or module_name.startswith("tensorrt_llm."):
                del sys.modules[module_name]

        from tensorrt_llm import LLM

        return LLM


def _run_trtllm_server_process(
    reasoning_parser: str, startup_queue: Any, stop_event: Any
) -> None:
    """Own the TRT-LLM engine and HTTP server in an isolated child process."""
    llm = None
    server = None
    server_thread = None
    startup_reported = False
    try:
        _sanitize_trtllm_child_environment()
        llm_cls = _import_trtllm_llm()

        from tensorrt_llm.llmapi import KvCacheConfig
        from transformers import AutoTokenizer

        from nemo_rl.models.generation.trtllm.trtllm_http_server import start_server

        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
        llm = llm_cls(
            model=MODEL,
            revision=MODEL_REVISION,
            tokenizer_revision=MODEL_REVISION,
            tensor_parallel_size=1,
            max_num_tokens=2048,
            kv_cache_config=KvCacheConfig(avg_seq_len=64),
        )
        server_thread, base_url, server = start_server(
            llm=llm,
            tokenizer=tokenizer,
            model_name=MODEL,
            max_seq_len=4096,
            sampling_config={"temperature": 0.0, "top_p": 1.0, "top_k": None},
            tool_parser="qwen3",
            reasoning_parser=TRTLLM_REASONING_PARSERS[reasoning_parser],
        )
        startup_queue.put(("ready", base_url))
        startup_reported = True

        while not stop_event.wait(0.2):
            if not server_thread.is_alive():
                raise RuntimeError("TRT-LLM HTTP server thread exited unexpectedly")
    except BaseException:
        if not startup_reported:
            startup_queue.put(("error", traceback.format_exc()))
        raise
    finally:
        if server is not None:
            server.should_exit = True
        server_stopped = True
        if server_thread is not None:
            server_thread.join(timeout=30)
            server_stopped = not server_thread.is_alive()
        if llm is not None:
            llm.shutdown()
        if not server_stopped:
            raise RuntimeError("TRT-LLM HTTP server thread did not stop cleanly")


def _stop_server_process(process: Any, stop_event: Any) -> None:
    """Stop the owned server process, escalating only if graceful exit stalls."""
    stop_event.set()
    process.join(timeout=SERVER_PROCESS_STOP_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10)
    if process.is_alive():
        process.kill()
        process.join(timeout=10)


@pytest.fixture(scope="module", params=tuple(PARSER_SCENARIOS))
def trtllm_server(
    request: pytest.FixtureRequest,
) -> Iterator[tuple[str, str]]:
    reasoning_parser = request.param
    if not _should_generate_golden():
        _load_golden(reasoning_parser)

    process_context = multiprocessing.get_context("spawn")
    startup_queue = process_context.Queue()
    stop_event = process_context.Event()
    process = process_context.Process(
        target=_run_trtllm_server_process,
        args=(reasoning_parser, startup_queue, stop_event),
        name=f"trtllm-parity-{reasoning_parser}",
    )

    process_started = False
    try:
        process.start()
        process_started = True
        try:
            status, payload = startup_queue.get(timeout=SERVER_PROCESS_START_TIMEOUT)
        except queue.Empty:
            pytest.fail(
                "TRT-LLM server process did not report startup within "
                f"{SERVER_PROCESS_START_TIMEOUT}s",
                pytrace=False,
            )
        if status == "error":
            pytest.fail(f"TRT-LLM server process failed:\n{payload}", pytrace=False)

        base_url = payload
        _wait_for_server(base_url)
        yield reasoning_parser, base_url
    finally:
        if process_started:
            _stop_server_process(process, stop_event)
        startup_queue.close()
        startup_queue.join_thread()

    assert process_started and process.exitcode == 0, (
        f"TRT-LLM server process exited with code {process.exitcode}"
    )


def _parse_with_trtllm(
    *,
    raw_output: str,
    reasoning_parser: str | None,
    enable_thinking: bool,
    reasoning_at_start: bool,
) -> dict[str, Any]:
    reasoning_content = ""
    content = raw_output
    if reasoning_parser is not None:
        parser = _build_reasoning_parser(
            TRTLLM_REASONING_PARSERS[reasoning_parser],
            {"enable_thinking": enable_thinking},
            reasoning_at_start=reasoning_at_start,
        )
        parsed = parser.parse(raw_output)
        reasoning_content = parsed.reasoning_content or ""
        content = parsed.content

    parse_tool_calls = _make_parse_tool_calls(_build_tool_parser("qwen3"))
    content, tool_calls = parse_tool_calls(content, [TOOL_DEF])
    normalized_calls = []
    for call in tool_calls:
        arguments = call["function"]["arguments"]
        normalized_calls.append(
            {
                "name": call["function"]["name"],
                "arguments": (
                    json.loads(arguments) if isinstance(arguments, str) else arguments
                ),
            }
        )

    return {
        "reasoning_content": reasoning_content,
        "content": content or None if normalized_calls else content,
        "tool_calls": normalized_calls,
    }


def test_ends_with_token_suffix() -> None:
    suffixes = ((7,), (7, 8), (7, 9, 9))

    assert _ends_with_token_suffix([1, 2, 7], suffixes)
    assert _ends_with_token_suffix([1, 2, 7, 8], suffixes)
    assert _ends_with_token_suffix([1, 7, 9, 9], suffixes)
    assert not _ends_with_token_suffix([1, 7, 9], suffixes)
    assert not _ends_with_token_suffix([], suffixes)


@pytest.mark.parametrize("reasoning_parser", tuple(TRTLLM_REASONING_PARSERS))
def test_reasoning_parser_contracts(reasoning_parser: str) -> None:
    for case in REASONING_PARSER_CONTRACT_CASES:
        actual = _parse_with_trtllm(
            raw_output=case["raw_output"],
            reasoning_parser=reasoning_parser,
            enable_thinking=case["enable_thinking"],
            reasoning_at_start=case["reasoning_at_start"],
        )
        assert actual == case["expected"], "TRT-LLM %s reasoning contract %r failed" % (
            reasoning_parser,
            case["name"],
        )


def test_tool_parser_contracts() -> None:
    for case in TOOL_PARSER_CONTRACT_CASES:
        actual = _parse_with_trtllm(
            raw_output=case["raw_output"],
            reasoning_parser=None,
            enable_thinking=False,
            reasoning_at_start=False,
        )
        actual_norm = {**actual, "content": (actual["content"] or "").strip() or None}
        assert actual_norm == case["expected"], (
            "TRT-LLM qwen3 tool contract %r failed" % case["name"]
        )


def test_parity(trtllm_server: tuple[str, str]) -> None:
    reasoning_parser, base_url = trtllm_server
    generate_golden = _should_generate_golden()
    golden = {} if generate_golden else _load_golden(reasoning_parser)

    for scenario_name, enable_thinking in PARSER_SCENARIOS[reasoning_parser]:
        actual_turns = _run_scenario(base_url, enable_thinking)

        if generate_golden:
            golden = json.loads(GOLDEN_PATH.read_text()) if GOLDEN_PATH.exists() else {}
            if golden.get("model") != MODEL:
                golden = {
                    "model": MODEL,
                    "model_revision": MODEL_REVISION,
                    "scenarios": {},
                }
            elif golden.get("model_revision") != MODEL_REVISION:
                golden = {
                    "model": MODEL,
                    "model_revision": MODEL_REVISION,
                    "scenarios": {},
                }
            golden.setdefault("scenarios", {})
            golden["scenarios"][scenario_name] = {"turns": actual_turns}
            GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN_PATH.write_text(json.dumps(golden, indent=2) + "\n")
            continue

        expected_turns = golden["scenarios"][scenario_name]["turns"]
        assert len(actual_turns) == len(expected_turns)

        assert (
            actual_turns[0]["prompt_token_ids"] == expected_turns[0]["prompt_token_ids"]
        ), f"scenario={scenario_name!r}: turn-one engine prompt mismatch"
        for turn_index in range(len(actual_turns) - 1):
            assert prompt_suffix_after_turn(
                actual_turns, turn_index
            ) == prompt_suffix_after_turn(expected_turns, turn_index), (
                f"scenario={scenario_name!r}: transition {turn_index + 1}->"
                f"{turn_index + 2} appended prompt suffix mismatch"
            )

        if enable_thinking:
            assert actual_turns[0]["reasoning_content"], (
                f"scenario={scenario_name!r}: reasoning parser was not exercised"
            )
            assert (
                not actual_turns[0]["reasoning_content"].lstrip().startswith("<think>")
            ), f"scenario={scenario_name!r}: reasoning marker leaked into response"
        else:
            assert not actual_turns[0]["reasoning_content"], (
                f"scenario={scenario_name!r}: reasoning leaked while thinking was disabled"
            )
