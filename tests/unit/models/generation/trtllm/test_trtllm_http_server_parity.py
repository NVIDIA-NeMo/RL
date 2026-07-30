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
"""TRT-LLM side of the cross-backend /v1/chat/completions parity test.

Regenerate golden: NEMO_RL_GENERATE_PARITY_GOLDEN=1 pytest <this file> -s --trtllm-only
"""

import json
import os
import time
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.trtllm

MODEL = os.environ.get("PARITY_MODEL", "Qwen/Qwen3.5-0.8B")
GOLDEN_PATH = Path(__file__).parent / "fixtures" / "chat_template_parity_golden.json"

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "run_bash",
        "description": "Run a bash command and return its stdout.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."}
            },
            "required": ["command"],
        },
    },
}

SCENARIOS = [
    ("no_thinking", False),
    ("with_thinking", True),
]

USER_MSG = "List the files in /tmp using run_bash."
TOOL_RESULT = "file1.txt\nfile2.txt\nREADME.md"


def _wait_for_server(base_url: str, timeout: int = 180) -> None:
    health_url = base_url.rstrip("/v1") + "/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(health_url, timeout=3).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"TRT-LLM server at {base_url} did not start within {timeout}s")


def _extract_fields(response_json: dict) -> dict:
    choice = response_json["choices"][0]
    msg = choice["message"]
    tool_calls = [
        {"function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
        for tc in (msg.get("tool_calls") or [])
    ]
    return {
        "prompt_token_ids": msg["prompt_token_ids"],
        "generation_token_ids": msg["generation_token_ids"],
        "content": msg.get("content"),
        "tool_calls": tool_calls,
        "finish_reason": choice["finish_reason"],
    }


def _run_scenario(base_url: str, enable_thinking: bool) -> list[dict]:
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
    asst_msg = {
        "role": "assistant",
        "content": raw_msg1.get("content"),
        "tool_calls": raw_msg1["tool_calls"],
        "prompt_token_ids": raw_msg1["prompt_token_ids"],
        "generation_token_ids": raw_msg1["generation_token_ids"],
        "generation_log_probs": raw_msg1["generation_log_probs"],
    }
    tool_result_msg = {
        "role": "tool",
        "tool_call_id": raw_msg1["tool_calls"][0]["id"],
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
    return [turn1, _extract_fields(r2.json())]


@pytest.fixture(scope="module")
def trtllm_server():
    import os

    # Strip MPI/SLURM vars to prevent TRT-LLM's MPI detection from crashing.
    for _k in list(os.environ):
        if _k.startswith(("PMIX_", "PMI_", "MPI_", "OMPI_", "SLURM_")):
            os.environ.pop(_k, None)

    # Limit to one GPU; Qwen3.5 is a Mamba hybrid and its state pool budget is
    # computed per visible GPU, causing "pool has only N slots" on multi-GPU nodes.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # tensorrt_llm._torch may resolve to a stale partial namespace package in site-packages;
    # fix by prepending the editable source tree and clearing the stale module cache.
    import sys

    try:
        from tensorrt_llm import LLM
    except (ImportError, ModuleNotFoundError):
        import importlib.metadata
        import json as _json

        try:
            dist = importlib.metadata.distribution("tensorrt-llm")
            _txt = dist.read_text("direct_url.json")
            _src = (_json.loads(_txt) if _txt else {}).get("url", "").removeprefix("file://")
            if _src and _src not in sys.path:
                sys.path.insert(0, _src)
            for _k in list(sys.modules):
                if _k == "tensorrt_llm" or _k.startswith("tensorrt_llm."):
                    del sys.modules[_k]
        except Exception:
            pass
        from tensorrt_llm import LLM

    from tensorrt_llm.llmapi import KvCacheConfig
    from transformers import AutoTokenizer

    from nemo_rl.models.generation.trtllm.trtllm_http_server import start_server

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # Qwen3.5-0.8B is Mamba hybrid; cap max_num_tokens and avg_seq_len to fit the
    # Mamba state pool (needs max_num_tokens+1 slots) in the GPU memory budget.
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        max_num_tokens=512,
        kv_cache_config=KvCacheConfig(avg_seq_len=64),
    )
    # Qwen3.5 uses XML tool calls (<function=...>); qwen3_coder handles that format.
    _thread, base_url, server = start_server(
        llm=llm,
        tokenizer=tokenizer,
        model_name=MODEL,
        max_seq_len=4096,
        sampling_config={"temperature": 0.0, "top_p": 1.0},
        tool_parser="qwen3_coder",
    )
    _wait_for_server(base_url)
    yield base_url
    server.should_exit = True


@pytest.mark.parametrize("scenario_name,enable_thinking", SCENARIOS)
def test_parity(trtllm_server, scenario_name, enable_thinking):
    base_url = trtllm_server
    turns = _run_scenario(base_url, enable_thinking)

    if os.environ.get("NEMO_RL_GENERATE_PARITY_GOLDEN", "").lower() in ("1", "true"):
        golden = json.loads(GOLDEN_PATH.read_text()) if GOLDEN_PATH.exists() else {}
        golden.setdefault("model", MODEL)
        golden.setdefault("scenarios", {})
        golden["scenarios"][scenario_name] = {"turns": turns}
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(golden, indent=2) + "\n")
        return

    assert GOLDEN_PATH.exists(), f"Golden missing; run with NEMO_RL_GENERATE_PARITY_GOLDEN=1"
    golden = json.loads(GOLDEN_PATH.read_text())
    assert scenario_name in golden.get("scenarios", {}), f"Scenario {scenario_name!r} missing from golden"
    golden_turns = golden["scenarios"][scenario_name]["turns"]
    assert len(turns) == len(golden_turns)

    for i, (actual, expected) in enumerate(zip(turns, golden_turns)):
        ctx = f"scenario={scenario_name!r} turn={i}"
        assert actual["prompt_token_ids"] == expected["prompt_token_ids"], f"{ctx}: prompt_token_ids mismatch"
        assert actual["generation_token_ids"] == expected["generation_token_ids"], f"{ctx}: generation_token_ids mismatch"
        assert actual["finish_reason"] == expected["finish_reason"], f"{ctx}: finish_reason mismatch"
        assert actual["tool_calls"] == expected["tool_calls"], f"{ctx}: tool_calls mismatch"
        assert actual["content"] == expected["content"], f"{ctx}: content mismatch"
