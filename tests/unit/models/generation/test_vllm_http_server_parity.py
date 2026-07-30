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
"""vLLM side of the cross-backend /v1/chat/completions parity test.

Asserts the vLLM HTTP server against the TRT-LLM golden from test_trtllm_http_server_parity.py.
generation_token_ids may differ across kernels (Mamba hybrid); content/tool_calls/finish_reason must match.
prompt_token_ids is not compared — vLLM does not embed it in the HTTP response.
"""

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
import requests

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration

MODEL = os.environ.get("PARITY_MODEL", "Qwen/Qwen3.5-0.8B")
GOLDEN_PATH = (
    Path(__file__).parent / "trtllm" / "fixtures" / "chat_template_parity_golden.json"
)

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
        "enforce_eager": "False",
        "kv_cache_dtype": "auto",
        # Qwen3.5 uses XML tool calls; qwen3_xml (lazily registered) handles that format.
        "http_server_serving_chat_kwargs": {"tool_parser": "qwen3_xml"},
    },
    "colocated": {
        "enabled": True,
        "resources": {"gpus_per_node": None, "num_nodes": None},
    },
    "vllm_kwargs": {},
}


def _wait_for_server(base_url: str) -> None:
    while True:
        try:
            requests.get(base_url, timeout=3)
            return
        except Exception:
            pass


def _extract_generation_token_ids(choice: dict) -> list[int]:
    # Parse token IDs from vLLM's "token_id:N" logprob format.
    return [
        int(lp["token"][len("token_id:"):])
        for lp in (choice.get("logprobs") or {}).get("content") or []
        if lp.get("token", "").startswith("token_id:")
    ]


def _extract_fields(response_json: dict) -> dict:
    choice = response_json["choices"][0]
    msg = choice["message"]
    tool_calls = [
        {"function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
        for tc in (msg.get("tool_calls") or [])
    ]
    return {
        "generation_token_ids": _extract_generation_token_ids(choice),
        "content": msg.get("content"),
        "tool_calls": tool_calls,
        "finish_reason": choice["finish_reason"],
    }


def _get_prompt_token_ids(base_url: str, messages: list[dict], tools: list[dict], template_kwargs: dict) -> list[int]:
    body = {"model": MODEL, "messages": messages, "tools": tools, "temperature": 0.0, "top_p": 1.0, "chat_template_kwargs": template_kwargs}
    r = requests.post(f"{base_url.rstrip('/')}/../tokenize", json=body, timeout=30)
    r.raise_for_status()
    return r.json()["tokens"]


def _run_scenario(base_url: str, enable_thinking: bool) -> list[dict]:
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
    # Reconstruct NeMo-Gym fields for the prefix splice on turn 2.
    prompt_ids_t1 = _get_prompt_token_ids(base, [{"role": "user", "content": USER_MSG}], [TOOL_DEF], template_kwargs)
    asst_msg = {
        "role": "assistant",
        "content": raw_msg1.get("content"),
        "tool_calls": raw_msg1["tool_calls"],
        "prompt_token_ids": prompt_ids_t1,
        "generation_token_ids": turn1["generation_token_ids"],
        "generation_log_probs": [-0.0] * len(turn1["generation_token_ids"]),
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
        "top_k": -1,
        "max_tokens": 512,
        "logprobs": True,
        "chat_template_kwargs": template_kwargs,
    }
    r2 = requests.post(f"{base}/chat/completions", json=body2, timeout=180)
    r2.raise_for_status()
    return [turn1, _extract_fields(r2.json())]


@pytest.fixture(scope="module")
def vllm_server():
    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[1],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=1,
        name="vllm-parity-cluster",
    )
    tokenizer = get_tokenizer({"name": MODEL})
    cfg = configure_generation_config(deepcopy(_BASE_VLLM_CFG), tokenizer, is_eval=True)
    gen = VllmGeneration(cluster, cfg)

    base_urls = gen.dp_openai_server_base_urls
    assert len(base_urls) == 1
    _wait_for_server(base_urls[0])
    yield base_urls[0]
    gen.shutdown()
    cluster.shutdown()


def _save_comparison(scenario_name: str, turns: list[dict], golden_turns: list[dict]) -> None:
    outfile = os.environ.get("VLLM_PARITY_OUTFILE", "")
    if not outfile:
        return
    import pathlib
    p = pathlib.Path(outfile)
    existing = json.loads(p.read_text()) if p.exists() else {"model": MODEL, "scenarios": {}}
    existing["scenarios"][scenario_name] = {
        "turns": [
            {
                "turn": i,
                "trtllm": g,
                "vllm": a,
                "match": {k: a[k] == g[k] for k in ("finish_reason", "tool_calls", "content", "generation_token_ids")},
            }
            for i, (a, g) in enumerate(zip(turns, golden_turns))
        ]
    }
    p.write_text(json.dumps(existing, indent=2) + "\n")


@pytest.mark.parametrize("scenario_name,enable_thinking", SCENARIOS)
def test_parity(vllm_server, scenario_name, enable_thinking):
    assert GOLDEN_PATH.exists(), f"Golden missing; run TRT-LLM test with NEMO_RL_GENERATE_PARITY_GOLDEN=1"
    golden = json.loads(GOLDEN_PATH.read_text())
    assert scenario_name in golden.get("scenarios", {}), f"Scenario {scenario_name!r} missing from golden"
    golden_turns = golden["scenarios"][scenario_name]["turns"]

    turns = _run_scenario(vllm_server, enable_thinking)
    assert len(turns) == len(golden_turns)
    _save_comparison(scenario_name, turns, golden_turns)

    for i, (actual, expected) in enumerate(zip(turns, golden_turns)):
        ctx = f"scenario={scenario_name!r} turn={i}"
        # generation_token_ids: log mismatch but don't fail — Mamba kernels can diverge across backends.
        if actual["generation_token_ids"] != expected["generation_token_ids"]:
            print(
                f"[WARN] {ctx}: generation_token_ids differ "
                f"(vLLM {len(actual['generation_token_ids'])} vs TRT-LLM {len(expected['generation_token_ids'])} tokens)"
            )
        assert actual["finish_reason"] == expected["finish_reason"], f"{ctx}: finish_reason mismatch"
        assert actual["tool_calls"] == expected["tool_calls"], f"{ctx}: tool_calls mismatch"
        # Skip content on tool_call turns — reasoning text handling differs between backends.
        if actual["finish_reason"] != "tool_calls":
            assert actual["content"] == expected["content"], f"{ctx}: content mismatch"
