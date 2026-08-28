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

import asyncio
import base64
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from nemo_rl.models.generation.trtllm.trtllm_http_server import (
    _build_prompt_token_ids,
    _build_sampling_params,
    _compute_splice_inputs,
    _context_leg_response,
    _decode_request_body,
    _make_parse_tool_calls,
    _parse_disaggregated_params,
    _resolve_tool_parser_name,
    _tokenizer_backend_name,
)

pytestmark = pytest.mark.trtllm


class _FakeToolParser:
    def __init__(self, *, calls):
        self.calls = calls

    def has_tool_call(self, text: str) -> bool:
        return "<tool_call>" in text

    def detect_and_parse(self, text: str, tools: list):
        del text, tools
        return SimpleNamespace(
            normal_text="visible before visible after",
            calls=self.calls,
        )


class _RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [1, 2, 3]


def test_tokenizer_backend_name_prefers_concrete_backend():
    backend = SimpleNamespace()
    tokenizer = SimpleNamespace(_tokenizer=backend)

    assert _tokenizer_backend_name(tokenizer) == "types.SimpleNamespace"


def test_tokenizer_backend_name_falls_back_to_tokenizer():
    tokenizer = _RecordingTokenizer()

    assert _tokenizer_backend_name(tokenizer).endswith("._RecordingTokenizer")


def test_http_sampling_params_match_the_direct_path():
    """The HTTP rollout path must sample exactly like the direct generate() path."""
    sampling_params_cls = MagicMock()

    _build_sampling_params(
        sampling_params_cls,
        sampling_config={"temperature": 0.8, "top_p": 0.9, "top_k": 20},
        stop_token_ids=[9],
        max_tokens=4,
    )

    sampling_params_cls.assert_called_once_with(
        temperature=0.8,
        top_p=0.9,
        top_k=20,
        max_tokens=4,
        stop_token_ids=[9],
        include_stop_str_in_output=True,
        logprobs=True,
        logprobs_simple_format=True,
    )


def test_http_sampling_params_map_null_top_k_and_empty_stop_tokens():
    """null top_k reaches TRT-LLM as 0, and unset stop tokens as None, not []."""
    sampling_params_cls = MagicMock()

    _build_sampling_params(
        sampling_params_cls,
        sampling_config={"temperature": 1.0, "top_p": 1.0, "top_k": None},
        stop_token_ids=None,
        max_tokens=8,
    )

    sampling_params_cls.assert_called_once_with(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_tokens=8,
        stop_token_ids=None,
        include_stop_str_in_output=True,
        logprobs=True,
        logprobs_simple_format=True,
    )


def test_explicit_tool_parser_overrides_model_auto_resolution():
    assert _resolve_tool_parser_name("qwen3_coder", "/missing/model") == "qwen3_coder"


def test_tool_parser_is_inferred_from_local_hf_config(tmp_path):
    pytest.importorskip("tensorrt_llm.serve.tool_parser.tool_parser_factory")
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_moe"}')

    assert _resolve_tool_parser_name(None, str(tmp_path)) == "qwen3"


def test_tool_parser_resolution_fails_loudly(monkeypatch):
    factory = pytest.importorskip("tensorrt_llm.serve.tool_parser.tool_parser_factory")
    monkeypatch.setattr(factory, "resolve_auto_tool_parser", lambda _: None)

    with pytest.raises(ValueError, match="set trtllm_cfg.tool_parser explicitly"):
        _resolve_tool_parser_name(None, "/missing/model")


def test_parse_tool_calls_uses_parser_normal_text():
    parser = _FakeToolParser(
        calls=[
            SimpleNamespace(
                name="run_command",
                parameters='{"command": "pwd"}',
            )
        ]
    )
    parse_tool_calls = _make_parse_tool_calls(parser)

    content, calls = parse_tool_calls(
        "visible before<tool_call>xml</tool_call>visible after",
        tools=None,
    )

    assert content == "visible before visible after"
    assert calls[0]["function"] == {
        "name": "run_command",
        "arguments": '{"command": "pwd"}',
    }


def test_parse_tool_calls_preserves_text_when_parser_returns_no_calls():
    parser = _FakeToolParser(calls=[])
    parse_tool_calls = _make_parse_tool_calls(parser)
    text = "visible before<tool_call>malformed</tool_call>visible after"

    content, calls = parse_tool_calls(text, tools=None)

    assert content == text
    assert calls == []


def _tool_call_messages():
    return [
        {"role": "user", "content": "List the current directory."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": '{"command": "pwd"}',
                    },
                }
            ],
            "prompt_token_ids": [10],
            "generation_token_ids": [11],
            "generation_log_probs": [-0.1],
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "/workspace"},
    ]


def _parse_with_trtllm(messages):
    chat_utils = pytest.importorskip("tensorrt_llm.serve.chat_utils")
    transformers = pytest.importorskip("transformers")
    conversation, mm_coroutine, *_ = chat_utils.parse_chat_messages_coroutines(
        messages, transformers.PretrainedConfig()
    )
    assert asyncio.run(mm_coroutine) == (None, None)
    return conversation


def test_trtllm_parser_normalizes_openai_tool_arguments_without_mutation():
    messages = _tool_call_messages()

    conversation = _parse_with_trtllm(messages)

    assert conversation[1]["tool_calls"][0]["function"]["arguments"] == {
        "command": "pwd"
    }
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == (
        '{"command": "pwd"}'
    )
    assert "prompt_token_ids" not in conversation[1]
    assert "generation_token_ids" not in conversation[1]
    assert "generation_log_probs" not in conversation[1]


def test_prompt_and_prefix_render_use_normalized_tool_arguments():
    messages = _tool_call_messages()
    conversation = _parse_with_trtllm(messages)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "parameters": {"type": "object"},
            },
        }
    ]
    tokenizer = _RecordingTokenizer()

    assert _build_prompt_token_ids(conversation, tokenizer, tools=tools) == [1, 2, 3]
    required_prefix_ids, template_prefix_ids = _compute_splice_inputs(
        messages, conversation, tokenizer, tools, {}
    )

    assert required_prefix_ids == [10, 11]
    assert template_prefix_ids == [1, 2, 3]
    assert len(tokenizer.calls) == 2
    for rendered_messages, kwargs in tokenizer.calls:
        assert rendered_messages[1]["tool_calls"][0]["function"]["arguments"] == {
            "command": "pwd"
        }
        assert kwargs["tools"] == tools


def test_trtllm_parser_rejects_invalid_tool_arguments():
    messages = _tool_call_messages()
    messages[1]["tool_calls"][0]["function"]["arguments"] = "not-json"

    with pytest.raises(ValueError, match="function.arguments"):
        _parse_with_trtllm(messages)


class _FakeRequest:
    """Minimal Request stand-in: headers plus the two body accessors."""

    def __init__(self, headers: dict, raw: bytes):
        self.headers = headers
        self._raw = raw

    async def body(self) -> bytes:
        return self._raw

    async def json(self):
        return json.loads(self._raw)


def test_a_plain_json_body_is_decoded_as_json():
    payload = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
    request = _FakeRequest({}, json.dumps(payload).encode())

    assert asyncio.run(_decode_request_body(request)) == payload


def test_a_msgpack_body_is_decoded_when_the_orchestrator_flags_it():
    """TRT-LLM's disagg orchestrator msgpacks the forwarded worker request.

    It leaves Content-Type as application/json so FastAPI still routes the body
    and signals the encoding with X-TRTLLM-Msgpack instead
    (tensorrt_llm/serve/openai_client.py: MSGPACK_HEADERS). Decoding on
    Content-Type alone therefore hands json.loads a msgpack frame, and every
    orchestrator-forwarded request dies on its first byte.
    """
    msgspec = pytest.importorskip("msgspec")
    payload = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
    raw = msgspec.msgpack.Encoder().encode(payload)
    # The header, not the content type, is what selects the decoder.
    request = _FakeRequest(
        {"content-type": "application/json", "x-trtllm-msgpack": "1"}, raw
    )

    assert asyncio.run(_decode_request_body(request)) == payload
    with pytest.raises(UnicodeDecodeError):
        json.loads(raw)


def test_no_disaggregated_params_means_an_aggregated_request():
    assert _parse_disaggregated_params(None) == (None, False)


@pytest.mark.parametrize("requested", [True, False])
def test_the_b64_request_survives_the_wire_to_engine_conversion(requested):
    # The orchestrator sets return_prompt_token_ids_b64 on the *wire* params
    # (gen_tokids_ctxbytes). The LLM-level params the engine takes have no
    # such field, so it must be read before converting, not after.
    pytest.importorskip("tensorrt_llm.serve.openai_protocol")

    disagg_params, return_b64 = _parse_disaggregated_params(
        {"request_type": "context_only", "return_prompt_token_ids_b64": requested}
    )

    assert disagg_params.request_type == "context_only"
    assert return_b64 is requested


@pytest.fixture
def stub_to_disaggregated_params(monkeypatch):
    protocol = types.ModuleType("tensorrt_llm.serve.openai_protocol")
    protocol.to_disaggregated_params = lambda params: SimpleNamespace(
        model_dump=lambda: {"request_type": "context_only"}
    )
    monkeypatch.setitem(sys.modules, protocol.__name__, protocol)


def _context_leg_body(return_prompt_token_ids_b64: bool) -> dict:
    gen = SimpleNamespace(disaggregated_params=object(), finish_reason="length")
    response = _context_leg_response("m", [1, 2, 3], gen, return_prompt_token_ids_b64)
    return json.loads(response.body)


def test_the_context_leg_returns_b64_prompt_ids_when_asked(
    stub_to_disaggregated_params,
):
    body = _context_leg_body(return_prompt_token_ids_b64=True)

    assert "prompt_token_ids" not in body
    decoded = np.frombuffer(base64.b64decode(body["prompt_token_ids_b64"]), np.int32)
    assert decoded.tolist() == [1, 2, 3]


def test_the_context_leg_returns_the_int_list_by_default(
    stub_to_disaggregated_params,
):
    body = _context_leg_body(return_prompt_token_ids_b64=False)

    assert body["prompt_token_ids"] == [1, 2, 3]
    assert "prompt_token_ids_b64" not in body


# What Gym's converter sends: each tool's keys in the OpenAI SDK's TypedDict
# order, "function" before "type".
_GYM_ORDER_TOOL = {
    "function": {
        "name": "execute_bash",
        "description": "Run a command.",
        "parameters": {"type": "object", "properties": {}},
        "strict": True,
    },
    "type": "function",
}


def test_normalized_tools_put_type_first_like_the_disagg_route_and_vllm():
    # Templates serialize each tool verbatim (`tool | tojson`), so key order is
    # part of the prompt.
    pytest.importorskip("tensorrt_llm.serve.openai_protocol")
    from nemo_rl.models.generation.trtllm.trtllm_http_server import (
        normalize_chat_inputs,
    )

    _, [tool] = normalize_chat_inputs(
        [{"role": "user", "content": "hi"}], [_GYM_ORDER_TOOL]
    )

    assert list(tool) == ["type", "function"]
    assert tool["function"] == _GYM_ORDER_TOOL["function"]


def test_normalized_messages_keep_the_ids_the_splice_reads():
    # _compute_splice_inputs takes the on-policy prefix from these; losing them
    # silently re-renders every later turn from the template.
    pytest.importorskip("tensorrt_llm.serve.openai_protocol")
    from nemo_rl.models.generation.trtllm.trtllm_http_server import (
        normalize_chat_inputs,
    )

    assistant = {
        "role": "assistant",
        "content": "done",
        "reasoning": "think",
        "reasoning_content": "think",
        "prompt_token_ids": [1, 2],
        "generation_token_ids": [3, 4],
    }

    messages, tools = normalize_chat_inputs(
        [{"role": "user", "content": "hi"}, assistant], None
    )

    assert messages[1]["prompt_token_ids"] == [1, 2]
    assert messages[1]["generation_token_ids"] == [3, 4]
    assert tools is None


def test_the_chat_route_renders_gym_tools_in_normalized_order(monkeypatch):
    # The engine answering Gym directly must render what the disagg frontend
    # renders; it used to pass Gym's raw tool dicts straight to the template.
    pytest.importorskip("tensorrt_llm.serve.openai_protocol")
    from fastapi.testclient import TestClient

    from nemo_rl.models.generation.trtllm import trtllm_http_server

    rendered = {}

    async def fake_build(messages, tools, tokenizer, model_config, template_kwargs):
        rendered["tools"] = tools
        # Stop before generation; the route answers a ValueError with a 400.
        raise ValueError("rendered")

    monkeypatch.setattr(trtllm_http_server, "build_spliced_prompt_ids", fake_build)
    app = trtllm_http_server.create_app(
        llm=SimpleNamespace(_hf_model_config=object(), _generation_config=None),
        tokenizer=SimpleNamespace(eos_token_id=0),
        model_name="m",
        max_seq_len=16,
        sampling_config={"temperature": 1.0, "top_p": 1.0, "top_k": None},
        tool_parser="qwen3",
    )

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [_GYM_ORDER_TOOL],
        },
    )

    assert response.status_code == 400
    assert [list(tool) for tool in rendered["tools"]] == [["type", "function"]]
