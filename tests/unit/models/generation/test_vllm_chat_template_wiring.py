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

"""Chat-template kwargs must reach every consumer that renders a conversation.

The async HTTP server builds three objects that each render chat messages:
OnlineRenderer, OpenAIServingChat and ServingTokenization. They do not share
one copy -- the chat serving builds its reasoning parser from its own, and the
tokenize path passes its own into preprocess_chat -- so a value handed to only
one makes /tokenize render differently from /v1/chat/completions on the same
conversation.

These tests drive the real _setup_vllm_openai_api_server against a fake vLLM
module tree, inspect consumer configuration, and exercise captured raw responses.
"""

import sys
import threading
import types
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from nemo_rl.models.generation.generation_cut_capture import _RequestCaptureState
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)

# The server subclasses each of these (NeMoRLOnlineRenderer, and so on), so the
# recording list is bound explicitly rather than looked up through the instance.
# A class attribute would be shadowed by the subclass and the construction would
# be recorded somewhere the assertions never look.
_BUILT: dict[str, list] = {"renderer": [], "chat": [], "tokenize": []}


def _recorder(slot: str):
    class _Stub:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            _BUILT[slot].append(self)

    return _Stub


_OnlineRenderer = _recorder("renderer")
_OpenAIServingChat = _recorder("chat")
_ServingTokenization = _recorder("tokenize")


class _FakeApp:
    """Minimal FastAPI stand-in: the server only registers routes on it."""

    def __init__(self):
        self.routes = []

    def _register(self, path):
        def decorator(fn):
            self.routes.append((path, fn))
            return fn

        return decorator

    def post(self, path, **_kwargs):
        return self._register(path)

    def get(self, path, **_kwargs):
        return self._register(path)


def _install_fake_vllm(monkeypatch):
    """Stub exactly the vLLM surface _setup_vllm_openai_api_server imports."""
    for name in (
        "vllm",
        "vllm.entrypoints",
        "vllm.entrypoints.openai",
        "vllm.entrypoints.openai.chat_completion",
        "vllm.entrypoints.openai.engine",
        "vllm.entrypoints.openai.models",
        "vllm.entrypoints.serve",
        "vllm.entrypoints.serve.engine",
        "vllm.entrypoints.serve.tokenize",
        "vllm.reasoning",
        "vllm.renderers",
        "vllm.tool_parsers",
        "vllm.v1",
        "vllm.v1.engine",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    def module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)

    def placeholder(name):
        return type(name, (), {})

    module(
        "vllm.entrypoints.chat_utils", load_chat_template=MagicMock(return_value=None)
    )
    module(
        "vllm.entrypoints.openai.chat_completion.protocol",
        ChatCompletionRequest=placeholder("ChatCompletionRequest"),
        ChatCompletionResponse=placeholder("ChatCompletionResponse"),
    )
    module(
        "vllm.entrypoints.openai.chat_completion.serving",
        OpenAIServingChat=_OpenAIServingChat,
    )
    # vLLM 0.29 moved the engine protocol out of the openai package
    # (vllm-project/vllm#54492); NeMo-RL imports ErrorResponse from the new
    # location, the old one is kept for the openai-side consumers.
    module(
        "vllm.entrypoints.openai.engine.protocol",
        ErrorResponse=placeholder("ErrorResponse"),
    )
    module(
        "vllm.entrypoints.serve.engine.protocol",
        ErrorResponse=placeholder("ErrorResponse"),
    )
    module(
        "vllm.entrypoints.openai.models.protocol",
        BaseModelPath=lambda **kwargs: kwargs,
    )
    module(
        "vllm.entrypoints.openai.models.serving",
        OpenAIServingModels=MagicMock(),
    )
    module(
        "vllm.entrypoints.serve.tokenize.protocol",
        TokenizeChatRequest=placeholder("TokenizeChatRequest"),
        TokenizeCompletionRequest=placeholder("TokenizeCompletionRequest"),
        TokenizeResponse=placeholder("TokenizeResponse"),
    )
    module(
        "vllm.entrypoints.serve.tokenize.serving",
        ServingTokenization=_ServingTokenization,
    )
    module("vllm.renderers.online_renderer", OnlineRenderer=_OnlineRenderer)
    module(
        "vllm.sampling_params",
        RequestOutputKind=type("RequestOutputKind", (), {"DELTA": "delta"}),
    )
    module(
        "vllm.exceptions",
        VLLMValidationError=type("VLLMValidationError", (Exception,), {}),
    )
    module(
        "vllm.reasoning.abs_reasoning_parsers",
        ReasoningParserManager=type(
            "ReasoningParserManager", (), {"import_reasoning_parser": MagicMock()}
        ),
    )
    module(
        "vllm.tool_parsers.abstract_tool_parser",
        ToolParserManager=type(
            "ToolParserManager", (), {"import_tool_parser": MagicMock()}
        ),
    )
    module("vllm.v1.engine.async_llm", logger=MagicMock())

    for built in _BUILT.values():
        built.clear()


def _build_server(monkeypatch, serving_chat_kwargs, *, prefix_cuts_enabled=False):
    """Run the real server setup and hand back the three consumer stubs."""
    _install_fake_vllm(monkeypatch)

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker._generation_prefix_cuts_enabled = prefix_cuts_enabled
    worker._capture_registry_lock = threading.Lock()
    worker.cfg = {
        "temperature": 1.0,
        "top_p": 1.0,
        "vllm_cfg": {"http_server_serving_chat_kwargs": serving_chat_kwargs},
    }
    worker.llm = MagicMock(model_config="model-config", renderer="renderer")
    worker._http_engine_client = MagicMock(
        model_config="http-model-config", renderer="http-renderer"
    )
    worker.llm_async_engine_args = MagicMock()
    worker.llm_async_engine_args.create_model_config.return_value = MagicMock(
        served_model_name="served-model", model="model-path"
    )

    worker._setup_vllm_openai_api_server(_FakeApp())
    assert _BUILT["chat"][0].kwargs["engine_client"] is worker._http_engine_client
    return _BUILT["renderer"], _BUILT["chat"], _BUILT["tokenize"]


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix_cuts", [False, True])
@pytest.mark.parametrize(
    "admitted,marker,stream,token_ids,default_token_ids,logprobs,bypass,invalid_raw",
    [
        (True, True, False, True, False, True, True, False),
        (True, True, False, None, True, True, True, False),
        (False, True, False, True, False, True, False, False),
        (False, False, False, True, False, True, False, False),
        (True, True, True, True, False, True, False, False),
        (True, True, False, False, True, True, False, False),
        (True, True, False, True, False, False, False, False),
        (True, True, False, True, False, True, True, True),
    ],
    ids=[
        "captured-raw",
        "server-default-token-ids",
        "unadmitted-marker",
        "ordinary",
        "streaming",
        "no-raw-arrays",
        "no-logprobs",
        "invalid-raw-logprobs",
    ],
)
async def test_captured_chat_logprobs_bypass(
    monkeypatch: Any,
    prefix_cuts: bool,
    admitted: bool,
    marker: bool,
    stream: bool,
    token_ids: bool | None,
    default_token_ids: bool,
    logprobs: bool,
    bypass: bool,
    invalid_raw: bool,
) -> None:
    """Drive the real serving MRO and retain engine-native arrays unchanged."""
    _, chats, _ = _build_server(monkeypatch, {}, prefix_cuts_enabled=prefix_cuts)
    serving = chats[0]
    serving.return_tokens_as_token_ids = default_token_ids
    serving.renderer = types.SimpleNamespace(tokenizer=None)
    capture_calls = {}
    monkeypatch.setattr(
        VllmAsyncGenerationWorkerImpl, "_capture_calls", capture_calls, raising=False
    )
    monkeypatch.setattr(
        VllmAsyncGenerationWorkerImpl,
        "_return_routed_experts_enabled",
        lambda _self: False,
    )
    request = types.SimpleNamespace(
        ng_capture={} if marker else None,
        stream=stream,
        logprobs=logprobs,
        top_logprobs=0,
        return_tokens_as_token_ids=token_ids,
        _restored_generation_token_ids=[],
        _restored_prefix_terminal=None,
    )
    if admitted:
        capture_calls[id(request)] = _RequestCaptureState(
            call=types.SimpleNamespace(), prompt_token_ids=[1, 2]
        )
    original_request_fields = vars(request).copy()
    ordinary_logprobs = object()
    builder = MagicMock(return_value=ordinary_logprobs)
    monkeypatch.setattr(
        _OpenAIServingChat, "_create_chat_logprobs", builder, raising=False
    )

    async def format_response(
        self: Any,
        upstream_request: Any,
        results: AsyncIterator[Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        assert upstream_request is request
        async for output in results:
            final_output = output
        response_type = sys.modules[
            "vllm.entrypoints.openai.chat_completion.protocol"
        ].ChatCompletionResponse
        response = response_type()
        response.usage = None
        response.choices = [
            types.SimpleNamespace(
                index=0,
                message=types.SimpleNamespace(),
                logprobs=self._create_chat_logprobs(
                    token_ids=final_output.outputs[0].token_ids,
                    top_logprobs=final_output.outputs[0].logprobs,
                )
                if upstream_request.logprobs
                else None,
            )
        ]
        return response

    monkeypatch.setattr(
        _OpenAIServingChat,
        "chat_completion_full_generator",
        format_response,
        raising=False,
    )

    async def results() -> AsyncIterator[Any]:
        for token, probability in [(7, -0.5)]:
            yield types.SimpleNamespace(
                prompt_token_ids=[1, 2],
                outputs=[
                    types.SimpleNamespace(
                        index=0,
                        token_ids=[token],
                        logprobs=[
                            {}
                            if invalid_raw
                            else {token: types.SimpleNamespace(logprob=probability)}
                        ],
                        text="tail",
                        finish_reason="stop",
                        stop_reason=None,
                    )
                ],
            )

    if invalid_raw:
        with pytest.raises(RuntimeError, match="did not include the selected token"):
            await serving.chat_completion_full_generator(request, results())
        assert builder.call_count == 0
        return

    response = await serving.chat_completion_full_generator(request, results())
    choice = response.choices[0]
    assert (choice.logprobs is None) == (bypass or not logprobs)
    assert builder.call_count == (0 if bypass or not logprobs else 1)
    assert vars(request) == original_request_fields
    if logprobs and (token_ids if token_ids is not None else default_token_ids):
        assert choice.message.prompt_token_ids == [1, 2]
        assert choice.message.generation_token_ids == [7]
        assert choice.message.generation_log_probs == [-0.5]


@pytest.mark.parametrize(
    "serving_chat_kwargs",
    [
        {"default_chat_template_kwargs": {"enable_thinking": False}},
        {"chat_template_kwargs": {"enable_thinking": False}},
    ],
    ids=["native-name", "legacy-name"],
)
def test_both_spellings_reach_all_three_consumers(monkeypatch, serving_chat_kwargs):
    renderer, serving_chat, tokenization = _build_server(
        monkeypatch, dict(serving_chat_kwargs)
    )

    expected = {"enable_thinking": False}
    assert renderer[0].kwargs["default_chat_template_kwargs"] == expected
    assert serving_chat[0].kwargs["default_chat_template_kwargs"] == expected
    assert tokenization[0].kwargs["default_chat_template_kwargs"] == expected


def test_legacy_spelling_does_not_survive_into_serving_chat(monkeypatch):
    """The legacy key must be renamed, not merely read.

    The kwargs bag is splatted into OpenAIServingChat, which rejects an
    argument it does not declare, so leaving chat_template_kwargs behind is a
    TypeError at construction.
    """
    _, serving_chat, _ = _build_server(
        monkeypatch, {"chat_template_kwargs": {"enable_thinking": False}}
    )

    assert "chat_template_kwargs" not in serving_chat[0].kwargs


def test_native_spelling_wins_and_legacy_is_dropped(monkeypatch):
    """Both spellings present: native wins, legacy is removed.

    Reading these as ``pop(native) or pop(legacy)`` short-circuits on a truthy
    native value and leaves the legacy key in the bag.
    """
    _, serving_chat, _ = _build_server(
        monkeypatch,
        {
            "default_chat_template_kwargs": {"enable_thinking": True},
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    chat_kwargs = serving_chat[0].kwargs
    assert chat_kwargs["default_chat_template_kwargs"] == {"enable_thinking": True}
    assert "chat_template_kwargs" not in chat_kwargs


def test_absent_kwargs_render_as_empty_dict(monkeypatch):
    """Neither spelling given: consumers get {} rather than None.

    preprocess_chat splats this, so None raises instead of letting the template
    apply its own defaults.
    """
    renderer, _, tokenization = _build_server(monkeypatch, {})

    assert renderer[0].kwargs["default_chat_template_kwargs"] == {}
    assert tokenization[0].kwargs["default_chat_template_kwargs"] == {}
