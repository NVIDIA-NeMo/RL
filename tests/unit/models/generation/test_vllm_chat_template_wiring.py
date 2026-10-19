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
module tree and inspect what each consumer was constructed with.
"""

import asyncio
import sys
import types
from copy import deepcopy
from unittest.mock import MagicMock

import pytest

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
    module("vllm.renderers.hf", HfRenderer=placeholder("HfRenderer"))
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


def _build_server(monkeypatch, serving_chat_kwargs):
    """Run the real server setup and hand back the three consumer stubs."""
    _install_fake_vllm(monkeypatch)

    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
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


@pytest.mark.parametrize(
    "mode",
    [
        "root",
        "candidate_continuation",
        "candidate_rewrite",
        "direct_prefix",
        "empty_prefix",
    ],
)
def test_media_marker_escape_is_discarded_or_replaced_by_exact_prefix(
    monkeypatch, mode
):
    """Exercise the real renderer override, including candidate-root fallback."""
    calls = []
    captures = []
    model_prefix = [900, 18, 2]
    messages = [
        {"role": "assistant", "content": "Count the <image> tags"},
        {"role": "tool", "content": "observation"},
        {"role": "user", "content": "new <image>"},
    ]
    original = deepcopy(messages)

    def render(items):
        tokens = []
        for message in items:
            if message["role"] == "assistant":
                tokens += [70, 18 if "<image>" in message["content"] else 50, 2]
            elif message["role"] == "tool":
                tokens += [80, 81, 2]
            else:
                tokens += [90, 18, 2]
        return tokens

    async def preprocess(_self, *, messages, **_kwargs):
        calls.append(deepcopy(messages))
        return messages, [{"prompt_token_ids": render(messages)}]

    monkeypatch.setattr(_OnlineRenderer, "preprocess_chat", preprocess, raising=False)
    monkeypatch.setattr(
        VllmAsyncGenerationWorkerImpl, "_capture_media", True, raising=False
    )
    admission = (
        types.SimpleNamespace(mode="candidate", prev_len=len(model_prefix))
        if mode.startswith("candidate")
        else None
    )
    monkeypatch.setattr(
        VllmAsyncGenerationWorkerImpl, "_capture_admission", lambda *_: admission
    )
    monkeypatch.setattr(
        VllmAsyncGenerationWorkerImpl,
        "_resolve_admission_prefix",
        lambda *_: model_prefix,
    )
    monkeypatch.setattr(
        VllmAsyncGenerationWorkerImpl,
        "_enter_request_prefix",
        lambda _self, request, prefix: setattr(
            request, "required_prefix_token_ids", prefix
        ),
    )
    monkeypatch.setattr(
        VllmAsyncGenerationWorkerImpl,
        "_capture_request_media",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        VllmAsyncGenerationWorkerImpl,
        "_begin_request_capture",
        lambda _self, _request, ids, **_kwargs: captures.append(list(ids)),
    )

    def decide(_admission, *, messages, **_kwargs):
        assert messages == original
        serving = (
            types.SimpleNamespace(mode="token_in")
            if mode == "candidate_continuation"
            else None
        )
        return types.SimpleNamespace(
            storage=serving, serving=serving, verify_retained_media=True
        )

    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async.decide_capture_input", decide
    )
    renderer = _build_server(monkeypatch, {})[0][0]
    renderer.renderer = types.SimpleNamespace(
        tokenizer=types.SimpleNamespace(eos_token_id=2)
    )

    class Request(types.SimpleNamespace):
        def model_copy(self, *, update):
            return Request(**(vars(self) | update))

    prefix = (
        model_prefix
        if mode == "direct_prefix"
        else []
        if mode == "empty_prefix"
        else None
    )
    request = Request(required_prefix_token_ids=prefix)
    result = asyncio.run(
        renderer.preprocess_chat(
            request=request,
            messages=messages,
            default_template=None,
            default_template_content_format="auto",
            default_template_kwargs={},
        )
    )
    assert messages == original
    actual_ids = result[1][0]["prompt_token_ids"]
    if mode in ("candidate_continuation", "direct_prefix"):
        assert all("<image>" not in call[0]["content"] for call in calls)
        assert actual_ids == model_prefix + render(original[1:])
        assert captures == [actual_ids]
    else:
        assert calls[0 if mode == "empty_prefix" else -1] == original
        assert actual_ids == render(original)
        assert captures == [actual_ids]
        if mode == "candidate_rewrite":
            assert "<image>" not in calls[0][0]["content"]
            assert len(calls) == 2
