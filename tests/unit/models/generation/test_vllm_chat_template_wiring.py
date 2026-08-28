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
from dataclasses import dataclass
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
    module(
        "vllm.entrypoints.openai.engine.protocol",
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
    worker.llm_async_engine_args = MagicMock()
    worker.llm_async_engine_args.create_model_config.return_value = MagicMock(
        served_model_name="served-model", model="model-path"
    )

    worker._setup_vllm_openai_api_server(_FakeApp())
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


# ---------------------------------------------------------------------------
# preprocess_chat: multimodal placeholders after exact-prefix replacement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PlaceholderRange:
    """Mirrors vllm.multimodal.inputs.PlaceholderRange (v0.25.1)."""

    offset: int
    length: int
    is_embed: object | None = None


class _Request:
    """Enough of ChatCompletionRequest for the prefix-replacement branch."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def model_copy(self, update):
        merged = dict(self.__dict__)
        merged.update(update)
        return _Request(**merged)


# eos = 2. History re-tokenizes 1002,1003 as the single token 1001, so the
# exact-token prompt is one longer than the template render and every media
# span after the splice point slides by one.
_TEMPLATE_PREFIX = [7, 8, 1001, 2]
_TEMPLATE_TOKENS = [7, 8, 1001, 2, 30, 90, 90, 90, 31]
_MODEL_PREFIX = [7, 8, 1002, 1003, 2]
_EXACT_TOKENS = [7, 8, 1002, 1003, 2, 30, 90, 90, 90, 31]


def _renderer_with_stubbed_super(monkeypatch, mm_placeholders):
    """Build the real NeMoRLOnlineRenderer over a scripted vLLM base class."""
    renderer, _, _ = _build_server(monkeypatch, {})
    instance = renderer[0]
    instance.renderer = MagicMock(tokenizer=MagicMock(eos_token_id=2))

    engine_prompt = {"prompt_token_ids": list(_TEMPLATE_TOKENS)}
    if mm_placeholders is not None:
        engine_prompt["mm_placeholders"] = mm_placeholders

    async def fake_preprocess_chat(self, *, request, messages, **_kwargs):
        # The history-only call passes the truncated message list.
        if len(messages) == len(_MESSAGES):
            return [], [engine_prompt]
        return [], [{"prompt_token_ids": list(_TEMPLATE_PREFIX)}]

    monkeypatch.setattr(
        _OnlineRenderer, "preprocess_chat", fake_preprocess_chat, raising=False
    )
    return instance, engine_prompt


_MESSAGES = [
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "reply"},
    {"role": "user", "content": "look at this"},
]


async def _run_preprocess_chat(instance):
    return await instance.preprocess_chat(
        request=_Request(required_prefix_token_ids=list(_MODEL_PREFIX)),
        messages=[dict(m) for m in _MESSAGES],
        default_template=None,
        default_template_content_format="auto",
        default_template_kwargs={},
    )


def test_preprocess_chat_moves_mm_placeholders_onto_the_exact_token_prompt(monkeypatch):
    """The remap must actually land on engine_prompt, under vLLM's key name.

    remap_multimodal_placeholders is unit-tested in isolation, but the whole
    repair is a no-op if the call site reads a key vLLM does not populate --
    ``mm_placeholders`` on MultiModalInput (vllm/inputs/engine.py, v0.25.1).
    """
    instance, engine_prompt = _renderer_with_stubbed_super(
        monkeypatch, {"image": [_PlaceholderRange(offset=5, length=3)]}
    )

    asyncio.run(_run_preprocess_chat(instance))

    assert engine_prompt["prompt_token_ids"] == _EXACT_TOKENS
    assert [r.offset for r in engine_prompt["mm_placeholders"]["image"]] == [6]
    assert [r.length for r in engine_prompt["mm_placeholders"]["image"]] == [3]


def test_preprocess_chat_leaves_text_only_prompts_without_mm_placeholders(monkeypatch):
    """A text-only request must not grow an empty mm_placeholders key."""
    instance, engine_prompt = _renderer_with_stubbed_super(monkeypatch, None)

    asyncio.run(_run_preprocess_chat(instance))

    assert engine_prompt["prompt_token_ids"] == _EXACT_TOKENS
    assert "mm_placeholders" not in engine_prompt
