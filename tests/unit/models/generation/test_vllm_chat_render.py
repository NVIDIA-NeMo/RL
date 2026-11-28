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

"""The disaggregation frontend's renderer runs the engine's render and splice."""

import asyncio
import copy
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.vllm
pytest.importorskip("vllm")

from nemo_rl.models.generation.vllm import (  # noqa: E402
    chat_server,
    prompt_and_capture,
)
from nemo_rl.models.generation.vllm.chat_render import (  # noqa: E402
    FrontendChatRenderer,
    FrontendRender,
)


class _OnlineRenderer:
    """Stands in for chat_server.FrontendRenderer: records what it rendered."""

    chat_template = None
    chat_template_content_format = "auto"
    default_chat_template_kwargs = {"enable_thinking": True}
    parser = None
    exclude_tools_when_tool_choice_none = False

    def __init__(self):
        self.calls = []

    async def _render_chat_call(self, request, messages, render_kwargs, *, admission):
        self.calls.append((request, copy.deepcopy(messages), render_kwargs, admission))
        # vLLM's renderer rewrites the conversation in place; emulate it.
        for message in messages:
            message["content"] = str(message.get("content", "")).upper()
        prefix = request.required_prefix_token_ids
        return chat_server.RenderedChat(
            template_token_ids=[10, 11, 12, 13],
            history_token_ids=None if prefix is None else [10, 11],
            required_prefix_token_ids=prefix,
            admission=None,
            serving_admission=None,
            verify_retained_media=False,
            request_max_tokens=None,
            reasoning_ended=False,
            conversation=[],
            engine_prompt={},
        )


@pytest.fixture
def renderer():
    r = FrontendChatRenderer.__new__(FrontendChatRenderer)
    r._settings = SimpleNamespace(model_name="the-model")
    r._tokenizer = object()
    r._model_config = SimpleNamespace(max_model_len=64)
    r._turn_end_token_ids = frozenset({0})
    r._online_renderer = _OnlineRenderer()
    return r


@pytest.fixture
def spliced(monkeypatch):
    calls = []

    def fake_splice(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(token_ids=[42])

    monkeypatch.setattr(prompt_and_capture, "splice_prefix_tokens", fake_splice)
    return calls


def _body(**extra):
    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "hello",
                "prompt_token_ids": [1, 2],
                "generation_token_ids": [3],
            },
            {"role": "user", "content": "more"},
        ],
        "max_tokens": 8,
        "ng_capture": {"rollout_id": "r"},
        "conversation_params": {"conversation_id": "c"},
        "required_prefix_token_ids": [1, 2, 3],
    }
    body.update(extra)
    return body


def test_render_is_the_engines_render_then_its_splice(renderer, spliced):
    body = _body()
    before = copy.deepcopy(body)

    rendered = asyncio.run(renderer.render(body))

    assert rendered == FrontendRender(token_ids=[42], reasoning_ended=False)
    [(request, messages, render_kwargs, admission)] = renderer._online_renderer.calls
    # The engine's own request model, so its fields shape the render alike.
    assert isinstance(request, chat_server.NeMoRLChatCompletionRequest)
    assert request.model == "m"
    assert [message["content"] for message in messages] == ["hi", "hello", "more"]
    # No capture admission is resolved here: the decode engine captures.
    assert request.ng_capture is None and admission is None
    assert render_kwargs["default_template_kwargs"] == {"enable_thinking": True}
    assert render_kwargs["tool_dicts"] is None
    [splice] = spliced
    assert splice["model_prefix_token_ids"] == [1, 2, 3]
    assert splice["template_prefix_token_ids"] == [10, 11]
    assert splice["template_token_ids"] == [10, 11, 12, 13]
    assert splice["eos_token_id"] == frozenset({0})
    # The caller's body survives vLLM's in-place rewrite.
    assert body == before


def test_render_derives_the_prefix_from_gym_token_fields(renderer, spliced):
    body = _body()
    del body["required_prefix_token_ids"]
    assert asyncio.run(renderer.render(body)).token_ids == [42]
    assert spliced[0]["model_prefix_token_ids"] == [1, 2, 3]


def test_render_without_a_prefix_splices_nothing(renderer, spliced):
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    assert asyncio.run(renderer.render(body)).token_ids == [10, 11, 12, 13]
    assert spliced == []
    [(request, *_)] = renderer._online_renderer.calls
    assert request.model == "the-model"
