"""Unit tests for the disaggregation frontend's chat renderer (no vLLM needed)."""

import asyncio
import copy
import sys
import types
from types import SimpleNamespace

import pytest

import nemo_rl.models.generation.vllm.chat_render as chat_render
from nemo_rl.models.generation.vllm.chat_render import (
    FrontendChatRenderer,
    last_assistant_prefix,
)


def test_last_assistant_prefix_stops_at_the_last_assistant_message():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
    ]
    assert last_assistant_prefix(messages) == messages[:5]
    assert last_assistant_prefix(messages[:2]) == messages[:2]
    assert last_assistant_prefix([]) == []


class _FakeChatCompletionRequest:
    """Stands in for vLLM's request model: records the fields it was built from."""

    built = []
    # The renderer forwards exactly the body keys the request model declares.
    model_fields = dict.fromkeys(
        (
            "model",
            "messages",
            "tools",
            "tool_choice",
            "chat_template",
            "chat_template_kwargs",
            "add_generation_prompt",
            "max_tokens",
            "max_completion_tokens",
            "reasoning_effort",
        )
    )

    def __init__(self, **fields):
        self.fields = fields
        self.add_generation_prompt = fields.get("add_generation_prompt", True)
        type(self).built.append(fields)


@pytest.fixture
def renderer(monkeypatch):
    # The renderer imports vLLM's request model lazily; provide a stand-in.
    protocol = types.ModuleType("vllm.entrypoints.openai.chat_completion.protocol")
    protocol.ChatCompletionRequest = _FakeChatCompletionRequest
    for name in (
        "vllm",
        "vllm.entrypoints",
        "vllm.entrypoints.openai",
        "vllm.entrypoints.openai.chat_completion",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, protocol.__name__, protocol)
    _FakeChatCompletionRequest.built = []

    r = FrontendChatRenderer.__new__(FrontendChatRenderer)
    r._settings = SimpleNamespace(model_name="the-model")
    r._tokenizer = object()
    r.renders = []

    async def _render(request, messages, *, add_generation_prompt):
        r.renders.append((copy.deepcopy(messages), add_generation_prompt))
        # vLLM's renderer rewrites the conversation in place; emulate it.
        for m in messages:
            m["content"] = str(m.get("content", "")).upper()
        return [100 + len(messages), int(add_generation_prompt)]

    r._render = _render
    return r


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
        "required_prefix_token_ids": [1, 2, 3],
    }
    body.update(extra)
    return body


def test_prompt_token_ids_two_render_splice(renderer, monkeypatch):
    spliced = []

    def fake_splice(**kw):
        spliced.append(kw)
        return SimpleNamespace(token_ids=[42])

    monkeypatch.setattr(chat_render, "splice_prefix_tokens", fake_splice)
    body = _body()
    before = copy.deepcopy(body)

    ids = asyncio.run(renderer.prompt_token_ids(body))

    assert ids == [42]
    # Full render with the generation prompt, then the prefix up to the last
    # assistant message without it.
    assert [(len(m), gen) for m, gen in renderer.renders] == [(3, True), (2, False)]
    assert spliced[0]["model_prefix_token_ids"] == [1, 2, 3]
    assert spliced[0]["template_token_ids"] == [103, 1]
    assert spliced[0]["template_prefix_token_ids"] == [102, 0]
    # Only the request model's own fields reach it; Gym-only ones do not.
    fields = _FakeChatCompletionRequest.built[0]
    assert "ng_capture" not in fields and "required_prefix_token_ids" not in fields
    assert fields["model"] == "m"
    # The output budget is 1 while rendering, as on the engine.
    assert fields["max_tokens"] == 1 and fields["max_completion_tokens"] is None
    # The caller's body survives vLLM's in-place rewrite.
    assert body == before


def test_prompt_token_ids_derives_the_prefix_from_gym_token_fields(
    renderer, monkeypatch
):
    spliced = []
    monkeypatch.setattr(
        chat_render,
        "splice_prefix_tokens",
        lambda **kw: spliced.append(kw) or SimpleNamespace(token_ids=[7]),
    )
    body = _body()
    del body["required_prefix_token_ids"]
    assert asyncio.run(renderer.prompt_token_ids(body)) == [7]
    assert spliced[0]["model_prefix_token_ids"] == [1, 2, 3]


def test_prompt_token_ids_without_a_prefix_is_a_single_render(renderer, monkeypatch):
    monkeypatch.setattr(
        chat_render, "splice_prefix_tokens", lambda **kw: pytest.fail("no splice")
    )
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    assert asyncio.run(renderer.prompt_token_ids(body)) == [101, 1]
    assert _FakeChatCompletionRequest.built[0]["model"] == "the-model"
    assert len(renderer.renders) == 1


def test_prompt_token_ids_forwards_declared_fields_and_leaves_the_budget_alone(
    renderer, monkeypatch
):
    monkeypatch.setattr(
        chat_render, "splice_prefix_tokens", lambda **kw: pytest.fail("no splice")
    )
    body = {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "low"}
    asyncio.run(renderer.prompt_token_ids(body))
    fields = _FakeChatCompletionRequest.built[0]
    assert fields["reasoning_effort"] == "low"
    assert "max_tokens" not in fields and "max_completion_tokens" not in fields
