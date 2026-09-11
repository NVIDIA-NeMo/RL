# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe/generation parity through actual worker setup and renderer overrides.

vLLM rendering is a deterministic dependency double. Worker prefix resolution,
EOS splicing, Gym capture, and TQ codecs are real; storage uses the existing
reference backend. No engine execution or GPU qualification is implied.
"""

import asyncio
import builtins
import copy
import json
import sys
import threading
from types import SimpleNamespace
from typing import Any, get_args
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ConfigDict

from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)
from tests.unit.models.generation import test_vllm_chat_template_wiring as wiring

HISTORY = [
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "<think>reasoning</think>answer"},
    {"role": "user", "content": "next question"},
]
PREFIX = [10, 2, 99, 98, 30, 2]


class _Request(BaseModel):
    model_config = ConfigDict(extra="allow")
    messages: list[dict[str, Any]]
    add_generation_prompt: bool = True
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    chat_template: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    def build_chat_params(self, default_template, default_template_content_format):
        return SimpleNamespace(
            chat_template_kwargs=dict(self.chat_template_kwargs or {})
        )


class _ChatRequest(_Request):
    """Model the pinned generation/tokenize distinction, not one shared fake."""

    documents: list[dict[str, str]] | None = None
    reasoning_effort: str | None = None
    return_assistant_tokens_mask: bool = False

    def build_chat_params(self, default_template, default_template_content_format):
        values = dict(self.chat_template_kwargs or {})
        values.update(documents=self.documents, reasoning_effort=self.reasoning_effort)
        if self.reasoning_effort is not None and "enable_thinking" not in values:
            values["enable_thinking"] = self.reasoning_effort != "none"
        return SimpleNamespace(chat_template_kwargs=values)


class _TokenizeResponse(BaseModel):
    count: int
    tokens: list[int]


async def render(renderer: Any, request: Any) -> dict:
    result = await renderer.preprocess_chat(
        request=request,
        messages=copy.deepcopy(request.messages),
        default_template=None,
        default_template_content_format="auto",
        default_template_kwargs={"enable_thinking": True},
    )
    return result[1][0]


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    wiring._install_fake_vllm(monkeypatch)
    sys.modules[
        "vllm.entrypoints.openai.chat_completion.protocol"
    ].ChatCompletionRequest = _ChatRequest
    protocol = sys.modules["vllm.entrypoints.serve.tokenize.protocol"]
    protocol.TokenizeChatRequest = _Request
    protocol.TokenizeResponse = _TokenizeResponse

    async def template_render(
        self: Any, *, request: Any, messages: list[dict], **kwargs: Any
    ) -> tuple[list[dict], list[dict]]:
        assert kwargs["default_template_kwargs"] == {"enable_thinking": True}
        if any(message["role"] == "assistant" for message in messages):
            # Qwen-style behavior: historical reasoning disappears when the
            # final message is a user, so the corresponding prefix is NOT an
            # exact token prefix of the full template. EOS counts still align.
            tokens = [10, 2, 30, 2, 40, 50] if request.add_generation_prompt else PREFIX
        else:
            tokens = [10, 50]
        # As in the real renderer, request.build_chat_params contributes template
        # inputs. Upstream tokenize intentionally lacks generation-only kwargs.
        parameters = request.build_chat_params(None, "auto").chat_template_kwargs
        if parameters.get("documents"):
            tokens += [700] * len(parameters["documents"][0]["text"])
        if parameters.get("reasoning_effort") is not None:
            tokens += [800, int(parameters["enable_thinking"])]
        return messages, [{"prompt_token_ids": list(tokens)}]

    async def tokenize(self: Any, request: Any, raw_request: Any) -> _TokenizeResponse:
        prompt = await render(self.kwargs["online_renderer"], request)
        return _TokenizeResponse(
            count=len(prompt["prompt_token_ids"]), tokens=prompt["prompt_token_ids"]
        )

    monkeypatch.setattr(
        wiring._OnlineRenderer, "preprocess_chat", template_render, raising=False
    )
    monkeypatch.setattr(
        wiring._OnlineRenderer,
        "renderer",
        SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=2)),
        raising=False,
    )
    monkeypatch.setattr(
        wiring._OnlineRenderer,
        "model_config",
        SimpleNamespace(max_model_len=128),
        raising=False,
    )
    monkeypatch.setattr(
        wiring._ServingTokenization, "create_tokenize", tokenize, raising=False
    )
    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = {
        "temperature": 1.0,
        "top_p": 1.0,
        "vllm_cfg": {
            "http_server_serving_chat_kwargs": {
                "default_chat_template_kwargs": {"enable_thinking": True}
            }
        },
    }
    worker.llm = MagicMock(model_config="model-config", renderer="renderer")
    worker._http_engine_client = MagicMock(
        model_config="model-config", renderer="renderer"
    )
    worker.llm_async_engine_args = MagicMock()
    worker.token_capture = None
    worker._capture_calls = {}
    worker._cc_capture_enabled = False
    worker._prefix_cache = {}
    worker._prefix_cache_lock = threading.Lock()
    worker._staging_source = None
    app = wiring._FakeApp()
    worker._setup_vllm_openai_api_server(app)
    routes = dict(app.routes)
    return SimpleNamespace(
        worker=worker,
        renderer=wiring._BUILT["renderer"][0],
        chat_type=routes["/v1/chat/completions"].__annotations__["request"],
        probe_type=get_args(routes["/tokenize"].__annotations__["request"])[1],
        tokenize=routes["/tokenize"],
    )


@pytest.fixture
def captured_server(
    server: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    # These optional imports belong only to the Gym capture fixture; the ordinary
    # server fixture/test below deliberately operates with Gym imports forbidden.
    from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter
    from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture
    from nemo_gym.token_id_capture.staging.records import CaptureAdmission

    from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
    from nemo_rl.data_plane.tq_token_sink import (
        STAGING_FIELDS,
        TQTokenSink,
        TQTokenSource,
    )

    data_plane = NoOpDataPlaneClient()
    data_plane.register_partition("staged", STAGING_FIELDS, 16, ["finalize"])
    capture = RolloutTokenCapture(
        sink=TQTokenSink(data_plane, staging_partition="staged"),
        weight_version_fn=lambda: 7,
        adapter=VLLMCaptureAdapter(),
    )
    parent = capture.begin_call(
        CaptureAdmission(rollout_id="scope", model_call_id="parent", mode="text")
    )
    coords = capture.complete_call(
        parent,
        prompt_token_ids=PREFIX[:2],
        generated_token_ids=PREFIX[2:],
        generated_logprobs=[-0.25] * 4,
    )
    assert coords.disposition == "staged"
    server.worker.token_capture = capture
    server.worker._staging_source = TQTokenSource(
        data_plane, staging_partition="staged"
    )
    server.data_plane = data_plane
    server.admission = CaptureAdmission(
        rollout_id="scope",
        model_call_id="next",
        parent_call_id="parent",
        mode="token_in",
        prev_len=len(PREFIX),
        staging_chain=["scope/parent"],
        parent_chain_hash=coords.chain_hash,
    ).model_dump()
    server.begin_call = MagicMock(wraps=capture.begin_call)
    monkeypatch.setattr(capture, "begin_call", server.begin_call)
    server.put_samples = MagicMock(wraps=data_plane.put_samples)
    monkeypatch.setattr(data_plane, "put_samples", server.put_samples)
    return server


@pytest.mark.nemo_gym
@pytest.mark.parametrize(
    "effort,thinking", [("high", None), ("none", None), ("high", False)]
)
def test_probe_uses_generation_document_and_reasoning_template_options(
    captured_server: SimpleNamespace, effort: str, thinking: bool | None
) -> None:
    server = captured_server
    options = {
        "messages": HISTORY[:1],
        "documents": [{"title": "source", "text": "abcd"}],
        "reasoning_effort": effort,
        "chat_template_kwargs": {}
        if thinking is None
        else {"enable_thinking": thinking},
    }
    probe = server.probe_type(**options, ng_prefix_staging_chain=[], ng_prefix_len=0)
    measured = asyncio.run(render(server.renderer, probe))
    assert server.worker._capture_calls == {}
    assert not server.put_samples.called
    generated = asyncio.run(render(server.renderer, server.chat_type(**options)))
    expected_thinking = effort != "none" if thinking is None else thinking
    assert (
        measured["prompt_token_ids"]
        == generated["prompt_token_ids"]
        == [10, 50, 700, 700, 700, 700, 800, int(expected_thinking)]
    )
    # Ordinary tokenize keeps its upstream request semantics, not this opt-in.
    ordinary = asyncio.run(render(server.renderer, server.probe_type(**options)))
    assert ordinary["prompt_token_ids"] == [10, 50]


@pytest.mark.nemo_gym
@pytest.mark.parametrize("probe", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_renderer_relocates_media_before_replacing_prompt(
    captured_server: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    probe: bool,
    missing: bool,
) -> None:
    server = captured_server
    original = {"image": [{"offset": 2, "length": 2, "is_embed": [True, True]}]}

    async def media_render(self: Any, *, request: Any, messages: list, **kwargs: Any):
        # History re-rendering changed the text length before the retained image.
        tokens = [10, 2, 18, 18, 30, 2]
        if request.add_generation_prompt:
            tokens += [40, 50]
        return messages, [{"prompt_token_ids": tokens, "mm_placeholders": original}]

    monkeypatch.setattr(wiring._OnlineRenderer, "preprocess_chat", media_render)
    prefix = [10, 99, 2, 17 if missing else 18, 18, 30, 2]
    if probe:
        monkeypatch.setattr(
            server.worker, "_resolve_probe_prefix", lambda request: prefix
        )
        request = server.probe_type(
            messages=HISTORY,
            ng_prefix_staging_chain=["scope/parent"],
            ng_prefix_len=len(prefix),
        )
    else:
        request = server.chat_type(messages=HISTORY, required_prefix_token_ids=prefix)
    if missing:
        with pytest.raises(ValueError, match="Could not locate image"):
            asyncio.run(render(server.renderer, request))
    else:
        result = asyncio.run(render(server.renderer, request))
        assert result["prompt_token_ids"] == prefix + [40, 50]
        assert result["mm_placeholders"]["image"][0] == {
            **original["image"][0],
            "offset": 3,
        }
    assert original["image"][0]["offset"] == 2
    assert server.worker._capture_calls == {}


@pytest.mark.nemo_gym
@pytest.mark.parametrize("mode", ["root", "history_root", "continuation"])
def test_probe_matches_generation_prepared_prompt_without_capture_mutation(
    captured_server: SimpleNamespace, mode: str
) -> None:
    server = captured_server
    messages = HISTORY[:1] if mode == "root" else HISTORY
    continuation = mode == "continuation"
    probe = server.probe_type(
        messages=messages,
        ng_prefix_staging_chain=["scope/parent"] if continuation else [],
        ng_prefix_len=len(PREFIX) if continuation else 0,
    )
    measured = asyncio.run(render(server.renderer, probe))
    assert measured["prompt_token_ids"] == (
        PREFIX + [40, 50]
        if continuation
        else [10, 2, 30, 2, 40, 50]
        if mode == "history_root"
        else [10, 50]
    )
    assert server.worker._capture_calls == {}
    server.begin_call.assert_not_called()
    server.put_samples.assert_not_called()
    generation = server.chat_type(
        messages=messages,
        ng_capture=server.admission
        if continuation
        else {"rollout_id": "root", "model_call_id": "root", "mode": "text"},
    )
    generated = asyncio.run(render(server.renderer, generation))
    assert generated == measured
    assert (
        server.worker._capture_calls[id(generation)][1] == measured["prompt_token_ids"]
    )
    server.begin_call.assert_called_once()
    server.put_samples.assert_not_called()


@pytest.mark.nemo_gym
def test_probe_endpoint_returns_counts_only_and_ordinary_tokenize_keeps_tokens(
    captured_server: SimpleNamespace,
) -> None:
    server = captured_server
    probe = server.probe_type(
        messages=HISTORY,
        ng_prefix_staging_chain=["scope/parent"],
        ng_prefix_len=len(PREFIX),
    )
    response = asyncio.run(server.tokenize(probe, None))
    assert json.loads(response.body) == {"prompt_token_count": 8, "ng_prefix_len": 6}
    ordinary = asyncio.run(server.tokenize(server.probe_type(messages=HISTORY), None))
    assert json.loads(ordinary.body) == {"count": 6, "tokens": [10, 2, 30, 2, 40, 50]}
    assert server.worker._capture_calls == {}
    server.begin_call.assert_not_called()
    server.put_samples.assert_not_called()


@pytest.mark.nemo_gym
@pytest.mark.parametrize(
    "failure",
    [
        "missing_length",
        "length_mismatch",
        "missing_row",
        "negative_token",
        "inline_prefix",
        "capture_context",
        "disabled_capture",
        "bool_length",
        "float_length",
        "negative_length",
        "nonstring_key",
        "empty_key",
        "root_length",
        "message_inline_prefix",
    ],
)
def test_probe_rejects_invalid_prefix_without_admission_or_write(
    captured_server: SimpleNamespace, failure: str
) -> None:
    server = captured_server
    body = {
        "messages": HISTORY,
        "ng_prefix_staging_chain": ["scope/parent"],
        "ng_prefix_len": len(PREFIX),
    }
    if failure == "missing_length":
        body.pop("ng_prefix_len")
    elif failure == "length_mismatch":
        body["ng_prefix_len"] += 1
    elif failure == "missing_row":
        body["ng_prefix_staging_chain"] = ["scope/missing"]
    elif failure == "negative_token":
        server.data_plane._partitions["staged"].rows["scope/parent"]["token_ids_delta"][
            0
        ] = -1
    elif failure == "inline_prefix":
        body["required_prefix_token_ids"] = PREFIX
    elif failure == "capture_context":
        body["ng_capture"] = server.admission
    elif failure == "disabled_capture":
        server.worker.token_capture = None
    elif failure == "message_inline_prefix":
        body["messages"] = copy.deepcopy(HISTORY)
        body["messages"][1].update(
            prompt_token_ids=PREFIX[:2], generation_token_ids=PREFIX[2:]
        )
    else:
        field, value = {
            "bool_length": ("ng_prefix_len", True),
            "float_length": ("ng_prefix_len", 6.0),
            "negative_length": ("ng_prefix_len", -1),
            "nonstring_key": ("ng_prefix_staging_chain", [123]),
            "empty_key": ("ng_prefix_staging_chain", [""]),
            "root_length": ("ng_prefix_staging_chain", []),
        }[failure]
        body[field] = value
    with pytest.raises((ValueError, KeyError)):
        asyncio.run(render(server.renderer, server.probe_type(**body)))
    assert server.worker._capture_calls == {}
    server.begin_call.assert_not_called()
    server.put_samples.assert_not_called()


@pytest.mark.parametrize("prefix", [None, PREFIX])
def test_ordinary_non_gym_rendering_does_not_import_gym(
    server: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, prefix: list[int] | None
) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "nemo_gym" or name.startswith("nemo_gym."):
            raise AssertionError("ordinary rendering must not import Gym")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    expected = PREFIX + [40, 50] if prefix else [10, 2, 30, 2, 40, 50]
    request = server.probe_type(messages=HISTORY, required_prefix_token_ids=prefix)
    assert asyncio.run(render(server.renderer, request))["prompt_token_ids"] == expected
    generation = server.chat_type(messages=HISTORY, required_prefix_token_ids=prefix)
    assert (
        asyncio.run(render(server.renderer, generation))["prompt_token_ids"] == expected
    )
    assert server.worker._capture_calls == {}
    assert "ng_prefix_staging_chain" not in server.chat_type.model_fields
    assert "ng_prefix_len" not in server.chat_type.model_fields
