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
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.models.generation.vllm import utils as vllm_utils
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


def _build_server(monkeypatch, serving_chat_kwargs, *, capture_requests=()):
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
    worker._capture_calls = {id(request): None for request in capture_requests}
    worker.routed_experts_dtype = torch.int16

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


@pytest.mark.vllm
def test_capture_bypasses_upstream_route_encoding_without_changing_plain_requests(
    monkeypatch,
):
    """Run the installed vLLM response generator with interleaved requests."""
    # Optional runtime: the existing scaffold replaces server initialization,
    # but this regression must execute vLLM's real superclass conversion.
    serving = pytest.importorskip("vllm.entrypoints.openai.chat_completion.serving")
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
    )
    from vllm.outputs import CompletionOutput, RequestOutput

    class ServingWithoutEngine(serving.OpenAIServingChat):
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            _BUILT["chat"].append(self)
            self.response_role = "assistant"
            self.return_tokens_as_token_ids = True
            self.parser_cls = None
            self.enable_auto_tools = False
            self._include_reasoning_tokens_details = False
            self.enable_prompt_tokens_details = False
            self.enable_per_request_metrics = False
            self.enable_log_outputs = False
            self.system_fingerprint = None

    real_install = _install_fake_vllm

    def install_with_real_response_type(monkeypatch):
        real_install(monkeypatch)
        monkeypatch.setattr(
            sys.modules["vllm.entrypoints.openai.chat_completion.protocol"],
            "ChatCompletionResponse",
            ChatCompletionResponse,
        )

    monkeypatch.setattr(
        sys.modules[__name__], "_OpenAIServingChat", ServingWithoutEngine
    )
    monkeypatch.setattr(
        sys.modules[__name__], "_install_fake_vllm", install_with_real_response_type
    )
    upstream_encode = MagicMock(wraps=serving.numpy2base64)
    rl_encode = MagicMock(wraps=vllm_utils.encode_routed_experts)
    monkeypatch.setattr(serving, "numpy2base64", upstream_encode)
    monkeypatch.setattr(vllm_utils, "encode_routed_experts", rl_encode)

    requests = [
        ChatCompletionRequest(
            model="model", messages=[{"role": "user", "content": "x"}], logprobs=False
        )
        for _ in range(2)
    ]
    _, chats, _ = _build_server(monkeypatch, {}, capture_requests=[requests[0]])
    chat = chats[0]
    results = []
    for request_index in range(2):
        per_request = []
        for step in range(2):
            routes = torch.tensor(
                [[[request_index * 10 + step, request_index * 10 + step + 1]]],
                dtype=torch.int16,
            ).numpy()
            output = CompletionOutput(
                index=0,
                text="answer",
                token_ids=[22],
                cumulative_logprob=None,
                logprobs=None,
                routed_experts=routes,
                finish_reason="stop",
            )
            per_request.append(
                RequestOutput(
                    request_id=str(request_index),
                    prompt="x",
                    prompt_token_ids=[10],
                    prompt_logprobs=None,
                    outputs=[output],
                    finished=True,
                )
            )
        results.append(per_request)

    async def respond(index):
        async def generate():
            for result in results[index]:
                await asyncio.sleep(0)
                yield result

        return await chat.chat_completion_full_generator(
            requests[index],
            generate(),
            str(index),
            "model",
            [],
            None,
            types.SimpleNamespace(),
        )

    async def respond_both():
        return await asyncio.gather(respond(0), respond(1))

    capture_response, plain_response = asyncio.run(respond_both())

    # Both upstream's numpy codec and RL's envelope codec run only for the
    # plain request, even though capture is installed on the shared worker.
    upstream_encode.assert_called_once()
    assert upstream_encode.call_args.args[0] is results[1][-1].outputs[0].routed_experts
    rl_encode.assert_called_once()
    assert capture_response.choices[0].routed_experts is None
    native = capture_response.choices[0].message.routed_experts
    torch.testing.assert_close(
        native, torch.tensor([[[1, 2]], [[0, 1]]], dtype=torch.int16)
    )
    assert isinstance(plain_response.choices[0].routed_experts, str)
    assert isinstance(plain_response.choices[0].message.routed_experts, str)
    # The wrapper must not clear the engine's originals or copy their buffers.
    for request_index, per_request in enumerate(results):
        for step, result in enumerate(per_request):
            assert result.outputs[0].routed_experts.tolist() == [
                [[request_index * 10 + step, request_index * 10 + step + 1]]
            ]
