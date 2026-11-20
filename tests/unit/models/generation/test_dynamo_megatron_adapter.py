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
"""A NeMo-Gym call through the frontend's chat engine to a Megatron engine's adapter.

Dynamo and Megatron-LM's Dynamo adapter are faked: the chat engine runs over a
fake render pool and router, and ``NemoRLMegatronLLMEngine`` over a fake
``MegatronLLMEngine`` whose legs replay scripted engine streams.
"""

import asyncio
import dataclasses
import importlib
import importlib.util
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_rl.models.generation.dynamo.render_pool import RenderedCall
from nemo_rl.models.generation.gym_call import GYM_EXTRAS_ANNOTATION_PREFIX

pytestmark = pytest.mark.mcore

GENERATION = {
    "model_name": "Qwen/Qwen3-0.6B",
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": None,
    "mcore_generation_config": {
        "expose_http_server": True,
        "parsers": [],
        "max_model_len": 16,
    },
}
EOS = 2
TOKEN_IN = {"mode": "token_in", "capture_key": "rollout-1", "model_call_id": "c2"}


class _HttpError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclasses.dataclass
class _EngineConfig:
    model: str
    runtime_data: dict | None = None


class _Stream:
    """A Megatron engine stream: partial items, then the final record."""

    def __init__(self, items):
        self._items = iter(items)
        self.request_id = 11
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self):
        self.closed = True


class _MegatronLLMEngine:
    """Megatron-LM's adapter: its legs read the stream through _stream_chunks."""

    def __init__(self, config, registration_model=None):
        self.config = config
        self.registration_model = registration_model
        self.requests: list[dict] = []
        self.items: list[dict] = []
        self.error: Exception | None = None

    async def start(self, worker_id):
        return _EngineConfig(
            model=self.registration_model,
            runtime_data={"role": self.config.role, "worker_id": worker_id},
        )

    async def generate(self, request, context):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if self.config.role == "prefill":
            yield {
                "token_ids": [],
                "finish_reason": "length",
                "disaggregated_params": {},
            }
            return
        stream = _Stream(self.items)
        async for chunk in self._stream_chunks(stream, request["token_ids"], None):
            yield chunk

    async def _stream_chunks(self, stream, prompt_token_ids, params):
        async for item in stream:
            if "partial" in item:
                yield {
                    "token_ids": item["partial"]["new_tokens"],
                    "log_probs": item["partial"]["new_log_probs"],
                }
            else:
                final = item["final"]
                yield {
                    "token_ids": [],
                    "log_probs": [],
                    "finish_reason": "stop",
                    "completion_usage": {
                        "completion_tokens": len(final["generated_tokens"])
                    },
                }


@pytest.fixture
def modules(monkeypatch):
    """Fake Dynamo and Megatron-LM's adapter; import the modules under test over them."""

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    async def fetch_model(model, ignore_weights=False):
        return f"/snapshots/{model}"

    module("dynamo")
    module("dynamo._core", Context=object, fetch_model=fetch_model)
    module("dynamo.common")
    worker = module("dynamo.common.backend.worker")
    module(
        "dynamo.common.backend",
        EngineConfig=_EngineConfig,
        GenerateChunk=dict,
        GenerateRequest=dict,
        LLMEngine=object,
        Worker=object,
        WorkerConfig=object,
        worker=worker,
    )
    module("dynamo.common.backend.publisher", KvEventSource=object)
    module("dynamo.llm", PythonAsyncEngine=object)
    module("dynamo.llm.exceptions", HttpError=_HttpError, InvalidArgument=ValueError)
    for name in (
        "megatron.inference",
        "megatron.inference.integrations",
        "megatron.inference.integrations.dynamo",
    ):
        module(name)
    module(
        "megatron.inference.integrations.dynamo.llm_engine",
        MegatronLLMEngine=_MegatronLLMEngine,
    )
    names = (
        "nemo_rl.models.generation.dynamo.engine_adapter",
        "nemo_rl.models.generation.dynamo.chat_engine",
        "nemo_rl.models.generation.dynamo.megatron.adapter",
    )
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    imported = SimpleNamespace(
        chat_engine=importlib.import_module(names[1]),
        adapter=importlib.import_module(names[2]),
    )
    yield imported
    for name in names:
        sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# The frontend: the chat engine with Megatron's backend
# ---------------------------------------------------------------------------


class _Pool:
    def __init__(self, rendered: RenderedCall):
        self.rendered = rendered
        self.responded: list[tuple] = []

    async def info(self):
        return {"turn_end_token_ids": [EOS], "max_model_len": 16, "bos_token_id": None}

    async def render(self, body, *, measure=False):
        self.body = body
        return self.rendered

    async def respond(self, request, prompt_token_ids, output, restored=None):
        self.responded.append((request, prompt_token_ids, output, restored))
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": len(prompt_token_ids)},
        }


class _Item:
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


class _Router:
    def __init__(self, engine_data):
        self.engine_data = engine_data
        self.requests: list[dict] = []

    async def generate(self, request, context=None):
        self.requests.append(request)

        async def stream():
            yield _Item({"engine_data": self.engine_data})

        return stream()


def _has_chat_renderer() -> bool:
    try:
        return (
            importlib.util.find_spec(
                "megatron.core.inference.text_generation_server."
                "dynamic_text_gen_server.endpoints.chat_renderer"
            )
            is not None
        )
    except ModuleNotFoundError:
        return False


# The chat backend applies Megatron-LM's BOS rule with its ChatRenderer.
@pytest.mark.skipif(
    not _has_chat_renderer(), reason="this Megatron-LM has no ChatRenderer"
)
def test_the_chat_engine_routes_a_megatron_gym_call_with_its_plan(modules):
    from nemo_rl.models.generation.dynamo.megatron.chat_backend import CHAT_BACKEND

    chat_engine = modules.chat_engine
    pool = _Pool(
        RenderedCall(
            template_token_ids=[1, 7, EOS, 3, 4],
            history_token_ids=[1, 7, EOS],
            required_prefix_token_ids=[1, 5, 6, EOS],
            admission=None,
            serving_admission=None,
            verify_retained_media=False,
            multimodal=None,
            request={"model": "model", "max_tokens": 20},
            request_max_tokens=20,
            reasoning_ended=None,
            reasoning_parser_kwargs=None,
        )
    )
    router = _Router(
        {
            "output": {"token_ids": [9, EOS], "finish_reason": "stop"},
            "result": {"generation_token_ids": [9, EOS]},
        }
    )
    engine = chat_engine.NemoRLChatEngine(
        pool=pool,
        routed_engine=router,
        spec={"generation": GENERATION},
        capture=SimpleNamespace(resolve_prefix=None),
        backend=CHAT_BACKEND,
    )
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 20,
        "nvext": {
            "annotations": [
                GYM_EXTRAS_ANNOTATION_PREFIX
                + '{"required_prefix_token_ids": [1, 5, 6, 2]}'
            ]
        },
    }

    async def run():
        return [chunk async for chunk in engine.generate(request, None)]

    (chunk,) = asyncio.run(run())

    # Omitted sampling took policy.generation's before the render.
    assert pool.body["temperature"] == 0.7 and pool.body["top_k"] == -1
    (routed,) = router.requests
    assert routed["token_ids"] == [1, 5, 6, EOS, 3, 4]
    assert routed["stop_conditions"] == {"max_tokens": 16 - 6}
    assert routed["annotations"] == [chat_engine.GYM_CALL_ANNOTATION]
    wire = routed["extra_args"][chat_engine.CALL_KEY]
    assert wire["prepared_prompt"]["prefix_len"] == 4
    assert wire["request"] == {"model": "model", "max_tokens": 20}
    assert chunk["nvext"]["nemo_rl"] == {"generation_token_ids": [9, EOS]}
    assert chunk["choices"][0]["delta"]["content"] == "hi"


# ---------------------------------------------------------------------------
# The engine: NemoRLMegatronLLMEngine
# ---------------------------------------------------------------------------


class _Host:
    def __init__(self):
        self.staged: list[tuple] = []

    def dynamo_chat_spec(self):
        return {"generation": {"model_name": "m"}}

    def stage_dynamo_gym_call(self, admission, **kwargs):
        self.staged.append((admission, kwargs))
        return {"partition": "staging", "row": 4}


def _engine(modules, role, host=None):
    config = SimpleNamespace(role=role)
    return modules.adapter.NemoRLMegatronLLMEngine(
        config, "/model", host=host or _Host()
    )


def _gym_request(modules, *, chat=None, admission=None, prefix_len=None):
    chat_engine = modules.chat_engine
    return {
        "token_ids": [1, 5, 6, EOS, 3, 4],
        "sampling_options": {"temperature": 0.7, "top_p": 0.9},
        "stop_conditions": {"max_tokens": 10},
        "output_options": {},
        "annotations": [chat_engine.GYM_CALL_ANNOTATION],
        "extra_args": {
            chat_engine.CALL_KEY: {
                "prepared_prompt": {
                    "splice": None,
                    "prefix_len": prefix_len,
                    "request_max_tokens": 20,
                    "max_tokens": 10,
                    "admission": admission,
                    "serving_admission": admission,
                    "verify_retained_media": False,
                },
                "request": chat
                or {"model": "m", "temperature": 0.7, "top_p": 0.9, "top_k": -1},
                "multimodal": None,
                "reasoning_ended": None,
                "reasoning_parser_kwargs": None,
            }
        },
    }


def _run(engine, request) -> list[dict[str, Any]]:
    async def run():
        return [chunk async for chunk in engine.generate(request, SimpleNamespace())]

    return asyncio.run(run())


_ITEMS = [
    {"partial": {"new_tokens": [9], "new_log_probs": [-0.5]}},
    {"partial": {"new_tokens": [EOS], "new_log_probs": [-0.1]}},
    {"final": {"generated_tokens": [9, EOS], "policy_epoch": [[0, 3]]}},
]


def test_registration_carries_the_chat_settings(modules):
    engine = _engine(modules, "decode")
    config = asyncio.run(engine.start(7))
    assert config.runtime_data == {
        "role": "decode",
        "worker_id": 7,
        "nemo_rl_chat": {"generation": {"model_name": "m"}},
    }


def test_rollouts_are_megatrons_own(modules):
    engine = _engine(modules, "decode")
    engine.items = _ITEMS
    request = {"token_ids": [1, 2], "annotations": ["nemo_rl:rollout:{}"]}
    chunks = _run(engine, request)
    assert engine.requests == [request]
    assert [c["token_ids"] for c in chunks] == [[9], [EOS], []]


def test_the_prefill_leg_samples_the_first_token_as_the_call_asks(modules):
    engine = _engine(modules, "prefill")
    chat = {"model": "m", "temperature": 0.6, "top_p": 0.95, "logprobs": True}
    chunks = _run(engine, _gym_request(modules, chat=chat))

    # Prefill samples the first token, whose log probability decode checks, so
    # it gets the call's sampling; its chunks are Megatron-LM's own.
    (sent,) = engine.requests
    assert sent["sampling_options"]["temperature"] == 0.6
    assert sent["sampling_options"]["top_p"] == 0.95
    assert sent["output_options"] == {"logprobs": 0}
    assert chunks == [
        {"token_ids": [], "finish_reason": "length", "disaggregated_params": {}}
    ]


def test_the_decode_leg_samples_as_the_rendered_request_asks(modules):
    engine = _engine(modules, "decode")
    engine.items = _ITEMS
    chat = {"model": "m", "temperature": 0.6, "top_p": 0.95, "top_k": 5, "stop": "x"}
    _run(engine, _gym_request(modules, chat=chat))

    (sent,) = engine.requests
    assert sent["sampling_options"] == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 5,
        "seed": None,
        "n": 1,
        "include_stop_str_in_output": False,
    }
    assert sent["stop_conditions"] == {
        "max_tokens": 10,
        "stop": ["x"],
        "ignore_eos": False,
    }
    assert sent["output_options"] == {}


@pytest.mark.parametrize("logprobs", [False, True])
def test_an_uncaptured_call_answers_with_its_tokens_in_one_chunk(modules, logprobs):
    engine = _engine(modules, "decode")
    engine.items = _ITEMS
    chat = {"model": "m", "temperature": 0.7, "top_p": 0.9, "logprobs": logprobs}
    (chunk,) = _run(engine, _gym_request(modules, chat=chat))

    (sent,) = engine.requests
    assert sent["output_options"] == ({"logprobs": 0} if logprobs else {})
    assert chunk["token_ids"] == [9, EOS] and chunk["finish_reason"] == "stop"
    output = chunk["engine_data"]["output"]
    assert output["token_ids"] == [9, EOS]
    assert ("log_probs" in output) is logprobs
    assert chunk["engine_data"]["result"] == {
        "prompt_token_ids": [1, 5, 6, EOS, 3, 4],
        "generation_token_ids": [9, EOS],
        "generation_log_probs": [-0.5, -0.1],
    }


def test_a_captured_call_is_staged_from_the_final_record(modules):
    host = _Host()
    engine = _engine(modules, "decode", host)
    engine.items = _ITEMS
    (chunk,) = _run(engine, _gym_request(modules, admission=TOKEN_IN, prefix_len=4))

    (sent,) = engine.requests
    # A captured call stages its log probabilities, so it always asks for them.
    assert sent["output_options"] == {"logprobs": 0}
    ((admission, staged),) = host.staged
    # Gym verifies the engine's prompt against the model's own prefix.
    assert admission == {**TOKEN_IN, "required_prefix_token_ids": [1, 5, 6, EOS]}
    assert staged == {
        "prompt_token_ids": [1, 5, 6, EOS, 3, 4],
        "generated_token_ids": [9, EOS],
        "generated_log_probs": [-0.5, -0.1],
        "final_record": {"generated_tokens": [9, EOS], "policy_epoch": [[0, 3]]},
    }
    assert chunk["engine_data"]["result"] == {
        "ng_commit_coords": {"partition": "staging", "row": 4}
    }
    assert "log_probs" not in chunk["engine_data"]["output"]


def test_a_captured_call_is_staged_once_and_sends_no_offload_params(modules):
    # Behind Dynamo the engine has no prompt preparer or payload stager: the
    # adapter stages the call itself, so nothing on the request may ask
    # Megatron-LM to prepare or stage it again.
    host = _Host()
    engine = _engine(modules, "decode", host)
    engine.items = _ITEMS
    _run(engine, _gym_request(modules, admission=TOKEN_IN, prefix_len=4))

    assert len(host.staged) == 1
    (sent,) = engine.requests
    assert "offload_params" not in sent
    assert "offload_params" not in str(sent["extra_args"])
    assert "offload_params" not in sent["sampling_options"]


def test_a_refused_call_is_answered_with_its_status(modules):
    engine = _engine(modules, "decode")
    engine.error = ValueError("Megatron request failed: too long")
    (chunk,) = _run(engine, _gym_request(modules))
    assert chunk["engine_data"] == {
        "refusal": {
            "status": 400,
            "message": "Megatron request failed: too long",
            "acknowledge": False,
        }
    }


def test_a_gym_call_needs_the_chat_engines_plan(modules):
    engine = _engine(modules, "decode")
    request = _gym_request(modules)
    request["extra_args"] = {}
    with pytest.raises(ValueError, match="through NeMo RL's chat engine"):
        _run(engine, request)
