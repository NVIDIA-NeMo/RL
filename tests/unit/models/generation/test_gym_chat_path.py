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
"""NeMo-Gym's chat calls take the same steps on both of NeMo RL's serving paths.

Each case sends one call to the native vLLM server (its real routes, over ASGI)
and through Dynamo (NeMo RL's chat engine, a spawned render pool, and the
engine adapter), over the same tiny Qwen3 checkpoint, the same scripted engine
and Gym's real capture core. Both must run the engine on the same prompt with
the same SamplingParams, stage the same capture record, and answer Gym with the
same message and token fields.

Not covered here: media (the checkpoint is text-only; the VLM functional
overlay runs it) and a model whose EOS is not its turn end.
"""

import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.vllm, pytest.mark.nemo_gym]
pytest.importorskip("dynamo._core")
pytest.importorskip("vllm")
pytest.importorskip("nemo_gym.token_id_capture.staging")

import httpx  # noqa: E402
from dynamo.llm.exceptions import HttpError  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter  # noqa: E402
from nemo_gym.token_id_capture.replay import (  # noqa: E402
    replay_context,
    summarize_replay,
)
from nemo_gym.token_id_capture.staging.capture import (  # noqa: E402
    RolloutTokenCapture,
)
from nemo_gym.token_id_capture.staging.records import (  # noqa: E402
    CaptureAdmission,
    StageResult,
)
from transformers import AutoTokenizer  # noqa: E402
from vllm.config import CacheConfig, LoadConfig, VllmConfig  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.logprobs import Logprob  # noqa: E402
from vllm.outputs import CompletionOutput, RequestOutput  # noqa: E402
from vllm.renderers.registry import renderer_from_config  # noqa: E402
from vllm.v1.engine.input_processor import InputProcessor  # noqa: E402

from nemo_rl.data_plane.tq_token_sink import ChainPrefixCache, StagedPrefix  # noqa: E402
from nemo_rl.models.generation.dynamo.vllm import chat_engine  # noqa: E402
from nemo_rl.models.generation.dynamo.vllm.adapter import (  # noqa: E402
    DynamoVllmAdapter,
    chat_runtime_data,
)
from nemo_rl.models.generation.dynamo.vllm.render_pool import RenderPool  # noqa: E402
from nemo_rl.models.generation.gym_call import GYM_EXTRAS_ANNOTATION_PREFIX  # noqa: E402
from nemo_rl.models.generation.vllm import chat_server  # noqa: E402
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    VllmAsyncGenerationWorkerImpl,
)

MAX_MODEL_LEN = 512
GENERATION = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": None,
    "val_temperature": 0.0,
    "val_top_p": 1.0,
    "val_top_k": None,
    "stop_token_ids": None,
    "max_new_tokens": 64,
    "vllm_cfg": {
        "max_model_len": MAX_MODEL_LEN,
        "http_server_serving_chat_kwargs": {
            "tool_parser": "hermes",
            "reasoning_parser": "qwen3",
            "enable_auto_tools": True,
            # Rendered beneath the request's own on both paths.
            "default_chat_template_kwargs": {"enable_thinking": False},
        },
    },
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "The weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
# Turn 1 as Gym sends it back: the template drops its reasoning when it
# re-renders the history, so the model's own tokens differ from the render.
HISTORY = [
    {"role": "user", "content": "Weather in Paris?"},
    {"role": "assistant", "content": "It is sunny.", "reasoning_content": "easy"},
]
OBSERVATION = {"role": "user", "content": "And in Lyon?"}
REPLY = (
    "<think>\nchecking\n</think>\n\n<tool_call>\n"
    '{"name": "get_weather", "arguments": {"city": "Lyon"}}\n</tool_call>'
)
TURN_1_REPLY = "<think>\neasy\n</think>\n\nIt is sunny."


@dataclass
class _Kit:
    """The checkpoint, and the vLLM objects both paths share."""

    path: str
    tokenizer: Any
    engine_args: Any
    model_config: Any
    vllm_config: Any
    renderer: Any
    pool: RenderPool

    def ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def turn_1_tokens(self) -> list[int]:
        """Turn 1's prompt and the model's own reply, as capture staged them."""
        prompt = self.tokenizer.apply_chat_template(
            HISTORY[:1], tools=TOOLS, add_generation_prompt=True, tokenize=False
        )
        return self.ids(prompt + TURN_1_REPLY + "<|im_end|>")


class _Engine:
    """vLLM's engine client: answers every call with REPLY and records its runs."""

    def __init__(self, kit: _Kit) -> None:
        self.model_config = kit.model_config
        self.renderer = kit.renderer
        self.vllm_config = kit.vllm_config
        self.input_processor = InputProcessor(kit.vllm_config)
        self.io_processor = None
        self.errored = False
        self._kit = kit
        self.runs: list[tuple[list[int], Any]] = []

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        prompt_token_ids = list(prompt["prompt_token_ids"])
        self.runs.append((prompt_token_ids, sampling_params))
        reply_ids = self._kit.ids(REPLY)
        logprobs = None
        if sampling_params.logprobs is not None:
            decode = self._kit.tokenizer.decode
            # The sampled token, and the runner-up a top_logprobs > 0 call sees.
            logprobs = [
                {
                    token_id: Logprob(-0.5, 1, decode([token_id])),
                    token_id + 1: Logprob(-2.0, 2, decode([token_id + 1])),
                }
                for token_id in reply_ids
            ]
        yield RequestOutput(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text=REPLY,
                    token_ids=reply_ids,
                    cumulative_logprob=None,
                    logprobs=logprobs,
                    finish_reason="stop",
                )
            ],
            finished=True,
            num_cached_tokens=0,
        )

    def check_admission(self, n, request_id=None) -> None:
        pass

    async def is_tracing_enabled(self) -> bool:
        return False

    async def abort(self, request_id) -> None:
        pass


class _Sink:
    def __init__(self) -> None:
        self.staged: list[tuple[Any, Any]] = []

    def stage(self, record, *, attachments=None) -> StageResult:
        self.staged.append((record, attachments))
        return StageResult(ok=True, staging_key=record.staging_key)


class _Staged:
    """TQ's staged calls: each call's token delta, by staging key."""

    def __init__(self, deltas: dict[str, list[int]]) -> None:
        self.deltas = deltas

    def fetch_prefix_chains(self, staging_keys: list[str]) -> StagedPrefix:
        return StagedPrefix(
            expanded=[token for key in staging_keys for token in self.deltas[key]]
        )


@dataclass
class _Outcome:
    """What one path did with a call."""

    status: int
    reply: dict[str, Any]
    runs: list[tuple[list[int], Any]]
    staged: list[tuple[Any, Any]]
    failed: list[str] = field(default_factory=list)


def _host(kit: _Kit, deltas: dict[str, list[int]], top_logprobs: int):
    host = object.__new__(VllmAsyncGenerationWorkerImpl)
    host.cfg = GENERATION
    host.llm = host._http_engine_client = _Engine(kit)
    host.llm_async_engine_args = kit.engine_args
    host.model_name = "model"
    host.routed_experts_dtype = torch.int16
    host._capture_media = False
    host._capture_calls = {}
    host._capture_top_logprobs = top_logprobs
    host._staging_source = None
    host._rollout_weight_version = 0
    host._chain_prefix = ChainPrefixCache()
    host._chain_prefix.install(_Staged(deltas))
    sink = _Sink()
    capture = RolloutTokenCapture(
        sink=sink, weight_version_fn=lambda: 0, adapter=VLLMCaptureAdapter()
    )
    failed: list[str] = []
    fail_call = capture.fail_call

    def record_failure(call, *, reason):
        failed.append(reason)
        return fail_call(call, reason=reason)

    capture.fail_call = record_failure
    host.token_capture = capture
    return host, sink, failed


async def _native(kit, host, sink, failed, body, *, measure: bool) -> _Outcome:
    server = chat_server.build_chat_server(
        host,
        engine_client=host._http_engine_client,
        model_config=kit.model_config,
        cfg=GENERATION,
    )
    app = FastAPI()
    chat_server.register_routes(app, server, host=host, generation_config=GENERATION)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://native"
    ) as client:
        response = await client.post(
            "/tokenize" if measure else "/v1/chat/completions", json=body
        )
    return _Outcome(
        response.status_code, response.json(), host.llm.runs, sink.staged, failed
    )


class _Router:
    """Dynamo's router to one engine: the request crosses the wire as JSON."""

    def __init__(self, adapter: DynamoVllmAdapter) -> None:
        self._adapter = adapter

    async def generate(self, request, context=None):
        wire = json.loads(json.dumps(request))
        chunks = [chunk async for chunk in self._adapter.generate(wire, _Context())]

        async def stream():
            for chunk in chunks:
                yield SimpleItem(chunk)

        return stream()


class _Context:
    def id(self) -> str:
        return "chatcmpl-0"

    def is_stopped(self) -> bool:
        return False


class SimpleItem:
    def __init__(self, data) -> None:
        self._data = data

    def is_error(self) -> bool:
        return False

    def data(self):
        return self._data


async def _dynamo(kit, host, sink, failed, body, extras) -> _Outcome:
    request = {k: v for k, v in body.items() if k not in extras}
    request["nvext"] = {
        "annotations": [GYM_EXTRAS_ANNOTATION_PREFIX + json.dumps(extras)]
    }
    capture = chat_engine._CaptureSource()
    capture._prefix_cache = host._chain_prefix
    engine = chat_engine.NemoRLChatEngine(
        pool=kit.pool,
        routed_engine=_Router(DynamoVllmAdapter(host, leader_worker_idx=0)),
        spec=json.loads(json.dumps(chat_runtime_data(host)))[
            chat_engine.RUNTIME_DATA_KEY
        ],
        capture=capture,
    )
    try:
        (chunk,) = [chunk async for chunk in engine.generate(request, None)]
    except HttpError as error:
        return _Outcome(
            error.code, {"error": error.message}, host.llm.runs, sink.staged, failed
        )
    return _Outcome(200, chunk, host.llm.runs, sink.staged, failed)


@pytest.fixture(scope="module")
def kit(tiny_qwen3_model_path):
    engine_args = AsyncEngineArgs(
        model=tiny_qwen3_model_path, max_model_len=MAX_MODEL_LEN
    )
    model_config = engine_args.create_model_config()
    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=LoadConfig(load_format="dummy"),
        cache_config=CacheConfig(),
    )
    # The pool renders with what an engine publishes at registration.
    spec = json.loads(
        json.dumps(
            chat_runtime_data(
                SimpleNamespace(llm_async_engine_args=engine_args, cfg=GENERATION)
            )
        )
    )[chat_engine.RUNTIME_DATA_KEY]
    pool = RenderPool(spec, tiny_qwen3_model_path, workers=1)
    yield _Kit(
        path=tiny_qwen3_model_path,
        tokenizer=AutoTokenizer.from_pretrained(tiny_qwen3_model_path),
        engine_args=engine_args,
        model_config=model_config,
        vllm_config=vllm_config,
        renderer=renderer_from_config(vllm_config),
        pool=pool,
    )
    pool.shutdown()


def _admission(**fields) -> dict[str, Any]:
    return CaptureAdmission(
        capture_key="rollout-1", model_call_id="c2", **fields
    ).model_dump(mode="json")


def _staged_turn_1(kit: _Kit, **fields) -> dict[str, Any]:
    """Turn 2's admission, chained to turn 1 staged under rollout-1/c1."""
    return _admission(
        parent_call_id="c1",
        prev_len=len(kit.turn_1_tokens()),
        parent_chain_hash="1" * 64,
        staging_chain=["rollout-1/c1"],
        **fields,
    )


def _candidate(kit: _Kit, messages: list[dict]) -> dict[str, Any]:
    """Gym's candidate: turn 1's source, and this request's, for RL to check the cut."""
    return _staged_turn_1(
        kit,
        mode="candidate",
        requires_empty_tool_content_equivalence=False,
        candidate_replay=summarize_replay(
            replay_context(HISTORY, render_digest="b" * 64)
        ),
        request_replay=replay_context(messages, render_digest="b" * 64),
    )


# id, messages, extras(kit), request fields, capture top-k
CASES = [
    # A plain call: the render and the reply's reasoning and tool parsers (C39).
    ("plain", [*HISTORY, OBSERVATION], lambda kit: {}, {}, 0),
    # Gym's echo of turn 1 without capture: the model's tokens replace the
    # template's re-render of that turn (C16's turn-end cut).
    (
        "inline_prefix",
        [*HISTORY, OBSERVATION],
        lambda kit: {"required_prefix_token_ids": kit.turn_1_tokens()},
        {},
        0,
    ),
    # Token capture: turn 1's staged tokens, read from TQ.
    (
        "token_in",
        [*HISTORY, OBSERVATION],
        lambda kit: {"ng_capture": _staged_turn_1(kit, mode="token_in")},
        {},
        0,
    ),
    # Gym's candidate whose cut holds is served token-in (C26)...
    (
        "candidate_cut_holds",
        [*HISTORY, OBSERVATION],
        lambda kit: {"ng_capture": _candidate(kit, [*HISTORY, OBSERVATION])},
        {},
        0,
    ),
    # ...and one whose conversion moved the cut roots a new text segment.
    (
        "candidate_cut_fails",
        [*HISTORY, OBSERVATION, HISTORY[-1]],
        lambda kit: {"ng_capture": _candidate(kit, [*HISTORY, OBSERVATION])},
        {},
        0,
    ),
    # Score centering stages the sampler's top-k with the call (C29).
    (
        "score_centering",
        [*HISTORY, OBSERVATION],
        lambda kit: {"ng_capture": _admission(mode="text")},
        {"top_logprobs": 2},
        2,
    ),
    # Off-policy sampling is refused before anything runs (C41).
    ("off_policy", [*HISTORY, OBSERVATION], lambda kit: {}, {"temperature": 0.5}, 0),
]


@pytest.mark.parametrize(
    ("messages", "extras", "fields", "top_logprobs"),
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_a_gym_call_takes_the_same_steps_through_dynamo_as_natively(
    kit, messages, extras, fields, top_logprobs
) -> None:
    extras = extras(kit)
    body = {
        "model": kit.path,
        "messages": messages,
        "tools": TOOLS,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 32,
        # Gym's token information.
        "logprobs": True,
        "top_logprobs": 0,
        **fields,
        **extras,
    }
    deltas = {"rollout-1/c1": kit.turn_1_tokens()}

    async def both():
        native = await _native(
            kit, *_host(kit, deltas, top_logprobs), body, measure=False
        )
        dynamo = await _dynamo(kit, *_host(kit, deltas, top_logprobs), body, extras)
        return native, dynamo

    native, dynamo = asyncio.run(both())

    assert (native.status, dynamo.status) == (
        native.status,
        native.status,
    ), (native.reply, dynamo.reply)
    if native.status != 200:
        assert native.runs == dynamo.runs == []
        return
    # The engine ran the same prompt with the same SamplingParams.
    ((native_prompt, native_params),) = native.runs
    ((dynamo_prompt, dynamo_params),) = dynamo.runs
    assert dynamo_prompt == native_prompt
    assert dynamo_params == native_params
    # Capture staged the same record and tensors.
    assert len(dynamo.staged) == len(native.staged)
    for (native_record, native_tensors), (dynamo_record, dynamo_tensors) in zip(
        native.staged, dynamo.staged
    ):
        assert dynamo_record == native_record
        assert (dynamo_tensors or {}).keys() == (native_tensors or {}).keys()
        for name, tensor in (native_tensors or {}).items():
            assert torch.equal(dynamo_tensors[name], tensor), name
    assert dynamo.failed == native.failed == []
    # Gym reads the same reply and token fields.
    message = native.reply["choices"][0]["message"]
    (choice,) = dynamo.reply["choices"]
    assert choice["finish_reason"] == native.reply["choices"][0]["finish_reason"]
    assert choice["delta"]["content"] == message["content"]
    assert choice["delta"].get("reasoning_content") == message.get("reasoning")
    assert [call["function"] for call in choice["delta"]["tool_calls"]] == [
        call["function"] for call in message["tool_calls"]
    ]
    result = dynamo.reply["nvext"]["nemo_rl"]
    if native.staged:
        assert result == {"ng_commit_coords": native.reply["ng_commit_coords"]}
    else:
        assert result == {
            key: message[key]
            for key in (
                "prompt_token_ids",
                "generation_token_ids",
                "generation_log_probs",
            )
        }


def test_a_measure_call_counts_the_prompt_the_call_would_run(kit) -> None:
    extras = {"required_prefix_token_ids": kit.turn_1_tokens()}
    body = {
        "model": kit.path,
        "messages": [*HISTORY, OBSERVATION],
        "tools": TOOLS,
        **extras,
    }

    async def both():
        native = await _native(kit, *_host(kit, {}, 0), body, measure=True)
        dynamo = await _dynamo(
            kit,
            *_host(kit, {}, 0),
            {**body, "temperature": 0.7, "top_p": 0.9},
            {**extras, "ng_measure": True},
        )
        return native, dynamo

    native, dynamo = asyncio.run(both())

    # Native Gym measures with /tokenize; through Dynamo, the chat call (C36).
    assert (
        dynamo.reply["nvext"]["nemo_rl"]["prompt_token_count"] == native.reply["count"]
    )
    assert dynamo.runs == []
