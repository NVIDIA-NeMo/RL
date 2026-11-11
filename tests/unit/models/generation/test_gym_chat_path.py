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
engine adapter), with the same kit (a checkpoint and the chat settings both
paths serve), the same scripted engine and Gym's real capture core. Both must
run the engine on the same prompt, media and SamplingParams, stage the same
capture record, and answer Gym with the same message and token fields.

Not covered here: staging an image's pixels, which needs the Omni processor
and its gated checkpoint.
"""

import asyncio
import base64
import io
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import pytest
import torch

pytestmark = [pytest.mark.vllm, pytest.mark.nemo_gym]
pytest.importorskip("dynamo._core")
pytest.importorskip("vllm")
pytest.importorskip("nemo_gym.token_id_capture.staging")

import httpx  # noqa: E402
from dynamo.llm.exceptions import HttpError  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from nemo_gym._checkpoint.generation_cut import (  # noqa: E402
    GenerationCutInventory,
    GenerationCutPrefix,
)
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
from PIL import Image  # noqa: E402
from transformers import AutoConfig, AutoProcessor, AutoTokenizer  # noqa: E402
from vllm.config import CacheConfig, LoadConfig, VllmConfig  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.exceptions import QueueOverflowError, VLLMValidationError  # noqa: E402
from vllm.logprobs import Logprob  # noqa: E402
from vllm.outputs import CompletionOutput, RequestOutput  # noqa: E402
from vllm.renderers.registry import renderer_from_config  # noqa: E402
from vllm.v1.engine.input_processor import InputProcessor  # noqa: E402

from nemo_rl.data_plane.tq_token_sink import (  # noqa: E402
    ChainPrefixCache,
    StagedPrefix,
)
from nemo_rl.models.generation.dynamo.vllm import chat_engine  # noqa: E402
from nemo_rl.models.generation.dynamo.vllm.adapter import (  # noqa: E402
    DynamoVllmAdapter,
    chat_runtime_data,
)
from nemo_rl.models.generation.dynamo.vllm.render_pool import RenderPool  # noqa: E402
from nemo_rl.models.generation.gym_call import (  # noqa: E402
    GYM_EXTRAS_ANNOTATION_PREFIX,
)
from nemo_rl.models.generation.prefix_recovery.restore import (  # noqa: E402
    _build_restored_prefix_terminal_output,
    restored_terminal,
)
from nemo_rl.models.generation.vllm import chat_server  # noqa: E402
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    VllmAsyncGenerationWorkerImpl,
)
from tests.unit.models.generation.test_vllm_token_capture_hosting import (  # noqa: E402
    _cuts,
    _MemoryPrefixSource,
    _MemorySink,
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
# A tool whose schema has a null default, as pydantic writes an Optional field.
NULL_DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            **TOOLS[0]["function"],
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                    },
                },
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
# Turn 1 as a tool call, which Gym's source history recorded with empty
# content and the harness now sends with none.
TOOL_TURN = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call-0",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
    ],
}
TOOL_HISTORY = [HISTORY[0], {**TOOL_TURN, "content": ""}]
TOOL_RESULT = {"role": "tool", "tool_call_id": "call-0", "content": "sunny"}


def _image_url() -> str:
    image = Image.new("RGB", (56, 56))
    image.putdata([(4 * x, 4 * y, 128) for y in range(56) for x in range(56)])
    png = io.BytesIO()
    image.save(png, format="PNG")
    return "data:image/png;base64," + base64.b64encode(png.getvalue()).decode()


IMAGE_TURN = {
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": _image_url()}},
        {"type": "text", "text": "And here?"},
    ],
}
NANO_V3_PLUGIN = str(
    Path(__file__).parents[4]
    / "nemo_rl/models/generation/vllm/reasoning_parsers/nano_v3_reasoning_parser.py"
)


@dataclass
class _Kit:
    """The checkpoint, and the vLLM objects both paths share."""

    path: str
    generation: dict[str, Any]
    capture_media: bool
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
    """vLLM's engine client: answers every call with REPLY and records its runs.

    ``refuse`` is what the engine refuses every call with, and where: its
    queue check ("admission") or its input check ("generate").
    """

    def __init__(self, kit: _Kit, refuse: Optional[tuple[str, Any]] = None) -> None:
        self.model_config = kit.model_config
        self.renderer = kit.renderer
        self.vllm_config = kit.vllm_config
        self.input_processor = InputProcessor(kit.vllm_config)
        self.io_processor = None
        self.errored = False
        self._kit = kit
        self._refuse = refuse
        self.runs: list[tuple[list[int], Any, Any]] = []
        # How many of REPLY's tokens the prompt already ends with: a call
        # resumed from a checkpoint cut generates only the rest.
        self.resumed = 0

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        if self._refuse is not None and self._refuse[0] == "generate":
            raise self._refuse[1]
        prompt_token_ids = list(prompt["prompt_token_ids"])
        self.runs.append((prompt_token_ids, sampling_params, _media(prompt)))
        terminal = restored_terminal()
        if terminal is not None:
            # As the worker's engine client answers a call whose saved prefix
            # had already finished it: without decoding.
            yield _build_restored_prefix_terminal_output(request_id, terminal)
            return
        reply_ids = self._kit.ids(REPLY)[self.resumed :]
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
                    text=self._kit.tokenizer.decode(reply_ids)
                    if self.resumed
                    else REPLY,
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
        if self._refuse is not None and self._refuse[0] == "admission":
            raise self._refuse[1]

    async def is_tracing_enabled(self) -> bool:
        return False

    async def abort(self, request_id) -> None:
        pass


def _media(prompt) -> Any:
    """The processed media an engine prompt runs: its token ranges and tensors."""
    if "mm_kwargs" not in prompt:
        return None
    return (
        {
            modality: [(item.offset, item.length) for item in ranges]
            for modality, ranges in prompt["mm_placeholders"].items()
        },
        {
            modality: [{key: elem.data for key, elem in item.items()} for item in items]
            for modality, items in prompt["mm_kwargs"].items()
        },
    )


def _assert_same_media(dynamo, native) -> None:
    if native is None:
        assert dynamo is None
        return
    assert dynamo[0] == native[0]
    assert dynamo[1].keys() == native[1].keys()
    for modality, items in native[1].items():
        assert len(dynamo[1][modality]) == len(items)
        for dynamo_item, item in zip(dynamo[1][modality], items):
            assert dynamo_item.keys() == item.keys()
            for key, tensor in item.items():
                assert torch.equal(dynamo_item[key], tensor), (modality, key)


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
    runs: list[tuple[list[int], Any, Any]]
    staged: list[tuple[Any, Any]]
    failed: list[str] = field(default_factory=list)


def _host(
    kit: _Kit,
    deltas: dict[str, list[int]],
    top_logprobs: int,
    refuse: Optional[tuple[str, Any]] = None,
):
    host = object.__new__(VllmAsyncGenerationWorkerImpl)
    host._init_token_capture_state()
    host.cfg = kit.generation
    host.llm = host._http_engine_client = _Engine(kit, refuse)
    host.llm_async_engine_args = kit.engine_args
    host.model_name = "model"
    host.routed_experts_dtype = torch.int16
    host._capture_media = kit.capture_media
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
        cfg=kit.generation,
    )
    app = FastAPI()
    chat_server.register_routes(
        app, server, host=host, generation_config=kit.generation
    )
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

    def data(self):
        return self._data


async def _dynamo(kit, host, sink, failed, body, extras) -> _Outcome:
    request = {k: v for k, v in body.items() if k not in extras}
    # Dynamo's request names the template kwargs chat_template_args.
    if "chat_template_kwargs" in request:
        request["chat_template_args"] = request.pop("chat_template_kwargs")
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


def _generation(*, vllm_cfg=None, **chat_settings) -> dict[str, Any]:
    """GENERATION with the native chat settings a recipe changes."""
    base = GENERATION["vllm_cfg"]
    return {
        **GENERATION,
        "vllm_cfg": {
            **base,
            **(vllm_cfg or {}),
            "http_server_serving_chat_kwargs": {
                **base["http_server_serving_chat_kwargs"],
                **chat_settings,
            },
        },
    }


def _turn_end_not_eos(path: str, tmp_path_factory) -> str:
    """The checkpoint, shaped like gpt-oss: its turns end on a generation-config
    EOS (<|im_end|>) that is not the tokenizer's (<|endoftext|>)."""
    copy = tmp_path_factory.mktemp("turn_end_not_eos")
    shutil.copytree(path, copy, dirs_exist_ok=True)
    tokenizer_config = json.loads((copy / "tokenizer_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(path)
    eos = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
    tokenizer_config["eos_token"] = "<|endoftext|>"
    (copy / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (copy / "generation_config.json").write_text(json.dumps({"eos_token_id": eos}))
    return str(copy)


def _vlm(path: str, tmp_path_factory) -> str:
    """Qwen2.5-VL's config and processor: rendering never loads weights."""
    del path
    checkpoint = tmp_path_factory.mktemp("qwen2_5_vl")
    AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct").save_pretrained(
        checkpoint
    )
    AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct").save_pretrained(
        checkpoint
    )
    return str(checkpoint)


@dataclass(frozen=True)
class _KitSpec:
    generation: dict[str, Any]
    # The checkpoint, from the tiny Qwen3's path.
    checkpoint: Callable[..., str] = lambda path, _: path
    capture_media: bool = False


KITS = {
    "qwen3": _KitSpec(GENERATION),
    "turn_end_not_eos": _KitSpec(GENERATION, _turn_end_not_eos),
    # The template gets structured content parts instead of strings.
    "structured_content": _KitSpec(_generation(chat_template_content_format="openai")),
    # Nemotron-3-Nano's parser, a plugin file the native recipe loads.
    "nano_v3": _KitSpec(
        _generation(
            vllm_cfg={"reasoning_parser_plugin": NANO_V3_PLUGIN},
            reasoning_parser="nano_v3",
        )
    ),
    # Qwen2.5's tokenizer has no thinking tokens for Qwen3's reasoning parser.
    "vlm": _KitSpec(_generation(reasoning_parser=""), _vlm),
    # A tool parser with no structural tag: a forced tool sets structured outputs.
    "pythonic": _KitSpec(_generation(tool_parser="pythonic")),
}


def _build_kit(kit_spec: _KitSpec, path: str) -> _Kit:
    generation = kit_spec.generation
    # No processor cache: each case's engine gets the media itself, not a handle
    # to media an earlier case processed.
    engine_args = AsyncEngineArgs(
        model=path, max_model_len=MAX_MODEL_LEN, mm_processor_cache_gb=0
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
                SimpleNamespace(llm_async_engine_args=engine_args, cfg=generation)
            )
        )
    )[chat_engine.RUNTIME_DATA_KEY]
    return _Kit(
        path=path,
        generation=generation,
        capture_media=kit_spec.capture_media,
        tokenizer=AutoTokenizer.from_pretrained(path),
        engine_args=engine_args,
        model_config=model_config,
        vllm_config=vllm_config,
        renderer=renderer_from_config(vllm_config),
        pool=RenderPool(spec, path, workers=1),
    )


@pytest.fixture(scope="module")
def kits(tiny_qwen3_model_path, tmp_path_factory):
    """Each kit in KITS, built on first use."""
    built: dict[str, _Kit] = {}

    def kit(name: str) -> _Kit:
        if name not in built:
            spec = KITS[name]
            built[name] = _build_kit(
                spec, spec.checkpoint(tiny_qwen3_model_path, tmp_path_factory)
            )
        return built[name]

    yield kit
    for built_kit in built.values():
        built_kit.pool.shutdown()


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


def _candidate(
    kit: _Kit,
    messages: list[dict],
    source: list[dict] = HISTORY,
    *,
    empty_tool_content: bool = False,
) -> dict[str, Any]:
    """Gym's candidate: turn 1's source, and this request's, for RL to check the cut."""
    return _staged_turn_1(
        kit,
        mode="candidate",
        requires_empty_tool_content_equivalence=empty_tool_content,
        candidate_replay=summarize_replay(
            replay_context(source, render_digest="b" * 64)
        ),
        request_replay=replay_context(messages, render_digest="b" * 64),
    )


@dataclass(frozen=True)
class _Case:
    """One Gym call, and what both paths must do with it."""

    id: str
    messages: list[dict] = field(default_factory=lambda: [*HISTORY, OBSERVATION])
    extras: Callable[[_Kit], dict[str, Any]] = lambda kit: {}
    fields: dict[str, Any] = field(default_factory=dict)
    # The sampler top-k capture stages, for score centering.
    top_logprobs: int = 0
    # 200, or the status both paths refuse the call with before it runs.
    status: int = 200
    # The mode capture stages the call in; None when the call is not captured.
    mode: Optional[str] = None
    kit: str = "qwen3"
    # What the engine refuses the call with, and where (see _Engine).
    refuse: Optional[tuple[str, Any]] = None
    # What a refused call's capture is failed with, on both paths.
    failed: tuple[str, ...] = ()


def _off_policy(name: str, value: Any) -> _Case:
    return _Case(f"off_policy_{name}", fields={name: value}, status=400)


CASES = [
    # A plain call: the render and the reply's reasoning and tool parsers.
    _Case("plain"),
    # No max_tokens: each path fills the budget the way vLLM's server does.
    _Case("no_max_tokens", fields={"max_tokens": None}),
    # Gym's echo of turn 1 without capture: the model's tokens replace the
    # template's re-render of that turn (the turn-end cut).
    _Case(
        "inline_prefix",
        extras=lambda kit: {"required_prefix_token_ids": kit.turn_1_tokens()},
    ),
    # The same, for a model whose turns end on a token that is not its
    # tokenizer's EOS, as gpt-oss's <|call|>.
    _Case(
        "turn_end_not_eos",
        extras=lambda kit: {"required_prefix_token_ids": kit.turn_1_tokens()},
        kit="turn_end_not_eos",
    ),
    # Token capture: turn 1's staged tokens, read from TQ.
    _Case(
        "token_in",
        extras=lambda kit: {"ng_capture": _staged_turn_1(kit, mode="token_in")},
        mode="token_in",
    ),
    # Gym's candidate whose cut holds is served token-in...
    _Case(
        "candidate_cut_holds",
        extras=lambda kit: {"ng_capture": _candidate(kit, [*HISTORY, OBSERVATION])},
        mode="token_in",
    ),
    # ...and one whose conversion moved the cut roots a new text segment.
    _Case(
        "candidate_cut_fails",
        messages=[*HISTORY, OBSERVATION, HISTORY[-1]],
        extras=lambda kit: {"ng_capture": _candidate(kit, [*HISTORY, OBSERVATION])},
        mode="text",
    ),
    # A candidate whose tool call had empty content and now has none: string
    # content renders both alike, so the cut holds...
    _Case(
        "empty_tool_content",
        messages=[HISTORY[0], TOOL_TURN, TOOL_RESULT],
        extras=lambda kit: {
            "ng_capture": _candidate(
                kit,
                [HISTORY[0], TOOL_TURN, TOOL_RESULT],
                TOOL_HISTORY,
                empty_tool_content=True,
            )
        },
        mode="token_in",
    ),
    # ...and structured content cannot prove that, so it roots a new segment.
    _Case(
        "empty_tool_content_structured",
        messages=[HISTORY[0], TOOL_TURN, TOOL_RESULT],
        extras=lambda kit: {
            "ng_capture": _candidate(
                kit,
                [HISTORY[0], TOOL_TURN, TOOL_RESULT],
                TOOL_HISTORY,
                empty_tool_content=True,
            )
        },
        mode="text",
        kit="structured_content",
    ),
    # Score centering stages the sampler's top-k with the call.
    _Case(
        "score_centering",
        extras=lambda kit: {"ng_capture": _admission(mode="text")},
        fields={"top_logprobs": 2},
        top_logprobs=2,
        mode="text",
    ),
    # Capture with a processor that has no media capture is refused before it
    # runs, with the native server's 400.
    _Case(
        "image_capture_unsupported",
        messages=[IMAGE_TURN],
        extras=lambda kit: {"ng_capture": _admission(mode="text")},
        status=400,
        kit="vlm",
    ),
    # An image after the echoed turn 1: processed once in the frontend, both
    # engines run the same pixels, and the splice moves its tokens.
    _Case(
        "image_after_echoed_turn",
        messages=[*HISTORY, IMAGE_TURN],
        extras=lambda kit: {"required_prefix_token_ids": kit.turn_1_tokens()},
        kit="vlm",
    ),
    # The client's nulls in a tool schema are rendered into the prompt.
    _Case("tool_schema_null_default", fields={"tools": NULL_DEFAULT_TOOLS}),
    # A named tool_choice, whose schema the tool parser sets as structured outputs.
    _Case(
        "named_tool_choice",
        fields={
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        },
        kit="pythonic",
    ),
    # Nemotron-3-Nano's reasoning parser plugin, with thinking off.
    _Case(
        "nano_v3_thinking_off",
        fields={"chat_template_kwargs": {"enable_thinking": False}},
        kit="nano_v3",
    ),
    # The sampling the trainer replays, set explicitly...
    _Case(
        "on_policy_sampling",
        fields={
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "logit_bias": {},
        },
    ),
    # ...and any it does not is refused before anything runs.
    _off_policy("temperature", 0.5),
    _off_policy("top_p", 0.5),
    _off_policy("top_k", 50),
    _off_policy("min_p", 0.1),
    _off_policy("repetition_penalty", 1.05),
    _off_policy("presence_penalty", 0.5),
    _off_policy("frequency_penalty", 0.5),
    _off_policy("logit_bias", {"10": 5.0}),
    # Capture stages one choice, and the trainer replays one.
    _Case("n_2", fields={"n": 2}, status=400),
    # Token information without top_logprobs, unset or sent as null.
    _Case("top_logprobs_unset", fields={"top_logprobs": None}, status=400),
    # A full queue: vLLM's 503, so Gym retries elsewhere; checked before capture.
    _Case(
        "queue_full",
        extras=lambda kit: {"ng_capture": _admission(mode="text")},
        status=503,
        refuse=("admission", QueueOverflowError()),
    ),
    # The engine's input check refuses a captured call after it began.
    _Case(
        "token_out_of_vocabulary",
        extras=lambda kit: {"ng_capture": _admission(mode="text")},
        status=400,
        refuse=(
            "generate",
            VLLMValidationError("Token id 999999 is out of vocabulary"),
        ),
        failed=("invalid_request",),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_a_gym_call_takes_the_same_steps_through_dynamo_as_natively(kits, case) -> None:
    kit = kits(case.kit)
    extras = case.extras(kit)
    body = {
        "model": kit.path,
        "messages": case.messages,
        "tools": TOOLS,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 32,
        # Gym's token information.
        "logprobs": True,
        "top_logprobs": 0,
        **case.fields,
        **extras,
    }
    deltas = {"rollout-1/c1": kit.turn_1_tokens()}

    async def both():
        native = await _native(
            kit,
            *_host(kit, deltas, case.top_logprobs, case.refuse),
            body,
            measure=False,
        )
        dynamo = await _dynamo(
            kit, *_host(kit, deltas, case.top_logprobs, case.refuse), body, extras
        )
        return native, dynamo

    native, dynamo = asyncio.run(both())

    assert (native.status, dynamo.status) == (case.status, case.status), (
        native.reply,
        dynamo.reply,
    )
    if case.status != 200:
        assert native.runs == dynamo.runs == []
        # Capture is failed alike, and neither path staged the call.
        assert native.failed == dynamo.failed == list(case.failed)
        assert native.staged == dynamo.staged == []
        return
    # The engine ran the same prompt, media and SamplingParams.
    ((native_prompt, native_params, native_media),) = native.runs
    ((dynamo_prompt, dynamo_params, dynamo_media),) = dynamo.runs
    assert dynamo_prompt == native_prompt
    _assert_same_media(dynamo_media, native_media)
    assert (native_media is not None) == (case.kit == "vlm")
    assert dynamo_params == native_params
    # Capture staged the same record and tensors, in the expected mode.
    assert [record.mode for record, _ in native.staged] == (
        [case.mode] if case.mode else []
    )
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
    assert [call["function"] for call in choice["delta"].get("tool_calls") or []] == [
        call["function"] for call in message.get("tool_calls") or []
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


@pytest.mark.parametrize(
    "observation",
    [
        OBSERVATION,
        # Past the context window: Gym measures to decide whether to compact.
        {"role": "user", "content": "And in Lyon?" + " Lyon" * MAX_MODEL_LEN},
    ],
    ids=["fits", "past_the_context"],
)
def test_a_measure_call_counts_the_prompt_the_call_would_run(kits, observation) -> None:
    kit = kits("qwen3")
    extras = {"required_prefix_token_ids": kit.turn_1_tokens()}
    body = {
        "model": kit.path,
        "messages": [*HISTORY, observation],
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

    # Native Gym measures with /tokenize; through Dynamo, the chat call.
    assert (native.status, dynamo.status) == (200, 200), dynamo.reply
    assert (
        dynamo.reply["nvext"]["nemo_rl"]["prompt_token_count"] == native.reply["count"]
    )
    assert dynamo.runs == []


def _cut_call(
    kit: _Kit, prompt_token_ids: list[int], saved: int, *, output_limit: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cut a call that has generated ``saved`` of REPLY's tokens, as a checkpoint does.

    Returns the cut's staged rows by key, and the continuation Gym puts in the
    admission of the call it re-sends after a restore.
    """
    host, _, _ = _host(kit, {}, 0)
    sink = _MemorySink()
    host._capture_sink = sink
    request = SimpleNamespace(ng_capture=_admission(mode="text"), stream=False)
    VllmAsyncGenerationWorkerImpl._begin_request_capture(
        host, request, prompt_token_ids
    )
    state = host._capture_registry.get(request)
    state.set_effective_output_limit(output_limit)
    reply_ids = kit.ids(REPLY)
    state.record(
        reply_ids[:saved],
        [-0.5] * saved,
        finish_reason="stop" if saved == len(reply_ids) else None,
    )
    (ack,) = (
        _cuts(host, tokenizer=kit.tokenizer)
        .checkpoint_cut(
            GenerationCutInventory.build(
                checkpoint_id="checkpoint-1",
                server_name="policy_model",
                active_prefixes=[
                    GenerationCutPrefix(
                        ticket_id="ticket-1",
                        rollout_id="rollout-1",
                        attempt=0,
                        model_call_id="c2",
                        admitted_at=1.0,
                    )
                ],
            )
        )
        .prefixes
    )
    assert ack.disposition == "durable_prefix"
    (cut_key,) = ack.staging_keys
    cut_record = sink.generation_prefix_records[-1][1]
    return {cut_key: cut_record}, {
        "source_capture_key": "rollout-1",
        "source_model_call_id": "c2",
        "staging_keys": [cut_key],
        "prefix_token_count": ack.prefix_token_count,
        "prefix_digest": cut_record.digest,
        "effective_output_limit": ack.effective_output_limit,
        "terminal_finish_reason": ack.terminal_finish_reason,
        "terminal_stop_reason": ack.terminal_stop_reason,
    }


def _restored_host(kit: _Kit, records: dict[str, Any], resumed: int):
    """A host after a restore: fresh, with the cut's rows in TQ."""
    host, sink, failed = _host(kit, {}, 0)
    host._capture_sink = _MemorySink()
    _cuts(
        host,
        staging_source=_MemoryPrefixSource({}, records=records),
        tokenizer=kit.tokenizer,
    )
    host.llm.resumed = resumed
    return host, sink, failed


def _reply_text(saved: int, kit: _Kit) -> str:
    return kit.tokenizer.decode(kit.ids(REPLY)[:saved])


@pytest.mark.parametrize(
    ("cut_after", "rows_kept"),
    [
        # The cut lands inside the reasoning, ...
        ("<think>\nchecking", True),
        # ... after it ended, so the reasoning parser must see the saved part, ...
        ("</think>\n\n<tool_call>", True),
        # ... inside the tool call's JSON, ...
        ('{"name": "get_weather"', True),
        # ... or after the call had finished: nothing is left to decode.
        (REPLY, True),
        # A cut whose rows are gone is dropped, and the call regenerates.
        ("</think>\n\n<tool_call>", False),
    ],
    ids=["in_reasoning", "after_reasoning", "in_tool_call", "finished", "rows_gone"],
)
def test_a_call_cut_at_a_checkpoint_resumes_to_the_uninterrupted_reply(
    kits, cut_after: str, rows_kept: bool
) -> None:
    kit = kits("qwen3")
    reply_ids = kit.ids(REPLY)
    # The cut saved the tokens up to the end of ``cut_after``.
    saved = next(
        count
        for count in range(1, len(reply_ids) + 1)
        if cut_after in _reply_text(count, kit)
    )
    body = {
        "model": kit.path,
        "messages": [*HISTORY, OBSERVATION],
        "tools": TOOLS,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 64,
        "logprobs": True,
        "top_logprobs": 0,
        "ng_capture": _admission(mode="text"),
    }
    # Each path in a task of its own, as each request has on a server: a
    # finished prefix is marked in a context variable.
    whole = asyncio.run(_native(kit, *_host(kit, {}, 0), body, measure=False))
    ((prompt, params, _),) = whole.runs
    ((whole_record, _),) = whole.staged
    records, continuation = _cut_call(
        kit, prompt, saved, output_limit=params.max_tokens
    )
    resumed = saved if rows_kept else 0
    extras = {
        "ng_capture": CaptureAdmission(
            capture_key="rollout-1-a1",
            model_call_id="c3",
            mode="text",
            generation_cut=continuation,
        ).model_dump(mode="json")
    }
    rows = records if rows_kept else {}
    native = asyncio.run(
        _native(
            kit, *_restored_host(kit, rows, resumed), {**body, **extras}, measure=False
        )
    )
    dynamo = asyncio.run(
        _dynamo(kit, *_restored_host(kit, rows, resumed), {**body, **extras}, extras)
    )

    assert (native.status, dynamo.status) == (200, 200), (native.reply, dynamo.reply)
    # The engine ran the saved rows as its prompt, with the budget that is left.
    ((native_prompt, native_params, _),) = native.runs
    ((dynamo_prompt, dynamo_params, _),) = dynamo.runs
    assert native_prompt == dynamo_prompt == prompt + reply_ids[:resumed]
    assert dynamo_params == native_params
    if resumed < len(reply_ids):
        assert native_params.max_tokens == params.max_tokens - resumed
    # Capture staged one row for the call: the saved tokens, then the rest.
    for outcome in (native, dynamo):
        ((record, _),) = outcome.staged
        assert record.token_ids_delta == whole_record.token_ids_delta
        assert record.token_mask_delta == whole_record.token_mask_delta
        assert (
            record.generation_log_probs_delta == whole_record.generation_log_probs_delta
        )
        assert outcome.failed == []
    # Gym reads the reply the uninterrupted call gave.
    whole_choice = whole.reply["choices"][0]
    native_choice = native.reply["choices"][0]
    (dynamo_choice,) = dynamo.reply["choices"]
    for choice, message in (
        (native_choice, native_choice["message"]),
        (dynamo_choice, dynamo_choice["delta"]),
    ):
        assert choice["finish_reason"] == whole_choice["finish_reason"]
        assert message["content"] == whole_choice["message"]["content"]
        assert (
            message.get("reasoning", message.get("reasoning_content"))
            == whole_choice["message"]["reasoning"]
        )
        assert [call["function"] for call in message["tool_calls"]] == [
            call["function"] for call in whole_choice["message"]["tool_calls"]
        ]
    counts = ("prompt_tokens", "completion_tokens", "total_tokens")
    for usage in (native.reply["usage"], dynamo.reply["usage"]):
        assert [usage[count] for count in counts] == [
            whole.reply["usage"][count] for count in counts
        ]
