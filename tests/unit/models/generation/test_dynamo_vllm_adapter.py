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
"""The vLLM Dynamo adapter over a fake AsyncLLM.

The host is a real ``VllmAsyncGenerationWorkerImpl`` built without its
constructor, so rollouts and NeMo-Gym calls run the native worker's own helpers.
"""

import ast
import asyncio
import base64
import importlib.util
import json
import logging
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.vllm
pytest.importorskip("dynamo._core")
vllm = pytest.importorskip("vllm")

from dynamo.llm.exceptions import InvalidArgument  # noqa: E402
from vllm.multimodal.inputs import PlaceholderRange  # noqa: E402

from nemo_rl.models.generation.dynamo import engine_adapter  # noqa: E402
from nemo_rl.models.generation.dynamo.config import (  # noqa: E402
    DynamoCfg,
    DynamoFrontendArgs,
)
from nemo_rl.models.generation.dynamo.rollout_client import (  # noqa: E402
    ROLLOUT_ANNOTATION,
)
from nemo_rl.models.generation.dynamo.vllm import (  # noqa: E402
    adapter as vllm_adapter,
)
from nemo_rl.models.generation.dynamo.vllm.chat_engine import (  # noqa: E402
    GYM_CALL_ANNOTATION,
    PLAN_KEY,
    RUNTIME_DATA_KEY,
    ForwardedCall,
)
from nemo_rl.models.generation.interfaces import SamplerReplayConfig  # noqa: E402
from nemo_rl.models.generation.rollout_result import (  # noqa: E402
    rollout_result_from_wire,
    rollout_result_to_wire,
)
from nemo_rl.models.generation.vllm.chat_server import (  # noqa: E402
    NeMoRLChatCompletionRequest,
)
from nemo_rl.models.generation.vllm.request_pipeline import (  # noqa: E402
    rollout_output,
    rollout_result,
)
from nemo_rl.models.generation.vllm.utils import R3_MISSING_ROUTE_SENTINEL  # noqa: E402
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    VllmAsyncGenerationWorkerImpl,
)
from nemo_rl.utils.routed_experts_codec import decode_routed_experts  # noqa: E402

MAX_MODEL_LEN = 16
EOS = 2
# A processed image's placeholder tokens.
IMAGE_TOKEN, IMAGE_TOKENS = 91, 3


class _Context:
    def id(self) -> str:
        return "request-0"

    def is_stopped(self) -> bool:
        return False


def _logprob(value: float, rank: int = 1):
    return SimpleNamespace(logprob=value, rank=rank, decoded_token=None)


def _completion(token_ids, finish_reason=None, index=0, routed_experts=None):
    return SimpleNamespace(
        index=index,
        text="".join(f"<{token_id}>" for token_id in token_ids),
        token_ids=token_ids,
        logprobs=[{t: _logprob(-0.5)} for t in token_ids],
        finish_reason=finish_reason,
        stop_reason=None,
        routed_experts=routed_experts,
    )


def _output(prompt, *completions, prompt_routed_experts=None):
    return SimpleNamespace(
        prompt_token_ids=prompt,
        num_cached_tokens=0,
        prompt_routed_experts=prompt_routed_experts,
        outputs=list(completions),
    )


class _FakeLLM:
    """Yields scripted RequestOutputs and records what it was asked to generate."""

    def __init__(self, outputs=(), *, defaults=None, admission_error=None):
        self.outputs = list(outputs)
        self.calls = []
        self.admission_error = admission_error
        self.tokenizer = None
        self.model_config = SimpleNamespace(
            get_diff_sampling_param=lambda: dict(defaults or {}),
            try_get_generation_config=lambda: {},
            generation_config="auto",
            override_generation_config={},
        )
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
            kv_events_config=None,
        )

    def check_admission(self, n, request_id):
        if self.admission_error is not None:
            raise self.admission_error

    async def generate(self, prompt, sampling_params, request_id, **kwargs):
        self.calls.append((prompt, sampling_params, request_id, kwargs))
        for output in self.outputs:
            yield output


def _host(llm: _FakeLLM, **cfg) -> VllmAsyncGenerationWorkerImpl:
    host = object.__new__(VllmAsyncGenerationWorkerImpl)
    host._init_token_capture_state()
    host.cfg = {
        "max_new_tokens": 8,
        "temperature": 0.7,
        "top_p": 0.9,
        "val_temperature": 0.0,
        "val_top_p": 1.0,
        "top_k": None,
        "val_top_k": None,
        "stop_token_ids": [EOS],
        "stop_strings": ["</done>"],
        "vllm_cfg": {"max_model_len": MAX_MODEL_LEN},
        **cfg,
    }
    host.SamplingParams = vllm.SamplingParams
    host.model_name = "model"
    host.routed_experts_dtype = torch.int16
    host.llm = llm
    host._http_engine_client = None
    host._capture_media = False
    host._capture_top_logprobs = 0
    host.token_capture = None
    return host


def _collect(engine, request, context=None) -> list[dict]:
    async def run():
        return [
            chunk async for chunk in engine.generate(request, context or _Context())
        ]

    return asyncio.run(run())


def _rollout_request(prompt: list[int], **rollout) -> dict:
    spec = {"greedy": False, "stop_strings": None, **rollout}
    return {"token_ids": prompt, "annotations": [ROLLOUT_ANNOTATION + json.dumps(spec)]}


def test_rollouts_are_served_like_generate_async() -> None:
    final = _output([1, 2, 3], _completion([10, 11, 12], "stop"))
    llm = _FakeLLM([final])
    host = _host(llm)
    engine = vllm_adapter.DynamoVllmAdapter(host, leader_worker_idx=3)

    (chunk,) = _collect(
        engine, _rollout_request([1, 2, 3], greedy=True, stop_strings=["</x>"])
    )

    ((_, sampling_params, _, _),) = llm.calls
    # The worker's _serve_rollout samples it, as it does for generate_async...
    assert sampling_params.max_tokens == 8
    assert sampling_params.temperature == 0.0
    assert sorted(sampling_params.stop) == ["</done>", "</x>"]
    # ...and builds the result from its final output.
    expected = rollout_result(
        final,
        input_length=3,
        return_routed_experts=False,
        routed_experts_dtype=torch.int16,
        device=torch.device("cpu"),
    )
    assert chunk["token_ids"] == [10, 11, 12]
    assert chunk["engine_data"] == {
        "rollout": rollout_result_to_wire(expected),
        "gen_leader_worker_idx": 3,
    }


def test_a_cancelled_rollout_aborts_the_vllm_request_it_runs() -> None:
    llm = _FakeLLM([_output([1, 2, 3], _completion([10], "stop"))])
    aborted = []

    async def abort(request_id):
        aborted.append(request_id)

    llm.abort = abort
    engine = vllm_adapter.DynamoVllmAdapter(_host(llm), leader_worker_idx=0)

    _collect(engine, _rollout_request([1, 2, 3]))
    # Dynamo's worker calls abort() when the frontend cancels a request.
    asyncio.run(engine.abort(_Context()))

    ((_, _, request_id, _),) = llm.calls
    assert aborted == [request_id]


def test_router_replay_routes_reach_rollouts_as_natively() -> None:
    routes = np.arange(5 * 2 * 2, dtype=np.int16).reshape(5, 2, 2)
    final = _output(
        [1, 2, 3],
        _completion([10, 11, 12], "stop", routed_experts=routes[3:]),
        prompt_routed_experts=routes[:3],
    )
    host = _host(_FakeLLM([final]), vllm_kwargs={"enable_return_routed_experts": True})
    engine = vllm_adapter.DynamoVllmAdapter(host, leader_worker_idx=0)

    (chunk,) = _collect(engine, _rollout_request([1, 2, 3]))

    input_ids = torch.tensor([1, 2, 3])
    shipped = rollout_output(
        rollout_result_from_wire(chunk["engine_data"]["rollout"]),
        input_ids=input_ids,
        input_length=3,
    )
    native = rollout_output(
        rollout_result(
            final,
            input_length=3,
            return_routed_experts=True,
            routed_experts_dtype=torch.int16,
            device=torch.device("cpu"),
        ),
        input_ids=input_ids,
        input_length=3,
    )
    assert shipped.keys() == native.keys()
    assert "routed_experts" in shipped
    for key in native:
        assert torch.equal(shipped[key], native[key]), key


def test_a_rollout_with_no_room_returns_the_prompt_untruncated() -> None:
    llm = _FakeLLM()
    engine = vllm_adapter.DynamoVllmAdapter(_host(llm), leader_worker_idx=0)

    (chunk,) = _collect(engine, _rollout_request(list(range(MAX_MODEL_LEN))))

    assert llm.calls == []
    rollout = chunk["engine_data"]["rollout"]
    assert rollout["token_ids"] == [] and rollout["truncated"] is False


def _start_worker_config(monkeypatch, host) -> engine_adapter.WorkerConfig:
    """The WorkerConfig ``start_dynamo_endpoint`` registers ``host``'s engine with."""
    configs = []

    class _Worker:
        def __init__(self, engine, config):
            configs.append(config)

        async def run(self):
            return None

    monkeypatch.setattr(engine_adapter, "Worker", _Worker)
    # Registered so the engine's replacement of Dynamo's logging setup is undone.
    monkeypatch.setattr(
        engine_adapter.dynamo_worker,
        "configure_dynamo_logging",
        engine_adapter.dynamo_worker.configure_dynamo_logging,
    )
    env = {
        "DYN_DISCOVERY_BACKEND": "etcd",
        "DYN_REQUEST_PLANE": "tcp",
        "DYN_EVENT_PLANE": "nats",
    }
    for key, value in env.items():
        # Registered here so the variables the engine exports are restored.
        monkeypatch.setenv(key, value)

    async def start():
        await engine_adapter.start_dynamo_endpoint(
            vllm_adapter.DynamoVllmAdapter(host, leader_worker_idx=0),
            model_path="/snapshots/model",
            served_model_name=host.model_name,
            namespace="ns",
            engine_env=env,
        )

    asyncio.run(start())
    (config,) = configs
    return config


def test_engines_register_for_rollouts_and_nemo_gym_chat(monkeypatch) -> None:
    config = _start_worker_config(monkeypatch, _host(_FakeLLM()))

    assert config.namespace == "ns"
    # Registered from the weight-free snapshot, served under the model's name.
    assert (config.model_name, config.served_model_name) == (
        "/snapshots/model",
        "model",
    )
    # NeMo RL's rollouts go to /v1/completions, NeMo-Gym's calls to chat.
    assert config.endpoint_types == "chat,completions"
    # Worker.run() starts with Dynamo's logging setup, which keeps the worker's.
    handlers = logging.getLogger().handlers[:]
    engine_adapter.dynamo_worker.configure_dynamo_logging()
    assert logging.getLogger().handlers == handlers


def test_worker_run_still_calls_the_logging_setup_the_engine_replaces() -> None:
    # Otherwise a Dynamo bump would silently take over the worker's logging again.
    run = engine_adapter.Worker.run
    assert run.__globals__ is vars(engine_adapter.dynamo_worker)
    assert "configure_dynamo_logging" in run.__code__.co_names


def test_requests_other_than_nemo_rls_are_refused() -> None:
    llm = _FakeLLM()
    engine = vllm_adapter.DynamoVllmAdapter(_host(llm), leader_worker_idx=0)

    with pytest.raises(InvalidArgument, match="only NeMo RL's own requests"):
        _collect(engine, {"token_ids": [1, 2, 3, 4]})

    assert llm.calls == []


def test_an_engine_error_reaches_the_driver_with_its_cause() -> None:
    host = _host(_FakeLLM())

    async def serve_rollout(*args, **kwargs):
        raise RuntimeError("the generation output did not include routed_experts")

    host._serve_rollout = serve_rollout
    engine = vllm_adapter.DynamoVllmAdapter(host, leader_worker_idx=0)

    # A 5xx would reach the driver as "Internal server error".
    with pytest.raises(InvalidArgument, match="RuntimeError: .*routed_experts"):
        _collect(engine, _rollout_request([1, 2, 3]))


def test_an_endpoint_that_stops_serving_reports_why() -> None:
    async def endpoint_errors() -> list:
        async def registration_fails() -> None:
            raise RuntimeError("etcd unreachable")

        host = SimpleNamespace(_dynamo_endpoint_task=None)
        errors = [await VllmAsyncGenerationWorkerImpl.dynamo_endpoint_error(host)]
        host._dynamo_endpoint_task = asyncio.get_running_loop().create_task(
            registration_fails()
        )
        await asyncio.sleep(0)
        errors.append(await VllmAsyncGenerationWorkerImpl.dynamo_endpoint_error(host))
        return errors

    # Not registered yet, then the driver's registration wait gets the cause.
    assert asyncio.run(endpoint_errors()) == [
        None,
        "RuntimeError('etcd unreachable')",
    ]


def test_engines_register_the_block_size_kv_events_are_hashed_with() -> None:
    llm = _FakeLLM()
    llm.vllm_config.cache_config = SimpleNamespace(block_size=16, num_gpu_blocks=64)
    llm.vllm_config.scheduler_config = SimpleNamespace(
        max_num_seqs=8, max_num_batched_tokens=256
    )

    async def call_utility_async(method):
        assert method == "get_kv_cache_group_metadata"
        # A hybrid model: the Mamba group's block size is not the attention one.
        return [
            {"kind": "mamba", "block_size": 1056},
            {"kind": "full_attention", "block_size": 528},
        ]

    llm.engine_core = SimpleNamespace(call_utility_async=call_utility_async)
    host = _host(llm)
    host.llm_async_engine_args = vllm.AsyncEngineArgs(model="/models/m")
    engine = vllm_adapter.DynamoVllmAdapter(host, leader_worker_idx=0)

    config = asyncio.run(engine.start(worker_id=0))
    registration = config.llm

    # The KV router hashes events in blocks of the main-attention group; a
    # wrong size would quietly turn KV routing into load-only routing.
    assert registration.kv_cache_block_size == 528
    assert registration.total_kv_blocks == 64
    assert registration.context_length == MAX_MODEL_LEN
    # The frontend's chat engine renders with what the engine publishes.
    assert config.runtime_data == vllm_adapter.chat_runtime_data(host)


def test_engines_publish_what_the_frontends_chat_engine_renders_with() -> None:
    host = _host(_FakeLLM())
    host.llm_async_engine_args = vllm.AsyncEngineArgs(
        model="/models/m",
        max_model_len=MAX_MODEL_LEN,
        enable_prefix_caching=True,
        skip_tokenizer_init=True,
    )
    serving_chat_kwargs = {"tool_parser": "hermes", "enable_auto_tools": True}
    host.cfg["vllm_cfg"] = {
        "max_model_len": MAX_MODEL_LEN,
        "http_server_serving_chat_kwargs": serving_chat_kwargs,
        "reasoning_parser_plugin": "/plugins/reasoning.py",
    }
    # MasterConfig validates this block into a pydantic model.
    host.cfg["sampler_replay"] = SamplerReplayConfig(enabled=True)

    # Dynamo carries runtime_data as JSON.
    spec = json.loads(json.dumps(vllm_adapter.chat_runtime_data(host)))[
        RUNTIME_DATA_KEY
    ]

    # The engine's own args rebuild the same model config, tokenizer included.
    published = vllm.AsyncEngineArgs(**spec["engine_args"])
    for field in ("model", "max_model_len", "enable_prefix_caching"):
        assert getattr(published, field) == getattr(
            host.llm_async_engine_args, field
        ), field
    assert published.skip_tokenizer_init is False
    # All of policy.generation, which the native server's chat helpers read.
    assert spec["generation"] == {**host.cfg, "sampler_replay": {"enabled": True}}


# --- NeMo-Gym calls, after the frontend's chat engine rendered them -----------


def _chat(**fields) -> dict:
    """A rendered chat request, as the chat engine sends it (without messages)."""
    return NeMoRLChatCompletionRequest(
        model="model",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
        top_p=0.9,
        **fields,
    ).model_dump(mode="json", exclude={"messages"})


def _gym_call(token_ids: list[int], **fields) -> dict:
    """A Gym call as the chat engine routes it: final token ids and the plan.

    ``fields`` set the prepared prompt's fields and the call's others by name.
    """
    plan = {
        "template_token_ids": token_ids,
        "splice": None,
        "prefix_len": None,
        "measure_only": False,
        "request_max_tokens": None,
        "max_tokens": None,
        "admission": None,
        "serving_admission": None,
        "verify_retained_media": False,
    }
    wire: ForwardedCall = {
        "prepared_prompt": plan,
        "request": _chat(),
        "multimodal": None,
        "reasoning_ended": None,
        "reasoning_parser_kwargs": None,
    }
    for name, value in fields.items():
        if name in plan:
            plan[name] = value
        else:
            assert name in wire, name
            wire[name] = value
    return {
        "token_ids": token_ids,
        "annotations": [GYM_CALL_ANNOTATION],
        "extra_args": {PLAN_KEY: wire},
    }


@pytest.mark.parametrize(
    ("max_tokens", "served_max_tokens"), [(5, 5), (None, MAX_MODEL_LEN - 3)]
)
def test_gym_calls_generate_with_the_native_servers_sampling_params(
    max_tokens, served_max_tokens
) -> None:
    final = _output([1, 3, 4], _completion([10, 11], "stop"))
    llm = _FakeLLM([final])
    engine = vllm_adapter.DynamoVllmAdapter(_host(llm), leader_worker_idx=0)
    reasoning = {"chat_template_kwargs": {"enable_thinking": True}}

    (chunk,) = _collect(
        engine,
        _gym_call(
            [1, 3, 4],
            # Gym's token information: logprobs=true, top_logprobs=0.
            request=_chat(
                max_tokens=max_tokens, stop=["</s>"], logprobs=True, top_logprobs=0
            ),
            reasoning_ended=False,
            reasoning_parser_kwargs=reasoning,
        ),
    )

    ((prompt, params, _, kwargs),) = llm.calls
    assert prompt == {"prompt_token_ids": [1, 3, 4]}
    # The request's own sampling, as vLLM's chat server builds it.
    assert (params.temperature, params.top_p) == (0.7, 0.9)
    assert (params.max_tokens, params.stop, params.logprobs) == (
        served_max_tokens,
        ["</s>"],
        0,
    )
    assert params.output_kind == vllm.sampling_params.RequestOutputKind.FINAL_ONLY
    # What vLLM's chat serving hands the engine besides the prompt.
    assert kwargs == {"reasoning_ended": False, "reasoning_parser_kwargs": reasoning}
    # The output the frontend builds the reply from, and the call's result.
    assert chunk["engine_data"] == {
        "output": {
            "text": "<10><11>",
            "token_ids": [10, 11],
            "finish_reason": "stop",
            "stop_reason": None,
            "logprobs": [{"10": [-0.5, 1, None]}, {"11": [-0.5, 1, None]}],
        },
        "result": {
            "prompt_token_ids": [1, 3, 4],
            "generation_token_ids": [10, 11],
            "generation_log_probs": [-0.5, -0.5],
        },
    }
    assert chunk["completion_usage"]["prompt_tokens"] == 3


def test_an_uncaptured_gym_call_carries_its_routes() -> None:
    # vLLM routes every token but the last sampled one.
    routes = np.arange(4 * 2 * 2, dtype=np.int16).reshape(4, 2, 2)
    final = _output(
        [1, 3, 4],
        _completion([10, 11], "stop", routed_experts=routes[3:]),
        prompt_routed_experts=routes[:3],
    )
    host = _host(_FakeLLM([final]), vllm_kwargs={"enable_return_routed_experts": True})
    engine = vllm_adapter.DynamoVllmAdapter(host, leader_worker_idx=0)

    (chunk,) = _collect(engine, _gym_call([1, 3, 4]))

    # As the native endpoint attaches them: one row per token of the sequence.
    expected = torch.full((5, 2, 2), R3_MISSING_ROUTE_SENTINEL, dtype=torch.int16)
    expected[:4] = torch.from_numpy(routes)
    shipped = chunk["engine_data"]["result"]["routed_experts"]
    assert torch.equal(decode_routed_experts(shipped, dtype=None), expected)


def test_a_gym_call_without_routes_fails_under_router_replay() -> None:
    final = _output([1, 3, 4], _completion([10, 11], "stop"))
    host = _host(_FakeLLM([final]), vllm_kwargs={"enable_return_routed_experts": True})
    engine = vllm_adapter.DynamoVllmAdapter(host, leader_worker_idx=0)

    # Training would replay the router on routes the call never returned.
    with pytest.raises(InvalidArgument, match="did not include routed_experts"):
        _collect(engine, _gym_call([1, 3, 4]))


def test_an_uncaptured_gym_call_with_non_finite_logprobs_fails_as_natively() -> None:
    completion = _completion([10, 11], "length")
    completion.logprobs[1] = {11: _logprob(float("nan"))}
    llm = _FakeLLM([_output([1, 3, 4], completion)])
    engine = vllm_adapter.DynamoVllmAdapter(_host(llm), leader_worker_idx=0)

    # Dynamo would hand the frontend a null log-prob. Fail the call instead: the
    # chat engine answers an engine's error with 500, as natively, and Gym retries.
    with pytest.raises(
        InvalidArgument, match=r"non-finite log-prob \(nan\) at token 1"
    ):
        _collect(engine, _gym_call([1, 3, 4]))


def test_a_gym_call_without_the_chat_engines_plan_is_refused() -> None:
    llm = _FakeLLM()
    engine = vllm_adapter.DynamoVllmAdapter(_host(llm), leader_worker_idx=0)

    with pytest.raises(InvalidArgument, match=r"extra_args\.nemo_rl"):
        _collect(engine, {"token_ids": [1, 3, 4], "annotations": [GYM_CALL_ANNOTATION]})

    assert llm.calls == []


@pytest.mark.parametrize(
    ("chat", "admission_error", "status", "acknowledge", "message"),
    [
        # vLLM's own validation errors keep their 400 and message, as natively,
        # and acknowledge a capture call's refusal.
        (_chat(min_tokens=10, max_tokens=5), None, 400, True, "min_tokens"),
        # The queue limits the native server checks: a 503 to retry elsewhere.
        (_chat(), vllm.exceptions.QueueOverflowError(), 503, False, "busy"),
    ],
    ids=["request_error", "queue_full"],
)
def test_a_gym_call_the_engine_refuses_answers_with_the_native_status(
    chat, admission_error, status, acknowledge, message
) -> None:
    llm = _FakeLLM(
        [_output([1, 3, 4], _completion([10], "stop"))],
        admission_error=admission_error,
    )
    engine = vllm_adapter.DynamoVllmAdapter(_host(llm), leader_worker_idx=0)

    # In the chunk: Dynamo hands a raised error to the chat engine without it.
    (chunk,) = _collect(engine, _gym_call([1, 3, 4], request=chat))

    refusal = chunk["engine_data"]["refusal"]
    assert (refusal["status"], refusal["acknowledge"]) == (status, acknowledge)
    assert message in refusal["message"]
    assert llm.calls == []


def test_a_spliced_media_call_keeps_placeholders_on_their_tokens() -> None:
    # Turn 1 ran [1, <image>, 3] and generated [5, 6, EOS]; the template
    # re-rendered that turn as [7, EOS] and appended a turn with a second image.
    image = [IMAGE_TOKEN] * IMAGE_TOKENS
    template = [1, *image, 3, 7, EOS, *image, 4]
    prefix = [1, *image, 3, 5, 6, EOS]
    spliced = [*prefix, *image, 4]
    multimodal = {
        "type": "multimodal",
        "mm_kwargs": {"image": ["pixels-0", "pixels-1"]},
        "mm_hashes": {"image": ["h0", "h1"]},
        "mm_placeholders": {
            "image": [
                PlaceholderRange(1, IMAGE_TOKENS),
                PlaceholderRange(7, IMAGE_TOKENS),
            ]
        },
    }
    llm = _FakeLLM([_output(spliced, _completion([10], "stop"))])
    engine = vllm_adapter.DynamoVllmAdapter(_host(llm), leader_worker_idx=0)

    _collect(
        engine,
        _gym_call(
            spliced,
            template_token_ids=template,
            # Each side cut at its turn end: prefix[:7] + template[6:].
            splice=[7, 6],
            multimodal=base64.b64encode(pickle.dumps(multimodal)).decode(),
        ),
    )

    ((engine_input, params, _, _),) = llm.calls
    assert engine_input["prompt_token_ids"] == spliced
    assert engine_input["mm_kwargs"] == multimodal["mm_kwargs"]
    # The first image sits in the model's own prefix; the second moved with the
    # longer generated turn, as the native endpoint remaps them.
    assert [
        (span.offset, span.length) for span in engine_input["mm_placeholders"]["image"]
    ] == [(1, IMAGE_TOKENS), (len(prefix), IMAGE_TOKENS)]
    assert params.max_tokens == MAX_MODEL_LEN - len(spliced)


@pytest.mark.parametrize(
    ("router_mode", "prefix_caching", "publishes"),
    [("kv", True, True), ("kv", False, False), ("round-robin", True, False)],
)
def test_engines_publish_kv_events_only_where_the_kv_router_can_use_them(
    router_mode, prefix_caching, publishes
) -> None:
    cfg = DynamoCfg(frontend_args=DynamoFrontendArgs(router_mode=router_mode))

    config = vllm_adapter.dynamo_kv_events_config(cfg, prefix_caching=prefix_caching)

    # vLLM publishes block events only for blocks its prefix cache stores.
    assert (config is not None) is publishes
    if publishes:
        assert config["enable_kv_cache_events"] and config["publisher"] == "zmq"
        # A per-engine abstract socket, which vLLM's publisher binds and Dynamo's
        # source connects to: no port for two engines on a node to collide on.
        assert config["endpoint"].startswith("ipc://@")
        other = vllm_adapter.dynamo_kv_events_config(cfg, prefix_caching=True)
        assert other["endpoint"] != config["endpoint"]


def test_main_attention_kinds_match_dynamos() -> None:
    # Read from source: importing dynamo.vllm would import Dynamo's vLLM integration.
    spec = importlib.util.find_spec("dynamo.vllm.cache_info")
    tree = ast.parse(Path(spec.origin).read_text())
    (dynamos,) = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            getattr(t, "id", None) == "MAIN_ATTENTION_KV_CACHE_KINDS"
            for t in node.targets
        )
    ]
    assert {kind.value for kind in vllm_adapter._MAIN_ATTENTION_KINDS} == dynamos
