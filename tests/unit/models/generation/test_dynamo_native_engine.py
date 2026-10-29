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
"""NeMo RL's Dynamo engine adapter over a fake AsyncLLM.

The host is a real ``VllmAsyncGenerationWorkerImpl`` built without its
constructor, so rollouts are sampled by the native worker's own helpers.
"""

import asyncio
import base64
import io
import json
from http import HTTPStatus
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.vllm
pytest.importorskip("dynamo._core")
vllm = pytest.importorskip("vllm")

from dynamo.llm.exceptions import HttpError, InvalidArgument  # noqa: E402
from vllm.exceptions import QueueOverflowError  # noqa: E402
from vllm.multimodal.inputs import PlaceholderRange  # noqa: E402

from nemo_rl.models.generation.dynamo import native_engine  # noqa: E402
from nemo_rl.models.generation.dynamo.config import DynamoWorkerArgs  # noqa: E402
from nemo_rl.models.generation.dynamo.rollout_client import (  # noqa: E402
    ROLLOUT_ANNOTATION,
)
from nemo_rl.models.generation.vllm.request_pipeline import (  # noqa: E402
    rollout_output,
    rollout_result,
    rollout_result_from_wire,
    rollout_result_to_wire,
)
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    VllmAsyncGenerationWorkerImpl,
)

MAX_MODEL_LEN = 16


EOS = 2
# The fake processor expands each image placeholder to IMAGE_TOKENS image tokens.
IMAGE_PLACEHOLDER, IMAGE_TOKEN, IMAGE_TOKENS = 90, 91, 3


class _Context:
    def id(self) -> str:
        return "request-0"

    def is_stopped(self) -> bool:
        return False


class _Tokenizer:
    eos_token_id = EOS

    def decode(self, token_ids) -> str:
        return "".join(f"<{token_id}>" for token_id in token_ids)


def _logprob(value: float):
    return SimpleNamespace(logprob=value, rank=1, decoded_token=None)


def _completion(token_ids, finish_reason=None, index=0, routed_experts=None):
    return SimpleNamespace(
        index=index,
        token_ids=token_ids,
        logprobs=[{t: _logprob(-0.5)} for t in token_ids],
        finish_reason=finish_reason,
        routed_experts=routed_experts,
    )


def _output(prompt, *completions, prompt_routed_experts=None, prompt_logprobs=None):
    return SimpleNamespace(
        prompt_token_ids=prompt,
        num_cached_tokens=0,
        prompt_logprobs=prompt_logprobs,
        prompt_routed_experts=prompt_routed_experts,
        outputs=list(completions),
    )


class _FakeRenderer:
    """vLLM's processing step: expands each image placeholder, as a VLM processor does."""

    def __init__(self) -> None:
        self.prompts = []

    async def process_for_engine_async(
        self, prompt, arrival_time, *, skip_mm_cache=False
    ):
        self.prompts.append((prompt, skip_mm_cache))
        if "multi_modal_data" not in prompt:
            return {"type": "token", "prompt_token_ids": prompt["prompt_token_ids"]}
        tokens, placeholders = [], []
        for token in prompt["prompt_token_ids"]:
            if token == IMAGE_PLACEHOLDER:
                placeholders.append(PlaceholderRange(len(tokens), IMAGE_TOKENS))
                tokens.extend([IMAGE_TOKEN] * IMAGE_TOKENS)
            else:
                tokens.append(token)
        return {
            "type": "multimodal",
            "prompt_token_ids": tokens,
            "mm_kwargs": {"image": [object() for _ in placeholders]},
            "mm_placeholders": {"image": placeholders},
        }


class _FakeLLM:
    """Yields scripted RequestOutputs and records what it was asked to generate."""

    def __init__(self, outputs=(), *, defaults=None, admission_error=None):
        self.outputs = list(outputs)
        self.calls = []
        self.admission_error = admission_error
        self.tokenizer = None
        self.renderer = _FakeRenderer()
        self.model_config = SimpleNamespace(
            get_diff_sampling_param=lambda: dict(defaults or {}),
            multimodal_config=None,
            allowed_local_media_path="",
            allowed_media_domains=None,
        )
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
            kv_events_config=None,
        )

    def check_admission(self, n, request_id):
        if self.admission_error is not None:
            raise self.admission_error

    async def generate(self, prompt, sampling_params, request_id):
        self.calls.append((prompt, sampling_params, request_id))
        for output in self.outputs:
            yield output


def _host(llm: _FakeLLM, **cfg) -> VllmAsyncGenerationWorkerImpl:
    host = object.__new__(VllmAsyncGenerationWorkerImpl)
    host.cfg = {
        "max_new_tokens": 8,
        "temperature": 0.7,
        "top_p": 0.9,
        "val_temperature": 0.0,
        "val_top_p": 1.0,
        "top_k": None,
        "stop_token_ids": [2],
        "stop_strings": ["</done>"],
        "vllm_cfg": {"max_model_len": MAX_MODEL_LEN},
        **cfg,
    }
    host.SamplingParams = vllm.SamplingParams
    host.model_name = "model"
    host.routed_experts_dtype = torch.int16
    host.llm = llm
    host._capture_media = False
    return host


def _collect(engine, request) -> list[dict]:
    async def run():
        return [chunk async for chunk in engine.generate(request, _Context())]

    return asyncio.run(run())


def _rollout_request(prompt: list[int], **rollout) -> dict:
    spec = {"greedy": False, "stop_strings": None, **rollout}
    return {"token_ids": prompt, "annotations": [ROLLOUT_ANNOTATION + json.dumps(spec)]}


def _openai_request(**fields) -> dict:
    return {"token_ids": [1, 2, 3, 4], **fields}


def test_rollouts_are_served_like_generate_async() -> None:
    final = _output([1, 2, 3], _completion([10, 11, 12], "stop"))
    llm = _FakeLLM([final])
    host = _host(llm)
    engine = native_engine.NativeVllmEngine(host, leader_worker_idx=3)

    (chunk,) = _collect(
        engine, _rollout_request([1, 2, 3], greedy=True, stop_strings=["</x>"])
    )

    ((_, sampling_params, _),) = llm.calls
    # The request the native worker's generate_async builds...
    assert sampling_params == host._rollout_sampling_params(
        max_new_tokens=host._rollout_max_new_tokens(3),
        greedy=True,
        stop_strings=["</x>"],
    )
    assert sorted(sampling_params.stop) == ["</done>", "</x>"]
    # ...and the result the shared pipeline builds from its final output.
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


def test_router_replay_routes_reach_rollouts_as_natively() -> None:
    routes = np.arange(5 * 2 * 2, dtype=np.int16).reshape(5, 2, 2)
    final = _output(
        [1, 2, 3],
        _completion([10, 11, 12], "stop", routed_experts=routes[3:]),
        prompt_routed_experts=routes[:3],
    )
    host = _host(_FakeLLM([final]), vllm_kwargs={"enable_return_routed_experts": True})
    engine = native_engine.NativeVllmEngine(host, leader_worker_idx=0)

    (chunk,) = _collect(engine, _rollout_request([1, 2, 3]))

    input_ids = torch.tensor([1, 2, 3])
    shipped = rollout_output(
        rollout_result_from_wire(chunk["engine_data"]["rollout"]),
        input_ids=input_ids,
        input_length=3,
        pad_token_id=0,
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
        pad_token_id=0,
    )
    assert shipped.keys() == native.keys()
    assert "routed_experts" in shipped
    for key in native:
        assert torch.equal(shipped[key], native[key]), key


def test_a_rollout_with_no_room_returns_the_prompt_untruncated() -> None:
    llm = _FakeLLM()
    engine = native_engine.NativeVllmEngine(_host(llm), leader_worker_idx=0)

    (chunk,) = _collect(engine, _rollout_request(list(range(MAX_MODEL_LEN))))

    assert llm.calls == []
    rollout = chunk["engine_data"]["rollout"]
    assert rollout["token_ids"] == [] and rollout["truncated"] is False


def test_openai_requests_map_onto_sampling_params_as_vllms_server_does() -> None:
    params = native_engine.build_sampling_params(
        _openai_request(
            sampling_options={"top_k": 5, "seed": 7, "n": 2},
            stop_conditions={
                "max_tokens": 100,
                "stop_token_ids": [9],
                "stop_token_ids_hidden": [2],
            },
            output_options={"logprobs": 2},
            extra_args={
                "sampling_options": {
                    "allowed_token_ids": [5, 6],
                    "bad_words_token_ids": [[7]],
                }
            },
        ),
        # The model's generation config, for what the request leaves unset.
        {"temperature": 0.6, "top_k": 20, "max_tokens": 9},
        output_room=MAX_MODEL_LEN - 4,
    )

    assert (params.temperature, params.top_k, params.seed, params.n) == (0.6, 5, 7, 2)
    # The request's max_tokens, capped by the context; the generation config's
    # max_tokens applies only when the request leaves it unset.
    assert params.max_tokens == MAX_MODEL_LEN - 4
    unset = native_engine.build_sampling_params(
        _openai_request(), {"max_tokens": 9}, output_room=MAX_MODEL_LEN - 4
    )
    assert unset.max_tokens == 9
    assert sorted(params.stop_token_ids) == [2, 9]
    assert params.logprobs == 2
    assert params.allowed_token_ids == [5, 6]
    assert params._bad_words_token_ids == [[7]]
    assert params.detokenize is False


def test_guided_decoding_maps_onto_structured_outputs() -> None:
    params = native_engine.build_sampling_params(
        _openai_request(
            sampling_options={"guided_decoding": {"json": {"type": "object"}}}
        ),
        {},
        output_room=MAX_MODEL_LEN - 4,
    )

    assert params.structured_outputs.json == {"type": "object"}


def test_best_of_is_rejected_as_by_vllm() -> None:
    with pytest.raises(InvalidArgument, match="best_of"):
        native_engine.build_sampling_params(
            _openai_request(sampling_options={"n": 1, "best_of": 2}),
            {},
            output_room=MAX_MODEL_LEN - 4,
        )


def test_openai_requests_stream_each_choice_by_index() -> None:
    prompt = [1, 2, 3, 4]
    llm = _FakeLLM(
        [
            _output(prompt, _completion([5], index=0), _completion([7], index=1)),
            _output(prompt, _completion([6], "stop", index=0)),
            _output(prompt, _completion([8], "length", index=1)),
        ]
    )
    engine = native_engine.NativeVllmEngine(_host(llm), leader_worker_idx=0)

    chunks = _collect(engine, _openai_request(sampling_options={"n": 2}))

    assert [(c["index"], c["token_ids"]) for c in chunks] == [
        (0, [5]),
        (1, [7]),
        (0, [6]),
        (1, [8]),
    ]
    assert [c.get("finish_reason") for c in chunks] == [None, None, "stop", "length"]
    assert chunks[-1]["completion_usage"]["completion_tokens"] == 4


def test_prompt_logprobs_are_returned_in_engine_data() -> None:
    prompt = [1, 2, 3, 4]
    llm = _FakeLLM(
        [
            # Under DELTA, vLLM sends prompt logprobs once, at the end of prefill,
            # and an empty list on every later output.
            _output(
                prompt,
                _completion([5]),
                prompt_logprobs=[None, {2: _logprob(-0.25)}, None, None],
            ),
            _output(prompt, _completion([6], "stop"), prompt_logprobs=[]),
        ]
    )
    engine = native_engine.NativeVllmEngine(_host(llm), leader_worker_idx=0)

    chunks = _collect(engine, _openai_request(output_options={"prompt_logprobs": 0}))

    assert chunks[-1]["engine_data"]["prompt_logprobs"][1]["2"]["logprob"] == -0.25


def test_queue_limits_reject_with_the_engines_status() -> None:
    llm = _FakeLLM(admission_error=QueueOverflowError())
    engine = native_engine.NativeVllmEngine(_host(llm), leader_worker_idx=0)

    with pytest.raises(HttpError) as raised:
        _collect(engine, _openai_request())

    assert raised.value.code == HTTPStatus.SERVICE_UNAVAILABLE
    assert llm.calls == []


def _start_worker_config(monkeypatch, host) -> SimpleNamespace:
    """The WorkerConfig ``start_dynamo_endpoint`` registers ``host``'s engine with."""
    configs = []

    class _Worker:
        def __init__(self, engine, config):
            configs.append(config)

        async def run(self):
            return None

    monkeypatch.setattr(native_engine, "Worker", _Worker)
    monkeypatch.setattr(native_engine, "WorkerConfig", SimpleNamespace)
    env = {
        "DYN_DISCOVERY_BACKEND": "etcd",
        "DYN_REQUEST_PLANE": "tcp",
        "DYN_EVENT_PLANE": "nats",
    }
    for key, value in env.items():
        # Registered here so the variables the engine exports are restored.
        monkeypatch.setenv(key, value)

    async def start():
        await native_engine.start_dynamo_endpoint(
            host, namespace="ns", engine_env=env, leader_worker_idx=0
        )

    asyncio.run(start())
    (config,) = configs
    return config


def test_every_worker_arg_reaches_the_dynamo_worker_config(
    monkeypatch, tmp_path
) -> None:
    template = tmp_path / "chat.jinja"
    template.write_text("{{ messages }}")
    worker_args = {
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": "nemotron_nano",
        "exclude_tools_when_tool_choice_none": True,
        "enable_structural_tag": True,
        "structural_tag_scope": "always",
        "structural_tag_schema": "strict",
        "custom_jinja_template": str(template),
        "endpoint_types": ["chat"],
    }
    assert set(worker_args) == set(DynamoWorkerArgs.model_fields)
    host = _host(_FakeLLM(), dynamo_cfg={"worker_args": worker_args})

    config = _start_worker_config(monkeypatch, host)

    assert config.namespace == "ns"
    assert config.endpoint_types == "chat"
    assert config.structural_tag_mode == "on"
    with open(config.custom_jinja_template) as template_file:
        assert template_file.read() == "{{ messages }}"
    for key in worker_args.keys() - {
        "endpoint_types",
        "enable_structural_tag",
        "custom_jinja_template",
    }:
        assert getattr(config, key) == worker_args[key], key


def test_the_native_servers_chat_settings_reach_the_frontend(monkeypatch) -> None:
    host = _host(_FakeLLM())
    host.cfg["vllm_cfg"] = {
        "max_model_len": MAX_MODEL_LEN,
        "reasoning_parser_plugin": "nemo_rl/models/generation/vllm/reasoning_parsers/nano_v3_reasoning_parser.py",
        "http_server_serving_chat_kwargs": {
            "tool_parser": "qwen3_coder",
            "reasoning_parser": "nano_v3",
            # Literal Jinja, which the native server also accepts.
            "chat_template": "{{ messages }}\n",
        },
    }

    config = _start_worker_config(monkeypatch, host)

    assert (config.tool_call_parser, config.reasoning_parser) == (
        "qwen3_coder",
        "nemotron_v3",
    )
    with open(config.custom_jinja_template) as template_file:
        assert template_file.read() == "{{ messages }}\n"


def test_a_parser_dynamo_lacks_fails_registration(monkeypatch) -> None:
    host = _host(
        _FakeLLM(), dynamo_cfg={"worker_args": {"tool_call_parser": "no_such_parser"}}
    )

    with pytest.raises(ValueError, match="no tool call parser named 'no_such_parser'"):
        _start_worker_config(monkeypatch, host)


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
    engine = native_engine.NativeVllmEngine(_host(llm), leader_worker_idx=0)

    registration = asyncio.run(engine.start(worker_id=0)).llm

    # The KV router hashes events in blocks of the main-attention group; a
    # wrong size would quietly turn KV routing into load-only routing.
    assert registration.kv_cache_block_size == 528
    assert registration.total_kv_blocks == 64
    assert registration.context_length == MAX_MODEL_LEN


def test_a_prompt_that_fills_the_context_is_refused_as_natively() -> None:
    llm = _FakeLLM()
    engine = native_engine.NativeVllmEngine(_host(llm), leader_worker_idx=0)

    with pytest.raises(InvalidArgument, match="maximum context length"):
        _collect(engine, _openai_request(token_ids=list(range(MAX_MODEL_LEN))))

    assert llm.calls == []


def _image_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color="red").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_media_is_loaded_and_processed_as_vllms_chat_parser_does() -> None:
    final = _output([1, *[IMAGE_TOKEN] * IMAGE_TOKENS, 3], _completion([5], "stop"))
    llm = _FakeLLM([final])
    engine = native_engine.NativeVllmEngine(_host(llm), leader_worker_idx=0)

    _collect(
        engine,
        _openai_request(
            token_ids=[1, IMAGE_PLACEHOLDER, 3],
            multi_modal_data={"image_url": [{"Url": _image_url()}]},
            mm_processor_kwargs={"max_num_tiles": 2},
        ),
    )

    ((prompt, skip_mm_cache),) = llm.renderer.prompts
    # The frontend's unexpanded tokens, the decoded image and the processor kwargs.
    assert prompt["prompt_token_ids"] == [1, IMAGE_PLACEHOLDER, 3]
    (image,) = prompt["multi_modal_data"]["image"]
    assert image.size == (8, 8) and image.mode == "RGB"
    assert prompt["multi_modal_uuids"] == {"image": [None]}
    assert prompt["mm_processor_kwargs"] == {"max_num_tiles": 2}
    assert skip_mm_cache is False
    # vLLM generates from the processed prompt, and the output cap counts its tokens.
    ((engine_input, params, _),) = llm.calls
    assert engine_input["prompt_token_ids"] == [1, *[IMAGE_TOKEN] * IMAGE_TOKENS, 3]
    assert params.max_tokens == MAX_MODEL_LEN - 2 - IMAGE_TOKENS


@pytest.mark.parametrize(
    ("fields", "error"),
    [
        ({"prompt_embeds": "AAAA"}, "Embedding prompts"),
        ({"multi_modal_data": {"image_url": [{"UuidOnly": "u"}]}}, "carry their URL"),
        (
            {
                "multi_modal_data": {
                    "image_url": [{"Url": "data:image/png;base64,eA=="}]
                }
            },
            "Cannot load image_url",
        ),
        # vLLM's own request errors (not ValueErrors) keep their message, as the
        # native server's 4xx does, instead of becoming a bare 500.
        ({"stop_conditions": {"min_tokens": 10, "max_tokens": 5}}, "min_tokens"),
    ],
)
def test_prompts_the_engine_cannot_serve_are_refused(fields, error) -> None:
    llm = _FakeLLM()
    engine = native_engine.NativeVllmEngine(_host(llm), leader_worker_idx=0)

    with pytest.raises(InvalidArgument, match=error):
        _collect(engine, _openai_request(**fields))

    assert llm.calls == []


@pytest.mark.parametrize(
    ("router_mode", "prefix_caching", "publishes"),
    [("kv", True, True), ("kv", False, False), ("round-robin", True, False)],
)
def test_engines_publish_kv_events_only_where_the_kv_router_can_use_them(
    router_mode, prefix_caching, publishes
) -> None:
    cfg = native_engine.DynamoCfg.model_validate(
        {"frontend_args": {"router_mode": router_mode}}
    )

    config = native_engine.dynamo_kv_events_config(cfg, prefix_caching=prefix_caching)

    # vLLM publishes block events only for blocks its prefix cache stores.
    assert (config is not None) is publishes
    if publishes:
        assert config["enable_kv_cache_events"] and config["publisher"] == "zmq"
        # A per-engine abstract socket, which vLLM's publisher binds and Dynamo's
        # source connects to: no port for two engines on a node to collide on.
        assert config["endpoint"].startswith("ipc://@")
