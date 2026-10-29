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
from pathlib import Path
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
from nemo_rl.models.generation.openai_server_utils import (  # noqa: E402
    derive_prefix_eos_count,
    splice_prefix_tokens,
)
from nemo_rl.models.generation.vllm.request_pipeline import (  # noqa: E402
    rollout_output,
    rollout_result,
    rollout_result_from_wire,
    rollout_result_to_wire,
)
from nemo_rl.models.generation.vllm.utils import chat_routed_experts  # noqa: E402
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

    async def process_for_engine_async(self, prompt, arrival_time, *, skip_mm_cache):
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


def test_router_replay_routes_reach_rollouts_and_gym_calls_as_natively() -> None:
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

    # A Gym call without capture carries the routes in engine_data, as the
    # native chat endpoint attaches them to the response.
    gym_final = _output(
        [1, 3, 4],
        _completion([10, 11], "stop", routed_experts=routes[3:]),
        prompt_routed_experts=routes[:3],
    )
    gym_host = _host(
        _FakeLLM([gym_final]), vllm_kwargs={"enable_return_routed_experts": True}
    )
    gym_engine = native_engine.NativeVllmEngine(gym_host, leader_worker_idx=0)

    (gym_chunk,) = _collect(gym_engine, _gym_request())

    assert gym_chunk["engine_data"]["routed_experts"] == chat_routed_experts(
        gym_final,
        gym_final.outputs[0],
        prompt_token_count=3,
        device=torch.device("cpu"),
        logger=native_engine.LOGGER,
        routed_experts_dtype=torch.int16,
    )


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


@pytest.mark.parametrize(
    ("reasoning_parser", "dynamo_reasoning_parser"),
    [
        # The plugin's own parser, which Dynamo ships as nemotron_v3.
        ("nano_v3", "nemotron_v3"),
        # A vLLM built-in that Dynamo also ships, beside an inherited plugin.
        ("deepseek_r1", "deepseek_r1"),
    ],
)
def test_the_native_servers_chat_settings_reach_the_frontend(
    monkeypatch, reasoning_parser, dynamo_reasoning_parser
) -> None:
    host = _host(_FakeLLM())
    host.cfg["vllm_cfg"] = {
        "max_model_len": MAX_MODEL_LEN,
        "reasoning_parser_plugin": str(
            Path(native_engine.__file__).parents[1]
            / "vllm/reasoning_parsers/nano_v3_reasoning_parser.py"
        ),
        "http_server_serving_chat_kwargs": {
            "tool_parser": "qwen3_coder",
            "reasoning_parser": reasoning_parser,
            # Literal Jinja, which the native server also accepts.
            "chat_template": "{{ messages }}\n",
        },
    }

    config = _start_worker_config(monkeypatch, host)

    assert (config.tool_call_parser, config.reasoning_parser) == (
        "qwen3_coder",
        dynamo_reasoning_parser,
    )
    with open(config.custom_jinja_template) as template_file:
        assert template_file.read() == "{{ messages }}\n"


_PLUGIN_PARSER = """
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser


@ReasoningParserManager.register_module("nemo_rl_test_plugin_parser")
class PluginParser(DeepSeekR1ReasoningParser):
    pass
"""


@pytest.mark.parametrize("cause", ["unknown_name", "plugin_parser"])
def test_a_parser_dynamo_lacks_fails_registration(monkeypatch, tmp_path, cause) -> None:
    if cause == "unknown_name":
        host = _host(
            _FakeLLM(),
            dynamo_cfg={"worker_args": {"tool_call_parser": "no_such_parser"}},
        )
        error = "no tool call parser named 'no_such_parser'"
    else:
        plugin = tmp_path / "nemo_rl_test_plugin.py"
        plugin.write_text(_PLUGIN_PARSER)
        host = _host(_FakeLLM())
        host.cfg["vllm_cfg"] = {
            "max_model_len": MAX_MODEL_LEN,
            "reasoning_parser_plugin": str(plugin),
            "http_server_serving_chat_kwargs": {
                "reasoning_parser": "nemo_rl_test_plugin_parser"
            },
        }
        error = "comes from vllm_cfg.reasoning_parser_plugin"

    with pytest.raises(ValueError, match=error):
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


# --- NeMo-Gym's Dynamo dialect -------------------------------------------------


def _f32(value: float) -> float:
    return float(np.float32(value))


def _gym_request(
    *, temperature=0.7, top_p=0.9, prefix=None, sampling=None, **fields
) -> dict:
    annotations = [native_engine.GYM_ROLLOUT_ANNOTATION]
    if prefix is not None:
        annotations.insert(
            0, native_engine.GYM_PREFIX_ANNOTATION_PREFIX + json.dumps(prefix)
        )
    sampling_options = {"temperature": temperature, "top_p": top_p, **(sampling or {})}
    return {
        "token_ids": [1, 3, 4],
        # The frontend carries sampling floats as f32.
        "sampling_options": {
            key: _f32(value) if isinstance(value, float) else value
            for key, value in sampling_options.items()
        },
        # Gym's token information: logprobs=true, top_logprobs=0.
        "output_options": {"logprobs": 0},
        "annotations": annotations,
        **fields,
    }


def test_gym_calls_sample_with_nemo_rls_exact_profile() -> None:
    final = _output([1, 3, 4], _completion([10, 11], "stop"))
    llm = _FakeLLM([final])
    engine = native_engine.NativeVllmEngine(_host(llm), leader_worker_idx=0)

    (chunk,) = _collect(
        engine, _gym_request(stop_conditions={"max_tokens": 100, "stop": ["</s>"]})
    )

    ((prompt, params, _),) = llm.calls
    assert prompt["prompt_token_ids"] == [1, 3, 4]
    # The configured values, not the f32 values the frontend delivered.
    assert (params.temperature, params.top_p) == (0.7, 0.9)
    assert params.top_k == vllm.SamplingParams(top_k=-1).top_k
    assert params.output_kind == vllm.sampling_params.RequestOutputKind.FINAL_ONLY
    # Output fits the context, and vLLM stops on the stop strings itself.
    assert params.max_tokens == MAX_MODEL_LEN - 3
    assert (params.stop, params.detokenize) == (["</s>"], True)
    assert chunk["engine_data"] == {
        "prompt_token_ids": [1, 3, 4],
        "completion_token_ids": [10, 11],
        "completion_logprobs": [-0.5, -0.5],
    }
    assert (chunk["token_ids"], chunk["finish_reason"]) == ([10, 11], "stop")
    assert chunk["log_probs"] == [-0.5, -0.5]


def test_gym_validation_calls_sample_with_the_validation_profile() -> None:
    llm = _FakeLLM([_output([1, 3, 4], _completion([10], "stop"))])
    engine = native_engine.NativeVllmEngine(_host(llm), leader_worker_idx=0)

    _collect(engine, _gym_request(temperature=0.0, top_p=1.0))

    ((_, params, _),) = llm.calls
    assert (params.temperature, params.top_p) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("request_fields", "error"),
    [
        ({"temperature": 0.5}, "matches neither"),
        ({"top_p": None}, "top_p must be set explicitly"),
        ({"sampling": {"top_k": 5}}, "Top k sampling parameter"),
        ({"sampling": {"n": 2}}, "take n=1"),
    ],
)
def test_gym_calls_that_would_sample_off_policy_are_refused(
    request_fields, error
) -> None:
    llm = _FakeLLM()
    engine = native_engine.NativeVllmEngine(_host(llm), leader_worker_idx=0)

    with pytest.raises(InvalidArgument, match=error):
        _collect(engine, _gym_request(**request_fields))

    assert llm.calls == []


def test_gym_echoed_prefix_replaces_the_rendered_history() -> None:
    # The model generated [5, 6, EOS]; the frontend re-rendered that turn as [7, EOS].
    prefix, rendered = [1, 5, 6, EOS], [1, 7, EOS, 3, 4]
    spliced = splice_prefix_tokens(
        tokenizer=None,
        model_prefix_token_ids=prefix,
        template_token_ids=rendered,
        template_prefix_eos_count=derive_prefix_eos_count(prefix, eos_token_id=EOS),
        eos_token_id=EOS,
    ).token_ids
    assert spliced != rendered
    llm = _FakeLLM([_output(spliced, _completion([10], "stop"))])
    llm.tokenizer = _Tokenizer()
    engine = native_engine.NativeVllmEngine(_host(llm), leader_worker_idx=0)

    (chunk,) = _collect(engine, _gym_request(prefix=prefix, token_ids=rendered))

    ((prompt, params, _),) = llm.calls
    assert prompt["prompt_token_ids"] == spliced
    assert params.max_tokens == MAX_MODEL_LEN - len(spliced)
    assert chunk["engine_data"]["prompt_token_ids"] == spliced


def test_gym_calls_whose_splice_fills_the_context_are_refused() -> None:
    prefix = [1, *range(20, 20 + MAX_MODEL_LEN), EOS]
    llm = _FakeLLM()
    llm.tokenizer = _Tokenizer()
    engine = native_engine.NativeVllmEngine(_host(llm), leader_worker_idx=0)

    with pytest.raises(InvalidArgument, match="maximum context length"):
        _collect(engine, _gym_request(prefix=prefix, token_ids=[1, 7, EOS, 3]))

    assert llm.calls == []


def test_gym_echoed_prefix_keeps_media_placeholders_on_their_tokens() -> None:
    # Turn 1 ran [1, <image>, 3] and generated [5, 6, EOS]; the frontend
    # re-rendered that turn as [7, EOS] and appended a turn with a second image.
    turn1 = [1, *[IMAGE_TOKEN] * IMAGE_TOKENS, 3]
    prefix = [*turn1, 5, 6, EOS]
    rendered = [1, IMAGE_PLACEHOLDER, 3, 7, EOS, IMAGE_PLACEHOLDER, 4]
    spliced = [*prefix, *[IMAGE_TOKEN] * IMAGE_TOKENS, 4]
    llm = _FakeLLM([_output(spliced, _completion([10], "stop"))])
    llm.tokenizer = _Tokenizer()
    engine = native_engine.NativeVllmEngine(_host(llm), leader_worker_idx=0)

    (chunk,) = _collect(
        engine,
        _gym_request(
            prefix=prefix,
            token_ids=rendered,
            multi_modal_data={"image_url": [{"Url": _image_url()}] * 2},
        ),
    )

    ((engine_input, params, _),) = llm.calls
    assert engine_input["prompt_token_ids"] == spliced
    # The first image sits in the model's own prefix; the second moved with the
    # longer generated turn, as the native endpoint remaps them.
    assert [
        (span.offset, span.length) for span in engine_input["mm_placeholders"]["image"]
    ] == [(1, IMAGE_TOKENS), (len(prefix), IMAGE_TOKENS)]
    assert params.max_tokens == MAX_MODEL_LEN - len(spliced)
    assert chunk["engine_data"]["prompt_token_ids"] == spliced


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
