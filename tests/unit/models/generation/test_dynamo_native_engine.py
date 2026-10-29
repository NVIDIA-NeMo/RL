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
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.vllm
pytest.importorskip("dynamo._core")
vllm = pytest.importorskip("vllm")

from dynamo.llm.exceptions import InvalidArgument  # noqa: E402

from nemo_rl.models.generation.dynamo import native_engine  # noqa: E402
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


class _FakeLLM:
    """Yields scripted RequestOutputs and records what it was asked to generate."""

    def __init__(self, outputs=()):
        self.outputs = list(outputs)
        self.calls = []
        self.tokenizer = None
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
            kv_events_config=None,
        )

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
    engine = native_engine.NativeVllmEngine(_host(llm), leader_worker_idx=0)

    (chunk,) = _collect(engine, _rollout_request(list(range(MAX_MODEL_LEN))))

    assert llm.calls == []
    rollout = chunk["engine_data"]["rollout"]
    assert rollout["token_ids"] == [] and rollout["truncated"] is False


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


def test_engines_register_for_rollouts_on_completions_only(monkeypatch) -> None:
    config = _start_worker_config(monkeypatch, _host(_FakeLLM()))

    assert config.namespace == "ns"
    # NeMo RL's rollouts go to /v1/completions; no chat surface is registered.
    assert config.endpoint_types == "completions"


def test_requests_other_than_rollouts_are_refused() -> None:
    llm = _FakeLLM()
    engine = native_engine.NativeVllmEngine(_host(llm), leader_worker_idx=0)

    with pytest.raises(InvalidArgument, match="only NeMo RL rollouts"):
        _collect(engine, {"token_ids": [1, 2, 3, 4]})

    assert llm.calls == []


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
