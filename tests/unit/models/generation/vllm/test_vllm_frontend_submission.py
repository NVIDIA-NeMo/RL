# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import time
from collections import deque
from types import SimpleNamespace

import pytest

import nemo_rl.models.generation.vllm.vllm_generation as generation_module
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorker,
    _await_model_worker_collective_rpc,
    _iterate_request_output_collector,
    _submit_vllm_request,
)


def test_submit_timestamp_is_after_add_request_returns() -> None:
    events: list[str] = []
    add_returned_at: list[float] = []
    collector = object()

    class FakeLlm:
        async def add_request(self, **kwargs):
            events.append("add_request")
            await asyncio.sleep(0)
            add_returned_at.append(time.monotonic())
            assert kwargs == {
                "request_id": "request-0",
                "prompt": "prompt",
                "params": "sampling-params",
                "priority": 7,
            }
            return collector

    async def run():
        return await _submit_vllm_request(
            FakeLlm(),
            prompt="prompt",
            sampling_params="sampling-params",
            request_id="request-0",
            priority=7,
        )

    (
        actual_collector,
        submitted_at_unix_s,
        submitted_at_monotonic_s,
        submitted_hostname,
    ) = asyncio.run(run())

    assert events == ["add_request"]
    assert actual_collector is collector
    assert submitted_at_unix_s > 0
    assert submitted_at_monotonic_s >= add_returned_at[0]
    assert submitted_hostname


def test_terminal_stream_sentinel_stops_without_being_yielded() -> None:
    class Output:
        def __init__(self, *, finished: bool) -> None:
            self.finished = finished

    first = Output(finished=False)
    stream_finished = Output(finished=True)

    class FakeCollector:
        def __init__(self) -> None:
            self.items = deque([first, stream_finished])
            self.get_nowait_calls = 0

        def get_nowait(self):
            self.get_nowait_calls += 1
            return self.items.popleft()

        async def get(self):
            raise AssertionError("all fake outputs should be immediately ready")

    collector = FakeCollector()

    async def collect():
        return [
            item
            async for item in _iterate_request_output_collector(
                collector,
                stream_finished,
            )
        ]

    outputs = asyncio.run(asyncio.wait_for(collect(), timeout=1))
    assert outputs == [first]
    assert collector.get_nowait_calls == 2


def test_terminal_request_output_is_yielded_once() -> None:
    class Output:
        finished = True

    final_output = Output()

    class FakeCollector:
        def get_nowait(self):
            return final_output

        async def get(self):
            raise AssertionError("the final output should be immediately ready")

    async def collect():
        return [
            item
            async for item in _iterate_request_output_collector(
                FakeCollector(),
                object(),
            )
        ]

    assert asyncio.run(collect()) == [final_output]


def test_async_gpu_profiler_collective_rpc_is_awaited() -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    class FakeLlm:
        async def collective_rpc(self, method: str, *, args: tuple[object, ...]):
            await asyncio.sleep(0)
            calls.append((method, args))

    asyncio.run(
        _await_model_worker_collective_rpc(
            FakeLlm(), "start_gpu_profiling"
        )
    )
    asyncio.run(
        _await_model_worker_collective_rpc(
            FakeLlm(), "stop_gpu_profiling"
        )
    )

    assert calls == [
        ("start_gpu_profiling", ()),
        ("stop_gpu_profiling", ()),
    ]


def test_gpu_profiler_selects_async_worker_entrypoints() -> None:
    class FakeWorkerGroup:
        def __init__(self) -> None:
            self.methods: list[str] = []

        def run_all_workers_single_data(self, method: str):
            self.methods.append(method)
            return []

    for async_engine, expected in (
        (False, ["start_gpu_profiling", "stop_gpu_profiling"]),
        (
            True,
            [
                "start_gpu_profiling_async",
                "stop_gpu_profiling_async",
            ],
        ),
    ):
        generation = object.__new__(VllmGeneration)
        generation.cfg = {"vllm_cfg": {"async_engine": async_engine}}
        generation.worker_group = FakeWorkerGroup()

        generation.start_gpu_profiling()
        generation.stop_gpu_profiling()

        assert generation.worker_group.methods == expected


def test_async_exact_model_step_rpc_returns_every_tp_proof() -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    class FakeLlm:
        async def collective_rpc(self, method: str, *, args: tuple[object, ...]):
            calls.append((method, args))
            return [{"rank": 0}, {"rank": 1}]

    worker = SimpleNamespace(llm=FakeLlm())
    worker_class = VllmAsyncGenerationWorker.__ray_metadata__.modified_class
    armed = asyncio.run(
        worker_class.arm_model_step_gpu_profile_async(
            worker,
            512,
            1024,
        )
    )
    completed = asyncio.run(
        worker_class.get_model_step_gpu_profile_async(worker)
    )

    assert armed == [{"rank": 0}, {"rank": 1}]
    assert completed == armed
    assert calls == [
        ("arm_model_step_gpu_profile", (512, 1024)),
        ("get_model_step_gpu_profile", ()),
    ]


def test_generation_exact_model_step_rpc_covers_every_dp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, dict[str, int]]] = []

    class FakeWorkerGroup:
        dp_size = 2

        @staticmethod
        def get_dp_leader_worker_idx(dp_idx: int) -> int:
            return 10 + dp_idx

        def run_single_worker_single_data(
            self,
            method: str,
            *,
            worker_idx: int,
            **kwargs: int,
        ):
            calls.append((method, worker_idx, kwargs))
            return [{"worker_idx": worker_idx, "method": method}]

    monkeypatch.setattr(generation_module.ray, "get", lambda values: values)
    generation = object.__new__(VllmGeneration)
    generation.cfg = {"vllm_cfg": {"async_engine": True}}
    generation.worker_group = FakeWorkerGroup()

    armed = generation.arm_model_step_gpu_profile(2, 6)
    completed = generation.get_model_step_gpu_profile()

    assert set(armed) == {0, 1}
    assert set(completed) == {0, 1}
    assert calls == [
        (
            "arm_model_step_gpu_profile_async",
            10,
            {"start_step": 2, "stop_step": 6},
        ),
        (
            "arm_model_step_gpu_profile_async",
            11,
            {"start_step": 2, "stop_step": 6},
        ),
        ("get_model_step_gpu_profile_async", 10, {}),
        ("get_model_step_gpu_profile_async", 11, {}),
    ]
