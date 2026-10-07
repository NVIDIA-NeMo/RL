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

"""Megatron retries must wait for the persistent engine, not just the Ray waiter."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.failures import RolloutDataFailure
from nemo_rl.models.generation.megatron.megatron_generation import MegatronGeneration
from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin

pytestmark = pytest.mark.mcore


@pytest.mark.parametrize("completed", [True, False])
def test_engine_failed_reply_is_not_a_successful_empty_rollout(monkeypatch, completed):
    from megatron.core.inference.inference_request import Status

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    async def run():
        result = SimpleNamespace(
            status=Status.COMPLETED if completed else Status.FAILED
        )
        reply = asyncio.get_running_loop().create_future()
        reply.set_result(result)
        worker = SimpleNamespace(
            inference_client=SimpleNamespace(add_request=lambda *args, **kwargs: reply)
        )
        call = MegatronGenerationMixin._generate_with_persistent_engine(
            worker, [[1]], [None], [None]
        )
        if completed:
            assert await call == [result]
        else:
            with pytest.raises(RuntimeError, match="unsuccessful request"):
                await call

    asyncio.run(run())


def _backend(stream):
    backend = object.__new__(MegatronGeneration)
    backend._owns_policy = False
    worker = MagicMock()
    worker.generate_async.options.return_value.remote.return_value = stream
    backend._policy = SimpleNamespace(worker_group=SimpleNamespace(workers=[worker]))
    return backend


@pytest.mark.parametrize("error", [TimeoutError("transient"), ValueError("bad output")])
def test_driver_drains_failure_and_preserves_original_error(monkeypatch, error):
    events = []

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise error

        async def completed(self):
            events.append("drained")
            raise ray.exceptions.TaskCancelledError()

    monkeypatch.setattr(ray, "cancel", lambda _: events.append("cancel"))
    backend = _backend(Stream())
    assert backend.supports_native_generation_retries()

    async def run():
        with pytest.raises(type(error)) as exc:
            async for _ in backend.generate_async(BatchedDataDict({})):
                pytest.fail("failed generation must not yield output")
        assert exc.value is error

    asyncio.run(run())
    assert events == ["cancel", "drained"]


def test_uncertain_cleanup_is_fatal(monkeypatch):
    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise TimeoutError("generation timed out")

        async def completed(self):
            raise TimeoutError("cleanup timed out")

    monkeypatch.setattr(ray, "cancel", lambda _: None)

    async def run():
        with pytest.raises(RolloutDataFailure, match="confirm cleanup"):
            async for _ in _backend(Stream()).generate_async(BatchedDataDict({})):
                pytest.fail("unexpected output")

    asyncio.run(run())


@pytest.mark.parametrize("cancellations", [0, 1, 2])
def test_worker_drains_persistent_engine_before_finishing(cancellations):
    async def run():
        started = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()

        async def engine(*_args):
            try:
                started.set()
                await release.wait()
                return ["reply"]
            finally:
                finished.set()

        worker = SimpleNamespace(
            _inference_loop=asyncio.get_running_loop(),
            _prepare_data_for_generation=lambda *_: ([[]], [None], [None]),
            _generate_with_persistent_engine=engine,
            _parse_result_to_batched_data_dict=lambda *args: args[-1],
        )
        stream = MegatronGenerationMixin.generate_async(
            worker, BatchedDataDict({"input_ids": torch.tensor([[1]])})
        )
        result = asyncio.create_task(anext(stream))
        await started.wait()
        for _ in range(cancellations):
            result.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not result.done()
            assert not finished.is_set()
        release.set()
        if cancellations:
            with pytest.raises(asyncio.CancelledError):
                await result
        else:
            assert await result == (0, ["reply"])
        await stream.aclose()
        assert finished.is_set()
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(run())
