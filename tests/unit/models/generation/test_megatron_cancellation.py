# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Cancellation must reach the inference engine, not only local futures."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.sampling_params import SamplingParams
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin


def _client():
    messages = []
    client = object.__new__(InferenceClient)
    client.next_request_id = 0
    client.completion_futures = {}
    client.request_submission_times = {}
    client.aborted_request_ids = set()
    client.streams = {}
    client.socket = SimpleNamespace(
        send_multipart=lambda frames: None, send=messages.append
    )
    client._pack_submit_frames = lambda request_id, *args, **kwargs: [request_id]
    return client, messages


@pytest.mark.parametrize("failure", ["cancel", "generation", "submission", "abort"])
def test_batch_failure_aborts_requests(failure):
    async def run():
        client, messages = _client()
        worker = SimpleNamespace(inference_client=client)
        original_submit = client.add_request_with_id
        error = RuntimeError("original failure")
        if failure == "submission":

            def submit(*args, **kwargs):
                if client.next_request_id == 1:
                    raise error
                return original_submit(*args, **kwargs)

            client.add_request_with_id = submit
        if failure == "abort":
            original_abort = client.abort_request

            def abort(request_id):
                original_abort(request_id)
                if request_id == 0:
                    raise RuntimeError("abort transport error")

            client.abort_request = abort
        with patch("torch.distributed.get_rank", return_value=0):
            task = asyncio.create_task(
                MegatronGenerationMixin._generate_with_persistent_engine(
                    worker,
                    [[1], [2]],
                    [None, None],
                    [SamplingParams(), SamplingParams()],
                )
            )
            await asyncio.sleep(0)
            futures = list(client.completion_futures.values())
            if failure in ("cancel", "abort"):
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                if failure == "generation":
                    futures[0].set_exception(error)
                with pytest.raises(RuntimeError) as caught:
                    await task
                assert caught.value is error
        assert len(messages) == (1 if failure == "submission" else 2)
        assert not client.completion_futures
        assert not client.request_submission_times
        assert all(future.done() for future in futures)

    asyncio.run(run())


def test_success_preserves_order_without_aborting():
    async def run():
        client, messages = _client()
        with patch("torch.distributed.get_rank", return_value=0):
            task = asyncio.create_task(
                MegatronGenerationMixin._generate_with_persistent_engine(
                    SimpleNamespace(inference_client=client),
                    [[1], [2]],
                    [None, None],
                    [SamplingParams(), SamplingParams()],
                )
            )
            await asyncio.sleep(0)
            client.completion_futures[1].set_result("second")
            client.completion_futures[0].set_result("first")
            assert await task == ["first", "second"]
        assert not messages

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["close", "generation"])
def test_stream_cleanup_cancels_siblings(failure):
    async def run():
        started = asyncio.Event()
        finished = asyncio.Event()

        async def generate(prompts, *args):
            if prompts[0][0] == 0:
                await started.wait()
                if failure == "generation":
                    raise RuntimeError("generation failure")
                return ["first"]
            started.set()
            try:
                await asyncio.Future()
            finally:
                finished.set()

        worker = SimpleNamespace(
            _inference_loop=asyncio.get_running_loop(),
            _prepare_data_for_generation=lambda datum, greedy: (
                datum["input_ids"].tolist(),
                [None],
                [SamplingParams()],
            ),
            _generate_with_persistent_engine=generate,
            _parse_result_to_batched_data_dict=lambda datum, result: result,
        )
        data = BatchedDataDict({"input_ids": torch.tensor([[0], [1]])})
        stream = MegatronGenerationMixin.generate_async(worker, data)
        if failure == "generation":
            with pytest.raises(RuntimeError, match="generation failure"):
                await anext(stream)
        else:
            assert await anext(stream) == (0, ["first"])
            await stream.aclose()
        await asyncio.wait_for(finished.wait(), timeout=1)

    asyncio.run(run())
