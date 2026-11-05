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

import asyncio

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.vllm.output_limits import per_request_output_limits


class _StopBeforeGeneration(Exception):
    pass


def _worker(worker_cls, vllm_cfg):
    worker = worker_cls.__new__(worker_cls)
    worker.cfg = {"_pad_token_id": 0, "max_new_tokens": 10, "vllm_cfg": vllm_cfg}
    worker.requested_max_new_tokens = []

    def build_sampling_params(*, max_new_tokens=None, **_kwargs):
        worker.requested_max_new_tokens.append(max_new_tokens)
        if len(worker.requested_max_new_tokens) == worker.expected_requests:
            raise _StopBeforeGeneration

    worker._build_sampling_params = build_sampling_params
    return worker


def test_mixed_caps_respect_global_and_remaining_context_without_changing_null_rows():
    assert per_request_output_limits(
        [None, 3, 200, 200, None],
        [98, 5, 5, 98, 99],
        configured_max_new_tokens=10,
        max_model_len=100,
    ) == [10, 3, 10, 2, 10]


@pytest.mark.parametrize(
    "caps", [[1], [0, None], [True, None], [1.5, None], [-2, None]]
)
def test_invalid_request_caps_are_rejected(caps):
    with pytest.raises(ValueError, match="max_output_tokens"):
        per_request_output_limits(
            caps, [5, 5], configured_max_new_tokens=10, max_model_len=100
        )


def test_explicit_cap_on_exhausted_context_is_rejected():
    with pytest.raises(ValueError, match="exhausts"):
        per_request_output_limits(
            [5], [100], configured_max_new_tokens=10, max_model_len=100
        )


def test_sync_worker_caps_compose_with_context_cap():
    from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl

    worker = _worker(
        VllmGenerationWorkerImpl,
        {"max_model_len": 100, "cap_max_tokens_to_context": True},
    )
    worker.expected_requests = 3
    worker.llm = object()
    input_ids = torch.ones((3, 95), dtype=torch.long)
    input_ids[1, 5:] = 0
    batch = BatchedDataDict(
        input_ids=input_ids,
        input_lengths=torch.tensor([95, 5, 95]),
        max_output_tokens=[None, 3, 200],
    )
    with pytest.raises(_StopBeforeGeneration):
        worker.generate(batch)
    assert worker.requested_max_new_tokens == [5, 3, 5]


def test_async_worker_applies_row_cap():
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    worker = _worker(
        VllmAsyncGenerationWorkerImpl, {"max_model_len": 100, "async_engine": True}
    )
    worker.expected_requests = 1
    batch = BatchedDataDict(
        input_ids=torch.ones((1, 5), dtype=torch.long),
        input_lengths=torch.tensor([5]),
        max_output_tokens=[3],
    )

    async def consume():
        async for _ in worker.generate_async(batch):
            pass

    with pytest.raises(_StopBeforeGeneration):
        asyncio.run(consume())
    assert worker.requested_max_new_tokens == [3]
