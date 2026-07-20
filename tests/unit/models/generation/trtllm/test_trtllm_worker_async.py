# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.trtllm.trtllm_worker_async import (
    TrtllmAsyncGenerationWorkerImpl,
)

pytestmark = pytest.mark.trtllm


def _config():
    return {
        "backend": "trtllm",
        "model_name": "test/model",
        "max_new_tokens": 4,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 20,
        "stop_token_ids": [9],
        "stop_strings": None,
        "_pad_token_id": 0,
        "colocated": {
            "enabled": True,
            "resources": {"gpus_per_node": None, "num_nodes": None},
        },
        "trtllm_cfg": {
            "tensor_parallel_size": 1,
            "max_model_len": 128,
            "precision": "bfloat16",
            "max_batch_size": 8,
            "max_num_tokens": 256,
            "async_engine": True,
        },
    }


def _worker():
    worker = TrtllmAsyncGenerationWorkerImpl.__new__(TrtllmAsyncGenerationWorkerImpl)
    worker.cfg = _config()
    worker.model_name = worker.cfg["model_name"]
    worker.is_model_owner = True
    worker.llm = AsyncMock()
    worker.TrtSamplingParams = MagicMock()
    return worker


def test_configure_worker_reserves_gpus_for_internal_trtllm_actors(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")

    resources, env_vars, init_kwargs, actor_options = (
        TrtllmAsyncGenerationWorkerImpl.configure_worker(
            num_gpus=2,
            bundle_indices=(0, [2, 3]),
        )
    )

    assert resources == {"num_gpus": 0, "num_cpus": 0}
    assert env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"
    assert env_vars["NCCL_CUMEM_ENABLE"] == "1"
    assert env_vars["PATH"].endswith(":/usr/bin")
    assert init_kwargs == {"bundle_indices": [2, 3]}
    assert actor_options == {}


@pytest.mark.asyncio
async def test_generate_async_converts_padded_batch_and_logprobs():
    worker = _worker()
    sampling_params = object()
    worker._build_sampling_params = MagicMock(return_value=sampling_params)

    responses = {
        (11, 12): SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=[21, 22], logprobs=[-0.1, -0.2])]
        ),
        (13,): SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[23],
                    logprobs=[{23: SimpleNamespace(logprob=-0.3)}],
                )
            ]
        ),
    }

    async def generate_async(*, inputs, sampling_params):
        assert sampling_params is not None
        return responses[tuple(inputs["prompt_token_ids"])]

    worker.llm.generate_async.side_effect = generate_async
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[11, 12], [13, 0]]),
            "input_lengths": torch.tensor([2, 1]),
        }
    )

    output = await worker.generate_async(data, greedy=True)

    worker._build_sampling_params.assert_called_once_with(greedy=True)
    assert worker.llm.generate_async.await_count == 2
    assert torch.equal(
        output["output_ids"], torch.tensor([[11, 12, 21, 22], [13, 23, 0, 0]])
    )
    assert torch.equal(output["generation_lengths"], torch.tensor([2, 1]))
    assert torch.equal(output["unpadded_sequence_lengths"], torch.tensor([4, 2]))
    assert torch.allclose(
        output["logprobs"],
        torch.tensor([[0.0, 0.0, -0.1, -0.2], [0.0, -0.3, 0.0, 0.0]]),
    )


@pytest.mark.asyncio
async def test_generate_async_rejects_non_right_padded_input():
    worker = _worker()
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[11, 0, 12]]),
            "input_lengths": torch.tensor([1]),
        }
    )

    with pytest.raises(ValueError, match="Non-padding values"):
        await worker.generate_async(data)
    worker.llm.generate_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_async_empty_batch_does_not_call_engine():
    worker = _worker()
    data = BatchedDataDict(
        {
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "input_lengths": torch.empty(0, dtype=torch.long),
        }
    )

    output = await worker.generate_async(data)

    assert output["output_ids"].shape == (0, 0)
    assert output["logprobs"].shape == (0, 0)
    worker.llm.generate_async.assert_not_awaited()


def test_build_sampling_params_supports_async_greedy_and_sampling_modes():
    worker = _worker()
    worker._resolve_end_id = MagicMock(return_value=2)

    worker._build_sampling_params(greedy=True)
    worker.TrtSamplingParams.assert_called_once_with(
        temperature=0.0,
        top_p=0.9,
        top_k=1,
        max_tokens=4,
        end_id=2,
        stop_token_ids=[9],
        include_stop_str_in_output=True,
        logprobs=True,
    )

    worker.TrtSamplingParams.reset_mock()
    worker._build_sampling_params(greedy=False)
    worker.TrtSamplingParams.assert_called_once_with(
        temperature=0.8,
        top_p=0.9,
        top_k=20,
        max_tokens=4,
        end_id=2,
        stop_token_ids=[9],
        include_stop_str_in_output=True,
        logprobs=True,
    )


@pytest.mark.asyncio
async def test_async_lifecycle_resets_cache_before_sleep_and_resumes_selected_tags():
    worker = _worker()
    worker._all_sleep_tags = MagicMock(return_value=["weights", "kv_cache"])
    worker._resolve_wake_tags = MagicMock(return_value=["weights"])

    await worker.sleep_async()
    await worker.wake_up_async(tags=["weights"])

    assert worker.llm.method_calls == [
        call.collective_rpc("reset_prefix_cache"),
        call.release(["weights", "kv_cache"]),
        call.resume(["weights"]),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [[True], [], [False]])
async def test_async_collective_refit_propagates_worker_result(result):
    worker = _worker()
    worker.llm.collective_rpc.return_value = result

    succeeded = await worker.update_weights_from_collective_async(
        drain=False,
        recompute_kv=True,
    )

    assert succeeded is (result != [False])
    worker.llm.collective_rpc.assert_awaited_once_with(
        "update_weights_from_collective",
        kwargs={"drain": False, "recompute_kv": True},
    )


@pytest.mark.asyncio
async def test_async_ipc_refit_returns_false_on_worker_exception():
    worker = _worker()
    worker.llm.collective_rpc.side_effect = RuntimeError("refit failed")

    assert await worker.update_weights_via_ipc_zmq_async() is False
