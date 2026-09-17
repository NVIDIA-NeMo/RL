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

"""Native workers must not assemble a trace from a different vLLM prompt."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.vllm.utils import validate_rollout_prompt


@pytest.mark.parametrize("actual", [None, [41], [41, 42, 43], [41, 99]])
def test_returned_prompt_mismatch_is_rejected(actual):
    with pytest.raises(ValueError, match="processed prompt differs"):
        validate_rollout_prompt([41, 42], actual)


def test_returned_prompt_exact_match_and_empty_prompt():
    validate_rollout_prompt([41, 42], [41, 42])
    validate_rollout_prompt([], [])


@pytest.mark.parametrize("async_engine", [False, True])
@pytest.mark.parametrize("mismatch", [False, True])
def test_workers_validate_returned_prompt_before_rebuilding_trace(
    async_engine, mismatch, monkeypatch
):
    # Import the real workers, but replace only the engine/sampler: no GPUs or
    # weights are needed to exercise request conversion and trace construction.
    from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    # Profiling annotations are the only CUDA calls in these adapter methods.
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)

    ids = [41, 3, 7, 7, 9, 42]
    batch = BatchedDataDict(
        input_ids=torch.tensor([ids + [0, 0]]),
        input_lengths=torch.tensor([len(ids)]),
        vllm_content=["text <image> question"],
        vllm_multi_modal_data=[{"image": "image"}],
    )
    completion = SimpleNamespace(
        token_ids=[101, 102], logprobs=None, finish_reason="stop"
    )
    response = SimpleNamespace(
        prompt_token_ids=ids + [999] if mismatch else ids.copy(), outputs=[completion]
    )
    submitted = []

    def generate(prompts, sampling_params, **kwargs):
        submitted.extend(prompts)
        return [response]

    async def generate_async(*, prompt, **kwargs):
        submitted.append(prompt)
        yield response

    worker_type = (
        VllmAsyncGenerationWorkerImpl if async_engine else VllmGenerationWorkerImpl
    )
    worker = worker_type.__new__(worker_type)
    worker.routed_experts_dtype = torch.int32
    worker.cfg = {
        "_pad_token_id": 0,
        "max_new_tokens": 2,
        "vllm_cfg": {"async_engine": async_engine, "max_model_len": 128},
    }
    worker._build_sampling_params = lambda **kwargs: None
    worker.llm = SimpleNamespace(
        generate=generate_async if async_engine else generate,
        llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=128)),
    )

    async def collect():
        return [result async for _, result in worker.generate_async(batch)]

    def run():
        return asyncio.run(collect())[0] if async_engine else worker.generate(batch)

    if mismatch:
        with pytest.raises(ValueError, match="processed prompt differs"):
            run()
    else:
        result = run()
        assert result["output_ids"][0, : len(ids) + 2].tolist() == ids + [101, 102]
        assert result["generation_lengths"].tolist() == [2]
    assert submitted[0] == {
        "prompt": "text <image> question",
        "multi_modal_data": {"image": "image"},
    }
    assert batch["input_ids"][0, : len(ids)].tolist() == ids
