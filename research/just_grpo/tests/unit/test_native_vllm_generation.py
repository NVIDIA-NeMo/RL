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
"""Reference-fork contracts on top of the native vLLM generation worker."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from just_grpo.generation.reference_vllm import (
    ReferenceVllmWorkerImpl,
    ReferenceVllmAsyncWorkerImpl,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl


@pytest.mark.parametrize(
    "worker_cls", [ReferenceVllmWorkerImpl, ReferenceVllmAsyncWorkerImpl]
)
@pytest.mark.parametrize("temperature", [0.7, 1.0, 2.0])
@pytest.mark.parametrize("greedy", [False, True])
def test_native_sampling_keeps_reference_temperature_and_commit_logprobs(
    temperature, greedy, worker_cls
):
    worker = object.__new__(worker_cls)
    worker.cfg = dict(
        temperature=temperature,
        top_p=1.0,
        top_k=None,
        max_new_tokens=512,
        stop_token_ids=[11],
    )
    worker.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    params = worker._build_sampling_params(
        greedy=greedy, stop_strings=None, max_new_tokens=32
    )
    assert params.temperature == (0.0 if greedy else 1.0)
    assert params.logprobs == 0
    assert params.max_tokens == 32
    assert params.stop_token_ids == [11]
    assert params.top_p == 1.0
    assert params.top_k == -1


@pytest.mark.parametrize("leftmost", [False, True])
def test_engine_requires_leftmost_support_and_reuses_native_creation(
    monkeypatch, leftmost
):
    for name in ("vllm", "vllm.config", "vllm.config.diffusion"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["vllm.config.diffusion"].DiffusionConfig = SimpleNamespace(
        __annotations__={"selection_policy": "leftmost" if leftmost else "confidence"}
    )
    create = Mock()
    monkeypatch.setattr(VllmGenerationWorkerImpl, "_create_engine", create)
    worker = object.__new__(ReferenceVllmWorkerImpl)
    kwargs = {"diffusion_config": {"temperature": 0.7}, "enable_sleep_mode": True}
    if leftmost:
        worker._create_engine(kwargs)
        create.assert_called_once_with(kwargs)
    else:
        with pytest.raises(RuntimeError, match="leftmost"):
            worker._create_engine(kwargs)
        create.assert_not_called()


@pytest.mark.parametrize("max_model_len", [47, 48])
def test_context_reserves_diffusion_canvas_and_delegates_output_packing(
    monkeypatch, max_model_len
):
    worker = object.__new__(ReferenceVllmWorkerImpl)
    worker.cfg = dict(
        max_new_tokens=16,
        vllm_cfg={"max_model_len": max_model_len},
        vllm_kwargs={"diffusion_config": {"canvas_length": 16}},
    )
    batch = BatchedDataDict(
        input_ids=torch.ones((1, 16), dtype=torch.long),
        input_lengths=torch.tensor([16]),
    )
    generate = Mock(return_value=object())
    monkeypatch.setattr(VllmGenerationWorkerImpl, "generate", generate)
    if max_model_len == 47:
        with pytest.raises(ValueError, match="context length"):
            worker.generate(batch)
        generate.assert_not_called()
    else:
        assert worker.generate(batch) is generate.return_value
        generate.assert_called_once_with(batch, greedy=False)


@pytest.mark.parametrize("max_model_len", [47, 48])
def test_async_context_check_and_native_stream_delegation(monkeypatch, max_model_len):
    import asyncio
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    worker = object.__new__(ReferenceVllmAsyncWorkerImpl)
    worker.cfg = dict(
        max_new_tokens=16,
        vllm_cfg={"max_model_len": max_model_len},
        vllm_kwargs={"diffusion_config": {"canvas_length": 16}},
    )
    batch = BatchedDataDict(
        input_ids=torch.ones((1, 16), dtype=torch.long),
        input_lengths=torch.tensor([16]),
    )
    called = []

    async def generate(self, data, greedy=False):
        called.append((data, greedy))
        yield 0, data

    monkeypatch.setattr(VllmAsyncGenerationWorkerImpl, "generate_async", generate)

    async def collect():
        return [item async for item in worker.generate_async(batch, greedy=True)]

    if max_model_len == 47:
        with pytest.raises(ValueError, match="context length"):
            asyncio.run(collect())
        assert not called
    else:
        assert asyncio.run(collect())[0][1] is batch
        assert called == [(batch, True)]
