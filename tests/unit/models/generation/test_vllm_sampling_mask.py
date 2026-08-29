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

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.vllm.config import resolve_vllm_sampling_mask_top_k
from nemo_rl.models.generation.vllm.utils import pad_and_align_sampling_mask
from nemo_rl.models.generation.vllm import vllm_worker as vllm_worker_module
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)


def _worker_config(*, async_engine: bool, max_model_len: int = 16) -> dict:
    return {
        "top_k": 3,
        "top_p": 0.9,
        "temperature": 1.0,
        "max_new_tokens": 2,
        "stop_token_ids": None,
        "stop_strings": None,
        "_pad_token_id": 0,
        "vllm_cfg": {
            "async_engine": async_engine,
            "max_model_len": max_model_len,
            "cap_max_tokens_to_context": False,
            "use_tqdm": False,
        },
        "vllm_kwargs": {"return_sampling_mask": True},
    }


def _completion(
    token_ids: list[int], supports: list[list[int]], *, finish_reason: str = "stop"
) -> SimpleNamespace:
    return SimpleNamespace(
        token_ids=token_ids,
        logprobs=[{token_id: SimpleNamespace(logprob=-0.25)} for token_id in token_ids],
        sampling_mask=SimpleNamespace(token_ids=supports),
        routed_experts=None,
        finish_reason=finish_reason,
    )


def test_pad_and_align_sampling_mask_uses_response_positions() -> None:
    completion = _completion([5, 6], [[5, 7], [6]])

    token_ids, sizes = pad_and_align_sampling_mask(
        completion,
        [5, 6],
        prompt_length=2,
        padded_length=5,
        top_k=3,
        device=torch.device("cpu"),
    )

    assert token_ids.dtype == torch.int32
    assert sizes.dtype == torch.int32
    assert token_ids.tolist() == [
        [0, 0, 0],
        [0, 0, 0],
        [5, 7, 0],
        [6, 0, 0],
        [0, 0, 0],
    ]
    assert sizes.tolist() == [0, 0, 2, 1, 0]


@pytest.mark.parametrize(
    (
        "completion",
        "generated_token_ids",
        "top_k",
        "exception_type",
        "error_match",
    ),
    [
        (
            SimpleNamespace(),
            [5],
            3,
            RuntimeError,
            "did not include sampling_mask",
        ),
        (
            SimpleNamespace(sampling_mask=SimpleNamespace()),
            [5],
            3,
            ValueError,
            "did not include token_ids",
        ),
        (
            _completion([5], [[5]]),
            [5, 6],
            3,
            ValueError,
            "token count does not match",
        ),
        (_completion([5], [[]]), [5], 3, ValueError, "empty sampling support"),
        (
            _completion([5], [[6]]),
            [5],
            3,
            ValueError,
            "does not contain the sampled token",
        ),
        (
            _completion([5], [[-1, 5]]),
            [5],
            3,
            ValueError,
            "outside the int32 range",
        ),
        (
            _completion([5], [[5, 6, 7, 8]]),
            [5],
            3,
            ValueError,
            "exceeds the configured top_k",
        ),
    ],
)
def test_pad_and_align_sampling_mask_rejects_malformed_output(
    completion: SimpleNamespace,
    generated_token_ids: list[int],
    top_k: int,
    exception_type: type[Exception],
    error_match: str,
) -> None:
    with pytest.raises(exception_type, match=error_match):
        pad_and_align_sampling_mask(
            completion,
            generated_token_ids,
            prompt_length=2,
            padded_length=5,
            top_k=top_k,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("top_k", [None, 0, -1, True])
def test_resolve_vllm_sampling_mask_top_k_requires_positive_integer(top_k) -> None:
    config = _worker_config(async_engine=False)
    config["top_k"] = top_k

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        resolve_vllm_sampling_mask_top_k(config)


def test_vllm_engine_receives_return_sampling_mask(monkeypatch) -> None:
    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    setattr(fake_vllm, "SamplingParams", object)
    fake_vllm_logger = ModuleType("vllm.logger")
    setattr(
        fake_vllm_logger,
        "init_logger",
        lambda _: SimpleNamespace(warning=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.logger", fake_vllm_logger)

    config = _worker_config(async_engine=False)
    config["model_name"] = "test-model"
    config["vllm_cfg"].update(
        {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.5,
            "precision": "bfloat16",
            "load_format": "dummy",
            "skip_tokenizer_init": True,
            "enforce_eager": True,
            "enable_prefix_caching": False,
        }
    )

    worker = object.__new__(VllmGenerationWorkerImpl)
    worker.cfg = config
    worker.model_name = config["model_name"]
    worker.tensor_parallel_size = 1
    worker.pipeline_parallel_size = 1
    worker.expert_parallel_size = 1
    worker.enable_expert_parallel = False
    worker.gpu_memory_utilization = 0.5
    worker.precision = "bfloat16"
    worker.fraction_of_gpus = 1.0

    captured_engine_kwargs = {}
    worker._create_engine = captured_engine_kwargs.update

    monkeypatch.setattr(
        vllm_worker_module, "log_gpu_memory_diagnostics", lambda **_: None
    )
    monkeypatch.setattr(
        vllm_worker_module, "checkpoint_engine_refit_config", lambda _: None
    )
    monkeypatch.setattr(
        vllm_worker_module,
        "resolve_distributed_executor_backend",
        lambda *_: "uni",
    )
    monkeypatch.setattr(vllm_worker_module, "is_vllm_v1_engine_enabled", lambda: True)
    monkeypatch.setattr(
        type(vllm_worker_module.ModelFlag.VLLM_LOAD_FORMAT_AUTO),
        "matches",
        lambda *_: False,
    )
    monkeypatch.setattr(
        vllm_worker_module.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(architectures=[]),
    )
    monkeypatch.setattr(vllm_worker_module, "get_num_routed_experts", lambda _: None)
    monkeypatch.setattr(
        vllm_worker_module,
        "resolve_routed_experts_dtype",
        lambda _: torch.int32,
    )
    monkeypatch.setattr(
        vllm_worker_module, "_maybe_enable_vllm_native_tracing", lambda _: None
    )
    monkeypatch.setenv("VLLM_USE_V1", "test-original")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "test-original")

    VllmGenerationWorkerImpl._load_model.__wrapped__(worker, [0], 123)

    assert captured_engine_kwargs["return_sampling_mask"] is True


def test_sync_vllm_worker_captures_and_aligns_sampling_mask() -> None:
    generation = _completion([5, 6], [[5, 7], [6]])
    request_output = SimpleNamespace(outputs=[generation])

    class FakeLLM:
        llm_engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=16))

        def generate(self, prompts, sampling_params, use_tqdm):
            return [request_output]

    worker = object.__new__(VllmGenerationWorkerImpl)
    worker.cfg = _worker_config(async_engine=False)
    worker.SamplingParams = lambda **kwargs: kwargs
    worker.llm = FakeLLM()
    worker.routed_experts_dtype = torch.int32
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[11, 12, 0]]),
            "input_lengths": torch.tensor([2]),
        }
    )

    result = worker.generate(data)

    assert result["sampling_mask_token_ids"].dtype == torch.int32
    assert result["sampling_mask_token_ids"].shape == (1, 5, 3)
    assert result["sampling_mask_token_ids"][0].tolist() == [
        [0, 0, 0],
        [0, 0, 0],
        [5, 7, 0],
        [6, 0, 0],
        [0, 0, 0],
    ]
    assert result["sampling_mask_sizes"].tolist() == [[0, 0, 2, 1, 0]]


def test_sync_vllm_worker_rejects_greedy_sampling_mask_replay() -> None:
    worker = object.__new__(VllmGenerationWorkerImpl)
    worker.cfg = _worker_config(async_engine=False)
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[11, 12]]),
            "input_lengths": torch.tensor([2]),
        }
    )

    with pytest.raises(ValueError, match="does not support greedy generation"):
        worker.generate(data, greedy=True)


@pytest.mark.asyncio
async def test_async_vllm_worker_captures_and_aligns_sampling_mask() -> None:
    generation = _completion([5, 6], [[5, 7], [6]])
    request_output = SimpleNamespace(outputs=[generation])

    class FakeAsyncLLM:
        def generate(self, *, prompt, sampling_params, request_id):
            async def responses():
                yield request_output

            return responses()

    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = _worker_config(async_engine=True)
    worker.SamplingParams = lambda **kwargs: kwargs
    worker.llm = FakeAsyncLLM()
    worker.routed_experts_dtype = torch.int32
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[11, 12, 0]]),
            "input_lengths": torch.tensor([2]),
        }
    )

    results = [result async for result in worker.generate_async(data)]

    assert len(results) == 1
    sample_idx, result = results[0]
    assert sample_idx == 0
    assert result["sampling_mask_token_ids"].dtype == torch.int32
    assert result["sampling_mask_token_ids"].tolist() == [
        [
            [0, 0, 0],
            [0, 0, 0],
            [5, 7, 0],
            [6, 0, 0],
        ]
    ]
    assert result["sampling_mask_sizes"].tolist() == [[0, 0, 2, 1]]


@pytest.mark.asyncio
async def test_async_vllm_worker_rejects_greedy_sampling_mask_replay() -> None:
    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = _worker_config(async_engine=True)
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[11, 12]]),
            "input_lengths": torch.tensor([2]),
        }
    )

    with pytest.raises(ValueError, match="does not support greedy generation"):
        _ = [result async for result in worker.generate_async(data, greedy=True)]


@pytest.mark.asyncio
async def test_async_vllm_worker_emits_zero_mask_without_generation() -> None:
    class UnexpectedLLM:
        def generate(self, **kwargs):
            raise AssertionError("vLLM should not be called without remaining context")

    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker.cfg = _worker_config(async_engine=True, max_model_len=2)
    worker.SamplingParams = lambda **kwargs: kwargs
    worker.llm = UnexpectedLLM()
    worker.routed_experts_dtype = torch.int32
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[11, 12]]),
            "input_lengths": torch.tensor([2]),
        }
    )

    results = [result async for result in worker.generate_async(data)]

    assert len(results) == 1
    _, result = results[0]
    assert result["generation_lengths"].tolist() == [0]
    assert result["sampling_mask_token_ids"].shape == (1, 2, 3)
    assert torch.count_nonzero(result["sampling_mask_token_ids"]) == 0
    assert result["sampling_mask_sizes"].tolist() == [[0, 0]]
