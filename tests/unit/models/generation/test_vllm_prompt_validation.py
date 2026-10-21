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
        renderer=SimpleNamespace(get_tokenizer=lambda: SimpleNamespace(bos_token=None)),
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


@pytest.fixture
def bos_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        models.WordLevel(
            {
                "<pad>": 0,
                "<bos>": 1,
                "<image>": 2,
                "question": 3,
                "answer": 4,
                "<unk>": 5,
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 1)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        pad_token="<pad>",
        unk_token="<unk>",
        additional_special_tokens=["<image>"],
    )


def test_bos_preparation_preserves_media_and_existing_token_inputs(bos_tokenizer):
    from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker

    worker = BaseVllmGenerationWorker.__new__(BaseVllmGenerationWorker)
    worker.llm = SimpleNamespace(
        renderer=SimpleNamespace(get_tokenizer=lambda: bos_tokenizer)
    )
    media = {"image": object()}
    prompt = {"prompt": "<bos> <image> question", "multi_modal_data": media}
    prepared = worker._tokenize_prompt_with_bos(prompt)
    assert bos_tokenizer.encode(prompt["prompt"]) == [1, 1, 2, 3]
    assert prepared["prompt_token_ids"] == [1, 2, 3]
    assert prepared["prompt"] == prompt["prompt"]
    assert prepared["multi_modal_data"] is media
    assert "prompt_token_ids" not in prompt
    assert worker._tokenize_prompt_with_bos("<bos> question") == {
        "prompt": "<bos> question",
        "prompt_token_ids": [1, 3],
    }
    assert worker._tokenize_prompt_with_bos("question") == "question"
    bos_tokenizer.bos_token = None
    assert worker._tokenize_prompt_with_bos(prompt) is prompt

    # Pretokenized and embedding inputs must not require a tokenizer at all.
    del worker.llm
    for existing in [
        {"prompt": "<bos> question", "prompt_token_ids": [1, 3]},
        {"prompt_token_ids": []},
        {"prompt_embeds": torch.zeros(1, 4)},
    ]:
        assert worker._tokenize_prompt_with_bos(existing) is existing


@pytest.mark.parametrize("async_engine", [False, True])
@pytest.mark.parametrize("evaluation", [False, True])
def test_workers_handle_mixed_bos_prompts(
    bos_tokenizer, monkeypatch, async_engine, evaluation
):
    from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    media = {"image": object()}
    raw = ["<bos> <image> question", "<image> question", None]
    expected = [[1, 2, 2, 2, 3], [1, 2, 2, 2, 3], [1, 3]]
    submitted = []

    def respond(prompt):
        submitted.append(prompt)
        if "prompt_token_ids" in prompt:
            ids = prompt["prompt_token_ids"].copy()
        else:
            ids = bos_tokenizer.encode(prompt["prompt"])
        if prompt.get("multi_modal_data"):
            # Model only the image expansion boundary; text tokenization is real.
            assert ids.count(2) == 1
            ids = [
                part for token in ids for part in ([2, 2, 2] if token == 2 else [token])
            ]
        return SimpleNamespace(
            prompt_token_ids=ids,
            outputs=[
                SimpleNamespace(
                    token_ids=[4], logprobs=None, finish_reason="stop", text=str(ids)
                )
            ],
        )

    def generate(prompts, sampling_params, **kwargs):
        return [respond(prompt) for prompt in prompts]

    async def generate_async(*, prompt, **kwargs):
        yield respond(prompt)

    worker_type = (
        VllmAsyncGenerationWorkerImpl if async_engine else VllmGenerationWorkerImpl
    )
    worker = worker_type.__new__(worker_type)
    worker.routed_experts_dtype = torch.int32
    worker.cfg = {
        "_pad_token_id": 0,
        "max_new_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": [],
        "vllm_cfg": {"async_engine": async_engine, "max_model_len": 128},
    }
    worker._build_sampling_params = lambda **kwargs: None
    worker.SamplingParams = lambda **kwargs: None
    worker.llm = SimpleNamespace(
        generate=generate_async if async_engine else generate,
        renderer=SimpleNamespace(get_tokenizer=lambda: bos_tokenizer),
        llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=128)),
    )
    batch = BatchedDataDict(
        input_ids=torch.tensor([row + [0] * (5 - len(row)) for row in expected]),
        input_lengths=torch.tensor([len(row) for row in expected]),
        vllm_content=raw,
        vllm_multi_modal_data=[media, media, {}],
    )
    if evaluation:
        batch = BatchedDataDict(
            prompts=[
                {"prompt": raw[0], "multi_modal_data": media},
                {"prompt": raw[1], "multi_modal_data": media},
                {"prompt_token_ids": expected[2]},
            ]
        )

    async def run_async():
        if evaluation:
            rows = {i: result async for i, result in worker.generate_text_async(batch)}
            return [rows[i] for i in range(3)]
        return [
            result
            for i in range(3)
            async for _, result in worker.generate_async(
                batch.select_indices(torch.tensor([i]))
            )
        ]

    if async_engine:
        results = asyncio.run(run_async())
    else:
        output = worker.generate_text(batch) if evaluation else worker.generate(batch)
        results = [output.select_indices(torch.tensor([i])) for i in range(3)]
    for result, ids in zip(results, expected):
        if evaluation:
            assert result["texts"] == [str(ids)]
        else:
            assert result["output_ids"][0, : len(ids) + 1].tolist() == ids + [4]
    assert submitted[0]["prompt_token_ids"] == [1, 2, 3]
    assert "prompt_token_ids" not in submitted[1]
    assert submitted[2]["prompt_token_ids"] == expected[2]
