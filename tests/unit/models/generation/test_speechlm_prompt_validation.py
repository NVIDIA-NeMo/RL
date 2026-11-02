# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SpeechLM's expanded audio prompt must preserve text and compact trace alignment."""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.vllm.utils import validate_rollout_prompt


@pytest.mark.parametrize("audio_length", [1, 2, 35, 2048])
def test_single_audio_placeholder_expansion(audio_length):
    validate_rollout_prompt(
        [41, 7, 42],
        [41] + [7] * audio_length + [42],
        expanded_placeholder_token_id=7,
    )


@pytest.mark.parametrize(
    "actual",
    [
        None,
        [41, 42],
        [99, 41, 7, 7, 42],
        [99, 7, 7, 42],
        [41, 7, 7, 99],
        [41, 7, 99, 7, 42],
        [41, 8, 8, 42],
        [41, 7, 7, 42, 99],
    ],
)
def test_audio_expansion_never_hides_text_or_position_changes(actual):
    with pytest.raises(ValueError, match="processed prompt differs"):
        validate_rollout_prompt([41, 7, 42], actual, expanded_placeholder_token_id=7)


def test_expansion_requires_explicit_single_placeholder_contract():
    with pytest.raises(ValueError, match="processed prompt differs"):
        validate_rollout_prompt([41, 7, 42], [41, 7, 7, 42])
    with pytest.raises(ValueError, match="processed prompt differs"):
        validate_rollout_prompt(
            [41, 7, 42, 7], [41, 7, 7, 42, 7], expanded_placeholder_token_id=7
        )


@pytest.mark.parametrize("native_speechlm", [False, True])
@pytest.mark.parametrize("change_text", [False, True])
def test_native_worker_keeps_compact_response_and_logprob_positions(
    monkeypatch, native_speechlm, change_text
):
    from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl

    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    batch = BatchedDataDict(
        input_ids=torch.tensor([[41, 7, 42, 0, 0]]),
        input_lengths=torch.tensor([3]),
        vllm_content=["instruction <|audio|>"],
        vllm_multi_modal_data=[{"audio": object()}],
    )
    completion = SimpleNamespace(
        token_ids=[101, 102],
        logprobs=[
            {101: SimpleNamespace(logprob=-0.25)},
            {102: SimpleNamespace(logprob=-0.5)},
        ],
        finish_reason="stop",
    )
    response = SimpleNamespace(
        prompt_token_ids=[99 if change_text else 41] + [7] * 35 + [42],
        outputs=[completion],
    )
    worker = VllmGenerationWorkerImpl.__new__(VllmGenerationWorkerImpl)
    worker.routed_experts_dtype = torch.int32
    worker.cfg = {
        "_pad_token_id": 0,
        "max_new_tokens": 2,
        "vllm_cfg": {"native_speechlm": native_speechlm, "max_model_len": 128},
    }
    worker._build_sampling_params = lambda **kwargs: None
    worker._native_lora_request = None
    tokenizer = SimpleNamespace(bos_token=None, get_vocab=lambda: {"<|audio|>": 7})
    worker.llm = SimpleNamespace(
        generate=lambda *args, **kwargs: [response],
        renderer=SimpleNamespace(get_tokenizer=lambda: tokenizer),
        llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=128)),
    )
    if not native_speechlm or change_text:
        with pytest.raises(ValueError, match="processed prompt differs"):
            worker.generate(batch)
    else:
        result = worker.generate(batch)
        assert result["output_ids"].tolist() == [[41, 7, 42, 101, 102, 0, 0]]
        assert result["logprobs"].tolist() == [[0, 0, 0, -0.25, -0.5, 0, 0]]
        assert result["generation_lengths"].tolist() == [2]
        assert result["unpadded_sequence_lengths"].tolist() == [5]
