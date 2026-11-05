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

"""An expanded media placeholder must preserve prompt text and response positions."""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.vllm.utils import validate_rollout_prompt

PLACEHOLDER = "<|media|>"
PLACEHOLDER_ID = 7


@pytest.mark.parametrize("expanded_length", [1, 2, 35, 2048])
def test_single_placeholder_expansion_is_accepted(expanded_length):
    validate_rollout_prompt(
        [41, PLACEHOLDER_ID, 42],
        [41] + [PLACEHOLDER_ID] * expanded_length + [42],
        expanded_placeholder_token_id=PLACEHOLDER_ID,
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
def test_expansion_never_hides_text_or_position_changes(actual):
    with pytest.raises(ValueError, match="processed prompt differs"):
        validate_rollout_prompt(
            [41, PLACEHOLDER_ID, 42],
            actual,
            expanded_placeholder_token_id=PLACEHOLDER_ID,
        )


def test_expansion_requires_explicit_single_placeholder():
    with pytest.raises(ValueError, match="processed prompt differs"):
        validate_rollout_prompt([41, 7, 42], [41, 7, 7, 42])
    with pytest.raises(ValueError, match="processed prompt differs"):
        validate_rollout_prompt(
            [41, 7, 42, 7], [41, 7, 7, 42, 7], expanded_placeholder_token_id=7
        )


def _worker(vllm_cfg, response):
    from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl

    worker = VllmGenerationWorkerImpl.__new__(VllmGenerationWorkerImpl)
    worker.routed_experts_dtype = torch.int32
    worker.cfg = {
        "_pad_token_id": 0,
        "max_new_tokens": 2,
        "vllm_cfg": {"max_model_len": 128, **vllm_cfg},
    }
    worker._build_sampling_params = lambda **kwargs: None
    worker._native_lora_request = None
    tokenizer = SimpleNamespace(
        bos_token=None, get_vocab=lambda: {PLACEHOLDER: PLACEHOLDER_ID}
    )
    worker.llm = SimpleNamespace(
        generate=lambda *args, **kwargs: [response],
        renderer=SimpleNamespace(get_tokenizer=lambda: tokenizer),
        llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=128)),
    )
    return worker


@pytest.mark.parametrize("declare_placeholder", [False, True])
@pytest.mark.parametrize("change_text", [False, True])
def test_worker_keeps_compact_response_and_logprob_positions(
    monkeypatch, declare_placeholder, change_text
):
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    batch = BatchedDataDict(
        input_ids=torch.tensor([[41, PLACEHOLDER_ID, 42, 0, 0]]),
        input_lengths=torch.tensor([3]),
        vllm_content=[f"instruction {PLACEHOLDER}"],
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
        prompt_token_ids=[99 if change_text else 41] + [PLACEHOLDER_ID] * 35 + [42],
        outputs=[completion],
    )
    vllm_cfg = (
        {"expanded_placeholder_token": PLACEHOLDER} if declare_placeholder else {}
    )
    worker = _worker(vllm_cfg, response)
    if not declare_placeholder or change_text:
        with pytest.raises(ValueError, match="processed prompt differs"):
            worker.generate(batch)
    else:
        result = worker.generate(batch)
        assert result["output_ids"].tolist() == [[41, 7, 42, 101, 102, 0, 0]]
        assert result["logprobs"].tolist() == [[0, 0, 0, -0.25, -0.5, 0, 0]]
        assert result["generation_lengths"].tolist() == [2]
        assert result["unpadded_sequence_lengths"].tolist() == [5]


def test_unknown_placeholder_token_is_rejected():
    worker = _worker({"expanded_placeholder_token": "<|missing|>"}, response=None)
    with pytest.raises(ValueError, match="expanded_placeholder_token"):
        worker._expanded_placeholder_token_id
