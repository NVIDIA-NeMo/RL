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

"""Engine completion reasons survive the generation and rollout adapters."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import nemo_rl.experience.rollouts as rollouts
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin


class Tokenizer:
    pad_token_id = 0

    def batch_decode(self, ids, **kwargs):
        return ["answer" for _ in ids]

    def __call__(self, text, **kwargs):
        assert text == ""
        return SimpleNamespace(input_ids=torch.empty((1, 0), dtype=torch.long))


@pytest.mark.parametrize("reason,expected", [("length", True), ("stop", False)])
def test_truncation_reaches_rollout_metrics(reason, expected):
    tokenizer = Tokenizer()
    worker = SimpleNamespace(tokenizer=tokenizer)
    # Equal-length outputs can either exhaust the response budget or stop at it.
    result = SimpleNamespace(
        prompt_tokens=[10, 11],
        generated_tokens=[20, 21, 22, 23],
        generated_log_probs=[-1.0] * 4,
        finish_reason=reason,
    )

    class Generation:
        def generate(self, data, **kwargs):
            return MegatronGenerationMixin._parse_result_to_batched_data_dict(
                worker, data, [result]
            )

    batch = BatchedDataDict(
        {
            "message_log": [
                [
                    {
                        "role": "user",
                        "content": "question",
                        "token_ids": torch.tensor([10, 11]),
                    }
                ]
            ],
            "task_name": ["math"],
            "extra_env_info": [{}],
        }
    )
    feedback = SimpleNamespace(
        rewards=torch.tensor([0.0]),
        terminateds=torch.tensor([True]),
        observations=[{"role": "environment", "content": ""}],
        next_stop_strings=[None],
        metadata=[None],
    )
    with patch.object(rollouts, "calculate_rewards", return_value=feedback):
        output, metrics = rollouts.run_multi_turn_rollout(
            Generation(), batch, tokenizer, {}, max_seq_len=16, max_rollout_turns=1
        )
    assert output["truncated"].tolist() == [expected]
    assert metrics["truncation_rate"] == float(expected)


@pytest.mark.parametrize("reason", [None, "unknown", "error"])
def test_missing_or_unknown_reason_is_not_reported_as_success(reason):
    data = BatchedDataDict(
        {"input_ids": torch.tensor([[10]]), "input_lengths": torch.tensor([1])}
    )
    request = SimpleNamespace(
        generated_tokens=[20], generated_log_probs=[-1.0], finish_reason=reason
    )
    with pytest.raises(RuntimeError, match="finish_reason"):
        MegatronGenerationMixin._parse_result_to_batched_data_dict(
            SimpleNamespace(tokenizer=Tokenizer()), data, [request]
        )


def test_old_engine_rejected_before_engine_construction():
    @dataclass
    class OldRequest:
        generated_length: int = 0

    with patch(
        "nemo_rl.models.generation.megatron.megatron_worker.DynamicInferenceRequest",
        OldRequest,
    ):
        with pytest.raises(RuntimeError, match="Upgrade Megatron-Core"):
            MegatronGenerationMixin._initialize_inference_engine(
                SimpleNamespace(_inference_engine_initialized=False), {}
            )
