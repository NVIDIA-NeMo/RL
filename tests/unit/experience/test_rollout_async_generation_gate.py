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
from nemo_rl.environments.interfaces import EnvironmentReturn
from nemo_rl.experience import rollouts
from nemo_rl.experience.rollouts import (
    generate_responses_async,
    run_sample_multi_turn_rollout,
)


class _FakeTokenizer:
    pad_token_id = 0

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return ["decoded" for _ in token_ids]

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        class _Tokenized:
            input_ids = torch.tensor([[4]], dtype=torch.long)

        return _Tokenized()


class _FakeDynamoGeneration:
    cfg = {"backend": "dynamo"}

    async def generate_async(self, data, greedy=False):
        yield (
            0,
            BatchedDataDict(
                {
                    "output_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                    "logprobs": torch.tensor([[0.0, 0.0, -0.5]], dtype=torch.float32),
                    "generation_lengths": torch.tensor([1], dtype=torch.long),
                    "unpadded_sequence_lengths": torch.tensor([3], dtype=torch.long),
                    "truncated": torch.tensor([False], dtype=torch.bool),
                }
            ),
        )


class _FakeSyncVllmGeneration:
    cfg = {"backend": "vllm", "vllm_cfg": {"async_engine": False}}

    async def generate_async(self, data, greedy=False):
        yield (
            0,
            BatchedDataDict(
                {
                    "output_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                    "logprobs": torch.tensor([[0.0, 0.0, -0.5]], dtype=torch.float32),
                    "generation_lengths": torch.tensor([1], dtype=torch.long),
                    "unpadded_sequence_lengths": torch.tensor([3], dtype=torch.long),
                }
            ),
        )


def test_generate_responses_async_accepts_dynamo_backend():
    generation_input_data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "input_lengths": torch.tensor([2], dtype=torch.long),
        }
    )
    batch = BatchedDataDict({"message_log": [[{"role": "user", "content": "x"}]]})

    updated_batch, generated_ids, gen_metrics = asyncio.run(
        generate_responses_async(
            _FakeDynamoGeneration(),
            generation_input_data,
            batch,
            _FakeTokenizer(),
            torch.tensor([2], dtype=torch.long),
        )
    )

    assert generated_ids[0].tolist() == [3]
    assert updated_batch["message_log"][0][-1]["content"] == "decoded"
    assert gen_metrics["total_generated_tokens"] == 1


def test_generate_responses_async_rejects_sync_vllm_backend():
    generation_input_data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "input_lengths": torch.tensor([2], dtype=torch.long),
        }
    )
    batch = BatchedDataDict({"message_log": [[{"role": "user", "content": "x"}]]})

    with pytest.raises(AssertionError, match="Async generation is not enabled"):
        asyncio.run(
            generate_responses_async(
                _FakeSyncVllmGeneration(),
                generation_input_data,
                batch,
                _FakeTokenizer(),
                torch.tensor([2], dtype=torch.long),
            )
        )


def test_dynamo_async_rollout_omits_empty_worker_token_counts(monkeypatch):
    def _calculate_rewards(batch, task_to_env):
        return EnvironmentReturn(
            observations=[{"role": "user", "content": "done"}],
            metadata=[{}],
            next_stop_strings=[None],
            rewards=torch.tensor([1.0]),
            terminateds=torch.tensor([True]),
            answers=[None],
        )

    monkeypatch.setattr(rollouts, "calculate_rewards", _calculate_rewards)
    initial_sample_state = {
        "message_log": [
            {
                "role": "user",
                "content": "x",
                "token_ids": torch.tensor([1, 2], dtype=torch.long),
            }
        ],
        "extra_env_info": {},
        "task_name": "fake_task",
    }

    _, sample_metrics = asyncio.run(
        run_sample_multi_turn_rollout(
            sample_idx=0,
            initial_sample_state=initial_sample_state,
            policy_generation=_FakeDynamoGeneration(),
            tokenizer=_FakeTokenizer(),
            task_to_env={},
            max_seq_len=8,
            max_rollout_turns=1,
        )
    )

    assert "per_worker_token_counts" not in sample_metrics
