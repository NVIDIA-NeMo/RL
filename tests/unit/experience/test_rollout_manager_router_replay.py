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

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentReturn
from nemo_rl.experience.rollout_manager import AsyncRolloutImpl


def _routes(count: int, *, start: int = 0) -> torch.Tensor:
    token_routes = torch.arange(start, start + count, dtype=torch.int16).view(
        count, 1, 1
    )
    topk_offsets = torch.arange(2, dtype=torch.int16).view(1, 1, 2)
    return (token_routes + topk_offsets).expand(count, 2, 2).contiguous()


def _fallback_routes(count: int) -> torch.Tensor:
    return torch.arange(2, dtype=torch.int16).view(1, 1, 2).expand(count, 2, 2)


class _FakeTokenizer:
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool) -> str:
        del token_ids, skip_special_tokens
        return "generated"

    def __call__(self, text: str, **kwargs) -> SimpleNamespace:
        del text, kwargs
        return SimpleNamespace(input_ids=torch.tensor([[31, 32]]))


class _FakeGeneration:
    def __init__(self, outputs: BatchedDataDict | list[BatchedDataDict]) -> None:
        self._outputs = outputs if isinstance(outputs, list) else [outputs]
        self._next_output = 0

    async def generate_async(self, data: BatchedDataDict):
        del data
        output = self._outputs[self._next_output]
        self._next_output += 1
        yield 0, output


def _rollout_impl(
    output: BatchedDataDict | list[BatchedDataDict],
    *,
    max_rollout_turns: int = 1,
) -> AsyncRolloutImpl:
    return AsyncRolloutImpl(
        tokenizer=_FakeTokenizer(),  # type: ignore[arg-type]
        task_to_env={},
        num_generations_per_prompt=1,
        max_seq_len=32,
        max_rollout_turns=max_rollout_turns,
        policy_generation=_FakeGeneration(output),  # type: ignore[arg-type]
    )


def _generation_output(
    token_ids: tuple[int, ...] = (10, 11, 12, 20, 21),
    *,
    route_start: int = 10,
) -> BatchedDataDict:
    routed_experts = _routes(len(token_ids), start=route_start)
    routed_experts[-1] = _fallback_routes(1)[0]
    logprobs = torch.zeros((1, len(token_ids)))
    logprobs[0, -2:] = torch.tensor([-0.1, -0.2])
    return BatchedDataDict(
        {
            "output_ids": torch.tensor([token_ids]),
            "logprobs": logprobs,
            "unpadded_sequence_lengths": torch.tensor([len(token_ids)]),
            "truncated": torch.tensor([False]),
            "routed_experts": routed_experts.unsqueeze(0),
        }
    )


def test_generate_response_attaches_prefix_and_generated_routes() -> None:
    impl = _rollout_impl(_generation_output())
    message_log = [
        {
            "role": "system",
            "content": "system",
            "token_ids": torch.tensor([10]),
        },
        {
            "role": "user",
            "content": "prompt",
            "token_ids": torch.tensor([11, 12]),
        },
    ]

    assistant_message, input_lengths, _ = asyncio.run(
        impl._generate_response(message_log, stop_strings=None)
    )

    assert input_lengths.tolist() == [3]
    assert torch.equal(message_log[0]["routed_experts"], _routes(1, start=10))
    assert torch.equal(message_log[1]["routed_experts"], _routes(2, start=11))
    assert torch.equal(
        assistant_message["routed_experts"],
        torch.cat((_routes(1, start=13), _fallback_routes(1))),
    )


def test_single_rollout_adds_fallback_routes_to_environment_tokens() -> None:
    impl = _rollout_impl(_generation_output())
    input_sample = {
        "idx": 0,
        "message_log": [
            {
                "role": "user",
                "content": "prompt",
                "token_ids": torch.tensor([10, 11, 12]),
            }
        ],
        "extra_env_info": None,
        "task_name": "test",
    }
    env_output = EnvironmentReturn(
        observations=[{"role": "user", "content": "environment"}],
        metadata=[None],
        next_stop_strings=[None],
        rewards=torch.tensor([1.0]),
        terminateds=torch.tensor([True]),
        answers=[None],
    )

    with patch(
        "nemo_rl.experience.rollout_manager.calculate_rewards",
        return_value=env_output,
    ):
        completion, _ = asyncio.run(impl._run_single_rollout(input_sample, traj_idx=0))

    env_message = completion.message_log[-1]
    assert env_message["role"] == "user"
    assert torch.equal(env_message["routed_experts"], _fallback_routes(2))


def test_second_turn_overwrites_prefix_fallback_routes() -> None:
    second_output = _generation_output(
        (10, 11, 12, 20, 21, 31, 32, 40, 41),
        route_start=100,
    )
    impl = _rollout_impl(
        [_generation_output(), second_output],
        max_rollout_turns=2,
    )
    input_sample = {
        "idx": 0,
        "message_log": [
            {
                "role": "user",
                "content": "prompt",
                "token_ids": torch.tensor([10, 11, 12]),
            }
        ],
        "extra_env_info": None,
        "task_name": "test",
    }
    first_env_output = EnvironmentReturn(
        observations=[{"role": "user", "content": "environment"}],
        metadata=[None],
        next_stop_strings=[None],
        rewards=torch.tensor([0.0]),
        terminateds=torch.tensor([False]),
        answers=[None],
    )
    final_env_output = EnvironmentReturn(
        observations=[{"role": "user", "content": "environment"}],
        metadata=[None],
        next_stop_strings=[None],
        rewards=torch.tensor([1.0]),
        terminateds=torch.tensor([True]),
        answers=[None],
    )

    with patch(
        "nemo_rl.experience.rollout_manager.calculate_rewards",
        side_effect=[first_env_output, final_env_output],
    ):
        completion, _ = asyncio.run(impl._run_single_rollout(input_sample, traj_idx=0))

    prompt, first_assistant, first_env, second_assistant, final_env = (
        completion.message_log
    )
    assert torch.equal(prompt["routed_experts"], _routes(3, start=100))
    assert torch.equal(first_assistant["routed_experts"], _routes(2, start=103))
    assert torch.equal(first_env["routed_experts"], _routes(2, start=105))
    assert torch.equal(
        second_assistant["routed_experts"],
        torch.cat((_routes(1, start=107), _fallback_routes(1))),
    )
    assert torch.equal(final_env["routed_experts"], _fallback_routes(2))
