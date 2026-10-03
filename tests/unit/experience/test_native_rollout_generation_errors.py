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

"""Native async generation failures must not become successful partial rollouts."""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience import rollouts


@pytest.fixture
def native_rollout(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    message_log = [
        {"role": "user", "content": "prompt", "token_ids": torch.tensor([1])}
    ]
    generation = AsyncMock(
        return_value=(
            message_log
            + [
                {
                    "role": "assistant",
                    "content": "answer",
                    "token_ids": torch.tensor([2]),
                }
            ],
            torch.tensor([2]),
            1,
            {},
        )
    )
    reward = MagicMock(
        return_value=SimpleNamespace(
            rewards=torch.tensor([1.0]),
            terminateds=torch.tensor([False]),
            observations=[{"role": "environment", "content": "continue"}],
            next_stop_strings=[None],
            metadata=[None],
        )
    )
    tokenizer = MagicMock(return_value=SimpleNamespace(input_ids=torch.tensor([[3]])))
    monkeypatch.setattr(rollouts, "async_generate_response_for_sample_turn", generation)
    monkeypatch.setattr(rollouts, "calculate_rewards", reward)
    batch = BatchedDataDict(
        {"message_log": [message_log], "extra_env_info": [{}], "task_name": ["math"]}
    )
    return SimpleNamespace(
        generation=generation, reward=reward, tokenizer=tokenizer, batch=batch
    )


async def _run_native_rollout(entrypoint: str, fixture: SimpleNamespace) -> Any:
    kwargs = {
        "policy_generation": MagicMock(),
        "tokenizer": fixture.tokenizer,
        "task_to_env": {},
        "max_seq_len": 32,
        "max_rollout_turns": 2,
    }
    if entrypoint == "sample":
        return await rollouts.run_sample_multi_turn_rollout(
            sample_idx=0,
            initial_sample_state={
                key: value[0] for key, value in fixture.batch.items()
            },
            **kwargs,
        )
    if entrypoint == "batch":
        return await rollouts._run_multi_turn_rollout_async(
            input_batch=fixture.batch, **kwargs
        )
    return [
        group
        async for group in rollouts.run_async_multi_turn_rollout_groups(
            input_batch=fixture.batch, num_generations=1, **kwargs
        )
    ]


@pytest.mark.parametrize("entrypoint", ["sample", "batch", "groups"])
@pytest.mark.parametrize("successful_turns", [0, 1])
def test_native_rollout_propagates_generation_failure(
    native_rollout: SimpleNamespace, entrypoint: str, successful_turns: int
) -> None:
    error = RuntimeError("generation unavailable")
    native_rollout.generation.side_effect = [
        *([native_rollout.generation.return_value] * successful_turns),
        error,
    ]

    with pytest.raises(RuntimeError, match="generation unavailable") as exc_info:
        asyncio.run(_run_native_rollout(entrypoint, native_rollout))

    if entrypoint == "sample":
        assert exc_info.value is error
    else:
        assert exc_info.value.__cause__ is error
    assert native_rollout.generation.await_count == successful_turns + 1
    assert native_rollout.reward.call_count == successful_turns


@pytest.mark.parametrize("entrypoint", ["sample", "batch", "groups"])
def test_native_rollout_success_still_returns_complete_trajectory(
    native_rollout: SimpleNamespace, entrypoint: str
) -> None:
    result = asyncio.run(_run_native_rollout(entrypoint, native_rollout))

    if entrypoint == "sample":
        state, metrics = result
        assert state["total_reward"].item() == 2.0
        assert metrics["turn_count"] == 2
    elif entrypoint == "batch":
        batch, metrics = result
        assert batch["total_reward"].tolist() == [2.0]
        assert metrics[0]["turn_count"] == 2
    else:
        assert len(result) == 1
        assert result[0].final_batch["total_reward"].tolist() == [2.0]
    assert native_rollout.generation.await_count == 2
    assert native_rollout.reward.call_count == 2
