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
from pydantic import ValidationError
from ray.exceptions import ActorDiedError

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience import rollouts
from nemo_rl.models.generation.interfaces import NativeGenerationRetryConfig


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


@pytest.mark.parametrize("successful_turns", [0, 1])
@pytest.mark.parametrize("error_kind", ["timeout", "actor_death"])
def test_retry_commits_only_success_and_never_replays_environment(
    native_rollout, successful_turns, error_kind
):
    calls = 0
    histories = []

    async def generate(_policy, messages, *_args, **_kwargs):
        nonlocal calls
        histories.append([m["content"] for m in messages])
        calls += 1
        if calls == successful_turns + 1:
            messages.append({"role": "assistant", "content": "uncommitted"})
            raise (
                TimeoutError("transient")
                if error_kind == "timeout"
                else ActorDiedError()
            )
        messages.append(
            {"role": "assistant", "content": "answer", "token_ids": torch.tensor([2])}
        )
        return messages, torch.tensor([2]), 1, {}

    native_rollout.generation.side_effect = generate
    result, metrics = asyncio.run(
        rollouts.run_sample_multi_turn_rollout(
            0,
            {key: value[0] for key, value in native_rollout.batch.items()},
            MagicMock(),
            native_rollout.tokenizer,
            {},
            32,
            max_rollout_turns=2,
            retry_config=NativeGenerationRetryConfig(backoff_seconds=0),
        )
    )
    assert calls == 3
    assert histories[successful_turns] == histories[successful_turns + 1]
    assert all("uncommitted" not in history for history in histories)
    assert [m["content"] for m in result["message_log"]] == [
        "prompt",
        "answer",
        "continue",
        "answer",
        "continue",
    ]
    assert native_rollout.reward.call_count == 2
    assert metrics["generation_retries"] == 1
    assert metrics["turn_count"] == 2
    assert len(native_rollout.batch["message_log"][0]) == 1


def test_exhausted_group_preserves_unrelated_complete_group(native_rollout):
    attempts = {}

    async def generate(_policy, messages, *_args, **_kwargs):
        prompt = messages[0]["content"]
        attempts[prompt] = attempts.get(prompt, 0) + 1
        if prompt == "bad":
            raise TimeoutError("engine unavailable")
        return native_rollout.generation.return_value

    native_rollout.generation.side_effect = generate
    batch = BatchedDataDict(
        {
            "message_log": [
                [{"role": "user", "content": prompt, "token_ids": torch.tensor([1])}]
                for prompt in ["bad", "sibling", "good", "good"]
            ],
            "extra_env_info": [{}, {}, {}, {}],
            "task_name": ["math"] * 4,
            "idx": [10, 11, 12, 13],
            "policy_version": torch.tensor([7] * 4),
        }
    )
    groups = []

    async def run():
        with pytest.raises(
            rollouts.NativeGenerationRetriesExhausted, match=r"groups \[0\]"
        ):
            async for group in rollouts.run_async_multi_turn_rollout_groups(
                MagicMock(),
                batch,
                native_rollout.tokenizer,
                {},
                32,
                2,
                max_rollout_turns=1,
                retry_config=NativeGenerationRetryConfig(backoff_seconds=0),
            ):
                groups.append(group)
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(run())
    assert attempts == {"bad": 3, "sibling": 1, "good": 2}
    assert native_rollout.reward.call_count == 3
    assert [g.group_index for g in groups] == [1]
    assert groups[0].final_batch["idx"] == [12, 13]
    assert groups[0].final_batch["policy_version"].tolist() == [7, 7]
    assert groups[0].final_batch.size == 2


@pytest.mark.parametrize("during_backoff", [False, True])
def test_cancellation_drains_generation_without_retry(native_rollout, during_backoff):
    async def run():
        entered = asyncio.Event()
        drained = asyncio.Event()

        async def generate(*_args, **_kwargs):
            try:
                entered.set()
                if during_backoff:
                    raise TimeoutError("transient")
                await asyncio.Event().wait()
            finally:
                drained.set()

        native_rollout.generation.side_effect = generate
        task = asyncio.create_task(
            rollouts._run_multi_turn_rollout_async(
                MagicMock(),
                native_rollout.batch,
                native_rollout.tokenizer,
                {},
                32,
                retry_config=NativeGenerationRetryConfig(backoff_seconds=60),
            )
        )
        await entered.wait()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert drained.is_set()
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(run())
    assert native_rollout.generation.await_count == 1
    native_rollout.reward.assert_not_called()


def test_generation_deadline_drains_attempt(native_rollout):
    drained = []

    async def generate(*_args, **_kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            drained.append(True)

    native_rollout.generation.side_effect = generate
    with pytest.raises(RuntimeError, match="deadline"):
        asyncio.run(
            rollouts._run_multi_turn_rollout_async(
                MagicMock(),
                native_rollout.batch,
                native_rollout.tokenizer,
                {},
                32,
                retry_config=NativeGenerationRetryConfig(deadline_seconds=0.01),
            )
        )
    assert drained == [True]
    native_rollout.reward.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_retries": -1},
        {"backoff_seconds": -1},
        {"deadline_seconds": 0},
        {"deadline_seconds": float("inf")},
    ],
)
def test_invalid_retry_config_rejected(kwargs):
    with pytest.raises(ValidationError):
        NativeGenerationRetryConfig(**kwargs)


@pytest.mark.parametrize("explicit_retry", [False, True])
def test_backend_must_opt_in_to_safe_retries(native_rollout, explicit_retry):
    policy = MagicMock()
    policy.supports_native_generation_retries.return_value = False
    native_rollout.generation.side_effect = TimeoutError(
        "unsupported backend timed out"
    )
    with pytest.raises(ValueError if explicit_retry else TimeoutError):
        asyncio.run(
            rollouts.run_sample_multi_turn_rollout(
                0,
                {key: value[0] for key, value in native_rollout.batch.items()},
                policy,
                native_rollout.tokenizer,
                {},
                32,
                retry_config=NativeGenerationRetryConfig() if explicit_retry else None,
            )
        )
    assert native_rollout.generation.await_count == (0 if explicit_retry else 1)
    native_rollout.reward.assert_not_called()


def test_unknown_failure_cancels_other_samples_immediately(native_rollout):
    async def run():
        entered = asyncio.Event()
        drained = asyncio.Event()

        async def generate(_policy, messages, *_args, **_kwargs):
            if messages[0]["content"] == "bad":
                await entered.wait()
                raise ValueError("invalid output shape")
            try:
                entered.set()
                await asyncio.Event().wait()
            finally:
                drained.set()

        native_rollout.generation.side_effect = generate
        batch = BatchedDataDict(
            {
                "message_log": [[{"content": prompt}] for prompt in ["bad", "waiting"]],
                "extra_env_info": [{}, {}],
                "task_name": ["math", "math"],
            }
        )
        with pytest.raises(RuntimeError, match="invalid output shape"):
            await asyncio.wait_for(
                rollouts._run_multi_turn_rollout_async(
                    MagicMock(),
                    batch,
                    native_rollout.tokenizer,
                    {},
                    32,
                ),
                timeout=1,
            )
        assert drained.is_set()
        assert len(asyncio.all_tasks()) == 1

    asyncio.run(run())
    native_rollout.reward.assert_not_called()
