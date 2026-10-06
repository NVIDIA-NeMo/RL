"""Gym prompt identity through real rollout, replay, and recycle/resume code.

Only Gym transport and remote storage are replaced; no GPU or external Gym service is needed.
"""

import asyncio
import copy
from collections import deque
from typing import Any
from unittest.mock import AsyncMock

import pytest
import torch

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.single_controller_utils.config import RolloutRecoveryConfig
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.payload import record_to_train_batch
from nemo_rl.experience.rollout_manager import RolloutManager, RolloutOutcome
from tests.unit.single_controller.test_queue_recycle import (
    buffer_and_sampler,
    restore,
    select,
    snapshot,
)
from tests.unit.single_controller.test_single_controller_actor import (
    _train_pump_controller,
)


def prompt(idx: int, task_name: str | None) -> DatumSpec:
    sample: DatumSpec = {
        "idx": idx,
        "length": 0,
        "loss_multiplier": 1.0,
        "message_log": [],
        "extra_env_info": {
            "task_source": task_name or "source-without-task-name",
            "agent_ref": {"name": "grading-agent"},
            "responses_create_params": {
                "input": [{"role": "user", "content": f"question-{idx}"}]
            },
        },
    }
    if task_name is not None:
        sample["task_name"] = task_name
    return sample


def gym_manager(
    buffer: TQReplayBuffer, monkeypatch: pytest.MonkeyPatch
) -> tuple[RolloutManager, list[list[dict[str, Any]]]]:
    gym_env = object()
    manager = RolloutManager(
        tokenizer=None,
        task_to_env={"nemo_gym": gym_env},
        num_generations_per_prompt=2,
        max_seq_len=100,
        rollout_recovery_config=RolloutRecoveryConfig(),
        generation_config={
            "temperature": 1.0,
            "top_p": 1.0,
            "max_new_tokens": 90,
            "stop_strings": None,
            "stop_token_ids": None,
            "top_k": None,
        },
        use_nemo_gym=True,
        tq_buffer=buffer,
    )
    manager.set_data_plane_checkpoint_barrier(buffer.data_plane_checkpoint_barrier)
    calls: list[list[dict[str, Any]]] = []

    async def stream_rows(
        env: object,
        inputs: list[dict[str, Any]],
        results: list[dict[str, Any] | None],
        shaping: list[Any],
        total_rows: int,
        timer_prefix: str,
        **kwargs: Any,
    ) -> dict[str, float]:
        # A source-specific task label must still route through the Gym actor.
        assert env is gym_env
        calls.append(copy.deepcopy(inputs))
        for row in inputs:
            user = {
                "role": "user",
                "content": row["responses_create_params"]["input"][0]["content"],
                "token_ids": [10, 11],
            }
            assistant = {
                "role": "assistant",
                "content": "answer",
                "token_ids": [20, 21],
                "generation_logprobs": [-0.1, -0.2],
            }
            results[row["_rowidx"]] = {
                "input_message_log": [copy.deepcopy(user)],
                "message_log": [user, assistant],
                "full_result": {"reward": float(row["_rowidx"])},
            }
        return {}

    monkeypatch.setattr(manager._impl, "_stream_rows", stream_rows)
    return manager, calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "task_name",
    ["math/dapo-math-p10-90", "code/competitive-coding", "stem/knowledge-mcqa", None],
)
async def test_gym_preserves_source_identity_without_changing_training_payload(
    monkeypatch: pytest.MonkeyPatch, task_name: str | None
) -> None:
    buffer, _, _ = buffer_and_sampler(capacity=3)
    manager, calls = gym_manager(buffer, monkeypatch)
    source = prompt(7, task_name)
    original = copy.deepcopy(source)
    record = await manager.run_rollout(source)
    assert record.prompt_idx == 7
    assert record.metadata == {"task_name": task_name}
    assert record.metadata["task_name"] is source.get("task_name")
    assert source == original
    assert record.extra_env_info == source["extra_env_info"]
    assert record.extra_env_info is not source["extra_env_info"]
    assert len(calls) == 1
    assert record.rollout_metrics["grading-agent/reward/mean"] == 0.5

    legacy = copy.deepcopy(record)
    legacy.metadata = {"task_name": "nemo_gym"}
    kwargs = {
        "pad_value_dict": {"token_ids": 0, "input_ids": 0},
        "include_message_violation_fields": False,
    }
    actual = record_to_train_batch(record, **kwargs)
    expected = record_to_train_batch(legacy, **kwargs)
    assert actual.keys() == expected.keys()
    for key in actual:
        if isinstance(actual[key], torch.Tensor):
            torch.testing.assert_close(actual[key], expected[key])
        else:
            assert actual[key] == expected[key]


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoint_at", ["none", "ready", "recycled_prefetch"])
async def test_gym_recycle_and_resume_preserve_prompt_and_prefetched_batch(
    monkeypatch: pytest.MonkeyPatch, checkpoint_at: str
) -> None:
    buffer, sampler, _ = buffer_and_sampler(capacity=3)
    manager, _ = gym_manager(buffer, monkeypatch)
    sources = [
        prompt(0, "code/competitive-coding"),
        prompt(1, "stem/knowledge-mcqa"),
        prompt(2, "math/dapo-math-p10-90"),
    ]
    for sample, version in zip(sources[:2], [9, 10], strict=True):
        manager.set_weight_version(version)
        assert await manager.generate_and_push(sample) is RolloutOutcome.COMMITTED
    old_group, prefetched_group = buffer.group_ids
    if checkpoint_at == "ready":
        buffer, sampler, _ = await restore(await snapshot(buffer))

    # Gap 8 is recycled; gap 7 enters the batch trained after the next update.
    sampler.start_prefetch(dequeue_version=17, num_groups=1)
    await asyncio.wait_for(sampler.finish_prefetch(), timeout=2)
    if checkpoint_at == "recycled_prefetch":
        buffer, sampler, _ = await restore(await snapshot(buffer))
    recycled = buffer.peek_recycled_prompt()
    assert recycled.group_id == old_group
    assert recycled.prompt_ref.sample_id == "0"
    assert recycled.prompt_ref.task_name == sources[0]["task_name"]
    selected, count = await select(sampler, 18)
    assert count == 1
    assert selected.sample_ids == [f"{prefetched_group}_g0", f"{prefetched_group}_g1"]

    manager, calls = gym_manager(buffer, monkeypatch)
    manager.set_weight_version(18)
    ctrl = _train_pump_controller(sampler=sampler)
    ctrl._buffer = buffer
    ctrl._rollout_manager = manager
    ctrl._data_plane_checkpoint_barrier = buffer.data_plane_checkpoint_barrier
    ctrl._async_cfg.max_inflight_prompts = 1
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._inflight_rollouts = 0
    ctrl._dispatched_rollouts = set()
    ctrl._inflight_by_group_id = {}
    ctrl._current_epoch = 0
    ctrl._rollout_completion_durations_s = deque()
    ctrl._rollout_queue_wait_durations_s = deque()

    class Loader:
        dataset = sources

        def __iter__(self):
            yield BatchedDataDict({key: [value] for key, value in sources[2].items()})

    ctrl._dataloader = Loader()
    finished = asyncio.Event()
    generate = manager.generate_and_push

    async def generate_then_pause(*args: Any, **kwargs: Any) -> RolloutOutcome:
        outcome = await generate(*args, **kwargs)
        if len(calls) == 2:
            ctrl._rollout_permitted.clear()
            finished.set()
        return outcome

    monkeypatch.setattr(
        manager, "generate_and_push", AsyncMock(side_effect=generate_then_pause)
    )
    task = asyncio.create_task(ctrl._rollout_pump())
    try:
        await asyncio.wait_for(finished.wait(), timeout=3)
        assert [
            rows[0]["responses_create_params"]["input"][0]["content"] for rows in calls
        ] == ["question-0", "question-2"]
        assert buffer.peek_recycled_prompt() is None
        assert len(manager.recovery_ledger) == 0
        assert old_group not in buffer.group_ids
        ready = buffer.ready_queue_indices()
        assert len(ready) == 2
        assert [buffer.meta_list[i].tags[0]["rollout_category"] for i in ready] == [
            sources[0]["task_name"],
            sources[2]["task_name"],
        ]
        assert all(buffer.start_weight_list[i] == 18 for i in ready)
        assert buffer.training_owned_group_ids() == {prefetched_group}
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
