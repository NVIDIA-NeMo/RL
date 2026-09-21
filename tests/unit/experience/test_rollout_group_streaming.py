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
"""Streaming prompt groups from the native multi-turn rollout collector."""

import asyncio

import pytest
import torch

import nemo_rl.experience.rollouts as rollouts_mod
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollouts import run_async_multi_turn_rollout_groups


def _group_test_batch() -> BatchedDataDict:
    return BatchedDataDict(
        {
            "message_log": [
                [{"role": "user", "content": f"prompt-{i}"}] for i in range(6)
            ],
            "idx": [100, 101, 102, 103, 104, 105],
        }
    )


def _fake_group_rollout_factory(release_events, calls):
    """Fake ``_run_multi_turn_rollout_async`` gated on per-group events.

    ``release_events`` maps the first ``idx`` of a slice to an ``asyncio.Event``;
    a group finishes only after its event is set, which makes completion order
    deterministic. The fake mirrors the real path in recording each sample's
    position within the slice it was handed as ``idx``.
    """

    async def fake_rollout(policy_generation, input_batch, **kwargs):
        first_idx = int(input_batch["idx"][0])
        calls.append(first_idx)
        event = release_events.get(first_idx)
        if event is not None:
            await event.wait()
        size = input_batch.size
        final_batch = BatchedDataDict(
            {
                "message_log": list(input_batch["message_log"]),
                "idx": list(range(size)),
                "total_reward": torch.arange(size, dtype=torch.float32),
            }
        )
        sample_metrics = [
            {
                "turn_count": 1,
                "total_tokens": 10,
                "assistant_tokens": 2,
                "env_tokens": 8,
                "terminated": True,
                "truncated": False,
                "max_turns_reached": False,
                "total_reward": float(i),
                "turn_gen_tokens": [2],
                "turn_input_tokens": [10],
                "turn_total_tokens": [12],
                "max_gen_tokens_per_turn": 2,
                "per_worker_token_counts": {"worker-0": 1},
            }
            for i in range(size)
        ]
        return final_batch, sample_metrics

    return fake_rollout


def _collect_groups(**overrides):
    kwargs = dict(
        policy_generation=None,
        input_batch=_group_test_batch(),
        tokenizer=None,
        task_to_env={},
        max_seq_len=128,
        num_generations=2,
    )
    kwargs.update(overrides)

    async def collect():
        return [group async for group in run_async_multi_turn_rollout_groups(**kwargs)]

    return asyncio.run(collect())


def test_stream_groups_yields_in_completion_order(monkeypatch):
    """Each prompt group runs on its own slice and is yielded as soon as it finishes."""
    calls: list[int] = []
    groups_seen: list = []

    async def run():
        # Release group 2 (idx 104) first, then group 1, then group 0.
        events = {100: asyncio.Event(), 102: asyncio.Event(), 104: asyncio.Event()}
        monkeypatch.setattr(
            rollouts_mod,
            "_run_multi_turn_rollout_async",
            _fake_group_rollout_factory(events, calls),
        )

        async def release():
            for first_idx in (104, 102, 100):
                await asyncio.sleep(0)
                events[first_idx].set()
                await asyncio.sleep(0)

        releaser = asyncio.ensure_future(release())
        async for group in run_async_multi_turn_rollout_groups(
            policy_generation=None,
            input_batch=_group_test_batch(),
            tokenizer=None,
            task_to_env={},
            max_seq_len=128,
            num_generations=2,
            stream_groups=True,
        ):
            groups_seen.append(group)
        await releaser

    asyncio.run(run())

    assert sorted(calls) == [100, 102, 104]
    assert [group.group_index for group in groups_seen] == [2, 1, 0]
    assert [group.final_batch.size for group in groups_seen] == [2, 2, 2]
    # idx keeps its whole-batch meaning: the position within the input batch.
    assert [group.final_batch["idx"] for group in groups_seen] == [
        [4, 5],
        [2, 3],
        [0, 1],
    ]
    assert all(
        group.rollout_metrics["avg_turns_per_sample"] == 1 for group in groups_seen
    )


def test_stream_groups_propagates_failure_and_cancels_the_rest(monkeypatch):
    """A failing group re-raises and cancels the groups still in flight."""
    started: list[int] = []
    cancelled: list[int] = []

    async def fake_rollout(policy_generation, input_batch, **kwargs):
        first_idx = int(input_batch["idx"][0])
        started.append(first_idx)
        if first_idx == 102:
            await asyncio.sleep(0.01)
            raise RuntimeError("boom")
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(first_idx)
            raise
        return None, []  # pragma: no cover

    monkeypatch.setattr(rollouts_mod, "_run_multi_turn_rollout_async", fake_rollout)

    with pytest.raises(RuntimeError, match="boom"):
        _collect_groups(stream_groups=True)
    assert sorted(started) == [100, 102, 104]
    assert sorted(cancelled) == [100, 104]


def test_stream_groups_retrieves_every_failed_group(monkeypatch):
    """Two groups failing: the first raised is reported, the other is awaited, not leaked."""
    import warnings

    async def fake_rollout(policy_generation, input_batch, **kwargs):
        first_idx = int(input_batch["idx"][0])
        if first_idx in (100, 102):
            raise RuntimeError(f"boom-{first_idx}")
        await asyncio.sleep(10)
        return None, []  # pragma: no cover

    monkeypatch.setattr(rollouts_mod, "_run_multi_turn_rollout_async", fake_rollout)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="boom-"):
            _collect_groups(stream_groups=True)
    assert not [w for w in caught if "never retrieved" in str(w.message)]


def test_stream_groups_single_group_uses_whole_batch_path(monkeypatch):
    """One group means there is nothing to stream; the barrier path runs once."""
    calls: list[int] = []
    monkeypatch.setattr(
        rollouts_mod,
        "_run_multi_turn_rollout_async",
        _fake_group_rollout_factory({}, calls),
    )

    groups = _collect_groups(stream_groups=True, num_generations=6)

    assert calls == [100]
    assert [group.group_index for group in groups] == [0]
    assert groups[0].final_batch.size == 6


def test_default_keeps_whole_batch_barrier_and_input_order(monkeypatch):
    """Without stream_groups the collector semantics are unchanged."""
    calls: list[int] = []
    monkeypatch.setattr(
        rollouts_mod,
        "_run_multi_turn_rollout_async",
        _fake_group_rollout_factory({}, calls),
    )

    groups = _collect_groups()

    assert calls == [100]  # one whole-batch call
    assert [group.group_index for group in groups] == [0, 1, 2]
    assert [group.final_batch["idx"] for group in groups] == [
        [0, 1],
        [2, 3],
        [4, 5],
    ]
