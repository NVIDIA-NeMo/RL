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

"""CUA baseline and FD update tests against explicit reference calculations."""

import copy
import pickle
from threading import Event

import pytest
import torch


def test_task_sampler_resume_preserves_oversampling_and_epoch_order():
    from logra.osworld import TaskSampler

    sampler = TaskSampler(7, 42)
    first = sampler.next(5)
    saved = copy.deepcopy(sampler.state_dict())
    rest = sampler.next(16)
    restored = TaskSampler(7, 999)
    restored.load_state_dict(saved)
    assert restored.next(16) == rest
    combined = first + rest
    for start in range(0, 21, 7):
        assert sorted(combined[start : start + 7]) == list(range(7))
    with pytest.raises(ValueError, match="dataset size"):
        TaskSampler(8, 42).load_state_dict(saved)


def test_rolling_pool_does_not_wait_for_a_straggler_or_lose_it_on_resume() -> None:
    from logra.osworld import (
        CuaSegment,
        DesktopTask,
        Episode,
        RollingTaskPool,
        RolloutTicket,
        TaskSampler,
    )

    release = Event()

    def generate(ticket: RolloutTicket) -> Episode:
        if ticket.serial == 0:
            assert release.wait(5)
        task = DesktopTask(str(ticket.task_index), "prompt", "test", {})
        segment = CuaSegment(
            torch.ones(2, dtype=torch.long),
            torch.tensor([False, True]),
            torch.zeros(2),
            torch.zeros(2, 1, 1, dtype=torch.long),
            [],
        )
        return Episode(task, "v1", 0.0, True, None, [segment])

    pool = RollingTaskPool(
        generate,
        TaskSampler(10, 43),
        batch_size=2,
        inflight=3,
        seed=43,
        start_step=0,
        resume=None,
    )
    try:
        first = pool.next_batch(step=0, prefetch_next=True).episodes
        consumed = {ticket.serial for ticket, _ in first}
        assert 0 not in consumed
        saved = pool.state_dict()
        pending = {ticket.serial for ticket in saved.pending}
        assert 0 in pending
        assert not consumed & pending
        assert consumed | pending == set(range(saved.next_serial))
    finally:
        release.set()
        pool.close()

    restored = RollingTaskPool(
        generate,
        TaskSampler(10, 43),
        batch_size=2,
        inflight=3,
        seed=43,
        start_step=1,
        resume=saved,
    )
    try:
        following = restored.next_batch(step=1, prefetch_next=False).episodes
        returned = {ticket.serial for ticket, _ in following}
        still_pending = {ticket.serial for ticket in restored.state_dict().pending}
        assert 0 in returned | still_pending
        assert not consumed & returned
        original = {ticket.serial: ticket for ticket in saved.pending}
        for ticket, _ in following:
            if ticket.serial in original:
                assert ticket == original[ticket.serial]
    finally:
        restored.close()


def test_rolling_pool_surfaces_episode_errors() -> None:
    from logra.osworld import (
        CuaSegment,
        DesktopTask,
        Episode,
        RollingTaskPool,
        RolloutTicket,
        TaskSampler,
    )

    def generate(ticket: RolloutTicket) -> Episode:
        if ticket.serial == 1:
            raise RuntimeError("capture failed")
        segment = CuaSegment(
            torch.ones(2, dtype=torch.long),
            torch.tensor([False, True]),
            torch.zeros(2),
            torch.zeros(2, 1, 1, dtype=torch.long),
            [],
        )
        return Episode(
            DesktopTask("a", "a", "test", {}), "v1", 0.0, True, None, [segment]
        )

    pool = RollingTaskPool(
        generate,
        TaskSampler(10, 43),
        batch_size=2,
        inflight=2,
        seed=43,
        start_step=0,
        resume=None,
    )
    try:
        with pytest.raises(RuntimeError, match="capture failed"):
            pool.next_batch(step=0, prefetch_next=True)
    finally:
        pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.next_batch(step=1, prefetch_next=False)


def test_rolling_pool_replaces_empty_episodes_without_dropping_failed_actions() -> None:
    from logra.osworld import (
        CuaSegment,
        DesktopTask,
        Episode,
        RollingTaskPool,
        RolloutTicket,
        TaskSampler,
    )

    def generate(ticket: RolloutTicket) -> Episode:
        segment = CuaSegment(
            torch.ones(2, dtype=torch.long),
            torch.tensor([False, ticket.serial != 0]),
            torch.zeros(2),
            torch.zeros(2, 1, 1, dtype=torch.long),
            [],
        )
        # A sandbox error after generating actions must retain the original
        # masked-batch semantics; a zero policy reward is also not a drop rule.
        return Episode(
            DesktopTask(str(ticket.serial), "prompt", "test", {}),
            "v1",
            0.0,
            ticket.serial != 1,
            "runtime failure" if ticket.serial == 1 else None,
            [segment],
        )

    pool = RollingTaskPool(
        generate,
        TaskSampler(10, 43),
        batch_size=1,
        inflight=1,
        seed=43,
        start_step=0,
        resume=None,
    )
    try:
        batch = pool.next_batch(step=0, prefetch_next=False)
        assert batch.dropped_no_action == 1
        assert batch.episodes[0][0].serial == 1
        assert not batch.episodes[0][1].valid
        following = pool.next_batch(step=1, prefetch_next=False)
        assert following.dropped_no_action == 0
        assert following.episodes[0][0].serial == 2
        assert following.episodes[0][1].valid
        assert following.episodes[0][1].score == 0
        assert not {0, 1, 2} & {ticket.serial for ticket in pool.state_dict().pending}
    finally:
        pool.close()


def test_legacy_pool_checkpoint_imports_remain_loadable() -> None:
    """Old torch/pickle checkpoints resolve their original logra.osworld names."""
    from logra.osworld import RolloutPoolState as LegacyState
    from logra.osworld import RolloutTicket as LegacyTicket
    from logra.rollout_pool import RolloutPoolState, RolloutTicket

    assert LegacyState is RolloutPoolState
    assert LegacyTicket is RolloutTicket
    state = RolloutPoolState({}, [RolloutTicket(3, 0, 46, 1)], 4)
    serialized = pickle.dumps(state, protocol=0)
    assert b"clogra.rollout_pool\n" in serialized
    legacy = serialized.replace(b"clogra.rollout_pool\n", b"clogra.osworld\n")
    assert pickle.loads(legacy) == state
