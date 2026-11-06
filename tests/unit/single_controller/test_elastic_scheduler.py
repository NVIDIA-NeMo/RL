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

"""Elastic scheduling policy: launches, cell transitions and burst-end retention."""

import heapq
import random
from collections import deque
from dataclasses import dataclass, field

import pytest

from nemo_rl.algorithms.single_controller_utils.elastic_scheduler import (
    CellRole,
    ElasticScheduler,
    ElasticSchedulerPolicy,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _policy(
    *,
    num_cells: int = 4,
    cell_capacity: int = 10,
    standalone_capacity: int = 0,
    batch_groups: int = 4,
    min_supply_fraction: float = 0.5,
    supply_hold_s: float = 5.0,
    burst_dispatch: int = 16,
) -> ElasticSchedulerPolicy:
    return ElasticSchedulerPolicy(
        num_cells=num_cells,
        cell_capacity=cell_capacity,
        standalone_capacity=standalone_capacity,
        batch_groups=batch_groups,
        min_supply_fraction=min_supply_fraction,
        supply_hold_s=supply_hold_s,
        burst_dispatch=burst_dispatch,
    )


def _scheduler(**policy_kwargs) -> tuple[ElasticScheduler, _Clock]:
    clock = _Clock()
    return ElasticScheduler(policy=_policy(**policy_kwargs), clock=clock), clock


def _run(scheduler: ElasticScheduler, *, dispatched: int, finished: int) -> None:
    """Dispatch and launch ``dispatched`` executions, then finish ``finished`` of them."""
    scheduler.record_dispatch(dispatched)
    scheduler.record_launch(scheduler.launchable())
    scheduler.record_finish(finished)


def _hold_supply(
    scheduler: ElasticScheduler,
    clock: _Clock,
    *,
    ready_groups: int = 4,
    groups_needed: int = 4,
):
    """Observe the supply, wait out tau, and return the resulting decision."""
    scheduler.next_training_cell(ready_groups=ready_groups, groups_needed=groups_needed)
    clock.now += scheduler.policy.supply_hold_s
    return scheduler.next_training_cell(
        ready_groups=ready_groups, groups_needed=groups_needed
    )


def _move_to_training(scheduler: ElasticScheduler, cell: int) -> None:
    scheduler.begin_training_transition(cell)
    scheduler.finish_training_transition(cell)


class TestPolicyValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"num_cells": 0},
            {"cell_capacity": 0},
            {"standalone_capacity": -1},
            {"batch_groups": 0},
            {"min_supply_fraction": 0.0},
            {"min_supply_fraction": 1.5},
            {"supply_hold_s": -1.0},
            {"burst_dispatch": -1},
        ],
    )
    def test_rejects_out_of_range_parameters(self, overrides):
        with pytest.raises(ValueError):
            _policy(**overrides)

    def test_accepts_full_supply_fraction_and_zero_hold(self):
        _policy(min_supply_fraction=1.0, supply_hold_s=0.0)


class TestLaunches:
    def test_capacity_counts_standalone_and_rollout_cells(self):
        scheduler, _ = _scheduler(num_cells=3, cell_capacity=10, standalone_capacity=5)
        assert scheduler.rollout_capacity == 35

    def test_launches_every_queued_execution_while_capacity_allows(self):
        scheduler, _ = _scheduler(num_cells=4, cell_capacity=10)
        scheduler.record_dispatch(25)
        assert scheduler.launchable() == 25

    def test_launches_are_capped_by_free_capacity(self):
        """Eq. 4: the excess stays queued."""
        scheduler, _ = _scheduler(num_cells=4, cell_capacity=10)
        scheduler.record_dispatch(55)
        assert scheduler.launchable() == 40
        scheduler.record_launch(40)
        assert scheduler.launchable() == 0
        scheduler.record_finish(3)
        assert scheduler.launchable() == 3
        assert scheduler.queued == 15

    def test_a_cell_leaving_rollout_takes_its_capacity_at_once(self):
        scheduler, clock = _scheduler(num_cells=4, cell_capacity=10)
        _run(scheduler, dispatched=40, finished=12)
        scheduler.open_batch()
        assert _hold_supply(scheduler, clock) == 0
        scheduler.begin_training_transition(0)
        assert scheduler.rollout_capacity == 30
        scheduler.record_dispatch(5)
        assert scheduler.launchable() == 2

    def test_launching_past_capacity_is_refused(self):
        scheduler, _ = _scheduler(num_cells=1, cell_capacity=4)
        scheduler.record_dispatch(6)
        with pytest.raises(ValueError, match="cannot launch 5"):
            scheduler.record_launch(5)

    def test_counters_keep_f_le_p_le_d(self):
        scheduler, _ = _scheduler()
        scheduler.record_dispatch(2)
        with pytest.raises(ValueError):
            scheduler.record_launch(3)
        scheduler.record_launch(2)
        with pytest.raises(ValueError):
            scheduler.record_finish(3)
        with pytest.raises(ValueError):
            scheduler.record_dispatch(-1)


class TestTransitionToTraining:
    def test_waterlevel_must_fit_in_the_capacity_left_behind(self):
        """Eq. 5, at its boundary: w == C_R - C_e is enough, one more is not."""
        scheduler, clock = _scheduler(num_cells=4, cell_capacity=10)
        _run(scheduler, dispatched=40, finished=9)
        scheduler.open_batch()
        assert scheduler.waterlevel == 31
        assert _hold_supply(scheduler, clock) is None
        scheduler.record_finish(1)
        assert scheduler.waterlevel_admits_next_cell()
        assert scheduler.next_training_cell(ready_groups=4, groups_needed=4) == 0

    def test_queued_executions_count_toward_the_waterlevel(self):
        """w = d - f includes executions that have not launched yet."""
        scheduler, clock = _scheduler(num_cells=4, cell_capacity=10)
        scheduler.record_dispatch(50)
        scheduler.record_launch(40)
        scheduler.record_finish(15)
        scheduler.open_batch()
        assert scheduler.waterlevel == 35
        assert _hold_supply(scheduler, clock) is None

    def test_single_cell_without_standalone_waits_for_every_execution(self):
        """K=1 and C_s=0 degenerates to Colocate: the pool switches only when empty."""
        scheduler, clock = _scheduler(num_cells=1, cell_capacity=10, burst_dispatch=0)
        _run(scheduler, dispatched=10, finished=9)
        scheduler.open_batch()
        assert _hold_supply(scheduler, clock) is None
        scheduler.record_finish(1)
        assert scheduler.next_training_cell(ready_groups=4, groups_needed=4) == 0

    def test_supply_must_cover_the_fraction_for_the_full_hold(self):
        scheduler, clock = _scheduler(min_supply_fraction=0.5, supply_hold_s=5.0)
        _run(scheduler, dispatched=10, finished=10)
        scheduler.open_batch()
        assert scheduler.next_training_cell(ready_groups=2, groups_needed=4) is None
        clock.now = 4.9
        assert scheduler.next_training_cell(ready_groups=2, groups_needed=4) is None
        assert scheduler.supply_hold_deadline() == 5.0
        clock.now = 5.0
        assert scheduler.next_training_cell(ready_groups=2, groups_needed=4) == 0

    def test_a_dip_in_supply_restarts_the_hold(self):
        scheduler, clock = _scheduler(min_supply_fraction=0.5, supply_hold_s=5.0)
        _run(scheduler, dispatched=10, finished=10)
        scheduler.open_batch()
        scheduler.next_training_cell(ready_groups=2, groups_needed=4)
        clock.now = 3.0
        scheduler.next_training_cell(ready_groups=1, groups_needed=4)
        assert scheduler.supply_hold_deadline() is None
        clock.now = 4.0
        scheduler.next_training_cell(ready_groups=2, groups_needed=4)
        clock.now = 8.9
        assert scheduler.next_training_cell(ready_groups=2, groups_needed=4) is None
        clock.now = 9.0
        assert scheduler.next_training_cell(ready_groups=2, groups_needed=4) == 0

    def test_supply_is_capped_by_what_the_batch_still_needs(self):
        """Plenty of ready groups do not justify a cell for the batch's last group."""
        scheduler, clock = _scheduler(batch_groups=4, min_supply_fraction=0.5)
        _run(scheduler, dispatched=10, finished=10)
        scheduler.open_batch()
        assert _hold_supply(scheduler, clock, ready_groups=9, groups_needed=1) is None

    def test_fractional_threshold_rounds_up(self):
        scheduler, clock = _scheduler(batch_groups=3, min_supply_fraction=0.5)
        _run(scheduler, dispatched=10, finished=10)
        scheduler.open_batch()
        assert _hold_supply(scheduler, clock, ready_groups=1, groups_needed=3) is None
        assert _hold_supply(scheduler, clock, ready_groups=2, groups_needed=3) == 0

    def test_no_transition_while_the_joining_window_is_closed(self):
        scheduler, clock = _scheduler()
        _run(scheduler, dispatched=10, finished=10)
        assert _hold_supply(scheduler, clock) is None
        scheduler.open_batch()
        assert _hold_supply(scheduler, clock) == 0

    def test_cells_move_serially_core_first(self):
        scheduler, clock = _scheduler(num_cells=3, cell_capacity=10)
        _run(scheduler, dispatched=5, finished=5)
        scheduler.open_batch()
        assert _hold_supply(scheduler, clock) == 0
        scheduler.begin_training_transition(0)
        clock.now += 100
        assert scheduler.next_training_cell(ready_groups=4, groups_needed=4) is None
        with pytest.raises(RuntimeError, match="serial"):
            scheduler.begin_training_transition(1)
        scheduler.finish_training_transition(0)
        assert _hold_supply(scheduler, clock) == 1
        with pytest.raises(RuntimeError, match="next cell in order is 1"):
            scheduler.begin_training_transition(2)
        _move_to_training(scheduler, 1)
        assert _hold_supply(scheduler, clock) == 2
        _move_to_training(scheduler, 2)
        assert _hold_supply(scheduler, clock) is None
        assert scheduler.roles == (CellRole.TRAINING,) * 3

    def test_finishing_a_transition_restarts_the_hold(self):
        scheduler, clock = _scheduler(num_cells=3, supply_hold_s=5.0)
        _run(scheduler, dispatched=5, finished=5)
        scheduler.open_batch()
        assert _hold_supply(scheduler, clock) == 0
        _move_to_training(scheduler, 0)
        assert scheduler.supply_hold_deadline() is None
        assert scheduler.next_training_cell(ready_groups=4, groups_needed=4) is None
        clock.now += 5.0
        assert scheduler.next_training_cell(ready_groups=4, groups_needed=4) == 1

    def test_opening_a_batch_restarts_the_hold(self):
        scheduler, clock = _scheduler(supply_hold_s=5.0)
        _run(scheduler, dispatched=5, finished=5)
        scheduler.open_batch()
        scheduler.next_training_cell(ready_groups=4, groups_needed=4)
        scheduler.close_batch()
        clock.now = 10.0
        scheduler.open_batch()
        assert scheduler.next_training_cell(ready_groups=4, groups_needed=4) is None
        clock.now = 15.0
        assert scheduler.next_training_cell(ready_groups=4, groups_needed=4) == 0

    def test_forced_transition_past_the_waterlevel_is_refused(self):
        scheduler, _ = _scheduler(num_cells=2, cell_capacity=10)
        _run(scheduler, dispatched=15, finished=0)
        with pytest.raises(RuntimeError, match="waterlevel"):
            scheduler.begin_training_transition(0)


class TestBurstEnd:
    def _all_training(
        self, *, standalone_capacity: int = 0, burst_dispatch: int = 16
    ) -> tuple[ElasticScheduler, _Clock]:
        scheduler, clock = _scheduler(
            num_cells=4,
            cell_capacity=10,
            standalone_capacity=standalone_capacity,
            burst_dispatch=burst_dispatch,
        )
        _run(scheduler, dispatched=20, finished=20)
        scheduler.open_batch()
        for cell in range(4):
            assert _hold_supply(scheduler, clock) == cell
            _move_to_training(scheduler, cell)
        scheduler.close_batch()
        return scheduler, clock

    def test_every_satellite_returns_in_descending_order(self):
        scheduler, _ = self._all_training()
        plan = scheduler.plan_burst_end(ready_groups=4)
        assert plan.core_stays_in_training
        assert plan.return_to_rollout == (3, 2, 1)

    def test_core_stays_when_capacity_covers_outstanding_and_pending_work(self):
        """Eq. 6 at its boundary: w + mu N B == C_s + (K - 1) C_e."""
        scheduler, _ = self._all_training(standalone_capacity=4, burst_dispatch=20)
        scheduler.record_dispatch(14)
        assert scheduler.plan_burst_end(ready_groups=4).core_stays_in_training
        scheduler.record_dispatch(1)
        plan = scheduler.plan_burst_end(ready_groups=4)
        assert not plan.core_stays_in_training
        assert plan.return_to_rollout == (3, 2, 1, 0)

    def test_core_returns_without_supply_for_the_next_batch(self):
        scheduler, _ = self._all_training()
        assert not scheduler.plan_burst_end(ready_groups=1).core_stays_in_training
        assert scheduler.plan_burst_end(ready_groups=2).core_stays_in_training

    def test_single_cell_always_returns_its_core(self):
        scheduler, clock = _scheduler(num_cells=1, cell_capacity=10, burst_dispatch=4)
        scheduler.open_batch()
        assert _hold_supply(scheduler, clock) == 0
        _move_to_training(scheduler, 0)
        scheduler.close_batch()
        plan = scheduler.plan_burst_end(ready_groups=4)
        assert plan == type(plan)(return_to_rollout=(0,), core_stays_in_training=False)

    def test_returned_cells_add_capacity_only_once_serving(self):
        scheduler, _ = self._all_training()
        scheduler.begin_rollout_transition(3)
        assert scheduler.rollout_capacity == 0
        scheduler.finish_rollout_transition(3)
        assert scheduler.rollout_capacity == 10

    def test_retained_core_restarts_the_next_burst_with_the_first_satellite(self):
        scheduler, clock = self._all_training()
        plan = scheduler.plan_burst_end(ready_groups=4)
        for cell in plan.return_to_rollout:
            scheduler.begin_rollout_transition(cell)
            scheduler.finish_rollout_transition(cell)
        scheduler.record_dispatch(16)
        scheduler.open_batch()
        assert scheduler.roles[0] is CellRole.TRAINING
        scheduler.record_launch(scheduler.launchable())
        scheduler.record_finish(6)
        assert _hold_supply(scheduler, clock) == 1

    def test_requires_a_closed_batch_and_settled_transitions(self):
        scheduler, clock = _scheduler(num_cells=2)
        scheduler.open_batch()
        assert _hold_supply(scheduler, clock) == 0
        scheduler.begin_training_transition(0)
        with pytest.raises(RuntimeError, match="close the batch"):
            scheduler.plan_burst_end(ready_groups=4)
        scheduler.close_batch()
        with pytest.raises(RuntimeError, match="mid-transition"):
            scheduler.plan_burst_end(ready_groups=4)
        scheduler.finish_training_transition(0)
        scheduler.plan_burst_end(ready_groups=4)

    def test_cells_cannot_leave_training_while_the_batch_is_open(self):
        scheduler, clock = _scheduler(num_cells=2)
        scheduler.open_batch()
        _hold_supply(scheduler, clock)
        _move_to_training(scheduler, 0)
        with pytest.raises(RuntimeError, match="after the batch closes"):
            scheduler.begin_rollout_transition(0)


# ── discrete-event simulation ────────────────────────────────────────────


@dataclass(frozen=True)
class _Workload:
    num_cells: int
    cell_capacity: int
    group_size: int
    batch_groups: int
    initial_groups: int
    steps: int
    # Seconds one cell needs to train one group; a cell's throughput scales with its size.
    train_s_per_group: float
    to_training_s: float = 8.0
    to_rollout_s: float = 4.0
    seed: int = 0


@dataclass
class _SimResult:
    finish_time: float = 0.0
    # Per batch: (cell, time its transition to training finished, f at that time).
    joins: list[list[tuple[int, float, int]]] = field(default_factory=list)
    # Largest excess of running executions over rollout capacity ever observed.
    max_running_over_capacity: int = 0


def _simulate(workload: _Workload) -> _SimResult:
    """Drive the scheduler with one-step bursts over a long-tailed execution workload.

    Groups become ready when all their executions finish. Training cells claim ready
    groups one at a time, standing in for the micro-step stream, and a step ends once
    ``batch_groups`` groups have trained. Execution durations are Pareto-distributed
    and capped like a timeout.
    """
    rng = random.Random(workload.seed)
    clock = _Clock()
    burst_executions = workload.batch_groups * workload.group_size
    scheduler = ElasticScheduler(
        policy=ElasticSchedulerPolicy(
            num_cells=workload.num_cells,
            cell_capacity=workload.cell_capacity,
            standalone_capacity=0,
            batch_groups=workload.batch_groups,
            min_supply_fraction=0.5,
            supply_hold_s=5.0,
            burst_dispatch=burst_executions,
        ),
        clock=clock,
    )
    events: list[tuple[float, int, str, int]] = []
    seq = 0

    def push(at: float, kind: str, payload: int = -1) -> None:
        nonlocal seq
        heapq.heappush(events, (at, seq, kind, payload))
        seq += 1

    queue: deque[int] = deque()
    group_of: dict[int, int] = {}
    unfinished_in_group: dict[int, int] = {}
    ready: deque[int] = deque()
    next_group = 0
    next_execution = 0

    def dispatch_groups(count: int) -> None:
        nonlocal next_group, next_execution
        for _ in range(count):
            unfinished_in_group[next_group] = workload.group_size
            for _ in range(workload.group_size):
                group_of[next_execution] = next_group
                queue.append(next_execution)
                next_execution += 1
            next_group += 1
        scheduler.record_dispatch(count * workload.group_size)

    result = _SimResult()
    steps_done = 0
    claimed = trained = 0
    idle_trainers: set[int] = set()
    returning: set[int] = set()
    burst_ending = False

    def open_next_batch() -> None:
        dispatch_groups(workload.batch_groups)
        scheduler.open_batch()
        result.joins.append([])

    def end_burst(now: float) -> None:
        nonlocal burst_ending
        if CellRole.TO_TRAINING in scheduler.roles:
            return  # A late joiner finishes its transition first.
        burst_ending = False
        plan = scheduler.plan_burst_end(ready_groups=len(ready))
        for cell in plan.return_to_rollout:
            scheduler.begin_rollout_transition(cell)
            idle_trainers.discard(cell)
            returning.add(cell)
            push(now + workload.to_rollout_s, "returned", cell)

    def settle(now: float) -> None:
        nonlocal claimed
        launch = scheduler.launchable()
        for _ in range(launch):
            execution = queue.popleft()
            duration = min(60.0 * rng.paretovariate(1.2), 1800.0)
            push(now + duration, "finished", execution)
        scheduler.record_launch(launch)
        result.max_running_over_capacity = max(
            result.max_running_over_capacity,
            scheduler.running - scheduler.rollout_capacity,
        )

        cell = scheduler.next_training_cell(
            ready_groups=len(ready), groups_needed=workload.batch_groups - claimed
        )
        if cell is not None:
            scheduler.begin_training_transition(cell)
            push(now + workload.to_training_s, "joined", cell)
        deadline = scheduler.supply_hold_deadline()
        if deadline is not None and deadline > now:
            push(deadline, "wake")

        while (
            scheduler.batch_open
            and idle_trainers
            and ready
            and claimed < workload.batch_groups
        ):
            trainer = min(idle_trainers)
            idle_trainers.discard(trainer)
            ready.popleft()
            claimed += 1
            push(now + workload.train_s_per_group, "trained", trainer)

    dispatch_groups(workload.initial_groups)
    scheduler.open_batch()
    result.joins.append([])
    settle(0.0)
    while events and steps_done < workload.steps:
        now, _, kind, payload = heapq.heappop(events)
        clock.now = now
        if kind == "finished":
            scheduler.record_finish(1)
            group = group_of[payload]
            unfinished_in_group[group] -= 1
            if unfinished_in_group[group] == 0:
                ready.append(group)
        elif kind == "joined":
            scheduler.finish_training_transition(payload)
            idle_trainers.add(payload)
            result.joins[-1].append((payload, now, scheduler.finished))
            if burst_ending:
                end_burst(now)
        elif kind == "trained":
            idle_trainers.add(payload)
            trained += 1
            if trained == workload.batch_groups:
                # One optimizer update per burst (mu = 1).
                scheduler.close_batch()
                steps_done += 1
                result.finish_time = now
                claimed = trained = 0
                burst_ending = True
                end_burst(now)
                if not burst_ending and not returning:
                    open_next_batch()
        elif kind == "returned":
            scheduler.finish_rollout_transition(payload)
            returning.discard(payload)
            if not returning:
                # Every engine now serves the published weights.
                open_next_batch()
        settle(now)
    assert steps_done == workload.steps, "simulation stalled before finishing"
    return result


_ELASTIC = _Workload(
    num_cells=4,
    cell_capacity=6,
    group_size=4,
    batch_groups=4,
    initial_groups=6,  # (phi + mu) B with phi = 0.5, mu = 1
    steps=4,
    train_s_per_group=120.0,
)


class TestSimulation:
    def test_cells_join_training_as_rollouts_drain(self):
        result = _simulate(_ELASTIC)
        assert result.max_running_over_capacity <= 0
        first = result.joins[0]
        cells = [cell for cell, _, _ in first]
        assert cells == list(range(len(cells))), "cells join core-first, in order"
        assert len(cells) >= 2, "satellites should join the first batch"
        times = [at for _, at, _ in first]
        finished = [f for _, _, f in first]
        assert times == sorted(times)
        assert finished[0] < finished[-1], (
            "later cells should join only after more executions have finished"
        )
        assert finished[0] < _ELASTIC.initial_groups * _ELASTIC.group_size, (
            "the core should start training before the rollout tail drains"
        )

    def test_elastic_cells_beat_switching_the_whole_pool(self):
        """The same GPUs as one colocated pool must wait for every execution."""
        elastic = _simulate(_ELASTIC)
        colocated = _simulate(
            _Workload(
                num_cells=1,
                cell_capacity=_ELASTIC.num_cells * _ELASTIC.cell_capacity,
                group_size=_ELASTIC.group_size,
                batch_groups=_ELASTIC.batch_groups,
                initial_groups=_ELASTIC.batch_groups,  # Colocate dispatches phi = 0
                steps=_ELASTIC.steps,
                train_s_per_group=_ELASTIC.train_s_per_group / _ELASTIC.num_cells,
            )
        )
        assert colocated.max_running_over_capacity <= 0
        assert elastic.finish_time < colocated.finish_time
