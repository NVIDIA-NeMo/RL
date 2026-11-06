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

"""Scheduling policy for elastic cells that switch between rollout and training.

GPUs are organized into ``K`` ordered *cells*, each able to serve rollouts or host a
complete training replica, plus an optional standalone pool that only serves rollouts.
Cell 0 is the *core*: it owns the optimizer and the authoritative weights, and it is
always the first cell to enter training. The other cells are *satellites* that join an
in-progress batch as rollouts drain. See ``docs/design-docs/elastic-scheduler.md``.

This module is the decision half only, in the shape of
:mod:`nemo_rl.models.generation.fleet_health`: a pure state machine over counters,
cell roles and a clock, with no Ray, GPU or asyncio. The caller performs every
transition and reports back when it has finished, so each rule is testable without a
cluster. Nothing in the SingleController drives it yet.

The rules implement equations 3-6 of the QwenGyre paper (arXiv 2609.33848, section 4).
All capacities and counters are in *execution* units: one rollout execution occupies one
slot of a cell's rollout capacity from launch until it terminates.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

CORE_CELL = 0


class CellRole(str, enum.Enum):
    """Role of one elastic cell."""

    # Serving rollouts; counts toward rollout capacity.
    ROLLOUT = "rollout"
    # Requests rerouted away, rollout state offloading, training state onloading.
    TO_TRAINING = "to_training"
    # Hosting a training replica and claiming micro-steps.
    TRAINING = "training"
    # Training state offloading, engine onloading and receiving the published weights.
    TO_ROLLOUT = "to_rollout"


@dataclass(frozen=True)
class ElasticSchedulerPolicy:
    """Static parameters of the elastic scheduling policy.

    Internal, like :class:`nemo_rl.models.generation.fleet_health.FleetHealthPolicy`:
    the user-facing BaseModel arrives with the controller wiring that consumes it.

    Attributes:
        num_cells: ``K``, the number of elastic cells, core included.
        cell_capacity: ``C_e``, concurrent executions one cell serves in rollout.
        standalone_capacity: ``C_s``, concurrent executions of the standalone rollout
            pool; 0 when there is none.
        batch_groups: ``B``, rollout groups consumed by one training step.
        min_supply_fraction: ``rho_min`` in (0, 1], the fraction of ``B`` that ready,
            unassigned groups must cover before a cell may move to training.
        supply_hold_s: ``tau``, seconds the supply condition must hold continuously.
        burst_dispatch: ``mu * N * B``, executions dispatched after each burst's weight
            publication. Only the core-retention rule (eq. 6) reads it.
    """

    num_cells: int
    cell_capacity: int
    standalone_capacity: int
    batch_groups: int
    min_supply_fraction: float
    supply_hold_s: float
    burst_dispatch: int

    def __post_init__(self) -> None:
        if self.num_cells < 1:
            raise ValueError(f"num_cells must be >= 1, got {self.num_cells}")
        if self.cell_capacity < 1:
            raise ValueError(f"cell_capacity must be >= 1, got {self.cell_capacity}")
        if self.standalone_capacity < 0:
            raise ValueError(
                f"standalone_capacity must be >= 0, got {self.standalone_capacity}"
            )
        if self.batch_groups < 1:
            raise ValueError(f"batch_groups must be >= 1, got {self.batch_groups}")
        if not 0.0 < self.min_supply_fraction <= 1.0:
            raise ValueError(
                f"min_supply_fraction must be in (0, 1], got {self.min_supply_fraction}"
            )
        if self.supply_hold_s < 0:
            raise ValueError(f"supply_hold_s must be >= 0, got {self.supply_hold_s}")
        if self.burst_dispatch < 0:
            raise ValueError(f"burst_dispatch must be >= 0, got {self.burst_dispatch}")


@dataclass(frozen=True)
class BurstEndPlan:
    """What happens to the training cells after a burst's final optimizer update.

    Attributes:
        return_to_rollout: Cells to move back to rollout, satellites in descending
            order and then the core if it is not retained.
        core_stays_in_training: Whether the core keeps its training state for the next
            burst instead of resharding into its rollout engine.
    """

    return_to_rollout: tuple[int, ...]
    core_stays_in_training: bool


class ElasticScheduler:
    """Tracks execution counters and cell roles and decides role transitions.

    Counters follow the paper's boundary dispatch: ``dispatched`` (d) grows when
    executions enter the rollout queue, ``launched`` (p) when they start, ``finished``
    (f) once per launched execution on completion, failure, timeout or cancellation.
    A model request retried on another engine stays inside its execution and changes
    no counter.

    Transitions are serial and in cell order, so the training cells always form a
    prefix ``0..k-1``. A cell stops counting toward rollout capacity as soon as its
    transition to training begins, and counts again only once its transition back to
    rollout has finished, which the caller reports after the engine serves the
    published weights.

    Args:
        policy: Static policy parameters.
        clock: Monotonic time source; injectable so tests need not sleep.
    """

    def __init__(
        self,
        *,
        policy: ElasticSchedulerPolicy,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._policy = policy
        self._clock = clock if clock is not None else time.monotonic
        self._roles: list[CellRole] = [CellRole.ROLLOUT] * policy.num_cells
        self._dispatched = 0
        self._launched = 0
        self._finished = 0
        self._batch_open = False
        # When the supply condition last became true without a break since; None
        # while it does not hold or since the last reset.
        self._supply_since: Optional[float] = None

    # ── read side ────────────────────────────────────────────────────────

    @property
    def policy(self) -> ElasticSchedulerPolicy:
        return self._policy

    @property
    def roles(self) -> tuple[CellRole, ...]:
        return tuple(self._roles)

    @property
    def dispatched(self) -> int:
        return self._dispatched

    @property
    def launched(self) -> int:
        return self._launched

    @property
    def finished(self) -> int:
        return self._finished

    @property
    def queued(self) -> int:
        """Executions dispatched but not launched, ``d - p``."""
        return self._dispatched - self._launched

    @property
    def running(self) -> int:
        """Executions launched and not yet terminated, ``p - f``."""
        return self._launched - self._finished

    @property
    def waterlevel(self) -> int:
        """Outstanding executions, ``w = d - f``."""
        return self._dispatched - self._finished

    @property
    def batch_open(self) -> bool:
        return self._batch_open

    @property
    def non_rollout_cells(self) -> int:
        """``k``: cells training or moving between roles, none of which serve rollouts."""
        return sum(role is not CellRole.ROLLOUT for role in self._roles)

    @property
    def rollout_capacity(self) -> int:
        """Eq. 3: ``C_R = C_s + (K - k) C_e``."""
        return (
            self._policy.standalone_capacity
            + (self._policy.num_cells - self.non_rollout_cells)
            * self._policy.cell_capacity
        )

    def supply_hold_deadline(self) -> Optional[float]:
        """Clock time at which the held supply satisfies ``tau``, or None if not held.

        Lets a caller that polls on events schedule one extra poll for the moment the
        hold timer can expire, instead of polling on a fixed interval.
        """
        if self._supply_since is None:
            return None
        return self._supply_since + self._policy.supply_hold_s

    def launchable(self) -> int:
        """Eq. 4: executions to launch now, ``min(d - p, C_R - (p - f))``."""
        return max(0, min(self.queued, self.rollout_capacity - self.running))

    def waterlevel_admits_next_cell(self) -> bool:
        """Eq. 5: whether rollout capacity still covers ``w`` with one cell fewer."""
        return self.waterlevel <= self.rollout_capacity - self._policy.cell_capacity

    def next_training_cell(
        self, *, ready_groups: int, groups_needed: int
    ) -> Optional[int]:
        """The cell that may begin moving to training now, or None.

        Also advances the supply timer, so call it whenever the counters or the ready
        supply change, and at :meth:`supply_hold_deadline`.

        A cell may move only while a batch's joining window is open, no other
        transition to training is in flight, and both hold:

        * eq. 5, :meth:`waterlevel_admits_next_cell`, and
        * the supply check: ``min(ready_groups, groups_needed) >= rho_min * B``
          continuously for ``tau`` seconds.

        Args:
            ready_groups: Ready rollout groups not yet assigned to a training batch.
            groups_needed: Groups the current batch still needs.
        """
        if ready_groups < 0 or groups_needed < 0:
            raise ValueError(
                "ready_groups and groups_needed must be >= 0, got "
                f"ready_groups={ready_groups}, groups_needed={groups_needed}"
            )
        now = self._clock()
        supplied = min(ready_groups, groups_needed) >= self._supply_threshold()
        if not supplied:
            self._supply_since = None
        elif self._supply_since is None:
            self._supply_since = now

        candidate = self._next_cell_in_order()
        if (
            candidate is None
            or not self._batch_open
            or CellRole.TO_TRAINING in self._roles
            or not self.waterlevel_admits_next_cell()
        ):
            return None
        deadline = self.supply_hold_deadline()
        if deadline is None or now < deadline:
            return None
        return candidate

    def plan_burst_end(self, *, ready_groups: int) -> BurstEndPlan:
        """Decide which training cells return to rollout after the final update.

        Every satellite returns. The core stays in training only if eq. 6 holds --
        ``w + mu N B <= C_s + (K - 1) C_e``, i.e. with only the core out of rollout the
        capacity covers outstanding work plus the dispatch that follows publication --
        and at least ``rho_min * B`` ready, unassigned groups exist for the next batch.

        Call it after the final update and before recording the post-publication
        dispatch, which eq. 6 already counts through ``burst_dispatch``.

        Args:
            ready_groups: Ready rollout groups not assigned to any batch.

        Raises:
            RuntimeError: If the joining window is still open, a cell is mid-transition,
                or the core is not training.
        """
        if self._batch_open:
            raise RuntimeError("close the batch before planning the burst end")
        in_flight = [
            idx
            for idx, role in enumerate(self._roles)
            if role in (CellRole.TO_TRAINING, CellRole.TO_ROLLOUT)
        ]
        if in_flight:
            raise RuntimeError(
                f"cells {in_flight} are mid-transition; a burst ends only once every "
                "transition has finished"
            )
        if self._roles[CORE_CELL] is not CellRole.TRAINING:
            raise RuntimeError("a burst cannot end without the core in training")

        policy = self._policy
        retained_capacity = (
            policy.standalone_capacity + (policy.num_cells - 1) * policy.cell_capacity
        )
        core_stays = (
            self.waterlevel + policy.burst_dispatch <= retained_capacity
            and min(ready_groups, policy.batch_groups) >= self._supply_threshold()
        )
        satellites = [
            idx
            for idx in range(policy.num_cells - 1, CORE_CELL, -1)
            if self._roles[idx] is CellRole.TRAINING
        ]
        if not core_stays:
            satellites.append(CORE_CELL)
        return BurstEndPlan(
            return_to_rollout=tuple(satellites), core_stays_in_training=core_stays
        )

    # ── write side ───────────────────────────────────────────────────────

    def record_dispatch(self, count: int) -> None:
        """Count executions entering the rollout queue."""
        self._require_non_negative(count, "dispatch")
        self._dispatched += count

    def record_launch(self, count: int) -> None:
        """Count queued executions that have started."""
        self._require_non_negative(count, "launch")
        if count > self.launchable():
            raise ValueError(
                f"cannot launch {count} execution(s): {self.queued} queued, "
                f"{self.running} running of {self.rollout_capacity} rollout capacity"
            )
        self._launched += count

    def record_finish(self, count: int) -> None:
        """Count launched executions that terminated, for any reason."""
        self._require_non_negative(count, "finish")
        if count > self.running:
            raise ValueError(
                f"cannot finish {count} execution(s) with only {self.running} running"
            )
        self._finished += count

    def open_batch(self) -> None:
        """Open a training batch's joining window and restart the supply timer."""
        if self._batch_open:
            raise RuntimeError("a batch is already open")
        self._batch_open = True
        self._supply_since = None

    def close_batch(self) -> None:
        """Close the joining window ahead of the batch's optimizer update."""
        if not self._batch_open:
            raise RuntimeError("no batch is open")
        self._batch_open = False

    def begin_training_transition(self, cell: int) -> None:
        """Mark ``cell`` as moving to training; it stops counting toward capacity.

        Raises:
            RuntimeError: If ``cell`` is not next in order, another transition to
                training is in flight, or eq. 5 does not hold.
        """
        expected = self._next_cell_in_order()
        if cell != expected:
            raise RuntimeError(
                f"cell {cell} cannot move to training; the next cell in order is "
                f"{expected}"
            )
        if CellRole.TO_TRAINING in self._roles:
            raise RuntimeError("transitions to training are serial")
        if not self.waterlevel_admits_next_cell():
            raise RuntimeError(
                f"waterlevel {self.waterlevel} exceeds the capacity left without cell "
                f"{cell} ({self.rollout_capacity - self._policy.cell_capacity})"
            )
        self._roles[cell] = CellRole.TO_TRAINING

    def finish_training_transition(self, cell: int) -> None:
        """Mark ``cell`` as training and restart the supply timer."""
        self._move(cell, frm=CellRole.TO_TRAINING, to=CellRole.TRAINING)
        self._supply_since = None

    def begin_rollout_transition(self, cell: int) -> None:
        """Mark a training ``cell`` as moving back to rollout."""
        if self._batch_open:
            raise RuntimeError("cells return to rollout only after the batch closes")
        self._move(cell, frm=CellRole.TRAINING, to=CellRole.TO_ROLLOUT)

    def finish_rollout_transition(self, cell: int) -> None:
        """Mark ``cell`` as serving rollouts again and restart the supply timer.

        Call it once the cell's engine serves the published weights: capacity it
        adds is immediately launchable.
        """
        self._move(cell, frm=CellRole.TO_ROLLOUT, to=CellRole.ROLLOUT)
        self._supply_since = None

    # ── internals ────────────────────────────────────────────────────────

    def _supply_threshold(self) -> float:
        return self._policy.min_supply_fraction * self._policy.batch_groups

    def _next_cell_in_order(self) -> Optional[int]:
        """First cell still in rollout, provided every cell before it is training."""
        for idx, role in enumerate(self._roles):
            if role is CellRole.ROLLOUT:
                return idx
            if role is not CellRole.TRAINING and role is not CellRole.TO_TRAINING:
                return None
        return None

    def _move(self, cell: int, *, frm: CellRole, to: CellRole) -> None:
        if not 0 <= cell < self._policy.num_cells:
            raise ValueError(f"cell {cell} is outside 0..{self._policy.num_cells - 1}")
        if self._roles[cell] is not frm:
            raise RuntimeError(
                f"cell {cell} is {self._roles[cell].value}, expected {frm.value}"
            )
        self._roles[cell] = to

    @staticmethod
    def _require_non_negative(count: int, what: str) -> None:
        if count < 0:
            raise ValueError(f"{what} count must be >= 0, got {count}")
