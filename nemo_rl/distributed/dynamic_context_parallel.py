# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only plans for Ray-dispatched hybrid DP/CP execution.

A lane is one complete TP replica. Synchronization groups partition the DP*CP
lanes into aligned, power-of-two attention groups. Each attention group owns
an ordered list of independently bounded packed tasks. Different attention
groups may execute different numbers of tasks before every lane meets at the
next group boundary, matching MCore's uneven hybrid-CP execution model.
"""

from dataclasses import dataclass
from math import lcm

from pydantic import BaseModel, PositiveInt


class DynamicContextParallelConfig(BaseModel, extra="forbid"):
    """Runtime CP scheduling; token budgets are per CP rank, before SP."""

    enabled: bool = False
    min_size: PositiveInt = 1
    max_size: PositiveInt | None = None
    tokens_per_rank: PositiveInt = 2048


@dataclass(frozen=True)
class CPAssignment:
    """One packed task, shared by all its participating lanes."""

    sample_indices: tuple[int, ...]
    lane_start: int
    cp_size: int
    pad_multiple: int
    padded_tokens: int


@dataclass(frozen=True)
class CPSyncGroup:
    """A fixed CP partition with one or more sequential tasks per lane."""

    assignments: tuple[CPAssignment, ...]


@dataclass(frozen=True)
class CPRankGroup:
    """The ordered packed tasks one lane executes in a synchronization group."""

    assignments: tuple[CPAssignment, ...]


@dataclass(frozen=True)
class CPRankStep:
    """Assignments and unique-data normalizers for one optimizer step."""

    groups: tuple[CPRankGroup, ...]
    valid_sequences: float
    valid_tokens: float

    @property
    def assignments(self) -> tuple[CPAssignment, ...]:
        """Flattened execution order, retained for metrics and compatibility."""
        return tuple(task for group in self.groups for task in group.assignments)


@dataclass(frozen=True)
class CPRankPlan:
    """Serializable plan; process groups are resolved only on the workers."""

    lane: int
    lane_ranks: tuple[int, ...]
    steps: tuple[CPRankStep, ...]


def _padding_for_cp(
    cp_size: int,
    *,
    sequence_parallel_size: int,
    user_pad_multiple: int,
    token_alignment: int,
) -> int:
    return lcm(
        user_pad_multiple,
        (2 * cp_size if cp_size > 1 else 1) * sequence_parallel_size * token_alignment,
    )


def _resize_assignment(
    task: CPAssignment,
    cp_size: int,
    *,
    lengths: list[int],
    tokens_per_rank: int,
    sequence_parallel_size: int,
    user_pad_multiple: int,
    token_alignment: int,
) -> CPAssignment:
    factor = _padding_for_cp(
        cp_size,
        sequence_parallel_size=sequence_parallel_size,
        user_pad_multiple=user_pad_multiple,
        token_alignment=token_alignment,
    )
    padded_tokens = sum(
        (lengths[index] + factor - 1) // factor * factor
        for index in task.sample_indices
    )
    if padded_tokens > tokens_per_rank * cp_size:
        raise ValueError("Expanded dynamic CP assignment exceeds its token budget")
    return CPAssignment(
        task.sample_indices,
        0,
        cp_size,
        factor,
        padded_tokens,
    )


def _fill_idle_lanes(
    tasks: list[CPAssignment],
    *,
    lanes: int,
    max_size: int,
    lengths: list[int],
    tokens_per_rank: int,
    sequence_parallel_size: int,
    user_pad_multiple: int,
    token_alignment: int,
) -> list[CPAssignment]:
    """Increase the smallest real CP groups until no legal expansion fits."""
    idle_lanes = lanes - sum(task.cp_size for task in tasks)
    while idle_lanes:
        candidates = [
            (task.cp_size, index)
            for index, task in enumerate(tasks)
            if task.cp_size < max_size and task.cp_size <= idle_lanes
        ]
        if not candidates:
            break
        _, index = min(candidates)
        task = tasks[index]
        tasks[index] = _resize_assignment(
            task,
            task.cp_size * 2,
            lengths=lengths,
            tokens_per_rank=tokens_per_rank,
            sequence_parallel_size=sequence_parallel_size,
            user_pad_multiple=user_pad_multiple,
            token_alignment=token_alignment,
        )
        idle_lanes -= task.cp_size

    # Descending powers of two guarantee that each consecutive lane start is
    # aligned for its group. Keep sample indices as a deterministic tie-breaker.
    tasks.sort(key=lambda task: (-task.cp_size, task.sample_indices))
    placed = []
    cursor = 0
    for task in tasks:
        placed.append(
            CPAssignment(
                task.sample_indices,
                cursor,
                task.cp_size,
                task.pad_multiple,
                task.padded_tokens,
            )
        )
        cursor += task.cp_size
    return placed


def _phase_topology(phase: tuple[CPAssignment, ...]) -> tuple[tuple[int, int], ...]:
    return tuple((task.lane_start, task.cp_size) for task in phase)


def _assignment_work(task: CPAssignment, lengths: list[int]) -> float:
    """Estimate packed attention work per participating CP rank."""
    return (
        sum(float(lengths[index] ** 2) for index in task.sample_indices) / task.cp_size
    )


def _merge_compatible_phases(
    phases: list[tuple[CPAssignment, ...]], lengths: list[int]
) -> tuple[CPSyncGroup, ...]:
    """Merge adjacent equal partitions and balance their tasks across subgroups.

    A topology change still requires a domain-wide barrier. Equal topologies do
    not: their independently bounded packed tasks can run sequentially inside
    the same fixed CP subgroups. LPT placement minimizes the longest estimated
    attention workload and naturally produces uneven task counts.
    """
    merged: list[CPSyncGroup] = []
    cursor = 0
    while cursor < len(phases):
        topology = _phase_topology(phases[cursor])
        end = cursor + 1
        while end < len(phases) and _phase_topology(phases[end]) == topology:
            end += 1

        slots_by_size: dict[int, list[int]] = {}
        for lane_start, cp_size in topology:
            slots_by_size.setdefault(cp_size, []).append(lane_start)
        assigned: dict[tuple[int, int], list[CPAssignment]] = {
            slot: [] for slot in topology
        }
        loads = {slot: 0.0 for slot in topology}

        real_tasks = [
            task
            for phase in phases[cursor:end]
            for task in phase
            if task.sample_indices
        ]
        for task in sorted(
            real_tasks,
            key=lambda item: (-_assignment_work(item, lengths), item.sample_indices),
        ):
            candidates = [
                (loads[(lane_start, task.cp_size)], lane_start)
                for lane_start in slots_by_size[task.cp_size]
            ]
            _, lane_start = min(candidates)
            slot = (lane_start, task.cp_size)
            placed = CPAssignment(
                task.sample_indices,
                lane_start,
                task.cp_size,
                task.pad_multiple,
                task.padded_tokens,
            )
            assigned[slot].append(placed)
            loads[slot] += _assignment_work(placed, lengths)

        group_assignments: list[CPAssignment] = []
        for phase_task in phases[cursor]:
            slot = (phase_task.lane_start, phase_task.cp_size)
            tasks = assigned[slot]
            if tasks:
                group_assignments.extend(tasks)
            else:
                # Every lane needs at least one local call so the final backward
                # on every DDP rank can participate in the same gradient sync.
                group_assignments.append(
                    CPAssignment(
                        (),
                        phase_task.lane_start,
                        phase_task.cp_size,
                        phase_task.pad_multiple,
                        (
                            (2 + phase_task.pad_multiple - 1)
                            // phase_task.pad_multiple
                            * phase_task.pad_multiple
                        ),
                    )
                )
        merged.append(CPSyncGroup(tuple(group_assignments)))
        cursor = end
    return tuple(merged)


def plan_cp_phases(
    lengths: list[int],
    *,
    lanes: int,
    min_size: int,
    max_size: int,
    tokens_per_rank: int,
    sequence_parallel_size: int,
    user_pad_multiple: int,
    token_alignment: int = 1,
) -> tuple[CPSyncGroup, ...]:
    """Pack sequences and form MCore-style synchronization groups.

    Sample indices refer to the original batch, never to a sorted copy.
    Padding is included when choosing CP size and admitting samples to bins.
    Every packed task remains within its per-rank token budget. Adjacent phases
    with the same CP partition are merged and balanced across their subgroups,
    allowing different lanes to execute different sequential task counts.
    """
    if lanes < 1 or lanes & (lanes - 1):
        raise ValueError("Dynamic CP requires a power-of-two DP*CP domain")
    sizes = (min_size, max_size)
    if any(s < 1 or s & (s - 1) for s in sizes) or not min_size <= max_size <= lanes:
        raise ValueError("CP bounds must be powers of two within the DP*CP domain")
    if (
        min(tokens_per_rank, sequence_parallel_size, user_pad_multiple, token_alignment)
        < 1
    ):
        raise ValueError("Token budgets and padding factors must be positive")
    bins: dict[int, list[tuple[list[int], int, int]]] = {}
    for index in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        length = lengths[index]
        if length < 2:
            raise ValueError("Dynamic CP requires at least two input tokens per sample")
        size = min_size
        while True:
            multiple = _padding_for_cp(
                size,
                sequence_parallel_size=sequence_parallel_size,
                user_pad_multiple=user_pad_multiple,
                token_alignment=token_alignment,
            )
            padded = (length + multiple - 1) // multiple * multiple
            if padded <= tokens_per_rank * size:
                break
            size *= 2
            if size > max_size:
                raise ValueError(
                    f"Sample {index} of length {length} exceeds the dynamic CP token budget"
                )
        size_bins = bins.setdefault(size, [])
        for bin_index, (members, used, factor) in enumerate(size_bins):
            if used + padded <= tokens_per_rank * size:
                members.append(index)
                size_bins[bin_index] = (members, used + padded, factor)
                break
        else:
            size_bins.append(([index], padded, multiple))

    pending = [
        CPAssignment(tuple(members), 0, size, factor, used)
        for size in sorted(bins, reverse=True)
        for members, used, factor in bins[size]
    ]
    phases = []
    while pending:
        selected = []
        remaining = []
        cursor = 0
        for task in pending:
            if cursor + task.cp_size <= lanes:
                selected.append(task)
                cursor += task.cp_size
            else:
                remaining.append(task)
        phase = _fill_idle_lanes(
            selected,
            lanes=lanes,
            max_size=max_size,
            lengths=lengths,
            tokens_per_rank=tokens_per_rank,
            sequence_parallel_size=sequence_parallel_size,
            user_pad_multiple=user_pad_multiple,
            token_alignment=token_alignment,
        )
        cursor = sum(task.cp_size for task in phase)
        while cursor < lanes:
            factor = _padding_for_cp(
                min_size,
                sequence_parallel_size=sequence_parallel_size,
                user_pad_multiple=user_pad_multiple,
                token_alignment=token_alignment,
            )
            phase.append(
                CPAssignment(
                    (), cursor, min_size, factor, ((2 + factor - 1) // factor) * factor
                )
            )
            cursor += min_size
        phases.append(tuple(phase))
        pending = remaining
    return _merge_compatible_phases(phases, lengths)


def assignments_for_lane(group: CPSyncGroup, lane: int) -> tuple[CPAssignment, ...]:
    """Resolve a lane's ordered tasks in one synchronization group."""
    matches = tuple(
        task
        for task in group.assignments
        if task.lane_start <= lane < task.lane_start + task.cp_size
    )
    if not matches:
        raise ValueError(f"Lane {lane} has no assignment in a CP synchronization group")
    topology = {(task.lane_start, task.cp_size) for task in matches}
    if len(topology) != 1:
        raise ValueError(
            f"Lane {lane} crosses {len(topology)} CP subgroups before synchronization"
        )
    return matches


def assignment_for_lane(group: CPSyncGroup, lane: int) -> CPAssignment:
    """Resolve one-task groups; callers needing uneven work use assignments_for_lane."""
    matches = assignments_for_lane(group, lane)
    if len(matches) != 1:
        raise ValueError(
            f"Lane {lane} has {len(matches)} sequential assignments in a CP group"
        )
    return matches[0]


def cp_loss_multiplier(
    *,
    active_cp_size: int,
    schedule_cp_size: int,
    num_microbatches: int,
    replicated_cp_loss: bool,
) -> float:
    """Cancel MCore's legacy averaging and the gathered loss's duplication.

    The no-pipeline executor scales by static CP / microbatches. Attention
    and the differentiable CP gather use active CP, which may differ.
    """
    if min(active_cp_size, schedule_cp_size, num_microbatches) < 1:
        raise ValueError("CP sizes and microbatch count must be positive")
    replicas = active_cp_size if replicated_cp_loss else 1
    return num_microbatches / (schedule_cp_size * replicas)
