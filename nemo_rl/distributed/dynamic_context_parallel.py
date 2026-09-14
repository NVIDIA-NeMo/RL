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

A lane is one complete TP replica. Phases partition the DP*CP lanes into
aligned, power-of-two attention groups. All lanes execute one packed forward
per phase; an empty assignment executes masked padding. This deliberately
uses more phase boundaries than Megatron's load-balancing scheduler, while
preserving its group-transition and gradient-accumulation semantics.
"""

from dataclasses import dataclass
from math import lcm

from pydantic import BaseModel, PositiveInt


class DynamicContextParallelConfig(BaseModel, extra="forbid"):
    """Runtime CP scheduling; token budgets are per CP rank, before SP."""

    enabled: bool = False
    min_size: PositiveInt = 1
    max_size: PositiveInt | None = None
    train_tokens_per_rank: PositiveInt = 2048
    logprob_tokens_per_rank: PositiveInt = 2048


@dataclass(frozen=True)
class CPAssignment:
    """One packed task, shared by all its participating lanes."""

    sample_indices: tuple[int, ...]
    lane_start: int
    cp_size: int
    pad_multiple: int
    padded_tokens: int


@dataclass(frozen=True)
class CPRankStep:
    """Assignments and unique-data normalizers for one optimizer step."""

    assignments: tuple[CPAssignment, ...]
    valid_sequences: float
    valid_tokens: float


@dataclass(frozen=True)
class CPRankPlan:
    """Serializable plan; process groups are resolved only on the workers."""

    lane: int
    lane_ranks: tuple[int, ...]
    steps: tuple[CPRankStep, ...]


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
) -> tuple[tuple[CPAssignment, ...], ...]:
    """Pack equal-CP sequences and place tasks into disjoint aligned groups.

    Sample indices refer to the original batch, never to a sorted copy.
    Padding is included when choosing CP size and admitting samples to bins.
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
            multiple = lcm(
                user_pad_multiple,
                (2 * size if size > 1 else 1)
                * sequence_parallel_size
                * token_alignment,
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
        phase = []
        remaining = []
        cursor = 0
        for task in pending:
            if cursor + task.cp_size <= lanes:
                phase.append(
                    CPAssignment(
                        task.sample_indices,
                        cursor,
                        task.cp_size,
                        task.pad_multiple,
                        task.padded_tokens,
                    )
                )
                cursor += task.cp_size
            else:
                remaining.append(task)
        while cursor < lanes:
            factor = lcm(
                user_pad_multiple,
                (2 * min_size if min_size > 1 else 1)
                * sequence_parallel_size
                * token_alignment,
            )
            phase.append(
                CPAssignment(
                    (), cursor, min_size, factor, ((2 + factor - 1) // factor) * factor
                )
            )
            cursor += min_size
        phases.append(tuple(phase))
        pending = remaining
    return tuple(phases)


def assignment_for_lane(phase: tuple[CPAssignment, ...], lane: int) -> CPAssignment:
    """Resolve exactly one task for a lane, including padding tasks."""
    matches = [a for a in phase if a.lane_start <= lane < a.lane_start + a.cp_size]
    if len(matches) != 1:
        raise ValueError(f"Lane {lane} has {len(matches)} assignments in a CP phase")
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
