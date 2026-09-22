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

import random

import pytest

from nemo_rl.distributed.dynamic_context_parallel import (
    DynamicContextParallelConfig,
    assignment_for_lane,
    assignments_for_lane,
    cp_loss_multiplier,
    plan_cp_phases,
)
from nemo_rl.distributed.named_sharding import REPLICATED_AXES, replicated_axes


def make_plan(lengths, *, lanes=8, minimum=1, maximum=None, budget=128, sp=2):
    return plan_cp_phases(
        lengths,
        lanes=lanes,
        min_size=minimum,
        max_size=maximum or lanes,
        tokens_per_rank=budget,
        sequence_parallel_size=sp,
        user_pad_multiple=sp,
    )


def test_one_token_budget_is_shared_by_score_and_train():
    config = DynamicContextParallelConfig(enabled=True, tokens_per_rank=64)
    assert config.tokens_per_rank == 64
    with pytest.raises(ValueError):
        DynamicContextParallelConfig(
            enabled=True,
            train_tokens_per_rank=64,
            logprob_tokens_per_rank=128,
        )


def check_plan(phases, lengths, lanes, budget, sp):
    owners = []
    for phase in phases:
        for lane in range(lanes):
            tasks = assignments_for_lane(phase, lane)
            assert len({(task.lane_start, task.cp_size) for task in tasks}) == 1
            for task in tasks:
                assert task.lane_start % task.cp_size == 0
                assert task.padded_tokens % (task.cp_size * sp) == 0
                assert task.padded_tokens // task.cp_size <= budget
        for task in phase.assignments:
            if not task.sample_indices:
                continue
            owners.extend(task.sample_indices)
            spans = [
                (lengths[i] + task.pad_multiple - 1)
                // task.pad_multiple
                * task.pad_multiple
                for i in task.sample_indices
            ]
            assert sum(spans) == task.padded_tokens
            # Reconstruct the zigzag partition and verify each real token's
            # ownership, independently of padding and concatenation order.
            for length, span in zip((lengths[i] for i in task.sample_indices), spans):
                owned = []
                for rank in range(task.cp_size):
                    if task.cp_size == 1:
                        positions = range(span)
                    else:
                        chunk = span // (2 * task.cp_size)
                        positions = [
                            p
                            for c in (rank, 2 * task.cp_size - rank - 1)
                            for p in range(c * chunk, (c + 1) * chunk)
                        ]
                    owned.extend(p for p in positions if p < length)
                assert sorted(owned) == list(range(length))
    assert sorted(owners) == list(range(len(lengths)))


def test_mixed_cp_and_cross_dp_groups():
    lengths = [400, 200, 90, 70, 9, 3, 1000]
    phases = make_plan(lengths)
    check_plan(phases, lengths, 8, 128, 2)
    assert (
        len({a.cp_size for p in phases for a in p.assignments if a.sample_indices}) > 1
    )
    assert any(a.cp_size == 8 for p in phases for a in p.assignments)


@pytest.mark.parametrize("lanes", [1, 2, 4, 8, 16])
def test_randomized_coverage_and_alignment(lanes):
    rng = random.Random(42)
    for _ in range(30):
        lengths = [rng.randint(2, 64 * lanes) for _ in range(rng.randint(1, 40))]
        phases = make_plan(lengths, lanes=lanes)
        check_plan(phases, lengths, lanes, 128, 2)
        assert phases == make_plan(lengths, lanes=lanes)


def test_padding_drives_cp_selection():
    phases = make_plan([127], lanes=2, budget=127)
    task = next(a for p in phases for a in p.assignments if a.sample_indices)
    assert task.cp_size == 2


def test_short_sequences_fill_existing_large_cp_packs():
    lengths = [6000, 2000]
    phases = make_plan(lengths, lanes=2, budget=4096, sp=4)
    tasks = [task for phase in phases for task in phase.assignments]
    assert len(tasks) == 1
    assert tasks[0].sample_indices == (0, 1)
    assert tasks[0].cp_size == 2
    assert tasks[0].padded_tokens == 8000
    check_plan(phases, lengths, 2, 4096, 4)


def test_cross_size_packing_uses_destination_padding_and_budget():
    # 4089 fits beside 4100 with CP1 padding (4100+4092), but not with
    # CP2's required multiple of 16 (4112+4096). Do not overfill that pack.
    lengths = [4100, 4089, 4000, 4000]
    phases = make_plan(lengths, lanes=2, budget=4096, sp=4)
    tasks = [task for phase in phases for task in phase.assignments]
    long_task = next(task for task in tasks if 0 in task.sample_indices)
    assert 1 not in long_task.sample_indices
    assert len(long_task.sample_indices) == 2
    assert long_task.padded_tokens == 8112
    check_plan(phases, lengths, 2, 4096, 4)


def test_cross_size_packing_preserves_small_cp_for_remaining_work():
    lengths = [8192, 4000, 4000]
    phases = make_plan(lengths, lanes=2, budget=4096, sp=4)
    tasks = [task for phase in phases for task in phase.assignments]
    assert sorted(task.cp_size for task in tasks) == [1, 1, 2]
    check_plan(phases, lengths, 2, 4096, 4)


def test_cross_size_packing_respects_expert_minimum():
    lengths = [24000, 8000, 16000, 16000]
    phases = make_plan(lengths, lanes=8, minimum=4, budget=4096, sp=2)
    assert all(task.cp_size >= 4 for phase in phases for task in phase.assignments)
    long_task = next(
        task
        for phase in phases
        for task in phase.assignments
        if 0 in task.sample_indices
    )
    assert long_task.sample_indices == (0, 1)
    check_plan(phases, lengths, 8, 4096, 2)


@pytest.mark.parametrize("tp", [1, 2])
def test_gpu_parity_fixture_still_exercises_cp1_after_cross_size_packing(tp):
    lengths = [7, 45, 101, 11, 55, 9, 29, 29, 29, 29]
    phases = make_plan(lengths, lanes=4 // tp, budget=32 * tp, sp=tp)
    sizes = {task.cp_size for phase in phases for task in phase.assignments}
    assert sizes == ({1, 2, 4} if tp == 1 else {1, 2})


def test_moe_minimum_expands_real_work_to_fill_lanes():
    phases = make_plan([3, 7], minimum=4)
    assert all(a.cp_size >= 4 for p in phases for a in p.assignments)
    assert not any(not a.sample_indices for p in phases for a in p.assignments)
    assert [a.cp_size for p in phases for a in p.assignments] == [8]
    check_plan(phases, [3, 7], 8, 128, 2)


def test_idle_lanes_expand_real_assignment_and_recompute_padding():
    phases = make_plan([3])
    task = phases[0].assignments[0]
    assert task.sample_indices == (0,)
    assert task.cp_size == 8
    assert task.pad_multiple == 32
    assert task.padded_tokens == 32
    assert not any(
        not assignment.sample_indices for assignment in phases[0].assignments
    )
    check_plan(phases, [3], 8, 128, 2)


def test_idle_lane_expansion_skips_task_when_larger_cp_padding_overflows():
    lengths = [100] * 32
    phases = plan_cp_phases(
        lengths,
        lanes=2,
        min_size=1,
        max_size=2,
        tokens_per_rank=4096,
        sequence_parallel_size=1,
        user_pad_multiple=1,
        token_alignment=128,
    )

    real = [task for task in phases[0].assignments if task.sample_indices]
    placeholders = [task for task in phases[0].assignments if not task.sample_indices]
    assert [(task.cp_size, task.padded_tokens) for task in real] == [(1, 4096)]
    assert [task.cp_size for task in placeholders] == [1]
    check_plan(phases, lengths, 2, 4096, 1)


def test_maximum_cp_keeps_placeholders_when_real_work_cannot_fill_lanes():
    phases = make_plan([3], maximum=4)
    real = [
        assignment for assignment in phases[0].assignments if assignment.sample_indices
    ]
    placeholders = [
        assignment
        for assignment in phases[0].assignments
        if not assignment.sample_indices
    ]
    assert [assignment.cp_size for assignment in real] == [4]
    assert sum(assignment.cp_size for assignment in placeholders) == 4
    check_plan(phases, [3], 8, 128, 2)


def test_equal_topologies_merge_into_uneven_sequential_task_lists():
    phases = make_plan([40, 40, 40, 40, 40], lanes=2, maximum=1, budget=64, sp=1)
    assert len(phases) == 1
    lane_zero = assignments_for_lane(phases[0], 0)
    lane_one = assignments_for_lane(phases[0], 1)
    assert sorted((len(lane_zero), len(lane_one))) == [2, 3]
    assert {
        index for task in lane_zero + lane_one for index in task.sample_indices
    } == {
        0,
        1,
        2,
        3,
        4,
    }
    with pytest.raises(ValueError, match="sequential assignments"):
        assignment_for_lane(phases[0], 0)


@pytest.mark.parametrize(
    "lanes,minimum,maximum", [(3, 1, 2), (8, 3, 8), (8, 4, 2), (8, 1, 16)]
)
def test_invalid_topologies_rejected(lanes, minimum, maximum):
    with pytest.raises(ValueError):
        make_plan([], lanes=lanes, minimum=minimum, maximum=maximum)


def test_oversized_sample_rejected():
    with pytest.raises(ValueError, match="token budget"):
        make_plan([1025])


@pytest.mark.parametrize("base_cp", [1, 2, 4])
@pytest.mark.parametrize("active_cp", [1, 2, 4, 8])
@pytest.mark.parametrize("microbatches", [1, 3, 7])
def test_gather_backward_and_schedule_scaling_cancel(base_cp, active_cp, microbatches):
    factor = cp_loss_multiplier(
        active_cp_size=active_cp,
        schedule_cp_size=base_cp,
        num_microbatches=microbatches,
        replicated_cp_loss=True,
    )
    # MCore scales the returned scalar; gather backward SUMs identical
    # contributions. Each owned token must reach DDP SUM with unit weight.
    assert factor * base_cp / microbatches * active_cp == pytest.approx(1.0)


def test_dynamic_outputs_are_not_static_cp_replicas():
    assert "context_parallel" in REPLICATED_AXES
    assert replicated_axes() == REPLICATED_AXES
    assert replicated_axes(dynamic_cp=True) == ("tensor_parallel", "pipeline_parallel")
