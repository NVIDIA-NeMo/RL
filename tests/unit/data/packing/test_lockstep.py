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

import pickle
from dataclasses import FrozenInstanceError

import pytest

from nemo_rl.data.packing import (
    LockstepPackingItem,
    SidePackingSpec,
    build_lockstep_packing_plan,
)


def _items(count: int) -> tuple[LockstepPackingItem, ...]:
    return tuple(
        LockstepPackingItem(sample_id=f"source-{i}", batch_item_id=i)
        for i in range(count)
    )


def _round_total_to(multiple: int):
    def physical_size(effective_lengths: tuple[int, ...]) -> int:
        total = sum(effective_lengths)
        return (total + multiple - 1) // multiple * multiple

    return physical_size


def test_builds_shared_bins_with_side_local_geometry() -> None:
    items = _items(4)
    plan = build_lockstep_packing_plan(
        batch_uid=9,
        items=items,
        data_parallel_size=1,
        sides=(
            SidePackingSpec(
                side_id="student",
                capacity=16,
                raw_lengths=(6, 6, 4, 4),
                effective_lengths=(8, 8, 4, 4),
            ),
            SidePackingSpec(
                side_id="teacher",
                capacity=16,
                raw_lengths=(5, 7, 3, 5),
                effective_lengths=(5, 7, 3, 5),
                physical_size_fn=_round_total_to(8),
            ),
        ),
    )

    assert plan.canonical_batch_item_ids == (0, 1, 2, 3)
    assert plan.bins == ((0, 1), (3, 2))
    assert plan.sides["student"].rank_bin_indices == ((0, 1),)
    assert plan.sides["teacher"].rank_bin_indices == ((0, 1),)

    student = plan.sides["student"]
    assert student.raw_cu_seqlens_by_bin == ((0, 6, 12), (0, 4, 8))
    assert student.padded_cu_seqlens_by_bin == ((0, 8, 16), (0, 4, 8))
    assert student.physical_tokens_by_bin == (16, 8)

    teacher = plan.sides["teacher"]
    assert teacher.raw_cu_seqlens_by_bin == ((0, 5, 12), (0, 5, 8))
    assert teacher.padded_cu_seqlens_by_bin == ((0, 5, 16), (0, 5, 8))
    assert teacher.physical_tokens_by_bin == (16, 8)


def test_item_that_fits_no_existing_bin_starts_the_next_bin() -> None:
    plan = build_lockstep_packing_plan(
        batch_uid=10,
        items=_items(3),
        data_parallel_size=1,
        sides=(
            SidePackingSpec(
                side_id="student",
                capacity=10,
                raw_lengths=(8, 4, 3),
                effective_lengths=(8, 4, 3),
            ),
            SidePackingSpec(
                side_id="teacher",
                capacity=10,
                raw_lengths=(3, 7, 3),
                effective_lengths=(3, 7, 3),
            ),
        ),
    )

    # Item 1 cannot join item 0 on the student side, so it starts bin 2.
    # Item 2 then fails bin 1 but fits bin 2, proving that all bins are tried.
    assert plan.bins == ((0,), (1, 2))


def test_duplicate_sample_ids_remain_distinct_batch_occurrences() -> None:
    items = (
        LockstepPackingItem(sample_id="same-source-row", batch_item_id=100),
        LockstepPackingItem(sample_id="same-source-row", batch_item_id=101),
    )
    plan = build_lockstep_packing_plan(
        batch_uid=3,
        items=items,
        data_parallel_size=1,
        sides=(
            SidePackingSpec(
                side_id="student",
                capacity=16,
                raw_lengths=(5, 5),
                effective_lengths=(5, 5),
            ),
        ),
    )

    assert plan.canonical_batch_item_ids == (100, 101)
    assert plan.bins == ((100, 101),)


def test_plan_is_deterministic_and_immutable() -> None:
    items = _items(6)
    sides = (
        SidePackingSpec(
            side_id="student",
            capacity=12,
            raw_lengths=(6, 4, 2, 5, 3, 2),
            effective_lengths=(6, 4, 2, 5, 3, 2),
        ),
        SidePackingSpec(
            side_id="teacher",
            capacity=12,
            raw_lengths=(3, 5, 4, 2, 6, 3),
            effective_lengths=(3, 5, 4, 2, 6, 3),
        ),
    )

    first = build_lockstep_packing_plan(
        batch_uid=4, items=items, data_parallel_size=2, sides=sides
    )
    second = build_lockstep_packing_plan(
        batch_uid=4, items=items, data_parallel_size=2, sides=sides
    )
    reversed_sides = build_lockstep_packing_plan(
        batch_uid=4,
        items=items,
        data_parallel_size=2,
        sides=tuple(reversed(sides)),
    )
    assert first == second
    assert first == reversed_sides
    assert hash(first) == hash(second)
    assert hash(first) == hash(reversed_sides)
    assert pickle.loads(pickle.dumps(first)) == first

    with pytest.raises(FrozenInstanceError):
        first.batch_uid = 5  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.sides["student"] = first.sides["student"]  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        first.sides["student"].capacity = 20  # type: ignore[misc]


def test_rejects_duplicate_batch_item_id() -> None:
    with pytest.raises(ValueError, match="batch_item_id must be unique"):
        build_lockstep_packing_plan(
            batch_uid=0,
            items=(
                LockstepPackingItem(sample_id="a", batch_item_id=7),
                LockstepPackingItem(sample_id="b", batch_item_id=7),
            ),
            data_parallel_size=1,
            sides=(
                SidePackingSpec(
                    side_id="student",
                    capacity=10,
                    raw_lengths=(4, 4),
                    effective_lengths=(4, 4),
                ),
            ),
        )


def test_rejects_effective_length_smaller_than_raw_length() -> None:
    with pytest.raises(
        ValueError, match="effective length 7 smaller than raw length 8"
    ):
        SidePackingSpec(
            side_id="student",
            capacity=10,
            raw_lengths=(8,),
            effective_lengths=(7,),
        )


def test_oversize_error_names_source_sample_and_side() -> None:
    with pytest.raises(
        ValueError,
        match=r"sample_id='raw-row-42'.*batch_item_id=3.*side 'teacher'.*capacity=100",
    ):
        build_lockstep_packing_plan(
            batch_uid=1,
            items=(LockstepPackingItem(sample_id="raw-row-42", batch_item_id=3),),
            data_parallel_size=1,
            sides=(
                SidePackingSpec(
                    side_id="student",
                    capacity=128,
                    raw_lengths=(90,),
                    effective_lengths=(96,),
                ),
                SidePackingSpec(
                    side_id="teacher",
                    capacity=100,
                    raw_lengths=(97,),
                    effective_lengths=(104,),
                ),
            ),
        )


def test_gbs48_dp4_equalizes_seven_bins_to_eight_by_shared_split() -> None:
    items = _items(48)
    student_lengths = (10,) * 12 + (6,) * 12 + (10,) * 12 + (6,) * 12
    teacher_lengths = (6,) * 12 + (10,) * 12 + (10,) * 12 + (7,) * 12

    initial_counts = []
    for rank in range(4):
        start = rank * 12
        stop = start + 12
        rank_plan = build_lockstep_packing_plan(
            batch_uid=100 + rank,
            items=items[start:stop],
            data_parallel_size=1,
            sides=(
                SidePackingSpec(
                    side_id="student",
                    capacity=100,
                    raw_lengths=student_lengths[start:stop],
                    effective_lengths=student_lengths[start:stop],
                ),
                SidePackingSpec(
                    side_id="teacher",
                    capacity=100,
                    raw_lengths=teacher_lengths[start:stop],
                    effective_lengths=teacher_lengths[start:stop],
                ),
            ),
        )
        initial_counts.append(len(rank_plan.bins))
    assert initial_counts == [2, 2, 2, 1]
    assert sum(initial_counts) == 7

    plan = build_lockstep_packing_plan(
        batch_uid=200,
        items=items,
        data_parallel_size=4,
        sides=(
            SidePackingSpec(
                side_id="student",
                capacity=100,
                raw_lengths=student_lengths,
                effective_lengths=student_lengths,
            ),
            SidePackingSpec(
                side_id="teacher",
                capacity=100,
                raw_lengths=teacher_lengths,
                effective_lengths=teacher_lengths,
            ),
        ),
    )

    assert plan.bins == (
        tuple(range(0, 10)),
        (10, 11),
        tuple(range(12, 22)),
        (22, 23),
        tuple(range(24, 34)),
        (34, 35),
        tuple(range(36, 47)),
        (47,),
    )
    expected_routing = ((0, 1), (2, 3), (4, 5), (6, 7))
    assert plan.sides["student"].rank_bin_indices == expected_routing
    assert plan.sides["teacher"].rank_bin_indices == expected_routing
    assert len(plan.bins) == 8
    assert plan.equalization_splits == 1
    assert sorted(item_id for bin_items in plan.bins for item_id in bin_items) == list(
        range(48)
    )
    for side in plan.sides.values():
        assert all(tokens <= side.capacity for tokens in side.physical_tokens_by_bin)
