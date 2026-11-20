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

"""Rank layout for a refit communicator rebuilt over the surviving shards.

Tested hard because it cannot be tested any other way here: exercising a real shard loss
needs at least three GPUs, so on a two-GPU workstation this arithmetic is the whole of
what can be verified. An off-by-one does not crash -- it silently points a receiver at
the wrong slice of the broadcast.
"""

import pytest

from nemo_rl.weight_sync.membership import (
    NoSurvivingShards,
    desired_membership,
    plan_refit_membership,
)


def _plan(
    surviving, *, dp_size=4, total_gen_workers=4, train_world_size=8, shard_widths=None
):
    return plan_refit_membership(
        surviving_shards=surviving,
        dp_size=dp_size,
        total_gen_workers=total_gen_workers,
        train_world_size=train_world_size,
        shard_widths=shard_widths,
    )


class TestWholeFleet:
    def test_nothing_moves_when_every_shard_survives(self):
        plan = _plan([0, 1, 2, 3])

        assert plan.shard_prefixes == {0: 0, 1: 1, 2: 2, 3: 3}
        assert plan.world_size == 12

    def test_a_single_shard_fleet_is_valid(self):
        plan = _plan([0], dp_size=1, total_gen_workers=1)

        assert plan.shard_prefixes == {0: 0}
        assert plan.world_size == 9


class TestSurvivorsAreCompacted:
    """The property the nccl_reshard mesh depends on: no holes.

    Its destination mesh is torch.arange(offset, offset + num_gpus), so a gap where the
    dead shard used to be misaligns every parameter's placements rather than erroring.
    """

    def test_losing_a_middle_shard_closes_the_gap(self):
        plan = _plan([0, 1, 3])

        assert plan.shard_prefixes == {0: 0, 1: 1, 3: 2}
        assert plan.world_size == 11

    def test_losing_the_first_shard_shifts_everyone_down(self):
        plan = _plan([1, 2, 3])

        assert plan.shard_prefixes == {1: 0, 2: 1, 3: 2}

    def test_prefixes_are_always_a_contiguous_range_from_zero(self):
        for surviving in ([0, 2], [1, 3], [3], [0, 1, 2, 3], [2, 3]):
            plan = _plan(surviving)
            width = plan.workers_per_shard[0]
            expected = list(range(0, len(surviving) * width, width))
            assert sorted(plan.shard_prefixes.values()) == expected, surviving

    def test_input_order_does_not_change_the_layout(self):
        """Survivors may arrive in discovery order; the layout must not depend on it."""
        assert _plan([3, 1, 0]).shard_prefixes == _plan([0, 1, 3]).shard_prefixes

    def test_duplicates_are_ignored(self):
        assert _plan([1, 1, 2]).shard_prefixes == {1: 0, 2: 1}


class TestMultiWorkerShards:
    """A shard is tp x pp workers, so prefixes step by that, not by one."""

    def test_prefixes_step_by_the_shard_width(self):
        plan = _plan([0, 1, 2, 3], dp_size=4, total_gen_workers=8)

        assert plan.workers_per_shard == (2, 2, 2, 2)
        assert plan.shard_prefixes == {0: 0, 1: 2, 2: 4, 3: 6}
        assert plan.world_size == 8 + 8

    def test_losing_a_shard_frees_its_whole_width(self):
        plan = _plan([0, 2, 3], dp_size=4, total_gen_workers=8)

        assert plan.shard_prefixes == {0: 0, 2: 2, 3: 4}
        assert plan.world_size == 8 + 6

    def test_a_tp4_fleet_of_two_shards(self):
        plan = _plan([1], dp_size=2, total_gen_workers=8, train_world_size=16)

        assert plan.workers_per_shard == (4, 4)
        assert plan.shard_prefixes == {1: 0}
        assert plan.world_size == 20


class TestHeterogeneousShards:
    """Shards of different widths: a prefill shard at TP 4 next to decode shards at TP 2.

    Prefixes are cumulative over the survivors rather than ``position * width``, and
    ``worker_offset`` counts every lower-indexed shard, dead or alive, because the
    worker list never shrinks.
    """

    @pytest.mark.parametrize(
        ("surviving", "prefixes", "world_size"),
        [
            ([0, 1, 2], {0: 0, 1: 4, 2: 6}, 8 + 8),
            ([0, 2], {0: 0, 2: 4}, 8 + 6),
            ([1, 2], {1: 0, 2: 2}, 8 + 4),
            ([2, 0, 1], {0: 0, 1: 4, 2: 6}, 8 + 8),
        ],
        ids=["all", "middle_absent", "first_absent", "discovery_order"],
    )
    def test_prefixes_accumulate_over_the_survivors(
        self, surviving, prefixes, world_size
    ):
        plan = _plan(surviving, dp_size=3, total_gen_workers=8, shard_widths=(4, 2, 2))

        assert plan.shard_prefixes == prefixes
        assert plan.world_size == world_size
        assert plan.workers_per_shard == (4, 2, 2)
        assert [plan.worker_offset(shard) for shard in range(3)] == [0, 4, 6]

    def test_equal_widths_match_the_evenly_sharded_plan(self):
        """Given explicitly, equal widths lay ranks out exactly as the even split does."""
        plan = _plan([0, 2, 3], total_gen_workers=8, shard_widths=(2, 2, 2, 2))
        even = _plan([0, 2, 3], total_gen_workers=8)

        assert plan.shard_prefixes == even.shard_prefixes
        assert plan.world_size == even.world_size
        assert plan.workers_per_shard == even.workers_per_shard == (2, 2, 2, 2)
        assert plan.worker_offset(3) == even.worker_offset(3) == 6

    @pytest.mark.parametrize(
        ("shard_widths", "match"),
        [
            ((4, 2), "2 entries for dp_size=3"),
            ((4, 2, 1), "sum to 7, not total_gen_workers=8"),
            ((4, 4, 0), "must be positive"),
        ],
        ids=["count", "sum", "zero"],
    )
    def test_inconsistent_widths_are_refused(self, shard_widths, match):
        with pytest.raises(ValueError, match=match):
            _plan([0, 1, 2], dp_size=3, total_gen_workers=8, shard_widths=shard_widths)

    def test_desired_membership_forwards_the_widths(self):
        desired = desired_membership(
            absent_shards=[1],
            dp_size=3,
            total_gen_workers=8,
            train_world_size=8,
            shard_widths=(4, 2, 2),
        )

        assert desired == _plan(
            [0, 2], dp_size=3, total_gen_workers=8, shard_widths=(4, 2, 2)
        )


class TestTrainersAreNeverExcluded:
    def test_train_world_size_survives_a_rebuild(self):
        """Rank 0 is a trainer and is the broadcast root; losing it would break refit."""
        plan = _plan([2], train_world_size=64)

        assert plan.train_world_size == 64
        assert plan.world_size == 65


class TestRejected:
    def test_an_empty_fleet_is_refused(self):
        with pytest.raises(NoSurvivingShards):
            _plan([])

    def test_an_unevenly_sharded_fleet_is_refused(self):
        """Would silently truncate a shard's worth of ranks."""
        with pytest.raises(ValueError, match="evenly sharded"):
            _plan([0, 1], dp_size=3, total_gen_workers=8)

    @pytest.mark.parametrize("bad", [-1, 4, 99])
    def test_a_shard_outside_the_fleet_is_refused(self, bad):
        with pytest.raises(ValueError, match="outside the fleet"):
            _plan([0, bad])

    @pytest.mark.parametrize("dp_size", [0, -2])
    def test_a_non_positive_dp_size_is_refused(self, dp_size):
        with pytest.raises(ValueError, match="dp_size"):
            _plan([0], dp_size=dp_size, total_gen_workers=4)


class TestDesiredMembership:
    """The absent-set view of the same plan.

    Exists so that "who should be in the communicator" has one answer whether a shard is
    leaving or rejoining. The re-admission bug this guards against was real: keyed off
    "is anything absent", an empty absent set reads as "nothing to do", and a restarted
    shard stays excluded from the refit forever while looking perfectly healthy.
    """

    def _desired(self, absent, **kw):
        return desired_membership(
            absent_shards=absent,
            dp_size=4,
            total_gen_workers=4,
            train_world_size=8,
            **kw,
        )

    def test_no_absent_shards_is_the_whole_fleet(self):
        """Not a no-op signal -- a concrete full-fleet plan, which is what re-admission compares against."""
        assert self._desired([]).shard_prefixes == {0: 0, 1: 1, 2: 2, 3: 3}

    def test_it_agrees_with_the_survivor_view(self):
        assert self._desired([1]) == _plan([0, 2, 3])

    def test_a_rejoining_shard_changes_the_plan_back(self):
        """The whole point: absent -> present must be visible as a different membership."""
        while_absent = self._desired([2])
        after_rejoin = self._desired([])

        assert while_absent != after_rejoin
        assert 2 not in while_absent.shard_prefixes
        assert 2 in after_rejoin.shard_prefixes

    def test_losing_everyone_is_refused(self):
        with pytest.raises(NoSurvivingShards):
            self._desired([0, 1, 2, 3])
