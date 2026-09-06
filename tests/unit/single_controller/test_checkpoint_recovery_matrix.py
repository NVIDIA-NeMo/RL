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

"""Checkpoint recovery contract across the built-in async samplers.

The six scenarios come from #3827. That PR covered windowed, weight_fifo, and
in_order; a zero-lag completed row and ready_first are included here to cover
the complete built-in recovery contract.

Unfinished prompt groups are recovered from the rollout ledger. Under sibling
recovery, completed siblings retain their captured staging rows and only their
missing siblings are redispatched.
"""

from __future__ import annotations

import pytest

from tests.unit.single_controller._checkpoint_scenarios import (
    ALL_SCENARIOS,
    FULLY_GENERATED,
    GROUPS_PER_STEP,
    ROLLOUTS_PER_GROUP,
    S_ALL_COMPLETE,
    S_LAG2,
    S_ZERO_LAG_ALL_COMPLETE,
    SAMPLERS,
    WITH_IN_FLIGHT,
    WITH_SEALED_SIBLINGS,
    Case,
    assert_completed_groups_survive,
    assert_no_data_loss,
    patch_converter,
    round_trip,
)

ALL_CASES = [
    Case(scenario, sampler) for sampler in SAMPLERS for scenario in ALL_SCENARIOS
]
COMPLETED_CASES = [
    Case(scenario, sampler) for sampler in SAMPLERS for scenario in FULLY_GENERATED
]
UNFINISHED_CASES = [
    Case(scenario, sampler) for sampler in SAMPLERS for scenario in WITH_IN_FLIGHT
]
SELECTABLE_CASES = [
    Case(scenario, sampler)
    for sampler in SAMPLERS
    for scenario in (S_ZERO_LAG_ALL_COMPLETE, S_ALL_COMPLETE, S_LAG2)
]
SEALED_SIBLING_CASES = [
    Case(scenario, sampler) for sampler in SAMPLERS for scenario in WITH_SEALED_SIBLINGS
]


@pytest.fixture(autouse=True)
def _converter(monkeypatch):
    patch_converter(monkeypatch)


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda case: case.id)
def test_completed_groups_survive_the_round_trip(case, tmp_path):
    """A pending sibling must not hide a different group that already committed."""
    result = assert_completed_groups_survive(
        case.scenario,
        case.sampler,
        tmp_path,
    )
    assert result.saved_sidecar


@pytest.mark.parametrize("case", COMPLETED_CASES, ids=lambda case: case.id)
def test_fully_generated_scenarios_have_no_data_loss(case, tmp_path):
    result = assert_no_data_loss(case.scenario, case.sampler, tmp_path)
    assert result.recovered == case.scenario.must_survive()


@pytest.mark.parametrize("case", UNFINISHED_CASES, ids=lambda case: case.id)
def test_unfinished_groups_are_owned_across_restart(case, tmp_path):
    """Handed-out unfinished groups remain recoverable."""
    assert_no_data_loss(case.scenario, case.sampler, tmp_path)


@pytest.mark.parametrize("case", SEALED_SIBLING_CASES, ids=lambda case: case.id)
def test_sealed_siblings_survive_and_only_missing_siblings_redispatch(case, tmp_path):
    """Preserve completed sibling work across samplers and gate lags 0, 1, and 2."""
    result = round_trip(case.scenario, case.sampler, tmp_path)
    partial_groups = {
        f"g{group.gid:02d}": group
        for group in case.scenario.groups
        if 0 < group.done < ROLLOUTS_PER_GROUP
        and not group.evicted
        and group.gid not in case.scenario.trained
    }

    assert partial_groups
    for group_id, group in partial_groups.items():
        expected_sealed = tuple(range(group.done))
        expected_missing = tuple(range(group.done, ROLLOUTS_PER_GROUP))
        assert result.sealed_before[group_id] == expected_sealed
        assert result.sealed_after[group_id] == expected_sealed
        assert result.redispatched[group_id] == expected_missing
    assert result.staging_rows_before
    assert result.staging_rows_after_restore == result.staging_rows_before


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda case: case.id)
def test_restore_preserves_sampler_stamps(case, tmp_path):
    """Every restored group retains the keys its sampler uses for selection."""
    result = round_trip(case.scenario, case.sampler, tmp_path)

    assert result.stamps == case.scenario.expected_stamps()


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_restore_reuses_the_same_tq_rows(sampler, tmp_path):
    """The replay sidecar restores the index; it must not duplicate tensor rows."""
    scenario = FULLY_GENERATED[0]
    result = round_trip(scenario, sampler, tmp_path)

    assert result.rows_before
    assert result.rows_after == result.rows_before


@pytest.mark.parametrize("sampler", SAMPLERS)
def test_intentionally_evicted_group_is_not_resurrected(sampler, tmp_path):
    """Recovery restores owned work, not work the sampler deliberately discarded."""
    scenario = next(s for s in WITH_IN_FLIGHT if "evicted" in s.name)
    result = round_trip(scenario, sampler, tmp_path)

    assert "g10" not in result.recovered


@pytest.mark.parametrize("case", SELECTABLE_CASES, ids=lambda case: case.id)
def test_restored_groups_are_selectable(case, tmp_path):
    """Each sampler can select the restored batch at gate lags zero, one, and two."""
    first_outstanding = next(
        group
        for group in case.scenario.groups
        if group.gid not in case.scenario.trained and not group.evicted
    )
    current_train_weight = (
        first_outstanding.target
        if case.sampler == "in_order"
        else first_outstanding.weight
    )
    assert current_train_weight is not None

    result = round_trip(
        case.scenario,
        case.sampler,
        tmp_path,
        select_current_train_weight=current_train_weight,
    )

    assert result.selected_count == GROUPS_PER_STEP
    assert result.selected == {"g12", "g13", "g14"}
