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
in_order; ready_first is included here because it now advertises the same
completed-buffer recovery capability.

Normal test runs keep the unfinished-group rows as strict xfails so #3480 can
land without claiming partial-rollout recovery. During development of the
dispatch ledger, run this file with --runxfail: those rows become the red TDD
contract and must all pass before the xfail marks are removed.
"""

from __future__ import annotations

import pytest

from tests.unit.single_controller._checkpoint_scenarios import (
    ALL_SCENARIOS,
    FULLY_GENERATED,
    SAMPLERS,
    WITH_IN_FLIGHT,
    Case,
    assert_completed_groups_survive,
    assert_no_data_loss,
    patch_converter,
    round_trip,
)

ALL_CASES = [
    Case(scenario, sampler)
    for sampler in SAMPLERS
    for scenario in ALL_SCENARIOS
]
COMPLETED_CASES = [
    Case(scenario, sampler)
    for sampler in SAMPLERS
    for scenario in FULLY_GENERATED
]
UNFINISHED_CASES = [
    Case(
        scenario,
        sampler,
        "The dataloader has advanced, but metadata_state_dict() omits every "
        "ready=False reservation. A dispatch ledger and recovery pump must "
        "redispatch the prompt group after restart.",
    )
    for sampler in SAMPLERS
    for scenario in WITH_IN_FLIGHT
]


def _known_gap(case: Case):
    return pytest.param(
        case,
        id=case.id,
        marks=pytest.mark.xfail(strict=True, reason=case.why),
    )


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


@pytest.mark.parametrize("case", [_known_gap(case) for case in UNFINISHED_CASES])
def test_unfinished_groups_are_owned_across_restart(case, tmp_path):
    """Desired end state: handed-out unfinished groups remain recoverable."""
    assert_no_data_loss(case.scenario, case.sampler, tmp_path)


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
