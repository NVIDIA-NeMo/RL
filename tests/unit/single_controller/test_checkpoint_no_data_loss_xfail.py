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

"""Gaps that were here before this change and are still here.

Only one cause is left in this file: **a prompt group that has not committed is
never saved.** That was true before this change too -- the old ``state_dict``
said so in as many words, *"Unready reservations are in-flight rollouts and are
dropped, matching legacy semantics"* -- so nothing here is a step backwards.
Things this change made worse live in ``test_checkpoint_regressions.py`` and
are **not** xfail.

The sharpest row is ``windowed::lag1-next-step-partly-generated``: group 13 has
finished both of its rollouts and is dropped anyway, because the commit had not
run when the snapshot was taken. A group is reserved or committed, with nothing
in between, so "one rollout still running" and "not started" are the same thing
to a checkpoint.

**Why these assert on presence, not on state.** Closing this gap needs a
durable copy of a half-finished group's tokens, which is what the token-capture
work in #3456 is building. How a restore then presents such a group is an open
design question -- it might come back already committed, with the missing
rollouts regenerated before the save, or as a reserved slot for the run to
finish. These tests deliberately do not care. ``recovered`` counts a group if
the restored buffer knows about it **at all**, so either design flips these to
XPASS. ``strict=True`` then turns that XPASS into a failure, and whoever built
it is told to move the row into ``PASSING``.
"""

from __future__ import annotations

import pytest

from tests.unit.single_controller._checkpoint_scenarios import (
    S_EVICTED,
    S_LAG2,
    S_PARTIAL,
    S_TRAINED_OUT_OF_ORDER,
    Case,
    assert_no_data_loss,
    assert_no_regression,
    patch_converter,
    round_trip,
)

IN_FLIGHT = (
    "a group that has not committed is never saved: reserve() marks the slot "
    "not-ready (replay_buffer.py:910) and metadata_state_dict skips it (:1093). "
    "Pre-existing -- the old state_dict dropped in-flight reservations too."
)

# ── the matrix ──────────────────────────────────────────────────────────────
# The recovering sampler only. Gated samplers are excluded on purpose: they
# fail for a second, worse reason, and that reason is a regression, so those
# rows belong in test_checkpoint_regressions.py.

EXPECTED_TO_FAIL = [
    Case(S_PARTIAL, "windowed", IN_FLIGHT),
    Case(S_LAG2, "windowed", IN_FLIGHT),
    Case(S_EVICTED, "windowed", IN_FLIGHT),
    Case(S_TRAINED_OUT_OF_ORDER, "windowed", IN_FLIGHT),
]


def _param(case: Case):
    return pytest.param(
        case, id=case.id, marks=pytest.mark.xfail(strict=True, reason=case.why)
    )


@pytest.fixture(autouse=True)
def _converter(monkeypatch):
    patch_converter(monkeypatch)


@pytest.mark.parametrize("case", [_param(c) for c in EXPECTED_TO_FAIL])
def test_restore_drops_groups_that_had_not_committed(case, tmp_path):
    assert_no_data_loss(case.scenario, case.sampler, tmp_path)


@pytest.mark.parametrize("case", EXPECTED_TO_FAIL, ids=lambda c: c.id)
def test_but_nothing_that_used_to_come_back_was_lost(case, tmp_path):
    """The same rows, held to the weaker bar -- and they pass.

    This is what makes the xfail above honest: under ``windowed`` every group
    the old code returned still comes back. The only losses are ones the old
    code lost too.
    """
    assert_no_regression(case.scenario, case.sampler, tmp_path)


def test_a_fully_generated_group_is_dropped_when_the_commit_had_not_run(tmp_path):
    """Pins the exact shape of the gap, so a partial fix cannot pass by accident.

    In ``S_PARTIAL`` group 13 committed and comes back; 12 and 14 did not and do
    not. If a future change brings back 12 or 14 in any form, ``recovered``
    grows and the xfail rows above flip to XPASS.
    """
    result = round_trip(S_PARTIAL, "windowed", tmp_path)
    assert result.recovered == {"g13"}
    assert result.ready == {"g13"}, "the one committed group comes back ready"
    assert result.pending == set(), "nothing comes back awaiting completion yet"
