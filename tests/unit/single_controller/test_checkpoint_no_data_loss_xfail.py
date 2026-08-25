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

"""Checkpoint cases that drop prompt groups today. Each one is real data loss.

Same property as ``test_checkpoint_no_data_loss.py``: pause, resume, and the run
should train on the same prompt groups as if it had never paused. This is the
table of combinations where it does not.

Marks are ``strict=True``, so the moment a fix lands the case turns into an
XPASS failure and whoever fixed it is told to move that row into ``PASSING`` in
the other module. Nothing here is a guess -- each row names the line that drops
the data.

Two independent causes:

* ``IN_FLIGHT`` -- ``reserve()`` books a slot marked not-ready and only
  ``commit()`` flips it, and ``metadata_state_dict`` skips anything not ready.
  A group that has generated some, or even all, of its rollouts without
  committing is skipped exactly like one that never started. Hits every sampler.
* ``GATED`` -- ``in_order`` and ``weight_fifo`` declare
  ``supports_buffer_checkpoint = False``, so no sidecar is written at all and
  the restore returns early. Hits every group, finished or not.
"""

from __future__ import annotations

import pytest

from tests.unit.single_controller._checkpoint_scenarios import (
    ALL_SCENARIOS,
    GATED_SAMPLERS,
    S_ALL_COMPLETE,
    S_EVICTED,
    S_LAG2,
    S_PARTIAL,
    S_TRAINED_OUT_OF_ORDER,
    Case,
    assert_no_data_loss,
    patch_converter,
    round_trip,
)

IN_FLIGHT = (
    "in-flight groups are never saved: reserve() marks the slot not-ready "
    "(replay_buffer.py:910) and metadata_state_dict skips it (:1093)"
)
GATED = (
    "gated sampler declares supports_buffer_checkpoint=False, so no sidecar is "
    "written (single_controller.py:981) and the restore returns early (:322)"
)

# ── the matrix ──────────────────────────────────────────────────────────────
# Read this as a table of setups. Left column is what the buffer held when the
# snapshot was taken; middle is the sampler; right is what loses the data.

EXPECTED_TO_FAIL = [
    # The recovering sampler still drops anything mid-generation.
    # S_PARTIAL is the sharpest: group 13 finished both rollouts and is dropped
    # anyway, because the commit had not run when the snapshot was taken.
    Case(S_PARTIAL, "windowed", IN_FLIGHT),
    Case(S_LAG2, "windowed", IN_FLIGHT),
    Case(S_EVICTED, "windowed", IN_FLIGHT),
    Case(S_TRAINED_OUT_OF_ORDER, "windowed", IN_FLIGHT),
    # Gated samplers keep nothing at all, however complete. This is the
    # behaviour change the review flagged: before this PR the buffer was saved
    # for every sampler and restored whenever the sampler name matched.
    *(Case(s, sampler, GATED) for sampler in GATED_SAMPLERS for s in ALL_SCENARIOS),
]


def _param(case: Case):
    return pytest.param(
        case, id=case.id, marks=pytest.mark.xfail(strict=True, reason=case.why)
    )


@pytest.fixture(autouse=True)
def _converter(monkeypatch):
    patch_converter(monkeypatch)


@pytest.mark.parametrize("case", [_param(c) for c in EXPECTED_TO_FAIL])
def test_restore_drops_groups_the_run_still_needs(case, tmp_path):
    assert_no_data_loss(case.scenario, case.sampler, tmp_path)


# ── why the rows above fail ─────────────────────────────────────────────────
# These pass on purpose. They pin the two causes, so a fix that changes either
# one fails here and points at the table to re-check.


def test_gated_samplers_write_no_sidecar_at_all(tmp_path):
    for sampler in GATED_SAMPLERS:
        result = round_trip(S_ALL_COMPLETE, sampler, tmp_path / sampler)
        assert not result.saved_sidecar, f"{sampler} unexpectedly wrote a sidecar"
        assert result.recovered == set(), f"{sampler} unexpectedly restored groups"

    windowed = round_trip(S_ALL_COMPLETE, "windowed", tmp_path / "windowed")
    assert windowed.saved_sidecar, "windowed should still write one"


def test_the_tensors_survive_even_when_the_index_does_not(tmp_path):
    """The loss is the index, not the rows -- so a fix can be index-only.

    TransferQueue still holds every committed row after a gated-sampler restore.
    The groups are unreachable only because nothing records which rows belong to
    which group.
    """
    result = round_trip(S_ALL_COMPLETE, "in_order", tmp_path)
    assert result.rows_before, "scenario should have written rows"
    assert result.rows_after == result.rows_before
    assert result.recovered == set(), "but no group index came back"
