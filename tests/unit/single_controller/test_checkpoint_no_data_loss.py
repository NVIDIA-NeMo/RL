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

"""Checkpoint save/restore must not drop prompt groups -- cases that hold today.

Pausing a run to checkpoint and resuming it should train on the same prompt
groups as never pausing at all. The matrix below is every combination that
holds right now.

Cases that do **not** hold live in ``test_checkpoint_no_data_loss_xfail.py``,
each marked with the line that drops the data. When a fix lands, its case moves
from that table into ``PASSING`` here.
"""

from __future__ import annotations

import asyncio

import pytest

from nemo_rl.algorithms.async_utils.staleness_sampler import create_sampler
from tests.unit.single_controller._checkpoint_scenarios import (
    CAPACITY,
    PARTITION,
    ROLLOUTS_PER_GROUP,
    S_ALL_COMPLETE,
    S_STALE_ONLY,
    Case,
    _fill,
    _fresh_client,
    _new_buffer,
    assert_no_data_loss,
    patch_converter,
    round_trip,
    sampler_config,
)

# ── the matrix ──────────────────────────────────────────────────────────────
# Read this as a table. Every row: restore returns every group the run still
# needs. Only ``windowed`` appears, and only with everything fully generated --
# that is the whole of what this PR recovers.

PASSING = [
    Case(S_ALL_COMPLETE, "windowed"),
    Case(S_STALE_ONLY, "windowed"),
]


@pytest.fixture(autouse=True)
def _converter(monkeypatch):
    patch_converter(monkeypatch)


@pytest.mark.parametrize("case", PASSING, ids=lambda c: c.id)
def test_restore_returns_every_group_the_run_still_needs(case, tmp_path):
    result = assert_no_data_loss(case.scenario, case.sampler, tmp_path)
    assert result.recovered == case.scenario.must_survive(), (
        "restore should return exactly the outstanding groups -- no more, no less"
    )


# ── properties that back the matrix up ──────────────────────────────────────


def test_restore_reuses_the_stored_rows_instead_of_rewriting_them(tmp_path):
    """TransferQueue keeps the tensors; the sidecar only names the rows."""
    result = round_trip(S_ALL_COMPLETE, "windowed", tmp_path)
    assert result.rows_before, "scenario should have written rows"
    assert result.rows_after == result.rows_before


def test_restored_groups_are_selectable_by_the_sampler(tmp_path):
    """Coming back is not enough -- they have to be usable on the next step."""

    async def exercise():
        dp = _fresh_client(register=True)
        buf = _new_buffer(dp)
        await _fill(buf, S_ALL_COMPLETE)
        sidecar = buf.metadata_state_dict(saved_capacity=CAPACITY)

        dp2 = _fresh_client(register=True)
        buf2 = _new_buffer(dp2)
        await buf2.load_state_dict(
            sidecar,
            max_groups=CAPACITY,
            expected_partition_id=PARTITION,
            expected_group_size=ROLLOUTS_PER_GROUP,
            expected_manifest_digest=sidecar["manifest_digest"],
        )
        sampler = create_sampler(buf2, sampler_config("windowed", 1))
        return await sampler.select(
            current_train_weight=1, min_prompt_groups=1, max_prompt_groups=3
        )

    meta, count = asyncio.run(exercise())
    assert count == 3, "the three restored groups should form the next batch"
    assert meta is not None


def test_a_stale_group_comes_back_so_the_sampler_can_decide_to_drop_it(tmp_path):
    """Restoring and then evicting is a decision; dropping at restore is a bug.

    Both look the same from outside, so this pins that the checkpoint hands the
    stale group back and leaves the staleness call to the sampler.
    """
    result = assert_no_data_loss(S_STALE_ONLY, "windowed", tmp_path)
    assert result.recovered == {"g09", "g10"}
