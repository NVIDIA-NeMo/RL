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

"""Groups that came back before this change and do not now.

**These are deliberately not xfail.** A gap that was always there is a missing
feature, and those live in ``test_checkpoint_no_data_loss_xfail.py``. Losing
something the previous code kept is a different thing, and marking it
"expected" would hide exactly the signal worth having.

What the previous code did, on the merge base:

* the buffer was saved on **every** checkpoint, with no sampler capability gate
  (``single_controller.py:802`` -- ``self._buffer.state_dict(...)``);
* the restore ran for **any** sampler, gated only on the saved sampler name
  matching the configured one (``:270``), which always holds on a same-sampler
  resume;
* ``state_dict`` saved the committed slots and dropped in-flight ones, saying
  so outright: *"Unready reservations are in-flight rollouts and are dropped,
  matching legacy semantics."*

So every fully generated group used to survive a restart under every sampler.
After this change, ``in_order`` and ``weight_fifo`` declare
``supports_buffer_checkpoint = False``, so no sidecar is written
(``single_controller.py:981``) and the restore returns early (``:322``) --
losing groups that used to come back.

The assertion is the weaker of the two in the harness: not "everything the run
needs" but "nothing that used to come back". Nothing here depends on the
in-flight gap, so these fail on the fully-generated scenarios too.

If dropping this is a deliberate trade -- one durable store instead of two --
then the right resolution is not to xfail these but to decide the old
behaviour is not coming back, and say so where a user configuring
``in_order`` will read it.
"""

from __future__ import annotations

import pytest

from tests.unit.single_controller._checkpoint_scenarios import (
    ALL_SCENARIOS,
    GATED_SAMPLERS,
    S_ALL_COMPLETE,
    Case,
    assert_no_regression,
    patch_converter,
    round_trip,
)

# ── the matrix ──────────────────────────────────────────────────────────────
# Every gated sampler, every scenario. The fully-generated rows are the
# cleanest evidence: nothing about them is in flight, so the only reason they
# do not come back is the capability gate.

REGRESSED = [
    Case(scenario, sampler)
    for sampler in GATED_SAMPLERS
    for scenario in ALL_SCENARIOS
]


@pytest.fixture(autouse=True)
def _converter(monkeypatch):
    patch_converter(monkeypatch)


@pytest.mark.parametrize("case", REGRESSED, ids=lambda c: c.id)
def test_restore_still_returns_what_it_used_to(case, tmp_path):
    assert_no_regression(case.scenario, case.sampler, tmp_path)


def test_the_rows_are_still_in_transfer_queue_after_a_gated_restore(tmp_path):
    """The regression is the index, not the tensors -- so a fix can be index-only.

    TransferQueue still holds every committed row. The groups are unreachable
    only because nothing wrote down which rows belong to which group.
    """
    result = round_trip(S_ALL_COMPLETE, "in_order", tmp_path)
    assert result.rows_before, "scenario should have written rows"
    assert result.rows_after == result.rows_before
    assert result.recovered == set(), "but no group index came back"
