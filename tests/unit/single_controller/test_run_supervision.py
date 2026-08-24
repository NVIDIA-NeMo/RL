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

"""run() must keep supervising the watchdog for as long as the train pump lives.

The rollout pump finishing first is the *normal* end-of-data path, not a
failure -- the train pump then drains the groups already committed. The
watchdog has to stay armed across that drain, because a wedged collective
during it is exactly the kind of stall nothing else detects.
"""

import asyncio
from types import SimpleNamespace

import pytest

from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.experience.failures import RolloutStall


def _bare_actor():
    """An actor with __init__ skipped, stubbed down to what run() touches."""
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._gen_fleet = None  # no fleet -> run() creates no probe task
    ctrl._train_steps = 0
    ctrl._trainer_version = 0
    ctrl._weight_synchronizer = SimpleNamespace(shutdown=lambda: None)
    ctrl._logger = SimpleNamespace(finish=lambda: None)
    ctrl._checkpointer = SimpleNamespace(shutdown=lambda: None)

    async def _noop():
        return None

    ctrl._sync_weights = _noop
    ctrl._maybe_restore_replay_buffer = _noop
    ctrl._maybe_restore_replacement_reserve = _noop
    return ctrl


async def _exhausts():
    """A rollout pump that reaches the end of the data and returns."""
    await asyncio.sleep(0)


async def _wedged():
    """A train pump blocked in a collective that never returns."""
    await asyncio.Event().wait()


def _stalls_after(delay):
    async def _watchdog():
        await asyncio.sleep(delay)
        raise RolloutStall("simulated stall")

    return _watchdog


def test_watchdog_still_aborts_after_the_rollout_pump_exhausts():
    """The drain phase is the window this regressed in: rollout done, train wedged."""
    ctrl = _bare_actor()
    ctrl._rollout_pump = _exhausts
    ctrl._train_pump = _wedged
    ctrl._stall_watchdog_pump = _stalls_after(0.02)

    with pytest.raises(RolloutStall):
        asyncio.run(asyncio.wait_for(ctrl.run(), timeout=5.0))


def test_watchdog_aborts_while_the_rollout_pump_is_still_running():
    """The pre-existing path, kept as a guard against fixing one and breaking the other."""
    ctrl = _bare_actor()
    ctrl._rollout_pump = _wedged
    ctrl._train_pump = _wedged
    ctrl._stall_watchdog_pump = _stalls_after(0.02)

    with pytest.raises(RolloutStall):
        asyncio.run(asyncio.wait_for(ctrl.run(), timeout=5.0))


def test_a_clean_run_still_returns_its_summary():
    """Both pumps finish, the watchdog never fires: run() returns normally."""
    ctrl = _bare_actor()
    ctrl._train_steps = 7
    ctrl._trainer_version = 7
    ctrl._rollout_pump = _exhausts
    ctrl._train_pump = _exhausts

    async def _quiet_watchdog():
        await asyncio.Event().wait()

    ctrl._stall_watchdog_pump = _quiet_watchdog

    result = asyncio.run(asyncio.wait_for(ctrl.run(), timeout=5.0))
    assert result == {"train_steps": 7, "trainer_version": 7}
