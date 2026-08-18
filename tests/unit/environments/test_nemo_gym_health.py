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

"""NemoGym liveness surface.

Two things are covered. First, health_check forwards to NeMo-Gym's own RunHelper.poll --
the check Gym already implements and calls every 60s from `gym env start`, but which
NeMo-RL never ran because it only calls rh.start. Without it a dead tool server shows up
as unexplained rollout timeouts instead of a named process.

Second, an actor that never spun up says so. Ray recreates a restarted actor through
__init__ only, which does not start the Gym servers, so a restarted NemoGym reaches that
state and previously surfaced it as an AttributeError from deep inside a rollout.
"""

import pytest

from nemo_rl.environments.nemo_gym import NemoGym

# NemoGym is a Ray actor; grab the plain class so these run without a cluster.
NemoGymClass = NemoGym.__ray_metadata__.modified_class


def _unspun() -> NemoGymClass:
    """A NemoGym exactly as Ray would recreate it after a restart."""
    return NemoGymClass(
        {"model_name": "m", "base_urls": [], "initial_global_config_dict": {}}
    )


class _FakeRunHelper:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.polls = 0
        self.shutdowns = 0

    def poll(self) -> None:
        self.polls += 1
        if self.error is not None:
            raise self.error

    def shutdown(self) -> None:
        self.shutdowns += 1


class TestHealthCheck:
    def test_a_healthy_gym_polls_the_run_helper(self):
        env = _unspun()
        env.rh = _FakeRunHelper()
        env.health_check()
        assert env.rh.polls == 1

    def test_a_dead_subprocess_server_propagates_with_its_name(self):
        env = _unspun()
        env.rh = _FakeRunHelper(
            RuntimeError("Process `workplace_assistant` finished unexpectedly!")
        )
        with pytest.raises(RuntimeError, match="workplace_assistant"):
            env.health_check()


class TestUnspunActor:
    def test_health_check_explains_the_restarted_actor_state(self):
        with pytest.raises(RuntimeError, match="_spinup\\(\\) was never called"):
            _unspun().health_check()

    def test_run_rollouts_refuses_rather_than_raising_attribute_error(self):
        env = _unspun()
        with pytest.raises(RuntimeError, match="_spinup\\(\\) was never called"):
            # run_rollouts is an async generator, so the guard fires on first advance.
            gen = env.run_rollouts([{"agent_ref": {"name": "a"}}], None, "timing/x")
            import asyncio

            asyncio.run(anext(gen))

    def test_shutdown_is_a_noop_so_teardown_does_not_mask_the_real_error(self):
        """shutdown() runs in a finally block; it must not raise over a training error."""
        _unspun().shutdown()

    def test_shutdown_still_forwards_when_spun_up(self):
        env = _unspun()
        env.rh = _FakeRunHelper()
        env.shutdown()
        assert env.rh.shutdowns == 1
