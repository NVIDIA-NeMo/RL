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

import asyncio
from collections.abc import AsyncIterator
from typing import Any

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


class _TaskSourceResolvingRolloutHelper:
    """Mimic Gym's synchronous task_source-to-agent_ref resolution."""

    def run_examples(
        self, examples: list[dict[str, Any]], head_server_config: str
    ) -> list[Any]:
        assert head_server_config == "head-server"
        assert all("agent_ref" not in example for example in examples)
        for example in examples:
            example["agent_ref"] = {
                "type": "responses_api_agents",
                "name": "workplace_assistant_simple_agent",
            }
        return []


class _TaskSourceResolvingRolloutHelperWithResult(_TaskSourceResolvingRolloutHelper):
    def run_examples(
        self, examples: list[dict[str, Any]], head_server_config: str
    ) -> list[Any]:
        super().run_examples(examples, head_server_config)

        async def completed(example):
            return example, {}

        return [completed(example) for example in examples]


async def _drain(async_generator: AsyncIterator[Any]) -> None:
    async for _ in async_generator:
        pass


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
        run_helper = _FakeRunHelper()
        env.rh = run_helper
        env.shutdown()
        env.shutdown()
        assert run_helper.shutdowns == 1
        assert env.rh is None


def test_run_rollouts_resolves_task_source_before_reading_agent_ref():
    """Gym 0.15 collated rows are task_source-routed until run_examples."""
    env = _unspun()
    env.rh = _FakeRunHelper()
    env._tokenizer = object()
    env.head_server_config = "head-server"
    env.rch = _TaskSourceResolvingRolloutHelper()

    rows = [{"task_source": "workplace_assistant"}]
    asyncio.run(_drain(env.run_rollouts(rows, "timing/test")))

    assert rows[0]["agent_ref"]["name"] == "workplace_assistant_simple_agent"


def test_run_rollouts_echoes_resolved_agent_ref_with_streamed_result():
    """The caller's serialized row copy cannot observe actor-local mutation."""
    env = _unspun()
    env.rh = _FakeRunHelper()
    env._tokenizer = object()
    env.head_server_config = "head-server"
    env.rch = _TaskSourceResolvingRolloutHelperWithResult()
    env._postprocess_nemo_gym_to_nemo_rl_result = lambda *_args, **_kwargs: {
        "message_log": []
    }

    async def collect():
        return [
            item
            async for item in env.run_rollouts(
                [{"task_source": "workplace_assistant", "_rowidx": 0}],
                "timing/test",
            )
        ]

    streamed = asyncio.run(collect())
    assert streamed[0][1] == {
        "type": "responses_api_agents",
        "name": "workplace_assistant_simple_agent",
    }
