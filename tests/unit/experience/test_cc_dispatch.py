# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve original CC actor replay guards under RL-owned selection."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import nemo_rl.environments.nemo_gym as gym_environment
from nemo_rl.environments.nemo_gym import GymTransportError, NemoGym
from nemo_rl.experience.rollout_manager import RolloutRetryPolicy
from nemo_rl.utils.timer import Timer
from tests.unit.experience.test_rollout_generation_failures import _make_gym_impl
from tests.unit.experience.test_rollout_manager import (
    _FakeCaptureBuffer,
    _make_capture_manager,
)

pytestmark = pytest.mark.nemo_gym


def _env(harness):
    env = object.__new__(NemoGym.__ray_metadata__.modified_class)
    env._context_compaction = True
    env.cfg = {}

    async def control(method, path):
        assert method == "GET"
        capture_id = path.split("/")[-2]
        return await harness.ledger.manifest(capture_id)

    env._control = AsyncMock(side_effect=control)
    return env


def test_cc_row_stream_disables_ray_replay_and_row_redispatch():
    class Remote:
        def __init__(self):
            self.calls = 0
            self.call_options = None

        def options(self, **kwargs):
            self.call_options = kwargs
            return self

        async def remote(self, *_args, per_prompt=False):
            assert per_prompt is False
            self.calls += 1
            raise GymTransportError("mutating agent ran; stream response lost")
            yield  # Ray streaming method shape, deliberately no result after failure.

    remote = Remote()
    impl = _make_gym_impl(remote, row_attempts=5)
    impl._context_compaction = True
    with pytest.raises(GymTransportError, match="stream response lost"):
        asyncio.run(
            impl._run_rollouts(
                [{"_rowidx": 0, "agent_ref": {"name": "agent"}}], Timer(), "test"
            )
        )
    assert remote.calls == 1
    assert remote.call_options == {"num_returns": "streaming", "max_task_retries": 0}


@pytest.mark.parametrize("failure_type", [GymTransportError, ValueError, TimeoutError])
def test_cc_group_never_redispatches_even_with_large_ordinary_budgets(failure_type):
    attempts = []

    async def execute_then_lose_response(_sample):
        attempts.append("mutation applied")
        raise failure_type("response lost")

    policy = RolloutRetryPolicy(
        max_infra_attempts=5,
        max_data_attempts=4,
        max_gym_row_attempts=3,
        backoff_base_s=0,
        max_backoff_s=0,
        max_skipped_prompts=10,
        max_consecutive_dropped_prompts=10,
    )
    buffer = _FakeCaptureBuffer()
    manager = _make_capture_manager(
        buffer, retry_policy=policy, on_run=execute_then_lose_response
    )
    manager._context_compaction = True
    with pytest.raises(failure_type, match="response lost"):
        asyncio.run(manager.generate_for_finalization({"idx": 1}))
    assert attempts == ["mutation applied"]
    assert len(buffer.abort_calls) == 1
    assert buffer.commit_calls == []
    assert manager.stats.skipped == 0


@pytest.mark.parametrize("cc", [False, True])
def test_actor_creation_preserves_foundation_no_replay_defaults(monkeypatch, cc):
    # The foundation now disables replay on the actor declaration for all runs.
    assert NemoGym._default_options["max_restarts"] == 0
    assert NemoGym._default_options["max_task_retries"] == 0
    options = MagicMock()
    actor = options.return_value.remote.return_value
    monkeypatch.setattr(
        gym_environment, "build_nemo_gym_config", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        gym_environment, "make_actor_runtime_env", lambda _: {"existing": "runtime"}
    )
    monkeypatch.setattr(gym_environment, "NemoGym", MagicMock(options=options))
    monkeypatch.setattr(gym_environment.ray, "get", lambda value: value)
    created = gym_environment.spinup_nemo_gym_actor(
        {"nemo_gym": {}},
        base_urls=[],
        model_name="model",
        tokenizer=None,
        enable_router_replay=False,
        use_fastokens=False,
        token_capture={"enabled": True, "context_compaction": cc},
    )
    assert created is actor
    expected = {"runtime_env": {"existing": "runtime"}}
    options.assert_called_once_with(**expected)
    actor._spinup.remote.assert_called_once()
    actor.set_tokenizer.remote.assert_called_once_with(None)
