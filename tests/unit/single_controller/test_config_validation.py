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

"""Unit tests for the SingleController capacity/concurrency config validators."""

from __future__ import annotations

import pytest

from nemo_rl.algorithms.single_controller_utils.config import (
    AsyncRLConfig,
    MasterConfig,
    validate_gym_actor_concurrency,
)


def _master_config(
    *,
    env: dict,
    max_inflight_prompts: int,
) -> MasterConfig:
    """Build the two sections validate_gym_actor_concurrency reads."""
    return MasterConfig.model_construct(
        env=env,
        async_rl=AsyncRLConfig(
            max_inflight_prompts=max_inflight_prompts,
            max_buffered_rollouts=max_inflight_prompts * 2,
        ),
    )


class TestValidateGymActorConcurrency:
    def test_no_op_without_a_gym_env(self) -> None:
        validate_gym_actor_concurrency(
            _master_config(env={"math": {}}, max_inflight_prompts=5120)
        )

    def test_no_op_when_env_is_absent_entirely(self) -> None:
        """model_construct only fills fields with defaults, and `env` has none."""
        master_config = _master_config(env={}, max_inflight_prompts=5120)
        del master_config.env
        assert not hasattr(master_config, "env")

        validate_gym_actor_concurrency(master_config)

    def test_no_op_when_max_concurrency_is_unset(self) -> None:
        """Unset means the actor derives its concurrency from the in-flight cap."""
        validate_gym_actor_concurrency(
            _master_config(env={"nemo_gym": {}}, max_inflight_prompts=5120)
        )

    def test_rejects_max_concurrency_below_inflight_cap(self) -> None:
        # Ray's silent default against the 6K in-flight depth: the pair that
        # produced zero training steps and zero rollout groups.
        master_config = _master_config(
            env={"nemo_gym": {"max_concurrency": 1000}},
            max_inflight_prompts=5120,
        )

        with pytest.raises(ValueError, match="below the rollout fan-in"):
            validate_gym_actor_concurrency(master_config)

    def test_accepts_max_concurrency_above_inflight_cap(self) -> None:
        validate_gym_actor_concurrency(
            _master_config(
                env={"nemo_gym": {"max_concurrency": 6144}},
                max_inflight_prompts=5120,
            )
        )
