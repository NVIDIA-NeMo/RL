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

from types import SimpleNamespace

import pytest

from nemo_rl.algorithms.single_controller_utils.config import (
    AsyncRLConfig,
    GRPOConfig,
    MasterConfig,
    validate_gym_actor_concurrency,
    validate_single_controller_config,
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


def _gym_env(nemo_gym: dict) -> dict:
    """A gym env block, which only counts when the run is taking that path."""
    return {"should_use_nemo_gym": True, "nemo_gym": nemo_gym}


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

    def test_no_op_when_the_run_is_not_taking_the_gym_path(self) -> None:
        """A leftover gym block is inert on a native run, so it must not fail it."""
        validate_gym_actor_concurrency(
            _master_config(
                env={"should_use_nemo_gym": False, "nemo_gym": {"max_concurrency": 1}},
                max_inflight_prompts=5120,
            )
        )

    def test_no_op_when_max_concurrency_is_unset(self) -> None:
        """Unset means the actor derives its concurrency from the in-flight cap."""
        validate_gym_actor_concurrency(
            _master_config(env=_gym_env({}), max_inflight_prompts=5120)
        )

    def test_rejects_max_concurrency_below_inflight_cap(self) -> None:
        # Ray's silent default against the 6K in-flight depth: the pair that
        # produced zero training steps and zero rollout groups.
        master_config = _master_config(
            env=_gym_env({"max_concurrency": 1000}),
            max_inflight_prompts=5120,
        )

        with pytest.raises(ValueError, match="below the rollout fan-in"):
            validate_gym_actor_concurrency(master_config)

    def test_rejects_max_concurrency_equal_to_inflight_cap(self) -> None:
        """Leaves the actor's own health_check and shutdown unable to be admitted."""
        master_config = _master_config(
            env=_gym_env({"max_concurrency": 5120}),
            max_inflight_prompts=5120,
        )

        with pytest.raises(ValueError, match="below the rollout fan-in"):
            validate_gym_actor_concurrency(master_config)

    def test_accepts_max_concurrency_above_inflight_cap(self) -> None:
        validate_gym_actor_concurrency(
            _master_config(
                env=_gym_env({"max_concurrency": 6144}),
                max_inflight_prompts=5120,
            )
        )


def _full_master_config(*, env: dict, max_inflight_prompts: int) -> MasterConfig:
    """Populate everything validate_single_controller_config reads on the way in."""
    return MasterConfig.model_construct(
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=2,
            max_inflight_prompts=max_inflight_prompts,
            max_buffered_rollouts=max_inflight_prompts * 2,
        ),
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=4,
            skip_reference_policy_logprobs_calculation=False,
        ),
        policy={"train_global_batch_size": 8},
        loss_fn=SimpleNamespace(reference_policy_kl_penalty=0),
        env=env,
        checkpointing={"enabled": False, "metric_name": None},
    )


class TestGymActorConcurrencyIsReachedFromTheEntryPoint:
    """The guard is only load-time protection if the entry point runs it."""

    def test_entry_point_rejects_an_actor_that_cannot_admit_the_fan_in(self) -> None:
        master_config = _full_master_config(
            env=_gym_env({"max_concurrency": 1}),
            max_inflight_prompts=2,
        )

        with pytest.raises(ValueError, match="below the rollout fan-in"):
            validate_single_controller_config(master_config)

    def test_entry_point_accepts_an_actor_sized_for_the_fan_in(self) -> None:
        validate_single_controller_config(
            _full_master_config(
                env=_gym_env({"max_concurrency": 64}),
                max_inflight_prompts=2,
            )
        )
