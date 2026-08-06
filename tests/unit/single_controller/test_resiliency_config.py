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

"""Tests for the SingleController resiliency config blocks.

Two things are pinned here. First, every default is inert: a config that does not
mention these fields must behave exactly as it did before they existed. Second, the
combinations that would silently do nothing are rejected at load time rather than at
hour three of a run.
"""

import pytest
from pydantic import ValidationError

from nemo_rl.algorithms.single_controller_utils.config import (
    AsyncRLConfig,
    RolloutFailureConfig,
    WatchdogConfig,
)


class TestDefaultsAreInert:
    def test_timeouts_default_to_disabled(self):
        cfg = AsyncRLConfig()
        assert cfg.rollout_timeout_s is None
        assert cfg.generation_timeout_s is None
        assert cfg.env_timeout_s is None

    def test_retry_budgets_have_documented_defaults(self):
        cfg = AsyncRLConfig().rollout_failure
        assert cfg.max_attempts_per_prompt == 5
        assert cfg.max_data_attempts_per_prompt == 2
        assert cfg.backoff_base_s == 1.0
        assert cfg.max_backoff_s == 30.0
        assert cfg.on_data_exhausted == "fail_fast"
        assert cfg.max_skipped_prompts == 0

    def test_watchdog_has_documented_defaults(self):
        cfg = AsyncRLConfig().watchdog
        assert cfg.interval_s == 30.0
        assert cfg.stall_timeout_s == 600.0
        assert cfg.stall_action == "warn"
        assert cfg.gym_subprocess_check is True

    def test_pre_existing_configs_still_load(self):
        """A config written before these fields existed must be unaffected."""
        cfg = AsyncRLConfig(
            **{
                "sampler": {"name": "in_order", "max_lookahead_versions": 0},
                "min_groups_for_streaming_train": 4,
                "max_inflight_prompts": 4,
                "max_buffered_rollouts": 4,
            }
        )
        assert cfg.rollout_timeout_s is None
        assert cfg.rollout_failure.max_attempts_per_prompt == 5


class TestRolloutFailureValidation:
    def test_backoff_ceiling_below_base_is_rejected(self):
        with pytest.raises(ValidationError, match="max_backoff_s"):
            RolloutFailureConfig(backoff_base_s=10.0, max_backoff_s=1.0)

    def test_equal_backoff_bounds_are_allowed(self):
        cfg = RolloutFailureConfig(backoff_base_s=5.0, max_backoff_s=5.0)
        assert cfg.max_backoff_s == 5.0

    def test_skip_without_a_budget_is_rejected(self):
        """`skip` with a zero budget behaves exactly like `fail_fast`.

        Accepting it would mean a user who asked to skip bad prompts silently gets a
        run that dies on the first one.
        """
        with pytest.raises(ValidationError, match="max_skipped_prompts"):
            RolloutFailureConfig(on_data_exhausted="skip", max_skipped_prompts=0)

    def test_skip_with_a_budget_is_accepted(self):
        cfg = RolloutFailureConfig(on_data_exhausted="skip", max_skipped_prompts=10)
        assert cfg.max_skipped_prompts == 10

    def test_fail_fast_does_not_require_a_skip_budget(self):
        assert (
            RolloutFailureConfig(on_data_exhausted="fail_fast").max_skipped_prompts == 0
        )

    @pytest.mark.parametrize("attempts", [0, -1])
    def test_non_positive_attempt_budgets_are_rejected(self, attempts):
        with pytest.raises(ValidationError):
            RolloutFailureConfig(max_attempts_per_prompt=attempts)
        with pytest.raises(ValidationError):
            RolloutFailureConfig(max_data_attempts_per_prompt=attempts)

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValidationError):
            RolloutFailureConfig(on_data_exhausted="drop")


class TestWatchdogValidation:
    def test_stall_timeout_must_exceed_the_tick(self):
        with pytest.raises(ValidationError, match="stall_timeout_s"):
            WatchdogConfig(interval_s=30.0, stall_timeout_s=30.0)

    def test_stall_timeout_below_the_tick_is_rejected(self):
        with pytest.raises(ValidationError, match="stall_timeout_s"):
            WatchdogConfig(interval_s=30.0, stall_timeout_s=5.0)

    def test_unknown_action_is_rejected(self):
        with pytest.raises(ValidationError):
            WatchdogConfig(stall_action="explode")


class TestWatchdogVersusRolloutTimeout:
    def test_watchdog_must_outlast_the_rollout_deadline(self):
        """A merely-slow rollout has its own deadline; the watchdog must let it fire."""
        with pytest.raises(ValidationError, match="stall_timeout_s"):
            AsyncRLConfig(
                rollout_timeout_s=900.0,
                watchdog={"interval_s": 30.0, "stall_timeout_s": 600.0},
            )

    def test_equal_deadlines_are_rejected(self):
        with pytest.raises(ValidationError, match="stall_timeout_s"):
            AsyncRLConfig(
                rollout_timeout_s=600.0,
                watchdog={"interval_s": 30.0, "stall_timeout_s": 600.0},
            )

    def test_watchdog_longer_than_the_rollout_deadline_is_accepted(self):
        cfg = AsyncRLConfig(
            rollout_timeout_s=900.0,
            watchdog={"interval_s": 30.0, "stall_timeout_s": 1200.0},
        )
        assert cfg.watchdog.stall_timeout_s == 1200.0

    def test_disabled_rollout_timeout_imposes_no_constraint(self):
        cfg = AsyncRLConfig(
            rollout_timeout_s=None,
            watchdog={"interval_s": 30.0, "stall_timeout_s": 60.0},
        )
        assert cfg.watchdog.stall_timeout_s == 60.0

    @pytest.mark.parametrize(
        "field", ["rollout_timeout_s", "generation_timeout_s", "env_timeout_s"]
    )
    def test_non_positive_timeouts_are_rejected(self, field):
        with pytest.raises(ValidationError):
            AsyncRLConfig(**{field: 0.0})
        with pytest.raises(ValidationError):
            AsyncRLConfig(**{field: -1.0})
