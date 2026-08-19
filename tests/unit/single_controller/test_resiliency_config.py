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

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.single_controller_utils.config import (
    AsyncRLConfig,
    FleetHealthConfig,
    GenerationRouterConfig,
    MasterConfig,
    RolloutFailureConfig,
    WatchdogConfig,
    validate_single_controller_config,
)


class TestDefaultsAreInert:
    def test_timeouts_default_to_disabled(self):
        cfg = AsyncRLConfig()
        assert cfg.rollout_failure.nemo_gym.rollout_timeout_s is None
        assert cfg.rollout_failure.native.generation_timeout_s is None
        assert cfg.rollout_failure.native.env_timeout_s is None

    def test_retry_budgets_have_documented_defaults(self):
        cfg = AsyncRLConfig().rollout_failure
        assert cfg.max_infra_attempts_per_prompt == 5
        assert cfg.max_data_attempts_per_prompt == 2
        assert cfg.backoff_base_s == 1.0
        assert cfg.max_backoff_s == 30.0
        assert cfg.max_skipped_prompts == 0
        assert cfg.nemo_gym.max_row_attempts == 3

    def test_watchdog_has_documented_defaults(self):
        cfg = AsyncRLConfig().stall_watchdog
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
        assert cfg.rollout_failure.nemo_gym.rollout_timeout_s is None
        assert cfg.rollout_failure.max_infra_attempts_per_prompt == 5


class TestRolloutFailureValidation:
    def test_backoff_ceiling_below_base_is_rejected(self):
        with pytest.raises(ValidationError, match="max_backoff_s"):
            RolloutFailureConfig(backoff_base_s=10.0, max_backoff_s=1.0)

    def test_equal_backoff_bounds_are_allowed(self):
        cfg = RolloutFailureConfig(backoff_base_s=5.0, max_backoff_s=5.0)
        assert cfg.max_backoff_s == 5.0

    def test_a_skip_budget_is_accepted(self):
        cfg = RolloutFailureConfig(max_skipped_prompts=10)
        assert cfg.max_skipped_prompts == 10

    def test_zero_is_the_default_and_means_fail_on_the_first_one(self):
        """One knob: the illegal "skip, but never skip anything" state is unrepresentable.

        It used to take an enum plus a count, which could contradict each other, and a
        validator existed purely to reject that one combination.
        """
        assert RolloutFailureConfig().max_skipped_prompts == 0

    @pytest.mark.parametrize("attempts", [0, -1])
    def test_non_positive_attempt_budgets_are_rejected(self, attempts):
        with pytest.raises(ValidationError):
            RolloutFailureConfig(max_infra_attempts_per_prompt=attempts)
        with pytest.raises(ValidationError):
            RolloutFailureConfig(max_data_attempts_per_prompt=attempts)

    @pytest.mark.parametrize(
        ("old", "value"),
        [
            ("max_attempts_per_prompt", 5),
            ("on_data_exhausted", "skip"),
            ("max_gym_row_attempts", 3),
        ],
    )
    def test_the_previous_key_names_are_rejected_not_ignored(self, old, value):
        """extra="allow" would let an old key parse and quietly do nothing.

        For on_data_exhausted="skip" that is a behaviour change -- prompts that used to
        be skipped start failing the run -- delivered with no diagnostic at all.
        """
        with pytest.raises(ValidationError, match="renamed"):
            RolloutFailureConfig(**{old: value})

    @pytest.mark.parametrize(
        "old", ["rollout_timeout_s", "generation_timeout_s", "env_timeout_s"]
    )
    def test_the_relocated_timeout_keys_are_rejected_not_ignored(self, old):
        """Same reasoning one level up: these moved into rollout_failure sub-blocks."""
        with pytest.raises(ValidationError, match="moved"):
            AsyncRLConfig(**{old: 60.0})


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


class TestGenerationRouterValidation:
    def test_the_default_status_is_outside_gyms_retry_set(self):
        assert GenerationRouterConfig().no_healthy_backend_status == 409

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 520])
    def test_a_status_gym_retries_is_rejected(self, status):
        """Returning one of these would make Gym retry forever.

        Gym retries 429/500/502/503/504/520, and for the rate-limit subset it raises its
        own retry ceiling on each attempt. Answering "no healthy backend" with one of
        them recreates the unbounded hang the router exists to prevent, so it is refused
        at config load rather than discovered in production.
        """
        with pytest.raises(ValidationError, match="NeMo-Gym retries internally"):
            GenerationRouterConfig(no_healthy_backend_status=status)

    @pytest.mark.parametrize("status", [400, 404, 409, 418, 422])
    def test_other_client_errors_are_allowed(self, status):
        assert (
            GenerationRouterConfig(
                no_healthy_backend_status=status
            ).no_healthy_backend_status
            == status
        )

    def test_it_is_off_by_default(self):
        assert AsyncRLConfig().generation_router.enabled is False


class TestFleetHealthValidation:
    def test_it_is_off_by_default(self):
        assert AsyncRLConfig().generation_fleet_health.enabled is False

    def test_a_probe_timeout_that_outlasts_the_interval_is_rejected(self):
        """Otherwise probes overlap and a slow fleet reads as a dead one."""
        with pytest.raises(ValidationError, match="probe_timeout_s"):
            FleetHealthConfig(probe_interval_s=2.0, probe_timeout_s=2.0)

    def test_unimplemented_recovery_modes_are_rejected(self):
        """They need the communicator rebuild; accepting them would do nothing."""
        with pytest.raises(ValidationError):
            FleetHealthConfig(on_dead_shard="degrade_and_restore")


class TestWatchdogVersusRolloutTimeout:
    """The watchdog must outlast EVERY deadline, not just the NeMo-Gym one.

    This guard used to compare stall_timeout_s against rollout_timeout_s alone, so on
    the native path -- where generation_timeout_s and env_timeout_s are the deadlines --
    the invariant it advertises went unchecked, and a stall_timeout_s below either
    produced exactly the false stall reports it exists to prevent.
    """

    # (sub-block, key) for each deadline the watchdog has to outlast.
    DEADLINES = [
        ("nemo_gym", "rollout_timeout_s"),
        ("native", "generation_timeout_s"),
        ("native", "env_timeout_s"),
    ]

    @staticmethod
    def _cfg(block, key, deadline, stall_timeout_s):
        return AsyncRLConfig(
            rollout_failure={block: {key: deadline}},
            stall_watchdog={"interval_s": 30.0, "stall_timeout_s": stall_timeout_s},
        )

    @pytest.mark.parametrize(("block", "key"), DEADLINES)
    def test_watchdog_must_outlast_every_deadline(self, block, key):
        with pytest.raises(ValidationError, match="stall_timeout_s"):
            self._cfg(block, key, 900.0, 600.0)

    @pytest.mark.parametrize(("block", "key"), DEADLINES)
    def test_equal_deadlines_are_rejected(self, block, key):
        with pytest.raises(ValidationError, match="stall_timeout_s"):
            self._cfg(block, key, 600.0, 600.0)

    @pytest.mark.parametrize(("block", "key"), DEADLINES)
    def test_a_longer_watchdog_is_accepted(self, block, key):
        assert (
            self._cfg(block, key, 900.0, 1200.0).stall_watchdog.stall_timeout_s
            == 1200.0
        )

    @pytest.mark.parametrize(("block", "key"), DEADLINES)
    def test_a_disabled_deadline_imposes_no_constraint(self, block, key):
        assert self._cfg(block, key, None, 60.0).stall_watchdog.stall_timeout_s == 60.0

    @pytest.mark.parametrize(("block", "key"), DEADLINES)
    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_non_positive_timeouts_are_rejected(self, block, key, value):
        with pytest.raises(ValidationError):
            AsyncRLConfig(rollout_failure={block: {key: value}})


class TestWrongPathFaultToleranceIsRejected:
    """Nesting shows which knob applies where; only a setup check enforces it.

    A populated sub-block for the path a run is not taking parses fine and then does
    nothing -- the silent no-op the restructure exists to remove. The chaos harness was
    itself guilty of this: it set the gym-only rollout_timeout_s on a native run.
    """

    @staticmethod
    def _master_config(*, use_nemo_gym: bool, rollout_failure: dict) -> MasterConfig:
        return MasterConfig.model_construct(
            async_rl=AsyncRLConfig(
                min_groups_for_streaming_train=2,
                max_inflight_prompts=2,
                rollout_failure=rollout_failure,
            ),
            grpo=GRPOConfig.model_construct(
                num_prompts_per_step=2,
                num_generations_per_prompt=4,
                skip_reference_policy_logprobs_calculation=False,
            ),
            policy={"train_global_batch_size": 8},
            loss_fn=SimpleNamespace(reference_policy_kl_penalty=0),
            env={"should_use_nemo_gym": use_nemo_gym},
            # Read by the metric_name check upstream #3429 added to this same
            # validator, which runs before the wrong-path check under test.
            checkpointing={"enabled": False, "metric_name": None},
        )

    def test_gym_only_knobs_on_a_native_run_are_rejected(self):
        cfg = self._master_config(
            use_nemo_gym=False,
            rollout_failure={"nemo_gym": {"rollout_timeout_s": 60.0}},
        )
        with pytest.raises(ValueError, match="silently ignored"):
            validate_single_controller_config(cfg)

    def test_native_only_knobs_on_a_gym_run_are_rejected(self):
        cfg = self._master_config(
            use_nemo_gym=True,
            rollout_failure={"native": {"generation_timeout_s": 60.0}},
        )
        with pytest.raises(ValueError, match="silently ignored"):
            validate_single_controller_config(cfg)

    def test_the_row_attempt_budget_counts_as_gym_only(self):
        """It is not a deadline, but it is just as inert on the native path."""
        cfg = self._master_config(
            use_nemo_gym=False, rollout_failure={"nemo_gym": {"max_row_attempts": 7}}
        )
        with pytest.raises(ValueError, match="max_row_attempts"):
            validate_single_controller_config(cfg)

    def test_the_applicable_block_is_accepted_on_each_path(self):
        validate_single_controller_config(
            self._master_config(
                use_nemo_gym=False,
                rollout_failure={"native": {"generation_timeout_s": 60.0}},
            )
        )
        validate_single_controller_config(
            self._master_config(
                use_nemo_gym=True,
                rollout_failure={"nemo_gym": {"rollout_timeout_s": 60.0}},
            )
        )

    @pytest.mark.parametrize("use_nemo_gym", [True, False])
    def test_defaults_are_accepted_on_both_paths(self, use_nemo_gym):
        """Inert by default: an untouched config must never trip this."""
        validate_single_controller_config(
            self._master_config(use_nemo_gym=use_nemo_gym, rollout_failure={})
        )

    def test_a_config_without_env_is_skipped_not_crashed(self):
        """`env` is required, so model_construct leaves it absent entirely.

        Reading it unguarded raised AttributeError from inside
        SingleControllerActor.__init__ and killed the actor -- for a check that only
        needs it to pick which half of the block is inert. Caught by an upstream
        vLLM test that builds its config this way, not by this suite.
        """
        cfg = self._master_config(use_nemo_gym=False, rollout_failure={})
        del cfg.env
        assert not hasattr(cfg, "env")
        validate_single_controller_config(cfg)


class TestGenerationRouterPortAndTimeoutValidation:
    def test_a_transposed_port_range_is_rejected(self):
        """Otherwise it surfaces as 'empty range for randrange()' far from the typo."""
        with pytest.raises(ValidationError, match="port_range_low"):
            GenerationRouterConfig(port_range_low=6099, port_range_high=6000)

    def test_an_equal_port_range_is_rejected(self):
        with pytest.raises(ValidationError, match="port_range_low"):
            GenerationRouterConfig(port_range_low=6000, port_range_high=6000)

    def test_the_connect_timeout_defaults_well_below_the_backend_timeout(self):
        """A handshake to a local vLLM is ms-or-never; the generation is minutes."""
        cfg = GenerationRouterConfig()
        assert cfg.connect_timeout_s == 5.0
        assert cfg.connect_timeout_s < cfg.backend_timeout_s

    def test_a_connect_timeout_beyond_the_total_is_rejected(self):
        with pytest.raises(ValidationError, match="connect_timeout_s"):
            GenerationRouterConfig(connect_timeout_s=100.0, backend_timeout_s=10.0)


class TestFleetHealthSelectionIsNotAdvertisedBeyondWhatItDoes:
    def test_an_unimplemented_selection_mode_is_rejected(self):
        """Nothing dispatches on this value, so accepting round_robin would silently
        hand the caller least_outstanding anyway -- the failure mode on_dead_shard's
        Literal already exists to prevent."""
        with pytest.raises(ValidationError):
            FleetHealthConfig(selection="round_robin")

    def test_the_implemented_mode_is_accepted(self):
        assert FleetHealthConfig(selection="least_outstanding").selection == (
            "least_outstanding"
        )


class TestRenamedBlocksAreRejected:
    """The old block names parsed fine under extra="allow" and then did nothing.

    async_rl.watchdog in particular shipped in the containment PR, so a config in the
    wild can carry it -- and silently losing stall detection is exactly the failure mode
    this series exists to remove.
    """

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            ("watchdog", "stall_watchdog"),
            ("fleet_health", "generation_fleet_health"),
            ("policy_router", "generation_router"),
        ],
    )
    def test_the_previous_block_names_are_rejected_not_ignored(self, old, new):
        with pytest.raises(ValidationError, match=new):
            AsyncRLConfig(**{old: {"enabled": True}})

    def test_the_new_names_are_accepted(self):
        cfg = AsyncRLConfig(
            stall_watchdog={"interval_s": 30.0, "stall_timeout_s": 600.0},
            generation_fleet_health={"enabled": True},
            generation_router={"enabled": True},
        )
        assert cfg.generation_fleet_health.enabled
        assert cfg.generation_router.enabled
        assert cfg.stall_watchdog.stall_timeout_s == 600.0


class TestRouterDeadlineFitsInsideTheRollout:
    """backend_timeout_s bounds one HTTP call; rollout_timeout_s bounds the whole stream."""

    def test_a_router_deadline_past_the_rollout_deadline_is_rejected(self):
        with pytest.raises(ValidationError, match="backend_timeout_s"):
            AsyncRLConfig(
                generation_router={"enabled": True, "backend_timeout_s": 600.0},
                rollout_failure={"nemo_gym": {"rollout_timeout_s": 300.0}},
            )

    def test_a_router_deadline_inside_it_is_accepted(self):
        cfg = AsyncRLConfig(
            generation_router={"enabled": True, "backend_timeout_s": 120.0},
            rollout_failure={"nemo_gym": {"rollout_timeout_s": 300.0}},
        )
        assert cfg.generation_router.backend_timeout_s == 120.0

    def test_an_unset_rollout_deadline_imposes_no_constraint(self):
        """rollout_timeout_s defaults to None -- disabled -- so there is nothing to fit in."""
        cfg = AsyncRLConfig(generation_router={"enabled": True})
        assert cfg.rollout_failure.nemo_gym.rollout_timeout_s is None

    def test_a_disabled_router_imposes_no_constraint(self):
        cfg = AsyncRLConfig(
            generation_router={"enabled": False, "backend_timeout_s": 600.0},
            rollout_failure={"nemo_gym": {"rollout_timeout_s": 60.0}},
        )
        assert cfg.generation_router.enabled is False
