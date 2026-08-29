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

import warnings
from types import SimpleNamespace
from typing import get_args

import pytest
from pydantic import ValidationError

from nemo_rl.algorithms.async_utils.staleness_sampler import SamplerConfig
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
from nemo_rl.algorithms.single_controller_utils.setup import _build_retry_policy


def _all_sampler_names() -> list[str]:
    """Every name the sampler union accepts, read off the union itself.

    ``SamplerConfig`` is ``Annotated[Union[...], Field(discriminator=...)]``, so the
    variants are one level in. Deriving the list rather than restating it is the point:
    a sampler added later shows up here on its own.
    """
    variants = get_args(get_args(SamplerConfig)[0])
    return [variant.model_fields["name"].default for variant in variants]


def _master_config(*, num_prompts_per_step: int = 8, **async_kwargs) -> MasterConfig:
    """A config the SC validator accepts, with the fields under test overridable."""
    return MasterConfig.model_construct(
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=num_prompts_per_step,
            **async_kwargs,
        ),
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=num_prompts_per_step,
            num_generations_per_prompt=4,
            skip_reference_policy_logprobs_calculation=False,
        ),
        policy={
            "train_global_batch_size": num_prompts_per_step * 4,
            "generation": {"colocated": {"enabled": False}},
        },
        # The last two are read only on the ready_first branch, which rejects a run
        # without them before it reaches anything under test here.
        loss_fn=SimpleNamespace(
            reference_policy_kl_penalty=0,
            use_importance_sampling_correction=True,
            force_on_policy_ratio=False,
        ),
        env={"should_use_nemo_gym": True},
        checkpointing={"enabled": False, "metric_name": None},
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
        assert cfg.max_consecutive_dropped_prompts == 0
        assert cfg.min_step_batch_fraction == 0.9
        assert cfg.on_dropped_prompt == "shrink"
        assert cfg.max_replacement_attempts == 1
        assert cfg.replacement_reserve_prompts == 1
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

    def test_a_consecutive_drop_budget_is_accepted(self):
        cfg = RolloutFailureConfig(max_consecutive_dropped_prompts=4)
        assert cfg.max_consecutive_dropped_prompts == 4

    def test_the_two_drop_budgets_are_independent_knobs(self):
        """Tolerating a bad dataset must not imply tolerating a dying fleet."""
        cfg = RolloutFailureConfig(max_skipped_prompts=100)
        assert cfg.max_consecutive_dropped_prompts == 0

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
    def test_an_out_of_range_step_floor_is_rejected(self, fraction):
        """0 would permit an empty step; above 1 could never be satisfied."""
        with pytest.raises(ValidationError):
            RolloutFailureConfig(min_step_batch_fraction=fraction)

    def test_a_full_step_floor_is_allowed_and_forbids_shrinking(self):
        assert RolloutFailureConfig(min_step_batch_fraction=1.0).min_step_batch_fraction

    def test_replace_mode_is_opt_in(self):
        """Shrinking is what the branch shipped with; replacing must be asked for."""
        assert RolloutFailureConfig().on_dropped_prompt == "shrink"

    @pytest.mark.parametrize("policy", ["regenerate", "promote"])
    def test_an_unknown_drop_policy_is_rejected(self, policy):
        """Borrowing is an optimization inside "replace", not a mode to select.

        Both paths hold the batch size, so "please hold it the slower way" is not a
        choice worth offering; promoted_prompt_groups reports which one ran.
        """
        with pytest.raises(ValidationError):
            RolloutFailureConfig(on_dropped_prompt=policy)

    @pytest.mark.parametrize(
        ("field", "message"),
        [
            ("max_replacement_attempts", "max_replacement_attempts"),
            ("replacement_reserve_prompts", "replacement_reserve_prompts"),
        ],
    )
    def test_replace_mode_that_could_never_replace_is_rejected(self, field, message):
        """Either zero leaves "replace" configured but behaving as "shrink".

        Silently degrading is the failure worth catching: the only reason to ask for
        replacement is the batch-size guarantee, so losing it without a word defeats
        the point of setting the knob. The spare pool gates borrowing too, since a
        group is only taken from a later step when a spare can repay it.
        """
        with pytest.raises(ValidationError, match=message):
            RolloutFailureConfig(on_dropped_prompt="replace", **{field: 0})

    def test_the_same_zeros_are_fine_while_shrinking(self):
        """They are only read in replace mode, so shrink runs must not trip on them."""
        cfg = RolloutFailureConfig(
            max_replacement_attempts=0, replacement_reserve_prompts=0
        )
        assert cfg.on_dropped_prompt == "shrink"

    def test_replace_mode_accepts_a_deeper_budget(self):
        cfg = RolloutFailureConfig(
            on_dropped_prompt="replace",
            max_replacement_attempts=3,
            replacement_reserve_prompts=16,
        )
        assert cfg.max_replacement_attempts == 3
        assert cfg.replacement_reserve_prompts == 16

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


class TestTheRefitDeadlineIsArmedByDefault:
    """The deadline is a precondition for recovery, so it may not default to None.

    A shard dying mid-collective leaves every trainer blocked inside NCCL. A Ray actor
    runs one task at a time, so the rebuild's own ``init_collective`` queues behind that
    blocked task and never runs -- the recovery wedges and the run ends on the stall
    watchdog instead. Only the abort releases those ranks.

    With ``None`` as the default, turning fleet health on bought detection and quarantine
    but no refit recovery, and nothing said so. These tests exist so that reverting the
    default to None fails loudly rather than silently removing recovery.
    """

    def test_the_documented_default_is_armed(self):
        assert AsyncRLConfig().generation_fleet_health.refit_timeout_s == 300.0

    def test_turning_fleet_health_on_does_not_have_to_ask_for_it(self):
        """The combination that used to be silently non-recovering."""
        cfg = FleetHealthConfig(enabled=True)
        assert cfg.refit_timeout_s is not None, (
            "enabled=True with no deadline is the configuration that advertises recovery "
            "and cannot perform it"
        )

    def test_it_clears_a_healthy_refit_by_a_wide_margin(self):
        """~1.9s measured for a 1.5B model on GB200, so it cannot fire on a slow one."""
        assert AsyncRLConfig().generation_fleet_health.refit_timeout_s >= 100 * 1.9

    def test_none_still_disarms_it_when_asked_explicitly(self):
        """The escape hatch stays: an explicit None starts no watchdog thread."""
        assert FleetHealthConfig(refit_timeout_s=None).refit_timeout_s is None


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
            policy={
                "train_global_batch_size": 8,
                "generation": {"colocated": {"enabled": False}},
            },
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


class TestTheDropBudgetsReachTheRolloutLayer:
    """A validated knob that no one reads is still a knob that does nothing.

    ``max_consecutive_dropped_prompts`` is enforced inside RolloutManager, which only
    ever sees ``RolloutRetryPolicy``. Nothing else in the suite crosses that seam, so a
    dropped assignment in `_build_retry_policy` would leave every config test green
    while the budget silently reverted to the default.
    """

    def test_the_configured_budgets_are_carried_across(self):
        policy = _build_retry_policy(
            _master_config(
                rollout_failure={
                    "max_consecutive_dropped_prompts": 8,
                    "max_skipped_prompts": 3,
                }
            )
        )
        assert policy.max_consecutive_dropped_prompts == 8
        assert policy.max_skipped_prompts == 3

    def test_the_default_still_fails_the_run_on_the_first_drop(self):
        policy = _build_retry_policy(_master_config())
        assert policy.max_consecutive_dropped_prompts == 0
        assert policy.max_skipped_prompts == 0


class TestCombinationsThatCannotDoWhatTheyWereSetFor:
    """Coherent configs whose knobs cancel out warn instead of failing.

    Each of these parses, and each is a defensible thing to ask for while debugging, so
    rejecting them would be wrong. What is not defensible is discovering hours in that
    the tolerance you configured could never have been exercised.
    """

    @staticmethod
    def _messages(cfg) -> list[str]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_single_controller_config(cfg)
        return [str(w.message) for w in caught]

    def test_a_floor_that_rounds_up_to_the_full_batch_cancels_the_drop_budget(self):
        """8 prompts at the 0.9 default floor to 8: no step may ever run short."""
        cfg = _master_config(rollout_failure={"max_consecutive_dropped_prompts": 4})
        with pytest.warns(UserWarning, match="min_step_batch_fraction"):
            validate_single_controller_config(cfg)

    def test_replacing_keeps_the_full_batch_legitimately(self):
        """A full floor is the point of replace mode, not a contradiction with it."""
        cfg = _master_config(
            rollout_failure={
                "max_consecutive_dropped_prompts": 4,
                "min_step_batch_fraction": 1.0,
                "on_dropped_prompt": "replace",
            }
        )
        assert not any("min_step_batch_fraction" in m for m in self._messages(cfg))

    def test_a_floor_that_leaves_room_to_shrink_is_not_warned_about(self):
        cfg = _master_config(
            rollout_failure={
                "max_consecutive_dropped_prompts": 4,
                "min_step_batch_fraction": 0.5,
            }
        )
        assert not any("min_step_batch_fraction" in m for m in self._messages(cfg))

    def test_replacing_under_a_sampler_that_never_stamps_falls_back_to_shrinking(self):
        """Replacement needs a step stamp to know which step it owes a group to."""
        cfg = _master_config(
            sampler={"name": "windowed"},
            rollout_failure={"on_dropped_prompt": "replace"},
        )
        with pytest.warns(UserWarning, match="on_dropped_prompt"):
            validate_single_controller_config(cfg)

    def test_a_stamping_sampler_is_not_warned_about(self):
        cfg = _master_config(rollout_failure={"on_dropped_prompt": "replace"})
        assert not any("on_dropped_prompt" in m for m in self._messages(cfg))

    def test_a_reserve_larger_than_a_step_can_never_fill(self):
        """The reserve is skimmed from completed steps, so a step bounds it."""
        cfg = _master_config(
            rollout_failure={
                "on_dropped_prompt": "replace",
                "replacement_reserve_prompts": 100,
            }
        )
        with pytest.warns(UserWarning, match="replacement_reserve_prompts"):
            validate_single_controller_config(cfg)

    def test_an_untouched_config_warns_about_nothing(self):
        assert self._messages(_master_config()) == []


class TestADropBudgetNeedsASamplerThatStamps:
    """The one combination in this area that is rejected rather than warned about.

    An unstamped drop is not a pair of knobs cancelling out: the prompt is given up on
    and no step is credited for it, so that step waits on a group nobody is generating.
    Under ``weight_fifo`` the gate that stops the rollout pump running ahead also stops
    it reaching exhaustion, so the "rollout exhausted" error never fires and the run
    stalls silently -- which is why config load is the last place that can object.
    """

    @pytest.mark.parametrize("sampler_name", ["windowed", "weight_fifo", "ready_first"])
    @pytest.mark.parametrize(
        "budget",
        [{"max_skipped_prompts": 1}, {"max_consecutive_dropped_prompts": 1}],
    )
    def test_either_budget_under_any_unstamped_sampler_is_rejected(
        self, sampler_name, budget
    ):
        cfg = _master_config(sampler={"name": sampler_name}, rollout_failure=budget)
        with pytest.raises(ValueError, match="stamps no target step"):
            validate_single_controller_config(cfg)

    def test_every_built_in_but_in_order_and_custom_is_covered(self):
        """Read off the union, so a sampler added later is caught by default.

        ready_first (#3582) landed between this PR's base and its rebase, inheriting the
        None stamp like the rest. An allow-list would have let it straight through, and a
        test that hardcoded the same list would have stayed green while it did.
        """
        exempt = {"in_order", "custom"}
        covered = [n for n in _all_sampler_names() if n not in exempt]
        assert covered, "the union should carry at least one unstamped sampler"
        for sampler_name in covered:
            cfg = _master_config(
                sampler={"name": sampler_name},
                rollout_failure={"max_consecutive_dropped_prompts": 1},
            )
            with pytest.raises(ValueError, match="stamps no target step"):
                validate_single_controller_config(cfg)

    def test_shrink_mode_is_rejected_too_not_only_replace(self):
        """The gap this closes: shrink is the default, so it was the silent one."""
        cfg = _master_config(
            sampler={"name": "weight_fifo"},
            rollout_failure={
                "max_consecutive_dropped_prompts": 2,
                "on_dropped_prompt": "shrink",
            },
        )
        with pytest.raises(ValueError, match="stamps no target step"):
            validate_single_controller_config(cfg)

    def test_a_stamping_sampler_carries_the_same_budget(self):
        cfg = _master_config(
            sampler={"name": "in_order"},
            rollout_failure={
                "max_consecutive_dropped_prompts": 2,
                "min_step_batch_fraction": 0.5,
            },
        )
        validate_single_controller_config(cfg)

    def test_a_zero_budget_is_left_to_the_warnings(self):
        """No drop is tolerated at all, so the shortfall path is unreachable anyway."""
        cfg = _master_config(
            sampler={"name": "windowed"},
            rollout_failure={"on_dropped_prompt": "replace"},
        )
        with pytest.warns(UserWarning, match="on_dropped_prompt"):
            validate_single_controller_config(cfg)

    def test_a_custom_sampler_is_not_second_guessed(self):
        """Whether it stamps is knowable only to its author, so neither set claims it."""
        cfg = _master_config(
            sampler={"name": "custom", "target": "my_pkg.samplers:MySampler"},
            rollout_failure={"max_consecutive_dropped_prompts": 2},
        )
        validate_single_controller_config(cfg)
