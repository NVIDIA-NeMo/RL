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

"""Fleet health state machine.

The transition that matters most is the one that does not exist: a shard cannot return
to service just because it started answering probes again. A restarted engine holds
whatever weights it loaded at init, so re-admitting it on liveness alone would feed
training rollouts generated from a checkpoint hundreds of steps stale -- invisible, and
worse than the outage that caused it.
"""

import pytest

from nemo_rl.experience.failures import NoHealthyShards
from nemo_rl.models.generation.fleet_health import (
    FleetHealthPolicy,
    GenerationFleetExhausted,
    GenerationFleetHealth,
    HealthyShardSelector,
    ShardState,
)


def _monitor(shard_count=4, **policy_kwargs) -> GenerationFleetHealth:
    ticks = iter(range(10_000))
    return GenerationFleetHealth(
        shard_count=shard_count,
        policy=FleetHealthPolicy(**policy_kwargs),
        clock=lambda: float(next(ticks)),
    )


def _fail(monitor, shard_idx, times):
    for _ in range(times):
        monitor.record_probe(shard_idx, ok=False, error="probe timeout")


class TestProbeTransitions:
    def test_all_shards_start_serving(self):
        monitor = _monitor()
        assert monitor.serving_shards() == [0, 1, 2, 3]

    def test_one_bad_probe_only_makes_a_shard_suspect(self):
        """A blip must not cost a shard's worth of throughput."""
        monitor = _monitor(unhealthy_threshold=3)
        _fail(monitor, 1, times=1)
        assert monitor.state_of(1) is ShardState.SUSPECT
        assert 1 in monitor.serving_shards(), "suspect shards still take traffic"

    def test_repeated_failures_condemn_the_shard(self):
        monitor = _monitor(unhealthy_threshold=3)
        _fail(monitor, 1, times=3)
        assert monitor.state_of(1) is ShardState.DEAD
        assert monitor.serving_shards() == [0, 2, 3]

    def test_a_recovering_shard_needs_consecutive_successes(self):
        monitor = _monitor(unhealthy_threshold=3, healthy_threshold=2)
        _fail(monitor, 1, times=1)
        monitor.record_probe(1, ok=True)
        assert monitor.state_of(1) is ShardState.SUSPECT, "one success is not enough"
        monitor.record_probe(1, ok=True)
        assert monitor.state_of(1) is ShardState.HEALTHY

    def test_an_interrupted_recovery_starts_over(self):
        monitor = _monitor(unhealthy_threshold=5, healthy_threshold=3)
        _fail(monitor, 1, times=1)
        monitor.record_probe(1, ok=True)
        monitor.record_probe(1, ok=True)
        _fail(monitor, 1, times=1)  # resets the success streak
        monitor.record_probe(1, ok=True)
        monitor.record_probe(1, ok=True)
        assert monitor.state_of(1) is ShardState.SUSPECT

    def test_reported_failures_use_the_same_threshold(self):
        """Adapters see failures a liveness probe cannot.

        Same threshold, separate streak -- see TestReportedFailureStreak for why the
        counters cannot be shared.
        """
        monitor = _monitor(unhealthy_threshold=2)
        monitor.report_failure(2, ConnectionResetError("boom"))
        monitor.report_failure(2, ConnectionResetError("boom"))
        assert monitor.state_of(2) is ShardState.DEAD


class TestStaleIsTheOnlyWayBack:
    def test_a_dead_shard_ignores_successful_probes(self):
        """Answering again says nothing about whether the weights are current."""
        monitor = _monitor(unhealthy_threshold=1, healthy_threshold=1)
        _fail(monitor, 0, times=1)
        assert monitor.state_of(0) is ShardState.DEAD

        for _ in range(10):
            monitor.record_probe(0, ok=True)
        assert monitor.state_of(0) is ShardState.DEAD
        assert 0 not in monitor.serving_shards()

    def test_a_loaded_replacement_is_stale_and_not_serving(self):
        monitor = _monitor(unhealthy_threshold=1)
        _fail(monitor, 0, times=1)
        monitor.mark_restarting(0)
        assert monitor.state_of(0) is ShardState.RESTARTING
        monitor.mark_loaded(0)
        assert monitor.state_of(0) is ShardState.STALE
        assert 0 not in monitor.serving_shards(), "stale weights must not serve"

    def test_a_stale_shard_ignores_successful_probes(self):
        monitor = _monitor(unhealthy_threshold=1, healthy_threshold=1)
        _fail(monitor, 0, times=1)
        monitor.mark_restarting(0)
        monitor.mark_loaded(0)
        for _ in range(10):
            monitor.record_probe(0, ok=True)
        assert monitor.state_of(0) is ShardState.STALE

    def test_only_a_refit_restores_service(self):
        monitor = _monitor(unhealthy_threshold=1)
        _fail(monitor, 0, times=1)
        monitor.mark_restarting(0)
        monitor.mark_loaded(0)
        monitor.report_refit(0, weight_version=42)
        assert monitor.state_of(0) is ShardState.HEALTHY
        assert 0 in monitor.serving_shards()
        assert monitor.snapshot()[0].weight_version == 42


class TestRetirement:
    def test_restarts_are_bounded_then_the_shard_retires(self):
        monitor = _monitor(unhealthy_threshold=1, max_restart_attempts_per_shard=2)
        _fail(monitor, 0, times=1)
        monitor.mark_restarting(0)
        monitor.mark_restarting(0)
        monitor.mark_restarting(0)  # one past the budget
        assert monitor.state_of(0) is ShardState.RETIRED

    def test_retirement_is_terminal(self):
        monitor = _monitor(unhealthy_threshold=1)
        monitor.retire(0, reason="node gone")
        for action in (
            lambda: monitor.mark_restarting(0),
            lambda: monitor.mark_loaded(0),
            lambda: monitor.report_refit(0, weight_version=9),
            lambda: monitor.record_probe(0, ok=True),
        ):
            action()
            assert monitor.state_of(0) is ShardState.RETIRED

    def test_training_continues_on_the_survivors(self):
        monitor = _monitor(shard_count=4)
        monitor.retire(3, reason="node gone")
        assert monitor.serving_shards() == [0, 1, 2]
        monitor.raise_if_exhausted()  # 3 >= min_healthy_shards=1

    def test_dropping_below_the_floor_is_fatal(self):
        monitor = _monitor(shard_count=2, min_healthy_shards=2)
        monitor.retire(1, reason="node gone")
        with pytest.raises(GenerationFleetExhausted, match="min_healthy_shards=2"):
            monitor.raise_if_exhausted()


class TestMembershipEpoch:
    def test_epoch_is_stable_while_the_serving_set_is(self):
        monitor = _monitor(unhealthy_threshold=3)
        before = monitor.membership_epoch
        _fail(monitor, 1, times=1)  # HEALTHY -> SUSPECT, still serving
        assert monitor.membership_epoch == before

    def test_epoch_advances_when_a_shard_leaves(self):
        monitor = _monitor(unhealthy_threshold=1)
        before = monitor.membership_epoch
        _fail(monitor, 1, times=1)
        assert monitor.membership_epoch == before + 1

    def test_epoch_advances_when_a_shard_returns(self):
        monitor = _monitor(unhealthy_threshold=1)
        _fail(monitor, 1, times=1)
        after_death = monitor.membership_epoch
        monitor.mark_restarting(1)
        monitor.mark_loaded(1)
        assert monitor.membership_epoch == after_death, "still not serving"
        monitor.report_refit(1, weight_version=7)
        assert monitor.membership_epoch == after_death + 1

    def test_several_changes_between_reads_collapse_into_the_delta(self):
        """Reconciliation compares epochs, so only the net change has to be applied."""
        monitor = _monitor(shard_count=4, unhealthy_threshold=1)
        before = monitor.membership_epoch
        _fail(monitor, 0, times=1)
        _fail(monitor, 1, times=1)
        assert monitor.membership_epoch > before
        assert monitor.serving_shards() == [2, 3]


class TestSelector:
    def test_a_dead_shard_is_never_selected(self):
        monitor = _monitor(shard_count=2, unhealthy_threshold=1)
        selector = HealthyShardSelector(monitor=monitor)
        _fail(monitor, 0, times=1)
        assert {selector.next_shard() for _ in range(10)} == {1}

    def test_least_outstanding_spreads_load(self):
        monitor = _monitor(shard_count=3)
        selector = HealthyShardSelector(monitor=monitor)
        picked = []
        for _ in range(6):
            shard = selector.next_shard()
            selector.acquire(shard)
            picked.append(shard)
        assert sorted(picked) == [0, 0, 1, 1, 2, 2]

    def test_load_steers_away_from_a_busy_shard(self):
        """The property that matters: a slow shard stops attracting new work."""
        monitor = _monitor(shard_count=2)
        selector = HealthyShardSelector(monitor=monitor)
        for _ in range(5):
            selector.acquire(0)
        assert selector.next_shard() == 1

    def test_releasing_makes_a_shard_eligible_again(self):
        monitor = _monitor(shard_count=2)
        selector = HealthyShardSelector(monitor=monitor)
        selector.acquire(0)
        selector.acquire(1)
        selector.release(0)
        assert selector.next_shard() == 0

    def test_no_eligible_shard_raises_a_retriable_failure(self):
        """NoHealthyShards is an infra failure, so the rollout policy re-dispatches."""
        monitor = _monitor(shard_count=1, unhealthy_threshold=1)
        selector = HealthyShardSelector(monitor=monitor)
        _fail(monitor, 0, times=1)
        with pytest.raises(NoHealthyShards, match="no generation shard is eligible"):
            selector.next_shard()

    def test_release_does_not_go_negative(self):
        monitor = _monitor(shard_count=1)
        selector = HealthyShardSelector(monitor=monitor)
        selector.release(0)
        assert selector.inflight(0) == 0


class TestPolicyValidation:
    @pytest.mark.parametrize(
        "kwargs", [{"unhealthy_threshold": 0}, {"healthy_threshold": 0}]
    )
    def test_non_positive_thresholds_are_rejected(self, kwargs):
        with pytest.raises(ValueError, match="thresholds must be >= 1"):
            FleetHealthPolicy(**kwargs)

    def test_a_zero_floor_is_rejected(self):
        with pytest.raises(ValueError, match="min_healthy_shards must be >= 1"):
            FleetHealthPolicy(min_healthy_shards=0)

    def test_shard_count_must_be_positive(self):
        with pytest.raises(ValueError, match="shard_count must be >= 1"):
            GenerationFleetHealth(shard_count=0, policy=FleetHealthPolicy())

    def test_base_urls_must_match_shard_count(self):
        with pytest.raises(ValueError, match="2 entries for 3 shards"):
            GenerationFleetHealth(
                shard_count=3, policy=FleetHealthPolicy(), base_urls=["a", "b"]
            )


class TestMetrics:
    def test_state_counts_and_epoch_are_published(self):
        monitor = _monitor(shard_count=3, unhealthy_threshold=1)
        _fail(monitor, 0, times=1)
        metrics = monitor.as_metrics()
        assert metrics["gen_fleet/shards/dead"] == 1.0
        assert metrics["gen_fleet/shards/healthy"] == 2.0
        assert metrics["gen_fleet/serving_shards"] == 2.0
        assert metrics["gen_fleet/membership_epoch"] >= 1.0

    def test_per_shard_weight_version_is_published(self):
        """A serving shard whose version trails the trainer is a correctness bug."""
        monitor = _monitor(shard_count=2, unhealthy_threshold=1)
        _fail(monitor, 0, times=1)
        monitor.mark_restarting(0)
        monitor.mark_loaded(0)
        monitor.report_refit(0, weight_version=11)
        assert monitor.as_metrics()["gen_fleet/shard_weight_version/0"] == 11.0


class TestReportedFailureStreak:
    """Failures seen on real requests, counted apart from probe failures.

    This is the wedged-engine case, and the reason the two streaks cannot share a
    counter: a wedged vLLM keeps answering ``is_alive`` from a perfectly healthy worker
    process, so every probe succeeds while every generation fails.
    """

    def test_reported_failures_condemn_a_shard_that_still_answers_probes(self):
        """The regression: ok-probes used to reset the streak, so this never fired.

        report_failure delegated to record_probe(ok=False), and record_probe(ok=True)
        zeroes the shared counter. A wedged shard answering a probe between each failure
        therefore never reached unhealthy_threshold, which is precisely the shard the
        reporting path exists to catch.
        """
        monitor = _monitor(unhealthy_threshold=3)
        for _ in range(3):
            monitor.record_probe(0, ok=True)  # the wedge answers every probe
            monitor.report_failure(0, RuntimeError("generation never returned"))
        assert monitor.state_of(0) is ShardState.DEAD
        assert 0 not in monitor.serving_shards()

    def test_one_reported_failure_only_makes_a_shard_suspect(self):
        monitor = _monitor(unhealthy_threshold=3)
        monitor.report_failure(1, RuntimeError("boom"))
        assert monitor.state_of(1) is ShardState.SUSPECT
        assert 1 in monitor.serving_shards(), "suspect shards still take traffic"

    def test_a_successful_generation_clears_the_streak(self):
        """Without this the counter is monotonic and every shard eventually dies."""
        monitor = _monitor(unhealthy_threshold=3)
        monitor.report_failure(2, RuntimeError("boom"))
        monitor.report_failure(2, RuntimeError("boom"))
        monitor.report_success(2)
        monitor.report_failure(2, RuntimeError("boom"))
        monitor.report_failure(2, RuntimeError("boom"))
        assert monitor.state_of(2) is not ShardState.DEAD

    def test_a_refit_clears_the_streak(self):
        """A refit means a fresh engine; its predecessor's failures are not evidence."""
        monitor = _monitor(unhealthy_threshold=3)
        monitor.report_failure(3, RuntimeError("boom"))
        monitor.report_failure(3, RuntimeError("boom"))
        monitor.report_refit(3, weight_version=7)
        monitor.report_failure(3, RuntimeError("boom"))
        monitor.report_failure(3, RuntimeError("boom"))
        assert monitor.state_of(3) is not ShardState.DEAD

    def test_a_probe_failure_and_a_reported_failure_are_independent(self):
        """Two failure kinds, two budgets: neither can quietly spend the other's."""
        monitor = _monitor(unhealthy_threshold=3)
        _fail(monitor, 0, times=2)
        monitor.report_failure(0, RuntimeError("boom"))
        monitor.report_failure(0, RuntimeError("boom"))
        assert monitor.state_of(0) is ShardState.SUSPECT

    def test_reports_are_ignored_once_a_shard_has_left_the_serving_set(self):
        """A late report must not disturb a shard the ledger has already condemned."""
        monitor = _monitor(unhealthy_threshold=2)
        _fail(monitor, 1, times=2)
        assert monitor.state_of(1) is ShardState.DEAD
        monitor.report_failure(1, RuntimeError("late report"))
        assert monitor.state_of(1) is ShardState.DEAD


class TestBaseUrlLookup:
    def test_a_known_url_maps_back_to_its_shard(self):
        monitor = GenerationFleetHealth(
            shard_count=2,
            policy=FleetHealthPolicy(),
            base_urls=["http://a:1/v1", "http://b:2/v1"],
        )
        assert monitor.shard_for_base_url("http://b:2/v1") == 1

    def test_an_unknown_url_maps_to_nothing(self):
        """The router may report a backend the ledger no longer tracks."""
        monitor = GenerationFleetHealth(
            shard_count=1, policy=FleetHealthPolicy(), base_urls=["http://a:1/v1"]
        )
        assert monitor.shard_for_base_url("http://gone:9/v1") is None
