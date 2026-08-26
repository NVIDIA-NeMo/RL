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

"""Re-dispatch policy: a prompt is retried before it is ever given up on.

An infra failure means the fleet is unwell, not the prompt, so the attempt is retried.
The retry re-enters generation-shard selection, which is what makes it land somewhere
else -- generate_and_push itself knows nothing about shard health. Deterministic
failures get a much smaller budget because another shard rejects the prompt identically.

A prompt that exhausts its budget is discarded only within an explicit allowance, and
the two classes get different allowances because they answer different questions.
``max_skipped_prompts`` is a lifetime total: a dataset does not get better.
``max_consecutive_dropped_prompts`` resets on every commit, so it asks whether the fleet
is answering *anyone* right now. Both default to 0, which is the fail-on-the-first-one
behaviour these budgets were added on top of.

The invariants every test here upholds:
  - nothing is committed unless the rollout actually succeeded
  - every reserved slot is released, whatever the outcome
  - each attempt gets a fresh group_id, so a failed attempt's rows cannot collide
    with the retry's
  - cancellation is never retried
"""

import asyncio
import uuid

import pytest
import ray.exceptions

from nemo_rl.experience.failures import (
    GenerationUnavailable,
    RolloutDataFailure,
    RolloutRedispatchExhausted,
)
from nemo_rl.experience.rollout_manager import (
    RolloutManager,
    RolloutOutcome,
    RolloutRetryPolicy,
    RolloutStats,
)


class _Buffer:
    """TQReplayBuffer stand-in recording the reserve/commit/remove sequence."""

    def __init__(self) -> None:
        self.reserved: list[str] = []
        self.commits: list[str] = []
        self.removals: list[str] = []

    def reserve(self, *, weight_version, target_step=None, group_id=None) -> str:
        del weight_version, target_step
        group_id = group_id or str(uuid.uuid4())
        self.reserved.append(group_id)
        return group_id

    async def commit(self, group_id, record, start_weight_version, end_weight_version):
        del start_weight_version, end_weight_version
        self.commits.append(group_id)
        return record

    async def remove_group(self, group_id, *, remove_in_dp: bool = False) -> int:
        del remove_in_dp
        self.removals.append(group_id)
        return 1


class _ScriptedImpl:
    """Rollout impl that raises a scripted sequence of failures, then succeeds."""

    def __init__(self, failures) -> None:
        self._failures = list(failures)
        self.attempts = 0

    async def run_rollout(self, input_sample):
        del input_sample
        index = self.attempts
        self.attempts += 1
        if index < len(self._failures):
            failure = self._failures[index]
            if failure is not None:
                raise failure
        return f"record-{index}"


def _make_manager(buffer, impl, policy) -> RolloutManager:
    manager = object.__new__(RolloutManager)
    manager._impl = impl
    manager._tokenizer = None
    manager._num_generations_per_prompt = 1
    manager._tq_buffer = buffer
    manager._weight_version = 0
    manager._retry_policy = policy
    manager._stats = RolloutStats()
    manager._skipped_prompts = 0
    manager._consecutive_infra_drops = 0
    return manager


def _sample(idx=3):
    return {"idx": idx}


@pytest.fixture
def no_sleep(monkeypatch):
    """Record backoff delays instead of waiting them out."""
    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def _fake(seconds, *args, **kwargs):
        delays.append(seconds)
        await real_sleep(0, *args, **kwargs)

    monkeypatch.setattr("nemo_rl.experience.rollout_manager.asyncio.sleep", _fake)
    return delays


class TestInfraRedispatch:
    def test_a_transient_infra_failure_is_retried_and_the_prompt_survives(
        self, no_sleep
    ):
        buffer = _Buffer()
        impl = _ScriptedImpl([GenerationUnavailable("shard down")])
        manager = _make_manager(
            buffer, impl, RolloutRetryPolicy.single_attempt(max_infra_attempts=3)
        )

        outcome = asyncio.run(manager.generate_and_push(_sample()))

        assert outcome is RolloutOutcome.COMMITTED
        assert impl.attempts == 2, "one failure, then one successful retry"
        assert len(buffer.commits) == 1

    def test_each_attempt_reserves_a_fresh_group_id(self, no_sleep):
        """A failed attempt's rows must not be able to collide with the retry's."""
        buffer = _Buffer()
        impl = _ScriptedImpl([GenerationUnavailable("x"), GenerationUnavailable("y")])
        manager = _make_manager(
            buffer, impl, RolloutRetryPolicy.single_attempt(max_infra_attempts=5)
        )

        asyncio.run(manager.generate_and_push(_sample()))

        assert len(buffer.reserved) == 3
        assert len(set(buffer.reserved)) == 3, "group ids must all differ"
        # The two failed attempts released their slots; the third committed.
        assert buffer.removals == buffer.reserved[:2]
        assert buffer.commits == buffer.reserved[2:]

    def test_exhausting_the_infra_budget_reports_fleet_wide_failure(self, no_sleep):
        buffer = _Buffer()
        impl = _ScriptedImpl([ray.exceptions.RayActorError()] * 10)
        manager = _make_manager(
            buffer, impl, RolloutRetryPolicy.single_attempt(max_infra_attempts=3)
        )

        with pytest.raises(RolloutRedispatchExhausted, match="after 3 attempt"):
            asyncio.run(manager.generate_and_push(_sample()))

        assert impl.attempts == 3
        assert buffer.commits == []
        assert len(buffer.removals) == 3, "every reserved slot released"

    def test_backoff_is_exponential_and_capped(self, no_sleep):
        buffer = _Buffer()
        impl = _ScriptedImpl([GenerationUnavailable("x")] * 10)
        manager = _make_manager(
            buffer,
            impl,
            RolloutRetryPolicy.single_attempt(
                max_infra_attempts=6, backoff_base_s=1.0, max_backoff_s=4.0
            ),
        )

        with pytest.raises(RolloutRedispatchExhausted):
            asyncio.run(manager.generate_and_push(_sample()))

        # 5 sleeps between 6 attempts, doubling until the ceiling bites.
        assert no_sleep == [1.0, 2.0, 4.0, 4.0, 4.0]

    def test_single_attempt_policy_reproduces_no_retry(self, no_sleep):
        buffer = _Buffer()
        impl = _ScriptedImpl([GenerationUnavailable("x")] * 3)
        manager = _make_manager(
            buffer, impl, RolloutRetryPolicy.single_attempt(max_infra_attempts=1)
        )

        with pytest.raises(RolloutRedispatchExhausted):
            asyncio.run(manager.generate_and_push(_sample()))

        assert impl.attempts == 1
        assert no_sleep == [], "no backoff when there is no retry"


class TestDataFailures:
    def test_one_retry_separates_transient_from_deterministic(self, no_sleep):
        """A shard under pressure can return an empty generation that is not the prompt."""
        buffer = _Buffer()
        impl = _ScriptedImpl([RolloutDataFailure("no generation data")])
        manager = _make_manager(
            buffer, impl, RolloutRetryPolicy.single_attempt(max_data_attempts=2)
        )

        outcome = asyncio.run(manager.generate_and_push(_sample()))

        assert outcome is RolloutOutcome.COMMITTED
        assert impl.attempts == 2

    def test_a_genuinely_bad_prompt_fails_the_run_by_default(self, no_sleep):
        buffer = _Buffer()
        impl = _ScriptedImpl([RolloutDataFailure("prompt too long")] * 5)
        manager = _make_manager(
            buffer, impl, RolloutRetryPolicy.single_attempt(max_data_attempts=2)
        )

        with pytest.raises(RolloutDataFailure, match="prompt too long"):
            asyncio.run(manager.generate_and_push(_sample()))

        assert impl.attempts == 2, "budget spent, then the original failure propagates"
        assert buffer.commits == []
        assert len(buffer.removals) == 2

    def test_data_failures_do_not_back_off(self, no_sleep):
        """Waiting cannot help a deterministic failure, so it should not cost time."""
        buffer = _Buffer()
        impl = _ScriptedImpl([RolloutDataFailure("x")] * 5)
        manager = _make_manager(
            buffer, impl, RolloutRetryPolicy.single_attempt(max_data_attempts=3)
        )

        with pytest.raises(RolloutDataFailure):
            asyncio.run(manager.generate_and_push(_sample()))
        assert no_sleep == []

    def test_skip_returns_a_typed_outcome_rather_than_committing(self, no_sleep):
        buffer = _Buffer()
        impl = _ScriptedImpl([RolloutDataFailure("bad row")] * 5)
        manager = _make_manager(
            buffer,
            impl,
            RolloutRetryPolicy.single_attempt(
                max_data_attempts=2, max_skipped_prompts=5
            ),
        )

        outcome = asyncio.run(manager.generate_and_push(_sample()))

        assert outcome is RolloutOutcome.SKIPPED
        assert buffer.commits == [], "a skipped prompt must never reach training"
        assert len(buffer.removals) == 2

    def test_the_skip_budget_is_run_wide_not_per_prompt(self, no_sleep):
        """Otherwise a systematically broken dataset would skip forever."""
        buffer = _Buffer()
        policy = RolloutRetryPolicy.single_attempt(
            max_data_attempts=1, max_skipped_prompts=2
        )
        impl = _ScriptedImpl([RolloutDataFailure("x")] * 50)
        manager = _make_manager(buffer, impl, policy)

        async def _drive():
            assert await manager.generate_and_push(_sample(1)) is RolloutOutcome.SKIPPED
            assert await manager.generate_and_push(_sample(2)) is RolloutOutcome.SKIPPED
            with pytest.raises(RolloutDataFailure, match="max_skipped_prompts=2"):
                await manager.generate_and_push(_sample(3))

        asyncio.run(_drive())


class TestInfraDropTolerance:
    """Exhausting the infra budget need not end the run.

    A fleet large enough to lose the occasional shard would otherwise fail a job over a
    single unlucky prompt. The budget is on *consecutive* drops so that tolerance never
    accumulates into ignoring a fleet that has stopped answering altogether -- the same
    shape as the v1 stack's ``max_generation_failures``.
    """

    def test_exhaustion_drops_the_prompt_instead_of_failing_the_run(self, no_sleep):
        buffer = _Buffer()
        impl = _ScriptedImpl([GenerationUnavailable("shard down")] * 10)
        manager = _make_manager(
            buffer,
            impl,
            RolloutRetryPolicy.single_attempt(
                max_infra_attempts=2, max_consecutive_dropped_prompts=3
            ),
        )

        outcome = asyncio.run(manager.generate_and_push(_sample()))

        assert outcome is RolloutOutcome.SKIPPED
        assert impl.attempts == 2, "the retry budget is still spent in full first"
        assert buffer.commits == [], "a dropped prompt must never reach training"
        assert len(buffer.removals) == 2, "every reserved slot released"

    def test_a_commit_clears_the_run_of_drops(self, no_sleep):
        """Tolerance is per-outage, so a recovered fleet starts from zero again."""
        buffer = _Buffer()
        # fail, succeed, fail, succeed, ... with a budget of 1 consecutive drop. Four
        # drops in total, none of them consecutive, so the run survives all of them.
        impl = _ScriptedImpl([GenerationUnavailable("x"), None] * 4)
        manager = _make_manager(
            buffer,
            impl,
            RolloutRetryPolicy.single_attempt(max_consecutive_dropped_prompts=1),
        )

        async def _drive():
            for _ in range(4):
                assert await manager.generate_and_push(_sample()) is (
                    RolloutOutcome.SKIPPED
                )
                assert await manager.generate_and_push(_sample()) is (
                    RolloutOutcome.COMMITTED
                )

        asyncio.run(_drive())

        assert manager.stats.committed == 4
        assert manager.stats.max_consecutive_infra_drops == 1

    def test_an_outage_that_does_not_recover_still_fails_the_run(self, no_sleep):
        buffer = _Buffer()
        impl = _ScriptedImpl([ray.exceptions.RayActorError()] * 20)
        manager = _make_manager(
            buffer,
            impl,
            RolloutRetryPolicy.single_attempt(max_consecutive_dropped_prompts=2),
        )

        async def _drive():
            assert await manager.generate_and_push(_sample()) is RolloutOutcome.SKIPPED
            assert await manager.generate_and_push(_sample()) is RolloutOutcome.SKIPPED
            with pytest.raises(
                RolloutRedispatchExhausted, match="max_consecutive_dropped_prompts=2"
            ):
                await manager.generate_and_push(_sample())

        asyncio.run(_drive())

    def test_the_default_budget_fails_on_the_first_drop(self, no_sleep):
        """The knob is opt-in: without it, exhaustion ends the run as it always did."""
        buffer = _Buffer()
        impl = _ScriptedImpl([GenerationUnavailable("x")] * 5)
        manager = _make_manager(
            buffer, impl, RolloutRetryPolicy.single_attempt(max_infra_attempts=2)
        )

        assert manager._retry_policy.max_consecutive_dropped_prompts == 0
        with pytest.raises(RolloutRedispatchExhausted):
            asyncio.run(manager.generate_and_push(_sample()))

    def test_the_infra_budget_is_not_spent_by_data_skips(self, no_sleep):
        """Sharing one allowance would let a bad dataset mask a dying fleet."""
        buffer = _Buffer()
        # Two data skips, then an infra exhaustion. With a shared counter the infra drop
        # would be the third and would abort; the budgets are independent, so it is the
        # first infra drop and is tolerated.
        impl = _ScriptedImpl(
            [
                RolloutDataFailure("bad prompt"),
                RolloutDataFailure("bad prompt"),
                GenerationUnavailable("shard down"),
            ]
        )
        manager = _make_manager(
            buffer,
            impl,
            RolloutRetryPolicy.single_attempt(
                max_skipped_prompts=5, max_consecutive_dropped_prompts=1
            ),
        )

        async def _drive():
            for _ in range(3):
                assert await manager.generate_and_push(_sample()) is (
                    RolloutOutcome.SKIPPED
                )

        asyncio.run(_drive())

        assert manager.stats.max_consecutive_infra_drops == 1


class TestCancellationIsNeverRetried:
    def test_cancellation_propagates_and_releases_the_slot(self, no_sleep):
        buffer = _Buffer()
        impl = _ScriptedImpl([asyncio.CancelledError()])
        manager = _make_manager(
            buffer, impl, RolloutRetryPolicy.single_attempt(max_infra_attempts=5)
        )

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(manager.generate_and_push(_sample()))

        assert impl.attempts == 1, "a cancelled rollout must not be re-dispatched"
        assert len(buffer.removals) == 1


class TestPolicyValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_infra_attempts": 0},
            {"max_data_attempts": 0},
            {"max_infra_attempts": -1},
        ],
    )
    def test_zero_attempt_budgets_are_rejected(self, kwargs):
        """A zero budget means "never attempt the rollout", which no caller wants."""
        with pytest.raises(ValueError, match="must be >= 1"):
            RolloutRetryPolicy.single_attempt(**kwargs)

    def test_single_attempt_is_the_no_retry_policy(self):
        """The attempt budgets are required, so no-retry has to be asked for by name."""
        policy = RolloutRetryPolicy.single_attempt()
        assert policy.max_infra_attempts == 1
        assert policy.max_data_attempts == 1
        # 0 skips allowed: the first data-budget exhaustion propagates the original
        # failure, which is what the retired on_data_exhausted="fail_fast" meant.
        assert policy.max_skipped_prompts == 0


class TestStats:
    def test_redispatches_and_commits_are_counted(self, no_sleep):
        buffer = _Buffer()
        impl = _ScriptedImpl([GenerationUnavailable("x")])
        manager = _make_manager(
            buffer, impl, RolloutRetryPolicy.single_attempt(max_infra_attempts=3)
        )

        asyncio.run(manager.generate_and_push(_sample()))

        metrics = manager.stats.as_metrics()
        assert metrics["rollout/committed_total"] == 1.0
        assert metrics["rollout/redispatch_total"] == 1.0
        assert metrics["rollout/redispatch_total/GenerationUnavailable"] == 1.0

    def test_skips_are_counted_separately_from_commits(self, no_sleep):
        buffer = _Buffer()
        impl = _ScriptedImpl([RolloutDataFailure("x")] * 5)
        manager = _make_manager(
            buffer,
            impl,
            RolloutRetryPolicy.single_attempt(
                max_data_attempts=1, max_skipped_prompts=5
            ),
        )

        asyncio.run(manager.generate_and_push(_sample()))

        metrics = manager.stats.as_metrics()
        assert metrics["rollout/skipped_total"] == 1.0
        assert metrics["rollout/committed_total"] == 0.0
        assert metrics["rollout/data_failures_total/RolloutDataFailure"] == 1.0

    def test_infra_drops_are_distinguishable_from_redispatches(self, no_sleep):
        """Redispatches alone cannot say whether the fleet recovered; drops can."""
        buffer = _Buffer()
        impl = _ScriptedImpl([GenerationUnavailable("x")] * 5)
        manager = _make_manager(
            buffer,
            impl,
            RolloutRetryPolicy.single_attempt(
                max_infra_attempts=2, max_consecutive_dropped_prompts=5
            ),
        )

        asyncio.run(manager.generate_and_push(_sample()))

        metrics = manager.stats.as_metrics()
        assert metrics["rollout/infra_drops_total"] == 1.0
        assert metrics["rollout/infra_drops_total/GenerationUnavailable"] == 1.0
        # One retry happened before the budget ran out, so both series move -- but by
        # different amounts, which is the whole reason they are separate.
        assert metrics["rollout/redispatch_total"] == 1.0

    def test_the_consecutive_high_water_mark_survives_recovery(self, no_sleep):
        """The live counter resets on commit, so only this records a near-miss."""
        buffer = _Buffer()
        impl = _ScriptedImpl([GenerationUnavailable("x")] * 3 + [None])
        manager = _make_manager(
            buffer,
            impl,
            RolloutRetryPolicy.single_attempt(max_consecutive_dropped_prompts=5),
        )

        async def _drive():
            for _ in range(3):
                await manager.generate_and_push(_sample())
            await manager.generate_and_push(_sample())

        asyncio.run(_drive())

        metrics = manager.stats.as_metrics()
        assert metrics["rollout/max_consecutive_infra_drops"] == 3.0
        assert manager._consecutive_infra_drops == 0, "the commit cleared the live run"
