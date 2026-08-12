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

"""Re-dispatch policy: no prompt is discarded for infrastructure reasons.

An infra failure means the fleet is unwell, not the prompt, so the attempt is retried.
The retry re-enters generation-shard selection, which is what makes it land somewhere
else -- generate_and_push itself knows nothing about shard health. Deterministic
failures get a much smaller budget because another shard rejects the prompt identically.

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
