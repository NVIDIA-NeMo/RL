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

"""Generation failures on the native GRPO rollout path must never reach training.

The regression these tests guard is subtle and silent. `_run_single_rollout` used to
catch every exception from `_generate_response`, print it, and `break` out of the turn
loop. Execution then fell through to build a Completion holding only the prompt, with
`reward=0.0`, which `generate_and_push` committed as a training row.

It did not even crash downstream: `add_grpo_token_loss_masks_and_generation_logprobs`
zero-fills a missing `generation_logprobs` and sets `token_loss_mask=0`, so the row is
well formed. It contributes no gradient -- but its zero reward *does* enter the
per-prompt GRPO baseline, so with `use_leave_one_out_baseline` one dead vLLM worker
silently shifts the advantage of every sibling in the group.
"""

import asyncio
import uuid

import pytest
import ray.exceptions
import torch

from nemo_rl.environments.interfaces import EnvironmentReturn
from nemo_rl.experience.failures import (
    FailureClass,
    GenerationUnavailable,
    GymTransportError,
    RolloutDataFailure,
    RolloutFailure,
    RolloutRedispatchExhausted,
    RolloutTimeout,
    classify_rollout_failure,
)
from nemo_rl.experience.rollout_manager import (
    AsyncRolloutImpl,
    RolloutManager,
    RolloutRetryPolicy,
    RolloutStats,
    RolloutTimeouts,
    _classify_generation_failure,
    _Deadline,
    _gather_cancelling_siblings,
)
from nemo_rl.utils.timer import Timer


@pytest.fixture
def terminating_env(monkeypatch):
    """Make calculate_rewards return a terminated, zero-reward step.

    Lets the success-path tests exercise a whole rollout turn without standing up a
    real environment actor.
    """

    def _fake(sample_batch, task_to_env):
        del sample_batch, task_to_env
        return EnvironmentReturn(
            observations=[{"role": "environment", "content": ""}],
            metadata=[None],
            next_stop_strings=[None],
            rewards=torch.tensor([0.0]),
            terminateds=torch.tensor([True]),
            answers=[None],
        )

    monkeypatch.setattr(
        "nemo_rl.experience.rollout_manager.calculate_rewards", _fake, raising=True
    )
    return _fake


class _TokenizedText:
    def __init__(self, input_ids: torch.Tensor) -> None:
        self.input_ids = input_ids


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return f"<{len(ids)} tokens>"

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        """Tokenize an environment observation to one id per character."""
        del return_tensors, add_special_tokens
        return _TokenizedText(torch.tensor([[7] * len(text)], dtype=torch.long))


class _FakeGeneration:
    """Generation stub whose behaviour is chosen per call.

    `behaviours` is consumed in call order; each entry is either an exception to raise
    or the string "hang" / "ok".
    """

    def __init__(self, behaviours):
        self._behaviours = list(behaviours)
        self.calls = 0
        self.cancelled = 0

    async def generate_async(self, data):
        del data
        index = min(self.calls, len(self._behaviours) - 1)
        behaviour = self._behaviours[index]
        self.calls += 1

        if isinstance(behaviour, BaseException):
            raise behaviour
        if behaviour == "hang":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        if behaviour == "slow":
            await asyncio.sleep(0.05)
        yield (
            0,
            {
                "output_ids": torch.tensor([[1, 2, 3, 4]]),
                "unpadded_sequence_lengths": torch.tensor([4]),
            },
        )


def _make_impl(
    generation, *, num_generations=1, max_turns=1, timeouts=None
) -> AsyncRolloutImpl:
    """Build an AsyncRolloutImpl without firing its real __init__."""
    impl = object.__new__(AsyncRolloutImpl)
    impl._tokenizer = _FakeTokenizer()
    impl._task_to_env = {}
    impl._num_generations_per_prompt = num_generations
    impl._max_seq_len = 128
    impl._max_rollout_turns = max_turns
    impl._policy_generation = generation
    impl._timeouts = timeouts if timeouts is not None else RolloutTimeouts()
    return impl


def _sample(idx=7):
    return {
        "idx": idx,
        "message_log": [
            {"role": "user", "content": "hi", "token_ids": torch.tensor([1, 2])}
        ],
        "extra_env_info": {},
        "task_name": "calc",
        "stop_strings": None,
    }


def _make_manager(buffer, impl, retry_policy=None) -> RolloutManager:
    """Build a RolloutManager without firing the real __init__."""
    manager = object.__new__(RolloutManager)
    manager._impl = impl
    manager._tokenizer = None
    manager._num_generations_per_prompt = 1
    manager._tq_buffer = buffer
    manager._weight_version = 0
    manager._retry_policy = (
        retry_policy
        if retry_policy is not None
        else RolloutRetryPolicy.single_attempt()
    )
    manager._stats = RolloutStats()
    manager._skipped_prompts = 0
    manager._consecutive_infra_drops = 0
    return manager


class _RecordingBuffer:
    """TQReplayBuffer stand-in that records whether anything was ever committed."""

    def __init__(self) -> None:
        self.commits: list[str] = []
        self.removals: list[str] = []

    def reserve(self, *, weight_version, target_step=None, group_id=None) -> str:
        del weight_version, target_step
        return group_id or str(uuid.uuid4())

    async def commit(self, group_id, record, start_weight_version, end_weight_version):
        del start_weight_version, end_weight_version
        self.commits.append(group_id)
        return record

    async def remove_group(self, group_id, *, remove_in_dp: bool = False) -> int:
        del remove_in_dp
        self.removals.append(group_id)
        return 1


class TestGenerationFailureIsClassifiedAndRaised:
    def test_dead_worker_becomes_generation_unavailable(self):
        impl = _make_impl(_FakeGeneration([ray.exceptions.RayActorError()]))
        with pytest.raises(GenerationUnavailable) as excinfo:
            asyncio.run(impl._run_single_rollout(_sample(), traj_idx=3))
        assert isinstance(excinfo.value.__cause__, ray.exceptions.RayActorError)

    def test_failure_message_names_the_prompt_and_trajectory(self):
        impl = _make_impl(_FakeGeneration([ray.exceptions.RayActorError()]))
        with pytest.raises(GenerationUnavailable, match=r"prompt_idx=7 traj_idx=3"):
            asyncio.run(impl._run_single_rollout(_sample(idx=7), traj_idx=3))

    def test_deterministic_error_becomes_a_data_failure(self):
        impl = _make_impl(_FakeGeneration([ValueError("prompt too long")]))
        with pytest.raises(RolloutDataFailure) as excinfo:
            asyncio.run(impl._run_single_rollout(_sample(), traj_idx=0))
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_cancellation_is_not_reclassified_as_a_rollout_failure(self):
        """CancelledError derives from BaseException, so `except Exception` must miss it.

        Reclassifying it would turn a shutdown into a data failure and, once retries
        exist, retry a rollout the controller is deliberately tearing down.
        """
        impl = _make_impl(_FakeGeneration([asyncio.CancelledError()]))
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(impl._run_single_rollout(_sample(), traj_idx=0))


class TestNoPartialCompletionSurvives:
    def test_run_rollout_raises_instead_of_returning_a_zero_reward_record(self):
        """The core regression guard: a failed generation yields no record at all."""
        impl = _make_impl(_FakeGeneration([ray.exceptions.RayActorError()]))
        with pytest.raises(RolloutFailure):
            asyncio.run(impl.run_rollout(_sample()))

    def test_generate_and_push_never_commits_a_failed_rollout(self):
        """End of the chain: nothing reaches the replay buffer, so nothing reaches training."""
        buffer = _RecordingBuffer()
        impl = _make_impl(_FakeGeneration([ray.exceptions.RayActorError()]))

        manager = _make_manager(buffer, impl)

        # A dead worker is an infra failure, so the single-attempt default budget is
        # exhausted immediately and surfaces as RolloutRedispatchExhausted.
        with pytest.raises(RolloutRedispatchExhausted):
            asyncio.run(manager.generate_and_push(_sample()))

        assert buffer.commits == [], "a failed generation must not be committed"
        assert len(buffer.removals) == 1, "the reserved slot must be released"


class TestSiblingCancellation:
    def test_one_failure_cancels_the_other_generations_in_the_group(self):
        """Siblings must not keep occupying generation capacity for a discarded group.

        asyncio.gather propagates the first exception but leaves the rest running
        detached; on this path that means N-1 generations still queued against the
        fleet for a prompt whose result is already being thrown away.
        """
        # First call fails, every later call hangs until cancelled.
        generation = _FakeGeneration([ray.exceptions.RayActorError(), "hang"])
        impl = _make_impl(generation, num_generations=4)

        with pytest.raises(RolloutFailure):
            asyncio.run(impl.run_rollout(_sample()))

        assert generation.calls == 4, "all four generations should have started"
        assert generation.cancelled == 3, (
            "the three surviving siblings must be cancelled"
        )


class TestGatherCancellingSiblings:
    def test_returns_results_in_input_order(self):
        async def _value(i):
            await asyncio.sleep((5 - i) * 0.001)
            return i

        result = asyncio.run(
            _gather_cancelling_siblings([_value(0), _value(1), _value(2)])
        )
        assert result == [0, 1, 2]

    def test_propagates_the_original_exception_type(self):
        async def _boom():
            raise ray.exceptions.RayActorError()

        async def _hang():
            await asyncio.Event().wait()

        with pytest.raises(ray.exceptions.RayActorError):
            asyncio.run(_gather_cancelling_siblings([_boom(), _hang()]))

    def test_drains_cancelled_siblings_before_unwinding(self):
        """The helper must not leave tasks pending when it returns control."""
        started = []
        finished = []

        async def _boom():
            await asyncio.sleep(0.01)
            raise ValueError("x")

        async def _slow():
            started.append(1)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                finished.append(1)
                raise

        async def _main():
            with pytest.raises(ValueError):
                await _gather_cancelling_siblings([_boom(), _slow()])
            # If the helper returned before draining, this would still be 0.
            assert finished == [1]
            assert not [
                t for t in asyncio.all_tasks() if t is not asyncio.current_task()
            ]

        asyncio.run(_main())


class TestGenerationDeadline:
    def test_a_hung_generation_raises_rollout_timeout(self):
        """The wedge this exists to prevent: a generation that never returns."""
        impl = _make_impl(
            _FakeGeneration(["hang"]),
            timeouts=RolloutTimeouts(generation_s=0.05),
        )
        with pytest.raises(GenerationUnavailable) as excinfo:
            asyncio.run(impl._run_single_rollout(_sample(), traj_idx=0))
        # Classified as infra, so the retry policy will re-dispatch it.
        assert isinstance(excinfo.value.__cause__, RolloutTimeout)

    def test_the_hung_generation_is_cancelled_not_abandoned(self):
        generation = _FakeGeneration(["hang"])
        impl = _make_impl(generation, timeouts=RolloutTimeouts(generation_s=0.05))
        with pytest.raises(GenerationUnavailable):
            asyncio.run(impl._run_single_rollout(_sample(), traj_idx=0))
        assert generation.cancelled == 1

    def test_no_deadline_configured_means_no_timeout(self, terminating_env):
        """Default config must behave exactly as before: wait indefinitely."""
        impl = _make_impl(
            _FakeGeneration(["slow"]), timeouts=RolloutTimeouts(generation_s=None)
        )
        completion, _ = asyncio.run(impl._run_single_rollout(_sample(), traj_idx=0))
        assert completion.reward == 0.0

    def test_slow_but_healthy_generation_does_not_trip_the_deadline(
        self, terminating_env
    ):
        """Guards against over-tightening: a slow success must still succeed."""
        impl = _make_impl(
            _FakeGeneration(["slow"]), timeouts=RolloutTimeouts(generation_s=5.0)
        )
        completion, metrics = asyncio.run(
            impl._run_single_rollout(_sample(), traj_idx=0)
        )
        assert metrics["turn_count"] == 1
        assert any(m["role"] == "assistant" for m in completion.message_log)

    def test_the_environment_step_has_its_own_deadline(self, monkeypatch):
        """A hung env actor leaks a rollout permit exactly like a hung generation.

        The sleep is deliberately short. calculate_rewards runs under asyncio.to_thread
        and Python cannot kill a running thread, so the deadline frees the rollout while
        the thread keeps its pool slot -- and asyncio.run waits for the default executor
        on the way out. A long sleep here would stall the test for exactly that reason,
        which is itself the documented caveat.
        """

        def _hang(sample_batch, task_to_env):
            del sample_batch, task_to_env
            import time

            time.sleep(1.0)

        monkeypatch.setattr(
            "nemo_rl.experience.rollout_manager.calculate_rewards", _hang, raising=True
        )
        impl = _make_impl(_FakeGeneration(["ok"]), timeouts=RolloutTimeouts(env_s=0.05))
        with pytest.raises(RolloutTimeout, match="environment step"):
            asyncio.run(impl._run_single_rollout(_sample(), traj_idx=0))


class TestDeadlineHelper:
    def test_expiry_is_reported_as_rollout_timeout(self):
        async def _main():
            async with _Deadline(0.01, "unit under test"):
                await asyncio.sleep(10)

        with pytest.raises(RolloutTimeout, match="unit under test exceeded 0.01s"):
            asyncio.run(_main())

    def test_an_inner_timeout_error_is_not_relabelled(self):
        """A TimeoutError from the wrapped code must not read as a deadline breach.

        Otherwise a genuine downstream timeout would be reported with our deadline's
        duration, sending anyone debugging it to the wrong knob.
        """

        async def _main():
            async with _Deadline(30.0, "unit under test"):
                raise TimeoutError("something downstream timed out")

        with pytest.raises(TimeoutError) as excinfo:
            asyncio.run(_main())
        assert not isinstance(excinfo.value, RolloutTimeout)

    def test_none_disables_the_deadline(self):
        async def _main():
            async with _Deadline(None, "unit under test"):
                await asyncio.sleep(0.01)
            return "completed"

        assert asyncio.run(_main()) == "completed"

    def test_outer_cancellation_still_propagates_as_cancellation(self):
        """Tearing down the controller must not look like a rollout timeout."""

        async def _main():
            async def _body():
                async with _Deadline(30.0, "unit under test"):
                    await asyncio.Event().wait()

            task = asyncio.ensure_future(_body())
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(_main())


def _gym_rows(count: int) -> list[dict]:
    """Rows shaped the way _build_inputs stamps them: each carries its own index."""
    return [{"_rowidx": i, "agent_ref": {"name": "agent"}} for i in range(count)]


class _PartialGymMethod:
    """Gym stream that fails partway through, then succeeds on re-dispatch.

    Reproduces the shape that matters: NeMo-Gym's stream dies on its first failing row,
    so every later row is lost with it even though nothing was wrong with them.
    """

    def __init__(self, fail_after_rows: int, failures_before_success: int) -> None:
        self._fail_after_rows = fail_after_rows
        self._failures_before_success = failures_before_success
        self.attempts = 0
        self.dispatched: list[list[int]] = []

    def options(self, **kwargs):
        del kwargs
        return self

    def remote(self, inputs, timer_prefix):
        del timer_prefix
        self.dispatched.append([row["_rowidx"] for row in inputs])
        attempt = self.attempts
        self.attempts += 1
        return self._stream(inputs, attempt)

    async def _stream(self, inputs, attempt):
        should_fail = attempt < self._failures_before_success
        for position, row in enumerate(inputs):
            if should_fail and position >= self._fail_after_rows:
                raise ConnectionResetError("gym stream died mid-flight")
            yield _row_result(row["_rowidx"])


async def _row_result(rowidx: int):
    """A minimally complete NeMo-Gym result, enough to build a Completion."""
    return (
        rowidx,
        {
            "input_message_log": [{"role": "user", "token_ids": [1]}],
            "message_log": [{"role": "assistant", "token_ids": [2]}],
            "full_result": {"reward": 1.0},
        },
        None,
    )


class _FakeGymMethod:
    """Mimics `env.run_rollouts.options(...).remote(...)` returning a result stream."""

    def __init__(self, rows_to_yield, hang_after) -> None:
        self._rows_to_yield = rows_to_yield
        self._hang_after = hang_after
        self.cancelled = 0

    def options(self, **kwargs):
        del kwargs
        return self

    def remote(self, inputs, timer_prefix):
        del timer_prefix
        return self._stream(len(inputs))

    async def _stream(self, num_inputs):
        async def _result(rowidx):
            return rowidx, {"input_message_log": [], "message_log": []}, None

        for rowidx in range(min(self._rows_to_yield, num_inputs)):
            yield _result(rowidx)
        if self._hang_after:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise


def _make_gym_impl(
    gym_method, *, num_generations=2, timeouts=None, row_attempts=1, stats=None
):
    from nemo_rl.experience.rollout_manager import AsyncNemoGymRolloutImpl

    impl = object.__new__(AsyncNemoGymRolloutImpl)
    impl._tokenizer = _FakeTokenizer()
    impl._task_to_env = {"nemo_gym": type("Env", (), {"run_rollouts": gym_method})()}
    impl._num_generations_per_prompt = num_generations
    impl._max_seq_len = 128
    impl._max_rollout_turns = 1
    impl._timeouts = timeouts if timeouts is not None else RolloutTimeouts()
    impl._max_gym_row_attempts = row_attempts
    # Real counters by default so row-level re-dispatches are observable; production
    # shares the owning RolloutManager's instance.
    impl._stats = stats if stats is not None else RolloutStats()
    # Upstream default; this fixture is about re-dispatch, not sample masking.
    impl._mask_env_flagged_samples = True
    # Effort-level reward shaping is off unless env.nemo_gym.effort_levels is set.
    impl._effort_config = None
    return impl


class TestGymRolloutDeadline:
    """The gym path is where the silent wedge described in the resiliency report lives.

    NeMo-Gym retries a dead vLLM endpoint forever with no HTTP timeout, so without a
    deadline here the rollout never returns and permanently holds a
    max_inflight_prompts permit.
    """

    def test_a_hung_gym_stream_raises_rollout_timeout(self):
        method = _FakeGymMethod(rows_to_yield=0, hang_after=True)
        impl = _make_gym_impl(method, timeouts=RolloutTimeouts(rollout_s=0.05))
        with pytest.raises(RolloutTimeout, match="NeMo-Gym prompt group"):
            asyncio.run(impl._run_rollouts(_gym_rows(2), Timer(), "timing/rollout"))
        assert method.cancelled == 1, "the hung stream must be cancelled"

    def test_the_deadline_spans_the_whole_stream_not_each_row(self):
        """A steady drip of fast rows must not keep resetting the budget.

        One slow row holding the group up is exactly the case that has to fire, and a
        per-await deadline would never see it.
        """
        method = _FakeGymMethod(rows_to_yield=1, hang_after=True)
        impl = _make_gym_impl(method, timeouts=RolloutTimeouts(rollout_s=0.05))
        with pytest.raises(RolloutTimeout):
            asyncio.run(impl._run_rollouts(_gym_rows(2), Timer(), "timing/rollout"))

    def test_a_truncated_stream_is_an_infra_failure_not_a_bare_runtime_error(self):
        """Rows going missing is a transport problem, so it must be retriable."""
        method = _FakeGymMethod(rows_to_yield=1, hang_after=False)
        impl = _make_gym_impl(method)
        with pytest.raises(GymTransportError, match=r"missing rows \[1\] of 2"):
            asyncio.run(impl._run_rollouts(_gym_rows(2), Timer(), "timing/rollout"))

    def test_no_deadline_configured_leaves_the_stream_unbounded(self):
        """Default config must not introduce a deadline where there was none."""
        method = _FakeGymMethod(rows_to_yield=1, hang_after=False)
        impl = _make_gym_impl(method, timeouts=RolloutTimeouts(rollout_s=None))
        # Reaches the missing-rows check rather than timing out first.
        with pytest.raises(GymTransportError):
            asyncio.run(impl._run_rollouts(_gym_rows(2), Timer(), "timing/rollout"))


class TestPartialGymRedispatch:
    """Only the rows that never arrived are re-sent.

    At num_generations_per_prompt=16, one bad row killing the stream costs 16
    regenerations if the whole group is redone. It should cost the rows that were lost.
    """

    def test_only_the_missing_rows_are_re_dispatched(self):
        # Rows 0-1 land, then the stream dies; the retry should carry rows 2-3 only.
        method = _PartialGymMethod(fail_after_rows=2, failures_before_success=1)
        impl = _make_gym_impl(method, num_generations=4, row_attempts=3)

        completions, _, _ = asyncio.run(
            impl._run_rollouts(_gym_rows(4), Timer(), "timing/rollout")
        )

        assert method.dispatched == [[0, 1, 2, 3], [2, 3]]
        assert len(completions) == 4

    def test_completed_rows_survive_across_attempts(self):
        """The point of the exercise: work already done is not thrown away.

        Attempt 1 lands rows 0-2 before the stream dies. Only 3-4 are re-sent, and the
        group finishes there -- 7 row-generations instead of the 10 a whole-group retry
        would have cost.
        """
        method = _PartialGymMethod(fail_after_rows=3, failures_before_success=2)
        impl = _make_gym_impl(method, num_generations=5, row_attempts=3)

        completions, _, _ = asyncio.run(
            impl._run_rollouts(_gym_rows(5), Timer(), "timing/rollout")
        )

        assert method.dispatched == [[0, 1, 2, 3, 4], [3, 4]]
        assert len(completions) == 5
        assert sum(len(d) for d in method.dispatched) == 7

    def test_a_stale_echo_of_a_landed_row_is_rejected(self):
        """Re-dispatch narrows the stream; an echo of an already-landed row must not win.

        Attempt 1 lands row 0 then dies. Attempt 2 carries only the rows that never
        arrived, so a stream replaying row 0 is echoing stale work -- and silently
        accepting it would overwrite a good result with an older one. This guard had no
        test at all: deleting the two lines it lives in left the suite green.
        """

        class _EchoingMethod(_PartialGymMethod):
            def __init__(self) -> None:
                super().__init__(fail_after_rows=1, failures_before_success=1)

            async def _stream(self, inputs, attempt):
                if attempt == 0:
                    yield _row_result(0)
                    raise ConnectionResetError("stream died after row 0")
                # Handed rows 1..n-1, but answers with row 0 all the same.
                yield _row_result(0)

        impl = _make_gym_impl(_EchoingMethod(), num_generations=3, row_attempts=3)

        with pytest.raises(ValueError, match="which was not dispatched"):
            asyncio.run(impl._run_rollouts(_gym_rows(3), Timer(), "timing/rollout"))

    def test_row_redispatches_are_counted(self):
        """Otherwise gym retries rows all run with every visible counter flat."""
        stats = RolloutStats()
        method = _PartialGymMethod(fail_after_rows=2, failures_before_success=1)
        impl = _make_gym_impl(method, num_generations=4, row_attempts=3, stats=stats)

        asyncio.run(impl._run_rollouts(_gym_rows(4), Timer(), "timing/rollout"))

        # Attempt 2 re-sent exactly the two rows that never arrived.
        assert method.dispatched == [[0, 1, 2, 3], [2, 3]]
        assert stats.gym_row_redispatches == 2
        assert stats.as_metrics()["rollout/gym_row_redispatch_total"] == 2.0

    def test_the_transport_error_that_lost_the_rows_is_chained(self):
        """A short-but-clean final stream otherwise reports missing rows with no cause.

        Attempt 1 dies with a transport error; attempt 2 ends cleanly without supplying
        the rest. Without the chain the operator reads "rows missing" at exactly the
        moment they need to know why.
        """

        class _ShortSecondAttempt(_PartialGymMethod):
            def __init__(self) -> None:
                super().__init__(fail_after_rows=1, failures_before_success=1)

            async def _stream(self, inputs, attempt):
                if attempt == 0:
                    yield _row_result(0)
                    raise ConnectionResetError("the transport error to blame")
                return
                yield  # unreachable; keeps this an async generator

        impl = _make_gym_impl(_ShortSecondAttempt(), num_generations=3, row_attempts=2)

        with pytest.raises(GymTransportError) as exc_info:
            asyncio.run(impl._run_rollouts(_gym_rows(3), Timer(), "timing/rollout"))
        cause = exc_info.value.__cause__
        assert isinstance(cause, ConnectionResetError)
        assert "the transport error to blame" in str(cause)

    def test_exhausting_the_row_budget_reports_a_transport_failure(self):
        method = _PartialGymMethod(fail_after_rows=1, failures_before_success=99)
        impl = _make_gym_impl(method, num_generations=4, row_attempts=2)

        with pytest.raises(ConnectionResetError):
            asyncio.run(impl._run_rollouts(_gym_rows(4), Timer(), "timing/rollout"))
        assert method.attempts == 2, "the budget bounds the re-dispatches"

    def test_a_data_failure_is_not_re_dispatched(self):
        """Another dispatch cannot help a prompt NeMo-Gym is unable to serve."""

        class _DataFailingMethod(_PartialGymMethod):
            async def _stream(self, inputs, attempt):
                del attempt
                yield _row_result(inputs[0]["_rowidx"])
                raise ValueError("NeMo Gym returned a result with no generation data")

        method = _DataFailingMethod(fail_after_rows=1, failures_before_success=99)
        impl = _make_gym_impl(method, num_generations=4, row_attempts=5)

        with pytest.raises(ValueError, match="no generation data"):
            asyncio.run(impl._run_rollouts(_gym_rows(4), Timer(), "timing/rollout"))
        assert method.attempts == 1, "a deterministic failure must not be retried"

    def test_a_single_attempt_budget_reproduces_the_old_behaviour(self):
        method = _PartialGymMethod(fail_after_rows=2, failures_before_success=1)
        impl = _make_gym_impl(method, num_generations=4, row_attempts=1)

        with pytest.raises(ConnectionResetError):
            asyncio.run(impl._run_rollouts(_gym_rows(4), Timer(), "timing/rollout"))
        assert method.attempts == 1

    def test_rows_must_carry_their_own_index(self):
        """Re-dispatch maps Gym's echoed _rowidx back onto the group, so this is a contract."""
        method = _PartialGymMethod(fail_after_rows=99, failures_before_success=0)
        impl = _make_gym_impl(method, num_generations=2, row_attempts=2)

        with pytest.raises(ValueError, match="must be stamped with their own position"):
            asyncio.run(
                impl._run_rollouts(
                    [{"agent_ref": {"name": "a"}}], Timer(), "timing/rollout"
                )
            )

    def test_the_group_deadline_spans_re_dispatches(self):
        """The budget belongs to the prompt group, not to each attempt.

        Otherwise N attempts silently multiply rollout_timeout_s by N.
        """

        class _SlowThenHang(_PartialGymMethod):
            async def _stream(self, inputs, attempt):
                if attempt == 0:
                    yield _row_result(inputs[0]["_rowidx"])
                    raise ConnectionResetError("died")
                await asyncio.Event().wait()
                yield _row_result(inputs[0]["_rowidx"])  # pragma: no cover

        method = _SlowThenHang(fail_after_rows=1, failures_before_success=1)
        impl = _make_gym_impl(
            method,
            num_generations=2,
            row_attempts=5,
            timeouts=RolloutTimeouts(rollout_s=0.1),
        )
        with pytest.raises(RolloutTimeout, match="NeMo-Gym prompt group"):
            asyncio.run(impl._run_rollouts(_gym_rows(2), Timer(), "timing/rollout"))


class TestClassifyGenerationFailure:
    def test_infra_maps_to_generation_unavailable(self):
        out = _classify_generation_failure(
            ConnectionResetError("x"), prompt_idx=1, traj_idx=2
        )
        assert isinstance(out, GenerationUnavailable)

    def test_data_maps_to_rollout_data_failure(self):
        out = _classify_generation_failure(
            AssertionError("non-contiguous"), prompt_idx=1, traj_idx=2
        )
        assert isinstance(out, RolloutDataFailure)

    @pytest.mark.parametrize(
        "cause",
        [
            RuntimeError("EngineDeadError"),  # vLLM, worker still alive
            RuntimeError("CUDA out of memory"),
            ValueError("something we have never seen"),
        ],
    )
    def test_a_ray_boundary_error_from_generation_is_infrastructure(self, cause):
        """The boundary destroys the cause, so classify by the call site instead.

        An exception raised inside a still-LIVING generation worker arrives as a bare
        RayTaskError -- not a RayActorError, and not in _INFRA_TYPES -- so the generic
        classifier can only fall through to DATA, and the prompt got 2 attempts instead
        of the 5 the fleet-failure path exists for.

        Scoped to this call site on purpose: globally, unrecognized-means-DATA is still
        the right default. Here we know the RPC was a generation, so "that shard could
        not serve it" is the correct reading whatever the cause was.
        """
        exc = ray.exceptions.RayTaskError("gen", "traceback", cause)
        out = _classify_generation_failure(exc, prompt_idx=1, traj_idx=2)
        assert isinstance(out, GenerationUnavailable)
        assert classify_rollout_failure(out) is FailureClass.INFRA

    def test_a_plain_data_error_is_still_data(self):
        """The widening is Ray-boundary-only; ordinary bad prompts are unaffected."""
        out = _classify_generation_failure(
            ValueError("token ids are not contiguous"), prompt_idx=1, traj_idx=2
        )
        assert isinstance(out, RolloutDataFailure)
        assert classify_rollout_failure(out) is FailureClass.DATA

    def test_both_branches_are_rollout_failures(self):
        for exc in (ConnectionResetError("x"), AssertionError("y")):
            out = _classify_generation_failure(exc, prompt_idx=0, traj_idx=0)
            assert isinstance(out, RolloutFailure)
