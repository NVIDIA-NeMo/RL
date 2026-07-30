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

from nemo_rl.experience.failures import (
    GenerationUnavailable,
    RolloutDataFailure,
    RolloutFailure,
)
from nemo_rl.experience.rollout_manager import (
    AsyncRolloutImpl,
    RolloutManager,
    _classify_generation_failure,
    _gather_cancelling_siblings,
)


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return f"<{len(ids)} tokens>"


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
        yield (
            0,
            {
                "output_ids": torch.tensor([[1, 2, 3, 4]]),
                "unpadded_sequence_lengths": torch.tensor([4]),
            },
        )


def _make_impl(generation, *, num_generations=1, max_turns=1) -> AsyncRolloutImpl:
    """Build an AsyncRolloutImpl without firing its real __init__."""
    impl = object.__new__(AsyncRolloutImpl)
    impl._tokenizer = _FakeTokenizer()
    impl._task_to_env = {}
    impl._num_generations_per_prompt = num_generations
    impl._max_seq_len = 128
    impl._max_rollout_turns = max_turns
    impl._policy_generation = generation
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

        manager = object.__new__(RolloutManager)
        manager._impl = impl
        manager._tokenizer = None
        manager._num_generations_per_prompt = 1
        manager._tq_buffer = buffer
        manager._weight_version = 0

        with pytest.raises(RolloutFailure):
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

    def test_both_branches_are_rollout_failures(self):
        for exc in (ConnectionResetError("x"), AssertionError("y")):
            out = _classify_generation_failure(exc, prompt_idx=0, traj_idx=0)
            assert isinstance(out, RolloutFailure)
