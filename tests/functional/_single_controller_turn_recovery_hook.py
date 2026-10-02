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

"""Test-only SC entrypoint that records turn-recovery ownership.

The wrapper does not alter scheduling or checkpoint timing. It records the
structured recovery identity once each Gym submission is marked dispatched,
when Gym refuses one at checkpoint admission, and after each completion has
been sealed in the RL ledger. A two-process functional test can
therefore prove that a restored episode returns to the same Gym instance under
``gym_attempt + 1`` without depending on log wording.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from examples import run_grpo_single_controller
from nemo_rl.experience.rollout_manager import RolloutCompletionCallback
from nemo_rl.experience.rollout_recovery import RecoveryGranularity


class _InstrumentedNemoGymRolloutImpl:
    """Delegate Gym rollouts while recording durable identity transitions."""

    def __init__(
        self,
        delegate: Any,
        *,
        recovery_ledger: Any,
        events_path: Path,
    ) -> None:
        self._delegate = delegate
        # Captured before the hook patches ``delegate.run_rollout`` to point here.
        self._delegate_run_rollout = delegate.run_rollout
        self._recovery_ledger = recovery_ledger
        self._events_path = events_path

    def __getattr__(self, name: str) -> Any:
        delegate = self.__dict__.get("_delegate")
        if delegate is None:
            raise AttributeError(name)
        return getattr(delegate, name)

    def _append_event(self, event: str, **fields: Any) -> None:
        self._events_path.parent.mkdir(parents=True, exist_ok=True)
        with self._events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": event, **fields}, sort_keys=True) + "\n")

    def _find_group(self, rollout_ids: list[str]) -> Any:
        rollout_id_set = set(rollout_ids)
        matches = [
            group
            for group in self._recovery_ledger.groups()
            if rollout_id_set.intersection(group.gate_rollout_ids)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "turn recovery hook could not uniquely resolve rollout IDs "
                f"to one ledger group: ids={rollout_ids!r}, matches="
                f"{[group.group_id for group in matches]!r}"
            )
        return matches[0]

    @staticmethod
    def _sibling_fields(group: Any, generation_index: int) -> dict[str, Any]:
        sibling = group.siblings[generation_index]
        attempt = sibling.current_attempt
        gym_rollout_id, gym_attempt = group.gym_episode(generation_index)
        return {
            "generation_index": generation_index,
            "logical_rollout_id": group.logical_rollout_id(generation_index),
            "gate_rollout_id": group.gate_rollout_id(generation_index),
            "gym_rollout_id": gym_rollout_id,
            "gym_instance_id": attempt.gym_instance_id,
            "gym_attempt": gym_attempt,
            "attempt_id": attempt.attempt_id,
            "status": attempt.status.value,
        }

    async def run_rollout(
        self,
        input_sample: Any,
        *,
        rollout_ids: list[str] | None = None,
        generation_indices: list[int] | None = None,
        on_completion: RolloutCompletionCallback | None = None,
        recovery_granularity: RecoveryGranularity = RecoveryGranularity.SIBLING,
        gym_instance_id: str | None = None,
        dispatch_recorder: Any = None,
    ) -> Any:
        if rollout_ids is None or generation_indices is None or on_completion is None:
            raise RuntimeError("turn recovery hook requires token capture")
        if gym_instance_id is None:
            raise RuntimeError("turn recovery hook requires a Gym checkpoint owner")
        if dispatch_recorder is None:
            raise RuntimeError("turn recovery hook requires a dispatch recorder")

        group = self._find_group(rollout_ids)
        indices = list(generation_indices)
        common = {
            "group_id": group.group_id,
            "prompt_idx": int(input_sample["idx"]),
            "target_step": group.target_step,
            "generation_indices": indices,
            "gym_instance_id": gym_instance_id,
        }

        def _record_submission(event: str, submitted: list[int]) -> None:
            refreshed = self._find_group(rollout_ids)
            self._append_event(
                event,
                **common,
                siblings=[
                    self._sibling_fields(refreshed, index) for index in submitted
                ],
            )

        async def _record_completion(generation_index: int, completion: Any) -> None:
            await on_completion(generation_index, completion)
            refreshed = self._find_group(rollout_ids)
            self._append_event(
                "completion",
                **common,
                sibling=self._sibling_fields(refreshed, generation_index),
            )

        result = await self._delegate_run_rollout(
            input_sample,
            rollout_ids=rollout_ids,
            generation_indices=indices,
            on_completion=_record_completion,
            recovery_granularity=recovery_granularity,
            gym_instance_id=gym_instance_id,
            dispatch_recorder=_RecordingDispatchRecorder(
                dispatch_recorder, record=_record_submission
            ),
        )
        self._append_event("rollout_returned", **common)
        return result


class _RecordingDispatchRecorder:
    """Record each physical Gym submission once the ledger marks it dispatched.

    A ``dispatch`` event is written right after Ray accepts the submission, so it
    sees the attempt's real owner and status. A ``refused`` event marks a
    submission Gym turned away at checkpoint admission; it never ran.
    """

    def __init__(
        self, delegate: Any, *, record: Callable[[str, list[int]], None]
    ) -> None:
        self._delegate = delegate
        self._record = record

    @asynccontextmanager
    async def submitting(
        self, generation_indices: Sequence[int]
    ) -> AsyncIterator[Callable[[], None]]:
        indices = list(generation_indices)
        async with self._delegate.submitting(indices) as mark_dispatched:

            def mark_and_record() -> None:
                mark_dispatched()
                self._record("dispatch", indices)

            yield mark_and_record

    async def refused(self, generation_index: int) -> None:
        await self._delegate.refused(generation_index)
        self._record("refused", [generation_index])


_original_setup_single_controller = run_grpo_single_controller.setup_single_controller


def _setup_with_turn_recovery_hook(*args: Any, **kwargs: Any) -> Any:
    actor_args, timing_metrics = _original_setup_single_controller(*args, **kwargs)
    manager = actor_args.rollout_manager
    impl = manager._impl
    instrumented = _InstrumentedNemoGymRolloutImpl(
        impl,
        recovery_ledger=manager.recovery_ledger,
        events_path=Path(os.environ["SC_TURN_RECOVERY_TEST_EVENTS"]),
    )
    # Patch the instance method rather than replacing ``_impl``: RolloutManager
    # gates checkpoint admission and turn recovery on
    # ``isinstance(self._impl, AsyncNemoGymRolloutImpl)``.
    impl.run_rollout = instrumented.run_rollout
    return actor_args, timing_metrics


run_grpo_single_controller.setup_single_controller = _setup_with_turn_recovery_hook


if __name__ == "__main__":
    run_grpo_single_controller.main()
