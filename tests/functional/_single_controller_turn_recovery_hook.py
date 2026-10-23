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

Gym's policy calls go through ``_gym_policy_proxy``, which holds the first one
carrying a tool result when ``SC_TURN_RECOVERY_HOLD_FIRST_TOOL_RESULT=1``, so
phase one always publishes a checkpoint at a post-tool Workplace boundary, and
forces ``SC_TURN_RECOVERY_FOLLOW_UP_TOOL`` as each episode's next call.
Otherwise the wrapper does not alter scheduling or checkpoint timing. It records the
structured recovery identity once each Gym submission is marked dispatched,
when Gym refuses one at checkpoint admission, after each completion has been
sealed in the RL ledger (with the model calls its receipt names), and which
episodes a Gym restore installed. A two-process functional test can therefore
prove that a restored episode returns to the same Gym instance under
``gym_attempt + 1`` and continues from its saved turn, without depending on log
wording.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from examples import run_grpo_single_controller
from nemo_rl.algorithms.single_controller_utils import setup as sc_setup
from nemo_rl.experience.rollout_manager import RolloutCompletionCallback
from nemo_rl.experience.rollout_recovery import RecoveryGranularity
from tests.functional._gym_policy_proxy import PolicyHoldProxy
from tests.functional._single_controller_hook_utils import append_event, find_group


class _InstrumentedNemoGymRolloutImpl:
    """Delegate Gym rollouts while recording durable identity transitions."""

    def __init__(
        self,
        delegate: Any,
        *,
        recovery_ledger: Any,
        events_path: Path,
    ) -> None:
        # Captured before the hook patches ``delegate.run_rollout`` to point here.
        self._delegate_run_rollout = delegate.run_rollout
        self._recovery_ledger = recovery_ledger
        self._events_path = events_path

    @staticmethod
    def _sibling_fields(group: Any, generation_index: int) -> dict[str, Any]:
        sibling = group.siblings[generation_index]
        attempt = sibling.current_attempt
        episode = group.environment_episode(generation_index)
        gym_rollout_id, gym_attempt = episode.rollout_id, episode.attempt
        return {
            "generation_index": generation_index,
            "logical_rollout_id": group.logical_rollout_id(generation_index),
            "gate_rollout_id": group.gate_rollout_id(generation_index),
            "gym_rollout_id": gym_rollout_id,
            "gym_instance_id": attempt.environment_instance_id,
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
        environment_instance_id: str | None = None,
        dispatch_recorder: Any = None,
    ) -> Any:
        if rollout_ids is None or generation_indices is None or on_completion is None:
            raise RuntimeError("turn recovery hook requires token capture")
        if environment_instance_id is None:
            raise RuntimeError("turn recovery hook requires a Gym checkpoint owner")
        if dispatch_recorder is None:
            raise RuntimeError("turn recovery hook requires a dispatch recorder")

        group = find_group(self._recovery_ledger, rollout_ids, hook="turn")
        indices = list(generation_indices)
        common = {
            "group_id": group.group_id,
            "prompt_idx": int(input_sample["idx"]),
            "target_step": group.target_step,
            "generation_indices": indices,
            "gym_instance_id": environment_instance_id,
        }

        def _record_submission(event: str, submitted: list[int]) -> None:
            refreshed = find_group(self._recovery_ledger, rollout_ids, hook="turn")
            append_event(
                self._events_path,
                event,
                **common,
                siblings=[
                    self._sibling_fields(refreshed, index) for index in submitted
                ],
            )

        async def _record_completion(generation_index: int, completion: Any) -> None:
            await on_completion(generation_index, completion)
            refreshed = find_group(self._recovery_ledger, rollout_ids, hook="turn")
            receipt = (completion.env_extras or {}).get("ng_receipt") or {}
            append_event(
                self._events_path,
                "completion",
                **common,
                sibling=self._sibling_fields(refreshed, generation_index),
                reward=completion.reward,
                # A continued episode keeps the calls it made before the cut.
                receipt_model_call_ids=[
                    str(record["model_call_id"])
                    for record in receipt.get("manifest") or []
                    if "model_call_id" in record
                ],
            )

        result = await self._delegate_run_rollout(
            input_sample,
            rollout_ids=rollout_ids,
            generation_indices=indices,
            on_completion=_record_completion,
            recovery_granularity=recovery_granularity,
            environment_instance_id=environment_instance_id,
            dispatch_recorder=_RecordingDispatchRecorder(
                dispatch_recorder, record=_record_submission
            ),
        )
        append_event(self._events_path, "rollout_returned", **common)
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


class _RecordingRestartLedger:
    """Record which episodes a Gym restore installed, as restart sees them.

    ``prepare_for_restart`` receives exactly the episodes Gym restored, after
    the restore succeeded, so a restored episode is distinguishable from one
    that started over under a fresh Gym session.
    """

    def __init__(self, ledger: Any, *, events_path: Path) -> None:
        self._prepare_for_restart = ledger.prepare_for_restart
        self._events_path = events_path

    def prepare_for_restart(self, cut: Any, **kwargs: Any) -> Any:
        restored = kwargs.get("continued_attempts") or set()
        append_event(
            self._events_path,
            "gym_restore",
            episodes=[
                {
                    "gym_instance_id": instance,
                    "gym_rollout_id": rollout,
                    "gym_attempt": attempt,
                }
                for instance, rollout, attempt in sorted(restored)
            ],
        )
        return self._prepare_for_restart(cut, **kwargs)


_original_spinup_gym = sc_setup._spinup_gym
# Kept alive for the life of the driver: Gym holds these URLs for the whole run.
_policy_proxies: list[PolicyHoldProxy] = []


def _spinup_gym_behind_policy_proxy(
    *args: Any, base_urls: list[str], **kwargs: Any
) -> Any:
    proxy = PolicyHoldProxy(
        base_urls,
        hold_first_tool_result=os.environ.get("SC_TURN_RECOVERY_HOLD_FIRST_TOOL_RESULT")
        == "1",
        follow_up_tool=os.environ.get("SC_TURN_RECOVERY_FOLLOW_UP_TOOL") or None,
        events_path=Path(os.environ["SC_TURN_RECOVERY_TEST_EVENTS"]),
    ).start()
    _policy_proxies.append(proxy)
    return _original_spinup_gym(*args, base_urls=proxy.base_urls, **kwargs)


sc_setup._spinup_gym = _spinup_gym_behind_policy_proxy

_original_setup_single_controller = run_grpo_single_controller.setup_single_controller


def _setup_with_turn_recovery_hook(*args: Any, **kwargs: Any) -> Any:
    actor_args, timing_metrics = _original_setup_single_controller(*args, **kwargs)
    manager = actor_args.rollout_manager
    impl = manager._impl
    events_path = Path(os.environ["SC_TURN_RECOVERY_TEST_EVENTS"])
    instrumented = _InstrumentedNemoGymRolloutImpl(
        impl,
        recovery_ledger=manager.recovery_ledger,
        events_path=events_path,
    )
    ledger = manager.recovery_ledger
    ledger.prepare_for_restart = _RecordingRestartLedger(
        ledger, events_path=events_path
    ).prepare_for_restart
    # Patch the instance method rather than replacing ``_impl``: RolloutManager
    # gates checkpoint admission and turn recovery on
    # ``isinstance(self._impl, AsyncNemoGymRolloutImpl)``.
    impl.run_rollout = instrumented.run_rollout
    return actor_args, timing_metrics


run_grpo_single_controller.setup_single_controller = _setup_with_turn_recovery_hook


if __name__ == "__main__":
    run_grpo_single_controller.main()
