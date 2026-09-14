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

"""Test-only SC entrypoint for deterministic sibling-level recovery.

The first process lets one sibling become ledger-sealed, then parks the next
completion before the ledger records it. Earlier-step completions wait for that
cut, guaranteeing that the step-1 checkpoint contains the selected partial
group. The second process records which generation indices are redispatched.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from examples import run_grpo_single_controller
from nemo_rl.experience.rollout_manager import RolloutCompletionCallback
from nemo_rl.experience.rollout_recovery import RecoveryGranularity


class _InstrumentedNemoGymRolloutImpl:
    """Delegate Gym rollouts while controlling one streamed-completion cut."""

    def __init__(
        self,
        delegate: Any,
        *,
        recovery_ledger: Any,
        events_path: Path,
        block_target_step: int | None,
    ) -> None:
        self._delegate = delegate
        self._recovery_ledger = recovery_ledger
        self._events_path = events_path
        self._block_target_step = block_target_step
        self._selected_group_id: str | None = None
        # Construct this lazily inside the Ray actor's event loop. The rollout
        # manager and this test wrapper are built driver-side and serialized into
        # that actor, so an eagerly created asyncio primitive could bind to the
        # wrong loop.
        self._selected_sibling_sealed: asyncio.Event | None = None

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
                "sibling recovery hook could not uniquely resolve rollout IDs "
                f"to one ledger group: ids={rollout_ids!r}, matches="
                f"{[group.group_id for group in matches]!r}"
            )
        return matches[0]

    def _sibling_sealed_event(self) -> asyncio.Event:
        if self._selected_sibling_sealed is None:
            self._selected_sibling_sealed = asyncio.Event()
        return self._selected_sibling_sealed

    async def run_rollout(
        self,
        input_sample: Any,
        *,
        rollout_ids: list[str] | None = None,
        generation_indices: list[int] | None = None,
        on_completion: RolloutCompletionCallback | None = None,
        recovery_granularity: RecoveryGranularity = RecoveryGranularity.SIBLING,
    ) -> Any:
        if rollout_ids is None or generation_indices is None or on_completion is None:
            raise RuntimeError(
                "sibling recovery hook requires the token-capture rollout path"
            )
        if recovery_granularity is not RecoveryGranularity.SIBLING:
            raise RuntimeError(
                "sibling recovery hook requires sibling recovery granularity"
            )

        group = self._find_group(rollout_ids)
        indices = list(generation_indices)
        fields = {
            "group_id": group.group_id,
            "prompt_idx": int(input_sample["idx"]),
            "target_step": group.target_step,
            "generation_indices": indices,
            "rollout_ids": list(rollout_ids),
        }
        self._append_event("dispatch", **fields)

        selected = False
        if (
            self._block_target_step is not None
            and group.target_step == self._block_target_step
            and self._selected_group_id is None
        ):
            # Selection occurs before the first await, so concurrent rollout tasks
            # cannot select two groups on this event loop.
            self._selected_group_id = group.group_id
            selected = True

        sealed_in_selected_call = False

        async def _instrumented_completion(
            generation_index: int, completion: Any
        ) -> None:
            nonlocal sealed_in_selected_call
            completion_fields = {
                **fields,
                "generation_index": generation_index,
                "rollout_id": rollout_ids[generation_index],
            }
            if selected:
                if not sealed_in_selected_call:
                    await on_completion(generation_index, completion)
                    sealed_in_selected_call = True
                    self._append_event("sibling_sealed", **completion_fields)
                    self._sibling_sealed_event().set()
                    return

                self._append_event("blocked_before_ledger_seal", **completion_fields)
                print(
                    "sibling recovery functional hook: blocked group="
                    f"{group.group_id} generation_index={generation_index}",
                    flush=True,
                )
                await asyncio.Event().wait()

            # Do not allow the preceding train step to complete until the selected
            # lookahead group has one durable sibling. This removes checkpoint timing
            # from the test: the step-1 save cannot occur before the intended cut.
            if (
                self._block_target_step is not None
                and group.target_step is not None
                and group.target_step < self._block_target_step
            ):
                await self._sibling_sealed_event().wait()
            await on_completion(generation_index, completion)

        result = await self._delegate.run_rollout(
            input_sample,
            rollout_ids=rollout_ids,
            generation_indices=indices,
            on_completion=_instrumented_completion,
            recovery_granularity=recovery_granularity,
        )
        self._append_event("capture_complete", **fields)
        return result


_original_setup_single_controller = run_grpo_single_controller.setup_single_controller


def _setup_with_sibling_recovery_hook(*args: Any, **kwargs: Any) -> Any:
    actor_args, timing_metrics = _original_setup_single_controller(*args, **kwargs)
    events_path = Path(os.environ["SC_SIBLING_RECOVERY_TEST_EVENTS"])
    raw_target_step = os.environ.get("SC_SIBLING_RECOVERY_BLOCK_TARGET_STEP")
    block_target_step = int(raw_target_step) if raw_target_step is not None else None
    manager = actor_args.rollout_manager
    manager._impl = _InstrumentedNemoGymRolloutImpl(
        manager._impl,
        recovery_ledger=manager.recovery_ledger,
        events_path=events_path,
        block_target_step=block_target_step,
    )
    return actor_args, timing_metrics


run_grpo_single_controller.setup_single_controller = _setup_with_sibling_recovery_hook


if __name__ == "__main__":
    run_grpo_single_controller.main()
