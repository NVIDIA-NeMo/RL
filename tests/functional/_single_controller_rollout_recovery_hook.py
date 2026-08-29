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

"""Test-only SC entrypoint for deterministic unfinished-rollout recovery.

The first process parks one selected rollout after controller admission. The
second process records its redispatch and successful canonical TQ commit. The
wrapper is injected driver-side into ``SingleControllerActorArgs`` so the
production request path has no environment-variable or timing hook.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, cast

from examples import run_grpo_single_controller
from nemo_rl.experience.rollout_manager import RolloutManager, RolloutOutcome


class _InstrumentedRolloutManager:
    """Delegate every operation except the deterministic recovery test cut."""

    def __init__(
        self,
        delegate: Any,
        *,
        events_path: Path,
        block_target_step: int | None,
    ) -> None:
        self._delegate = delegate
        self._events_path = events_path
        self._block_target_step = block_target_step
        self._blocked = False

    def __getattr__(self, name: str) -> Any:
        delegate = self.__dict__.get("_delegate")
        if delegate is None:
            raise AttributeError(name)
        return getattr(delegate, name)

    @property
    def _tq_buffer(self) -> Any:
        return self._delegate._tq_buffer

    @_tq_buffer.setter
    def _tq_buffer(self, value: Any) -> None:
        self._delegate._tq_buffer = value

    def _append_event(self, event: str, **fields: Any) -> None:
        self._events_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"event": event, **fields}
        with self._events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    async def generate_and_push(
        self,
        input_sample: Any,
        *,
        target_step: int | None = None,
        inflight_registry: Any = None,
        lineage_group_id: str | None = None,
    ) -> RolloutOutcome:
        fields = {
            "group_id": lineage_group_id,
            "prompt_idx": int(input_sample["idx"]),
            "target_step": target_step,
        }
        self._append_event("dispatch", **fields)

        if (
            not self._blocked
            and self._block_target_step is not None
            and target_step == self._block_target_step
        ):
            if lineage_group_id is None:
                raise RuntimeError("recovery test expected a lineage-tracked group")
            self._blocked = True
            self._append_event("blocked_before_tq_commit", **fields)
            print(
                "recovery functional hook: blocked admitted "
                f"group_id={lineage_group_id} target_step={target_step}",
                flush=True,
            )
            # The phase-1 timeout checkpoint terminates the process and cancels
            # this task. No polling or wall-clock race controls the checkpoint cut.
            await asyncio.Event().wait()

        outcome = await self._delegate.generate_and_push(
            input_sample,
            target_step=target_step,
            inflight_registry=inflight_registry,
            lineage_group_id=lineage_group_id,
        )
        if outcome is RolloutOutcome.COMMITTED:
            self._append_event("canonical_tq_commit", **fields)
        return outcome


_original_setup_single_controller = run_grpo_single_controller.setup_single_controller


def _setup_with_recovery_hook(*args: Any, **kwargs: Any) -> Any:
    actor_args, timing_metrics = _original_setup_single_controller(*args, **kwargs)
    events_path = Path(os.environ["SC_RECOVERY_TEST_EVENTS"])
    raw_target_step = os.environ.get("SC_RECOVERY_TEST_BLOCK_TARGET_STEP")
    block_target_step = int(raw_target_step) if raw_target_step is not None else None
    actor_args.rollout_manager = cast(
        RolloutManager,
        _InstrumentedRolloutManager(
            actor_args.rollout_manager,
            events_path=events_path,
            block_target_step=block_target_step,
        ),
    )
    return actor_args, timing_metrics


run_grpo_single_controller.setup_single_controller = _setup_with_recovery_hook


if __name__ == "__main__":
    run_grpo_single_controller.main()
