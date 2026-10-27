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

"""The RolloutCheckpointing contract the single controller relies on."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Optional

import pytest
import torch

from nemo_rl.algorithms.async_utils.replay_buffer import (
    REPLACEMENT_RESERVE_FILENAME,
    REPLAY_BUFFER_METADATA_FILENAME,
    DataPlaneCheckpointBarrier,
)
from nemo_rl.algorithms.rollout_checkpointing import (
    ControllerRolloutState,
    ControllerSidecars,
    DataPlaneRolloutCheckpointing,
    NoRolloutCheckpointing,
    RolloutCheckpointing,
    write_controller_sidecars,
)
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.experience.environment_checkpoint import (
    EnvironmentCheckpoint,
    EnvironmentSnapshot,
    NoEnvironmentCheckpoint,
    RestoredEnvironmentRollouts,
)
from nemo_rl.experience.rollout_recovery import ROLLOUT_RECOVERY_STATE_FILENAME

_ACTOR_CLS = SingleControllerActor.__ray_metadata__.modified_class


class _StatefulEnvironment(EnvironmentCheckpoint):
    """An environment backend that keeps rollout state between turns."""

    @property
    def holds_rollout_state(self) -> bool:
        return True

    async def check_startup(self) -> None:
        return None

    @contextlib.asynccontextmanager
    async def snapshot(
        self, checkpoint_path: Path
    ) -> AsyncIterator[Optional[EnvironmentSnapshot]]:
        del checkpoint_path
        yield None

    @contextlib.asynccontextmanager
    async def restored(
        self, checkpoint_path: Path
    ) -> AsyncIterator[RestoredEnvironmentRollouts]:
        del checkpoint_path
        yield RestoredEnvironmentRollouts()

    def is_unavailable(self, error: BaseException) -> bool:
        del error
        return False


def _component(
    environment_checkpoint: Optional[EnvironmentCheckpoint] = None,
) -> DataPlaneRolloutCheckpointing:
    return DataPlaneRolloutCheckpointing(
        host=None,  # type: ignore[arg-type]
        master_config=None,  # type: ignore[arg-type]
        dp_client=None,
        partition_id="rollout_data",
        buffer=None,  # type: ignore[arg-type]
        rollout_manager=None,  # type: ignore[arg-type]
        checkpointer=None,  # type: ignore[arg-type]
        timeout=None,  # type: ignore[arg-type]
        data_plane_barrier=DataPlaneCheckpointBarrier(),
        environment_checkpoint=environment_checkpoint or NoEnvironmentCheckpoint(),
        bootstrap_identity=None,
        teacher_coordinator=None,
    )


def _state(**overrides: Any) -> ControllerRolloutState:
    fields: dict[str, Any] = {
        "train_steps": 3,
        "trainer_version": 3,
        "current_epoch": 1,
        "dataloader_state": {"next_batch": 4},
        "sampler_dispatch_index": 4,
        "sampler_supports_buffer_checkpoint": True,
        "replacement_reserve": [],
        "batch_shortfall": {},
        "sampler_stamps_target_steps": False,
        "finalizer_metrics_by_group": {},
        "finalizer_rewards_by_group": {},
        "rollout_recovery_enabled": True,
    }
    fields.update(overrides)
    return ControllerRolloutState(**fields)


def test_an_implementation_missing_a_member_fails_when_built() -> None:
    class _Partial(RolloutCheckpointing):
        def background_tasks(self) -> list:
            return []

    with pytest.raises(TypeError, match="abstract"):
        _Partial()  # type: ignore[abstract]


def test_no_rollout_checkpointing_runs_nothing_and_keeps_plain_sidecars() -> None:
    component = NoRolloutCheckpointing()

    async def exercise() -> ControllerSidecars:
        async with component.trainer_checkpoint():
            async with component.optimizer_commit(drain_snapshot=True):
                pass
            async with DataPlaneCheckpointBarrier().checkpoint() as cut:
                return await component.save_with_trainer_checkpoint(
                    cut, "unused", _state(replacement_reserve=[{"idx": 1}])
                )

    sidecars = asyncio.run(exercise())

    assert component.background_tasks() == []
    assert not component.stop_requested
    # A trainer checkpoint still needs the dataloader and spare prompts.
    assert sidecars == ControllerSidecars(
        dataloader_state={"next_batch": 4},
        replacement_reserve=[{"idx": 1}],
    )


def test_optimizer_commit_window_opens_and_closes() -> None:
    component = _component()

    async def exercise() -> None:
        async with component.optimizer_commit(drain_snapshot=False):
            assert component._optimizer_commit_in_progress
            assert component._optimizer_commit_started.is_set()
        assert not component._optimizer_commit_in_progress

    asyncio.run(exercise())


def test_optimizer_commit_stays_open_when_the_step_fails() -> None:
    """As before the move: a failed step leaves the window open, so no snapshot runs."""
    component = _component()

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="step failed"):
            async with component.optimizer_commit(drain_snapshot=False):
                raise RuntimeError("step failed")

    asyncio.run(exercise())

    assert component._optimizer_commit_in_progress


def test_draining_optimizer_commit_waits_for_an_inflight_save() -> None:
    component = _component()

    async def exercise() -> None:
        async with component._checkpoint_save_lock:
            commit = asyncio.create_task(
                component.optimizer_commit(drain_snapshot=True).__aenter__()
            )
            await asyncio.sleep(0.01)
            assert not commit.done()
            assert not component._optimizer_commit_in_progress
        await asyncio.wait_for(commit, timeout=1.0)
        assert component._optimizer_commit_in_progress

    asyncio.run(exercise())


@pytest.mark.parametrize("holds_rollout_state", [False, True])
def test_trainer_checkpoint_wakes_the_snapshot_pump_only_for_environment_state(
    holds_rollout_state: bool,
) -> None:
    component = _component(
        _StatefulEnvironment() if holds_rollout_state else NoEnvironmentCheckpoint()
    )

    async def exercise() -> None:
        async with component.trainer_checkpoint():
            assert component._checkpoint_save_lock.locked()
        assert not component._checkpoint_save_lock.locked()

    asyncio.run(exercise())

    assert component._rollout_checkpoint_wakeup.is_set() is holds_rollout_state


def test_a_failed_trainer_checkpoint_does_not_wake_the_snapshot_pump() -> None:
    component = _component(_StatefulEnvironment())

    async def exercise() -> None:
        with pytest.raises(OSError):
            async with component.trainer_checkpoint():
                raise OSError("disk full")

    asyncio.run(exercise())

    assert not component._rollout_checkpoint_wakeup.is_set()


def test_write_controller_sidecars_writes_only_what_is_present(tmp_path: Path) -> None:
    sidecars = ControllerSidecars(
        dataloader_state={"next_batch": 4},
        replacement_reserve=[],
        replay_metadata={"groups": [], "manifest_digest": "d"},  # type: ignore[typeddict-item]
        rollout_recovery_payload=b"ledger",
    )

    written = asyncio.run(write_controller_sidecars(tmp_path, sidecars))

    assert sorted(Path(path).name for path in written) == sorted(
        [
            "train_dataloader.pt",
            REPLAY_BUFFER_METADATA_FILENAME,
            ROLLOUT_RECOVERY_STATE_FILENAME,
        ]
    )
    assert not (tmp_path / REPLACEMENT_RESERVE_FILENAME).exists()
    assert torch.load(tmp_path / "train_dataloader.pt") == {"next_batch": 4}
    assert (tmp_path / ROLLOUT_RECOVERY_STATE_FILENAME).read_bytes() == b"ledger"


@pytest.mark.parametrize(
    ("interval_s", "enabled", "save_data_plane", "expected"),
    [
        (None, False, False, NoRolloutCheckpointing),
        (None, True, False, NoRolloutCheckpointing),
        (None, True, True, DataPlaneRolloutCheckpointing),
        (60.0, True, True, DataPlaneRolloutCheckpointing),
    ],
)
def test_controller_gets_no_component_when_it_saves_no_rollout_state(
    interval_s: Optional[float],
    enabled: bool,
    save_data_plane: bool,
    expected: type,
) -> None:
    controller = object.__new__(_ACTOR_CLS)
    controller._master_config = SimpleNamespace(
        checkpointing={"enabled": enabled, "save_data_plane": save_data_plane},
        rollout_checkpointing=SimpleNamespace(snapshot_attempt_interval_s=interval_s),
    )
    for name in (
        "_dp_client",
        "_buffer",
        "_rollout_manager",
        "_checkpointer",
        "_timeout",
        "_bootstrap_identity",
        "_teacher_coordinator",
    ):
        setattr(controller, name, None)
    controller._partition_id = "rollout_data"
    controller._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
    controller._environment_checkpoint = NoEnvironmentCheckpoint()

    assert type(controller._build_rollout_checkpointing()) is expected
