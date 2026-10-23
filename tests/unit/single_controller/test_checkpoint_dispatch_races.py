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

"""Controller checkpoint cuts around in-order rollout admission.

These tests isolate the liveness hole that a replay-buffer-only checkpoint
cannot close:

* the dataloader has advanced past a batch;
* the sampler has admitted that batch and persisted dispatch_index=7;
* none of its prompt groups committed before the data-plane snapshot.

Restoring only the cursor correctly makes the next *new* admission step 8.
Recovery must therefore replay the owned batch at its saved target step 7
without admitting it a second time.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import threading
import warnings
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, TypeVar, cast
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.algorithms.async_utils.replay_buffer import (
    REPLAY_BUFFER_METADATA_FILENAME,
    DataPlaneCheckpointBarrier,
    DataPlaneMutationCut,
    TQReplayBuffer,
)
from nemo_rl.algorithms.async_utils.staleness_sampler import InOrderSampler
from nemo_rl.algorithms.grpo import _initial_grpo_save_state
from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
from nemo_rl.algorithms.rollout_checkpointing import (
    CheckpointToRestore,
    DataPlaneRolloutCheckpointing,
    RolloutRestorer,
)
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.gym_checkpoint_adapter import (
    GymCheckpointEpisode,
)
from nemo_rl.environments.gym_checkpoint_coordinator import (
    GYM_CHECKPOINT_SCHEMA_VERSION,
    GymCheckpointCommitResult,
    GymCheckpointCoordinator,
    GymCheckpointManifest,
    GymCheckpointNotReady,
    load_gym_checkpoint_manifest,
    write_gym_checkpoint_manifest,
)
from nemo_rl.environments.gym_environment_checkpoint import GymEnvironmentCheckpoint
from nemo_rl.experience.environment_checkpoint import NoEnvironmentCheckpoint
from nemo_rl.experience.rollout_manager import (
    RequestDeadlineRegistry,
    RolloutOutcome,
    _Deadline,
)
from nemo_rl.experience.rollout_recovery import (
    ROLLOUT_RECOVERY_SCHEMA_VERSION,
    ROLLOUT_RECOVERY_STATE_FILENAME,
    PromptGroupPhase,
    PromptGroupRecoveryRecord,
    RecoveryTargetLevel,
    RolloutAttemptStatus,
    RolloutRecoveryLedger,
    build_rollout_recovery_state,
)
from tests.unit.environments.gym_checkpoint_fakes import FakeGymActor, gym_coordinator
from tests.unit.single_controller._restore_helpers import (
    restorer_for,
    use_dispatch,
)
from tests.unit.single_controller._checkpoint_scenarios import (
    _record,
    patch_converter,
)
from tests.unit.single_controller.test_checkpointing import (
    _actor_master_config,
    _FakeDataloader,
    _make_actor_args,
)

_ASYNC_TEST_TIMEOUT_S = 10.0
_T = TypeVar("_T")


def _with_mutation_cut(callback: Callable[[DataPlaneMutationCut], _T]) -> _T:
    async def apply() -> _T:
        async with DataPlaneCheckpointBarrier().mutation() as cut:
            return callback(cut)

    return asyncio.run(apply())


def _init_recovery_telemetry(controller: Any, *, train_steps: int = 0) -> None:
    """Initialize constructor-owned telemetry state for hand-built controllers."""
    controller._train_steps = train_steps
    controller._logger = MagicMock()


async def _wait_for_event_or_pump(
    event: asyncio.Event,
    pump: asyncio.Task[None],
) -> None:
    """Wait for a test hook while surfacing an early rollout-pump failure."""
    event_waiter = asyncio.create_task(event.wait())
    try:
        done, _ = await asyncio.wait(
            {event_waiter, pump},
            timeout=_ASYNC_TEST_TIMEOUT_S,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            raise TimeoutError("rollout pump did not reach the expected test hook")
        if pump in done:
            await pump
            raise AssertionError("rollout pump completed before the expected test hook")
        await event_waiter
    finally:
        if not event_waiter.done():
            event_waiter.cancel()
            await asyncio.gather(event_waiter, return_exceptions=True)


class _CountingInOrderSampler(InOrderSampler):
    """Real in-order sampler with observable admission calls."""

    def __init__(self) -> None:
        super().__init__(None, max_lookahead_versions=1)
        self.admit_calls = 0
        self.admission_commits = 0

    async def admit(self, *, trainer_version_fn):
        self.admit_calls += 1
        return await super().admit(trainer_version_fn=trainer_version_fn)

    def commit_admission(self, cut: DataPlaneMutationCut):
        self.admission_commits += 1
        return super().commit_admission(cut)


class _BlockingBeforeAdmissionSampler(_CountingInOrderSampler):
    """Pause after the dataloader advances but before admission mutates state."""

    def __init__(self) -> None:
        super().__init__()
        self.admission_entered = asyncio.Event()
        self.release_admission = asyncio.Event()

    async def wait_until_admissible(self, *, trainer_version_fn):
        self.admission_entered.set()
        await self.release_admission.wait()
        await super().wait_until_admissible(trainer_version_fn=trainer_version_fn)


@dataclass(frozen=True)
class _PendingGroup:
    group_id: str
    target_step: int | None
    prompt_payload: dict[str, Any]


class _PendingLedger:
    """Small stand-in for the group-level recovery ledger contract."""

    def __init__(self, group: _PendingGroup | None = None) -> None:
        self._groups = [group] if group is not None else []
        self.prepare_calls = 0

    def prepare_for_restart(self) -> None:
        self.prepare_calls += 1

    def groups(self) -> list[_PendingGroup]:
        return list(self._groups)

    def expected_staging_keys(self) -> set[str]:
        return set()

    def record(self, group: _PendingGroup) -> None:
        self._groups.append(group)

    def assign_target_step(self, group_id: str, target_step: int) -> None:
        self._groups = [
            _PendingGroup(
                group_id=group.group_id,
                target_step=target_step,
                prompt_payload=group.prompt_payload,
            )
            if group.group_id == group_id
            else group
            for group in self._groups
        ]

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
            "groups": [
                {
                    "group_id": group.group_id,
                    "admission_id": group.group_id,
                    "prompt_id": str(group.prompt_payload.get("idx", "unknown")),
                    "target_step": group.target_step,
                    "prompt_ref": {
                        "sample_id": str(group.prompt_payload.get("idx", "unknown")),
                        "task_name": group.prompt_payload.get("task_name"),
                    },
                    "expected_generations": 2,
                    "start_weight_version": 7,
                    "phase": ("reserved" if group.target_step is None else "admitted"),
                }
                for group in self._groups
            ],
        }

    def release(self, group_id: str) -> None:
        self._groups = [group for group in self._groups if group.group_id != group_id]


def _sidecar_state(ledger: RolloutRecoveryLedger) -> dict[str, Any]:
    """Ledger state plus the controller-owned keys every sidecar must carry."""
    return {
        **ledger.state_dict(),
        "reward_settings": None,
        "finalizer_metrics_by_group": {},
        "finalizer_rewards_by_group": {},
    }


class _RecoveryRolloutManager:
    def __init__(self, ledger: RolloutRecoveryLedger) -> None:
        self.recovery_ledger = ledger
        self.recovered: list[tuple[str, int | None]] = []

    async def complete_recovery(
        self,
        cut: DataPlaneMutationCut,
        group_id: str,
    ) -> None:
        group = self.recovery_ledger.get_group(group_id)
        self.recovered.append((group.group_id, group.target_step))
        self.recovery_ledger.discard_group(cut, group_id)

    def mark_prompt_group_admitted(
        self,
        cut: DataPlaneMutationCut,
        group_id: str,
        *,
        target_step: int | None,
    ) -> None:
        self.recovery_ledger.mark_group_admitted(
            cut,
            group_id,
            target_step=target_step,
            start_weight_version=7,
        )

    def discard_prompt_group(
        self,
        cut: DataPlaneMutationCut,
        group_id: str,
    ) -> None:
        self.recovery_ledger.discard_group(cut, group_id)

    def record_recovery_siblings(self, *, reused: int, redispatched: int) -> None:
        del reused, redispatched


def _deadline_pausing_manager(
    registry: Optional[RequestDeadlineRegistry] = None,
) -> SimpleNamespace:
    """Rollout-manager stand-in for the deadline pause a Gym checkpoint takes."""
    registry = registry if registry is not None else RequestDeadlineRegistry()
    return SimpleNamespace(
        suspend_request_deadlines=registry.suspend,
        resume_request_deadlines=registry.resume,
    )


async def _no_staging_keys() -> list[str]:
    return []


def _controller_for(
    coordinator: GymCheckpointCoordinator,
    ledger: Optional[RolloutRecoveryLedger] = None,
    *,
    manager: Optional[Any] = None,
    list_staging_keys: Callable[[], Awaitable[list[str]]] = _no_staging_keys,
    allow_restart_only_servers: bool = False,
) -> Any:
    """A bare controller wired for Gym checkpoint coordination, and nothing else."""
    controller = object.__new__(SingleControllerActor.__ray_metadata__.modified_class)
    controller._rollout_manager = (
        manager if manager is not None else _deadline_pausing_manager()
    )
    controller._rollout_recovery_ledger = (
        ledger if ledger is not None else RolloutRecoveryLedger()
    )
    controller._environment_checkpoint = GymEnvironmentCheckpoint(
        coordinator,
        ledger=controller._rollout_recovery_ledger,
        rollout_manager=controller._rollout_manager,
        list_staging_keys=list_staging_keys,
        allow_restart_only_servers=allow_restart_only_servers,
    )
    return controller


def _environment_phase_checkpointing(
    environment_checkpoint: Any,
) -> DataPlaneRolloutCheckpointing:
    """Rollout checkpointing with only what the environment phase and commit use."""
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
        environment_checkpoint=environment_checkpoint,
        bootstrap_identity=None,
        teacher_coordinator=None,
        restorer=None,  # type: ignore[arg-type]
        dispatch=None,  # type: ignore[arg-type]
    )


def _gym_environment_for(
    controller: Any, coordinator: GymCheckpointCoordinator
) -> GymEnvironmentCheckpoint:
    """The Gym checkpoint a turn-level controller builds, over ``coordinator``."""
    return GymEnvironmentCheckpoint(
        coordinator,
        ledger=controller._rollout_manager.recovery_ledger,
        rollout_manager=controller._rollout_manager,
        list_staging_keys=_no_staging_keys,
    )


@contextlib.asynccontextmanager
async def _gym_commit(
    controller: Any, checkpoint_path: Path, *, checkpoint_id: str
) -> AsyncIterator[GymCheckpointCommitResult]:
    """Gym's commit for one snapshot, while Gym is parked."""
    async with controller._environment_checkpoint.snapshot(
        checkpoint_path, checkpoint_id=checkpoint_id
    ) as snapshot:
        yield snapshot.commit


class _BlockingRolloutManager:
    """Hold one admitted rollout unfinished while the checkpoint is written."""

    def __init__(self, ledger: _PendingLedger) -> None:
        self.recovery_ledger = ledger
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.weight_version = 0

    def set_weight_version(self, version: int) -> None:
        self.weight_version = version

    def set_data_plane_checkpoint_barrier(
        self, barrier: DataPlaneCheckpointBarrier
    ) -> None:
        """Accept the controller-owned barrier used by the production manager."""
        del barrier

    def reserve_prompt_group(
        self,
        cut: DataPlaneMutationCut | None,
        prompt: DatumSpec,
        *,
        target_step: int | None = None,
        admitted: bool = True,
        admission_id: str | None = None,
    ) -> str:
        del admitted, admission_id
        batch_label = "fetched" if target_step is None else str(target_step)
        group_id = f"batch-{batch_label}-prompt-{prompt['idx']}"
        if not self.recovery_ledger.groups():
            self.recovery_ledger.record(
                _PendingGroup(
                    group_id=group_id,
                    target_step=target_step,
                    prompt_payload=dict(prompt),
                )
            )
        return group_id

    def mark_prompt_group_admitted(
        self,
        cut: DataPlaneMutationCut,
        group_id: str,
        *,
        target_step: int | None,
    ) -> None:
        del cut
        if target_step is None:
            return
        self.recovery_ledger.assign_target_step(group_id, target_step)

    def discard_prompt_group(
        self,
        cut: DataPlaneMutationCut,
        group_id: str,
    ) -> None:
        del cut
        self.recovery_ledger.release(group_id)

    async def generate_and_push(
        self,
        prompt: DatumSpec,
        *,
        target_step: int | None = None,
        inflight_registry: dict[str, Any] | None = None,
        lineage_group_id: str | None = None,
    ) -> RolloutOutcome:
        del inflight_registry
        if lineage_group_id is None:
            lineage_group_id = self.reserve_prompt_group(
                None,
                prompt,
                target_step=target_step,
            )
        self.started.set()
        await self.release.wait()
        return RolloutOutcome.COMMITTED


class _BlockingNoOpDataPlaneClient(NoOpDataPlaneClient):
    """Hold the native data-plane save while a commit tries to publish."""

    def __init__(self) -> None:
        super().__init__()
        self.save_started = threading.Event()
        self.release_save = threading.Event()

    def save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.save_started.set()
        assert self.release_save.wait(timeout=30.0), "test never released TQ save"
        super().save_checkpoint(checkpoint_dir, metadata=metadata)


class _LedgerFacade:
    """Minimal RolloutManager ownership surface for reserve-pool cuts."""

    def __init__(self) -> None:
        self.recovery_ledger = RolloutRecoveryLedger()

    def reserve_prompt_group(
        self,
        cut: DataPlaneMutationCut,
        prompt: DatumSpec,
        *,
        target_step: int | None = None,
        admitted: bool = True,
        admission_id: str | None = None,
    ) -> str:
        record = self.recovery_ledger.reserve_group(
            cut,
            prompt_id=str(prompt["idx"]),
            prompt_payload=prompt,
            expected_generations=2,
            target_step=target_step,
            start_weight_version=7,
            task_source=None,
            restore_level=RecoveryTargetLevel.SIBLING,
            admitted=admitted,
            admission_id=admission_id,
        )
        return record.group_id

    def mark_prompt_group_admitted(
        self,
        cut: DataPlaneMutationCut,
        group_id: str,
        *,
        target_step: int | None,
    ) -> None:
        self.recovery_ledger.mark_group_admitted(
            cut,
            group_id,
            target_step=target_step,
            start_weight_version=7,
        )

    def discard_prompt_group(
        self,
        cut: DataPlaneMutationCut,
        group_id: str,
    ) -> None:
        self.recovery_ledger.discard_group(cut, group_id)


def _reserve_prompt(idx: int) -> DatumSpec:
    return {
        "idx": idx,
        "message_log": [{"role": "user", "content": f"prompt {idx}"}],
        "length": 1,
        "extra_env_info": None,
        "loss_multiplier": 1.0,
    }


def _identity_dict_collator(batch: list[DatumSpec]) -> DatumSpec:
    """Return one directly usable prompt rather than a BatchedDataDict."""
    assert len(batch) == 1
    prompt = dict(batch[0])
    prompt["length"] = 99
    return cast(DatumSpec, prompt)


def _two_row_collator(_batch: list[DatumSpec]) -> BatchedDataDict:
    """Return an invalid two-row recovery batch."""
    return BatchedDataDict({"idx": [7, 8]})


def _non_mapping_collator(_batch: list[DatumSpec]) -> list[str]:
    """Return an invalid collator result type."""
    return ["not-a-prompt"]


def _rehydration_controller(
    collate_fn: Callable[[list[DatumSpec]], Any],
) -> tuple[Any, RolloutRecoveryLedger]:
    """Build a restored ledger whose prompt must be resolved from the dataset."""
    dataset_prompt = _reserve_prompt(7)
    saved_ledger = RolloutRecoveryLedger()
    _with_mutation_cut(
        lambda cut: saved_ledger.reserve_group(
            cut,
            group_id="rehydrate-7",
            prompt_id="7",
            prompt_payload=dataset_prompt,
            expected_generations=2,
            target_step=7,
            start_weight_version=7,
            admitted=True,
        )
    )
    restored_ledger = RolloutRecoveryLedger()
    _with_mutation_cut(
        lambda cut: restored_ledger.load_state_dict(cut, saved_ledger.state_dict())
    )

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    controller = object.__new__(controller_cls)
    controller._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
    controller._rollout_manager = SimpleNamespace(recovery_ledger=restored_ledger)
    controller._dataloader = SimpleNamespace(
        dataset={7: dataset_prompt},
        collate_fn=collate_fn,
    )
    return controller, restored_ledger


def _run_rehydration(controller: Any) -> None:
    async def rehydrate() -> None:
        async with controller._data_plane_checkpoint_barrier.mutation() as cut:
            await restorer_for(controller)._rehydrate_rollout_recovery_prompts(cut)

    asyncio.run(rehydrate())


def test_recovery_rehydration_accepts_an_identity_dict_collator() -> None:
    controller, ledger = _rehydration_controller(_identity_dict_collator)

    _run_rehydration(controller)

    assert ledger.get_group("rehydrate-7").prompt_payload["length"] == 99


@pytest.mark.parametrize(
    ("collate_fn", "expected_error", "match"),
    [
        pytest.param(
            _two_row_collator,
            ValueError,
            "must return exactly one prompt",
            id="multiple-prompts",
        ),
        pytest.param(
            _non_mapping_collator,
            TypeError,
            "expected a mapping",
            id="non-mapping",
        ),
    ],
)
def test_recovery_rehydration_rejects_invalid_collator_results(
    collate_fn: Callable[[list[DatumSpec]], Any],
    expected_error: type[Exception],
    match: str,
) -> None:
    controller, _ = _rehydration_controller(collate_fn)

    with pytest.raises(expected_error, match=match):
        _run_rehydration(controller)


def _reserve_controller() -> Any:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    controller = object.__new__(controller_cls)
    controller._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
    controller._replacement_reserve = deque()
    controller._sampler_stamps_target_steps = True
    controller._async_cfg = SimpleNamespace(
        rollout_failure=SimpleNamespace(
            on_dropped_prompt="replace",
            replacement_reserve_prompts=2,
            max_replacement_attempts=1,
        )
    )
    controller._algo_cfg = SimpleNamespace(num_prompts_per_step=2)
    controller._rollout_manager = _LedgerFacade()
    use_dispatch(controller, records_prompt_groups=True)
    return controller


def test_dispatch_cursor_alone_assigns_the_next_batch_to_step_8() -> None:
    """The exact cursor is correct; it cannot recreate the missing step-7 batch."""

    async def exercise() -> int | None:
        sampler = _CountingInOrderSampler()
        sampler.restore_dispatch_index(7)
        return await sampler.admit(trainer_version_fn=lambda: 7)

    assert asyncio.run(exercise()) == 8


def test_checkpoint_after_fetch_before_admit_owns_the_prompt(tmp_path) -> None:
    """A checkpoint cut inside admit retains the fetched batch for recovery."""

    async def exercise() -> None:
        save_state = _initial_grpo_save_state()
        save_state.current_step = 7
        save_state.total_steps = 7
        save_state.trainer_version = 7
        save_state.sampler_dispatch_index = 6

        ledger = _PendingLedger()
        rollout_manager = _BlockingRolloutManager(ledger)
        dataloader = _FakeDataloader(
            [
                BatchedDataDict(
                    {
                        "idx": [70],
                        "message_log": [[{"role": "user", "content": "batch 7"}]],
                    }
                )
            ],
            state={"next_batch": 8},
        )
        config = _actor_master_config(
            tmp_path,
            max_num_steps=8,
            num_prompts_per_step=1,
            max_num_epochs=1,
            data_plane_checkpoint=True,
        )
        actor_args = _make_actor_args(
            save_state=save_state,
            dataloader=dataloader,
        )
        actor_args.rollout_manager = rollout_manager  # type: ignore[assignment]

        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        controller = controller_cls(config, actor_args, SetupTimingMetrics())
        sampler = _BlockingBeforeAdmissionSampler()
        sampler.restore_dispatch_index(6)
        controller._sampler = sampler
        pump = asyncio.create_task(controller._rollout_pump())
        await _wait_for_event_or_pump(sampler.admission_entered, pump)
        assert controller._sampler.dispatch_index == 6

        try:
            await controller._save_checkpoint(
                {"loss": 1.0},
                is_policy_training_step=True,
                is_final_checkpoint=False,
            )
        finally:
            sampler.release_admission.set()
            await _wait_for_event_or_pump(rollout_manager.started, pump)
            rollout_manager.release.set()
            await asyncio.wait_for(pump, timeout=_ASYNC_TEST_TIMEOUT_S)
            controller._checkpointer.shutdown()

        checkpoint = tmp_path / "checkpoints" / "step_7"
        recovery_state = torch.load(
            checkpoint / "rollout_recovery.pt",
            weights_only=False,
        )
        assert len(recovery_state["groups"]) == 1
        assert recovery_state["groups"][0]["target_step"] is None
        assert recovery_state["groups"][0]["prompt_ref"]["sample_id"] == "70"
        assert "prompt_payload" not in recovery_state["groups"][0]
        assert torch.load(
            checkpoint / "train_dataloader.pt",
            weights_only=False,
        ) == {"next_batch": 8}

    asyncio.run(exercise())


def test_turn_trainer_checkpoint_leaves_gym_state_to_the_rollout_snapshot(
    tmp_path: Path,
) -> None:
    """Gym never holds up a trainer save; the next rollout snapshot captures it.

    The trainer checkpoint says it holds no Gym state, does not count as a
    rollout snapshot, and wakes the snapshot pump so the gap stays short.
    """

    gym_actor = FakeGymActor()

    async def exercise() -> Any:
        save_state = _initial_grpo_save_state()
        save_state.current_step = 7
        save_state.total_steps = 7
        save_state.trainer_version = 7
        save_state.sampler_dispatch_index = 7
        config = _actor_master_config(
            tmp_path,
            max_num_steps=8,
            num_prompts_per_step=1,
            max_num_epochs=1,
            data_plane_checkpoint=True,
        )
        actor_args = _make_actor_args(
            save_state=save_state,
            dataloader=_FakeDataloader([], state={"next_batch": 8}),
        )
        actor_args.rollout_manager = _BlockingRolloutManager(  # type: ignore[assignment]
            _PendingLedger()
        )
        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        controller = controller_cls(config, actor_args, SetupTimingMetrics())
        controller._environment_checkpoint = _gym_environment_for(
            controller, gym_coordinator(gym_actor)
        )
        controller._rollout_checkpointing = controller._build_rollout_checkpointing(
            CheckpointToRestore()
        )
        try:
            await controller._save_checkpoint(
                {"loss": 1.0},
                is_policy_training_step=True,
                is_final_checkpoint=False,
            )
            await asyncio.to_thread(controller._checkpointer.finalize_pending)
        finally:
            controller._checkpointer.shutdown()
        return controller

    controller = asyncio.run(exercise())

    checkpoint = tmp_path / "checkpoints" / "step_7"
    # A trainer checkpoint never holds Gym state, and never calls Gym.
    assert gym_actor.calls == []
    assert not list(checkpoint.glob("gym*"))
    assert controller._rollout_checkpointing._rollout_checkpoint_wakeup.is_set()
    assert (
        controller._rollout_checkpointing._last_rollout_snapshot_mutation_version
        is None
    )


def test_turn_checkpoint_drains_on_wire_reply_before_tq_cut(tmp_path: Path) -> None:
    async def exercise() -> None:
        ledger = RolloutRecoveryLedger()
        barrier = DataPlaneCheckpointBarrier()
        async with barrier.mutation() as cut:
            group = _dispatched_turn_group(ledger, cut, generations=1)

        actor = FakeGymActor(exported_episodes=())
        controller = _controller_for(gym_coordinator(actor), ledger)

        async def complete_on_wire_reply() -> None:
            while "commit" not in actor.operations():
                await asyncio.sleep(0)
            async with barrier.mutation("sibling_seals") as cut:
                ledger.mark_sibling_sealed(
                    cut,
                    group.group_id,
                    generation_index=0,
                    gate_rollout_id=group.gate_rollout_id(0),
                    receipt={
                        "rollout_id": group.gate_rollout_id(0),
                        "manifest": [{"staging_key": "on-wire/call-0"}],
                    },
                    reward=1.0,
                    mask_sample=False,
                )

        completion = asyncio.create_task(complete_on_wire_reply())
        async with _gym_commit(
            controller,
            tmp_path,
            checkpoint_id="save-1",
        ) as gym_commit:
            assert gym_commit is not None
            assert gym_commit.manifest.instances == {"tools/replica-0": ()}
            async with barrier.checkpoint() as cut:
                await controller._environment_checkpoint.validate_cut(
                    cut, gym_commit, checkpoint_path=tmp_path
                )
                assert ledger.expected_staging_keys() == {"on-wire/call-0"}
                actor.mark("tq-cut")
        await asyncio.wait_for(completion, _ASYNC_TEST_TIMEOUT_S)

        assert actor.operations() == ["prepare", "renew", "commit", "tq-cut", "resume"]
        (commit_call,) = [call for call in actor.calls if call[0] == "commit"]
        assert commit_call[1] == (
            "save-1",
            str(tmp_path),
            (group.environment_episode(0),),
        )

    asyncio.run(exercise())


@pytest.mark.parametrize("step_commits", [True, False])
def test_gym_phase_stops_when_a_train_step_starts_committing(
    tmp_path: Path, step_commits: bool
) -> None:
    """Stopping mid-commit resumes Gym and reopens dispatch; otherwise it enters."""

    async def exercise() -> None:
        actor = FakeGymActor(exported_episodes=(), block_commit=True)
        controller = _controller_for(gym_coordinator(actor))
        rollout_checkpointing = _environment_phase_checkpointing(
            controller._environment_checkpoint
        )
        async with contextlib.AsyncExitStack() as stack:
            phase = asyncio.create_task(
                rollout_checkpointing._enter_environment_phase_unless_a_step_commits(
                    stack,
                    _gym_commit(controller, tmp_path, checkpoint_id="save-1"),
                )
            )
            await asyncio.wait_for(
                actor.commit_started.wait(), timeout=_ASYNC_TEST_TIMEOUT_S
            )
            if step_commits:
                rollout_checkpointing._begin_optimizer_commit()
            else:
                actor.release_commit.set()
            entered, gym_commit = await asyncio.wait_for(
                phase, timeout=_ASYNC_TEST_TIMEOUT_S
            )
            assert entered is not step_commits
            assert (gym_commit is None) is step_commits
            if step_commits:
                # Gym resumed and dispatch reopened before anything else ran.
                assert actor.operations()[-1] == "resume"
                assert not actor.checkpoint_open
                async with asyncio.timeout(_ASYNC_TEST_TIMEOUT_S):
                    async with (
                        controller._environment_checkpoint.admission_gate.mutation(
                            "gym_dispatch"
                        )
                    ):
                        pass
            else:
                assert actor.checkpoint_open
        assert not actor.checkpoint_open
        assert actor.operations().count("resume") == 1

    asyncio.run(exercise())


def test_completion_callback_can_run_while_gym_commit_is_pending(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        ledger = RolloutRecoveryLedger()
        barrier = DataPlaneCheckpointBarrier()
        async with barrier.mutation() as cut:
            group = _dispatched_turn_group(ledger, cut, generations=1)

        actor = FakeGymActor(exported_episodes=(), block_commit=True)
        controller = _controller_for(gym_coordinator(actor), ledger)

        async def checkpoint() -> GymCheckpointCommitResult:
            async with _gym_commit(
                controller,
                tmp_path,
                checkpoint_id="save-1",
            ) as gym_commit:
                assert gym_commit is not None
                return gym_commit

        checkpoint_task = asyncio.create_task(checkpoint())
        await asyncio.wait_for(
            actor.commit_started.wait(),
            timeout=_ASYNC_TEST_TIMEOUT_S,
        )

        async def complete_rollout() -> None:
            async with barrier.mutation("sibling_seals") as cut:
                ledger.mark_sibling_sealed(
                    cut,
                    group.group_id,
                    generation_index=0,
                    gate_rollout_id=group.gate_rollout_id(0),
                    receipt={
                        "rollout_id": group.gate_rollout_id(0),
                        "manifest": [{"staging_key": "on-wire/call-0"}],
                    },
                    reward=1.0,
                    mask_sample=False,
                )

        await asyncio.wait_for(
            complete_rollout(),
            timeout=_ASYNC_TEST_TIMEOUT_S,
        )

        actor.release_commit.set()
        commit = await asyncio.wait_for(
            checkpoint_task,
            timeout=_ASYNC_TEST_TIMEOUT_S,
        )

        assert commit.manifest.instances == {"tools/replica-0": ()}
        assert ledger.expected_staging_keys() == {"on-wire/call-0"}
        assert actor.operations()[-1] == "resume"

    asyncio.run(exercise())


def test_rollout_dispatch_gate_blocks_only_new_rpc_submissions() -> None:
    async def exercise() -> None:
        gate = DataPlaneCheckpointBarrier()
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        checkpoint_closed = asyncio.Event()
        release_checkpoint = asyncio.Event()
        second_entered = asyncio.Event()

        async def first_submission() -> None:
            async with gate.mutation("gym_dispatch"):
                first_entered.set()
                await release_first.wait()

        async def checkpoint() -> None:
            async with gate.checkpoint():
                checkpoint_closed.set()
                await release_checkpoint.wait()

        async def second_submission() -> None:
            async with gate.mutation("gym_dispatch"):
                second_entered.set()

        first = asyncio.create_task(first_submission())
        await first_entered.wait()
        closing = asyncio.create_task(checkpoint())
        await asyncio.sleep(0)
        assert not checkpoint_closed.is_set()

        release_first.set()
        await checkpoint_closed.wait()
        second = asyncio.create_task(second_submission())
        await asyncio.sleep(0)
        assert not second_entered.is_set()

        release_checkpoint.set()
        await asyncio.gather(first, closing, second)
        assert second_entered.is_set()

    asyncio.run(exercise())


def test_turn_checkpoint_aborts_when_candidate_is_neither_exported_nor_drained(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        ledger = RolloutRecoveryLedger()
        async with DataPlaneCheckpointBarrier().mutation() as cut:
            group = _dispatched_turn_group(ledger, cut, generations=1)

        actor = FakeGymActor(exported_episodes=())
        controller = _controller_for(
            gym_coordinator(actor, control_timeout_s=0.01), ledger
        )

        with pytest.raises(TimeoutError, match="neither|classify every candidate"):
            async with _gym_commit(
                controller,
                tmp_path,
                checkpoint_id="save-1",
            ):
                raise AssertionError("unreachable")

        assert actor.operations()[-1] == "resume"

    asyncio.run(exercise())


def test_turn_checkpoint_fails_fast_on_parked_session_without_an_owner(
    tmp_path: Path,
) -> None:
    """A session Gym parks but cannot export never drains; do not wait for it."""

    async def exercise() -> None:
        ledger = RolloutRecoveryLedger()
        async with DataPlaneCheckpointBarrier().mutation() as cut:
            group = _dispatched_turn_group(ledger, cut, generations=1)

        (stranded,) = ledger.environment_inventory(frozenset({"tools/replica-0"}))[
            "tools/replica-0"
        ]
        actor = FakeGymActor(exported_episodes=(), unowned_live_episodes=(stranded,))
        controller = _controller_for(
            gym_coordinator(actor, control_timeout_s=60.0), ledger
        )

        async def checkpoint() -> None:
            async with _gym_commit(
                controller,
                tmp_path,
                checkpoint_id="save-1",
            ):
                raise AssertionError("unreachable")

        with pytest.raises(RuntimeError, match="parked agent sessions"):
            await asyncio.wait_for(checkpoint(), timeout=5.0)

        assert actor.operations()[-1] == "resume"

    asyncio.run(exercise())


def _dispatched_turn_group(
    ledger: RolloutRecoveryLedger,
    cut: DataPlaneMutationCut,
    *,
    generations: int = 2,
) -> PromptGroupRecoveryRecord:
    """An admitted turn-level group, dispatched to ``tools/replica-0``."""
    group = ledger.reserve_group(
        cut,
        group_id="batch-7-prompt-0",
        admission_id="batch-7",
        prompt_id="70",
        prompt_payload={"idx": 70, "message_log": []},
        expected_generations=generations,
        target_step=7,
        start_weight_version=7,
        restore_level=RecoveryTargetLevel.TURN,
        admitted=True,
    )
    ledger.mark_group_dispatched(
        cut, group.group_id, environment_instance_id="tools/replica-0"
    )
    return ledger.get_group(group.group_id)


def _gym_episodes(group: PromptGroupRecoveryRecord) -> set[GymCheckpointEpisode]:
    return {
        group.environment_episode(sibling.generation_index)
        for sibling in group.siblings
    }


def test_turn_checkpoint_retires_abandoned_attempts_gym_is_still_running(
    tmp_path: Path,
) -> None:
    """A cancelled group's /run keeps running in Gym; RL must retire it.

    Abandoning drops the attempts from RL's inventory, but nothing stops the
    episodes Gym already admitted. Unretired, Gym exports them at the next
    commit and every checkpoint fails until they finish on their own.
    """

    async def exercise() -> None:
        barrier = DataPlaneCheckpointBarrier()
        ledger = RolloutRecoveryLedger()
        async with barrier.mutation() as cut:
            group = _dispatched_turn_group(ledger, cut)
        abandoned = _gym_episodes(group)
        # The group task was cancelled after Ray accepted its submission.
        async with barrier.mutation() as cut:
            ledger.abandon_unsealed(cut, group.group_id)

        actor = FakeGymActor(live=abandoned)
        controller = _controller_for(gym_coordinator(actor), ledger)

        async with _gym_commit(
            controller,
            tmp_path,
            checkpoint_id="save-1",
        ) as gym_commit:
            assert gym_commit is not None
            assert gym_commit.manifest.instances == {"tools/replica-0": ()}

        # Retired, then forgotten, while Gym is idle, before prepare parks anything.
        assert actor.operations() == [
            "retire",
            "forget",
            "prepare",
            "renew",
            "commit",
            "resume",
        ]
        assert frozenset(actor.calls[0][1][1]) == frozenset(abandoned)
        assert actor.calls[1][1][1] == tuple(
            sorted(episode.rollout_id for episode in abandoned)
        )
        assert actor.live == set()
        assert ledger.environment_forgets({"tools/replica-0"}) == {
            "tools/replica-0": ()
        }

    asyncio.run(exercise())


def test_turn_checkpoint_retires_each_dropped_attempt_only_once(
    tmp_path: Path,
) -> None:
    """A retired attempt leaves the queue, so the next checkpoint skips it."""

    async def exercise() -> None:
        barrier = DataPlaneCheckpointBarrier()
        ledger = RolloutRecoveryLedger()
        async with barrier.mutation() as cut:
            group = _dispatched_turn_group(ledger, cut)
        async with barrier.mutation() as cut:
            ledger.abandon_unsealed(cut, group.group_id)

        actor = FakeGymActor(live=_gym_episodes(group))
        controller = _controller_for(gym_coordinator(actor), ledger)
        for checkpoint_id in ("save-1", "save-2"):
            (tmp_path / checkpoint_id).mkdir()
            async with _gym_commit(
                controller,
                tmp_path / checkpoint_id,
                checkpoint_id=checkpoint_id,
            ):
                pass

        assert ledger.environment_retirements({"tools/replica-0"}) == {
            "tools/replica-0": ()
        }
        retires = [call for call in actor.calls if call[0] == "retire"]
        assert [call[1][0] for call in retires] == ["save-1"]

    asyncio.run(exercise())


@pytest.mark.parametrize("forget_failures", [1, 2])
def test_turn_checkpoint_retires_again_and_forgets_a_rollout_gym_refused_to_forget(
    tmp_path: Path, forget_failures: int
) -> None:
    """A refused forget gets the retire again, then the forget, as Gym asks.

    A deployment that still refuses keeps the rollouts queued for the next
    snapshot: a mark costs one entry, so the snapshot itself goes ahead.
    """

    async def exercise() -> None:
        barrier = DataPlaneCheckpointBarrier()
        ledger = RolloutRecoveryLedger()
        async with barrier.mutation() as cut:
            group = _dispatched_turn_group(ledger, cut)
        abandoned = _gym_episodes(group)
        async with barrier.mutation() as cut:
            ledger.abandon_unsealed(cut, group.group_id)

        actor = FakeGymActor(
            live=abandoned,
            forget_errors=(RuntimeError("retire_incomplete"),) * forget_failures,
        )
        controller = _controller_for(gym_coordinator(actor), ledger)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            async with _gym_commit(
                controller, tmp_path, checkpoint_id="save-1"
            ) as gym_commit:
                assert gym_commit is not None

        operations = actor.operations()
        assert operations[:4] == ["retire", "forget", "retire", "forget"]
        assert operations[4] == "prepare"
        assert frozenset(actor.calls[2][1][1]) == frozenset(abandoned)
        pending = ledger.environment_forgets({"tools/replica-0"})
        forgot_warnings = [w for w in caught if "could not forget" in str(w.message)]
        if forget_failures == 1:
            assert pending == {"tools/replica-0": ()}
            assert not forgot_warnings
        else:
            assert set(pending["tools/replica-0"]) == abandoned
            assert forgot_warnings

    asyncio.run(exercise())


def test_turn_checkpoint_retires_only_the_abandoned_attempts_of_a_retried_group(
    tmp_path: Path,
) -> None:
    """The retry's fresh attempts stay live and are committed normally."""

    async def exercise() -> None:
        barrier = DataPlaneCheckpointBarrier()
        ledger = RolloutRecoveryLedger()
        async with barrier.mutation() as cut:
            group = _dispatched_turn_group(ledger, cut)
        abandoned = _gym_episodes(group)
        async with barrier.mutation() as cut:
            ledger.abandon_unsealed(cut, group.group_id)
            ledger.prepare_incomplete_retry(cut, group.group_id)
            ledger.mark_group_dispatched(
                cut, group.group_id, environment_instance_id="tools/replica-0"
            )
        replacements = _gym_episodes(ledger.get_group(group.group_id))
        assert replacements.isdisjoint(abandoned)

        actor = FakeGymActor(live=abandoned | replacements)
        controller = _controller_for(gym_coordinator(actor), ledger)

        async with _gym_commit(
            controller,
            tmp_path,
            checkpoint_id="save-1",
        ) as gym_commit:
            assert gym_commit is not None
            assert set(gym_commit.manifest.instances["tools/replica-0"]) == (
                replacements
            )

        assert actor.calls[0][0] == "retire"
        assert frozenset(actor.calls[0][1][1]) == frozenset(abandoned)
        assert actor.live == replacements

    asyncio.run(exercise())


def test_turn_checkpoint_leaves_gym_restarts_out_and_records_them(
    tmp_path: Path,
) -> None:
    """Gym refuses a commit that names a restart; the snapshot records it instead.

    The restart is still running in Gym, so the drain must not wait for it and
    the cut must accept RL still owning it.
    """

    async def exercise() -> None:
        barrier = DataPlaneCheckpointBarrier()
        ledger = RolloutRecoveryLedger()
        async with barrier.mutation() as cut:
            group = _dispatched_turn_group(ledger, cut)
        saved, restart = sorted(_gym_episodes(group))
        actor = FakeGymActor(live={saved, restart}, restarts=(restart,))
        controller = _controller_for(gym_coordinator(actor), ledger)

        async with _gym_commit(
            controller,
            tmp_path,
            checkpoint_id="save-1",
        ) as gym_commit:
            assert gym_commit is not None
            async with barrier.checkpoint() as cut:
                await controller._environment_checkpoint.validate_cut(
                    cut, gym_commit, checkpoint_path=tmp_path
                )

        (commit,) = [call for call in actor.calls if call[0] == "commit"]
        assert commit[1][2] == (saved,)
        manifest = load_gym_checkpoint_manifest(tmp_path)
        assert manifest.instances == {"tools/replica-0": (saved,)}
        assert manifest.restarts == {"tools/replica-0": (restart,)}
        assert actor.operations()[-1] == "resume"

    asyncio.run(exercise())


def test_turn_checkpoint_rejects_missing_gym_staging_key(tmp_path: Path) -> None:
    async def exercise() -> None:
        ledger = RolloutRecoveryLedger()
        barrier = DataPlaneCheckpointBarrier()
        async with barrier.mutation() as cut:
            group = _dispatched_turn_group(ledger, cut, generations=1)

        episode = group.environment_episode(0)
        actor = FakeGymActor(staging_keys_by_episode={episode: ("missing-call-row",)})
        # TQ holds no staging rows, so the one Gym references is missing.
        controller = _controller_for(gym_coordinator(actor), ledger)

        with pytest.raises(RuntimeError, match="staging rows missing"):
            async with _gym_commit(
                controller,
                tmp_path,
                checkpoint_id="save-1",
            ) as gym_commit:
                assert gym_commit is not None
                async with barrier.checkpoint() as cut:
                    await controller._environment_checkpoint.validate_cut(
                        cut, gym_commit, checkpoint_path=tmp_path
                    )

        assert actor.operations()[-1] == "resume"

    asyncio.run(exercise())


def test_checkpoint_owns_batch_7_while_its_rollout_is_unfinished(tmp_path) -> None:
    """A finalized checkpoint cannot contain a cursor hole for target step 7."""

    async def exercise() -> None:
        save_state = _initial_grpo_save_state()
        save_state.current_step = 7
        save_state.total_steps = 7
        save_state.trainer_version = 7
        save_state.sampler_dispatch_index = 6

        # Start empty: generate_and_push records the batch-7 prompt only after
        # the sampler has admitted it.
        ledger = _PendingLedger()
        rollout_manager = _BlockingRolloutManager(ledger)
        dataloader = _FakeDataloader(
            [
                BatchedDataDict(
                    {
                        "idx": [70],
                        "message_log": [[{"role": "user", "content": "batch 7"}]],
                    }
                )
            ],
            state={"next_batch": 8},
        )
        config = _actor_master_config(
            tmp_path,
            max_num_steps=8,
            num_prompts_per_step=1,
            max_num_epochs=1,
            data_plane_checkpoint=True,
        )
        actor_args = _make_actor_args(
            save_state=save_state,
            dataloader=dataloader,
        )
        actor_args.rollout_manager = rollout_manager  # type: ignore[assignment]

        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        controller = controller_cls(config, actor_args, SetupTimingMetrics())

        pump = asyncio.create_task(controller._rollout_pump())
        await _wait_for_event_or_pump(rollout_manager.started, pump)
        assert controller._sampler.dispatch_index == 7

        try:
            await controller._save_checkpoint(
                {"loss": 1.0},
                is_policy_training_step=True,
                is_final_checkpoint=False,
            )
        finally:
            rollout_manager.release.set()
            await asyncio.wait_for(pump, timeout=_ASYNC_TEST_TIMEOUT_S)
            controller._checkpointer.shutdown()

        checkpoint = tmp_path / "checkpoints" / "step_7"
        recovery_path = checkpoint / "rollout_recovery.pt"
        assert recovery_path.is_file()
        recovery_state = torch.load(recovery_path, weights_only=False)
        assert [group["target_step"] for group in recovery_state["groups"]] == [7]
        assert torch.load(
            checkpoint / "train_dataloader.pt",
            weights_only=False,
        ) == {"next_batch": 8}

    asyncio.run(exercise())


def test_commit_contending_with_checkpoint_has_exactly_one_saved_owner(
    tmp_path,
    monkeypatch,
) -> None:
    """The checkpoint records the group as canonical or pending, never neither."""
    patch_converter(monkeypatch)

    async def exercise() -> None:
        dp_client = _BlockingNoOpDataPlaneClient()
        dp_client.register_partition(
            partition_id="rollout_data",
            fields=["input_ids", "input_lengths", "total_reward"],
            num_samples=8,
            consumer_tasks=["train"],
            grpo_group_size=2,
        )
        buffer = TQReplayBuffer(
            dp_client,
            partition_id="rollout_data",
            pad_value_dict={"input_ids": 0},
            include_message_violation_fields=False,
            require_routed_experts=False,
        )
        group_id = buffer.reserve(
            weight_version=7,
            target_step=7,
            group_id="batch-7-prompt-70",
        )
        ledger = _PendingLedger(
            _PendingGroup(
                group_id=group_id,
                target_step=7,
                prompt_payload={"idx": 70, "message_log": []},
            )
        )
        rollout_manager = _BlockingRolloutManager(ledger)

        save_state = _initial_grpo_save_state()
        save_state.current_step = 7
        save_state.total_steps = 7
        save_state.trainer_version = 7
        save_state.sampler_dispatch_index = 7
        config = _actor_master_config(
            tmp_path,
            max_num_steps=8,
            num_prompts_per_step=1,
            data_plane_checkpoint=True,
        )
        actor_args = _make_actor_args(
            save_state=save_state,
            dataloader=_FakeDataloader(state={"next_batch": 8}),
            tq_buffer=buffer,  # type: ignore[arg-type]
            dp_client=dp_client,  # type: ignore[arg-type]
        )
        actor_args.rollout_manager = rollout_manager  # type: ignore[assignment]

        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        controller = controller_cls(config, actor_args, SetupTimingMetrics())
        save_task = asyncio.create_task(
            controller._save_checkpoint(
                {"loss": 1.0},
                is_policy_training_step=True,
                is_final_checkpoint=False,
            )
        )
        save_started = await asyncio.to_thread(dp_client.save_started.wait, 5.0)
        assert save_started

        commit_task = asyncio.create_task(
            controller._buffer.commit(
                group_id,
                _record(),
                start_weight_version=7,
                end_weight_version=7,
            )
        )
        await asyncio.sleep(0)
        assert not commit_task.done()
        assert controller._buffer.ready_list == [False]

        dp_client.release_save.set()
        await asyncio.wait_for(save_task, timeout=5.0)
        await asyncio.wait_for(commit_task, timeout=5.0)
        controller._checkpointer.shutdown()

        checkpoint = tmp_path / "checkpoints" / "step_7"
        replay_state = torch.load(
            checkpoint / REPLAY_BUFFER_METADATA_FILENAME,
            weights_only=False,
        )
        recovery_state = torch.load(
            checkpoint / "rollout_recovery.pt",
            weights_only=False,
        )
        canonical_ids = {group["group_id"] for group in replay_state["groups"]}
        pending_ids = {group["group_id"] for group in recovery_state["groups"]}

        assert int(group_id in canonical_ids) + int(group_id in pending_ids) == 1
        assert group_id not in canonical_ids
        assert group_id in pending_ids
        assert controller._buffer.ready_list == [True]

    asyncio.run(exercise())


def test_canonical_replay_wins_over_stale_ledger_entry(
    tmp_path,
    monkeypatch,
) -> None:
    """A completed group appears exactly once when ledger cleanup loses the cut."""
    patch_converter(monkeypatch)

    async def exercise() -> None:
        dp_client = NoOpDataPlaneClient()
        dp_client.register_partition(
            partition_id="rollout_data",
            fields=["input_ids", "input_lengths", "total_reward"],
            num_samples=8,
            consumer_tasks=["train"],
            grpo_group_size=2,
        )
        buffer = TQReplayBuffer(
            dp_client,
            partition_id="rollout_data",
            pad_value_dict={"input_ids": 0},
            include_message_violation_fields=False,
            require_routed_experts=False,
        )
        group_id = buffer.reserve(
            weight_version=7,
            target_step=7,
            group_id="batch-7-prompt-70",
        )

        # Model the narrow cut after the canonical commit but before the live
        # ledger entry is released. The checkpoint must not persist both owners.
        ledger = _PendingLedger(
            _PendingGroup(
                group_id=group_id,
                target_step=7,
                prompt_payload={"idx": 70, "message_log": []},
            )
        )
        rollout_manager = _BlockingRolloutManager(ledger)

        save_state = _initial_grpo_save_state()
        save_state.current_step = 7
        save_state.total_steps = 7
        save_state.trainer_version = 7
        save_state.sampler_dispatch_index = 7
        config = _actor_master_config(
            tmp_path,
            max_num_steps=8,
            num_prompts_per_step=1,
            data_plane_checkpoint=True,
        )
        actor_args = _make_actor_args(
            save_state=save_state,
            dataloader=_FakeDataloader(state={"next_batch": 8}),
            tq_buffer=buffer,  # type: ignore[arg-type]
            dp_client=dp_client,  # type: ignore[arg-type]
        )
        actor_args.rollout_manager = rollout_manager  # type: ignore[assignment]

        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        controller = controller_cls(config, actor_args, SetupTimingMetrics())
        await buffer.commit(
            group_id,
            _record(),
            start_weight_version=7,
            end_weight_version=7,
        )
        assert buffer.ready_list == [True]

        try:
            await controller._save_checkpoint(
                {"loss": 1.0},
                is_policy_training_step=True,
                is_final_checkpoint=False,
            )
        finally:
            controller._checkpointer.shutdown()

        checkpoint = tmp_path / "checkpoints" / "step_7"
        replay_state = torch.load(
            checkpoint / REPLAY_BUFFER_METADATA_FILENAME,
            weights_only=False,
        )
        recovery_state = torch.load(
            checkpoint / "rollout_recovery.pt",
            weights_only=False,
        )
        canonical_ids = {group["group_id"] for group in replay_state["groups"]}
        pending_ids = {group["group_id"] for group in recovery_state["groups"]}

        assert group_id in canonical_ids
        assert group_id not in pending_ids
        assert int(group_id in canonical_ids) + int(group_id in pending_ids) == 1

    asyncio.run(exercise())


def test_recovery_replays_step_7_without_readmitting_the_batch(tmp_path) -> None:
    """An admitted batch keeps target_step=7 across a process restart."""

    async def exercise() -> None:
        sampler = _CountingInOrderSampler()
        sampler.restore_dispatch_index(7)
        dataset_prompt: DatumSpec = {
            "idx": 70,
            "message_log": [],
            "length": 1,
            "extra_env_info": None,
            "loss_multiplier": 1.0,
        }
        prompt_batch = rl_collate_fn([dataset_prompt])
        dispatched_prompt = cast(
            DatumSpec,
            {key: value[0] for key, value in prompt_batch.items()},
        )
        saved_ledger = RolloutRecoveryLedger()
        async with DataPlaneCheckpointBarrier().mutation() as cut:
            saved_ledger.reserve_group(
                cut,
                group_id="batch-7-prompt-0",
                admission_id="batch-7",
                prompt_id="70",
                prompt_payload=dispatched_prompt,
                expected_generations=2,
                target_step=7,
                start_weight_version=7,
                admitted=True,
            )
        saved_state = build_rollout_recovery_state(
            saved_ledger,
            batch_shortfall={6: 1},
            sampler_stamps_target_steps=True,
        )
        recovery_path = tmp_path / ROLLOUT_RECOVERY_STATE_FILENAME
        torch.save(saved_state, recovery_path)
        payload_sha256 = hashlib.sha256(recovery_path.read_bytes()).hexdigest()

        ledger = RolloutRecoveryLedger()
        rollout_manager = _RecoveryRolloutManager(ledger)

        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        controller = object.__new__(controller_cls)
        _init_recovery_telemetry(controller, train_steps=7)
        controller._sampler = sampler
        controller._rollout_manager = rollout_manager
        controller._master_config = SimpleNamespace(
            token_capture=SimpleNamespace(enabled=False)
        )
        checkpoint = CheckpointToRestore(
            path=str(tmp_path),
            data_plane_metadata={
                "rollout_recovery_schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
                "rollout_recovery_payload_sha256": payload_sha256,
                "rollout_recovery_group_count": 1,
            },
        )
        controller._async_cfg = SimpleNamespace(
            max_buffered_rollouts=4,
            max_inflight_prompts=2,
        )
        controller._buffer_capacity = asyncio.Semaphore(4)
        controller._trainer_version = 7
        controller._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
        controller._dataloader = SimpleNamespace(
            dataset={70: dataset_prompt},
            collate_fn=rl_collate_fn,
        )
        controller._buffer = SimpleNamespace(
            count_for_target_step=lambda _target_step: 0,
            metadata_state_dict=lambda *, saved_capacity: {
                "groups": [],
                "saved_capacity": saved_capacity,
            },
        )

        async def _recover(
            _prompt: dict[str, Any],
            _target_step: int | None,
            group_id: str,
        ) -> None:
            async with controller._data_plane_checkpoint_barrier.mutation() as cut:
                await rollout_manager.complete_recovery(cut, group_id)

        restorer = restorer_for(controller, checkpoint)
        await restorer._maybe_restore_rollout_recovery(restored_replay_groups=0)
        await restorer.redispatch_restored(
            _recover, controller._admit_reserved_prompt_groups
        )

        assert rollout_manager.recovered == [("batch-7-prompt-0", 7)]
        assert sampler.admit_calls == 0
        assert sampler.admission_commits == 0
        assert sampler.dispatch_index == 7
        # The restorer reports these for the controller to take back.
        assert restorer._batch_shortfall == {6: 1}
        assert restorer._sampler_stamps_target_steps is True
        recovery_metrics = controller._logger.log_metrics.call_args.args[0]
        assert recovery_metrics["groups_unfinished_found"] == 1.0
        assert recovery_metrics["siblings_reused"] == 0.0
        assert recovery_metrics["siblings_rerun"] == 2.0
        assert recovery_metrics["siblings_continued"] == 0.0
        assert recovery_metrics["siblings_restarted"] == 2.0
        assert "groups_redispatched" not in recovery_metrics

    asyncio.run(exercise())


async def _turn_recovery_restart(
    tmp_path: Path, *, gym_actor: FakeGymActor | None = None
) -> tuple[Any, _RecoveryRolloutManager, FakeGymActor, GymCheckpointEpisode]:
    """A restarted controller whose checkpoint holds one dispatched turn rollout.

    The caller adds the Gym manifest or the restart fallback it is testing.
    """
    dataset_prompt: DatumSpec = {
        "idx": 70,
        "message_log": [],
        "length": 1,
        "extra_env_info": None,
        "loss_multiplier": 1.0,
    }
    prompt_batch = rl_collate_fn([dataset_prompt])
    dispatched_prompt = cast(
        DatumSpec,
        {key: value[0] for key, value in prompt_batch.items()},
    )
    saved_ledger = RolloutRecoveryLedger()
    async with DataPlaneCheckpointBarrier().mutation() as cut:
        saved_group = saved_ledger.reserve_group(
            cut,
            group_id="batch-7-prompt-0",
            admission_id="batch-7",
            prompt_id="70",
            prompt_payload=dispatched_prompt,
            expected_generations=1,
            target_step=7,
            start_weight_version=7,
            restore_level=RecoveryTargetLevel.TURN,
            admitted=True,
        )
        saved_ledger.mark_group_dispatched(
            cut,
            saved_group.group_id,
            environment_instance_id="tools/replica-0",
        )
    saved_state = build_rollout_recovery_state(
        saved_ledger,
        batch_shortfall={},
        sampler_stamps_target_steps=True,
    )
    recovery_path = tmp_path / ROLLOUT_RECOVERY_STATE_FILENAME
    torch.save(saved_state, recovery_path)
    payload_sha256 = hashlib.sha256(recovery_path.read_bytes()).hexdigest()

    rollout_manager = _RecoveryRolloutManager(RolloutRecoveryLedger())
    gym_actor = gym_actor if gym_actor is not None else FakeGymActor()
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    controller = object.__new__(controller_cls)
    _init_recovery_telemetry(controller, train_steps=7)
    controller._sampler = _CountingInOrderSampler()
    controller._rollout_manager = rollout_manager
    controller._rollout_recovery_ledger = rollout_manager.recovery_ledger
    controller._environment_checkpoint = _gym_environment_for(
        controller, gym_coordinator(gym_actor)
    )
    controller._master_config = SimpleNamespace(
        token_capture=SimpleNamespace(enabled=False)
    )
    controller._async_cfg = SimpleNamespace(
        max_buffered_rollouts=4,
        max_inflight_prompts=2,
    )
    controller._buffer_capacity = asyncio.Semaphore(4)
    controller._trainer_version = 7
    controller._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
    controller._dataloader = SimpleNamespace(
        dataset={70: dataset_prompt},
        collate_fn=rl_collate_fn,
    )
    controller._buffer = SimpleNamespace(
        count_for_target_step=lambda _target_step: 0,
        metadata_state_dict=lambda *, saved_capacity: {
            "groups": [],
            "saved_capacity": saved_capacity,
        },
    )
    return controller, rollout_manager, gym_actor, saved_group.environment_episode(0)


async def _restore_turn_recovery(
    controller: Any, tmp_path: Path, *, from_rollout_snapshot: bool = True
) -> RolloutRestorer:
    """Restore the ledger ``_turn_recovery_restart`` saved under ``tmp_path``.

    The default is a rollout snapshot, which holds Gym state; a test of a
    trainer checkpoint passes ``from_rollout_snapshot=False``.
    """
    payload = (tmp_path / ROLLOUT_RECOVERY_STATE_FILENAME).read_bytes()
    restorer = restorer_for(
        controller,
        CheckpointToRestore(
            path=str(tmp_path),
            from_rollout_snapshot=from_rollout_snapshot,
            data_plane_metadata={
                "rollout_recovery_schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
                "rollout_recovery_payload_sha256": hashlib.sha256(payload).hexdigest(),
                "rollout_recovery_group_count": 1,
            },
        ),
    )
    await restorer._maybe_restore_rollout_recovery(restored_replay_groups=0)
    return restorer


def test_turn_recovery_restores_gym_before_redispatch(tmp_path: Path) -> None:
    async def exercise() -> tuple[Any, Any, GymCheckpointManifest]:
        controller, rollout_manager, gym_actor, episode = await _turn_recovery_restart(
            tmp_path
        )
        manifest = _write_turn_manifest(tmp_path, episode)
        await _restore_turn_recovery(controller, tmp_path)
        return rollout_manager, gym_actor, manifest

    rollout_manager, gym_actor, manifest = asyncio.run(exercise())

    restored_group = rollout_manager.recovery_ledger.get_group("batch-7-prompt-0")
    restored_attempt = restored_group.siblings[0].current_attempt
    assert restored_attempt.status is RolloutAttemptStatus.RESERVED
    assert restored_attempt.environment_instance_id == "tools/replica-0"
    assert restored_attempt.environment_attempt == 1
    assert gym_actor.operations() == ["restore", "resume"]
    (restore_id, checkpoint_root, episodes), restore_kwargs = gym_actor.calls[0][1:]
    assert restore_id.startswith("restore-")
    assert checkpoint_root == str(tmp_path)
    assert episodes == manifest.instances["tools/replica-0"]
    assert restore_kwargs["source_checkpoint_id"] == manifest.checkpoint_id
    assert gym_actor.calls[1][1] == (restore_id,)


def test_turn_recovery_starts_a_gym_restart_over_from_its_task(tmp_path: Path) -> None:
    """A rollout Gym could not capture is not restored; it starts over."""

    async def exercise() -> tuple[Any, FakeGymActor]:
        controller, rollout_manager, gym_actor, episode = await _turn_recovery_restart(
            tmp_path
        )
        _write_turn_manifest(tmp_path, episode, restarted=True)
        await _restore_turn_recovery(controller, tmp_path)
        return rollout_manager, gym_actor

    rollout_manager, gym_actor = asyncio.run(exercise())

    restored_group = rollout_manager.recovery_ledger.get_group("batch-7-prompt-0")
    assert restored_group.siblings[0].attempts[0].status is (
        RolloutAttemptStatus.ABANDONED
    )
    assert gym_actor.operations() == ["restore", "resume"]
    assert gym_actor.calls[0][1][2] == ()


def _write_turn_manifest(
    tmp_path: Path,
    episode: GymCheckpointEpisode,
    *,
    attempt_delta: int = 0,
    restarted: bool = False,
) -> GymCheckpointManifest:
    saved = GymCheckpointEpisode(episode.rollout_id, episode.attempt + attempt_delta)
    manifest = GymCheckpointManifest(
        schema_version=GYM_CHECKPOINT_SCHEMA_VERSION,
        checkpoint_id="save-1",
        instances={"tools/replica-0": () if restarted else (saved,)},
        staging_keys={"tools/replica-0": ()},
        restarts={"tools/replica-0": (saved,) if restarted else ()},
    )
    write_gym_checkpoint_manifest(tmp_path, manifest)
    return manifest


def test_failure_after_gym_restore_retires_the_replacements_then_resumes(
    tmp_path: Path,
) -> None:
    """A restored topology is never released when the outer restore fails."""

    async def exercise() -> tuple[FakeGymActor, GymCheckpointEpisode]:
        controller, _, gym_actor, episode = await _turn_recovery_restart(tmp_path)
        _write_turn_manifest(tmp_path, episode)
        controller._dataloader = SimpleNamespace(collate_fn=rl_collate_fn)
        with pytest.raises(RuntimeError, match="source dataset"):
            await _restore_turn_recovery(controller, tmp_path)
        return gym_actor, episode

    gym_actor, episode = asyncio.run(exercise())

    assert gym_actor.operations() == ["restore", "retire", "resume"]
    restore_id = gym_actor.calls[0][1][0]
    assert gym_actor.calls[1][1][:2] == (restore_id, (episode.next_attempt(),))


def test_failed_gym_restore_is_cleaned_up_once(
    tmp_path: Path,
) -> None:
    """The restore cleans up its own failure; nothing retires or resumes again."""

    async def exercise() -> FakeGymActor:
        controller, _, gym_actor, episode = await _turn_recovery_restart(
            tmp_path,
            gym_actor=FakeGymActor(restore_error=RuntimeError("gym restore failed")),
        )
        _write_turn_manifest(tmp_path, episode)
        with pytest.raises(RuntimeError, match="gym restore failed"):
            await _restore_turn_recovery(controller, tmp_path)
        return gym_actor

    gym_actor = asyncio.run(exercise())

    # Only the coordinator's own cleanup: one retire, one resume.
    assert gym_actor.operations() == ["restore", "retire", "resume"]


def test_manifest_that_disagrees_with_the_ledger_fails_before_any_gym_call(
    tmp_path: Path,
) -> None:
    async def exercise() -> FakeGymActor:
        controller, _, gym_actor, episode = await _turn_recovery_restart(tmp_path)
        _write_turn_manifest(tmp_path, episode, attempt_delta=1)
        with pytest.raises(
            ValueError,
            match=r"does not match the rollout recovery ledger for Gym instances "
            r"\['tools/replica-0'\]",
        ):
            await _restore_turn_recovery(controller, tmp_path)
        return gym_actor

    assert asyncio.run(exercise()).calls == []


def test_turn_state_without_gym_coordination_is_refused(tmp_path: Path) -> None:
    async def exercise() -> None:
        controller, _, _, episode = await _turn_recovery_restart(tmp_path)
        controller._environment_checkpoint = NoEnvironmentCheckpoint()
        _write_turn_manifest(tmp_path, episode)
        await _restore_turn_recovery(controller, tmp_path)

    with pytest.raises(RuntimeError, match="keeps none to restore"):
        asyncio.run(exercise())


def test_turn_recovery_from_a_trainer_checkpoint_restarts_unfinished_turns(
    tmp_path: Path,
) -> None:
    """A trainer checkpoint holds no Gym state, so turns restart from their tasks."""

    async def exercise() -> tuple[Any, Any]:
        controller, rollout_manager, gym_actor, _ = await _turn_recovery_restart(
            tmp_path
        )
        await _restore_turn_recovery(controller, tmp_path, from_rollout_snapshot=False)
        return rollout_manager, gym_actor

    rollout_manager, gym_actor = asyncio.run(exercise())

    restored_group = rollout_manager.recovery_ledger.get_group("batch-7-prompt-0")
    assert (
        restored_group.siblings[0].current_attempt.status
        is RolloutAttemptStatus.ABANDONED
    )
    assert gym_actor.calls == []


@pytest.mark.parametrize(
    ("gym_state", "continued", "restarted"),
    [("manifest", 1.0, 0.0), ("trainer_checkpoint", 0.0, 1.0)],
)
def test_turn_recovery_counts_continued_and_restarted_siblings(
    tmp_path: Path, gym_state: str, continued: float, restarted: float
) -> None:
    """Re-run siblings split by whether they resume from Gym state or their prompt."""

    async def exercise() -> dict[str, float]:
        controller, rollout_manager, _, episode = await _turn_recovery_restart(tmp_path)
        if gym_state == "manifest":
            _write_turn_manifest(tmp_path, episode)
        restorer = await _restore_turn_recovery(
            controller, tmp_path, from_rollout_snapshot=gym_state == "manifest"
        )

        async def _recover(
            _prompt: dict[str, Any], _target_step: int | None, group_id: str
        ) -> None:
            async with controller._data_plane_checkpoint_barrier.mutation() as cut:
                await rollout_manager.complete_recovery(cut, group_id)

        await restorer.redispatch_restored(
            _recover, controller._admit_reserved_prompt_groups
        )
        return controller._logger.log_metrics.call_args.args[0]

    metrics = asyncio.run(exercise())

    assert metrics["siblings_rerun"] == 1.0
    assert metrics["siblings_continued"] == continued
    assert metrics["siblings_restarted"] == restarted


def test_turn_recovery_refuses_a_rollout_snapshot_without_its_gym_manifest(
    tmp_path: Path,
) -> None:
    """A snapshot holds Gym state, so restarting its turns would lose it silently."""

    async def exercise() -> None:
        controller, _, gym_actor, _ = await _turn_recovery_restart(tmp_path)
        try:
            await _restore_turn_recovery(controller, tmp_path)
        finally:
            assert gym_actor.calls == []

    with pytest.raises(FileNotFoundError, match="gym_checkpoint.json"):
        asyncio.run(exercise())


def test_turn_checkpoint_commits_owned_gym_episode_before_tq_cut(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        ledger = RolloutRecoveryLedger()
        async with DataPlaneCheckpointBarrier().mutation() as cut:
            group = _dispatched_turn_group(ledger, cut, generations=1)

        actor = FakeGymActor()
        controller = _controller_for(gym_coordinator(actor), ledger)

        async with _gym_commit(
            controller,
            tmp_path,
            checkpoint_id="save-1",
        ):
            async with DataPlaneCheckpointBarrier().checkpoint():
                actor.mark("tq-cut")

        assert actor.operations() == ["prepare", "renew", "commit", "tq-cut", "resume"]
        (commit_call,) = [call for call in actor.calls if call[0] == "commit"]
        assert commit_call[1] == (
            "save-1",
            str(tmp_path),
            (group.environment_episode(0),),
        )

    asyncio.run(exercise())


def test_recovery_rejects_an_unhandled_phase_before_redispatch() -> None:
    """A future phase must fail loudly instead of remaining owned forever."""

    async def exercise() -> None:
        recovery_ledger = SimpleNamespace(
            groups=lambda: [
                SimpleNamespace(group_id="future-group", phase="future-phase")
            ]
        )
        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        controller = object.__new__(controller_cls)
        _init_recovery_telemetry(controller)
        controller._rollout_manager = SimpleNamespace(recovery_ledger=recovery_ledger)
        launched = False

        async def _recover(
            _prompt: dict[str, Any],
            _target_step: int | None,
            _group_id: str,
        ) -> None:
            nonlocal launched
            launched = True

        with pytest.raises(
            RuntimeError,
            match=r"unrecognized rollout recovery phase.*future-group='future-phase'",
        ):
            await restorer_for(controller).redispatch_restored(
                _recover, controller._admit_reserved_prompt_groups
            )

        assert not launched

    asyncio.run(exercise())


def test_recovery_readmits_one_reserved_batch_only_once(tmp_path) -> None:
    """Two prompts fetched together consume one sampler admission on restart."""

    async def exercise() -> None:
        saved_ledger = RolloutRecoveryLedger()
        async with DataPlaneCheckpointBarrier().mutation() as cut:
            for prompt_idx in (70, 71):
                saved_ledger.reserve_group(
                    cut,
                    group_id=f"batch-7-prompt-{prompt_idx}",
                    admission_id="batch-7",
                    prompt_id=str(prompt_idx),
                    prompt_payload={"idx": prompt_idx, "message_log": []},
                    expected_generations=2,
                    target_step=None,
                    start_weight_version=7,
                    admitted=False,
                )
        recovery_path = tmp_path / ROLLOUT_RECOVERY_STATE_FILENAME
        torch.save(_sidecar_state(saved_ledger), recovery_path)

        sampler = _CountingInOrderSampler()
        sampler.restore_dispatch_index(6)
        rollout_manager = _RecoveryRolloutManager(RolloutRecoveryLedger())
        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        controller = object.__new__(controller_cls)
        _init_recovery_telemetry(controller, train_steps=7)
        controller._sampler = sampler
        controller._rollout_manager = rollout_manager
        controller._master_config = SimpleNamespace(
            token_capture=SimpleNamespace(enabled=False)
        )
        checkpoint = CheckpointToRestore(
            path=str(tmp_path),
            data_plane_metadata={
                "rollout_recovery_schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
                "rollout_recovery_payload_sha256": hashlib.sha256(
                    recovery_path.read_bytes()
                ).hexdigest(),
                "rollout_recovery_group_count": 2,
            },
        )
        controller._async_cfg = SimpleNamespace(
            max_buffered_rollouts=4,
            max_inflight_prompts=2,
        )
        controller._buffer_capacity = asyncio.Semaphore(4)
        controller._trainer_version = 7
        controller._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
        controller._dataloader = SimpleNamespace(
            dataset={
                prompt_idx: {"idx": prompt_idx, "message_log": []}
                for prompt_idx in (70, 71)
            }
        )
        controller._buffer = SimpleNamespace(
            count_for_target_step=lambda _target_step: 0,
            metadata_state_dict=lambda *, saved_capacity: {
                "groups": [],
                "saved_capacity": saved_capacity,
            },
        )

        async def _recover(
            _prompt: dict[str, Any],
            _target_step: int | None,
            group_id: str,
        ) -> None:
            async with controller._data_plane_checkpoint_barrier.mutation() as cut:
                await rollout_manager.complete_recovery(cut, group_id)

        restorer = restorer_for(controller, checkpoint)
        # Re-admission reserves through the controller's ledger dispatch.
        use_dispatch(controller, records_prompt_groups=True)
        await restorer._maybe_restore_rollout_recovery(restored_replay_groups=0)
        await restorer.redispatch_restored(
            _recover, controller._admit_reserved_prompt_groups
        )

        assert sampler.admit_calls == 0
        assert sampler.admission_commits == 1
        assert sampler.dispatch_index == 7
        assert set(rollout_manager.recovered) == {
            ("batch-7-prompt-70", 7),
            ("batch-7-prompt-71", 7),
        }

    asyncio.run(exercise())


def test_recovery_launches_admitted_groups_before_waiting_to_readmit() -> None:
    """Recovered work can open the gate that a reserved batch is waiting on."""

    async def exercise() -> None:
        sampler = _CountingInOrderSampler()
        sampler.restore_dispatch_index(7)
        ledger = RolloutRecoveryLedger()
        barrier = DataPlaneCheckpointBarrier()
        async with barrier.mutation() as cut:
            ledger.reserve_group(
                cut,
                group_id="admitted-step-7",
                admission_id="batch-7",
                prompt_id="70",
                prompt_payload={"idx": 70, "message_log": []},
                expected_generations=2,
                target_step=7,
                start_weight_version=7,
                admitted=True,
            )
            ledger.reserve_group(
                cut,
                group_id="reserved-step-8",
                admission_id="batch-8",
                prompt_id="80",
                prompt_payload={"idx": 80, "message_log": []},
                expected_generations=2,
                target_step=None,
                start_weight_version=7,
                admitted=False,
            )
        rollout_manager = _RecoveryRolloutManager(ledger)

        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        controller = object.__new__(controller_cls)
        _init_recovery_telemetry(controller, train_steps=6)
        controller._sampler = sampler
        controller._rollout_manager = rollout_manager
        controller._trainer_version = 6
        controller._data_plane_checkpoint_barrier = barrier
        controller._buffer = SimpleNamespace(
            count_for_target_step=lambda _target_step: 0,
        )
        # Re-admission reserves through the controller's ledger dispatch.
        use_dispatch(controller, records_prompt_groups=True)

        launched: list[tuple[str, int | None]] = []

        async def _recover(
            _prompt: dict[str, Any],
            target_step: int | None,
            group_id: str,
        ) -> None:
            launched.append((group_id, target_step))
            async with controller._data_plane_checkpoint_barrier.mutation() as cut:
                await rollout_manager.complete_recovery(cut, group_id)
            if group_id == "admitted-step-7":
                # Model the concurrent train pump consuming recovered step 7. This
                # opens the in-order gate so the reserved batch can become step 8.
                controller._trainer_version = 7

        await asyncio.wait_for(
            restorer_for(controller).redispatch_restored(
                _recover, controller._admit_reserved_prompt_groups
            ),
            timeout=1.0,
        )

        assert launched == [
            ("admitted-step-7", 7),
            ("reserved-step-8", 8),
        ]
        assert sampler.admission_commits == 1
        assert sampler.dispatch_index == 8
        assert len(ledger) == 0

    asyncio.run(exercise())


def test_recovery_load_does_not_require_every_unfinished_group_to_fit_at_once(
    tmp_path,
) -> None:
    """The train pump may free replay slots while recovery is redispatching."""

    async def exercise() -> None:
        saved_ledger = RolloutRecoveryLedger()
        async with DataPlaneCheckpointBarrier().mutation() as cut:
            for prompt_idx in (70, 71):
                saved_ledger.reserve_group(
                    cut,
                    group_id=f"batch-7-prompt-{prompt_idx}",
                    admission_id="batch-7",
                    prompt_id=str(prompt_idx),
                    prompt_payload={"idx": prompt_idx, "message_log": []},
                    expected_generations=2,
                    target_step=7,
                    start_weight_version=7,
                    admitted=True,
                )
        recovery_path = tmp_path / ROLLOUT_RECOVERY_STATE_FILENAME
        torch.save(_sidecar_state(saved_ledger), recovery_path)

        rollout_manager = _RecoveryRolloutManager(RolloutRecoveryLedger())
        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        controller = object.__new__(controller_cls)
        controller._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
        controller._rollout_manager = rollout_manager
        controller._master_config = SimpleNamespace(
            token_capture=SimpleNamespace(enabled=False)
        )
        checkpoint = CheckpointToRestore(
            path=str(tmp_path),
            data_plane_metadata={
                "rollout_recovery_schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
                "rollout_recovery_payload_sha256": hashlib.sha256(
                    recovery_path.read_bytes()
                ).hexdigest(),
                "rollout_recovery_group_count": 2,
            },
        )
        controller._async_cfg = SimpleNamespace(max_buffered_rollouts=4)
        controller._dataloader = SimpleNamespace(
            dataset={
                prompt_idx: {"idx": prompt_idx, "message_log": []}
                for prompt_idx in (70, 71)
            }
        )
        controller._buffer = SimpleNamespace(
            metadata_state_dict=lambda *, saved_capacity: {
                "groups": [{"group_id": f"canonical-{idx}"} for idx in range(3)],
                "saved_capacity": saved_capacity,
            }
        )

        # Three canonical groups plus two unfinished groups exceed capacity four,
        # but only the canonical groups occupy slots at restore time. Recovery is
        # launched beside the train pump, which releases capacity as it consumes.
        await restorer_for(controller, checkpoint)._maybe_restore_rollout_recovery(
            restored_replay_groups=3
        )

        assert len(rollout_manager.recovery_ledger) == 2

    asyncio.run(exercise())


def test_checkpoint_waits_for_replacement_reserve_refill() -> None:
    """The dataloader-owned batch is visible in the pool after the mutation cut."""

    async def exercise() -> None:
        controller = _reserve_controller()
        mutation_applied = asyncio.Event()
        release_mutation = asyncio.Event()
        checkpoint_entered = asyncio.Event()
        batch = BatchedDataDict(
            {
                "idx": [20, 21],
                "message_log": [
                    [{"role": "user", "content": "prompt 20"}],
                    [{"role": "user", "content": "prompt 21"}],
                ],
                "length": [1, 1],
                "extra_env_info": [None, None],
                "loss_multiplier": [1.0, 1.0],
            }
        )

        async def refill() -> None:
            async with controller._data_plane_checkpoint_barrier.mutation():
                assert controller._divert_batch_to_reserve(batch)
                mutation_applied.set()
                await release_mutation.wait()

        async def checkpoint_snapshot() -> list[int]:
            async with controller._data_plane_checkpoint_barrier.checkpoint():
                checkpoint_entered.set()
                return [prompt["idx"] for prompt in controller._replacement_reserve]

        refill_task = asyncio.create_task(refill())
        await mutation_applied.wait()
        checkpoint_task = asyncio.create_task(checkpoint_snapshot())
        await asyncio.sleep(0)
        assert not checkpoint_entered.is_set()

        release_mutation.set()
        assert await checkpoint_task == [20, 21]
        await refill_task

    asyncio.run(exercise())


def test_checkpoint_waits_for_replacement_pop_and_reownership() -> None:
    """A skipped owner becomes its replacement atomically at checkpoint time."""

    async def exercise() -> None:
        controller = _reserve_controller()
        manager = controller._rollout_manager
        async with controller._data_plane_checkpoint_barrier.mutation() as cut:
            old_group_id = manager.reserve_prompt_group(
                cut,
                _reserve_prompt(20),
                target_step=7,
            )
        controller._replacement_reserve.append(_reserve_prompt(21))
        mutation_applied = asyncio.Event()
        release_mutation = asyncio.Event()
        checkpoint_entered = asyncio.Event()

        async def replace() -> None:
            async with controller._data_plane_checkpoint_barrier.mutation() as cut:
                replacement = controller._take_replacement(7, 0)
                assert replacement is not None
                manager.discard_prompt_group(cut, old_group_id)
                manager.reserve_prompt_group(cut, replacement, target_step=7)
                mutation_applied.set()
                await release_mutation.wait()

        async def checkpoint_snapshot() -> tuple[list[int], list[str]]:
            async with controller._data_plane_checkpoint_barrier.checkpoint():
                checkpoint_entered.set()
                reserve_ids = [
                    prompt["idx"] for prompt in controller._replacement_reserve
                ]
                ledger_prompt_ids = [
                    group.prompt_id for group in manager.recovery_ledger.groups()
                ]
                return reserve_ids, ledger_prompt_ids

        replace_task = asyncio.create_task(replace())
        await mutation_applied.wait()
        checkpoint_task = asyncio.create_task(checkpoint_snapshot())
        await asyncio.sleep(0)
        assert not checkpoint_entered.is_set()

        release_mutation.set()
        reserve_ids, ledger_prompt_ids = await checkpoint_task
        await replace_task

        assert reserve_ids == []
        assert ledger_prompt_ids == ["21"]

    asyncio.run(exercise())


def test_reserve_drain_is_recoverable_before_sampler_admission() -> None:
    """After pool removal, RESERVED ledger records own the whole batch."""

    async def exercise() -> None:
        controller = _reserve_controller()
        controller._replacement_reserve.extend(
            [_reserve_prompt(20), _reserve_prompt(21)]
        )
        admission_started = asyncio.Event()
        release_admission = asyncio.Event()
        launched: list[tuple[int, int | None, str | None]] = []

        async def block_admission(
            group_ids: list[str],
        ) -> int:
            admission_started.set()
            await release_admission.wait()
            async with controller._data_plane_checkpoint_barrier.mutation() as cut:
                for group_id in group_ids:
                    controller._rollout_manager.mark_prompt_group_admitted(
                        cut,
                        group_id,
                        target_step=7,
                    )
            return 7

        async def launch(
            prompt: DatumSpec,
            target_step: int | None,
            group_id: str | None,
        ) -> None:
            launched.append((prompt["idx"], target_step, group_id))

        controller._admit_reserved_prompt_groups = block_admission
        drain_task = asyncio.create_task(controller._drain_reserve_into_steps(launch))
        await admission_started.wait()

        async with controller._data_plane_checkpoint_barrier.checkpoint():
            assert list(controller._replacement_reserve) == []
            groups = controller._rollout_manager.recovery_ledger.groups()
            assert [group.prompt_id for group in groups] == ["20", "21"]
            assert all(group.phase is PromptGroupPhase.RESERVED for group in groups)
            assert len({group.admission_id for group in groups}) == 1

        release_admission.set()
        await drain_task
        assert {(idx, target_step) for idx, target_step, _ in launched} == {
            (20, 7),
            (21, 7),
        }

    asyncio.run(exercise())


def test_recovery_rejects_a_corrupt_ledger_sidecar(tmp_path) -> None:
    recovery_path = tmp_path / ROLLOUT_RECOVERY_STATE_FILENAME
    recovery_path.write_bytes(b"corrupt checkpoint payload")

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    controller = object.__new__(controller_cls)
    restorer = restorer_for(
        controller,
        CheckpointToRestore(
            path=str(tmp_path),
            data_plane_metadata={
                "rollout_recovery_schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
                "rollout_recovery_payload_sha256": "0" * 64,
                "rollout_recovery_group_count": 1,
            },
        ),
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        asyncio.run(restorer._maybe_restore_rollout_recovery(restored_replay_groups=0))


def test_recovery_rejects_a_missing_advertised_ledger_sidecar(tmp_path) -> None:
    """Do not combine an older/missing ledger with the restored trainer step."""

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    controller = object.__new__(controller_cls)
    restorer = restorer_for(
        controller,
        CheckpointToRestore(
            path=str(tmp_path),
            data_plane_metadata={
                "rollout_recovery_schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
                "rollout_recovery_payload_sha256": "0" * 64,
                "rollout_recovery_group_count": 1,
            },
        ),
    )

    with pytest.raises(FileNotFoundError, match="sidecar is missing"):
        asyncio.run(restorer._maybe_restore_rollout_recovery(restored_replay_groups=0))


class TestGymCheckpointPausesRolloutDeadlines:
    """Parked rollouts cannot progress, so a checkpoint must not use up their budget."""

    def test_a_parked_rollout_outlives_a_checkpoint_longer_than_its_budget(
        self, tmp_path: Path
    ) -> None:
        registry = RequestDeadlineRegistry()
        controller = _controller_for(
            gym_coordinator(FakeGymActor(exported_episodes=())),
            manager=_deadline_pausing_manager(registry),
        )

        async def exercise() -> None:
            entered = asyncio.Event()
            checkpoint_done = asyncio.Event()

            async def parked_rollout() -> None:
                async with _Deadline(0.2, "NeMo-Gym prompt group", registry=registry):
                    entered.set()
                    await checkpoint_done.wait()
                    await asyncio.sleep(0.05)

            rollout = asyncio.create_task(parked_rollout())
            await entered.wait()
            async with _gym_commit(controller, tmp_path, checkpoint_id="save-1"):
                assert registry.suspended
                # Twice the rollout's whole budget.
                await asyncio.sleep(0.4)
            checkpoint_done.set()
            # Raises RolloutTimeout if the checkpoint was charged to the rollout.
            await rollout
            assert not registry.suspended

        asyncio.run(exercise())

    def test_ending_a_checkpoint_keeps_a_generation_stand_down_pause(
        self, tmp_path: Path
    ) -> None:
        registry = RequestDeadlineRegistry()
        controller = _controller_for(
            gym_coordinator(FakeGymActor(exported_episodes=())),
            manager=_deadline_pausing_manager(registry),
        )

        async def exercise() -> None:
            # The colocated engine is training; its own pause must outlast ours.
            registry.suspend()
            async with _gym_commit(controller, tmp_path, checkpoint_id="save-1"):
                pass
            assert registry.suspended
            registry.resume()
            assert not registry.suspended

        asyncio.run(exercise())

    def test_a_checkpoint_that_cannot_park_releases_its_pause(
        self, tmp_path: Path
    ) -> None:
        registry = RequestDeadlineRegistry()
        controller = _controller_for(
            gym_coordinator(FakeGymActor(prepared=False)),
            manager=_deadline_pausing_manager(registry),
        )

        async def exercise() -> None:
            with pytest.raises(GymCheckpointNotReady):
                async with _gym_commit(controller, tmp_path, checkpoint_id="save-1"):
                    raise AssertionError("unreachable")
            assert not registry.suspended


class TestARowDroppedDuringACheckpointDoesNotFailIt:
    """RL can drop rows while Gym is prepared; the checkpoint must hold.

    The drop marks the rows abandoned at once, but Gym refuses a retire until it
    resumes, so it still holds and exports them. Restore requires the saved Gym
    manifest to equal RL's inventory, so the saved checkpoint must leave the
    dropped episodes out, and they stay queued for the retire after resume.
    """

    @staticmethod
    async def _drop(
        ledger: RolloutRecoveryLedger,
        barrier: DataPlaneCheckpointBarrier,
        group_id: str,
    ) -> None:
        # A cancelled group task queues its retire for the next checkpoint.
        async with barrier.mutation() as cut:
            ledger.abandon_unsealed(cut, group_id)

    @staticmethod
    async def _save(
        controller: Any, barrier: DataPlaneCheckpointBarrier, tmp_path: Path
    ) -> None:
        async with _gym_commit(
            controller, tmp_path, checkpoint_id="save-1"
        ) as gym_commit:
            assert gym_commit is not None
            async with barrier.checkpoint() as cut:
                await controller._environment_checkpoint.validate_cut(
                    cut, gym_commit, checkpoint_path=tmp_path
                )

    @staticmethod
    def _assert_restorable(
        controller: Any,
        coordinator: GymCheckpointCoordinator,
        tmp_path: Path,
        *,
        kept: set[GymCheckpointEpisode],
    ) -> None:
        saved = load_gym_checkpoint_manifest(tmp_path).instances
        assert saved == controller._environment_checkpoint.inventory()
        assert set(saved["tools/replica-0"]) == kept

    def test_a_drop_after_the_pre_prepare_retire(self, tmp_path: Path) -> None:
        """Gym exports the dropped row it still holds, outside RL's candidates."""

        async def exercise() -> None:
            barrier = DataPlaneCheckpointBarrier()
            ledger = RolloutRecoveryLedger()
            async with barrier.mutation() as cut:
                group = _dispatched_turn_group(ledger, cut)
            row_0, row_1 = (group.environment_episode(index) for index in (0, 1))

            coordinator = gym_coordinator(
                FakeGymActor(
                    live={row_0, row_1},
                    while_prepared=lambda: self._drop(ledger, barrier, group.group_id),
                )
            )
            controller = _controller_for(coordinator, ledger)
            controller._data_plane_checkpoint_barrier = barrier

            await asyncio.wait_for(
                self._save(controller, barrier, tmp_path), _ASYNC_TEST_TIMEOUT_S
            )

            self._assert_restorable(controller, coordinator, tmp_path, kept=set())
            assert ledger.environment_retirements({"tools/replica-0"}) == {
                "tools/replica-0": tuple(sorted((row_0, row_1)))
            }

        asyncio.run(exercise())

    def test_a_drop_while_the_commit_drains(self, tmp_path: Path) -> None:
        """The group is dropped while its on-wire reply drains, with a row Gym exported."""

        async def exercise() -> None:
            barrier = DataPlaneCheckpointBarrier()
            ledger = RolloutRecoveryLedger()
            async with barrier.mutation() as cut:
                group = _dispatched_turn_group(ledger, cut)
            row_0, row_1 = (group.environment_episode(index) for index in (0, 1))
            # Row 0 is parked in Gym and exported; row 1's reply is on the wire.
            actor = FakeGymActor(live={row_0})
            coordinator = gym_coordinator(actor)
            controller = _controller_for(coordinator, ledger)
            controller._data_plane_checkpoint_barrier = barrier

            async def fail_the_on_wire_reply() -> None:
                while "commit" not in actor.operations():
                    await asyncio.sleep(0)
                await self._drop(ledger, barrier, group.group_id)

            failing = asyncio.create_task(fail_the_on_wire_reply())
            await asyncio.wait_for(
                self._save(controller, barrier, tmp_path), _ASYNC_TEST_TIMEOUT_S
            )
            await asyncio.wait_for(failing, _ASYNC_TEST_TIMEOUT_S)

            self._assert_restorable(controller, coordinator, tmp_path, kept=set())
            assert ledger.environment_retirements({"tools/replica-0"}) == {
                "tools/replica-0": tuple(sorted((row_0, row_1)))
            }

        asyncio.run(exercise())

    def test_a_dropped_episode_takes_its_staging_keys_with_it(
        self, tmp_path: Path
    ) -> None:
        """The cut needs, and the saved manifest keeps, only the kept episodes' rows."""

        async def exercise() -> None:
            barrier = DataPlaneCheckpointBarrier()
            ledger = RolloutRecoveryLedger()
            async with barrier.mutation() as cut:
                group = _dispatched_turn_group(ledger, cut)
            row_0, row_1 = (group.environment_episode(index) for index in (0, 1))
            coordinator = gym_coordinator(
                FakeGymActor(
                    live={row_0, row_1},
                    while_prepared=lambda: self._drop(ledger, barrier, group.group_id),
                    staging_keys_by_episode={
                        row_0: (f"{row_0.rollout_id}/call-0",),
                        row_1: (f"{row_1.rollout_id}/call-0",),
                    },
                )
            )

            async def list_staging_keys() -> list[str]:
                # The dropped group's staging is gone by the time the cut looks.
                return []

            controller = _controller_for(
                coordinator, ledger, list_staging_keys=list_staging_keys
            )
            controller._data_plane_checkpoint_barrier = barrier

            await asyncio.wait_for(
                self._save(controller, barrier, tmp_path), _ASYNC_TEST_TIMEOUT_S
            )

            saved = load_gym_checkpoint_manifest(tmp_path)
            assert saved.instances == {"tools/replica-0": ()}
            assert saved.staging_keys == {"tools/replica-0": ()}

        asyncio.run(exercise())

    def test_only_a_queued_drop_may_differ_from_the_saved_manifest(
        self, tmp_path: Path
    ) -> None:
        """An exported episode RL dropped without queuing a retire still fails."""

        async def exercise() -> None:
            barrier = DataPlaneCheckpointBarrier()
            ledger = RolloutRecoveryLedger()
            async with barrier.mutation() as cut:
                group = _dispatched_turn_group(ledger, cut)
            row_0, row_1 = (group.environment_episode(index) for index in (0, 1))
            coordinator = gym_coordinator(FakeGymActor(live={row_0, row_1}))
            controller = _controller_for(coordinator, ledger)
            controller._data_plane_checkpoint_barrier = barrier

            async with _gym_commit(
                controller, tmp_path, checkpoint_id="save-1"
            ) as gym_commit:
                assert gym_commit is not None
                await self._drop(ledger, barrier, group.group_id)
                # As if the drop's retire had already been acknowledged.
                ledger.mark_environment_retired({"tools/replica-0": [row_0]})
                async with barrier.checkpoint() as cut:
                    with pytest.raises(RuntimeError, match="ownership changed"):
                        await controller._environment_checkpoint.validate_cut(
                            cut, gym_commit, checkpoint_path=tmp_path
                        )

        asyncio.run(exercise())


def _restart_only_check(
    restart_only_servers: tuple[str, ...], *, allowed: bool
) -> tuple[Any, FakeGymActor]:
    actor = FakeGymActor(restart_only_servers=restart_only_servers)
    controller = _controller_for(
        gym_coordinator(actor), allow_restart_only_servers=allowed
    )
    return controller, actor


def test_turn_recovery_refuses_a_restart_only_gym_server_by_default() -> None:
    """Its rollouts would start over after a crash without anyone being told."""
    controller, actor = _restart_only_check(("tools_resources",), allowed=False)

    with pytest.raises(ValueError, match="restart-only Gym servers") as raised:
        asyncio.run(controller._environment_checkpoint.check_startup())

    assert "tools/replica-0" in str(raised.value)
    assert "tools_resources" in str(raised.value)
    assert "allow_restart_only_gym_servers=true" in str(raised.value)
    assert actor.operations() == ["status"]


def test_an_allowed_restart_only_gym_server_is_named_in_a_warning() -> None:
    controller, _ = _restart_only_check(("tools_agent",), allowed=True)

    with pytest.warns(UserWarning, match="start over after a crash.*tools_agent"):
        asyncio.run(controller._environment_checkpoint.check_startup())


def test_allowing_restart_only_gym_servers_warns_when_there_are_none() -> None:
    controller, _ = _restart_only_check((), allowed=True)

    with pytest.warns(UserWarning, match="no Gym server is restart-only"):
        asyncio.run(controller._environment_checkpoint.check_startup())


def test_no_restart_only_gym_server_and_no_opt_in_passes_silently() -> None:
    controller, _ = _restart_only_check((), allowed=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        asyncio.run(controller._environment_checkpoint.check_startup())


def test_without_turn_recovery_there_are_no_gym_servers_to_check() -> None:
    controller = object.__new__(SingleControllerActor.__ray_metadata__.modified_class)
    controller._environment_checkpoint = NoEnvironmentCheckpoint()

    asyncio.run(controller._environment_checkpoint.check_startup())
