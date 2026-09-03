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
import hashlib
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar, cast

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
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollout_manager import RolloutOutcome
from nemo_rl.experience.rollout_recovery import (
    ROLLOUT_RECOVERY_SCHEMA_VERSION,
    ROLLOUT_RECOVERY_STATE_FILENAME,
    PromptGroupPhase,
    RolloutRecoveryLedger,
    build_rollout_recovery_state,
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


class _BlockingRolloutManager:
    """Hold one admitted rollout unfinished while the checkpoint is written."""

    def __init__(self, ledger: _PendingLedger) -> None:
        self.recovery_ledger = ledger
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.weight_version = 0

    def set_weight_version(self, version: int) -> None:
        self.weight_version = version

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
            await controller._rehydrate_rollout_recovery_prompts(cut)

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
    controller._rollout_recovery_enabled = True
    controller._async_cfg = SimpleNamespace(
        rollout_failure=SimpleNamespace(
            on_dropped_prompt="replace",
            replacement_reserve_prompts=2,
            max_replacement_attempts=1,
        )
    )
    controller._algo_cfg = SimpleNamespace(num_prompts_per_step=2)
    controller._rollout_manager = _LedgerFacade()
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
        controller._sampler = sampler
        controller._rollout_manager = rollout_manager
        controller._last_checkpoint_path = str(tmp_path)
        controller._data_plane_checkpoint_metadata = {
            "rollout_recovery_schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
            "rollout_recovery_payload_sha256": payload_sha256,
            "rollout_recovery_group_count": 1,
        }
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

        await controller._maybe_restore_rollout_recovery(restored_replay_groups=0)
        await controller._redispatch_restored_rollouts(_recover)

        assert rollout_manager.recovered == [("batch-7-prompt-0", 7)]
        assert sampler.admit_calls == 0
        assert sampler.admission_commits == 0
        assert sampler.dispatch_index == 7
        assert controller._batch_shortfall == {6: 1}
        assert controller._sampler_stamps_target_steps is True

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
            await controller._redispatch_restored_rollouts(_recover)

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
        torch.save(saved_ledger.state_dict(), recovery_path)

        sampler = _CountingInOrderSampler()
        sampler.restore_dispatch_index(6)
        rollout_manager = _RecoveryRolloutManager(RolloutRecoveryLedger())
        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        controller = object.__new__(controller_cls)
        controller._sampler = sampler
        controller._rollout_manager = rollout_manager
        controller._last_checkpoint_path = str(tmp_path)
        controller._data_plane_checkpoint_metadata = {
            "rollout_recovery_schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
            "rollout_recovery_payload_sha256": hashlib.sha256(
                recovery_path.read_bytes()
            ).hexdigest(),
            "rollout_recovery_group_count": 2,
        }
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

        await controller._maybe_restore_rollout_recovery(restored_replay_groups=0)
        await controller._redispatch_restored_rollouts(_recover)

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
        controller._sampler = sampler
        controller._rollout_manager = rollout_manager
        controller._trainer_version = 6
        controller._data_plane_checkpoint_barrier = barrier
        controller._buffer = SimpleNamespace(
            count_for_target_step=lambda _target_step: 0,
        )

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
            controller._redispatch_restored_rollouts(_recover),
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
        torch.save(saved_ledger.state_dict(), recovery_path)

        rollout_manager = _RecoveryRolloutManager(RolloutRecoveryLedger())
        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        controller = object.__new__(controller_cls)
        controller._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
        controller._rollout_manager = rollout_manager
        controller._last_checkpoint_path = str(tmp_path)
        controller._data_plane_checkpoint_metadata = {
            "rollout_recovery_schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
            "rollout_recovery_payload_sha256": hashlib.sha256(
                recovery_path.read_bytes()
            ).hexdigest(),
            "rollout_recovery_group_count": 2,
        }
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
        await controller._maybe_restore_rollout_recovery(restored_replay_groups=3)

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
        ) -> tuple[int, list[str], int]:
            admission_started.set()
            await release_admission.wait()
            async with controller._data_plane_checkpoint_barrier.mutation() as cut:
                for group_id in group_ids:
                    controller._rollout_manager.mark_prompt_group_admitted(
                        cut,
                        group_id,
                        target_step=7,
                    )
            return 7, group_ids, 0

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
    controller._last_checkpoint_path = str(tmp_path)
    controller._data_plane_checkpoint_metadata = {
        "rollout_recovery_schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
        "rollout_recovery_payload_sha256": "0" * 64,
        "rollout_recovery_group_count": 1,
    }

    with pytest.raises(ValueError, match="checksum mismatch"):
        asyncio.run(
            controller._maybe_restore_rollout_recovery(restored_replay_groups=0)
        )


def test_recovery_rejects_a_missing_advertised_ledger_sidecar(tmp_path) -> None:
    """Do not combine an older/missing ledger with the restored trainer step."""

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    controller = object.__new__(controller_cls)
    controller._last_checkpoint_path = str(tmp_path)
    controller._data_plane_checkpoint_metadata = {
        "rollout_recovery_schema_version": ROLLOUT_RECOVERY_SCHEMA_VERSION,
        "rollout_recovery_payload_sha256": "0" * 64,
        "rollout_recovery_group_count": 1,
    }

    with pytest.raises(FileNotFoundError, match="sidecar is missing"):
        asyncio.run(
            controller._maybe_restore_rollout_recovery(restored_replay_groups=0)
        )
