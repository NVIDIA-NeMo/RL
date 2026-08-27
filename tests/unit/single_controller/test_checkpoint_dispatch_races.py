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
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from nemo_rl.algorithms.async_utils.replay_buffer import (
    REPLAY_BUFFER_METADATA_FILENAME,
    TQReplayBuffer,
)
from nemo_rl.algorithms.async_utils.staleness_sampler import InOrderSampler
from nemo_rl.algorithms.grpo import _initial_grpo_save_state
from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollout_manager import RolloutOutcome
from tests.unit.single_controller._checkpoint_scenarios import (
    _record,
    patch_converter,
)
from tests.unit.single_controller.test_checkpointing import (
    _FakeDataloader,
    _actor_master_config,
    _make_actor_args,
)


class _CountingInOrderSampler(InOrderSampler):
    """Real in-order sampler with observable admission calls."""

    def __init__(self) -> None:
        super().__init__(None, max_lookahead_versions=1)
        self.admit_calls = 0

    async def admit(self, *, trainer_version_fn):
        self.admit_calls += 1
        return await super().admit(trainer_version_fn=trainer_version_fn)


class _BlockingBeforeAdmissionSampler(_CountingInOrderSampler):
    """Pause after the dataloader advances but before admission mutates state."""

    def __init__(self) -> None:
        super().__init__()
        self.admission_entered = asyncio.Event()
        self.release_admission = asyncio.Event()

    async def admit(self, *, trainer_version_fn):
        self.admission_entered.set()
        await self.release_admission.wait()
        return await super().admit(trainer_version_fn=trainer_version_fn)


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
            "schema_version": 1,
            "groups": [
                {
                    "group_id": group.group_id,
                    "target_step": group.target_step,
                    "prompt_payload": group.prompt_payload,
                }
                for group in self._groups
            ],
        }

    def release(self, group_id: str) -> None:
        self._groups = [group for group in self._groups if group.group_id != group_id]


class _RecoveryRolloutManager:
    def __init__(self, ledger: _PendingLedger) -> None:
        self.recovery_ledger = ledger
        self.recovered: list[tuple[str, int | None]] = []

    async def recover_group(self, group_id: str) -> bool:
        group = next(
            group
            for group in self.recovery_ledger.groups()
            if group.group_id == group_id
        )
        self.recovered.append((group.group_id, group.target_step))
        self.recovery_ledger.release(group_id)
        return True


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
        self, prompt: dict[str, Any], *, target_step: int | None = None
    ) -> str:
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

    def mark_prompt_group_admitted(self, group_id: str, *, target_step: int) -> None:
        self.recovery_ledger.assign_target_step(group_id, target_step)

    async def generate_and_push(
        self,
        prompt: dict[str, Any],
        *,
        target_step: int | None = None,
        inflight_registry: dict[str, Any] | None = None,
        recovery_group_id: str | None = None,
    ) -> RolloutOutcome:
        del inflight_registry
        if recovery_group_id is None:
            recovery_group_id = self.reserve_prompt_group(
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


def _enable_recovery_checkpoint_capture(controller: Any) -> None:
    """Install the narrow recovery hooks expected by the future foundation."""

    async def _inventory_is_valid(**_: Any) -> None:
        return None

    controller._validate_rollout_recovery_inventory = _inventory_is_valid
    controller._master_config.__dict__["token_capture"] = SimpleNamespace(
        enabled=True,
        staging_partition="rollout_staging",
    )


def test_dispatch_cursor_alone_assigns_the_next_batch_to_step_8() -> None:
    """The exact cursor is correct; it cannot recreate the missing step-7 batch."""

    async def exercise() -> int | None:
        sampler = _CountingInOrderSampler()
        sampler.restore_dispatch_index(7)
        return await sampler.admit(trainer_version_fn=lambda: 7)

    assert asyncio.run(exercise()) == 8


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The dataloader cursor can advance before unfinished prompt ownership "
        "is recorded in the checkpoint bundle."
    ),
)
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
        _enable_recovery_checkpoint_capture(controller)

        pump = asyncio.create_task(controller._rollout_pump())
        await asyncio.wait_for(sampler.admission_entered.wait(), timeout=1.0)
        assert controller._sampler.dispatch_index == 6

        try:
            await controller._save_checkpoint(
                {"loss": 1.0},
                is_policy_training_step=True,
            )
        finally:
            sampler.release_admission.set()
            await asyncio.wait_for(rollout_manager.started.wait(), timeout=1.0)
            rollout_manager.release.set()
            await asyncio.wait_for(pump, timeout=1.0)
            controller._checkpointer.shutdown()

        checkpoint = tmp_path / "checkpoints" / "step_7"
        recovery_state = torch.load(
            checkpoint / "rollout_recovery.pt",
            weights_only=False,
        )
        assert len(recovery_state["groups"]) == 1
        assert recovery_state["groups"][0]["target_step"] is None
        assert recovery_state["groups"][0]["prompt_payload"]["idx"] == 70
        assert torch.load(
            checkpoint / "train_dataloader.pt",
            weights_only=False,
        ) == {"next_batch": 8}

    asyncio.run(exercise())


@pytest.mark.xfail(
    strict=True,
    reason=(
        "An unfinished admitted batch is not yet persisted alongside the native "
        "TQ checkpoint."
    ),
)
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

        _enable_recovery_checkpoint_capture(controller)

        pump = asyncio.create_task(controller._rollout_pump())
        await asyncio.wait_for(rollout_manager.started.wait(), timeout=1.0)
        assert controller._sampler.dispatch_index == 7

        try:
            await controller._save_checkpoint(
                {"loss": 1.0},
                is_policy_training_step=True,
            )
        finally:
            rollout_manager.release.set()
            await asyncio.wait_for(pump, timeout=1.0)
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A commit that loses the checkpoint-barrier race is not yet retained in "
        "a durable unfinished-group ledger."
    ),
)
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
        _enable_recovery_checkpoint_capture(controller)

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
        canonical_ids = {
            group["group_id"] for group in replay_state["groups"]
        }
        pending_ids = {
            group["group_id"] for group in recovery_state["groups"]
        }

        assert int(group_id in canonical_ids) + int(group_id in pending_ids) == 1
        assert group_id not in canonical_ids
        assert group_id in pending_ids
        assert controller._buffer.ready_list == [True]

    asyncio.run(exercise())


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A canonical replay group is not yet filtered out of the checkpointed "
        "unfinished-group ledger."
    ),
)
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
        _enable_recovery_checkpoint_capture(controller)

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
        canonical_ids = {
            group["group_id"] for group in replay_state["groups"]
        }
        pending_ids = {
            group["group_id"] for group in recovery_state["groups"]
        }

        assert group_id in canonical_ids
        assert group_id not in pending_ids
        assert int(group_id in canonical_ids) + int(group_id in pending_ids) == 1

    asyncio.run(exercise())


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The controller does not yet persist and replay unfinished prompt-group "
        "ownership alongside the TQ checkpoint."
    ),
)
def test_recovery_replays_step_7_without_readmitting_the_batch() -> None:
    """An admitted batch keeps target_step=7 across a process restart."""

    async def exercise() -> None:
        sampler = _CountingInOrderSampler()
        sampler.restore_dispatch_index(7)
        ledger = _PendingLedger(
            _PendingGroup(
                group_id="batch-7-prompt-0",
                target_step=7,
                prompt_payload={"idx": 70, "message_log": []},
            )
        )
        rollout_manager = _RecoveryRolloutManager(ledger)

        controller_cls = SingleControllerActor.__ray_metadata__.modified_class
        controller = object.__new__(controller_cls)
        controller._sampler = sampler
        controller._rollout_manager = rollout_manager
        controller._data_plane_checkpoint_metadata = {
            "rollout_recovery_payload_sha256": "checkpoint-cut-digest"
        }
        controller._async_cfg = SimpleNamespace(max_buffered_rollouts=4)
        controller._buffer_capacity = asyncio.Semaphore(4)

        async def _inventory_is_valid(*, clear_unreferenced: bool) -> None:
            assert clear_unreferenced

        controller._validate_rollout_recovery_inventory = _inventory_is_valid

        await controller._maybe_restore_rollout_recovery(restored_replay_groups=0)

        assert ledger.prepare_calls == 1
        assert rollout_manager.recovered == [("batch-7-prompt-0", 7)]
        assert sampler.admit_calls == 0
        assert sampler.dispatch_index == 7

    asyncio.run(exercise())
