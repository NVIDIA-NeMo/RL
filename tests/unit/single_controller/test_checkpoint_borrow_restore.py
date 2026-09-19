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

"""Round-trip a borrowed and repaid rollout through an SC checkpoint cut."""

from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.algorithms.async_utils.replay_buffer import (
    REPLAY_BUFFER_METADATA_FILENAME,
    DataPlaneCheckpointBarrier,
    TQReplayBuffer,
)
from nemo_rl.algorithms.async_utils.staleness_sampler import InOrderSampler
from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.single_controller import (
    DATA_PLANE_CHECKPOINT_DIR,
    SingleControllerActor,
)
from nemo_rl.algorithms.single_controller_utils.config import RolloutRecoveryConfig
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.failures import RolloutDataFailure
from nemo_rl.experience.interfaces import PromptGroupRecord
from nemo_rl.experience.rollout_manager import (
    RolloutManager,
    RolloutRetryPolicy,
    RolloutStats,
)
from nemo_rl.experience.rollout_recovery import (
    ROLLOUT_RECOVERY_STATE_FILENAME,
    PromptGroupPhase,
    RolloutRecoveryLedger,
)
from tests.unit.single_controller import _checkpoint_scenarios as scenarios

_GROUP_SIZE = 2
_PROMPTS_PER_STEP = 3
_CAPACITY = 64
_TIMEOUT_S = 20.0


class _Generation:
    """Finish immediately unless a prompt is deliberately held or failed."""

    def __init__(self) -> None:
        self.hold: dict[int, asyncio.Event] = {}
        self.fail: set[int] = set()

    async def run_rollout(self, input_sample: dict[str, Any]) -> PromptGroupRecord:
        prompt_idx = input_sample["idx"]
        gate = self.hold.get(prompt_idx)
        if gate is not None:
            await gate.wait()
        if prompt_idx in self.fail:
            raise RolloutDataFailure(f"prompt {prompt_idx} is bad on purpose")
        return PromptGroupRecord(
            prompt_idx=prompt_idx,
            prompt=[],
            extra_env_info=None,
            metadata={},
            completions=[],
            rollout_metrics={},
        )


class _Loader:
    def __init__(
        self,
        batches: list[BatchedDataDict],
        dataset: dict[int, dict[str, Any]],
    ) -> None:
        self._batches = batches
        self.dataset = dataset

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)

    def state_dict(self) -> dict[str, bool]:
        return {"fake": True}


class _TrackingRolloutManager(RolloutManager):
    """Associate stable recovery group IDs with their source prompt indexes."""

    def reserve_prompt_group(self, cut, input_sample, **kwargs) -> str:
        group_id = super().reserve_prompt_group(cut, input_sample, **kwargs)
        self.prompt_by_group_id[group_id] = int(input_sample["idx"])
        return group_id


def _prompt(prompt_idx: int) -> dict[str, Any]:
    return {
        "idx": prompt_idx,
        "message_log": [{"role": "user", "content": f"p{prompt_idx}"}],
    }


def _batch(prompt_indices: list[int]) -> BatchedDataDict:
    return BatchedDataDict(
        {
            "idx": prompt_indices,
            "message_log": [
                [{"role": "user", "content": f"p{prompt_idx}"}]
                for prompt_idx in prompt_indices
            ],
        }
    )


def _client(*, register: bool) -> NoOpDataPlaneClient:
    client = NoOpDataPlaneClient()
    if register:
        client.register_partition(
            partition_id=scenarios.PARTITION,
            fields=list(scenarios._FIELDS),
            num_samples=_CAPACITY * _GROUP_SIZE,
            consumer_tasks=["train"],
        )
    return client


def _manager(
    buffer: TQReplayBuffer,
    barrier: DataPlaneCheckpointBarrier,
    generation: _Generation,
) -> _TrackingRolloutManager:
    manager = object.__new__(_TrackingRolloutManager)
    manager._impl = generation
    manager._tokenizer = None
    manager._num_generations_per_prompt = _GROUP_SIZE
    manager._rollout_recovery_config = RolloutRecoveryConfig()
    manager._tq_buffer = buffer
    manager._recovery_ledger = RolloutRecoveryLedger()
    manager._data_plane_checkpoint_barrier = barrier
    manager._env_handles = {}
    manager._weight_version = 0
    manager._retry_policy = RolloutRetryPolicy(
        max_infra_attempts=1,
        max_data_attempts=1,
        max_gym_row_attempts=1,
        max_skipped_prompts=8,
    )
    manager._stats = RolloutStats()
    manager._canonical_groups_finalized = 0
    manager._canonical_output_tokens = 0
    manager._recovery_siblings_reused = 0
    manager._recovery_siblings_redispatched = 0
    manager._skipped_prompts = 0
    manager._consecutive_infra_drops = 0
    manager.prompt_by_group_id: dict[str, int] = {}
    return manager


def _controller(
    *,
    client: NoOpDataPlaneClient,
    loader: _Loader,
    generation: _Generation,
    dispatch_index: int | None = None,
    replacement_reserve: list[dict[str, Any]] | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> Any:
    barrier = DataPlaneCheckpointBarrier()
    buffer = TQReplayBuffer(
        client,
        partition_id=scenarios.PARTITION,
        pad_value_dict={"input_ids": 0},
        include_message_violation_fields=False,
        require_routed_experts=False,
    )
    buffer.set_data_plane_checkpoint_barrier(barrier)

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    controller = object.__new__(controller_cls)
    controller._data_plane_checkpoint_barrier = barrier
    controller._buffer = buffer
    controller._dp_client = client
    controller._partition_id = scenarios.PARTITION
    controller._rollout_manager = _manager(buffer, barrier, generation)
    controller._sampler = InOrderSampler(buffer, max_lookahead_versions=2)
    if dispatch_index is not None:
        controller._sampler.restore_dispatch_index(dispatch_index)
    elif checkpoint_path is not None:
        controller._sampler.set_dispatch_index(0)
    controller._async_cfg = SimpleNamespace(
        max_inflight_prompts=16,
        max_buffered_rollouts=_CAPACITY,
        diagnostics=False,
        sampler=SimpleNamespace(name="in_order"),
        rollout_failure=SimpleNamespace(
            on_dropped_prompt="replace",
            max_replacement_attempts=1,
            replacement_reserve_prompts=_PROMPTS_PER_STEP,
            min_step_batch_fraction=0.9,
        ),
    )
    controller._algo_cfg = GRPOConfig.model_construct(
        max_num_epochs=1,
        num_prompts_per_step=_PROMPTS_PER_STEP,
        num_generations_per_prompt=_GROUP_SIZE,
    )
    controller._master_config = SimpleNamespace(
        grpo=controller._algo_cfg,
        token_capture=SimpleNamespace(enabled=False),
    )
    controller._dataloader = loader
    controller._rollout_permitted = asyncio.Event()
    controller._rollout_permitted.set()
    controller._rollout_exhausted = asyncio.Event()
    controller._buffer_capacity = asyncio.Semaphore(_CAPACITY)
    controller._inflight_rollouts = 0
    controller._inflight_by_group_id = {}
    controller._dispatched_rollouts = set()
    controller._trainer_version = 0
    controller._train_steps = 0
    controller._current_epoch = 0
    controller._sampler_stamps_target_steps = False
    controller._rollout_recovery_enabled = True
    controller._batch_shortfall = {}
    controller._batch_replacements = {}
    controller._batch_promotions = {}
    controller._finalizer_actors = []
    controller._replacement_reserve = deque(replacement_reserve or [])
    controller._rollout_slot_waiters = 0
    controller._rollout_permitted_waiters = 0
    controller._buffer_capacity_waiters = 0
    controller._rollout_completion_durations_s = deque(maxlen=10_000)
    controller._rollout_queue_wait_durations_s = deque(maxlen=10_000)
    controller._logger = MagicMock()
    controller._last_checkpoint_path = (
        str(checkpoint_path) if checkpoint_path is not None else None
    )
    controller._data_plane_checkpoint_metadata = checkpoint_metadata
    return controller


async def _wait_for(predicate, description: str, pump: asyncio.Task[None]) -> None:
    deadline = asyncio.get_running_loop().time() + _TIMEOUT_S
    while not predicate():
        if pump.done():
            pump.result()
            raise AssertionError(f"rollout pump exited before {description}")
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError(f"timed out waiting for {description}")
        await asyncio.sleep(0.005)


def _stamps(
    buffer: TQReplayBuffer,
    prompt_by_group_id: dict[str, int],
) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for group_id, target_step in zip(buffer._group_ids, buffer.target_step_list):
        assert target_step is not None
        result.setdefault(target_step, []).append(prompt_by_group_id[group_id])
    return {
        target_step: sorted(prompt_indices)
        for target_step, prompt_indices in sorted(result.items())
    }


async def _exercise_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scenarios._rb, "record_to_train_batch", scenarios._stub_converter
    )

    dataset = {prompt_idx: _prompt(prompt_idx) for prompt_idx in range(15)}
    batches = [
        _batch([0, 1, 2]),
        _batch([3, 4, 5]),
        _batch([6, 7, 8]),
        _batch([9, 10, 11]),
        _batch([12, 13, 14]),
    ]
    generation = _Generation()
    generation.hold[1] = asyncio.Event()
    generation.fail.add(1)
    generation.hold[3] = asyncio.Event()

    first_client = _client(register=True)
    first = _controller(
        client=first_client,
        loader=_Loader(batches, dataset),
        generation=generation,
    )
    first_pump = asyncio.create_task(first._rollout_pump())
    first_ledger = first._rollout_manager.recovery_ledger
    await _wait_for(
        lambda: sum(first._buffer.ready_list) == 8
        and sum(
            group.phase is PromptGroupPhase.RESERVED for group in first_ledger.groups()
        )
        == 3,
        "lookahead work and the reserved batch",
        first_pump,
    )

    generation.hold[1].set()
    await _wait_for(
        lambda: first._batch_promotions == {0: 1}
        and any(
            first._rollout_manager.prompt_by_group_id.get(group_id) == 3
            for group_id in first._buffer._group_ids
        ),
        "the borrow and its repayment dispatch",
        first_pump,
    )

    checkpoint_path = tmp_path / "step_0"
    checkpoint_path.mkdir()
    async with first._data_plane_checkpoint_barrier.checkpoint() as cut:
        snapshot = await first._capture_rollout_checkpoint_cut(cut, checkpoint_path)
    assert snapshot.replay_metadata is not None
    assert snapshot.rollout_recovery_payload is not None
    torch.save(
        snapshot.replay_metadata,
        checkpoint_path / REPLAY_BUFFER_METADATA_FILENAME,
    )
    (checkpoint_path / ROLLOUT_RECOVERY_STATE_FILENAME).write_bytes(
        snapshot.rollout_recovery_payload
    )
    first_pump.cancel()
    await asyncio.gather(first_pump, return_exceptions=True)

    second_client = _client(register=False)
    checkpoint_metadata = second_client.load_checkpoint(
        checkpoint_path / DATA_PLANE_CHECKPOINT_DIR
    )
    restored = _controller(
        client=second_client,
        loader=_Loader([], dataset),
        generation=_Generation(),
        dispatch_index=snapshot.sampler_dispatch_index,
        replacement_reserve=snapshot.replacement_reserve,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
    )
    restored_group_count = await restored._maybe_restore_replay_buffer()
    await restored._maybe_restore_rollout_recovery(
        restored_replay_groups=restored_group_count
    )
    restored._validate_restored_sampler_cursor()
    assert restored._sampler.dispatch_index == snapshot.sampler_dispatch_index
    restored._rollout_manager.prompt_by_group_id.update(
        first._rollout_manager.prompt_by_group_id
    )

    restored_pump = asyncio.create_task(restored._rollout_pump())
    await _wait_for(
        lambda: 3 in restored._rollout_manager.prompt_by_group_id.values()
        and any(
            restored._rollout_manager.prompt_by_group_id.get(group_id) == 3
            for group_id in restored._buffer._group_ids
        ),
        "the restored repayment rollout",
        restored_pump,
    )
    restored._trainer_version = 1
    await asyncio.wait_for(restored_pump, timeout=_TIMEOUT_S)

    present = {
        prompt_idx
        for prompt_indices in _stamps(
            restored._buffer,
            restored._rollout_manager.prompt_by_group_id,
        ).values()
        for prompt_idx in prompt_indices
    }
    spares = {prompt["idx"] for prompt in restored._replacement_reserve}
    assert present | spares | {1} == set(range(15))

    recovery_metrics: dict[str, float] = {}
    for call in restored._logger.log_metrics.call_args_list:
        recovery_metrics.update(
            {
                key: value
                for key, value in call.args[0].items()
                if key
                in {
                    "groups_unfinished_found",
                    "siblings_reused",
                    "siblings_rerun",
                }
            }
        )
    assert recovery_metrics["groups_unfinished_found"] == 4.0
    assert recovery_metrics["siblings_reused"] == 0.0
    assert recovery_metrics["siblings_rerun"] == 8.0


def test_borrow_and_repayment_survive_controller_checkpoint_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A saved sampler cursor preserves every prompt after borrow and repayment."""
    asyncio.run(_exercise_round_trip(tmp_path, monkeypatch))
