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

"""Tests for SingleController initialization and pump lifecycle."""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

import nemo_rl.algorithms.single_controller as single_controller
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig,
    AsyncRLConfig,
    MasterConfig,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.timer import Timer


class FakeWeightSynchronizer:
    pass


class FakeCheckpointDataPlane:
    def __init__(self) -> None:
        self.saves: list[dict[str, Any]] = []
        self.loads: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}

    def save_checkpoint(self, **kwargs: Any) -> None:
        self.saves.append(kwargs)
        self.metadata = dict(kwargs["metadata"])

    def load_checkpoint(self, **kwargs: Any) -> dict[str, Any]:
        self.loads.append(kwargs)
        return dict(self.metadata)


def _checkpoint_controller(
    dp_client: Any | None = None,
) -> tuple[Any, Any]:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    if dp_client is None:
        dp_client = FakeCheckpointDataPlane()
    ctrl._dp_client = dp_client
    ctrl._data_plane_checkpoint_lock = asyncio.Lock()
    ctrl._run_started = False
    ctrl._train_steps = 4
    ctrl._trainer_version = 5
    ctrl._current_epoch = 2
    ctrl._partition_id = "rollout-data"
    return ctrl, dp_client


def test_data_plane_checkpoint_hooks_forward_through_client(tmp_path) -> None:
    ctrl, dp_client = _checkpoint_controller()
    checkpoint_dir = str(tmp_path / "step-4")

    asyncio.run(
        ctrl.save_data_plane_checkpoint(
            checkpoint_dir,
            metadata={"run_id": "test-run"},
        )
    )
    restored_metadata = asyncio.run(
        ctrl.load_data_plane_checkpoint(checkpoint_dir)
    )

    assert dp_client.saves == [
        {
            "checkpoint_dir": checkpoint_dir,
            "metadata": {
                "run_id": "test-run",
                "data_plane_checkpoint_schema_version": 1,
                "single_controller_train_steps": 4,
                "single_controller_trainer_version": 5,
                "single_controller_epoch": 2,
            },
        }
    ]
    assert dp_client.loads == [{"checkpoint_dir": checkpoint_dir}]
    assert restored_metadata == dp_client.saves[0]["metadata"]


def test_data_plane_checkpoint_restore_rejected_after_run_starts(tmp_path) -> None:
    ctrl, dp_client = _checkpoint_controller()
    ctrl._run_started = True

    with pytest.raises(RuntimeError, match="before SingleControllerActor.run"):
        asyncio.run(
            ctrl.load_data_plane_checkpoint(
                str(tmp_path / "step-4"),
            )
        )
    assert dp_client.loads == []


def test_run_waits_for_inflight_data_plane_restore(tmp_path) -> None:
    class BlockingCheckpointDataPlane:
        def __init__(self) -> None:
            self.load_started = asyncio.Event()
            self.allow_load = asyncio.Event()

        async def load_checkpoint(self, checkpoint_dir) -> dict[str, Any]:
            self.load_started.set()
            await self.allow_load.wait()
            return {}

    async def exercise() -> None:
        dp_client = BlockingCheckpointDataPlane()
        ctrl, _ = _checkpoint_controller(dp_client)
        sync_started = asyncio.Event()
        block_sync = asyncio.Event()

        async def blocking_sync_weights() -> None:
            sync_started.set()
            await block_sync.wait()

        ctrl._sync_weights = blocking_sync_weights

        load_task = asyncio.create_task(
            ctrl.load_data_plane_checkpoint(str(tmp_path / "step-4"))
        )
        await dp_client.load_started.wait()

        run_task = asyncio.create_task(ctrl.run())
        await asyncio.sleep(0)
        assert not ctrl._run_started

        dp_client.allow_load.set()
        await load_task
        await sync_started.wait()
        assert ctrl._run_started

        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task

    asyncio.run(exercise())


def test_data_plane_checkpoint_saves_serialize_and_snapshot_inside_lock(
    tmp_path,
) -> None:
    class BlockingCheckpointDataPlane:
        def __init__(self) -> None:
            self.save_started = asyncio.Event()
            self.allow_first_save = asyncio.Event()
            self.active_saves = 0
            self.max_active_saves = 0
            self.saves: list[dict[str, Any]] = []

        async def save_checkpoint(self, **kwargs: Any) -> None:
            self.active_saves += 1
            self.max_active_saves = max(
                self.max_active_saves, self.active_saves
            )
            self.saves.append(kwargs)
            if len(self.saves) == 1:
                self.save_started.set()
                await self.allow_first_save.wait()
            self.active_saves -= 1

    async def exercise() -> None:
        dp_client = BlockingCheckpointDataPlane()
        ctrl, _ = _checkpoint_controller(dp_client)
        first = asyncio.create_task(
            ctrl.save_data_plane_checkpoint(str(tmp_path / "first"))
        )
        await dp_client.save_started.wait()

        second = asyncio.create_task(
            ctrl.save_data_plane_checkpoint(str(tmp_path / "second"))
        )
        await asyncio.sleep(0)
        ctrl._train_steps = 9
        ctrl._trainer_version = 10
        assert len(dp_client.saves) == 1

        dp_client.allow_first_save.set()
        await asyncio.gather(first, second)

        assert dp_client.max_active_saves == 1
        assert dp_client.saves[0]["metadata"][
            "single_controller_train_steps"
        ] == 4
        assert dp_client.saves[1]["metadata"][
            "single_controller_train_steps"
        ] == 9
        assert dp_client.saves[1]["metadata"][
            "single_controller_trainer_version"
        ] == 10

    asyncio.run(exercise())


def test_data_plane_clear_waits_for_checkpoint_save(tmp_path) -> None:
    class BlockingCheckpointDataPlane:
        def __init__(self) -> None:
            self.save_started = asyncio.Event()
            self.allow_save = asyncio.Event()
            self.clears: list[dict[str, Any]] = []

        async def save_checkpoint(self, **kwargs: Any) -> None:
            self.save_started.set()
            await self.allow_save.wait()

        def clear_samples(self, **kwargs: Any) -> None:
            self.clears.append(kwargs)

    async def exercise() -> None:
        dp_client = BlockingCheckpointDataPlane()
        ctrl, _ = _checkpoint_controller(dp_client)
        save = asyncio.create_task(
            ctrl.save_data_plane_checkpoint(str(tmp_path / "checkpoint"))
        )
        await dp_client.save_started.wait()

        clear = asyncio.create_task(
            ctrl._clear_data_plane_samples(["sample-1"])
        )
        await asyncio.sleep(0)
        assert dp_client.clears == []

        dp_client.allow_save.set()
        await asyncio.gather(save, clear)
        assert dp_client.clears == [
            {
                "sample_ids": ["sample-1"],
                "partition_id": "rollout-data",
            }
        ]

    asyncio.run(exercise())


def test_data_plane_checkpoint_save_failure_is_reported(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingCheckpointDataPlane:
        def save_checkpoint(self, **kwargs: Any) -> None:
            raise OSError("storage unavailable")

    ctrl, _ = _checkpoint_controller(FailingCheckpointDataPlane())
    checkpoint_dir = str(tmp_path / "failed")

    with pytest.raises(OSError, match="storage unavailable"):
        asyncio.run(ctrl.save_data_plane_checkpoint(checkpoint_dir))

    output = capsys.readouterr().out
    assert "data-plane checkpoint save failed" in output
    assert checkpoint_dir in output


def test_data_plane_checkpoint_load_failure_is_reported(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FailingCheckpointDataPlane:
        def load_checkpoint(self, **kwargs: Any) -> dict[str, Any]:
            raise FileNotFoundError("checkpoint missing")

    ctrl, _ = _checkpoint_controller(FailingCheckpointDataPlane())
    checkpoint_dir = str(tmp_path / "missing")

    with pytest.raises(FileNotFoundError, match="checkpoint missing"):
        asyncio.run(ctrl.load_data_plane_checkpoint(checkpoint_dir))

    output = capsys.readouterr().out
    assert "data-plane checkpoint load failed" in output
    assert checkpoint_dir in output


def test_rejects_multiple_optimizer_steps_per_rl_step(monkeypatch) -> None:
    monkeypatch.setattr(single_controller, "Logger", lambda _: object())
    master_config = MasterConfig.model_construct(
        policy={"train_global_batch_size": 4},
        grpo={
            "num_prompts_per_step": 2,
            "num_generations_per_prompt": 4,
        },
        async_rl=AsyncRLConfig(min_groups_for_streaming_train=1),
        logger={},
    )
    actor_args = SimpleNamespace(
        partition_id="rollout_data",
        dp_client=None,
        gen_handle=None,
        trainer_handle=None,
        dataloader=None,
        weight_synchronizer=None,
        advantage_estimator=None,
        loss_fn=None,
        tq_buffer=None,
        rollout_manager=SimpleNamespace(_tq_buffer=None),
        train_cluster=None,
        inference_cluster=None,
    )
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class

    with pytest.raises(
        ValueError,
        match=(
            r"num_prompts_per_step \* num_generations_per_prompt \(8\) "
            r"must equal policy.train_global_batch_size \(4\)"
        ),
    ):
        controller_cls(
            master_config=master_config,
            actor_args=actor_args,
        )


def test_logs_concrete_weight_synchronizer(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(single_controller, "Logger", lambda _: object())
    master_config = MasterConfig.model_construct(
        policy={"train_global_batch_size": 8},
        grpo={
            "num_prompts_per_step": 2,
            "num_generations_per_prompt": 4,
        },
        loss_fn=SimpleNamespace(force_on_policy_ratio=False),
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=1,
            max_buffered_rollouts=4,
        ),
        logger={},
    )
    actor_args = SimpleNamespace(
        partition_id="rollout_data",
        dp_client=None,
        gen_handle=None,
        trainer_handle=None,
        dataloader=None,
        weight_synchronizer=FakeWeightSynchronizer(),
        advantage_estimator=None,
        loss_fn=None,
        tq_buffer=None,
        rollout_manager=SimpleNamespace(_tq_buffer=None),
        train_cluster=None,
        inference_cluster=None,
    )
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class

    controller_cls(
        master_config=master_config,
        actor_args=actor_args,
    )

    output = capsys.readouterr().out
    assert "weight_sync=FakeWeightSynchronizer" in output
    assert "transport=stub" not in output


@pytest.mark.parametrize(
    ("recompute_kv_cache", "expected_invalidation_calls"),
    [(False, 0), (True, 1)],
)
def test_sync_weights_honors_recompute_kv_cache_config(
    recompute_kv_cache: bool,
    expected_invalidation_calls: int,
) -> None:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._async_cfg = AsyncRLConfig(
        recompute_kv_cache_after_weight_updates=recompute_kv_cache
    )
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._weight_synchronizer = SimpleNamespace(sync_weights=MagicMock())
    ctrl._gen = SimpleNamespace(
        invalidate_kv_cache=MagicMock(),
        requires_kv_scale_sync=False,
    )
    ctrl._rollout_manager = SimpleNamespace(set_weight_version=MagicMock())
    ctrl._trainer_version = 3

    asyncio.run(ctrl._sync_weights())

    ctrl._weight_synchronizer.sync_weights.assert_called_once_with(kv_scales=None)
    assert ctrl._gen.invalidate_kv_cache.call_count == expected_invalidation_calls
    ctrl._rollout_manager.set_weight_version.assert_called_once_with(3)
    assert ctrl._rollout_permitted.is_set()


def test_sync_weights_calibrates_and_forwards_fp8_kv_scales() -> None:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._async_cfg = AsyncRLConfig()
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._weight_synchronizer = SimpleNamespace(sync_weights=MagicMock())
    ctrl._gen = SimpleNamespace(
        invalidate_kv_cache=MagicMock(),
        requires_kv_scale_sync=True,
    )
    ctrl._trainer = SimpleNamespace(
        calibrate_qkv_fp8_scales=MagicMock(return_value={"layers": {"layer.0": 0.5}})
    )
    ctrl._rollout_manager = SimpleNamespace(set_weight_version=MagicMock())
    ctrl._trainer_version = 3
    calibration_data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "input_lengths": torch.tensor([2]),
        }
    )

    asyncio.run(ctrl._sync_weights(calibration_data=calibration_data))

    ctrl._trainer.calibrate_qkv_fp8_scales.assert_called_once_with(
        calibration_data,
        include_q=True,
    )
    ctrl._weight_synchronizer.sync_weights.assert_called_once_with(
        kv_scales={"layer.0": 0.5}
    )


class _EmptySampler:
    async def evict(self, *, current_train_weight: int) -> int:
        del current_train_weight
        return 0

    async def select(self, **kwargs):
        del kwargs
        return None, 0


class _OneThenEmptySampler(_EmptySampler):
    def __init__(self, meta: KVBatchMeta) -> None:
        self._meta: KVBatchMeta | None = meta

    async def select(self, **kwargs):
        del kwargs
        if self._meta is None:
            return None, 0
        meta = self._meta
        self._meta = None
        return meta, 1


class _EmptyBuffer:
    def __len__(self) -> int:
        return 0


class _NoOpTrainer:
    def prepare_for_lp_inference(self) -> None:
        pass

    def prepare_for_training(self) -> None:
        pass

    def begin_train_step(self, loss_fn) -> None:
        del loss_fn

    def train_microbatches_from_meta(self, meta: KVBatchMeta) -> None:
        del meta


class _NoOpDataPlane:
    def clear_samples(self, **kwargs) -> None:
        del kwargs


def _train_pump_controller(*, sampler) -> object:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._master_config = SimpleNamespace(
        grpo={
            "num_prompts_per_step": 2,
            "max_num_steps": 1,
        }
    )
    ctrl._async_cfg = SimpleNamespace(min_groups_for_streaming_train=1)
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._policy_logprobs_required = False
    ctrl._reference_logprobs_required = False
    ctrl._advantage_estimator = None
    ctrl._partition_id = "rollout_data"
    ctrl._sampler = sampler
    ctrl._buffer = _EmptyBuffer()
    ctrl._buffer_capacity = asyncio.Semaphore(2)
    ctrl._rollout_exhausted = asyncio.Event()
    ctrl._rollout_exhausted.set()
    ctrl._trainer = _NoOpTrainer()
    ctrl._gen = SimpleNamespace(requires_kv_scale_sync=False)
    ctrl._loss_fn = None
    ctrl._dp_client = _NoOpDataPlane()
    ctrl._timer = Timer()
    ctrl._trainer_version = 0
    ctrl._train_steps = 0
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "sequence_lengths": [],
    }
    return ctrl


def test_train_pump_stops_after_rollout_exhaustion_and_buffer_drain() -> None:
    ctrl = _train_pump_controller(sampler=_EmptySampler())

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert ctrl._train_steps == 0


def test_train_pump_fails_if_rollout_exhausts_during_partial_step() -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    ctrl = _train_pump_controller(sampler=_OneThenEmptySampler(meta))

    with pytest.raises(
        RuntimeError,
        match=(
            r"rollout exhausted before a complete training step was assembled: "
            r"dispatched 1/2 prompt groups"
        ),
    ):
        asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))
