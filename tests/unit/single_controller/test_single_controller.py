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
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

import nemo_rl.algorithms.single_controller as single_controller
from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneCheckpointBarrier
from nemo_rl.algorithms.grpo import GRPOConfig, _initial_grpo_save_state
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig,
    AsyncRLConfig,
    MasterConfig,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.timer import TimeoutChecker, Timer


class FakeWeightSynchronizer:
    pass


def _checkpointing_config(tmp_path) -> dict:
    """Minimal checkpointing block for actors built through __init__."""
    return {
        "enabled": False,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "metric_name": None,
        "higher_is_better": True,
        "keep_top_k": None,
        "save_period": 10,
        "save_optimizer": True,
        "checkpoint_must_save_by": None,
    }


def test_rejects_multiple_optimizer_steps_per_rl_step(monkeypatch) -> None:
    monkeypatch.setattr(single_controller, "Logger", lambda _: object())
    master_config = MasterConfig.model_construct(
        policy={"train_global_batch_size": 4},
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=4,
        ),
        async_rl=AsyncRLConfig(min_groups_for_streaming_train=1),
        logger={},
        env={},
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
        rollout_manager=SimpleNamespace(_tq_buffer=None, recovery_ledger=None),
        env_handles={},
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
            setup_timing_metrics=SetupTimingMetrics(),
        )


def test_logs_hyperparameters_and_concrete_weight_synchronizer(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    logger = MagicMock()
    monkeypatch.setattr(single_controller, "Logger", lambda _: logger)
    master_config = MasterConfig.model_construct(
        policy={"train_global_batch_size": 8},
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=4,
        ),
        loss_fn=ClippedPGLossConfig(force_on_policy_ratio=False),
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=1,
            max_buffered_rollouts=4,
        ),
        logger={},
        env={},
        # __init__ builds a CheckpointManager + TimeoutChecker from this block.
        checkpointing=_checkpointing_config(tmp_path),
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
        rollout_manager=SimpleNamespace(_tq_buffer=None, recovery_ledger=None),
        env_handles={},
        train_cluster=None,
        inference_cluster=None,
        save_state=_initial_grpo_save_state(),
        last_checkpoint_path=None,
        data_plane_checkpoint_metadata=None,
        finalizer_actors=[],
    )
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class

    controller_cls(
        master_config=master_config,
        actor_args=actor_args,
        setup_timing_metrics=SetupTimingMetrics(),
    )

    logger.log_hyperparams.assert_called_once_with(master_config.model_dump())
    output = capsys.readouterr().out
    assert "weight_sync=FakeWeightSynchronizer" in output
    assert "transport=stub" not in output


def test_logs_setup_timing_metrics(monkeypatch, tmp_path) -> None:
    """setup_timing_metrics is forwarded to Logger.log_metrics under timing/setup."""
    logger = MagicMock()
    monkeypatch.setattr(single_controller, "Logger", lambda _: logger)
    master_config = MasterConfig.model_construct(
        policy={"train_global_batch_size": 8},
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=4,
        ),
        loss_fn=ClippedPGLossConfig(force_on_policy_ratio=False),
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=1,
            max_buffered_rollouts=4,
        ),
        logger={},
        env={},
        # __init__ builds a CheckpointManager + TimeoutChecker from this block.
        checkpointing=_checkpointing_config(tmp_path),
    )
    setup_metrics = SetupTimingMetrics(
        generation_init_time_s=1.5, policy_init_time_s=2.5
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
        rollout_manager=SimpleNamespace(_tq_buffer=None, recovery_ledger=None),
        train_cluster=None,
        inference_cluster=None,
        # A real field of SingleControllerActorArgs. Read directly rather than via a
        # getattr default, so omitting it breaks here instead of silently degrading
        # watchdog.gym_subprocess_check into a no-op at runtime.
        env_handles={},
        save_state=_initial_grpo_save_state(),
        last_checkpoint_path=None,
        data_plane_checkpoint_metadata=None,
        finalizer_actors=[],
    )
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class

    controller_cls(
        master_config=master_config,
        actor_args=actor_args,
        setup_timing_metrics=setup_metrics,
    )

    logger.log_metrics.assert_called_once_with(
        setup_metrics.to_metrics_dict(), step=0, prefix="timing/setup"
    )


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
    ctrl._inflight_by_group_id = {}
    # env={} -> _should_use_nemo_gym is False, so _sync_weights takes the native
    # abort path (empty registry -> no-op) instead of the gym gate.
    ctrl._master_config = SimpleNamespace(
        env={}, token_capture=SimpleNamespace(enabled=False)
    )

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
    ctrl._inflight_by_group_id = {}
    # env={} -> _should_use_nemo_gym is False, so _sync_weights takes the native
    # abort path (empty registry -> no-op) instead of the gym gate.
    ctrl._master_config = SimpleNamespace(
        env={}, token_capture=SimpleNamespace(enabled=False)
    )
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


class _EvictingSampler(_OneThenEmptySampler):
    async def evict(self, *, current_train_weight: int) -> int:
        del current_train_weight
        return 2

    async def select(self, **kwargs):
        meta, num_groups = await super().select(**kwargs)
        return meta, 2 if num_groups else 0


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

    def finish_train_step(self) -> dict:
        return {}


class _NoOpDataPlane:
    def clear_samples(self, **kwargs) -> None:
        del kwargs


def _train_pump_controller(*, sampler) -> object:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._master_config = SimpleNamespace(
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            max_num_steps=1,
        ),
        # The pump's step epilogue reads the save triggers even when saving
        # is disabled.
        checkpointing={"enabled": False, "save_period": 10},
        token_capture=SimpleNamespace(enabled=False),
    )
    ctrl._async_cfg = SimpleNamespace(min_groups_for_streaming_train=1)
    ctrl._consumed_samples = 0
    ctrl._total_valid_tokens = 0
    ctrl._timeout = TimeoutChecker(timeout=None, fit_last_save_time=True)
    ctrl._timeout.start_iterations()
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
    ctrl._rollout_recovery_complete = asyncio.Event()
    ctrl._rollout_recovery_complete.set()
    ctrl._trainer = _NoOpTrainer()
    # Continuous-serving default; the pump asks before every step's trainer work.
    ctrl._gen = SimpleNamespace(
        requires_kv_scale_sync=False,
        blocks_training=lambda: False,
    )
    ctrl._loss_fn = None
    ctrl._dp_client = _NoOpDataPlane()
    ctrl._timer = Timer()
    ctrl._trainer_version = 0
    ctrl._train_steps = 0
    ctrl._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
    ctrl._rollout_recovery_ledger = None
    ctrl._finalizer_actors = []
    ctrl._available_finalizers = asyncio.Queue()
    ctrl._active_finalizers = 0
    ctrl._finalizer_waiters = 0
    ctrl._finalizer_unknown_outcomes = 0
    ctrl._finalizer_metrics_by_group = {}
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


def test_train_pump_waits_for_recovery_after_dataloader_exhaustion() -> None:
    async def _main() -> None:
        ctrl = _train_pump_controller(sampler=_EmptySampler())
        ctrl._rollout_recovery_complete.clear()

        train_task = asyncio.create_task(ctrl._train_pump())
        await asyncio.sleep(0.02)
        assert not train_task.done()

        ctrl._rollout_recovery_complete.set()
        await asyncio.wait_for(train_task, timeout=1.0)

    asyncio.run(_main())


def test_train_pump_fails_if_rollout_exhausts_during_partial_step() -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["g0_g0"],
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


def test_train_pump_logs_nonzero_stale_group_metrics(monkeypatch) -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["g0_g0", "g1_g0"],
        fields=[],
        sequence_lengths=[1, 1],
        tags=[{"weight_version": 0}, {"weight_version": 0}],
    )
    ctrl = _train_pump_controller(sampler=_EvictingSampler(meta))
    ctrl._sync_weights = AsyncMock(return_value=1)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    ctrl._sync_weights.assert_awaited_once_with(
        calibration_data=None, defer_engine_wake=False
    )
    train_metrics = ctrl._logger.log_metrics.call_args_list[0].args[0]
    assert train_metrics["evicted_stale_prompt_groups"] == 2
    assert train_metrics["aborted_stale_inflight_groups"] == 1


@pytest.mark.parametrize(
    "engine_blocks_training", [False, True], ids=["streaming", "blocking"]
)
def test_train_pump_chunked_step_by_engine_regime(
    monkeypatch, engine_blocks_training
) -> None:
    """One two-chunk step, observed under both engine regimes.

    Both regimes: the logprob detour between chunks must not offload the
    trainer's grad buffers — mcore's offload frees the gradients earlier
    chunks accumulated rather than copying them out. First chunk: no step
    open, the offload is worth taking; later chunks: buffers stay resident.

    Streaming: the engine is never stood down, the rollout gate stays open,
    and the configured streaming minimum reaches the sampler.

    Blocking (colocated Megatron): the pump closes the gate and sleeps the
    engine exactly once per step, before any trainer GPU work, and demands
    whole steps from the sampler (min == max). The chunked delivery here is
    a fake-permitted shape — real samplers honor the min — proving
    release-once is robust to partial deliveries.
    """
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    select_bounds: list[tuple[int, int]] = []

    class _BoundsRecordingSampler(_ChunkedSampler):
        async def select(self, **kwargs):
            select_bounds.append(
                (kwargs["min_prompt_groups"], kwargs["max_prompt_groups"])
            )
            return await super().select(**kwargs)

    # num_prompts_per_step is 2 in the harness, so two single-group chunks
    # close the step.
    ctrl = _train_pump_controller(sampler=_BoundsRecordingSampler(meta, chunks=2))
    ctrl._policy_logprobs_required = True
    calls: list[object] = []

    class _RecordingTrainer(_LpRecordingTrainer):
        def prepare_for_lp_inference(self, keep_train_buffers: bool = False) -> None:
            super().prepare_for_lp_inference(keep_train_buffers)
            calls.append("lp_inference_prep")

        def prepare_for_training(self) -> None:
            calls.append("prepare_for_training")

        def train_microbatches_from_meta(self, meta: KVBatchMeta) -> None:
            del meta
            calls.append("train")

    def _finish_generation() -> None:
        calls.append(("finish_generation", ctrl._rollout_permitted.is_set()))

    trainer = _RecordingTrainer()
    ctrl._trainer = trainer
    ctrl._gen = SimpleNamespace(
        requires_kv_scale_sync=False,
        blocks_training=lambda: engine_blocks_training,
        finish_generation=_finish_generation,
    )
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert ctrl._train_steps == 1
    assert trainer.keep_train_buffers_calls == [False, True]
    chunk = ["lp_inference_prep", "prepare_for_training", "train"]
    if engine_blocks_training:
        # Released exactly once, with the gate already closed, before the
        # trainer touched the GPUs; both chunks then ran without a second
        # release. The (mocked) post-step _sync_weights reopens the gate.
        assert calls == [("finish_generation", False)] + chunk * 2
        assert not ctrl._rollout_permitted.is_set()
        assert select_bounds and all(lo == hi for lo, hi in select_bounds)
    else:
        assert calls == chunk * 2
        assert ctrl._rollout_permitted.is_set()
        assert select_bounds[0] == (1, 2)
    ctrl._sync_weights.assert_awaited_once_with(
        calibration_data=None, defer_engine_wake=False
    )
