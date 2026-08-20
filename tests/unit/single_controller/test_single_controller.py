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
from nemo_rl.algorithms.async_utils.staleness_sampler import BaseSampler
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
        rollout_manager=SimpleNamespace(_tq_buffer=None),
        env_handles={},
        fleet_monitor=None,
        generation_router=None,
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
        rollout_manager=SimpleNamespace(_tq_buffer=None),
        env_handles={},
        fleet_monitor=None,
        generation_router=None,
        train_cluster=None,
        inference_cluster=None,
        save_state=_initial_grpo_save_state(),
        last_checkpoint_path=None,
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
        rollout_manager=SimpleNamespace(_tq_buffer=None),
        train_cluster=None,
        inference_cluster=None,
        # A real field of SingleControllerActorArgs. Read directly rather than via a
        # getattr default, so omitting it breaks here instead of silently degrading
        # watchdog.gym_subprocess_check into a no-op at runtime.
        env_handles={},
        save_state=_initial_grpo_save_state(),
        last_checkpoint_path=None,
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
    # env={} -> should_use_nemo_gym is False, so _sync_weights takes the native
    # abort path (empty registry -> no-op) instead of the gym gate.
    ctrl._master_config = SimpleNamespace(env={})

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
    # env={} -> should_use_nemo_gym is False, so _sync_weights takes the native
    # abort path (empty registry -> no-op) instead of the gym gate.
    ctrl._master_config = SimpleNamespace(env={})
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


class _ChunkedSampler(_EmptySampler):
    """Assembles one step out of several single-group chunks, then goes empty.

    This is the shape the streaming path actually produces and the reason
    ``keep_train_buffers`` exists: every chunk after the first runs against an
    already-open train step.
    """

    def __init__(self, meta: KVBatchMeta, chunks: int) -> None:
        self._meta = meta
        self._remaining = chunks

    async def select(self, **kwargs):
        del kwargs
        if self._remaining == 0:
            return None, 0
        self._remaining -= 1
        return self._meta, 1


class _EmptyBuffer:
    def __len__(self) -> int:
        return 0


class _NoOpTrainer:
    def prepare_for_lp_inference(self, keep_train_buffers: bool = False) -> None:
        del keep_train_buffers

    def prepare_for_training(self) -> None:
        pass

    def begin_train_step(self, loss_fn) -> None:
        del loss_fn

    def train_microbatches_from_meta(self, meta: KVBatchMeta) -> None:
        del meta

    def finish_train_step(self) -> dict:
        return {}


class _LpRecordingTrainer(_NoOpTrainer):
    """Records the ``keep_train_buffers`` flag the pump passes on each chunk."""

    def __init__(self) -> None:
        self.keep_train_buffers_calls: list[bool] = []

    def prepare_for_lp_inference(self, keep_train_buffers: bool = False) -> None:
        self.keep_train_buffers_calls.append(keep_train_buffers)

    def get_logprobs_from_meta(self, meta: KVBatchMeta) -> None:
        del meta


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
    )
    ctrl._async_cfg = SimpleNamespace(
        min_groups_for_streaming_train=1,
        rollout_failure=SimpleNamespace(min_step_batch_fraction=0.9),
    )
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
    ctrl._trainer = _NoOpTrainer()
    ctrl._gen = SimpleNamespace(requires_kv_scale_sync=False)
    ctrl._loss_fn = None
    ctrl._dp_client = _NoOpDataPlane()
    ctrl._timer = Timer()
    ctrl._trainer_version = 0
    ctrl._train_steps = 0
    ctrl._batch_shortfall = {}
    ctrl._batch_replacements = {}
    ctrl._batch_promotions = {}
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


class _DroppingSampler(_OneThenEmptySampler):
    """Yields one group, then reports the second as never coming.

    Stands in for a prompt that was stamped for this step and then given up on: the
    credit lands while the pump is already waiting for the group, which is the only
    way it happens in a real run and the case that waits forever without the fix.

    ``select`` validates its bounds through the real sampler's own check, so a pump
    that asks for a non-positive batch fails here exactly as it would in production
    rather than being quietly tolerated by a permissive fake.
    """

    def __init__(self, meta: KVBatchMeta, *, credit_in_evict: bool) -> None:
        super().__init__(meta)
        self._credit_in_evict = credit_in_evict
        self.ctrl = None

    def _credit(self) -> None:
        self.ctrl._batch_shortfall[self.ctrl._trainer_version] = 1

    async def evict(self, *, current_train_weight: int) -> int:
        del current_train_weight
        # The window between the pump's two reads of the step target. A credit landing
        # here shrinks the target after the loop condition has already passed.
        if self._credit_in_evict and self._meta is None:
            self._credit()
        return 0

    async def select(self, **kwargs):
        BaseSampler._validate_group_bounds(
            kwargs["min_prompt_groups"], kwargs["max_prompt_groups"]
        )
        meta, num_groups = await super().select(**kwargs)
        if meta is None and not self._credit_in_evict:
            self._credit()
        return meta, num_groups


def _dropping_controller(*, credit_in_evict: bool):
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    sampler = _DroppingSampler(meta, credit_in_evict=credit_in_evict)
    ctrl = _train_pump_controller(sampler=sampler)
    sampler.ctrl = ctrl
    # ceil(0.9 * 2) is 2, so the harness's default floor forbids any short step at
    # all; at 0.5 the floor is 1 and closing on the one group it got is legal.
    ctrl._async_cfg.rollout_failure.min_step_batch_fraction = 0.5
    # Rollouts are still running, so the "rollout exhausted" escape is unavailable:
    # a pump that does not act on the credit waits on a group nobody is generating.
    ctrl._rollout_exhausted.clear()
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    return ctrl


@pytest.mark.parametrize("credit_in_evict", [False, True])
def test_train_pump_closes_a_step_short_when_a_stamped_prompt_is_dropped(
    monkeypatch, credit_in_evict
) -> None:
    """Both windows a shortfall can be credited in have to close the step.

    Credited before the loop condition is re-read, the target simply falls to what is
    already dispatched. Credited after it, between the two reads, the batch the pump
    would ask for is empty -- which the sampler rejects, so the pump has to notice
    instead of asking. Either way the step trains on what it got.
    """
    ctrl = _dropping_controller(credit_in_evict=credit_in_evict)
    ctrl._batch_replacements = {0: 1}
    ctrl._batch_promotions = {0: 2, 3: 1}
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert ctrl._train_steps == 1
    # One group, not the configured two: the step is what shrank, not the run.
    assert ctrl._consumed_samples == 1
    train_metrics = ctrl._logger.log_metrics.call_args_list[0].args[0]
    assert train_metrics["dropped_prompt_groups"] == 1
    assert train_metrics["replaced_prompt_groups"] == 1
    assert train_metrics["promoted_prompt_groups"] == 2
    # Read against version_during_step, not the already-incremented _trainer_version,
    # which would report 0 for every step forever.
    assert ctrl._trainer_version == 1
    # This step's counts are pruned; a later step's survive to be reported by it.
    assert ctrl._batch_shortfall == {}
    assert ctrl._batch_replacements == {}
    assert ctrl._batch_promotions == {3: 1}


def test_train_pump_prunes_stamps_older_than_the_step_that_just_closed(
    monkeypatch,
) -> None:
    """A straggler credited for a step that already closed must not outlive it.

    Popping only the current step's entry would leak those, and they are unreachable
    afterwards: the target is only ever read for the step being assembled.
    """
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 3}],
    )
    ctrl = _train_pump_controller(sampler=_OneThenEmptySampler(meta))
    ctrl._async_cfg.rollout_failure.min_step_batch_fraction = 0.5
    ctrl._trainer_version = 3
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    ctrl._batch_shortfall = {2: 1, 3: 1, 5: 1}
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert ctrl._batch_shortfall == {5: 1}


def test_train_pump_logs_nonzero_stale_group_metrics(monkeypatch) -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0", "sample-1"],
        fields=[],
        sequence_lengths=[1, 1],
        tags=[{"weight_version": 0}, {"weight_version": 0}],
    )
    ctrl = _train_pump_controller(sampler=_EvictingSampler(meta))
    ctrl._sync_weights = AsyncMock(return_value=1)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    ctrl._sync_weights.assert_awaited_once_with(calibration_data=None)
    train_metrics = ctrl._logger.log_metrics.call_args_list[0].args[0]
    assert train_metrics["evicted_stale_prompt_groups"] == 2
    assert train_metrics["aborted_stale_inflight_groups"] == 1


def test_train_pump_keeps_train_buffers_once_the_step_is_open(monkeypatch) -> None:
    """The logprob detour between chunks must not offload the trainer's grad
    buffers, because mcore's offload frees the gradients the earlier chunks of
    this step accumulated rather than copying them out.

    First chunk: no step open yet, nothing to preserve, so the offload is still
    worth taking. Every later chunk: step open, buffers must stay resident.
    """
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    # num_prompts_per_step is 2 in the harness, so two single-group chunks close
    # the step.
    ctrl = _train_pump_controller(sampler=_ChunkedSampler(meta, chunks=2))
    ctrl._policy_logprobs_required = True
    trainer = _LpRecordingTrainer()
    ctrl._trainer = trainer
    ctrl._sync_weights = AsyncMock(return_value=1)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert ctrl._train_steps == 1
    assert trainer.keep_train_buffers_calls == [False, True]
