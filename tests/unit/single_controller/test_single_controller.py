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
import math
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from tensordict import TensorDict

import nemo_rl.algorithms.single_controller as single_controller
from nemo_rl.algorithms.grpo import GRPOConfig, _initial_grpo_save_state
from nemo_rl.algorithms.loss import ClippedPGLossConfig
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
        train_cluster=None,
        inference_cluster=None,
        save_state=_initial_grpo_save_state(),
        last_checkpoint_path=None,
    )
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class

    controller_cls(
        master_config=master_config,
        actor_args=actor_args,
    )

    logger.log_hyperparams.assert_called_once_with(master_config.model_dump())
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


class _AdvantageDataPlane:
    def __init__(self, data: TensorDict) -> None:
        self._data = data
        self.selected_fields: list[str] | None = None
        self.written_fields: TensorDict | None = None

    def get_samples(self, *, select_fields, **kwargs):
        del kwargs
        self.selected_fields = list(select_fields)
        return self._data

    def put_samples(self, *, fields, **kwargs) -> None:
        del kwargs
        self.written_fields = fields


class _MaskRecordingAdvantageEstimator:
    def __init__(self) -> None:
        self.mask: torch.Tensor | None = None

    def compute_advantage(self, *, rewards, mask, **kwargs) -> torch.Tensor:
        del kwargs
        self.mask = mask.clone()
        return rewards.unsqueeze(-1).expand_as(mask).clone()


def test_advantage_stage_applies_seq_logprob_error_mask_before_streaming_train(
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch_size, sequence_length = 4, 5
    generation_logprobs = torch.zeros(batch_size, sequence_length)
    # exp(abs(1 - 0)) > the configured threshold of 2, so only row 2
    # should be removed from the loss while the other rows remain trainable.
    generation_logprobs[2, 1:] = 1.0
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.zeros(
                batch_size, sequence_length, dtype=torch.long
            ),
            "total_reward": torch.tensor([0.0, 0.0, 1.0, 0.0]),
            "token_mask": torch.ones(batch_size, sequence_length),
            "sample_mask": torch.ones(batch_size),
            "prev_logprobs": torch.zeros(batch_size, sequence_length),
            "generation_logprobs": generation_logprobs,
        },
        batch_size=[batch_size],
    )
    data_plane = _AdvantageDataPlane(data)
    estimator = _MaskRecordingAdvantageEstimator()

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = data_plane
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = estimator
    ctrl._policy_logprobs_required = True
    ctrl._reference_logprobs_required = False
    ctrl._master_config = SimpleNamespace(
        grpo=SimpleNamespace(seq_logprob_error_threshold=2.0)
    )
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "sequence_lengths": [],
        "seq_logprob_error_metrics": [],
    }
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"sample-{i}" for i in range(batch_size)],
        fields=list(data.keys()),
    )

    result_meta = asyncio.run(ctrl._advantage_stage(meta))
    capsys.readouterr()

    assert data_plane.selected_fields is not None
    assert "prev_logprobs" in data_plane.selected_fields
    assert "generation_logprobs" in data_plane.selected_fields
    assert data_plane.written_fields is not None
    assert torch.equal(
        data_plane.written_fields["sample_mask"],
        torch.tensor([1.0, 1.0, 0.0, 1.0]),
    )
    assert estimator.mask is not None
    assert estimator.mask[2].count_nonzero() == 0
    assert estimator.mask[[0, 1, 3]].all()
    metrics = ctrl._step_log_dict["seq_logprob_error_metrics"]
    assert len(metrics) == 1
    assert metrics[0]["num_masked_seqs_by_logprob_error"] == 1
    assert metrics[0]["max_seq_mult_prob_error"] == pytest.approx(math.e)
    assert metrics[0]["max_seq_mult_prob_error_after_mask"] == pytest.approx(1.0)
    assert "advantages" in (result_meta.fields or [])


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
    def prepare_for_lp_inference(self, keep_train_buffers: bool = False) -> None:
        del keep_train_buffers

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
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            max_num_steps=1,
        ),
        # The pump's step epilogue reads the save triggers even when saving
        # is disabled.
        checkpointing={"enabled": False, "save_period": 10},
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


def _retry_controller(**async_cfg_kwargs):
    """A controller with just enough state for _rollout_with_retries."""
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._async_cfg = AsyncRLConfig(**async_cfg_kwargs)
    ctrl._dropped_rollouts = 0
    ctrl._consecutive_rollout_failures = 0
    ctrl._rollout_failure_counts = Counter()
    return ctrl


def test_rollout_retries_a_transient_failure_then_commits() -> None:
    # Gym surfaces HTTP 5xx and truncated bodies to us rather than retrying them
    # itself, and they are usually transient.
    calls = []

    async def generate_and_push(prompt, *, target_step):
        calls.append(prompt)
        if len(calls) == 1:
            raise RuntimeError("Server disconnected")

    ctrl = _retry_controller(rollout_retries=1)
    ctrl._rollout_manager = SimpleNamespace(generate_and_push=generate_and_push)

    assert asyncio.run(ctrl._rollout_with_retries({"idx": 0}, None)) is True
    assert len(calls) == 2
    assert ctrl._dropped_rollouts == 0
    # The streak resets on success, so scattered failures never accumulate into
    # a spurious abort over a long run.
    assert ctrl._consecutive_rollout_failures == 0


def test_rollout_drops_the_prompt_once_retries_are_exhausted() -> None:
    async def generate_and_push(prompt, *, target_step):
        raise RuntimeError("HTTP 500")

    ctrl = _retry_controller(rollout_retries=1)
    ctrl._rollout_manager = SimpleNamespace(generate_and_push=generate_and_push)

    # False, not an exception: escaping here would cancel every sibling rollout
    # in the pump's TaskGroup and tear down the run over one bad prompt.
    assert asyncio.run(ctrl._rollout_with_retries({"idx": 0}, None)) is False
    assert ctrl._dropped_rollouts == 1
    assert ctrl._rollout_failure_counts["RuntimeError"] == 2


def test_rollout_aborts_once_failures_are_consecutive_enough() -> None:
    async def generate_and_push(prompt, *, target_step):
        raise RuntimeError("HTTP 500")

    ctrl = _retry_controller(rollout_retries=0, max_consecutive_rollout_failures=2)
    ctrl._rollout_manager = SimpleNamespace(generate_and_push=generate_and_push)

    async def _drive():
        return [await ctrl._rollout_with_retries({"idx": i}, None) for i in range(2)]

    # A streak means the env servers or generation backend are down, and failing
    # fast beats burning the rest of the allocation.
    assert asyncio.run(_drive()) == [False, False]
    with pytest.raises(RuntimeError, match="3 consecutive rollouts failed"):
        asyncio.run(ctrl._rollout_with_retries({"idx": 2}, None))


def test_rollout_never_retries_cancellation() -> None:
    # Cancellation is how the TaskGroup and weight sync stop the pump; retrying
    # it would defeat both.
    calls = []

    async def generate_and_push(prompt, *, target_step):
        calls.append(prompt)
        raise asyncio.CancelledError()

    ctrl = _retry_controller(rollout_retries=5)
    ctrl._rollout_manager = SimpleNamespace(generate_and_push=generate_and_push)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ctrl._rollout_with_retries({"idx": 0}, None))
    assert len(calls) == 1
    assert ctrl._dropped_rollouts == 0
