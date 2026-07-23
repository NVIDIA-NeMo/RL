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

import pytest

import nemo_rl.algorithms.single_controller as single_controller
from nemo_rl.algorithms.single_controller import (
    SingleControllerActor,
    SingleControllerConfig,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.utils.timer import Timer


def test_rejects_multiple_optimizer_steps_per_rl_step(monkeypatch) -> None:
    monkeypatch.setattr(single_controller, "Logger", lambda _: object())
    cfg = SingleControllerConfig.model_construct(
        min_groups_per_batch=1,
        target_groups_per_step=2,
        group_size=4,
        train_global_batch_size=4,
        advantage_enabled=False,
        logger={},
    )
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class

    with pytest.raises(
        ValueError,
        match=(
            r"train_global_batch_size \(4\) must equal "
            r"target_groups_per_step \(2\) \* group_size \(4\) = 8; "
            r"multiple optimizer steps per RL step are not supported"
        ),
    ):
        controller_cls(
            cfg=cfg,
            dp_client_handle=None,
            gen_handle=None,
            trainer_handle=None,
            weight_synchronizer=None,
            loss_fn=None,
            rollout_manager=SimpleNamespace(_tq_buffer=None),
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
    ctrl._cfg = SimpleNamespace(
        target_groups_per_step=2,
        min_groups_per_batch=1,
        max_train_steps=1,
        advantage_policy_logprobs_field=None,
        advantage_reference_logprobs_field=None,
        advantage_enabled=False,
        partition_id="rollout_data",
    )
    ctrl._sampler = sampler
    ctrl._buffer = _EmptyBuffer()
    ctrl._buffer_capacity = asyncio.Semaphore(2)
    ctrl._rollout_exhausted = asyncio.Event()
    ctrl._rollout_exhausted.set()
    ctrl._trainer = _NoOpTrainer()
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
