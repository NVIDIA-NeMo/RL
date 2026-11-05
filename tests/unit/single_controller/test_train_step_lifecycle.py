# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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


"""Cancellation must drain actual worker calls before discarding gradients."""

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import ray

from nemo_rl.algorithms import single_controller as implementation
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.models.policy.tq_policy import TQPolicy

_ACTOR_CLASS = SingleControllerActor.__ray_metadata__.modified_class
pytestmark = pytest.mark.timeout(15)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["begin", "dispatch", "finish"])
async def test_cancel_drains_rpc_before_cleanup(phase: str) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocked(*args: Any, **kwargs: Any) -> dict[str, Any]:
        entered.set()
        assert release.wait(timeout=5)
        return {}

    trainer = SimpleNamespace(
        begin_train_step=Mock(),
        train_microbatches_from_meta=Mock(),
        finish_train_step=Mock(return_value={}),
        abort_train_step=Mock(),
    )
    if phase == "begin":
        trainer.begin_train_step.side_effect = blocked
    elif phase == "dispatch":
        trainer.train_microbatches_from_meta.side_effect = lambda *args, **kwargs: (
            blocked()
        )
    else:
        trainer.finish_train_step.side_effect = blocked

    async def body(lifecycle: implementation._TrainPumpLifecycle) -> None:
        await lifecycle.begin(trainer, None)
        await lifecycle.dispatch(trainer, None, ())
        await lifecycle.finish(trainer)

    actor = SimpleNamespace(_trainer=trainer, _train_pump_body=body)
    task = asyncio.create_task(_ACTOR_CLASS._train_pump(actor))
    try:
        while not entered.is_set():
            if task.done():
                await task
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
        trainer.abort_train_step.assert_not_called()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    if phase == "finish":
        trainer.abort_train_step.assert_not_called()
        trainer.finish_train_step.assert_called_once()
    else:
        trainer.abort_train_step.assert_called_once()
        trainer.finish_train_step.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("commit_failure", [False, True])
async def test_failure_aborts_only_before_commit(commit_failure: bool) -> None:
    trainer = SimpleNamespace(
        begin_train_step=Mock(),
        train_microbatches_from_meta=Mock(),
        finish_train_step=Mock(),
        abort_train_step=Mock(),
    )
    if commit_failure:
        trainer.finish_train_step.side_effect = RuntimeError("commit uncertain")
    else:
        trainer.train_microbatches_from_meta.side_effect = RuntimeError(
            "dispatch failed"
        )

    async def body(lifecycle: implementation._TrainPumpLifecycle) -> None:
        await lifecycle.begin(trainer, None)
        await lifecycle.dispatch(trainer, None, ())
        await lifecycle.finish(trainer)

    actor = SimpleNamespace(_trainer=trainer, _train_pump_body=body)
    with pytest.raises(RuntimeError, match="uncertain|failed"):
        await _ACTOR_CLASS._train_pump(actor)
    assert trainer.abort_train_step.call_count == (0 if commit_failure else 1)


@pytest.mark.asyncio
async def test_cancel_unsettled_rpc_requires_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(implementation, "_TRAIN_PUMP_DRAIN_TIMEOUT_S", 0.02)
    entered = threading.Event()
    release = threading.Event()

    def begin(*args: Any) -> None:
        entered.set()
        assert release.wait(timeout=5)

    trainer = SimpleNamespace(begin_train_step=begin, abort_train_step=Mock())

    async def body(lifecycle: implementation._TrainPumpLifecycle) -> None:
        await lifecycle.begin(trainer, None)
        lifecycle.check_cancelled()

    actor = SimpleNamespace(_trainer=trainer, _train_pump_body=body)
    task = asyncio.create_task(_ACTOR_CLASS._train_pump(actor))
    try:
        while not entered.is_set():
            if task.done():
                await task
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(RuntimeError, match="recover workers"):
            await task
        trainer.abort_train_step.assert_not_called()
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_during_drain", [False, True])
async def test_cancel_again_or_failure_while_draining(fail_during_drain: bool) -> None:
    entered = threading.Event()
    release = threading.Event()

    def begin(*args: object) -> None:
        entered.set()
        assert release.wait(timeout=5)
        if fail_during_drain:
            raise ValueError("begin failed during drain")

    trainer = SimpleNamespace(begin_train_step=begin, abort_train_step=Mock())

    async def body(lifecycle: implementation._TrainPumpLifecycle) -> None:
        await lifecycle.begin(trainer, None)
        lifecycle.check_cancelled()

    actor = SimpleNamespace(_trainer=trainer, _train_pump_body=body)
    task = asyncio.create_task(_ACTOR_CLASS._train_pump(actor))
    try:
        while not entered.is_set():
            if task.done():
                await task
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0.02)
        if fail_during_drain:
            release.set()
            with pytest.raises(ValueError, match="during drain"):
                await task
            trainer.abort_train_step.assert_called_once()
        else:
            task.cancel()
            with pytest.raises(RuntimeError, match="drain was interrupted"):
                await task
            trainer.abort_train_step.assert_not_called()
    finally:
        release.set()


def test_automodel_selection_aligns_whole_prompt_groups() -> None:
    trainer = SimpleNamespace(sharding_annotations=Mock())
    trainer.sharding_annotations.get_axis_size.return_value = 16
    actor = SimpleNamespace(
        _streaming_ppo=False,
        _master_config=SimpleNamespace(
            policy={
                "automodel_cfg": {"enabled": True},
                "train_micro_batch_size": 1,
                "logprob_batch_size": 1,
            }
        ),
        _trainer=trainer,
        _async_cfg=SimpleNamespace(min_groups_for_streaming_train=4),
        _algo_cfg=SimpleNamespace(num_prompts_per_step=8, num_generations_per_prompt=4),
    )
    assert _ACTOR_CLASS._streaming_selection_kwargs(actor) == {"group_multiple": 4}
    actor._algo_cfg.num_prompts_per_step = 6
    with pytest.raises(ValueError, match="divisible by 4"):
        _ACTOR_CLASS._streaming_selection_kwargs(actor)


def test_tq_begin_drains_other_ranks_before_error() -> None:
    @ray.remote
    def rank_call(fail: bool) -> None:
        if fail:
            raise ValueError("rank failed")
        time.sleep(0.2)

    failed = rank_call.remote(True)
    pending = rank_call.remote(False)
    policy = object.__new__(TQPolicy)
    policy.cfg = {"train_global_batch_size": 4, "train_micro_batch_size": 1}
    policy.flops_tracker = None
    policy.worker_group = Mock()
    policy.worker_group.run_all_workers_single_data.return_value = [failed, pending]
    with pytest.raises(ray.exceptions.RayTaskError, match="rank failed"):
        policy.begin_train_step(None)
    ready, remaining = ray.wait([pending], timeout=0)
    assert ready == [pending] and not remaining
