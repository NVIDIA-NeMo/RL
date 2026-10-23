# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A slow following rollout must not strand completed, unsaved updates."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
from tests.unit.single_controller.test_checkpointing import (
    _ACTOR_CLS,
    _FakeGeneration,
    _FakeRolloutManager,
    _FakeSampler,
    _RefitRecordingTrainer,
    _actor_master_config,
    _make_actor_args,
    _step_dir_names,
    _training_info,
)


@pytest.mark.parametrize("blocking", [False, True])
@pytest.mark.parametrize("metric_name", [None, "train:loss"])
def test_idle_deadline_saves_completed_update_without_waiting_for_next_rollout(
    tmp_path, blocking, metric_name
):
    """Run one complete update, then make the next rollout unavailable forever."""

    async def run():
        events = []
        trainer = _RefitRecordingTrainer(events)
        trainer._step_metrics = {"loss": 0.25}
        trainer.begin_train_step = MagicMock(wraps=trainer.begin_train_step)
        trainer.finish_train_step = MagicMock(wraps=trainer.finish_train_step)
        manager = _FakeRolloutManager(events)
        actor = _ACTOR_CLS(
            _actor_master_config(
                tmp_path, max_num_steps=200, save_period=10, metric_name=metric_name
            ),
            _make_actor_args(
                trainer=trainer,
                gen=_FakeGeneration(events if blocking else None),
                rollout_manager=manager,
            ),
            SetupTimingMetrics(),
        )
        trainer.set_gate_probe(actor._rollout_permitted.is_set)

        class SlowNextRollout(_FakeSampler):
            deadline_due = False

            async def select(self, **kwargs):
                if self._step:
                    self.deadline_due = True
                    return None, 0
                return await super().select(**kwargs)

        sampler = SlowNextRollout()
        actor._sampler = sampler
        actor._timeout = SimpleNamespace(
            mark_iteration=lambda: None,
            would_save=lambda: sampler.deadline_due,
            check_save=lambda: sampler.deadline_due,
        )
        try:
            with patch("ray.cluster_resources", return_value={"GPU": 0}):
                await asyncio.wait_for(actor._train_pump(), timeout=2)
        finally:
            actor._checkpointer.shutdown()
        return actor, trainer, manager

    actor, trainer, manager = asyncio.run(run())
    assert actor._train_steps == actor._trainer_version == 1
    assert actor._algo_cfg.max_num_steps == 200
    trainer.begin_train_step.assert_called_once()
    trainer.finish_train_step.assert_called_once()
    assert _step_dir_names(tmp_path / "checkpoints") == {"step_1"}
    info = _training_info(tmp_path / "checkpoints", 1)
    assert info["current_step"] == info["trainer_version"] == 1
    assert info["consumed_samples"] == 2
    if metric_name:
        assert info[metric_name] == 0.25
    assert trainer.gate_open_during_save == [False]
    assert not actor._rollout_permitted.is_set()
    # Saving a committed update does not publish another generation version.
    assert manager.weight_versions == [1]
