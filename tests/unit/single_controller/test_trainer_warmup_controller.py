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
"""Controller side of the trainer warm-up (NRL_TRAINER_WARMUP=1)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.utils.timer import Timer


def _controller(
    *,
    megatron: bool = True,
    colocated: bool = False,
    is_ppo: bool = False,
    opd: bool = False,
    train_steps: int = 0,
    max_num_steps: int = 10,
    ran_in_setup: bool = False,
):
    """Bare actor carrying only what the warm-up start and wait read."""
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._master_config = SimpleNamespace(
        policy={
            "megatron_cfg": {"enabled": megatron},
            "generation": {"colocated": {"enabled": colocated}},
        }
    )
    ctrl._is_ppo = is_ppo
    ctrl._teacher_logprobs_required = opd
    ctrl._train_steps = train_steps
    ctrl._algo_cfg = SimpleNamespace(max_num_steps=max_num_steps)
    ctrl._trainer = MagicMock(name="trainer")
    ctrl._trainer.warmup_first_pass.return_value = {"total_s": 1.0}
    ctrl._loss_fn = MagicMock(name="loss_fn")
    ctrl._train_fields = ("input_ids", "sample_mask")
    ctrl._policy_logprobs_required = True
    ctrl._trainer_warmup_task = None
    ctrl._trainer_warmup_ran = ran_in_setup
    ctrl._timer = Timer()
    return ctrl


async def _start_and_wait(ctrl) -> None:
    ctrl._maybe_start_trainer_warmup()
    await ctrl._finish_trainer_warmup()


def test_the_warmup_is_off_by_default(monkeypatch):
    monkeypatch.delenv("NRL_TRAINER_WARMUP", raising=False)
    ctrl = _controller()

    asyncio.run(_start_and_wait(ctrl))

    ctrl._trainer.warmup_first_pass.assert_not_called()
    with pytest.raises(KeyError):
        ctrl._timer.get_elapsed("trainer_warmup_wait")


def test_the_warmup_runs_the_step_fields_and_its_wait_is_timed(monkeypatch):
    monkeypatch.setenv("NRL_TRAINER_WARMUP", "1")
    ctrl = _controller()

    asyncio.run(_start_and_wait(ctrl))

    ctrl._trainer.warmup_first_pass.assert_called_once_with(
        ctrl._loss_fn, train_fields=("input_ids", "sample_mask"), run_logprobs=True
    )
    assert ctrl._trainer_warmup_task is None
    assert len(ctrl._timer.get_elapsed("trainer_warmup_wait")) == 1


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"megatron": False}, "not Megatron"),
        ({"colocated": True}, "shares the trainer's GPUs"),
        ({"is_ppo": True}, "PPO"),
        ({"opd": True}, "distillation"),
        ({"train_steps": 10}, "no train step"),
    ],
)
def test_the_warmup_is_skipped_where_it_does_not_apply(
    monkeypatch, capsys, overrides, reason
):
    monkeypatch.setenv("NRL_TRAINER_WARMUP", "1")
    ctrl = _controller(**overrides)

    asyncio.run(_start_and_wait(ctrl))

    ctrl._trainer.warmup_first_pass.assert_not_called()
    assert reason in capsys.readouterr().out


def test_the_controller_leaves_a_warmup_setup_ran_alone(monkeypatch, capsys):
    monkeypatch.setenv("NRL_TRAINER_WARMUP", "1")
    ctrl = _controller(ran_in_setup=True)

    asyncio.run(_start_and_wait(ctrl))

    ctrl._trainer.warmup_first_pass.assert_not_called()
    assert "ignored" not in capsys.readouterr().out


def test_a_failed_warmup_fails_at_the_first_trainer_call(monkeypatch):
    monkeypatch.setenv("NRL_TRAINER_WARMUP", "1")
    ctrl = _controller()
    ctrl._trainer.warmup_first_pass.side_effect = RuntimeError("warm-up OOM")

    with pytest.raises(RuntimeError, match="warm-up OOM"):
        asyncio.run(_start_and_wait(ctrl))
