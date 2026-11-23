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
"""Controller side of the trainer warm-up (async_rl.trainer_warmup)."""

import asyncio
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.utils.timer import Timer

_KWARGS = {
    "train_fields": ("input_ids", "sample_mask"),
    "run_logprobs": True,
    "seqlens": None,
}


def _controller(warmup_kwargs: Optional[dict[str, Any]]):
    """Bare actor carrying only what the warm-up start and wait read."""
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._trainer = MagicMock(name="trainer")
    ctrl._trainer.warmup_first_pass.return_value = {"total_s": 1.0}
    ctrl._loss_fn = MagicMock(name="loss_fn")
    ctrl._trainer_warmup_task = None
    ctrl._trainer_warmup_kwargs = warmup_kwargs
    ctrl._timer = Timer()
    return ctrl


async def _start_and_wait(ctrl) -> None:
    ctrl._maybe_start_trainer_warmup()
    await ctrl._finish_trainer_warmup()


def test_no_warmup_when_setup_hands_over_none():
    # Setup hands over None when the warm-up is off, skipped, or already ran.
    ctrl = _controller(None)

    asyncio.run(_start_and_wait(ctrl))

    ctrl._trainer.warmup_first_pass.assert_not_called()
    with pytest.raises(KeyError):
        ctrl._timer.get_elapsed("trainer_warmup_wait")


def test_the_warmup_runs_setup_arguments_and_its_wait_is_timed():
    ctrl = _controller(_KWARGS)

    asyncio.run(_start_and_wait(ctrl))

    ctrl._trainer.warmup_first_pass.assert_called_once_with(
        ctrl._loss_fn,
        train_fields=("input_ids", "sample_mask"),
        run_logprobs=True,
        seqlens=None,
    )
    assert ctrl._trainer_warmup_task is None
    assert len(ctrl._timer.get_elapsed("trainer_warmup_wait")) == 1


def test_a_failed_warmup_fails_at_the_first_trainer_call():
    ctrl = _controller(_KWARGS)
    ctrl._trainer.warmup_first_pass.side_effect = RuntimeError("warm-up OOM")

    with pytest.raises(RuntimeError, match="warm-up OOM"):
        asyncio.run(_start_and_wait(ctrl))
