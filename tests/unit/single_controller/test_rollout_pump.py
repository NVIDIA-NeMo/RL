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

"""Focused unit test for ``SingleControllerActor._rollout_pump``."""

from __future__ import annotations

import os
from typing import Any

import pytest
import ray

# Ray temp dir: must be SHORT on macOS (AF_UNIX path limit = 103 bytes).
_RAY_TEMP = "/tmp/nrl_rp_test"
os.makedirs(_RAY_TEMP, exist_ok=True)
os.environ.setdefault("RAY_TEMP_DIR", _RAY_TEMP)
os.environ.setdefault("RAY_TMPDIR", _RAY_TEMP)

from nemo_rl.algorithms.single_controller import (
    SingleControllerActor,
    SingleControllerConfig,
)
from nemo_rl.experience.rollout_manager import RolloutManager


@ray.remote(num_cpus=0)
class NoOpDataPlane:
    """DataPlane stub. ``_rollout_pump`` only forwards the handle."""

    def ping(self) -> bool:
        return True


@ray.remote(num_cpus=0)
class _CallCounter:
    """Ray actor exposing a call counter across the SC actor boundary."""

    def __init__(self) -> None:
        self._n = 0

    def tick(self) -> None:
        self._n += 1

    def get(self) -> int:
        return self._n


class _CountingRolloutManager(RolloutManager):
    """``RolloutManager`` subclass that tallies ``generate_and_push`` calls.

    Real ``RolloutManager.__init__`` runs (so the impl plumbing is exercised);
    only ``generate_and_push`` is instrumented to bump a Ray-actor counter
    that the test can read from the driver process.
    """

    def __init__(self, counter: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._counter = counter

    async def generate_and_push(self, input_sample: Any) -> None:
        await super().generate_and_push(input_sample)
        await self._counter.tick.remote()


@pytest.fixture(scope="module")
def ray_init():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)
    yield


def test_rollout_pump_dispatches_max_rollout_prompts(ray_init):
    """``_rollout_pump`` issues exactly ``max_rollout_prompts`` calls."""
    max_rollout_prompts = 4
    cfg = SingleControllerConfig(
        max_train_steps=1,
        max_rollout_prompts=max_rollout_prompts,
        min_prompt_groups_per_batch=1,
        generations_per_prompt=1,
        max_buffered_rollouts=max_rollout_prompts,
        max_inflight_prompts=max_rollout_prompts,
        max_weight_staleness_versions=0,
        advantage_enabled=False,
        diagnostics=False,
    )

    dp_client_handle = NoOpDataPlane.remote()
    counter = _CallCounter.remote()
    rollout_manager = _CountingRolloutManager(
        counter=counter,
        tokenizer=None,
        task_to_env={},
        num_generations_per_prompt=cfg.generations_per_prompt,
        max_seq_len=0,
        policy_generation=object(),
        dp_client=dp_client_handle,
    )
    ctrl = SingleControllerActor.remote(
        cfg=cfg,
        prompts=["only_prompt"],
        dp_client_handle=dp_client_handle,
        rollout_manager=rollout_manager,
        trainer_handle=object(),
        weight_synchronizer=object(),
    )

    ray.get(ctrl._rollout_pump.remote())

    assert ray.get(counter.get.remote()) == max_rollout_prompts
