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


@ray.remote(num_cpus=0)
class NoOpDataPlane:
    """DataPlane stub. ``_rollout_pump`` only forwards the handle."""

    def ping(self) -> bool:
        return True


@ray.remote(num_cpus=0)
class CountingGenWorker:
    """Counts the number of ``generate_and_push`` calls."""

    def __init__(self) -> None:
        self._call_count = 0

    async def generate_and_push(self, prompt: str, dp_client: Any) -> None:
        del prompt, dp_client
        self._call_count += 1

    def get_call_count(self) -> int:
        return self._call_count


@pytest.fixture(scope="module")
def ray_init():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4)
    yield


def test_rollout_pump_dispatches_max_rollout_prompts(ray_init):
    """`_rollout_pump` issues exactly `max_rollout_prompts` calls."""
    # Set up config
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

    # Set up DataPlane, GenWorker and SingleController
    dp = NoOpDataPlane.remote()
    gen = CountingGenWorker.remote()
    ctrl = SingleControllerActor.remote(
        cfg,
        ["only_prompt"],
        dp,
        gen,
        object(),
        object(),
    )

    # Run rollout pump
    ray.get(ctrl._rollout_pump.remote())

    # Check result
    assert ray.get(gen.get_call_count.remote()) == max_rollout_prompts
