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

"""Rank 0's actor must serve more in-flight rows than Ray's default async concurrency.

Every rollout sample is one ``generate_async`` call on rank 0 of the Megatron
policy worker. Calls past the actor's ``max_concurrency`` wait in Ray's FIFO,
which caps the engine's in-flight rows and delays control calls such as
``finish_generation`` (which pauses the engines) until earlier rows finish.
"""

import asyncio

import pytest
import ray

from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorker

pytestmark = pytest.mark.mcore

IN_FLIGHT_ROWS = 1500  # above Ray's async-actor default of 1000


def test_control_call_is_not_queued_behind_in_flight_generation():
    options = {
        k: v
        for k, v in MegatronPolicyWorker._default_options.items()
        if k == "max_concurrency"
    }

    @ray.remote(num_cpus=0, **options)
    class _Rank0:
        def __init__(self):
            self.released = None

        def _event(self):
            if self.released is None:
                self.released = asyncio.Event()
            return self.released

        async def generate_async(self):
            await self._event().wait()

        async def finish_generation(self):
            self._event().set()
            return True

    actor = _Rank0.remote()
    try:
        rows = [actor.generate_async.remote() for _ in range(IN_FLIGHT_ROWS)]
        assert ray.get(actor.finish_generation.remote(), timeout=60)
        ray.get(rows, timeout=60)
    finally:
        ray.kill(actor)
