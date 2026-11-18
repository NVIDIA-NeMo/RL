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

pytestmark = pytest.mark.mcore

IN_FLIGHT_ROWS = 1500  # above Ray's async-actor default of 1000


def test_control_call_is_not_queued_behind_in_flight_generation():
    # Other backend lanes collect this module without the optional Bridge dependency.
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorker,
    )

    options = {
        k: v
        for k, v in MegatronPolicyWorker._default_options.items()
        if k == "max_concurrency"
    }

    @ray.remote(num_cpus=0, **options)
    class _Rank0:
        def __init__(self) -> None:
            self.released = asyncio.Event()
            self.all_started = asyncio.Event()
            self.started = 0

        async def generate_async(self) -> None:
            self.started += 1
            if self.started == IN_FLIGHT_ROWS:
                self.all_started.set()
            await self.released.wait()

        async def wait_for_all_started(self) -> None:
            await self.all_started.wait()

        async def finish_generation(self) -> bool:
            self.released.set()
            return True

    actor = _Rank0.remote()
    try:
        rows = [actor.generate_async.remote() for _ in range(IN_FLIGHT_ROWS)]
        # Async actor calls may start out of submission order.
        ray.get(actor.wait_for_all_started.remote(), timeout=60)
        assert ray.get(actor.finish_generation.remote(), timeout=60)
        ray.get(rows, timeout=60)
    finally:
        ray.kill(actor)
