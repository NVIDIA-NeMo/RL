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
"""Streaming generation calls in their own Ray concurrency group (megatron_worker.GENERATION_CONCURRENCY_GROUP): with
more generation calls in flight than Ray's async default (1000), a control call still runs at once."""

import asyncio
import time

import pytest

ray = pytest.importorskip("ray")

IN_FLIGHT = 1100  # above Ray's async-actor default max_concurrency (1000)


class _PlainWorker:
    def __init__(self):
        self.release = asyncio.Event()

    async def generate_async(self, i):
        await self.release.wait()
        yield i

    def control(self):
        return "ok"


class _GroupedWorker(_PlainWorker):
    # A new function object: ray.method annotates the function it is given.
    async def generate_async(self, i):
        await self.release.wait()
        yield i


@pytest.fixture(scope="module")
def ray_local():
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, _temp_dir="/tmp/ray_gcg_test")
    yield
    ray.shutdown()


def _control_latency(actor_cls) -> float:
    actor = actor_cls.remote()
    gens = [actor.generate_async.options(num_returns="streaming").remote(i) for i in range(IN_FLIGHT)]
    time.sleep(5.0)  # let the calls reach the actor
    t0 = time.perf_counter()
    done, _ = ray.wait([actor.control.remote()], timeout=15.0)
    latency = time.perf_counter() - t0 if done else float("inf")
    ray.kill(actor)
    del gens
    return latency


def test_control_call_passes_with_a_generation_group(ray_local):
    from nemo_rl.models.generation.megatron.megatron_worker import (
        GENERATION_CONCURRENCY_GROUP,
        WORKER_CONCURRENCY_GROUPS,
    )

    ray.method(concurrency_group=GENERATION_CONCURRENCY_GROUP)(_GroupedWorker.generate_async)
    assert _control_latency(ray.remote(concurrency_groups=WORKER_CONCURRENCY_GROUPS)(_GroupedWorker)) < 5.0


def test_control_call_queues_without_it(ray_local):
    """Negative control: in one default group, the 1001st call (the control call) waits behind the generation calls
    (probe 8802004: blocked at 1100 and 2500 calls in flight, passes at 900)."""
    assert _control_latency(ray.remote(_PlainWorker)) == float("inf")


def test_megatron_workers_put_generate_async_in_the_generation_group():
    pytest.importorskip("megatron.core")
    from nemo_rl.models.generation.megatron.megatron_worker import (
        GENERATION_CONCURRENCY_GROUP,
        MegatronGenerationMixin,
    )
    from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorker

    assert MegatronPolicyWorker._default_options["concurrency_groups"][GENERATION_CONCURRENCY_GROUP] >= IN_FLIGHT
    assert MegatronGenerationMixin.generate_async.__ray_concurrency_group__ == GENERATION_CONCURRENCY_GROUP


def test_check_generation_calls():
    from nemo_rl.models.generation.megatron.megatron_worker import MAX_GENERATION_CALLS, check_generation_calls

    check_generation_calls(MAX_GENERATION_CALLS // 8, 8)
    with pytest.raises(ValueError, match="generation calls in flight"):
        check_generation_calls(MAX_GENERATION_CALLS // 8 + 1, 8)
