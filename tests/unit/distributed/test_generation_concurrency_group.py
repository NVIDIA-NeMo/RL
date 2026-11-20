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
"""Check that a blocked Megatron stream leaves the actor available to control RPCs."""

import asyncio

import pytest

ray = pytest.importorskip("ray")


@pytest.mark.mcore
@pytest.mark.parametrize("grouped", [False, True])
def test_control_during_blocked_stream(init_ray_cluster, grouped: bool) -> None:
    from nemo_rl.models.generation.megatron.megatron_worker import (
        GENERATION_CONCURRENCY_GROUP,
        WORKER_CONCURRENCY_GROUPS,
    )

    class Worker:
        async def generate_async(self):
            yield "started"
            await asyncio.Event().wait()

        def control(self):
            return "ok"

    if grouped:
        ray.method(concurrency_group=GENERATION_CONCURRENCY_GROUP)(
            Worker.generate_async
        )
    # One blocked stream fills the default group without submitting 1000 requests.
    actor = ray.remote(
        max_concurrency=1,
        concurrency_groups=WORKER_CONCURRENCY_GROUPS if grouped else {},
    )(Worker).remote()
    try:
        stream = actor.generate_async.options(num_returns="streaming").remote()
        ready, _ = ray.wait([stream], timeout=30)
        assert ready and ray.get(next(stream)) == "started"
        control = actor.control.remote()
        if grouped:
            assert ray.get(control, timeout=10) == "ok"
        else:
            with pytest.raises(ray.exceptions.GetTimeoutError):
                ray.get(control, timeout=1)
    finally:
        ray.kill(actor)


@pytest.mark.mcore
def test_megatron_workers_put_generate_async_in_the_generation_group():
    from nemo_rl.models.generation.megatron.config import MAX_GENERATION_CALLS
    from nemo_rl.models.generation.megatron.megatron_worker import (
        GENERATION_CONCURRENCY_GROUP,
        MegatronGenerationMixin,
    )
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorker,
    )

    assert (
        MegatronPolicyWorker._default_options["concurrency_groups"][
            GENERATION_CONCURRENCY_GROUP
        ]
        == MAX_GENERATION_CALLS
    )
    assert (
        MegatronGenerationMixin.generate_async.__ray_concurrency_group__
        == GENERATION_CONCURRENCY_GROUP
    )


def test_check_generation_calls():
    from nemo_rl.models.generation.megatron.config import (
        MAX_GENERATION_CALLS,
        check_generation_calls,
    )

    check_generation_calls(MAX_GENERATION_CALLS // 8, 8)
    with pytest.raises(ValueError, match="generation calls in flight"):
        check_generation_calls(MAX_GENERATION_CALLS // 8 + 1, 8)
