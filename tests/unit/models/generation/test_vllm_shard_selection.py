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

"""Health-aware data-parallel shard selection on the native GRPO generation path.

`_async_generate_base` used to advance a counter modulo dp_size and send the request
wherever it landed, with no idea whether that shard was alive. A dead shard therefore
kept receiving a fixed share of traffic for the rest of the run.

Two properties are pinned here. Without a fleet selector attached the old round-robin is
reproduced exactly, so an unconfigured run is unchanged. With one attached, a quarantined
shard stops receiving work, and a Ray failure is both reported to the monitor and
re-raised as GenerationUnavailable so the rollout retry policy re-dispatches the prompt.
"""

import asyncio

import pytest
import ray.exceptions
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.failures import GenerationUnavailable, NoHealthyShards
from nemo_rl.models.generation.fleet_health import (
    FleetHealthPolicy,
    GenerationFleetMonitor,
    HealthyShardSelector,
    ShardState,
)
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration


class _WorkerGroup:
    """Minimal RayWorkerGroup stand-in that records where requests were sent."""

    def __init__(self, dp_size: int, fail_on_workers=()) -> None:
        self.dp_size = dp_size
        self.fail_on_workers = set(fail_on_workers)
        self.dispatched: list[int] = []

    def get_dp_leader_worker_idx(self, dp_shard_idx: int) -> int:
        # One leader per shard keeps shard index and worker index aligned, so the
        # assertions below can talk about either.
        return dp_shard_idx

    def run_single_worker_single_data(self, *, method_name, worker_idx, data, greedy):
        del method_name, data, greedy
        self.dispatched.append(worker_idx)
        if worker_idx in self.fail_on_workers:
            raise ray.exceptions.ActorDiedError()
        return _one_result(worker_idx)


async def _one_result(worker_idx: int):
    async def _ref():
        return (0, {"served_by": worker_idx})

    yield _ref()


def _make_generation(dp_size: int, fail_on_workers=()) -> VllmGeneration:
    """Build a VllmGeneration without firing its real __init__."""
    gen = object.__new__(VllmGeneration)
    gen.worker_group = _WorkerGroup(dp_size, fail_on_workers)
    gen.current_generate_dp_shard_idx = 0
    gen.fleet_monitor = None
    gen.fleet_selector = None
    gen.cfg = {"vllm_cfg": {"async_engine": True}}
    return gen


def _attach(gen: VllmGeneration, **policy_kwargs) -> GenerationFleetMonitor:
    monitor = GenerationFleetMonitor(
        shard_count=gen.worker_group.dp_size,
        policy=FleetHealthPolicy(**policy_kwargs),
    )
    gen.attach_fleet_health(monitor, HealthyShardSelector(monitor=monitor))
    return monitor


def _one_sample() -> BatchedDataDict:
    return BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "input_lengths": torch.tensor([2], dtype=torch.int32),
        }
    )


def _generate(gen: VllmGeneration):
    """Drive one generation through the real entry point, selection included."""

    async def _run():
        results = []
        async for item in gen._async_generate_base(
            _one_sample(),
            "generate_async",
            lambda _data: True,
        ):
            results.append(item)
        return results

    return asyncio.run(_run())


class TestWithoutFleetHealth:
    def test_selection_is_unchanged_round_robin(self):
        """An unconfigured run must behave exactly as it did before."""
        gen = _make_generation(dp_size=3)
        picked = [gen._next_dp_shard_idx() for _ in range(7)]
        assert picked == [0, 1, 2, 0, 1, 2, 0]

    def test_no_monitor_means_no_reporting(self):
        gen = _make_generation(dp_size=2, fail_on_workers={0})
        with pytest.raises(GenerationUnavailable):
            _generate(gen)
        # Nothing to report to; the failure still surfaces typed.
        assert gen.fleet_monitor is None


class TestWithFleetHealth:
    def test_a_quarantined_shard_stops_receiving_work(self):
        gen = _make_generation(dp_size=3)
        monitor = _attach(gen, unhealthy_threshold=1)
        monitor.record_probe(1, ok=False, error="dead")

        # Least-outstanding is deterministic when nothing is in flight, so what
        # matters is that the quarantined shard is never chosen.
        picked = {gen._next_dp_shard_idx() for _ in range(20)}
        assert 1 not in picked
        assert picked <= {0, 2}

    def test_a_ray_failure_is_reported_and_retyped(self):
        """Reporting is what stops the next request rediscovering the same corpse."""
        gen = _make_generation(dp_size=2, fail_on_workers={0})
        monitor = _attach(gen, unhealthy_threshold=1)

        with pytest.raises(GenerationUnavailable, match="shard 0"):
            _generate(gen)

        assert monitor.state_of(0) is ShardState.DEAD
        assert monitor.serving_shards() == [1]

    def test_the_failure_keeps_the_ray_error_as_its_cause(self):
        gen = _make_generation(dp_size=2, fail_on_workers={0})
        _attach(gen, unhealthy_threshold=1)
        with pytest.raises(GenerationUnavailable) as excinfo:
            _generate(gen)
        assert isinstance(excinfo.value.__cause__, ray.exceptions.RayError)

    def test_traffic_moves_to_the_survivor_after_a_failure(self):
        gen = _make_generation(dp_size=2, fail_on_workers={0})
        _attach(gen, unhealthy_threshold=1)

        with pytest.raises(GenerationUnavailable):
            _generate(gen)
        gen.worker_group.dispatched.clear()

        for _ in range(3):
            _generate(gen)
        assert gen.worker_group.dispatched == [1, 1, 1]

    def test_losing_every_shard_raises_a_retriable_failure(self):
        """NoHealthyShards is infra-classified, so the prompt is re-dispatched."""
        gen = _make_generation(dp_size=1, fail_on_workers={0})
        _attach(gen, unhealthy_threshold=1)

        with pytest.raises(GenerationUnavailable):
            _generate(gen)
        with pytest.raises(NoHealthyShards):
            gen._next_dp_shard_idx()

    def test_inflight_is_released_even_when_generation_fails(self):
        """A leaked in-flight count would permanently bias selection away from a shard."""
        gen = _make_generation(dp_size=2, fail_on_workers={0})
        _attach(gen, unhealthy_threshold=5)

        with pytest.raises(GenerationUnavailable):
            _generate(gen)
        assert gen.fleet_selector.inflight(0) == 0

    def test_attach_rejects_a_mismatched_shard_count(self):
        gen = _make_generation(dp_size=4)
        monitor = GenerationFleetMonitor(shard_count=2, policy=FleetHealthPolicy())
        with pytest.raises(ValueError, match="tracks 2 shards"):
            gen.attach_fleet_health(monitor, HealthyShardSelector(monitor=monitor))
