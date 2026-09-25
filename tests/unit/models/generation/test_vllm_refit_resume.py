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

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import ray

from nemo_rl.models.generation.fleet_health import (
    FleetHealthPolicy,
    GenerationFleetExhausted,
    GenerationFleetHealth,
    ShardState,
)
from nemo_rl.models.generation.interfaces import (
    _warn_unsupported_in_flight_refit_pause_once,
)
from nemo_rl.models.generation.vllm import VllmGeneration


@pytest.fixture
def generation(monkeypatch: pytest.MonkeyPatch) -> VllmGeneration:
    gen = VllmGeneration.__new__(VllmGeneration)
    gen.cfg = {"vllm_cfg": {"async_engine": True}}
    gen.dp_size = 3
    gen.worker_group = SimpleNamespace(workers=[MagicMock() for _ in range(6)])
    gen._refit_membership = SimpleNamespace(
        shard_prefixes={0: 0, 2: 2}, workers_per_shard=2
    )
    gen.fleet_monitor = GenerationFleetHealth(
        shard_count=3, policy=FleetHealthPolicy(min_healthy_shards=1)
    )
    gen.fleet_monitor.record_actor_death(1)
    monkeypatch.setattr(
        ray, "wait", MagicMock(side_effect=lambda refs, **_: (refs, []))
    )
    return gen


@pytest.mark.parametrize("dead_shard", [0, 2])
def test_resume_records_death_and_waits_for_survivors(
    generation: VllmGeneration, monkeypatch: pytest.MonkeyPatch, dead_shard: int
) -> None:
    leaders = generation._refit_leader_workers()
    futures = [worker.resume_generation_async.remote.return_value for worker in leaders]
    dead_future = futures[dead_shard // 2]

    def result(ref: object) -> bool:
        # Even when the dead leader comes first, every RPC has settled already.
        ray.wait.assert_called_once_with(futures, num_returns=2, timeout=0.25)
        if ref is dead_future:
            raise ray.exceptions.ActorDiedError()
        return True

    ray_get = MagicMock(side_effect=result)
    monkeypatch.setattr(ray, "get", ray_get)

    assert generation.resume_generation_after_refit(timeout_s=0.25)

    assert ray_get.call_count == 2
    assert generation.fleet_monitor.state_of(dead_shard) is ShardState.DEAD
    assert generation.fleet_monitor.serving_shards() == [2 - dead_shard]


def test_late_resume_error_does_not_condemn_replacement(
    generation: VllmGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    monitor = generation.fleet_monitor

    def replace_actor(
        refs: list[object], **kwargs: object
    ) -> tuple[list[object], list]:
        monitor.record_actor_death(0)
        monitor.mark_restarting(0)
        generation.worker_group.workers[0] = MagicMock()
        monitor.mark_loaded(0)
        return refs, []

    monkeypatch.setattr(ray, "wait", MagicMock(side_effect=replace_actor))
    monkeypatch.setattr(
        ray, "get", MagicMock(side_effect=[ray.exceptions.ActorDiedError(), True])
    )

    assert generation.resume_generation_after_refit(timeout_s=0.25)
    assert monitor.state_of(0) is ShardState.STALE
    assert monitor.serving_shards() == [2]


def test_resume_fails_if_the_remaining_fleet_is_exhausted(
    generation: VllmGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ray, "get", MagicMock(side_effect=ray.exceptions.ActorDiedError())
    )
    with pytest.raises(GenerationFleetExhausted):
        generation.resume_generation_after_refit(timeout_s=0.25)
    assert generation.fleet_monitor.serving_shards() == []


def test_resume_does_not_suppress_death_without_fleet_health(
    generation: VllmGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation.fleet_monitor = None
    monkeypatch.setattr(
        ray, "get", MagicMock(side_effect=ray.exceptions.ActorDiedError())
    )
    with pytest.raises(ray.exceptions.ActorDiedError):
        generation.resume_generation_after_refit(timeout_s=0.25)


def test_resume_does_not_treat_unavailability_as_death(
    generation: VllmGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = ray.exceptions.ActorUnavailableError("temporarily unavailable", None)
    monkeypatch.setattr(ray, "get", MagicMock(side_effect=failure))
    with pytest.raises(ray.exceptions.ActorUnavailableError):
        generation.resume_generation_after_refit(timeout_s=0.25)
    assert generation.fleet_monitor.serving_shards() == [0, 2]


def test_controller_continues_after_a_confirmed_resume_death(
    generation: VllmGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.single_controller.test_refit_recovery import _make_controller

    ctrl, monitor, sync = _make_controller(None, shard_count=3, dead_shards=())
    monitor.record_actor_death(1)
    generation.fleet_monitor = monitor
    ctrl._gen.resume_generation_after_refit = generation.resume_generation_after_refit
    sync.sync_weights = MagicMock()
    monkeypatch.setattr(
        ray, "get", MagicMock(side_effect=[ray.exceptions.ActorDiedError(), True])
    )

    asyncio.run(ctrl._sync_weights())

    sync.sync_weights.assert_called_once()
    assert monitor.state_of(0) is ShardState.DEAD
    assert monitor.state_of(2) is ShardState.HEALTHY
    assert monitor.snapshot()[2].weight_version == ctrl._trainer_version
    assert ctrl._rollout_permitted.is_set()
    ctrl._rollout_manager.resume_request_deadlines.assert_called_once()


def test_sync_engine_uses_unsupported_pause_contract(
    generation: VllmGeneration, capsys: pytest.CaptureFixture[str]
) -> None:
    generation.cfg["vllm_cfg"]["async_engine"] = False
    _warn_unsupported_in_flight_refit_pause_once.cache_clear()
    try:
        assert generation.pause_generation_for_refit(clear_cache=False) is False
        assert generation.resume_generation_after_refit() is False
        assert (
            capsys.readouterr().out.count("no native generation pause/resume support")
            == 1
        )
        for worker in generation.worker_group.workers:
            worker.pause_generation_async.remote.assert_not_called()
            worker.resume_generation_async.remote.assert_not_called()
    finally:
        _warn_unsupported_in_flight_refit_pause_once.cache_clear()
