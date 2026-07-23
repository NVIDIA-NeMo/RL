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

import pickle
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from nemo_rl.models.generation.sglang import fault_tolerance
from nemo_rl.models.generation.sglang.fault_tolerance import RolloutHealthMonitor
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.sglang.sglang_worker import (
    _is_absent_weight_update_group_response,
)
from nemo_rl.models.generation.sglang.utils import refit_deadline
from nemo_rl.models.generation.sglang.utils.ray_utils import _OwnerTokenLock
from nemo_rl.models.generation.sglang.utils.refit_deadline import (
    SGLangRefitDeadline,
    SGLangRefitTimeoutError,
)


class _RemoteMethod:
    def __init__(self, result=None):
        self.calls = []
        self.result = result if result is not None else object()

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class _FakeHealthMonitor:
    def __init__(self, *, pause_error=None):
        self.default_quiesce_timeout_s = 7.0
        self.pause_error = pause_error
        self.pause_timeouts = []
        self.resume_count = 0

    def pause(self, timeout_s=None):
        self.pause_timeouts.append(timeout_s)
        if self.pause_error is not None:
            raise self.pause_error

    def resume(self):
        self.resume_count += 1


def _minimal_generation(*, needs_offload=True, monitor=None):
    generation = SGLangGeneration.__new__(SGLangGeneration)
    generation.needs_offload = needs_offload
    generation.num_gpus_per_engine = 1
    generation.num_gpus_per_node = 1
    generation.all_engines = []
    generation._health_monitor = monitor
    generation._health_monitor_state_lock = threading.Lock()
    generation._health_monitor_suspensions = {}
    if needs_offload:
        generation._health_monitor_suspensions["offload"] = "offload"
    return generation


def _health_monitor_config():
    return {
        "sglang_cfg": {
            "rollout_health_check_interval": 1,
            "rollout_health_check_timeout": 1,
            "rollout_health_check_first_wait": 0,
        }
    }


def _started_monitor():
    generation = SimpleNamespace(
        all_engines=[object()],
        engines=[],
        nodes_per_engine=1,
    )
    monitor = RolloutHealthMonitor(generation, _health_monitor_config())
    monitor._stop_event = threading.Event()
    monitor._pause_event = threading.Event()
    monitor._is_checking_enabled = True
    return monitor


def test_refit_deadline_uses_one_decreasing_monotonic_budget(monkeypatch):
    clock = iter([10.0, 12.0, 15.5, 20.0])
    monkeypatch.setattr(refit_deadline.time, "monotonic", lambda: next(clock))

    deadline = SGLangRefitDeadline(10.0)

    assert deadline.remaining("first stage") == pytest.approx(8.0)
    assert deadline.remaining("second stage") == pytest.approx(4.5)
    with pytest.raises(SGLangRefitTimeoutError, match="final stage"):
        deadline.remaining("final stage")


@pytest.mark.parametrize(
    ("status_code", "payload", "expected"),
    [
        (
            400,
            b'{"success": false, "message": "The group to be destroyed does not exist."}',
            True,
        ),
        (
            500,
            b'{"success": false, "message": "The group to be destroyed does not exist."}',
            False,
        ),
        (400, b'{"success": false, "message": "NCCL teardown failed."}', False),
        (400, b"not-json", False),
    ],
)
def test_absent_weight_update_group_response(status_code, payload, expected):
    response = requests.Response()
    response.status_code = status_code
    response._content = payload

    assert _is_absent_weight_update_group_response(response) is expected


def test_owner_token_lock_delayed_release_cannot_unlock_another_owner():
    lock = _OwnerTokenLock()

    assert lock.acquire("first")
    assert lock.acquire("first")
    assert not lock.acquire("second")
    assert not lock.release("second")
    assert lock.release("first")
    assert lock.acquire("second")


def test_health_pause_drains_inflight_check_and_blocks_new_admission(monkeypatch):
    monitor = _started_monitor()
    rpc_entered = threading.Event()
    rpc_release = threading.Event()
    remote = _RemoteMethod()
    engine = SimpleNamespace(health_generate=remote)

    def blocking_get(_ref, *, timeout):
        assert timeout == 2
        rpc_entered.set()
        assert rpc_release.wait(timeout=1)

    monkeypatch.setattr(fault_tolerance.ray, "get", blocking_get)
    check_thread = threading.Thread(
        target=monitor._check_engine_health,
        args=(0, engine),
    )
    check_thread.start()
    assert rpc_entered.wait(timeout=1)

    pause_done = threading.Event()

    def pause():
        monitor.pause(timeout_s=1)
        pause_done.set()

    pause_thread = threading.Thread(target=pause)
    pause_thread.start()
    assert not pause_done.wait(timeout=0.05)
    rpc_release.set()
    check_thread.join(timeout=1)
    pause_thread.join(timeout=1)

    assert pause_done.is_set()
    assert monitor._checks_in_flight == 0
    monitor._check_engine_health(0, engine)
    assert len(remote.calls) == 1


def test_health_pause_timeout_stays_paused():
    monitor = _started_monitor()
    monitor._checks_in_flight = 1

    with pytest.raises(TimeoutError, match="remains paused"):
        monitor.pause(timeout_s=0.01)

    assert monitor._pause_event is not None
    assert monitor._pause_event.is_set()
    assert not monitor.is_checking_enabled()


def test_health_pause_drains_engine_kill_cleanup(monkeypatch):
    monitor = _started_monitor()
    kill_entered = threading.Event()
    kill_release = threading.Event()
    engine = SimpleNamespace(health_generate=_RemoteMethod())

    monkeypatch.setattr(
        fault_tolerance.ray,
        "get",
        Mock(side_effect=RuntimeError("health RPC failed")),
    )

    def blocking_kill(*, rollout_engine_id):
        assert rollout_engine_id == 0
        kill_entered.set()
        assert kill_release.wait(timeout=1)

    monkeypatch.setattr(monitor, "_kill_engine", blocking_kill)
    check_thread = threading.Thread(
        target=monitor._check_engine_health,
        args=(0, engine),
    )
    check_thread.start()
    assert kill_entered.wait(timeout=1)

    pause_done = threading.Event()
    pause_thread = threading.Thread(
        target=lambda: (monitor.pause(timeout_s=1), pause_done.set())
    )
    pause_thread.start()
    assert not pause_done.wait(timeout=0.05)
    kill_release.set()
    check_thread.join(timeout=1)
    pause_thread.join(timeout=1)

    assert pause_done.is_set()


def test_kv_onload_cannot_override_refit_suspension(monkeypatch):
    monitor = _FakeHealthMonitor()
    generation = _minimal_generation(monitor=monitor)
    engine = SimpleNamespace(resume_memory_occupation=_RemoteMethod())
    generation.all_engines = [engine]
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.ray.get",
        lambda _refs: None,
    )

    lease = generation.health_monitoring_suspend_for_refit()
    generation.prepare_for_generation(tags=["kv_cache"])

    assert "offload" not in generation._health_monitor_suspensions
    assert lease in generation._health_monitor_suspensions
    assert monitor.resume_count == 0

    generation.health_monitoring_release_refit(lease)
    assert monitor.resume_count == 1


def test_failed_refit_quiesce_is_persistently_latched(monkeypatch):
    monitor = _FakeHealthMonitor(pause_error=TimeoutError("busy health action"))
    generation = _minimal_generation(monitor=monitor)
    engine = SimpleNamespace(resume_memory_occupation=_RemoteMethod())
    generation.all_engines = [engine]
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.ray.get",
        lambda _refs: None,
    )

    with pytest.raises(TimeoutError, match="busy health action"):
        generation.health_monitoring_suspend_for_refit()
    generation.prepare_for_generation(tags=["kv_cache"])

    assert any(
        reason == "refit" for reason in generation._health_monitor_suspensions.values()
    )
    assert monitor.resume_count == 0


def test_failed_memory_release_keeps_offload_suspension(monkeypatch):
    monitor = _FakeHealthMonitor()
    generation = _minimal_generation(monitor=monitor)
    engine = SimpleNamespace(release_memory_occupation=_RemoteMethod())
    generation.all_engines = [engine]
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.ray.get",
        Mock(side_effect=RuntimeError("release failed")),
    )

    with pytest.raises(RuntimeError, match="release failed"):
        generation.finish_generation(tags=["weights"])

    assert generation._health_monitor_suspensions["offload"] == "offload"
    assert monitor.resume_count == 0
    assert monitor.pause_timeouts == [monitor.default_quiesce_timeout_s]


def test_pickle_drops_and_rebuilds_health_monitor_lock(monkeypatch):
    class _AsyncLoop:
        pass

    class _HttpClient:
        def __init__(self, cfg):
            self.cfg = cfg

    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.AsyncLoopThread",
        _AsyncLoop,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.HttpClient",
        _HttpClient,
    )
    generation = _minimal_generation(monitor=object())
    generation.sglang_cfg = {"sglang_cfg": {}}
    generation._async_loop = object()
    generation._http_client = object()

    clone = pickle.loads(pickle.dumps(generation))

    assert clone._health_monitor is None
    assert clone._health_monitor_state_lock.acquire(blocking=False)
    clone._health_monitor_state_lock.release()
    assert isinstance(clone._async_loop, _AsyncLoop)
    assert isinstance(clone._http_client, _HttpClient)
