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
import sys
import threading
import time
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from nemo_rl.models.generation.sglang import fault_tolerance
from nemo_rl.models.generation.sglang import sglang_router
from nemo_rl.models.generation.sglang.fault_tolerance import RolloutHealthMonitor
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.sglang.sglang_worker import (
    SGLangGenerationWorker,
    _is_absent_weight_update_group_response,
    _is_missing_post_process_weights_response,
)
from nemo_rl.models.generation.sglang.utils import refit_deadline
from nemo_rl.models.generation.sglang.utils import startup_deadline
from nemo_rl.models.generation.sglang.utils.ip_port_utils import (
    _allocate_rollout_engine_addr_and_ports_normal,
)
from nemo_rl.models.generation.sglang.utils.ray_utils import _OwnerTokenLock
from nemo_rl.models.generation.sglang.utils.refit_deadline import (
    SGLangRefitDeadline,
    SGLangRefitTimeoutError,
)
from nemo_rl.models.generation.sglang.utils.startup_deadline import (
    SGLangStartupDeadline,
    SGLangStartupTimeoutError,
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
    generation._engine_cleanup_error = None
    generation.sglang_cfg = {
        "sglang_cfg": {
            "engine_startup_timeout_s": 1,
        }
    }
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


def test_startup_deadline_uses_one_decreasing_monotonic_budget(monkeypatch):
    clock = iter([10.0, 12.0, 15.5, 20.0])
    monkeypatch.setattr(startup_deadline.time, "monotonic", lambda: next(clock))

    deadline = SGLangStartupDeadline(10.0)

    assert deadline.remaining("router startup") == pytest.approx(8.0)
    assert deadline.remaining("engine startup") == pytest.approx(4.5)
    with pytest.raises(SGLangStartupTimeoutError, match="worker registration"):
        deadline.remaining("worker registration")


def test_router_startup_timeout_stops_and_kills_owned_actor(monkeypatch):
    init_ref = object()
    stop_ref = object()
    actor = SimpleNamespace(
        init=_RemoteMethod(init_ref),
        stop=_RemoteMethod(stop_ref),
    )
    actor_factory = SimpleNamespace(remote=Mock(return_value=actor))
    router_actor = SimpleNamespace(options=Mock(return_value=actor_factory))
    deadline = Mock()
    startup_error = SGLangStartupTimeoutError("router startup timed out")
    deadline.ray_get.side_effect = startup_error
    ray_get = Mock(return_value=None)
    ray_kill = Mock()

    monkeypatch.setattr(sglang_router, "RouterActor", router_actor)
    monkeypatch.setattr(sglang_router, "make_actor_runtime_env", Mock(return_value={}))
    monkeypatch.setattr(sglang_router.ray, "get", ray_get)
    monkeypatch.setattr(sglang_router.ray, "kill", ray_kill)

    with pytest.raises(SGLangStartupTimeoutError) as exc_info:
        sglang_router._start_router({}, deadline=deadline)

    assert exc_info.value is startup_error
    deadline.ray_get.assert_called_once_with(
        init_ref,
        stage="starting the rollout router",
        cancel_on_error=True,
    )
    assert actor.stop.calls == [((), {})]
    ray_get.assert_called_once_with(stop_ref, timeout=5)
    ray_kill.assert_called_once_with(actor, no_restart=True)


def test_engine_address_and_port_discovery_uses_supplied_deadline():
    engine = SimpleNamespace(
        _get_current_free_port=_RemoteMethod(3100),
        _get_current_node_ip=_RemoteMethod("127.0.0.1"),
    )
    deadline = Mock()
    deadline.ray_get.side_effect = lambda ref, **_kwargs: ref

    addr_and_ports, _ = _allocate_rollout_engine_addr_and_ports_normal(
        gpus_per_node=1,
        sglang_cfg={
            "sglang_cfg": {
                "dp_size": 1,
                "sglang_server_config": {
                    "num_gpus_per_engine": 1,
                },
            }
        },
        local_all_engines=[(0, engine)],
        deadline=deadline,
    )

    assert addr_and_ports[0] == {
        "host": "127.0.0.1",
        "port": 3100,
        "nccl_port": 3100,
        "dist_init_addr": "127.0.0.1:3100",
    }
    assert deadline.ray_get.call_count == 5
    assert all(
        call.kwargs["cancel_on_error"] is True
        for call in deadline.ray_get.call_args_list
    )


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


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(404, True), (400, False), (500, False)],
)
def test_missing_post_process_weights_response(status_code, expected):
    response = requests.Response()
    response.status_code = status_code

    assert _is_missing_post_process_weights_response(response) is expected


@pytest.mark.parametrize("status_code", [404, 400, 500])
def test_post_process_weights_skips_only_missing_endpoint(status_code):
    response = requests.Response()
    response.status_code = status_code
    error = requests.exceptions.HTTPError(response=response)
    worker_cls = SGLangGenerationWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker._make_request = Mock(side_effect=error)

    if status_code == 404:
        result = worker.post_process_weights(timeout_s=1)
        assert result["success"] is True
    else:
        with pytest.raises(requests.exceptions.HTTPError):
            worker.post_process_weights(timeout_s=1)


def test_worker_shutdown_kills_process_when_router_cleanup_fails(monkeypatch):
    kill_process_tree = Mock()
    sglang_module = ModuleType("sglang")
    srt_module = ModuleType("sglang.srt")
    utils_module = ModuleType("sglang.srt.utils")
    utils_module.kill_process_tree = kill_process_tree
    monkeypatch.setitem(sys.modules, "sglang", sglang_module)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils_module)
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.get",
        Mock(side_effect=requests.Timeout("router hung")),
    )

    worker_cls = SGLangGenerationWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker.node_rank = 0
    worker.server_host = "127.0.0.1"
    worker.server_port = 3000
    worker.server_base_url = "http://127.0.0.1:3000"
    worker.router_ip = "127.0.0.1"
    worker.router_port = 4000
    worker.process = SimpleNamespace(pid=123)
    worker._shutdown_requested = threading.Event()
    worker._process_lock = threading.Lock()
    worker._router_registration_lock = threading.Lock()

    assert worker.shutdown(timeout_s=0.01) is False
    assert kill_process_tree.call_args.args == (123,)
    assert 0 < kill_process_tree.call_args.kwargs["wait_timeout"] <= 0.01
    assert worker.process is None

    # Idempotence: a second cleanup cannot signal a reused PID.
    assert worker.shutdown(timeout_s=0.01) is False
    assert kill_process_tree.call_count == 1


@pytest.mark.parametrize(
    ("stuck_endpoint", "timeout_stage"),
    [
        ("health_generate", "generation health endpoint"),
        ("flush_cache", "initial cache flush"),
    ],
)
def test_worker_startup_http_requests_share_a_bounded_deadline(
    monkeypatch,
    stuck_endpoint,
    timeout_stage,
):
    request_timeouts = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, url, timeout, **_kwargs):
            request_timeouts.append((url, timeout))
            if stuck_endpoint in url:
                raise requests.Timeout(
                    "server accepted the connection but never replied"
                )
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.Session",
        _Session,
    )

    worker_cls = SGLangGenerationWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker._shutdown_requested = threading.Event()

    with pytest.raises(TimeoutError, match=timeout_stage):
        worker._wait_server_healthy(
            base_url="http://127.0.0.1:3000",
            api_key=None,
            process_alive_fn=lambda: True,
            startup_deadline=time.monotonic() + 0.05,
        )

    assert any(stuck_endpoint in url for url, _ in request_timeouts)
    assert all(0 < timeout <= 0.05 for _, timeout in request_timeouts)


def test_driver_startup_timeout_cancels_refs_and_shuts_down(monkeypatch):
    generation = _minimal_generation(needs_offload=False)
    generation.shutdown = Mock(return_value=False)
    init_handles = [object(), object()]
    driver_cancel_refs = Mock()
    worker_cancel_refs = Mock()
    deadline = SGLangStartupDeadline(0.25)

    monkeypatch.setattr(
        startup_deadline.ray,
        "get",
        Mock(side_effect=refit_deadline.ray.exceptions.GetTimeoutError()),
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.cancel_ray_refs",
        driver_cancel_refs,
    )
    monkeypatch.setattr(
        startup_deadline,
        "cancel_ray_refs",
        worker_cancel_refs,
    )

    with pytest.raises(SGLangStartupTimeoutError, match="0.250s") as exc_info:
        try:
            generation._wait_for_engine_startup(
                init_handles,
                deadline=deadline,
            )
        except BaseException as exc:
            generation._cleanup_failed_startup(init_handles, exc)
            raise

    worker_cancel_refs.assert_called_once_with(init_handles)
    driver_cancel_refs.assert_called_once_with(init_handles)
    generation.shutdown.assert_called_once_with()
    assert any(
        "terminate the enclosing Ray runtime" in note
        for note in exc_info.value.__notes__
    )


def test_startup_cleanup_exceptions_do_not_replace_original_error(monkeypatch):
    generation = _minimal_generation(needs_offload=False)
    generation.shutdown = Mock(side_effect=RuntimeError("shutdown failed"))
    startup_error = SGLangStartupTimeoutError("original startup timeout")
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.cancel_ray_refs",
        Mock(side_effect=RuntimeError("cancel failed")),
    )

    with pytest.raises(SGLangStartupTimeoutError) as exc_info:
        try:
            raise startup_error
        except BaseException as exc:
            generation._cleanup_failed_startup([object()], exc)
            raise

    assert exc_info.value is startup_error
    assert any(
        "terminate the enclosing Ray runtime" in note
        for note in startup_error.__notes__
    )
    generation.shutdown = Mock(return_value=True)


def test_worker_shutdown_during_startup_cannot_orphan_child(monkeypatch):
    health_entered = threading.Event()
    health_release = threading.Event()
    kill_process_tree = Mock()

    class _ServerArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.api_key = None

        def url(self):
            return "http://127.0.0.1:3000"

    class _Process:
        def __init__(self, *_args, **_kwargs):
            self.pid = None

        def start(self):
            self.pid = 456

        def is_alive(self):
            return True

    sglang_module = ModuleType("sglang")
    srt_module = ModuleType("sglang.srt")
    entrypoints_module = ModuleType("sglang.srt.entrypoints")
    http_server_module = ModuleType("sglang.srt.entrypoints.http_server")
    server_args_module = ModuleType("sglang.srt.server_args")
    utils_module = ModuleType("sglang.srt.utils")
    http_server_module.launch_server = Mock()
    server_args_module.ServerArgs = _ServerArgs
    utils_module.kill_process_tree = kill_process_tree
    monkeypatch.setitem(sys.modules, "sglang", sglang_module)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints", entrypoints_module)
    monkeypatch.setitem(
        sys.modules, "sglang.srt.entrypoints.http_server", http_server_module
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils_module)
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_worker.multiprocessing.Process",
        _Process,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_worker.multiprocessing.set_start_method",
        Mock(),
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.get",
        Mock(side_effect=requests.Timeout("router unavailable")),
    )

    worker_cls = SGLangGenerationWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker.node_rank = 0
    worker.server_host = "127.0.0.1"
    worker.server_port = 3000
    worker.server_base_url = "http://127.0.0.1:3000"
    worker.router_ip = "127.0.0.1"
    worker.router_port = 4000
    worker.process = None
    worker._shutdown_requested = threading.Event()
    worker._process_lock = threading.Lock()
    worker._router_registration_lock = threading.Lock()

    def wait_server_healthy(**_kwargs):
        # The PID must already be discoverable by the concurrent control
        # method before startup enters a potentially long health wait.
        assert worker.process is not None
        assert worker.process.pid == 456
        health_entered.set()
        assert health_release.wait(timeout=1)

    worker._wait_server_healthy = wait_server_healthy
    startup_errors = []

    def launch():
        try:
            worker._launch_server_process(
                {
                    "node_rank": 0,
                    "host": "127.0.0.1",
                    "port": 3000,
                },
                startup_timeout_s=1,
            )
        except BaseException as exc:
            startup_errors.append(exc)

    launch_thread = threading.Thread(target=launch)
    launch_thread.start()
    assert health_entered.wait(timeout=1)

    assert worker.shutdown(timeout_s=0.01) is False
    health_release.set()
    launch_thread.join(timeout=1)

    assert kill_process_tree.call_args.args == (456,)
    assert 0 < kill_process_tree.call_args.kwargs["wait_timeout"] <= 0.01
    assert worker.process is None
    assert worker._shutdown_requested.is_set()
    assert startup_errors
    assert "shutdown interrupted engine startup" in str(startup_errors[0])


def test_worker_shutdown_removes_concurrent_router_registration(monkeypatch):
    post_entered = threading.Event()
    post_release = threading.Event()
    kill_process_tree = Mock()

    class _ServerArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.api_key = None

        def url(self):
            return "http://127.0.0.1:3000"

    class _Process:
        def __init__(self, *_args, **_kwargs):
            self.pid = None

        def start(self):
            self.pid = 789

        def is_alive(self):
            return True

    sglang_module = ModuleType("sglang")
    srt_module = ModuleType("sglang.srt")
    entrypoints_module = ModuleType("sglang.srt.entrypoints")
    http_server_module = ModuleType("sglang.srt.entrypoints.http_server")
    server_args_module = ModuleType("sglang.srt.server_args")
    utils_module = ModuleType("sglang.srt.utils")
    http_server_module.launch_server = Mock()
    server_args_module.ServerArgs = _ServerArgs
    utils_module.kill_process_tree = kill_process_tree
    monkeypatch.setitem(sys.modules, "sglang", sglang_module)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints", entrypoints_module)
    monkeypatch.setitem(
        sys.modules, "sglang.srt.entrypoints.http_server", http_server_module
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils_module)
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_worker.multiprocessing.Process",
        _Process,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_worker.multiprocessing.set_start_method",
        Mock(),
    )

    post_response = Mock()
    post_response.raise_for_status = Mock()

    def register(*_args, **_kwargs):
        post_entered.set()
        assert post_release.wait(timeout=1)
        return post_response

    workers_response = Mock()
    workers_response.json.return_value = {
        "workers": [
            {
                "id": "worker-id",
                "url": "http://127.0.0.1:3000",
            }
        ]
    }
    delete_response = Mock()
    delete_response.raise_for_status = Mock()
    delete = Mock(return_value=delete_response)
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.post",
        register,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.get",
        Mock(return_value=workers_response),
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.delete",
        delete,
    )

    worker_cls = SGLangGenerationWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker.node_rank = 0
    worker.server_host = "127.0.0.1"
    worker.server_port = 3000
    worker.server_base_url = "http://127.0.0.1:3000"
    worker.router_ip = "127.0.0.1"
    worker.router_port = 4000
    worker.process = None
    worker._shutdown_requested = threading.Event()
    worker._process_lock = threading.Lock()
    worker._router_registration_lock = threading.Lock()
    worker._wait_server_healthy = Mock()

    startup_errors = []

    def launch():
        try:
            worker._launch_server_process(
                {
                    "node_rank": 0,
                    "host": "127.0.0.1",
                    "port": 3000,
                },
                startup_timeout_s=1,
            )
        except BaseException as exc:
            startup_errors.append(exc)

    launch_thread = threading.Thread(target=launch)
    launch_thread.start()
    assert post_entered.wait(timeout=1)

    shutdown_results = []
    shutdown_thread = threading.Thread(
        target=lambda: shutdown_results.append(worker.shutdown(timeout_s=1))
    )
    shutdown_thread.start()
    assert worker._shutdown_requested.wait(timeout=1)
    post_release.set()
    launch_thread.join(timeout=1)
    shutdown_thread.join(timeout=1)

    assert shutdown_results == [True]
    assert kill_process_tree.call_args.args == (789,)
    assert 0 < kill_process_tree.call_args.kwargs["wait_timeout"] <= 1
    delete.assert_called_once_with(
        "http://127.0.0.1:4000/workers/worker-id",
        timeout=pytest.approx(1, abs=0.1),
    )
    assert startup_errors
    assert "shutdown interrupted router registration" in str(startup_errors[0])


def test_recover_keyboard_interrupt_rolls_back_staged_actors(monkeypatch):
    generation = _minimal_generation(needs_offload=False)
    generation.all_engines = [None]
    engine = SimpleNamespace(shutdown=_RemoteMethod(True))

    def start_engines(_port_cursors, *, deadline):
        del deadline
        generation.all_engines[:] = [engine]
        generation.num_new_engines = 1
        return [object()], {}

    generation._start_engines = start_engines
    generation._wait_for_refs = Mock(side_effect=KeyboardInterrupt)
    ray_kill = Mock()
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.ray.get",
        lambda refs, *, timeout: [True] * len(refs),
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.ray.kill",
        ray_kill,
    )

    with pytest.raises(KeyboardInterrupt):
        generation._recover()

    assert generation.all_engines == [None]
    assert generation.num_new_engines == 0
    ray_kill.assert_called_once_with(engine, no_restart=True)


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


def test_unconfirmed_health_cleanup_permanently_disables_recovery(monkeypatch):
    generation = _minimal_generation(needs_offload=False)
    engine = SimpleNamespace(shutdown=_RemoteMethod(False))
    generation.all_engines = [engine]
    monitor = RolloutHealthMonitor(generation, _health_monitor_config())
    monkeypatch.setattr(
        fault_tolerance.ray,
        "get",
        lambda _ref, *, timeout: False,
    )
    monkeypatch.setattr(fault_tolerance.ray, "kill", Mock())

    monitor._kill_engine(rollout_engine_id=0)

    assert generation.all_engines == [None]
    assert generation._engine_cleanup_error is not None
    with pytest.raises(RuntimeError, match="automatic recovery is disabled"):
        generation._recover()


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


def test_quarantine_discards_every_physical_engine_rank(monkeypatch):
    generation = _minimal_generation(needs_offload=False)
    generation.num_gpus_per_engine = 2
    generation.num_gpus_per_node = 1
    engines = [SimpleNamespace(shutdown=_RemoteMethod(True)) for _ in range(4)]
    generation.all_engines = engines.copy()
    ray_kill = Mock()
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.ray.get",
        lambda refs, *, timeout: [True] * len(refs),
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.ray.kill",
        ray_kill,
    )

    assert generation.quarantine_all_engines(timeout_s=1)

    assert generation.all_engines == [None, None, None, None]
    assert generation.num_new_engines == 0
    for engine in engines:
        assert engine.shutdown.calls == [((), {"timeout_s": 1.0})]
        ray_kill.assert_any_call(engine, no_restart=True)


def test_unconfirmed_quarantine_latch_blocks_every_recovery_attempt(monkeypatch):
    generation = _minimal_generation(needs_offload=False)
    engine = SimpleNamespace(shutdown=_RemoteMethod(False))
    generation.all_engines = [engine]
    generation._start_engines = Mock()
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.ray.get",
        lambda _refs, *, timeout: [False],
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.ray.kill",
        Mock(),
    )

    assert not generation.quarantine_all_engines(timeout_s=1)
    assert generation.all_engines == [None]
    assert generation._engine_cleanup_error is not None

    for _ in range(2):
        with pytest.raises(RuntimeError, match="automatic recovery is disabled"):
            generation._recover()
    generation._start_engines.assert_not_called()


@pytest.mark.parametrize(
    "failing_stage",
    [
        "restarting failed rollout engines",
        "offloading recovered engine weights",
        "offloading recovered engine KV caches",
        "restoring recovered engine weights",
    ],
)
def test_recover_failure_rolls_back_and_retries(monkeypatch, failing_stage):
    generation = _minimal_generation(needs_offload=True)
    generation.all_engines = [None, None]
    created_engines = []

    def start_engines(_port_cursors, *, deadline):
        del deadline
        new_engines = [
            SimpleNamespace(
                shutdown=_RemoteMethod(True),
                release_memory_occupation=_RemoteMethod(True),
                resume_memory_occupation=_RemoteMethod(True),
            )
            for _ in range(2)
        ]
        created_engines.extend(new_engines)
        generation.all_engines[:] = new_engines
        generation.num_new_engines = len(new_engines)
        return [object(), object()], {}

    fail_enabled = True

    def wait_for_refs(_refs, *, deadline, stage, cancel_on_error=True):
        del deadline, cancel_on_error
        if fail_enabled and stage == failing_stage:
            raise RuntimeError(f"failed at {stage}")
        return None

    generation._start_engines = start_engines
    generation._wait_for_refs = wait_for_refs
    ray_kill = Mock()
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.ray.get",
        lambda refs, *, timeout: [True] * len(refs),
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.ray.kill",
        ray_kill,
    )

    with pytest.raises(RuntimeError, match="failed at"):
        generation._recover()

    assert generation.all_engines == [None, None]
    assert generation.num_new_engines == 0
    assert ray_kill.call_count == 2

    fail_enabled = False
    generation._recover()

    assert all(engine is not None for engine in generation.all_engines)
    assert len(created_engines) == 4
    assert generation.num_new_engines == 2


def test_recover_expands_missing_rank_to_complete_logical_engine(monkeypatch):
    generation = _minimal_generation(needs_offload=False)
    generation.num_gpus_per_engine = 2
    generation.num_gpus_per_node = 1
    old_peer = SimpleNamespace(shutdown=_RemoteMethod(True))
    untouched = [
        SimpleNamespace(shutdown=_RemoteMethod(True)),
        SimpleNamespace(shutdown=_RemoteMethod(True)),
    ]
    generation.all_engines = [old_peer, None, *untouched]
    created = []

    def start_engines(_port_cursors, *, deadline):
        del deadline
        for index, engine in enumerate(generation.all_engines):
            if engine is None:
                replacement = SimpleNamespace(shutdown=_RemoteMethod(True))
                generation.all_engines[index] = replacement
                created.append(replacement)
        generation.num_new_engines = len(created)
        return [object() for _ in created], {}

    generation._start_engines = start_engines
    generation._wait_for_refs = Mock(return_value=None)
    ray_kill = Mock()
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.ray.get",
        lambda refs, *, timeout: [True] * len(refs),
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.sglang.sglang_generation.ray.kill",
        ray_kill,
    )

    generation._recover()

    assert old_peer.shutdown.calls
    ray_kill.assert_called_once_with(old_peer, no_restart=True)
    assert len(created) == 2
    assert generation.all_engines[2:] == untouched
