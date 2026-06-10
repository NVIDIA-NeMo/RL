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

import sys
import types

import pytest

sys.modules.setdefault("decord", types.ModuleType("decord"))

import nemo_rl.models.generation.sglang.sglang_worker as sglang_worker


class _FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _FakeSession:
    def __init__(self, status_codes: list[int], requested_urls: list[str]):
        self.status_codes = status_codes
        self.requested_urls = requested_urls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def get(self, url, *, headers, timeout):
        self.requested_urls.append(url)
        status_code = self.status_codes.pop(0)
        return _FakeResponse(status_code)


class _FakeProcess:
    def __init__(self, target, args):
        self.target = target
        self.args = args
        self.pid = 12345
        self.started = False
        self.alive = True

    def start(self):
        self.started = True

    def is_alive(self):
        return self.alive


def _make_worker(sglang_cfg=None):
    worker_cls = sglang_worker.SGLangGenerationWorker.__ray_metadata__.modified_class
    worker = worker_cls.__new__(worker_cls)
    worker.global_rank = 0
    worker.base_url = "http://127.0.0.1:30000"
    worker.sglang_cfg = sglang_cfg or {}
    return worker


def _patch_launch_dependencies(monkeypatch, status_codes=None):
    requested_urls = []
    killed_pids = []
    processes = []

    def launch_server(*args, **kwargs):
        return None

    def kill_process_tree(pid):
        killed_pids.append(pid)

    def make_process(target, args):
        process = _FakeProcess(target, args)
        processes.append(process)
        return process

    def make_session():
        return _FakeSession(list(status_codes or [200]), requested_urls)

    monkeypatch.setattr(
        sglang_worker,
        "_require_sglang",
        lambda: (launch_server, object, kill_process_tree),
    )
    monkeypatch.setattr(sglang_worker.multiprocessing, "Process", make_process)
    monkeypatch.setattr(sglang_worker.requests, "Session", make_session)

    return requested_urls, killed_pids, processes


def test_sglang_startup_health_endpoint_defaults_to_liveness_probe():
    assert sglang_worker._normalize_startup_health_endpoint(None) == "health"
    assert sglang_worker._normalize_startup_health_endpoint("/health") == "health"


def test_sglang_startup_health_endpoint_rejects_unknown_endpoint():
    with pytest.raises(ValueError, match="startup_health_endpoint"):
        sglang_worker._normalize_startup_health_endpoint("ready")


def test_sglang_server_startup_timeout_validation():
    assert sglang_worker._normalize_server_startup_timeout(None) == 300.0
    assert sglang_worker._normalize_server_startup_timeout(12) == 12.0

    with pytest.raises(ValueError, match="server_startup_timeout"):
        sglang_worker._normalize_server_startup_timeout(0)


def test_sglang_launch_server_process_polls_liveness_health_by_default(monkeypatch):
    requested_urls, killed_pids, processes = _patch_launch_dependencies(monkeypatch)

    process = _make_worker()._launch_server_process(object())

    assert process is processes[0]
    assert process.started
    assert requested_urls == ["http://127.0.0.1:30000/health"]
    assert killed_pids == []


def test_sglang_launch_server_process_falls_back_for_old_health_api(monkeypatch):
    requested_urls, killed_pids, _processes = _patch_launch_dependencies(
        monkeypatch,
        status_codes=[404, 200],
    )

    _make_worker()._launch_server_process(object())

    assert requested_urls == [
        "http://127.0.0.1:30000/health",
        "http://127.0.0.1:30000/health_generate",
    ]
    assert killed_pids == []


def test_sglang_launch_server_process_can_use_generation_health(monkeypatch):
    requested_urls, _killed_pids, _processes = _patch_launch_dependencies(monkeypatch)
    worker = _make_worker({"startup_health_endpoint": "/health_generate"})

    worker._launch_server_process(object())

    assert requested_urls == ["http://127.0.0.1:30000/health_generate"]


def test_sglang_launch_server_process_kills_process_on_timeout(monkeypatch):
    _requested_urls, killed_pids, _processes = _patch_launch_dependencies(monkeypatch)
    times = iter([0.0, 2.0])
    monkeypatch.setattr(sglang_worker.time, "time", lambda: next(times))
    worker = _make_worker({"server_startup_timeout": 1})

    with pytest.raises(TimeoutError, match="polling /health"):
        worker._launch_server_process(object())

    assert killed_pids == [12345]
