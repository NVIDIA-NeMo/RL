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
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.dynamo.config import DynamoCfg
from nemo_rl.models.generation.dynamo import managed_runtime
from nemo_rl.models.generation.dynamo.managed_runtime import (
    FrontendHost,
    ManagedDynamoRuntime,
    _managed_namespace,
    _service_executable,
)

FRONTEND_URLS = ["http://10.0.0.2:3000/v1", "http://10.0.0.3:3000/v1"]


def _runtime(**frontend_args) -> ManagedDynamoRuntime:
    return ManagedDynamoRuntime(
        dynamo_cfg=DynamoCfg.model_validate(
            {"startup_timeout_s": 5, "frontend_args": frontend_args}
        ),
        model_name="model",
        frontend_python="/venvs/vllm/bin/python",
    )


class _FakeProcess:
    returncode = None

    @staticmethod
    def poll():
        return None


class _Remote:
    """A fake actor method: ``.remote()`` returns its value, or raises it."""

    def __init__(self, value=None):
        self._value = value

    def remote(self):
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


class _FakeFrontendHost:
    def __init__(self, returncode=None, stop_error=None):
        self.returncode = _Remote(returncode)
        self.stop = _Remote(stop_error)


@pytest.fixture
def fake_ray(monkeypatch):
    """Frontend hosts are plain objects; ``ray.get`` returns what they return."""
    killed = []
    monkeypatch.setattr(
        managed_runtime,
        "ray",
        SimpleNamespace(get=lambda refs: refs, kill=killed.append),
    )
    return killed


class _FakeHttpResponse:
    status = 200

    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self._payload).encode()


def _started(
    runtime: ManagedDynamoRuntime, *hosts: _FakeFrontendHost
) -> ManagedDynamoRuntime:
    """``runtime`` as started, with one frontend per host (default: one, alive)."""
    hosts = hosts or (_FakeFrontendHost(),)
    runtime._started = True
    runtime._host = "10.0.0.1"
    runtime._namespace = "nemo-rl-test"
    runtime._etcd_process = _FakeProcess()
    runtime._nats_process = _FakeProcess()
    runtime._frontend_hosts = list(hosts)
    runtime._frontend_urls = FRONTEND_URLS[: len(hosts)]
    return runtime


def test_dynamo_package_directory_does_not_shadow_stdlib_http() -> None:
    package_dir = (
        Path(__file__).resolve().parents[4]
        / "nemo_rl"
        / "models"
        / "generation"
        / "dynamo"
    )
    result = subprocess.run(
        [sys.executable, "-c", "import http.client"],
        cwd=package_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_runtime_construction_is_inert_and_namespace_is_driver_owned(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "Job/123.4")
    assert _managed_namespace() == "nemo-rl-job-123-4"
    runtime = _runtime()
    assert runtime._started is False
    assert runtime._etcd_process is None
    with pytest.raises(RuntimeError, match="not been started"):
        _ = runtime.frontend_urls
    with pytest.raises(RuntimeError, match="not been started"):
        _ = runtime.engine_env


def test_managed_service_and_frontend_environments_are_runtime_owned(
    monkeypatch,
) -> None:
    runtime = _runtime()
    runtime._manager_env = {
        "DYN_NAMESPACE": "managed",
        "DYN_DISCOVERY_BACKEND": "etcd",
        "ETCD_ENDPOINTS": "http://managed-etcd:2379",
        "NATS_SERVER": "nats://managed-nats:4222",
    }
    monkeypatch.setenv("DYN_NAMESPACE", "stale")
    monkeypatch.setenv("DYN_STALE_SETTING", "remove-me")
    monkeypatch.setenv("ETCD_STALE_SETTING", "remove-me")
    monkeypatch.setenv("NATS_STALE_SETTING", "remove-me")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")

    service_env = runtime._service_env()
    assert service_env["DYN_NAMESPACE"] == "managed"
    assert service_env["ETCD_ENDPOINTS"] == "http://managed-etcd:2379"
    assert service_env["NATS_SERVER"] == "nats://managed-nats:4222"
    assert "DYN_STALE_SETTING" not in service_env
    assert "ETCD_STALE_SETTING" not in service_env
    assert "NATS_STALE_SETTING" not in service_env
    assert service_env["NCCL_DEBUG"] == "INFO"

    runtime._started = True
    engine_env = runtime.engine_env
    assert engine_env["DYN_NAMESPACE"] == "managed"
    # Each engine serves its own model card, so a frontend on another node can
    # read model files that exist only on the engine's node.
    assert engine_env["DYN_SYSTEM_PORT"] == "0"
    assert "DYN_SYSTEM_PORT" not in service_env


def test_dynamo_services_resolve_from_bin_dir_then_path(monkeypatch, tmp_path) -> None:
    etcd = tmp_path / "etcd"
    etcd.write_text("#!/bin/sh\n")
    etcd.chmod(0o755)
    monkeypatch.setenv("NEMO_RL_DYNAMO_BIN_DIR", str(tmp_path))
    assert _service_executable("etcd") == str(etcd)
    with pytest.raises(FileNotFoundError, match="NEMO_RL_DYNAMO_BIN_DIR"):
        _service_executable("nats-server")

    monkeypatch.delenv("NEMO_RL_DYNAMO_BIN_DIR")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert _service_executable("etcd") == str(etcd)
    with pytest.raises(FileNotFoundError, match="on PATH"):
        _service_executable("nats-server")


def test_stop_process_escalates_its_process_group(monkeypatch) -> None:
    class EscalatingProcess:
        pid = 1234

        def __init__(self):
            self.wait_count = 0

        @staticmethod
        def poll():
            return None

        def wait(self, timeout):
            self.wait_count += 1
            if self.wait_count == 1:
                raise subprocess.TimeoutExpired("dynamo", timeout)
            return 0

    signals = []
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.managed_runtime.os.killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )
    process = EscalatingProcess()
    ManagedDynamoRuntime._stop_process(process, "frontend", timeout_s=0.01)
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]


def test_shutdown_guards_each_owned_resource_independently(
    monkeypatch, fake_ray
) -> None:
    failing = _FakeFrontendHost(stop_error=RuntimeError("frontend failure"))
    healthy = _FakeFrontendHost()
    runtime = _started(_runtime(), failing, healthy)
    monkeypatch.setattr(runtime, "_log_router_kv_events", lambda: None)
    runtime._nats_process = object()
    runtime._etcd_process = object()
    calls = []

    def stop_process(process, label, timeout_s=15):
        if process is None:
            return
        calls.append(label)

    monkeypatch.setattr(runtime, "_stop_process", stop_process)
    runtime.shutdown()
    runtime.shutdown()

    # The failing frontend does not keep the others, NATS or etcd running.
    assert fake_ray == [healthy]
    assert calls == ["NATS", "etcd"]
    assert runtime._frontend_hosts == [] and runtime._frontend_urls == []
    assert runtime._nats_process is None
    assert runtime._etcd_process is None


def test_every_frontend_must_route_to_the_engines_and_serve_the_model(
    monkeypatch, fake_ray
) -> None:
    runtime = _started(_runtime(), _FakeFrontendHost(), _FakeFrontendHost())
    health = {
        "instances": [
            {
                "namespace": "nemo-rl-test",
                "component": "backend",
                "endpoint": "generate",
                "instance_id": f"engine-{idx}",
            }
            for idx in range(2)
        ]
        # Another namespace's engine does not count toward this fleet.
        + [
            {
                "namespace": "other",
                "component": "backend",
                "endpoint": "generate",
                "instance_id": "foreign",
            }
        ]
    }
    responses = iter(
        [
            health,
            {"data": []},
            health,
            {"data": [{"id": "model"}]},
            health,
            {"data": [{"id": "model"}]},
        ]
    )
    urls: list[str] = []

    def fake_urlopen(url, timeout):
        urls.append(url)
        return _FakeHttpResponse(next(responses))

    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.managed_runtime.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.managed_runtime.time.sleep", lambda _: None
    )
    runtime.wait_for_engines(2, check_engines=lambda: None)
    first, second = (url.removesuffix("/v1") for url in FRONTEND_URLS)
    assert urls == [
        f"{first}/health?instances=true",
        f"{first}/v1/models",
    ] * 2 + [f"{second}/health?instances=true", f"{second}/v1/models"]


def _engine_stopped() -> None:
    raise RuntimeError("endpoint of generation shard 0 stopped serving")


@pytest.mark.parametrize(
    "frontend_code, check_engines, error",
    [
        (None, _engine_stopped, "shard 0 stopped serving"),
        (2, lambda: None, "frontend http://10.0.0.2:3000/v1 is not alive .code=2"),
    ],
)
def test_engine_wait_fails_immediately_when_an_engine_or_frontend_stops(
    fake_ray, frontend_code, check_engines, error
) -> None:
    runtime = _started(_runtime(), _FakeFrontendHost(returncode=frontend_code))

    with pytest.raises(RuntimeError, match=error):
        runtime.wait_for_engines(1, check_engines=check_engines)


def test_frontend_host_serves_on_a_free_port_of_its_node(monkeypatch) -> None:
    launched = []
    monkeypatch.setattr(
        managed_runtime.subprocess,
        "Popen",
        lambda command, **kwargs: launched.append((command, kwargs)) or _FakeProcess(),
    )
    monkeypatch.setattr(
        managed_runtime, "_get_free_port_local", lambda *args, **kwargs: 3000
    )
    monkeypatch.setattr(managed_runtime, "_get_node_ip_local", lambda: "10.0.0.2")
    monkeypatch.setenv("DYN_STALE_SETTING", "remove-me")
    host = FrontendHost()

    url = host.start(
        frontend_python="/venvs/vllm/bin/python",
        namespace="nemo-rl-test",
        dynamo_cfg=DynamoCfg(),
        env={"DYN_NAMESPACE": "nemo-rl-test"},
    )

    assert url == "http://10.0.0.2:3000/v1"
    ((command, kwargs),) = launched
    assert command[:3] == ["/venvs/vllm/bin/python", "-m", "dynamo.frontend"]
    assert command[command.index("--http-port") + 1] == "3000"
    assert kwargs["env"]["DYN_NAMESPACE"] == "nemo-rl-test"
    assert "DYN_STALE_SETTING" not in kwargs["env"]
    assert host.returncode() is None


def test_shutdown_logs_the_kv_events_the_router_applied(
    monkeypatch, capsys, fake_ray
) -> None:
    runtime = _started(_runtime())
    metrics = (
        "# HELP dynamo_component_kv_cache_events_applied KV events applied\n"
        'dynamo_component_kv_cache_events_applied{status="ok",worker_id="1"} 40\n'
        'dynamo_component_kv_cache_events_applied{status="ok",worker_id="2"} 2\n'
        'dynamo_component_kv_cache_events_applied{status="parent_block_not_found"} 3\n'
        'dynamo_component_requests_total{status="ok"} 9\n'
    )

    class _Metrics(_FakeHttpResponse):
        def read(self):
            return metrics.encode()

    urls = []
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout: urls.append(url) or _Metrics(None),
    )
    monkeypatch.setattr(runtime, "_stop_process", lambda *args, **kwargs: None)

    runtime.shutdown()

    assert urls == ["http://10.0.0.2:3000/metrics"]
    assert (
        '[Dynamo] router applied KV events: {"ok": 42.0, "parent_block_not_found": 3.0}'
        in capsys.readouterr().out
    )
