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
import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.dynamo import managed_runtime
from nemo_rl.models.generation.dynamo.config import DynamoCfg, DynamoFrontendArgs
from nemo_rl.models.generation.dynamo.managed_runtime import (
    FrontendHost,
    FrontendUrls,
    ManagedDynamoRuntime,
    _managed_namespace,
    _service_executable,
)

FRONTEND_URLS = ["http://10.0.0.2:3000/v1", "http://10.0.0.3:3000/v1"]


def _runtime(serves_chat: bool = False, **frontend_args) -> ManagedDynamoRuntime:
    return ManagedDynamoRuntime(
        dynamo_cfg=DynamoCfg(
            startup_timeout_s=5, frontend_args=DynamoFrontendArgs(**frontend_args)
        ),
        model_name="model",
        frontend_python="/venvs/vllm/bin/python",
        ray_address="10.0.0.1:6379",
        serves_chat=serves_chat,
    )


class _FakeProcess:
    returncode = None

    @staticmethod
    def poll():
        return None

    @staticmethod
    def stop():
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
    managed_runtime._stop_process(process, "frontend", timeout_s=0.01)
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]


def _exited(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return True
    return stat.rsplit(")", 1)[1].split()[0] == "Z"


def test_a_service_stops_when_the_process_that_started_it_is_sigkilled() -> None:
    # The service prints its own PID (exec keeps it) to the starter's stdout.
    starter_code = (
        "import os, time\n"
        "from nemo_rl.models.generation.dynamo.managed_runtime import _ServiceProcess\n"
        "_ServiceProcess(['sh', '-c', 'echo $$; exec sleep 600'],"
        " env=dict(os.environ), label='sleep')\n"
        "time.sleep(600)\n"
    )
    with subprocess.Popen(
        [sys.executable, "-c", starter_code],
        cwd=Path(__file__).resolve().parents[4],
        stdout=subprocess.PIPE,
        text=True,
    ) as starter:
        assert starter.stdout is not None
        line = starter.stdout.readline()
        assert line, "the starter exited before it started the service"
        service_pid = int(line)
        # SIGKILL runs no shutdown or atexit handler, and the service's own
        # session keeps it out of the starter's process group.
        starter.send_signal(signal.SIGKILL)
    deadline = time.monotonic() + 20
    try:
        while not _exited(service_pid):
            assert time.monotonic() < deadline, "the service outlived its starter"
            time.sleep(0.1)
    finally:
        if not _exited(service_pid):
            os.killpg(service_pid, signal.SIGKILL)


def test_stopping_a_service_also_stops_its_watchdog() -> None:
    service = managed_runtime._ServiceProcess(
        ["sleep", "600"], env=dict(os.environ), label="sleep"
    )
    service.stop()
    assert service.poll() == -signal.SIGTERM
    assert service._watchdog.returncode is not None


def test_shutdown_guards_each_owned_resource_independently(
    monkeypatch, fake_ray
) -> None:
    failing = _FakeFrontendHost(stop_error=RuntimeError("frontend failure"))
    healthy = _FakeFrontendHost()
    runtime = _started(_runtime(), failing, healthy)
    monkeypatch.setattr(runtime, "_log_router_kv_events", lambda: None)
    stopped = []

    class _Service(_FakeProcess):
        def __init__(self, label):
            self.label = label

        def stop(self):
            stopped.append(self.label)

    runtime._nats_process = _Service("NATS")
    runtime._etcd_process = _Service("etcd")
    runtime.shutdown()
    runtime.shutdown()

    # The failing frontend does not keep the others, NATS or etcd running.
    assert fake_ray == [healthy]
    assert stopped == ["NATS", "etcd"]
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


class _FakeFrontendHostActor:
    """Records where each frontend is placed and what it is started with."""

    def __init__(self):
        self.node_ids: list[str] = []
        self.envs: list[dict[str, str]] = []
        self.serve_chat: list[bool] = []
        self.captures: list[dict] = []
        self.chat_settings: list[dict] = []

    def options(self, scheduling_strategy):
        self.node_ids.append(scheduling_strategy.node_id)
        return self

    def remote(self):
        actor = self

        class _Host:
            class start:
                @staticmethod
                def remote(**kwargs):
                    actor.envs.append(kwargs["env"])
                    actor.serve_chat.append(kwargs["serves_chat"])
                    host = f"frontend-{len(actor.envs)}"
                    return FrontendUrls(
                        base_url=f"http://{host}:3000/v1",
                        generation_cut_control_url=f"http://{host}:2400",
                    )

            class publish_chat_capture:
                @staticmethod
                def remote(capture):
                    actor.captures.append(capture)

            class publish_chat_settings:
                @staticmethod
                def remote(settings):
                    actor.chat_settings.append(settings)

        return _Host()


def test_frontends_spread_over_the_nodes_with_the_extra_environment(
    monkeypatch, fake_ray
) -> None:
    hosts = _FakeFrontendHostActor()
    monkeypatch.setattr(managed_runtime, "_FrontendHostActor", hosts)
    runtime = _runtime(
        True,
        num_frontends=3,
        render_workers=3,
        extra_env={"DYN_TCP_LISTEN_BACKLOG": "8192"},
    )
    runtime._manager_env = {"DYN_NAMESPACE": "nemo-rl-test"}

    node_a, node_b = "a" * 56, "b" * 56  # Ray node IDs are hex
    runtime._start_frontends([node_a, node_b])
    runtime._started = True
    runtime.publish_chat_capture(
        dp_cfg={"backend": "simple"},
        staging_partition="staging",
        capture_media=False,
        generation_cut_control_token="secret",
        generation_cut_control_timeout_s=60.0,
    )
    runtime.publish_chat_settings({"nemo_rl_chat": {"engine_args": {}}})

    assert hosts.node_ids == [node_a, node_b, node_a]
    assert runtime.frontend_urls == [f"http://frontend-{i}:3000/v1" for i in (1, 2, 3)]
    # Gym pairs each base URL with the cut port of the frontend behind it.
    assert runtime.generation_cut_control_urls == [
        f"http://frontend-{i}:2400" for i in (1, 2, 3)
    ]
    for env in hosts.envs:
        assert env["DYN_TCP_LISTEN_BACKLOG"] == "8192"
        assert env["DYN_NAMESPACE"] == "nemo-rl-test"
        assert env["VIRTUAL_ENV"] == "/venvs/vllm"
        assert env[managed_runtime.RENDER_WORKERS_ENV] == "3"
        assert env[managed_runtime.RAY_ADDRESS_ENV] == "10.0.0.1:6379"
    # Every frontend's chat engine reads token-capture prefixes, and checks a
    # cut's bearer token, the same way.
    assert hosts.captures == 3 * [
        {
            "data_plane": {"backend": "simple"},
            "staging_partition": "staging",
            "capture_media": False,
            "generation_cut_control_token": "secret",
            "generation_cut_control_timeout_s": 60.0,
        }
    ]

    # Every frontend starts its render workers at launch, and builds them from
    # the chat settings an engine sends before it loads.
    assert hosts.serve_chat == 3 * [True]
    assert hosts.chat_settings == 3 * [{"nemo_rl_chat": {"engine_args": {}}}]

    clashing = _runtime(extra_env={"DYN_NAMESPACE": "mine"})
    clashing._manager_env = {"DYN_NAMESPACE": "nemo-rl-test"}
    with pytest.raises(ValueError, match=r"\['DYN_NAMESPACE'\] is set by both"):
        clashing._start_frontends([node_a])


@pytest.mark.parametrize("serves_chat", [True, False])
def test_frontend_host_launches_nemo_rls_chat_engine_on_a_free_port(
    monkeypatch, tmp_path, serves_chat: bool
) -> None:
    launched = []
    monkeypatch.setattr(
        managed_runtime,
        "_ServiceProcess",
        lambda command, *, env, label: launched.append((command, env))
        or _FakeProcess(),
    )
    ports = {
        managed_runtime.DEFAULT_DYNAMO_HTTP_PORT_RANGE_LOW: 3000,
        managed_runtime.DEFAULT_DYNAMO_CONTROL_PORT_RANGE_LOW: 2400,
    }
    monkeypatch.setattr(
        managed_runtime, "_get_free_port_local", lambda low, high, **kwargs: ports[low]
    )
    monkeypatch.setattr(managed_runtime, "_get_node_ip_local", lambda: "10.0.0.2")
    monkeypatch.setattr(managed_runtime.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("DYN_STALE_SETTING", "remove-me")
    host = FrontendHost()

    urls = host.start(
        frontend_python="/venvs/vllm/bin/python",
        namespace="nemo-rl-test",
        dynamo_cfg=DynamoCfg(),
        env={"DYN_NAMESPACE": "nemo-rl-test"},
        serves_chat=serves_chat,
    )
    host.publish_chat_capture({"staging_partition": "staging"})

    assert urls == FrontendUrls(
        base_url="http://10.0.0.2:3000/v1",
        generation_cut_control_url="http://10.0.0.2:2400",
    )
    ((command, env),) = launched
    # Dynamo's frontend, with NeMo RL's chat engine serving chat calls.
    assert command[:3] == [
        "/venvs/vllm/bin/python",
        "-m",
        "nemo_rl.models.generation.dynamo.vllm.frontend",
    ]
    assert command[command.index("--http-port") + 1] == "3000"
    assert env["DYN_NAMESPACE"] == "nemo-rl-test"
    assert "DYN_STALE_SETTING" not in env
    # NeMo-Gym's checkpoint generation cuts go to a port of the frontend's own.
    assert env[managed_runtime.GENERATION_CUT_PORT_ENV] == "2400"
    assert host.returncode() is None
    # Capture is set up after the engines register; the chat engine reads it
    # from the file on its own node that the frontend was told about.
    with open(env[managed_runtime.CAPTURE_FILE_ENV], "rb") as capture_file:
        assert pickle.load(capture_file) == {"staging_partition": "staging"}
    # Only a run that serves chat names the file an engine's chat settings
    # arrive in before it loads; the frontend starts its render workers on it.
    assert (managed_runtime.CHAT_SETTINGS_FILE_ENV in env) is serves_chat
    if serves_chat:
        monkeypatch.setenv(
            managed_runtime.CHAT_SETTINGS_FILE_ENV,
            env[managed_runtime.CHAT_SETTINGS_FILE_ENV],
        )
        assert managed_runtime.read_chat_settings() is None
        host.publish_chat_settings({"nemo_rl_chat": {"engine_args": {}}})
        assert managed_runtime.read_chat_settings() == {
            "nemo_rl_chat": {"engine_args": {}}
        }


def test_shutdown_logs_the_kv_events_the_router_applied(
    monkeypatch, capsys, fake_ray
) -> None:
    runtime = _started(_runtime())
    metrics = (
        "# HELP dynamo_component_kv_cache_events_applied KV events applied\n"
        'dynamo_component_kv_cache_events_applied{event_type="stored",status="ok",worker_id="1"} 40\n'
        'dynamo_component_kv_cache_events_applied{event_type="stored",status="ok",worker_id="2"} 2\n'
        'dynamo_component_kv_cache_events_applied{event_type="stored",status="parent_block_not_found"} 3\n'
        # Every prefix-cache reset applies one; it says nothing about block events.
        'dynamo_component_kv_cache_events_applied{event_type="cleared",status="ok",worker_id="1"} 5\n'
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

    runtime.shutdown()

    assert urls == ["http://10.0.0.2:3000/metrics"]
    assert (
        '[Dynamo] router applied KV events: {"ok": 42.0, "parent_block_not_found": 3.0}'
        in capsys.readouterr().out
    )
