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
from unittest.mock import MagicMock

import pytest

from nemo_rl.models.generation.dynamo import venv as dynamo_venv
from nemo_rl.models.generation.dynamo.config import DynamoCfg
from nemo_rl.models.generation.dynamo.managed_runtime import (
    ManagedDynamoRuntime,
    _managed_namespace,
)


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


def _started(runtime: ManagedDynamoRuntime) -> ManagedDynamoRuntime:
    runtime._started = True
    runtime._host = "10.0.0.1"
    runtime._frontend_port = 3000
    runtime._namespace = "nemo-rl-test"
    runtime._etcd_process = _FakeProcess()
    runtime._nats_process = _FakeProcess()
    runtime._frontend_process = _FakeProcess()
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
        _ = runtime.frontend_url
    with pytest.raises(RuntimeError, match="not been started"):
        _ = runtime.engine_env


def test_managed_service_and_frontend_environments_are_runtime_owned(
    monkeypatch,
) -> None:
    runtime = _runtime(
        tokenizer="fastokens", tokenizer_cache=True, tokenizer_cache_bytes=4096
    )
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

    frontend_env = runtime._frontend_env()
    assert frontend_env["DYN_TOKENIZER"] == "fastokens"
    assert frontend_env["DYN_TOKENIZER_CACHE"] == "1"
    assert frontend_env["DYN_TOKENIZER_CACHE_BYTES"] == "4096"


def test_dynamo_python_resolves_the_vllm_worker_venv_as_workers_do(
    monkeypatch,
) -> None:
    built = []
    monkeypatch.setattr(
        dynamo_venv, "get_actor_python_env", lambda fqn: "uv run --locked --extra vllm"
    )
    # The workers' resolver: reuses an existing venv, builds a missing one per node.
    monkeypatch.setattr(
        dynamo_venv,
        "create_local_venv_on_each_node",
        lambda py_executable, venv_name: built.append((py_executable, venv_name))
        or "/venvs/vllm/bin/python",
    )

    assert dynamo_venv.get_dynamo_python() == "/venvs/vllm/bin/python"
    assert built == [
        ("uv run --locked --extra vllm", dynamo_venv.VLLM_ASYNC_WORKER_FQN)
    ]


def test_dynamo_python_passes_an_interpreter_through(monkeypatch) -> None:
    monkeypatch.setattr(
        dynamo_venv, "get_actor_python_env", lambda fqn: "/usr/bin/python3"
    )
    monkeypatch.setattr(
        dynamo_venv,
        "create_local_venv_on_each_node",
        MagicMock(side_effect=AssertionError("no venv for an interpreter")),
    )

    assert dynamo_venv.get_dynamo_python() == "/usr/bin/python3"


def test_dynamo_services_resolve_from_bin_dir_then_path(monkeypatch, tmp_path) -> None:
    etcd = tmp_path / "etcd"
    etcd.write_text("#!/bin/sh\n")
    etcd.chmod(0o755)
    monkeypatch.setenv("NEMO_RL_DYNAMO_BIN_DIR", str(tmp_path))
    assert dynamo_venv.get_dynamo_service_executable("etcd") == str(etcd)
    with pytest.raises(FileNotFoundError, match="NEMO_RL_DYNAMO_BIN_DIR"):
        dynamo_venv.get_dynamo_service_executable("nats-server")

    monkeypatch.delenv("NEMO_RL_DYNAMO_BIN_DIR")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert dynamo_venv.get_dynamo_service_executable("etcd") == str(etcd)
    with pytest.raises(FileNotFoundError, match="not on PATH"):
        dynamo_venv.get_dynamo_service_executable("nats-server")


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


def test_shutdown_guards_each_owned_resource_independently(monkeypatch) -> None:
    runtime = _runtime()
    runtime._frontend_process = object()
    runtime._nats_process = object()
    runtime._etcd_process = object()
    calls = []

    def stop_process(process, label, timeout_s=15):
        if process is None:
            return
        calls.append(label)
        if label == "frontend":
            raise RuntimeError("frontend failure")

    monkeypatch.setattr(runtime, "_stop_process", stop_process)
    runtime.shutdown()
    runtime.shutdown()

    assert calls == ["frontend", "NATS", "etcd"]
    assert runtime._frontend_process is None
    assert runtime._nats_process is None
    assert runtime._etcd_process is None


def test_engines_are_ready_once_registered_and_the_model_is_served(
    monkeypatch,
) -> None:
    runtime = _started(_runtime())
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
    assert (
        urls
        == [
            "http://127.0.0.1:3000/health",
            "http://127.0.0.1:3000/v1/models",
        ]
        * 2
    )


def test_engine_wait_fails_immediately_with_the_engines_error() -> None:
    runtime = _started(_runtime())

    def endpoint_stopped() -> None:
        raise RuntimeError("endpoint of generation shard 0 stopped serving")

    with pytest.raises(RuntimeError, match="shard 0 stopped serving"):
        runtime.wait_for_engines(1, check_engines=endpoint_stopped)


def test_frontend_launches_from_the_given_interpreter(monkeypatch, capsys) -> None:
    runtime = _runtime(tokenizer="fastokens", tokenizer_cache=True)
    runtime._frontend_port = 3000
    runtime._namespace = "nemo-rl-test"
    launched = []
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.managed_runtime.subprocess.Popen",
        lambda command, **kwargs: launched.append(command) or _FakeProcess(),
    )

    runtime._start_frontend()

    assert launched[0][:3] == ["/venvs/vllm/bin/python", "-m", "dynamo.frontend"]
    output = capsys.readouterr().out
    assert "DYN_TOKENIZER': 'fastokens" in output


def test_shutdown_logs_the_kv_events_the_router_applied(monkeypatch, capsys) -> None:
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

    assert urls == ["http://127.0.0.1:3000/metrics"]
    assert (
        '[Dynamo] router applied KV events: {"ok": 42.0, "parent_block_not_found": 3.0}'
        in capsys.readouterr().out
    )
