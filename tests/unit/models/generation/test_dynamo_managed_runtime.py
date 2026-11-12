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

import pytest

from nemo_rl.models.generation.dynamo import managed_runtime as runtime_module
from nemo_rl.models.generation.dynamo.config import DynamoCfg, DynamoFrontendArgs
from nemo_rl.models.generation.dynamo.managed_runtime import (
    ManagedDynamoRuntime,
    _managed_namespace,
    _service_executable,
    frontend_argv,
    wait_for_engine_registrations,
)
from nemo_rl.models.generation.megatron.dynamo_config import MegatronDynamoCfg

_MODULE = "nemo_rl.models.generation.dynamo.managed_runtime"


def _runtime(**frontend_args) -> ManagedDynamoRuntime:
    return ManagedDynamoRuntime(
        dynamo_cfg=DynamoCfg(
            startup_timeout_s=5, frontend_args=DynamoFrontendArgs(**frontend_args)
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


@pytest.mark.parametrize(
    ("stopped", "error"),
    [
        ("engine", "shard 0 stopped serving"),
        # A bad frontend_args.extra_cli_args flag makes the frontend exit at once.
        ("frontend", r"frontend is not alive \(code=2\)"),
    ],
)
def test_engine_wait_fails_immediately_when_an_engine_or_service_stops(
    stopped, error
) -> None:
    runtime = _started(_runtime())
    if stopped == "frontend":
        runtime._frontend_process.poll = lambda: 2
        runtime._frontend_process.returncode = 2

    def check_engines() -> None:
        if stopped == "engine":
            raise RuntimeError("endpoint of generation shard 0 stopped serving")

    with pytest.raises(RuntimeError, match=error):
        runtime.wait_for_engines(1, check_engines=check_engines)


def test_frontend_launches_from_the_given_interpreter(monkeypatch) -> None:
    runtime = _runtime()
    runtime._frontend_port = 3000
    runtime._namespace = "nemo-rl-test"
    launched = []
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.managed_runtime.subprocess.Popen",
        lambda command, **kwargs: launched.append(command) or _FakeProcess(),
    )

    runtime._start_frontend()

    assert launched[0][:3] == ["/venvs/vllm/bin/python", "-m", "dynamo.frontend"]


def test_shutdown_logs_the_kv_events_the_router_applied(monkeypatch, capsys) -> None:
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
    monkeypatch.setattr(runtime, "_stop_process", lambda *args, **kwargs: None)

    runtime.shutdown()

    assert urls == ["http://127.0.0.1:3000/metrics"]
    assert (
        '[Dynamo] router applied KV events: {"ok": 42.0, "parent_block_not_found": 3.0}'
        in capsys.readouterr().out
    )


# The runtime reads only startup_timeout_s and frontend_args, so the Megatron
# engines' dynamo_cfg serves it as well as the vLLM one above.
def _megatron_runtime(**overrides) -> ManagedDynamoRuntime:
    return ManagedDynamoRuntime(
        dynamo_cfg=MegatronDynamoCfg(**overrides),
        model_name="model",
        frontend_python="/venvs/engine/bin/python",
    )


def _instance(component: str, instance_id: str, namespace: str = "ns") -> dict:
    return {
        "namespace": namespace,
        "component": component,
        "endpoint": "generate",
        "instance_id": instance_id,
    }


def test_frontend_argv_rejects_repeated_options() -> None:
    cfg = MegatronDynamoCfg(
        frontend_args={"extra_cli_args": ["--kv-overlap-score-weight=2"]}
    )
    argv = frontend_argv(host="0.0.0.0", port=3000, namespace="ns", dynamo_cfg=cfg)
    assert argv[:4] == ["--http-host", "0.0.0.0", "--http-port", "3000"]
    assert "--router-mode" in argv and argv[-1] == "--kv-overlap-score-weight=2"
    cfg = MegatronDynamoCfg(frontend_args={"extra_cli_args": ["--router_mode=kv"]})
    with pytest.raises(ValueError, match="--router-mode is set by both"):
        frontend_argv(host="0.0.0.0", port=3000, namespace="ns", dynamo_cfg=cfg)
    # Spellings that differ only in "_" vs "-" are the same option.
    cfg = MegatronDynamoCfg(
        frontend_args={
            "extra_cli_args": [
                "--kv-overlap-score-weight=1",
                "--kv_overlap_score_weight=2",
            ]
        }
    )
    with pytest.raises(ValueError, match="--kv-overlap-score-weight is repeated"):
        frontend_argv(host="0.0.0.0", port=3000, namespace="ns", dynamo_cfg=cfg)


def test_wait_counts_every_component_then_the_model(monkeypatch) -> None:
    """Both roles must register and the model must be routable before success."""
    prefill_only = {"instances": [_instance("prefill", "p0")]}
    both = {
        "instances": [
            _instance("prefill", "p0"),
            _instance("backend", "d0"),
            _instance("backend", "d1"),
            # Another namespace's engine does not count.
            _instance("backend", "x", namespace="other"),
        ]
    }
    responses = iter(
        [
            prefill_only,
            {"data": [{"id": "model"}]},
            both,
            {"data": []},
            both,
            {"data": [{"id": "model"}]},
        ]
    )
    urls: list[str] = []

    def fake_urlopen(url, timeout):
        urls.append(url)
        return _FakeHttpResponse(next(responses))

    monkeypatch.setattr(f"{_MODULE}.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(f"{_MODULE}.time.sleep", lambda _: None)
    probes = []

    wait_for_engine_registrations(
        frontend_url="http://127.0.0.1:3000/v1/",
        namespace="ns",
        expected_components={"prefill": 1, "backend": 2},
        model_name="model",
        timeout_s=5,
        check_alive=lambda: probes.append(True),
    )

    assert (
        urls == ["http://127.0.0.1:3000/health", "http://127.0.0.1:3000/v1/models"] * 3
    )
    assert len(probes) == 3


def test_wait_fails_at_once_when_an_engine_died(monkeypatch) -> None:
    def dead():
        raise RuntimeError("Megatron engines died: decode-0")

    monkeypatch.setattr(f"{_MODULE}.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="died: decode-0"):
        wait_for_engine_registrations(
            frontend_url="http://127.0.0.1:3000/v1",
            namespace="ns",
            expected_components={"prefill": 1, "backend": 1},
            model_name="model",
            timeout_s=5,
            check_alive=dead,
        )


def test_wait_times_out_naming_what_is_missing(monkeypatch) -> None:
    health = {"instances": [_instance("prefill", "p0")]}
    monkeypatch.setattr(
        f"{_MODULE}.urllib.request.urlopen",
        lambda url, timeout: _FakeHttpResponse(
            health if "health" in url else {"data": []}
        ),
    )
    monkeypatch.setattr(f"{_MODULE}.time.sleep", lambda _: None)
    clock = iter([0.0, 0.0, 10.0])
    monkeypatch.setattr(f"{_MODULE}.time.monotonic", lambda: next(clock))
    with pytest.raises(
        RuntimeError, match=r"registered=\{'prefill': 1, 'backend': 0\}"
    ):
        wait_for_engine_registrations(
            frontend_url="http://127.0.0.1:3000/v1",
            namespace="ns",
            expected_components={"prefill": 1, "backend": 1},
            model_name="model",
            timeout_s=5,
            check_alive=lambda: None,
        )


@pytest.mark.parametrize(
    ("expected_engines", "components"),
    [
        # vLLM: one count, every engine under the backend component.
        pytest.param(2, {"backend": 2}, id="engine_count"),
        # Megatron prefill/decode: a count per component.
        pytest.param(
            {"prefill": 1, "backend": 1}, {"prefill": 1, "backend": 1}, id="per_role"
        ),
    ],
)
def test_runtime_wait_takes_a_count_or_one_per_component(
    monkeypatch, expected_engines, components
) -> None:
    runtime = _megatron_runtime(startup_timeout_s=7.0)
    runtime._frontend_port = 3000
    runtime._namespace = "ns"
    monkeypatch.setattr(runtime, "_assert_services_alive", lambda: None)
    waits = []
    monkeypatch.setattr(
        runtime_module,
        "wait_for_engine_registrations",
        lambda **kwargs: waits.append(kwargs),
    )
    probes = []

    runtime.wait_for_engines(expected_engines, check_engines=lambda: probes.append(1))

    (wait,) = waits
    wait.pop("check_alive")()
    assert probes == [1]
    assert wait == {
        "frontend_url": "http://127.0.0.1:3000/v1",
        "namespace": "ns",
        "expected_components": components,
        "model_name": "model",
        "timeout_s": 7.0,
    }


def test_runtime_wait_checks_its_own_processes(monkeypatch) -> None:
    runtime = _megatron_runtime()
    runtime._frontend_port = 3000
    runtime._namespace = "ns"
    # No service process was started: the probe says so before any HTTP call.
    monkeypatch.setattr(
        f"{_MODULE}.urllib.request.urlopen",
        lambda url, timeout: pytest.fail("no HTTP before the liveness probe"),
    )
    with pytest.raises(RuntimeError, match="etcd is not alive"):
        runtime.wait_for_engines({"prefill": 1}, check_engines=lambda: None)
