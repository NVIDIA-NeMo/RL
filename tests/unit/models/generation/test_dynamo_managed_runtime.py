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

"""Driver-owned etcd, NATS and frontend that NeMo RL's own engines register with."""

import json
import signal
import subprocess

import pytest

from nemo_rl.models.generation.dynamo import managed_runtime as runtime_module
from nemo_rl.models.generation.dynamo.managed_runtime import (
    ManagedDynamoRuntime,
    _managed_namespace,
    _service_executable,
    frontend_argv,
    wait_for_engine_registrations,
)
from nemo_rl.models.generation.megatron.dynamo_config import MegatronDynamoCfg

_MODULE = "nemo_rl.models.generation.dynamo.managed_runtime"


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


def _instance(component: str, instance_id: str, namespace: str = "ns") -> dict:
    return {
        "namespace": namespace,
        "component": component,
        "endpoint": "generate",
        "instance_id": instance_id,
    }


def _runtime(**overrides) -> ManagedDynamoRuntime:
    # Megatron's dynamo_cfg stands in for any backend's: the runtime reads only
    # startup_timeout_s and frontend_args.
    return ManagedDynamoRuntime(
        dynamo_cfg=MegatronDynamoCfg(**overrides),
        model_name="model",
        frontend_python="/venvs/engine/bin/python",
    )


def test_construction_is_inert_and_namespace_is_driver_owned(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "Job/123.4")
    assert _managed_namespace() == "nemo-rl-job-123-4"
    runtime = _runtime()
    assert runtime._started is False and runtime._etcd_process is None
    assert runtime.namespace == "nemo-rl-job-123-4"
    for attribute in ("frontend_url", "engine_env"):
        with pytest.raises(RuntimeError, match="not been started"):
            getattr(runtime, attribute)


def test_start_launches_every_service_and_hands_engines_their_env(monkeypatch):
    runtime = _runtime()
    launched: list[str] = []
    for service in ("etcd", "nats", "frontend"):
        monkeypatch.setattr(
            runtime,
            f"_start_{service}",
            lambda service=service: launched.append(service),
        )
    monkeypatch.setattr(f"{_MODULE}._get_node_ip_local", lambda: "10.0.0.1")
    ports = iter([2379, 2380, 4222, 3000])
    monkeypatch.setattr(
        f"{_MODULE}._get_free_port_local", lambda *args, **kwargs: next(ports)
    )
    monkeypatch.setattr(f"{_MODULE}.atexit.register", lambda fn: None)

    runtime.start()

    assert launched == ["etcd", "nats", "frontend"]
    assert runtime.frontend_url == "http://10.0.0.1:3000/v1"
    env = runtime.engine_env
    assert env["ETCD_ENDPOINTS"] == "http://10.0.0.1:2379"
    assert env["NATS_SERVER"] == "nats://10.0.0.1:4222"
    assert env["DYN_NAMESPACE"] == runtime.namespace
    assert env["DYN_SYSTEM_PORT"] == "0"
    with pytest.raises(RuntimeError, match="already started"):
        runtime.start()


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


def test_service_env_is_runtime_owned(monkeypatch) -> None:
    runtime = _runtime()
    runtime._manager_env = {
        "DYN_NAMESPACE": "managed",
        "ETCD_ENDPOINTS": "http://managed-etcd:2379",
    }
    monkeypatch.setenv("DYN_NAMESPACE", "stale")
    monkeypatch.setenv("DYN_STALE_SETTING", "remove-me")
    monkeypatch.setenv("ETCD_ENDPOINTS", "http://stale-etcd:2379")
    monkeypatch.setenv("NATS_STALE_SETTING", "remove-me")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")

    env = runtime._service_env()

    assert env["DYN_NAMESPACE"] == "managed"
    assert env["ETCD_ENDPOINTS"] == "http://managed-etcd:2379"
    assert "DYN_STALE_SETTING" not in env and "NATS_STALE_SETTING" not in env
    assert env["NCCL_DEBUG"] == "INFO"


def test_service_executable_lookup_order(monkeypatch, tmp_path) -> None:
    binary = tmp_path / "etcd"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setenv("NEMO_RL_DYNAMO_BIN_DIR", str(tmp_path))
    assert _service_executable("etcd") == str(binary)
    with pytest.raises(FileNotFoundError, match="NEMO_RL_DYNAMO_BIN_DIR"):
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
        f"{_MODULE}.os.killpg", lambda pid, sig: signals.append((pid, sig))
    )
    process = EscalatingProcess()
    ManagedDynamoRuntime._stop_process(process, "frontend", timeout_s=0.01)
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]


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
    runtime = _runtime(startup_timeout_s=7.0)
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
    runtime = _runtime()
    runtime._frontend_port = 3000
    runtime._namespace = "ns"
    # No service process was started: the probe says so before any HTTP call.
    monkeypatch.setattr(
        f"{_MODULE}.urllib.request.urlopen",
        lambda url, timeout: pytest.fail("no HTTP before the liveness probe"),
    )
    with pytest.raises(RuntimeError, match="etcd is not alive"):
        runtime.wait_for_engines({"prefill": 1}, check_engines=lambda: None)


def test_shutdown_guards_each_owned_resource(monkeypatch, tmp_path) -> None:
    runtime = _runtime()
    stopped: list[str] = []

    def fake_stop(process, label, timeout_s=15):
        stopped.append(label)
        if label == "NATS":
            raise RuntimeError("boom")

    monkeypatch.setattr(
        runtime_module.ManagedDynamoRuntime, "_stop_process", staticmethod(fake_stop)
    )
    runtime._frontend_process = object()
    runtime._nats_process = object()
    runtime._etcd_process = object()
    etcd_dir = tmp_path / "etcd"
    etcd_dir.mkdir()
    runtime._etcd_data_dir = str(etcd_dir)
    runtime._started = True

    runtime.shutdown()

    assert stopped == ["frontend", "NATS", "etcd"]
    assert not etcd_dir.exists()
    assert runtime._started is False and runtime._frontend_process is None
    runtime.shutdown()
    assert stopped == ["frontend", "NATS", "etcd"] * 2
