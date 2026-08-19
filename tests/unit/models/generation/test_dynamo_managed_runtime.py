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
from unittest.mock import MagicMock

import pytest

from nemo_rl.models.generation.dynamo.dynamo_worker import (
    DynamoGpuReservation,
    DynamoVllmWorker,
)
from nemo_rl.models.generation.dynamo.managed_runtime import (
    ManagedDynamoRuntime,
    _managed_namespace,
)
from nemo_rl.models.generation.dynamo.venv import get_dynamo_venv_dir
from nemo_rl.models.generation.dynamo.worker_pool import (
    FixedDynamoWorkerPool,
    _vllm_port_for_node_slot,
)


class _Cluster:
    num_gpus_per_node = 4


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


def _config(*, tp: int = 1) -> dict:
    return {
        "backend": "dynamo",
        "model_name": "model",
        "colocated": {"enabled": False},
        "dynamo_cfg": {
            "engine": "vllm",
            "startup_timeout_s": 5,
            "request_timeout_s": 30,
            "control_timeout_s": 10,
            "metrics_include_prefixes": None,
            "metrics_exclude_prefixes": None,
            "worker_args": {
                "tool_call_parser": None,
                "reasoning_parser": None,
                "exclude_tools_when_tool_choice_none": True,
                "enable_structural_tag": False,
                "structural_tag_scope": "auto",
                "structural_tag_schema": "auto",
                "custom_jinja_template": None,
                "endpoint_types": ["chat", "completions"],
                "extra_cli_args": [],
            },
            "frontend_args": {
                "tokenizer": "default",
                "tokenizer_cache": False,
                "tokenizer_cache_bytes": 50 * 1024 * 1024,
                "router_mode": "kv",
                "router_reset_states": True,
                "extra_cli_args": [],
            },
        },
        "vllm_cfg": {
            "async_engine": True,
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": tp,
            "gpu_memory_utilization": 0.8,
            "max_model_len": 512,
            "precision": "bfloat16",
            "kv_cache_dtype": "auto",
            "load_format": "auto",
            "enforce_eager": False,
            "expose_http_server": False,
            "enable_vllm_metrics_logger": True,
            "vllm_metrics_logger_interval": 1.0,
            "env_vars": None,
        },
        "vllm_kwargs": {},
    }


class _RemoteMethod:
    def __init__(self, result):
        self._result = result
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._result


class _FakeWorker:
    def __init__(self, alive=True, metadata=None):
        self.is_alive = _RemoteMethod(alive)
        self.metadata = _RemoteMethod(metadata or {})
        self.shutdown = _RemoteMethod(True)


class _FakeReservation:
    def __init__(self, metadata=None, system_port=4000):
        self.metadata = _RemoteMethod(metadata or {})
        self.select_free_port = _RemoteMethod(system_port)
        self.register_process_group = _RemoteMethod(True)
        self.cleanup_process_group = _RemoteMethod(True)


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


def test_runtime_construction_is_inert_and_namespace_is_driver_owned(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "Job/123.4")
    assert _managed_namespace() == "nemo-rl-job-123-4"
    runtime = ManagedDynamoRuntime(cluster=_Cluster(), config=_config())
    assert runtime._started is False
    assert runtime._etcd_process is None
    with pytest.raises(RuntimeError, match="not been started"):
        _ = runtime.frontend_url


def test_runtime_rejects_multinode_engine_group_before_spawning() -> None:
    with pytest.raises(ValueError, match="fit on one node"):
        ManagedDynamoRuntime(cluster=_Cluster(), config=_config(tp=8))


def test_managed_service_and_frontend_environments_are_runtime_owned(
    monkeypatch,
) -> None:
    config = _config()
    config["dynamo_cfg"]["frontend_args"].update(
        {
            "tokenizer": "fastokens",
            "tokenizer_cache": True,
            "tokenizer_cache_bytes": 4096,
        }
    )
    runtime = ManagedDynamoRuntime(cluster=_Cluster(), config=config)
    runtime._manager_env = {
        "DYN_NAMESPACE": "managed",
        "DYN_DISCOVERY_BACKEND": "etcd",
        "ETCD_ENDPOINTS": "http://managed-etcd:2379",
        "NATS_SERVER": "nats://managed-nats:4222",
    }
    monkeypatch.setenv("DYN_NAMESPACE", "stale")
    monkeypatch.setenv("DYN_STALE_SETTING", "remove-me")
    monkeypatch.setenv("ETCD_ENDPOINTS", "http://stale-etcd:2379")
    monkeypatch.setenv("ETCD_STALE_SETTING", "remove-me")
    monkeypatch.setenv("NATS_SERVER", "nats://stale-nats:4222")
    monkeypatch.setenv("NATS_STALE_SETTING", "remove-me")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")

    service_env = runtime._service_env()
    assert service_env["DYN_NAMESPACE"] == "managed"
    assert service_env["DYN_DISCOVERY_BACKEND"] == "etcd"
    assert "DYN_STALE_SETTING" not in service_env
    assert service_env["ETCD_ENDPOINTS"] == "http://managed-etcd:2379"
    assert service_env["NATS_SERVER"] == "nats://managed-nats:4222"
    assert "ETCD_STALE_SETTING" not in service_env
    assert "NATS_STALE_SETTING" not in service_env
    assert service_env["NCCL_DEBUG"] == "INFO"
    assert "ALLOW_NONE_AUTHENTICATION" not in service_env

    frontend_env = runtime._frontend_env()
    assert frontend_env["DYN_TOKENIZER"] == "fastokens"
    assert frontend_env["DYN_TOKENIZER_CACHE"] == "1"
    assert frontend_env["DYN_TOKENIZER_CACHE_BYTES"] == "4096"


def test_dynamo_venv_uses_explicit_env_or_repository_fallback(monkeypatch) -> None:
    monkeypatch.setenv("NEMO_RL_DYNAMO_VENV_DIR", "/custom/dynamo")
    assert get_dynamo_venv_dir() == Path("/custom/dynamo")

    monkeypatch.delenv("NEMO_RL_DYNAMO_VENV_DIR")
    monkeypatch.setenv("NRL_CONTAINER", "1")
    assert get_dynamo_venv_dir().parts[-2:] == ("venvs", "dynamo")


def test_vllm_node_local_port_bands_are_deterministic() -> None:
    assert [_vllm_port_for_node_slot(slot) for slot in range(3)] == [7000, 7100, 7200]


def test_startup_failure_cleans_up_partial_worker_pool(monkeypatch, tmp_path) -> None:
    calls = []
    exit_hooks = []
    pool_init_kwargs = {}

    class FailingPool:
        def __init__(self, **kwargs):
            calls.append("pool-init")
            pool_init_kwargs.update(kwargs)

        def start(self):
            calls.append("pool-start")
            raise RuntimeError("worker failed")

        def shutdown(self):
            calls.append("pool-shutdown")

    runtime = ManagedDynamoRuntime(cluster=_Cluster(), config=_config())
    etcd_dir = tmp_path / "etcd"
    nats_dir = tmp_path / "nats"
    etcd_dir.mkdir()
    nats_dir.mkdir()
    ports = iter([1313, 1314, 1315, 3000])
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.managed_runtime._get_node_ip_local",
        lambda: "10.0.0.1",
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.managed_runtime._get_free_port_local",
        lambda low, high, **kwargs: next(ports),
    )

    def start_etcd():
        calls.append("etcd")
        runtime._etcd_process = object()
        runtime._etcd_data_dir = str(etcd_dir)

    def start_nats():
        calls.append("nats")
        runtime._nats_process = object()
        runtime._nats_data_dir = str(nats_dir)

    def stop_process(process, label, timeout_s=15):
        if process is not None:
            calls.append(f"stop-{label}")

    monkeypatch.setattr(runtime, "_start_etcd", start_etcd)
    monkeypatch.setattr(runtime, "_start_nats", start_nats)
    monkeypatch.setattr(runtime, "_stop_process", stop_process)
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.managed_runtime.atexit.register",
        lambda hook: exit_hooks.append(hook),
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.managed_runtime.atexit.unregister",
        lambda hook: exit_hooks.remove(hook),
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.managed_runtime.FixedDynamoWorkerPool",
        FailingPool,
    )

    with pytest.raises(RuntimeError, match="worker failed"):
        runtime.start()
    assert calls == [
        "etcd",
        "nats",
        "pool-init",
        "pool-start",
        "pool-shutdown",
        "stop-NATS",
        "stop-etcd",
    ]
    assert runtime._pool is None
    assert runtime._started is False
    assert exit_hooks == []
    assert pool_init_kwargs["manager_env"]["DYN_ENABLE_EXPERIMENTAL_PARSERS_V2"] == "1"
    assert pool_init_kwargs["manager_env"]["DYN_RL_INIT_WEIGHTS_TIMEOUT_S"] == "10.0"
    assert not etcd_dir.exists()
    assert not nats_dir.exists()


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
    ManagedDynamoRuntime._stop_process(process, "worker", timeout_s=0.01)
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]


def test_reservation_records_and_cleans_registered_process_group(monkeypatch) -> None:
    reservation_cls = DynamoGpuReservation.__ray_metadata__.modified_class
    reservation = reservation_cls()
    signals = []
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.dynamo_worker.os.killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.dynamo_worker.time.sleep", lambda _: None
    )

    assert reservation.register_process_group(4321)
    assert reservation.cleanup_process_group()
    assert reservation.cleanup_process_group()
    assert signals == [(4321, signal.SIGTERM), (4321, signal.SIGKILL)]


def test_reservation_selects_a_free_nonexcluded_system_port(monkeypatch) -> None:
    reservation_cls = DynamoGpuReservation.__ray_metadata__.modified_class
    reservation = reservation_cls()
    calls = []
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.dynamo_worker._get_free_port_local",
        lambda low, high, **kwargs: calls.append((low, high, kwargs)) or 4002,
    )

    assert (
        reservation.select_free_port(
            port_range_low=4000,
            port_range_high=4003,
            excluded_ports=[4001],
        )
        == 4002
    )
    assert calls == [
        (
            4000,
            4003,
            {"max_retries": None, "excluded_ports": {4001}},
        )
    ]


def test_worker_argument_validation_has_startup_timeout(monkeypatch) -> None:
    worker_cls = DynamoVllmWorker.__ray_metadata__.modified_class
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.dynamo_worker.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("validator", kwargs["timeout"])
        ),
    )

    with pytest.raises(RuntimeError, match="startup_timeout_s=7"):
        worker_cls._validate_argv(
            "/opt/dynamo_venv/bin/python",
            ["--model", "model"],
            {},
            timeout_s=7,
        )


def test_worker_registers_process_group_immediately_after_launch(monkeypatch) -> None:
    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def bind(self, address):
            if address[1] == 7000:
                raise AssertionError("VLLM_PORT must not be probed")
            return None

    process = SimpleNamespace(pid=4321)
    reservation = _FakeReservation()
    worker_cls = DynamoVllmWorker.__ray_metadata__.modified_class
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.dynamo_worker._get_node_ip_local",
        lambda: "10.0.0.1",
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.dynamo_worker.socket.socket",
        lambda *args, **kwargs: FakeSocket(),
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.dynamo_worker.get_dynamo_python",
        lambda: "/opt/dynamo_venv/bin/python",
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.dynamo_worker.get_dynamo_venv_dir",
        lambda: Path("/opt/dynamo_venv"),
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.dynamo_worker.subprocess.Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.dynamo_worker.ray.get", lambda ref: ref
    )
    monkeypatch.setattr(worker_cls, "_validate_argv", MagicMock())
    monkeypatch.setattr(worker_cls, "_wait_for_system_port", MagicMock())

    worker_cls(
        _config(),
        namespace="nemo-rl-test",
        group_name="worker-0",
        cuda_devices=[0],
        system_port=4000,
        vllm_port=7000,
        manager_env={},
        startup_timeout_s=5,
        seed=0,
        cleanup_reservation=reservation,
    )

    assert reservation.register_process_group.calls == [((4321,), {})]


def test_shutdown_guards_each_owned_resource_independently(monkeypatch) -> None:
    runtime = ManagedDynamoRuntime(cluster=_Cluster(), config=_config())
    runtime._frontend_process = object()
    runtime._nats_process = object()
    runtime._etcd_process = object()
    calls = []

    class FailingPool:
        def shutdown(self):
            calls.append("pool")
            raise RuntimeError("pool failure")

    runtime._pool = FailingPool()

    def stop_process(process, label, timeout_s=15):
        if process is None:
            return
        calls.append(label)
        if label == "frontend":
            raise RuntimeError("frontend failure")

    monkeypatch.setattr(runtime, "_stop_process", stop_process)
    runtime.shutdown()
    runtime.shutdown()

    assert calls[:4] == ["frontend", "pool", "NATS", "etcd"]
    assert runtime._frontend_process is None
    assert runtime._pool is None
    assert runtime._nats_process is None
    assert runtime._etcd_process is None


def test_fixed_pool_detects_worker_membership_change(monkeypatch) -> None:
    expected = [{"instance_id": "worker-0", "system_url": "http://10.0.0.1:4000"}]
    reservation_metadata = {"node_ip": "10.0.0.1", "gpu_id": 0}
    reservation = _FakeReservation(metadata=reservation_metadata)
    pool = object.__new__(FixedDynamoWorkerPool)
    pool._workers = [_FakeWorker(metadata={"instance_id": "changed"})]
    pool._reservations = [reservation]
    pool._cleanup_reservations = [reservation]
    pool._reservation_metadata = [reservation_metadata]

    def fake_get(refs, **kwargs):
        if refs == [True]:
            return [True]
        if refs == [reservation_metadata]:
            return [reservation_metadata]
        return [{"instance_id": "changed"}]

    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.ray.get", fake_get
    )
    with pytest.raises(RuntimeError, match="worker membership changed"):
        pool.validate(expected)


def test_fixed_pool_detects_reservation_membership_change(monkeypatch) -> None:
    expected_worker = {
        "instance_id": "worker-0",
        "system_url": "http://10.0.0.1:4000",
    }
    expected_reservation = {"node_ip": "10.0.0.1", "gpu_id": 0}
    changed_reservation = {"node_ip": "10.0.0.2", "gpu_id": 0}
    reservation = _FakeReservation(metadata=changed_reservation)
    pool = object.__new__(FixedDynamoWorkerPool)
    pool._workers = [_FakeWorker(metadata=expected_worker)]
    pool._reservations = [reservation]
    pool._cleanup_reservations = [reservation]
    pool._reservation_metadata = [expected_reservation]

    def fake_get(refs, **kwargs):
        if refs == [True]:
            return [True]
        if refs == [changed_reservation]:
            return [changed_reservation]
        return [expected_worker]

    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.ray.get", fake_get
    )
    with pytest.raises(RuntimeError, match="GPU reservation membership changed"):
        pool.validate([expected_worker])


def test_fixed_pool_tracks_worker_before_metadata_failure(monkeypatch) -> None:
    reservation = _FakeReservation(metadata={"node_ip": "10.0.0.1", "gpu_id": 0})
    worker = _FakeWorker()

    class RemoteFactory:
        def __init__(self, actor):
            self.actor = actor

        def remote(self, *args, **kwargs):
            return self.actor

    class ReservationFactory:
        @staticmethod
        def options(**kwargs):
            return RemoteFactory(reservation)

    class WorkerFactory:
        @staticmethod
        def options(**kwargs):
            return RemoteFactory(worker)

    class Cluster:
        @staticmethod
        def get_placement_groups():
            return [SimpleNamespace(bundle_count=1)]

    pool = FixedDynamoWorkerPool(
        cluster=Cluster(),
        config=_config(),
        namespace="nemo-rl-test",
        engine_world_size=1,
        manager_env={},
        startup_timeout_s=5,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.DynamoGpuReservation",
        ReservationFactory,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.DynamoVllmWorker",
        WorkerFactory,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.get_actor_python_env",
        lambda _fqn: "python",
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.PlacementGroupSchedulingStrategy",
        lambda **kwargs: kwargs,
    )

    def fake_get(refs, **kwargs):
        if refs == [reservation.metadata.remote()]:
            return [{"node_ip": "10.0.0.1", "gpu_id": 0}]
        if refs == 4000:
            return refs
        raise RuntimeError("metadata failed")

    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.ray.get", fake_get
    )
    with pytest.raises(RuntimeError, match="metadata failed"):
        pool.start()

    assert pool._workers == [worker]
    assert pool._cleanup_reservations == [reservation]
    assert pool._metadata == [{}]


def test_fixed_pool_launches_all_workers_before_waiting_for_model_metadata(
    monkeypatch,
) -> None:
    reservation_objects = [
        _FakeReservation(
            metadata={"node_ip": "10.0.0.1", "gpu_id": 0}, system_port=4001
        ),
        _FakeReservation(
            metadata={"node_ip": "10.0.0.1", "gpu_id": 1}, system_port=4002
        ),
    ]
    reservations = iter(reservation_objects)
    workers = [
        _FakeWorker(metadata={"instance_id": "worker-0"}),
        _FakeWorker(metadata={"instance_id": "worker-1"}),
    ]
    workers_to_launch = iter(workers)
    launch_kwargs = []

    class RemoteFactory:
        def __init__(self, actor):
            self.actor = actor

        def remote(self, *args, **kwargs):
            return self.actor

    class ReservationFactory:
        @staticmethod
        def options(**kwargs):
            return RemoteFactory(next(reservations))

    class WorkerFactory:
        @staticmethod
        def options(**kwargs):
            worker = next(workers_to_launch)

            class CapturingRemoteFactory(RemoteFactory):
                def remote(self, *args, **kwargs):
                    launch_kwargs.append(kwargs)
                    return super().remote(*args, **kwargs)

            return CapturingRemoteFactory(worker)

    class Cluster:
        @staticmethod
        def get_placement_groups():
            return [SimpleNamespace(bundle_count=2)]

    pool = FixedDynamoWorkerPool(
        cluster=Cluster(),
        config=_config(),
        namespace="nemo-rl-test",
        engine_world_size=1,
        manager_env={},
        startup_timeout_s=5,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.DynamoGpuReservation",
        ReservationFactory,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.DynamoVllmWorker",
        WorkerFactory,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.get_actor_python_env",
        lambda _fqn: "python",
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.PlacementGroupSchedulingStrategy",
        lambda **kwargs: kwargs,
    )

    def fake_get(refs, **kwargs):
        if isinstance(refs, int):
            return refs
        if isinstance(refs, list) and refs and "node_ip" in refs[0]:
            return refs
        assert pool._workers == workers
        return refs

    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.ray.get", fake_get
    )

    pool.start()

    assert pool._metadata == [
        {"instance_id": "worker-0"},
        {"instance_id": "worker-1"},
    ]
    assert [kwargs["system_port"] for kwargs in launch_kwargs] == [4001, 4002]
    assert [kwargs["vllm_port"] for kwargs in launch_kwargs] == [7000, 7100]
    assert [kwargs["cleanup_reservation"] for kwargs in launch_kwargs] == (
        reservation_objects
    )
    assert reservation_objects[0].select_free_port.calls == [
        (
            (),
            {
                "port_range_low": 4000,
                "port_range_high": 4100,
                "excluded_ports": [],
            },
        )
    ]
    assert reservation_objects[1].select_free_port.calls == [
        (
            (),
            {
                "port_range_low": 4000,
                "port_range_high": 4100,
                "excluded_ports": [4001],
            },
        )
    ]


def test_fixed_pool_shutdown_releases_workers_and_reservations(monkeypatch) -> None:
    pool = object.__new__(FixedDynamoWorkerPool)
    worker = _FakeWorker()
    reservation = _FakeReservation()
    pool._workers = [worker]
    pool._reservations = [reservation]
    pool._cleanup_reservations = [reservation]
    pool._reservation_metadata = []
    pool._metadata = [{"instance_id": "worker-0", "process_pid": 1234}]
    killed = []
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.ray.get",
        lambda refs, **kwargs: [True],
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.ray.kill",
        lambda actor, **kwargs: killed.append(actor),
    )
    pool.shutdown()
    assert killed == [worker, reservation]
    assert pool._workers == []
    assert pool._reservations == []


def test_fixed_pool_shutdown_uses_registered_pid_when_worker_dies(monkeypatch) -> None:
    worker = _FakeWorker()
    reservation = _FakeReservation()
    pool = object.__new__(FixedDynamoWorkerPool)
    pool._workers = [worker]
    pool._reservations = [reservation]
    pool._cleanup_reservations = [reservation]
    pool._reservation_metadata = []
    pool._metadata = [{}]
    shutdown_ref = worker.shutdown.remote()

    def fake_get(ref, **kwargs):
        if ref == shutdown_ref:
            raise RuntimeError("worker died")
        return True

    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.ray.get", fake_get
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.ray.kill",
        lambda actor, **kwargs: None,
    )

    pool.shutdown()

    assert reservation.cleanup_process_group.calls == [((), {})]


def test_frontend_waits_for_model_after_endpoint_registration(monkeypatch) -> None:
    runtime = ManagedDynamoRuntime(cluster=_Cluster(), config=_config())
    runtime._frontend_port = 3000
    runtime._frontend_process = _FakeProcess()
    runtime._namespace = "nemo-rl-test"
    runtime._pool = SimpleNamespace(is_alive=lambda: True)
    health = {
        "instances": [
            {
                "namespace": "nemo-rl-test",
                "component": "backend",
                "endpoint": endpoint,
                "instance_id": "worker-0",
            }
            for endpoint in ("generate", "rl")
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
    runtime._wait_for_frontend(expected_workers=1)
    assert urls == [
        "http://127.0.0.1:3000/health",
        "http://127.0.0.1:3000/v1/models",
        "http://127.0.0.1:3000/health",
        "http://127.0.0.1:3000/v1/models",
    ]


def test_frontend_wait_fails_immediately_when_worker_exits(monkeypatch) -> None:
    runtime = ManagedDynamoRuntime(cluster=_Cluster(), config=_config())
    runtime._frontend_port = 3000
    runtime._frontend_process = _FakeProcess()
    runtime._pool = SimpleNamespace(is_alive=lambda: False)

    with pytest.raises(RuntimeError, match="worker exited while the frontend"):
        runtime._wait_for_frontend(expected_workers=1)


def test_frontend_logs_resolved_tokenizer_environment(monkeypatch, capsys) -> None:
    config = _config()
    config["dynamo_cfg"]["frontend_args"].update(
        {"tokenizer": "fastokens", "tokenizer_cache": True}
    )
    runtime = ManagedDynamoRuntime(cluster=_Cluster(), config=config)
    runtime._frontend_port = 3000
    runtime._namespace = "nemo-rl-test"
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.managed_runtime.get_dynamo_python",
        lambda: "/opt/dynamo_venv/bin/python",
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.managed_runtime.subprocess.Popen",
        lambda *args, **kwargs: _FakeProcess(),
    )

    runtime._start_frontend()

    output = capsys.readouterr().out
    assert "DYN_TOKENIZER': 'fastokens" in output
    assert "<redacted>" not in output
