# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json

import pytest

from nemo_rl.models.generation.dynamo.config import DynamoCfg
from nemo_rl.models.generation.dynamo.managed_runtime import (
    ManagedDynamoRuntime,
    _managed_namespace,
)
from nemo_rl.models.generation.dynamo.worker_pool import (
    FixedDynamoWorkerPool,
    _system_port_for_group,
)


class _RemoteMethod:
    def __init__(self, result):
        self._result = result

    def remote(self, *args, **kwargs):
        return self._result


class _FakeWorker:
    def __init__(self, alive, metadata):
        self.is_alive = _RemoteMethod(alive)
        self.metadata = _RemoteMethod(metadata)
        self.shutdown = _RemoteMethod(True)


class _FakeReservation:
    def __init__(self):
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


def test_namespace_defaults_to_sanitized_slurm_job_id(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "Job/123.4")
    assert _managed_namespace(None) == "nemo-rl-job-123-4"
    assert _managed_namespace("My Namespace") == "my-namespace"


def test_system_ports_are_unique_across_tp1_groups() -> None:
    assert [_system_port_for_group(29000, idx) for idx in range(8)] == list(
        range(29000, 29008)
    )
    with pytest.raises(ValueError, match="exceeds 65535"):
        _system_port_for_group(65535, 1)


def test_runtime_rejects_engine_world_size_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.managed_runtime._get_node_ip_local",
        lambda: "10.0.0.1",
    )
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        ManagedDynamoRuntime(
            cluster=object(),
            config={
                "model_name": "model",
                "dynamo_cfg": {"deployment": "ray", "engine_world_size": 1},
                "vllm_cfg": {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 1,
                },
            },
            dynamo_cfg=DynamoCfg.model_validate(
                {"deployment": "ray", "engine_world_size": 1}
            ),
        )


def test_fixed_pool_detects_actor_or_membership_change(monkeypatch) -> None:
    expected = [{"instance_id": "worker-0", "system_url": "http://10.0.0.1:29000"}]
    pool = object.__new__(FixedDynamoWorkerPool)
    pool._workers = [_FakeWorker("alive", "metadata")]
    pool._reservations = []
    pool._cleanup_reservations = []
    pool._reservation_metadata = []

    def get_changed(refs, **kwargs):
        if refs == ["alive"]:
            return [True]
        if refs == []:
            return []
        return [{"instance_id": "worker-0", "system_url": "http://10.0.0.1:29001"}]

    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.ray.get", get_changed
    )
    with pytest.raises(RuntimeError, match="membership changed"):
        pool.validate(expected)

    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.ray.get",
        lambda refs, **kwargs: [False],
    )
    with pytest.raises(RuntimeError, match="worker exited"):
        pool.validate(expected)


def test_fixed_pool_shutdown_releases_workers_and_reservations(monkeypatch) -> None:
    pool = object.__new__(FixedDynamoWorkerPool)
    worker = _FakeWorker("alive", "metadata")
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
    assert pool._cleanup_reservations == []
    assert pool._reservation_metadata == []
    assert pool._metadata == []


def test_fixed_pool_fallback_kills_orphaned_process_group(monkeypatch) -> None:
    pool = object.__new__(FixedDynamoWorkerPool)
    worker = _FakeWorker("alive", "metadata")
    reservation = _FakeReservation()
    pool._workers = [worker]
    pool._reservations = [reservation]
    pool._cleanup_reservations = [reservation]
    pool._reservation_metadata = []
    pool._metadata = [{"instance_id": "worker-0", "process_pid": 1234}]
    calls = []

    def get_with_shutdown_failure(refs, **kwargs):
        calls.append(refs)
        if refs == [True] and len(calls) == 1:
            raise RuntimeError("actor died")
        return [True]

    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.ray.get",
        get_with_shutdown_failure,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.worker_pool.ray.kill",
        lambda actor, **kwargs: None,
    )

    pool.shutdown()

    assert calls == [[True], [True]]


def test_runtime_startup_failure_cleans_up_worker_pool(monkeypatch) -> None:
    calls = []

    class FailingPool:
        def __init__(self, **kwargs):
            calls.append("init")

        def start(self):
            calls.append("start")
            raise RuntimeError("worker failed")

        def shutdown(self):
            calls.append("shutdown")

    runtime = object.__new__(ManagedDynamoRuntime)
    runtime._cluster = object()
    runtime._config = {"model_name": "model"}
    runtime._dynamo_cfg = DynamoCfg.model_validate(
        {"deployment": "ray", "engine_world_size": 1}
    )
    runtime._namespace = "nemo-rl-test"
    runtime._manager_env = {}
    runtime._etcd_process = None
    runtime._nats_process = None
    runtime._frontend_process = None
    runtime._etcd_data_dir = None
    runtime._nats_data_dir = None
    runtime._pool = None
    monkeypatch.setattr(runtime, "_start_etcd", lambda: calls.append("etcd"))
    monkeypatch.setattr(runtime, "_start_nats", lambda: calls.append("nats"))
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.managed_runtime.FixedDynamoWorkerPool",
        FailingPool,
    )

    with pytest.raises(RuntimeError, match="worker failed"):
        runtime._start()

    assert calls == ["etcd", "nats", "init", "start", "shutdown"]
    assert runtime._pool is None


def test_runtime_waits_for_frontend_model_route_after_registrations(
    monkeypatch,
) -> None:
    runtime = object.__new__(ManagedDynamoRuntime)
    runtime._frontend_port = 8000
    runtime._frontend_process = _FakeProcess()
    runtime._namespace = "nemo-rl-test"
    runtime._config = {"model_name": "org/model"}
    runtime._dynamo_cfg = DynamoCfg.model_validate(
        {
            "deployment": "ray",
            "engine_world_size": 1,
            "startup_timeout_s": 5,
        }
    )
    health_payload = {
        "instances": [
            {
                "namespace": "nemo-rl-test",
                "component": "backend",
                "endpoint": "generate",
                "instance_id": "worker-0",
            },
            {
                "namespace": "nemo-rl-test",
                "component": "backend",
                "endpoint": "rl",
                "instance_id": "worker-0",
            },
        ]
    }
    responses = [
        health_payload,
        {"object": "list", "data": []},
        health_payload,
        {"object": "list", "data": [{"id": "org/model"}]},
    ]
    urls = []

    def fake_urlopen(url, timeout):
        urls.append(url)
        return _FakeHttpResponse(responses.pop(0))

    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.managed_runtime.urllib.request.urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.dynamo.managed_runtime.time.sleep",
        lambda _: None,
    )

    runtime._wait_for_frontend(expected_workers=1)

    assert urls == [
        "http://127.0.0.1:8000/health",
        "http://127.0.0.1:8000/v1/models",
        "http://127.0.0.1:8000/health",
        "http://127.0.0.1:8000/v1/models",
    ]
