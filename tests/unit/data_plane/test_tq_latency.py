# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Run pinned native TQ methods with fake transport, preserving both executors."""

from __future__ import annotations

import ast
import asyncio
import functools
import itertools
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from nemo_rl.data_plane import tq_latency

G_PINNED = Path(
    os.environ.get(
        "NRL_TEST_TQ_SERIAL_UTILS_SOURCE",
        "/tmp/nano35-tq-image-source/opt/nemo_rl_venv/lib/python3.13/site-packages/transfer_queue/utils/serial_utils.py",
    )
).parents[1]


def native_method(path, class_name, method, namespace, *, preserve_decorator=False):
    if not path.exists():
        pytest.skip("Pinned container TQ sources unavailable")
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    definition = next(
        n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method
    )
    if not preserve_decorator:
        definition.decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, definition], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[method]


@pytest.fixture
def client():
    def allocate(dtypes, shapes):
        tensors = [
            torch.zeros(shape, dtype=dtype)
            for dtype, shape in zip(dtypes, shapes, strict=True)
        ]
        return (
            tensors,
            None,
            [t.data_ptr() for t in tensors],
            [t.numel() for t in tensors],
        )

    namespace = dict(
        asyncio=asyncio,
        torch=torch,
        Tensor=torch.Tensor,
        cast=cast,
        Any=Any,
        itertools=itertools,
        ThreadPoolExecutor=ThreadPoolExecutor,
        MAX_BATCH_WORKER_THREADS=2,
        BATCH_SIZE_LIMIT=2,
        MAX_SERIAL_WORKER_THREADS=2,
        extract_field_schema=lambda data: {},
        allocate_empty_tensors=allocate,
        merge_contiguous_memory=lambda ptrs, sizes: (ptrs, sizes),
        serial_utils=SimpleNamespace(
            batch_encode_into=lambda values, alloc, **kw: (
                alloc([8 for _ in values]),
                [8 for _ in values],
            )
        ),
    )
    manager_put = native_method(
        G_PINNED / "storage/managers/base.py",
        "KVStorageManager",
        "put_data",
        dict(namespace),
    )
    storage_path = G_PINNED / "storage/clients/mooncake_client.py"

    class Storage:
        use_gdr = False
        _gdr_staging = None
        fail = False

        def _preprocess_tensors_for_put(self, tensors):
            return (
                [t.data_ptr() for t in tensors],
                [t.numel() * t.element_size() for t in tensors],
                tensors,
            )

        def _register_all_buffers(self, ptrs, sizes):
            pass

        def _unregister_all_buffers(self, ptrs):
            pass

        def _batch_upsert_with_retry(self, keys, ptrs, sizes):
            assert tq_latency.G_TQ_TRACE.get() is not None
            if self.fail:
                raise ValueError("native transport failed")

    for name in ["put", "_put_bytes_thread_worker", "_put_tensors_thread_worker"]:
        setattr(
            Storage,
            name,
            native_method(storage_path, "MooncakeStoreClient", name, dict(namespace)),
        )

    class Manager:
        put_data = manager_put

        def __init__(self):
            self.storage_client = Storage()

        def _generate_keys(self, names, indices):
            return [f"{n}/{i}" for n in names for i in indices]

        def _generate_values(self, data):
            return [torch.tensor([1, 2]), {"payload": "unchanged"}]

        async def notify_data_update(self, *args):
            await self._notify_and_wait(None)

        async def _notify_and_wait(self, message):
            await asyncio.sleep(0)

    class ControllerSocket:
        async def send_multipart(self, frames):
            assert frames == [b"controller request"]

        async def recv_multipart(self, copy):
            await asyncio.sleep(0)
            return [b"controller response"]

    class ControllerMessage:
        @classmethod
        def create(cls, **kwargs):
            return cls()

        def serialize(self):
            return [b"controller request"]

        @classmethod
        def deserialize(cls, frames):
            return SimpleNamespace(request_type="reply", body={})

    controller_request = native_method(
        G_PINNED / "client.py",
        "AsyncTransferQueueClient",
        "_request_controller",
        dict(ZMQMessage=ControllerMessage, logger=logging.getLogger(__name__)),
    )

    class Client:
        def __init__(self):
            self.storage_manager = Manager()
            self.client_id = "test-client"
            self._controller = SimpleNamespace(id="test-controller")
            self.loop = asyncio.new_event_loop()
            self._loop = self.loop
            self.thread = threading.Thread(target=self.loop.run_forever)
            self.thread.start()

        _request_controller = controller_request

        async def async_kv_retrieve_meta(self):
            await self._request_controller(
                socket=ControllerSocket(),
                request_type="get_meta",
                response_type="reply",
                body={},
            )
            return SimpleNamespace(
                global_indexes=[0], partition_ids=["test"], _custom_backend_meta=[{}]
            )

        async def async_set_custom_meta(self, metadata):
            await self._request_controller(
                socket=ControllerSocket(),
                request_type="set_meta",
                response_type="reply",
                body={},
            )

        async def async_put(self, data, metadata):
            await self.storage_manager.put_data(data, metadata)
            await self.async_set_custom_meta(metadata)
            return metadata

        def _run_coroutine(self, coro):
            return asyncio.run_coroutine_threadsafe(coro, self.loop).result(5)

    value = Client()
    yield value
    if tq_latency.G_TQ_HOOKS is not None:
        tq_latency.G_TQ_HOOKS.close()
    value.loop.call_soon_threadsafe(value.loop.stop)
    value.thread.join(3)
    value.loop.close()


@pytest.mark.parametrize("fail", [False, True])
def test_native_put_phases_survive_both_executor_hops_and_preserve_errors(
    client, caplog, fail
):
    caplog.set_level(logging.INFO)
    hooks = tq_latency.install_transfer_queue_latency(
        client, max_events=256, slow_call_s=0.000001
    )
    client.storage_manager.storage_client.fail = fail
    data = SimpleNamespace(batch_size=[1], keys=lambda: ["tensor", "object"])

    async def put():
        meta = await client.async_kv_retrieve_meta()
        return await client.async_put(data, meta)

    if fail:
        with (
            pytest.raises(ValueError, match="native transport failed"),
            tq_latency.trace_tq_put(),
        ):
            client._run_coroutine(put())
    else:
        with tq_latency.trace_tq_put():
            assert client._run_coroutine(put()).global_indexes == [0]
    records = [
        json.loads(r.message.split(" ", 1)[1])
        for r in caplog.records
        if r.message.startswith("tq_put_internal_phases ")
    ]
    assert len(records) == 1 and records[0]["failed"] == fail
    phases = records[0]["phases"]
    names = {e["phase"] for e in phases}
    assert {
        "client_loop_dispatch_wait",
        "metadata_lookup",
        "controller_rpc",
        "executor_dispatch_wait",
        "backend_put",
        "serialization",
        "buffer_register",
        "backend_upsert_with_retries",
    } <= names
    if not fail:
        assert {"data_ready_ack", "custom_metadata_ack"} <= names
    assert len({e["thread_id"] for e in phases}) >= 3
    assert all(
        e["thread_cpu_s"] is None for e in phases if e["phase"] == "metadata_lookup"
    )
    hooks.close()
    assert tq_latency.G_TQ_HOOKS is None
    assert "async_put" not in vars(client)
    assert "put" not in vars(client.storage_manager.storage_client)


def test_bounded_phase_events():
    trace = tq_latency._PutTrace(16, 0)
    for _ in range(50):
        with trace.phase("phase"):
            pass
    assert len(trace.events) == 16 and trace.dropped == 34


def test_unknown_native_seam_is_rejected_without_execution():
    def unsupported():
        raise AssertionError("must not execute")

    with pytest.raises(ValueError, match="Unsupported pinned TQ"):
        tq_latency.instrument_native_function(unsupported, "manager")


def test_disabled_root_probe_is_a_noop():
    assert tq_latency.G_TQ_HOOKS is None
    with tq_latency.trace_tq_put():
        assert tq_latency.G_TQ_TRACE.get() is None


@pytest.mark.parametrize("mode", ["success", "retry_once", "bad_response"])
def test_pinned_simple_storage_keeps_socket_leases_retries_and_values(
    client, caplog, mode
):
    caplog.set_level(logging.INFO)
    sent, leases = [], []

    class Again(Exception):
        pass

    class StorageTimeout(RuntimeError):
        pass

    class Message:
        def __init__(self, body):
            self.body = body

        @classmethod
        def create(cls, **kwargs):
            return cls(kwargs["body"])

        def serialize(self):
            sent.append(self.body)
            return [b"original-wire-payload"]

        @staticmethod
        def deserialize(frames):
            return SimpleNamespace(
                request_type="bad" if mode == "bad_response" else "reply", body={}
            )

    class Socket:
        async def send_multipart(self, frames, copy):
            assert frames == [b"original-wire-payload"] and copy is False
            if mode == "retry_once" and len(leases) == 1:
                raise Again("transient send timeout")

        async def recv_multipart(self, copy):
            await asyncio.sleep(0)
            return [b"reply"]

    def socket_decorator(function):
        @functools.wraps(function)
        async def lease(self, *args, **kwargs):
            leases.append(kwargs["target_storage_unit"])
            kwargs["socket"] = Socket()
            return await function(self, *args, **kwargs)

        return lease

    namespace = dict(
        asyncio=asyncio,
        time=time,
        partial=functools.partial,
        extract_field_schema=lambda data: {"tokens": "schema"},
        logger=logging.getLogger(__name__),
        ZMQMessage=Message,
        ZMQRequestType=SimpleNamespace(PUT_DATA="put", PUT_DATA_RESPONSE="reply"),
        StorageUnitTimeout=StorageTimeout,
        TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT=1,
        TQ_SIMPLE_STORAGE_MAX_ATTEMPTS=3,
        frame_nbytes=len,
        log_heavy_operation=lambda *args: None,
        zmq=SimpleNamespace(error=SimpleNamespace(Again=Again)),
        with_storage_unit_socket=socket_decorator,
    )
    source = G_PINNED / "storage/managers/simple_storage_manager.py"
    old_manager = type(client.storage_manager)

    class Manager:
        storage_manager_id = "simple-test"
        notify_data_update = old_manager.notify_data_update
        _notify_and_wait = old_manager._notify_and_wait

        def _group_by_hash(self, indexes):
            return {
                "unit": SimpleNamespace(global_indexes=indexes, batch_positions=[0, 1])
            }

        def _select_by_positions(self, value, positions):
            return value[positions]

        def _describe_storage_unit(self, unit):
            return unit

        async def _diagnose_storage_unit(self, unit):
            return "test unit"

    for name in ["put_data", "_request_with_retry", "_put_to_single_storage_unit"]:
        setattr(
            Manager,
            name,
            native_method(
                source,
                "AsyncSimpleStorageManager",
                name,
                dict(namespace),
                preserve_decorator=name.startswith("_put_to"),
            ),
        )
    client.storage_manager = Manager()
    hooks = tq_latency.install_transfer_queue_latency(
        client, max_events=256, slow_call_s=0.000001
    )
    data = {"tokens": torch.tensor([[1, 2], [3, 4]])}
    metadata = SimpleNamespace(size=2, global_indexes=[0, 1], partition_ids=["test"])
    with tq_latency.trace_tq_put():
        if mode == "bad_response":
            with pytest.raises(RuntimeError, match="Failed to put data"):
                client._run_coroutine(client.async_put(data, metadata))
        else:
            assert client._run_coroutine(client.async_put(data, metadata)) is metadata
    assert len(leases) == (2 if mode == "retry_once" else 1)
    assert all(torch.equal(body["data"]["tokens"], data["tokens"]) for body in sent)
    record = next(
        json.loads(r.message.split(" ", 1)[1])
        for r in caplog.records
        if r.message.startswith("tq_put_internal_phases ")
    )
    phases = record["phases"]
    assert {
        "storage_schema",
        "storage_routing",
        "storage_slice",
        "storage_serialize",
        "storage_send",
    } <= {e["phase"] for e in phases}
    if mode != "bad_response":
        assert {"storage_response_wait", "data_ready_ack"} <= {
            e["phase"] for e in phases
        }
    assert all(
        e["storage_unit"] == "unit" for e in phases if e["phase"] == "storage_serialize"
    )
    hooks.close()
    assert "_put_to_single_storage_unit" not in vars(client.storage_manager)


def test_controller_probe_retains_request_kind_and_response_validation(
    client, caplog, monkeypatch
):
    import inspect

    caplog.set_level(logging.INFO)
    hooks = tq_latency.install_transfer_queue_latency(
        client, max_events=256, slow_call_s=0.000001
    )
    with tq_latency.trace_tq_put():
        client._run_coroutine(client.async_set_custom_meta(None))
    record = next(
        json.loads(r.message.split(" ", 1)[1])
        for r in caplog.records
        if r.message.startswith("tq_put_internal_phases ")
    )
    selected = [
        e
        for e in record["phases"]
        if e["phase"].startswith("controller_") and e.get("controller_request")
    ]
    assert {e["phase"] for e in selected} == {
        "controller_build",
        "controller_serialize",
        "controller_send",
        "controller_response_wait",
        "controller_deserialize",
    }
    assert all(e["controller_request"] == "set_meta" for e in selected)
    native = inspect.unwrap(type(client)._request_controller)
    message = native.__globals__["ZMQMessage"]
    monkeypatch.setattr(
        message,
        "deserialize",
        lambda frames: SimpleNamespace(
            request_type="wrong", body={"message": "bad reply"}
        ),
    )
    with pytest.raises(RuntimeError, match="Expected reply, got wrong"):
        client._run_coroutine(client.async_set_custom_meta(None))
    hooks.close()


def test_tq_client_monitor_reports_blocked_loop_and_retires_on_close(client, caplog):
    from nemo_rl.utils.checkpoint_latency import (
        CheckpointLatencyConfig,
        CheckpointLatencyDiagnostics,
    )

    caplog.set_level(logging.INFO)
    diag = CheckpointLatencyDiagnostics(
        CheckpointLatencyConfig(enabled=True, gc_intervals=False), worker_ip="test"
    )
    diag.begin("monitor-checkpoint")
    hooks = tq_latency.install_transfer_queue_latency(
        client, max_events=256, slow_call_s=0.001, diagnostics=diag
    )
    try:
        client._run_coroutine(asyncio.sleep(0))

        async def block_client_loop():
            time.sleep(1.1)

        client._run_coroutine(block_client_loop())
        client._run_coroutine(asyncio.sleep(0.01))
        records = [
            json.loads(r.message.split(" ", 1)[1])
            for r in caplog.records
            if r.message.startswith("tq_client_loop_delay ")
        ]
        assert records and records[0]["lag_s"] > 0.05
        assert records[0]["checkpoint_id"] == "monitor-checkpoint"
    finally:
        hooks.close()
        client._run_coroutine(asyncio.sleep(0.01))
        diag.close()
    assert hooks._monitor_task.done()
