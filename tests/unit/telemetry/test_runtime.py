# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime diagnostics with actual executors and an in-memory OTel reader."""

import ast
import asyncio
import contextvars
import logging
import threading
import time
from collections.abc import AsyncIterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from nemo_rl.telemetry import runtime


@pytest.fixture
def reader():
    from nemo.lens import NemoLensConfig, setup_telemetry
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    from nemo_rl.telemetry import setup

    metric_reader = InMemoryMetricReader()
    setup._TELEMETRY_HANDLE = setup_telemetry(
        NemoLensConfig(enabled=True, traces_enabled=False), metric_reader=metric_reader
    )
    setup._METRICS_ENABLED = True
    return metric_reader


def points(reader, name):
    data = reader.get_metrics_data()
    return (
        [
            point
            for resource in data.resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
            if metric.name == name
            for point in metric.data.data_points
        ]
        if data is not None
        else []
    )


async def wait_until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


def test_queue_wait_is_separate_from_execution_and_counts_balance(reader):
    async def scenario():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        release = threading.Event()
        blocker = loop.run_in_executor(None, release.wait, 2)
        task = asyncio.create_task(runtime.run_in_thread("terminal", lambda: 42))
        try:
            await asyncio.sleep(0.04)
            queued = points(reader, "rl.executor.pending")
            assert queued[0].attributes["rl.executor.state"] == "queued"
            assert queued[0].value == 1
        finally:
            release.set()
        assert await task == 42
        await blocker

    asyncio.run(scenario())
    queue = points(reader, "rl.executor.queue_wait")
    execution = points(reader, "rl.executor.execution")
    assert queue[0].count == execution[0].count == 1
    assert queue[0].sum >= 30
    assert execution[0].sum < queue[0].sum
    assert all(point.value == 0 for point in points(reader, "rl.executor.pending"))
    assert points(reader, "rl.executor.result_delivery")[0].count == 1


def test_cancellation_before_start_retires_queued_work(reader):
    ran = threading.Event()

    async def scenario():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        release = threading.Event()
        blocker = loop.run_in_executor(None, release.wait, 2)
        task = asyncio.create_task(runtime.run_in_thread("terminal", ran.set))
        try:
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0)
            assert all(p.value == 0 for p in points(reader, "rl.executor.pending"))
        finally:
            release.set()
            await blocker

    asyncio.run(scenario())
    assert not ran.is_set()
    assert points(reader, "rl.executor.execution") == []


def test_cancelled_caller_keeps_running_thread_counted(reader):
    started, release, finished = (threading.Event() for _ in range(3))

    def work():
        started.set()
        release.wait(2)
        finished.set()

    async def scenario():
        task = asyncio.create_task(runtime.run_in_thread("terminal", work))
        try:
            await wait_until(started.is_set)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            pending = points(reader, "rl.executor.pending")
            assert (
                next(
                    p.value
                    for p in pending
                    if p.attributes["rl.executor.state"] == "running"
                )
                == 1
            )
        finally:
            release.set()
            await wait_until(finished.is_set)

    asyncio.run(scenario())
    assert all(p.value == 0 for p in points(reader, "rl.executor.pending"))
    assert points(reader, "rl.executor.execution")[0].count == 1
    assert points(reader, "rl.executor.result_delivery") == []


def test_callable_exception_and_context_are_preserved(reader):
    marker = contextvars.ContextVar("marker")
    failure = ValueError("original failure")

    def work():
        assert marker.get() == "request context"
        raise failure

    async def scenario():
        marker.set("request context")
        with pytest.raises(ValueError) as caught:
            await runtime.run_in_thread("terminal", work)
        assert caught.value is failure

    asyncio.run(scenario())
    execution = points(reader, "rl.executor.execution")
    assert execution[0].attributes["rl.executor.status"] == "error"
    assert all(p.value == 0 for p in points(reader, "rl.executor.pending"))


def test_cancel_racing_wrapper_entry_still_counts_execution(reader, monkeypatch):
    launched, allow_entry, started, release = (threading.Event() for _ in range(4))
    original_start = runtime._Submission.start

    def delayed_entry(submission):
        launched.set()  # Native executor has started, before our state transition.
        allow_entry.wait(2)
        return original_start(submission)

    def work():
        started.set()
        release.wait(2)

    monkeypatch.setattr(runtime._Submission, "start", delayed_entry)

    async def scenario():
        task = asyncio.create_task(runtime.run_in_thread("terminal", work))
        try:
            await wait_until(launched.is_set)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0)
            allow_entry.set()
            await wait_until(started.is_set)
            pending = points(reader, "rl.executor.pending")
            assert (
                next(
                    p.value
                    for p in pending
                    if p.attributes["rl.executor.state"] == "running"
                )
                == 1
            )
            assert (
                next(
                    p.value
                    for p in pending
                    if p.attributes["rl.executor.state"] == "queued"
                )
                == 0
            )
        finally:
            allow_entry.set()
            release.set()

    asyncio.run(scenario())
    assert points(reader, "rl.executor.execution")[0].count == 1
    assert all(p.value == 0 for p in points(reader, "rl.executor.pending"))


def test_submit_failure_retires_queue(reader, monkeypatch):
    async def scenario():
        def fail(*args):
            raise RuntimeError("executor closed")

        monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", fail)
        with pytest.raises(RuntimeError, match="executor closed"):
            await runtime.run_in_thread("terminal", lambda: None)

    asyncio.run(scenario())
    assert all(p.value == 0 for p in points(reader, "rl.executor.pending"))


def test_telemetry_failure_does_not_change_result(reader, monkeypatch):
    def fail(*args):
        raise RuntimeError("broken exporter")

    monkeypatch.setattr(runtime, "record_rl_metrics", fail)
    assert asyncio.run(runtime.run_in_thread("terminal", lambda x: x + 1, 4)) == 5


@pytest.mark.parametrize("metrics_only_off", [False, True])
def test_disabled_diagnostics_delegate_and_create_no_probe(
    monkeypatch, metrics_only_off
):
    from nemo_rl.telemetry import setup

    if metrics_only_off:
        from nemo.lens import NemoLensConfig, setup_telemetry

        setup._TELEMETRY_HANDLE = setup_telemetry(
            NemoLensConfig(enabled=True, metrics_enabled=False, exporter="console")
        )
    calls = []

    async def delegated(function, *args, **kwargs):
        calls.append((args, kwargs))
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", delegated)

    async def scenario():
        before = asyncio.all_tasks()
        async with runtime.monitor_event_loop("controller"):
            assert asyncio.all_tasks() == before
            assert await runtime.run_in_thread("terminal", lambda *, x: x, x=7) == 7

    asyncio.run(scenario())
    assert calls == [((), {"x": 7})]


def test_probe_detects_blocked_loop_and_stops_on_exit(reader, monkeypatch):
    monkeypatch.setattr(runtime, "_EVENT_LOOP_PROBE_INTERVAL_S", 0.005)

    async def scenario():
        async with runtime.monitor_event_loop("controller"):
            await asyncio.sleep(0.002)  # Let the probe establish its deadline.
            time.sleep(0.04)  # Deliberately block the monitored loop.
            await asyncio.sleep(0.01)
        assert not any(
            t.get_name().startswith("rl-loop-probe-") for t in asyncio.all_tasks()
        )

    asyncio.run(scenario())
    lag = points(reader, "rl.event_loop.lag")
    assert lag[0].attributes == {"rl.event_loop": "controller"}
    assert lag[0].sum >= 30


def test_probe_is_cleaned_up_when_owner_is_cancelled(reader):
    async def scenario():
        entered = asyncio.Event()

        async def owner():
            async with runtime.monitor_event_loop("controller"):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(owner())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not any(
            t.get_name().startswith("rl-loop-probe-") for t in asyncio.all_tasks()
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("enabled", [False, True])
def test_vllm_http_lifespan_owns_probe_on_serving_loop(reader, monkeypatch, enabled):
    """Exercise the real server setup without importing a vLLM engine or GPU stack."""
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    from nemo_rl.telemetry import setup

    setup._METRICS_ENABLED = enabled
    path = (
        Path(__file__).resolve().parents[3]
        / "nemo_rl/models/generation/vllm/vllm_worker_async.py"
    )
    node = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_setup_vllm_server"
    )
    # Compile the production setup method with its lightweight dependencies;
    # constructors for networking/thread launch are replaced, not its lifespan.
    namespace = {
        "monitor_event_loop": runtime.monitor_event_loop,
        "asynccontextmanager": asynccontextmanager,
        "AsyncIterator": AsyncIterator,
        "Mapping": Mapping,
        "Any": Any,
        "cast": cast,
        "asyncio": asyncio,
        "_get_node_ip_local": lambda: "127.0.0.1",
        "_get_free_port_local": lambda *_: 9000,
        "DEFAULT_GENERATION_PORT_RANGE_LOW": 9000,
        "DEFAULT_GENERATION_PORT_RANGE_HIGH": 9010,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    lifecycle = []
    original_state = {"api_state": object()}

    @asynccontextmanager
    async def api_lifespan(_app):
        lifecycle.append("startup")
        try:
            yield original_state
        finally:
            lifecycle.append("shutdown")

    def configure_app(app):
        app.router.lifespan_context = api_lifespan
        return app

    worker = SimpleNamespace(
        _reserved_socket=None,
        _sparse_refit_receiver=None,
        _setup_vllm_openai_api_server=configure_app,
        cfg={"port_range_low": 9000, "port_range_high": 9010},
    )
    server_factory = Mock(
        side_effect=lambda *, config: SimpleNamespace(config=config, run=lambda: None)
    )
    monkeypatch.setattr(uvicorn, "Server", server_factory)
    monkeypatch.setattr(logging.getLogger("uvicorn.access"), "filters", [])
    with monkeypatch.context() as patch:
        patch.setattr(threading, "Thread", Mock())
        _, _, server = namespace["_setup_vllm_server"](worker)
    app = server.config.app

    async def scenario():
        async with app.router.lifespan_context(app) as state:
            assert state is original_state
            assert lifecycle == ["startup"]
            probes = [
                t
                for t in asyncio.all_tasks()
                if t.get_name() == "rl-loop-probe-vllm_http"
            ]
            assert len(probes) == int(enabled)
            assert all(t.get_loop() is asyncio.get_running_loop() for t in probes)
        assert not any(
            t.get_name() == "rl-loop-probe-vllm_http" for t in asyncio.all_tasks()
        )
        assert lifecycle == ["startup", "shutdown"]

    asyncio.run(scenario())
