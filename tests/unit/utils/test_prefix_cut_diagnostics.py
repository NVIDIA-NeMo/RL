"""Standalone regression tests for bounded prefix-cut timing summaries."""

import ast
import asyncio
import contextvars
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest

from nemo_rl.utils import prefix_cut_diagnostics as diagnostics


def probe(*, enabled=True):
    return diagnostics.prefix_cut_probe(
        enabled=enabled,
        checkpoint_id="checkpoint-1",
        inventory_digest="inventory-1",
        calls=3,
        deadline=time.monotonic() + 120,
    )


def summary(caplog):
    messages = [
        record.message
        for record in caplog.records
        if record.message.startswith("prefix_cut_probe ")
    ]
    assert len(messages) == 1
    return json.loads(messages[0].split(" ", 1)[1])


def test_disabled_probe_does_not_read_clocks_or_log(monkeypatch, caplog):
    manager = probe(enabled=False)

    def unexpected_clock():
        raise AssertionError("disabled instrumentation read a clock")

    monkeypatch.setattr(diagnostics.time, "perf_counter", unexpected_clock)
    monkeypatch.setattr(diagnostics.time, "thread_time", unexpected_clock)
    with manager, diagnostics.prefix_cut_timing("unused"):
        diagnostics.prefix_cut_count("bytes", 100)
        assert not diagnostics.prefix_cut_probe_active()
    assert not caplog.records


def test_summary_aggregates_sizes_and_inclusive_timings(caplog):
    caplog.set_level(logging.INFO)
    with probe():
        for amount in (10, 20):
            with diagnostics.prefix_cut_timing("record"):
                diagnostics.prefix_cut_count("route_bytes", amount)
                diagnostics.prefix_cut_count("max_route_bytes", amount, maximum=True)
        diagnostics.prefix_cut_wait("control_queue_wait", 0.25)
    result = summary(caplog)
    assert result["calls"] == 3
    assert result["outcome"] == "completed"
    assert result["counts"] == {"route_bytes": 30, "max_route_bytes": 20}
    assert result["phases"]["record"]["calls"] == 2
    assert result["phases"]["record"]["wall_s"] >= 0
    assert result["phases"]["control_queue_wait"]["wall_s"] == 0.25
    assert result["phases"]["control_queue_wait"]["thread_cpu_s"] == 0
    assert not diagnostics.prefix_cut_probe_active()


def test_phase_failure_is_measured_without_changing_exception(caplog):
    caplog.set_level(logging.INFO)
    error = ValueError("injected")
    with pytest.raises(ValueError) as caught:
        with probe(), diagnostics.prefix_cut_timing("route_decode"):
            raise error
    assert caught.value is error
    result = summary(caplog)
    assert result["outcome"] == "interrupted"
    assert result["phases"]["route_decode"]["errors"] == 1
    assert not diagnostics.prefix_cut_probe_active()


def test_cancelled_probe_is_labelled_interrupted(caplog):
    caplog.set_level(logging.INFO)
    with pytest.raises(asyncio.CancelledError):
        with probe(), diagnostics.prefix_cut_timing("batch"):
            raise asyncio.CancelledError
    result = summary(caplog)
    assert result["outcome"] == "interrupted"
    assert result["phases"]["batch"]["errors"] == 1


def test_context_propagates_into_existing_executor(caplog):
    caplog.set_level(logging.INFO)
    with ThreadPoolExecutor(max_workers=1) as executor, probe():
        context = contextvars.copy_context()

        def work():
            with diagnostics.prefix_cut_timing("worker"):
                diagnostics.prefix_cut_count("rows", 4)

        executor.submit(context.run, work).result(timeout=5)
    result = summary(caplog)
    assert result["counts"]["rows"] == 4
    assert result["phases"]["worker"]["calls"] == 1


def test_queued_work_is_distinct_from_execution(caplog):
    caplog.set_level(logging.INFO)
    release = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        blocker = executor.submit(release.wait, 5)
        with probe():
            submitted = time.perf_counter()
            context = contextvars.copy_context()

            def work():
                diagnostics.prefix_cut_wait("queue", time.perf_counter() - submitted)
                with diagnostics.prefix_cut_timing("execution"):
                    diagnostics.prefix_cut_count("rows", 1)

            result = executor.submit(context.run, work)
            try:
                time.sleep(0.04)
            finally:
                release.set()
            result.result(timeout=5)
        blocker.result(timeout=5)
    phases = summary(caplog)["phases"]
    assert phases["queue"]["wall_s"] >= 0.03
    assert phases["execution"]["wall_s"] < phases["queue"]["wall_s"]


def test_logging_failure_does_not_replace_capture_failure(monkeypatch):
    def broken_logger(*args, **kwargs):
        raise RuntimeError("logger unavailable")

    monkeypatch.setattr(diagnostics.logger, "info", broken_logger)
    with pytest.raises(ValueError, match="capture error"):
        with probe():
            raise ValueError("capture error")


def test_actual_cut_control_wrapper_preserves_queue_context_and_result(caplog):
    """Execute the production method without importing GPU-only worker modules."""
    source = (
        Path(__file__).resolve().parents[3]
        / "nemo_rl/models/generation/generation_cut_capture.py"
    )
    tree = ast.parse(source.read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GenerationCutCaptureMixin"
    )
    method = next(
        node
        for node in owner.body
        if getattr(node, "name", None) == "_run_generation_cut_control"
    )
    namespace = dict(
        Any=Any,
        Callable=Callable,
        cast=cast,
        asyncio=asyncio,
        contextvars=contextvars,
        time=time,
        checkpoint_phase=lambda phase: nullcontext(),
        prefix_cut_wait=diagnostics.prefix_cut_wait,
        prefix_cut_timing=diagnostics.prefix_cut_timing,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )
    run_control = namespace["_run_generation_cut_control"]
    caplog.set_level(logging.INFO)

    async def scenario():
        with ThreadPoolExecutor(max_workers=1) as executor:
            worker = SimpleNamespace(_generation_cut_control_executor=executor)
            release = threading.Event()
            blocker = executor.submit(release.wait, 5)

            def operation():
                diagnostics.prefix_cut_count("executed", 1)
                return 42

            with probe():
                task = asyncio.create_task(run_control(worker, operation))
                try:
                    await asyncio.sleep(0.04)
                finally:
                    release.set()
                assert await task == 42
            blocker.result(timeout=5)

    asyncio.run(scenario())
    result = summary(caplog)
    assert result["counts"]["executed"] == 1
    assert result["phases"]["control_queue_wait"]["wall_s"] >= 0.03
    assert result["phases"]["control_execution"]["calls"] == 1
    assert result["phases"]["control_result_delivery"]["calls"] == 1


@pytest.fixture
def owner_loop():
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run)
    thread.start()
    assert ready.wait(2)
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        assert not thread.is_alive()
        loop.close()


def test_snapshot_splits_owner_dispatch_callback_and_return(owner_loop, caplog):
    caplog.set_level(logging.INFO)
    blocked = threading.Event()
    release = threading.Event()

    def block_owner():
        blocked.set()
        release.wait(5)

    owner_loop.call_soon_threadsafe(block_owner)
    assert blocked.wait(2)
    value = object()

    def callback():
        assert asyncio.get_running_loop() is owner_loop
        time.sleep(0.01)
        return value

    def unblock():
        time.sleep(0.04)
        release.set()

    releaser = threading.Thread(target=unblock)
    releaser.start()
    try:
        with probe():
            assert (
                diagnostics.run_engine_snapshot(callback, loop=owner_loop, timeout_s=2)
                is value
            )
    finally:
        release.set()
        releaser.join(2)
    phases = summary(caplog)["phases"]
    assert phases["engine_snapshot_dispatch_wait"]["wall_s"] >= 0.02
    assert phases["engine_snapshot_callback"]["wall_s"] >= 0.008
    assert phases["engine_snapshot_callback"]["errors"] == 0
    assert phases["engine_snapshot_return_wakeup"]["wall_s"] >= 0


def test_snapshot_preserves_callback_exception(owner_loop, caplog):
    caplog.set_level(logging.INFO)
    error = ValueError("lookup failed")

    def callback():
        raise error

    with pytest.raises(ValueError) as caught:
        with probe():
            diagnostics.run_engine_snapshot(callback, loop=owner_loop, timeout_s=2)
    assert caught.value is error
    assert summary(caplog)["phases"]["engine_snapshot_callback"]["errors"] == 1


def test_snapshot_timeout_records_an_unresolved_round_trip(owner_loop, caplog):
    caplog.set_level(logging.INFO)
    blocked = threading.Event()
    release = threading.Event()

    def block_owner():
        blocked.set()
        release.wait(5)

    owner_loop.call_soon_threadsafe(block_owner)
    assert blocked.wait(2)
    try:
        with pytest.raises(TimeoutError):
            with probe():
                diagnostics.run_engine_snapshot(
                    lambda: None, loop=owner_loop, timeout_s=0.02
                )
    finally:
        release.set()
    result = summary(caplog)
    assert result["counts"]["engine_snapshot_timeouts"] == 1
    assert result["phases"]["engine_snapshot_unresolved_wait"]["errors"] == 1
    # Let the scheduled cancellation settle before stopping the fixture's loop.
    asyncio.run_coroutine_threadsafe(asyncio.sleep(0.01), owner_loop).result(2)


def test_unmeasured_snapshot_keeps_the_same_result(owner_loop, caplog):
    value = object()
    assert (
        diagnostics.run_engine_snapshot(lambda: value, loop=owner_loop, timeout_s=2)
        is value
    )
    assert not caplog.records


def test_snapshot_distinguishes_cut_thread_wakeup_from_callback(
    owner_loop, caplog, monkeypatch
):
    caplog.set_level(logging.INFO)
    submit = asyncio.run_coroutine_threadsafe

    def delayed_result(coroutine, loop):
        future = submit(coroutine, loop)
        result = future.result

        def wait(timeout=None):
            value = result(timeout=timeout)
            time.sleep(0.03)
            return value

        monkeypatch.setattr(future, "result", wait)
        return future

    monkeypatch.setattr(diagnostics.asyncio, "run_coroutine_threadsafe", delayed_result)
    with probe():
        assert (
            diagnostics.run_engine_snapshot(lambda: 42, loop=owner_loop, timeout_s=2)
            == 42
        )
    phases = summary(caplog)["phases"]
    assert phases["engine_snapshot_return_wakeup"]["wall_s"] >= 0.02
    assert phases["engine_snapshot_callback"]["calls"] == 1


def test_actual_vllm_adapter_uses_the_split_snapshot_lookup(owner_loop, caplog):
    caplog.set_level(logging.INFO)
    source = (
        Path(__file__).resolve().parents[3]
        / "nemo_rl/models/generation/vllm/vllm_worker_async.py"
    )
    owner = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "_AsyncLLMHTTPClient"
    )
    method = next(
        node
        for node in owner.body
        if getattr(node, "name", None) == "snapshot_routed_experts"
    )
    namespace = dict(Any=Any, run_engine_snapshot=diagnostics.run_engine_snapshot)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )
    chunks = [object(), object()]
    client = SimpleNamespace(
        _engine_loop=owner_loop,
        _direct_route_snapshot_enabled=lambda: False,
        _routed_expert_chunks=lambda request_id: chunks,
    )
    with probe():
        assert (
            namespace["snapshot_routed_experts"](client, "request", timeout_s=2)
            is chunks
        )
    assert summary(caplog)["phases"]["engine_snapshot_callback"]["calls"] == 1


def test_generator_completion_is_not_an_acknowledgement_error(caplog):
    caplog.set_level(logging.INFO)
    with probe():
        with pytest.raises(StopIteration):
            with diagnostics.prefix_cut_timing("seal", expected_stop_iteration=True):
                raise StopIteration
    assert summary(caplog)["phases"]["seal"]["errors"] == 0
