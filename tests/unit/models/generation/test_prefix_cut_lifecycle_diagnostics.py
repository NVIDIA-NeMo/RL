# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""CPU regressions for cut cleanup ordering and non-reentrant GC diagnostics."""

from __future__ import annotations

import ast
import asyncio
import gc
import json
import logging
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_rl.models.generation.vllm.packed_logprobs import PackedSelectedLogprobs
from tools.model_diagnostics import prefix_cut_benchmark as benchmark
from tools.model_diagnostics.prefix_cut_benchmark_diagnostics import (
    BenchmarkDiagnostics,
)

G_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def shutdown_worker() -> Any:
    """Run the production lifecycle methods without importing GPU packages."""
    source = G_ROOT / "nemo_rl/models/generation/vllm/vllm_worker_async.py"
    tree = ast.parse(source.read_text())
    worker_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "VllmAsyncGenerationWorkerImpl"
    )
    methods = [
        node
        for node in worker_class.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name in {"shutdown", "_drain_generation_cut_cleanup"}
    ]
    extracted = ast.ClassDef(
        name="Worker", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    namespace = {
        "asyncio": asyncio,
        "gc": SimpleNamespace(collect=lambda: None),
        "torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
        "shutdown_telemetry": lambda: None,
        "LOGGER": logging.getLogger(__name__),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )
    worker = namespace["Worker"]()
    worker.server_thread = None
    worker._http_engine_client = None
    worker._sparse_refit_receiver = None
    worker.llm = None
    return worker


@pytest.mark.asyncio
async def test_http_cancellation_can_submit_and_drain_cleanup_before_reader_closes(
    shutdown_worker: Any,
) -> None:
    worker = shutdown_worker
    executor = ThreadPoolExecutor(max_workers=1)
    worker._generation_cut_cleanup_executor = executor
    reopened, started, cleaned = threading.Event(), threading.Event(), threading.Event()
    worker._token_capture_snapshot_gate = SimpleNamespace(reopen=reopened.set)
    worker.http_server = SimpleNamespace(
        config=SimpleNamespace(timeout_graceful_shutdown=None), should_exit=False
    )
    errors = []

    async def request() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            assert reopened.is_set()

            def cleanup() -> None:
                time.sleep(0.02)
                cleaned.set()

            try:
                executor.submit(cleanup)
            except RuntimeError as error:
                errors.append(error)

    async def http_main() -> None:
        asyncio.create_task(request())
        while not worker.http_server.should_exit:
            await asyncio.sleep(0.001)
        # asyncio.run cancels/drains the live request before thread.join returns.

    async def close_reader() -> None:
        assert cleaned.is_set(), "Reader closed before abort cleanup drained"

    worker._generation_prefix_reader = SimpleNamespace(aclose=close_reader)
    worker.server_thread = threading.Thread(
        target=lambda: asyncio.run(http_main()), daemon=True
    )
    worker.server_thread.start()
    assert await asyncio.to_thread(started.wait, 2)
    try:
        assert await worker.shutdown()
        assert not errors
        assert cleaned.is_set()
        assert worker._generation_cut_cleanup_executor is None
        assert worker.http_server.config.timeout_graceful_shutdown == 30
        assert await worker.shutdown(), (
            "Shutdown must tolerate an already drained executor"
        )
    finally:
        worker.http_server.should_exit = True
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_cleanup_drain_times_out_and_reports_failure(
    shutdown_worker: Any,
) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    executor.submit(release.wait)
    shutdown_worker._generation_cut_cleanup_executor = executor
    try:
        with pytest.raises(TimeoutError):
            await shutdown_worker._drain_generation_cut_cleanup(timeout_s=0.01)
        assert shutdown_worker._generation_cut_cleanup_executor is None
    finally:
        release.set()
        executor.shutdown(wait=True)

    async def failed_drain() -> None:
        raise TimeoutError("cleanup deadline")

    shutdown_worker._drain_generation_cut_cleanup = failed_drain
    assert await shutdown_worker.shutdown() is False


def read_events(directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in next(directory.glob("events-*.jsonl")).read_text().splitlines()
    ]


def test_gc_callback_records_cpu_counts_and_final_drain_without_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diagnostics = BenchmarkDiagnostics(str(tmp_path))
    diagnostics._closed.set()
    diagnostics._watchdog.join(timeout=2)
    monkeypatch.setattr(gc, "get_threshold", lambda: (700, 10, 10))
    monkeypatch.setattr(gc, "get_count", lambda: (701, 2, 3))
    # A callback can interrupt either lock, including while queue.put holds its mutex.
    with diagnostics._events.mutex, diagnostics._guard:
        diagnostics._gc_callback("start", {"generation": 2})
        diagnostics._gc_callback(
            "stop", {"generation": 2, "collected": 17, "uncollectable": 2}
        )
    diagnostics.close()
    events = [event for event in read_events(tmp_path) if event["event"] == "gc_pause"]
    assert len(events) == 1
    event = events[0]
    assert event["collected"] == 17 and event["uncollectable"] == 2
    assert event["thresholds"] == [700, 10, 10]
    assert event["counts_before"] == [701, 2, 3]
    assert event["thread_cpu_s"] >= 0 and event["process_cpu_s"] >= 0
    assert diagnostics.summary()["gc"]["2"]["collected"] == 17


def test_gc_zero_collected_full_scan_remains_visible(tmp_path: Path) -> None:
    diagnostics = BenchmarkDiagnostics(str(tmp_path))
    try:
        diagnostics._gc_callback("start", {"generation": 2})
        diagnostics._gc_callback(
            "stop", {"generation": 2, "collected": 0, "uncollectable": 0}
        )
        assert diagnostics.summary()["gc"]["2"]["count"] >= 1
    finally:
        diagnostics.close()
    assert any(
        event["event"] == "gc_pause" and event["collected"] == 0
        for event in read_events(tmp_path)
    )


def test_heap_census_is_explicit_and_does_not_retain_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diagnostics = BenchmarkDiagnostics(str(tmp_path))
    refs = []

    class Tracked:
        def __sizeof__(self) -> int:
            pytest.fail("Census must not call arbitrary user sizing methods")

    def objects() -> list[Any]:
        value = Tracked()
        refs.append(weakref.ref(value))
        return [value, [None] * 1000]

    monkeypatch.setattr(gc, "get_objects", objects)
    try:
        # The callback only collects lightweight timings, never a heap census.
        diagnostics._gc_callback("start", {"generation": 2})
        diagnostics._gc_callback("stop", {"generation": 2})
        assert not refs
        result = diagnostics.heap_summary()
        assert result["tracked_objects"] == 2
        assert result["containers"]["list"]["items"] == 1000
        assert result["containers"]["list"]["shallow_bytes"] >= 8000
        assert refs[0]() is None
        json.dumps(result)
    finally:
        diagnostics.close()


@pytest.mark.asyncio
async def test_heap_rpc_reports_retained_engine_counts(tmp_path: Path) -> None:
    source = G_ROOT / "tools/model_diagnostics/prefix_cut_benchmark_worker.py"
    tree = ast.parse(source.read_text())
    worker_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(
        node
        for node in worker_class.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "benchmark_heap_summary"
    )

    class FlatLogprobs:
        start_indices = [0, 1, 2]
        logprobs = [0.0, 0.1, 0.2, 0.3]

    namespace = {
        "Any": Any,
        "asyncio": asyncio,
        "FlatLogprobs": FlatLogprobs,
        "PackedSelectedLogprobs": PackedSelectedLogprobs,
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    diagnostics = BenchmarkDiagnostics(str(tmp_path))
    state = SimpleNamespace(
        detokenizer=SimpleNamespace(output_token_ids=[1, 2, 3]),
        routed_experts_chunks=[None, None],
        logprobs_processor=SimpleNamespace(logprobs=FlatLogprobs()),
    )
    worker = SimpleNamespace(
        _bench_diagnostics=diagnostics,
        llm=SimpleNamespace(
            output_processor=SimpleNamespace(request_states={"request": state}),
            engine_core=SimpleNamespace(outputs_queue=SimpleNamespace(qsize=lambda: 4)),
        ),
        _capture_registry_lock=threading.Lock(),
        _capture_calls={1: None},
        _completed_capture_calls={},
    )
    try:
        result = await namespace["benchmark_heap_summary"](worker)
        assert result["engine_retained"] == {
            "request_states": 1,
            "output_queue_depth": 4,
            "detokenizer_output_tokens": 3,
            "routed_expert_fragments": 2,
            "flat_logprob_positions": 3,
            "flat_logprob_values": 4,
            "active_capture_calls": 1,
            "completed_capture_calls": 0,
        }
    finally:
        diagnostics.close()


@pytest.mark.asyncio
async def test_heap_artifacts_keep_diagnostic_failure_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def worker_call(generation: Any, method: str) -> dict[str, int]:
        assert method == "benchmark_heap_summary"
        if generation == "failed":
            raise TimeoutError("census exceeded RPC deadline")
        return {"tracked_objects": 7}

    monkeypatch.setattr(benchmark, "worker_call", worker_call)
    await benchmark.collect_heap_summaries(
        ["good", "failed"], tmp_path, label="after-000"
    )
    artifact = json.loads((tmp_path / "heap-after-000.json").read_text())
    assert artifact["workers"][0] == {"tracked_objects": 7}
    assert "TimeoutError" in artifact["workers"][1]["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("heap_census", [True, False])
@pytest.mark.parametrize("slots", [[2, 4], [4, 4]])
async def test_driver_census_runs_between_snapshot_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, heap_census: bool, slots: list[int]
) -> None:
    events = []
    names = []

    class Load:
        errors = completed = 0

        def __init__(self, *args: Any) -> None:
            names.append(args[-1])

        def resize(self, size: int) -> None:
            pass

        def check_healthy(self) -> None:
            pass

        async def close(self) -> None:
            pass

    async def census(generations: Any, output: Path, *, label: str) -> None:
        events.append(label)

    async def snapshot(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append(f"snapshot-{kwargs['round_index']}")
        return {"timings": {}, "workers": []}

    def worker_status(generation: Any, method: str) -> dict[str, int]:
        assert method == "benchmark_status"
        events.append("status")
        return {"output_queue_depth": 7}

    monkeypatch.setattr(benchmark, "Load", Load)
    monkeypatch.setattr(benchmark, "collect_heap_summaries", census)
    monkeypatch.setattr(benchmark, "run_snapshot", snapshot)
    monkeypatch.setattr(benchmark, "worker_call", worker_status)
    args = SimpleNamespace(
        output=str(tmp_path),
        outstanding=2,
        warmup_seconds=0,
        rounds=2,
        interval_seconds=0,
        inventories=64,
        cut_budget=30,
        heap_census=heap_census,
        slots=slots,
    )
    generation = SimpleNamespace(
        dp_openai_server_base_urls=["http://unused"], http_server_api_key="unused"
    )
    await benchmark.exercise(
        generations=[generation, generation],
        storage=None,
        config={
            "policy": {"model_name": "unused"},
            "token_capture": {"staging_partition": "test"},
        },
        args=args,
        auth_token="unused",
        wandb_run=SimpleNamespace(log=lambda *args, **kwargs: None),
    )
    assert names == [
        f"engine-{index}-slots-{count}" for index, count in enumerate(slots)
    ]
    assert len(set(names)) == len(slots)
    if heap_census:
        assert events == [
            "warmup",
            "snapshot-0",
            "after-000",
            "snapshot-1",
            "after-001",
        ]
        assert not (tmp_path / "status-warmup.json").exists()
    else:
        assert events == ["status", "status", "snapshot-0", "snapshot-1"]
        status = json.loads((tmp_path / "status-warmup.json").read_text())
        assert status["workers"] == [{"output_queue_depth": 7}] * 2
