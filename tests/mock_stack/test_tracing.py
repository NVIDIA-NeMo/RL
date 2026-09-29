# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import pytest
from opentelemetry import trace as otel

from tests.mock_stack.tracing import TraceRun


def read_spans(trace):
    return [json.loads(line) for line in trace.path.read_text().splitlines()]


def test_trace_keeps_concurrent_calls_and_failure(tmp_path):
    with TraceRun(tmp_path) as trace:

        async def call(index):
            with trace.span("test.call", index=index):
                await asyncio.sleep(0.01)
                if index == 1:
                    raise ValueError("test failure")
                return index

        async def calls():
            return await asyncio.gather(call(0), call(1), return_exceptions=True)

        results = asyncio.run(calls())
        assert results[0] == 0
        assert isinstance(results[1], ValueError)
    spans = read_spans(trace)
    calls = sorted(
        (s for s in spans if s["name"] == "test.call"),
        key=lambda s: s["attributes"]["index"],
    )
    assert len(calls) == 2
    assert calls[1]["start_time"] < calls[0]["end_time"]
    assert calls[0]["parent_id"] == calls[1]["parent_id"]
    assert calls[1]["status"]["status_code"] == "ERROR"
    assert len({s["context"]["trace_id"] for s in spans}) == 1
    root = next(s for s in spans if s["name"] == "test.cpu_run")
    assert root["status"]["status_code"] == "UNSET"


def test_trace_records_despite_disabled_ambient_sampling(tmp_path, monkeypatch):
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "always_off")
    parent = otel.NonRecordingSpan(otel.SpanContext(123, 456, False))
    with otel.use_span(parent):
        with TraceRun(tmp_path) as trace:
            # Exercise the HTTP server's separate-thread context behavior.
            def work():
                with trace.span("test.thread"):
                    pass

            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(work).result()
        assert otel.get_current_span() is parent
    spans = read_spans(trace)
    assert {s["name"] for s in spans} == {"test.cpu_run", "test.thread"}
    assert len({s["context"]["trace_id"] for s in spans}) == 1
    root = next(s for s in spans if s["name"] == "test.cpu_run")
    assert root["parent_id"] is None


def test_trace_flushes_when_run_raises(tmp_path):
    previous = otel.get_current_span()
    with pytest.raises(ValueError, match="run failed"):
        with TraceRun(tmp_path) as trace:
            with trace.span("test.failure"):
                raise ValueError("run failed")
    assert otel.get_current_span() is previous
    spans = read_spans(trace)
    assert len(spans) == 2
    assert all(s["status"]["status_code"] == "ERROR" for s in spans)
