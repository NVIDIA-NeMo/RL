"""Owner-loop diagnostics with real asyncio and lightweight output objects."""

import asyncio
import json
import logging
import time
from types import SimpleNamespace

import pytest

from nemo_rl.utils import vllm_engine_diagnostics as diagnostics


def objects():
    class State:
        def __init__(self):
            self.routed_experts_chunks = [object(), object()]
            self.calls = 0

        def _new_completion_output(self, token_ids, finish_reason, stop_reason):
            self.calls += 1
            return SimpleNamespace(routed_experts=SimpleNamespace(nbytes=1024))

    class Processor:
        def __init__(self):
            self.state = State()
            self.result = object()
            self.calls = 0

        def process_outputs(self, outputs, *args, **kwargs):
            self.calls += 1
            for output in outputs:
                self.state._new_completion_output([], output.finish_reason, None)
            return self.result

    processor = Processor()
    observer = diagnostics.EngineLoopDiagnostics(
        processor, State, chunk_size=128, torch_threads=16
    )
    return State, processor, observer


def messages(caplog, event):
    prefix = event + " "
    return [
        json.loads(record.message[len(prefix) :])
        for record in caplog.records
        if record.message.startswith(prefix)
    ]


def test_original_results_counts_and_hooks_are_preserved(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    State, processor, observer = objects()
    original_state = State._new_completion_output
    original_process = processor.process_outputs

    async def scenario():
        observer.start()
        try:
            result = processor.process_outputs(
                [
                    SimpleNamespace(finish_reason=None),
                    SimpleNamespace(finish_reason="stop"),
                ]
            )
            assert result is processor.result
            assert processor.calls == 1 and processor.state.calls == 2
        finally:
            await observer.aclose()
            await observer.aclose()
        assert observer._task is None

    asyncio.run(scenario())
    assert State._new_completion_output is original_state
    assert processor.process_outputs == original_process
    start = messages(caplog, "vllm_engine_probe_start")[0]
    assert start["output_proc_chunk_size"] == 128
    assert start["torch_num_threads"] == 16
    assert start["omp_num_threads"] == "16"
    report = messages(caplog, "vllm_engine_probe")[-1]
    assert report["counts"] == dict(
        output_chunks=1,
        request_outputs=2,
        finished_outputs=1,
        terminal_route_chunks=2,
        terminal_route_bytes=1024,
    )
    assert report["phases"]["terminal_output_build"]["calls"] == 1
    assert report["phases"]["process_outputs"]["calls"] == 1


def test_output_errors_are_preserved_and_counted(caplog):
    caplog.set_level(logging.INFO)
    State, processor, _ = objects()
    error = ValueError("output processing failed")

    def failing(state, *args, **kwargs):
        raise error

    State._new_completion_output = failing
    observer = diagnostics.EngineLoopDiagnostics(
        processor, State, chunk_size=128, torch_threads=16
    )

    async def scenario():
        observer.start()
        try:
            with pytest.raises(ValueError) as caught:
                processor.process_outputs([SimpleNamespace(finish_reason="stop")])
            assert caught.value is error
        finally:
            await observer.aclose()

    asyncio.run(scenario())
    phases = messages(caplog, "vllm_engine_probe")[-1]["phases"]
    assert phases["process_outputs"]["errors"] == 1
    assert phases["terminal_output_build"]["errors"] == 1


def test_observation_failure_does_not_fail_generation(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    _, processor, observer = objects()

    def broken_record(*args, **kwargs):
        raise RuntimeError("diagnostic failed")

    monkeypatch.setattr(observer, "_record", broken_record)

    async def scenario():
        observer.start()
        try:
            assert (
                processor.process_outputs([SimpleNamespace(finish_reason="stop")])
                is processor.result
            )
        finally:
            await observer.aclose()

    asyncio.run(scenario())
    assert len(messages(caplog, "vllm_engine_probe_error")) == 1


def test_completion_hook_is_shared_but_measurements_are_isolated(caplog):
    caplog.set_level(logging.INFO)
    State, processor, first = objects()
    second_processor = type(processor)()
    second = diagnostics.EngineLoopDiagnostics(
        second_processor, State, chunk_size=128, torch_threads=16
    )
    original = State._new_completion_output

    async def scenario():
        first.start()
        second.start()
        try:
            processor.process_outputs([SimpleNamespace(finish_reason="stop")])
            await first.aclose()
            assert State._new_completion_output is not original
            second_processor.process_outputs([SimpleNamespace(finish_reason="stop")])
        finally:
            await first.aclose()
            await second.aclose()

    asyncio.run(scenario())
    assert State._new_completion_output is original
    reports = messages(caplog, "vllm_engine_probe")
    assert len(reports) == 2
    assert all(report["counts"]["finished_outputs"] == 1 for report in reports)


def test_owner_loop_probe_detects_a_block_and_logs_are_bounded(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(diagnostics, "_LOOP_SAMPLE_INTERVAL_S", 0.005)
    _, processor, observer = objects()

    async def scenario():
        observer.start()
        try:
            await asyncio.sleep(0)
            time.sleep(0.04)
            await asyncio.sleep(0.015)
            for _ in range(100):
                processor.process_outputs([SimpleNamespace(finish_reason=None)])
        finally:
            await observer.aclose()

    asyncio.run(scenario())
    reports = messages(caplog, "vllm_engine_probe")
    assert len(reports) == 1
    assert reports[0]["phases"]["owner_loop_lag"]["max_wall_s"] >= 0.02
    assert reports[0]["counts"]["output_chunks"] == 100


def test_install_failure_restores_the_shared_completion_hook():
    State, processor, _ = objects()
    original = State._new_completion_output

    class ReadOnly:
        @property
        def process_outputs(self):
            return processor.process_outputs

    observer = diagnostics.EngineLoopDiagnostics(
        ReadOnly(), State, chunk_size=128, torch_threads=16
    )

    async def scenario():
        with pytest.raises(AttributeError):
            observer.start()
        await observer.aclose()

    asyncio.run(scenario())
    assert State._new_completion_output is original


def test_slow_output_logging_is_rate_limited(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(diagnostics, "_SLOW_OPERATION_S", 0)
    _, processor, observer = objects()

    async def scenario():
        observer.start()
        observer.start()
        try:
            for _ in range(100):
                processor.process_outputs([SimpleNamespace(finish_reason="stop")])
        finally:
            await observer.aclose()

    asyncio.run(scenario())
    slow = messages(caplog, "vllm_engine_slow")
    assert len(slow) == 2
    assert {record["phase"] for record in slow} == {
        "terminal_output_build",
        "process_outputs",
    }
    assert len(messages(caplog, "vllm_engine_probe_start")) == 1


def test_periodic_report_and_uninstrumented_calls(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(diagnostics, "_LOOP_SAMPLE_INTERVAL_S", 0.002)
    monkeypatch.setattr(diagnostics, "_REPORT_INTERVAL_S", 0.008)
    _, processor, observer = objects()

    async def scenario():
        observer.start()
        try:
            # A class-level wrapper must pass through calls outside this owner.
            processor.state._new_completion_output([], "stop", None)
            await asyncio.sleep(0.025)
        finally:
            await observer.aclose()

    asyncio.run(scenario())
    reports = messages(caplog, "vllm_engine_probe")
    assert any(record["reason"] == "periodic" for record in reports)
    assert all(record["counts"].get("finished_outputs", 0) == 0 for record in reports)
    assert all(
        record["finished_unix_s"] >= record["started_unix_s"] for record in reports
    )
