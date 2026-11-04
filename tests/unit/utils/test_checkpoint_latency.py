# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Fence ownership and bounded collector observations."""

from __future__ import annotations

import ast
import gc
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from nemo_rl.utils import checkpoint_latency as latency


@pytest.fixture
def gate_type():
    # Exercise the production gate without importing the optional GPU/Gym stack.
    source = (
        Path(__file__).parents[3]
        / "nemo_rl/models/generation/generation_cut_capture.py"
    )
    gate = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "_TokenCaptureSnapshotGate"
    )
    namespace = dict(vars(latency))
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[gate], type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )
    return namespace[gate.name]


def events(caplog, name):
    return [
        json.loads(r.message.split(" ", 1)[1])
        for r in caplog.records
        if r.message.startswith(name + " ")
    ]


def test_fence_identifies_running_holder_and_separates_reply_delay(gate_type, caplog):
    caplog.set_level(logging.INFO)
    gate = gate_type()
    diag = latency.CheckpointLatencyDiagnostics(
        latency.CheckpointLatencyConfig(
            enabled=True, gc_intervals=False, progress_interval_s=0.01
        ),
        worker_ip="test",
    )
    gate.diagnostics = diag
    entered, release, fenced = threading.Event(), threading.Event(), threading.Event()

    def terminal():
        gate.enter(operation="terminal", model_call_id="slow-call")
        assert latency.G_GATE_WORK.get().model_call_id == "slow-call"
        entered.set()
        assert release.wait(3)
        gate.exit()
        assert latency.G_GATE_WORK.get() is None

    holder = threading.Thread(target=terminal)
    holder.start()
    assert entered.wait(3)
    epoch = gate.begin_epoch("checkpoint-1")
    closer = threading.Thread(target=lambda: (gate.close_and_wait(epoch), fenced.set()))
    closer.start()
    deadline = time.monotonic() + 3
    while gate._open and time.monotonic() < deadline:
        time.sleep(0.001)
    assert not gate._open and not fenced.is_set()
    release.set()
    holder.join(3)
    closer.join(3)
    assert fenced.is_set()
    gate.reply_ready()
    gate.reopen()
    diag.end("checkpoint-1")
    diag.close()
    rows = events(caplog, "checkpoint_fence")
    closed = next(r for r in rows if r["event"] == "closed")
    drained = next(r for r in rows if r["event"] == "drained")
    reply = next(r for r in rows if r["event"] == "reply_ready")
    assert closed["holders"][0]["model_call_id"] == "slow-call"
    assert drained["last_holder"]["operation"] == "terminal"
    assert reply["drained_to_reply_s"] >= 0
    assert all(r["checkpoint_id"] == "checkpoint-1" for r in rows)
    assert gate._holders == {} and gate._active == 0


def test_late_close_and_nested_permits_keep_gate_semantics(gate_type):
    gate = gate_type()
    diag = latency.CheckpointLatencyDiagnostics(
        latency.CheckpointLatencyConfig(enabled=True, gc_intervals=False),
        worker_ip="test",
    )
    gate.diagnostics = diag
    gate.enter(operation="terminal")
    gate.enter(operation="background_cleanup")
    assert latency.G_GATE_WORK.get().operation == "background_cleanup"
    gate.exit()
    assert latency.G_GATE_WORK.get().operation == "terminal"
    gate.exit()
    epoch = gate.begin_epoch("cancelled-checkpoint")
    gate.reopen()
    gate.close_and_wait(epoch)
    assert gate._open
    assert latency.G_GATE_WORK.get() is None
    diag.close()


def test_gc_reports_real_intervals_without_retaining_objects(caplog):
    caplog.set_level(logging.INFO)
    diag = latency.CheckpointLatencyDiagnostics(
        latency.CheckpointLatencyConfig(enabled=True, max_events=16), worker_ip="test"
    )
    try:
        diag.begin("gc-checkpoint")
        for _ in range(20):
            gc.collect(0)
        diag.end("gc-checkpoint")
        report = events(caplog, "checkpoint_gc_intervals")[-1]
        assert len(report["events"]) <= 16
        assert report["dropped_events"] >= 4
        assert all(
            e["wall_s"] >= 0 and e["thread_cpu_s"] >= 0 for e in report["events"]
        )
    finally:
        diag.close()
    assert diag._on_gc not in gc.callbacks


def test_disabled_diagnostics_install_no_collector_callback():
    before = list(gc.callbacks)
    diag = latency.CheckpointLatencyDiagnostics(
        latency.CheckpointLatencyConfig(), worker_ip="test"
    )
    diag.begin("ignored")
    assert diag.checkpoint_id is None and gc.callbacks == before
    diag.close()


def test_logging_failure_does_not_change_work(monkeypatch):
    def broken(*args: Any, **kwargs: Any):
        raise RuntimeError("logging unavailable")

    monkeypatch.setattr(latency.LOGGER, "info", broken)
    latency.emit_latency("test", checkpoint_id="checkpoint")
