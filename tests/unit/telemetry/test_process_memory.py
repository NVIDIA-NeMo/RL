# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-process memory reporting.

The process tree is faked for the attribution rules, and one test forks a real
child so the "a forked child is charged to its parent" claim is checked against
/proc rather than against the fake.
"""

import os
import subprocess
import sys
import types
from types import SimpleNamespace

import psutil
import pytest

import nemo_rl.telemetry.process_memory as process_memory
from nemo_rl.telemetry.process_memory import (
    DRIVER_WORKER_GROUP,
    PROCESS_PID_ATTR,
    RL_PROCESS_MEMORY_PSS_METRIC,
    RL_WORKER_GROUP_ATTR,
    process_tree_pss,
    start_process_memory_reporting,
    stop_process_memory_reporting,
)

_MIB = 1024 * 1024


@pytest.fixture(autouse=True)
def _reset_process_memory_state():
    """Whether this process is reporting is process-global; none leaks."""
    stop_process_memory_reporting()
    yield
    stop_process_memory_reporting()


class _FakeProcess:
    """A ``psutil.Process`` reduced to the two calls ``process_tree_pss`` makes."""

    def __init__(self, pss=0, *, children=(), error=None):
        self._pss = pss
        self._children = list(children)
        self._error = error

    def memory_full_info(self):
        if self._error is not None:
            raise self._error
        return SimpleNamespace(pss=self._pss)

    def children(self, recursive=False):
        assert recursive, "grandchildren are the parent's too"
        return self._children


def _start_exporting_telemetry():
    """Install an exporting telemetry handle; returns its metric reader."""
    from nemo.lens import NemoLensConfig, setup_telemetry
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    import nemo_rl.telemetry.setup as setup_mod

    reader = InMemoryMetricReader()
    setup_mod._TELEMETRY_HANDLE = setup_telemetry(
        NemoLensConfig(enabled=True), metric_reader=reader
    )
    return reader


def _metrics(data, name):
    """Every metric called *name* in one already-collected metrics batch."""
    if data is None:
        return []
    return [
        metric
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == name
    ]


def _points(data, name):
    return [
        point for metric in _metrics(data, name) for point in metric.data.data_points
    ]


def test_descendants_are_charged_to_the_process_that_forked_them():
    root = _FakeProcess(100, children=[_FakeProcess(20), _FakeProcess(30)])
    assert process_tree_pss(root) == 150


@pytest.mark.parametrize(
    "error", [psutil.NoSuchProcess(pid=1234), psutil.AccessDenied(pid=1234)]
)
def test_a_descendant_that_cannot_be_read_costs_only_its_own_share(error):
    root = _FakeProcess(100, children=[_FakeProcess(error=error), _FakeProcess(30)])
    assert process_tree_pss(root) == 130


def test_an_unreadable_root_raises_rather_than_reporting_its_children_alone():
    root = _FakeProcess(error=psutil.AccessDenied(pid=1), children=[_FakeProcess(30)])
    with pytest.raises(psutil.AccessDenied):
        process_tree_pss(root)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PSS is Linux-only")
def test_a_real_forked_child_is_counted_in_its_parents_pss():
    before = process_tree_pss()
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            # bytes() of a repeated non-zero byte, so every page is touched and
            # private: zero pages could stay shared with the kernel's zero page.
            "import sys, time; blob = b'x' * (64 << 20); "
            "print('ready', flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        grown = process_tree_pss() - before
    finally:
        child.kill()
        child.wait()
    # The child's 64 MiB is all private, so all of it is charged here; the
    # interpreter pages it shares with this process are split, not doubled.
    assert grown > 56 * _MIB


def test_each_process_reports_one_point_with_its_group_rank_and_pid(monkeypatch):
    monkeypatch.setattr(process_memory, "process_tree_pss", lambda: 123 * _MIB)
    reader = _start_exporting_telemetry()

    start_process_memory_reporting(rank=3, worker_group="lm_policy")

    (point,) = _points(reader.get_metrics_data(), RL_PROCESS_MEMORY_PSS_METRIC)
    assert point.value == 123 * _MIB
    assert dict(point.attributes) == {
        RL_WORKER_GROUP_ATTR: "lm_policy",
        "nv.dl.rank": 3,
        PROCESS_PID_ATTR: os.getpid(),
    }


def test_the_series_is_exported_as_a_non_monotonic_sum(monkeypatch):
    """Summable by group or host, which a gauge does not promise a backend."""
    from opentelemetry.sdk.metrics.export import Sum

    monkeypatch.setattr(process_memory, "process_tree_pss", lambda: 1)
    reader = _start_exporting_telemetry()
    start_process_memory_reporting(rank=0, worker_group="lm_policy")

    (metric,) = _metrics(reader.get_metrics_data(), RL_PROCESS_MEMORY_PSS_METRIC)
    assert isinstance(metric.data, Sum)
    assert not metric.data.is_monotonic
    assert metric.unit == "By"


def test_a_process_outside_any_worker_group_is_reported_without_the_label(
    monkeypatch,
):
    monkeypatch.setattr(process_memory, "process_tree_pss", lambda: 1)
    reader = _start_exporting_telemetry()

    start_process_memory_reporting(rank=0, worker_group=None)

    (point,) = _points(reader.get_metrics_data(), RL_PROCESS_MEMORY_PSS_METRIC)
    assert RL_WORKER_GROUP_ATTR not in point.attributes


def test_nothing_is_reported_before_start_or_after_stop(monkeypatch):
    monkeypatch.setattr(process_memory, "process_tree_pss", lambda: 1)
    reader = _start_exporting_telemetry()
    from nemo_rl.telemetry.metrics import create_rl_metric_instruments

    create_rl_metric_instruments()
    assert _points(reader.get_metrics_data(), RL_PROCESS_MEMORY_PSS_METRIC) == []

    start_process_memory_reporting(rank=0, worker_group="lm_policy")
    assert len(_points(reader.get_metrics_data(), RL_PROCESS_MEMORY_PSS_METRIC)) == 1

    stop_process_memory_reporting()
    assert _points(reader.get_metrics_data(), RL_PROCESS_MEMORY_PSS_METRIC) == []


def test_a_failed_read_costs_this_point_and_does_not_raise(monkeypatch):
    def _explode():
        raise OSError("smaps_rollup unreadable")

    monkeypatch.setattr(process_memory, "process_tree_pss", _explode)
    reader = _start_exporting_telemetry()
    start_process_memory_reporting(rank=0, worker_group="lm_policy")

    assert _points(reader.get_metrics_data(), RL_PROCESS_MEMORY_PSS_METRIC) == []


def test_the_point_label_is_the_one_the_resource_uses():
    """Typed in two modules; a rename on one side would split the join."""
    import nemo_rl.telemetry.setup as setup_mod

    assert RL_WORKER_GROUP_ATTR == setup_mod._WORKER_GROUP_ATTR


def _telemetry_env(monkeypatch):
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "true")
    monkeypatch.setenv("NEMO_RL_OTEL_EXPORTER", "console")


def test_the_driver_reports_itself_as_the_driver(monkeypatch):
    from nemo_rl.telemetry.setup import init_telemetry_driver, shutdown_telemetry

    _telemetry_env(monkeypatch)
    # The driver also starts the host-memory monitor, which asks Ray for nodes.
    ray = types.ModuleType("ray")
    ray.is_initialized = lambda: True
    ray.nodes = lambda: []
    monkeypatch.setitem(sys.modules, "ray", ray)
    config = SimpleNamespace(telemetry=None, logger=None, policy=None, cluster=None)

    handle = init_telemetry_driver(config, algorithm="grpo")
    assert handle is not None and handle.is_exporting
    assert process_memory._POINT_ATTRIBUTES is not None
    assert process_memory._POINT_ATTRIBUTES[RL_WORKER_GROUP_ATTR] == DRIVER_WORKER_GROUP
    assert process_memory._POINT_ATTRIBUTES["nv.dl.rank"] == 0

    shutdown_telemetry(timeout_ms=100)
    assert process_memory._POINT_ATTRIBUTES is None


def test_a_worker_reports_the_group_and_rank_it_was_launched_with(monkeypatch):
    from nemo_rl.telemetry.setup import init_telemetry_worker, shutdown_telemetry

    _telemetry_env(monkeypatch)
    monkeypatch.setenv("NRL_WORKER_GROUP", "lm_policy")

    init_telemetry_worker(rank=2, world_size=4)
    assert process_memory._POINT_ATTRIBUTES is not None
    assert process_memory._POINT_ATTRIBUTES[RL_WORKER_GROUP_ATTR] == "lm_policy"
    assert process_memory._POINT_ATTRIBUTES["nv.dl.rank"] == 2

    shutdown_telemetry(timeout_ms=100)


def test_an_explicit_worker_group_wins_over_the_environment(monkeypatch):
    """As it does on the resource, for actors built outside RayWorkerGroup."""
    from nemo_rl.telemetry.setup import init_telemetry_worker, shutdown_telemetry

    _telemetry_env(monkeypatch)
    monkeypatch.setenv("NRL_WORKER_GROUP", "lm_policy")

    init_telemetry_worker(rank=0, world_size=1, worker_group="reward_model")
    assert process_memory._POINT_ATTRIBUTES[RL_WORKER_GROUP_ATTR] == "reward_model"

    shutdown_telemetry(timeout_ms=100)


def test_a_worker_with_telemetry_off_reports_nothing(monkeypatch):
    from nemo_rl.telemetry.setup import init_telemetry_worker

    monkeypatch.delenv("NEMO_RL_OTEL_ENABLED", raising=False)

    assert init_telemetry_worker(rank=0, world_size=1) is None
    assert process_memory._POINT_ATTRIBUTES is None
