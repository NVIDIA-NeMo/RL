# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the host-memory monitor.

Ray and its Prometheus endpoints are both faked: the point of these is the
attribution and the never-raises guarantee, neither of which needs a cluster.
"""

import sys
import types
from types import SimpleNamespace

import pytest

import nemo_rl.telemetry.host_memory as host_memory
from nemo_rl.telemetry.host_memory import (
    HOST_NAME_ATTR,
    NO_WORKER_GROUPS,
    RL_HOST_MEMORY_TOTAL_METRIC,
    RL_HOST_MEMORY_USED_METRIC,
    RL_WORKER_GROUPS_ATTR,
    HostMemoryMonitor,
    forget_worker_group_nodes,
    parse_memory_samples,
    ray_nodes,
    register_worker_group_nodes,
    start_host_memory_monitoring,
    stop_host_memory_monitoring,
    worker_groups_on_node,
)

_USED_BYTES = 16_000_000_000.0
_TOTAL_BYTES = 540_000_000_000.0

#: A raylet's payload, cut down to the two series that are read plus one that
#: is not, so "ignores everything else" is actually exercised.
_SCRAPE = f"""\
# HELP ray_node_mem_used Memory in use on the node.
# TYPE ray_node_mem_used gauge
ray_node_mem_used{{SessionName="s"}} {_USED_BYTES}
# HELP ray_node_mem_total Memory installed on the node.
# TYPE ray_node_mem_total gauge
ray_node_mem_total{{SessionName="s"}} {_TOTAL_BYTES}
# HELP ray_node_cpu_utilization CPU percentage.
# TYPE ray_node_cpu_utilization gauge
ray_node_cpu_utilization{{SessionName="s"}} 42.0
"""


@pytest.fixture(autouse=True)
def _reset_host_memory_state():
    """Both the placement map and the monitor are process-global; neither leaks."""
    host_memory._WORKER_GROUP_NODES.clear()
    stop_host_memory_monitoring()
    yield
    host_memory._WORKER_GROUP_NODES.clear()
    stop_host_memory_monitoring()


def _fake_ray(monkeypatch, nodes):
    """Install a ``ray`` module reporting *nodes* from ``ray.nodes()``."""
    module = types.ModuleType("ray")
    module.is_initialized = lambda: True
    module.nodes = lambda: nodes
    monkeypatch.setitem(sys.modules, "ray", module)
    return module


def _node(node_id="n0", *, hostname="worker-0", address="10.0.0.1", alive=True):
    return {
        "NodeID": node_id,
        "NodeManagerHostname": hostname,
        "NodeManagerAddress": address,
        "MetricsExportPort": 8080,
        "Alive": alive,
    }


def _fake_scrape(monkeypatch, by_url):
    """Serve *by_url* from ``requests.get``; a missing URL raises like a dead node."""
    import requests

    import nemo_rl.telemetry.host_memory as host_memory

    class _Response:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    def _get(url, timeout=None):
        if url not in by_url:
            raise requests.ConnectionError(url)
        return _Response(by_url[url])

    monkeypatch.setattr(host_memory.requests, "get", _get)


def _gauge_points(data, name):
    """Data points for gauge *name* in one already-collected metrics batch."""
    if data is None:
        return []
    return [
        point
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


def _start_exporting_telemetry():
    """Install an exporting telemetry handle; returns its metric reader."""
    from nemo.lens import NemoLensConfig, setup_telemetry
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    import nemo_rl.telemetry.setup as setup_mod

    instruments = pytest.importorskip("nemo.lens.instruments")
    if not hasattr(instruments, "register_metric_group"):
        pytest.skip("installed nemo-lens has no metric registry (lens PR #46)")

    reader = InMemoryMetricReader()
    setup_mod._TELEMETRY_HANDLE = setup_telemetry(
        NemoLensConfig(enabled=True), metric_reader=reader
    )
    return reader


def test_only_the_two_node_memory_series_are_read():
    """A raylet publishes hundreds of series; all but two are noise here."""
    assert parse_memory_samples(_SCRAPE) == {
        "host_memory_used": _USED_BYTES,
        "host_memory_total": _TOTAL_BYTES,
    }


def test_bytes_are_reported_unscaled():
    """The W&B path divides to GB on the way out; OTel wants the base unit."""
    values = parse_memory_samples(_SCRAPE)

    assert values["host_memory_used"] == _USED_BYTES
    assert values["host_memory_used"] > 1e9


def test_a_node_is_named_by_hostname_rather_than_ip(monkeypatch):
    """``NodeManagerAddress`` is the IP; the hostname is what a reader knows."""
    _fake_ray(monkeypatch, [_node(hostname="dgx-042", address="10.0.0.7")])

    (node,) = ray_nodes()

    assert node.host == "dgx-042"
    # The IP is still what the endpoint is reached on.
    assert node.metrics_url == "http://10.0.0.7:8080/metrics"


def test_a_node_falls_back_to_its_ip_when_ray_resolved_no_hostname(monkeypatch):
    """Better an address than an unlabelled point."""
    _fake_ray(monkeypatch, [_node(hostname="", address="10.0.0.7")])

    (node,) = ray_nodes()

    assert node.host == "10.0.0.7"


def test_dead_nodes_are_not_scraped(monkeypatch):
    """``ray.nodes()`` keeps dead entries, whose endpoints only ever time out."""
    _fake_ray(
        monkeypatch,
        [_node("n0", hostname="live"), _node("n1", hostname="dead", alive=False)],
    )

    assert [node.host for node in ray_nodes()] == ["live"]


def test_no_endpoint_means_no_node(monkeypatch):
    """A node without a metrics port publishes nothing to scrape."""
    node = _node()
    del node["MetricsExportPort"]
    _fake_ray(monkeypatch, [node])

    assert ray_nodes() == []


def test_every_group_resident_on_a_node_is_named():
    """Colocation puts several groups on one host; the label says so."""
    register_worker_group_nodes("vllm_policy", ["n0", "n1"])
    register_worker_group_nodes("lm_policy", ["n0"])

    # Sorted, so the attribute value is stable rather than insertion-ordered --
    # an unstable value would split one host's series in two.
    assert worker_groups_on_node("n0") == "lm_policy,vllm_policy"
    assert worker_groups_on_node("n1") == "vllm_policy"


def test_a_node_running_no_worker_group_is_still_labelled():
    """The head node hosts only the driver, and its memory is just as real."""
    register_worker_group_nodes("lm_policy", ["n1"])

    assert worker_groups_on_node("head") == NO_WORKER_GROUPS


def test_a_regrouped_worker_group_is_not_reported_on_both_placements():
    """A restart can move a group; the old nodes must stop claiming it."""
    register_worker_group_nodes("lm_policy", ["n0"])
    register_worker_group_nodes("lm_policy", ["n1"])

    assert worker_groups_on_node("n0") == NO_WORKER_GROUPS
    assert worker_groups_on_node("n1") == "lm_policy"


def test_a_shut_down_group_stops_claiming_its_nodes():
    register_worker_group_nodes("teacher", ["n0"])
    forget_worker_group_nodes("teacher")

    assert worker_groups_on_node("n0") == NO_WORKER_GROUPS


def test_each_host_is_reported_with_its_name_and_its_worker_groups(monkeypatch):
    pytest.importorskip("nemo.lens")
    reader = _start_exporting_telemetry()

    _fake_ray(
        monkeypatch,
        [
            _node("n0", hostname="dgx-000", address="10.0.0.1"),
            _node("n1", hostname="dgx-001", address="10.0.0.2"),
        ],
    )
    _fake_scrape(
        monkeypatch,
        {
            "http://10.0.0.1:8080/metrics": _SCRAPE,
            "http://10.0.0.2:8080/metrics": _SCRAPE,
        },
    )
    register_worker_group_nodes("lm_policy", ["n0"])

    HostMemoryMonitor().collect_once()

    data = reader.get_metrics_data()
    used = {
        point.attributes[HOST_NAME_ATTR]: point
        for point in _gauge_points(data, RL_HOST_MEMORY_USED_METRIC)
    }
    assert set(used) == {"dgx-000", "dgx-001"}
    assert used["dgx-000"].value == _USED_BYTES
    assert used["dgx-000"].attributes[RL_WORKER_GROUPS_ATTR] == "lm_policy"
    assert used["dgx-001"].attributes[RL_WORKER_GROUPS_ATTR] == NO_WORKER_GROUPS

    total = _gauge_points(data, RL_HOST_MEMORY_TOTAL_METRIC)
    assert {point.value for point in total} == {_TOTAL_BYTES}


def test_an_unreachable_node_costs_only_its_own_point(monkeypatch):
    """The node that is wedged is exactly the one whose host memory matters."""
    pytest.importorskip("nemo.lens")
    reader = _start_exporting_telemetry()

    _fake_ray(
        monkeypatch,
        [
            _node("n0", hostname="dgx-000", address="10.0.0.1"),
            _node("n1", hostname="wedged", address="10.0.0.2"),
        ],
    )
    _fake_scrape(monkeypatch, {"http://10.0.0.1:8080/metrics": _SCRAPE})

    HostMemoryMonitor().collect_once()

    points = _gauge_points(reader.get_metrics_data(), RL_HOST_MEMORY_USED_METRIC)
    assert [point.attributes[HOST_NAME_ATTR] for point in points] == ["dgx-000"]


def test_collection_never_raises(monkeypatch):
    """It runs on a timer beside a training run and must not be able to end it."""
    import nemo_rl.telemetry.host_memory as host_memory

    def _explode():
        raise RuntimeError("GCS is unreachable")

    monkeypatch.setattr(host_memory, "ray_nodes", _explode)
    _start_exporting_telemetry()

    HostMemoryMonitor().collect_once()


def test_nothing_is_scraped_when_telemetry_is_not_exporting(monkeypatch):
    """No handle means no consumer, so the HTTP calls are wasted work."""
    import nemo_rl.telemetry.setup as setup_mod

    setup_mod._TELEMETRY_HANDLE = None
    scraped = []
    _fake_ray(monkeypatch, [_node()])
    monkeypatch.setattr(
        HostMemoryMonitor,
        "_scrape",
        lambda self, node: scraped.append(node) or {},
    )

    HostMemoryMonitor().collect_once()

    assert scraped == []


def test_stopping_a_monitor_that_was_never_started_is_harmless():
    HostMemoryMonitor().stop()


def test_a_process_runs_at_most_one_monitor(monkeypatch):
    """``init_telemetry_driver`` is idempotent, so starting twice must be too."""
    _fake_ray(monkeypatch, [])

    start_host_memory_monitoring(collection_interval=60.0)
    first = host_memory._MONITOR
    start_host_memory_monitoring(collection_interval=60.0)

    assert host_memory._MONITOR is first


def test_stopping_releases_the_monitor_so_a_later_run_can_start_one(monkeypatch):
    _fake_ray(monkeypatch, [])

    start_host_memory_monitoring(collection_interval=60.0)
    stop_host_memory_monitoring()

    assert host_memory._MONITOR is None


def test_stopping_without_a_monitor_is_a_no_op():
    """Every worker calls ``shutdown_telemetry``, and none of them started one."""
    stop_host_memory_monitoring()

    assert host_memory._MONITOR is None


def test_the_driver_starts_the_monitor_that_the_worker_groups_register_with(
    monkeypatch, tmp_path
):
    """The whole point of owning it here: one process writes and reads the map.

    ``RayWorkerGroup`` registers placements from the driver into a module
    global, so a monitor anywhere else -- the controller actor that owns the
    ``Logger`` under single-controller GRPO and SFT v2 -- would read an empty
    map and label every host ``none``.
    """
    pytest.importorskip("nemo.lens")
    from nemo_rl.telemetry.setup import init_telemetry_driver, shutdown_telemetry

    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "true")
    monkeypatch.setenv("NEMO_RL_OTEL_EXPORTER", "console")
    _fake_ray(monkeypatch, [])

    config = SimpleNamespace(
        telemetry=None,
        logger=SimpleNamespace(
            gpu_monitoring=SimpleNamespace(collection_interval=60.0)
        ),
        policy=None,
        cluster=None,
    )
    handle = init_telemetry_driver(config, algorithm="grpo")
    if handle is None or not handle.is_exporting:
        pytest.skip("telemetry did not come up exporting in this environment")

    assert host_memory._MONITOR is not None
    # Taken from the logger's GPU cadence rather than restated, so the two
    # families land on the same points of a dashboard.
    assert host_memory._MONITOR.collection_interval == 60.0

    shutdown_telemetry(timeout_ms=100)
    assert host_memory._MONITOR is None
