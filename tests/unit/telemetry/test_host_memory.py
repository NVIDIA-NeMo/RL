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
    register_worker_group_placement,
    start_host_memory_monitoring,
    stop_host_memory_monitoring,
    worker_groups_on_node,
)
from nemo_rl.telemetry.metrics import create_rl_metric_instruments

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
# HELP ray_node_mem_used_percent Share of memory in use.
# TYPE ray_node_mem_used_percent gauge
ray_node_mem_used_percent{{SessionName="s"}} 3.0
"""


@pytest.fixture(autouse=True)
def _reset_host_memory_state():
    """Placement map, snapshot and monitor are process-global; none leaks."""
    host_memory._WORKER_GROUP_NODES.clear()
    host_memory._SNAPSHOT.clear()
    stop_host_memory_monitoring()
    yield
    host_memory._WORKER_GROUP_NODES.clear()
    host_memory._SNAPSHOT.clear()
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

    reader = InMemoryMetricReader()
    setup_mod._TELEMETRY_HANDLE = setup_telemetry(
        NemoLensConfig(enabled=True), metric_reader=reader
    )
    # These series are observable, so nothing records them into existence --
    # the reader only sees them once their instruments exist. In a run this is
    # start_host_memory_monitoring's job; here the tests drive collection by
    # hand, so it is the handle's.
    create_rl_metric_instruments()
    return reader


def test_only_the_two_node_memory_series_are_read():
    """A raylet publishes hundreds of series; all but two are noise here.

    ``ray_node_mem_used_percent`` is the near miss that matters: the parser
    drops most lines on a prefix before parsing them, and that prefix keeps it.
    """
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


def _fake_placement_groups(monkeypatch, tables, *, raises=False):
    """Install a ``placement_group_table`` answering from *tables*.

    Keyed by ``id(pg)``, like the code under test, because a real placement
    group is not hashable either.

    Returns the list of placement groups it was asked about, in order, so a
    test can tell one lookup from several.
    """
    asked = []

    def placement_group_table(pg):
        asked.append(pg)
        if raises:
            raise RuntimeError("GCS is unreachable")
        return {"bundles_to_node_id": tables[id(pg)]}

    ray = types.ModuleType("ray")
    util = types.ModuleType("ray.util")
    module = types.ModuleType("ray.util.placement_group")
    module.placement_group_table = placement_group_table
    util.placement_group = module
    ray.util = util
    # All three levels, since the import under test walks the package path.
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util", util)
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", module)
    return asked


def test_a_groups_bundles_name_the_nodes_it_is_on(monkeypatch):
    """The form the worker groups actually call with: (pg, bundle index)."""
    pg = SimpleNamespace(name="policy_pg")
    _fake_placement_groups(monkeypatch, {id(pg): {0: "n0", 1: "n1"}})

    register_worker_group_placement("lm_policy", [(pg, 0), (pg, 1)])

    assert worker_groups_on_node("n0") == "lm_policy"
    assert worker_groups_on_node("n1") == "lm_policy"


def test_a_placement_group_is_looked_up_once_however_many_bundles_it_holds(
    monkeypatch,
):
    """One table covers every bundle, and the lookup is a GCS round trip."""
    pg = SimpleNamespace(name="policy_pg")
    asked = _fake_placement_groups(monkeypatch, {id(pg): {i: "n0" for i in range(4)}})

    register_worker_group_placement("lm_policy", [(pg, i) for i in range(4)])

    assert asked == [pg]


def test_a_bundle_the_table_does_not_place_is_left_out(monkeypatch):
    """A pending bundle has no node yet; the ones that do are still labelled."""
    pg = SimpleNamespace(name="policy_pg")
    _fake_placement_groups(monkeypatch, {id(pg): {0: "n0"}})

    register_worker_group_placement("lm_policy", [(pg, 0), (pg, 1)])

    assert worker_groups_on_node("n0") == "lm_policy"


def test_an_unresolvable_placement_costs_only_the_label(monkeypatch):
    """Attribution is descriptive; failing to resolve it must not reach Ray."""
    pg = SimpleNamespace(name="policy_pg")
    _fake_placement_groups(monkeypatch, {id(pg): {0: "n0"}}, raises=True)

    register_worker_group_placement("lm_policy", [(pg, 0)])

    assert worker_groups_on_node("n0") == NO_WORKER_GROUPS


def test_each_host_is_reported_with_its_name_and_its_worker_groups(monkeypatch):
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

    HostMemoryMonitor(collection_interval=10.0).collect_once()

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


def test_a_label_the_scrape_stops_reporting_stops_being_exported(monkeypatch):
    """Why these series are observable instead of pushed to a gauge.

    A host is labelled ``none`` until its worker groups are built, which is
    minutes into a run. Pushed to a synchronous gauge, that startup point would
    keep being re-exported next to the correct one for the rest of the run: the
    SDK holds a gauge's last value per attribute set, and nothing can retract
    one. ``sum by (host.name)`` would then count every host twice, and
    filtering on ``none`` would show every training host as idle.
    """
    reader = _start_exporting_telemetry()

    _fake_ray(monkeypatch, [_node("n0", hostname="dgx-000", address="10.0.0.1")])
    _fake_scrape(monkeypatch, {"http://10.0.0.1:8080/metrics": _SCRAPE})
    monitor = HostMemoryMonitor(collection_interval=10.0)

    monitor.collect_once()
    at_startup = _gauge_points(reader.get_metrics_data(), RL_HOST_MEMORY_USED_METRIC)
    assert [point.attributes[RL_WORKER_GROUPS_ATTR] for point in at_startup] == [
        NO_WORKER_GROUPS
    ]

    register_worker_group_nodes("lm_policy", ["n0"])
    monitor.collect_once()

    once_grouped = _gauge_points(reader.get_metrics_data(), RL_HOST_MEMORY_USED_METRIC)
    assert [point.attributes[RL_WORKER_GROUPS_ATTR] for point in once_grouped] == [
        "lm_policy"
    ]


def test_two_raylets_on_one_host_are_one_point_scraped_once(monkeypatch):
    """A local multi-node cluster must not report its host twice.

    Each raylet has its own endpoint but publishes the same whole-host memory,
    so a point per node would double count under ``sum by (host.name)``. The
    worker groups of both are merged, since the label describes the host.
    """
    reader = _start_exporting_telemetry()

    _fake_ray(
        monkeypatch,
        [
            _node("n0", hostname="dev-box", address="10.0.0.1"),
            _node("n1", hostname="dev-box", address="10.0.0.1"),
        ],
    )
    _fake_scrape(monkeypatch, {"http://10.0.0.1:8080/metrics": _SCRAPE})
    served = host_memory.requests.get
    scraped: list[str] = []

    def _counting(url, timeout=None):
        scraped.append(url)
        return served(url, timeout=timeout)

    monkeypatch.setattr(host_memory.requests, "get", _counting)
    register_worker_group_nodes("lm_policy", ["n0"])
    register_worker_group_nodes("vllm_policy", ["n1"])

    HostMemoryMonitor(collection_interval=10.0).collect_once()

    points = _gauge_points(reader.get_metrics_data(), RL_HOST_MEMORY_USED_METRIC)
    assert len(points) == 1
    assert points[0].attributes[HOST_NAME_ATTR] == "dev-box"
    assert points[0].attributes[RL_WORKER_GROUPS_ATTR] == "lm_policy,vllm_policy"
    assert len(scraped) == 1


def test_stopping_abandons_the_sweep_instead_of_finishing_it(monkeypatch):
    """Shutdown costs the scrape in flight, not one per remaining node.

    The sweep is sequential and each node has its own timeout, so finishing it
    on the way out would make shutdown scale with the size of the cluster. The
    partial result is thrown away rather than published: it would otherwise end
    the series of every host the sweep had not reached.
    """
    _start_exporting_telemetry()

    nodes = [
        _node(f"n{index}", hostname=f"dgx-{index}", address=f"10.0.0.{index}")
        for index in range(5)
    ]
    _fake_ray(monkeypatch, nodes)
    monitor = HostMemoryMonitor(collection_interval=10.0)
    scraped: list[str] = []

    class _Response:
        text = _SCRAPE

        def raise_for_status(self):
            return None

    def _get(url, timeout=None):
        scraped.append(url)
        # Shutdown lands while the first node's request is in flight.
        monitor._stop.set()
        return _Response()

    monkeypatch.setattr(host_memory.requests, "get", _get)

    monitor.collect_once()

    assert len(scraped) == 1
    assert host_memory._SNAPSHOT == {}


def test_a_stopped_monitor_leaves_nothing_behind_to_export(monkeypatch):
    """The callbacks outlive the monitor: OTel cannot detach an instrument.

    So whatever the last scrape left behind would otherwise be exported, frozen,
    by every cycle between shutdown and process exit.
    """
    reader = _start_exporting_telemetry()

    _fake_ray(monkeypatch, [_node("n0", hostname="dgx-000", address="10.0.0.1")])
    _fake_scrape(monkeypatch, {"http://10.0.0.1:8080/metrics": _SCRAPE})

    # An interval long enough that the loop samples once and then sleeps, so
    # the points under test are the ones collected below and not a race.
    start_host_memory_monitoring(collection_interval=3600.0)
    HostMemoryMonitor(collection_interval=10.0).collect_once()
    assert _gauge_points(reader.get_metrics_data(), RL_HOST_MEMORY_USED_METRIC)

    stop_host_memory_monitoring()

    assert not _gauge_points(reader.get_metrics_data(), RL_HOST_MEMORY_USED_METRIC)


def test_an_unparsable_page_costs_only_its_own_point(monkeypatch):
    """One broken page must not abandon the sweep and freeze the snapshot.

    A proxy in front of a raylet can answer 200 with an HTML error page. Left to
    the guard around the sweep, the ValueError would discard every other node's
    sample, and the exporter would keep publishing the previous snapshot --
    stale worker-group labels included -- for as long as the page stayed broken.
    """
    reader = _start_exporting_telemetry()

    _fake_ray(
        monkeypatch,
        [
            _node("n0", hostname="dgx-000", address="10.0.0.1"),
            _node("n1", hostname="proxied", address="10.0.0.2"),
        ],
    )
    _fake_scrape(
        monkeypatch,
        {
            "http://10.0.0.1:8080/metrics": _SCRAPE,
            "http://10.0.0.2:8080/metrics": "<html><body>502 Bad Gateway</body></html>",
        },
    )

    HostMemoryMonitor(collection_interval=10.0).collect_once()

    points = _gauge_points(reader.get_metrics_data(), RL_HOST_MEMORY_USED_METRIC)
    assert [point.attributes[HOST_NAME_ATTR] for point in points] == ["dgx-000"]


def test_an_unreachable_node_costs_only_its_own_point(monkeypatch):
    """The node that is wedged is exactly the one whose host memory matters."""
    reader = _start_exporting_telemetry()

    _fake_ray(
        monkeypatch,
        [
            _node("n0", hostname="dgx-000", address="10.0.0.1"),
            _node("n1", hostname="wedged", address="10.0.0.2"),
        ],
    )
    _fake_scrape(monkeypatch, {"http://10.0.0.1:8080/metrics": _SCRAPE})

    HostMemoryMonitor(collection_interval=10.0).collect_once()

    points = _gauge_points(reader.get_metrics_data(), RL_HOST_MEMORY_USED_METRIC)
    assert [point.attributes[HOST_NAME_ATTR] for point in points] == ["dgx-000"]


def test_collection_never_raises(monkeypatch):
    """It runs on a timer beside a training run and must not be able to end it."""
    import nemo_rl.telemetry.host_memory as host_memory

    def _explode():
        raise RuntimeError("GCS is unreachable")

    monkeypatch.setattr(host_memory, "ray_nodes", _explode)
    _start_exporting_telemetry()

    HostMemoryMonitor(collection_interval=10.0).collect_once()


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

    HostMemoryMonitor(collection_interval=10.0).collect_once()

    assert scraped == []


def test_stopping_a_monitor_that_was_never_started_is_harmless():
    HostMemoryMonitor(collection_interval=10.0).stop()


def test_a_process_runs_at_most_one_monitor(monkeypatch):
    """``init_telemetry_driver`` is idempotent, so starting twice must be too."""
    _fake_ray(monkeypatch, [])

    start_host_memory_monitoring(collection_interval=60.0)
    first = host_memory._MONITOR
    start_host_memory_monitoring(collection_interval=60.0)

    assert host_memory._MONITOR is first


@pytest.mark.parametrize("interval", [0.0, -5.0])
def test_a_nonpositive_interval_starts_nothing_rather_than_spinning(
    monkeypatch, interval
):
    """At zero the loop would scrape every raylet as fast as HTTP allows."""
    _fake_ray(monkeypatch, [])

    start_host_memory_monitoring(collection_interval=interval)

    assert host_memory._MONITOR is None


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
    # Asserted, not skipped: lens is a core dependency, so a handle that is not
    # exporting is the regression this test exists to catch.
    assert handle is not None and handle.is_exporting

    assert host_memory._MONITOR is not None
    # Taken from the logger's GPU cadence rather than restated, so the two
    # families land on the same points of a dashboard.
    assert host_memory._MONITOR.collection_interval == 60.0

    shutdown_telemetry(timeout_ms=100)
    assert host_memory._MONITOR is None
