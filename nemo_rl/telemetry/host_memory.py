# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host memory per node, scraped from Ray and exported through lens.

Same source as the W&B path: every raylet publishes ``ray_node_mem_used`` and
``ray_node_mem_total`` on its own Prometheus endpoint, and
``RayGpuMonitorLogger`` in :mod:`nemo_rl.utils.logger` already reads them.
Nothing else in NeMo-RL reports host memory -- ``utils/memory_tracker.py``
samples the driver's own RSS, prints it, and is wired only into PPO -- so
without this a run's OTel stream has GPU-side numbers and no host-side ones.

Scraped here rather than teed out of the GPU monitor because that path flattens
the node's identity into the metric *name* (``node.3.mem_gb``) and discards the
hostname, and because it is gated on ``logger.monitor_gpus``: reading it would
make an OTel series disappear when someone turns off a W&B feature. The cost is
a second HTTP GET per node per interval, against a local endpoint, which is not
measurable next to a training step.

What the number is, and is not
------------------------------
``ray_node_mem_used`` is whole-host memory. It covers the raylet, the Ray object
store (often tens of GB, and frequently the thing that actually ran the node out
of memory), every worker resident on the host, and -- on the head node -- the
driver. It is not a per-process or per-worker-group figure, and there is no
breakdown inside it; see :data:`RL_WORKER_GROUPS_ATTR` for what the worker-group
attribute can and cannot be used for.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import requests
from prometheus_client.parser import text_string_to_metric_families

from nemo_rl.telemetry.metrics import record_rl_metrics, warn_once
from nemo_rl.telemetry.setup import get_telemetry_handle
from nemo_rl.telemetry.vocabulary import (
    RecordedMetric,
    register_recorded_metrics,
    registry_key,
)

logger = logging.getLogger(__name__)

#: Resident and installed memory on one host, in bytes.
RL_HOST_MEMORY_USED_METRIC = "rl.host.memory.used"
RL_HOST_MEMORY_TOTAL_METRIC = "rl.host.memory.total"

#: Which host a point describes. OTel's ``host.name`` is normally a *resource*
#: attribute naming the host a process runs on, and lens already sets it that
#: way. Here it is a metric attribute instead, because the driver is reporting
#: about hosts other than its own -- the resource would say "the head node" for
#: every point.
HOST_NAME_ATTR = "host.name"

#: Worker groups resident on the host, sorted and comma-joined.
#:
#: A descriptor of the host, not a partition key. The value it labels is the
#: whole host's memory, so ``sum by (rl.worker_groups)`` double-counts under
#: colocation (where ``lm_policy`` and ``vllm_policy`` share GPUs and therefore
#: a host) and over-counts always, since no worker group owns the raylet, the
#: object store or the driver. Plural, and joined rather than emitted as one
#: point per group, so that this reads as "what is on this box" rather than as
#: something summable. Group it to find *which* hosts to look at; read
#: ``host.name`` for the number itself.
RL_WORKER_GROUPS_ATTR = "rl.worker_groups"

#: Value of :data:`RL_WORKER_GROUPS_ATTR` for a host running no worker group:
#: a head node that only hosts the driver, or any node whose groups have not
#: been built yet when the sample is taken.
NO_WORKER_GROUPS = "none"

#: Prometheus series published by each raylet, and the key each is recorded
#: against. Bytes already, despite the GB the W&B path reports: that divides by
#: 1024**3 on the way out.
_SERIES_BY_RAY_NAME = {
    "ray_node_mem_used": RL_HOST_MEMORY_USED_METRIC,
    "ray_node_mem_total": RL_HOST_MEMORY_TOTAL_METRIC,
}

HOST_MEMORY_RECORDED_METRICS = (
    RecordedMetric(
        RL_HOST_MEMORY_USED_METRIC,
        kind="gauge",
        unit="By",
        description=(
            "Memory in use on one host, across every process on it. Labelled by "
            "host.name; see rl.worker_groups before aggregating."
        ),
    ),
    RecordedMetric(
        RL_HOST_MEMORY_TOTAL_METRIC,
        kind="gauge",
        unit="By",
        description="Memory installed on one host.",
    ),
)

register_recorded_metrics(HOST_MEMORY_RECORDED_METRICS)

_USED_KEY = HOST_MEMORY_RECORDED_METRICS[0].key
_TOTAL_KEY = HOST_MEMORY_RECORDED_METRICS[1].key

#: Matches ``GPUMonitoringConfig.collection_interval``, so the host-memory and
#: GPU series line up on a dashboard rather than interleaving.
DEFAULT_COLLECTION_INTERVAL = 10.0

#: Short, because the endpoint is a raylet on the cluster's own network and the
#: loop would rather miss a sample than hold the interval open behind a node
#: that is already wedged.
_SCRAPE_TIMEOUT_S = 5.0


# --- which worker groups sit on which node --------------------------------
#
# Filled by RayWorkerGroup as each group is built, because the mapping is not
# knowable when the monitor starts: the driver initialises telemetry before
# init_ray(), and the groups are placed later still. A node missing from here
# is reported as NO_WORKER_GROUPS rather than skipped -- its memory is just as
# real, and the head node never appears here at all.
_WORKER_GROUP_NODES: dict[str, frozenset[str]] = {}
_WORKER_GROUP_LOCK = threading.Lock()


def register_worker_group_nodes(name_prefix: str, node_ids: Iterable[str]) -> None:
    """Record that worker group *name_prefix* occupies *node_ids*.

    Args:
        name_prefix: The group's ``RayWorkerGroup.name_prefix`` -- the same
            value its workers carry as the ``rl.worker_group`` resource
            attribute, so a reader can join the two.
        node_ids: Ray ``NodeID`` hex strings the group's workers are placed on.

    Replaces any previous entry for the group, so a group rebuilt onto different
    nodes after a worker restart is not reported on both sets.
    """
    with _WORKER_GROUP_LOCK:
        _WORKER_GROUP_NODES[name_prefix] = frozenset(node_ids)


def register_worker_group_placement(
    name_prefix: str, bundles: Iterable[tuple[Any, int]]
) -> None:
    """Resolve *bundles* to nodes and register them. Never raises.

    The form :class:`~nemo_rl.distributed.worker_groups.RayWorkerGroup` can
    supply directly: it knows each worker's ``(placement group, bundle index)``
    but not the node underneath, and ``placement_group_table`` answers that from
    the GCS's own scheduling record -- driver-side, with no worker asked
    anything. Each placement group is looked up once however many bundles it
    contributes, since the table covers all of them.

    Args:
        name_prefix: See :func:`register_worker_group_nodes`.
        bundles: ``(placement_group, bundle_index)`` pairs the group occupies.
    """
    # Deferred with the rest of Ray: see ray_nodes().
    from ray.util.placement_group import placement_group_table

    try:
        node_ids = set()
        tables: dict[int, dict[int, str]] = {}
        for pg, bundle_index in bundles:
            table = tables.get(id(pg))
            if table is None:
                table = placement_group_table(pg)["bundles_to_node_id"]
                tables[id(pg)] = table
            node_id = table.get(bundle_index)
            if node_id:
                node_ids.add(node_id)
    except Exception:
        warn_once(
            f"host_memory_placement:{name_prefix}",
            f"could not resolve which nodes worker group {name_prefix!r} is on; "
            "its host memory will be reported without a worker-group label",
        )
    else:
        register_worker_group_nodes(name_prefix, node_ids)


def forget_worker_group_nodes(name_prefix: str) -> None:
    """Drop *name_prefix*'s placement, for a group that has been shut down."""
    with _WORKER_GROUP_LOCK:
        _WORKER_GROUP_NODES.pop(name_prefix, None)


def worker_groups_on_node(node_id: str) -> str:
    """Value of :data:`RL_WORKER_GROUPS_ATTR` for *node_id*."""
    with _WORKER_GROUP_LOCK:
        groups = sorted(
            prefix
            for prefix, node_ids in _WORKER_GROUP_NODES.items()
            if node_id in node_ids
        )
    return ",".join(groups) if groups else NO_WORKER_GROUPS


@dataclass(frozen=True)
class RayNode:
    """One live Ray node: who it is, and where to scrape it."""

    node_id: str
    host: str
    metrics_url: str


def ray_nodes() -> list[RayNode]:
    """Live Ray nodes that publish a metrics endpoint.

    Returns an empty list when Ray is not up, which makes the collection loop
    idle rather than fail: the driver starts telemetry before ``init_ray()``.
    """
    # Deferred: this module is imported from nemo_rl.utils.logger at module
    # scope, and must stay importable in a process that never initialises Ray.
    import ray

    if not ray.is_initialized():
        return []

    nodes = []
    for node in ray.nodes():
        # "Alive" because ray.nodes() keeps dead entries, whose endpoints would
        # be scraped every interval until they time out.
        if not node.get("Alive"):
            continue
        port = node.get("MetricsExportPort")
        address = node.get("NodeManagerAddress")
        if not port or not address:
            continue
        nodes.append(
            RayNode(
                node_id=node.get("NodeID", ""),
                # NodeManagerAddress is the IP; the hostname is a sibling key.
                # Preferred because it is what a user recognises and what the
                # cluster's other tooling reports, and it falls back to the IP
                # rather than going unlabelled when Ray has not resolved one.
                host=node.get("NodeManagerHostname") or address,
                metrics_url=f"http://{address}:{port}/metrics",
            )
        )
    return nodes


def parse_memory_samples(metrics_text: str) -> dict[str, float]:
    """Extract ``{registry key: bytes}`` from one node's Prometheus payload.

    Pure function (no OTel side effects) so it is trivially unit-testable.
    Unknown series are ignored, which is nearly all of them: a raylet publishes
    hundreds, and only the two in :data:`_SERIES_BY_RAY_NAME` are wanted.
    """
    values: dict[str, float] = {}
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            name = _SERIES_BY_RAY_NAME.get(sample.name)
            if name is None:
                continue
            values[registry_key(name)] = float(sample.value)
    return values


class HostMemoryMonitor:
    """Samples every Ray node's host memory on a timer and records it to lens.

    Driver-side and read-only: it scrapes endpoints the cluster already serves
    and touches no worker. Start it once the cluster is up; it tolerates being
    started earlier and simply finds no nodes until there are some.
    """

    def __init__(
        self, collection_interval: float = DEFAULT_COLLECTION_INTERVAL
    ) -> None:
        """Initialise the monitor.

        Args:
            collection_interval: Seconds between samples of every node.
        """
        self.collection_interval = collection_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Begin sampling on a background thread. Idempotent."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._collection_loop,
            name="nemo-rl-host-memory",
            # Daemon so a run that exits without calling stop() -- a crash, a
            # KeyboardInterrupt -- is not held open by an observability thread.
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "host memory monitoring started (interval=%ss)", self.collection_interval
        )

    def stop(self) -> None:
        """Stop sampling and wait for the thread to finish. Idempotent."""
        thread = self._thread
        self._stop.set()
        self._thread = None
        if thread is not None:
            # One interval plus the scrape budget: the loop can be asleep on
            # Event.wait or mid-scrape, and this covers both without the join
            # becoming the thing that holds up shutdown.
            thread.join(timeout=self.collection_interval + _SCRAPE_TIMEOUT_S)

    def collect_once(self) -> None:
        """Sample every node and record one point each. Never raises.

        The guarantee lives here rather than in the loop so that the loop cannot
        be killed by a single bad sample, and so a caller driving the monitor
        by hand gets the same protection.
        """
        try:
            self._collect_once()
        except Exception:
            warn_once("host_memory", "failed to collect host memory from Ray")

    def _collect_once(self) -> None:
        """Body of :meth:`collect_once`, inside its exception guard."""
        handle = get_telemetry_handle()
        if handle is None or not handle.is_exporting:
            return
        for node in ray_nodes():
            values = self._scrape(node)
            if not values:
                continue
            record_rl_metrics(
                values,
                {
                    HOST_NAME_ATTR: node.host,
                    RL_WORKER_GROUPS_ATTR: worker_groups_on_node(node.node_id),
                },
            )

    def _scrape(self, node: RayNode) -> dict[str, float]:
        """Fetch and parse one node's memory series, or ``{}`` if unreachable.

        A node that cannot be scraped is skipped rather than retried: the next
        interval is the retry, and a node that is down or overloaded is exactly
        the one that must not be allowed to stall the loop for the others.
        """
        try:
            response = requests.get(node.metrics_url, timeout=_SCRAPE_TIMEOUT_S)
            response.raise_for_status()
        except requests.RequestException:
            warn_once(
                f"host_memory_scrape:{node.host}",
                f"could not scrape host memory from {node.metrics_url}",
            )
            return {}
        return parse_memory_samples(response.text)

    def _collection_loop(self) -> None:
        """Sample until stopped, on :attr:`collection_interval`."""
        # Sampled before the first wait, so a run shorter than one interval
        # still reports something. Event.wait rather than sleep so stop()
        # returns promptly instead of blocking for up to a whole interval.
        while True:
            self.collect_once()
            if self._stop.wait(self.collection_interval):
                return


#: The one monitor a process runs, owned by the telemetry lifecycle below.
_MONITOR: Optional[HostMemoryMonitor] = None


def start_host_memory_monitoring(
    collection_interval: float = DEFAULT_COLLECTION_INTERVAL,
) -> None:
    """Start this process's monitor, if it has none. Never raises.

    Called from ``init_telemetry_driver``, which is what puts the monitor in
    the **driver** process on every entry point. That is not a detail: the
    placement map above is an ordinary module global, so it is only visible to
    the process that wrote it, and ``RayWorkerGroup`` writes it from the driver.
    Owning the monitor anywhere else reports the right memory under the wrong
    label -- the single-controller entry points build their worker groups on
    the driver (``examples/run_grpo_single_controller.py``,
    ``examples/run_sft_v2.py``) but their ``Logger`` inside the controller
    actor, whose copy of the map would never be anything but empty.

    Args:
        collection_interval: Seconds between samples of every node.
    """
    global _MONITOR
    if _MONITOR is not None:
        return
    monitor = HostMemoryMonitor(collection_interval=collection_interval)
    try:
        monitor.start()
    except Exception:
        warn_once("host_memory_start", "could not start host memory monitoring")
    else:
        _MONITOR = monitor


def stop_host_memory_monitoring() -> None:
    """Stop this process's monitor. Never raises; a no-op if none is running.

    Called from ``shutdown_telemetry``, which every worker also calls -- hence
    the no-op rather than an assertion that one is running.
    """
    global _MONITOR
    monitor = _MONITOR
    _MONITOR = None
    if monitor is None:
        return
    try:
        monitor.stop()
    except Exception:
        warn_once("host_memory_stop", "could not stop host memory monitoring")
