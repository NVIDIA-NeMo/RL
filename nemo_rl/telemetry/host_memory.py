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

What the number is, and is not: ``ray_node_mem_used`` is whole-host memory. It
covers the raylet, the Ray object store (often tens of GB, and frequently the
thing that actually ran the node out of memory), every worker resident on the
host, and -- on the head node -- the driver. It is not a per-process or
per-worker-group figure, and there is no breakdown inside it; see
:data:`RL_WORKER_GROUPS_ATTR` for what the worker-group attribute can and
cannot be used for.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Optional

import requests
from prometheus_client.parser import text_string_to_metric_families

from nemo_rl.telemetry.metrics import create_rl_metric_instruments, warn_once
from nemo_rl.telemetry.setup import get_telemetry_handle
from nemo_rl.telemetry.vocabulary import (
    RecordedMetric,
    register_recorded_metrics,
    registry_key,
)

if TYPE_CHECKING:
    from opentelemetry.metrics import CallbackOptions, Observation

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

#: Ray's own series names, public because ``RayGpuMonitorLogger`` reads the
#: same two for the W&B path. Typed once: renaming only one side would leave
#: this module matching nothing and its series silently empty.
RAY_NODE_MEM_USED = "ray_node_mem_used"
RAY_NODE_MEM_TOTAL = "ray_node_mem_total"

#: Prometheus series published by each raylet, and the key each is recorded
#: against. Bytes already, despite the GB the W&B path reports: that divides by
#: 1024**3 on the way out.
_SERIES_BY_RAY_NAME = {
    RAY_NODE_MEM_USED: RL_HOST_MEMORY_USED_METRIC,
    RAY_NODE_MEM_TOTAL: RL_HOST_MEMORY_TOTAL_METRIC,
}

_USED_KEY = registry_key(RL_HOST_MEMORY_USED_METRIC)
_TOTAL_KEY = registry_key(RL_HOST_MEMORY_TOTAL_METRIC)

#: The most recent scrape, keyed by the attributes the points carry.
#:
#: Replaced wholesale each cycle rather than updated, which is what makes a
#: host's label stop being exported when it changes: the exporter's points are
#: exactly what the callbacks below return, so a key that is no longer in here
#: is a series that ends. Read under :data:`_SNAPSHOT_LOCK` from the exporter's
#: thread and written from the monitor's.
_SNAPSHOT: dict[tuple[str, str], dict[str, float]] = {}
_SNAPSHOT_LOCK = threading.Lock()


def _observations(key: str) -> list[Observation]:
    """One observation per host in the latest scrape, for registry *key*."""
    # Deferred: this module is imported to declare its rows, which must not
    # require OTel to be installed. By the time a callback runs, it is.
    from opentelemetry.metrics import Observation

    with _SNAPSHOT_LOCK:
        latest = list(_SNAPSHOT.items())
    return [
        Observation(
            values[key],
            {HOST_NAME_ATTR: host, RL_WORKER_GROUPS_ATTR: groups},
        )
        for (host, groups), values in latest
        if key in values
    ]


# Two callbacks rather than one parameterised closure: lens holds the function
# for the life of the process, so each has to be a stable module-level object.
def _observe_used(options: CallbackOptions) -> list[Observation]:
    return _observations(_USED_KEY)


def _observe_total(options: CallbackOptions) -> list[Observation]:
    return _observations(_TOTAL_KEY)


#: Observable rather than pushed, because this samples once per collection
#: interval rather than on the exporter's cycle. A synchronous gauge is wrong
#: for that either way the SDK leans: on the locked 1.44.0 it drops its value
#: at each collect, so the cycles between two samples carry no point at all,
#: and from 1.45 it holds the last value per attribute set instead, which
#: re-exports a host's startup-time ``rl.worker_groups="none"`` label long
#: after its groups are built. An observable gauge is read on every export, so
#: it always reports the latest snapshot and a label the callback stops
#: yielding simply ends.
HOST_MEMORY_RECORDED_METRICS = (
    RecordedMetric(
        RL_HOST_MEMORY_USED_METRIC,
        kind="observable_gauge",
        unit="By",
        description=(
            "Memory in use on one host, across every process on it. Labelled by "
            "host.name; see rl.worker_groups before aggregating."
        ),
        callback=_observe_used,
    ),
    RecordedMetric(
        RL_HOST_MEMORY_TOTAL_METRIC,
        kind="observable_gauge",
        unit="By",
        description="Memory installed on one host.",
        callback=_observe_total,
    ),
)

register_recorded_metrics(HOST_MEMORY_RECORDED_METRICS)

#: Used only when the config carries no ``logger.gpu_monitoring`` block, which
#: is why it is not the default of any parameter below -- a run that *does*
#: configure a cadence must not be able to silently fall back to this one. Set
#: to match ``GPUMonitoringConfig.collection_interval`` so the host-memory and
#: GPU series line up on a dashboard rather than interleaving.
FALLBACK_COLLECTION_INTERVAL = 10.0

#: Short, because the endpoint is a raylet on the cluster's own network and the
#: loop would rather miss a sample than hold the interval open behind a node
#: that is already wedged.
_SCRAPE_TIMEOUT_S = 5.0

#: How long :meth:`HostMemoryMonitor.stop` waits for the loop to notice.
#:
#: Twice the scrape budget because ``requests`` applies its timeout to the
#: connect and the read separately, plus a margin to parse the response it has
#: already fetched. Deliberately not a function of the collection interval: the
#: loop is either asleep on an :class:`threading.Event` that ``stop`` has
#: already set, or inside one node's scrape, and it checks between nodes.
_SHUTDOWN_GRACE_S = _SCRAPE_TIMEOUT_S * 2 + 1.0


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


def worker_group_names_on_node(node_id: str) -> set[str]:
    """Worker groups with at least one worker on *node_id*, unformatted."""
    with _WORKER_GROUP_LOCK:
        return {
            prefix
            for prefix, node_ids in _WORKER_GROUP_NODES.items()
            if node_id in node_ids
        }


def format_worker_groups(names: Iterable[str]) -> str:
    """Render *names* as :data:`RL_WORKER_GROUPS_ATTR` carries them."""
    return ",".join(sorted(names)) or NO_WORKER_GROUPS


def worker_groups_on_node(node_id: str) -> str:
    """Value of :data:`RL_WORKER_GROUPS_ATTR` for *node_id*."""
    return format_worker_groups(worker_group_names_on_node(node_id))


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
    # Deferred: this module is imported at module scope by
    # nemo_rl.distributed.worker_groups, and must stay importable in a process
    # that never initialises Ray.
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
    # Parsing costs ~16 us a line and only two series are wanted, so drop the
    # other lines first; each kept series keeps its HELP/TYPE lines.
    wanted = tuple(_SERIES_BY_RAY_NAME)
    kept = "\n".join(
        line
        for line in metrics_text.splitlines()
        if line.removeprefix("# HELP ").removeprefix("# TYPE ").startswith(wanted)
    )
    for family in text_string_to_metric_families(kept):
        for sample in family.samples:
            name = _SERIES_BY_RAY_NAME.get(sample.name)
            if name is None:
                continue
            values[registry_key(name)] = float(sample.value)
    return values


class HostMemoryMonitor:
    """Samples every Ray node's host memory on a timer, for lens to export.

    Driver-side and read-only: it scrapes endpoints the cluster already serves
    and touches no worker. Start it once the cluster is up; it tolerates being
    started earlier and simply finds no nodes until there are some.

    It publishes into :data:`_SNAPSHOT` rather than recording: the series are
    observable, so the exporter pulls the latest sample on its own schedule.
    The two intervals are independent, which means a sample can be up to one
    collection interval old when it is exported, and that samples taken
    between two export cycles are never seen. Both are properties of a gauge
    and neither is worth coupling the threads to avoid.
    """

    def __init__(self, collection_interval: float) -> None:
        """Initialise the monitor.

        Args:
            collection_interval: Seconds between samples of every node. Must be
                positive; :func:`start_host_memory_monitoring` is what enforces
                that, since it is the only caller a run goes through.
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
            thread.join(timeout=_SHUTDOWN_GRACE_S)

    def collect_once(self) -> None:
        """Sample every node and publish the result for export. Never raises.

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
        # Scraped here, not in the callbacks, so the exporter never waits on HTTP.
        # Keyed by host, not node: raylets on one box all report the same memory.
        values_by_host: dict[str, dict[str, float]] = {}
        groups_by_host: dict[str, set[str]] = {}
        for node in ray_nodes():
            # Checked between nodes; return without publishing, or unreached
            # hosts' series end.
            if self._stop.is_set():
                return
            groups_by_host.setdefault(node.host, set()).update(
                worker_group_names_on_node(node.node_id)
            )
            if node.host in values_by_host:
                continue
            values = self._scrape(node)
            if values:
                values_by_host[node.host] = values
        sampled = {
            (host, format_worker_groups(groups_by_host[host])): values
            for host, values in values_by_host.items()
        }
        with _SNAPSHOT_LOCK:
            # Replaced, not merged, so a host that left or changed label stops
            # reporting.
            _SNAPSHOT.clear()
            _SNAPSHOT.update(sampled)

    def _scrape(self, node: RayNode) -> dict[str, float]:
        """Fetch and parse one node's memory series, or ``{}`` if unusable.

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
        try:
            return parse_memory_samples(response.text)
        except ValueError:
            # Caught here so one bad page (an HTML error page, a truncated
            # body) costs only this node's point, not the whole sweep.
            warn_once(
                f"host_memory_parse:{node.host}",
                f"could not parse host memory from {node.metrics_url}",
            )
            return {}

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


def start_host_memory_monitoring(collection_interval: float) -> None:
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
    # Refused rather than clamped to something sane: at zero the loop would
    # scrape every raylet as fast as HTTP allows, and a descriptive series is
    # not worth hammering the cluster for. Clamping would report a cadence the
    # config did not ask for, which is harder to notice than no series at all.
    if collection_interval <= 0:
        warn_once(
            "host_memory_interval",
            "host memory monitoring needs a positive collection interval, not "
            f"{collection_interval}; not starting it",
        )
        return
    monitor = HostMemoryMonitor(collection_interval=collection_interval)
    try:
        # Before start(), so the instruments exist no matter when the first
        # export cycle lands. Nothing records these series, so this call is the
        # only thing that ever creates them -- without it the monitor would
        # scrape happily into a snapshot no exporter ever reads.
        create_rl_metric_instruments()
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
    # Outside the guard: the callbacks outlive the monitor and would export this.
    with _SNAPSHOT_LOCK:
        _SNAPSHOT.clear()
