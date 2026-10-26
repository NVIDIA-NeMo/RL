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

"""Memory held by one process and everything it forked, reported by itself.

:mod:`nemo_rl.telemetry.host_memory` says how full each host is, not who filled
it. Ray does publish per-process series on the same page, but it keys worker
processes on their title, which comes from the actor *class*: two worker groups
built from one class -- distillation's teacher and student, say -- land in the
same series, and a process a worker forks itself (vLLM's EngineCore, a compile
pool) is in none. So each process measures itself instead. Every process that
sets up telemetry, the driver included, reports one point for itself and its
descendants, labelled with its worker group, rank and pid.

The value is PSS, not RSS. RSS counts a shared page once for every process that
maps it -- CUDA mappings and the object store's shared memory above all -- so
RSS summed over a group or a host over-counts. PSS splits each shared page
between the processes sharing it, so the points can be summed.

Read inside the exporter's callback, not on a side thread. ``psutil`` reads PSS
from ``/proc/<pid>/smaps_rollup``, whose cost grows with the process's mapping
count: measured at about 5 ms for 10k mappings and 11 ms for 30k (Linux 6.18),
once per export. The full ``smaps`` file is ~20x slower and is never read.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any, Optional

from nemo_rl.telemetry.metrics import create_rl_metric_instruments, warn_once
from nemo_rl.telemetry.vocabulary import RecordedMetric, register_recorded_metrics

if TYPE_CHECKING:
    import psutil
    from opentelemetry.metrics import CallbackOptions, Observation

#: Proportional set size of one process and its descendants, in bytes.
RL_PROCESS_MEMORY_PSS_METRIC = "rl.process.memory.pss"

#: Which worker group a point's process belongs to. Also a resource attribute
#: on workers, but repeated on the point: ranks restart at zero in every group,
#: and a backend that keys series on point attributes alone would otherwise
#: merge rank 0 of every group into one series.
RL_WORKER_GROUP_ATTR = "rl.worker_group"

#: The reporting process. The value covers its descendants too, so this is the
#: pid of the process that owns the number, not of every process inside it.
PROCESS_PID_ATTR = "process.pid"

#: :data:`RL_WORKER_GROUP_ATTR` for the driver, which belongs to no worker group
#: but holds memory of its own -- the training loop, the logger, datasets.
DRIVER_WORKER_GROUP = "driver"

#: Attributes of this process's point, or ``None`` while it is not reporting.
#: Written by :func:`start_process_memory_reporting` and read from the
#: exporter's thread.
_POINT_ATTRIBUTES: Optional[dict[str, Any]] = None
_POINT_ATTRIBUTES_LOCK = threading.Lock()


def process_tree_pss(process: Optional[psutil.Process] = None) -> int:
    """PSS of *process* (default: this one) and every descendant, in bytes.

    A descendant that exits, or turns out to be unreadable, between being
    listed and being read is skipped rather than failing the sum: it holds
    nothing worth reporting by then. Failing to read the root raises, since
    there is no number to report at all.
    """
    import psutil

    root = process if process is not None else psutil.Process()
    total = root.memory_full_info().pss
    for child in root.children(recursive=True):
        try:
            total += child.memory_full_info().pss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _observe_pss(options: CallbackOptions) -> list[Observation]:
    """This process's one point, or none while it is not reporting."""
    # Deferred: this module is imported to declare its row, which must not
    # require OTel to be installed. By the time a callback runs, it is.
    from opentelemetry.metrics import Observation

    with _POINT_ATTRIBUTES_LOCK:
        attributes = _POINT_ATTRIBUTES
    if attributes is None:
        return []
    # Broad on purpose: this runs on the exporter's thread, beside every other
    # series the process exports. No PSS on this platform, or a /proc that
    # refuses the read, has to cost this point and nothing else.
    try:
        value = process_tree_pss()
    except Exception as exc:
        warn_once(
            "process_memory",
            f"could not read this process's PSS, so {RL_PROCESS_MEMORY_PSS_METRIC} "
            f"is not reported from it: {exc!r}",
        )
        return []
    return [Observation(value, attributes)]


#: An up-down counter rather than a gauge, because the points are meant to be
#: summed: by worker group, by host, or across the job. That is the kind OTel
#: uses for memory usage, and backends aggregate it as a sum.
PROCESS_MEMORY_RECORDED_METRICS = (
    RecordedMetric(
        RL_PROCESS_MEMORY_PSS_METRIC,
        kind="observable_up_down_counter",
        unit="By",
        description=(
            "Proportional set size of one process and its descendants. "
            "Labelled by rl.worker_group, nv.dl.rank and process.pid; safe to sum."
        ),
        callback=_observe_pss,
    ),
)

register_recorded_metrics(PROCESS_MEMORY_RECORDED_METRICS)


def start_process_memory_reporting(*, rank: int, worker_group: Optional[str]) -> None:
    """Report this process's memory from the next export on. Never raises.

    Call once telemetry is exporting. A *worker_group* of ``None`` or ``""``
    leaves :data:`RL_WORKER_GROUP_ATTR` off the point, as it is left off the
    resource, for actors built outside ``RayWorkerGroup``.
    """
    global _POINT_ATTRIBUTES
    try:
        from nemo.lens.semconv import NV_DL_RANK

        attributes: dict[str, Any] = {
            NV_DL_RANK: rank,
            PROCESS_PID_ATTR: os.getpid(),
        }
        if worker_group:
            attributes[RL_WORKER_GROUP_ATTR] = worker_group
        # Observable, so nothing records it into existence: without its
        # instrument the callback above is never called.
        create_rl_metric_instruments()
    except Exception:
        warn_once(
            "process_memory_start", "could not start per-process memory reporting"
        )
        return
    with _POINT_ATTRIBUTES_LOCK:
        _POINT_ATTRIBUTES = attributes


def stop_process_memory_reporting() -> None:
    """Stop reporting. The series ends at the next export.

    OTel cannot remove an observable instrument once created, so this does not
    try to: the callback simply stops returning a point.
    """
    global _POINT_ATTRIBUTES
    with _POINT_ATTRIBUTES_LOCK:
        _POINT_ATTRIBUTES = None
