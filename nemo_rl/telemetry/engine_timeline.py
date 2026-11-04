# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Per-request engine timeline switch and cross-host clock calibration.

``NRL_COLLECT_TIMELINE_FIELDS=1`` makes the vLLM and TRT-LLM servers stamp each response with
epoch-microsecond milestones and the host whose clock stamped them. Hosts' clocks disagree, so the
GRPO driver measures each Ray host's offset once at startup (:func:`calibrate_cluster_clocks`) and a
consumer such as :mod:`nemo_rl.telemetry.llm_span_breakdown_hook` maps the stamps onto one clock.
"""

import json
import logging
import os
import socket
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


def timeline_collection_enabled() -> bool:
    """Whether NRL_COLLECT_TIMELINE_FIELDS opts in to timeline collection.

    One switch, off by default, for the generation servers' per-request timing
    fields and the clock calibration that lets a consumer line them up.
    """
    return os.environ.get("NRL_COLLECT_TIMELINE_FIELDS", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def timeline_clock_id() -> str:
    """Identify this process's wall clock in timeline records.

    Processes on one host share a clock, so the host name is the id that the
    offsets in ``clock_calibration.jsonl`` are keyed by.
    """
    return socket.gethostname()


def append_record(path: str, record: dict[str, Any]) -> None:
    """Append one best-effort JSON line to *path*; a failed write is dropped."""
    try:
        payload = (json.dumps(record, separators=(",", ":")) + "\n").encode()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
    except (OSError, TypeError, ValueError):
        pass


def clock_sample(
    t0_ns: int, t3_ns: int, arrival_ns: int, departure_ns: int
) -> tuple[float, float]:
    """Offset and uncertainty of a remote clock from one round trip, in µs.

    The caller sends at ``t0_ns`` and receives at ``t3_ns`` on its own clock; the
    remote clock reads ``arrival_ns`` on receipt and ``departure_ns`` on reply.

    Args:
        t0_ns: Caller clock when the probe was sent.
        t3_ns: Caller clock when the reply arrived.
        arrival_ns: Remote clock when the probe arrived.
        departure_ns: Remote clock when the reply left.

    Returns:
        ``(offset_us, uncertainty_us)``. The offset is remote minus caller, so a
        remote timestamp maps to the caller's clock as ``ts - offset``. The
        uncertainty is half the network round trip.
    """
    offset_us = ((arrival_ns - t0_ns) + (departure_ns - t3_ns)) / 2_000
    uncertainty_us = max(0, (t3_ns - t0_ns) - (departure_ns - arrival_ns)) / 2_000
    return offset_us, uncertainty_us


def _clock_probe() -> dict[str, Any]:  # pragma: no cover
    """Runs on a Ray node: report that host's clock on arrival and on reply."""
    arrival_ns = time.time_ns()
    return {
        "clock_id": timeline_clock_id(),
        "arrival_ns": arrival_ns,
        "departure_ns": time.time_ns(),
    }


def _probe_node_clock(
    node_id: str, *, samples: int, timeout_s: float
) -> Optional[dict[str, Any]]:
    """Measure one Ray node's clock; keep the sample with the least uncertainty.

    Returns:
        The calibration record, or None if the node could not be probed.
    """
    # Imported here so the engine servers can import this module without Ray.
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    probe = ray.remote(num_cpus=0)(_clock_probe).options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)
    )
    best: Optional[dict[str, Any]] = None
    pending = None
    try:
        pending = probe.remote()
        ray.get(pending, timeout=timeout_s)  # warm-up: worker start-up is no sample
        for _ in range(samples):
            t0_ns = time.time_ns()
            pending = probe.remote()
            reply = ray.get(pending, timeout=timeout_s)
            t3_ns = time.time_ns()
            offset_us, uncertainty_us = clock_sample(
                t0_ns, t3_ns, reply["arrival_ns"], reply["departure_ns"]
            )
            if best is None or uncertainty_us < best["uncertainty_us"]:
                best = {
                    "event": "clock_calibration",
                    "host_clock_id": reply["clock_id"],
                    "offset_us": offset_us,
                    "uncertainty_us": uncertainty_us,
                    "measured_at_us": t3_ns // 1_000,
                    "sample_count": samples,
                    "ray_node_id": node_id,
                }
    except (ray.exceptions.RayError, TimeoutError) as error:
        logger.warning(
            "Timeline clock probe failed for node %s: %s", node_id, type(error).__name__
        )
        if pending is not None:
            ray.cancel(pending, force=True)
        return None
    return best


def calibrate_cluster_clocks(
    path: str, *, samples: int = 8, timeout_s: float = 10.0
) -> int:
    """Measure every live Ray host's clock against this process and record it.

    Hosts' wall clocks disagree by up to milliseconds, which distorts any span
    that crosses hosts. Run once at startup, this appends one
    ``clock_calibration`` record per host to *path*. A host that cannot be
    probed is skipped with a warning and keeps its raw timestamps.

    Args:
        path: ``clock_calibration.jsonl`` to append to.
        samples: Round trips per host; the one with the least uncertainty is kept.
        timeout_s: Per round trip.

    Returns:
        The number of hosts recorded.
    """
    import ray  # Imported here so the engine servers can import this module without Ray.

    reference = timeline_clock_id()
    nodes = [node for node in ray.nodes() if node["Alive"]]
    recorded: set[str] = set()
    for node in nodes:
        record = _probe_node_clock(node["NodeID"], samples=samples, timeout_s=timeout_s)
        if record is None:
            continue
        host = record["host_clock_id"]
        if host in recorded:
            logger.warning(
                "Timeline clock %s was reported by more than one Ray node; keeping the first",
                host,
            )
            continue
        recorded.add(host)
        append_record(
            path,
            {
                **record,
                "reference_clock_id": reference,
                "expected_ray_node_count": len(nodes),
            },
        )
    logger.info(
        "Timeline clocks calibrated for %d of %d Ray nodes", len(recorded), len(nodes)
    )
    return len(recorded)
