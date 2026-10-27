# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Helpers shared by the timeline analysis and export scripts. Standard library only."""

import json
import math
from pathlib import Path
from typing import Any, Optional

# A model_call is cut at these milestones, in the order a request passes them.
# Each segment is (name, start milestone, end milestone, class). ``class`` is
# "compute" for engine work, "queue" for waiting on an engine scheduler, and
# "plumbing" for everything else (network, proxies, (de)serialisation).
# Names are looked up with the ``llm_`` prefix first, then the older ``vllm_``.
SEGMENTS = [
    ("client_to_frontend", "start", "frontend_recv", "plumbing"),
    ("frontend_route", "frontend_recv", "ctx_arrival", "plumbing"),
    ("ctx_submit", "ctx_arrival", "ctx_queued", "plumbing"),
    ("ctx_sched_queue", "ctx_queued", "ctx_first_scheduled", "queue"),
    ("ctx_prefill", "ctx_first_scheduled", "ctx_done", "compute"),
    ("kv_handoff", "ctx_done", "llm_arrival", "plumbing"),
    ("gen_submit", "llm_arrival", "queued", "plumbing"),
    ("gen_sched_queue", "queued", "first_scheduled", "queue"),
    ("gen_first_token", "first_scheduled", "first_token", "compute"),
    ("decode", "first_token", "last_token", "compute"),
    ("response_build", "last_token", "llm_response_ready", "plumbing"),
    ("llm_to_gym", "llm_response_ready", "gym_response_received", "plumbing"),
    ("gym_proxy", "gym_response_received", "gym_response_ready", "plumbing"),
    (
        "gym_to_openhands",
        "gym_response_ready",
        "openhands_response_received",
        "plumbing",
    ),
    ("openhands_tail", "openhands_response_received", "end", "plumbing"),
]
# Without PD disaggregation there is no frontend/ctx leg: the request goes
# straight from the client to the one engine.
AGG_SEGMENTS = [("client_to_frontend", "start", "llm_arrival", "plumbing")] + [
    segment
    for segment in SEGMENTS
    if segment[1]
    in (
        "llm_arrival",
        "queued",
        "first_scheduled",
        "first_token",
        "last_token",
        "llm_response_ready",
        "gym_response_received",
        "gym_response_ready",
        "openhands_response_received",
    )
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every valid JSON line of *path*; a missing file is an empty list."""
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def milestone(args: dict[str, Any], name: str) -> Optional[int]:
    """A milestone timestamp in epoch µs, tolerating the old ``vllm_`` prefix."""
    for key in (f"{name}_ts_us", f"{name.replace('llm_', 'vllm_', 1)}_ts_us"):
        value = args.get(key)
        if isinstance(value, int) and value > 0:
            return value
    # gym_llm_response_received_ts_us was gym_vllm_response_received_ts_us.
    if name == "gym_response_received":
        for key in (
            "gym_llm_response_received_ts_us",
            "gym_vllm_response_received_ts_us",
        ):
            value = args.get(key)
            if isinstance(value, int) and value > 0:
                return value
    return None


def breakdown(
    event: dict[str, Any],
) -> Optional[tuple[bool, list[tuple[str, int, int, str]]]]:
    """Cut one ok ``model_call`` into segments.

    Returns:
        ``(disaggregated, [(name, start_us, end_us, class), ...])``, or None if the
        call lacks the milestones. A segment's duration is ``end_us - start_us``;
        it is negative when the two milestones come from clocks that disagree.
    """
    args = event["args"]
    points: dict[str, Optional[int]] = {
        "start": event["ts"],
        "end": event["ts"] + event["dur"],
    }
    for name in {m for s in SEGMENTS for m in s[1:3]} - {"start", "end"}:
        points[name] = milestone(args, name)
    disagg = points["ctx_arrival"] is not None
    plan = SEGMENTS if disagg else AGG_SEGMENTS
    if any(points[s[1]] is None or points[s[2]] is None for s in plan):
        return None
    return disagg, [(n, points[a], points[b], c) for n, a, b, c in plan]  # type: ignore[misc]


def inflight_series(
    intervals: list[tuple[float, float]], bin_us: float, *, t0: Optional[float] = None
) -> list[float]:
    """Mean in-flight count per bin from ``(start, end)`` intervals (sweep line).

    Args:
        intervals: Request intervals in µs.
        bin_us: Bin width in µs.
        t0: Start of the first bin; defaults to the earliest interval start.
    """
    if not intervals:
        return []
    if t0 is None:
        t0 = min(start for start, _ in intervals)
    t1 = max(end for _, end in intervals)
    series = [0.0] * (int((t1 - t0) // bin_us) + 1)
    for start, end in intervals:
        if end <= start:
            continue
        first, last = max(0, int((start - t0) // bin_us)), int((end - t0) // bin_us)
        for b in range(first, last + 1):
            lo = max(start, t0 + b * bin_us)
            hi = min(end, t0 + (b + 1) * bin_us)
            series[b] += max(0.0, hi - lo) / bin_us
    return series


def read_clock_offsets(path: Optional[Path]) -> tuple[Optional[str], dict[str, float]]:
    """Reference clock id and ``{host_clock_id: offset_us}`` from clock_calibration.jsonl.

    The offset is host minus reference, so a host timestamp maps to the reference
    clock as ``ts - offset``. The reference's own offset is 0. Returns
    ``(None, {})`` if the file is missing or names more than one reference.
    """
    rows = load_jsonl(path) if path is not None else []
    references = {r["reference_clock_id"] for r in rows if "reference_clock_id" in r}
    if len(references) != 1:
        return None, {}
    reference = next(iter(references))
    offsets = {
        r["host_clock_id"]: float(r["offset_us"])
        for r in rows
        if "host_clock_id" in r and math.isfinite(r.get("offset_us", math.nan))
    }
    offsets[reference] = 0.0
    return reference, offsets
