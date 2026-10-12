#!/usr/bin/env python3
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

"""NemoRL Efficiency Log Parser & Perfetto Trace Generator.

Parses Timer events from NemoRL log files and produces:
  1. Chrome Trace Event Format JSON  (loadable in ui.perfetto.dev or chrome://tracing)
  2. Self-contained HTML with an interactive timeline visualisation

Usage:
    python generate_trace.py /path/to/logs/ray-driver.log
    python generate_trace.py /path/to/logs/ray-driver.log -o output_dir/
    python generate_trace.py /path/to/logs/  # scans all *.log files
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RAY_DEDUP_RE = re.compile(r"\[repeated (\d+)x across cluster\]")

TIMER_RE = re.compile(
    r"INFO:nemo_rl\.utils\.timer:timer\.py:\d+:\s*"
    r"\[(?P<context>[^\]]+)\]\s+"
    r"(?P<label>\S+)\s+"
    r"(?P<event>start|end|record|mark)"
    r"(?:\s+elapsed=(?P<elapsed>[\d.]+)s)?"
    r"(?:\s+meta=(?P<meta>\{.*?\}))?"
    r"\s+ts=(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)"
)

CONTEXT_KV_RE = re.compile(r"(\w+)=(\S+)")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_context(ctx: str) -> dict[str, str]:
    return dict(CONTEXT_KV_RE.findall(ctx))


def _ts_to_us(iso: str) -> int:
    """ISO-8601 UTC string -> microseconds since epoch."""
    dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)


def parse_log_file(path: str | Path) -> list[dict]:
    events: list[dict] = []
    with open(path, errors="replace") as fh:
        for raw in fh:
            clean = ANSI_RE.sub("", raw)
            dedup_match = RAY_DEDUP_RE.search(clean)
            dedup_count = int(dedup_match.group(1)) if dedup_match else 0
            line = RAY_DEDUP_RE.sub("", clean)
            m = TIMER_RE.search(line)
            if not m:
                continue
            ctx = _parse_context(m.group("context"))
            elapsed = float(m.group("elapsed")) if m.group("elapsed") else None
            events.append(
                {
                    "worker": ctx.get("worker", "unknown"),
                    "rank": int(ctx["rank"]) if "rank" in ctx else None,
                    "hostname": ctx.get("hostname", "unknown"),
                    "label": m.group("label"),
                    "event": m.group("event"),
                    "elapsed": elapsed,
                    "meta": m.group("meta"),
                    "ts_us": _ts_to_us(m.group("ts")),
                    "ts_iso": m.group("ts"),
                    "source": os.path.basename(str(path)),
                    "dedup_count": dedup_count,
                }
            )
    return events


def parse_log_directory(dirpath: str | Path) -> list[dict]:
    all_events: list[dict] = []
    dp = Path(dirpath)
    for p in sorted(dp.glob("*.log")):
        all_events.extend(parse_log_file(p))
    all_events.sort(key=lambda e: e["ts_us"])
    return all_events


# ---------------------------------------------------------------------------
# Chrome Trace Event Format (for Perfetto / chrome://tracing)
# ---------------------------------------------------------------------------

WORKER_ORDER = [
    "driver",
    "megatron_policy",
    "rollout",
    "collector",
    "nemo_gym",
]


def _worker_sort_key(w: str) -> int:
    try:
        return WORKER_ORDER.index(w)
    except ValueError:
        return len(WORKER_ORDER)


def _match_spans(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """Match start/end pairs into complete spans, with cross-rank fallback.

    Returns (spans, unmatched) where each span is a dict with ts_start,
    ts_end, and original event metadata.
    """
    pending: dict[str, list[dict]] = defaultdict(list)
    spans: list[dict] = []
    unmatched_ends: list[dict] = []

    for ev in events:
        if ev["event"] == "start":
            key = f"{ev['worker']}|{ev['rank']}|{ev['label']}"
            pending[key].append(ev)
        elif ev["event"] == "end":
            key = f"{ev['worker']}|{ev['rank']}|{ev['label']}"
            if pending[key]:
                s = pending[key].pop(0)
                spans.append(
                    {
                        **ev,
                        "ts_start": s["ts_us"],
                        "ts_end": ev["ts_us"],
                        "matched_rank": ev["rank"],
                    }
                )
            else:
                unmatched_ends.append(ev)
        elif ev["event"] == "record" and ev["elapsed"] is not None:
            dur_us = int(ev["elapsed"] * 1_000_000)
            spans.append(
                {
                    **ev,
                    "ts_start": ev["ts_us"] - dur_us,
                    "ts_end": ev["ts_us"],
                    "matched_rank": ev["rank"],
                }
            )
        elif ev["event"] == "mark":
            spans.append(
                {
                    **ev,
                    "ts_start": ev["ts_us"],
                    "ts_end": ev["ts_us"],
                    "matched_rank": ev["rank"],
                }
            )

    for ev in unmatched_ends:
        wl_prefix = f"{ev['worker']}|"
        suffix = f"|{ev['label']}"
        matched = False
        best_key, best_idx, best_dist = None, None, float("inf")
        for key in list(pending.keys()):
            if key.startswith(wl_prefix) and key.endswith(suffix) and pending[key]:
                for i, cand in enumerate(pending[key]):
                    dist = ev["ts_us"] - cand["ts_us"]
                    if 0 < dist < best_dist:
                        best_dist = dist
                        best_key = key
                        best_idx = i
        if best_key is not None:
            s = pending[best_key].pop(best_idx)
            if not pending[best_key]:
                del pending[best_key]
            spans.append(
                {
                    **ev,
                    "ts_start": s["ts_us"],
                    "ts_end": ev["ts_us"],
                    "matched_rank": None,
                    "dedup_count": max(
                        ev.get("dedup_count", 0), s.get("dedup_count", 0)
                    ),
                }
            )
            matched = True
        if not matched:
            spans.append(
                {
                    **ev,
                    "ts_start": ev["ts_us"],
                    "ts_end": ev["ts_us"],
                    "matched_rank": ev["rank"],
                }
            )

    flat_unmatched = []
    for q in pending.values():
        flat_unmatched.extend(q)

    return spans, flat_unmatched


def _consolidate_dedup_workers(spans: list[dict]) -> list[dict]:
    """For workers with any dedup events, merge all ranks into one row.

    Remove overlapping duplicate spans for the same label.
    """
    workers_with_dedup: set[str] = set()
    for sp in spans:
        if sp.get("dedup_count", 0) > 0:
            workers_with_dedup.add(sp["worker"])

    result: list[dict] = []
    to_dedup: list[dict] = []
    for sp in spans:
        if sp["worker"] in workers_with_dedup:
            sp["matched_rank"] = None
            to_dedup.append(sp)
        else:
            result.append(sp)

    groups: dict[str, list[dict]] = defaultdict(list)
    for sp in to_dedup:
        groups[f"{sp['worker']}|{sp['label']}"].append(sp)

    for grp in groups.values():
        grp.sort(
            key=lambda s: (-s.get("dedup_count", 0), -(s["ts_end"] - s["ts_start"]))
        )
        kept: list[dict] = []
        for sp in grp:
            if not any(
                sp["ts_start"] < k["ts_end"] and sp["ts_end"] > k["ts_start"]
                for k in kept
            ):
                kept.append(sp)
        result.extend(kept)

    return result


def build_chrome_trace(events: list[dict]) -> dict:
    workers = sorted({e["worker"] for e in events}, key=_worker_sort_key)
    pid_map = {w: i + 1 for i, w in enumerate(workers)}

    trace_events: list[dict] = []

    for w, pid in pid_map.items():
        trace_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"name": w},
            }
        )

    spans, _ = _match_spans(events)
    spans = _consolidate_dedup_workers(spans)

    tid_max_dedup: dict[tuple[int, int], int] = defaultdict(int)
    tid_has_rank: dict[tuple[int, int], int | None] = {}
    span_records: list[tuple[dict, int, int]] = []

    for sp in spans:
        pid = pid_map[sp["worker"]]
        rank = sp.get("matched_rank")
        tid = rank if rank is not None else 0
        key = (pid, tid)
        tid_max_dedup[key] = max(tid_max_dedup[key], sp.get("dedup_count", 0))
        if key not in tid_has_rank:
            tid_has_rank[key] = rank
        span_records.append((sp, pid, tid))

    for key, rank in tid_has_rank.items():
        pid, tid = key
        dedup = tid_max_dedup[key]
        if rank is not None:
            tname = f"rank-{tid}"
        elif dedup > 0:
            tname = f"all {dedup + 1} ranks (aggregated)"
        else:
            tname = "main"
        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": tid,
                "args": {"name": tname},
            }
        )

    for sp, pid, tid in span_records:
        args = {"hostname": sp["hostname"], "source": sp["source"]}
        if sp.get("dedup_count"):
            args["repeated_across_cluster"] = sp["dedup_count"]
        cat = sp["label"].split("/")[0] if "/" in sp["label"] else "misc"

        if sp["event"] == "mark":
            args["meta"] = sp.get("meta")
            trace_events.append(
                {
                    "name": sp["label"],
                    "cat": "mark",
                    "ph": "i",
                    "ts": sp["ts_us"],
                    "s": "g",
                    "pid": pid,
                    "tid": tid,
                    "args": args,
                }
            )
        else:
            dur_us = sp["ts_end"] - sp["ts_start"]
            ts_start = sp["ts_start"]
            if sp.get("elapsed") is not None:
                args["elapsed_s"] = sp["elapsed"]
                elapsed_us = int(sp["elapsed"] * 1_000_000)
                # For cross-rank aggregated events, start/end timestamps may
                # come from different ranks, producing wildly inflated dur.
                # Use the actual perf_counter elapsed and anchor at ts_end.
                if elapsed_us < dur_us:
                    dur_us = elapsed_us
                    ts_start = sp["ts_end"] - elapsed_us
            if dur_us > 0:
                trace_events.append(
                    {
                        "name": sp["label"],
                        "cat": cat,
                        "ph": "X",
                        "ts": ts_start,
                        "dur": dur_us,
                        "pid": pid,
                        "tid": tid,
                        "args": args,
                    }
                )

    return {"traceEvents": trace_events}


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

EFFICIENCY_CATEGORIES = [
    "init/total",
    "idle/buffer_starvation",
    "idle/buffer_full_backoff",
    "idle/refit_bubble",
    "idle/generation_limit_pause",
    "idle/refit_event_wait",
    "idle/validation",
    "wasted/failed_trajectory",
]


def compute_summary(events: list[dict]) -> dict:
    """Aggregate durations by label for efficiency reporting.

    Keyed as 'label' for driver events and 'worker:label' for non-driver
    efficiency events so the UI can distinguish driver-timeline waste from
    collector-side waste.
    """
    durations: dict[str, float] = defaultdict(float)
    pending: dict[str, list[float]] = defaultdict(list)

    for ev in events:
        is_driver = ev["worker"] == "driver"
        is_efficiency = ev["label"] in EFFICIENCY_CATEGORIES

        if not is_driver and not is_efficiency:
            continue

        key = ev["label"] if is_driver else f"{ev['worker']}:{ev['label']}"

        if ev["event"] == "start":
            pending[key].append(ev["ts_us"])
        elif ev["event"] == "end" and ev["elapsed"] is not None:
            durations[key] += ev["elapsed"]
            if pending[key]:
                pending[key].pop(0)
        elif ev["event"] == "record" and ev["elapsed"] is not None:
            durations[key] += ev["elapsed"]

    return dict(durations)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def _build_html(events: list[dict], trace_json: dict, summary: dict) -> str:
    trace_b64 = base64.b64encode(json.dumps(trace_json).encode()).decode()
    events_json = json.dumps(events, default=str)
    summary_json = json.dumps(summary)

    return (
        _HTML_TEMPLATE.replace("__TRACE_B64__", trace_b64)
        .replace("__EVENTS_JSON__", events_json)
        .replace("__SUMMARY_JSON__", summary_json)
    )


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NemoRL Efficiency Trace</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0d1117;
  --surface: #161b22;
  --border: #30363d;
  --text: #e6edf3;
  --text-dim: #8b949e;
  --accent: #58a6ff;
  --green: #3fb950;
  --red: #f85149;
  --orange: #d29922;
  --purple: #bc8cff;
  --pink: #f778ba;
  --cyan: #39d2c0;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
  overflow-x: hidden;
}

.header {
  padding: 24px 32px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 16px;
}

.header h1 { font-size: 20px; font-weight: 700; letter-spacing: -0.3px; }
.header .subtitle { color: var(--text-dim); font-size: 14px; flex: 1; }
.header .actions { display: flex; gap: 8px; }

.btn {
  background: #21262d;
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 14px;
  cursor: pointer;
  font-size: 12px;
  font-family: 'JetBrains Mono', monospace;
  transition: background .15s;
}
.btn:hover { background: #30363d; }
.btn-primary { background: #238636; border-color: #238636; color: #fff; }
.btn-primary:hover { background: #2ea043; }

.stats-bar {
  display: flex;
  gap: 32px;
  padding: 16px 32px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
}

.stat { display: flex; flex-direction: column; gap: 2px; }

.stat-label {
  font-size: 11px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-family: 'JetBrains Mono', monospace;
}

.stat-value {
  font-size: 22px;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
}

.stat-value.good { color: var(--green); }
.stat-value.warn { color: var(--orange); }
.stat-value.bad { color: var(--red); }

.controls-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 32px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
  font-size: 12px;
}
.controls-bar label { color: var(--text-dim); font-family: 'JetBrains Mono', monospace; }
.controls-bar input[type=range] { width: 140px; accent-color: var(--accent); }
.controls-bar select, .controls-bar input[type=text] {
  background: var(--bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 4px; padding: 3px 8px; font-size: 12px;
  font-family: 'JetBrains Mono', monospace;
}

.timeline-container { padding: 24px 32px; }

.timeline-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 16px;
}

.timeline-header h2 { font-size: 15px; font-weight: 600; }

.legend {
  display: flex;
  gap: 14px;
  font-size: 12px;
  margin-left: auto;
  flex-wrap: wrap;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 5px;
  color: var(--text-dim);
  font-family: 'JetBrains Mono', monospace;
}

.legend-swatch {
  width: 12px;
  height: 12px;
  border-radius: 2px;
}

.timeline-scroll { overflow-x: auto; padding-bottom: 12px; }

.track-group { margin-bottom: 20px; }

.track-group-label {
  font-size: 13px;
  font-weight: 600;
  color: var(--accent);
  padding: 6px 10px;
  background: rgba(88, 166, 255, 0.08);
  border-left: 3px solid var(--accent);
  margin-bottom: 8px;
  font-family: 'JetBrains Mono', monospace;
}

.track {
  display: flex;
  align-items: stretch;
  margin-bottom: 3px;
}

.track-label {
  width: 160px;
  min-width: 160px;
  font-size: 11px;
  color: var(--text-dim);
  font-family: 'JetBrains Mono', monospace;
  padding-right: 8px;
  text-align: right;
  display: flex;
  align-items: center;
  justify-content: flex-end;
}

.track-content {
  position: relative;
  flex: 1;
  background: rgba(255,255,255,0.02);
  border-radius: 3px;
}

.event-bar {
  position: absolute;
  height: 22px;
  border-radius: 3px;
  cursor: pointer;
  display: flex;
  align-items: center;
  padding: 0 4px;
  font-size: 10px;
  font-family: 'JetBrains Mono', monospace;
  color: rgba(255,255,255,0.9);
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  transition: opacity 0.15s, border-color 0.15s;
  border: 1px solid rgba(255,255,255,0.1);
}

.event-bar:hover {
  opacity: 0.85;
  z-index: 10;
  border-color: rgba(255,255,255,0.5);
}

.event-bar.cat-init        { background: #1f6feb; }
.event-bar.cat-idle        { background: var(--orange); }
.event-bar.cat-wasted      { background: var(--red); }
.event-bar.cat-training    { background: var(--green); }
.event-bar.cat-generation  { background: #388bfd; }
.event-bar.cat-checkpoint  { background: var(--purple); }
.event-bar.cat-timing      { background: #238636; }
.event-bar.cat-misc        { background: #79c0ff; }

.event-bar.cat-idle,
.event-bar.cat-wasted {
  border-style: solid;
  border-width: 1px;
}
.event-bar.cat-idle {
  background: repeating-linear-gradient(
    -45deg, var(--orange), var(--orange) 3px,
    rgba(210,153,34,0.6) 3px, rgba(210,153,34,0.6) 6px
  );
  border-color: var(--orange);
}
.event-bar.cat-wasted {
  background: repeating-linear-gradient(
    -45deg, var(--red), var(--red) 3px,
    rgba(248,81,73,0.6) 3px, rgba(248,81,73,0.6) 6px
  );
  border-color: var(--red);
}

.event-bar.depth-0 { opacity: 0.65; }
.event-bar.depth-1 { opacity: 0.9; }
.event-bar.depth-2 { opacity: 1.0; }
.event-bar.depth-0.cat-idle,
.event-bar.depth-0.cat-wasted { opacity: 0.9; }

.time-axis {
  display: flex;
  align-items: flex-end;
  height: 24px;
  margin-bottom: 8px;
  padding-left: 160px;
  position: relative;
}

.time-tick {
  position: absolute;
  font-size: 10px;
  color: var(--text-dim);
  font-family: 'JetBrains Mono', monospace;
  border-left: 1px solid var(--border);
  padding-left: 4px;
  height: 100%;
  display: flex;
  align-items: flex-end;
}

.tooltip {
  display: none;
  position: fixed;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px 16px;
  font-size: 12px;
  z-index: 1000;
  max-width: 420px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.4);
  pointer-events: none;
}
.tooltip.visible { display: block; }
.tooltip-name {
  font-weight: 700;
  font-size: 13px;
  margin-bottom: 6px;
  font-family: 'JetBrains Mono', monospace;
  color: var(--accent);
}
.tooltip-row {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  margin: 2px 0;
}
.tooltip-key {
  color: var(--text-dim);
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
}
.tooltip-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  text-align: right;
}

.waste-breakdown {
  padding: 24px 32px;
  border-top: 1px solid var(--border);
}
.waste-breakdown h2 {
  font-size: 15px;
  font-weight: 600;
  margin-bottom: 12px;
}
.waste-bars {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-width: 800px;
}
.waste-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  font-family: 'JetBrains Mono', monospace;
}
.waste-name {
  width: 260px;
  min-width: 260px;
  color: var(--text-dim);
  text-align: right;
}
.waste-bar-bg {
  flex: 1;
  height: 18px;
  background: rgba(255,255,255,0.04);
  border-radius: 2px;
  overflow: hidden;
}
.waste-bar-fill { height: 100%; border-radius: 2px; }
.waste-dur {
  width: 80px;
  min-width: 80px;
  text-align: right;
  color: var(--text-dim);
}
.waste-pct {
  width: 50px;
  min-width: 50px;
  text-align: right;
}

.perfetto-note {
  padding: 16px 32px;
  border-top: 1px solid var(--border);
  color: var(--text-dim);
  font-size: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.perfetto-note code {
  background: var(--surface);
  padding: 2px 6px;
  border-radius: 3px;
  font-family: 'JetBrains Mono', monospace;
}
</style>
</head>
<body>

<div class="header">
  <h1>NemoRL Efficiency Trace</h1>
  <span class="subtitle">Post-Training Efficiency Analysis</span>
  <div class="actions">
    <button class="btn" id="btn-download-json">Download JSON</button>
    <button class="btn btn-primary" id="btn-perfetto">Open in Perfetto</button>
  </div>
</div>

<div class="stats-bar" id="stats-bar"></div>

<div class="controls-bar">
  <label>Zoom:</label>
  <input type="range" id="zoom-slider" min="1" max="50" value="1">
  <button class="btn" id="btn-reset">Full View</button>
  <button class="btn" id="btn-training">Training Phase</button>
  <label>Filter:</label>
  <select id="worker-filter"><option value="all">All Workers</option></select>
  <span style="border-left:1px solid var(--border);height:20px;margin:0 4px"></span>
  <label>Step:</label>
  <select id="step-filter"><option value="all">All Steps</option></select>
</div>

<div class="timeline-container">
  <div class="timeline-header">
    <h2>Timeline</h2>
    <div class="legend" id="legend"></div>
  </div>
  <div class="timeline-scroll" id="timeline-scroll">
    <div id="timeline"></div>
  </div>
</div>

<div class="waste-breakdown" id="waste-breakdown"></div>

<div class="perfetto-note">
  For full interactive analysis, download the JSON and open at
  <a href="https://ui.perfetto.dev" style="color:var(--accent)" target="_blank">ui.perfetto.dev</a>
</div>

<div class="tooltip" id="tooltip"></div>

<script>
const TRACE_B64 = "__TRACE_B64__";
const ALL_EVENTS = __EVENTS_JSON__;
const SUMMARY = __SUMMARY_JSON__;

const WORKER_NAMES = {
  'driver': 'Driver Process',
  'megatron_policy': 'MegatronPolicyWorker',
  'rollout': 'AsyncTrajectoryCollector',
  'collector': 'AsyncTrajectoryCollector',
  'nemo_gym': 'NemoGym Environment',
};

// Within the AsyncTrajectoryCollector, label threads by their role
const THREAD_NAMES = {
  'rollout': 'Rollout Pipeline',
  'collector': 'Collection Loop',
};

const CAT_MAP = {
  'init/':          'init',
  'idle/':          'idle',
  'wasted/':        'wasted',
  'total_step_time':'training',
  'policy_':        'training',
  'train':          'training',
  'get_logprobs':   'training',
  'advantage_':     'training',
  'training_prep':  'training',
  'logprob_':       'training',
  'data_processing':'training',
  'overlong_filter':'training',
  'add_loss_mask':  'training',
  'reward_':        'training',
  'exposed_':       'generation',
  'weight_sync':    'checkpoint',
  'offload_':       'checkpoint',
  'timing/':        'timing',
  '_run_rollouts':  'generation',
};

function getCat(name) {
  for (const [prefix, cat] of Object.entries(CAT_MAP)) {
    if (name.startsWith(prefix)) return cat;
  }
  return 'misc';
}

// --- Build spans ---
function buildSpans(events) {
  const pending = {};
  const spans = [];
  const unmatchedEnds = [];
  for (const ev of events) {
    const key = `${ev.worker}|${ev.rank ?? ''}|${ev.label}`;
    if (ev.event === 'start') {
      if (!pending[key]) pending[key] = [];
      pending[key].push(ev);
    } else if (ev.event === 'end') {
      const q = pending[key];
      if (q && q.length > 0) {
        const s = q.shift();
        if (q.length === 0) delete pending[key];
        const dc = Math.max(ev.dedup_count || 0, s.dedup_count || 0);
        let dur = ev.ts_us - s.ts_us;
        let tsStart = s.ts_us;
        if (ev.elapsed != null) {
          const elUs = Math.round(ev.elapsed * 1e6);
          if (elUs < dur) { dur = elUs; tsStart = ev.ts_us - elUs; }
        }
        spans.push({ worker: ev.worker, rank: ev.rank, hostname: ev.hostname, label: ev.label,
          ts_start: tsStart, ts_end: ev.ts_us, duration_us: dur, elapsed: ev.elapsed,
          dedup_count: dc });
      } else { unmatchedEnds.push(ev); }
    } else if (ev.event === 'record' && ev.elapsed != null) {
      const dur = Math.round(ev.elapsed * 1e6);
      spans.push({ worker: ev.worker, rank: ev.rank, hostname: ev.hostname, label: ev.label,
        ts_start: ev.ts_us - dur, ts_end: ev.ts_us, duration_us: dur, elapsed: ev.elapsed,
        dedup_count: ev.dedup_count || 0 });
    }
  }
  for (const ev of unmatchedEnds) {
    const wl = `${ev.worker}|`, sl = `|${ev.label}`;
    // Cross-rank fallback: find the closest preceding start by timestamp
    let bestKey = null, bestStart = null, bestDist = Infinity;
    for (const key of Object.keys(pending)) {
      if (key.startsWith(wl) && key.endsWith(sl) && pending[key].length > 0) {
        for (let i = 0; i < pending[key].length; i++) {
          const cand = pending[key][i];
          const dist = ev.ts_us - cand.ts_us;
          if (dist > 0 && dist < bestDist) {
            bestDist = dist; bestKey = key; bestStart = { idx: i, ev: cand };
          }
        }
      }
    }
    if (bestStart) {
      pending[bestKey].splice(bestStart.idx, 1);
      if (pending[bestKey].length === 0) delete pending[bestKey];
      const s = bestStart.ev;
      const dc = Math.max(ev.dedup_count || 0, s.dedup_count || 0);
      let dur = ev.ts_us - s.ts_us;
      let tsStart = s.ts_us;
      if (ev.elapsed != null) {
        const elUs = Math.round(ev.elapsed * 1e6);
        if (elUs < dur) { dur = elUs; tsStart = ev.ts_us - elUs; }
      }
      spans.push({ worker: ev.worker, rank: null, hostname: ev.hostname, label: ev.label,
        ts_start: tsStart, ts_end: ev.ts_us, duration_us: dur, elapsed: ev.elapsed,
        dedup_count: dc });
    }
  }
  return consolidateDedupWorkers(spans);
}

function consolidateDedupWorkers(spans) {
  const workerHasDedup = {};
  for (const sp of spans) {
    if (sp.dedup_count > 0) workerHasDedup[sp.worker] = true;
  }

  const result = [];
  const toDedup = [];
  for (const sp of spans) {
    if (workerHasDedup[sp.worker]) {
      sp.rank = null;
      toDedup.push(sp);
    } else {
      result.push(sp);
    }
  }

  const groups = {};
  for (const sp of toDedup) {
    const key = `${sp.worker}|${sp.label}`;
    if (!groups[key]) groups[key] = [];
    groups[key].push(sp);
  }
  for (const grp of Object.values(groups)) {
    grp.sort((a, b) => b.dedup_count - a.dedup_count || b.duration_us - a.duration_us);
    const kept = [];
    for (const sp of grp) {
      if (!kept.some(k => sp.ts_start < k.ts_end && sp.ts_end > k.ts_start)) kept.push(sp);
    }
    result.push(...kept);
  }
  return result;
}

function assignDepths(spans) {
  const sortFn = (a, b) => {
    if (a.ts_start !== b.ts_start) return a.ts_start - b.ts_start;
    return (b.ts_end - b.ts_start) - (a.ts_end - a.ts_start);
  };
  const placeFn = (sp, lanes) => {
    for (let d = 0; d < lanes.length; d++) {
      if (sp.ts_start >= lanes[d]) { lanes[d] = sp.ts_end; sp.depth = d; return; }
    }
    sp.depth = lanes.length; lanes.push(sp.ts_end);
  };

  const isTotal = (sp) => sp.label === 'total_step_time' || sp.label === 'total' || sp.label.endsWith('/total');
  const totals = spans.filter(isTotal);
  const others = spans.filter(sp => !isTotal(sp));

  if (totals.length > 0) {
    totals.sort(sortFn);
    const totalLanes = [];
    for (const sp of totals) placeFn(sp, totalLanes);

    // Only separate if totals are sequential containers (few lanes).
    // Highly concurrent totals (e.g. rollout/total with 60+ lanes) stay interleaved.
    if (totalLanes.length <= 3) {
      others.sort(sortFn);
      const otherLanes = [];
      const offset = totalLanes.length;
      for (const sp of others) {
        placeFn(sp, otherLanes);
        sp.depth += offset;
      }
      return totalLanes.length + otherLanes.length;
    }
  }

  const sorted = spans.slice().sort(sortFn);
  const lanes = [];
  for (const sp of sorted) placeFn(sp, lanes);
  return lanes.length;
}

const WORKER_ORDER = ['driver','megatron_policy','rollout','collector','nemo_gym'];
function workerSortKey(w) { const i = WORKER_ORDER.indexOf(w); return i >= 0 ? i : WORKER_ORDER.length; }

// Group key: merge rollout + collector under the same track group
function workerGroupKey(w) { return (w === 'collector') ? 'rollout' : w; }

function buildRows(spans) {
  const groups = {};
  for (const sp of spans) {
    const key = `${sp.worker}|${sp.rank ?? ''}`;
    if (!groups[key]) groups[key] = { worker: sp.worker, rank: sp.rank, spans: [], dedup_count: 0 };
    groups[key].spans.push(sp);
    groups[key].dedup_count = Math.max(groups[key].dedup_count, sp.dedup_count || 0);
  }
  const rows = Object.values(groups);
  rows.sort((a, b) => {
    const wa = workerSortKey(a.worker), wb = workerSortKey(b.worker);
    if (wa !== wb) return wa - wb;
    return (a.rank ?? 0) - (b.rank ?? 0);
  });
  for (const row of rows) row.maxDepth = assignDepths(row.spans);
  return rows;
}

const allSpans = buildSpans(ALL_EVENTS);
let rows = buildRows(allSpans);

// Detect step boundaries from total_step_time spans on the driver
const stepSpans = allSpans
  .filter(s => s.worker === 'driver' && s.label === 'total_step_time')
  .sort((a, b) => a.ts_start - b.ts_start);
const steps = stepSpans.map((s, i) => ({
  num: i + 1, ts_start: s.ts_start, ts_end: s.ts_end, duration_us: s.duration_us
}));

function filterSpansByStep(spans, stepNum) {
  if (stepNum === 'all' || steps.length === 0) return spans;
  if (stepNum === 'init') {
    const firstStepStart = steps[0].ts_start;
    return spans.filter(s => s.ts_end <= firstStepStart);
  }
  const step = steps[parseInt(stepNum) - 1];
  if (!step) return spans;
  return spans.filter(s => s.ts_start < step.ts_end && s.ts_end > step.ts_start);
}

const globalMinTs = allSpans.length > 0 ? Math.min(...allSpans.map(s => s.ts_start)) : 0;
const globalMaxTs = allSpans.length > 0 ? Math.max(...allSpans.map(s => s.ts_end)) : 1e6;
const globalRange = globalMaxTs - globalMinTs || 1;
const totalWallS = globalRange / 1e6;

function fmtS(s) {
  if (s < 0.001) return (s * 1e6).toFixed(0) + 'us';
  if (s < 1) return (s * 1000).toFixed(1) + 'ms';
  if (s < 60) return s.toFixed(1) + 's';
  return (s / 60).toFixed(1) + 'm';
}

// Helper: get summary value for a key, checking both driver-level (bare key)
// and any worker-prefixed keys (e.g. "collector:idle/generation_limit_pause").
function driverVal(key) { return SUMMARY[key] || 0; }
function anyWorkerVal(key) {
  let total = SUMMARY[key] || 0;
  for (const k of Object.keys(SUMMARY)) {
    if (k.endsWith(':' + key)) total += SUMMARY[k];
  }
  return total;
}

// --- Stats bar ---
function renderStats() {
  const bar = document.getElementById('stats-bar');
  const initT = driverVal('setup') + driverVal('init/total');
  const training = driverVal('policy_training');
  const logprobs = driverVal('policy_and_reference_logprobs');

  // Driver-level idle: only waste that happened ON the driver timeline
  const driverIdleCats = ['idle/buffer_starvation','idle/buffer_full_backoff','idle/refit_bubble',
    'idle/refit_event_wait','idle/validation'];
  const driverIdle = driverIdleCats.reduce((s, k) => s + driverVal(k), 0);
  const driverWasted = driverVal('wasted/failed_trajectory');

  // Collector-level idle (shown separately)
  const collectorIdle = anyWorkerVal('idle/generation_limit_pause') - driverVal('idle/generation_limit_pause')
    + anyWorkerVal('idle/buffer_full_backoff') - driverVal('idle/buffer_full_backoff')
    + anyWorkerVal('idle/refit_event_wait') - driverVal('idle/refit_event_wait');

  const stepTime = driverVal('total_step_time');
  const weightSync = driverVal('weight_sync');

  // Step Efficiency = (step_time - driver_idle - driver_wasted) / step_time
  const productive = stepTime - driverIdle - driverWasted;
  const stepPct = stepTime > 0 ? (productive / stepTime * 100) : 0;

  // Goodput = (wall_time - init - driver_idle - driver_wasted) / wall_time
  // Accounts for ALL non-productive time: init/setup cost, in-step idle, and failed work
  const totalWaste = initT + driverIdle + driverWasted;
  const goodputPct = totalWallS > 0 ? ((totalWallS - totalWaste) / totalWallS * 100) : 0;

  const stats = [
    { label: 'Total Wall Time', value: fmtS(totalWallS), cls: '' },
    { label: 'Init', value: fmtS(initT), cls: '' },
    { label: 'Step Time', value: fmtS(stepTime), cls: '' },
    { label: 'Training', value: fmtS(training), cls: 'good' },
    { label: 'Logprobs', value: fmtS(logprobs), cls: 'good' },
    { label: 'Weight Sync', value: fmtS(weightSync), cls: weightSync > 10 ? 'warn' : '' },
    { label: 'Driver Idle', value: fmtS(driverIdle), cls: driverIdle > 0 ? 'warn' : 'good' },
    { label: 'Failed Work', value: fmtS(driverWasted), cls: driverWasted > 0 ? 'bad' : 'good' },
    { label: 'Step Efficiency', value: stepPct.toFixed(1) + '%',
      cls: stepPct > 70 ? 'good' : stepPct > 40 ? 'warn' : 'bad' },
    { label: 'Goodput', value: goodputPct.toFixed(1) + '%',
      cls: goodputPct > 60 ? 'good' : goodputPct > 30 ? 'warn' : 'bad' },
    { label: 'Events', value: String(allSpans.length), cls: '' },
  ];
  bar.innerHTML = stats.map(s =>
    `<div class="stat"><span class="stat-label">${s.label}</span><span class="stat-value ${s.cls}">${s.value}</span></div>`
  ).join('');
}

// --- Legend ---
function renderLegend() {
  const legend = document.getElementById('legend');
  const cats = [
    { name: 'init', color: '#1f6feb' },
    { name: 'training', color: '#3fb950' },
    { name: 'idle (waste)', color: '#d29922' },
    { name: 'generation', color: '#388bfd' },
    { name: 'checkpoint', color: '#bc8cff' },
    { name: 'wasted / error', color: '#f85149' },
  ];
  legend.innerHTML = cats.map(c =>
    `<div class="legend-item"><div class="legend-swatch" style="background:${c.color}"></div>${c.name}</div>`
  ).join('');
}

// --- Coalesce dense same-label spans for rendering ---
// Coalesce idle/waste labels (20+ spans) and any high-frequency polling
// artifacts (100+ spans) into contiguous blocks for visual clarity.
function coalesceSpans(spans, viewRange) {
  const gapThreshold = viewRange * 0.002;
  const groups = {};
  const keep = [];
  for (const sp of spans) {
    const key = `${sp.worker}|${sp.rank ?? ''}|${sp.label}`;
    if (!groups[key]) groups[key] = [];
    groups[key].push(sp);
  }
  for (const [key, grp] of Object.entries(groups)) {
    const label = grp[0].label;
    const isIdleOrWaste = label.startsWith('idle/') || label.startsWith('wasted/');
    const isDriverPolling = grp[0].worker === 'driver' && grp.length >= 100;
    const shouldCoalesce = (isIdleOrWaste && grp.length >= 20) || isDriverPolling;
    if (!shouldCoalesce) { keep.push(...grp); continue; }
    grp.sort((a, b) => a.ts_start - b.ts_start);
    let cur = { ...grp[0], _coalesced: 1 };
    for (let i = 1; i < grp.length; i++) {
      const sp = grp[i];
      if (sp.ts_start - cur.ts_end <= gapThreshold) {
        cur.ts_end = Math.max(cur.ts_end, sp.ts_end);
        cur.duration_us = cur.ts_end - cur.ts_start;
        cur._coalesced++;
      } else {
        keep.push(cur);
        cur = { ...sp, _coalesced: 1 };
      }
    }
    keep.push(cur);
  }
  return keep;
}

// --- Timeline rendering ---
let zoomLevel = 1;
const LANE_H = 24;
const LANE_GAP = 2;

function renderTimeline() {
  const timelineEl = document.getElementById('timeline');
  const wf = document.getElementById('worker-filter').value;
  const sf = document.getElementById('step-filter').value;
  let filtered = wf === 'all' ? allSpans : allSpans.filter(s => s.worker === wf);
  filtered = filterSpansByStep(filtered, sf);

  // Determine view range early so we can coalesce
  let viewMinTs = globalMinTs, viewMaxTs = globalMaxTs;
  const sfParsed = sf;
  if (sfParsed !== 'all' && filtered.length > 0) {
    viewMinTs = Math.min(...filtered.map(s => s.ts_start));
    viewMaxTs = Math.max(...filtered.map(s => s.ts_end));
  }
  const viewRange = viewMaxTs - viewMinTs || 1;

  filtered = coalesceSpans(filtered, viewRange);
  rows = buildRows(filtered);

  const viewWallS = viewRange / 1e6;

  const baseWidth = Math.max(1200, window.innerWidth - 250);
  const timelineWidth = baseWidth * zoomLevel;
  const pxPerUs = timelineWidth / viewRange;

  // Time axis
  let html = `<div class="time-axis" style="position:relative;min-width:${160 + timelineWidth}px">`;
  const idealTicks = Math.max(4, Math.floor(timelineWidth / 100));
  const rawInterval = viewWallS / idealTicks;
  const niceIntervals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600, 900, 1800, 3600];
  const tickInterval = niceIntervals.find(n => n >= rawInterval) || rawInterval;
  for (let t = 0; t <= viewWallS; t += tickInterval) {
    const left = 160 + (t * 1e6 * pxPerUs);
    html += `<div class="time-tick" style="left:${left}px">${fmtS(t)}</div>`;
  }
  html += '</div>';

  // Group rows by worker group (rollout + collector merge into one group)
  const workerGroups = {};
  for (const row of rows) {
    const gk = workerGroupKey(row.worker);
    if (!workerGroups[gk]) workerGroups[gk] = [];
    workerGroups[gk].push(row);
  }

  const workerOrder = [...new Set(rows.map(r => workerGroupKey(r.worker)))].sort((a, b) => workerSortKey(a) - workerSortKey(b));

  for (const worker of workerOrder) {
    const wRows = workerGroups[worker];
    const groupName = WORKER_NAMES[worker] || worker;
    html += `<div class="track-group">`;
    html += `<div class="track-group-label">${groupName}</div>`;

    for (const row of wRows) {
      const threadLabel = THREAD_NAMES[row.worker];
      let label;
      if (threadLabel) {
        label = threadLabel;
      } else if (row.rank != null) {
        label = `rank-${row.rank}`;
      } else if (row.dedup_count > 0) {
        label = `all ${row.dedup_count + 1} ranks (aggr.)`;
      } else {
        label = 'main';
      }
      const contentH = row.maxDepth * (LANE_H + LANE_GAP);

      html += `<div class="track">`;
      html += `<div class="track-label">${label}</div>`;
      html += `<div class="track-content" style="min-width:${timelineWidth}px;height:${contentH}px">`;

      // Find parent (nearest enclosing span) for each span and mark parents
      const byDepth = row.spans.slice().sort((a, b) => a.depth - b.depth);
      for (const sp of byDepth) {
        sp._parent = null;
        sp._isParent = false;
        for (const candidate of byDepth) {
          if (candidate.depth < sp.depth
              && candidate.ts_start <= sp.ts_start
              && candidate.ts_end >= sp.ts_end) {
            if (!sp._parent || candidate.depth > sp._parent.depth) sp._parent = candidate;
          }
        }
        if (sp._parent) sp._parent._isParent = true;
      }

      // Group spans by lane to clamp min-width and prevent overlap
      const byLane = {};
      for (const sp of byDepth) {
        if (!byLane[sp.depth]) byLane[sp.depth] = [];
        byLane[sp.depth].push(sp);
      }
      for (const lane of Object.values(byLane)) {
        lane.sort((a, b) => a.ts_start - b.ts_start);
        for (let i = 0; i < lane.length; i++) {
          lane[i]._nextStartPx = lane[i + 1]
            ? (lane[i + 1].ts_start - viewMinTs) * pxPerUs
            : Infinity;
        }
      }

      for (const sp of byDepth) {
        const left = (sp.ts_start - viewMinTs) * pxPerUs;
        const naturalWidth = sp.duration_us * pxPerUs;
        const maxWidth = sp._nextStartPx - left;
        const width = maxWidth > 0 ? Math.min(Math.max(naturalWidth, 2), maxWidth) : Math.max(naturalWidth, 1);
        const top = sp.depth * (LANE_H + LANE_GAP);
        const cat = getCat(sp.label);
        const depthCls = `depth-${Math.min(sp.depth, 2)}`;
        const durS = sp.duration_us >= 1e6
          ? (sp.duration_us / 1e6).toFixed(1) + 's'
          : (sp.duration_us / 1e3).toFixed(0) + 'ms';
        const shortLabel = sp.label.split('/').pop();
        const barLabel = width > 120 ? `${shortLabel} (${durS})`
          : width > 50 ? shortLabel : '';

        const parentAttr = sp._parent
          ? `data-parent-label="${sp._parent.label}" data-parent-dur="${sp._parent.duration_us}"`
          : '';

        const coalAttr = sp._coalesced > 1 ? `data-coalesced="${sp._coalesced}"` : '';
        html += `<div class="event-bar cat-${cat} ${depthCls}" `
          + `style="left:${left}px;width:${width}px;top:${top}px" `
          + `data-label="${sp.label}" data-worker="${sp.worker}" `
          + `data-rank="${sp.rank ?? ''}" data-host="${sp.hostname}" `
          + `data-dur="${sp.duration_us}" data-start="${sp.ts_start}" `
          + `data-end="${sp.ts_end}" data-depth="${sp.depth}" `
          + `data-dedup="${sp.dedup_count || 0}" ${coalAttr} ${parentAttr}>`
          + `${barLabel}</div>`;
      }
      html += '</div></div>';
    }
    html += '</div>';
  }

  timelineEl.innerHTML = html;
  attachTooltips();
}

// --- Tooltip ---
function attachTooltips() {
  const tip = document.getElementById('tooltip');
  document.querySelectorAll('.event-bar').forEach(bar => {
    bar.addEventListener('mouseenter', () => {
      const d = bar.dataset;
      const durUs = parseInt(d.dur);
      const durS = (durUs / 1e6).toFixed(3);
      const startUs = parseInt(d.start);
      const endUs = parseInt(d.end);
      const fmtTs = (us) => {
        const d = new Date(us / 1000);
        const yyyy = d.getUTCFullYear();
        const mo = String(d.getUTCMonth() + 1).padStart(2, '0');
        const dd = String(d.getUTCDate()).padStart(2, '0');
        const hh = String(d.getUTCHours()).padStart(2, '0');
        const mm = String(d.getUTCMinutes()).padStart(2, '0');
        const ss = String(d.getUTCSeconds()).padStart(2, '0');
        const ms = String(d.getUTCMilliseconds()).padStart(3, '0');
        return `${yyyy}-${mo}-${dd} ${hh}:${mm}:${ss}.${ms} UTC`;
      };
      const startStr = fmtTs(startUs);
      const endStr = fmtTs(endUs);
      const offsetS = ((startUs - globalMinTs) / 1e6).toFixed(2);
      const rank = d.rank ? ` / rank-${d.rank}` : '';
      const dedup = parseInt(d.dedup || '0');
      const dedupRow = dedup > 0
        ? `<div class="tooltip-row"><span class="tooltip-key">scope</span><span class="tooltip-val" style="color:var(--orange)">aggregated across ${dedup + 1} ranks</span></div>`
        : '';
      const coalesced = parseInt(d.coalesced || '0');
      const coalRow = coalesced > 1
        ? `<div class="tooltip-row"><span class="tooltip-key">coalesced</span><span class="tooltip-val" style="color:var(--cyan)">${coalesced} events merged for display</span></div>`
        : '';

      let parentRows = '';
      if (d.parentLabel) {
        const parentDur = parseInt(d.parentDur);
        const diffUs = parentDur - durUs;
        const pct = parentDur > 0 ? (durUs / parentDur * 100) : 0;
        const diffStr = diffUs >= 1e6 ? (diffUs/1e6).toFixed(3)+'s'
          : diffUs >= 1000 ? (diffUs/1000).toFixed(1)+'ms'
          : diffUs+'us';
        const parentShort = d.parentLabel.split('/').pop();
        parentRows = `
          <div style="border-top:1px solid var(--border);margin:4px 0"></div>
          <div class="tooltip-row"><span class="tooltip-key">parent</span><span class="tooltip-val">${d.parentLabel}</span></div>
          <div class="tooltip-row"><span class="tooltip-key">% of ${parentShort}</span><span class="tooltip-val">${pct.toFixed(2)}%</span></div>
          <div class="tooltip-row"><span class="tooltip-key">overhead</span><span class="tooltip-val" style="color:var(--text-dim)">${diffStr}</span></div>`;
      }

      tip.innerHTML = `
        <div class="tooltip-name">${d.label}</div>
        <div class="tooltip-row"><span class="tooltip-key">duration</span><span class="tooltip-val">${durS}s (${(durUs/1000).toFixed(0)}ms)</span></div>
        <div class="tooltip-row"><span class="tooltip-key">start</span><span class="tooltip-val">${startStr}</span></div>
        <div class="tooltip-row"><span class="tooltip-key">end</span><span class="tooltip-val">${endStr}</span></div>
        <div class="tooltip-row"><span class="tooltip-key">offset</span><span class="tooltip-val">+${offsetS}s from trace start</span></div>
        <div class="tooltip-row"><span class="tooltip-key">process</span><span class="tooltip-val">${WORKER_NAMES[d.worker] || d.worker}${rank}</span></div>
        <div class="tooltip-row"><span class="tooltip-key">hostname</span><span class="tooltip-val">${d.host}</span></div>
        ${dedupRow}${coalRow}${parentRows}
      `;
      tip.classList.add('visible');
    });
    bar.addEventListener('mousemove', (e) => {
      const tw = tip.offsetWidth, th = tip.offsetHeight;
      tip.style.left = Math.min(e.clientX + 12, window.innerWidth - tw - 8) + 'px';
      tip.style.top = Math.min(e.clientY + 12, window.innerHeight - th - 8) + 'px';
    });
    bar.addEventListener('mouseleave', () => { tip.classList.remove('visible'); });
  });
}

// --- Waste breakdown ---
function renderWasteBreakdown() {
  const panel = document.getElementById('waste-breakdown');

  // Driver-level waste (affects step efficiency)
  const fullInitKey = null;
  const fullInitVal = driverVal('setup') + driverVal('init/total');
  const driverCats = [
    { key: fullInitKey,                   label: 'initialization (setup + init)',  color: '#1f6feb', override: fullInitVal },
    { key: 'idle/buffer_starvation',      label: 'idle/buffer_starvation',      color: '#d29922' },
    { key: 'idle/refit_bubble',           label: 'idle/refit_bubble',           color: '#d29922' },
    { key: 'idle/validation',             label: 'idle/validation',             color: '#d29922' },
    { key: 'wasted/failed_trajectory',    label: 'wasted/failed_trajectory',    color: '#f85149' },
  ];

  // Collector-level waste (separate system, does not affect driver step efficiency)
  const collectorCats = [
    { key: 'idle/generation_limit_pause', label: 'idle/generation_limit_pause', color: '#d29922' },
    { key: 'idle/buffer_full_backoff',    label: 'idle/buffer_full_backoff',    color: '#d29922' },
    { key: 'idle/refit_event_wait',       label: 'idle/refit_event_wait',       color: '#d29922' },
  ];

  function renderSection(title, cats, useAnyWorker) {
    const vals = cats.map(c => c.override != null ? c.override : (useAnyWorker ? anyWorkerVal(c.key) : driverVal(c.key)));
    const maxDur = Math.max(...vals, 0.001);
    const totalWaste = vals.reduce((s, v) => s + v, 0);

    let html = `<h3 style="font-size:13px;color:var(--text-dim);margin:12px 0 8px;font-family:'JetBrains Mono',monospace">${title}</h3>`;
    for (let i = 0; i < cats.length; i++) {
      const c = cats[i];
      const dur = vals[i];
      const pct = totalWallS > 0 ? (dur / totalWallS * 100) : 0;
      const barPct = maxDur > 0 ? (dur / maxDur * 100) : 0;
      const color = dur > 0 ? c.color : '#21262d';
      const valColor = dur > 0 ? (c.key && c.key.startsWith('wasted') ? 'color:var(--red)' : 'color:var(--orange)') : 'color:var(--text-dim)';
      html += `<div class="waste-row">
        <span class="waste-name">${c.label}</span>
        <div class="waste-bar-bg"><div class="waste-bar-fill" style="width:${barPct}%;background:${color}"></div></div>
        <span class="waste-dur">${dur > 0 ? fmtS(dur) : '---'}</span>
        <span class="waste-pct" style="${valColor}">${dur > 0 ? pct.toFixed(1) + '%' : '0%'}</span>
      </div>`;
    }
    if (totalWaste > 0) {
      html += `<div class="waste-row" style="margin-top:4px;border-top:1px solid var(--border);padding-top:4px">
        <span class="waste-name" style="color:var(--text);font-weight:600">Subtotal</span>
        <div class="waste-bar-bg"><div class="waste-bar-fill" style="width:100%;background:${totalWaste > 0 ? 'var(--orange)' : '#21262d'};opacity:0.5"></div></div>
        <span class="waste-dur" style="color:var(--text)">${fmtS(totalWaste)}</span>
        <span class="waste-pct" style="color:var(--orange)">${(totalWaste / totalWallS * 100).toFixed(1)}%</span>
      </div>`;
    }
    return html;
  }

  let html = '<h2>Efficiency Breakdown &mdash; Waste Categories</h2><div class="waste-bars">';
  html += renderSection('Driver Timeline (affects step efficiency)', driverCats, false);
  html += renderSection('Collector / Worker Timeline (parallel system)', collectorCats, true);
  html += '</div>';
  panel.innerHTML = html;
}

// --- Controls ---
document.getElementById('zoom-slider').addEventListener('input', (e) => {
  zoomLevel = parseInt(e.target.value);
  renderTimeline();
});

document.getElementById('btn-reset').addEventListener('click', () => {
  zoomLevel = 1;
  document.getElementById('zoom-slider').value = 1;
  renderTimeline();
  document.getElementById('timeline-scroll').scrollLeft = 0;
});

document.getElementById('btn-training').addEventListener('click', () => {
  const initSpans = allSpans.filter(s => s.label === 'init/total');
  if (initSpans.length > 0) {
    const initEnd = Math.max(...initSpans.map(s => s.ts_end));
    const trainingStart = (initEnd - globalMinTs) / globalRange;
    zoomLevel = Math.max(3, Math.ceil(globalRange / (globalMaxTs - initEnd)));
    document.getElementById('zoom-slider').value = Math.min(zoomLevel, 50);
    renderTimeline();
    const scroll = document.getElementById('timeline-scroll');
    const baseWidth = Math.max(1200, window.innerWidth - 250);
    scroll.scrollLeft = trainingStart * baseWidth * zoomLevel - 100;
  }
});

document.getElementById('worker-filter').addEventListener('change', () => renderTimeline());
document.getElementById('step-filter').addEventListener('change', () => {
  zoomLevel = 1;
  document.getElementById('zoom-slider').value = 1;
  document.getElementById('timeline-scroll').scrollLeft = 0;
  renderTimeline();
});

function populateWorkerFilter() {
  const sel = document.getElementById('worker-filter');
  const workers = [...new Set(allSpans.map(s => s.worker))].sort((a, b) => workerSortKey(a) - workerSortKey(b));
  for (const w of workers) {
    const opt = document.createElement('option');
    opt.value = w;
    opt.textContent = WORKER_NAMES[w] || w;
    sel.appendChild(opt);
  }
}

function populateStepFilter() {
  const sel = document.getElementById('step-filter');
  if (steps.length === 0) { sel.disabled = true; return; }
  const initOpt = document.createElement('option');
  initOpt.value = 'init';
  initOpt.textContent = 'Init Only';
  sel.appendChild(initOpt);
  for (const step of steps) {
    const opt = document.createElement('option');
    opt.value = String(step.num);
    opt.textContent = `Step ${step.num} (${fmtS(step.duration_us / 1e6)})`;
    sel.appendChild(opt);
  }
}

// --- Download / Perfetto ---
document.getElementById('btn-download-json').addEventListener('click', () => {
  const raw = atob(TRACE_B64);
  const blob = new Blob([raw], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'nemorl_trace.json';
  a.click();
});

document.getElementById('btn-perfetto').addEventListener('click', () => {
  const raw = atob(TRACE_B64);
  const blob = new Blob([raw], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'nemorl_trace.json';
  a.click();
  setTimeout(() => window.open('https://ui.perfetto.dev', '_blank'), 200);
});

// --- Init ---
renderStats();
renderLegend();
populateWorkerFilter();
populateStepFilter();
renderTimeline();
renderWasteBreakdown();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse NemoRL timer logs and generate Perfetto-compatible trace files."
    )
    parser.add_argument(
        "input",
        help="Path to a single .log file or a directory containing .log files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory to write output files (default: current directory).",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only generate the Chrome Trace JSON, skip HTML.",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    if inp.is_dir():
        print(f"Scanning directory: {inp}")
        events = parse_log_directory(inp)
    elif inp.is_file():
        print(f"Parsing file: {inp}")
        events = parse_log_file(inp)
    else:
        print(f"Error: {inp} not found", file=sys.stderr)
        sys.exit(1)

    if not events:
        print("No timer events found.", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(events)} timer events")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    trace = build_chrome_trace(events)
    summary = compute_summary(events)

    json_path = outdir / "nemorl_trace.json"
    with open(json_path, "w") as f:
        json.dump(trace, f)
    print(
        f"Wrote Chrome Trace JSON: {json_path}  ({len(trace['traceEvents'])} trace events)"
    )

    if not args.json_only:
        html_path = outdir / "nemorl_trace.html"
        html = _build_html(events, trace, summary)
        with open(html_path, "w") as f:
            f.write(html)
        print(f"Wrote HTML visualization: {html_path}")

    print("\nTo view in Perfetto:")
    print("  1. Open https://ui.perfetto.dev")
    print(f"  2. Drag & drop {json_path}")
    print(
        f"\nOr open {outdir / 'nemorl_trace.html'} in a browser for the built-in viewer."
    )

    # Print summary
    print("\n--- Efficiency Summary (driver) ---")
    for label in sorted(summary.keys()):
        print(f"  {label:40s} {summary[label]:10.3f}s")


if __name__ == "__main__":
    main()
