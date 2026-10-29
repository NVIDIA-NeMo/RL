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
"""Derive statistics from per-rollout ``timeline.jsonl`` files.

Applies only to rollouts written by the NeMo Gym SWE agent and OpenHands (see
the "Applicability" section of SKILL.md). Standard library only.

Reports:
  * per-trajectory shape: turns, model/tool/other time share, tokens, cache hit
  * the ``model_call`` latency breakdown, from OpenHands down to the engine
  * system-level in-flight counts over time (client, queued, running, per replica)
  * the estimated cost of the tracing itself
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from timeline_common import (
    breakdown,
    inflight_series,
    load_jsonl,
    milestone,
)


def percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def analyze_rollout(events):
    spans = defaultdict(list)
    for e in events:
        spans[e["name"]].append(e)
    calls = [e for e in spans["model_call"] if e["args"].get("status") == "ok"]
    controller = spans["run_controller"]
    tools = [e for e in events if e.get("cat") == "tool"]
    run_controller = controller[0]["dur"] if controller else None
    model_time = sum(e["dur"] for e in spans["model_call"])
    tool_time = sum(e["dur"] for e in tools)
    prompt = sum(e["args"].get("prompt_tokens", 0) for e in calls)
    # Under PD disaggregation the prefill leg decides reuse, so a call that
    # carries ctx_computed_tokens is scored from it, not from the decode leg's
    # cached_prompt_tokens.
    cached = sum(
        e["args"]["prompt_tokens"] - e["args"]["ctx_computed_tokens"]
        if "ctx_computed_tokens" in e["args"] and "prompt_tokens" in e["args"]
        else e["args"].get("cached_prompt_tokens", 0)
        for e in calls
    )
    return {
        "turns": len(calls),
        "run_controller_us": run_controller,
        "model_us": model_time,
        "tool_us": tool_time,
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "decode_tokens": sum(e["args"].get("decode_tokens", 0) for e in calls),
        "max_prompt": max(
            (e["args"].get("prompt_tokens", 0) for e in calls), default=0
        ),
        "openhands_run_us": spans["openhands_run"][0]["dur"]
        if spans["openhands_run"]
        else None,
        "ray_queue_us": spans["ray_queue"][0]["dur"] if spans["ray_queue"] else None,
        "final_eval_us": spans["final_eval"][0]["dur"] if spans["final_eval"] else None,
        "tool_by_name": {
            n: sum(e["dur"] for e in v)
            for n, v in (
                (n, [e for e in tools if e["name"] == n])
                for n in {e["name"] for e in tools}
            )
        },
        "calls": calls,
    }


def summarize(values):
    if not values:
        return "n/a"
    return (
        f"mean {sum(values) / len(values):.1f}  p50 {percentile(values, 50):.1f}  "
        f"p95 {percentile(values, 95):.1f}  max {max(values):.1f}"
    )


def tracing_cost(root, rollout_files):
    sizes = [f.stat().st_size for f in rollout_files]
    return {"files": len(sizes), "mean_bytes": sum(sizes) / max(len(sizes), 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--bin-seconds", type=float, default=10.0)
    parser.add_argument("--output-json", type=Path)
    parsed = parser.parse_args()

    files = sorted(parsed.rollout_root.rglob("timeline.jsonl"))
    files = [f for f in files if f.stat().st_size > 0]
    if not files:
        raise SystemExit(f"no non-empty timeline.jsonl under {parsed.rollout_root}")

    rollouts, n_events = [], 0
    seg_total = defaultdict(float)
    seg_vals = defaultdict(list)
    seg_class = {}
    negative = defaultdict(int)
    n_broken_down = n_calls = n_disagg = 0
    client, queued, running, ctx_running = [], [], [], []
    per_replica = defaultdict(list)
    for path in files:
        events = load_jsonl(path)
        n_events += len(events)
        r = analyze_rollout(events)
        rollouts.append(r)
        for call in r["calls"]:
            n_calls += 1
            a = call["args"]
            client.append((call["ts"], call["ts"] + call["dur"]))
            q, s, l = (
                milestone(a, "queued"),
                milestone(a, "first_scheduled"),
                milestone(a, "last_token"),
            )
            if q and s:
                queued.append((q, s))
            if s and l:
                running.append((s, l))
                per_replica[a.get("generation_replica_idx", 0)].append((s, l))
            cs, cd = milestone(a, "ctx_first_scheduled"), milestone(a, "ctx_done")
            if cs and cd:
                ctx_running.append((cs, cd))
            cut = breakdown(call)
            if cut is None:
                continue
            n_broken_down += 1
            n_disagg += cut[0]
            for name, start, end, klass in cut[1]:
                dur = end - start
                seg_total[name] += dur
                seg_vals[name].append(dur)
                seg_class[name] = klass
                if dur < 0:
                    negative[name] += 1

    model_total = sum(r["model_us"] for r in rollouts)
    cut_total = sum(seg_total.values())
    ok = [r for r in rollouts if r["run_controller_us"]]
    print(
        f"rollouts: {len(rollouts)}  events: {n_events}  model_calls(ok): {n_calls}  "
        f"with full milestones: {n_broken_down} ({n_disagg} disaggregated)"
    )

    print("\n== per-trajectory shape (over run_controller) ==")
    turns = [r["turns"] for r in ok]
    print(f"turns          {summarize(turns)}")
    for label, key in (("model_call", "model_us"), ("tool", "tool_us")):
        share = [100 * r[key] / r["run_controller_us"] for r in ok]
        print(f"{label:<14} time share %: {summarize(share)}")
    other = [
        100 * (1 - (r["model_us"] + r["tool_us"]) / r["run_controller_us"]) for r in ok
    ]
    print(
        f"{'agent other':<14} time share %: {summarize(other)}   (loop overhead between calls and tools)"
    )
    print(f"run_controller s: {summarize([r['run_controller_us'] / 1e6 for r in ok])}")
    print(
        f"prompt tokens/call {summarize([r['prompt_tokens'] / max(r['turns'], 1) for r in ok])}"
    )
    print(
        f"decode tokens/call {summarize([r['decode_tokens'] / max(r['turns'], 1) for r in ok])}"
    )
    hit = [
        100 * r["cached_tokens"] / r["prompt_tokens"] for r in ok if r["prompt_tokens"]
    ]
    print(f"prefix hit rate % per trajectory (token-weighted): {summarize(hit)}")
    tool_total = defaultdict(float)
    for r in ok:
        for name, dur in r["tool_by_name"].items():
            tool_total[name] += dur
    top = sorted(tool_total.items(), key=lambda kv: -kv[1])[:5]
    print(
        "tool time by action: "
        + ", ".join(f"{n} {100 * d / sum(tool_total.values()):.1f}%" for n, d in top)
    )

    print("\n== model_call breakdown: OpenHands -> engine -> OpenHands ==")
    print(
        f"{'segment':<22}{'class':<10}{'% of call':>10}{'mean ms':>10}{'p50 ms':>9}{'p95 ms':>9}{'<0':>6}"
    )
    for name, total in sorted(seg_total.items(), key=lambda kv: -kv[1]):
        vals = [v / 1000 for v in seg_vals[name]]
        print(
            f"{name:<22}{seg_class[name]:<10}{100 * total / cut_total:>9.2f}%"
            f"{sum(vals) / len(vals):>10.1f}{percentile(vals, 50):>9.1f}{percentile(vals, 95):>9.1f}"
            f"{negative[name]:>6}"
        )
    by_class = defaultdict(float)
    for name, total in seg_total.items():
        by_class[seg_class[name]] += total
    print(
        "by class: "
        + "  ".join(
            f"{k} {100 * v / cut_total:.2f}%" for k, v in sorted(by_class.items())
        )
    )
    print(
        "('<0' counts segments that came out negative: cross-host clock skew, not real)"
    )

    decode_tok = [
        (c["args"]["last_token_ts_us"] - c["args"]["first_scheduled_ts_us"])
        / (c["args"]["decode_tokens"] - 1)
        for r in rollouts
        for c in r["calls"]
        if c["args"].get("decode_tokens", 0) > 1
        and milestone(c["args"], "last_token")
        and milestone(c["args"], "first_scheduled")
    ]
    if decode_tok:
        print(
            f"TPOT ms (last_token-first_scheduled)/(decode_tokens-1): {summarize([v / 1000 for v in decode_tok])}"
        )

    print("\n== system in-flight (mean concurrent requests per bin) ==")
    bin_us = parsed.bin_seconds * 1e6
    series = {
        "client (OpenHands waiting)": inflight_series(client, bin_us),
        "engine queued": inflight_series(queued, bin_us),
        "engine running (gen batch)": inflight_series(running, bin_us),
        "ctx prefill running": inflight_series(ctx_running, bin_us),
    }
    for name, values in series.items():
        print(f"{name:<28}{summarize(values)}")
    for replica in sorted(per_replica):
        print(
            f"  gen replica {replica}: running {summarize(inflight_series(per_replica[replica], bin_us))}"
        )

    print("\n== tracing cost ==")
    cost = tracing_cost(parsed.rollout_root, files)
    print(
        f"events/rollout {n_events / len(files):.0f}   bytes/rollout {cost['mean_bytes'] / 1024:.0f} KiB   "
        f"total {sum(f.stat().st_size for f in files) / 2**20:.0f} MiB"
    )

    if parsed.output_json:
        parsed.output_json.write_text(
            json.dumps(
                {
                    "rollouts": len(rollouts),
                    "model_calls": n_calls,
                    "segment_share_pct": {
                        n: 100 * t / cut_total for n, t in seg_total.items()
                    },
                    "class_share_pct": {
                        k: 100 * v / cut_total for k, v in by_class.items()
                    },
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
