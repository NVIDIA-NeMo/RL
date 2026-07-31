#!/usr/bin/env python3
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
"""Aggregate the instrumented legacy-vs-capture perf pair.

Usage: python aggregate_perf.py <legacy_run_dir> <capture_run_dir>

Each run dir: http_bytes/*.json (per-server per-route byte counters),
metrics.json (TB dump), train_rows_*.jsonl (row dump for token counts).
"""

import glob
import json
import os
import sys


def load_bytes(run_dir):
    per_server = {}
    for path in glob.glob(os.path.join(run_dir, "http_bytes", "*.json")):
        name = os.path.basename(path).rsplit("_", 1)[0]
        data = json.load(open(path))
        agg = per_server.setdefault(name, {})
        for route, c in data.items():
            e = agg.setdefault(route, {"requests": 0, "req_bytes": 0, "resp_bytes": 0})
            for k in e:
                e[k] += c[k]
    return per_server


def total_trained_tokens(run_dir):
    total = 0
    for path in glob.glob(os.path.join(run_dir, "train_rows_*.jsonl")):
        for line in open(path):
            if line.strip():
                total += json.loads(line)["input_lengths"]
    return total


def timing(run_dir):
    d = json.load(open(os.path.join(run_dir, "metrics.json")))
    out = {}
    for key in (
        "timing/train/total_step_time",
        "timing/train/exposed_generation",
        "timing/train/policy_training",
        "timing/train/weight_sync",
        "timing/train/valid_tokens_per_sec_per_gpu",
        "train/gen_kl_error",
        "train/reward",
        "gate/token_in_rate",
        "gate/token_in",
        "gate/fallback_no_marker",
        "gate/fallback_fingerprint_miss",
    ):
        s = d.get(key)
        if isinstance(s, dict) and s:
            vals = [v for _, v in sorted(s.items(), key=lambda kv: int(kv[0]))]
            out[key] = vals
    return out


def main():
    legacy_dir, capture_dir = sys.argv[1], sys.argv[2]
    report = {}
    for tag, run_dir in (("legacy", legacy_dir), ("capture", capture_dir)):
        servers = load_bytes(run_dir)
        toks = total_trained_tokens(run_dir)
        grand = {"requests": 0, "req_bytes": 0, "resp_bytes": 0}
        print(f"\n===== {tag} =====")
        for name, routes in sorted(servers.items()):
            s = {"requests": 0, "req_bytes": 0, "resp_bytes": 0}
            for route, c in routes.items():
                for k in s:
                    s[k] += c[k]
            for k in grand:
                grand[k] += s[k]
            print(f"{name:32s} req={s['requests']:6d}  in={s['req_bytes']/1e6:8.2f} MB  out={s['resp_bytes']/1e6:8.2f} MB")
            for route, c in sorted(routes.items(), key=lambda kv: -(kv[1]["req_bytes"] + kv[1]["resp_bytes"]))[:4]:
                print(f"    {route:40s} n={c['requests']:5d} in={c['req_bytes']/1e6:7.2f} MB out={c['resp_bytes']/1e6:7.2f} MB")
        total_bytes = grand["req_bytes"] + grand["resp_bytes"]
        print(f"{'TOTAL':32s} req={grand['requests']:6d}  bytes={total_bytes/1e6:.2f} MB  trained_tokens={toks}  bytes/token={total_bytes/max(toks,1):.1f}")
        report[tag] = {"total_bytes": total_bytes, "tokens": toks, "timing": timing(run_dir)}

    lt, ct = report["legacy"], report["capture"]
    print("\n===== comparison =====")
    bt_l = lt["total_bytes"] / max(lt["tokens"], 1)
    bt_c = ct["total_bytes"] / max(ct["tokens"], 1)
    print(f"HTTP bytes/trained token: legacy {bt_l:.1f} -> capture {bt_c:.1f}  ({(bt_c/bt_l-1)*100:+.1f}%)")
    for key in ("timing/train/total_step_time", "timing/train/exposed_generation", "timing/train/valid_tokens_per_sec_per_gpu"):
        lv, cv = lt["timing"].get(key), ct["timing"].get(key)
        if lv and cv:
            import statistics
            ml, mc = statistics.median(lv), statistics.median(cv)
            print(f"{key}: legacy median {ml:.3f} -> capture median {mc:.3f} ({(mc/ml-1)*100:+.1f}%)")
    tir = ct["timing"].get("gate/token_in_rate")
    if tir:
        print(f"capture token_in_rate (cumulative, final): {tir[-1]:.3f}")


if __name__ == "__main__":
    main()
