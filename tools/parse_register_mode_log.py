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
"""Break down register-mode per-call timings from a Ray driver log.

The adapter prints one line per put/get/clear. Ray collapses identical
consecutive lines into ``[repeated Nx across cluster]``, so a raw line count
undercounts calls; this reports the collapse so the counts are not read as
exact.
"""

from __future__ import annotations

import re
import statistics
import sys
from collections import defaultdict

FIELD = re.compile(r"(\w+)=(\d+)")
REPEATED = re.compile(r"repeated (\d+)x")


def parse(path: str) -> tuple[dict[str, list[dict]], dict[str, int]]:
    calls: dict[str, list[dict]] = defaultdict(list)
    collapsed: dict[str, int] = defaultdict(int)
    for line in open(path, errors="replace"):
        m = re.search(r"register_mode (put|get|clear): (.*)", line)
        if not m:
            continue
        op, rest = m.group(1), m.group(2)
        row = {k: int(v) for k, v in FIELD.findall(rest)}
        row["device"] = "cuda" if "device=cuda" in rest else (
            "cpu" if "device=cpu" in rest else ""
        )
        rep = REPEATED.search(rest)
        if rep:
            collapsed[op] += int(rep.group(1)) - 1
        calls[op].append(row)
    return calls, collapsed


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def dist(name: str, vals_us: list[float]) -> str:
    ms = [v / 1e3 for v in vals_us]
    return (
        f"    {name:<10} n={len(ms):<4} sum={sum(ms):8.1f}ms  mean={statistics.mean(ms):7.2f}  "
        f"min={min(ms):7.2f}  p50={pct(ms, 0.5):7.2f}  p90={pct(ms, 0.9):7.2f}  max={max(ms):7.2f}"
    )


def main() -> None:
    path = sys.argv[1]
    step_time_s = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    calls, collapsed = parse(path)

    grand_ms = 0.0
    for op in ("put", "get", "clear"):
        rows = calls.get(op, [])
        if not rows:
            continue
        totals = [r["total"] for r in rows]
        grand_ms += sum(totals) / 1e3
        nbytes = sum(r.get("bytes", 0) for r in rows)
        print(f"\n  {op.upper()}  ({len(rows)} lines logged"
              f"{f', +{collapsed[op]} collapsed by Ray dedup' if collapsed.get(op) else ''})")
        print(dist("total", totals))
        for part in ("alloc", "register", "move", "local", "send"):
            if part in rows[0]:
                print(dist(part, [r[part] for r in rows]))
        if nbytes and op == "put":
            # Not a throughput: register mode's put copies nothing, so these
            # are bytes published (made addressable to readers), and dividing
            # by the pin time would invent a transfer rate for a transfer
            # that never happened.
            print(f"    bytes={nbytes / 2**20:8.1f} MB published, none copied")
        elif nbytes:
            secs = sum(totals) / 1e6
            print(f"    bytes={nbytes / 2**20:8.1f} MB   "
                  f"effective={nbytes / 2**20 / secs:6.2f} MB/s over the op's own wall time")
        if op == "get":
            for dev in ("cuda", "cpu"):
                sub = [r for r in rows if r["device"] == dev]
                if sub:
                    mb = sum(r["bytes"] for r in sub) / 2**20
                    loc = sum(r["local"] for r in sub)
                    rem = sum(r["remote"] for r in sub)
                    print(f"    device={dev:<5} n={len(sub):<3} "
                          f"{sum(r['total'] for r in sub) / 1e3:8.1f}ms  {mb:7.1f} MB  "
                          f"keys local={loc} remote={rem}  "
                          f"peers={sorted({r['peers'] for r in sub})}")

    print(f"\n  data-plane wall (sum of logged calls): {grand_ms / 1e3:.2f}s")
    if step_time_s:
        print(f"  training wall (5 steps):               {step_time_s:.2f}s")
        print(f"  share:                                 {100 * grand_ms / 1e3 / step_time_s:.2f}%")
    print("\n  Caveat: these are per-process prints multiplexed into one driver log,")
    print("  so the sum overlaps in wall-clock across processes and is an upper")
    print("  bound on elapsed data-plane time, not a serial total.")


if __name__ == "__main__":
    main()
