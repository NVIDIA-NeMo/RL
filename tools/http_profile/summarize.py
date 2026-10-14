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
"""Aggregate uninstrumented benchmark runs; keep diagnostic timings separate."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize(directory: Path) -> list[dict[str, Any]]:
    options = json.loads((directory / "environment.json").read_text())["options"]
    if options.get("profile") or options.get("trace"):
        raise ValueError("instrumented runs are diagnostic, not speedup measurements")
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in json.loads((directory / "results.json").read_text()):
        groups[(row["mode"], row["prompt_tokens"], row["concurrency"])].append(row)
    summaries = []
    for (mode, prompt, concurrency), rows in sorted(groups.items()):
        queues = [row["servers"][0]["queue"] for row in rows]
        p50s = [row["latency_ms"]["p50"] for row in rows]
        summaries.append(
            {
                "mode": mode,
                "initial_prompt_tokens": prompt,
                "concurrency": concurrency,
                "rounds": len(rows),
                "requests": sum(row["requests"] for row in rows),
                "p50_ms": statistics.median(p50s),
                "p50_min_ms": min(p50s),
                "p50_max_ms": max(p50s),
                "p95_ms": statistics.median(row["latency_ms"]["p95"] for row in rows),
                "first_body_p50_ms": statistics.median(
                    row["first_body_byte_ms"]["p50"] for row in rows
                ),
                "requests_per_s": statistics.median(
                    row["requests_per_s"] for row in rows
                ),
                "server_cpu_ms_per_request": statistics.median(
                    row["server_cpu_ms_per_request"] for row in rows
                ),
                "drain_ms": statistics.median(
                    queue.get("drain_ms", 0) for queue in queues
                ),
                "queue_lag_max_ms": max(
                    queue.get("queue_lag_max_ms", 0) for queue in queues
                ),
                "peak_pending_bytes": max(
                    queue.get("peak_pending_bytes", 0) for queue in queues
                ),
                "peak_pending_records": max(
                    queue.get("peak_pending", 0) for queue in queues
                ),
            }
        )
    with (directory / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (directory / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    for row in summarize(args.directory):
        print(
            f"{row['mode']:15s} P={row['initial_prompt_tokens']:6d} "
            f"C={row['concurrency']:2d} p50={row['p50_ms']:8.3f}ms "
            f"p95={row['p95_ms']:8.3f}ms "
            f"RPS={row['requests_per_s']:8.1f} "
            f"CPU={row['server_cpu_ms_per_request']:7.3f}ms "
            f"drain={row['drain_ms']:7.2f}ms"
        )


if __name__ == "__main__":
    main()
