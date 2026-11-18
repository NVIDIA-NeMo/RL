#!/usr/bin/env -S uv run --script -q
# /// script
# dependencies = [
#   "rich"
# ]
# ///
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
"""Summarize how the metric-check values of one functional test vary across repeated runs.

The checks are parsed out of the functional test script (the quoted expressions passed to
``tests/check_metrics.py``). Each run's ``metrics.json`` is evaluated with the same
``evaluate_check`` the test uses, and the resulting values are summarized per check.

Example:
    uv run tests/aggregate_metric_stats.py --script tests/functional/grpo.sh \
        --metrics-dir metrics --output-md stats.md --output-json stats.json
"""

import argparse
import json
import math
import re
import shlex
import statistics
import sys
from pathlib import Path

from check_metrics import evaluate_check


def parse_checks(script_path: Path) -> list[str]:
    """Return the check expressions passed to check_metrics.py in a test script.

    Expressions that still contain an unresolved shell variable are skipped, since their
    value is only known when the script runs.
    """
    text = script_path.read_text().replace("\\\n", " ")
    checks: list[str] = []
    for line in text.splitlines():
        if "check_metrics.py" not in line or line.lstrip().startswith("#"):
            continue
        tokens = shlex.split(line.split("check_metrics.py", 1)[1])
        # First positional is the metrics json path; the rest are check expressions.
        for token in tokens[1:]:
            if token.startswith("--"):
                continue
            if re.search(r"\$\{?\w", token):
                print(f"skipping check with shell variable: {token}", file=sys.stderr)
                continue
            checks.append(token)
    return checks


def summarize(values: list[float]) -> dict:
    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n >= 2 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "median": statistics.median(values),
        "std": std,
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "cv_pct": abs(std / mean) * 100 if n >= 2 and mean != 0 else float("nan"),
    }


def fmt(x: float) -> str:
    return "n/a" if isinstance(x, float) and math.isnan(x) else f"{x:.6g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help="Directory searched recursively for metrics.json files, one per run",
    )
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    checks = parse_checks(args.script)
    if not checks:
        print(f"No check_metrics expressions found in {args.script}", file=sys.stderr)
        return 1

    run_files = sorted(args.metrics_dir.rglob("metrics.json"))
    if not run_files:
        print(f"No metrics.json found under {args.metrics_dir}", file=sys.stderr)
        return 1

    results = []
    for check in checks:
        values, per_run, passes = [], {}, 0
        for run_file in run_files:
            run = run_file.parent.name
            passed, _, value = evaluate_check(json.loads(run_file.read_text()), check)
            passes += passed
            is_num = isinstance(value, (int, float)) and not isinstance(value, bool)
            per_run[run] = {"passed": passed, "value": value if is_num else None}
            if is_num and math.isfinite(value):
                values.append(float(value))
        results.append(
            {
                "check": check,
                "runs": len(run_files),
                "passed": passes,
                "stats": summarize(values) if values else None,
                "per_run": per_run,
            }
        )

    lines = [
        f"### Metric variance: `{args.script.stem}` ({len(run_files)} runs)",
        "",
        "| check | pass | n | mean | median | std | min | max | range | CV % |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        s = r["stats"]
        stat_cells = (
            [
                fmt(s[k])
                for k in ("mean", "median", "std", "min", "max", "range", "cv_pct")
            ]
            if s
            else ["n/a"] * 7
        )
        n = s["n"] if s else 0
        check = r["check"].replace("|", "\\|")
        lines.append(
            f"| `{check}` | {r['passed']}/{r['runs']} | {n} | "
            + " | ".join(stat_cells)
            + " |"
        )
    markdown = "\n".join(lines) + "\n"

    print(markdown)
    if args.output_md:
        args.output_md.write_text(markdown)
    if args.output_json:
        args.output_json.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
