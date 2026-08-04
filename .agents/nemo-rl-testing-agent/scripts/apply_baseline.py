#!/usr/bin/env python3
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

# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""Reclassify a PR's failures that also fail in the megatron-core `main` baseline.

A test that fails on `main` too is not the labeled PR's fault, and saying so is the
difference between a useful report and one that wastes the author's afternoon. This
downgrades those rows to `fail (pre-existing)` and records which baseline decided
it, so nobody has to re-litigate the call.

Usage:
    uv run --script apply_baseline.py --results l1.json --baseline l1-baseline.json \
        --baseline-meta l1-baseline.env
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

FAILED_STATUSES = {"fail", "incomplete", "pass (suspect)"}


def load_meta(path: Path | None) -> dict[str, str]:
    meta: dict[str, str] = {}
    if path is None or not path.is_file():
        return meta
    for line in path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, required=True, help="PR results JSON, edited in place."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Baseline results JSON from megatron-core main.",
    )
    parser.add_argument(
        "--baseline-meta", type=Path, help="Baseline .env recording the main sha."
    )
    parser.add_argument(
        "--out", type=Path, help="Write here instead of editing --results in place."
    )
    args = parser.parse_args()

    for path in (args.results, args.baseline):
        if not path.is_file():
            print(f"apply_baseline: missing {path}", file=sys.stderr)
            return 1

    payload: dict[str, Any] = json.loads(args.results.read_text())
    baseline: dict[str, Any] = json.loads(args.baseline.read_text())

    baseline_status = {
        test.get("name"): test.get("status", "unknown")
        for test in baseline.get("tests", [])
    }
    meta = load_meta(args.baseline_meta)
    main_sha = meta.get("MCORE_SHA", "")
    short_sha = main_sha[:8] if main_sha else "main"

    reclassified = 0
    unknown = 0
    for test in payload.get("tests", []):
        if test.get("status") not in FAILED_STATUSES:
            continue
        name = test.get("name")
        if name not in baseline_status:
            unknown += 1
            continue
        if baseline_status[name] in FAILED_STATUSES:
            # A suspect pass keeps its status: calling it `fail` would contradict the
            # rc=0 the author can see in the log, and the point of the label is that
            # the two disagree.
            if test.get("status") != "pass (suspect)":
                test["status"] = "fail (pre-existing)"
            note = (
                f"Also broken on megatron-core `main` ({short_sha}), so this is not caused "
                "by this PR. Being fixed separately."
            )
            existing = (test.get("comment") or "").strip()
            test["comment"] = f"{note} {existing}".strip() if existing else note
            reclassified += 1

    payload.setdefault("baseline", {})
    payload["baseline"].update(
        {
            "mcore_sha": main_sha,
            "run_name": meta.get("RUN_NAME", ""),
            "created_utc": meta.get("CREATED_UTC", ""),
        }
    )

    destination = args.out or args.results
    destination.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"apply_baseline: {reclassified} failure(s) reclassified as pre-existing -> {destination}"
    )
    if unknown:
        print(
            f"apply_baseline: {unknown} failing test(s) absent from the baseline; left as-is"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
