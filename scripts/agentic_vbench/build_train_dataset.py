#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training rows for the gym-native Agentic-VBench agent.

Rows carry the verbatim benchmark prompt and verifier metadata exactly as
``benchmarks/agentic_vbench/prepare.py`` exports them, plus the Gym routing stamp
NeMo-RL needs. Repeats provide enough rows for async prefetch; they are repeated
tasks, not distinct examples, and training on benchmark tasks is for debugging.
"""

import argparse
import json
import sys
from pathlib import Path

GYM_ROOT = Path(__file__).resolve().parents[2] / "3rdparty/Gym-workspace/Gym"
sys.path.insert(0, str(GYM_ROOT))

from responses_api_agents.agentic_vbench_agent.core import REVISION, dataset_rows, ensure_checkout, inventory  # noqa: E402

AGENT_REF = {"type": "responses_api_agents", "name": "agentic_vbench_agent"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=None, help=f"Pinned checkout at {REVISION[:12]}")
    parser.add_argument("--tasks", default="all", help="'all', or space-separated families / task ids")
    parser.add_argument("--limit", type=int, default=None, help="Keep the first N selected tasks")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat every row this many times")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = ensure_checkout(args.benchmark_root)
    tasks = inventory(root)
    rows = dataset_rows(tasks, args.tasks)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No tasks selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for _ in range(args.repeats):
            for row in rows:
                handle.write(json.dumps({**row, "agent_ref": AGENT_REF}) + "\n")
    manifest = {
        "benchmark_root": str(root),
        "benchmark_revision": REVISION,
        "selection": args.tasks,
        "tasks": [row["verifier_metadata"]["task_id"] for row in rows],
        "repeats": args.repeats,
        "rows": len(rows) * args.repeats,
    }
    args.output.with_name(args.output.stem + "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest['rows']} rows ({len(rows)} tasks x {args.repeats}) to {args.output}")


if __name__ == "__main__":
    main()
