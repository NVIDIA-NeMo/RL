#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build prompt-only RL rows from successful OpenCode-aligned ClawEval traces.

Recorded assistant/tool turns are deliberately excluded: online GRPO must sample
fresh policy trajectories. Native task fixtures and hidden graders are resolved
by verifier metadata inside the Gym agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

GYM_ROOT = Path(__file__).resolve().parents[2] / "3rdparty" / "Gym-workspace" / "Gym"
sys.path.insert(0, str(GYM_ROOT))

from responses_api_agents.claweval_agent.core import dataset_rows, dataset_summary, inventory  # noqa: E402

TASK_SOURCE = "claweval_agent"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, required=True, help="OpenCode-aligned successful ClawEval trajectories")
    parser.add_argument(
        "--task-root",
        action="append",
        required=True,
        metavar="VARIANT=PATH",
        help="Repeat for original_100, brief_100, and brief_1000",
    )
    parser.add_argument("--source-prefix", action="append", help="Optional id prefix such as original_100 or brief_1000")
    parser.add_argument("--limit", type=int, default=None, help="Debug-only row limit after deterministic filtering")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    task_roots = {}
    for value in args.task_root:
        variant, separator, raw_path = value.partition("=")
        if not separator or not variant or not raw_path or variant in task_roots:
            raise SystemExit(f"Invalid or duplicate --task-root {value!r}; expected VARIANT=PATH")
        task_roots[variant] = Path(raw_path)
    tasks = inventory(task_roots)
    rows = dataset_rows(args.sft, tasks)
    if args.source_prefix:
        allowed = tuple(prefix + "__" for prefix in args.source_prefix)
        rows = [row for row in rows if row["verifier_metadata"]["source_id"].startswith(allowed)]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No ClawEval rows selected")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps({**row, "task_source": TASK_SOURCE}, ensure_ascii=False) + "\n")

    manifest = {
        **dataset_summary(rows),
        "schema": "nemo_gym_prompt_only_v1",
        "task_source": TASK_SOURCE,
        "sft_source": str(args.sft.resolve()),
        "sft_source_sha256": file_sha256(args.sft),
        "task_roots": {variant: str(root.resolve()) for variant, root in task_roots.items()},
        "source_prefix": args.source_prefix,
        "limit": args.limit,
        "assistant_turns_replayed": False,
    }
    manifest_path = args.output.with_name(args.output.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), **manifest}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
