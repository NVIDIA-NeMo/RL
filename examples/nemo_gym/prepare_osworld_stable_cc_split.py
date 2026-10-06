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

"""Build stable OSWorld train and held-out datasets for v2 context compaction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from examples.nemo_gym.prepare_osworld_context_compaction_data import (
    prepare_osworld_rows,
    read_jsonl,
    task_id,
    write_jsonl,
)


def is_stable_train_task(row: dict[str, Any]) -> bool:
    """Return whether a task belongs to the low-volatility training subset."""
    metadata = row["verifier_metadata"]
    return (
        metadata.get("proxy") is False
        and metadata.get("possibility_of_env_change") == "low"
    )


def is_stable_eval_task(row: dict[str, Any]) -> bool:
    """Return whether a held-out task avoids known high-volatility cases."""
    return row["verifier_metadata"].get("possibility_of_env_change") != "high"


def sha256_path(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--eval-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-repeats", type=int, default=2)
    parser.add_argument("--eval-repeats", type=int, default=1)
    parser.add_argument("--expected-train-tasks", type=int, default=251)
    parser.add_argument("--expected-eval-tasks", type=int, default=71)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    """Filter, transform, and write the stable split plus provenance manifest."""
    args = parse_args()
    if args.train_repeats < 1 or args.eval_repeats < 1:
        raise ValueError("Repeat counts must be positive")

    train_source = read_jsonl(args.train_input)
    eval_source = read_jsonl(args.eval_input)
    train_tasks = [row for row in train_source if is_stable_train_task(row)]
    eval_tasks = [row for row in eval_source if is_stable_eval_task(row)]

    if len(train_tasks) != args.expected_train_tasks:
        raise ValueError(
            f"Expected {args.expected_train_tasks} stable train tasks, "
            f"found {len(train_tasks)}"
        )
    if len(eval_tasks) != args.expected_eval_tasks:
        raise ValueError(
            f"Expected {args.expected_eval_tasks} stable eval tasks, "
            f"found {len(eval_tasks)}"
        )

    train_ids = {task_id(row) for row in train_tasks}
    eval_ids = {task_id(row) for row in eval_tasks}
    overlap = sorted(train_ids & eval_ids)
    if overlap:
        raise ValueError(f"Train/eval task ID overlap: {overlap}")

    generation_kwargs = {
        "agent_name": "nemotron_osworld",
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    train_rows = prepare_osworld_rows(
        train_tasks,
        num_repeats=args.train_repeats,
        **generation_kwargs,
    )
    eval_rows = prepare_osworld_rows(
        eval_tasks,
        num_repeats=args.eval_repeats,
        **generation_kwargs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"train-{args.train_repeats}x.jsonl"
    eval_path = args.output_dir / f"heldout-{args.eval_repeats}x.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    manifest = {
        "format": "nemo_gym_osworld_single_controller",
        "format_version": 1,
        "filters": {
            "train": {
                "proxy": False,
                "possibility_of_env_change": "low",
            },
            "eval": {
                "possibility_of_env_change": "not high",
            },
        },
        "sources": {
            "train": {
                "path": str(args.train_input),
                "sha256": sha256_path(args.train_input),
                "rows": len(train_source),
            },
            "eval": {
                "path": str(args.eval_input),
                "sha256": sha256_path(args.eval_input),
                "rows": len(eval_source),
            },
        },
        "logical_tasks": {
            "train": len(train_tasks),
            "eval": len(eval_tasks),
            "overlap": len(overlap),
        },
        "outputs": {
            "train": {
                "path": str(train_path),
                "sha256": sha256_path(train_path),
                "rows": len(train_rows),
                "repeats": args.train_repeats,
            },
            "eval": {
                "path": str(eval_path),
                "sha256": sha256_path(eval_path),
                "rows": len(eval_rows),
                "repeats": args.eval_repeats,
            },
        },
        "generation": generation_kwargs,
        "train_task_ids": sorted(train_ids),
        "eval_task_ids": sorted(eval_ids),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Wrote {len(train_rows)} train rows ({len(train_tasks)} tasks) and "
        f"{len(eval_rows)} held-out rows ({len(eval_tasks)} tasks) to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
