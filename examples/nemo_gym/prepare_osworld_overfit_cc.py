#!/usr/bin/env python3
"""Build an OSWorld overfit dataset with identical train/eval task IDs."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from prepare_osworld_stable_cc_split import (
    add_cc_contract,
    read_jsonl,
    source_sha256,
    task_id,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-repeats", type=int, default=5)
    parser.add_argument("--eval-repeats", type=int, default=1)
    parser.add_argument("--expected-tasks", type=int, default=361)
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Keep only the selected OSWorld task ID; may be repeated.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def combined_source_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode())
        digest.update(b"\0")
        digest.update(source_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.train_repeats < 1 or args.eval_repeats < 1:
        raise ValueError("Repeat counts must be positive")

    rows: list[dict[str, Any]] = []
    source_records = []
    for path in args.input:
        source_rows = read_jsonl(path)
        rows.extend(source_rows)
        source_records.append(
            {
                "path": str(path),
                "sha256": source_sha256(path),
                "rows": len(source_rows),
            }
        )

    if args.task_id:
        selected_ids = set(args.task_id)
        rows = [row for row in rows if task_id(row) in selected_ids]
        missing_ids = selected_ids.difference(task_id(row) for row in rows)
        if missing_ids:
            raise ValueError(f"Requested task IDs were not found: {sorted(missing_ids)}")

    ids = [task_id(row) for row in rows]
    if len(ids) != args.expected_tasks:
        raise ValueError(f"Expected {args.expected_tasks} tasks, found {len(ids)}")
    if len(set(ids)) != len(ids):
        duplicates = sorted({value for value in ids if ids.count(value) > 1})
        raise ValueError(f"Duplicate task IDs: {duplicates}")

    train_rows = add_cc_contract(
        rows,
        args.train_repeats,
        args.max_output_tokens,
        args.temperature,
        args.top_p,
    )
    eval_rows = add_cc_contract(
        rows,
        args.eval_repeats,
        args.max_output_tokens,
        args.temperature,
        args.top_p,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"train-{args.train_repeats}x.jsonl"
    eval_path = args.output_dir / f"validation-{args.eval_repeats}x.jsonl"
    train_sha256 = write_jsonl(train_path, train_rows)
    eval_sha256 = write_jsonl(eval_path, eval_rows)

    manifest = {
        "contract_version": 2,
        "mode": "overfit",
        "sources": source_records,
        "combined_source_sha256": combined_source_sha256(args.input),
        "logical_tasks": {
            "train": len(ids),
            "eval": len(ids),
            "overlap": len(ids),
        },
        "outputs": {
            "train": {
                "path": str(train_path),
                "sha256": train_sha256,
                "rows": len(train_rows),
                "repeats": args.train_repeats,
            },
            "eval": {
                "path": str(eval_path),
                "sha256": eval_sha256,
                "rows": len(eval_rows),
                "repeats": args.eval_repeats,
            },
        },
        "generation": {
            "agent_name": "nemotron_osworld",
            "max_output_tokens": args.max_output_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "train_task_ids": sorted(ids),
        "eval_task_ids": sorted(ids),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"Wrote {len(train_rows)} train rows and {len(eval_rows)} eval rows "
        f"for {len(ids)} shared OSWorld tasks to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
