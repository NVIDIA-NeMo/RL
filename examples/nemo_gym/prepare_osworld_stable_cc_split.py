#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Build the stable OSWorld train/held-out split with the CC v2 contract."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("verifier_metadata")
            task_id = metadata.get("id") if isinstance(metadata, dict) else None
            if "responses_create_params" not in row or not task_id:
                raise ValueError(f"{path}:{line_number} is not an OSWorld task row")
            rows.append(row)
    return rows


def task_id(row: dict[str, Any]) -> str:
    return row["verifier_metadata"]["id"]


def is_stable_train_task(row: dict[str, Any]) -> bool:
    metadata = row["verifier_metadata"]
    return (
        metadata.get("proxy") is False
        and metadata.get("possibility_of_env_change") == "low"
    )


def is_stable_eval_task(row: dict[str, Any]) -> bool:
    return row["verifier_metadata"].get("possibility_of_env_change") != "high"


def add_cc_contract(
    rows: list[dict[str, Any]],
    repeats: int,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict[str, Any]]:
    output = []
    for repeat_index in range(repeats):
        for source_row in rows:
            row = copy.deepcopy(source_row)
            current_task_id = task_id(row)
            row["responses_create_params"].update(
                {
                    "max_output_tokens": max_output_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
            row["agent_ref"] = {
                "type": "responses_api_agents",
                "name": "nemotron_osworld",
            }
            row.update(
                {
                    "context_compaction_contract_version": 2,
                    "context_compaction_group_id": (
                        f"osworld:{current_task_id}:repeat:{repeat_index}"
                    ),
                    "context_compaction_task_id": current_task_id,
                    "context_compaction_rollout_index": 0,
                    "context_compaction_attempt_index": 0,
                }
            )
            output.append(row)
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as destination:
        for row in rows:
            encoded = (
                json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n"
            ).encode()
            destination.write(encoded)
            digest.update(encoded)
    return digest.hexdigest()


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
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

    train_rows = add_cc_contract(
        train_tasks,
        args.train_repeats,
        args.max_output_tokens,
        args.temperature,
        args.top_p,
    )
    eval_rows = add_cc_contract(
        eval_tasks,
        args.eval_repeats,
        args.max_output_tokens,
        args.temperature,
        args.top_p,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"train-{args.train_repeats}x.jsonl"
    eval_path = args.output_dir / f"heldout-{args.eval_repeats}x.jsonl"
    train_sha256 = write_jsonl(train_path, train_rows)
    eval_sha256 = write_jsonl(eval_path, eval_rows)

    manifest = {
        "contract_version": 2,
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
                "sha256": source_sha256(args.train_input),
                "rows": len(train_source),
            },
            "eval": {
                "path": str(args.eval_input),
                "sha256": source_sha256(args.eval_input),
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
        "train_task_ids": sorted(train_ids),
        "eval_task_ids": sorted(eval_ids),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        f"Wrote {len(train_rows)} train rows ({len(train_tasks)} tasks) and "
        f"{len(eval_rows)} eval rows ({len(eval_tasks)} tasks) to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
