#!/usr/bin/env python3
"""Prepare byte-faithful measured and warmup JSONL workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--warmup", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--warmup-requests", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_prompts <= 0 or args.warmup_requests <= 0:
        raise ValueError("prompt and warmup counts must be positive")
    if args.warmup_requests > args.num_prompts:
        raise ValueError("warmup requests cannot exceed measured prompts")

    lines: list[str] = []
    with args.source.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise TypeError(f"{args.source}:{line_number}: expected a JSON object")
            lines.append(raw_line.rstrip("\n") + "\n")
            if len(lines) == args.num_prompts:
                break
    if len(lines) != args.num_prompts:
        raise ValueError(
            f"source has {len(lines)} usable rows, expected {args.num_prompts}"
        )

    for path in (args.workload, args.warmup):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing workload: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    args.workload.write_text("".join(lines), encoding="utf-8")
    args.warmup.write_text("".join(lines[: args.warmup_requests]), encoding="utf-8")
    print(
        json.dumps(
            {
                "workload_records": args.num_prompts,
                "workload_sha256": sha256(args.workload),
                "warmup_records": args.warmup_requests,
                "warmup_sha256": sha256(args.warmup),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
