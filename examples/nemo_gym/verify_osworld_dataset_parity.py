#!/usr/bin/env python3
"""Verify that the converted NeMo-Gym OSWorld set matches Jianh's source tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict[str, str]:
    tasks: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            metadata = row.get("verifier_metadata") or {}
            task_id = (
                row.get("example_id")
                or metadata.get("example_id")
                or metadata.get("id")
            )
            instruction = metadata.get("instruction")
            if instruction is None:
                inputs = (row.get("responses_create_params") or {}).get("input") or []
                instruction = inputs[-1].get("content") if inputs else None
            if not isinstance(task_id, str) or not isinstance(instruction, str):
                raise ValueError(f"{path}:{line_number}: missing task ID or instruction")
            if task_id in tasks:
                raise ValueError(f"{path}:{line_number}: duplicate task ID {task_id}")
            tasks[task_id] = instruction
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("canonical", type=Path)
    parser.add_argument("converted", type=Path)
    args = parser.parse_args()

    canonical = _load(args.canonical)
    converted = _load(args.converted)
    missing = sorted(canonical.keys() - converted.keys())
    extra = sorted(converted.keys() - canonical.keys())
    changed = sorted(
        task_id
        for task_id in canonical.keys() & converted.keys()
        if canonical[task_id] != converted[task_id]
    )
    print(
        f"canonical={len(canonical)} converted={len(converted)} "
        f"missing={len(missing)} extra={len(extra)} changed_instructions={len(changed)}"
    )
    if missing or extra or changed:
        for label, values in (
            ("missing", missing),
            ("extra", extra),
            ("changed_instructions", changed),
        ):
            if values:
                print(f"{label}: {values[:20]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
