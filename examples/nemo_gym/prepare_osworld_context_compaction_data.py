#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Convert OSWorld tasks into context-compaction-aware NeMo-Gym rows."""

import argparse
import copy
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Include only this verifier task ID; may be passed more than once.",
    )
    parser.add_argument("--agent-name", default="nemotron_osworld")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_repeats < 1:
        raise ValueError("--num-repeats must be at least 1")

    source_rows = []
    with args.input.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            verifier_metadata = row.get("verifier_metadata")
            if "responses_create_params" not in row or not isinstance(
                verifier_metadata, dict
            ):
                raise ValueError(
                    f"{args.input}:{line_number} is not an OSWorld Gym task row"
                )
            task_id = verifier_metadata.get("id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(
                    f"{args.input}:{line_number} has no verifier task ID"
                )
            if args.task_id and task_id not in args.task_id:
                continue
            source_rows.append(row)
            if args.limit is not None and len(source_rows) >= args.limit:
                break

    if args.task_id:
        found_task_ids = {
            row["verifier_metadata"]["id"] for row in source_rows
        }
        missing_task_ids = set(args.task_id) - found_task_ids
        if missing_task_ids:
            raise ValueError(
                "Requested task IDs were not found: "
                + ", ".join(sorted(missing_task_ids))
            )

    output_rows = []
    for repeat_index in range(args.num_repeats):
        for row in source_rows:
            prepared = copy.deepcopy(row)
            task_id = prepared["verifier_metadata"]["id"]
            prepared["responses_create_params"].update(
                {
                    "max_output_tokens": args.max_output_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
            )
            prepared["agent_ref"] = {
                "type": "responses_api_agents",
                "name": args.agent_name,
            }
            prepared.update(
                {
                    "context_compaction_contract_version": 2,
                    "context_compaction_group_id": (
                        f"osworld:{task_id}:repeat:{repeat_index}"
                    ),
                    "context_compaction_task_id": task_id,
                    "context_compaction_rollout_index": 0,
                    "context_compaction_attempt_index": 0,
                }
            )
            output_rows.append(prepared)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as destination:
        for row in output_rows:
            destination.write(json.dumps(row, separators=(",", ":")) + "\n")

    print(
        f"Wrote {len(output_rows)} rows to {args.output} "
        f"({len(source_rows)} unique tasks x {args.num_repeats} repeats)."
    )


if __name__ == "__main__":
    main()
