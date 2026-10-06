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

"""Prepare OSWorld Gym rows for SingleController context compaction."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

# SingleController creates these fields for each live dispatch. Dataset values
# would alias independent attempts or logical generations onto the same ledger.
_FRAMEWORK_OWNED_FIELDS = frozenset(
    {
        "_ng_attempt_index",
        "_ng_group_attempt",
        "_ng_group_id",
        "_ng_rollout_id",
        "_ng_rollout_index",
        "_ng_task_index",
    }
)

# These v1 fields described physical trace materialization. The v2 runtime gets
# its authoritative logical segments from Gym's LogicalCCResult instead.
_LEGACY_TRACE_FIELDS = frozenset(
    {
        "context_compaction_attempt_index",
        "context_compaction_contract_version",
        "context_compaction_rollout_index",
    }
)


def task_id(row: dict[str, Any]) -> str:
    """Return the stable verifier task ID from an OSWorld row."""
    verifier_metadata = row.get("verifier_metadata")
    value = verifier_metadata.get("id") if isinstance(verifier_metadata, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError("OSWorld row has no non-empty verifier_metadata.id")
    return value


def validate_source_row(
    row: Any,
    *,
    source: str,
) -> dict[str, Any]:
    """Validate and narrow one raw Gym OSWorld row."""
    if not isinstance(row, dict):
        raise ValueError(f"{source} is not a JSON object")
    responses_create_params = row.get("responses_create_params")
    if not isinstance(responses_create_params, dict):
        raise ValueError(f"{source} has no responses_create_params object")
    if "input" not in responses_create_params:
        raise ValueError(f"{source} has no responses_create_params.input")
    try:
        task_id(row)
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from error
    return row


def read_jsonl(
    path: Path,
    *,
    selected_task_ids: Sequence[str] = (),
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read unique OSWorld rows, retaining source order."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    selected = set(selected_task_ids)
    rows: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = validate_source_row(
                json.loads(line),
                source=f"{path}:{line_number}",
            )
            current_task_id = task_id(row)
            if selected and current_task_id not in selected:
                continue
            if current_task_id in seen_task_ids:
                raise ValueError(
                    f"{path}:{line_number} has duplicate task ID {current_task_id}"
                )
            seen_task_ids.add(current_task_id)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break

    if selected:
        missing = selected.difference(seen_task_ids)
        if missing:
            raise ValueError(
                "Requested task IDs were not found: " + ", ".join(sorted(missing))
            )
    return rows


def prepare_osworld_rows(
    source_rows: Iterable[dict[str, Any]],
    *,
    num_repeats: int,
    agent_name: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict[str, Any]]:
    """Add Gym routing and stable provenance without runtime-owned IDs."""
    if num_repeats < 1:
        raise ValueError("num_repeats must be at least 1")
    if not agent_name:
        raise ValueError("agent_name must not be empty")
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be positive")

    rows = list(source_rows)
    output_rows: list[dict[str, Any]] = []
    for repeat_index in range(num_repeats):
        for source_index, source_row in enumerate(rows, start=1):
            validate_source_row(source_row, source=f"source row {source_index}")
            prepared = copy.deepcopy(source_row)
            current_task_id = task_id(prepared)

            for field in _FRAMEWORK_OWNED_FIELDS | _LEGACY_TRACE_FIELDS:
                prepared.pop(field, None)

            prepared["responses_create_params"].update(
                {
                    "max_output_tokens": max_output_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
            prepared["agent_ref"] = {
                "type": "responses_api_agents",
                "name": agent_name,
            }

            # Informational dataset provenance only. SingleController separately
            # owns every _ng_* dispatch identity and Gym owns LogicalCCResult.
            prepared["context_compaction_task_id"] = current_task_id
            prepared["context_compaction_group_id"] = (
                f"osworld:{current_task_id}:repeat:{repeat_index}"
            )
            output_rows.append(prepared)
    return output_rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write compact UTF-8 JSONL and return its row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(
                json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n"
            )
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--limit", type=int)
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
    """Prepare and write the requested OSWorld dataset."""
    args = parse_args()
    source_rows = read_jsonl(
        args.input,
        selected_task_ids=args.task_id,
        limit=args.limit,
    )
    output_rows = prepare_osworld_rows(
        source_rows,
        num_repeats=args.num_repeats,
        agent_name=args.agent_name,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    count = write_jsonl(args.output, output_rows)
    print(
        f"Wrote {count} rows to {args.output} "
        f"({len(source_rows)} unique tasks x {args.num_repeats} repeats)."
    )


if __name__ == "__main__":
    main()
