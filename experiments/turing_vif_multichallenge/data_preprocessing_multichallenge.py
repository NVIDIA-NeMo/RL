#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Preprocesses MultiChallenge dataset from raw per-task JSON files into
the Turing VIF JSONL format.

Reads individual JSON files from ``{data_dir}/{split}/*.json`` and produces
a single JSONL file per split where each line is a turing_vif-format task.

Key transformations:
- Filters out ``"thinking"`` role messages
- Prepends the ``system`` field as a system message (when present)
- Builds a conversation context string for the LLM judge
- Converts rubric items into ``llm_judge`` items with context embedded
  in the ``content`` field (preserving the information the old global
  judge template provided)
"""

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_RAW_DATA_DIR = Path(
    "/lustre/fsw/portfolios/llmservice/users/mfathi/data/multichallenge"
)


def build_input_messages(task: dict) -> list[dict]:
    """Build ``responses_create_params.input`` from the raw task."""
    messages = task.get("messages", [])
    system_prompt = task.get("system", None)

    input_msgs: list[dict] = []

    if system_prompt:
        input_msgs.append({"role": "system", "content": system_prompt})

    for msg in messages:
        role = msg.get("role", "")
        if role == "thinking":
            continue
        input_msgs.append({"role": role, "content": msg.get("content", "")})

    return input_msgs


def build_context_string(task: dict) -> str:
    """Build a human-readable context string for the judge (same format
    as the old ``dataset_preprocess.py``)."""
    messages = task.get("messages", [])
    system_prompt = task.get("system", None)

    parts: list[str] = []
    if system_prompt:
        parts.append(f"[SYSTEM]: {system_prompt}")

    for msg in messages:
        role = msg.get("role", "")
        if role == "thinking":
            continue
        parts.append(f"[{role.upper()}]: {msg.get('content', '')}")

    return "\n\n".join(parts)


def build_judge_content(context: str, question: str, pass_criteria: str) -> str:
    """Embed the full conversation context, rubric question, and expected
    answer into a single ``content`` string for the turing_vif ``llm_judge``
    item.  This preserves the information the old global judge template
    provided via its ``{context}``, ``{question}``, and ``{pass_criteria}``
    placeholders."""
    return (
        f"Given the following conversation:\n\n"
        f"{context}\n\n"
        f"Does the model's final response satisfy this criterion?\n\n"
        f"Criterion: {question}\n\n"
        f"Expected answer: {pass_criteria}"
    )


def process_task(task: dict, fallback_id: str = "unknown") -> dict[str, Any]:
    """Convert a single raw task dict into a turing_vif JSONL record."""
    metadata = task.get("metadata", {})
    task_id = metadata.get("taskId", fallback_id)

    input_messages = build_input_messages(task)
    context = build_context_string(task)

    llm_judge_items: list[dict[str, Any]] = []
    for i, rubric_item in enumerate(task.get("rubric", []), start=1):
        question = rubric_item.get("question", "")
        pass_criteria = rubric_item.get("pass_criteria", "YES")
        llm_judge_items.append(
            {
                "uid": i,
                "content": build_judge_content(context, question, pass_criteria),
                "pass_criteria": pass_criteria,
                "source": "system",
                "is_misalignment_check": False,
            }
        )

    return {
        "id": int(task_id) if isinstance(task_id, (int, float)) else task_id,
        "instructions": [],
        "llm_judge": llm_judge_items,
        "responses_create_params": {"input": input_messages},
        "language": "en",
        "agent_ref": {
            "type": "responses_api_agents",
            "name": "turing_vif_simple_agent",
        },
    }


def process_task_file(filepath: Path) -> dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        task = json.load(f)
    return process_task(task, fallback_id=filepath.stem)


def process_split(data_dir: Path, split: str, output_dir: Path) -> int:
    split_dir = data_dir / split
    if not split_dir.exists():
        print(f"Warning: split directory not found: {split_dir}")
        return 0

    json_files = sorted(split_dir.glob("*.json"))
    if not json_files:
        print(f"Warning: no JSON files in {split_dir}")
        return 0

    output_file = output_dir / f"multichallenge_{split}.jsonl"
    print(f"Processing {len(json_files)} files from {split_dir} ...")

    count = 0
    errors = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        for filepath in json_files:
            try:
                record = process_task_file(filepath)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                print(f"  Error processing {filepath.name}: {e}")
                errors += 1

    status = f"  Wrote {count} records to {output_file}"
    if errors:
        status += f" ({errors} errors)"
    print(status)
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw MultiChallenge JSON files to Turing VIF JSONL format"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help=f"Directory containing split subdirectories (default: {DEFAULT_RAW_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Output directory for JSONL files (default: ./data)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["advanced", "vanilla"],
        help="Splits to process (default: advanced vanilla)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory:  {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Splits:           {args.splits}")
    print()

    total = 0
    for split in args.splits:
        total += process_split(args.data_dir, split, args.output_dir)

    print(f"\nTotal: {total} records processed")


if __name__ == "__main__":
    main()
