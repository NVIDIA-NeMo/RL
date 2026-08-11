# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Search-R1's official NQ/HotpotQA parquet files to NeMo Gym JSONL."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


AGENT_REF = {
    "type": "responses_api_agents",
    "name": "ai_search_search_r1_agent",
}


def _as_list(value: Any, field_name: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be list-valued, got {type(value)}")
    return value


def _convert_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt = _as_list(row["prompt"], "prompt")
    messages = []
    for message in prompt:
        if not isinstance(message, dict):
            raise ValueError("prompt entries must be objects")
        role = message.get("role")
        content = message.get("content")
        if role not in ("user", "assistant", "system", "developer"):
            raise ValueError(f"unsupported prompt role: {role!r}")
        if not isinstance(content, str):
            raise ValueError("prompt message content must be text")
        messages.append({"role": role, "content": content})

    answers = [
        str(answer) for answer in _as_list(row["golden_answers"], "golden_answers")
    ]
    if not answers:
        raise ValueError("golden_answers must not be empty")

    return {
        "question": str(row["question"]),
        "answers": answers,
        "supporting_doc_ids": [],
        "data_source": str(row["data_source"]),
        "source_id": str(row["id"]),
        "agent_ref": AGENT_REF,
        "responses_create_params": {
            "input": messages,
            "tools": [],
            "parallel_tool_calls": False,
            "tool_choice": "none",
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            encoded = (
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode()
            output.write(encoded.decode())
            digest.update(encoded)
    return digest.hexdigest()


def convert_split(source: Path, destination: Path, limit: int | None) -> dict[str, Any]:
    """Convert one official parquet split and return reproducibility metadata."""
    frame = pd.read_parquet(source)
    required = {"id", "question", "golden_answers", "data_source", "prompt"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    if limit is not None:
        frame = frame.head(limit)

    rows = [_convert_row(row) for row in frame.to_dict(orient="records")]
    digest = _write_jsonl(destination, rows)
    source_counts: dict[str, int] = {}
    for row in rows:
        source_name = row["data_source"]
        source_counts[source_name] = source_counts.get(source_name, 0) + 1
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "rows": len(rows),
        "data_sources": source_counts,
        "sha256": digest,
    }


def prepare(
    source_dir: Path,
    output_dir: Path,
    max_train_rows: int | None = None,
    max_validation_rows: int | None = None,
) -> dict[str, Any]:
    """Convert the official train/test pair without changing prompt text."""
    train = convert_split(
        source_dir / "train.parquet",
        output_dir / "train.jsonl",
        max_train_rows,
    )
    validation = convert_split(
        source_dir / "test.parquet",
        output_dir / "validation.jsonl",
        max_validation_rows,
    )
    manifest = {
        "schema_version": 1,
        "upstream_dataset": "PeterJinGo/nq_hotpotqa_train",
        "train": train,
        "validation": validation,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    plugin_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing official train.parquet and test.parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            plugin_root / "resources_servers" / "ai_search" / "data" / "search_r1"
        ),
    )
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=None)
    args = parser.parse_args()
    manifest = prepare(
        args.source_dir.resolve(),
        args.output_dir.resolve(),
        args.max_train_rows,
        args.max_validation_rows,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
