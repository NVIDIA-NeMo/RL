#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit local W&B circle-count result tables without printing image payloads."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


def _response_text(response: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        generation_str = item.get("generation_str")
        if isinstance(generation_str, str):
            chunks.append(generation_str)
        for summary in item.get("summary", []):
            if isinstance(summary, dict) and isinstance(summary.get("text"), str):
                chunks.append(summary["text"])
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "\n".join(chunks)


def _request_details(request: dict[str, Any]) -> tuple[int, str]:
    image_count = 0
    query = ""
    for message in request.get("input", []):
        content = message.get("content", []) if isinstance(message, dict) else []
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "input_image":
                image_count += 1
            elif item.get("type") == "input_text":
                query = str(item.get("text", ""))
    return image_count, query


def _percentile(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table_dir", type=Path)
    parser.add_argument("--show-longest", type=int, default=8)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for table_path in sorted(args.table_dir.glob("*.table.json")):
        table = json.loads(table_path.read_text())
        for row_index, row in enumerate(table.get("data", [])):
            if not row or not isinstance(row[0], str):
                continue
            result = json.loads(row[0])
            request = result.get("responses_create_params", {})
            response = result.get("response", {})
            image_count, query = _request_details(request)
            text = _response_text(response)
            usage = response.get("usage") or {}
            incomplete = response.get("incomplete_details") or {}
            request_json = json.dumps(request)
            rows.append(
                {
                    "file": table_path.name,
                    "row": row_index,
                    "image_count": image_count,
                    "query": query,
                    "text": text,
                    "output_tokens": int(usage.get("output_tokens") or 0),
                    "incomplete_reason": incomplete.get("reason"),
                    "leaked_metadata": any(
                        marker in request_json
                        for marker in ('"circles"', '"target_color"', '"expected_count"')
                    ),
                    "boxed": bool(re.search(r"\\boxed\{[^}]+\}", text)),
                    "image_grounded": "circle" in text.lower()
                    and any(
                        word in text.lower()
                        for word in (
                            "image",
                            "red",
                            "blue",
                            "green",
                            "yellow",
                            "purple",
                            "orange",
                            "cyan",
                            "pink",
                        )
                    ),
                    "reward": result.get("reward"),
                    "expected": result.get("expected_count"),
                    "predicted": result.get("predicted_count"),
                    "correct": result.get("correct"),
                }
            )

    if not rows:
        raise SystemExit(f"No rollout rows found in {args.table_dir}")

    token_counts = [row["output_tokens"] for row in rows]
    print(f"rows={len(rows)} files={len({row['file'] for row in rows})}")
    print(
        "images_exactly_one="
        f"{sum(row['image_count'] == 1 for row in rows)}/{len(rows)} "
        f"metadata_leaks={sum(row['leaked_metadata'] for row in rows)}"
    )
    print(
        f"boxed={sum(row['boxed'] for row in rows)}/{len(rows)} "
        f"image_grounded={sum(row['image_grounded'] for row in rows)}/{len(rows)} "
        f"correct={sum(row['correct'] is True for row in rows)}/{len(rows)}"
    )
    print(
        "output_tokens "
        f"min={min(token_counts)} median={statistics.median(token_counts):.1f} "
        f"p95={_percentile(token_counts, 0.95)} max={max(token_counts)}"
    )
    reasons: dict[str, int] = {}
    for row in rows:
        reason = str(row["incomplete_reason"] or "complete")
        reasons[reason] = reasons.get(reason, 0) + 1
    print(f"termination={json.dumps(reasons, sort_keys=True)}")

    print("longest_rollouts:")
    for row in sorted(rows, key=lambda item: item["output_tokens"], reverse=True)[
        : args.show_longest
    ]:
        excerpt = " ".join(row["text"].split())[-500:]
        print(
            f"- {row['file']}[{row['row']}] tokens={row['output_tokens']} "
            f"reason={row['incomplete_reason'] or 'complete'} "
            f"expected={row['expected']} predicted={row['predicted']} "
            f"reward={row['reward']} query={row['query']!r}"
        )
        print(f"  tail={excerpt!r}")


if __name__ == "__main__":
    main()
