#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit and split image-tool GRPO rows without changing prompts or grader routing.

Run on the machine hosting the source images. Does not copy images, contact
models, infer dataset licenses, or overwrite an existing output directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


AGENT = {"type": "responses_api_agents", "name": "image_tools_simple_agent"}
GRADERS = {
    "string_match_simple_agent",
    "math_with_judge_simple_agent",
    "mcqa_simple_agent",
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def image_parts(row: dict) -> list[dict]:
    """Return mutable image parts in the model input; never inspect metadata images."""
    parts = []
    for message in row["responses_create_params"]["input"]:
        if not isinstance(message, dict):
            raise ValueError("Input messages must be objects")
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_image":
                    parts.append(part)
    return parts


def validate_row(row: dict, line: int) -> None:
    """Validate the supported full-episode format rather than silently converting pivots."""
    if row.get("agent_ref") != AGENT or "expected_action" in row:
        raise ValueError(
            f"Row {line}: expected full-episode image-tool GRPO, not pivot data"
        )
    grader = row.get("image_tools_base_agent_ref")
    if (
        not isinstance(grader, dict)
        or grader.get("name") not in GRADERS
        or grader.get("type") != "responses_api_agents"
    ):
        raise ValueError(f"Row {line}: unknown grader route")
    params = row.get("responses_create_params")
    if not isinstance(params, dict) or not isinstance(params.get("input"), list):
        raise ValueError(f"Row {line}: missing Responses input list")
    if params.get("tools") != [] or params.get("parallel_tool_calls") is not False:
        raise ValueError(
            f"Row {line}: expected textual tools with tools=[] and parallel_tool_calls=false"
        )
    if (
        not isinstance(row.get("expected_answer"), str)
        or not row["expected_answer"].strip()
    ):
        raise ValueError(f"Row {line}: missing answer")
    if grader["name"] == "math_with_judge_simple_agent" and not isinstance(
        row.get("question"), str
    ):
        raise ValueError(f"Row {line}: math grader requires question")
    if grader["name"] == "mcqa_simple_agent":
        options = row.get("options")
        if not isinstance(options, list) or not all(
            isinstance(option, dict) for option in options
        ):
            raise ValueError(f"Row {line}: MCQA grader requires top-level options")
        letters = {
            key.upper()
            for option in options
            for key, value in option.items()
            if isinstance(key, str)
            and len(key) == 1
            and key.isalpha()
            and value is not None
        }
        if row["expected_answer"].strip().upper() not in letters:
            raise ValueError(
                f"Row {line}: MCQA answer absent from allowed option letters"
            )
    cap = params.get("max_output_tokens")
    if cap is not None and (type(cap) is not int or cap <= 0):
        raise ValueError(f"Row {line}: invalid output token cap")
    if not image_parts(row):
        raise ValueError(f"Row {line}: missing image")
    for part in image_parts(row):
        path = part.get("image_url")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError(f"Row {line}: staging requires absolute local image paths")
    if row.get("metadata") is not None and not isinstance(row["metadata"], dict):
        raise ValueError(f"Row {line}: metadata must be an object")


def prepare(
    source: Path | list[Path],
    *,
    validation_fraction: float,
    seed: int,
    max_output_tokens: int,
    asset_root: Path | None = None,
    hash_images: bool = False,
) -> tuple[dict[str, list[dict]], list[dict], dict]:
    """Build transitive source/image groups and a deterministic group-held-out split.

    With hash_images, identical image bytes at different paths are grouped too.
    Without it, leakage protection covers canonical paths and declared source IDs.
    """
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one")
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be a positive integer")
    if asset_root is not None and not asset_root.is_absolute():
        raise ValueError("asset_root must be an absolute destination path")
    rows = []
    source_hash = hashlib.sha256()
    sources = [source] if isinstance(source, Path) else source
    number = 0
    for source_path in sources:
        with source_path.open("rb") as stream:
            for line in stream:
                number += 1
                source_hash.update(line)
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Row {number}: expected an object")
                validate_row(row, number)
                rows.append(row)
    if not rows:
        raise ValueError("Empty dataset")

    parents: dict[str, str] = {}

    def find(key):
        parents.setdefault(key, key)
        root = key
        while parents[root] != root:
            root = parents[root]
        while key != root:
            next_key = parents[key]
            parents[key] = root
            key = next_key
        return root

    def join(left, right):
        a, b = find(left), find(right)
        parents[max(a, b)] = min(a, b)

    assets = {}
    row_keys = []
    for row in rows:
        keys = []
        for part in image_parts(row):
            path = Path(part["image_url"]).resolve(strict=True)
            if not path.is_file():
                raise ValueError("Image reference must be a regular file")
            canonical = str(path)
            if canonical not in assets:
                content_hash = None
                if hash_images:
                    hasher = hashlib.sha256()
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            hasher.update(block)
                    content_hash = hasher.hexdigest()
                asset_id = content_hash or digest(canonical)
                relative = f"{asset_id[:2]}/{asset_id}{path.suffix.lower()}"
                assets[canonical] = {
                    "source_path": canonical,
                    "relative_path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": content_hash,
                }
            asset = assets[canonical]
            keys.append("image:" + (asset["sha256"] or canonical))
        source_id = (row.get("metadata") or {}).get("source_id")
        if source_id is not None and str(source_id):
            keys.append("source:" + str(source_id))
        for key in keys:
            join(keys[0], key)
        row_keys.append(keys[0])

    groups: dict[str, list[int]] = {}
    for index, key in enumerate(row_keys):
        groups.setdefault(find(key), []).append(index)
    if len(groups) < 2:
        raise ValueError(
            "Need at least two independent source/image groups for held-out evaluation"
        )
    ordered = sorted(groups, key=lambda key: digest(f"{seed}:{key}"))
    n_validation = min(
        len(ordered) - 1, max(1, round(len(ordered) * validation_fraction))
    )
    held_out = set(ordered[:n_validation])
    splits = {"train": [], "validation": []}
    counts = {"train": Counter(), "validation": Counter()}
    # Stable within-split order too, independent of source-file shuffle.
    for key in ordered:
        split = "validation" if key in held_out else "train"
        for index in sorted(
            groups[key], key=lambda i: digest(json.dumps(rows[i], sort_keys=True))
        ):
            row = copy.deepcopy(rows[index])
            dataset = row.get("dataset") or "unspecified"
            label = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(dataset))
            row["env_id"] = f"image_tools/{label}"
            row["task_source"] = AGENT["name"]
            row["task_id"] = "image-tools-" + digest(
                json.dumps(rows[index], sort_keys=True)
            )
            params = row["responses_create_params"]
            params["max_output_tokens"] = min(
                params.get("max_output_tokens") or max_output_tokens, max_output_tokens
            )
            row["task_metadata"] = {
                **(row.get("task_metadata") or {}),
                "suite": "image_tools",
                "split": split,
                "source_sha256": source_hash.hexdigest(),
                "split_group": digest(key),
            }
            if asset_root is not None:
                for part in image_parts(row):
                    canonical = str(Path(part["image_url"]).resolve(strict=True))
                    part["image_url"] = str(
                        asset_root / assets[canonical]["relative_path"]
                    )
            splits[split].append(row)
            counts[split][row["image_tools_base_agent_ref"]["name"]] += 1
    report = {
        "source_sha256": source_hash.hexdigest(),
        "rows": len(rows),
        "groups": len(groups),
        "seed": seed,
        "requested_validation_group_fraction": validation_fraction,
        "split_rows": {s: len(r) for s, r in splits.items()},
        "grader_counts": counts,
        "unique_source_images": len(assets),
        "image_bytes": sum(a["bytes"] for a in assets.values()),
        "image_content_hashes_checked": hash_images,
        "max_output_tokens": max_output_tokens,
        "asset_root": str(asset_root) if asset_root else None,
    }
    return splits, sorted(assets.values(), key=lambda a: a["source_path"]), report


def write_bundle(output_dir: Path, splits: dict, assets: list, report: dict) -> None:
    """Write a new bundle; existing directories are never reused or overwritten."""
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in {**splits, "assets": assets}.items():
        with (output_dir / f"{name}.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        action="append",
        help="Input JSONL; repeat to regroup existing splits without copying images",
    )
    parser.add_argument("--output-dir", type=Path, help="Omit to audit without writing")
    parser.add_argument(
        "--asset-root",
        type=Path,
        help="Optional absolute destination image root; does not transfer images",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument(
        "--hash-images",
        action="store_true",
        help="Also group identical bytes at different image paths",
    )
    args = parser.parse_args()
    splits, assets, report = prepare(
        args.source,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        max_output_tokens=args.max_output_tokens,
        asset_root=args.asset_root,
        hash_images=args.hash_images,
    )
    if args.output_dir:
        write_bundle(args.output_dir, splits, assets, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
