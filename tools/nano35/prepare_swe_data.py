#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow==21.0.0"]
# ///
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recreate the ordered Nano SWE dataset from a pinned public HF snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

PROFILE = Path(__file__).with_name("swe_verified_403.json")


def make_rows(
    source_rows: list[dict[str, Any]], instance_ids: list[str]
) -> list[dict[str, Any]]:
    """Select exact IDs in recorded order and restore the Gym request envelope."""
    by_id = {row["instance_id"]: row for row in source_rows}
    if len(by_id) != len(source_rows) or len(set(instance_ids)) != len(instance_ids):
        raise ValueError("Duplicate instance IDs in source or selection")
    missing = set(instance_ids) - by_id.keys()
    if missing:
        raise ValueError(f"Missing public instances: {sorted(missing)}")
    rows = []
    for instance_id in instance_ids:
        source = by_id[instance_id]
        metadata = {
            "instance_id": instance_id,
            "base_commit": source["base_commit"],
            "dataset_name": "princeton-nlp/SWE-bench_Verified",
            "split": "test",
            "problem_statement": source["problem_statement"],
            "golden_patch": source["patch"],
        }
        metadata["instance_dict"] = json.dumps({**metadata, **source})
        rows.append(
            {
                "responses_create_params": {
                    "input": [],
                    "metadata": metadata,
                    # Preserve the historical request template for byte-identical data.
                    "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                    "temperature": 1.0,
                    "top_p": 1.0,
                },
                "agent_ref": {
                    "type": "responses_api_agents",
                    "name": "swe_agents_train",
                },
                **source,
            }
        )
    return rows


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Serialize with the same field order and encoding as the training input."""
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def write_matching(path: Path, content: bytes) -> None:
    """Reuse identical assets while rejecting conflicting existing files."""
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(
                f"Existing file differs; choose a new output directory: {path}"
            )
        return
    with path.open("xb") as output:
        output.write(content)


def prepare(*, parquet: Path, output_dir: Path) -> None:
    """Check the public input, reconstruct all rows, and verify the output hash."""
    # The Arrow reader is only needed for preparation, not the pure row helpers.
    import pyarrow.parquet as pq

    profile = json.loads(PROFILE.read_text())
    source = profile["source"]
    if hashlib.sha256(parquet.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("Public parquet checksum differs from the pinned snapshot")
    raw_rows = pq.read_table(parquet).to_pylist()
    rows = make_rows(raw_rows, profile["instance_ids"])
    content = jsonl_bytes(rows)
    if hashlib.sha256(content).hexdigest() != profile["output_sha256"]:
        raise ValueError("Reconstructed data differs from the recorded 403-row input")
    by_id = {row["instance_id"]: row for row in raw_rows}
    raw = jsonl_bytes([by_id[instance_id] for instance_id in profile["instance_ids"]])
    if hashlib.sha256(raw).hexdigest() != profile["raw_sha256"]:
        raise ValueError("Raw builder input differs from the pinned rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_matching(output_dir / "swe_verified_403.jsonl", content)
    write_matching(output_dir / "swe_verified_403.raw.jsonl", raw)
    report = {
        "source": source,
        "rows": len(rows),
        "data_sha256": profile["output_sha256"],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "instance_ids": profile["instance_ids"],
    }
    write_matching(
        output_dir / "data-manifest.json",
        (json.dumps(report, indent=2) + "\n").encode(),
    )
    print(f"Prepared {len(rows)} rows in {output_dir}")
    print(f"Training JSONL SHA-256: {profile['output_sha256']}")


def main() -> None:
    """Download the pinned public parquet or verify a previously downloaded copy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--parquet", type=Path, help="Use a local pinned parquet offline"
    )
    args = parser.parse_args()
    if args.parquet is not None:
        prepare(parquet=args.parquet, output_dir=args.output_dir)
        return
    profile = json.loads(PROFILE.read_text())
    with tempfile.TemporaryDirectory(prefix="nano35-data-") as temp:
        parquet = Path(temp) / "verified.parquet"
        with urllib.request.urlopen(profile["source"]["url"], timeout=120) as response:
            parquet.write_bytes(response.read())
        prepare(parquet=parquet, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
