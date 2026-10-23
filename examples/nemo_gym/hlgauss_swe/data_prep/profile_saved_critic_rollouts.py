#!/usr/bin/env python3
"""Profile stored GPP8 Gym groups and export a PPO curriculum by pass count.

Uses the per-shard manifest written after each atomic group file. This script
never modifies the banked data and needs no training environment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpp", type=int, default=8)
    parser.add_argument("--min-pass", type=int, default=2)
    parser.add_argument("--max-pass", type=int, default=6)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.min_pass <= args.max_pass <= args.gpp:
        parser.error("Require 0 <= min-pass <= max-pass <= gpp")

    rows = args.dataset.read_bytes().splitlines(keepends=True)
    manifest = [json.loads(line) for line in args.image_manifest.open()]
    if len(rows) != len(manifest):
        raise ValueError("Dataset and image manifest row counts differ")
    groups: dict[int, Path] = {}
    records: dict[int, dict] = {}
    for path in args.rollouts.glob("shard_*/group_*.pt"):
        index = int(path.stem.removeprefix("group_"))
        if index in groups:
            raise ValueError(f"Duplicate saved group for dataset index {index}")
        groups[index] = path
    for manifest_path in args.rollouts.glob("shard_*/manifest.jsonl"):
        for line in manifest_path.open():
            record = json.loads(line)
            index = int(record["dataset_idx"])
            if index in records:
                raise ValueError(f"Duplicate manifest record for dataset index {index}")
            records[index] = record
    if not groups:
        raise ValueError(f"No saved groups found under {args.rollouts}")
    if len(groups) != len(rows) and not args.allow_incomplete:
        raise ValueError(
            f"Only {len(groups)}/{len(rows)} groups collected; "
            "use --allow-incomplete for an interim profile"
        )
    if groups.keys() != records.keys():
        raise ValueError(
            f"Group files and manifest indices differ: "
            f"{len(groups.keys() - records.keys())} files without records, "
            f"{len(records.keys() - groups.keys())} records without files"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.output_dir / "passrate_profile.csv"
    selected_path = args.output_dir / f"ppo_pass_{args.min_pass}_to_{args.max_pass}_of_{args.gpp}.jsonl"
    selected_manifest = args.output_dir / "selected_image_manifest.jsonl"
    with (
        profile_path.open("w", newline="") as profile_file,
        selected_path.open("wb") as selected_file,
        selected_manifest.open("w") as selected_manifest_file,
    ):
        writer = csv.DictWriter(
            profile_file,
            fieldnames=(
                "dataset_index", "instance_id", "dataset_name", "pass_count",
                "gpp", "pass_rate", "reward_mean", "truncated_count", "group_file",
            ),
        )
        writer.writeheader()
        selected_count = 0
        for index, path in sorted(groups.items()):
            if index >= len(rows):
                raise ValueError(f"Out-of-range saved group {index}")
            item = manifest[index]
            if item["dataset_index"] != index:
                raise ValueError(f"Image manifest index mismatch at {index}")
            record = records[index]
            if int(record["num_samples"]) != args.gpp:
                raise ValueError(f"Incomplete group in manifest: {path}")
            pass_count = int(record["pass_count"])
            if not 0 <= pass_count <= args.gpp:
                raise ValueError(f"Invalid pass count in manifest: {path}")
            writer.writerow(
                {
                    "dataset_index": index,
                    "instance_id": item["instance_id"],
                    "dataset_name": item["dataset_name"],
                    "pass_count": pass_count,
                    "gpp": args.gpp,
                    "pass_rate": pass_count / args.gpp,
                    "reward_mean": float(record["reward_mean"]),
                    "truncated_count": round(float(record["truncated_frac"]) * args.gpp),
                    "group_file": str(path),
                }
            )
            if args.min_pass <= pass_count <= args.max_pass:
                selected_file.write(rows[index])
                selected_manifest_file.write(json.dumps(item) + "\n")
                selected_count += 1
    print(
        json.dumps(
            {
                "available_groups": len(groups),
                "missing_groups": len(rows) - len(groups),
                "selected": selected_count,
                "profile": str(profile_path),
                "ppo_dataset": str(selected_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
