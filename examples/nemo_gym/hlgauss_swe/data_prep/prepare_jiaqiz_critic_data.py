#!/usr/bin/env python3
"""Keep Jiaqi curriculum rows with a locally available OpenHands SIF image.

The input JSONL is copied byte-for-byte. The manifest records the resolved image
and original row index so later rollout shards can be traced back to the source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


SOURCE = Path(
    "/lustre/fsw/portfolios/llmservice/users/jiaqiz/research/ppo/data/"
    "intersection_pass_ge1of16_no-critic-heldout_shuf42.jsonl"
)
OUT_DIR = Path(
    "/lustre/fsw/portfolios/coreai/users/bihu/repos/multi-turn-swe/data/"
    "jiaqiz_intersection_openhands_images"
)
REB_PATH = Path("/lustre/fsw/portfolios/llmservice/users/sdevare/images/swerebench")
SWE_PATH = Path(
    "/lustre/fsw/portfolios/llmservice/users/jiaqiz/swe/singularity_images/swe-gym"
)
R2E_PATH = Path("/lustre/fsw/portfolios/llmservice/users/igitman/images/swe-bench")


def find_image(dataset_name: str, instance_id: str) -> Path | None:
    if dataset_name == "SWE-rebench-V2":
        candidate = REB_PATH / f"{instance_id}.sif"
    elif dataset_name == "R2E-Gym/R2E-Gym-Subset":
        modified = re.sub(
            r"[^_]+__([^-]+)-",
            lambda match: match.group(1).lower() + "_final_",
            instance_id,
        )
        candidate = R2E_PATH / f"namanjain12_{modified}.sif"
    elif dataset_name == "SWE-Gym/SWE-Gym":
        candidate = SWE_PATH / (
            "xingyaoww_sweb.eval.x86_64."
            + instance_id.replace("__", "_s_")
            + ".sif"
        )
    else:
        return None
    return candidate if candidate.is_file() and candidate.stat().st_size > 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "train.jsonl"
    mapping = args.out_dir / "image_manifest.jsonl"
    summary = args.out_dir / "summary.json"
    source_hash = hashlib.sha256()
    output_hash = hashlib.sha256()
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    with (
        args.source.open("rb") as source_file,
        output.with_suffix(".jsonl.tmp").open("wb") as output_file,
        mapping.with_suffix(".jsonl.tmp").open("w") as mapping_file,
    ):
        for source_index, line in enumerate(source_file):
            source_hash.update(line)
            row = json.loads(line)
            metadata = row["responses_create_params"]["metadata"]
            instance_id = metadata["instance_id"]
            dataset_name = metadata["dataset_name"]
            if instance_id in seen:
                raise ValueError(f"Duplicate instance_id: {instance_id}")
            seen.add(instance_id)
            image = find_image(dataset_name, instance_id)
            counts[f"source/{dataset_name}"] += 1
            if image is None:
                counts[f"excluded/{dataset_name}"] += 1
                continue
            dataset_index = counts["included/total"]
            output_file.write(line)
            output_hash.update(line)
            mapping_file.write(
                json.dumps(
                    {
                        "dataset_index": dataset_index,
                        "source_index": source_index,
                        "instance_id": instance_id,
                        "dataset_name": dataset_name,
                        "image_path": str(image),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            counts["included/total"] += 1
            counts[f"included/{dataset_name}"] += 1
    output.with_suffix(".jsonl.tmp").replace(output)
    mapping.with_suffix(".jsonl.tmp").replace(mapping)
    summary.write_text(
        json.dumps(
            {
                "source": str(args.source),
                "source_sha256": source_hash.hexdigest(),
                "train_sha256": output_hash.hexdigest(),
                "counts": dict(sorted(counts.items())),
                "image_resolution": "Gym swe_agents/app.py:_find_container exact naming",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"output": str(output), "counts": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
