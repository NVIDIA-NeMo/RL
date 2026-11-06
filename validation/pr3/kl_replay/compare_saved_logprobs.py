# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Compare saved valid-token policy logprobs without loading a model or a GPU."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def compare(first: torch.Tensor, second: torch.Tensor) -> dict[str, float | int | bool]:
    """Describe logprob differences on an identical ordered valid-token vector."""
    if (
        first.shape != second.shape
        or not torch.isfinite(first).all()
        or not torch.isfinite(second).all()
    ):
        raise ValueError("Expected matching finite valid-token vectors")
    error = (first.double() - second.double()).abs()
    return {
        "exact": torch.equal(first, second),
        "valid_tokens": error.numel(),
        "nonzero_differences": int(torch.count_nonzero(error)),
        "max_abs": float(error.max()),
        "p99_abs": float(torch.quantile(error, 0.99)),
        "p999_abs": float(torch.quantile(error, 0.999)),
        "rms": float(error.square().mean().sqrt()),
        "tokens_abs_gt_0_1": int((error > 0.1).sum()),
        "tokens_abs_gt_1": int((error > 1).sum()),
    }


def main() -> None:
    """Read the two completed diagnostic exports in a CPU allocation."""
    if os.environ["USER"] != "root" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run in an allocated CPU container")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for path in sorted((args.directory / "stream-capture/step_210").glob("chunk-*.pt")):
        packet = torch.load(path, map_location="cpu", weights_only=False)
        if packet["trainer_version"] != 210:
            raise ValueError("Expected checkpoint210 capture")
        for index, sample in enumerate(packet["ids"]):
            if sample in rows:
                raise ValueError(f"Duplicate response {sample}")
            data = packet["data"]
            length = int(data["input_lengths"][index])
            mask = (
                data["token_mask"][index, :length] * data["sample_mask"][index]
            ).bool()
            mask[0] = False
            rows[sample] = {"length": length, "mask": mask}
    if len(rows) != 32:
        raise ValueError("Expected32 unique real responses")
    files = {}

    def load(directory: Path, case: str, suffix: str = "eval") -> torch.Tensor:
        path = directory / "replay" / f"{case}-{suffix}-lp.pt"
        values = torch.load(path, map_location="cpu", weights_only=False)
        if set(values) != set(rows):
            raise ValueError(f"Response IDs differ in {path}")
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return torch.cat(
            [
                values[sample][: row["length"]][row["mask"]]
                for sample, row in sorted(rows.items())
            ]
        )

    old_full = load(args.reference, "full")
    old_production = load(args.reference, "split_production_shards")
    new_full = load(args.directory, "full")
    new_production = load(args.directory, "split_production_shards")
    checks = {
        "old_full_vs_unified_full": compare(old_full, new_full),
        "old_production_vs_unified_full": compare(old_production, new_full),
        "unified_full_vs_production": compare(new_full, new_production),
        "unified_recomputed_full_vs_production": compare(
            load(args.directory, "full_recompute"),
            load(args.directory, "split_production_recompute"),
        ),
        "post_update_full_vs_numerator": compare(
            load(args.directory, "full", "post-update"),
            load(args.directory, "numerator_full", "post-update"),
        ),
        "post_update_numerator_vs_fixed_split": compare(
            load(args.directory, "numerator_full", "post-update"),
            load(args.directory, "split_fixed_shards", "post-update"),
        ),
        "post_update_fixed_vs_production_split": compare(
            load(args.directory, "split_fixed_shards", "post-update"),
            load(args.directory, "split_production_shards", "post-update"),
        ),
    }
    report = {
        "cpu_job": os.environ["SLURM_JOB_ID"],
        "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Saved policy LPs on identical valid response tokens; post-update LPs use the same old rollout, not live LA0 gen KL",
        "checks": checks,
        "files_sha256": files,
    }
    (args.directory / "lp-comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
