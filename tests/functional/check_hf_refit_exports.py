"""Require repeated HF exports and changes to actual model tensors after training."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def read_probe_weights(directory: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    projection_count = 0
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            for name in weights.keys():
                if "norm" in name and name.endswith("weight"):
                    tensors[name] = weights.get_tensor(name)
                elif projection_count < 4 and name.endswith("_proj.weight"):
                    tensors[name] = weights.get_tensor(name)
                    projection_count += 1
    if not tensors:
        raise ValueError(f"No probe weights found in {directory}")
    return tensors


def validate(root: Path, minimum_versions: int) -> dict:
    versions = sorted(path for path in root.glob("version_*") if path.is_dir())
    if len(versions) < minimum_versions:
        raise ValueError(
            f"Expected at least {minimum_versions} exports; found {len(versions)}"
        )
    unfinished = list(root.glob(".version_*.tmp"))
    if unfinished:
        raise ValueError(f"Unfinished checkpoint exports remain: {unfinished}")
    first = read_probe_weights(versions[0])
    last = read_probe_weights(versions[-1])
    if first.keys() != last.keys():
        raise ValueError("Exported model tensor names changed between versions")
    changed = [name for name in first if not torch.equal(first[name], last[name])]
    if not changed:
        raise ValueError("Exported probe weights never changed after training")
    return {
        "versions": len(versions),
        "first": str(versions[0]),
        "last": str(versions[-1]),
        "changed_tensors": changed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--minimum-versions", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.root, args.minimum_versions)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
