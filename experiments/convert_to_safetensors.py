#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file, load_file as st_load_file

def _to_cpu_tensors_only(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            # Ensure on CPU and contiguous for safetensors
            t = v.detach().to("cpu").contiguous()
            out[k] = t
        # non-tensor entries (e.g., metadata) are skipped
    return out

def convert_single_bin(bin_path: Path, out_path: Path, verify: bool = False):
    print(f"[INFO] Loading {bin_path} ...")
    sd = torch.load(bin_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and all(
        isinstance(k, str) for k in sd["state_dict"].keys()
    ):
        # Some training frameworks wrap weights under "state_dict"
        sd = sd["state_dict"]

    sd = _to_cpu_tensors_only(sd)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving safetensors to {out_path} ...")
    save_file(sd, str(out_path))

    if verify:
        print("[INFO] Verifying round-trip (keys & shapes)...")
        st = st_load_file(str(out_path))
        if set(st.keys()) != set(sd.keys()):
            missing = set(sd.keys()) - set(st.keys())
            extra = set(st.keys()) - set(sd.keys())
            raise RuntimeError(f"Key mismatch after conversion. Missing={len(missing)} Extra={len(extra)}")
        for k in sd.keys():
            if tuple(st[k].shape) != tuple(sd[k].shape):
                raise RuntimeError(f"Shape mismatch for {k}: {tuple(st[k].shape)} vs {tuple(sd[k].shape)}")
        print("[INFO] Verification passed.")

def convert_sharded(index_path: Path, verify: bool = False):
    print(f"[INFO] Reading index {index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    weight_map: Dict[str, str] = index.get("weight_map") or index.get("weight_map", {})
    if not weight_map:
        raise ValueError("Index JSON missing 'weight_map'.")

    # Build mapping of shard file -> list of tensor keys
    shards: Dict[str, list] = {}
    for k, shard_file in weight_map.items():
        shards.setdefault(shard_file, []).append(k)

    base_dir = index_path.parent
    new_weight_map: Dict[str, str] = {}

    for shard_file, keys in shards.items():
        src = base_dir / shard_file
        if not src.exists():
            raise FileNotFoundError(f"Shard not found: {src}")

        dst = base_dir / shard_file.replace(".bin", ".safetensors")
        print(f"[INFO] Converting shard {src.name} -> {dst.name} ({len(keys)} tensors)")

        # Load entire shard then keep only the tensors for these keys
        sd = torch.load(src, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = _to_cpu_tensors_only(sd)

        # Keep only tensors that belong to this shard (defensive: intersect by keys)
        sd = {k: v for k, v in sd.items() if k in keys}

        save_file(sd, str(dst))

        # Update new weight map
        for k in keys:
            new_weight_map[k] = dst.name

        if verify:
            st = st_load_file(str(dst))
            missing = set(keys) - set(st.keys())
            if missing:
                raise RuntimeError(f"[VERIFY] Missing {len(missing)} keys in {dst.name}: e.g. {next(iter(missing))}")

    # Write new safetensors index side-by-side
    new_index = dict(index)
    new_index["weight_map"] = new_weight_map
    # Optional: annotate format
    metadata = new_index.get("metadata", {}) or {}
    metadata["format"] = "safetensors"
    new_index["metadata"] = metadata

    new_index_path = base_dir / index_path.name.replace(".bin.index.json", ".safetensors.index.json")
    with open(new_index_path, "w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=2)
    print(f"[INFO] Wrote safetensors index: {new_index_path.name}")

def main():
    ap = argparse.ArgumentParser(description="Convert Hugging Face pytorch_model.bin(.index.json) -> .safetensors")
    ap.add_argument("input", type=str,
                    help="Path to either pytorch_model.bin or pytorch_model.bin.index.json (or any *.bin / *.bin.index.json shard/index).")
    ap.add_argument("-o", "--output", type=str, default=None,
                    help="Output path for single-file conversion (defaults to same name with .safetensors). Ignored for sharded index.")
    ap.add_argument("--verify", action="store_true",
                    help="Verify keys & shapes after conversion.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    # Sharded case: *.bin.index.json
    if in_path.name.endswith(".bin.index.json"):
        convert_sharded(in_path, verify=args.verify)
        return

    # Single-file case: *.bin
    if in_path.suffix == ".bin":
        out_path = Path(args.output) if args.output else in_path.with_suffix(".safetensors")
        convert_single_bin(in_path, out_path, verify=args.verify)
        return

    raise ValueError("Input must be a '.bin' or a '.bin.index.json' file.")

if __name__ == "__main__":
    main()