#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Reinitialize near-zero embeddings in a HuggingFace checkpoint.

During pretraining with weight decay, tokens that are never seen in the training
data have their embeddings continuously pushed toward zero by the optimizer. By
the end of pretraining these unused-token rows can be effectively zero. When such
a checkpoint is then used for fine-tuning on a dataset where those tokens do
appear, the zero embeddings cause gradient spikes because the model must learn
the embedding from scratch under a large loss signal.

This script detects those near-zero rows and reinitializes them from N(0, sigma)
where sigma is estimated from the healthy (non-zero) embedding rows in the same
tensor, matching the scale of the surrounding pretrained embeddings.

The script handles sharded safetensors checkpoints and only touches the files
that contain embedding weights; all other files are either symlinked or copied
unchanged.

Usage:
    # Inspect only (no writes):
    uv run tools/reinit_hf_embeddings.py \
        --input  /path/to/hf_checkpoint \
        --dry-run

    # Write patched checkpoint to a new directory:
    uv run tools/reinit_hf_embeddings.py \
        --input  /path/to/hf_checkpoint \
        --output /path/to/patched_checkpoint

    # Patch in-place (overwrites the source shard files):
    uv run tools/reinit_hf_embeddings.py \
        --input  /path/to/hf_checkpoint \
        --in-place

    # Use a custom norm threshold and seed:
    uv run tools/reinit_hf_embeddings.py \
        --input  /path/to/hf_checkpoint \
        --output /path/to/patched_checkpoint \
        --threshold 1e-2 --seed 42

    # Inspect embedding health without modifying anything:
    uv run tools/reinit_hf_embeddings.py \
        --input  /path/to/hf_checkpoint \
        --stats-only

    # Pass a Hub model ID directly (downloads to the HF cache automatically):
    uv run tools/reinit_hf_embeddings.py \
        --input  nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16 \
        --output /path/to/patched_checkpoint
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Optional safetensors import – preferred for large sharded models
# ---------------------------------------------------------------------------
try:
    from safetensors import safe_open
    from safetensors.torch import save_file as safetensors_save

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print(
        "WARNING: safetensors not available; falling back to torch.load / torch.save "
        "which is slower and does not support memory-mapped loading of individual tensors.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBEDDING_KEY_CANDIDATES = [
    # Transformer-style (Llama, Mistral, Nemotron, …)
    "model.embed_tokens.weight",
    # NemotronH (Mamba-based)
    "backbone.embeddings.weight",
    # GPT-NeoX / Falcon
    "gpt_neox.embed_in.weight",
    # BLOOM / OPT
    "model.decoder.embed_tokens.weight",
    # Phi
    "model.embed.weight",
    # Generic fallback checked last
    "transformer.wte.weight",
]


def format_index_ranges(indices: list[int]) -> str:
    """Format a list of indices into compact range strings like '0-1,3-6'."""
    if not indices:
        return "(none)"
    ranges: list[str] = []
    start = end = indices[0]
    for i in range(1, len(indices)):
        if indices[i] == end + 1:
            end = indices[i]
        else:
            ranges.append(str(start) if start == end else f"{start}-{end}")
            start = end = indices[i]
    ranges.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(ranges)


def describe_token(tokenizer, idx: int) -> str:
    if tokenizer is None:
        return "?"
    try:
        return repr(tokenizer.decode([idx]))
    except Exception:
        return "?"


def load_embedding_tensor_from_shard(shard_path: Path, key: str) -> Optional[torch.Tensor]:
    """Load a single tensor from a safetensors shard without loading the whole file."""
    if not HAS_SAFETENSORS:
        return None
    try:
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            if key in f.keys():
                return f.get_tensor(key)
    except Exception:
        pass
    return None


def find_key_in_shards(
    index: dict, key: str
) -> Optional[str]:
    """Return the shard filename that contains *key*, or None."""
    return index.get("weight_map", {}).get(key)


# ---------------------------------------------------------------------------
# Core reinit logic
# ---------------------------------------------------------------------------

def compute_healthy_std(weights: torch.Tensor, zero_indices: list[int], threshold: float) -> float:
    """Estimate std from rows whose norm is >= threshold (i.e. the healthy rows)."""
    norms = weights.float().norm(dim=1)
    healthy_mask = norms >= threshold
    if healthy_mask.sum() == 0:
        # Extreme fallback: use the global std of the whole matrix
        return weights.float().std().item()
    healthy = weights[healthy_mask].float()
    return healthy.std().item()


def reinit_near_zero_rows(
    weights: torch.Tensor,
    threshold: float,
    seed: int,
    key: str,
    tokenizer=None,
) -> tuple[torch.Tensor, list[int]]:
    """
    Identify rows with L2-norm < threshold and replace them with random samples
    drawn from N(0, sigma) where sigma = std of healthy rows.

    Returns (patched_weights, list_of_reinitialized_indices).
    """
    orig_dtype = weights.dtype
    wf = weights.float()
    norms = wf.norm(dim=1)

    zero_mask = norms < threshold
    zero_indices = zero_mask.nonzero(as_tuple=True)[0].tolist()

    if not zero_indices:
        print(f"  [{key}] No near-zero rows found (threshold={threshold:.2e}). Nothing to do.")
        return weights, []

    sigma = compute_healthy_std(wf, zero_indices, threshold)
    print(f"  [{key}] Found {len(zero_indices)} near-zero row(s). "
          f"Reinitializing with N(0, sigma={sigma:.6f})")

    rng = torch.Generator()
    rng.manual_seed(seed)

    patched = wf.clone()
    for idx in zero_indices:
        token_repr = describe_token(tokenizer, idx)
        new_row = torch.empty(wf.shape[1]).normal_(mean=0.0, std=sigma, generator=rng)
        patched[idx] = new_row
        print(f"    ID {idx:6d}: {token_repr}  "
              f"norm_before={norms[idx]:.4e}  norm_after={new_row.norm():.4e}")

    # Restore original dtype
    return patched.to(orig_dtype), zero_indices


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def load_safetensors_index(ckpt_dir: Path) -> Optional[dict]:
    for name in ["model.safetensors.index.json", "pytorch_model.bin.index.json"]:
        p = ckpt_dir / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def patch_safetensors_checkpoint(
    input_dir: Path,
    output_dir: Optional[Path],
    in_place: bool,
    threshold: float,
    seed: int,
    tokenizer=None,
    dry_run: bool = False,
) -> None:
    index = load_safetensors_index(input_dir)
    weight_map = index.get("weight_map", {}) if index else {}

    # Discover the input-embedding key present in this checkpoint
    keys_to_patch: dict[str, str] = {}  # key -> shard filename
    for key in EMBEDDING_KEY_CANDIDATES:
        if key in weight_map:
            keys_to_patch[key] = weight_map[key]
            break  # only one input-embedding key expected

    if not keys_to_patch:
        # Single-file checkpoint (model.safetensors) – no index
        single = input_dir / "model.safetensors"
        if single.exists():
            keys_to_patch = {}  # will be detected per-tensor below
            weight_map = {}     # signal single-file mode
            print("  Single-file safetensors checkpoint detected.")
        else:
            raise FileNotFoundError(
                f"No safetensors index or model.safetensors found in {input_dir}"
            )

    # -------------------------------------------------------------------
    # Group keys by the shard that contains them so we open each shard
    # only once.
    # -------------------------------------------------------------------
    shard_to_keys: dict[str, list[str]] = {}
    if weight_map:
        for key, shard in keys_to_patch.items():
            shard_to_keys.setdefault(shard, []).append(key)
    else:
        # Single-file: load all keys from model.safetensors
        shard_to_keys["model.safetensors"] = list(keys_to_patch.keys()) or _discover_embedding_keys(input_dir / "model.safetensors")

    # -------------------------------------------------------------------
    # Process each relevant shard
    # -------------------------------------------------------------------
    patched_shards: set[str] = set()

    for shard_name, keys in shard_to_keys.items():
        shard_path = input_dir / shard_name
        print(f"\nProcessing shard: {shard_name}")

        # Load ALL tensors from the shard (needed to re-save correctly)
        all_tensors: dict[str, torch.Tensor] = {}
        metadata: dict[str, str] = {}
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for k in f.keys():
                all_tensors[k] = f.get_tensor(k)
            metadata = f.metadata() or {}

        modified = False
        for key in keys:
            if key not in all_tensors:
                # Key wasn't actually in this shard – skip gracefully
                continue
            print(f"\n  Checking key: {key}  shape={tuple(all_tensors[key].shape)}")
            patched, changed_ids = reinit_near_zero_rows(
                all_tensors[key], threshold=threshold, seed=seed,
                key=key, tokenizer=tokenizer
            )
            if changed_ids:
                all_tensors[key] = patched
                modified = True

        if not modified:
            print(f"  No changes needed in {shard_name}.")
            continue

        patched_shards.add(shard_name)
        if dry_run:
            print(f"  [DRY-RUN] Would write patched {shard_name}")
            continue

        # Determine output path
        if in_place:
            out_shard = shard_path
        else:
            out_shard = output_dir / shard_name
            out_shard.parent.mkdir(parents=True, exist_ok=True)

        print(f"  Writing patched shard: {out_shard}")
        safetensors_save(all_tensors, str(out_shard), metadata=metadata)

    # -------------------------------------------------------------------
    # Copy remaining files unchanged (or symlink for non-in-place)
    # -------------------------------------------------------------------
    if not in_place and not dry_run and output_dir is not None:
        print(f"\nCopying remaining files to {output_dir} ...")
        for src in input_dir.iterdir():
            dst = output_dir / src.name
            if dst.exists():
                continue  # already written above
            if src.is_file():
                shutil.copy2(src, dst)
                print(f"  Copied: {src.name}")

    if dry_run:
        print(f"\n[DRY-RUN] Would have patched shards: {sorted(patched_shards)}")
    else:
        if patched_shards:
            print(f"\nPatched shards: {sorted(patched_shards)}")
        print("Done.")


def _discover_embedding_keys(shard_path: Path) -> list[str]:
    """Return embedding keys found in a single safetensors file."""
    found = []
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
    for c in EMBEDDING_KEY_CANDIDATES:
        if c in keys:
            found.append(c)
    return found


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def print_embedding_stats(
    weights: torch.Tensor,
    key: str,
    threshold: float,
    tokenizer=None,
    identical_threshold: float = 1e-8,
) -> dict:
    wf = weights.float()
    norms = wf.norm(dim=1)
    zero_mask = norms < threshold
    zero_indices = zero_mask.nonzero(as_tuple=True)[0].tolist()

    # Identical embeddings: rows whose std dev is below the identical_threshold
    row_stds = wf.std(dim=1)
    identical_mask = row_stds < identical_threshold
    identical_indices = identical_mask.nonzero(as_tuple=True)[0].tolist()

    total = weights.shape[0]

    print(f"\n=== {key} ===")
    print(f"  Shape  : {tuple(weights.shape)}")
    print(f"  Dtype  : {weights.dtype}")
    print(f"  Norm   : min={norms.min():.4e}  mean={norms.mean():.4e}  max={norms.max():.4e}")
    print(f"  Stats  : mean_abs={wf.abs().mean():.6f}  max_abs={wf.abs().max():.6f}"
          f"  std_range=[{row_stds.min():.6f}, {row_stds.max():.6f}]")
    print(f"  Near-zero rows (norm < {threshold:.2e}): {len(zero_indices)}/{total}"
          f" ({100 * len(zero_indices) / total:.1f}%)")
    if zero_indices:
        print(f"  Indices: {format_index_ranges(zero_indices)}")
        print("  Tokens :")
        for i in zero_indices[:30]:
            emb = wf[i]
            first_two = emb[:2].tolist()
            last_two = emb[-2:].tolist()
            print(f"    ID {i:6d}: {describe_token(tokenizer, i)}  norm={norms[i]:.4e}"
                  f"  values=[{first_two[0]:.2e},{first_two[1]:.2e},...,"
                  f"{last_two[0]:.2e},{last_two[1]:.2e}]")
        if len(zero_indices) > 30:
            print(f"    ... and {len(zero_indices) - 30} more")

    print(f"  Identical rows (std < {identical_threshold:.2e}): {len(identical_indices)}/{total}"
          f" ({100 * len(identical_indices) / total:.1f}%)")
    if identical_indices:
        print(f"  Indices: {format_index_ranges(identical_indices)}")
        print("  Tokens :")
        for i in identical_indices[:30]:
            emb = wf[i]
            first_two = emb[:2].tolist()
            last_two = emb[-2:].tolist()
            print(f"    ID {i:6d}: {describe_token(tokenizer, i)}  std={row_stds[i]:.4e}"
                  f"  values=[{first_two[0]:.2e},{first_two[1]:.2e},...,"
                  f"{last_two[0]:.2e},{last_two[1]:.2e}]")
        if len(identical_indices) > 30:
            print(f"    ... and {len(identical_indices) - 30} more")

    issues = []
    if zero_indices:
        issues.append(f"{len(zero_indices)} near-zero embeddings")
    if identical_indices:
        issues.append(f"{len(identical_indices)} identical embeddings")

    return {
        "key": key,
        "shape": tuple(weights.shape),
        "dtype": weights.dtype,
        "total": total,
        "num_near_zero": len(zero_indices),
        "near_zero_indices": zero_indices,
        "num_identical": len(identical_indices),
        "identical_indices": identical_indices,
        "near_zero_threshold": threshold,
        "identical_threshold": identical_threshold,
        "mean_abs": wf.abs().mean().item(),
        "max_abs": wf.abs().max().item(),
        "min_std": row_stds.min().item(),
        "max_std": row_stds.max().item(),
        "issues": issues,
    }


def run_stats(
    input_dir: Path,
    threshold: float,
    tokenizer=None,
    identical_threshold: float = 1e-8,
) -> None:
    """Print embedding health statistics without modifying anything."""
    index = load_safetensors_index(input_dir)
    weight_map = index.get("weight_map", {}) if index else {}

    summaries = []
    for key in EMBEDDING_KEY_CANDIDATES:
        shard_name = weight_map.get(key)
        if shard_name is None:
            # Try single-file
            shard_name = "model.safetensors"
        shard_path = input_dir / shard_name
        if not shard_path.exists():
            continue
        tensor = load_embedding_tensor_from_shard(shard_path, key)
        if tensor is not None:
            summary = print_embedding_stats(
                tensor, key, threshold, tokenizer, identical_threshold
            )
            summaries.append(summary)

    if summaries:
        print("\n" + "=" * 80)
        print("EMBEDDING SUMMARIES")
        print("=" * 80)
        for s in summaries:
            print(f"\n--- {s['key']} ---")
            print(f"Shape: {s['shape']}, Dtype: {s['dtype']}")
            print(
                f"Near-zero (norm < {s['near_zero_threshold']:.2e}): "
                f"{s['num_near_zero']}/{s['total']} "
                f"({100 * s['num_near_zero'] / s['total']:.1f}%)"
            )
            if s["near_zero_indices"]:
                print(f"  Indices: {format_index_ranges(s['near_zero_indices'])}")
            print(
                f"Identical (std < {s['identical_threshold']:.2e}): "
                f"{s['num_identical']}/{s['total']} "
                f"({100 * s['num_identical'] / s['total']:.1f}%)"
            )
            if s["identical_indices"]:
                print(f"  Indices: {format_index_ranges(s['identical_indices'])}")
            print(
                f"Statistics: mean_abs={s['mean_abs']:.6f}  max_abs={s['max_abs']:.6f}"
                f"  std_range=[{s['min_std']:.6f}, {s['max_std']:.6f}]"
            )
            if s["issues"]:
                print(f"WARNING - POTENTIAL ISSUES: {', '.join(s['issues'])}")
            else:
                print("OK - No obvious untrained patterns detected")

        print(f"\n=== Final Summary ===")
        print(f"Checkpoint: {input_dir}")
        print("Analysis complete.")


# ---------------------------------------------------------------------------
# Hub resolution
# ---------------------------------------------------------------------------

def resolve_input(input_str: str) -> Path:
    """Return a local Path for *input_str*.

    If *input_str* is an existing local directory it is returned as-is.
    Otherwise it is treated as a HuggingFace Hub model ID and downloaded to
    the local HF cache via ``snapshot_download``.
    """
    local = Path(input_str)
    if local.is_dir():
        return local.resolve()

    # Not a local path – try Hub download.
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "ERROR: huggingface_hub is required to download Hub models. "
            "Install it with:\n  pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"'{input_str}' is not a local directory – downloading from HuggingFace Hub ...")
    local_dir = snapshot_download(repo_id=input_str)
    print(f"Downloaded to: {local_dir}")
    return Path(local_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reinitialize near-zero embedding rows in a HuggingFace checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        help="Local path to a HuggingFace checkpoint directory, or a Hub model ID "
             "(e.g. nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16). "
             "Hub model IDs are downloaded to the local HF cache automatically.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output directory for the patched checkpoint. "
             "Required unless --in-place or --dry-run is set.",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input checkpoint shard files in place. "
             "Make a backup first!",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed but do not write anything.",
    )
    p.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print embedding statistics; do not reinitialize anything.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="L2-norm threshold below which a row is considered near-zero.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for reinitialization.",
    )
    p.add_argument(
        "--identical-threshold",
        type=float,
        default=1e-8,
        help="Std-dev threshold below which a row is considered identical/constant "
             "(reported in --stats-only mode).",
    )
    p.add_argument(
        "--tokenizer",
        default=None,
        help="HuggingFace model name or local path to use as the tokenizer. "
             "Defaults to --input (checkpoint directories already bundle tokenizer files). "
             "Only needed if you want to use a different tokenizer than the one in the checkpoint.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not HAS_SAFETENSORS:
        print(
            "ERROR: safetensors is required. Install it with:\n"
            "  pip install safetensors",
            file=sys.stderr,
        )
        sys.exit(1)

    input_dir = resolve_input(args.input)

    if args.in_place and not Path(args.input).is_dir():
        print(
            "ERROR: --in-place cannot be used with a Hub model ID – that would modify "
            "the shared HF cache. Download the model to a local directory first and "
            "then re-run with --in-place.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.in_place and not args.dry_run and not args.stats_only and args.output is None:
        print(
            "ERROR: Specify --output <dir>, --in-place, --dry-run, or --stats-only.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir: Optional[Path] = None
    if args.output:
        output_dir = Path(args.output).resolve()
        if output_dir == input_dir:
            print(
                "ERROR: --output must differ from --input. Use --in-place to overwrite.",
                file=sys.stderr,
            )
            sys.exit(1)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer for human-readable token names.
    # Default: load from the checkpoint directory itself (HF checkpoints bundle tokenizer files).
    # Override with --tokenizer if you want a different tokenizer.
    tokenizer_source = args.tokenizer or str(input_dir)
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer from {tokenizer_source} ...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"WARNING: Could not load tokenizer ({e}). Token IDs will not be decoded.")

    print(f"\nCheckpoint : {input_dir}")
    print(f"Threshold  : {args.threshold:.2e}")
    print(f"Seed       : {args.seed}")

    if args.stats_only:
        run_stats(input_dir, args.threshold, tokenizer, identical_threshold=args.identical_threshold)
        return

    patch_safetensors_checkpoint(
        input_dir=input_dir,
        output_dir=output_dir,
        in_place=args.in_place,
        threshold=args.threshold,
        seed=args.seed,
        tokenizer=tokenizer,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
