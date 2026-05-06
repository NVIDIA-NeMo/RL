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

"""Slice Nemotron-3-Nano-30B-A3B-BF16 down to its first ``MEMEM*`` block.

The full model is far too large for a unit test (~60GB BF16). The Nemotron-H
hybrid layer pattern is::

    MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME

The first segment ``MEMEM*`` covers six layers (Mamba / MoE alternating with
one attention layer ``*``). Keeping just those six layers gives a model that
still exercises every Nemotron module class (Mamba, MoE, attention) under
both colocate and disaggregate refit, while small enough to fit on the test
hosts we use (~8 GPU H100).

This module is invoked from a session-scoped fixture in ``conftest.py``; the
sliced checkpoint is cached at ``${HF_HOME or ~/.cache/huggingface}/
nemotron-3-nano-sliced-MEMEM`` so the heavy work only runs once per host.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

# Source HF repo to slice from.
SOURCE_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# Layer pattern of the slice we keep. ``MEMEM*EMEMEM*E`` = 14 layers (the
# first 14 characters of the Nemotron-3-Nano-30B-A3B hybrid pattern).
#
# Why 14 specifically?
#   1. Sglang's NemotronH model implementation has a corner case where, with
#      PP > 1, **every** PP rank must get at least one attention layer
#      (``*``). Otherwise the weight checker's GPU→CPU copy of the rank's
#      ``embed_tokens.weight`` fails with ``CUDA error: invalid argument``.
#   2. Megatron requires ``num_layers % pp_size == 0`` (or an explicit
#      ``|`` separator in the hybrid layer pattern). 14 splits cleanly 7+7
#      for PP=2 — the only PP value used by our megatron configs.
#   3. The first 14 chars of the upstream pattern are ``MEMEM*EMEMEM*E``,
#      which under a 7+7 PP split puts exactly one ``*`` in each half
#      (rank 0: ``MEMEM*E``; rank 1: ``MEMEM*E`` — same shape).
#
# A 6-layer slice (``MEMEM*``) trips (1); a 13-layer slice (``MEMEM*EMEMEM*``)
# satisfies (1) but trips (2). 14 is the smallest layer count satisfying
# both.
SLICED_PATTERN = "MEMEM*EMEMEM*E"
SLICED_NUM_LAYERS = len(SLICED_PATTERN)

# Cache directory inside the user's HF cache root. The trailing suffix is
# included in the directory name so a future change to ``SLICED_PATTERN``
# auto-creates a fresh cache instead of silently reusing a stale slice.
SLICED_DIR_NAME = f"nemotron-3-nano-sliced-{SLICED_PATTERN.replace('*', 'A')}"


def _hf_cache_root() -> Path:
    """Return the same root that HuggingFace would use, honoring HF_HOME."""
    root = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(root)


def sliced_model_path() -> Path:
    """Absolute path to the sliced checkpoint (whether or not it exists yet).

    Lives under ``${HF_HOME or ~/.cache/huggingface}/hub/`` so it sits next to
    the other HF Hub-cached models (``models--<org>--<name>/...``) instead of
    polluting the parent ``huggingface/`` directory.
    """
    return _hf_cache_root() / "hub" / SLICED_DIR_NAME


# ---------------------------------------------------------------------------
# Layer-key matching
# ---------------------------------------------------------------------------
# Match weight keys that live under a transformer layer indexed by an integer.
# Examples:
#   model.layers.0.self_attn.q_proj.weight       (layer 0)
#   model.layers.42.mlp.experts.7.w1.weight      (layer 42)
#   backbone.layers.5.mixer.A_log                (Mamba style)
#
# Anything that does not match (embeddings, lm_head, final norm, rotary cache)
# is a non-layer tensor and is preserved unchanged.
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _layer_index_for_key(key: str) -> int | None:
    m = _LAYER_RE.search(key)
    return int(m.group(1)) if m else None


def _should_keep_key(key: str, kept_layers: int) -> bool:
    idx = _layer_index_for_key(key)
    return idx is None or idx < kept_layers


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def ensure_sliced_model(force: bool = False) -> Path:
    """Materialize the sliced checkpoint and return its path.

    Idempotent: if the target directory already contains a populated
    ``config.json`` and weight files we assume a previous run produced it.
    Set ``force=True`` to nuke and rebuild.
    """
    out_dir = sliced_model_path()
    if not force and out_dir.is_dir() and (out_dir / "config.json").is_file():
        # Cheap sanity check: at least one safetensors shard.
        if any(out_dir.glob("*.safetensors")):
            return out_dir

    # Heavy imports kept local so importing this module costs nothing.
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pull every file we need from HF (config, tokenizer, safetensors,
    #    safetensors index). This populates the standard HF cache; we then
    #    re-emit a slimmed copy at ``out_dir``.
    src = Path(
        snapshot_download(
            repo_id=SOURCE_MODEL_ID,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.model",
                "*.safetensors",
                "*.py",  # NemotronH ships custom modeling/configuration code
                "tokenizer.*",
                "special_tokens_map.json",
                "generation_config.json",
            ],
        )
    )

    # 2. Patch config.json: shrink to ``SLICED_NUM_LAYERS`` and rewrite the
    #    hybrid layer pattern.
    config_path = src / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    # Both keys exist on Nemotron-H configs in the wild (older checkpoints
    # used ``hybrid_override_pattern``, newer ones ``hybrid_layer_pattern``);
    # rewrite whichever is set so we don't depend on which one HF picked.
    for pattern_key in ("hybrid_override_pattern", "hybrid_layer_pattern"):
        if pattern_key in cfg:
            cfg[pattern_key] = SLICED_PATTERN
    cfg["num_hidden_layers"] = SLICED_NUM_LAYERS
    # Some converters look at this instead.
    if "num_layers" in cfg:
        cfg["num_layers"] = SLICED_NUM_LAYERS

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # 3. Copy non-weight files verbatim (tokenizer, generation_config, ...).
    for item in src.iterdir():
        if item.suffix == ".safetensors" or item.name == "config.json":
            continue
        if item.name.endswith(".safetensors.index.json"):
            continue
        if item.is_file():
            shutil.copy2(item, out_dir / item.name)

    # 4. Walk every safetensors shard and write a pruned copy. We keep tensors
    #    that either don't reference a layer index, or reference layer < N.
    src_index_path = src / "model.safetensors.index.json"
    if src_index_path.is_file():
        with open(src_index_path) as f:
            src_index = json.load(f)
        weight_map: dict[str, str] = src_index.get("weight_map", {})
        # Group by source shard to amortize the open() cost.
        per_shard: dict[str, list[str]] = {}
        for k, shard in weight_map.items():
            per_shard.setdefault(shard, []).append(k)
        new_weight_map: dict[str, str] = {}
        new_total_size = 0
        for shard, keys in per_shard.items():
            keep = [k for k in keys if _should_keep_key(k, SLICED_NUM_LAYERS)]
            if not keep:
                continue
            with safe_open(src / shard, framework="pt") as reader:
                tensors = {k: reader.get_tensor(k) for k in keep}
                metadata = reader.metadata() or {}
            save_file(tensors, str(out_dir / shard), metadata=metadata)
            for k in keep:
                new_weight_map[k] = shard
                new_total_size += tensors[k].numel() * tensors[k].element_size()
            del tensors
        with open(out_dir / "model.safetensors.index.json", "w") as f:
            json.dump(
                {
                    "metadata": {"total_size": new_total_size},
                    "weight_map": new_weight_map,
                },
                f,
                indent=2,
            )
    else:
        # Single-file checkpoint path.
        single = src / "model.safetensors"
        with safe_open(single, framework="pt") as reader:
            keep = [k for k in reader.keys() if _should_keep_key(k, SLICED_NUM_LAYERS)]
            tensors = {k: reader.get_tensor(k) for k in keep}
            metadata = reader.metadata() or {}
        save_file(tensors, str(out_dir / "model.safetensors"), metadata=metadata)

    return out_dir


if __name__ == "__main__":  # pragma: no cover — manual one-shot use
    path = ensure_sliced_model(force="--force" in os.sys.argv)
    print(path)
