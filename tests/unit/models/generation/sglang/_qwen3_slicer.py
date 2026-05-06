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

"""Slice ``Qwen/Qwen3-30B-A3B-Instruct-2507`` down to the first N layers.

The full model has 48 transformer layers and 30B parameters (~60 GiB bf16),
which doesn't fit on a single H200 with the Megatron-side full
params+grads+optimizer footprint (~240 GiB total) needed by the test
fixtures' single-GPU variants.

Slicing to ``SLICED_NUM_LAYERS`` keeps the same architecture (Qwen3MoE,
128 experts per layer, 8 active per token) but drops all but the first N
transformer blocks. This:

* Preserves every module class the refit path exercises (attention, MoE
  router, MoE experts).
* Brings the parameter count to roughly ``base + N * per_layer ≈
  embeddings + small`` — small enough for the ``mcore_dp1``/``sgl_tp1``
  parametrization to fit on one GPU.
* Leaves the tokenizer, generation config, and all non-weight files
  untouched.

Idempotent: the sliced checkpoint is cached at
``${HF_HOME or ~/.cache/huggingface}/qwen3-30b-a3b-sliced-N``. Set
``force=True`` to rebuild.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

# Source HF repo to slice from.
SOURCE_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Number of transformer layers to keep (full model has 48).
SLICED_NUM_LAYERS = 4

# Tensors with names like ``model.layers.<idx>.<...>`` carry per-layer
# weights. Anything that doesn't match (embeddings, norm, lm_head,
# rotary buffers) is kept unconditionally.
_LAYER_RE = re.compile(r"\bmodel\.layers\.(\d+)\.")


def sliced_model_path() -> Path:
    """Where the sliced checkpoint lives on disk."""
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return Path(hf_home) / "hub" / f"qwen3-30b-a3b-sliced-{SLICED_NUM_LAYERS}"


def _layer_index_for_key(key: str) -> int | None:
    m = _LAYER_RE.search(key)
    return int(m.group(1)) if m else None


def _should_keep_key(key: str, kept_layers: int) -> bool:
    idx = _layer_index_for_key(key)
    return idx is None or idx < kept_layers


def ensure_sliced_model(force: bool = False) -> Path:
    """Materialize the sliced Qwen3-30B-A3B checkpoint and return its path.

    Idempotent: if the target directory already contains a populated
    ``config.json`` and at least one safetensors shard we assume a
    previous run produced it. Set ``force=True`` to nuke and rebuild.
    """
    out_dir = sliced_model_path()
    if not force and out_dir.is_dir() and (out_dir / "config.json").is_file():
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
    #    index). This populates the standard HF cache; we then re-emit a
    #    slimmed copy at ``out_dir``.
    src = Path(
        snapshot_download(
            repo_id=SOURCE_MODEL_ID,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.model",
                "*.safetensors",
                "tokenizer.*",
                "special_tokens_map.json",
                "generation_config.json",
            ],
        )
    )

    # 2. Patch config.json: shrink ``num_hidden_layers``.
    config_path = src / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["num_hidden_layers"] = SLICED_NUM_LAYERS
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
        single = next(src.glob("*.safetensors"), None)
        if single is None:
            raise RuntimeError(
                f"no safetensors files found under {src}; cannot slice"
            )
        with safe_open(single, framework="pt") as reader:
            keys = list(reader.keys())
            keep = [k for k in keys if _should_keep_key(k, SLICED_NUM_LAYERS)]
            tensors = {k: reader.get_tensor(k) for k in keep}
            metadata = reader.metadata() or {}
        save_file(tensors, str(out_dir / single.name), metadata=metadata)

    return out_dir
