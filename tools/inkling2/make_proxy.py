# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Slice the released Inkling2 checkpoint into a small real-format proxy.

Keeps the release's storage formats (NVFP4 + MXFP8 packed experts, FP32 groups,
BF16 dense) so refit/serving paths are exercised exactly, at 1-node scale:
  layers  [0..5] (NVFP4; layer 5 is a global-attention layer) + [46, 47] (MXFP8)
  experts first N of 384 (router weight/bias rows sliced to match)
Weights are not a meaningful model; this checks plumbing and train/serve agreement.

usage: make_proxy.py SRC DST [N_EXPERTS]
"""
import json
import re
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
n_experts = int(sys.argv[3]) if len(sys.argv) > 3 else 96
LAYERS = [0, 1, 2, 3, 4, 5, 46, 47]
new_of = {old: new for new, old in enumerate(LAYERS)}
dst.mkdir(parents=True, exist_ok=False)

config = json.loads((src / "config.json").read_text())
text = config["text_config"]
text["num_hidden_layers"] = len(LAYERS)
text["local_layer_ids"] = [new_of[i] for i in LAYERS if i in text["local_layer_ids"]]
text["n_routed_experts"] = n_experts
config.pop("mtp_config", None)
(dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def rename(name: str) -> str | None:
    """Source name -> proxy name, or None when the tensor is dropped."""
    if m := re.fullmatch(r"model\.llm\.layers\.(\d+)\.(.+)", name):
        old = int(m[1])
        return f"model.llm.layers.{new_of[old]}.{m[2]}" if old in new_of else None
    if m := re.fullmatch(r"model\.llm\.attn_res\.attn_queries\.(\d+)", name):
        # attn_queries[i] belongs to layer i + 1.
        layer = int(m[1]) + 1
        return f"model.llm.attn_res.attn_queries.{new_of[layer] - 1}" if layer in new_of and new_of[layer] > 0 else None
    if m := re.fullmatch(r"model\.llm\.attn_res\.mlp_queries\.(\d+)", name):
        layer = int(m[1])
        return f"model.llm.attn_res.mlp_queries.{new_of[layer]}" if layer in new_of else None
    return name


def expert_sliced(name: str) -> bool:
    return bool(re.search(r"\.mlp\.(experts\.w(13|2)_weight(_scale|_amax)?|gate\.(weight|bias))$", name))


quant = json.loads((src / "hf_quant_config.json").read_text())
quant.pop("mtp", None)
for group in quant["model"]["parameter_groups"]:
    group["tensors"] = sorted(n for n in map(rename, group["tensors"]) if n is not None)
(dst / "hf_quant_config.json").write_text(json.dumps(quant, indent=2) + "\n")

index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
by_file: dict[str, list[str]] = {}
for name, file in index.items():
    if rename(name) is not None:
        by_file.setdefault(file, []).append(name)

weight_map, shard, shard_bytes, shard_id = {}, {}, 0, 0


def flush() -> None:
    global shard, shard_bytes, shard_id
    if shard:
        shard_id += 1
        fname = f"model-{shard_id:05d}.safetensors"
        save_file(shard, str(dst / fname), metadata={"format": "pt"})
        weight_map.update({k: fname for k in shard})
        print(f"wrote {fname}: {len(shard)} tensors, {shard_bytes / 2**30:.1f} GiB", flush=True)
    shard, shard_bytes = {}, 0


for file in sorted(by_file):
    with safe_open(str(src / file), "pt", device="cpu") as f:
        for name in by_file[file]:
            sl = f.get_slice(name)
            tensor = sl[:n_experts] if expert_sliced(name) else sl[:]
            new = rename(name)
            shard[new] = tensor.contiguous()
            shard_bytes += tensor.numel() * tensor.element_size()
            if shard_bytes > 8 * 2**30:
                flush()
flush()

expected = {n for g in quant["model"]["parameter_groups"] for n in g["tensors"]}
missing_meta = {n for n in weight_map if not n.endswith(("_scale", "_amax", "_outer_scale_denominator"))} - expected
assert not missing_meta, f"tensors without precision entry: {sorted(missing_meta)[:5]}"
(dst / "model.safetensors.index.json").write_text(
    json.dumps({"metadata": {}, "weight_map": dict(sorted(weight_map.items()))}, indent=2) + "\n"
)
for extra in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "image_preprocessing.json"):
    shutil.copy2(src / extra, dst / extra)
shutil.copytree(src / "tiktoken", dst / "tiktoken")
print("done:", dst, "tensors:", len(weight_map), "layers:", LAYERS, "experts:", n_experts, "local:", text["local_layer_ids"])
