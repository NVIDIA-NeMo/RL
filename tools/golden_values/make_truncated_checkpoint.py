#!/usr/bin/env python3
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
"""Build a small random-weight HF checkpoint with a real model's architecture.

Copies a checkpoint's config, remote code and tokenizer (never its weights),
edits the config with --set, builds the model through Megatron-Bridge with
random weights, and saves them through the bridge's own HF export. Going
through the bridge keeps tensor names and layouts exactly what the bridge loads,
which the model's HF modeling code does not always reproduce (for example,
Nemotron 3.5 Super VL's MTP block). Golden-value tests use the result to run a
production architecture on two GPUs. Needs one GPU:

    torchrun --nproc-per-node 1 tools/golden_values/make_truncated_checkpoint.py \
        --hf-model nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 --out /tmp/super-vl-tiny \
        --set 'llm_config.layers_block_type=["mamba","moe","attention","moe"]' \
        --set llm_config.n_routed_experts=8 --set llm_config.num_experts_per_tok=2
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

# Everything in a checkpoint directory except the weights.
_NON_WEIGHT_PATTERNS = ["*.json", "*.py", "*.jinja", "*.txt", "*.model", "*.tiktoken"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--hf-model", required=True, help="HF checkpoint path or hub id"
    )
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="DOTTED.KEY=JSON",
        help="Config edit; the value is parsed as JSON (repeatable)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _source_dir(hf_model: str) -> Path:
    if Path(hf_model).is_dir():
        return Path(hf_model)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(hf_model, allow_patterns=_NON_WEIGHT_PATTERNS))


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    *parents, leaf = dotted_key.split(".")
    node = config
    for key in parents:
        if not isinstance(node.get(key), dict):
            raise KeyError(f"{dotted_key}: {key!r} is not a config section")
        node = node[key]
    if leaf not in node:
        raise KeyError(f"{dotted_key}: no such config key")
    node[leaf] = value


def main() -> None:
    args = _parse_args()
    source, out = _source_dir(args.hf_model), Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for pattern in _NON_WEIGHT_PATTERNS:
        for path in source.glob(pattern):
            if path.name != "model.safetensors.index.json":
                shutil.copy2(path, out / path.name)

    config = json.loads((out / "config.json").read_text())
    for item in args.set:
        key, _, value = item.partition("=")
        _set_dotted(config, key, json.loads(value))
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    from megatron.bridge import AutoBridge
    from safetensors.torch import save_file
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(out, trust_remote_code=True)
    # Some bridges copy source-only buffers (for example RADIO's input
    # normalization) from the checkpoint the config names. Name the source when
    # it has weights; otherwise an empty checkpoint, so they are skipped.
    empty_source = out / "_empty_source"
    if any(source.glob("*.safetensors")):
        hf_config._name_or_path = str(source)
    else:
        empty_source.mkdir()
        save_file({}, empty_source / "model.safetensors")
        hf_config._name_or_path = str(empty_source)
    bridge = AutoBridge.from_hf_config(hf_config)
    provider = bridge.to_megatron_provider(load_weights=False)
    # Needs APEX's fused weight-gradient extension, and nothing here trains.
    provider.gradient_accumulation_fusion = False
    provider.finalize()
    provider.initialize_model_parallel(seed=args.seed)
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    bridge.save_hf_weights(model, out, show_progress=False)
    shutil.rmtree(empty_source, ignore_errors=True)
    num_parameters = sum(
        parameter.numel() for chunk in model for parameter in chunk.parameters()
    )
    print(f"Wrote {out}: {num_parameters / 1e6:.1f}M parameters")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
