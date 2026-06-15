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

"""Convert a Megatron-Bridge LoRA DCP checkpoint to a Hugging Face PEFT adapter.

This script converts the adapter-only policy checkpoint produced by Megatron
LoRA training, for example:

  step_*/policy/weights/iter_0000000

The output directory contains the files expected by PEFT and vLLM LoRA serving:

  adapter_model.safetensors
  adapter_config.json

Megatron-Bridge stores Ultra attention adapters under fused QKV modules:

  decoder.layers.N.self_attention.linear_qkv.adapter.linear_in.weight
  decoder.layers.N.self_attention.linear_qkv.adapter.linear_out.weight

PEFT/vLLM validates adapters against the unfused Hugging Face module names, so
the converter maps those entries to q_proj, k_proj, v_proj, and o_proj LoRA
keys. The fused QKV LoRA-B tensor is split into q, k, and v slices.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def load_dcp_state(dcp_dir: Path) -> dict[str, torch.Tensor]:
    """Load a Torch DCP checkpoint into a flat CPU tensor state dict."""
    if not (dcp_dir / ".metadata").exists():
        raise FileNotFoundError(f"No DCP .metadata found in {dcp_dir}")

    with tempfile.TemporaryDirectory(prefix="megatron_lora_hf_") as tmp:
        tmp_path = Path(tmp) / "adapter.pt"
        dcp_to_torch_save(str(dcp_dir), str(tmp_path))
        state = torch.load(tmp_path, map_location="cpu", weights_only=False)

    if set(state.keys()) == {"model"} and isinstance(state["model"], dict):
        state = state["model"]

    tensors = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    if not tensors:
        raise RuntimeError(f"No tensor entries found in {dcp_dir}")
    return tensors


@dataclass(frozen=True)
class QKVLayout:
    """Attention dimensions needed to split Megatron fused QKV rows."""

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    @property
    def q_per_group(self) -> int:
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads, "
                f"got {self.num_attention_heads} and {self.num_key_value_heads}"
            )
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def fused_dim(self) -> int:
        return self.q_dim + 2 * self.kv_dim


def load_qkv_layout(base_model_name_or_path: str) -> QKVLayout:
    """Load attention dimensions from a Hugging Face-style config.json."""
    if not base_model_name_or_path:
        raise ValueError(
            "--base-model-name-or-path is required to derive Ultra QKV layout"
        )

    config_path = Path(base_model_name_or_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find config.json under {base_model_name_or_path}"
        )

    config = json.loads(config_path.read_text())
    num_attention_heads = config.get("num_attention_heads")
    num_key_value_heads = config.get("num_key_value_heads", num_attention_heads)
    head_dim = config.get("head_dim")
    hidden_size = config.get("hidden_size")

    if num_attention_heads is None:
        raise ValueError(f"num_attention_heads is missing from {config_path}")
    if head_dim is None:
        if hidden_size is None:
            raise ValueError(
                f"Either head_dim or hidden_size must be present in {config_path}"
            )
        head_dim = hidden_size // num_attention_heads
    if hidden_size is not None and head_dim * num_attention_heads != hidden_size:
        raise ValueError(
            f"Inconsistent attention config in {config_path}: "
            f"head_dim={head_dim}, num_attention_heads={num_attention_heads}, "
            f"hidden_size={hidden_size}"
        )

    return QKVLayout(
        num_attention_heads=int(num_attention_heads),
        num_key_value_heads=int(num_key_value_heads),
        head_dim=int(head_dim),
    )


def split_qkv_lora_b(
    tensor: torch.Tensor,
    *,
    layout: QKVLayout,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split fused Megatron QKV LoRA-B weights into HF q/k/v projections."""
    if tensor.shape[0] != layout.fused_dim:
        raise ValueError(
            f"Unexpected fused QKV LoRA-B shape {tuple(tensor.shape)} "
            f"for layout={layout}"
        )

    rank_shape = tensor.shape[1:]
    grouped = tensor.reshape(
        layout.num_key_value_heads,
        layout.q_per_group + 2,
        layout.head_dim,
        *rank_shape,
    )
    q = grouped[:, : layout.q_per_group].reshape(layout.q_dim, *rank_shape)
    k = grouped[:, layout.q_per_group].reshape(layout.kv_dim, *rank_shape)
    v = grouped[:, layout.q_per_group + 1].reshape(layout.kv_dim, *rank_shape)
    return q, k, v


def to_hf_entries(
    key: str,
    tensor: torch.Tensor,
    *,
    qkv_layout: QKVLayout,
) -> dict[str, torch.Tensor]:
    """Map one Megatron-Bridge LoRA tensor to one or more PEFT tensors."""
    prefix = "decoder.layers."
    if not key.startswith(prefix):
        return {}

    rest = key[len(prefix) :]
    layer, _, suffix = rest.partition(".")
    if not layer.isdigit() or not suffix:
        return {}

    base = f"base_model.model.model.layers.{layer}.self_attn"

    if suffix == "self_attention.linear_proj.adapter.linear_in.weight":
        return {f"{base}.o_proj.lora_A.weight": tensor}
    if suffix == "self_attention.linear_proj.adapter.linear_out.weight":
        return {f"{base}.o_proj.lora_B.weight": tensor}

    if suffix == "self_attention.linear_qkv.adapter.linear_in.weight":
        # The same low-rank input projection feeds q, k, and v.
        return {
            f"{base}.q_proj.lora_A.weight": tensor.clone(),
            f"{base}.k_proj.lora_A.weight": tensor.clone(),
            f"{base}.v_proj.lora_A.weight": tensor.clone(),
        }
    if suffix == "self_attention.linear_qkv.adapter.linear_out.weight":
        q, k, v = split_qkv_lora_b(tensor, layout=qkv_layout)
        return {
            f"{base}.q_proj.lora_B.weight": q,
            f"{base}.k_proj.lora_B.weight": k,
            f"{base}.v_proj.lora_B.weight": v,
        }

    return {}


def write_adapter_config(
    output_dir: Path,
    *,
    base_model_name_or_path: str,
    rank: int,
    alpha: int,
) -> None:
    """Write the minimal PEFT adapter config used by vLLM LoRA serving."""
    config = {
        "base_model_name_or_path": base_model_name_or_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "task_type": "CAUSAL_LM",
    }
    (output_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Megatron LoRA DCP checkpoint to a HF PEFT adapter"
    )
    parser.add_argument(
        "--dcp-dir",
        required=True,
        type=Path,
        help="Path to step_*/policy/weights/iter_0000000",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for adapter_model.safetensors and adapter_config.json",
    )
    parser.add_argument(
        "--base-model-name-or-path",
        default="",
        help="Optional base model path to write into adapter_config.json",
    )
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} is not empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state = load_dcp_state(args.dcp_dir)
    qkv_layout = load_qkv_layout(args.base_model_name_or_path)
    hf_state: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, tensor in state.items():
        entries = to_hf_entries(key, tensor, qkv_layout=qkv_layout)
        if not entries:
            skipped.append(key)
            continue
        for hf_key, hf_tensor in entries.items():
            hf_state[hf_key] = hf_tensor.contiguous()

    if not hf_state:
        raise RuntimeError("No LoRA tensors were mapped to HF adapter keys")

    save_file(hf_state, args.output_dir / "adapter_model.safetensors")
    write_adapter_config(
        args.output_dir,
        base_model_name_or_path=args.base_model_name_or_path,
        rank=args.rank,
        alpha=args.alpha,
    )

    print(f"Wrote {len(hf_state)} tensors to {args.output_dir}")
    print(f"Skipped {len(skipped)} non-LoRA/unmapped entries")
    if skipped:
        print("First skipped keys:")
        for key in skipped[:20]:
            print(f"  {key}")


if __name__ == "__main__":
    main()
