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
"""Synthesise a randomly-initialised LiLiCorr drafter for a given target geometry.

For PIPECLEANING the co-training loop, not for quality. The released LiLiCorr drafters
are tied to Super 3.5, whose architecture the Megatron backend does not support (every
3.5-Super recipe in this repo is automodel), so there is no real LiLiCorr checkpoint for
any Megatron-supported target. A random one exercises the same code: strict load, anchor
plan, body forward, reranker, lattice loss, refit export.

Writes under SERVING names, like a released drafter, so the loader's rename runs for
real rather than being bypassed.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    # Target geometry; the body runs at the target's width.
    ap.add_argument("--target-hidden-size", type=int, default=4096)
    ap.add_argument("--target-vocab-size", type=int, default=151936)
    ap.add_argument("--target-num-layers", type=int, default=36)
    # Drafter body
    ap.add_argument("--body-layers", type=int, default=5)
    ap.add_argument("--body-heads", type=int, default=32)
    ap.add_argument("--body-kv-heads", type=int, default=8)
    ap.add_argument("--body-head-dim", type=int, default=128)
    ap.add_argument("--body-intermediate", type=int, default=12288)
    ap.add_argument("--rms-norm-eps", type=float, default=1e-6)
    ap.add_argument("--rope-theta", type=float, default=1_000_000.0)
    # Block / lattice
    ap.add_argument("--block-size", type=int, default=6)
    ap.add_argument("--mask-token-id", type=int, default=151669)
    ap.add_argument("--target-layer-ids", default="1,9,17,25,33")
    # Reranker
    ap.add_argument("--lilicorr-hidden", type=int, default=1024)
    ap.add_argument("--lilicorr-layers", type=int, default=2)
    ap.add_argument("--lilicorr-heads", type=int, default=8)
    ap.add_argument("--lilicorr-mlp-ratio", type=float, default=2.0)
    ap.add_argument("--lilicorr-topk", type=int, default=4)
    ap.add_argument("--lilicorr-factor-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from safetensors.torch import save_file
    from megatron.core.model_parallel_config import ModelParallelConfig

    from nemo_rl.models.megatron.draft.dflash import DFlashBody, DFlashBodyConfig
    from nemo_rl.models.megatron.draft.lilicorr import (
        LiLiCorrHeadConfig,
        build_lilicorr_head,
    )
    from nemo_rl.models.megatron.draft.lilicorr_model import _from_positional_name

    taps = [int(x) for x in args.target_layer_ids.split(",") if x.strip()]
    invalid = [t for t in taps if t >= args.target_num_layers]
    if invalid:
        raise SystemExit(
            f"target layer ids {invalid} exceed the target's {args.target_num_layers} layers"
        )

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_type": "qwen3",
        "architectures": ["DFlashQwen3ForCausalLM"],
        "block_size": args.block_size,
        "hidden_size": args.target_hidden_size,
        "vocab_size": args.target_vocab_size,
        "draft_vocab_size": args.target_vocab_size,
        "num_target_layers": args.target_num_layers,
        "num_hidden_layers": args.body_layers,
        "num_attention_heads": args.body_heads,
        "num_key_value_heads": args.body_kv_heads,
        "head_dim": args.body_head_dim,
        "intermediate_size": args.body_intermediate,
        "rms_norm_eps": args.rms_norm_eps,
        "rope_theta": args.rope_theta,
        "torch_dtype": "bfloat16",
        "dflash_config": {
            "projector_type": "lilicorr",
            "target_layer_ids": taps,
            "mask_token_id": args.mask_token_id,
            "lilicorr_enabled": True,
            "lilicorr_hidden_size": args.lilicorr_hidden,
            "lilicorr_num_layers": args.lilicorr_layers,
            "lilicorr_num_heads": args.lilicorr_heads,
            "lilicorr_mlp_ratio": args.lilicorr_mlp_ratio,
            "lilicorr_candidate_topk": args.lilicorr_topk,
            "lilicorr_factor_dim": args.lilicorr_factor_dim,
            "lilicorr_vector_eps": 1e-8,
            "lilicorr_logit_scale": 1.0,
        },
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    torch.manual_seed(args.seed)
    parallel_config = ModelParallelConfig(
        tensor_model_parallel_size=1,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        sequence_parallel=False,
    )
    body = DFlashBody(
        config=DFlashBodyConfig(
            hidden_size=args.target_hidden_size,
            intermediate_size=args.body_intermediate,
            num_attention_heads=args.body_heads,
            num_key_value_heads=args.body_kv_heads,
            head_dim=args.body_head_dim,
            num_hidden_layers=args.body_layers,
            num_target_taps=len(taps),
            rope_theta=args.rope_theta,
            rms_norm_eps=args.rms_norm_eps,
        ),
        parallel_config=parallel_config,
    )
    head = build_lilicorr_head(LiLiCorrHeadConfig.from_checkpoint_config(cfg))

    tensors = {k: v.detach().clone().to(torch.bfloat16) for k, v in body.state_dict().items()}
    for k, v in head.state_dict().items():
        tensors[f"lilicorr.{_from_positional_name(k)}"] = v.detach().clone().to(torch.bfloat16)
    save_file(tensors, str(out / "model.safetensors"))
    total = sum(t.numel() for t in tensors.values())
    print(f"wrote {len(tensors)} tensors ({total:,} params) to {out}")
    print(f"  block_size={args.block_size} gamma={args.block_size - 1} taps={taps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
