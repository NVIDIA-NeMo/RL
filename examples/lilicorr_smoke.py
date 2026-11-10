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
"""Exercise the LiLiCorr draft path on ONE GPU, without a 120B policy.

The 16-node pipeclean is the real test, but it queues for hours and every failure
costs a full cluster bring-up. Everything specific to this drafter -- strict checkpoint
load, the anchor plan, the body/reranker split, the lattice objective, the export
naming -- needs only a GPU and Megatron's tensor-parallel machinery at TP=1.

Deliberately uses a SMALL geometry rather than the real one: the point is the contract,
not the weights. A mismatch here is a shape or name error, which is exactly what the
static checks could not reach.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import torch


def _init_megatron(tp: int = 1) -> None:
    from megatron.core import parallel_state

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29571")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp)


# Mirrors the real Super 3.5 drafter's SHAPE at small scale: the body and the reranker
# have DIFFERENT head counts (32 vs 8 there), which is exactly the confusion that put the
# reranker's head count into the body config. Keep them different here so the test would
# catch it again.
GEOMETRY = dict(
    model_hidden_size=512,   # target width; the body runs at this
    hidden_size=128,         # reranker width
    num_layers=2,
    num_heads=4,             # RERANKER heads
    mlp_ratio=2.0,
    block_size=4,            # lattice slots = block_size - 1 = 3
    candidate_topk=3,
    factor_dim=16,
    rms_norm_eps=1e-6,
    vector_eps=1e-8,
    logit_scale=1.0,
)
# Body geometry, deliberately distinct from the reranker's above.
BODY_HEADS = 8
BODY_KV_HEADS = 4
BODY_INTERMEDIATE = 1024
BODY_HEAD_DIM = 64
BODY_RMS_EPS = 1e-5      # NOT DFlashBodyConfig's 1e-6 default
NUM_TARGET_LAYERS = 16
TARGET_LAYER_IDS = [1, 5, 9, 13]
MASK_TOKEN_ID = 7
VOCAB = 128


def _write_checkpoint(path: pathlib.Path, dtype: torch.dtype) -> None:
    """Write a drafter checkpoint whose tensors match what the builder will construct.

    Built by instantiating the same modules, so the names and shapes come from the code
    under test rather than from a guess. Saved under SERVING names, which is what a
    released drafter uses -- so the strict loader's rename is exercised too.
    """
    from safetensors.torch import save_file

    from nemo_rl.models.megatron.draft.dflash import DFlashBody, DFlashBodyConfig
    from nemo_rl.models.megatron.draft.lilicorr import (
        LiLiCorrHeadConfig,
        build_lilicorr_head,
    )
    from nemo_rl.models.megatron.draft.lilicorr_model import _from_positional_name
    from megatron.core.model_parallel_config import ModelParallelConfig

    path.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model_type": "qwen3",
        "block_size": GEOMETRY["block_size"],
        "hidden_size": GEOMETRY["model_hidden_size"],
        "vocab_size": VOCAB,
        "num_target_layers": NUM_TARGET_LAYERS,
        "num_hidden_layers": GEOMETRY["num_layers"],
        "num_attention_heads": BODY_HEADS,
        "num_key_value_heads": BODY_KV_HEADS,
        "intermediate_size": BODY_INTERMEDIATE,
        "head_dim": BODY_HEAD_DIM,
        "rms_norm_eps": BODY_RMS_EPS,
        "torch_dtype": "bfloat16",
        "dflash_config": {
            "projector_type": "lilicorr",
            "target_layer_ids": TARGET_LAYER_IDS,
            "mask_token_id": MASK_TOKEN_ID,
            "lilicorr_enabled": True,
            "lilicorr_hidden_size": GEOMETRY["hidden_size"],
            "lilicorr_num_layers": GEOMETRY["num_layers"],
            "lilicorr_num_heads": GEOMETRY["num_heads"],
            "lilicorr_mlp_ratio": GEOMETRY["mlp_ratio"],
            "lilicorr_candidate_topk": GEOMETRY["candidate_topk"],
            "lilicorr_factor_dim": GEOMETRY["factor_dim"],
            "lilicorr_vector_eps": GEOMETRY["vector_eps"],
            "lilicorr_logit_scale": GEOMETRY["logit_scale"],
        },
    }
    (path / "config.json").write_text(json.dumps(cfg, indent=1))

    parallel_config = ModelParallelConfig(
        tensor_model_parallel_size=1,
        use_cpu_initialization=True,
        bf16=(dtype == torch.bfloat16),
        params_dtype=dtype,
        sequence_parallel=False,
    )
    body = DFlashBody(
        config=DFlashBodyConfig(
            hidden_size=GEOMETRY["model_hidden_size"],
            intermediate_size=BODY_INTERMEDIATE,
            num_attention_heads=BODY_HEADS,
            num_key_value_heads=BODY_KV_HEADS,
            head_dim=BODY_HEAD_DIM,
            num_hidden_layers=GEOMETRY["num_layers"],
            num_target_taps=len(TARGET_LAYER_IDS),
            rms_norm_eps=BODY_RMS_EPS,
        ),
        parallel_config=parallel_config,
    )
    head = build_lilicorr_head(LiLiCorrHeadConfig.from_checkpoint_config(cfg))

    tensors = {f"{k}": v.detach().clone().to(dtype) for k, v in body.state_dict().items()}
    for k, v in head.state_dict().items():
        tensors[f"lilicorr.{_from_positional_name(k)}"] = v.detach().clone().to(dtype)
    save_file(tensors, str(path / "model.safetensors"))
    print(f"[smoke] wrote {len(tensors)} tensors to {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/tmp/lilicorr_smoke")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    _init_megatron()
    work = pathlib.Path(args.workdir)
    _write_checkpoint(work / "drafter", dtype)

    from types import SimpleNamespace

    from nemo_rl.models.megatron.draft.block_plan import build_dflash_batch_plan
    from nemo_rl.models.megatron.draft.lilicorr import (
        gold_index_from_candidates,
        lilicorr_lattice_nll,
    )
    from nemo_rl.models.megatron.draft.lilicorr_model import (
        build_lilicorr_draft_model,
        export_lilicorr_weights_to_hf,
    )
    from nemo_rl.models.policy.draft_config import _DRAFT_CONFIG_ADAPTER

    # 1. config resolution, exactly as a recipe delivers it
    draft_cfg = _DRAFT_CONFIG_ADAPTER.validate_python(
        {
            "enabled": True,
            "speculator_type": "lilicorr",
            "model_name": str(work / "drafter"),
            "loss_weight": 0.1,
            "anchors_per_sample": 2,
        }
    )
    assert draft_cfg.gamma == GEOMETRY["block_size"] - 1, draft_cfg.gamma
    assert draft_cfg.mask_token_id == MASK_TOKEN_ID
    assert draft_cfg.target_hidden_state_layer_ids == TARGET_LAYER_IDS
    print(f"[smoke] config: gamma={draft_cfg.gamma} taps={draft_cfg.target_hidden_state_layer_ids}")

    # 2. build + STRICT checkpoint load
    provider = SimpleNamespace(
        tensor_model_parallel_size=1,
        use_cpu_initialization=True,
        fp16=False,
        bf16=(dtype == torch.bfloat16),
        params_dtype=dtype,
    )
    model = build_lilicorr_draft_model(
        model_provider=provider,
        draft_config=draft_cfg,
        pg_collection=SimpleNamespace(tp=None),
        policy_model_chunk=None,
    )
    model = model.cuda().eval()
    print(f"[smoke] built + loaded: {sum(p.numel() for p in model.parameters()):,} params")
    # The body must take ITS geometry from the checkpoint, not DFlashBodyConfig's
    # defaults and not the reranker's. rms_norm_eps is the silent one: wrong value,
    # weights still load, outputs quietly differ.
    bc = model.body.config
    assert bc.num_attention_heads == BODY_HEADS, (bc.num_attention_heads, BODY_HEADS)
    assert bc.num_key_value_heads == BODY_KV_HEADS, bc.num_key_value_heads
    assert bc.intermediate_size == BODY_INTERMEDIATE, bc.intermediate_size
    assert bc.head_dim == BODY_HEAD_DIM, bc.head_dim
    assert abs(bc.rms_norm_eps - BODY_RMS_EPS) < 1e-12, bc.rms_norm_eps
    assert bc.num_target_taps == len(TARGET_LAYER_IDS), bc.num_target_taps
    print(f"[smoke] body geometry from checkpoint: heads={bc.num_attention_heads} "
          f"kv={bc.num_key_value_heads} eps={bc.rms_norm_eps}")

    # 3. anchor plan
    batch, seq = 2, 32
    token_valid = torch.ones(batch, seq, dtype=torch.bool, device="cuda")
    sample_ids = torch.arange(batch, dtype=torch.int64, device="cuda")
    plan = build_dflash_batch_plan(
        token_valid,
        sample_ids,
        anchors_per_sample=draft_cfg.anchors_per_sample,
        gamma=draft_cfg.gamma,
        optimizer_step=0,
        seed=draft_cfg.seed,
    )
    rows = plan.sample_rows.numel()
    print(f"[smoke] plan: {rows} anchor rows, block_size={plan.block_size}")

    # 4. body forward
    # [batch, seq, taps, hidden] -- the layout DFlashBody validates against. The real
    # provider reaches it as hidden_states.unflatten(-1,(taps,hidden)).permute(1,0,2,3),
    # i.e. from Megatron's [seq, batch, taps*hidden]; getting the axis order wrong here
    # is caught by DFlashBody's own check, which is why it validates shapes at all.
    taps = len(TARGET_LAYER_IDS)
    target_taps = torch.randn(
        batch, seq, taps, GEOMETRY["model_hidden_size"], device="cuda", dtype=dtype
    )
    # [num_blocks, block_size, hidden]; block_size == 1 anchor + gamma mask slots.
    assert plan.block_size == 1 + draft_cfg.gamma, (plan.block_size, draft_cfg.gamma)
    block_embeddings = torch.randn(
        rows, plan.block_size, GEOMETRY["model_hidden_size"], device="cuda", dtype=dtype
    )
    block_hidden = model(
        target_taps=target_taps, block_embeddings=block_embeddings, plan=plan
    )
    print(f"[smoke] body out: {tuple(block_hidden.shape)}")
    assert block_hidden.shape[0] == rows
    assert block_hidden.shape[1] == 1 + draft_cfg.gamma, block_hidden.shape

    # 5. reranker over the MASK slots only
    output_weight = torch.randn(VOCAB, GEOMETRY["model_hidden_size"], device="cuda", dtype=dtype)
    embed = torch.nn.Embedding(VOCAB, GEOMETRY["model_hidden_size"]).cuda().to(dtype)
    out = model.score_candidates(
        block_hidden=block_hidden,
        output_weight=output_weight,
        embed_tokens=embed,
        anchor_hidden=block_hidden[:, :1],
        anchor_valid=plan.slot_valid[:, :1],
    )
    print(f"[smoke] student_logits {tuple(out.student_logits.shape)}")
    print(f"[smoke] start_scores   {tuple(out.start_scores.shape)}")
    print(f"[smoke] pair_scores    {tuple(out.pair_scores.shape)}")
    assert out.candidate_ids.shape[-2] == GEOMETRY["block_size"] - 1, out.candidate_ids.shape
    # The head's outputs carry [batch, blocks, ...]; the body's rows are row-major.
    assert out.start_scores.shape[:2] == (1, rows), out.start_scores.shape
    assert out.pair_scores.shape[:3] == (1, rows, GEOMETRY["block_size"] - 2), out.pair_scores.shape

    # 6. lattice objective. Two cases, because they exercise different code.
    #
    # (a) Targets drawn FROM the candidates, so every block's gold path exists. This is
    #     the regime real training is in -- a drafter proposes the right token often --
    #     and it is the only one where the lattice can learn anything.
    # (b) Targets that are absent from the candidates, where the objective must return
    #     exactly zero rather than scoring an arbitrary path.
    #
    # With uniformly random targets over a 128-token vocab and topk=3, (b) is what you
    # get by accident, and a lattice that silently trains on nothing looks identical to
    # one that works.
    chosen = torch.randint(0, out.candidate_ids.shape[-1], out.candidate_ids.shape[:-1])
    target_ids = out.candidate_ids.gather(-1, chosen.unsqueeze(-1).cuda()).squeeze(-1)
    gold = gold_index_from_candidates(
        out.candidate_ids.unsqueeze(0), target_ids.unsqueeze(0)
    )
    assert (gold >= 0).all(), "targets drawn from the candidates must all be found"
    loss = lilicorr_lattice_nll(model.lilicorr, out.start_scores, out.pair_scores, gold)
    loss.backward()
    grads = [p for p in model.lilicorr.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    print(f"[smoke] lattice nll = {float(loss):.6f}; {len(grads)} reranker tensors got gradient")
    assert torch.isfinite(loss), loss
    assert float(loss) > 0.0, f"a real path NLL must be positive, got {float(loss)}"
    assert grads, "lattice loss produced no gradient in the reranker"

    # (b): nothing proposed, nothing to learn, and nothing to blow up on.
    model.zero_grad(set_to_none=True)
    absent = torch.full_like(target_ids, VOCAB - 1)
    absent_gold = gold_index_from_candidates(
        out.candidate_ids.unsqueeze(0), absent.unsqueeze(0)
    )
    if (absent_gold < 0).any():
        zero_loss = lilicorr_lattice_nll(
            model.lilicorr, out.start_scores, out.pair_scores, absent_gold
        )
        assert torch.isfinite(zero_loss), zero_loss
        print(f"[smoke] incomplete blocks -> nll {float(zero_loss):.6f} (expected 0)")

    # 7. export naming
    exported = dict(export_lilicorr_weights_to_hf(model))
    head_names = {k for k in exported if k.startswith("lilicorr.")}
    assert any("feature_norm" in k for k in head_names), sorted(head_names)[:6]
    assert not any(".mlp.0." in k or ".mlp.2." in k for k in head_names), "positional names leaked"
    print(f"[smoke] exported {len(exported)} tensors; head names in serving form")

    print("\n[smoke] PASS -- lilicorr draft path works end to end on one GPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
