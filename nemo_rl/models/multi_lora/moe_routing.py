"""Per-token adapter routing through MoE expert dispatch.

Problem this solves (worklog 2026-07-02): ``MultiLinearLoRA.forward`` needs a
per-row adapter id for every row of its input. Inside MoE expert MLPs the
input rows are a router-selected SUBSET of flattened tokens
(``hidden_states[token_indices]`` in ``NemotronHMOE.moe``), so the
microbatch-level ``[B]`` routing buffer never matches ``x.shape[0]`` and MLL
fell back to slot 0 for ALL rows. Result: slot 0's expert LoRA weights
absorbed gradient from EVERY adapter's data (measured: expert lora_B slot
norms 21.6/3.1/2.9/3.4 after code5), silently breaking adapter isolation
whenever adapters train on different data.

Fix: wrap each MoE module's ``moe(hidden_states, topk_indices, topk_weights)``
method. The wrapper

1. reads the current microbatch's per-ROW ids (``[B]``) stashed on the root
   model by the microbatch routing pre-hook (see ``routing.py``),
2. expands them to per-TOKEN ids (``[B*S]``) via ``repeat_interleave`` —
   valid because ``NemotronHMOE.forward`` flattens ``[B, S, H]`` row-major,
3. recomputes the same one-hot expert mask the stock loop uses (cheap,
   deterministic) to find each expert's ``token_indices``,
4. calls ``set_routing(token_ids[token_indices])`` on every
   ``MultiLinearLoRA`` inside that expert,
5. then calls the ORIGINAL ``moe`` — whose per-expert forwards now see ids
   exactly aligned with their input rows.

The stock modeling loop runs unmodified (no duplication of expert compute).
For the empty-expert dummy path (1 zero-row forward to keep FSDP collectives
symmetric) we set a length-1 id; LoRA(0-input) == 0 so numerics are unchanged.

Activation checkpointing: recompute re-enters the wrapper with identical
inputs and identical stashed row ids (the root-model buffer only changes at
the NEXT microbatch's pre-hook), so recomputed routing is deterministic.

Known limitation: assumes expert LoRA params are replicated or sharded on the
rank dim only (FSDP2 dim-1 sharding). True expert-parallel (EP>1) placement
shards adapter slots across ranks and needs id localization — detect and warn.
"""

from __future__ import annotations

import functools
import logging
import os
import weakref
from typing import Optional

import torch
import torch.nn as nn

from nemo_rl.models.multi_lora.adapter import MultiLinearLoRA

logger = logging.getLogger(__name__)


def _looks_like_moe(mod: nn.Module) -> bool:
    """Duck-type NemotronHMOE-style modules: .experts ModuleList + .gate +
    callable .moe(hidden_states, topk_indices, topk_weights)."""
    return (
        hasattr(mod, "experts")
        and isinstance(getattr(mod, "experts"), nn.ModuleList)
        and hasattr(mod, "gate")
        and callable(getattr(mod, "moe", None))
    )


def _expert_mlls(moe_mod: nn.Module) -> list[tuple[int, list[MultiLinearLoRA]]]:
    out = []
    for i, expert in enumerate(moe_mod.experts):
        mlls = [m for m in expert.modules() if isinstance(m, MultiLinearLoRA)]
        if mlls:
            out.append((i, mlls))
    return out


def _wrap_moe_module(moe_mod: nn.Module, root_model: nn.Module) -> bool:
    """Instance-patch ``moe_mod.moe`` with routing injection. Idempotent."""
    if getattr(moe_mod, "_nousnet_moe_routing_wrapped", False):
        return False
    mll_experts = _expert_mlls(moe_mod)
    if not mll_experts:
        return False  # no multi-LoRA inside this MoE's experts — nothing to do

    orig_moe = moe_mod.moe  # bound method
    model_wr = weakref.ref(root_model)
    n_experts = len(moe_mod.experts)

    @functools.wraps(orig_moe)
    def moe_with_routing(hidden_states, topk_indices, topk_weights):
        model = model_wr()
        chunk: Optional[torch.Tensor] = (
            getattr(model, "_nousnet_current_chunk", None) if model is not None else None
        )
        if chunk is None or chunk.numel() == 0:
            # No routing seeded (eval/generation) — MLLs have no ids and run
            # base-only; nothing to inject.
            return orig_moe(hidden_states, topk_indices, topk_weights)

        n_tok = hidden_states.shape[0]
        B = int(chunk.shape[0])
        if n_tok % B != 0:
            if not getattr(moe_mod, "_nousnet_moe_shape_warned", False):
                logger.warning(
                    "moe token routing: n_tok=%d not divisible by B=%d — "
                    "cannot map tokens to rows; falling back to legacy "
                    "(slot-0) expert behavior for this module.",
                    n_tok, B,
                )
                moe_mod._nousnet_moe_shape_warned = True
            return orig_moe(hidden_states, topk_indices, topk_weights)

        seq = n_tok // B
        token_ids = chunk.to(hidden_states.device).repeat_interleave(seq)  # [B*S]

        # Same mask construction as the stock loop (cheap: one_hot on [N, top_k]).
        expert_mask = torch.nn.functional.one_hot(
            topk_indices, num_classes=n_experts
        ).permute(2, 0, 1)

        for expert_idx, mlls in mll_experts:
            token_indices, _ = torch.where(expert_mask[expert_idx])
            if token_indices.numel() > 0:
                ids = token_ids[token_indices]
            else:
                # Empty-expert dummy path: stock loop feeds 1 zero-row.
                # LoRA on a zero input contributes 0 forward and 0 grad, so
                # the id value is irrelevant — length must match (1).
                ids = token_ids[:1]
            for sub in mlls:
                # Group-by-adapter forward on experts: the per-row bmm
                # gather materializes [rows, dim, in_f] weight copies and
                # saves them for backward -> OOM at 30B scale (smoke run
                # 223170/223171). Grouped F.linear per unique id is
                # mathematically identical and memory-flat.
                sub._nousnet_route_group_mode = True
                sub.set_routing(ids)

        return orig_moe(hidden_states, topk_indices, topk_weights)

    moe_mod.moe = moe_with_routing
    moe_mod._nousnet_moe_routing_wrapped = True
    return True


def install_moe_expert_routing(model: nn.Module) -> int:
    """Wrap every MoE module under ``model`` for per-token expert routing.

    Returns the number of MoE modules wrapped (0 on re-install / non-MoE
    models / when disabled via NOUSNET_DISABLE_MOE_TOKEN_ROUTING=1).
    Safe to call multiple times.
    """
    if os.environ.get("NOUSNET_DISABLE_MOE_TOKEN_ROUTING") == "1":
        if not getattr(model, "_nousnet_moe_routing_disable_logged", False):
            logger.warning(
                "moe expert routing DISABLED via NOUSNET_DISABLE_MOE_TOKEN_ROUTING=1 "
                "— legacy slot-0 expert fallback stays active"
            )
            model._nousnet_moe_routing_disable_logged = True
        return 0
    if getattr(model, "_nousnet_moe_routing_installed", False):
        return 0
    n = 0
    for mod in model.modules():
        if _looks_like_moe(mod):
            # EP>1 detection: adapter slots sharded across ranks break global
            # id indexing. Warn loudly rather than corrupt silently.
            for _, mlls in _expert_mlls(mod):
                p = mlls[0].lora_A
                placements = getattr(p, "placements", None)
                if placements is not None and any(
                    getattr(pl, "dim", None) == 0 for pl in placements
                    if pl.__class__.__name__ == "Shard"
                ):
                    logger.warning(
                        "moe expert routing: lora_A sharded on dim 0 "
                        "(expert-parallel adapter slots). Per-token routing "
                        "with global ids is NOT valid under EP>1 slot "
                        "sharding — review before trusting isolation."
                    )
                break
            if _wrap_moe_module(mod, model):
                n += 1
    model._nousnet_moe_routing_installed = True
    if n:
        logger.info("moe expert routing: wrapped %d MoE modules", n)
        print(f"[NOUSNET] moe expert routing: wrapped {n} MoE modules "
              f"(per-token adapter routing active)", flush=True)
    return n
