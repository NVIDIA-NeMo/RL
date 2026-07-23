"""Per-microbatch routing for :class:`MultiLinearLoRA` (MLL) layers.

NeMo-RL's train loop iterates over microbatches, calling
``model.forward(...)`` once per microbatch. Each microbatch is a slice
``[mb_start : mb_start + mb_size]`` of the global batch. Routing must
push the matching slice of per-row adapter ids to every MLL layer
before each forward.

Contract:

- The caller (worker ``train``) calls :func:`seed_microbatch_routing` once
  per ``policy.train(...)`` invocation, passing the FULL batch's per-row
  adapter ids as a ``LongTensor[B]`` indexed into the canonical adapter
  slot order (see :class:`MultiLinearLoRA`'s ``lora_A[ids]`` semantics).

- The pre-hook installed on ``model`` then advances a cursor by
  ``input_ids.shape[0]`` each forward, slices the next chunk, and calls
  ``sub.set_routing(chunk_ids)`` on every MLL submodule.

Ids are GLOBAL — i.e., the same int means the same adapter across every
forward in the step. No local renumbering, no name→id map.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from nemo_rl.models.multi_lora.adapter import MultiLinearLoRA

logger = logging.getLogger(__name__)


def set_microbatch_routing_full(model: nn.Module, adapter_ids: torch.Tensor) -> None:
    """Seed the whole-batch per-row adapter-id buffer + reset the mb cursor.

    Call this once per ``policy.train(...)`` invocation. The
    :func:`install_microbatch_routing_hook` pre-hook will then advance the
    cursor by ``input_ids.shape[0]`` on each forward, reading the next
    chunk of ids out of this tensor and pushing it to every MLL submodule.
    """
    if not (isinstance(adapter_ids, torch.Tensor) and adapter_ids.dtype == torch.long):
        raise TypeError(
            f"adapter_ids must be a LongTensor; got {type(adapter_ids).__name__}"
        )
    if adapter_ids.dim() != 1:
        raise ValueError(
            f"adapter_ids must be 1-D [B]; got shape {tuple(adapter_ids.shape)}"
        )
    # Keep on CPU; per-forward slicing then .to(device) happens in the hook
    # so we don't carry a device assumption from the caller.
    model._nousnet_routing_full = adapter_ids.detach().cpu()
    model._nousnet_mb_cursor = 0


def clear_microbatch_routing(model: nn.Module) -> None:
    model._nousnet_routing_full = None
    model._nousnet_mb_cursor = 0
    model._nousnet_current_chunk = None
    for sub in model.modules():
        if isinstance(sub, MultiLinearLoRA):
            sub.clear_routing()


def install_microbatch_routing_hook(model: nn.Module):
    """Install a forward pre-hook that pushes per-microbatch routing into
    every :class:`MultiLinearLoRA` submodule.

    Reads ``model._nousnet_routing_full`` (``LongTensor[B]`` for the WHOLE
    batch) and advances ``model._nousnet_mb_cursor`` by the current
    forward's leading dim. Idempotent: re-installs are no-ops.
    """
    existing = getattr(model, "_nousnet_mb_routing_handle", None)
    if existing is not None:
        return existing

    def _hook(mod: nn.Module, args, kwargs):  # noqa: ARG001
        full: Optional[torch.Tensor] = getattr(mod, "_nousnet_routing_full", None)
        if full is None:
            mod._nousnet_current_chunk = None
            for sub in mod.modules():
                if isinstance(sub, MultiLinearLoRA):
                    sub.clear_routing()
            return None

        # Determine the current microbatch's row count from the input.
        # NeMo-RL workers call ``self.model(**model_args)`` with kwargs only;
        # tests call ``model(x)`` with a positional tensor. Handle both.
        first = args[0] if args else None
        if first is None:
            for cand_key in ("input_ids", "inputs_embeds"):
                if cand_key in kwargs:
                    first = kwargs[cand_key]
                    break
        if first is None:
            return None
        if isinstance(first, torch.Tensor):
            mb_size = first.shape[0]
        else:
            return None

        L = int(full.shape[0])
        if L == 0:
            for sub in mod.modules():
                if isinstance(sub, MultiLinearLoRA):
                    sub.clear_routing()
            return None

        # Routing-buffer slicing strategy.
        #
        # A single global step may invoke ``model.forward`` MORE times than
        # the buffer was sized for: NeMo-RL pads the data iterator with
        # dummy microbatches to equalize across DP ranks. When the cursor
        # would overrun, we MUST keep routing populated — clearing it
        # makes MultiLinearLoRA fall back to base-only (no LoRA), which
        # is a structurally different kernel pattern from the multi-adapter
        # path. Under FSDP/EP, that asymmetry across ranks produces
        # divergent collective sequences and deadlocks NCCL.
        #
        # On overrun, repeat the last ``mb_size``-window cyclically: every
        # rank stays on the same code path through MLL.forward, and dummy
        # microbatches contribute zero loss so adapter routing is moot.
        cursor = int(getattr(mod, "_nousnet_mb_cursor", 0) or 0)
        end = cursor + mb_size
        if end <= L:
            chunk = full[cursor:end]
            mod._nousnet_mb_cursor = end
        else:
            logger.debug(
                "microbatch routing cursor exhausted: cursor=%d mb_size=%d "
                "full_len=%d — using cyclic fallback (repeat last window)",
                cursor, mb_size, L,
            )
            if mb_size <= L:
                chunk = full[L - mb_size : L]
            else:
                # mb_size > L: tile to fill, then trim.
                reps = (mb_size + L - 1) // L
                chunk = full.repeat(reps)[:mb_size]
            # Cursor stays exhausted; subsequent forwards keep getting the
            # last window. Reset only happens via set_microbatch_routing_full
            # (start of next train() call).

        # Move ids to the same device the layer's params live on; do it
        # lazily so we don't need to know the device at seed time.
        if isinstance(first, torch.Tensor) and chunk.device != first.device:
            chunk = chunk.to(first.device)

        # Stash the current microbatch's row ids on the root model so the
        # MoE expert-routing wrapper (moe_routing.py) can derive per-TOKEN
        # ids for router-scattered expert inputs.
        mod._nousnet_current_chunk = chunk

        for sub in mod.modules():
            if isinstance(sub, MultiLinearLoRA):
                sub.set_routing(chunk)
        return None

    handle = model.register_forward_pre_hook(_hook, with_kwargs=True)
    model._nousnet_mb_routing_handle = handle
    return handle


def seed_microbatch_routing(model: nn.Module, adapter_ids: torch.Tensor):
    """Install the microbatch routing hook and seed per-row adapter ids.

    This is the canonical worker-side entry point used by the wrapped
    ``train`` method. It keeps the install-then-seed ordering in one place.

    Also installs per-token MoE expert routing (moe_routing.py) so expert
    MLP LoRA rows route to their owning adapter instead of the legacy
    slot-0 fallback. No-op on non-MoE models.
    """
    handle = install_microbatch_routing_hook(model)
    set_microbatch_routing_full(model, adapter_ids)
    try:
        from nemo_rl.models.multi_lora.moe_routing import install_moe_expert_routing
        install_moe_expert_routing(model)
    except Exception as e:  # noqa: BLE001 — never break training on install
        logger.warning("moe expert routing install failed (legacy slot-0 "
                       "fallback stays active): %s", e)
    return handle
