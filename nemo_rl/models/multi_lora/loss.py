"""Per-adapter token-weighted cross-entropy for multi-LoRA SFT.

Replicates NeMo-RL ``NLLLossFn``'s exact reduction kernel chain so the
autograd graph from logits → token_logprobs → loss is bit-identical to
single-LoRA. Per-adapter normalization is by each adapter's own
valid-token count (not global ``global_valid_toks``), so each adapter's
gradient scale matches a standalone single-LoRA run on the same rows.

Convention: NeMo-RL kwargs API only. NeMo-RL calls us as ::

    loss, metrics = loss_fn(
        data=batched_data_dict,
        global_valid_seqs=...,            # absorbed via **_
        global_valid_toks=...,            # absorbed via **_
        logits=...,                       # OR
        next_token_logprobs=...,          # when input_type=LOGPROB
    )

``data`` must carry ``adapter_names: list[str]`` of length B,
block-contiguous per :meth:`MultiAdapterDataLoader._pack`. Optional:
``token_mask`` (``[B, T]``), ``sample_mask`` (``[B]``), and ``seq_index``
(for DTensor logprob under CP).

Subtleties pinned by tests (see ``tests/rl/lora/test_loss.py``):
    1. **Op-order bit-match.** NLL chain is exactly
       ``log_softmax(fp32) → [:,:-1] → gather`` — same ops, same order
       as ``loss_functions.py:660-667``.
    2. **Slice, don't mask.** Per-adapter reduction narrows the logprob
       tensor to that adapter's contiguous row range before ``torch.sum``.
       Masking the full ``[B, T-1]`` tensor and summing has a different
       reduction tree than summing the narrowed ``[n_rows, T-1]`` tensor,
       producing 1-ULP rounding errors that compound through the optimizer.
    3. **Lazy accumulation.** ``total`` seeds from the first per-adapter
       ``l_i`` (not from ``logprobs.sum() * 0.0``). The
       ``logprobs.sum() * 0.0`` seed would add an extra autograd reduction
       over all logprob positions whose backward in fp32 perturbs
       grad-accumulation kernel order — mathematically zero, not
       bit-equivalent.
    4. **Mask materialize-before-shift.** A DTensor ``token_mask`` must be
       materialized to a full tensor BEFORE the ``[:, 1:]`` shift,
       otherwise the per-rank CP shard gets shifted (dropping the first
       token of each shard, not one global token).
    5. **DTensor logprobs go through nemo_rl's vocab-parallel helper.**
       ``full_tensor()`` + local ``log_softmax`` produces a different
       result than the DTensor-aware ``get_logprobs_from_vocab_parallel_logits``
       under CP (different gather/permute order).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn.functional as F

from nemo_rl.models.multi_lora._compat import optional_dtensor_type

logger = logging.getLogger(__name__)

# ``process_global_batch`` attaches one globally all-reduced denominator per
# row when a batch carries canonical ``adapter_ids``. The row-aligned shape is
# intentional: ``BatchedDataDict.slice`` then preserves the value through
# microbatching without special metadata handling.
GLOBAL_ADAPTER_TOKEN_COUNTS_KEY = "_nousnet_adapter_global_valid_toks"


# NeMo-RL loss-interface enums, resolved at import time. ``None`` when
# NeMo-RL isn't installed (test environments without container).
try:
    from nemo_rl.algorithms.loss.interfaces import LossInputType, LossType
except ImportError:
    try:
        from nemo_rl.algorithms.interfaces import LossType  # type: ignore[no-redef]
        LossInputType = None  # older container has LossType only
    except ImportError:
        LossType = None
        LossInputType = None


# ---------------------------------------------------------------------------
# Helpers — exposed at module scope so tests can target them directly.
# ---------------------------------------------------------------------------


def adapter_row_ranges(adapter_names: list[str]) -> list[tuple[str, int, int]]:
    """Find contiguous ``(name, start, end)`` blocks in ``adapter_names``.

    Multi-LoRA's :meth:`MultiAdapterDataLoader._pack` guarantees a
    block-contiguous layout (``['a','a','b','b','b']``, never
    ``['a','b','a','b','b']``). If a name reappears after a different name
    interrupted it, raise — silently slicing in non-contiguous order would
    route rows to the wrong adapter in the reduction loop.
    """
    if not adapter_names:
        return []
    seen: set[str] = set()
    out: list[tuple[str, int, int]] = []
    i = 0
    n = len(adapter_names)
    while i < n:
        name = adapter_names[i]
        if name in seen:
            raise ValueError(
                f"adapter_names not block-contiguous: {adapter_names!r}; "
                f"adapter {name!r} appears in multiple non-contiguous blocks"
            )
        seen.add(name)
        j = i + 1
        while j < n and adapter_names[j] == name:
            j += 1
        out.append((name, i, j))
        i = j
    return out


def build_loss_mask(data: Any) -> torch.Tensor | None:
    """Materialize and shift the per-token loss mask to shape ``[B, T-1]``.

    Reads optional ``data['token_mask']`` (``[B, T]``) and
    ``data['sample_mask']`` (``[B]``). DTensors are materialized via
    ``full_tensor()`` **before** the ``[:, 1:]`` shift. Returns ``None``
    when ``token_mask`` isn't present (caller falls back to label padding).
    """
    tm = data.get("token_mask") if hasattr(data, "get") else None
    if tm is None or not isinstance(tm, torch.Tensor) or tm.dim() != 2 or tm.size(1) < 2:
        return None

    sm = data.get("sample_mask") if hasattr(data, "get") else None
    DT = optional_dtensor_type()
    if DT is not None:
        if isinstance(tm, DT):
            tm = tm.full_tensor()
        if sm is not None and isinstance(sm, DT):
            sm = sm.full_tensor()

    mask = tm[:, 1:].to(torch.float32)
    if sm is not None:
        mask = mask * sm.to(torch.float32).unsqueeze(-1)
    return mask


def compute_token_logprobs(logits: torch.Tensor, data: Any) -> torch.Tensor:
    """Compute next-token logprobs ``[B, T-1]`` matching NLLLossFn exactly.

    - DTensor logits → ``get_logprobs_from_vocab_parallel_logits``
      (CP/TP-aware via NeMo-RL's distributed helper).
    - Plain logits → ``log_softmax(fp32) → [:,:-1] → gather``, the exact
      op chain in ``loss_functions.py:660-667``.
    """
    DT = optional_dtensor_type()
    logits_fp32 = logits.to(torch.float32)

    if DT is not None and isinstance(logits_fp32, DT):
        from nemo_rl.distributed.model_utils import (
            get_logprobs_from_vocab_parallel_logits,
        )
        return get_logprobs_from_vocab_parallel_logits(
            logits_fp32, data["input_ids"], seq_index=data.get("seq_index"),
        )

    input_ids = data["input_ids"]
    next_tokens = input_ids[:, 1:].to(logits_fp32.device)
    logp = F.log_softmax(logits_fp32, dim=-1)
    shifted = logp[:, :-1]
    return shifted.gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)


def reduce_per_adapter(
    token_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    ranges: list[tuple[str, int, int]],
    global_valid_toks: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Per-adapter NLL with NLLLossFn's exact reduction kernel chain.

    For each ``(name, start, end)``:
        ``narrow`` the logprob + mask tensors to the contiguous row range,
        then accumulate ``-sum(logp * mask)`` into ``total``. Slicing
        before sum is critical: summing the same elements after
        zero-masking the full ``[B, T-1]`` tensor uses a different
        reduction tree and produces 1-ULP rounding errors.

    Normalization (``global_valid_toks``):
        - When provided (the production path, supplied by the worker as an
          all-reduced scalar over the *full* batch), the returned ``total``
          is ``-sum(logp * mask) / global_valid_toks`` summed across
          adapters. This matches NeMo-RL ``NLLLossFn`` exactly and
          DP-all-reduce-sums correctly to the global per-token mean (no
          extra DP_world_size scaling).
        - When ``None`` (tests + legacy callers without an all-reduce), the
          per-adapter denominator falls back to the local ``adapter_mask.sum()``
          for human-readable per-adapter means. Note this does NOT DP-reduce
          correctly under multi-rank training.

    Per-adapter metrics (``means``) are always reported as the LOCAL per-token
    mean for that adapter (for wandb visibility), independent of the
    backward-path normalization.

    Returns ``(total, per_adapter_means)``. ``total`` is ``None`` when no
    adapter contributed any rows (caller is responsible for emitting a
    connected zero scalar for backward).
    """
    total: torch.Tensor | None = None
    means: dict[str, float] = {}
    for name, start, end in ranges:
        n_rows = end - start
        adapter_logprobs = token_logprobs.narrow(0, start, n_rows)
        adapter_mask = loss_mask.narrow(0, start, n_rows)
        num = -torch.sum(adapter_logprobs * adapter_mask)
        # Per-adapter local mean for metrics (always available, BF16/FP32 safe).
        local_den = adapter_mask.sum() + 1e-8
        means[name] = float((num / local_den).detach())
        # Backward-path normalization must use the adapter's GLOBAL token
        # count across DP ranks, exactly like standalone NLLLoss uses
        # ``global_valid_toks``. ``process_global_batch`` materializes that
        # value row-wise under GLOBAL_ADAPTER_TOKEN_COUNTS_KEY before the batch
        # is sliced into microbatches. A local denominator is only correct for
        # DP=1 (or when every adapter is present on exactly one rank); under the
        # rank-64/DP=8 topology it caused each two-rank adapter contribution to
        # be added twice, producing the observed ~2x loss and gradients.
        den = local_den
        if hasattr(global_valid_toks, "get"):
            den_by_name = global_valid_toks
            if name in den_by_name:
                den = den_by_name[name]
        l_i = num / (den + 1e-8)
        # Lazy accumulation: seeding `total` with `logprobs.sum() * 0.0`
        # would add an extra autograd reduction whose backward in fp32
        # perturbs grad-accumulation kernel order (mathematically zero,
        # not bit-equivalent to NLLLossFn).
        total = l_i if total is None else total + l_i
    return total, means


# ---------------------------------------------------------------------------
# Public class — the NeMo-RL LossFunction adapter.
# ---------------------------------------------------------------------------


class MultiAdapterLoss:
    """NeMo-RL ``LossFunction`` for per-adapter token-weighted CE.

    Returns ``(loss, metrics)``:
        - ``loss``: scalar, sum of per-adapter token-mean CE values.
        - ``metrics['per_adapter_loss/<name>']``: float per active adapter.
        - ``metrics['num_unmasked_tokens']``: int total valid label tokens.
        - ``metrics['num_valid_samples']``: int routed rows (``= len(adapter_names)``).
    """

    loss_type = LossType.TOKEN_LEVEL if LossType is not None else None
    # LOGPROB tells NeMo-RL to pre-gather via the DTensor-aware path so we
    # receive plain ``[B, T-1]`` next_token_logprobs. Older containers
    # without LossInputType ignore this and pass raw logits — we handle
    # both inputs.
    input_type = LossInputType.LOGPROB if LossInputType is not None else None

    def __call__(
        self,
        next_token_logits: torch.Tensor | None = None,
        data: Any = None,
        global_valid_seqs: torch.Tensor | None = None,
        global_valid_toks: torch.Tensor | None = None,
        *,
        logits: torch.Tensor | None = None,
        next_token_logprobs: torch.Tensor | None = None,
        **_unused: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # NeMo-RL's LossFunction Protocol passes positional (next_token_logits,
        # data, global_valid_seqs, global_valid_toks). We also accept keyword
        # `logits=` / `next_token_logprobs=` for tests + the rare path where
        # the worker pre-gathers logprobs (input_type=LOGPROB).
        if logits is None and next_token_logits is not None:
            logits = next_token_logits
        if data is None:
            raise TypeError("MultiAdapterLoss requires `data` (positional or keyword)")
        if (logits is None) == (next_token_logprobs is None):
            raise TypeError(
                "MultiAdapterLoss requires exactly one of `logits` or "
                "`next_token_logprobs` (got both or neither)"
            )

        adapter_names = list(data["adapter_names"])
        ranges = adapter_row_ranges(adapter_names)

        # Extract per-adapter global token counts if the batch carries them.
        # The worker stores one denominator per row so BatchedDataDict slicing
        # preserves it through microbatching. Convert the row-aligned tensor to
        # a name->scalar mapping for ``reduce_per_adapter``.
        adapter_global_toks = None
        if hasattr(data, "get"):
            row_global_toks = data.get(GLOBAL_ADAPTER_TOKEN_COUNTS_KEY)
            if isinstance(row_global_toks, torch.Tensor):
                adapter_global_toks = {
                    name: row_global_toks[start]
                    for name, start, _end in ranges
                }

        # token_logprobs: [B, T-1]
        if logits is not None:
            token_logprobs = compute_token_logprobs(logits, data)
        else:
            DT = optional_dtensor_type()
            tl = next_token_logprobs
            token_logprobs = (
                tl.full_tensor() if (DT is not None and isinstance(tl, DT)) else tl
            )

        # Loss mask: prefer token_mask + sample_mask. Fall back to label
        # padding (anywhere != -100) when neither is provided.
        loss_mask = build_loss_mask(data)
        if loss_mask is None:
            labels = data.get("labels", data["input_ids"]) if hasattr(data, "get") else data["input_ids"]
            loss_mask = (labels[..., 1:] != -100).to(token_logprobs.dtype)

        total, per_adapter = reduce_per_adapter(
            token_logprobs, loss_mask, ranges,
            global_valid_toks=adapter_global_toks if adapter_global_toks is not None else global_valid_toks,
        )
        if total is None:
            # No adapter contributed any row — return a connected zero so
            # ``loss.backward()`` doesn't error. Pathological case.
            total = token_logprobs.sum() * 0.0

        # ----- Diagnostic instrumentation (no-op unless NOUSNET_DIAG_ENABLED=1) -----
        # Per-adapter forward fingerprints. Both the singles' NLLLoss hook
        # and this hook emit identically-named scalars so a direct compare
        # of single_X step N vs multi_adapter_X step N is byte-meaningful.
        from nemo_rl.models.multi_lora import diag as _diag
        if _diag.is_enabled():
            try:
                step = _diag.next_step("MultiAdapterLoss")
                rank = int(os.environ.get("RANK", "0"))
                trace_only = os.environ.get("NOUSNET_DIAG_TRACE_ONLY", "0") == "1"
                scalars_extra: dict[str, Any] = {
                    f"diag/global/global_valid_toks": float(
                        global_valid_toks.item() if hasattr(global_valid_toks, "item")
                        else float(global_valid_toks or 0.0)
                    ),
                    f"diag/global/loss_mask_sum_local": float(loss_mask.sum().item()),
                }
                for name, start, end in ranges:
                    who = f"multi_adapter_{name.replace('adapter_', '')}"
                    adapter_ids = data["input_ids"][start:end] if "input_ids" in data else None
                    adapter_logprobs = token_logprobs.narrow(0, start, end - start)
                    adapter_mask = loss_mask.narrow(0, start, end - start)
                    if not trace_only:
                        scalars_extra.update(_diag.build_step_scalars(
                            "diag", who,
                            input_ids=adapter_ids,
                            token_logprobs=adapter_logprobs,
                            loss_mask=adapter_mask,
                        ))
                    _diag.append_loss_trace(
                        step=step,
                        rank=rank,
                        who=who,
                        input_ids=adapter_ids,
                        token_logprobs=adapter_logprobs,
                        loss_mask=adapter_mask,
                    )
                    if not trace_only:
                        _diag.dump_tensors(step, rank, who,
                            input_ids=adapter_ids,
                            token_logprobs=adapter_logprobs,
                            loss_mask=adapter_mask,
                        )
                # Attach extras to metrics so the worker's wandb logger picks them up.
                # (NeMo-RL's worker logs every key in this dict.)
                metrics_extra = scalars_extra
            except Exception as e:
                logger.warning("diag hook failed in MultiAdapterLoss: %s", e)
                metrics_extra = {}
        else:
            metrics_extra = {}
        # ---------------------------------------------------------------------------

        metrics: dict[str, Any] = {
            "num_unmasked_tokens": int(loss_mask.sum().item()),
            "num_valid_samples": len(adapter_names),
        }
        for name, val in per_adapter.items():
            metrics[f"per_adapter_loss/{name}"] = val
        metrics.update(metrics_extra)
        return total, metrics
