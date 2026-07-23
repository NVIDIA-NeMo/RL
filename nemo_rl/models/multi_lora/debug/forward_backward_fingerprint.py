"""Forward + backward activation/gradient fingerprint hooks for multi-LoRA bit-eq.

Method documented in `~/.hermes/skills/ml-training/multi-lora-bit-equivalence-debugging/references/per-layer-activation-grad-fingerprint-bisection.md`.
Original helper at `/home/phuc/workspace/moe/worklogs/20251124_deepep_gradient_explosion_investigation/gradient_logging.py`.

Gating: only installs hooks when `NOUSNET_FORWARD_BACKWARD_DIAG=1` in env.
Output: `[ACT_LOG]` / `[GRAD_LOG]` / `[PARAM_GRAD_LOG]` lines on rank=`log_rank` only
(default rank 0). Each line: `name`, `norm`, `mean(abs)`, `max(abs)`, `shape`.

To diff two runs (single_a vs multi_abcd, 1 step each):

    diff -u <(grep -E "^\\[(ACT|GRAD|PARAM_GRAD)_LOG\\]" $SLOG | sort) \\
            <(grep -E "^\\[(ACT|GRAD|PARAM_GRAD)_LOG\\]" $MLOG | sort) | head -100

First DIFFER line (sorted by module name → layer index) is the divergence site.
"""

from __future__ import annotations

import os
from typing import Iterable

import torch
import torch.nn as nn

_HANDLES: list = []


def _to_local_if_dtensor(t: torch.Tensor) -> torch.Tensor:
    """Return the local shard if this is a DTensor, else identity. Used to keep
    fingerprint computation cheap and rank-local (skip the all-gather)."""
    if hasattr(t, "to_local"):
        try:
            return t.to_local()
        except Exception:
            return t
    return t


def _fmt(t: torch.Tensor) -> str:
    t = _to_local_if_dtensor(t)
    if t.numel() == 0:
        return f"norm=         0.0000 mean=    0.000000 max=    0.000000 shape={tuple(t.shape)}"
    td = t.detach()
    return (
        f"norm={td.norm().item():12.4f} "
        f"mean={td.abs().mean().item():12.6f} "
        f"max={td.abs().max().item():12.6f} "
        f"shape={tuple(t.shape)}"
    )


def _gated_print(rank: int, log_rank: int | None, msg: str) -> None:
    if log_rank is None or rank == log_rank:
        print(msg, flush=True)


def _matches_filters(name: str, cls_name: str, filters: tuple[str, ...]) -> bool:
    if not filters:
        return False
    return any(f in cls_name or f in name for f in filters)


def install_fingerprint_hooks(
    model: nn.Module,
    *,
    log_rank: int | None = 0,
    module_filters: tuple[str, ...] = (),
    param_filters: tuple[str, ...] = (),
) -> None:
    """Install forward + param-grad hooks. Idempotent across calls (clears
    prior handles first).

    Args:
        model: the model whose modules / params to hook. On the worker side this
            is `self.model` (post-FSDP wrap, post-LoRA injection).
        log_rank: rank to print on. None = every rank with `[rank=N]` prefix.
        module_filters: substring filters on either module class name or fq
            module name. Both forward output and forward input are captured.
        param_filters: substring filters on parameter fq names.
    """
    if os.environ.get("NOUSNET_FORWARD_BACKWARD_DIAG") != "1":
        return
    # Idempotent re-install
    remove_fingerprint_hooks()

    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    prefix = f"[rank={rank}] " if log_rank is None else ""

    # Forward hooks (activation fingerprints).
    hooked_modules: list[str] = []
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if not _matches_filters(name, cls, module_filters):
            continue

        def _hook(m, inp, out, name=name, cls=cls):
            if isinstance(out, torch.Tensor):
                _gated_print(rank, log_rank, f"{prefix}[ACT_LOG] {cls}::{name:60s} {_fmt(out)}")
            elif isinstance(out, (tuple, list)):
                for i, x in enumerate(out):
                    if isinstance(x, torch.Tensor):
                        _gated_print(
                            rank, log_rank,
                            f"{prefix}[ACT_LOG] {cls}::{name}[{i}]:".ljust(70) + " " + _fmt(x),
                        )

        _HANDLES.append(mod.register_forward_hook(_hook))
        hooked_modules.append(f"{cls}::{name}")

    # Param-grad hooks (backward fingerprints) — fire as each param's grad is computed.
    hooked_params: list[str] = []
    for name, p in model.named_parameters():
        if not _matches_filters(name, "", param_filters):
            continue
        if not p.requires_grad:
            # Still log this for diagnostic visibility — frozen param shouldn't trigger,
            # but knowing it's frozen tells us autograd never tries.
            _gated_print(rank, log_rank, f"{prefix}[PARAM_DIAG] {name} requires_grad=False (will not receive grad)")
            continue

        def _gh(g, name=name):
            if g is not None:
                _gated_print(rank, log_rank, f"{prefix}[GRAD_LOG] {name:60s} {_fmt(g)}")
            else:
                _gated_print(rank, log_rank, f"{prefix}[GRAD_LOG] {name:60s} grad=None")
            return g

        _HANDLES.append(p.register_hook(_gh))
        hooked_params.append(name)

    # Summary print so we know hooks armed correctly
    _gated_print(
        rank, log_rank,
        f"{prefix}[FP_INSTALL] hooked {len(hooked_modules)} modules, {len(hooked_params)} params",
    )
    if log_rank is None or rank == log_rank:
        for n in hooked_modules[:10]:
            print(f"{prefix}[FP_INSTALL]   module: {n}", flush=True)
        if len(hooked_modules) > 10:
            print(f"{prefix}[FP_INSTALL]   ... +{len(hooked_modules) - 10} more modules", flush=True)
        for n in hooked_params[:10]:
            print(f"{prefix}[FP_INSTALL]   param:  {n}", flush=True)
        if len(hooked_params) > 10:
            print(f"{prefix}[FP_INSTALL]   ... +{len(hooked_params) - 10} more params", flush=True)


def log_all_param_grads(model: nn.Module, step: int, *, log_rank: int | None = 0) -> None:
    """Post-backward dump of EVERY trainable param's grad. Use sparingly — call
    once after `loss.backward()` on step 1 to confirm which params received grads.

    This is the smoking-gun probe for the "weights frozen" symptom: if the
    multi-LoRA `lora_A`/`lora_B` weights show up with `grad=None` while singles
    show non-zero norm, autograd is severed somewhere on the loss path.
    """
    if os.environ.get("NOUSNET_FORWARD_BACKWARD_DIAG") != "1":
        return
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    if log_rank is not None and rank != log_rank:
        return
    print(f"\n{'=' * 80}\n[PARAM_GRAD_LOG] STEP {step} — post-backward grad dump\n{'=' * 80}", flush=True)
    n_with_grad = 0
    n_with_grad_none = 0
    n_frozen = 0
    total_sq = 0.0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            n_frozen += 1
            continue
        if p.grad is None:
            n_with_grad_none += 1
            print(f"[PARAM_GRAD_LOG] {name:60s} grad=None (requires_grad=True but no grad arrived)", flush=True)
            continue
        g = _to_local_if_dtensor(p.grad)
        gn = g.detach().norm().item()
        gm = g.detach().abs().mean().item()
        gx = g.detach().abs().max().item()
        total_sq += gn * gn
        print(
            f"[PARAM_GRAD_LOG] {name:60s} norm={gn:12.4f} mean={gm:12.6f} max={gx:12.6f} shape={tuple(p.shape)}",
            flush=True,
        )
        n_with_grad += 1
    print(
        f"[PARAM_GRAD_LOG] SUMMARY step={step}: "
        f"with_grad={n_with_grad}  grad_none={n_with_grad_none}  frozen={n_frozen}  "
        f"total_grad_norm={total_sq ** 0.5:.4f}",
        flush=True,
    )


def remove_fingerprint_hooks() -> None:
    for h in _HANDLES:
        try:
            h.remove()
        except Exception:
            pass
    _HANDLES.clear()
