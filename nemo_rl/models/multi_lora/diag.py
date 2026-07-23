"""Bit-equivalence diagnostic instrumentation for multi-LoRA vs single-LoRA SFT.

Single source of truth: both ``MultiAdapterLoss`` (nousnet) and ``NLLLoss``
(NeMo-RL) call into the same helpers here so the per-step fingerprints they
produce are byte-comparable. Hooks are all NO-OPs unless
``NOUSNET_DIAG_ENABLED`` is set, so production runs pay zero cost.

Environment variables
---------------------
NOUSNET_DIAG_ENABLED      : "1" to enable any diag at all (default off).
NOUSNET_DIAG_DUMP         : "1" to additionally write raw tensor .pt files
                            under ${NOUSNET_RUN_DIR}/diag_tensors/step_NNNN/.
                            Scalars always go to wandb when diag enabled.
NOUSNET_DIAG_DETERMINISM  : "1" to call torch.use_deterministic_algorithms(True).
                            Skipped by default because it requires
                            CUBLAS_WORKSPACE_CONFIG and can break some kernels.

Public API
----------
enable_absolute_determinism(seed)
    Sets torch + numpy + Python RNG seeds, deterministic algorithms,
    cuDNN flags, and validates the env vars CUBLAS_WORKSPACE_CONFIG and
    PYTHONHASHSEED are set. Idempotent.

fingerprint_tensor(t, *, n_bytes=16) -> str
    SHA256-truncated hex fingerprint of a tensor's raw bytes. Materializes
    DTensors via ``full_tensor()`` first so all ranks agree.

tensor_stats(t) -> dict
    {"sha16", "shape", "dtype", "norm", "sum", "min", "max", "abs_mean"}.
    Norm/sum/min/max are computed in float32 to avoid BF16 noise in the
    diagnostic itself. All values are Python floats / strs, JSON-safe.

build_step_scalars(prefix, who, **named_tensors) -> dict[str, float|str]
    Flat dict suitable for direct wandb logging. Keys are
    ``"{prefix}/{who}/{name}_{stat}"`` for stat in
    {"sha16","norm","sum","min","max","abs_mean","shape"}.
    Tensors that are None are skipped.

dump_tensors(step, rank, who, **named_tensors)
    Writes named tensors to ``${NOUSNET_RUN_DIR}/diag_tensors/step_NNNN/rank_RR/<who>__<name>.pt``
    only when ``NOUSNET_DIAG_DUMP=1``. Always a no-op when diag not enabled.

is_enabled() -> bool
is_dump_enabled() -> bool
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import torch

from nemo_rl.models.multi_lora._compat import optional_dtensor_type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Toggles
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """True iff NOUSNET_DIAG_ENABLED=1. Cached per-call; cheap."""
    return os.environ.get("NOUSNET_DIAG_ENABLED", "0") == "1"


def is_dump_enabled() -> bool:
    """True iff NOUSNET_DIAG_ENABLED=1 AND NOUSNET_DIAG_DUMP=1."""
    return is_enabled() and os.environ.get("NOUSNET_DIAG_DUMP", "0") == "1"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


_DETERMINISM_APPLIED = False


def enable_absolute_determinism(seed: int = 42) -> dict[str, Any]:
    """Set every knob torch/numpy/python expose for bit-reproducibility.

    Idempotent: safe to call from every rank, every entry point. Returns a
    status dict so callers can log/verify what was actually applied.

    Caller MUST have:
        CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8)
        PYTHONHASHSEED=0 (or any fixed value)
    set in env BEFORE Python interpreter start. We warn if either is missing
    but don't crash — some platforms patch these inside container init and
    we still want the run to proceed with best-effort determinism.
    """
    global _DETERMINISM_APPLIED

    status: dict[str, Any] = {
        "seed": seed,
        "already_applied": _DETERMINISM_APPLIED,
        "warnings": [],
    }

    cublas_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas_cfg not in (":4096:8", ":16:8"):
        status["warnings"].append(
            f"CUBLAS_WORKSPACE_CONFIG={cublas_cfg!r}, expected ':4096:8' or ':16:8'"
        )

    pyhash = os.environ.get("PYTHONHASHSEED")
    if pyhash is None:
        status["warnings"].append("PYTHONHASHSEED unset (set before python start)")

    # Seed every RNG we know about.
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        status["numpy_seeded"] = True
    except ImportError:
        status["numpy_seeded"] = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic kernels.
    if os.environ.get("NOUSNET_DIAG_DETERMINISM", "1") == "1":
        try:
            # warn_only=False forces deterministic algorithms. If Flash
            # Attention backward has no deterministic impl, PyTorch will
            # raise RuntimeError — which is what we want for bit-eq work.
            # The worker catches and prints the error, so training won't
            # silently proceed with non-deterministic gradients.
            torch.use_deterministic_algorithms(True, warn_only=False)
            status["use_deterministic_algorithms"] = True
        except Exception as e:
            status["use_deterministic_algorithms"] = f"failed: {e}"
    else:
        status["use_deterministic_algorithms"] = "disabled by env"

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    status["cudnn_deterministic"] = True
    status["cudnn_benchmark"] = False

    # Best-effort: deterministic TF32 disable (cuBLAS still uses TF32 unless
    # the workspace config is set; we already validated that above).
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        status["tf32_disabled"] = True
    except AttributeError:
        status["tf32_disabled"] = "n/a"

    _DETERMINISM_APPLIED = True
    return status


# ---------------------------------------------------------------------------
# Tensor fingerprints
# ---------------------------------------------------------------------------


def _materialize(t: torch.Tensor) -> torch.Tensor:
    """Return a plain Tensor on the original device, materializing DTensors."""
    DT = optional_dtensor_type()
    if DT is not None and isinstance(t, DT):
        return t.full_tensor()
    return t


def fingerprint_tensor(t: torch.Tensor, *, n_bytes: int = 16) -> str:
    """SHA256-truncated hex fingerprint of a tensor's raw bytes.

    Materializes DTensors first (every rank sees the same global tensor).
    The fingerprint is computed on CPU to avoid GPU sync cost when callers
    already have a CPU view. We cast to contiguous to defeat strided layouts
    producing different byte sequences for the "same" tensor.
    """
    if t is None:
        return "none"
    t = _materialize(t)
    if t.is_cuda:
        t_cpu = t.detach().to("cpu", non_blocking=False)
    else:
        t_cpu = t.detach()
    t_cpu = t_cpu.contiguous()
    h = hashlib.sha256(t_cpu.numpy().tobytes()).hexdigest()
    return h[: n_bytes * 2]


def tensor_stats(t: torch.Tensor | None) -> dict[str, Any]:
    """Compact, JSON-safe summary of a tensor for wandb logging."""
    if t is None:
        return {
            "sha16": "none", "shape": "none", "dtype": "none",
            "norm": 0.0, "sum": 0.0, "min": 0.0, "max": 0.0, "abs_mean": 0.0,
        }
    t = _materialize(t)
    t32 = t.detach().to(torch.float32)
    return {
        "sha16": fingerprint_tensor(t),
        "shape": "x".join(str(d) for d in t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "norm": float(t32.norm().item()),
        "sum": float(t32.sum().item()),
        "min": float(t32.min().item()) if t32.numel() else 0.0,
        "max": float(t32.max().item()) if t32.numel() else 0.0,
        "abs_mean": float(t32.abs().mean().item()) if t32.numel() else 0.0,
    }


# ---------------------------------------------------------------------------
# wandb scalar emission
# ---------------------------------------------------------------------------


_FLOAT_KEYS = ("norm", "sum", "min", "max", "abs_mean")
_STR_KEYS = ("sha16", "shape", "dtype")


def build_step_scalars(
    prefix: str,
    who: str,
    **named_tensors: torch.Tensor | None,
) -> dict[str, Any]:
    """Build a flat dict for wandb.

    Example:
        scalars = build_step_scalars(
            "diag", "multi_adapter_a",
            input_ids=ids, token_logprobs=tlp, lora_A=A, lora_A_grad=Agrad,
        )

    Output keys (numeric only — string-valued sha16/shape/dtype were dropped
    on 2026-05-29 because the downstream NeMo-RL metric aggregator at
    ``automodel/train.py:439`` does ``metrics[k] /= num_global_batches`` and
    ``algorithms/sft.py:594`` does ``np.sum(v).item()`` over the per-mb
    list — both crash on str. Bit-level identity is preserved via
    ``dump_tensors`` which writes ``.pt`` files to disk — diff those, not
    wandb.

        "diag/multi_adapter_a/input_ids_norm"   : float
        ...
        "diag/multi_adapter_a/lora_A_grad_norm" : float (or 0.0 if grad=None)

    Skips entries whose value is None entirely (no zero-padding noise),
    EXCEPT for the grad keys (callers want explicit zero for "grad not set").
    """
    out: dict[str, Any] = {}
    for name, t in named_tensors.items():
        stats = tensor_stats(t)
        for k in _FLOAT_KEYS:
            out[f"{prefix}/{who}/{name}_{k}"] = stats[k]
    return out


# ---------------------------------------------------------------------------
# Disk dumps (opt-in, large)
# ---------------------------------------------------------------------------


def append_loss_trace(
    *,
    step: int,
    rank: int,
    who: str,
    input_ids: torch.Tensor,
    token_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
) -> None:
    """Append one compact pairwise-loss record per loss invocation.

    Enabled only by ``NOUSNET_DIAG_LOSS_TRACE=1``. Unlike raw tensor dumps,
    this writes a few hundred bytes per row and is safe for horizon probes.
    The loss is recomputed from the exact token log-probabilities and mask used
    by the backward loss. ``input_sha256`` makes row alignment explicit.
    """
    if not is_enabled() or os.environ.get("NOUSNET_DIAG_LOSS_TRACE", "0") != "1":
        return
    run_dir = os.environ.get("NOUSNET_RUN_DIR")
    if not run_dir:
        return
    ids = _materialize(input_ids).detach().cpu().contiguous()
    logp = _materialize(token_logprobs).detach().to(torch.float32).cpu()
    mask = _materialize(loss_mask).detach().to(torch.float32).cpu()
    den = mask.sum()
    loss = float((-(logp * mask).sum() / den).item()) if float(den.item()) else None
    record = {
        "step": int(step),
        "rank": int(rank),
        "who": str(who),
        "loss": loss,
        "num_tokens": int(den.item()),
        "input_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        "input_shape": list(ids.shape),
    }
    path = Path(run_dir) / "diag_loss_trace" / f"rank_{rank:02d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def dump_tensors(
    step: int,
    rank: int,
    who: str,
    **named_tensors: torch.Tensor | None,
) -> None:
    """Optionally write raw tensors to disk for post-hoc bit-comparison.

    Writes to ``${NOUSNET_RUN_DIR}/diag_tensors/step_NNNN/rank_RR/<who>__<name>.pt``
    Only fires when both NOUSNET_DIAG_ENABLED=1 and NOUSNET_DIAG_DUMP=1.
    Tensors are saved on CPU (materializing DTensors first) so other ranks /
    other runs can compare without device coordination.
    """
    if not is_dump_enabled():
        return
    run_dir = os.environ.get("NOUSNET_RUN_DIR")
    if not run_dir:
        return
    out_dir = Path(run_dir) / "diag_tensors" / f"step_{step:04d}" / f"rank_{rank:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in named_tensors.items():
        if t is None:
            continue
        t = _materialize(t)
        path = out_dir / f"{who}__{name}.pt"
        try:
            torch.save(t.detach().cpu().contiguous(), path)
        except Exception as e:
            logger.warning("diag dump failed for %s: %s", path, e)


# ---------------------------------------------------------------------------
# Step-counter helper (used by callers that don't have one handy)
# ---------------------------------------------------------------------------


_STEP_COUNTER = {"NLLLoss": 0, "MultiAdapterLoss": 0}


def next_step(name: str) -> int:
    """Monotonic per-name step counter, scoped to this process.

    Used by loss hooks that don't have a step number from the trainer. Callers
    that DO have the real trainer step number should pass it directly to
    ``dump_tensors``/``build_step_scalars`` instead of using this.
    """
    n = _STEP_COUNTER.get(name, 0) + 1
    _STEP_COUNTER[name] = n
    return n


# ---------------------------------------------------------------------------
# Model-side LoRA tensor enumeration
# ---------------------------------------------------------------------------


def iter_lora_tensors(model: Any, adapter_names: list[str] | None = None):
    """Iterate ``(who, module_path, tensor_name, tensor)`` for LoRA params.

    Handles both shapes:

    - Single LoRA (``LinearLoRA``): ``self.lora_A.weight`` ``[dim, in_f]``,
      ``self.lora_B.weight`` ``[out_f, dim]``. Emitted with
      ``who = NOUSNET_DIAG_WHO`` (set per-launch) for a single adapter.
    - Multi LoRA (``MultiLinearLoRA``): ``self.lora_A`` ``[n_adapters, dim, in_f]``,
      ``self.lora_B`` ``[n_adapters, out_f, dim]``. Emitted as one slice per
      adapter with ``who = f"multi_adapter_{name.replace('adapter_','')}"``.

    Adapter name resolution for multi: ``adapter_names`` must be passed in
    canonical adapter order (the same list passed to MultiAdapterDataLoader),
    so slice index 0 → adapter_names[0]. If None, falls back to ``"idx{i}"``.
    """
    who_single = os.environ.get("NOUSNET_DIAG_WHO", "single_unknown")
    for path, module in model.named_modules():
        A = getattr(module, "lora_A", None)
        B = getattr(module, "lora_B", None)
        if A is None or B is None:
            continue

        # Single LoRA can be a stock LinearLoRA OR the runtime-created
        # PatchedLinearLoRA class used by apply_lora_to_linear_modules.  Detect
        # it from its storage contract rather than exact class-name equality.
        Aw = getattr(A, "weight", None)
        Bw = getattr(B, "weight", None)
        if isinstance(Aw, torch.Tensor) and isinstance(Bw, torch.Tensor):
            if Aw.ndim == 2 and Bw.ndim == 2:
                yield (who_single, path, "lora_A", Aw)
                yield (who_single, path, "lora_B", Bw)
                yield (who_single, path, "lora_A_grad", Aw.grad)
                yield (who_single, path, "lora_B_grad", Bw.grad)
                continue

        # Multi LoRA: stacked tensor [n_adapters, ...]. Emit one slice per
        # adapter.  Shape detection keeps this robust to wrappers/subclasses.
        if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
            if A.ndim != 3 or B.ndim != 3:
                continue
            n = A.shape[0]
            for i in range(n):
                name = adapter_names[i] if adapter_names and i < len(adapter_names) else f"idx{i}"
                who = f"multi_adapter_{name.replace('adapter_', '')}"
                yield (who, path, "lora_A", A[i])
                yield (who, path, "lora_B", B[i])
                # Grads share the leading-dim slice indexing iff .grad is the
                # same shape as the full parameter. autograd writes [n_adapters, …].
                Ag = A.grad[i] if A.grad is not None else None
                Bg = B.grad[i] if B.grad is not None else None
                yield (who, path, "lora_A_grad", Ag)
                yield (who, path, "lora_B_grad", Bg)


def aggregate_lora_fingerprints(
    model: Any,
    adapter_names: list[str] | None = None,
    *,
    grads: bool = True,
    dump_phase: str | None = None,
    step: int | None = None,
) -> dict[str, Any]:
    """Walk the model and produce per-``who`` aggregate fingerprints.

    Aggregates per-(who, name) by concatenating all module slices into one
    flat tensor and hashing/normalizing the concat. This collapses
    "the per-layer A's" into one fingerprint per adapter, which is what we
    actually want to compare across runs.

    When ``NOUSNET_DIAG_DUMP=1`` and ``dump_phase`` is provided (e.g.
    ``"pre_step"`` or ``"post_step"``), also writes the flat concatenated
    tensor for each (who, tname) to disk via :func:`dump_tensors`. This
    captures the full LoRA weight / grad state for byte-level offline
    comparison across runs. ``step`` is required when dumping — pass the
    trainer's step counter (1-indexed).

    Returns a flat dict suitable for direct wandb.log.
    """
    if not is_enabled():
        return {}
    buckets: dict[tuple[str, str], list[torch.Tensor]] = {}
    for who, path, tname, t in iter_lora_tensors(model, adapter_names):
        if t is None:
            if tname.endswith("_grad") and grads:
                # Record grad-missing explicitly so we can spot silent no-op.
                buckets.setdefault((who, tname + "_missing_count"), []).append(
                    torch.tensor([1.0])
                )
            continue
        if not grads and tname.endswith("_grad"):
            continue
        buckets.setdefault((who, tname), []).append(t.detach().reshape(-1).to(torch.float32).cpu())

    out: dict[str, Any] = {}
    # Disk dumps are gated by NOUSNET_DIAG_DUMP and a caller-supplied phase tag.
    do_dump = is_dump_enabled() and dump_phase is not None and step is not None
    rank = int(os.environ.get("RANK", "0"))
    for (who, tname), parts in buckets.items():
        if tname.endswith("_missing_count"):
            out[f"diag/{who}/{tname}"] = float(sum(p.numel() for p in parts))
            continue
        flat = torch.cat(parts) if len(parts) > 1 else parts[0]
        stats = tensor_stats(flat)
        for k in _STR_KEYS:
            out[f"diag/{who}/{tname}_{k}"] = stats[k]
        for k in _FLOAT_KEYS:
            out[f"diag/{who}/{tname}_{k}"] = stats[k]
        if do_dump:
            # Write the full concatenated tensor to disk. Phase-prefixed key
            # so caller can drop pre_step / post_step alongside per-loss dumps
            # without colliding. Tensor is already on CPU + fp32 + flat.
            dump_tensors(
                step,
                rank,
                who,
                **{f"{dump_phase}__{tname}": flat},
            )
    return out
