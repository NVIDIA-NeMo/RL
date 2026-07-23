"""Full-fidelity tensor dump: every leaf module, every step, every microbatch.

This is the "log absolutely everything" diagnostic. For each leaf module M in
``model.named_modules()`` it installs:

- forward_pre_hook:        saves M's inputs to disk
- forward_hook:            saves M's output(s) to disk
- full_backward_pre_hook:  saves grad_output to disk
- full_backward_hook:      saves grad_input to disk

For each parameter it saves ``p.data`` and ``p.grad`` once per step (called
explicitly from outside the hooks). Optimizer state (``exp_avg``,
``exp_avg_sq``, etc) is also dumped per step.

Output layout::

    ${NOUSNET_DUMP_DIR}/step_NNNN/mb_MMM/forward/<module_fqn>/<inN_or_out>.pt
    ${NOUSNET_DUMP_DIR}/step_NNNN/mb_MMM/backward/<module_fqn>/<grad_outN_or_grad_inN>.pt
    ${NOUSNET_DUMP_DIR}/step_NNNN/params/<param_fqn>/data.pt
    ${NOUSNET_DUMP_DIR}/step_NNNN/params/<param_fqn>/grad.pt
    ${NOUSNET_DUMP_DIR}/step_NNNN/opt/<param_fqn>/<state_key>.pt

A microbatch counter is auto-advanced by the root module's forward_pre_hook so
each forward pass through the top-level model gets a fresh ``mb_MMM`` dir.

Gating: only fires when ``NOUSNET_DUMP_EVERYTHING=1``. When that env var is
unset (default), the install function is a no-op and there's zero runtime cost.

Disk-pressure caveats:

- Every leaf forward output is saved per microbatch. For Super-120B with ~1k
  Linear layers and seq=1024 hidden=12288 BF16, expect ~25 MB per Linear
  output, ~25 GB per microbatch, ~200 GB per training step over 8 ranks.
- You almost certainly want to constrain steps and microbatches before running
  this at 120B. The diag config that goes with this module sets
  ``sft.max_num_steps=1`` so we get one full step's worth of data and stop.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


_HANDLES: list = []
_STEP: dict[str, int] = {"step": 0, "mb": -1}  # mb starts at -1; incremented on each root-forward entry


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            return int(torch.distributed.get_rank())
        except Exception:
            pass
    return int(os.environ.get("RANK", "0") or 0)


def _is_enabled() -> bool:
    return os.environ.get("NOUSNET_DUMP_EVERYTHING", "0") == "1"


def _dump_root() -> Path:
    root = os.environ.get("NOUSNET_DUMP_DIR")
    if not root:
        run_dir = os.environ.get("NOUSNET_RUN_DIR", ".")
        root = str(Path(run_dir) / "dump_everything")
    return Path(root)


def _to_local(t: torch.Tensor) -> torch.Tensor:
    """Get the local shard of a DTensor; identity for plain tensors.

    We never call ``full_tensor()``/all_gather here because the hook runs on
    every leaf and gathering would be cripplingly slow at 120B. Each rank's
    dump is a snapshot of its own local shard; cross-rank reconstruction is
    done post-hoc by the analyzer reading rank_00/..rank_07 files.
    """
    if hasattr(t, "to_local"):
        try:
            return t.to_local()
        except Exception:
            return t
    return t


def _safe_save(obj: Any, path: Path) -> None:
    """Save anything tensor-like to disk, robust to DTensor wrappers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, torch.Tensor):
        try:
            t = _to_local(obj.detach()).contiguous().cpu()
            torch.save(t, path)
        except Exception as e:
            # Don't take training down over a dump failure
            with open(path.with_suffix(".err"), "w") as f:
                f.write(f"save_failed: {type(e).__name__}: {e}\n")
    elif isinstance(obj, (int, float, bool)):
        torch.save(obj, path)
    else:
        # Best-effort: save repr for non-tensor scalars (e.g. optimizer step ints)
        try:
            torch.save(obj, path)
        except Exception:
            with open(path.with_suffix(".repr"), "w") as f:
                f.write(repr(obj))


def _sanitize(name: str) -> str:
    """FS-safe module path. Keeps dots since they aid readability."""
    return name.replace("/", "_") or "_root"


def _is_leaf(mod: nn.Module) -> bool:
    """Strictly: no submodules."""
    return next(mod.children(), None) is None


def _save_tensor_tuple(t: Any, dest_dir: Path, prefix: str) -> None:
    """Save a tensor or a tuple/list/dict of tensors into dest_dir.

    Naming:
      bare tensor                -> {prefix}.pt
      tuple/list of N tensors    -> {prefix}_{i}.pt
      dict                       -> {prefix}_{key}.pt
      None / non-tensor          -> {prefix}.none.txt
    """
    if t is None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        with open(dest_dir / f"{prefix}.none.txt", "w") as f:
            f.write("None\n")
        return
    if isinstance(t, torch.Tensor):
        _safe_save(t, dest_dir / f"{prefix}.pt")
        return
    if isinstance(t, (tuple, list)):
        for i, x in enumerate(t):
            _save_tensor_tuple(x, dest_dir, f"{prefix}_{i}")
        return
    if isinstance(t, dict):
        for k, v in t.items():
            _save_tensor_tuple(v, dest_dir, f"{prefix}_{k}")
        return
    # Non-tensor scalar; write repr
    dest_dir.mkdir(parents=True, exist_ok=True)
    with open(dest_dir / f"{prefix}.repr.txt", "w") as f:
        f.write(repr(t)[:4096])


def _step_dir() -> Path:
    return _dump_root() / f"step_{_STEP['step']:04d}" / f"rank_{_rank():02d}"


def _mb_dir() -> Path:
    mb = max(_STEP["mb"], 0)
    return _step_dir() / f"mb_{mb:03d}"


def install_dump_hooks(model: nn.Module) -> int:
    """Install pre/post forward + backward hooks on every leaf module.

    Returns the number of leaf modules hooked. Idempotent: clears previous
    handles. No-op when ``NOUSNET_DUMP_EVERYTHING != "1"``.
    """
    remove_dump_hooks()
    if not _is_enabled():
        return 0

    rank = _rank()
    # Root forward_pre_hook advances the microbatch counter so every leaf hook
    # files into the same mb directory for that pass. We hook the top-level
    # model itself for this; it isn't a leaf so it won't be re-hooked below.
    def _root_pre(_m, _inp):
        _STEP["mb"] += 1
        # Also dump the model-level inputs (input_ids etc.)
        d = _mb_dir() / "model_input"
        _save_tensor_tuple(_inp, d, "in")
    _HANDLES.append(model.register_forward_pre_hook(_root_pre))

    def _root_post(_m, _inp, out):
        d = _mb_dir() / "model_output"
        _save_tensor_tuple(out, d, "out")
    _HANDLES.append(model.register_forward_hook(_root_post))

    n_leaves = 0
    for name, mod in model.named_modules():
        if mod is model:
            continue
        if not _is_leaf(mod):
            continue
        n_leaves += 1
        safe = _sanitize(name)

        def _pre(_m, inp, name=safe):
            d = _mb_dir() / "forward" / name
            _save_tensor_tuple(inp, d, "in")

        def _post(_m, inp, out, name=safe):
            d = _mb_dir() / "forward" / name
            _save_tensor_tuple(out, d, "out")

        def _bpre(_m, gout, name=safe):
            d = _mb_dir() / "backward" / name
            _save_tensor_tuple(gout, d, "grad_out")

        def _bpost(_m, ginp, gout, name=safe):
            d = _mb_dir() / "backward" / name
            _save_tensor_tuple(ginp, d, "grad_in")

        _HANDLES.append(mod.register_forward_pre_hook(_pre))
        _HANDLES.append(mod.register_forward_hook(_post))
        try:
            _HANDLES.append(mod.register_full_backward_pre_hook(_bpre))
        except Exception:
            pass
        try:
            _HANDLES.append(mod.register_full_backward_hook(_bpost))
        except Exception:
            pass

    if rank == 0:
        print(
            f"[DUMP_EVERYTHING] installed hooks on {n_leaves} leaf modules; "
            f"root forward_pre_hook will advance microbatch counter; "
            f"output -> {_dump_root()}",
            flush=True,
        )
    return n_leaves


def remove_dump_hooks() -> None:
    for h in _HANDLES:
        try:
            h.remove()
        except Exception:
            pass
    _HANDLES.clear()


def set_step(step: int) -> None:
    """Set the step counter and reset the microbatch counter for this step.

    Called from the train loop (or the worker) once per ``policy.train()``
    iteration so the dump tree segregates by step.
    """
    _STEP["step"] = int(step)
    _STEP["mb"] = -1


def get_step() -> int:
    return _STEP["step"]


def dump_params_and_opt(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    phase: str,
) -> None:
    """Dump every parameter's data + grad, plus optimizer state per param.

    ``phase`` is a free-form tag (e.g. ``"pre_step"``, ``"post_forward"``,
    ``"post_backward"``, ``"post_optstep"``) used to scope the dir.
    No-op when ``NOUSNET_DUMP_EVERYTHING != "1"``.
    """
    if not _is_enabled():
        return
    root = _step_dir() / f"params_{phase}"
    root.mkdir(parents=True, exist_ok=True)
    # Map param->fqn for opt-state lookup
    param_to_name: dict[int, str] = {}
    for name, p in model.named_parameters():
        param_to_name[id(p)] = name
        d = root / _sanitize(name)
        _safe_save(p.data, d / "data.pt")
        if p.grad is not None:
            _safe_save(p.grad, d / "grad.pt")
        else:
            with open(d / "grad.none.txt", "w") as f:
                f.write("None\n")

    if optimizer is None:
        return
    opt_root = _step_dir() / f"opt_{phase}"
    opt_root.mkdir(parents=True, exist_ok=True)
    for group in optimizer.param_groups:
        for p in group.get("params", []):
            name = param_to_name.get(id(p), f"unknown_{id(p):x}")
            state = optimizer.state.get(p, {})
            d = opt_root / _sanitize(name)
            for k, v in state.items():
                _safe_save(v, d / f"{k}.pt")
