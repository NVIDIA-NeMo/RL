"""Exact initial-LoRA export/import for matched single-vs-multi probes.

This is a diagnostic-only, environment-gated shim.  A canonical fresh
``MultiLinearLoRA`` run can export each slot's *local FSDP shard* immediately
post-setup; true standalone ``LinearLoRA`` runs (or another multi run) then
copy those exact bytes before the first forward/backward.

Environment variables
---------------------
``NOUSNET_INIT_EXPORT_DIR``
    Export all multi slots to this directory.  Files are written atomically as
    ``rank_RR_slot_II.pt``.
``NOUSNET_INIT_IMPORT_DIR``
    Import from an export directory.
``NOUSNET_INIT_IMPORT_SLOT``
    Required for a true-single model; selects the canonical multi slot.  Omit
    for a multi model to import all slots.

The helper must run after model/FSDP/PEFT setup and before the first training
step.  It copies parameter data in-place, so the optimizer's parameter
references remain valid and its state is still empty on a fresh probe.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", "0") or 0)


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if hasattr(t, "to_local") else t


def _lora_kind(module: nn.Module) -> Literal["single", "multi"] | None:
    """Classify by storage contract, not fragile runtime class-name equality."""
    A = getattr(module, "lora_A", None)
    B = getattr(module, "lora_B", None)
    if A is None or B is None:
        return None
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if A.ndim == 3 and B.ndim == 3:
            return "multi"
    Aw = getattr(A, "weight", None)
    Bw = getattr(B, "weight", None)
    if isinstance(Aw, torch.Tensor) and isinstance(Bw, torch.Tensor):
        if Aw.ndim == 2 and Bw.ndim == 2:
            return "single"
    return None


def _storage(module: nn.Module, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    if kind == "multi":
        return module.lora_A, module.lora_B
    return module.lora_A.weight, module.lora_B.weight


def _canonical_name(name: str) -> str:
    # FSDP/composable wrappers may expose one or more implementation prefixes
    # without changing the underlying model traversal.
    parts = [p for p in name.split(".") if p not in {"_fsdp_wrapped_module", "module"}]
    return ".".join(parts)


def _inventory(model: nn.Module) -> tuple[str, list[tuple[str, nn.Module]]]:
    rows: list[tuple[str, nn.Module]] = []
    kinds: set[str] = set()
    for name, module in model.named_modules():
        kind = _lora_kind(module)
        if kind is not None:
            kinds.add(kind)
            rows.append((_canonical_name(name), module))
    if not rows:
        raise RuntimeError("no LoRA modules found for exact-init transfer")
    if len(kinds) != 1:
        raise RuntimeError(f"mixed LoRA storage kinds are unsupported: {sorted(kinds)}")
    # Match source/destination by stable canonical FQN, not traversal order.
    rows.sort(key=lambda item: item[0])
    return next(iter(kinds)), rows


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def export_initial_lora(model: nn.Module, output_dir: str | os.PathLike[str]) -> None:
    kind, rows = _inventory(model)
    if kind != "multi":
        raise RuntimeError("canonical init export requires a MultiLinearLoRA model")

    rank = _rank()
    first_A, _ = _storage(rows[0][1], kind)
    first_local = _local(first_A.detach())
    n_adapters = int(first_local.shape[0])
    root = Path(output_dir)

    for slot in range(n_adapters):
        tensors: list[dict[str, Any]] = []
        for ordinal, (name, module) in enumerate(rows):
            A, B = _storage(module, kind)
            Al = _local(A.detach())
            Bl = _local(B.detach())
            if int(Al.shape[0]) != n_adapters or int(Bl.shape[0]) != n_adapters:
                raise RuntimeError(f"adapter-count mismatch at {name}: A={tuple(Al.shape)} B={tuple(Bl.shape)}")
            tensors.append(
                {
                    "ordinal": ordinal,
                    "source_name": name,
                    "in_features": int(getattr(module, "in_features", -1)),
                    "out_features": int(getattr(module, "out_features", -1)),
                    "lora_A": Al[slot].contiguous().cpu(),
                    "lora_B": Bl[slot].contiguous().cpu(),
                }
            )
        payload: dict[str, Any] = {
            "format": 2,
            "source_kind": "multi",
            "rank": rank,
            "slot": slot,
            "n_adapters": n_adapters,
            "module_count": len(tensors),
            "tensors": tensors,
        }
        _atomic_torch_save(payload, root / f"rank_{rank:02d}_slot_{slot:02d}.pt")

    _barrier()
    print(
        f"[INIT_TRANSFER] exported {len(rows)} LoRA modules x {n_adapters} slots "
        f"for rank {rank} to {root}",
        flush=True,
    )


@torch.no_grad()
def import_initial_lora(
    model: nn.Module,
    input_dir: str | os.PathLike[str],
    *,
    single_slot: int | None,
) -> None:
    kind, rows = _inventory(model)
    rank = _rank()
    root = Path(input_dir)

    if kind == "single":
        if single_slot is None:
            raise RuntimeError("NOUSNET_INIT_IMPORT_SLOT is required for true-single import")
        slots = [int(single_slot)]
    else:
        if single_slot is not None:
            raise RuntimeError("omit NOUSNET_INIT_IMPORT_SLOT when importing all slots into multi")
        A0, _ = _storage(rows[0][1], kind)
        slots = list(range(int(_local(A0.detach()).shape[0])))

    payloads: dict[int, dict[str, Any]] = {}
    for slot in slots:
        path = root / f"rank_{rank:02d}_slot_{slot:02d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing exact-init shard: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != 2 or int(payload.get("rank", -1)) != rank or int(payload.get("slot", -1)) != slot:
            raise RuntimeError(f"invalid exact-init payload metadata in {path}")
        if int(payload.get("module_count", -1)) != len(rows) or len(payload.get("tensors", [])) != len(rows):
            raise RuntimeError(
                f"LoRA module-count mismatch for {path}: source={payload.get('module_count')} "
                f"destination={len(rows)}"
            )
        payloads[slot] = payload

    # Match by canonical FQN + in/out dimensions + exact local tensor shapes.
    # A deterministic ordinal is retained as a corruption check, but never used
    # to silently pair differently-named modules.
    for ordinal, (name, module) in enumerate(rows):
        A, B = _storage(module, kind)
        Al = _local(A.data)
        Bl = _local(B.data)
        check_slots = slots if kind == "multi" else [slots[0]]
        for slot in check_slots:
            src = payloads[slot]["tensors"][ordinal]
            if int(src.get("ordinal", -1)) != ordinal:
                raise RuntimeError(f"ordinal mismatch at destination {name}: source row={src.get('ordinal')}")
            if str(src.get("source_name")) != name:
                raise RuntimeError(
                    f"canonical LoRA FQN mismatch at ordinal {ordinal}: "
                    f"source={src.get('source_name')} destination={name}"
                )
            if int(src.get("in_features", -1)) != int(getattr(module, "in_features", -2)):
                raise RuntimeError(f"in_features mismatch at destination {name}")
            if int(src.get("out_features", -1)) != int(getattr(module, "out_features", -2)):
                raise RuntimeError(f"out_features mismatch at destination {name}")
            dst_A = Al[slot] if kind == "multi" else Al
            dst_B = Bl[slot] if kind == "multi" else Bl
            if tuple(dst_A.shape) != tuple(src["lora_A"].shape) or tuple(dst_B.shape) != tuple(src["lora_B"].shape):
                raise RuntimeError(
                    f"shape mismatch at destination {name}, source {src.get('source_name')}: "
                    f"dst A/B={tuple(dst_A.shape)}/{tuple(dst_B.shape)} "
                    f"src={tuple(src['lora_A'].shape)}/{tuple(src['lora_B'].shape)}"
                )
            dst_A.copy_(src["lora_A"].to(device=dst_A.device, dtype=dst_A.dtype))
            dst_B.copy_(src["lora_B"].to(device=dst_B.device, dtype=dst_B.dtype))

    _barrier()
    slot_desc = str(slots[0]) if kind == "single" else "all"
    print(
        f"[INIT_TRANSFER] imported exact initial LoRA bytes into {kind} model "
        f"slot={slot_desc}, modules={len(rows)}, rank={rank}, source={root}",
        flush=True,
    )


def maybe_transfer_initial_lora(model: nn.Module) -> None:
    """Environment-dispatched entry point called by the NeMo-RL worker."""
    export_dir = os.environ.get("NOUSNET_INIT_EXPORT_DIR", "").strip()
    import_dir = os.environ.get("NOUSNET_INIT_IMPORT_DIR", "").strip()
    if export_dir and import_dir:
        raise RuntimeError("set only one of NOUSNET_INIT_EXPORT_DIR / NOUSNET_INIT_IMPORT_DIR")
    if export_dir:
        export_initial_lora(model, export_dir)
    elif import_dir:
        raw_slot = os.environ.get("NOUSNET_INIT_IMPORT_SLOT", "").strip()
        slot = int(raw_slot) if raw_slot else None
        import_initial_lora(model, import_dir, single_slot=slot)
