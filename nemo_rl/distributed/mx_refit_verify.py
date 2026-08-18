# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Parameter-equality verification for the ``mx_reshard`` refit path.

Every other refit transport has a way to check the weights it moved: SGLang has
``check_weights(compare)`` and the sparse transports emit ``delta_verify/*``.
``mx_reshard`` had only the end-to-end logprob metrics, which conflate refit
fidelity with Megatron-vs-vLLM implementation divergence and so cannot answer
"did the transport install the right bytes" on their own.

The check exploits a property of the first refit. A fresh run loads vLLM from the
HF checkpoint, and the trainer's Megatron weights are converted from that same
checkpoint with no optimizer step taken yet. So **the first refit should leave
every parameter it touches unchanged.** Anything it does change is refit or
conversion error, and it is named.

From the second refit on the trainer has genuinely moved, so parameters are
*expected* to change and the same records instead answer a different question:
which parameters the refit reaches at all. A parameter that never changes across
many steps of training is one the refit is silently not updating - the failure
mode the coverage counters are meant to catch, cross-checked against the values.

Fingerprints rather than copies: retaining a pre-refit copy of a 30B model's
shard costs ~15 GB per rank, while these are two allocation-free reductions.
"""

from __future__ import annotations

import json
import os

_ENV_FLAG = "MX_REFIT_VERIFY"


def enabled() -> bool:
    """Off unless asked for. The reductions are cheap but not free, and this runs
    on the refit critical path."""
    return os.environ.get(_ENV_FLAG, "0") not in ("", "0", "false", "False")


def fingerprint(tensor) -> tuple[int, int, int]:
    """``(numel, sum of all raw bytes, sum of the high byte of each element)``.

    Computed over the raw bytes, so it is sensitive to any bit change rather than
    only to changes large enough to move a float statistic. Both sums accumulate
    in int64 via ``sum(dtype=...)``, which avoids materializing a promoted copy of
    a tensor that can be hundreds of MB.

    The second statistic strides by element size to pick out each element's most
    significant byte -- sign, exponent and the top mantissa bits for a
    little-endian float -- so a change confined to the high bits cannot cancel
    against an offsetting change elsewhere in the flat byte sum.

    This is a fingerprint, not a hash: collisions are possible in principle. Two
    independent statistics make an accidental collision unlikely, and a *silent*
    collision would have to survive both.
    """
    import torch

    flat = tensor.detach().reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    raw = flat.view(torch.uint8)
    elsize = max(1, raw.numel() // max(1, flat.numel()))
    high = raw[elsize - 1 :: elsize] if elsize > 1 else raw
    return (
        int(flat.numel()),
        int(raw.sum(dtype=torch.int64).item()),
        int(high.sum(dtype=torch.int64).item()),
    )


def fingerprint_model(model) -> dict[str, tuple[int, int, int]]:
    """Fingerprint every parameter, or ``{}`` if the model cannot be walked.

    Never raises: this is verification, and a verification failure must not be
    able to fail the refit it is verifying.
    """
    try:
        return {name: fingerprint(p) for name, p in model.named_parameters()}
    except Exception:  # noqa: BLE001 - see above
        return {}


def compare(
    before: dict[str, tuple[int, int, int]],
    after: dict[str, tuple[int, int, int]],
) -> dict:
    """Which parameters the refit changed, and which it left alone."""
    shared = [name for name in after if name in before]
    changed = [name for name in shared if before[name] != after[name]]
    unchanged = [name for name in shared if before[name] == after[name]]
    return {
        "params_compared": len(shared),
        "params_changed": len(changed),
        "params_unchanged": len(unchanged),
        "changed_sample": sorted(changed)[:10],
        "unchanged_sample": sorted(unchanged)[:10],
    }


def report(step: int, rank: int, before: dict, after: dict) -> dict:
    """Emit one ``MX_REFIT_VERIFY`` record and return it.

    ``first_refit`` marks the record where "unchanged" is the passing outcome, so a
    later reader does not have to reconstruct which expectation applied.
    """
    record = {
        "schema": "mx-refit-verify-v1",
        "step": step,
        "rank": rank,
        "first_refit": step <= 1,
        **compare(before, after),
    }
    print("MX_REFIT_VERIFY " + json.dumps(record), flush=True)
    return record
