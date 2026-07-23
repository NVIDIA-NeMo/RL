"""DTensor compat helpers shared by multi-LoRA modules.

Kept lightweight: stdlib + torch only, no nemo_automodel / nemo_rl deps,
so this module imports cleanly on the Ray driver process where the
worker-only packages aren't available.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch


@lru_cache(maxsize=1)
def optional_dtensor_type() -> type | None:
    """Return ``torch.distributed.tensor.DTensor`` or ``None``.

    Cached so repeated lookups are free. Returns ``None`` when DTensor
    isn't importable (e.g., torch build without distributed.tensor).
    Callers use this for ``isinstance`` checks only; required
    distributed semantics should still fail loudly at their own boundary.
    """
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return None
    return DTensor


def is_dtensor(value: Any) -> bool:
    """``isinstance(value, DTensor)`` with the DTensor-missing case folded in."""
    cls = optional_dtensor_type()
    return cls is not None and isinstance(value, cls)


def to_local(t: torch.Tensor) -> torch.Tensor:
    """``t.to_local()`` if ``t`` is a DTensor, else ``t`` unchanged."""
    return t.to_local() if is_dtensor(t) else t
