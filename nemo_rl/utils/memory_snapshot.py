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
"""Opt-in CUDA allocator history and out-of-memory snapshots for policy workers.

A CUDA OOM in a training worker reports only the failing allocation. Knowing
which *live* tensors filled the GPU, and which code allocated them, is what
turns the report into a fix. PyTorch can record the allocation history with a
stack per block; this module wraps that in a small, config-gated hook:

* :func:`enable_memory_history` starts recording (once per process).
* :func:`snapshot_on_oom` decorates a worker method so that a
  ``torch.OutOfMemoryError`` prints the largest live blocks and writes a full
  snapshot (loadable with ``pickle`` and viewable with PyTorch's memory
  visualizer) before re-raising.

Recording costs a few percent of CPU time per allocation and is off by default.
"""

import functools
import os
import tempfile
import time
import uuid
from collections.abc import Mapping
from typing import Any, Callable, Optional

import torch
from pydantic import BaseModel

GIB = float(2**30)
DEFAULT_MAX_ENTRIES = 200_000


class MemorySnapshotConfig(BaseModel, extra="allow"):
    """``policy.dtensor_cfg.memory_snapshot`` settings (defaults live here)."""

    enabled: bool = False
    # Directory for the snapshot pickles; None means a subdirectory of the temp dir.
    directory: Optional[str] = None
    # Number of allocation events kept in the recorded history.
    max_entries: int = DEFAULT_MAX_ENTRIES
    # Record allocation stacks (needed to attribute blocks to code). When
    # False, the OOM dump still lists live block sizes.
    record_history: bool = True


def default_snapshot_directory() -> str:
    return os.path.join(tempfile.gettempdir(), "nemo_rl_memory_snapshots")


def enable_memory_history(max_entries: int = DEFAULT_MAX_ENTRIES) -> bool:
    """Start recording CUDA allocation history with stacks. Returns False without CUDA."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.memory._record_memory_history(max_entries=max_entries)
    except Exception as exc:  # pragma: no cover - depends on the torch build
        print(f"[MEMORY_SNAPSHOT] could not enable allocation history: {exc!r}")
        return False
    return True


def summarize_live_blocks(snapshot: Mapping[str, Any], top_k: int = 20) -> dict:
    """Summarize the live (active, allocated) blocks of a ``torch.cuda.memory._snapshot()``."""
    sizes: list[int] = []
    for segment in snapshot.get("segments", []):
        for block in segment.get("blocks", []):
            if block.get("state") == "active_allocated":
                sizes.append(int(block.get("size", 0)))
    sizes.sort(reverse=True)
    return {
        "live_blocks": len(sizes),
        "live_total_bytes": sum(sizes),
        "top_block_bytes": sizes[:top_k],
    }


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def report_oom(tag: str, cfg: MemorySnapshotConfig) -> Optional[str]:
    """Print the largest live blocks and dump a snapshot. Never raises.

    Returns the snapshot path on success, ``None`` when the dump failed.
    """
    rank = _rank()
    path: Optional[str] = None
    try:
        summary = summarize_live_blocks(torch.cuda.memory._snapshot())
        top = ",".join(f"{b / GIB:.2f}" for b in summary["top_block_bytes"])
        print(
            f"[MEMORY_SNAPSHOT] {tag} rank={rank} live_blocks={summary['live_blocks']} "
            f"live_total={summary['live_total_bytes'] / GIB:.2f} GiB top_blocks_GiB={top}",
            flush=True,
        )
        directory = cfg.directory or default_snapshot_directory()
        os.makedirs(directory, exist_ok=True)
        # rank + pid + a short random suffix: two ranks (or two nodes writing to
        # shared storage) that OOM in the same second must not clobber each other.
        candidate = os.path.join(
            directory,
            f"oom_{tag}_rank{rank}_pid{os.getpid()}_"
            f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}.pickle",
        )
        torch.cuda.memory._dump_snapshot(candidate)
        path = candidate
        print(
            f"[MEMORY_SNAPSHOT] wrote {path} (open with torch.cuda._memory_viz or "
            "https://pytorch.org/memory_viz)",
            flush=True,
        )
    except Exception as exc:
        print(f"[MEMORY_SNAPSHOT] {tag} rank={rank} dump failed: {exc!r}", flush=True)
    return path


def snapshot_on_oom(tag: str) -> Callable:
    """Decorate a worker method: on ``torch.OutOfMemoryError`` dump a snapshot, then re-raise.

    The worker must expose ``self._memory_snapshot_cfg`` (a
    :class:`MemorySnapshotConfig` or ``None``); when it is missing or disabled
    the method runs untouched.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            cfg = getattr(self, "_memory_snapshot_cfg", None)
            if cfg is None or not cfg.enabled:
                return fn(self, *args, **kwargs)
            try:
                return fn(self, *args, **kwargs)
            except torch.OutOfMemoryError:
                report_oom(tag, cfg)
                raise

        return wrapper

    return decorator
