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
"""NixlStorageUnit: a CPU Ray actor that owns one NIXL-registered DRAM slab.

Passive on the data path: clients WRITE and READ one-sided. The unit only
serves small control RPCs (alloc, resolve, release) and tracks pins and
refcounts so a region is never reused under an in-flight read.

Started by :func:`nemo_rl.data_plane.adapters.tq_nixl.initialize_nixl_storage`;
the store-side pieces it uses live in :mod:`nemo_rl.data_plane.nixl`.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import ray

from nemo_rl.data_plane.nixl import blob_format
from nemo_rl.data_plane.nixl.control import UnitServer
from nemo_rl.data_plane.nixl.allocator import SlabAllocator
from nemo_rl.data_plane.nixl.errors import UnitFull
from nemo_rl.data_plane.nixl.nixl_io import NixlEndpoint
from nemo_rl.distributed import numa_utils

log = logging.getLogger(__name__)


@dataclass
class BlobRec:
    off: int
    nbytes: int
    refs: int
    created_at: float
    last_resolve_at: float = 0.0
    pins: dict[str, float] = field(default_factory=dict)  # pin_id → expiry (monotonic)
    holds: int = 0  # checkpoints copying this blob right now (save_shard)

    def pin_expiry(self, now: float) -> float:
        live = [t for t in self.pins.values() if t > now]
        return max(live) if live else now


def _locked(fn):
    """Serialise a unit method under the unit lock.

    Unit state is touched by the ZMQ server thread and by (rare) Ray-called
    methods; one re-entrant lock serialises both.
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._mu:
            return fn(self, *args, **kwargs)

    return wrapper


@ray.remote(num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0)
class NixlStorageUnit:
    """Owns one registered DRAM slab.

    Clients reach it over ZMQ (``info()["zmq"]``)
    for alloc / resolve / release; Ray is only its lifecycle and admin path.
    """

    def __init__(
        self,
        unit_id: int,
        slab_bytes: int,
        *,
        read_pin_s: float,  # from tq_nixl.check_timeouts, the one source
        nixl_backend: str = "UCX",
        nixl_init_params: dict[str, Any] | None = None,
        directory: Any = None,
        require_rdma: bool = True,
        numa_mode: str = "auto",
        numa_index: int = 0,
    ) -> None:
        self.unit_id = int(unit_id)
        self.read_pin_s = float(read_pin_s)
        # Socket binding (nemo_rl/distributed/numa_utils.py): with numa=auto on a node
        # with >= 2 sockets, the k-th unit on the node (numa_index) pins its threads
        # and slab pages to socket k % sockets; a no-op on single-socket nodes.
        self.numa_node: int | None = None
        nixl_init_params = dict(nixl_init_params or {})
        sockets = numa_utils.cpu_nodes() if numa_mode == "auto" else {}
        if len(sockets) >= 2:
            node = sorted(sockets)[int(numa_index) % len(sockets)]
            numa_utils.bind_process(sockets[node])
            # No NIC restriction: limiting a unit to its socket's rails broke
            # cross-node endpoint creation with clients on other rails (86n job
            # 1072049: "UCX endpoint create failed", NIXL_ERR_BACKEND). CPU and
            # memory stay socket-local; NICs stay as UCX_NET_DEVICES allows.
            self.numa_node = node
        self._nics = nixl_init_params.get("device_list", "env")
        self.slab = np.empty(int(slab_bytes), dtype=np.uint8)
        if self.numa_node is not None:
            numa_utils.first_touch(self.slab)
        self.alloc_ = SlabAllocator(int(slab_bytes), on_free=self._wipe_footer)
        self.ep = NixlEndpoint(
            f"unit-{self.unit_id}-{uuid.uuid4().hex[:6]}",
            backend=nixl_backend,
            init_params=nixl_init_params,
            prog_thread=True,
            require_rdma=require_rdma,
        )
        self.slab_base = self.ep.register(self.slab)
        self.blobs: dict[str, BlobRec] = {}
        self._pins: dict[str, list[str]] = {}  # pin_id → blob_ids
        self.retired_bytes = 0  # regions of WRITEs that never reached DONE
        # Blobs released while a checkpoint still copies them: freed once the
        # save's holds drop (save_shard's finally).
        self._released_held: dict[str, BlobRec] = {}
        self.checkpointing = False
        self.clears_during_save = 0
        self._directory = directory  # admin/restore only; never on the op path
        self.node_id = ray.get_runtime_context().get_node_id()
        self._mu = threading.RLock()
        self._server = UnitServer(
            {
                "alloc": self.alloc,
                "abort": self.abort,
                "release_many": self.release_many,
                "resolve": self.resolve,
                "unpin": self.unpin,
            },
            self._mu,
        )

    # ------------------------------------------------------------------ identity
    @_locked
    def info(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "node_id": self.node_id,
            "numa": self.numa_node,
            "nics": self._nics,
            "zmq": self._server.address,
            "agent_name": self.ep.name,
            "agent_md": self.ep.metadata(),
            "slab_base": self.slab_base,
            "slab_bytes": int(self.slab.nbytes),
            "state": "ACTIVE",
        }

    @_locked
    def ping(self) -> int:
        return self.unit_id

    # ------------------------------------------------------------------ allocation
    @_locked
    def alloc(self, blob_id: str, nbytes: int, n_entries: int) -> int:
        now = time.monotonic()
        self.alloc_.reclaim(now)
        if blob_id in self.blobs:
            raise ValueError(f"blob {blob_id} already allocated")
        off = self.alloc_.alloc(
            nbytes
        )  # raises UnitFull; the client spills to the next unit
        self.blobs[blob_id] = BlobRec(
            off=off, nbytes=nbytes, refs=int(n_entries), created_at=now
        )
        return off

    @_locked
    def abort(self, blob_id: str, in_flight: bool = True) -> None:
        """Drop a blob whose WRITE failed; nothing was published.

        If the WRITE may still be running (``in_flight``), the region is never
        freed: a late WRITE would land in whatever blob reused it. The leak is
        counted in ``retired_bytes``.
        """
        rec = self.blobs.pop(blob_id, None)
        if rec is None:
            return
        if in_flight:
            self.retired_bytes += rec.nbytes
        else:
            self.alloc_.free(rec.off, rec.nbytes)

    @_locked
    def release(self, blob_id: str, n_entries: int) -> int:
        """Decrement the live-entry count; at zero quarantine the region.

        Returns the remaining count (0 when the blob is gone).
        """
        rec = self.blobs.get(blob_id)
        if rec is None:
            return 0
        rec.refs -= int(n_entries)
        if rec.refs > 0:
            return rec.refs
        self.blobs.pop(blob_id, None)
        if self.checkpointing:
            self.clears_during_save += 1  # TQ says no clears during a save
        if rec.holds:
            # A checkpoint is still copying it: free it when the save is done.
            self._released_held[blob_id] = rec
        else:
            self._free_region(rec)
        return 0

    def _free_region(self, rec: BlobRec) -> None:
        now = time.monotonic()
        # Always quarantine for read_pin_s, pinned or not: clients read with
        # cached (unit, offset) hints and no pin RPC, so a freed region must not
        # be reused while such a read could still be in flight. The blob tag in
        # the footer catches anything older than that: _wipe_footer clears it
        # before the region can be reused (a smaller blob would not overwrite it).
        release_at = max(rec.pin_expiry(now), now + self.read_pin_s)
        if release_at <= now:
            self.alloc_.free(rec.off, rec.nbytes)
        else:
            self.alloc_.quarantine(rec.off, rec.nbytes, release_at)

    def _wipe_footer(self, off: int, nbytes: int) -> None:
        """Zero a freed blob's footer so a stale hinted read fails its tag check."""
        if nbytes >= blob_format.FOOTER_SIZE:
            self.slab[off + nbytes - blob_format.FOOTER_SIZE : off + nbytes] = 0

    @_locked
    def release_many(self, items: list[tuple[str, int]]) -> None:
        """Batched :meth:`release` -- one RPC per unit per clear, sent without waiting."""
        for blob_id, n in items:
            self.release(str(blob_id), int(n))

    # ------------------------------------------------------------------ reads
    @_locked
    def resolve(self, blob_ids: list[str]) -> dict[str, Any]:
        """Return current slab offsets for blobs and pin them for ``read_pin_s``.

        Unknown blobs are listed under ``missing`` rather than raising, so the
        client can report exactly which keys are lost.
        """
        now = time.monotonic()
        pin_id = uuid.uuid4().hex[:12]
        expiry = now + self.read_pin_s
        offsets: dict[str, int] = {}
        missing: list[str] = []
        pinned: list[str] = []
        for b in blob_ids:
            rec = self.blobs.get(b)
            if rec is None:
                missing.append(b)
                continue
            rec.pins[pin_id] = expiry
            rec.last_resolve_at = now
            offsets[b] = rec.off
            pinned.append(b)
        if pinned:
            self._pins[pin_id] = pinned
        return {
            "pin_id": pin_id,
            "offsets": offsets,
            "missing": missing,
            "slab_base": self.slab_base,
        }

    @_locked
    def unpin(self, pin_id: str) -> None:
        for b in self._pins.pop(pin_id, []):
            rec = self.blobs.get(b)
            if rec is not None:
                rec.pins.pop(pin_id, None)

    # ------------------------------------------------------------------ introspection
    @_locked
    def scan(self) -> list[dict[str, Any]]:
        out = []
        for b, rec in self.blobs.items():
            tail = memoryview(self.slab)[rec.off : rec.off + rec.nbytes]
            entries = (
                blob_format.read_index(tail)
                if blob_format.footer_is_valid(tail)
                else []
            )
            out.append(
                {
                    "blob_id": b,
                    "off": rec.off,
                    "nbytes": rec.nbytes,
                    "refs": rec.refs,
                    "entries": len(entries),
                }
            )
        return out

    @_locked
    def stats(self) -> dict[str, Any]:
        self.alloc_.reclaim(time.monotonic())
        return {
            "unit_id": self.unit_id,
            "blobs": len(self.blobs),
            "used_bytes": self.alloc_.used_bytes,
            "free_bytes": self.alloc_.free_bytes,
            "quarantined_bytes": self.alloc_.quarantined_bytes,
            "fragments": self.alloc_.fragments,
            "pins": len(self._pins),
            "retired_bytes": self.retired_bytes,
        }

    @_locked
    def read_bytes(self, off: int, nbytes: int) -> bytes:
        """Test helper: copy bytes out of the slab through the actor."""
        return bytes(self.slab[off : off + nbytes])

    # ------------------------------------------------------------------ checkpoint (A2)
    def save_shard(self, path: str) -> dict[str, Any]:
        """Dump every live blob back to back into ``path``.

        Returns ``{"blobs": {blob_id: [off_in_file, nbytes, refs]}, "nbytes":
        total, "clears_during_save": n}``.

        The snapshot is taken under the unit lock; the file I/O runs without it
        (blocking syscalls, design §2.5), so puts, reads and clears keep being
        served. Each snapshotted blob carries a hold: one released meanwhile is
        freed only after it is copied. Every exit, including an exception at
        any step, drops exactly the holds taken (one outer try).
        """
        held: list[tuple[str, BlobRec, int]] = []
        try:
            with self._mu:
                self.checkpointing = True
                self.clears_during_save = 0
                for b, rec in self.blobs.items():
                    rec.holds += 1
                    held.append((b, rec, rec.refs))
            index: dict[str, list[int]] = {}
            tmp = path + ".tmp"
            pos = 0
            view = memoryview(self.slab)
            with open(tmp, "wb") as f:
                for b, rec, refs in held:
                    f.write(view[rec.off : rec.off + rec.nbytes])
                    index[b] = [pos, rec.nbytes, refs]
                    pos += rec.nbytes
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            with self._mu:
                for _, rec, _ in held:
                    rec.holds -= 1
                for b, rec in list(self._released_held.items()):
                    if rec.holds == 0:
                        del self._released_held[b]
                        self._free_region(rec)
                self.checkpointing = False
                clears = self.clears_during_save
        if clears:
            log.warning(
                "unit %d: %d blob(s) released during save_shard; TransferQueue "
                "does not support clears during a checkpoint",
                self.unit_id,
                clears,
            )
        return {"blobs": index, "nbytes": pos, "clears_during_save": clears}

    def load_shard(
        self, path: str, blobs: dict[str, list[int]]
    ) -> dict[str, list[str]]:
        """Read ``blobs`` (``blob_id -> [off_in_file, nbytes, refs]``) from ``path`` into fresh regions.

        Blobs that do not fit are returned under ``failed`` so the caller can
        spill them to another unit; already-present blobs count as loaded.

        Regions are allocated under the unit lock but filled without it; a
        blob is published only once its bytes are in place.
        """
        loaded: list[str] = []
        failed: list[str] = []
        view = memoryview(self.slab)
        with open(path, "rb", buffering=0) as f:
            for b, (off, nbytes, refs) in blobs.items():
                nbytes = int(nbytes)
                with self._mu:
                    if b in self.blobs:
                        loaded.append(b)
                        continue
                    self.alloc_.reclaim(time.monotonic())
                    try:
                        dst = self.alloc_.alloc(nbytes)
                    except UnitFull:
                        failed.append(b)
                        continue
                try:
                    f.seek(int(off))
                    target = view[dst : dst + nbytes]
                    done = 0
                    while done < nbytes:
                        n = f.readinto(target[done:])
                        if not n:
                            raise IOError(f"short read restoring blob {b} from {path}")
                        done += n
                except BaseException:
                    with self._mu:
                        self.alloc_.free(dst, nbytes)
                    raise
                with self._mu:
                    self.blobs[b] = BlobRec(
                        off=dst,
                        nbytes=nbytes,
                        refs=int(refs),
                        created_at=time.monotonic(),
                    )
                loaded.append(b)
        return {"loaded": loaded, "failed": failed}
