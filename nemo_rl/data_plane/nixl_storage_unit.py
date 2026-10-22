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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import ray

from nemo_rl.data_plane.nixl import blob_format
from nemo_rl.data_plane.nixl.control import UnitServer
from nemo_rl.data_plane.nixl.allocator import SlabAllocator, align_up
from nemo_rl.data_plane.nixl.errors import DataPlaneBusy, UnitFull
from nemo_rl.data_plane.nixl.nixl_io import NixlEndpoint
from nemo_rl.distributed import numa_utils

log = logging.getLogger(__name__)

# IO jobs (spill writes, stage reads) queued or running at once. Each holds an IO-area
# region, so the area bounds it; this caps the count when blobs are tiny.
_MAX_IO_JOBS = 256


@dataclass
class BlobRec:
    off: int
    nbytes: int
    refs: int
    created_at: float
    last_resolve_at: float = 0.0
    pins: dict[str, float] = field(default_factory=dict)  # pin_id → expiry (monotonic)
    holds: int = 0  # copies of the slab region running now (save_shard)
    # SSD tier (only with ssd enabled). A blob lives in the put area (in_slab), or
    # in the IO area while its put is written to disk (staged_off set, disk None),
    # or on disk (disk = fd of an already unlinked file) with an optional staged
    # copy in the IO area (staged_off).
    in_slab: bool = True
    disk: int | None = None
    staged_off: int | None = (
        None  # offset in the IO area: the put's region or the staged copy
    )
    staging: bool = False  # an IO job (spill write or stage read) owns staged_off now
    stage_failed: bool = (
        False  # the last stage read failed; the next resolve says so once
    )
    staged_at: float = 0.0
    picked_up: bool = False  # a reader pinned the staged copy since it was staged
    wait_since: float = 0.0  # first pending reply of the current stage

    def pin_expiry(self, now: float) -> float:
        live = [t for t in self.pins.values() if t > now]
        return max(live) if live else now


def _free_at(a: SlabAllocator, off: int, nbytes: int, at: float, now: float) -> None:
    if at <= now:
        a.free(off, nbytes)
    else:
        a.quarantine(off, nbytes, at)


def _close(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


def _write_all(fd: int, view: memoryview) -> None:
    done = 0
    while done < len(view):
        n = os.pwrite(fd, view[done:], done)
        if n <= 0:
            raise OSError("write made no progress")
        done += n


_COPY_CHUNK = 64 << 20


def _copy_out(fd: int, f, nbytes: int) -> None:
    """Append the first ``nbytes`` of file ``fd`` to open file ``f``."""
    done = 0
    while done < nbytes:
        chunk = os.pread(fd, min(nbytes - done, _COPY_CHUNK), done)
        if not chunk:
            raise OSError("short read copying a spilled blob")
        f.write(chunk)
        done += len(chunk)


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
        ssd_dir: str | None = None,
        staging_bytes: int = 0,
        pickup_s: float = 10.0,
    ) -> None:
        self.unit_id = int(unit_id)
        self.read_pin_s = float(read_pin_s)
        # Socket binding (nemo_rl/distributed/numa_utils.py): with numa=auto on a node
        # with >= 2 sockets, the k-th unit on the node (numa_index) pins its threads
        # and slab pages to socket k % sockets; a no-op on single-socket nodes.
        self.numa_node: int | None = None
        nixl_init_params = dict(nixl_init_params or {})
        auto = numa_mode == "auto" and os.environ.get("NRL_DISABLE_NUMA_BINDING") != "1"
        sockets = numa_utils.cpu_nodes() if auto else {}
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
        self._mu = threading.RLock()
        self._init_ssd(int(slab_bytes), ssd_dir, int(staging_bytes), float(pickup_s))
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
        self._server = UnitServer(
            {
                "alloc": self.alloc,
                "commit": self.commit,
                "abort": self.abort,
                "release_many": self.release_many,
                "resolve": self.resolve,
                "unpin": self.unpin,
            },
            self._mu,
        )
        self._start_ticker()

    # ------------------------------------------------------------------ SSD tier
    # Spill, not demote. Data is read oldest first (the FIFO contract), so a put
    # the put area cannot take goes to disk from birth and data already in the put
    # area never moves: no victims, no pin drain, no eviction policy. The put's
    # WRITE lands in the IO area (carved from the slab, as the staging area was);
    # its commit is "pending" while ONE worker thread writes the file, and
    # "done" once it is on disk. A read of a disk blob gets "pending" while the
    # same worker reads the file into the IO area, then reads it pinned. Disk
    # I/O is plain blocking file I/O on that worker, outside _mu: a hung disk
    # stalls spills and cold reads, never the put area. An IO job owns its
    # IO-area region until it ends, so a region is never freed under it.
    def _init_ssd(
        self,
        slab_bytes: int,
        ssd_dir: str | None,
        staging_bytes: int,
        pickup_s: float,
    ) -> None:
        self.ssd_dir: str | None = None
        self.staging_: SlabAllocator | None = None
        self.staging_base = slab_bytes
        self.pickup_s = pickup_s
        self.ssd_stats: dict[str, float] = {
            "spilled_bytes": 0,
            "spills": 0,
            "spill_failures": 0,
            "staged_bytes": 0,
            "stages": 0,
            "stage_failures": 0,
            "stage_wait_ms_max": 0.0,
            "busy_replies": 0,
        }
        self._io_jobs = 0  # IO jobs queued or running
        self._max_io_jobs = _MAX_IO_JOBS
        self._staged: dict[str, BlobRec] = {}  # blobs with a staged copy
        if not ssd_dir:
            self.alloc_ = SlabAllocator(slab_bytes, on_free=self._wipe_footer)
            return
        main = (slab_bytes - staging_bytes) // 4096 * 4096
        if staging_bytes <= 0 or main <= 0:
            raise ValueError(
                f"ssd staging_bytes={staging_bytes} must be in (0, slab {slab_bytes})"
            )
        self.alloc_ = SlabAllocator(main, on_free=self._wipe_footer)
        self.staging_ = SlabAllocator(slab_bytes - main)
        self.staging_base = main
        self.ssd_dir = os.path.join(
            ssd_dir, f"nixl-unit-{self.unit_id}-{uuid.uuid4().hex[:6]}"
        )
        os.makedirs(self.ssd_dir, exist_ok=True)
        self._worker = ThreadPoolExecutor(1, thread_name_prefix="nvdp-io")

    def _start_ticker(self) -> None:
        """Start the SSD ticker; call once the unit is fully built."""
        if self.ssd_dir is not None:
            threading.Thread(
                target=self._tick, name="nvdp-ssd-tick", daemon=True
            ).start()

    def _tick(self, every: float = 1.0, report_s: float = 30.0) -> None:
        """SSD ticker thread: free what has aged out, print stats when they change."""
        last, last_at = None, 0.0
        while True:
            time.sleep(every)
            try:
                with self._mu:
                    self._reclaim(time.monotonic())
            except Exception:  # noqa: BLE001 - keep ticking; the next tick retries
                log.exception("unit %d: SSD tick failed", self.unit_id)
                continue
            now = time.monotonic()
            if now - last_at < report_s:
                continue
            last_at = now
            with self._mu:
                st = {**self._ssd_stats(), "slab_used": self.alloc_.used_bytes}
            if st != last:
                print(f"[nvdp-ssd] unit {self.unit_id} stats {st}", flush=True)
                last = st

    def _ssd_stats(self) -> dict[str, Any]:
        return {
            **self.ssd_stats,
            "staging_used_bytes": self.staging_.used_bytes,
            "disk_only_blobs": sum(r.disk is not None for r in self.blobs.values()),
        }

    def _reclaim(self, now: float) -> None:
        self.alloc_.reclaim(now)
        if self.staging_ is not None:
            self.staging_.reclaim(now)

    def _alloc_io(self, nbytes: int, now: float) -> int:
        """IO-area region for a put the put area cannot take.

        Busy if something will free room (jobs, staged copies, pins, quarantine),
        UnitFull if nothing will.
        """
        io = self.staging_
        busy = False
        if align_up(nbytes, io.align) <= io.capacity:
            try:
                return io.alloc(nbytes)
            except UnitFull:
                self._evict_staged(nbytes, now)
            try:
                return io.alloc(nbytes)
            except UnitFull:
                busy = io.used_bytes > 0
        if busy or self.alloc_.quarantined_bytes:
            self.ssd_stats["busy_replies"] += 1
            raise DataPlaneBusy(
                f"unit {self.unit_id} has no room for {nbytes} B yet", reason="pressure"
            )
        raise UnitFull(
            f"unit {self.unit_id}: {nbytes} B fits neither the put area nor the IO area"
        )

    def _submit_io(self, fn: Any, b: str, rec: BlobRec) -> None:
        """Queue an IO job that owns ``rec.staged_off`` (caller holds ``_mu``).

        Raises before any state changes, so a failed submit leaves nothing to undo.
        """
        self._worker.submit(fn, b, rec)
        self._io_jobs += 1
        rec.staging = True

    def _spill(self, b: str, rec: BlobRec) -> None:
        """IO worker: write a put's IO-area region to a file, then publish it as a disk blob."""
        fd = None
        ok = False
        try:
            fd = self._new_file()
            at = self.staging_base + rec.staged_off
            _write_all(fd, memoryview(self.slab)[at : at + rec.nbytes])
            ok = True
        except Exception as e:  # noqa: BLE001 - anything else would leave the job owning the region
            log.warning("unit %d: spilling %s failed: %s", self.unit_id, b, e)
        now = time.monotonic()
        with self._mu:
            self._io_jobs -= 1
            rec.staging = False
            gone = self.blobs.get(b) is not rec
            self._drop_staged(b, rec, now, now)  # nobody ever read the IO region
            if ok and not gone:
                rec.disk = fd
                self.ssd_stats["spilled_bytes"] += rec.nbytes
                self.ssd_stats["spills"] += 1
            elif not gone:
                # Not acknowledged, so nothing is lost: the put fails ("lost") and
                # the client tries another unit.
                del self.blobs[b]
                self.ssd_stats["spill_failures"] += 1
        if fd is not None and not (ok and not gone):
            _close(fd)

    def _new_file(self) -> int:
        """Open a fresh file in the SSD dir and unlink it at once; the fd keeps it.

        Nothing is left on the node's disk when the unit dies, however it dies.
        Named by a fresh uuid, not the client-supplied blob id, so no id can
        steer the path.
        """
        path = os.path.join(self.ssd_dir, f"{uuid.uuid4().hex}.blob")
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        os.unlink(path)
        return fd

    def _start_stage(self, b: str, rec: BlobRec, now: float) -> None:
        """Stage a disk blob into the IO area, if room can be found."""
        if not rec.wait_since:
            rec.wait_since = now
        if self._io_jobs >= self._max_io_jobs:
            return
        try:
            off = self.staging_.alloc(rec.nbytes)
        except UnitFull:
            self._evict_staged(rec.nbytes, now)
            try:
                off = self.staging_.alloc(rec.nbytes)
            except UnitFull:
                return  # IO area full: stays pending until pins end / copies age
        rec.staged_off = off
        try:
            self._submit_io(self._stage, b, rec)
        except BaseException:
            rec.staged_off = None
            self.staging_.free(off, rec.nbytes)
            raise
        self._staged[b] = rec

    def _evict_staged(self, nbytes: int, now: float) -> None:
        """Drop unpinned staged copies, oldest first.

        A copy no reader has pinned yet is kept for ``pickup_s`` after its
        stage, so the reader polling for it finds it (design §4.2).
        """
        need = align_up(nbytes, self.staging_.align)
        old = sorted(
            (r.staged_at, b, r)
            for b, r in self._staged.items()
            if not r.staging
            and r.pin_expiry(now) <= now
            and (r.picked_up or now - r.staged_at >= self.pickup_s)
        )
        for _, b, r in old:
            if self.staging_.largest_free >= need:
                return
            self._drop_staged(b, r, now, now)
            r.pins.clear()

    def _drop_staged(self, b: str, rec: BlobRec, at: float, now: float) -> None:
        """Free a blob's IO-area region at ``at`` (caller holds ``_mu``)."""
        _free_at(self.staging_, rec.staged_off, rec.nbytes, at, now)
        rec.staged_off = None
        if self._staged.get(b) is rec:
            del self._staged[b]

    def _stage(self, b: str, rec: BlobRec) -> None:
        """IO worker: read a disk blob's file into its IO-area region.

        While a stage runs, the fd is closed only here (``_dispose`` skips it).
        """
        fd = rec.disk
        at = self.staging_base + rec.staged_off
        view = memoryview(self.slab)[at : at + rec.nbytes]
        ok = True
        try:
            done = 0
            while done < rec.nbytes:
                n = os.preadv(fd, [view[done:]], done)
                if not n:
                    raise OSError(f"short read staging {b}")
                done += n
        except Exception as e:  # noqa: BLE001 - anything else would leave staging set
            ok = False
            log.error("unit %d: staging %s failed: %s", self.unit_id, b, e)
        now = time.monotonic()
        with self._mu:
            self._io_jobs -= 1
            rec.staging = False
            gone = self.blobs.get(b) is not rec
            if gone or not ok:
                self._drop_staged(b, rec, now, now)
                if gone:
                    rec.disk = None
                else:
                    # The file may be fine: keep it and tell the next resolve once.
                    rec.stage_failed = True
                    self.ssd_stats["stage_failures"] += 1
            else:
                rec.staged_at = now
                rec.picked_up = False
                wait_ms = (now - rec.wait_since) * 1000.0
                rec.wait_since = 0.0
                st = self.ssd_stats
                st["staged_bytes"] += rec.nbytes
                st["stages"] += 1
                st["stage_wait_ms_max"] = max(st["stage_wait_ms_max"], wait_ms)
        if gone:
            _close(fd)

    # ------------------------------------------------------------------ identity
    @_locked
    def info(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "node_id": self.node_id,
            "nics": self._nics,
            "zmq": self._server.address,
            "agent_name": self.ep.name,
            "agent_md": self.ep.metadata(),
            "slab_base": self.slab_base,
            "slab_bytes": int(self.slab.nbytes),
            "put_bytes": int(self.staging_base),
            "state": "ACTIVE",
        }

    @_locked
    def ping(self) -> int:
        return self.unit_id

    # ------------------------------------------------------------------ allocation
    @_locked
    def alloc(
        self, blob_id: str, nbytes: int, n_entries: int, spill: bool = True
    ) -> int:
        now = time.monotonic()
        self._reclaim(now)
        if blob_id in self.blobs:
            raise ValueError(f"blob {blob_id} already allocated")
        try:
            off = self.alloc_.alloc(nbytes)
        except UnitFull:
            # Best-effort RAM-first: the client asks spill=False of every unit
            # before it asks any unit to spill.
            if self.ssd_dir is None or not spill:
                raise
            # Spill: the put goes to disk from birth, through the IO area. The
            # client cannot tell: it just WRITEs at the offset it is given.
            io_off = self._alloc_io(int(nbytes), now)
            self.blobs[blob_id] = BlobRec(
                off=0,
                nbytes=nbytes,
                refs=int(n_entries),
                created_at=now,
                in_slab=False,
                staged_off=io_off,
            )
            return self.staging_base + io_off
        self.blobs[blob_id] = BlobRec(
            off=off,
            nbytes=nbytes,
            refs=int(n_entries),
            created_at=now,
        )
        return off

    @_locked
    def commit(self, blob_id: str) -> str:
        """The put's WRITE is DONE: ``"done"`` once the blob is durable.

        A spilled blob is ``"pending"`` while its disk write runs (the client asks
        again); ``"lost"`` if the blob is gone (aborted or its write failed), and
        the put then fails.
        """
        rec = self.blobs.get(blob_id)
        if rec is None:
            return "lost"
        if rec.in_slab or rec.disk is not None:
            return "done"
        if not rec.staging and self._io_jobs < self._max_io_jobs:
            self._submit_io(self._spill, blob_id, rec)  # exactly once: staging is set
        return "pending"

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
        if rec.staging:
            self._drop(blob_id, rec)  # its IO job frees the region when it ends
        elif in_flight:
            self.retired_bytes += rec.nbytes
        elif self.ssd_dir is not None:
            # A committed blob (commit reply lost) may be on disk or staged by
            # now, a spill not yet committed still holds its IO region: drop it
            # like a release.
            self._drop(blob_id, rec)
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
        self._drop(blob_id, rec)
        return 0

    def _drop(self, blob_id: str, rec: BlobRec) -> None:
        """Free a gone blob's copies, or after the running copy of it ends."""
        if rec.holds:
            # A checkpoint is still copying it: free it when that copy is done.
            self._released_held[blob_id] = rec
        else:
            self._dispose(blob_id, rec)

    def _dispose(self, blob_id: str, rec: BlobRec) -> None:
        """Free every copy of a released blob (caller holds ``_mu``)."""
        if self.ssd_dir is None:
            self._free_region(rec)
            return
        # SSD on: a copy is free once its pins end. Put-area blobs are also read
        # by hint, without a pin: a hinted read issued after the release fails
        # its footer tag check and raises. A READ already in flight while the
        # region is reused is not guarded; consumers do not read a key while it
        # is being cleared or overwritten, so a reasonable run never does this.
        now = time.monotonic()
        at = rec.pin_expiry(now)
        if rec.in_slab:
            _free_at(self.alloc_, rec.off, rec.nbytes, at, now)
            rec.in_slab = False
        if rec.staged_off is not None and not rec.staging:
            self._drop_staged(blob_id, rec, at, now)
        elif self._staged.get(blob_id) is rec:
            # A running stage frees its own region when it sees the blob gone.
            del self._staged[blob_id]
        if rec.disk is not None and not rec.staging:
            self._worker.submit(_close, rec.disk)  # no blocking syscall under _mu
            rec.disk = None

    def _free_region(self, rec: BlobRec) -> None:
        now = time.monotonic()
        # Always quarantine for read_pin_s, pinned or not: clients read with
        # cached (unit, offset) hints and no pin RPC, so a freed region must not
        # be reused while such a read could still be in flight. The blob tag in
        # the footer catches anything older than that: _wipe_footer clears it
        # before the region can be reused (a smaller blob would not overwrite it).
        at = max(rec.pin_expiry(now), now + self.read_pin_s)
        _free_at(self.alloc_, rec.off, rec.nbytes, at, now)

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
        pending: list[str] = []
        failed: list[str] = []
        for b in blob_ids:
            rec = self.blobs.get(b)
            if rec is None:
                missing.append(b)
                continue
            rec.last_resolve_at = now
            if rec.in_slab:
                off = rec.off
            elif rec.disk is None:
                missing.append(b)  # a spill whose put is not acknowledged yet
                continue
            elif rec.staged_off is not None and not rec.staging:
                off = self.staging_base + rec.staged_off
                rec.picked_up = True
            elif rec.staging:
                pending.append(b)  # a stage is running: ask again shortly
                continue
            elif rec.stage_failed:
                rec.stage_failed = False  # reported once; the next resolve retries
                failed.append(b)
                continue
            else:
                self._start_stage(b, rec, now)
                pending.append(b)
                continue
            rec.pins[pin_id] = expiry
            offsets[b] = off
            pinned.append(b)
        if pinned:
            self._pins[pin_id] = pinned
        return {
            "pin_id": pin_id,
            "offsets": offsets,
            "missing": missing,
            "pending": pending,
            "failed": failed,
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
            if not rec.in_slab:
                continue
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
        self._reclaim(time.monotonic())
        ssd = self._ssd_stats() if self.ssd_dir is not None else {}
        return {
            **ssd,
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

        SSD on: a disk blob is copied from its file, through a dup of its fd
        taken just before the copy. A disk blob released meanwhile may already
        have its file closed; the save then raises (a clear during a checkpoint
        is outside TQ's contract). A spill not acknowledged yet is not
        published, so it is skipped.
        """
        held: list[tuple[str, BlobRec, int]] = []
        try:
            with self._mu:
                self.checkpointing = True
                self.clears_during_save = 0
                for b, rec in self.blobs.items():
                    if not rec.in_slab and rec.disk is None:
                        continue
                    rec.holds += 1
                    held.append((b, rec, rec.refs))
            index: dict[str, list[int]] = {}
            tmp = path + ".tmp"
            pos = 0
            view = memoryview(self.slab)
            with open(tmp, "wb") as f:
                for b, rec, refs in held:
                    if rec.in_slab:
                        f.write(view[rec.off : rec.off + rec.nbytes])
                    else:
                        with self._mu:
                            fd = None if rec.disk is None else os.dup(rec.disk)
                        if fd is None:
                            raise RuntimeError(
                                f"unit {self.unit_id}: blob {b} was released during "
                                "save_shard; TransferQueue does not support clears "
                                "during a checkpoint"
                            )
                        try:
                            _copy_out(fd, f, rec.nbytes)
                        finally:
                            _close(fd)
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
                        self._dispose(b, rec)
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
        self, path: str, blobs: dict[str, list[int]], spill: bool = False
    ) -> dict[str, list[str]]:
        """Read ``blobs`` (``blob_id -> [off_in_file, nbytes, refs]``) from ``path`` into fresh regions.

        Blobs that do not fit are returned under ``failed`` so the caller can
        try another unit; already-present blobs count as loaded. With ``spill``
        and the SSD tier on, a blob the put area cannot take goes to a file
        instead, as a spilled put would (the caller asks for that only once
        every unit's RAM is full, as puts do).

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
                        dst = None
                if dst is None:
                    ok = spill and self._load_to_disk(f, b, int(off), nbytes, int(refs))
                    (loaded if ok else failed).append(b)
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

    def _load_to_disk(self, f, b: str, off: int, nbytes: int, refs: int) -> bool:
        """Restore one blob from checkpoint file ``f`` into a fresh SSD file.

        False (the caller tries another unit) with the SSD tier off, if the blob
        could never be staged back for a read, or if the disk write fails.
        """
        io = self.staging_
        if io is None or align_up(nbytes, io.align) > io.capacity:
            return False
        fd = None
        try:
            fd = self._new_file()
            f.seek(off)
            done = 0
            while done < nbytes:
                chunk = f.read(min(nbytes - done, _COPY_CHUNK))
                if not chunk:  # a bad checkpoint, not a full disk: fail the restore
                    raise EOFError(f"short read restoring blob {b}")
                view, w = memoryview(chunk), 0
                while w < len(view):  # pwrite may write less than asked
                    w += os.pwrite(fd, view[w:], done + w)
                done += len(chunk)
        except OSError as e:
            if fd is not None:
                _close(fd)
            log.warning("unit %d: restoring %s to disk failed: %s", self.unit_id, b, e)
            return False
        except BaseException:
            if fd is not None:
                _close(fd)
            raise
        with self._mu:
            self.blobs[b] = BlobRec(
                off=0,
                nbytes=nbytes,
                refs=refs,
                created_at=time.monotonic(),
                in_slab=False,
                disk=fd,
            )
            self.ssd_stats["spilled_bytes"] += nbytes
            self.ssd_stats["spills"] += 1
        return True
