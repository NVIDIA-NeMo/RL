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
"""BlobStore: where a blob's bytes live. The client never knows.

The client lays values out as raw bytes in a registered local buffer, then
hands the store ``(local_addr, plan)``; on read it hands the store byte
ranges and registered destination addresses. Everything about placement,
pins, refcounts and the medium is the store's business, so the same client
serves:

- :class:`UnitSlabStore` — pinned-DRAM slabs on CPU StorageUnit actors, one-sided
  RDMA over UCX. The hot path.
- :class:`FileStore` — one file per blob on a shared filesystem (Lustre, NVMe),
  written and read with NIXL's POSIX backend in loopback. No storage actors;
  the BlobDirectory only keeps refcounts.

Both return the same location meta, so TransferQueue and NeMo-RL cannot
tell them apart, and a blob can be moved between stores by copying bytes
and rewriting its meta (that is what tiering and checkpoint shards are).
"""

from __future__ import annotations

import os
import uuid
from collections import OrderedDict
from typing import Any, Protocol

import ray

from nemo_rl.data_plane.nixl import blob_format
from nemo_rl.data_plane.nixl.control import UnitClient
from nemo_rl.data_plane.nixl.directory import LOST
from nemo_rl.data_plane.nixl.errors import TransferError, UnitFull
from nemo_rl.data_plane.nixl.nixl_io import NixlEndpoint
from nemo_rl.data_plane.nixl.placement import Placement


class LostBlobs(Exception):
    """Some blobs are unreadable; the client maps them back to keys."""

    def __init__(self, blob_ids: list[str], reason: str) -> None:
        self.blob_ids = list(blob_ids)
        self.reason = reason
        super().__init__(f"{len(self.blob_ids)} blob(s) lost: {reason}")


# (off_in_blob, nbytes, local_addr)
ReadRange = tuple[int, int, int]


class BlobStore(Protocol):
    def put(
        self, local_addr: int, p: blob_format.BlobPlan
    ) -> tuple[str, dict[str, Any]]:
        """Store the blob at ``local_addr``; return ``(blob_id, extra_meta)``."""
        ...

    def read(
        self,
        requests: dict[str, list[ReadRange]],
        metas: dict[str, dict[str, Any]],
        scratch: tuple[int, memoryview] | None = None,
    ) -> None:
        """Fill every local range from its blob. Raises :class:`LostBlobs`."""
        ...

    def release(
        self, counts: dict[str, int], hints: dict[str, int] | None = None
    ) -> None:
        """Drop ``counts[blob]`` live entries per blob; free the blob at zero."""
        ...

    def close(self) -> None: ...


# ============================================================================ unit slabs
class UnitSlabStore:
    """Pinned-DRAM slabs on NixlStorageUnit actors, one-sided RDMA.

    No Ray on the op path: control messages go to the units over ZMQ
    (nemo_rl/data_plane/nixl/control.py), bytes over NIXL.

    - put:   1 ZMQ request/reply (``alloc``) + NIXL WRITE.
    - get:   0 control messages when the key's location meta carries a hint
             ``(u, so, z)``: READ the requested ranges plus the blob's footer
             straight from ``slab_base + so`` and check the footer's blob tag.
             A mismatch (blob moved by a checkpoint restore) or a missing hint
             falls back to a directory lookup + ZMQ ``resolve`` pin.
    - clear: 0 replies -- one ZMQ ``release_many`` per unit, fire-and-forget.

    Ray remains only off the op path: the unit list at start (and if every
    cached unit refuses a put), and directory lookups for restored blobs.

    Reads without a pin are safe because units quarantine every freed region
    for ``read_pin_s`` before reusing it.
    """

    def __init__(
        self,
        ep: NixlEndpoint,
        directory: Any,
        *,
        namespace: str,
        placement: Placement,
        node_id: str,
        timeout_s: float,
    ) -> None:
        self.ep = ep
        self.directory = directory
        self.namespace = namespace
        self.placement = placement
        self.node_id = node_id
        self.timeout_s = timeout_s
        self._units: dict[int, dict[str, Any]] = {}
        self._epoch = 0
        self._blob_unit: dict[str, int] = {}
        self.ctl = UnitClient(timeout_s=min(timeout_s, 30.0))
        self.stats = {
            "fast_reads": 0,
            "slow_reads": 0,
            "tag_mismatch": 0,
            "unit_refresh": 0,
        }
        self._refresh_units()

    # -- membership
    def _refresh_units(self) -> None:
        self.stats["unit_refresh"] += 1
        epoch, units = ray.get(self.directory.units.remote(self._epoch))
        if units:
            self._units = {int(u["unit_id"]): u for u in units}
            self._epoch = epoch

    def _unit(self, unit_id: int) -> dict[str, Any]:
        u = self._units.get(unit_id)
        if u is None:
            self._refresh_units()
            u = self._units.get(unit_id)
        if u is None:
            raise LostBlobs([], f"unit {unit_id} unknown to the directory")
        if not self.ep.has_remote(u["agent_name"]):
            try:
                self.ep.add_remote(u["agent_md"])
            except Exception as e:  # noqa: BLE001 - NIXL cannot reach the unit's agent
                raise LostBlobs(
                    [], f"unit {unit_id} unreachable: {type(e).__name__}: {e}"
                ) from e
        return u

    def _gen(self, unit_id: int) -> str | None:
        """Instance stamp of the live unit with this id (its NIXL agent suffix).

        Hints carry the stamp of the unit that wrote them; after a checkpoint
        restore the units are new instances, so stale hints stop matching.
        """
        u = self._units.get(unit_id)
        return u["agent_name"].rsplit("-", 1)[-1] if u else None

    def _hint_live(self, m: dict[str, Any]) -> bool:
        return "u" in m and m.get("g") is not None and self._gen(int(m["u"])) == m["g"]

    def _addr(self, unit_id: int) -> str:
        u = self._units.get(unit_id)
        if u is None:
            self._refresh_units()
            u = self._units.get(unit_id)
        if u is None or not u.get("zmq"):
            raise LostBlobs([], f"unit {unit_id} unknown to the directory")
        return u["zmq"]

    def _where(self, blob_ids: list[str]) -> dict[str, int]:
        unknown = [b for b in blob_ids if b not in self._blob_unit]
        if unknown:
            for b, u in ray.get(self.directory.where.remote(unknown)).items():
                if u != LOST:
                    self._blob_unit[b] = int(u)
        return {b: self._blob_unit[b] for b in blob_ids if b in self._blob_unit}

    # -- BlobStore
    def put(
        self, local_addr: int, p: blob_format.BlobPlan
    ) -> tuple[str, dict[str, Any]]:
        return self.put_segments([(local_addr, 0, p.nbytes)], p.nbytes, len(p.entries))

    def put_segments(
        self,
        segments: list[tuple[int, int, int]],
        nbytes: int,
        n_entries: int,
        blob_id: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Write one blob of ``nbytes`` from scattered registered regions.

        ``segments`` are ``(local_addr, blob_off, n)``: a value sent zero-copy
        from the caller's own (registered) memory, or a run copied into a pool
        buffer. All go out in one NIXL transfer. Returns ``(blob_id, hint)``;
        the hint ``{"u", "so", "z"}`` lets readers skip every RPC.
        """
        blob_id = blob_id or uuid.uuid4().hex
        if not self._units:
            self._refresh_units()
        last_err: Exception | None = None
        for attempt in range(2):
            for cand in self.placement.order(
                list(self._units.values()), node_id=self.node_id, nbytes=nbytes
            ):
                unit_id = int(cand["unit_id"])
                try:
                    off = self.ctl.call(
                        self._addr(unit_id),
                        "alloc",
                        blob_id=blob_id,
                        nbytes=int(nbytes),
                        n_entries=int(n_entries),
                    )
                except (UnitFull, TransferError, LostBlobs) as e:
                    last_err = e
                    continue
                u = self._unit(unit_id)
                base = int(u["slab_base"]) + off
                try:
                    self.ep.transfer(
                        "WRITE",
                        [(a, n) for a, _, n in segments],
                        [(base + bo, n) for _, bo, n in segments],
                        u["agent_name"],
                        timeout_s=self.timeout_s,
                    )
                except TransferError as e:
                    self.ctl.notify(
                        u["zmq"], "abort", blob_id=blob_id, in_flight=e.in_flight
                    )
                    last_err = e
                    continue
                self._blob_unit[blob_id] = unit_id
                return blob_id, {
                    "u": unit_id,
                    "so": int(off),
                    "z": int(nbytes),
                    "g": self._gen(unit_id),
                }
            if attempt == 0:
                self._refresh_units()  # membership may have changed; one retry
        raise UnitFull(f"no unit could take {nbytes} B: {last_err}")

    def read(
        self,
        requests: dict[str, list[ReadRange]],
        metas: dict[str, dict[str, Any]],
        scratch: tuple[int, memoryview] | None = None,
    ) -> None:
        """Fill every local range from its blob.

        ``scratch`` gives registered room for one footer per blob
        (``len(requests) * FOOTER_SIZE`` bytes) and enables the RPC-free path for blobs whose meta carries a hint.
        """
        slow: dict[str, list[ReadRange]] = {}
        fast: dict[int, list[str]] = {}
        for b, ranges in requests.items():
            m = metas.get(b) or {}
            if scratch is not None and "so" in m and "z" in m and self._hint_live(m):
                fast.setdefault(int(m["u"]), []).append(b)
            else:
                slow[b] = ranges
        if fast:
            slow.update(self._read_hinted(requests, metas, fast, scratch))
        if slow:
            self.stats["slow_reads"] += len(slow)
            self._read_pinned(slow)

    def _read_hinted(
        self, requests, metas, fast, scratch
    ) -> dict[str, list[ReadRange]]:
        sbase, sview = scratch
        fsz = blob_format.FOOTER_SIZE
        retry: dict[str, list[ReadRange]] = {}
        slot = 0
        for unit_id, blobs in fast.items():
            try:
                uinfo = self._unit(unit_id)
            except LostBlobs:
                retry.update({b: requests[b] for b in blobs})
                continue
            loc: list[tuple[int, int]] = []
            rem: list[tuple[int, int]] = []
            slots: dict[str, int] = {}
            for b in blobs:
                m = metas[b]
                base = int(uinfo["slab_base"]) + int(m["so"])
                for off, n, laddr in requests[b]:
                    loc.append((laddr, n))
                    rem.append((base + off, n))
                slots[b] = slot
                loc.append((sbase + slot * fsz, fsz))
                rem.append((base + int(m["z"]) - fsz, fsz))
                slot += 1
            try:
                self.ep.transfer(
                    "READ", loc, rem, uinfo["agent_name"], timeout_s=self.timeout_s
                )
            except TransferError as e:
                if e.in_flight:
                    # The READ may still land in these local ranges, so they
                    # cannot be refilled by the pinned path; fail the read.
                    raise LostBlobs(blobs, str(e)) from e
                retry.update({b: requests[b] for b in blobs})
                continue
            for b in blobs:
                s = slots[b]
                tag = blob_format.footer_tag(sview[s * fsz : (s + 1) * fsz])
                if tag is not None and tag.hex() == b:
                    self.stats["fast_reads"] += 1
                else:
                    self.stats["tag_mismatch"] += 1
                    retry[b] = requests[b]
        return retry

    def _read_pinned(self, requests: dict[str, list[ReadRange]]) -> None:
        """Directory lookup + ``unit.resolve`` pin: blobs without a usable hint."""
        where = self._where(list(requests))
        if len(where) < len(requests):
            raise LostBlobs(
                [b for b in requests if b not in where], "blob location unknown"
            )
        by_unit: dict[int, list[str]] = {}
        for b, u in where.items():
            by_unit.setdefault(u, []).append(b)

        resolved: dict[int, dict[str, Any]] = {}
        for u, bs in by_unit.items():
            try:
                resolved[u] = self.ctl.call(self._addr(u), "resolve", blob_ids=bs)
            except (TransferError, LostBlobs) as e:
                for done_u, r in resolved.items():
                    self.ctl.notify(
                        self._units[done_u]["zmq"], "unpin", pin_id=r["pin_id"]
                    )
                raise LostBlobs(bs, f"unit {u} is unreachable: {e}")
        try:
            missing = [b for r in resolved.values() for b in r["missing"]]
            if missing:
                raise LostBlobs(missing, "blob not on unit")
            for u, blobs in by_unit.items():
                uinfo, r = self._unit(u), resolved[u]
                loc: list[tuple[int, int]] = []
                rem: list[tuple[int, int]] = []
                for b in blobs:
                    base = int(uinfo["slab_base"]) + int(r["offsets"][b])
                    for off, n, laddr in requests[b]:
                        loc.append((laddr, n))
                        rem.append((base + off, n))
                try:
                    self.ep.transfer(
                        "READ", loc, rem, uinfo["agent_name"], timeout_s=self.timeout_s
                    )
                except TransferError as e:
                    raise LostBlobs(blobs, str(e))
        finally:
            for u, r in resolved.items():
                self.ctl.notify(self._units[u]["zmq"], "unpin", pin_id=r["pin_id"])

    def release(
        self, counts: dict[str, int], hints: dict[str, dict[str, Any]] | None = None
    ) -> None:
        """Drop live-entry counts: one ``release_many`` per unit, not awaited.

        ``hints`` maps blob -> its location meta. A live hint (stamp matches the
        current unit instance) needs no lookup; anything else (restored blobs)
        asks the directory, which the restore keeps current.
        """
        metas = hints or {}
        loc: dict[str, int] = {}
        unknown = []
        for b in counts:
            m = metas.get(b) or {}
            if self._hint_live(m):
                loc[b] = int(m["u"])
            elif b in self._blob_unit:
                loc[b] = self._blob_unit[b]
            else:
                unknown.append(b)
        if unknown:
            loc.update(self._where(unknown))
        hints = loc
        by_unit: dict[int, list[tuple[str, int]]] = {}
        for b, n in counts.items():
            u = hints.get(b, self._blob_unit.get(b))
            if u is not None:
                by_unit.setdefault(int(u), []).append((b, int(n)))
        for u, items in by_unit.items():
            try:
                self.ctl.notify(
                    self._addr(u), "release_many", items=[[b, n] for b, n in items]
                )
            except LostBlobs:
                continue
        for b in counts:
            self._blob_unit.pop(b, None)

    def close(self) -> None:
        self.ctl.close()


# ============================================================================ files (Lustre / NVMe)
class FileStore:
    """One file per blob under ``root``; NIXL POSIX loopback for WRITE/READ.

    No storage actors. The BlobDirectory keeps per-blob refcounts so the last
    ``release`` deletes the file. Registered fds are cached per process with
    an LRU cap so reads of hot blobs skip the open+register cost.

    A blob whose WRITE or READ failed short of DONE moves to ``retained``:
    the transfer may still use its fd, so the fd stays open and registered
    and the file stays in place until the process exits (never evicted,
    released or closed).
    """

    def __init__(
        self,
        ep: NixlEndpoint,
        directory: Any,
        *,
        root: str,
        timeout_s: float,
        backend: str = "POSIX",
        max_open: int = 256,
    ) -> None:
        self.ep = ep
        self.directory = directory
        self.root = root
        self.timeout_s = timeout_s
        self.backend = backend
        self.max_open = max_open
        os.makedirs(root, exist_ok=True)
        self._open: OrderedDict[str, tuple[int, int]] = (
            OrderedDict()
        )  # blob → (fd, nbytes)
        self.retained: dict[str, tuple[int, int]] = {}  # blob → (fd, nbytes)

    def path(self, blob_id: str) -> str:
        return os.path.join(self.root, f"{blob_id}.blob")

    def _fd(self, blob_id: str, nbytes: int, *, create: bool) -> int:
        item = self._open.get(blob_id)
        if item is not None:
            self._open.move_to_end(blob_id)
            return item[0]
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        try:
            fd = os.open(self.path(blob_id), flags, 0o644)
        except FileNotFoundError:
            raise LostBlobs([blob_id], "file missing")
        if create:
            os.ftruncate(fd, nbytes)
        self.ep.register_file(fd, nbytes, backend=self.backend)
        self._open[blob_id] = (fd, nbytes)
        while len(self._open) > self.max_open:
            old, (ofd, _) = self._open.popitem(last=False)
            self._close_fd(ofd)
        return fd

    def _close_fd(self, fd: int) -> None:
        try:
            self.ep.deregister_file(fd)
        finally:
            os.close(fd)

    def _retain(self, blob_id: str) -> None:
        item = self._open.pop(blob_id, None)
        if item is not None:
            self.retained[blob_id] = item

    def _forget(self, blob_id: str) -> None:
        item = self._open.pop(blob_id, None)
        if item is not None:
            self._close_fd(item[0])

    # -- BlobStore
    def put(
        self, local_addr: int, p: blob_format.BlobPlan
    ) -> tuple[str, dict[str, Any]]:
        blob_id = uuid.uuid4().hex
        fd = self._fd(blob_id, p.nbytes, create=True)
        try:
            self.ep.transfer(
                "WRITE",
                [(local_addr, p.nbytes)],
                [(0, p.nbytes)],
                self.ep.name,
                timeout_s=self.timeout_s,
                remote_mem="FILE",
                remote_dev=fd,
            )
        except TransferError as e:
            if e.in_flight:
                self._retain(blob_id)  # the WRITE may still land in the file
                raise
            self._forget(blob_id)
            try:
                os.remove(self.path(blob_id))
            except FileNotFoundError:
                pass
            raise
        ray.get(self.directory.put_blobs.remote([(blob_id, -1, len(p.entries))]))
        return blob_id, {
            "z": p.nbytes
        }  # blob size: needed to register the file for reads

    def read(
        self,
        requests: dict[str, list[ReadRange]],
        metas: dict[str, dict[str, Any]],
        scratch: tuple[int, memoryview] | None = None,
    ) -> None:
        for b, ranges in requests.items():
            nbytes = int(metas[b]["z"])
            fd = self._fd(b, nbytes, create=False)
            loc = [(laddr, n) for _, n, laddr in ranges]
            rem = [(off, n) for off, n, _ in ranges]
            try:
                self.ep.transfer(
                    "READ",
                    loc,
                    rem,
                    self.ep.name,
                    timeout_s=self.timeout_s,
                    remote_mem="FILE",
                    remote_dev=fd,
                )
            except TransferError as e:
                if e.in_flight:
                    self._retain(b)  # the READ may still use the fd
                raise LostBlobs([b], str(e))

    def release(
        self, counts: dict[str, int], hints: dict[str, int] | None = None
    ) -> None:
        gone = ray.get(self.directory.release.remote(list(counts.items())))
        for b in gone:
            if b in self.retained:
                continue  # a failed transfer may still use the file
            self._forget(b)
            try:
                os.remove(self.path(b))
            except FileNotFoundError:
                pass

    def close(self) -> None:
        for b in list(self._open):
            self._forget(b)


def make_store(kind: str, ep: NixlEndpoint, directory: Any, **kw: Any) -> BlobStore:
    if kind == "unit":
        return UnitSlabStore(ep, directory, **kw)
    if kind == "file":
        return FileStore(ep, directory, **kw)
    raise ValueError(f"unknown store kind {kind!r}; choose 'unit' or 'file'")
