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
"""UnitSlabStore: where a blob's bytes live. The client never knows.

The client lays values out as raw bytes in a registered local buffer, then
hands the store ``(local_addr, plan)``; on read it hands the store byte
ranges and registered destination addresses. Everything about placement,
pins, refcounts and the medium is the store's business: pinned-DRAM slabs on
CPU StorageUnit actors, one-sided RDMA over UCX, with an optional SSD tier on
each unit.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import ray

from nemo_rl.data_plane.nixl import blob_format
from nemo_rl.data_plane.nixl.control import UnitClient
from nemo_rl.data_plane.nixl.directory import DEAD, LOST
from nemo_rl.data_plane.nixl.errors import (
    DataPlaneBusy,
    DataPlaneTimeout,
    LostBlobs,
    TransferError,
    UnitFull,
)
from nemo_rl.data_plane.nixl.nixl_io import NixlEndpoint
from nemo_rl.data_plane.nixl.placement import Placement

__all__ = [
    "LostBlobs",
    "TransferFailed",
    "UnitSlabStore",
]

# Client backoff while units answer Busy (design §5.3): doubling, capped.
BUSY_RETRY_S = 0.2
BUSY_RETRY_MAX_S = 2.0
# First re-ask for blobs a unit reports ``pending`` (staging from disk takes ms).
PENDING_POLL_S = 0.01


class TransferFailed(Exception):
    """A transfer or a control call failed, but the unit still has the blobs.

    The caller may retry. ``in_flight``: a NIXL transfer may still be running.
    """

    def __init__(self, blob_ids: list[str], reason: str, *, in_flight: bool = False):
        self.blob_ids = list(blob_ids)
        self.reason = reason
        self.in_flight = in_flight
        super().__init__(f"{len(self.blob_ids)} blob(s): {reason}")


def remaining(deadline: float | None, cap: float) -> float:
    """``min(cap, deadline - now)``; raise :class:`DataPlaneTimeout` if spent."""
    if deadline is None:
        return cap
    left = deadline - time.monotonic()
    if left <= 0:
        raise DataPlaneTimeout("operation deadline passed")
    return min(cap, left)


def busy_wait(deadline: float | None, backoff: float) -> float:
    """Sleep one Busy backoff (bounded by ``deadline``); return the next one."""
    time.sleep(remaining(deadline, backoff))
    return min(2 * backoff, BUSY_RETRY_MAX_S)


# (off_in_blob, nbytes, local_addr)
ReadRange = tuple[int, int, int]


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
    - clear: one awaited ZMQ ``release_rows`` per unit (row-wide).

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
        ssd: bool = False,
    ) -> None:
        self.ep = ep
        # SSD tier on the units: a put commits after its WRITE is DONE. Put-area
        # blobs keep their hints; a spilled put has none, so its reads resolve
        # through the unit.
        self.ssd = ssd
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

    def _locate(
        self, blob_ids: list[str], metas: dict[str, dict[str, Any]]
    ) -> dict[str, int]:
        """Unit of each blob: its live hint (any writer), else cache / directory.

        The directory only knows restored blobs, so a blob another client wrote
        is found through its meta.
        """
        loc: dict[str, int] = {}
        rest = []
        for b in blob_ids:
            m = metas.get(b) or {}
            if self._hint_live(m):
                loc[b] = int(m["u"])
            else:
                rest.append(b)
        loc.update(self._where(rest))
        return loc

    def put(
        self,
        local_addr: int,
        p: blob_format.BlobPlan,
        deadline: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        return self.put_segments(
            [(local_addr, 0, p.nbytes)], p.nbytes, len(p.entries), deadline=deadline
        )

    def put_segments(
        self,
        segments: list[tuple[int, int, int]],
        nbytes: int,
        n_entries: int,
        blob_id: str | None = None,
        deadline: float | None = None,
        unit_id: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Write one blob of ``nbytes`` from scattered registered regions.

        ``unit_id`` pins the blob to the unit that owns its rows (index
        addressing); without it the placement picks any unit.

        ``segments`` are ``(local_addr, blob_off, n)``: a value sent zero-copy
        from the caller's own (registered) memory, or a run copied into a pool
        buffer. All go out in one NIXL transfer. Returns ``(blob_id, hint)``;
        the hint ``{"u", "so", "z"}`` lets readers skip every RPC.

        Every wait is bounded by ``deadline`` (monotonic). Raises
        :class:`UnitFull` only when every unit refused for lack of space,
        :class:`TransferFailed` when units failed otherwise, and
        :class:`DataPlaneTimeout` when the deadline passes (also while units
        keep answering Busy, design §5.3).
        """
        blob_id = blob_id or uuid.uuid4().hex
        if not self._units:
            self._refresh_units()
        backoff = BUSY_RETRY_S
        refreshed = False
        while True:
            errs: list[Exception] = []
            if unit_id is not None:
                if unit_id not in self._units:
                    self._refresh_units()
                if unit_id not in self._units:
                    raise LostBlobs(
                        [blob_id], f"unit {unit_id} unknown to the directory"
                    )
                order = [self._units[unit_id]]
            else:
                order = self.placement.order(
                    list(self._units.values()), node_id=self.node_id, nbytes=nbytes
                )
            # With the SSD tier on, every unit's RAM is tried before any unit spills.
            tries = [(c, True) for c in order]
            if self.ssd:
                tries = [(c, False) for c in order] + tries
            for cand, spill in tries:
                unit_id = int(cand["unit_id"])
                try:
                    off = self.ctl.call(
                        self._addr(unit_id),
                        "alloc",
                        timeout_s=remaining(deadline, self.ctl.timeout_ms / 1000),
                        blob_id=blob_id,
                        nbytes=int(nbytes),
                        n_entries=int(n_entries),
                        spill=spill,
                    )
                except TransferError as e:
                    # No reply in time: the unit may still allocate. Nothing
                    # was posted, so it can free the region whenever it does.
                    self.ctl.notify(
                        self._addr(unit_id), "abort", blob_id=blob_id, in_flight=False
                    )
                    errs.append(e)
                    continue
                except (UnitFull, DataPlaneBusy, LostBlobs) as e:
                    errs.append(e)
                    continue
                try:
                    u = self._unit(unit_id)
                except LostBlobs as e:  # NIXL cannot reach it: nothing was posted
                    self.ctl.notify(
                        self._addr(unit_id), "abort", blob_id=blob_id, in_flight=False
                    )
                    errs.append(e)
                    continue
                base = int(u["slab_base"]) + off
                try:
                    left = remaining(deadline, self.timeout_s)
                except DataPlaneTimeout:
                    # Nothing was posted: the unit can free the region at once.
                    self.ctl.notify(u["zmq"], "abort", blob_id=blob_id, in_flight=False)
                    raise
                try:
                    self.ep.transfer(
                        "WRITE",
                        [(a, n) for a, _, n in segments],
                        [(base + bo, n) for _, bo, n in segments],
                        u["agent_name"],
                        timeout_s=left,
                    )
                except TransferError as e:
                    self.ctl.notify(
                        u["zmq"], "abort", blob_id=blob_id, in_flight=e.in_flight
                    )
                    errs.append(e)
                    continue
                if not self._commit(u, blob_id, deadline, errs):
                    continue
                self._blob_unit[blob_id] = unit_id
                hint = {"u": unit_id, "g": self._gen(unit_id)}
                if off < int(u.get("put_bytes", off + 1)):  # not a spilled blob
                    hint |= {"so": int(off), "z": int(nbytes)}
                return blob_id, hint
            if any(isinstance(e, DataPlaneBusy) for e in errs):
                backoff = busy_wait(deadline, backoff)
                continue
            if not refreshed:
                self._refresh_units()  # membership may have changed; one retry
                refreshed = True
                continue
            if errs and all(isinstance(e, UnitFull) for e in errs):
                raise UnitFull(f"no unit could take {nbytes} B: {errs[-1]}")
            raise TransferFailed(
                [blob_id],
                f"no unit took {nbytes} B: {errs[-1] if errs else 'no units'}",
                in_flight=any(getattr(e, "in_flight", False) for e in errs),
            )

    def _commit(
        self, u: dict[str, Any], blob_id: str, deadline: float | None, errs: list
    ) -> bool:
        """Awaited commit: the blob's rows are published once it returns True.

        A blob the unit spilled to disk answers ``"pending"`` while its file is
        written: ask again, inside the put deadline. Commit is idempotent on the
        unit, so a call whose reply was lost is asked again too. On failure the
        client aborts, which the unit ignores for a blob it did publish.
        """
        ok = False
        try:
            backoff = PENDING_POLL_S
            while True:
                try:
                    status = self.ctl.call(
                        u["zmq"],
                        "commit",
                        timeout_s=remaining(deadline, self.ctl.timeout_ms / 1000),
                        blob_id=blob_id,
                    )
                except TransferError:
                    if deadline is None:
                        raise
                    continue  # no reply: ask again until the deadline
                if status != "pending":
                    break
                backoff = busy_wait(deadline, backoff)
            ok = status == "done"
            if not ok:
                errs.append(TransferError(f"unit {u['unit_id']} lost the blob"))
        except (TransferError, LostBlobs, DataPlaneBusy) as e:
            errs.append(e)
        finally:
            if not ok:  # every failure, including a deadline: free the region
                self.ctl.notify(u["zmq"], "abort", blob_id=blob_id, in_flight=False)
        return bool(ok)

    def read(
        self,
        requests: dict[str, list[ReadRange]],
        metas: dict[str, dict[str, Any]],
        scratch: tuple[int, memoryview] | None = None,
        deadline: float | None = None,
    ) -> None:
        """Fill every local range from its blob.

        ``scratch`` gives registered room for one footer per blob
        (``len(requests) * FOOTER_SIZE`` bytes) and enables the RPC-free path for blobs whose meta carries a hint.

        Raises :class:`LostBlobs` only for confirmed loss (the unit lacks the
        blob, or the unit is dead) and :class:`TransferFailed` when a transfer
        or reply failed while the unit is still alive.
        """
        slow: dict[str, list[ReadRange]] = {}
        fast: dict[int, list[str]] = {}
        for b, ranges in requests.items():
            m = metas.get(b) or {}
            # A hint exists only for blobs in the put area, which never move; a
            # spilled blob has none and is read pinned.
            if scratch is not None and "so" in m and "z" in m and self._hint_live(m):
                fast.setdefault(int(m["u"]), []).append(b)
            else:
                slow[b] = ranges
        if fast:
            slow.update(self._read_hinted(requests, metas, fast, scratch, deadline))
        if slow:
            self.stats["slow_reads"] += len(slow)
            self._read_pinned(slow, deadline, metas)

    def _failed(self, unit_id: int, blobs: list[str], e: Exception) -> Exception:
        """Classify a failed read: lost if the unit is dead, else retryable."""
        if self._unit_dead(unit_id):
            return LostBlobs(blobs, f"unit {unit_id} is dead: {e}")
        return TransferFailed(blobs, str(e), in_flight=getattr(e, "in_flight", False))

    def _unit_dead(self, unit_id: int) -> bool:
        """Ask Ray whether a unit's actor is gone; mark it DEAD if so.

        Only runs after a failure, so the op path stays Ray-free. A unit that
        is alive but slow is not dead: its blobs are still there.
        """
        try:
            actor = ray.get_actor(
                f"NixlStorageUnit#{unit_id}", namespace=self.namespace
            )
            ray.get(actor.ping.remote(), timeout=10)
            return False
        except (ValueError, ray.exceptions.RayActorError):
            pass  # name gone or the actor died
        except Exception:  # noqa: BLE001 - a timeout or Ray hiccup: still alive
            return False
        try:
            ray.get(self.directory.set_unit_state.remote(unit_id, DEAD))
            self._refresh_units()
        except Exception:  # noqa: BLE001 - the verdict stands without the directory
            pass
        return True

    def _read_hinted(
        self, requests, metas, fast, scratch, deadline=None
    ) -> dict[str, list[ReadRange]]:
        sbase, sview = scratch
        fsz = blob_format.FOOTER_SIZE
        retry: dict[str, list[ReadRange]] = {}
        slot = 0
        # One READ per unit, all posted before any is waited for.
        reads: list[tuple[int, list[str], dict[str, int], tuple[Any, ...]]] = []
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
            reads.append(
                (unit_id, blobs, slots, ("READ", loc, rem, uinfo["agent_name"]))
            )
        errs = self.ep.transfer_many(
            [r[3] for r in reads], timeout_s=remaining(deadline, self.timeout_s)
        )
        for (unit_id, blobs, slots, _), e in zip(reads, errs):
            if e is not None:
                if e.in_flight:
                    # The READ may still land in these local ranges, so they
                    # cannot be refilled by the pinned path; fail the read.
                    raise self._failed(unit_id, blobs, e) from e
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

    def _read_pinned(
        self,
        requests: dict[str, list[ReadRange]],
        deadline: float | None = None,
        metas: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Locate + ``unit.resolve`` pin: blobs without a usable hint.

        Blobs a unit reports ``pending`` (being demoted or staged back from
        disk) are asked for again until the deadline: at once after a round
        that read something (its unpin frees staging room for the rest), else
        on a doubling backoff from ``PENDING_POLL_S``.
        """
        backoff = PENDING_POLL_S
        while requests:
            rest = self._read_pinned_once(requests, deadline, metas or {})
            if len(rest) < len(requests):
                backoff = PENDING_POLL_S
            elif rest:
                backoff = busy_wait(deadline, backoff)
            requests = rest

    def _read_pinned_once(
        self,
        requests: dict[str, list[ReadRange]],
        deadline: float | None,
        metas: dict[str, dict[str, Any]],
    ) -> dict[str, list[ReadRange]]:
        """One resolve + READ round; returns the requests still pending."""
        where = self._locate(list(requests), metas)
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
                resolved[u] = self.ctl.call(
                    self._addr(u),
                    "resolve",
                    timeout_s=remaining(deadline, self.ctl.timeout_ms / 1000),
                    blob_ids=bs,
                )
            except (TransferError, LostBlobs, DataPlaneTimeout) as e:
                for done_u, r in resolved.items():
                    self.ctl.notify(
                        self._units[done_u]["zmq"], "unpin", pin_id=r["pin_id"]
                    )
                if isinstance(e, DataPlaneTimeout):
                    raise
                raise self._failed(u, bs, e) from e
        try:
            missing = [b for r in resolved.values() for b in r["missing"]]
            if missing:
                raise LostBlobs(missing, "blob not on unit")
            failed = [b for r in resolved.values() for b in r.get("failed", ())]
            if failed:  # the unit still has the file, so the caller may retry
                raise TransferFailed(failed, "disk read failed on the unit")
            pending = {b for r in resolved.values() for b in r.get("pending", ())}
            reads: list[tuple[int, list[str], tuple[Any, ...]]] = []
            for u, blobs in by_unit.items():
                blobs = [b for b in blobs if b not in pending]
                if not blobs:
                    continue
                uinfo, r = self._unit(u), resolved[u]
                loc: list[tuple[int, int]] = []
                rem: list[tuple[int, int]] = []
                for b in blobs:
                    base = int(uinfo["slab_base"]) + int(r["offsets"][b])
                    for off, n, laddr in requests[b]:
                        loc.append((laddr, n))
                        rem.append((base + off, n))
                reads.append((u, blobs, ("READ", loc, rem, uinfo["agent_name"])))
            errs = self.ep.transfer_many(
                [r[2] for r in reads], timeout_s=remaining(deadline, self.timeout_s)
            )
            for (u, blobs, _), e in zip(reads, errs):
                if e is not None:
                    raise self._failed(u, blobs, e) from e
        finally:
            for u, r in resolved.items():
                self.ctl.notify(self._units[u]["zmq"], "unpin", pin_id=r["pin_id"])
        return {b: requests[b] for b in pending}

    # -- index-addressed reads and clears (rows live on unit gidx % N)
    def resolve_keys(
        self, by_unit: dict[int, list[str]], deadline: float | None
    ) -> dict[int, dict[str, Any]]:
        """One ``resolve_keys`` per unit, in turn; pins what it finds.

        On a failure, units already resolved are unpinned before the error is
        raised (``blob_ids`` of the error are the keys).
        """
        resolved: dict[int, dict[str, Any]] = {}
        for u, keys in by_unit.items():
            try:
                resolved[u] = self.ctl.call(
                    self._addr(u),
                    "resolve_keys",
                    timeout_s=remaining(deadline, self.ctl.timeout_ms / 1000),
                    keys=keys,
                )
            except (TransferError, LostBlobs, DataPlaneTimeout) as e:
                self.unpin_all(resolved)
                if isinstance(e, DataPlaneTimeout):
                    raise
                raise self._failed(u, keys, e) from e
        return resolved

    def read_entries(
        self,
        reads: dict[int, list[tuple[int, int, int]]],
        keys_of: dict[int, list[str]],
        deadline: float | None,
    ) -> None:
        """READ ``(slab_off, nbytes, local_addr)`` ranges, all units at once."""
        ops = []
        units = []
        for u, ranges in reads.items():
            uinfo = self._unit(u)
            base = int(uinfo["slab_base"])
            ops.append(
                (
                    "READ",
                    [(laddr, n) for _, n, laddr in ranges],
                    [(base + off, n) for off, n, _ in ranges],
                    uinfo["agent_name"],
                )
            )
            units.append(u)
        errs = self.ep.transfer_many(ops, timeout_s=remaining(deadline, self.timeout_s))
        for u, e in zip(units, errs):
            if e is not None:
                raise self._failed(u, keys_of[u], e) from e

    def unpin_all(self, resolved: dict[int, dict[str, Any]]) -> None:
        for u, r in resolved.items():
            self.ctl.notify(self._addr(u), "unpin", pin_id=r["pin_id"])

    def release_rows(
        self, by_unit: dict[int, list[int]], deadline: float | None
    ) -> None:
        """Awaited row-wide clear: one ``release_rows`` per unit.

        Rows on a dead unit are already gone, so clearing them succeeds; a live
        unit that does not answer fails the clear (retryable).
        """
        for u, gidxs in by_unit.items():
            try:
                self.ctl.call(
                    self._addr(u),
                    "release_rows",
                    timeout_s=remaining(deadline, self.ctl.timeout_ms / 1000),
                    gidxs=gidxs,
                )
            except (TransferError, LostBlobs) as e:
                err = self._failed(u, [str(g) for g in gidxs], e)
                if isinstance(err, LostBlobs):
                    continue
                raise err from e

    def close(self) -> None:
        self.ctl.close()
