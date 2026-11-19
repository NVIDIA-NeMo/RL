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
"""UnitSlabStore: blob bytes on the storage units' slabs.

The client lays values out as raw bytes in registered local memory and hands
the store the segments and the owner unit; on read it hands the store slab
ranges and registered destination addresses. Pins, refcounts and the medium
are the units' business: pinned-DRAM slabs on CPU StorageUnit actors,
one-sided RDMA over UCX, with an optional SSD tier on each unit.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from typing import Any

import ray

from nemo_rl.data_plane.nixl.control import UnitClient
from nemo_rl.data_plane.nixl.directory import DEAD
from nemo_rl.data_plane.nixl.errors import (
    DataPlaneBusy,
    DataPlaneTimeout,
    LostBlobs,
    TransferError,
    UnitFull,
)
from nemo_rl.data_plane.nixl.nixl_io import NixlEndpoint

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


# ============================================================================ unit slabs
class UnitSlabStore:
    """Pinned-DRAM slabs on NixlStorageUnit actors, one-sided RDMA.

    No Ray on the op path: control messages go to the units over ZMQ
    (nemo_rl/data_plane/nixl/control.py), bytes over NIXL. The caller names
    the unit of every op (rows live on unit ``gidx % N``).

    - put:   ZMQ ``alloc`` + NIXL WRITE + ZMQ ``commit`` (publishes the rows).
    - get:   one ZMQ ``resolve_keys`` per unit (pins) + one NIXL READ per unit.
    - clear: one awaited ZMQ ``release_rows`` per unit (row-wide).

    Ray remains only off the op path: the unit list at start (and when a unit
    is unknown), and the liveness check after a failure.
    """

    def __init__(
        self,
        ep: NixlEndpoint,
        directory: Any,
        *,
        namespace: str,
        timeout_s: float,
        ssd: bool = False,
        lock: Any = None,
    ) -> None:
        self.ep = ep
        # SSD tier on the units: a put that RAM cannot take spills to disk.
        self.ssd = ssd
        # The caller's lock (TimedLock), released while a put sleeps on Busy or
        # a pending commit, so the process's other ops do not wait on a disk write.
        self.idle = lock.released if lock is not None else contextlib.nullcontext
        self.directory = directory
        self.namespace = namespace
        self.timeout_s = timeout_s
        self._units: dict[int, dict[str, Any]] = {}
        self._epoch = 0
        self.ctl = UnitClient(timeout_s=min(timeout_s, 30.0))
        self.stats = {"unit_refresh": 0}
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

    def _addr(self, unit_id: int) -> str:
        u = self._units.get(unit_id)
        if u is None:
            self._refresh_units()
            u = self._units.get(unit_id)
        if u is None or not u.get("zmq"):
            raise LostBlobs([], f"unit {unit_id} unknown to the directory")
        return u["zmq"]

    def put_segments(
        self,
        segments: list[tuple[int, int, int]],
        nbytes: int,
        n_entries: int,
        unit_id: int,
        blob_id: str | None = None,
        deadline: float | None = None,
    ) -> str:  # pyrefly: ignore[bad-return]  the retry loop only returns or raises
        """Write one blob of ``nbytes`` to unit ``unit_id``; return its id.

        ``segments`` are ``(local_addr, blob_off, n)``: a value sent zero-copy
        from the caller's own (registered) memory, or a run copied into a pool
        buffer. All go out in one NIXL transfer.

        Every wait is bounded by ``deadline`` (monotonic). Raises
        :class:`UnitFull` when the unit refused for lack of space,
        :class:`TransferFailed` when it failed otherwise, and
        :class:`DataPlaneTimeout` when the deadline passes (also while the
        unit keeps answering Busy, design §5.3).
        """
        blob_id = blob_id or uuid.uuid4().hex
        # With the SSD tier on, the unit's RAM is tried before it spills.
        spills = (False, True) if self.ssd else (True,)
        backoff = BUSY_RETRY_S
        refreshed = False
        while True:
            errs: list[Exception] = []
            if unit_id not in self._units:
                self._refresh_units()
            if unit_id not in self._units:
                raise LostBlobs([blob_id], f"unit {unit_id} unknown to the directory")
            for spill in spills:
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
                return blob_id
            if any(isinstance(e, DataPlaneBusy) for e in errs):
                with self.idle():  # nothing is posted
                    backoff = busy_wait(deadline, backoff)
                continue
            if not refreshed:
                self._refresh_units()  # the unit's address may have changed; one retry
                refreshed = True
                continue
            if errs and all(isinstance(e, UnitFull) for e in errs):
                raise UnitFull(f"unit {unit_id} could not take {nbytes} B: {errs[-1]}")
            raise TransferFailed(
                [blob_id],
                f"unit {unit_id} did not take {nbytes} B: {errs[-1] if errs else '?'}",
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
                with self.idle():  # the WRITE is done
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

    def put_many(
        self,
        blobs: list[tuple[list[tuple[int, int, int]], int, int, int, str]],
        deadline: float | None = None,
    ) -> None:
        """``put_segments`` for several ``(segments, nbytes, n_entries, unit_id, blob_id)``.

        The first blob of each unit goes out in one round over all units (alloc,
        one WRITE batch, commit), so a put waits for the slowest unit, not the
        sum. A blob that does not finish that round (busy, full, spill, no reply,
        lost) and any further blob of a unit take ``put_segments``, one by one.
        """
        first = {b[3]: b for b in reversed(blobs)}
        done = self._put_round(first, deadline) if len(first) > 1 else set()
        for segments, nbytes, n_entries, unit_id, blob_id in blobs:
            if blob_id not in done:
                self.put_segments(
                    segments, nbytes, n_entries, unit_id, blob_id, deadline
                )

    def _put_round(self, by_unit: dict[int, tuple], deadline: float | None) -> set[str]:
        """One alloc / WRITE / commit round, one blob per unit; the blob ids committed."""
        offs, failed = self._call_units(
            "alloc",
            {
                u: {
                    "blob_id": b[4],
                    "nbytes": int(b[1]),
                    "n_entries": int(b[2]),
                    "spill": not self.ssd,
                }  # with the SSD tier, RAM first
                for u, b in by_unit.items()
            },
            deadline,
        )

        def abort(u: int, in_flight: bool = False) -> None:
            self.ctl.notify(
                self._addr(u), "abort", blob_id=by_unit[u][4], in_flight=in_flight
            )

        for u, e in failed:
            if isinstance(e, TransferError):  # no reply: the unit may still allocate
                abort(u)
        ops, infos = [], {}
        for u, off in offs.items():
            try:
                infos[u] = info = self._unit(u)
            except LostBlobs:  # NIXL cannot reach it: nothing was posted
                abort(u)
                continue
            base, segs = int(info["slab_base"]) + off, by_unit[u][0]
            remote = [(base + bo, n) for _, bo, n in segs]
            ops.append(
                ("WRITE", [(a, n) for a, _, n in segs], remote, info["agent_name"])
            )
        try:
            left = remaining(deadline, self.timeout_s)
        except DataPlaneTimeout:  # nothing was posted: free the regions at once
            for u in infos:
                abort(u)
            raise
        written = []
        for u, e in zip(
            infos, self.ep.transfer_many(ops, timeout_s=left) if ops else []
        ):
            if e is None:
                written.append(u)
            else:
                abort(u, e.in_flight)
        if not written:
            return set()
        try:
            status, _ = self._call_units(
                "commit", {u: {"blob_id": by_unit[u][4]} for u in written}, deadline
            )
        except BaseException:
            for u in written:
                abort(u)
            raise
        stored: set[str] = set()
        for u in written:
            blob_id = by_unit[u][4]
            # Not "done" in the round (a spill still pending, no reply): commit's own polling.
            if status.get(u) == "done" or self._commit(infos[u], blob_id, deadline, []):
                stored.add(blob_id)
        return stored

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

    # -- index-addressed reads and clears (rows live on unit gidx % N)
    def resolve_keys(
        self, by_unit: dict[int, list[str]], deadline: float | None
    ) -> dict[int, dict[str, Any]]:
        """One ``resolve_keys`` per unit, all units at once; pins what it finds.

        Sent to every unit before any reply is read, so a get waits for the
        slowest unit rather than the sum of them. On a failure, the units that
        resolved are unpinned before the error is raised (``blob_ids`` of the
        error are the keys).
        """
        resolved, failed = self._call_units(
            "resolve_keys", {u: {"keys": k} for u, k in by_unit.items()}, deadline
        )
        if failed:
            self.unpin_all(resolved)
            u, e = failed[0]
            if isinstance(e, (TransferError, LostBlobs)):
                raise self._failed(u, by_unit[u], e) from e
            raise e
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
        """Awaited row-wide clear: one ``release_rows`` per unit, all units at once.

        Rows on a dead unit are already gone, so clearing them succeeds; a live
        unit that does not answer fails the clear (retryable).
        """
        _, failed = self._call_units(
            "release_rows", {u: {"gidxs": g} for u, g in by_unit.items()}, deadline
        )
        for u, e in failed:
            if not isinstance(e, (TransferError, LostBlobs)):
                raise e
            err = self._failed(u, [str(g) for g in by_unit[u]], e)
            if not isinstance(err, LostBlobs):
                raise err from e

    def _call_units(
        self, op: str, args_by_unit: dict[int, dict[str, Any]], deadline: float | None
    ) -> tuple[dict[int, Any], list[tuple[int, Exception]]]:
        """``op`` on every unit at once: results by unit, and ``(unit, error)`` failures.

        A unit unknown to the directory fails without being asked.
        """
        unit_of: dict[str, int] = {}
        failed: list[tuple[int, Exception]] = []
        for u in args_by_unit:
            try:
                unit_of[self._addr(u)] = u
            except LostBlobs as e:
                failed.append((u, e))
        replies = self.ctl.call_many(
            op,
            {a: args_by_unit[u] for a, u in unit_of.items()},
            timeout_s=remaining(deadline, self.ctl.timeout_ms / 1000),
        )
        failed += [
            (unit_of[a], r) for a, r in replies.items() if isinstance(r, Exception)
        ]
        return {
            unit_of[a]: r for a, r in replies.items() if not isinstance(r, Exception)
        }, failed

    def close(self) -> None:
        self.ctl.close()
