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
"""SSD tier on a storage unit: a put the put area cannot take is spilled to a
file from birth (its commit is pending until the file is written); data already
in the put area never moves; reads of a spilled blob are pending until it is
staged back, then pinned. CPU-only, no Ray actor.
"""

import os
import threading
import time

import pytest

from nemo_rl.data_plane.adapters import tq_nixl
from nemo_rl.data_plane.nixl import blobstore
from nemo_rl.data_plane.nixl.errors import DataPlaneBusy, UnitFull
from nemo_rl.data_plane import nixl_storage_unit
from tests.unit.data_plane._nixl_helpers import StubCtl as _Ctl
from nemo_rl.data_plane.nixl import blob_format
from tests.unit.data_plane._nixl_helpers import (
    bare_unit,
    gidx_of,
    key_of,
    stub_store,
    write_blob,
)


def _store(ctl, ssd=True):
    return stub_store(ctl, ssd=ssd)


KB = 1 << 10
BLOB = 128 * KB


def _unit(tmp_path, *, slab=1 << 20, staging=256 * KB, pickup_s=10.0, read_pin_s=60.0):
    """1 MiB slab: 768 KiB put area (six 128 KiB blobs), 256 KiB IO area (two)."""
    return bare_unit(
        slab,
        read_pin_s=read_pin_s,
        ssd_dir=str(tmp_path),
        staging_bytes=staging,
        pickup_s=pickup_s,
    )


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < end, "condition not reached"
        time.sleep(0.005)


def _commit(u, b):
    """What the client does: ask again while the unit says pending."""
    end = time.monotonic() + 5
    while (status := u.commit(b)) == "pending":
        assert time.monotonic() < end, "commit still pending"
        time.sleep(0.005)
    return status


def _write(u, b, off, fill, n=BLOB):
    """The client's WRITE: a valid one-key blob whose value bytes are ``fill``."""
    write_blob(u.slab, off, n, key_of(b), fill)


def _value(data):
    """The value bytes of a one-key blob (everything before its index)."""
    return data[: blob_format._footer(memoryview(data))[0]]


def _put(u, b, fill, n=BLOB):
    off = u.alloc(b, n, 1)
    _write(u, b, off, fill, n)
    assert _commit(u, b) == "done"
    return off


def _fill(u, n=6):
    for i in range(n):
        _put(u, f"b{i}", i + 1)


def _read(u, b):
    """Resolve until ready, return the pinned copy's bytes and the pin id."""
    end = time.monotonic() + 5
    while True:
        r = u.resolve([b])
        if b in r["offsets"]:
            off = r["offsets"][b]
            return _value(bytes(u.slab[off : off + u.blobs[b].nbytes])), r["pin_id"]
        assert b in r["pending"] and time.monotonic() < end, r
        u.unpin(r["pin_id"])
        time.sleep(0.005)


def _gate_pwrite(monkeypatch):
    """Make the first disk write block until ``go`` is set."""
    go, entered = threading.Event(), threading.Event()
    real = os.pwrite

    def gated(fd, data, off):
        entered.set()
        go.wait(5)
        return real(fd, data, off)

    monkeypatch.setattr(nixl_storage_unit.os, "pwrite", gated)
    return go, entered


def _open_fds():
    return len(os.listdir("/proc/self/fd"))


# ------------------------------------------------------------------ spill
def test_a_put_the_put_area_cannot_take_spills_and_reads_back(tmp_path):
    u = _unit(tmp_path)
    _fill(u)  # six blobs fill the put area
    off = _put(u, "s", 7)
    assert off >= u.staging_base  # the client wrote into the IO area
    rec = u.blobs["s"]
    assert not rec.in_slab and rec.disk is not None
    assert set(_value(os.pread(rec.disk, BLOB, 0))) == {7}
    assert not os.listdir(u.ssd_dir)  # unlinked at once: nothing left on disk
    assert u.staging_.used_bytes == 0  # the IO region went back at once
    assert u.stats()["spills"] == 1

    r = u.resolve(["s"])
    assert r["pending"] == ["s"] and not r["offsets"]
    data, pin = _read(u, "s")
    assert set(data) == {7}
    assert u.resolve(["s"])["offsets"]["s"] >= u.staging_base
    u.unpin(pin)
    assert u.stats()["stages"] == 1


def test_data_in_the_put_area_never_moves_so_the_newest_goes_to_disk(tmp_path):
    u = _unit(tmp_path)
    _fill(u)
    for i in range(4):  # the IO area is free again after each write
        _put(u, f"s{i}", 10 + i)
    assert all(u.blobs[f"b{i}"].in_slab for i in range(6))  # FIFO: oldest stay in RAM
    assert all(u.blobs[f"s{i}"].disk is not None for i in range(4))
    assert u.stats()["spills"] == 4
    assert set(_read(u, "b0")[0]) == {1}


def test_commit_is_pending_until_the_file_is_written_and_enqueues_once(
    tmp_path, monkeypatch
):
    u = _unit(tmp_path)
    _fill(u)
    go, entered = _gate_pwrite(monkeypatch)
    off = u.alloc("s", BLOB, 1)
    _write(u, "s", off, 5)
    assert u.commit("s") == "pending"
    assert entered.wait(5)
    assert u.commit("s") == "pending" and u._io_jobs == 1  # asked again: no 2nd job
    assert u.resolve(["s"])["missing"] == ["s"]  # not acknowledged yet: invisible
    go.set()
    assert _commit(u, "s") == "done"
    assert u.stats()["spills"] == 1


@pytest.mark.parametrize(
    "fail",
    [
        pytest.param(
            lambda *a: (_ for _ in ()).throw(OSError(28, "ENOSPC")), id="enospc"
        ),
        pytest.param(lambda *a: (_ for _ in ()).throw(ValueError("bug")), id="bug"),
        pytest.param(lambda *a: 0, id="zero-progress"),
    ],
)
def test_a_failed_write_fails_the_put_and_frees_the_io_area(
    tmp_path, monkeypatch, fail
):
    u = _unit(tmp_path)
    _fill(u)
    fds = _open_fds()
    monkeypatch.setattr(nixl_storage_unit.os, "pwrite", fail)
    _write(u, "s", u.alloc("s", BLOB, 1), 5)
    assert _commit(u, "s") == "lost"  # never acknowledged, so nothing is lost
    assert "s" not in u.blobs and u._io_jobs == 0
    assert u.staging_.used_bytes == 0 and u.stats()["spill_failures"] == 1
    assert _open_fds() == fds  # the partial file's fd is closed


def test_a_put_that_fits_nowhere_is_unitfull_and_a_full_io_area_is_busy(tmp_path):
    u = _unit(tmp_path)
    _fill(u)
    with pytest.raises(UnitFull):  # bigger than the IO area, put area full
        u.alloc("big", u.staging_.capacity + BLOB, 1)
    u.alloc("s", u.staging_.capacity, 1)  # a put in progress holds the IO area
    with pytest.raises(DataPlaneBusy):
        u.alloc("t", BLOB, 1)
    assert u.stats()["busy_replies"] == 1


def test_alloc_without_spill_is_unitfull_and_leaves_the_io_area_alone(tmp_path):
    u = _unit(tmp_path)
    _fill(u)
    with pytest.raises(UnitFull):
        u.alloc("x", BLOB, 1, spill=False)
    assert u.staging_.used_bytes == 0 and "x" not in u.blobs
    u.alloc("y", BLOB, 1)  # the default still spills
    assert u.stats()["spills"] == 0 and u.blobs["y"].staged_off is not None


class _SpillCtl(_Ctl):
    """Records each alloc as ``(address, spill)``; ``full`` units refuse no-spill ones."""

    def __init__(self, full):
        super().__init__({"commit": lambda a: "done"})
        self.full, self.allocs = full, []

    def call(self, address, op, *, timeout_s=None, **args):
        if op != "alloc":
            return super().call(address, op, timeout_s=timeout_s, **args)
        self.allocs.append((address, args["spill"]))
        if address in self.full and not args["spill"]:
            raise UnitFull("ram full")
        return 64


@pytest.mark.parametrize(
    ("full", "allocs"),
    [
        (set(), [("tcp://u1", False)]),  # RAM has room: no spill
        ({"tcp://u1"}, [("tcp://u1", False), ("tcp://u1", True)]),  # then it spills
    ],
    ids=["ram", "spill_after_ram_refused"],
)
def test_a_put_tries_its_units_ram_before_it_spills(full, allocs):
    ctl = _SpillCtl(full)
    assert _store(ctl).put_segments([(1000, 0, 8)], 8, 1, 1, blob_id="b") == "b"
    assert ctl.allocs == allocs


def test_commit_waits_when_too_many_io_jobs_are_running(tmp_path, monkeypatch):
    u = _unit(tmp_path)
    u._max_io_jobs = 1
    _fill(u)
    go, entered = _gate_pwrite(monkeypatch)
    for b in ("s0", "s1"):
        _write(u, b, u.alloc(b, 64 * KB, 1), 3, 64 * KB)
    assert u.commit("s0") == "pending" and entered.wait(5)
    assert u.commit("s1") == "pending" and not u.blobs["s1"].staging  # not queued yet
    go.set()
    assert _commit(u, "s0") == "done" and _commit(u, "s1") == "done"


# ------------------------------------------------------------------ abort / release
def test_abort_before_commit_restores_the_io_area(tmp_path):
    u = _unit(tmp_path)
    _fill(u)
    cap = u.staging_.capacity
    u.alloc("s", cap, 1)
    u.abort("s", in_flight=False)  # no WRITE was posted
    u.alloc("t", cap, 1)  # the whole area is back


def test_abort_with_a_write_in_flight_retires_the_io_region(tmp_path):
    u = _unit(tmp_path)
    _fill(u)
    cap = u.staging_.capacity
    u.alloc("s", cap, 1)
    u.abort("s", in_flight=True)  # a late WRITE may still land: never reused
    assert u.retired_bytes == cap and u.staging_.used_bytes == cap


def test_abort_during_the_write_leaves_the_cleanup_to_the_job(tmp_path, monkeypatch):
    u = _unit(tmp_path)
    _fill(u)
    fds = _open_fds()
    go, entered = _gate_pwrite(monkeypatch)
    _write(u, "s", u.alloc("s", BLOB, 1), 5)
    u.commit("s")
    assert entered.wait(5)
    used = u.staging_.used_bytes
    u.abort("s", in_flight=False)  # the client gave up on a slow commit
    assert u.staging_.used_bytes == used  # the job still reads the region
    go.set()
    _wait(lambda: u._io_jobs == 0)
    assert u.staging_.used_bytes == 0 and "s" not in u.blobs
    assert u.stats()["spills"] == 0 and _open_fds() == fds


def test_released_disk_blob_closes_its_file(tmp_path):
    u = _unit(tmp_path)
    _fill(u)
    _put(u, "s", 7)
    rec = u.blobs["s"]
    assert u.release_rows([gidx_of("s")]) == 1
    assert rec.disk is None and "s" not in u.blobs and gidx_of("s") not in u.rows


def test_abort_after_a_lost_commit_reply_keeps_the_published_blob(tmp_path):
    u = _unit(tmp_path)
    _fill(u)
    _put(u, "s", 7)  # committed and on disk, but the client never saw "done"
    rec = u.blobs["s"]
    u.abort("s", in_flight=False)  # its rows are live: only a clear drops them
    assert u.blobs["s"] is rec and rec.disk is not None
    assert u.commit("s") == "done"  # asking again is safe
    assert u.release_rows([gidx_of("s")]) == 1 and rec.disk is None


# ------------------------------------------------------------------ submit failure
def test_a_failed_commit_submit_leaves_the_blob_reserved(tmp_path):
    u = _unit(tmp_path)
    _fill(u)
    _write(u, "s", u.alloc("s", BLOB, 1), 5)
    u._worker.shutdown(wait=True)
    with pytest.raises(RuntimeError):
        u.commit("s")
    assert not u.blobs["s"].staging and u._io_jobs == 0  # nothing to undo
    u.abort("s", in_flight=False)
    assert u.staging_.used_bytes == 0


def test_a_failed_stage_submit_rolls_the_staging_back(tmp_path):
    u = _unit(tmp_path)
    _fill(u)
    _put(u, "s", 7)
    u._worker.shutdown(wait=True)
    with pytest.raises(RuntimeError):
        u.resolve(["s"])
    rec = u.blobs["s"]
    assert u.staging_.used_bytes == 0 and rec.staged_off is None
    assert not rec.staging and "s" not in u._staged and u._io_jobs == 0


# ------------------------------------------------------------------ reads
def test_staged_copies_are_evicted_only_after_pickup(tmp_path):
    u = _unit(tmp_path)
    _fill(u)
    for i in range(3):
        _put(u, f"s{i}", 10 + i)
    _, p0 = _read(u, "s0")
    _read(u, "s1")  # the IO area (two blobs) is now full; s1 stays pinned
    r = u.resolve(["s2"])
    assert r["pending"] == ["s2"]
    u.unpin(r["pin_id"])
    assert u.blobs["s2"].staged_off is None  # nothing evictable yet
    u.unpin(p0)  # s0 was picked up and is unpinned: evictable
    data, _ = _read(u, "s2")
    assert set(data) == {12}
    assert u.blobs["s0"].staged_off is None and u.blobs["s1"].staged_off is not None


def test_a_failed_stage_read_keeps_the_file_and_the_retry_succeeds(
    tmp_path, monkeypatch
):
    u = _unit(tmp_path)
    _fill(u)
    _put(u, "s", 7)
    real, calls = os.preadv, []

    def flaky(fd, bufs, off):
        calls.append(1)
        if len(calls) == 1:
            raise OSError(5, "Input/output error")
        return real(fd, bufs, off)

    monkeypatch.setattr(nixl_storage_unit.os, "preadv", flaky)
    assert u.resolve(["s"])["pending"] == ["s"]
    _wait(lambda: u.blobs["s"].stage_failed)
    r = u.resolve(["s"])
    assert r["failed"] == ["s"] and not r["pending"] and not r["offsets"]
    assert u.blobs["s"].disk is not None  # the only copy is kept
    data, _ = _read(u, "s")  # reported once; the next resolve retries
    assert set(data) == {7}
    assert u.stats()["stage_failures"] == 1


def _spilled(u):
    """Six blobs in the put area, two on disk: fills 1..8."""
    _fill(u)
    _put(u, "b6", 7)
    _put(u, "b7", 8)
    assert u.stats()["disk_only_blobs"] == 2


def _fill_of(b):
    return int(b[1:]) + 1


def test_checkpoint_saves_put_area_and_disk_blobs(tmp_path):
    u = _unit(tmp_path / "u")
    _spilled(u)
    fds = _open_fds()
    res = u.save_shard(str(tmp_path / "s"))
    assert _open_fds() == fds  # each dup'd fd is closed after its copy
    assert sorted(res["blobs"]) == [f"b{i}" for i in range(8)]
    raw = (tmp_path / "s").read_bytes()
    assert res["nbytes"] == len(raw) == 8 * BLOB
    for b, (off, n, refs) in res["blobs"].items():
        assert set(_value(raw[off : off + n])) == {_fill_of(b)} and refs == 1
    assert all(r.holds == 0 for r in u.blobs.values())


def test_checkpoint_raises_if_a_disk_blob_is_released_during_it(tmp_path, monkeypatch):
    u = _unit(tmp_path / "u")
    _spilled(u)
    dup = os.dup

    def release_b7_first(fd):
        rec = u.blobs.pop("b7", None)  # a clear, then its stage closes the file
        if rec is not None:
            os.close(rec.disk)
            rec.disk = None
        return dup(fd)

    monkeypatch.setattr(nixl_storage_unit.os, "dup", release_b7_first)
    with pytest.raises(RuntimeError, match="released during save_shard"):
        u.save_shard(str(tmp_path / "s"))
    assert all(r.holds == 0 for r in u.blobs.values())
    assert not u.checkpointing


def test_checkpoint_skips_a_spill_not_acknowledged_yet(tmp_path, monkeypatch):
    u = _unit(tmp_path / "u")
    _fill(u)
    go, entered = _gate_pwrite(monkeypatch)
    _write(u, "s", u.alloc("s", BLOB, 1), 9)
    assert u.commit("s") == "pending"
    assert entered.wait(5)
    res = u.save_shard(str(tmp_path / "s"))
    go.set()
    assert "s" not in res["blobs"] and len(res["blobs"]) == 6


def test_restore_spills_only_when_asked_and_reads_back(tmp_path):
    src = _unit(tmp_path / "a")
    _spilled(src)
    res = src.save_shard(str(tmp_path / "s"))
    dst = _unit(tmp_path / "b")
    first = dst.load_shard(str(tmp_path / "s"), res["blobs"])
    assert len(first["loaded"]) == 6 and len(first["failed"]) == 2  # RAM only
    left = {b: res["blobs"][b] for b in first["failed"]}
    second = dst.load_shard(str(tmp_path / "s"), left, spill=True)
    assert sorted(second["loaded"]) == sorted(left) and not second["failed"]
    assert dst.stats()["disk_only_blobs"] == 2
    for b in res["blobs"]:
        data, pin = _read(dst, b)
        assert set(data) == {_fill_of(b)}, b
        dst.unpin(pin)


def test_restore_with_a_full_disk_leaves_the_blob_to_another_unit(
    tmp_path, monkeypatch
):
    src = _unit(tmp_path / "a")
    _spilled(src)
    res = src.save_shard(str(tmp_path / "s"))
    dst = _unit(tmp_path / "b")
    _fill(dst)  # put area full

    def no_space(*a):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(nixl_storage_unit.os, "pwrite", no_space)
    blobs = {f"r{b}": v for b, v in res["blobs"].items()}
    out = dst.load_shard(str(tmp_path / "s"), blobs, spill=True)
    assert not out["loaded"] and len(out["failed"]) == 8
    assert not any(b.startswith("r") for b in dst.blobs)


# ------------------------------------------------------------------ client
@pytest.mark.parametrize("ssd", [True, False])
def test_every_put_commits(ssd):
    # The commit publishes the blob's rows on the unit, SSD tier or not.
    ctl = _Ctl({"alloc": lambda a: 64, "commit": lambda a: "done"})
    _store(ctl, ssd=ssd).put_segments([(1000, 0, 8)], 8, 1, 0, blob_id="b")
    assert ctl.calls == ["alloc", "commit"]


def test_commit_is_asked_again_while_the_unit_writes_the_file():
    replies = iter(["pending", "pending", "done"])
    ctl = _Ctl({"alloc": lambda a: 64, "commit": lambda a: next(replies)})
    _store(ctl).put_segments([(1000, 0, 8)], 8, 1, 0, blob_id="b")
    assert ctl.calls == ["alloc", "commit", "commit", "commit"]
    assert not [n for n in ctl.notified if n[0] == "abort"]


def test_failed_commit_fails_the_put_and_frees_the_region():
    ctl = _Ctl({"alloc": lambda a: 64, "commit": lambda a: "lost"})
    with pytest.raises(blobstore.TransferFailed):
        _store(ctl).put_segments([(1000, 0, 8)], 8, 1, 0, blob_id="b")
    assert ("abort", {"blob_id": "b", "in_flight": False}) in ctl.notified


def test_deadline_during_commit_aborts_the_blob():
    from nemo_rl.data_plane.nixl.errors import DataPlaneTimeout

    ctl = _Ctl({"alloc": lambda a: 64, "commit": lambda a: DataPlaneTimeout("late")})
    with pytest.raises(DataPlaneTimeout):
        _store(ctl).put_segments([(1000, 0, 8)], 8, 1, 0, blob_id="b")
    assert ("abort", {"blob_id": "b", "in_flight": False}) in ctl.notified


# ------------------------------------------------------------------ config
def test_ssd_config_validation():
    assert tq_nixl.ssd_config({}) is None
    on = {"unit_slab_bytes": 1 << 30, "ssd": {"enabled": True, "dir": "/x"}}
    assert tq_nixl.ssd_config(on) == {
        "ssd_dir": "/x",
        "staging_bytes": 1 << 28,
        "pickup_s": 10.0,
    }
    with pytest.raises(ValueError, match="pickup_s"):
        tq_nixl.ssd_config({**on, "ssd": {"enabled": True, "pickup_s": 1.0}})
    with pytest.raises(ValueError, match="staging_bytes"):
        tq_nixl.ssd_config({**on, "ssd": {"enabled": True, "staging_bytes": 1 << 30}})
