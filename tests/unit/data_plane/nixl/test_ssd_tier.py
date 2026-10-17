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
"""SSD tier on a storage unit (design PR 2, minimal): a full slab answers
Busy while committed blobs are demoted to files; reads of a demoted blob are
pending until it is staged back, then pinned. CPU-only, no Ray actor.
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
from tests.unit.data_plane._nixl_helpers import bare_unit, stub_store


def _store(ctl, ssd=True):
    return stub_store(ctl, ssd=ssd)


KB = 1 << 10
BLOB = 128 * KB


def _unit(tmp_path, *, slab=1 << 20, staging=256 * KB, pickup_s=10.0, read_pin_s=60.0):
    """1 MiB slab: 768 KiB for puts (six 128 KiB blobs), 256 KiB staging (two)."""
    return bare_unit(
        slab,
        read_pin_s=read_pin_s,
        ssd_dir=str(tmp_path),
        staging_bytes=staging,
        pickup_s=pickup_s,
    )


def _put(u, b, fill, n=BLOB, commit=True):
    off = u.alloc(b, n, 1)
    u.slab[off : off + n] = fill
    if commit:
        assert u.commit(b)
    return off


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < end, "condition not reached"
        time.sleep(0.005)


def _fill(u, n=6):
    for i in range(n):
        _put(u, f"b{i}", i + 1)


def _alloc_eventually(u, b, n=BLOB):
    """What the client does: retry Busy until the unit admits the put."""
    end = time.monotonic() + 5
    while True:
        try:
            return u.alloc(b, n, 1)
        except DataPlaneBusy:
            assert time.monotonic() < end
            time.sleep(0.005)


def _read(u, b):
    """Resolve until ready, return the pinned copy's bytes and the pin id."""
    end = time.monotonic() + 5
    while True:
        r = u.resolve([b])
        if b in r["offsets"]:
            off = r["offsets"][b]
            return bytes(u.slab[off : off + u.blobs[b].nbytes]), r["pin_id"]
        assert b in r["pending"] and time.monotonic() < end, r
        u.unpin(r["pin_id"])
        time.sleep(0.005)


def test_full_slab_is_busy_while_demoting_then_admits(tmp_path):
    u = _unit(tmp_path)
    _fill(u)
    with pytest.raises(DataPlaneBusy):
        u.alloc("new", BLOB, 1)
    _alloc_eventually(u, "new")
    st = u.stats()
    assert st["demotions"] >= 1 and st["busy_replies"] >= 1
    rec = u.blobs["b0"]  # oldest first
    assert not rec.in_slab and rec.disk is not None
    assert set(os.pread(rec.disk, BLOB, 0)) == {1}
    assert not os.listdir(u.ssd_dir)  # unlinked at once: nothing left on disk


def test_demoted_blob_reads_back_through_staging(tmp_path):
    u = _unit(tmp_path)
    _fill(u)
    _alloc_eventually(u, "new")
    r = u.resolve(["b0"])
    assert r["pending"] == ["b0"] and not r["offsets"]
    data, pin = _read(u, "b0")
    assert set(data) == {1}
    assert u.resolve(["b0"])["offsets"]["b0"] >= u.staging_base
    u.unpin(pin)
    assert u.stats()["stages"] == 1


def test_victim_keeps_its_slab_copy_until_its_pins_end(tmp_path):
    u = _unit(tmp_path)
    _fill(u)
    pin = u.resolve(["b0"])["pin_id"]  # a reader holds b0's slab copy
    with pytest.raises(DataPlaneBusy):
        u.alloc("new", BLOB, 1)
    _wait(lambda: u.blobs["b0"].disk is not None)
    rec = u.blobs["b0"]
    assert rec.in_slab  # still pinned: not dropped
    assert u.resolve(["b0"])["pending"] == ["b0"]  # no new slab pins (§5.1)
    u.unpin(pin)
    assert not rec.in_slab


def test_uncommitted_blobs_are_never_demoted(tmp_path):
    u = _unit(tmp_path)
    for i in range(6):
        _put(u, f"w{i}", 1, commit=False)  # WRITEs not yet DONE
    with pytest.raises(UnitFull):
        u.alloc("new", BLOB, 1)
    assert u.stats()["demotions"] == 0


def test_release_during_demotion_frees_slab_and_file(tmp_path, monkeypatch):
    u = _unit(tmp_path)
    _fill(u)
    free0 = u.alloc_.free_bytes
    go, entered = threading.Event(), threading.Event()
    real = os.pwrite

    def slow_pwrite(fd, data, off):
        entered.set()
        go.wait(5)
        return real(fd, data, off)

    monkeypatch.setattr(nixl_storage_unit.os, "pwrite", slow_pwrite)
    with pytest.raises(DataPlaneBusy):
        u.alloc("new", BLOB, 1)
    assert entered.wait(5)
    u.release_many([("b0", 1)])  # cleared mid-copy: the demotion owns the free
    assert u.alloc_.free_bytes == free0
    go.set()
    _wait(lambda: not u._victims)
    _wait(lambda: not os.listdir(u.ssd_dir))
    assert "b0" not in u.blobs and not u._released_held
    assert u.alloc_.free_bytes == free0 + BLOB


def test_staging_evicts_only_copies_already_picked_up(tmp_path):
    u = _unit(tmp_path)
    _fill(u)
    for b in ("x", "y", "z"):  # demote b0, b1, b2 (one per Busy round)
        _alloc_eventually(u, b)
    assert all(not u.blobs[f"b{i}"].in_slab for i in range(3))
    _, p0 = _read(u, "b0")
    _read(u, "b1")  # staging (two blobs) now full; b1 stays pinned
    r = u.resolve(["b2"])
    assert r["pending"] == ["b2"]
    u.unpin(r["pin_id"])
    assert u.blobs["b2"].staged_off is None  # nothing evictable yet
    u.unpin(p0)  # b0 was picked up and is unpinned: evictable
    data, _ = _read(u, "b2")
    assert set(data) == {3}
    assert u.blobs["b0"].staged_off is None and u.blobs["b1"].staged_off is not None


def test_unreadable_file_is_reported_missing(tmp_path, monkeypatch):
    u = _unit(tmp_path)
    _fill(u)
    _alloc_eventually(u, "new")

    def eio(*a):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(nixl_storage_unit.os, "preadv", eio)
    assert u.resolve(["b0"])["pending"] == ["b0"]
    _wait(lambda: u.resolve(["b0"])["missing"] == ["b0"])


def test_released_disk_only_blob_closes_its_file(tmp_path):
    u = _unit(tmp_path)
    _fill(u)
    _alloc_eventually(u, "new")
    rec = u.blobs["b0"]
    u.release_many([("b0", 1)])
    assert rec.disk is None and "b0" not in u.blobs


def test_demote_after_s_demotes_old_blobs_without_pressure(tmp_path):
    u = _unit(tmp_path)
    u.demote_after_s = 0.0
    _put(u, "b0", 7)
    _wait(lambda: not u.blobs["b0"].in_slab, timeout=5)  # the ticker demoted it
    data, _ = _read(u, "b0")
    assert set(data) == {7}


def test_abort_after_a_lost_commit_reply_frees_each_copy_once(tmp_path):
    u = _unit(tmp_path)
    _fill(u)
    _alloc_eventually(u, "new")  # b0 committed, then demoted: slab copy freed
    rec = u.blobs["b0"]
    u.abort("b0", in_flight=False)  # the client never saw the commit reply
    assert rec.disk is None
    ranges = [(r.off, r.off + r.size) for r in u.alloc_._free]
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))  # no overlap
    assert u.alloc_.used_bytes == 6 * BLOB  # b1..b5 + "new"


def test_failed_demotion_is_not_retried_so_a_full_disk_ends_in_unitfull(
    tmp_path, monkeypatch
):
    u = _unit(tmp_path)
    _fill(u)

    def enospc(*a):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(nixl_storage_unit.os, "pwrite", enospc)
    with pytest.raises(UnitFull):
        _alloc_eventually(u, "new")
    assert u.stats()["demote_failures"] == 6


def test_unexpected_demotion_error_fails_like_a_disk_error(tmp_path, monkeypatch):
    u = _unit(tmp_path)
    _fill(u)

    def bug(*a):
        raise ValueError("not an OSError")

    monkeypatch.setattr(nixl_storage_unit.os, "pwrite", bug)
    with pytest.raises(UnitFull):  # not Busy forever on a stuck victim
        _alloc_eventually(u, "new")
    assert u.stats()["demote_failures"] == 6
    assert all(r.holds == 0 and not r.victim for r in u.blobs.values())


def test_freed_region_loses_its_footer():
    from nemo_rl.data_plane.nixl.blob_format import FOOTER_SIZE

    u = bare_unit(1 << 16)  # read_pin_s=0: freed at once
    off = u.alloc("a", 1000, 1)
    u.slab[off : off + 1000] = 7
    u.release("a", 1)
    assert not u.slab[off + 1000 - FOOTER_SIZE : off + 1000].any()
    assert u.slab[off] == 7  # only the footer is wiped


def test_checkpoint_is_refused_with_ssd_on(tmp_path):
    u = _unit(tmp_path)
    with pytest.raises(NotImplementedError):
        u.save_shard(str(tmp_path / "s"))


# ------------------------------------------------------------------ client
def test_put_commits_only_with_ssd_on():
    ctl = _Ctl({"alloc": lambda a: 64, "commit": lambda a: True})
    _store(ctl).put_segments([(1000, 0, 8)], 8, 1, blob_id="b")
    assert ctl.calls == ["alloc", "commit"]
    ctl = _Ctl({"alloc": lambda a: 64})
    _store(ctl, ssd=False).put_segments([(1000, 0, 8)], 8, 1, blob_id="b")
    assert ctl.calls == ["alloc"]


def test_failed_commit_fails_the_put_and_frees_the_region():
    ctl = _Ctl({"alloc": lambda a: 64, "commit": lambda a: False})
    with pytest.raises(blobstore.TransferFailed):
        _store(ctl).put_segments([(1000, 0, 8)], 8, 1, blob_id="b")
    assert ("abort", {"blob_id": "b", "in_flight": False}) in ctl.notified


def test_deadline_during_commit_aborts_the_blob():
    from nemo_rl.data_plane.nixl.errors import DataPlaneTimeout

    ctl = _Ctl({"alloc": lambda a: 64, "commit": lambda a: DataPlaneTimeout("late")})
    with pytest.raises(DataPlaneTimeout):
        _store(ctl).put_segments([(1000, 0, 8)], 8, 1, blob_id="b")
    assert ("abort", {"blob_id": "b", "in_flight": False}) in ctl.notified


def test_pending_blobs_are_asked_again_until_ready():
    replies = iter(
        [
            {"pin_id": "p1", "offsets": {}, "missing": [], "pending": ["b"]},
            {"pin_id": "p2", "offsets": {}, "missing": [], "pending": ["b"]},
            {"pin_id": "p3", "offsets": {"b": 0}, "missing": [], "pending": []},
        ]
    )
    ctl = _Ctl({"resolve": lambda a: next(replies)})
    _store(ctl)._read_pinned({"b": [(0, 8, 1000)]}, deadline=time.monotonic() + 5)
    assert ctl.calls == ["resolve"] * 3
    assert [a["pin_id"] for op, a in ctl.notified if op == "unpin"] == [
        "p1",
        "p2",
        "p3",
    ]


def test_pending_reads_reask_at_once_after_progress_and_reset_the_backoff(monkeypatch):
    # A round that read blobs unpins them, freeing staging for the rest, so
    # the next ask goes out at once; only rounds that read nothing back off.
    pending = iter([["b", "c"], ["b", "c"], ["b", "c"], ["c"], ["c"], []])

    def resolve(a):
        p = next(pending)
        offs = {b: 0 for b in "abc" if b not in p}
        return {"pin_id": "p", "offsets": offs, "missing": [], "pending": p}

    sleeps = []

    def wait(deadline, backoff):
        sleeps.append(backoff)
        return 2 * backoff

    monkeypatch.setattr(blobstore, "busy_wait", wait)
    ctl = _Ctl({"resolve": resolve})
    s = _store(ctl)
    s._blob_unit = {b: 0 for b in "abc"}
    s._read_pinned({b: [(0, 8, 1000)] for b in "abc"}, deadline=time.monotonic() + 5)
    assert ctl.calls == ["resolve"] * 6
    p0 = blobstore.PENDING_POLL_S
    assert sleeps == [p0, 2 * p0, p0]


def test_with_ssd_on_reads_are_pinned_and_find_other_clients_blobs_by_meta():
    s = _store(
        _Ctl({"resolve": lambda a: {"pin_id": "p", "offsets": {"b": 0}, "missing": []}})
    )
    s._read_hinted = lambda *a, **k: pytest.fail("hinted read with ssd on")
    s._blob_unit = {}  # this client never wrote "b"
    s.directory = None  # asking the directory would raise
    meta = {"u": 0, "so": 0, "z": 64, "g": "abc"}
    s.read(
        {"b": [(0, 8, 1000)]}, {"b": meta}, scratch=(2000, memoryview(bytearray(64)))
    )
    assert s.stats["slow_reads"] == 1


# ------------------------------------------------------------------ config
def test_ssd_config_validation():
    assert tq_nixl.ssd_config({}) is None
    on = {"unit_slab_bytes": 1 << 30, "ssd": {"enabled": True, "dir": "/x"}}
    assert tq_nixl.ssd_config(on) == {
        "ssd_dir": "/x",
        "staging_bytes": 1 << 28,
        "pickup_s": 10.0,
        "demote_after_s": None,
    }
    with pytest.raises(ValueError, match="pickup_s"):
        tq_nixl.ssd_config({**on, "ssd": {"enabled": True, "pickup_s": 1.0}})
    with pytest.raises(ValueError, match="staging_bytes"):
        tq_nixl.ssd_config({**on, "ssd": {"enabled": True, "staging_bytes": 1 << 30}})
    with pytest.raises(ValueError, match="store.kind"):
        tq_nixl.ssd_config({**on, "store": {"kind": "file"}})
