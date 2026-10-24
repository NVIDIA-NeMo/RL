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
"""save_shard does its file I/O outside the unit lock and holds what it copies
(design §7): a blob released mid-save is freed only after it was copied, and
every exit drops exactly the holds taken. CPU-only, no Ray actor.
"""

import os
import threading

import pytest

import msgspec

from nemo_rl.data_plane import nixl_storage_unit
from nemo_rl.data_plane.nixl import blob_format
from tests.unit.data_plane._nixl_helpers import bare_unit, gidx_of, key_of, write_blob


def _unit(slab_bytes: int = 1 << 16):
    # read_pin_s 0: free at once, so the test sees the region come back
    return bare_unit(slab_bytes)


def _put(u, blob_id: str, nbytes: int, fill: int) -> int:
    off = u.alloc(blob_id, nbytes, 1)
    write_blob(u.slab, off, nbytes, key_of(blob_id), fill)
    assert u.commit(blob_id) == "done"
    return off


def _value(data) -> bytes:
    """The value bytes of a one-key blob (everything before its index)."""
    data = bytes(data)
    return data[: blob_format._footer(memoryview(data))[0]]


def _lock_is_free(u) -> bool:
    """True if another thread can take the unit lock right now."""
    got = []

    def probe():
        ok = u._mu.acquire(timeout=0.5)
        got.append(ok)
        if ok:
            u._mu.release()  # an RLock must be released by the thread that owns it

    t = threading.Thread(target=probe)
    t.start()
    t.join()
    return got[0]


def test_blob_released_mid_save_is_copied_then_freed(tmp_path, monkeypatch):
    u = _unit()
    _put(u, "a", 4096, 0xAA)
    _put(u, "b", 4096, 0xBB)
    free_before = u.alloc_.free_bytes
    seen = {}
    real_fsync = os.fsync

    def during_io(fd):
        # Step 4: file I/O runs without the unit lock; a clear arrives.
        seen["lock_free"] = _lock_is_free(u)
        u.release_rows([gidx_of("b")])
        seen["free_mid_save"] = u.alloc_.free_bytes
        real_fsync(fd)

    monkeypatch.setattr(nixl_storage_unit.os, "fsync", during_io)
    out = u.save_shard(str(tmp_path / "unit-0.shard"))

    assert seen["lock_free"]
    assert seen["free_mid_save"] == free_before  # held: not reusable yet
    assert u.alloc_.free_bytes == free_before + 4096  # freed after the copy
    assert out["clears_during_save"] == 1
    assert not u.checkpointing and not u._released_held
    data = (tmp_path / "unit-0.shard").read_bytes()
    off_b, n_b, _ = out["blobs"]["b"]
    assert set(_value(data[off_b : off_b + n_b])) == {0xBB}


@pytest.mark.parametrize("where", ["open", "fsync", "replace"])
def test_every_failure_drops_exactly_the_holds_taken(tmp_path, monkeypatch, where):
    u = _unit()
    _put(u, "a", 4096, 1)
    _put(u, "b", 4096, 2)
    path = tmp_path / "unit-0.shard"
    if where == "open":
        path = tmp_path / "no-such-dir" / "unit-0.shard"  # open itself fails
    else:

        def boom(*a, **k):
            raise OSError(f"injected at {where}")

        monkeypatch.setattr(nixl_storage_unit.os, where, boom)
    with pytest.raises(OSError):
        u.save_shard(str(path))
    assert all(rec.holds == 0 for rec in u.blobs.values())
    assert not u.checkpointing
    assert not path.exists()


def test_release_outside_a_save_frees_at_once():
    u = _unit()
    _put(u, "a", 4096, 1)
    free_before = u.alloc_.free_bytes
    u.release_rows([gidx_of("a")])
    assert u.alloc_.free_bytes == free_before + 4096
    assert u.clears_during_save == 0


def test_load_shard_fills_regions_outside_the_lock(tmp_path):
    src = _unit()
    _put(src, "a", 4096, 0x5A)
    out = src.save_shard(str(tmp_path / "s"))
    dst = _unit()
    res = dst.load_shard(str(tmp_path / "s"), out["blobs"])
    assert res == {"loaded": ["a"], "failed": []}
    rec = dst.blobs["a"]
    assert set(_value(dst.slab[rec.off : rec.off + rec.nbytes])) == {0x5A}
    assert dst.rows[gidx_of("a")]["x"][0] == "a"  # its live key is indexed again


def test_load_shard_short_read_frees_its_region(tmp_path):
    (tmp_path / "s").write_bytes(b"\x00" * 100)
    (tmp_path / "s.rows").write_bytes(msgspec.msgpack.encode({"a": [key_of("a")]}))
    u = _unit()
    free_before = u.alloc_.free_bytes
    with pytest.raises(IOError, match="short read"):
        u.load_shard(str(tmp_path / "s"), {"a": [0, 4096, 1]})
    assert u.alloc_.free_bytes == free_before and "a" not in u.blobs
