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
"""Index-addressed rows on a unit: commit publishes a blob's keys, an overwrite
drops the old blob's reference, clears are row-wide. CPU-only, no Ray actor.
"""

import asyncio
from types import SimpleNamespace

import numpy as np

from nemo_rl.data_plane.adapters.tq_nixl import NixlStorageManager
from nemo_rl.data_plane.nixl import blob_format
from tests.unit.data_plane._nixl_helpers import bare_unit


def _put(u, blob_id, kv):
    """What a client does: lay out the blob, WRITE it into its region, commit."""
    p = blob_format.plan(list(kv), [np.full(16, v, np.uint8) for v in kv.values()])
    off = u.alloc(blob_id, p.nbytes, len(kv))
    blob_format.write(p, memoryview(u.slab)[off : off + p.nbytes])
    assert u.commit(blob_id) == "done"


def test_commit_publishes_every_key_and_is_idempotent():
    u = bare_unit(1 << 16)
    _put(u, "A", {"5@x": 1, "5@y": 2, "6@x": 3})
    assert set(u.rows) == {5, 6} and set(u.rows[5]) == {"x", "y"}
    assert u.blobs["A"].refs == 3
    assert u.commit("A") == "done" and u.blobs["A"].refs == 3  # asked again


def test_overwrite_releases_old_blob():
    u = bare_unit(1 << 16)
    _put(u, "A", {"5@x": 1})
    _put(u, "B", {"5@x": 2})
    assert "A" not in u.blobs and u.rows[5]["x"][0] == "B"


def test_only_overwritten_fields_drop_their_reference():
    u = bare_unit(1 << 16)
    _put(u, "A", {"5@x": 1, "5@y": 2})
    _put(u, "B", {"5@x": 3})
    assert u.blobs["A"].refs == 1 and u.rows[5]["y"][0] == "A"


def test_release_rows_is_row_wide_and_idempotent():
    u = bare_unit(1 << 16)
    _put(u, "A", {"5@x": 1, "6@x": 2})
    _put(u, "B", {"5@y": 3})  # a field only row 5 has
    assert u.release_rows([5]) == 2
    assert 5 not in u.rows and "B" not in u.blobs and u.blobs["A"].refs == 1
    assert u.release_rows([5, 99]) == 0


def test_resolve_keys_returns_entries_and_lists_missing():
    u = bare_unit(1 << 16)
    _put(u, "A", {"5@x": 7})
    r = u.resolve_keys(["5@x", "5@z", "8@x"])
    assert sorted(r["missing"]) == ["5@z", "8@x"]
    off, n, kind, _dtype, shape = r["entries"]["5@x"]
    assert n == 16 and kind == blob_format.KIND_NUMPY and shape == [16]
    assert set(u.slab[off : off + n]) == {7}
    u.unpin(r["pin_id"])


def test_expired_pins_are_swept():
    u = bare_unit(1 << 16, read_pin_s=0.0)
    _put(u, "A", {"5@x": 7})
    u.resolve_keys(["5@x"])  # a reader that never unpins
    u._sweep(float("inf"))
    assert not u._pins and not u._pin_exp and not u.blobs["A"].pins


def test_uncommitted_allocation_is_retired_after_its_lease():
    u = bare_unit(1 << 16)
    u.alloc_lease_s = 1.0
    u.alloc("dead", 4096, 1)  # its writer crashed before committing
    u._sweep(u.blobs["dead"].created_at + 2.0)
    assert "dead" not in u.blobs and u.retired_bytes == 4096
    assert u.alloc_.used_bytes == 4096  # never reused: a late WRITE may land


def test_abort_of_a_published_blob_is_ignored():
    u = bare_unit(1 << 16)
    _put(u, "A", {"5@x": 1})
    u.abort("A", in_flight=False)
    assert "A" in u.blobs and u.rows[5]["x"][0] == "A"


def test_manager_clears_whole_rows_by_global_index(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.rows = []

        def clear_rows(self, gidxs):
            self.rows.append(gidxs)

    mgr = object.__new__(NixlStorageManager)
    mgr.storage_client = FakeClient()
    monkeypatch.setattr(NixlStorageManager, "__del__", lambda self: None)
    md = SimpleNamespace(global_indexes=[3, 4], field_names=["a"])
    asyncio.run(mgr.clear_data(md))
    assert mgr.storage_client.rows == [[3, 4]]
