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
"""Two-phase nixl capture: file layout, completion marker order, cleanup. No Ray.

The controller snapshot RPCs are faked; their real behavior is in the Simple tests.
"""

import concurrent.futures
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_rl.data_plane.adapters import tq_nixl_checkpoint as ckpt
from tests.unit.data_plane._nixl_helpers import bare_unit, gidx_of, key_of, write_blob


class _Remote:
    def __init__(self, method):
        self.method = method

    def remote(self, *args):
        future = concurrent.futures.Future()
        try:
            future.set_result(self.method(*args))
        except Exception as error:  # noqa: BLE001 - delivered through the future
            future.set_exception(error)
        return future


class _Controller:
    """The controller's snapshot RPCs: capture keeps bytes, persist writes them."""

    def __init__(self):
        self.views = {}
        self.nrl_capture_snapshot = _Remote(lambda t: self.views.setdefault(t, b"c"))
        self.nrl_persist_snapshot = _Remote(
            lambda t, path: Path(path).write_bytes(self.views[t])
        )
        self.nrl_release_snapshot = _Remote(lambda t: self.views.pop(t, None))


def _actor(unit):
    return SimpleNamespace(
        capture_shard=_Remote(unit.capture_shard),
        persist_shard=_Remote(unit.persist_shard),
        release_shard=_Remote(unit.release_shard),
    )


def _put(u, blob_id, fill):
    off = u.alloc(blob_id, 4096, 1)
    write_blob(u.slab, off, 4096, key_of(blob_id), fill)
    assert u.commit(blob_id) == "done"


@pytest.fixture
def cluster(monkeypatch):
    units = [bare_unit(1 << 16), bare_unit(1 << 16)]
    units[1].unit_id = 1
    actors = {0: _actor(units[0]), 1: _actor(units[1])}
    monkeypatch.setattr(ckpt.ray, "get", lambda ref: ref.result())
    monkeypatch.setattr(ckpt, "_unit_actor", lambda store, u: actors[u])
    monkeypatch.setattr(ckpt, "_active_units", lambda store: [0, 1])
    storage = SimpleNamespace(store=object(), unit_ids=[0, 1])
    controller = _Controller()
    monkeypatch.setattr(ckpt, "_controller_actor", lambda: controller)
    client = SimpleNamespace(
        storage_manager=SimpleNamespace(storage_client=storage), controller=controller
    )
    return units, client


def test_capture_persist_release_layout(cluster, tmp_path):
    units, client = cluster
    _put(units[0], "a", 1)
    _put(units[1], "b", 2)
    path = tmp_path / "dp"
    cap = ckpt.capture_nixl_checkpoint(client, path, metadata={"step": 3})
    assert not (path / "metadata.json").exists()
    _put(units[0], "n", 3)  # a new key after the cut
    cap.persist()
    manifest = json.loads((path / "nixl_store" / ckpt.MANIFEST).read_text())
    assert manifest["version"] == ckpt.VERSION and manifest["unit_ids"] == [0, 1]
    assert [s["unit_id"] for s in manifest["shards"]] == [0, 1]
    assert set(manifest["shards"][0]["blobs"]) == {"a"}  # still in the cut
    assert json.loads((path / "metadata.json").read_text()) == {
        "storage_saved": True,
        "user_metadata": {"step": 3},
    }
    assert (path / "controller_state.pkl").read_bytes() == b"c"
    cap.release()
    assert all(not u._captures for u in units) and not client.controller.views


def test_clear_after_capture_fails_persist_with_no_marker(cluster, tmp_path):
    units, client = cluster
    _put(units[0], "a", 1)
    cap = ckpt.capture_nixl_checkpoint(client, tmp_path / "dp", metadata=None)
    units[0].release_rows([gidx_of("a")])  # a clear of a key of the cut
    with pytest.raises(RuntimeError, match="updated or cleared"):
        cap.persist()
    assert not (tmp_path / "dp" / "metadata.json").exists()
    cap.release()
    assert all(not u._captures for u in units)


def test_failed_capture_releases_every_unit(cluster, tmp_path):
    units, client = cluster
    units[1].capture_shard("prior")  # unit 1 refuses a second capture
    with pytest.raises(RuntimeError, match="Only one"):
        ckpt.capture_nixl_checkpoint(client, tmp_path / "dp", metadata=None)
    assert not units[0]._captures  # unit 0 captured, then was released
    assert set(units[1]._captures) == {"prior"}  # the prior hold is untouched
    assert not client.controller.views  # the controller bytes were released too
