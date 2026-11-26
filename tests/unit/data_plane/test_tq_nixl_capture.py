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
from nemo_rl.data_plane.checkpoint import CheckpointReleaseError
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


def _fake_get(ref, timeout=None):
    """ray.get on a fake future; a wait that expires raises Ray's own error."""
    try:
        return ref.result(timeout)
    except concurrent.futures.TimeoutError as error:
        raise ckpt.ray.exceptions.GetTimeoutError("fake wait expired") from error


def _actor(unit, **overrides):
    """A fake unit actor; `overrides` replace single RPCs (for failure cases)."""
    rpcs = {
        "capture_shard": _Remote(unit.capture_shard),
        "persist_shard": _Remote(unit.persist_shard),
        "release_shard": _Remote(unit.release_shard),
    }
    return SimpleNamespace(**{**rpcs, **overrides})


def _put(u, blob_id, fill):
    off = u.alloc(blob_id, 4096, 1)
    write_blob(u.slab, off, 4096, key_of(blob_id), fill)
    assert u.commit(blob_id) == "done"


@pytest.fixture
def cluster(monkeypatch):
    units = [bare_unit(1 << 16), bare_unit(1 << 16)]
    units[1].unit_id = 1
    actors = {0: _actor(units[0]), 1: _actor(units[1])}
    monkeypatch.setattr(ckpt.ray, "get", _fake_get)
    monkeypatch.setattr(ckpt, "_unit_actor", lambda store, u: actors[u])
    monkeypatch.setattr(ckpt, "_active_units", lambda store, *, timeout_s=None: [0, 1])
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
    cap = ckpt.capture_nixl_checkpoint(
        client, path, metadata={"step": 3}, timeout_s=30.0
    )
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
    cap = ckpt.capture_nixl_checkpoint(
        client, tmp_path / "dp", metadata=None, timeout_s=30.0
    )
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
        ckpt.capture_nixl_checkpoint(
            client, tmp_path / "dp", metadata=None, timeout_s=30.0
        )
    assert not units[0]._captures  # unit 0 captured, then was released
    assert set(units[1]._captures) == {"prior"}  # the prior hold is untouched
    assert not client.controller.views  # the controller bytes were released too


def test_persist_is_single_attempt_and_release_is_idempotent(cluster, tmp_path):
    units, client = cluster
    _put(units[0], "a", 1)
    cap = ckpt.capture_nixl_checkpoint(
        client, tmp_path / "dp", metadata=None, timeout_s=30.0
    )
    cap.persist()
    with pytest.raises(RuntimeError, match="single-attempt"):
        cap.persist()
    cap.release()
    cap.release()  # a second release is a no-op
    with pytest.raises(RuntimeError, match="released"):
        cap.persist()


class _NeverDone:
    """An RPC whose reply never arrives, as with a frozen actor."""

    def remote(self, *args):
        return concurrent.futures.Future()


def test_unanswered_capture_rpc_times_out_and_releases_the_rest(
    cluster, tmp_path, monkeypatch
):
    # The capture reply never arrives. The release still runs at once here, so
    # this does not model a serial actor that queues the release behind it.
    units, client = cluster
    actors = {
        0: _actor(units[0]),
        1: _actor(units[1], capture_shard=_NeverDone()),
    }
    monkeypatch.setattr(ckpt, "_unit_actor", lambda store, u: actors[u])
    with pytest.raises(TimeoutError, match="timed out"):
        ckpt.capture_nixl_checkpoint(
            client, tmp_path / "dp", metadata=None, timeout_s=0.2
        )
    assert not units[0]._captures  # unit 0 captured, then was released
    assert not client.controller.views  # the controller bytes were released too


def test_capture_that_cannot_release_raises_checkpoint_release_error(
    cluster, tmp_path, monkeypatch
):
    units, client = cluster
    units[1].capture_shard("prior")  # unit 1 refuses a second capture
    boom = _Remote(lambda token: (_ for _ in ()).throw(RuntimeError("release boom")))
    actors = {0: _actor(units[0], release_shard=boom), 1: _actor(units[1])}
    monkeypatch.setattr(ckpt, "_unit_actor", lambda store, u: actors[u])
    with pytest.raises(CheckpointReleaseError, match="stop checkpointing") as info:
        ckpt.capture_nixl_checkpoint(
            client, tmp_path / "dp", metadata=None, timeout_s=30.0
        )
    assert isinstance(info.value.__cause__, RuntimeError)
    assert "release boom" in str(info.value.__cause__)


def _capture(client, tmp_path, timeout_s=0.1):
    return ckpt.capture_nixl_checkpoint(
        client, tmp_path / "dp", metadata=None, timeout_s=timeout_s
    )


def test_persist_submission_failure_keeps_the_original_error(cluster, tmp_path):
    units, client = cluster
    _put(units[0], "a", 1)
    cap = _capture(client, tmp_path)

    def refuse(*args):
        raise RuntimeError("submit failed")

    client.controller.nrl_persist_snapshot = SimpleNamespace(remote=refuse)
    # Nothing was submitted, so the wait has no result to unpack. The submission
    # error must not be replaced by an unpacking error.
    with pytest.raises(RuntimeError, match="submit failed"):
        cap.persist()
    assert not (tmp_path / "dp" / "metadata.json").exists()
    cap.release()
    assert all(not u._captures for u in units) and not client.controller.views


def test_frozen_directory_times_out_before_any_hold(cluster, tmp_path, monkeypatch):
    units, client = cluster
    seen = []

    def frozen(store, *, timeout_s=None):
        seen.append(timeout_s)
        raise ckpt.ray.exceptions.GetTimeoutError("directory frozen")

    monkeypatch.setattr(ckpt, "_active_units", frozen)
    with pytest.raises(TimeoutError, match="unit lookup timed out"):
        _capture(client, tmp_path, timeout_s=0.2)
    assert seen == [0.2]  # the lookup got the capture budget
    assert all(not u._captures for u in units) and not client.controller.views


def test_active_units_passes_its_budget_to_the_directory_query():
    seen = []
    store = SimpleNamespace(
        _units={0: {"unit_id": 0}, 1: {"unit_id": 1, "state": "draining"}},
        _refresh_units=lambda *, timeout_s=None: seen.append(timeout_s),
    )
    assert ckpt._active_units(store, timeout_s=1.5) == [0]
    assert seen == [1.5]


def test_persist_and_release_timeouts_keep_the_holds_until_the_write_settles(
    cluster, tmp_path
):
    units, client = cluster
    _put(units[0], "a", 1)
    cap = _capture(client, tmp_path)
    pending = concurrent.futures.Future()  # the write of unit 0 is still running
    cap.units[0].persist_shard = SimpleNamespace(remote=lambda *args: pending)
    with pytest.raises(TimeoutError, match="timed out"):
        cap.persist()
    assert not (tmp_path / "dp" / "metadata.json").exists()
    with pytest.raises(TimeoutError, match="timed out"):
        cap.release()  # must not drop holds under a running write
    assert units[0]._captures and units[1]._captures
    assert client.controller.views
    with pytest.raises(RuntimeError, match="single-attempt"):
        cap.persist()
    pending.set_result({"blobs": {}, "nbytes": 0})  # the write settles
    cap.release()
    assert all(not u._captures for u in units) and not client.controller.views
    assert not (tmp_path / "dp" / "metadata.json").exists()


def test_release_ack_timeout_can_be_retried(cluster, tmp_path):
    units, client = cluster
    _put(units[0], "a", 1)
    cap = _capture(client, tmp_path)
    original = cap.units[0].release_shard
    cap.units[0].release_shard = SimpleNamespace(
        remote=lambda token: concurrent.futures.Future()  # never answers
    )
    with pytest.raises(TimeoutError, match="timed out"):
        cap.release()
    assert not units[1]._captures and not client.controller.views  # released
    assert units[0]._captures  # the unanswered release did not run
    cap.units[0].release_shard = original
    cap.release()  # the retry finishes the job
    assert all(not u._captures for u in units)
