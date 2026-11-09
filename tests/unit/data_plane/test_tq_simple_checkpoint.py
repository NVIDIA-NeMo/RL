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
"""Stable views, native-format restore, RPC failures and capture ownership."""

import concurrent.futures
import json
import pickle
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transfer_queue")
from transfer_queue.storage.simple_storage import StorageUnitData  # noqa: E402

from nemo_rl.data_plane.adapters import tq_simple_checkpoint as snapshot  # noqa: E402
from nemo_rl.data_plane.interfaces import (  # noqa: E402
    SimpleStorageConfig,
    SsdOffloadConfig,
)


class _StorageBase:
    def __init__(self):
        self.storage_unit_id = "unit-0"
        self.storage_data = StorageUnitData(1024)

    def _handle_put(self, parts):
        self.storage_data.put_data(parts.body["data"], parts.body["global_indexes"])

    def _handle_clear(self, parts):
        self.storage_data.clear(parts.body["global_indexes"])

    def _handle_load_checkpoint(self, parts):
        self.storage_data.load_checkpoint(parts.body["path"])


class _Storage(snapshot.SnapshotStorageMixin, _StorageBase):
    pass


def _put(unit, indexes, **fields):
    unit._handle_put(SimpleNamespace(body={"global_indexes": indexes, "data": fields}))


def test_view_survives_clear_replacement_and_added_fields(tmp_path):
    unit = _Storage()
    routes = torch.arange(12).reshape(2, 2, 3)
    _put(unit, [5, 7], routes=routes, label=[{"task": "math"}, {"task": "code"}])
    assert unit.nrl_capture_snapshot("one") == "unit-0"
    # Capture does not clone multi-GB route tensors.
    retained = unit._snapshot_views["one"]["field_data"]["routes"][5]
    assert retained.untyped_storage().data_ptr() == routes.untyped_storage().data_ptr()
    unit._handle_clear(SimpleNamespace(body={"global_indexes": [5]}))
    _put(unit, [7], routes=[torch.full((2, 3), -1)])
    _put(unit, [9], new_field=["new"])
    path = tmp_path / "unit.pkl"
    assert unit.nrl_persist_snapshot("one", str(path)) > 0
    restored = StorageUnitData()
    assert restored.load_checkpoint(path) == 1024
    assert restored._active_keys == {5, 7}
    assert set(restored.field_data) == {"routes", "label"}
    torch.testing.assert_close(restored.field_data["routes"][5], routes[0])
    torch.testing.assert_close(restored.field_data["routes"][7], routes[1])
    assert restored.field_data["label"][5] == {"task": "math"}
    assert unit.storage_data._active_keys == {7, 9}
    unit.nrl_release_snapshot("one")
    unit.nrl_release_snapshot("one")
    assert not unit._snapshot_views


def test_only_one_capture_is_allowed_and_restore_is_rejected(tmp_path):
    unit = _Storage()
    unit.nrl_capture_snapshot("one")
    with pytest.raises(RuntimeError, match="Only one"):
        unit.nrl_capture_snapshot("two")
    with pytest.raises(RuntimeError, match="Cannot restore"):
        unit._handle_load_checkpoint(SimpleNamespace(body={"path": tmp_path / "x"}))
    unit.nrl_release_snapshot("one")
    unit.nrl_capture_snapshot("two")


def test_hybrid_or_unknown_storage_is_rejected():
    unit = _Storage()
    unit.storage_data = SimpleNamespace(ssd_offload_enabled=True)
    with pytest.raises(TypeError, match="in-memory"):
        unit.nrl_capture_snapshot("one")
    assert not unit._snapshot_views


def test_capture_waits_for_complete_put_transaction():
    unit = _Storage()
    entered, finish = threading.Event(), threading.Event()
    original_put = unit.storage_data.put_data

    def slow_put(fields, indexes):
        original_put({"first": fields["first"]}, indexes)
        entered.set()
        assert finish.wait(5)
        original_put({"second": fields["second"]}, indexes)

    unit.storage_data.put_data = slow_put
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        put = pool.submit(_put, unit, [0], first=[1], second=[2])
        assert entered.wait(5)
        capture = pool.submit(unit.nrl_capture_snapshot, "one")
        try:
            with pytest.raises(concurrent.futures.TimeoutError):
                capture.result(timeout=0.05)
        finally:
            finish.set()
        put.result(timeout=5)
        capture.result(timeout=5)
    assert unit._snapshot_views["one"]["field_data"] == {
        "first": {0: 1},
        "second": {0: 2},
    }


def test_persistence_does_not_hold_mutation_lock(monkeypatch, tmp_path):
    unit = _Storage()
    _put(unit, [0], field=[torch.tensor([1])])
    unit.nrl_capture_snapshot("one")
    entered, finish = threading.Event(), threading.Event()
    original_dump = pickle.dump

    def slow_dump(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        original_dump(*args, **kwargs)

    monkeypatch.setattr(snapshot.pickle, "dump", slow_dump)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        write = pool.submit(unit.nrl_persist_snapshot, "one", str(tmp_path / "state"))
        assert entered.wait(5)
        try:
            pool.submit(_put, unit, [0], field=[torch.tensor([2])]).result(timeout=2)
        finally:
            finish.set()
        write.result(timeout=5)
    restored = StorageUnitData()
    restored.load_checkpoint(tmp_path / "state")
    assert restored.field_data["field"][0].item() == 1
    assert unit.storage_data.field_data["field"][0].item() == 2


def test_async_config_rejects_ssd():
    with pytest.raises(ValueError, match="does not support SSD"):
        SimpleStorageConfig(
            num_storage_units=2,
            async_checkpoint=True,
            ssd_offload=SsdOffloadConfig(enabled=True, path="/tmp/tq"),
        )


def test_bootstrap_patch_is_idempotent(monkeypatch):
    original = snapshot.bootstrap.SimpleStorageUnit
    monkeypatch.setattr(snapshot.bootstrap, "SimpleStorageUnit", original)
    snapshot.configure_simple_checkpoint()
    patched = snapshot.bootstrap.SimpleStorageUnit
    assert issubclass(
        patched.__ray_metadata__.modified_class, snapshot.SnapshotStorageMixin
    )
    assert patched._default_options == original._default_options
    snapshot.configure_simple_checkpoint()
    assert snapshot.bootstrap.SimpleStorageUnit is patched


class _RemoteMethod:
    def __init__(self, method):
        self.method = method

    def remote(self, *args):
        future = concurrent.futures.Future()
        try:
            future.set_result(self.method(*args))
        except Exception as error:
            future.set_exception(error)
        return future


def _remote_unit(unit):
    return SimpleNamespace(
        nrl_capture_snapshot=_RemoteMethod(unit.nrl_capture_snapshot),
        nrl_persist_snapshot=_RemoteMethod(unit.nrl_persist_snapshot),
        nrl_release_snapshot=_RemoteMethod(unit.nrl_release_snapshot),
    )


@pytest.fixture
def fake_cluster(monkeypatch):
    units = [_Storage(), _Storage()]
    units[1].storage_unit_id = "unit-1"
    actors = [_remote_unit(unit) for unit in units]
    monkeypatch.setattr(snapshot.ray, "get", lambda ref: ref.result())
    monkeypatch.setattr(
        snapshot.ray, "get_actor", lambda name: actors[int(name.split("#")[1])]
    )
    # Deliberately reverse manager order to catch positional restore mistakes.
    client = SimpleNamespace(
        storage_manager=SimpleNamespace(
            storage_unit_infos={"unit-1": {}, "unit-0": {}}
        ),
        save_controller_checkpoint=lambda path: Path(path).write_bytes(b"controller"),
    )
    monkeypatch.setattr(snapshot.interface, "_maybe_create_tq_client", lambda: client)
    return units, client


def test_distributed_capture_native_layout_and_metadata_detachment(
    fake_cluster, tmp_path
):
    units, _ = fake_cluster
    for index, unit in enumerate(units):
        _put(unit, [index], field=[torch.tensor([index])])
    metadata = {"step": 5, "nested": {"keys": [1]}}
    path = tmp_path / "data_plane"
    captured = snapshot.capture_simple_checkpoint(
        path, num_storage_units=2, metadata=metadata
    )
    assert not (path / "metadata.json").exists()
    metadata["nested"]["keys"].append(2)
    for unit in units:
        unit.storage_data.clear([0, 1])
    captured.persist()
    info = json.loads((path / "simple_storage/storage_unit_info.json").read_text())
    assert info == [
        {"position": 0, "storage_unit_id": "unit-1"},
        {"position": 1, "storage_unit_id": "unit-0"},
    ]
    assert json.loads((path / "metadata.json").read_text()) == {
        "storage_saved": True,
        "user_metadata": {"step": 5, "nested": {"keys": [1]}},
    }
    restored = StorageUnitData()
    restored.load_checkpoint(path / "simple_storage/su_0_unit-1.pkl")
    assert restored.field_data["field"][1].item() == 1
    captured.release()
    assert all(not unit._snapshot_views for unit in units)


def test_partial_capture_failure_releases_every_unit(fake_cluster, tmp_path):
    units, _ = fake_cluster
    # One already-retained snapshot must not be disturbed by failed new capture.
    units[0].nrl_capture_snapshot("prior")
    with pytest.raises(RuntimeError, match="Only one"):
        snapshot.capture_simple_checkpoint(
            tmp_path / "state", num_storage_units=2, metadata=None
        )
    assert set(units[0]._snapshot_views) == {"prior"}
    assert not units[1]._snapshot_views


def test_persist_failure_has_no_completion_marker(fake_cluster, monkeypatch, tmp_path):
    units, _ = fake_cluster
    captured = snapshot.capture_simple_checkpoint(
        tmp_path / "state", num_storage_units=2, metadata=None
    )

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(units[0], "nrl_persist_snapshot", fail)
    captured.units = [_remote_unit(units[1]), _remote_unit(units[0])]
    with pytest.raises(OSError, match="disk full"):
        captured.persist()
    assert not (tmp_path / "state/metadata.json").exists()
    captured.release()
    assert all(not unit._snapshot_views for unit in units)


def test_existing_checkpoint_is_not_overwritten(fake_cluster, tmp_path):
    path = tmp_path / "state"
    path.mkdir()
    (path / "marker").write_text("durable")
    with pytest.raises(FileExistsError):
        snapshot.capture_simple_checkpoint(path, num_storage_units=2, metadata=None)
    assert (path / "marker").read_text() == "durable"


def test_wait_all_settles_later_writes_after_early_failure(monkeypatch):
    failed, completed = concurrent.futures.Future(), concurrent.futures.Future()
    failed.set_exception(OSError("first failed"))
    completed.set_result(42)
    observed = []

    def get(ref):
        observed.append(ref)
        return ref.result()

    monkeypatch.setattr(snapshot.ray, "get", get)
    with pytest.raises(OSError, match="first failed"):
        snapshot._wait_all([failed, completed])
    assert observed == [failed, completed]


def test_release_attempts_all_units_after_submission_failure(fake_cluster, tmp_path):
    units, _ = fake_cluster
    captured = snapshot.capture_simple_checkpoint(
        tmp_path / "state", num_storage_units=2, metadata=None
    )

    def fail_submit(*args):
        raise RuntimeError("actor submission failed")

    captured.units[0].nrl_release_snapshot = SimpleNamespace(remote=fail_submit)
    with pytest.raises(RuntimeError, match="actor submission failed"):
        captured.release()
    # Manager order is reversed: unit 0 must still receive its release RPC.
    assert not units[0]._snapshot_views
