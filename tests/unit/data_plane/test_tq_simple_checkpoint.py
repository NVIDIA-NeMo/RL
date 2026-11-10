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
import copy
import json
import pickle
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transfer_queue")
from transfer_queue.controller import (  # noqa: E402
    DataPartitionStatus,
    FieldMeta,
    PartitionIndexManager,
    TransferQueueController,
)
from transfer_queue.storage.simple_storage import StorageUnitData  # noqa: E402

from nemo_rl.data_plane.adapters import (
    tq_controller_checkpoint as controller_snapshot,  # noqa: E402
)
from nemo_rl.data_plane.adapters import (
    tq_controller_patch as controller_patch,  # noqa: E402
)
from nemo_rl.data_plane.adapters import tq_simple_checkpoint as snapshot  # noqa: E402
from nemo_rl.data_plane.interfaces import SimpleStorageConfig  # noqa: E402


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


class _ControllerBase:
    load_checkpoint = (
        TransferQueueController.__ray_metadata__.modified_class.load_checkpoint
    )

    def __init__(self):
        self.controller_id = "controller-0"
        self.partitions = {"data": DataPartitionStatus(partition_id="data")}
        self.index_manager = PartitionIndexManager()
        self.sampler_state = {"cursor": [1]}
        self.sampler = SimpleNamespace(
            save_checkpoint=lambda: self.sampler_state,
            load_checkpoint=lambda state: setattr(self, "sampler_state", state),
        )

    def _handle_request(self, request, monitor):
        return request()


class _Controller(controller_snapshot.SnapshotControllerMixin, _ControllerBase):
    pass


class _CustomPartition(DataPartitionStatus):
    def to_snapshot(self):
        saved = DataPartitionStatus(partition_id=self.partition_id)
        saved.custom_meta[0] = {"custom_snapshot": True}
        return saved


def test_custom_partition_keeps_native_snapshot_semantics(tmp_path):
    controller = _Controller()
    controller.partitions["custom"] = _CustomPartition(partition_id="custom")
    controller.nrl_capture_snapshot("one")
    path = tmp_path / "controller.pkl"
    controller.nrl_persist_snapshot("one", str(path))
    restored = _Controller()
    restored.load_checkpoint(str(path))
    assert restored.partitions["custom"].custom_meta[0] == {"custom_snapshot": True}
    assert not controller.partitions["custom"].custom_meta


def test_controller_capture_waits_for_complete_request():
    controller = _Controller()
    entered, finish = threading.Event(), threading.Event()

    def mutate():
        controller.partitions["data"].keys_mapping["first"] = 1
        entered.set()
        assert finish.wait(5)
        controller.partitions["data"].keys_mapping["second"] = 2

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        request = pool.submit(controller._handle_request, mutate, None)
        assert entered.wait(5)
        capture = pool.submit(controller.nrl_capture_snapshot, "one")
        try:
            with pytest.raises(concurrent.futures.TimeoutError):
                capture.result(timeout=0.05)
        finally:
            finish.set()
        request.result(timeout=5)
        capture.result(timeout=5)
    assert pickle.loads(controller._controller_snapshot_views["one"])["partitions"][
        "data"
    ].keys_mapping == {"first": 1, "second": 2}


def test_byte_capture_detaches_all_metadata_and_uses_native_loader(
    tmp_path, monkeypatch
):
    controller = _Controller()
    partition = controller.partitions["data"]
    shared = [2, 3]
    partition.keys_mapping["old"] = 0
    partition.field_metadata["routes"] = FieldMeta(
        global_indexes={0}, per_sample_shapes={0: shared}
    )
    partition.custom_meta[0] = {"nested": [torch.tensor([7])], "shape": shared}
    partition.field_custom_backend_meta[0] = {"key": [1]}
    controller.index_manager.partition_to_indexes["data"].add(0)
    controller.index_manager.allocated_indexes.add(0)
    controller.index_manager.reusable_indexes.append(2)

    def no_copy(*args, **kwargs):
        raise AssertionError("Byte capture must not use deepcopy/to_snapshot")

    with monkeypatch.context() as capture_only:
        capture_only.setattr(copy, "deepcopy", no_copy)
        capture_only.setattr(DataPartitionStatus, "to_snapshot", no_copy)
        assert controller.nrl_capture_snapshot("bytes") == "controller-0"
    payload = controller._controller_snapshot_views["bytes"]
    assert isinstance(payload, bytes)
    assert not list(tmp_path.iterdir())
    partition.keys_mapping.clear()
    partition.production_status.fill_(1)
    shared.append(9)
    partition.field_metadata["routes"].global_indexes.clear()
    partition.custom_meta[0]["nested"][0].fill_(99)
    partition.field_custom_backend_meta[0]["key"].append(2)
    controller.index_manager.partition_to_indexes["data"].clear()
    controller.index_manager.allocated_indexes.clear()
    controller.index_manager.reusable_indexes.clear()
    controller.sampler_state["cursor"].append(2)
    path = tmp_path / "controller.pkl"
    assert controller.nrl_persist_snapshot("bytes", str(path)) == len(payload)
    assert path.read_bytes() == payload
    restored = _Controller()
    restored.load_checkpoint(str(path))
    saved = restored.partitions["data"]
    assert saved.keys_mapping == {"old": 0}
    assert not saved.production_status.any()
    assert saved.field_metadata["routes"].global_indexes == {0}
    assert saved.field_metadata["routes"].per_sample_shapes[0] == [2, 3]
    assert saved.custom_meta[0]["nested"][0].item() == 7
    assert saved.field_custom_backend_meta[0] == {"key": [1]}
    assert restored.index_manager.partition_to_indexes["data"] == {0}
    assert restored.index_manager.allocated_indexes == {0}
    assert list(restored.index_manager.reusable_indexes) == [2]
    assert restored.sampler_state == {"cursor": [1]}
    with pytest.raises(RuntimeError, match="Only one"):
        controller.nrl_capture_snapshot("copy")
    with pytest.raises(RuntimeError, match="Cannot restore"):
        controller.load_checkpoint(str(path))
    controller.nrl_release_snapshot("bytes")
    controller.nrl_release_snapshot("bytes")


def test_failed_byte_serialization_does_not_retain_partial_capture(monkeypatch):
    controller = _Controller()

    def fail(*args, **kwargs):
        raise ValueError("Cannot serialize metadata")

    with monkeypatch.context() as patch:
        patch.setattr(controller_snapshot.pickle, "dumps", fail)
        with pytest.raises(ValueError, match="Cannot serialize"):
            controller.nrl_capture_snapshot("one")
    assert not controller._controller_snapshot_views
    controller.nrl_capture_snapshot("retry")


def test_byte_serialization_excludes_live_mutation(monkeypatch):
    controller = _Controller()
    entered, finish = threading.Event(), threading.Event()
    original = pickle.dumps

    def paused(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(controller_snapshot.pickle, "dumps", paused)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        capture = pool.submit(controller.nrl_capture_snapshot, "one")
        assert entered.wait(5)
        request = pool.submit(
            controller._handle_request,
            lambda: controller.partitions["data"].keys_mapping.update(new=1),
            None,
        )
        try:
            with pytest.raises(concurrent.futures.TimeoutError):
                request.result(timeout=0.05)
        finally:
            finish.set()
        capture.result(timeout=5)
        request.result(timeout=5)
    saved = pickle.loads(controller._controller_snapshot_views["one"])
    assert saved["partitions"]["data"].keys_mapping == {}
    assert controller.partitions["data"].keys_mapping == {"new": 1}


def test_byte_persistence_never_reserializes(monkeypatch, tmp_path):
    controller = _Controller()
    controller.nrl_capture_snapshot("one")

    def fail(*args, **kwargs):
        raise AssertionError("Persistence must write existing bytes only")

    monkeypatch.setattr(controller_snapshot.pickle, "dump", fail)
    monkeypatch.setattr(controller_snapshot.pickle, "dumps", fail)
    controller.nrl_persist_snapshot("one", str(tmp_path / "state"))


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


def test_async_config_defaults_off_and_rejects_ssd():
    assert not SimpleStorageConfig(num_storage_units=2).async_checkpoint
    with pytest.raises(ValueError, match="does not support SSD"):
        SimpleStorageConfig(
            num_storage_units=2,
            async_checkpoint=True,
            ssd_offload={"enabled": True, "path": "/tmp/tq"},
        )


def test_bootstrap_patch_is_idempotent(monkeypatch):
    original = snapshot.bootstrap.SimpleStorageUnit
    monkeypatch.setattr(snapshot.bootstrap, "SimpleStorageUnit", original)
    monkeypatch.setattr(
        snapshot.interface, "TransferQueueController", TransferQueueController
    )
    snapshot.configure_simple_checkpoint()
    patched = snapshot.bootstrap.SimpleStorageUnit
    assert issubclass(
        patched.__ray_metadata__.modified_class, snapshot.SnapshotStorageMixin
    )
    assert patched._default_options == original._default_options
    snapshot.configure_simple_checkpoint()
    assert snapshot.bootstrap.SimpleStorageUnit is patched
    controller = snapshot.interface.TransferQueueController
    snapshot.configure_simple_checkpoint()
    assert snapshot.interface.TransferQueueController is controller


@pytest.mark.parametrize("metadata_patch_first", [False, True])
def test_controller_bootstrap_preserves_both_patches(monkeypatch, metadata_patch_first):
    monkeypatch.setattr(
        snapshot.interface, "TransferQueueController", TransferQueueController
    )
    if metadata_patch_first:
        controller_patch.configure_controller_patch()
    controller_snapshot.configure_controller_checkpoint()
    controller_patch.configure_controller_patch()
    patched = snapshot.interface.TransferQueueController
    assert issubclass(
        patched.__ray_metadata__.modified_class,
        controller_snapshot.SnapshotControllerMixin,
    )
    assert patched.__ray_metadata__.modified_class._nrl_kv_meta_patched
    assert patched._default_options == TransferQueueController._default_options


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


def _fake_get(ref, timeout=None):
    try:
        return ref.result(timeout)
    except TimeoutError:
        if ref.done():
            raise
        raise snapshot.ray.exceptions.GetTimeoutError("RPC is not ready") from None


@pytest.fixture
def fake_cluster(monkeypatch):
    units = [_Storage(), _Storage()]
    units[1].storage_unit_id = "unit-1"
    actors = [_remote_unit(unit) for unit in units]
    controller = _Controller()
    monkeypatch.setattr(snapshot.ray, "get", _fake_get)
    monkeypatch.setattr(
        snapshot.ray,
        "get_actor",
        lambda name, **kwargs: (
            _remote_unit(controller)
            if name == "TransferQueueController"
            else actors[int(name.split("#")[1])]
        ),
    )
    # Deliberately reverse manager order to catch positional restore mistakes.
    client = SimpleNamespace(
        storage_manager=SimpleNamespace(
            storage_unit_infos={"unit-1": {}, "unit-0": {}}
        ),
        save_controller_checkpoint=lambda path: Path(path).write_bytes(b"controller"),
        controller=controller,
    )
    monkeypatch.setattr(snapshot.interface, "_maybe_create_tq_client", lambda: client)
    return units, client


def test_distributed_capture_native_layout_and_metadata_detachment(
    fake_cluster, tmp_path
):
    units, client = fake_cluster
    for index, unit in enumerate(units):
        _put(unit, [index], field=[torch.tensor([index])])
    metadata = {"step": 5, "nested": {"keys": [1]}}
    path = tmp_path / "data_plane"
    captured = snapshot.capture_simple_checkpoint(
        path, num_storage_units=2, timeout_s=5.0, metadata=metadata
    )
    assert not (path / "metadata.json").exists()
    assert not (path / "controller_state.pkl").exists()
    client.controller.partitions["data"].keys_mapping["new"] = 1
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
    restored_controller = _Controller()
    restored_controller.load_checkpoint(str(path / "controller_state.pkl"))
    assert restored_controller.partitions["data"].keys_mapping == {}
    captured.release()
    assert all(not unit._snapshot_views for unit in units)
    assert not client.controller._controller_snapshot_views


def test_partial_capture_failure_releases_every_unit(fake_cluster, tmp_path):
    units, client = fake_cluster
    # One already-retained snapshot must not be disturbed by failed new capture.
    units[0].nrl_capture_snapshot("prior")
    with pytest.raises(RuntimeError, match="Only one"):
        snapshot.capture_simple_checkpoint(
            tmp_path / "state", num_storage_units=2, timeout_s=5.0, metadata=None
        )
    assert set(units[0]._snapshot_views) == {"prior"}
    assert not units[1]._snapshot_views
    assert not client.controller._controller_snapshot_views


def test_controller_serialization_failure_does_not_retain_partial_view(
    fake_cluster, monkeypatch, tmp_path
):
    units, client = fake_cluster

    def fail(*args, **kwargs):
        raise ValueError("invalid metadata")

    monkeypatch.setattr(controller_snapshot.pickle, "dumps", fail)
    with pytest.raises(ValueError, match="invalid metadata"):
        snapshot.capture_simple_checkpoint(
            tmp_path / "state", num_storage_units=2, timeout_s=5.0, metadata=None
        )
    assert not client.controller._controller_snapshot_views
    assert all(not unit._snapshot_views for unit in units)


def test_controller_release_submission_failure_still_releases_shards(
    fake_cluster, tmp_path
):
    units, _ = fake_cluster
    captured = snapshot.capture_simple_checkpoint(
        tmp_path / "state", num_storage_units=2, timeout_s=5.0, metadata=None
    )

    def fail(*args):
        raise RuntimeError("controller release submission failed")

    original = captured.controller.nrl_release_snapshot
    captured.controller.nrl_release_snapshot = SimpleNamespace(remote=fail)
    with pytest.raises(RuntimeError, match="controller release submission failed"):
        captured.release()
    assert all(not unit._snapshot_views for unit in units)
    captured.controller.nrl_release_snapshot = original
    captured.release()


def test_controller_write_failure_settles_shards_and_has_no_completion_marker(
    fake_cluster, monkeypatch, tmp_path
):
    units, client = fake_cluster
    captured = snapshot.capture_simple_checkpoint(
        tmp_path / "state", num_storage_units=2, timeout_s=5.0, metadata=None
    )

    def fail(*args):
        raise OSError("controller write failed")

    monkeypatch.setattr(client.controller, "nrl_persist_snapshot", fail)
    captured.controller = _remote_unit(client.controller)
    with pytest.raises(OSError, match="controller write failed"):
        captured.persist()
    assert len(list((tmp_path / "state/simple_storage").glob("su_*.pkl"))) == 2
    assert not (tmp_path / "state/metadata.json").exists()
    captured.release()
    assert not client.controller._controller_snapshot_views
    assert all(not unit._snapshot_views for unit in units)


def test_persist_failure_has_no_completion_marker(fake_cluster, monkeypatch, tmp_path):
    units, _ = fake_cluster
    captured = snapshot.capture_simple_checkpoint(
        tmp_path / "state", num_storage_units=2, timeout_s=5.0, metadata=None
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


@pytest.mark.parametrize("fail_write", [False, True])
def test_persist_is_single_attempt_even_after_failure(
    fake_cluster, monkeypatch, tmp_path, fail_write
):
    units, _ = fake_cluster
    captured = snapshot.capture_simple_checkpoint(
        tmp_path / "state", num_storage_units=2, metadata=None, timeout_s=5.0
    )
    if fail_write:

        def fail(*args):
            raise OSError("disk full")

        monkeypatch.setattr(units[0], "nrl_persist_snapshot", fail)
        captured.units = [_remote_unit(units[1]), _remote_unit(units[0])]
        with pytest.raises(OSError, match="disk full"):
            captured.persist()
    else:
        captured.persist()
    with pytest.raises(RuntimeError, match="single-attempt"):
        captured.persist()
    captured.release()
    captured.release()
    with pytest.raises(RuntimeError, match="released"):
        captured.persist()


def test_persist_and_release_timeouts_retain_inflight_views(fake_cluster, tmp_path):
    units, client = fake_cluster
    captured = snapshot.capture_simple_checkpoint(
        tmp_path / "state", num_storage_units=2, metadata=None, timeout_s=0.03
    )
    entered, finish = threading.Event(), threading.Event()

    def slow_write(*args):
        entered.set()
        assert finish.wait(5)
        return units[1].nrl_persist_snapshot(*args)

    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        captured.units[0].nrl_persist_snapshot = SimpleNamespace(
            remote=lambda *args: pool.submit(slow_write, *args)
        )
        started = time.monotonic()
        try:
            with pytest.raises(TimeoutError, match="RPC wait timed out"):
                captured.persist()
            assert entered.is_set()
            with pytest.raises(TimeoutError, match="RPC wait timed out"):
                captured.release()
            assert time.monotonic() - started < 2
            assert all(unit._snapshot_views for unit in units)
            assert client.controller._controller_snapshot_views
            assert not (captured.checkpoint_dir / "metadata.json").exists()
            with pytest.raises(RuntimeError, match="single-attempt"):
                captured.persist()
        finally:
            finish.set()
    assert not (captured.checkpoint_dir / "metadata.json").exists()
    captured.release()
    assert not client.controller._controller_snapshot_views
    assert all(not unit._snapshot_views for unit in units)


def test_release_ack_timeout_can_be_retried(fake_cluster, tmp_path):
    units, _ = fake_cluster
    captured = snapshot.capture_simple_checkpoint(
        tmp_path / "state", num_storage_units=2, metadata=None, timeout_s=0.02
    )
    original = captured.units[0].nrl_release_snapshot
    pending = concurrent.futures.Future()
    captured.units[0].nrl_release_snapshot = SimpleNamespace(
        remote=lambda token: pending
    )
    with pytest.raises(TimeoutError, match="RPC wait timed out"):
        captured.release()
    assert units[1]._snapshot_views
    assert not units[0]._snapshot_views
    captured.units[0].nrl_release_snapshot = original
    captured.release()
    assert all(not unit._snapshot_views for unit in units)


def test_finished_remote_timeout_error_does_not_prevent_release(fake_cluster, tmp_path):
    units, client = fake_cluster
    captured = snapshot.capture_simple_checkpoint(
        tmp_path / "state", num_storage_units=2, metadata=None, timeout_s=5.0
    )

    def fail(*args):
        raise TimeoutError("filesystem operation failed")

    captured.units[0].nrl_persist_snapshot = _RemoteMethod(fail)
    with pytest.raises(TimeoutError, match="filesystem operation failed"):
        captured.persist()
    captured.release()
    assert not client.controller._controller_snapshot_views
    assert all(not unit._snapshot_views for unit in units)


def test_wait_deadline_covers_all_refs_and_takes_priority_over_write_error(monkeypatch):
    failed, waiting, completed = (
        concurrent.futures.Future(),
        concurrent.futures.Future(),
        concurrent.futures.Future(),
    )
    failed.set_exception(OSError("failed write"))
    completed.set_result(42)
    seen = []

    def get(ref, *, timeout):
        seen.append(ref)
        return _fake_get(ref, timeout)

    monkeypatch.setattr(snapshot.ray, "get", get)
    with pytest.raises(TimeoutError, match="remote writes may still be running"):
        snapshot._wait_all([failed, waiting, completed], timeout_s=0.01)
    assert seen == [failed, waiting, completed]


def test_existing_checkpoint_is_not_overwritten(fake_cluster, tmp_path):
    path = tmp_path / "state"
    path.mkdir()
    (path / "marker").write_text("durable")
    with pytest.raises(FileExistsError):
        snapshot.capture_simple_checkpoint(
            path, num_storage_units=2, timeout_s=5.0, metadata=None
        )
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
        tmp_path / "state", num_storage_units=2, timeout_s=5.0, metadata=None
    )

    def fail_submit(*args):
        raise RuntimeError("actor submission failed")

    captured.units[0].nrl_release_snapshot = SimpleNamespace(remote=fail_submit)
    with pytest.raises(RuntimeError, match="actor submission failed"):
        captured.release()
    # Manager order is reversed: unit 0 must still receive its release RPC.
    assert not units[0]._snapshot_views
