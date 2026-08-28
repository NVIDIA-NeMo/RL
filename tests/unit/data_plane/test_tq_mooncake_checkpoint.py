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

from __future__ import annotations

import asyncio
import ctypes
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import nemo_rl.data_plane.adapters.tq_mooncake_checkpoint as checkpoint_plugin
from nemo_rl.data_plane.adapters.tq_mooncake_checkpoint import (
    _load_storage_checkpoint,
    _master_command,
    _physical_keys,
    _save_storage_checkpoint,
    _wait_for_disk_objects,
    install_tq_mooncake_checkpoint_plugin,
)


class _StatusMatrix:
    def __init__(self, values: dict[tuple[int, int], int]) -> None:
        self._values = values

    def __getitem__(self, index: tuple[int, int]) -> int:
        return self._values[index]


def _controller_state() -> dict[str, Any]:
    partition = SimpleNamespace(
        global_indexes={0, 1},
        field_name_mapping={"tokens": 0, "router_indices": 1, "metadata": 2},
        production_status=_StatusMatrix(
            {
                (0, 0): 1,
                (0, 1): 1,
                (0, 2): 0,
                (1, 0): 1,
                (1, 1): 0,
                (1, 2): 1,
            }
        ),
        field_custom_backend_meta={0: {"tokens": {"n_chunks": 2}}},
    )
    return {"partitions": {"train": partition}}


class _DiskDescriptor:
    def __init__(self, path: Path, size: int) -> None:
        self.status = SimpleNamespace(name="COMPLETE")
        self._disk = SimpleNamespace(file_path=str(path), object_size=size)

    def is_disk_replica(self) -> bool:
        return True

    def get_disk_descriptor(self) -> Any:
        return self._disk


class _FakeStore:
    def __init__(self, live_root: Path, objects: dict[str, bytes]) -> None:
        self.live_root = live_root
        self.objects = dict(objects)
        self.paths: dict[str, Path] = {}
        self.registered: set[int] = set()
        self.mapped_upserts: list[str] = []
        live_root.mkdir(parents=True, exist_ok=True)
        for index, (key, value) in enumerate(sorted(objects.items())):
            path = live_root / f"source-{index}.bin"
            path.write_bytes(value)
            self.paths[key] = path

    def batch_get_replica_desc(self, keys: list[str]) -> dict[str, list[Any]]:
        return {
            key: [_DiskDescriptor(self.paths[key], len(self.objects[key]))]
            for key in keys
            if key in self.paths
        }

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return [1 if key in self.objects else 0 for key in keys]

    def register_buffer(self, pointer: int, _size: int) -> int:
        self.registered.add(pointer)
        return 0

    def unregister_buffer(self, pointer: int) -> int:
        self.registered.remove(pointer)
        return 0

    def upsert_from(self, key: str, pointer: int, size: int, _config: Any) -> int:
        assert pointer in self.registered
        value = ctypes.string_at(pointer, size)
        self.objects[key] = value
        self.mapped_upserts.append(key)
        path = self.live_root / f"restored-{len(self.paths)}.bin"
        path.write_bytes(value)
        self.paths[key] = path
        return 0


def _manager(storage_root: Path, store: _FakeStore) -> Any:
    config = {
        "use_gdr": False,
        "gdr_staging_buffer_mb": 1024,
        "checkpoint": {
            "enabled": True,
            "storage_root": str(storage_root),
            "durability_timeout_s": 1.0,
            "poll_interval_s": 0.001,
        },
    }
    client = SimpleNamespace(_store=store, replica_config=object())
    return SimpleNamespace(config=config, storage_client=client)


def _write_controller(checkpoint_dir: Path) -> None:
    from transfer_queue import interface as tq_interface

    with (checkpoint_dir / tq_interface._CONTROLLER_FILE).open("wb") as output:
        pickle.dump(_controller_state(), output)


def test_physical_keys_include_all_produced_fields_and_gdr_chunks() -> None:
    assert _physical_keys(_controller_state()) == [
        "0@router_indices",
        "0@tokens:c0",
        "0@tokens:c1",
        "1@metadata",
        "1@tokens",
    ]


def test_storage_checkpoint_round_trip_uses_raw_mooncake_objects(
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "storage"
    live_root = storage_root / "live"
    payloads = {
        "0@router_indices": b"router",
        "0@tokens:c0": b"token-chunk-0",
        "0@tokens:c1": b"token-chunk-1",
        "1@metadata": b"pickled non-tensor bytes",
        "1@tokens": b"tokens",
    }
    source_store = _FakeStore(live_root, payloads)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _write_controller(checkpoint_dir)

    _save_storage_checkpoint(_manager(storage_root, source_store), str(checkpoint_dir))

    manifest = json.loads(
        (checkpoint_dir / "mooncake_storage" / "manifest.json").read_text()
    )
    assert [entry["key"] for entry in manifest["objects"]] == sorted(payloads)

    restored_store = _FakeStore(live_root, {})
    _load_storage_checkpoint(
        _manager(storage_root, restored_store), str(checkpoint_dir)
    )
    assert restored_store.objects == payloads
    assert restored_store.mapped_upserts == sorted(payloads)
    assert restored_store.registered == set()


def test_restore_rejects_a_corrupt_payload(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    payloads = {
        "0@router_indices": b"router",
        "0@tokens:c0": b"token-chunk-0",
        "0@tokens:c1": b"token-chunk-1",
        "1@metadata": b"pickled non-tensor bytes",
        "1@tokens": b"tokens",
    }
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _write_controller(checkpoint_dir)
    _save_storage_checkpoint(
        _manager(storage_root, _FakeStore(storage_root / "live", payloads)),
        str(checkpoint_dir),
    )
    first_payload = next((checkpoint_dir / "mooncake_storage" / "objects").iterdir())
    first_payload.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="Corrupt Mooncake checkpoint payload"):
        _load_storage_checkpoint(
            _manager(storage_root, _FakeStore(storage_root / "live", {})),
            str(checkpoint_dir),
        )


def test_save_rejects_storage_root_inside_checkpoint_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "checkpoint"
    checkpoint_tmp = tmp_path / "checkpoint.tmp"
    checkpoint_tmp.mkdir()
    _write_controller(checkpoint_tmp)
    storage_root = destination / "active-storage"
    store = _FakeStore(storage_root / "live", {})

    with pytest.raises(ValueError, match="inside the TQ checkpoint destination"):
        _save_storage_checkpoint(_manager(storage_root, store), str(checkpoint_tmp))


def test_disk_wait_rejects_distinct_keys_with_the_same_path(tmp_path: Path) -> None:
    live_root = tmp_path / "live"
    store = _FakeStore(live_root, {"key-a": b"a", "key-b": b"b"})
    store.paths["key-b"] = store.paths["key-a"]

    with pytest.raises(RuntimeError, match="distinct keys to the same DISK path"):
        _wait_for_disk_objects(
            store,
            ["key-a", "key-b"],
            live_root=live_root,
            timeout_s=1.0,
            poll_interval_s=0.001,
        )


def test_restore_rejects_a_different_gdr_layout(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _write_controller(checkpoint_dir)
    payloads = {key: key.encode() for key in _physical_keys(_controller_state())}
    _save_storage_checkpoint(
        _manager(storage_root, _FakeStore(storage_root / "live", payloads)),
        str(checkpoint_dir),
    )
    restore_manager = _manager(storage_root, _FakeStore(storage_root / "live", {}))
    restore_manager.config["use_gdr"] = True

    with pytest.raises(ValueError, match="storage layout does not match"):
        _load_storage_checkpoint(restore_manager, str(checkpoint_dir))


def test_master_command_enables_direct_disk_without_offload(tmp_path: Path) -> None:
    command = _master_command(
        executable="/opt/mooncake_master",
        metadata_server="10.0.0.1:50050",
        master_server_address="10.0.0.1:50051",
        live_root=tmp_path / "live",
        session_id="session-1",
    )
    assert f"--root_fs_dir={tmp_path / 'live'}" in command
    assert "--cluster_id=session-1" in command
    assert "--enable_disk_eviction=false" in command
    assert "--enable_offload=false" in command
    assert not any(argument.startswith("--offload_on_evict") for argument in command)


def test_installed_put_fence_waits_for_the_clients_disk_replica(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transfer_queue.storage.clients import mooncake_client
    from transfer_queue.storage.managers.base import StorageManagerFactory

    calls: list[tuple[str, list[str]]] = []

    def original_upsert(
        _client: Any, keys: list[str], _pointers: list[int], _sizes: list[int]
    ) -> None:
        calls.append(("upsert", keys))

    def wait_for_disk(_store: Any, keys: list[str], **_kwargs: Any) -> list[Any]:
        calls.append(("wait", keys))
        return []

    monkeypatch.setattr(
        mooncake_client.MooncakeStoreClient,
        "_batch_upsert_with_retry",
        original_upsert,
    )
    monkeypatch.setitem(
        StorageManagerFactory._registry,
        "MooncakeStore",
        StorageManagerFactory._registry["MooncakeStore"],
    )
    monkeypatch.setattr(checkpoint_plugin, "_wait_for_disk_objects", wait_for_disk)
    install_tq_mooncake_checkpoint_plugin()

    storage_root = tmp_path / "storage"
    fake_client = SimpleNamespace(
        config={
            "checkpoint": {
                "enabled": True,
                "storage_root": str(storage_root),
                "durability_timeout_s": 1.0,
                "poll_interval_s": 0.001,
            }
        },
        _store=object(),
    )
    mooncake_client.MooncakeStoreClient._batch_upsert_with_retry(
        fake_client, ["0@tokens"], [1], [6]
    )
    assert calls == [("upsert", ["0@tokens"]), ("wait", ["0@tokens"])]


def test_installed_manager_dispatches_storage_save_and_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transfer_queue.storage.managers.base import StorageManagerFactory

    monkeypatch.setitem(
        StorageManagerFactory._registry,
        "MooncakeStore",
        StorageManagerFactory._registry["MooncakeStore"],
    )
    install_tq_mooncake_checkpoint_plugin()
    manager_type = StorageManagerFactory._registry["MooncakeStore"]
    manager = object.__new__(manager_type)
    manager.config = {"checkpoint": {"enabled": True}}
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        checkpoint_plugin,
        "_save_storage_checkpoint",
        lambda _manager, path: calls.append(("save", path)),
    )
    monkeypatch.setattr(
        checkpoint_plugin,
        "_load_storage_checkpoint",
        lambda _manager, path: calls.append(("load", path)),
    )

    asyncio.run(manager.save_checkpoint("/checkpoint-save"))
    asyncio.run(manager.load_checkpoint("/checkpoint-load"))

    assert calls == [
        ("save", "/checkpoint-save"),
        ("load", "/checkpoint-load"),
    ]
