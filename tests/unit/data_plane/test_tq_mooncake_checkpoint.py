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
import hashlib
import json
import pickle
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import nemo_rl.data_plane.adapters.tq_mooncake_checkpoint as checkpoint_plugin
from nemo_rl.data_plane.adapters.tq_mooncake_checkpoint import (
    _load_storage_checkpoint,
    _physical_keys,
    _save_storage_checkpoint,
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


class _FakeStore:
    def __init__(
        self, objects: dict[str, bytes], *, unregister_result: int = 0
    ) -> None:
        self.objects = dict(objects)
        self.unregister_result = unregister_result
        self.registered: dict[int, int] = {}
        self.saved_gets: list[str] = []
        self.mapped_upserts: list[str] = []

    def get_size(self, key: str) -> int:
        value = self.objects.get(key)
        return -1 if value is None else len(value)

    def register_buffer(self, pointer: int, size: int) -> int:
        assert pointer not in self.registered
        self.registered[pointer] = size
        return 0

    def get_into(self, key: str, pointer: int, size: int) -> int:
        assert self.registered[pointer] == size
        value = self.objects[key]
        assert len(value) == size
        ctypes.memmove(pointer, value, size)
        self.saved_gets.append(key)
        return size

    def unregister_buffer(self, pointer: int) -> int:
        if self.unregister_result == 0:
            self.registered.pop(pointer)
        return self.unregister_result

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return [1 if key in self.objects else 0 for key in keys]

    def upsert_from(self, key: str, pointer: int, size: int, _config: Any) -> int:
        assert self.registered[pointer] == size
        self.objects[key] = ctypes.string_at(pointer, size)
        self.mapped_upserts.append(key)
        return 0


class _ShortGetStore(_FakeStore):
    def __init__(self, objects: dict[str, bytes], short_key: str) -> None:
        super().__init__(objects)
        self.short_key = short_key

    def get_into(self, key: str, pointer: int, size: int) -> int:
        result = super().get_into(key, pointer, size)
        return result - 1 if key == self.short_key else result


def _manager(store: _FakeStore) -> Any:
    config = {
        "use_gdr": False,
        "gdr_staging_buffer_mb": 1024,
        "checkpoint": {"enabled": True},
    }
    client = SimpleNamespace(_store=store, replica_config=object())
    return SimpleNamespace(config=config, storage_client=client)


@pytest.fixture
def quarantined_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[Any]]:
    buffers: list[Any] = []
    monkeypatch.setattr(
        checkpoint_plugin, "_QUARANTINED_BUFFERS", buffers, raising=False
    )
    yield buffers
    for buffer in buffers:
        if not buffer.closed:
            buffer.close()
    buffers.clear()


def _write_controller(checkpoint_dir: Path) -> None:
    from transfer_queue import interface as tq_interface

    with (checkpoint_dir / tq_interface._CONTROLLER_FILE).open("wb") as output:
        pickle.dump(_controller_state(), output)


def _payloads() -> dict[str, bytes]:
    return {
        "0@router_indices": b"router",
        "0@tokens:c0": b"token-chunk-0",
        "0@tokens:c1": b"token-chunk-1",
        "1@metadata": b"pickled non-tensor bytes",
        "1@tokens": b"tokens",
    }


def _manifest(checkpoint_dir: Path) -> dict[str, Any]:
    manifest_path = checkpoint_dir / "mooncake_storage" / "manifest.json"
    return json.loads(manifest_path.read_text())


def _saved_payloads(checkpoint_dir: Path) -> dict[str, bytes]:
    storage_dir = checkpoint_dir / "mooncake_storage"
    packed = (storage_dir / "objects.bin").read_bytes()
    return {
        entry["key"]: packed[entry["offset"] : entry["offset"] + entry["size"]]
        for entry in _manifest(checkpoint_dir)["objects"]
    }


def _corrupt_first_object(checkpoint_dir: Path) -> None:
    storage_dir = checkpoint_dir / "mooncake_storage"
    packed_path = storage_dir / "objects.bin"
    first = _manifest(checkpoint_dir)["objects"][0]
    packed = bytearray(packed_path.read_bytes())
    packed[first["offset"]] ^= 0xFF
    packed_path.write_bytes(packed)


def test_physical_keys_include_all_produced_fields_and_gdr_chunks() -> None:
    assert _physical_keys(_controller_state()) == [
        "0@router_indices",
        "0@tokens:c0",
        "0@tokens:c1",
        "1@metadata",
        "1@tokens",
    ]


def test_explicit_storage_checkpoint_round_trip_uses_raw_mooncake_objects(
    tmp_path: Path,
) -> None:
    payloads = _payloads()
    source_store = _FakeStore(payloads)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _write_controller(checkpoint_dir)

    _save_storage_checkpoint(_manager(source_store), str(checkpoint_dir))

    manifest = _manifest(checkpoint_dir)
    assert manifest["version"] == 2
    assert sorted(
        path.name for path in (checkpoint_dir / "mooncake_storage").iterdir()
    ) == [
        "manifest.json",
        "objects.bin",
    ]
    assert source_store.saved_gets == sorted(payloads)
    assert source_store.registered == {}
    assert _saved_payloads(checkpoint_dir) == payloads
    offset = 0
    for entry in manifest["objects"]:
        value = payloads[entry["key"]]
        assert entry == {
            "key": entry["key"],
            "offset": offset,
            "size": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
        offset += len(value)

    restored_store = _FakeStore({})
    _load_storage_checkpoint(_manager(restored_store), str(checkpoint_dir))

    assert restored_store.objects == payloads
    assert restored_store.mapped_upserts == sorted(payloads)
    assert restored_store.registered == {}


def test_save_rejects_a_missing_mooncake_object(tmp_path: Path) -> None:
    payloads = _payloads()
    payloads.pop("1@tokens")
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _write_controller(checkpoint_dir)

    with pytest.raises(RuntimeError, match="positive size.*'1@tokens'"):
        _save_storage_checkpoint(_manager(_FakeStore(payloads)), str(checkpoint_dir))


def test_save_rejects_a_short_get_and_unregisters_the_buffer(tmp_path: Path) -> None:
    payloads = _payloads()
    store = _ShortGetStore(payloads, "0@tokens:c0")
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _write_controller(checkpoint_dir)

    with pytest.raises(RuntimeError):
        _save_storage_checkpoint(_manager(store), str(checkpoint_dir))

    assert store.registered == {}


def test_save_quarantines_a_buffer_when_unregister_fails(
    tmp_path: Path,
    quarantined_buffers: list[Any],
) -> None:
    store = _FakeStore(_payloads(), unregister_result=-1)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _write_controller(checkpoint_dir)

    with pytest.raises(RuntimeError, match="buffer cleanup failed"):
        _save_storage_checkpoint(_manager(store), str(checkpoint_dir))

    assert len(quarantined_buffers) == 1
    assert quarantined_buffers[0].closed is False
    pointer, size = next(iter(store.registered.items()))
    assert ctypes.string_at(pointer, size) == _payloads()["0@router_indices"]


def test_restore_rejects_a_corrupt_payload(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _write_controller(checkpoint_dir)
    _save_storage_checkpoint(_manager(_FakeStore(_payloads())), str(checkpoint_dir))
    _corrupt_first_object(checkpoint_dir)

    with pytest.raises(ValueError, match="Corrupt Mooncake checkpoint payload"):
        _load_storage_checkpoint(_manager(_FakeStore({})), str(checkpoint_dir))


def test_restore_rechecks_the_bytes_loaded_after_manifest_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _write_controller(checkpoint_dir)
    _save_storage_checkpoint(_manager(_FakeStore(_payloads())), str(checkpoint_dir))
    original_load_manifest = checkpoint_plugin._load_manifest

    def load_manifest_then_corrupt(checkpoint_root: Path) -> Any:
        result = original_load_manifest(checkpoint_root)
        _corrupt_first_object(checkpoint_dir)
        return result

    monkeypatch.setattr(checkpoint_plugin, "_load_manifest", load_manifest_then_corrupt)

    with pytest.raises(ValueError, match="Corrupt Mooncake checkpoint payload"):
        _load_storage_checkpoint(_manager(_FakeStore({})), str(checkpoint_dir))


def test_restore_rejects_a_different_gdr_layout(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _write_controller(checkpoint_dir)
    _save_storage_checkpoint(_manager(_FakeStore(_payloads())), str(checkpoint_dir))
    restore_manager = _manager(_FakeStore({}))
    restore_manager.config["use_gdr"] = True

    with pytest.raises(ValueError, match="storage layout does not match"):
        _load_storage_checkpoint(restore_manager, str(checkpoint_dir))


def test_restore_requires_a_clean_mooncake_store(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _write_controller(checkpoint_dir)
    _save_storage_checkpoint(_manager(_FakeStore(_payloads())), str(checkpoint_dir))

    with pytest.raises(RuntimeError, match="requires a clean store"):
        _load_storage_checkpoint(
            _manager(_FakeStore({"0@router_indices": b"stale"})),
            str(checkpoint_dir),
        )


def test_restore_quarantines_a_buffer_when_unregister_fails(
    tmp_path: Path,
    quarantined_buffers: list[Any],
) -> None:
    payloads = _payloads()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _write_controller(checkpoint_dir)
    _save_storage_checkpoint(_manager(_FakeStore(payloads)), str(checkpoint_dir))
    store = _FakeStore({}, unregister_result=-1)

    with pytest.raises(RuntimeError, match="buffer cleanup failed"):
        _load_storage_checkpoint(_manager(store), str(checkpoint_dir))

    assert len(quarantined_buffers) == 1
    assert quarantined_buffers[0].closed is False
    pointer, size = next(iter(store.registered.items()))
    assert ctypes.string_at(pointer, size) == payloads["0@router_indices"]


def test_plugin_install_does_not_change_the_normal_put_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transfer_queue.storage.clients import mooncake_client
    from transfer_queue.storage.managers.base import StorageManagerFactory

    original_upsert = mooncake_client.MooncakeStoreClient._batch_upsert_with_retry
    monkeypatch.setitem(
        StorageManagerFactory._registry,
        "MooncakeStore",
        StorageManagerFactory._registry["MooncakeStore"],
    )

    install_tq_mooncake_checkpoint_plugin()

    assert (
        mooncake_client.MooncakeStoreClient._batch_upsert_with_retry is original_upsert
    )


def test_installed_manager_dispatches_explicit_storage_save_and_load(
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
