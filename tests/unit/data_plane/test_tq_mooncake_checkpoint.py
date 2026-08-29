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


class _FakeMemoryReplica:
    def __init__(self, endpoint: str, size: int) -> None:
        self.status = SimpleNamespace(name="COMPLETE")
        self._memory = SimpleNamespace(
            buffer_descriptor=SimpleNamespace(
                transport_endpoint=endpoint,
                size=size,
            )
        )

    def is_memory_replica(self) -> bool:
        return True

    def get_memory_descriptor(self) -> Any:
        return self._memory


class _FakeCluster:
    def __init__(
        self,
        objects: dict[str, bytes],
        owners: dict[str, tuple[str, ...]],
    ) -> None:
        self.objects = dict(objects)
        self.owners = dict(owners)
        self.get_calls: list[tuple[str, str]] = []
        self.upsert_calls: list[tuple[str, str, str]] = []


class _FakeStore:
    def __init__(
        self,
        cluster: _FakeCluster,
        endpoint: str,
        *,
        unregister_result: int = 0,
        short_key: str | None = None,
        owner_override: str | None = None,
    ) -> None:
        self.cluster = cluster
        self.endpoint = endpoint
        self.unregister_result = unregister_result
        self.short_key = short_key
        self.owner_override = owner_override
        self.registered: dict[int, int] = {}

    def get_hostname(self) -> str:
        return self.endpoint

    def get_size(self, key: str) -> int:
        value = self.cluster.objects.get(key)
        return -1 if value is None else len(value)

    def _replicas(self, key: str) -> list[_FakeMemoryReplica]:
        value = self.cluster.objects.get(key)
        if value is None:
            return []
        return [
            _FakeMemoryReplica(endpoint, len(value))
            for endpoint in self.cluster.owners.get(key, ())
        ]

    def get_replica_desc(self, key: str) -> list[_FakeMemoryReplica]:
        return self._replicas(key)

    def batch_get_replica_desc(
        self, keys: list[str]
    ) -> dict[str, list[_FakeMemoryReplica]]:
        return {key: self._replicas(key) for key in keys}

    def register_buffer(self, pointer: int, size: int) -> int:
        assert pointer not in self.registered
        self.registered[pointer] = size
        return 0

    def get_into(self, key: str, pointer: int, size: int) -> int:
        assert self.registered[pointer] == size
        assert self.endpoint in self.cluster.owners[key], (
            f"{self.endpoint} tried to checkpoint remotely owned {key}"
        )
        value = self.cluster.objects[key]
        assert len(value) == size
        ctypes.memmove(pointer, value, size)
        self.cluster.get_calls.append((self.endpoint, key))
        return size - 1 if key == self.short_key else size

    def unregister_buffer(self, pointer: int) -> int:
        if self.unregister_result == 0:
            self.registered.pop(pointer)
        return self.unregister_result

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return [1 if key in self.cluster.objects else 0 for key in keys]

    def upsert_from(self, key: str, pointer: int, size: int, config: Any) -> int:
        assert self.registered[pointer] == size
        assert config.preferred_segment == self.endpoint
        self.cluster.objects[key] = ctypes.string_at(pointer, size)
        self.cluster.owners[key] = (self.owner_override or self.endpoint,)
        self.cluster.upsert_calls.append((self.endpoint, key, config.preferred_segment))
        return 0


def _manager(store: _FakeStore, manager_id: str = "manager-a") -> Any:
    config = {
        "use_gdr": False,
        "gdr_staging_buffer_mb": 1024,
        "checkpoint": {
            "enabled": True,
            "timeout_s": 10.0,
            "max_parallel": 8,
        },
    }
    replica_config = SimpleNamespace(
        replica_num=1,
        with_soft_pin=False,
        with_hard_pin=True,
        prefer_alloc_in_same_node=False,
        data_type=None,
    )
    client = SimpleNamespace(
        _store=store,
        replica_config=replica_config,
        metadata_server="http://metadata.example/metadata",
    )
    return SimpleNamespace(
        config=config,
        storage_client=client,
        storage_manager_id=manager_id,
        controller_info=SimpleNamespace(
            id="controller-test",
            ip="10.0.0.100",
            ports={"request": 15001, "response": 15002},
        ),
    )


def _participant(manager: Any) -> Any:
    participant = checkpoint_plugin._CheckpointParticipant(manager)
    info = participant.info
    participant.info = checkpoint_plugin._ParticipantInfo(
        participant_id=info.participant_id,
        incarnation=info.incarnation,
        controller_session=info.controller_session,
        control_endpoint=f"inproc://{info.participant_id}",
        segment_name=info.segment_name,
        transport_endpoint=info.transport_endpoint,
    )
    return participant


def _managers(
    cluster: _FakeCluster,
    identities: list[tuple[str, str]],
    *,
    store_options: dict[str, dict[str, Any]] | None = None,
) -> list[Any]:
    options = store_options or {}
    return [
        _manager(
            _FakeStore(cluster, endpoint, **options.get(manager_id, {})),
            manager_id,
        )
        for manager_id, endpoint in identities
    ]


def _wire_participants(
    monkeypatch: pytest.MonkeyPatch,
    participants: list[Any],
    *,
    observe_response: Any = None,
    transform_responses: Any = None,
) -> list[list[Any]]:
    by_id = {
        participant.info.participant_id: participant for participant in participants
    }
    calls: list[list[Any]] = []

    monkeypatch.setattr(
        checkpoint_plugin,
        "_live_participants",
        lambda _manager: [participant.info for participant in participants],
    )
    monkeypatch.setattr(
        checkpoint_plugin,
        "_local_replica_config",
        lambda _manager, segment_name: SimpleNamespace(preferred_segment=segment_name),
    )

    def fanout(requests: list[Any], **_kwargs: Any) -> dict[str, dict[str, Any]]:
        calls.append(list(requests))
        responses: dict[str, dict[str, Any]] = {}
        for request in requests:
            participant_id = request.participant.participant_id
            response = by_id[participant_id]._dispatch(request.body)
            if observe_response is not None:
                observe_response(request, response)
            responses[participant_id] = response
        if transform_responses is not None:
            return transform_responses(requests, responses)
        return responses

    monkeypatch.setattr(checkpoint_plugin, "_fanout_requests", fanout)
    return calls


_SOURCE_IDENTITIES = [
    ("manager-a", "10.0.0.1:12301"),
    ("manager-b", "10.0.0.2:12302"),
]


def _source_cluster() -> _FakeCluster:
    payloads = _payloads()
    endpoints = [endpoint for _, endpoint in _SOURCE_IDENTITIES]
    owners = {
        key: (endpoints[index % len(endpoints)],)
        for index, key in enumerate(sorted(payloads))
    }
    return _FakeCluster(payloads, owners)


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
    payloads: dict[str, bytes] = {}
    shards: dict[str, bytes] = {}
    for entry in _manifest(checkpoint_dir)["objects"]:
        packed = shards.get(entry["shard"])
        if packed is None:
            packed = (storage_dir / entry["shard"]).read_bytes()
            shards[entry["shard"]] = packed
        payloads[entry["key"]] = packed[
            entry["offset"] : entry["offset"] + entry["size"]
        ]
    return payloads


def _corrupt_first_object(checkpoint_dir: Path) -> None:
    storage_dir = checkpoint_dir / "mooncake_storage"
    first = _manifest(checkpoint_dir)["objects"][0]
    packed_path = storage_dir / first["shard"]
    packed = bytearray(packed_path.read_bytes())
    packed[first["offset"]] ^= 0xFF
    packed_path.write_bytes(packed)


def _checkpoint_dir(tmp_path: Path) -> Path:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _write_controller(checkpoint_dir)
    return checkpoint_dir


def _save_distributed_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, _FakeCluster, list[Any], list[Any]]:
    cluster = _source_cluster()
    managers = _managers(cluster, _SOURCE_IDENTITIES)
    participants = [_participant(manager) for manager in managers]
    calls = _wire_participants(monkeypatch, participants)
    checkpoint_dir = _checkpoint_dir(tmp_path)
    _save_storage_checkpoint(managers[0], str(checkpoint_dir))
    return checkpoint_dir, cluster, managers, calls


def test_physical_keys_include_all_produced_fields_and_gdr_chunks() -> None:
    assert _physical_keys(_controller_state()) == [
        "0@router_indices",
        "0@tokens:c0",
        "0@tokens:c1",
        "1@metadata",
        "1@tokens",
    ]


@pytest.mark.parametrize(
    ("unsupported_mode", "message"),
    [
        ("unpinned", "hard-pinned"),
        ("offload", "offload"),
    ],
)
def test_checkpoint_participant_rejects_unsupported_runtime_modes(
    unsupported_mode: str,
    message: str,
) -> None:
    cluster = _source_cluster()
    manager = _managers(cluster, [_SOURCE_IDENTITIES[0]])[0]
    if unsupported_mode == "unpinned":
        manager.storage_client.replica_config.with_hard_pin = False
    else:
        manager.config["offload"] = {"enabled": True}

    with pytest.raises(NotImplementedError, match=message):
        _participant(manager)


def test_owner_distributed_checkpoint_round_trip_uses_current_participants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_cluster = _source_cluster()
    source_managers = _managers(source_cluster, _SOURCE_IDENTITIES)
    source_participants = [_participant(manager) for manager in source_managers]
    checkpoint_dir = _checkpoint_dir(tmp_path)
    manifest_path = checkpoint_dir / "mooncake_storage" / "manifest.json"

    def observe_save_ack(_request: Any, _response: Any) -> None:
        # A participant ACK means only its own shard is durable. The global
        # commit record must not exist until every ACK has been validated.
        assert not manifest_path.exists()

    _wire_participants(
        monkeypatch,
        source_participants,
        observe_response=observe_save_ack,
    )
    _save_storage_checkpoint(source_managers[0], str(checkpoint_dir))

    manifest = _manifest(checkpoint_dir)
    assert manifest["version"] == 3
    assert sorted(
        path.name for path in (checkpoint_dir / "mooncake_storage").iterdir()
    ) == [
        "manifest.json",
        "part-00000.bin",
        "part-00001.bin",
    ]
    assert _saved_payloads(checkpoint_dir) == _payloads()
    assert set(source_cluster.get_calls) == {
        (source_cluster.owners[key][0], key) for key in _payloads()
    }
    assert all(
        manager.storage_client._store.registered == {} for manager in source_managers
    )

    offsets: dict[str, int] = {}
    for entry in manifest["objects"]:
        value = _payloads()[entry["key"]]
        assert entry == {
            "key": entry["key"],
            "shard": entry["shard"],
            "offset": offsets.get(entry["shard"], 0),
            "size": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
            "saved_owner": source_cluster.owners[entry["key"]][0],
        }
        offsets[entry["shard"]] = entry["offset"] + len(value)

    current_identities = [
        ("current-a", "10.1.0.1:13301"),
        ("current-b", "10.1.0.2:13302"),
    ]
    restored_cluster = _FakeCluster({}, {})
    restore_managers = _managers(restored_cluster, current_identities)
    restore_participants = [_participant(manager) for manager in restore_managers]
    _wire_participants(monkeypatch, restore_participants)
    _load_storage_checkpoint(restore_managers[0], str(checkpoint_dir))

    current_endpoints = {endpoint for _, endpoint in current_identities}
    assert restored_cluster.objects == _payloads()
    assert set(restored_cluster.owners.values()) <= {
        (endpoint,) for endpoint in current_endpoints
    }
    assert {
        endpoint for endpoint, _, _ in restored_cluster.upsert_calls
    } == current_endpoints
    assert all(
        endpoint == preferred_segment
        for endpoint, _, preferred_segment in restored_cluster.upsert_calls
    )
    assert not current_endpoints.intersection(
        endpoint for _, endpoint in _SOURCE_IDENTITIES
    )
    assert all(
        manager.storage_client._store.registered == {} for manager in restore_managers
    )


def test_restore_rejects_an_object_not_placed_on_its_assigned_participant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir, _, _, _ = _save_distributed_checkpoint(monkeypatch, tmp_path)
    endpoint_a = "10.1.0.1:13301"
    endpoint_b = "10.1.0.2:13302"
    restored_cluster = _FakeCluster({}, {})
    managers = _managers(
        restored_cluster,
        [("current-a", endpoint_a), ("current-b", endpoint_b)],
        store_options={"current-a": {"owner_override": endpoint_b}},
    )
    _wire_participants(monkeypatch, [_participant(manager) for manager in managers])

    with pytest.raises(RuntimeError, match="assigned checkpoint participant"):
        _load_storage_checkpoint(managers[0], str(checkpoint_dir))


def test_save_rejects_an_object_without_a_live_memory_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = _source_cluster()
    cluster.owners["1@tokens"] = ("10.9.9.9:19999",)
    managers = _managers(cluster, _SOURCE_IDENTITIES)
    participants = [_participant(manager) for manager in managers]
    calls = _wire_participants(monkeypatch, participants)
    checkpoint_dir = _checkpoint_dir(tmp_path)

    with pytest.raises(RuntimeError, match="no COMPLETE memory replica"):
        _save_storage_checkpoint(managers[0], str(checkpoint_dir))

    assert calls == []
    assert cluster.get_calls == []
    assert not (checkpoint_dir / "mooncake_storage" / "manifest.json").exists()


@pytest.mark.parametrize(
    "response_mode",
    [
        "missing_ack",
        "extra_ack",
        "missing_object",
        "duplicate_object",
        "wrong_size",
        "missing_shard",
        "short_shard",
    ],
)
def test_save_commits_no_manifest_without_exact_participant_acks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response_mode: str,
) -> None:
    cluster = _source_cluster()
    managers = _managers(cluster, _SOURCE_IDENTITIES)
    participants = [_participant(manager) for manager in managers]
    checkpoint_dir = _checkpoint_dir(tmp_path)
    manifest_path = checkpoint_dir / "mooncake_storage" / "manifest.json"

    def observe_ack(_request: Any, _response: Any) -> None:
        assert not manifest_path.exists()

    def transform(
        requests: list[Any], responses: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        last_id = requests[-1].participant.participant_id
        if response_mode == "missing_ack":
            responses.pop(last_id)
        elif response_mode == "extra_ack":
            responses["unexpected-participant"] = dict(responses[last_id])
        elif response_mode == "missing_object":
            responses[last_id]["objects"].pop()
        elif response_mode == "duplicate_object":
            objects = responses[last_id]["objects"]
            objects[-1] = dict(objects[0])
        elif response_mode == "wrong_size":
            responses[last_id]["objects"][0]["size"] += 1
        else:
            shard = responses[last_id]["objects"][0]["shard"]
            path = checkpoint_dir / "mooncake_storage" / shard
            if response_mode == "missing_shard":
                path.unlink()
            else:
                path.write_bytes(path.read_bytes()[:-1])
        return responses

    _wire_participants(
        monkeypatch,
        participants,
        observe_response=observe_ack,
        transform_responses=transform,
    )

    with pytest.raises(RuntimeError, match="ACK|mismatch|response set"):
        _save_storage_checkpoint(managers[0], str(checkpoint_dir))

    assert not manifest_path.exists()


def test_save_rejects_a_short_owner_get_and_unregisters_the_buffer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = _source_cluster()
    managers = _managers(
        cluster,
        _SOURCE_IDENTITIES,
        store_options={"manager-a": {"short_key": "0@tokens:c1"}},
    )
    participants = [_participant(manager) for manager in managers]
    _wire_participants(monkeypatch, participants)
    checkpoint_dir = _checkpoint_dir(tmp_path)

    with pytest.raises(RuntimeError):
        _save_storage_checkpoint(managers[0], str(checkpoint_dir))

    assert all(manager.storage_client._store.registered == {} for manager in managers)
    assert not (checkpoint_dir / "mooncake_storage" / "manifest.json").exists()


def test_save_quarantines_a_buffer_when_unregister_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quarantined_buffers: list[Any],
) -> None:
    cluster = _source_cluster()
    managers = _managers(
        cluster,
        _SOURCE_IDENTITIES,
        store_options={"manager-a": {"unregister_result": -1}},
    )
    participants = [_participant(manager) for manager in managers]
    _wire_participants(monkeypatch, participants)
    checkpoint_dir = _checkpoint_dir(tmp_path)

    with pytest.raises(RuntimeError, match="buffer cleanup failed"):
        _save_storage_checkpoint(managers[0], str(checkpoint_dir))

    assert len(quarantined_buffers) == 1
    assert quarantined_buffers[0].closed is False
    store = managers[0].storage_client._store
    pointer, size = next(iter(store.registered.items()))
    assert ctypes.string_at(pointer, size) == _payloads()["0@router_indices"]
    assert not (checkpoint_dir / "mooncake_storage" / "manifest.json").exists()


def test_restore_rejects_a_corrupt_owner_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir, _, _, _ = _save_distributed_checkpoint(monkeypatch, tmp_path)
    _corrupt_first_object(checkpoint_dir)
    cluster = _FakeCluster({}, {})
    managers = _managers(cluster, [("current-a", "10.1.0.1:13301")])
    _wire_participants(monkeypatch, [_participant(managers[0])])

    with pytest.raises(ValueError, match="Corrupt Mooncake checkpoint payload"):
        _load_storage_checkpoint(managers[0], str(checkpoint_dir))


def test_restore_rejects_a_different_gdr_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir, _, _, _ = _save_distributed_checkpoint(monkeypatch, tmp_path)
    cluster = _FakeCluster({}, {})
    restore_manager = _managers(cluster, [("current-a", "10.1.0.1:13301")])[0]
    restore_manager.config["use_gdr"] = True

    with pytest.raises(ValueError, match="storage layout does not match"):
        _load_storage_checkpoint(restore_manager, str(checkpoint_dir))


def test_restore_requires_a_clean_mooncake_store_before_fanout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir, _, _, _ = _save_distributed_checkpoint(monkeypatch, tmp_path)
    endpoint = "10.1.0.1:13301"
    cluster = _FakeCluster(
        {"0@router_indices": b"stale"},
        {"0@router_indices": (endpoint,)},
    )
    restore_manager = _managers(cluster, [("current-a", endpoint)])[0]
    fanout_called = False

    def unexpected_fanout(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal fanout_called
        fanout_called = True
        raise AssertionError("restore fanout ran before the clean-store preflight")

    monkeypatch.setattr(checkpoint_plugin, "_fanout_requests", unexpected_fanout)

    with pytest.raises(RuntimeError, match="requires a clean store"):
        _load_storage_checkpoint(restore_manager, str(checkpoint_dir))

    assert fanout_called is False


def test_restore_quarantines_a_buffer_when_unregister_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quarantined_buffers: list[Any],
) -> None:
    checkpoint_dir, _, _, _ = _save_distributed_checkpoint(monkeypatch, tmp_path)
    endpoint = "10.1.0.1:13301"
    cluster = _FakeCluster({}, {})
    managers = _managers(
        cluster,
        [("current-a", endpoint)],
        store_options={"current-a": {"unregister_result": -1}},
    )
    participant = _participant(managers[0])
    _wire_participants(monkeypatch, [participant])

    with pytest.raises(RuntimeError, match="buffer cleanup failed"):
        _load_storage_checkpoint(managers[0], str(checkpoint_dir))

    assert len(quarantined_buffers) == 1
    assert quarantined_buffers[0].closed is False
    store = managers[0].storage_client._store
    pointer, size = next(iter(store.registered.items()))
    assert ctypes.string_at(pointer, size) == _payloads()["0@router_indices"]


def test_plugin_install_does_not_change_the_normal_put_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transfer_queue.storage.clients import mooncake_client
    from transfer_queue.storage.managers import mooncake_manager
    from transfer_queue.storage.managers.base import StorageManagerFactory

    original_upsert = mooncake_client.MooncakeStoreClient._batch_upsert_with_retry
    monkeypatch.setitem(
        StorageManagerFactory._registry,
        "MooncakeStore",
        mooncake_manager.MooncakeStorageManager,
    )

    install_tq_mooncake_checkpoint_plugin()

    assert (
        mooncake_client.MooncakeStoreClient._batch_upsert_with_retry is original_upsert
    )


def test_installed_manager_registers_its_actual_segment_and_closes_participant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ray

    from transfer_queue.storage.managers import mooncake_manager
    from transfer_queue.storage.managers.base import StorageManagerFactory

    endpoint = "10.3.0.7:14321"
    cluster = _FakeCluster({}, {})
    store = _FakeStore(cluster, endpoint)
    registered: list[dict[str, str]] = []
    unregistered: list[tuple[str, str]] = []
    manager_closes: list[str] = []

    class RemoteMethod:
        def __init__(self, function: Any) -> None:
            self.function = function

        def remote(self, *args: Any) -> Any:
            return self.function(*args)

    registry = SimpleNamespace(
        register=RemoteMethod(lambda info: registered.append(info) or "registered"),
        unregister=RemoteMethod(
            lambda participant_id, incarnation: unregistered.append(
                (participant_id, incarnation)
            )
            or "unregistered"
        ),
    )

    def base_init(self: Any, *_args: Any, **_kwargs: Any) -> None:
        fake = _manager(store, "manager-live")
        self.config = fake.config
        self.storage_client = fake.storage_client
        self.storage_manager_id = fake.storage_manager_id
        self.controller_info = fake.controller_info

    def run_endpoint(self: Any) -> None:
        info = self.info
        self.info = checkpoint_plugin._ParticipantInfo(
            participant_id=info.participant_id,
            incarnation=info.incarnation,
            controller_session=info.controller_session,
            control_endpoint="tcp://10.3.0.7:26321",
            segment_name=info.segment_name,
            transport_endpoint=info.transport_endpoint,
        )
        self._ready.set()

    monkeypatch.setattr(mooncake_manager.MooncakeStorageManager, "__init__", base_init)
    monkeypatch.setattr(
        mooncake_manager.MooncakeStorageManager,
        "close",
        lambda _self: manager_closes.append("manager"),
    )
    monkeypatch.setattr(checkpoint_plugin._CheckpointParticipant, "_run", run_endpoint)
    monkeypatch.setattr(checkpoint_plugin, "_registry_actor", lambda: registry)
    monkeypatch.setattr(ray, "get", lambda value, **_kwargs: value)
    monkeypatch.setitem(
        StorageManagerFactory._registry,
        "MooncakeStore",
        mooncake_manager.MooncakeStorageManager,
    )

    install_tq_mooncake_checkpoint_plugin()
    manager_type = StorageManagerFactory._registry["MooncakeStore"]
    manager = manager_type(object(), {})
    participant = manager._checkpoint_participant
    assert participant is not None

    assert registered == [
        {
            "participant_id": "manager-live",
            "incarnation": participant.info.incarnation,
            "controller_session": checkpoint_plugin._controller_session(manager),
            "control_endpoint": "tcp://10.3.0.7:26321",
            "segment_name": endpoint,
            "transport_endpoint": endpoint,
        }
    ]
    assert registered[0]["control_endpoint"] != registered[0]["segment_name"]

    manager.close()

    assert unregistered == [
        (participant.info.participant_id, participant.info.incarnation)
    ]
    assert manager._checkpoint_participant is None
    assert manager_closes == ["manager"]


def test_installed_manager_dispatches_explicit_storage_save_and_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transfer_queue.storage.managers import mooncake_manager
    from transfer_queue.storage.managers.base import StorageManagerFactory

    monkeypatch.setitem(
        StorageManagerFactory._registry,
        "MooncakeStore",
        mooncake_manager.MooncakeStorageManager,
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
