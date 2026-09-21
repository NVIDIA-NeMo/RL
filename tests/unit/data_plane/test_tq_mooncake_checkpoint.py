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
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

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
    def __init__(self, endpoint: str, size: int, address: int = 1) -> None:
        self.status = SimpleNamespace(name="COMPLETE")
        self._memory = SimpleNamespace(
            buffer_descriptor=SimpleNamespace(
                transport_endpoint=endpoint,
                size=size,
                buffer_address=address,
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
        self.buffers: dict[str, Any] = {
            key: ctypes.create_string_buffer(value, len(value))
            for key, value in objects.items()
        }


class _FakeStore:
    def __init__(
        self,
        cluster: _FakeCluster,
        endpoint: str,
        *,
        unregister_result: int = 0,
        owner_override: str | None = None,
    ) -> None:
        self.cluster = cluster
        self.endpoint = endpoint
        self.unregister_result = unregister_result
        self.owner_override = owner_override
        self.registered: dict[int, int] = {}
        self.registrations: list[tuple[int, int]] = []
        self.replica_batches: list[list[str]] = []
        self.get_batches: list[list[str]] = []
        self.upsert_batches: list[list[str]] = []

    def get_hostname(self) -> str:
        return self.endpoint

    def get_size(self, key: str) -> int:
        pytest.fail("checkpoint made a per-object size query")

    def _replicas(self, key: str) -> list[_FakeMemoryReplica]:
        value = self.cluster.objects.get(key)
        if value is None:
            return []
        buffer = self.cluster.buffers.get(key)
        if buffer is None or bytes(buffer) != value:
            buffer = ctypes.create_string_buffer(value, len(value))
            self.cluster.buffers[key] = buffer
        return [
            _FakeMemoryReplica(endpoint, len(value), ctypes.addressof(buffer))
            for endpoint in self.cluster.owners.get(key, ())
        ]

    def get_replica_desc(self, key: str) -> list[_FakeMemoryReplica]:
        pytest.fail("checkpoint made a per-object replica query")

    def batch_get_replica_desc(
        self, keys: list[str]
    ) -> dict[str, list[_FakeMemoryReplica]]:
        self.replica_batches.append(list(keys))
        return {key: self._replicas(key) for key in keys}

    def register_buffer(self, pointer: int, size: int) -> int:
        assert pointer not in self.registered
        self.registered[pointer] = size
        self.registrations.append((pointer, size))
        return 0

    def _assert_registered(self, pointer: int, size: int) -> None:
        assert pointer % 256 == 0
        assert any(
            base <= pointer and pointer + size <= base + capacity
            for base, capacity in self.registered.items()
        )

    def batch_get_into(
        self, keys: list[str], pointers: list[int], sizes: list[int]
    ) -> list[int]:
        pytest.fail(
            "owner-local SAVE must not fetch or copy payload through native GET"
        )

    def unregister_buffer(self, pointer: int) -> int:
        if self.unregister_result == 0:
            self.registered.pop(pointer)
        return self.unregister_result

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return [1 if key in self.cluster.objects else 0 for key in keys]

    def upsert_from(self, key: str, pointer: int, size: int, config: Any) -> int:
        self._assert_registered(pointer, size)
        assert config.preferred_segment == self.endpoint
        self.cluster.objects[key] = ctypes.string_at(pointer, size)
        self.cluster.owners[key] = (self.owner_override or self.endpoint,)
        self.cluster.upsert_calls.append((self.endpoint, key, config.preferred_segment))
        return 0

    def batch_upsert_from(
        self, keys: list[str], pointers: list[int], sizes: list[int], config: Any
    ) -> list[int]:
        self.upsert_batches.append(list(keys))
        return [
            self.upsert_from(key, pointer, size, config)
            for key, pointer, size in zip(keys, pointers, sizes, strict=True)
        ]


def _manager(store: _FakeStore, manager_id: str = "manager-a") -> Any:
    config = {
        "use_gdr": False,
        "gdr_staging_buffer_mb": 1024,
        "checkpoint": {
            "enabled": True,
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
        _checkpoint_participant=None,
        _checkpoint_workers=[],
        controller_info=SimpleNamespace(
            id="controller-test",
            ip="10.0.0.100",
            ports={"request": 15001, "response": 15002},
        ),
    )


def _participant(manager: Any) -> Any:
    participant = checkpoint_plugin._CheckpointParticipant(manager)
    manager._checkpoint_participant = participant
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
        lambda _manager: ([participant.info for participant in participants], {}),
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


class _RemoteMethod:
    def __init__(self, function: Any) -> None:
        self.function = function

    def remote(self, *, body: dict[str, Any]) -> Any:
        # Ray's cross-environment placeholder handles accept keyword calls only.
        return self.function(body)


class _FakeObjectRef:
    def __init__(self, value: Any) -> None:
        self.value = value


def _fake_ray_get(value: Any, **_kwargs: Any) -> Any:
    if isinstance(value, _FakeObjectRef):
        return value.value
    if isinstance(value, list):
        return [_fake_ray_get(item) for item in value]
    return value


@pytest.fixture(autouse=True)
def fake_ray_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise command and reference routing without launching a Ray cluster."""
    monkeypatch.setattr(checkpoint_plugin.ray, "put", _FakeObjectRef)
    monkeypatch.setattr(checkpoint_plugin.ray, "get", _fake_ray_get)
    monkeypatch.setattr(checkpoint_plugin.ray, "ObjectRef", _FakeObjectRef)


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
    for entry in _manifest_objects(checkpoint_dir):
        packed = shards.get(entry["shard"])
        if packed is None:
            packed = (storage_dir / entry["shard"]).read_bytes()
            shards[entry["shard"]] = packed
        payloads[entry["key"]] = packed[
            entry["offset"] : entry["offset"] + entry["size"]
        ]
    return payloads


def _manifest_objects(checkpoint_dir: Path) -> list[dict[str, Any]]:
    manifest = _manifest(checkpoint_dir)
    if "objects" in manifest:
        return manifest["objects"]
    storage_dir = checkpoint_dir / "mooncake_storage"
    return [
        entry
        for shard in manifest["shards"]
        for entry in json.loads(
            (storage_dir / f"{shard['shard']}.index.json").read_text()
        )["objects"]
    ]


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


@pytest.mark.parametrize("dtype", [torch.int8, torch.int64])
@pytest.mark.parametrize("indexes", [[1, 0, 1], {0, 1}])
def test_physical_keys_bulk_status_and_fallback_preserve_sorted_deduplication(
    dtype: torch.dtype, indexes: Any
) -> None:
    state = _controller_state()
    partition = state["partitions"]["train"]
    partition.global_indexes = indexes
    partition.production_status = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=dtype)
    state["partitions"]["overlapping"] = partition
    assert _physical_keys(state) == _physical_keys(_controller_state())


@pytest.mark.parametrize("published_before_cut", [True, False])
def test_save_uses_controller_production_cut_during_fresh_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    published_before_cut: bool,
) -> None:
    cluster = _source_cluster()
    managers = _managers(cluster, _SOURCE_IDENTITIES)
    state = _controller_state()
    staging = SimpleNamespace(
        global_indexes={2},
        field_name_mapping={"tokens": 0},
        production_status=_StatusMatrix({(2, 0): 0}),
        field_custom_backend_meta={},
    )
    state["partitions"]["rollout_staging"] = staging
    staging_key = "2@tokens"
    staging_payload = b"new staging tokens"

    def publish_staging() -> None:
        # TQ publishes production only after the native payload write completes.
        cluster.objects[staging_key] = staging_payload
        cluster.owners[staging_key] = (_SOURCE_IDENTITIES[0][1],)
        staging.production_status = _StatusMatrix({(2, 0): 1})

    if published_before_cut:
        publish_staging()
    checkpoint_dir = _checkpoint_dir(tmp_path)
    with checkpoint_plugin._controller_path(checkpoint_dir).open("wb") as output:
        pickle.dump(state, output)

    def publish_after_discovery(request: Any, _response: Any) -> None:
        if not published_before_cut and request.body["operation"] == "DISCOVER_SAVE":
            # The row was allocated in the cut, but its payload was not published.
            # A fresh write during SAVE must not change the selected old objects.
            publish_staging()

    _wire_participants(
        monkeypatch,
        [_participant(manager) for manager in managers],
        observe_response=publish_after_discovery,
    )
    _save_storage_checkpoint(managers[0], str(checkpoint_dir))

    expected = _payloads()
    if published_before_cut:
        expected[staging_key] = staging_payload
    assert staging.production_status[2, 0] == 1
    assert cluster.objects[staging_key] == staging_payload
    assert checkpoint_plugin._controller_keys(checkpoint_dir) == sorted(expected)
    assert _saved_payloads(checkpoint_dir) == expected

    restored_cluster = _FakeCluster({}, {})
    restored_manager = _manager(_FakeStore(restored_cluster, "10.1.0.1:13301"))
    _wire_participants(monkeypatch, [_participant(restored_manager)])
    _load_storage_checkpoint(restored_manager, str(checkpoint_dir))
    assert restored_cluster.objects == expected


def test_empty_checkpoint_needs_no_participants_or_load_controller_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(_FakeStore(_FakeCluster({}, {}), "127.0.0.1:12301"))
    checkpoint_dir = _checkpoint_dir(tmp_path)
    monkeypatch.setattr(checkpoint_plugin, "_controller_keys", lambda _root: [])

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail(
            "empty storage checkpoint unexpectedly inspected participants/controller"
        )

    monkeypatch.setattr(checkpoint_plugin, "_live_participants", unexpected)
    _save_storage_checkpoint(manager, str(checkpoint_dir))
    assert _manifest(checkpoint_dir)["shards"] == []
    monkeypatch.setattr(checkpoint_plugin, "_controller_keys", unexpected)
    _load_storage_checkpoint(manager, str(checkpoint_dir))


def test_load_has_one_restore_round_and_no_extra_controller_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir, _, _, _ = _save_distributed_checkpoint(monkeypatch, tmp_path)
    cluster = _FakeCluster({}, {})
    manager = _manager(_FakeStore(cluster, "127.0.0.1:12301"))
    calls = _wire_participants(monkeypatch, [_participant(manager)])

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("storage LOAD independently reread the TQ controller snapshot")

    monkeypatch.setattr(checkpoint_plugin, "_controller_keys", unexpected)
    _load_storage_checkpoint(manager, str(checkpoint_dir))
    assert cluster.objects == _payloads()
    assert len(calls) == 1
    assert calls[0][0].body["operation"] == "LOAD_SHARDS"


def test_distributed_checkpoint_round_trip_over_command_only_rpc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise command routing with owner-local stores and no coordinator I/O."""
    import ray

    monkeypatch.setattr(checkpoint_plugin, "_BATCH_KEYS", 2)
    payloads = _payloads()
    identities = [("owner-a", "127.0.0.1:12301"), ("owner-b", "127.0.0.1:12302")]
    owners = {
        key: (identities[index % len(identities)][1],)
        for index, key in enumerate(sorted(payloads))
    }
    cluster = _FakeCluster(payloads, owners)
    managers = _managers(cluster, identities)
    participants = [_participant(manager) for manager in managers]
    coordinator = _manager(_FakeStore(cluster, "127.0.0.1:12303"), "coordinator")
    commands: list[dict[str, Any]] = []

    def command(participant: Any, body: dict[str, Any]) -> dict[str, Any]:
        # Only opaque metadata references may accompany the JSON command fields.
        def encode_reference(value: Any) -> str:
            assert isinstance(value, _FakeObjectRef)
            return "metadata-reference"

        json.dumps(body, default=encode_reference)
        commands.append(body)
        response = participant._dispatch(body)
        json.dumps(response, default=encode_reference)
        return response

    coordinator._checkpoint_workers = [
        SimpleNamespace(
            mooncake_checkpoint=_RemoteMethod(
                lambda body, participant=participant: command(participant, body)
            )
        )
        for participant in participants
    ]
    monkeypatch.setattr(ray, "get", _fake_ray_get)

    def unexpected_payload_io(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("checkpoint coordinator performed payload I/O")

    monkeypatch.setattr(
        coordinator.storage_client._store, "batch_get_into", unexpected_payload_io
    )
    monkeypatch.setattr(
        coordinator.storage_client._store, "batch_upsert_from", unexpected_payload_io
    )
    monkeypatch.setattr(
        checkpoint_plugin,
        "_local_replica_config",
        lambda _manager, segment_name: SimpleNamespace(preferred_segment=segment_name),
    )
    checkpoint_dir = _checkpoint_dir(tmp_path)
    _save_storage_checkpoint(coordinator, str(checkpoint_dir))

    assert _saved_payloads(checkpoint_dir) == payloads
    shard_owners: dict[str, set[str]] = {}
    for entry in _manifest_objects(checkpoint_dir):
        assert entry["saved_owner"] == owners[entry["key"]][0]
        shard_owners.setdefault(entry["shard"], set()).add(entry["saved_owner"])
    assert len(shard_owners) == 2
    assert all(len(endpoints) == 1 for endpoints in shard_owners.values())
    assert cluster.get_calls == []
    assert all(not manager.storage_client._store.registrations for manager in managers)
    assert all(manager.storage_client._store.replica_batches for manager in managers)
    assert sorted(
        key
        for manager in managers
        for batch in manager.storage_client._store.replica_batches
        for key in batch
    ) == sorted(payloads)

    cluster.objects.clear()
    cluster.owners.clear()
    _load_storage_checkpoint(coordinator, str(checkpoint_dir))

    assert cluster.objects == payloads
    assert {endpoint for endpoint, _, _ in cluster.upsert_calls} == {
        endpoint for _, endpoint in identities
    }
    assert all(manager.storage_client._store.registered == {} for manager in managers)
    assert {body["operation"] for body in commands} == {
        "DESCRIBE",
        "DISCOVER_SAVE",
        "SAVE_SHARD",
        "LOAD_SHARDS",
    }


def test_checkpoint_timeout_matches_simple_default() -> None:
    assert checkpoint_plugin._DEFAULT_TIMEOUT_S == 200.0


def test_participant_timeout_is_fatal_to_the_checkpoint_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SC must not treat uncertain participant writes as a retryable local timeout."""
    import ray

    manager = _manager(_FakeStore(_FakeCluster({}, {}), "127.0.0.1:12301"))
    participant = _participant(manager)
    request = checkpoint_plugin._ParticipantRequest(
        participant.info, checkpoint_plugin._request_body(manager, "save")
    )

    def timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise TimeoutError("participant still writing")

    worker = SimpleNamespace(mooncake_checkpoint=_RemoteMethod(lambda _body: object()))
    monkeypatch.setattr(ray, "get", timeout)
    with pytest.raises(RuntimeError, match="checkpoint fanout failed") as error:
        checkpoint_plugin._fanout_requests(
            [request],
            workers={participant.info.participant_id: worker},
            local=None,
            timeout_s=0.1,
        )
    assert isinstance(error.value.__cause__, TimeoutError)


def test_command_fanout_dispatches_all_workers_before_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ray

    managers = _managers(
        _FakeCluster({}, {}),
        [(f"owner-{index}", f"127.0.0.1:{12300 + index}") for index in range(65)],
    )
    requests = [
        checkpoint_plugin._ParticipantRequest(
            _participant(manager).info, {"participant_id": manager.storage_manager_id}
        )
        for manager in managers
    ]
    expected = [request.participant.participant_id for request in requests]
    submitted: list[str] = []

    def submit(body: dict[str, Any]) -> dict[str, Any]:
        submitted.append(body["participant_id"])
        return {"ok": True, "participant_id": body["participant_id"]}

    def gather(values: list[Any], *, timeout: float) -> list[Any]:
        assert submitted == expected
        assert len(values) == len(requests)
        assert timeout == 10.0
        return values

    workers = {
        participant_id: SimpleNamespace(mooncake_checkpoint=_RemoteMethod(submit))
        for participant_id in expected
    }
    monkeypatch.setattr(ray, "get", gather)
    responses = checkpoint_plugin._fanout_requests(
        requests, workers=workers, local=None, timeout_s=10.0
    )
    assert responses == {
        participant_id: {"ok": True, "participant_id": participant_id}
        for participant_id in expected
    }


def test_command_fanout_dispatches_local_owner_without_ray(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ray

    manager = _manager(_FakeStore(_FakeCluster({}, {}), "10.0.0.1:12301"))
    participant = _participant(manager)
    request = checkpoint_plugin._ParticipantRequest(
        participant.info, checkpoint_plugin._request_body(manager, "DESCRIBE")
    )

    def empty_get(values: list[Any], **_kwargs: Any) -> list[Any]:
        assert values == [], "local participant was routed through Ray"
        return values

    monkeypatch.setattr(ray, "get", empty_get)
    responses = checkpoint_plugin._fanout_requests(
        [request], workers={}, local=participant, timeout_s=1.0
    )
    assert responses[participant.info.participant_id]["participant"] == asdict(
        participant.info
    )


def test_command_rejects_a_different_controller_session() -> None:
    manager = _manager(_FakeStore(_FakeCluster({}, {}), "10.0.0.1:12301"))
    participant = _participant(manager)
    body = checkpoint_plugin._request_body(manager, "DESCRIBE")
    body["controller_session"] = "another-run"

    with pytest.raises(ValueError, match="controller session mismatch"):
        participant._dispatch(body)


def test_live_participants_does_not_silently_drop_an_unavailable_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ray

    manager = _manager(_FakeStore(_FakeCluster({}, {}), "10.0.0.9:12309"))
    manager._checkpoint_workers = [
        SimpleNamespace(mooncake_checkpoint=_RemoteMethod(lambda _body: object()))
    ]

    def unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("owner process unavailable")

    monkeypatch.setattr(ray, "get", unavailable)
    with pytest.raises(RuntimeError):
        checkpoint_plugin._live_participants(manager)


def test_live_participants_includes_local_owner_without_self_rpc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ray

    manager = _manager(_FakeStore(_FakeCluster({}, {}), "10.0.0.9:12309"))
    local = _participant(manager)
    remote_manager = _manager(
        _FakeStore(_FakeCluster({}, {}), "10.0.0.1:12301"), "remote-owner"
    )
    remote = _participant(remote_manager)
    worker = SimpleNamespace(mooncake_checkpoint=_RemoteMethod(remote._dispatch))
    manager._checkpoint_workers = [worker]
    monkeypatch.setattr(ray, "get", lambda value, **_kwargs: value)
    participants, workers = checkpoint_plugin._live_participants(manager)
    assert set(participants) == {local.info, remote.info}
    assert workers == {remote.info.participant_id: worker}


def test_live_participants_rejects_duplicate_endpoints_that_are_both_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ray

    manager = _manager(_FakeStore(_FakeCluster({}, {}), "10.0.0.9:12309"))
    session = checkpoint_plugin._controller_session(manager)
    participants = [
        checkpoint_plugin._ParticipantInfo(
            participant_id=f"owner-{index}",
            controller_session=session,
            segment_name="10.0.0.1:12301",
            transport_endpoint="10.0.0.1:12301",
        )
        for index in range(2)
    ]
    manager._checkpoint_workers = [
        SimpleNamespace(
            mooncake_checkpoint=_RemoteMethod(
                lambda _body, info=info: {"ok": True, "participant": asdict(info)}
            )
        )
        for info in participants
    ]
    monkeypatch.setattr(ray, "get", lambda value, **_kwargs: value)

    with pytest.raises(RuntimeError, match="duplicate endpoints"):
        checkpoint_plugin._live_participants(manager)


def test_save_fsyncs_the_storage_directory_and_published_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced: list[tuple[Path, set[str]]] = []

    def observe_directory(path: Path) -> None:
        path = Path(path)
        synced.append((path, {entry.name for entry in path.iterdir()}))

    monkeypatch.setattr(
        checkpoint_plugin,
        "_fsync_directory",
        observe_directory,
    )

    checkpoint_dir, _, _, _ = _save_distributed_checkpoint(monkeypatch, tmp_path)
    storage_dir = checkpoint_dir / "mooncake_storage"
    storage_snapshots = [entries for path, entries in synced if path == storage_dir]

    assert [path for path, _ in synced].count(checkpoint_dir) == 1
    assert [path for path, _ in synced].count(storage_dir) == (
        2 * len(_SOURCE_IDENTITIES) + 1
    )
    assert [
        len({name for name in entries if name.endswith(".bin")})
        for entries in storage_snapshots
    ] == [1, 1, 2, 2, 2]
    assert all(
        not any(name.endswith(".partial") for name in entries)
        for entries in storage_snapshots
    )
    assert "manifest.json" not in storage_snapshots[-2]
    assert "manifest.json" in storage_snapshots[-1]


def test_save_does_not_commit_a_manifest_when_directory_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir = _checkpoint_dir(tmp_path)
    storage_dir = checkpoint_dir / "mooncake_storage"
    cluster = _source_cluster()
    managers = _managers(cluster, _SOURCE_IDENTITIES)
    _wire_participants(monkeypatch, [_participant(manager) for manager in managers])

    def fail_storage_sync(path: Path) -> None:
        if Path(path) == storage_dir:
            raise OSError("directory fsync failed")

    monkeypatch.setattr(checkpoint_plugin, "_fsync_directory", fail_storage_sync)

    with pytest.raises(OSError, match="directory fsync failed"):
        _save_storage_checkpoint(managers[0], str(checkpoint_dir))

    assert not (storage_dir / "manifest.json").exists()


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


@pytest.mark.parametrize("restore_count", [2, 3])
def test_owner_distributed_checkpoint_round_trip_uses_current_participants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_count: int,
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
    assert set(manifest) == {"format_version", "storage_layout", "shards"}
    assert manifest["format_version"] == 3
    assert all("index_sha256" not in shard for shard in manifest["shards"])
    assert sorted(
        path.name for path in (checkpoint_dir / "mooncake_storage").iterdir()
    ) == [
        "manifest.json",
        "part-00000.bin",
        "part-00000.bin.index.json",
        "part-00001.bin",
        "part-00001.bin.index.json",
    ]
    assert _saved_payloads(checkpoint_dir) == _payloads()
    assert source_cluster.get_calls == []
    assert all(
        manager.storage_client._store.registrations == [] for manager in source_managers
    )

    offsets: dict[str, int] = {}
    for entry in _manifest_objects(checkpoint_dir):
        value = _payloads()[entry["key"]]
        assert entry == {
            "key": entry["key"],
            "shard": entry["shard"],
            "offset": offsets.get(entry["shard"], 0),
            "size": len(value),
            "saved_owner": source_cluster.owners[entry["key"]][0],
        }
        offsets[entry["shard"]] = entry["offset"] + len(value)

    current_identities = [
        ("current-a", "10.1.0.1:13301"),
        ("current-b", "10.1.0.2:13302"),
        ("current-c", "10.1.0.3:13303"),
    ][:restore_count]
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
    upsert_endpoints = {endpoint for endpoint, _, _ in restored_cluster.upsert_calls}
    assert upsert_endpoints <= current_endpoints
    # Whole shards are indivisible: a third participant has no work for two shards.
    assert len(upsert_endpoints) == min(restore_count, len(manifest["shards"]))
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


@pytest.mark.parametrize("owner_is_participant", [True, False])
def test_restore_accepts_only_current_participant_placement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_is_participant: bool,
) -> None:
    checkpoint_dir, _, _, _ = _save_distributed_checkpoint(monkeypatch, tmp_path)
    endpoint_a = "10.1.0.1:13301"
    endpoint_b = "10.1.0.2:13302"
    restored_cluster = _FakeCluster({}, {})
    managers = _managers(
        restored_cluster,
        [("current-a", endpoint_a), ("current-b", endpoint_b)],
        store_options={
            "current-a": {
                "owner_override": endpoint_b
                if owner_is_participant
                else "10.9.9.9:19999"
            }
        },
    )
    _wire_participants(monkeypatch, [_participant(manager) for manager in managers])

    if not owner_is_participant:
        with pytest.raises(RuntimeError, match="current checkpoint participant"):
            _load_storage_checkpoint(managers[0], str(checkpoint_dir))
    else:
        _load_storage_checkpoint(managers[0], str(checkpoint_dir))
        assert restored_cluster.objects == _payloads()
        assert any(
            endpoint == endpoint_a and restored_cluster.owners[key] == (endpoint_b,)
            for endpoint, key, _ in restored_cluster.upsert_calls
        )


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

    assert all(
        request.body["operation"] == "DISCOVER_SAVE"
        for call in calls
        for request in call
    )
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
        if requests[0].body["operation"] != "SAVE_SHARD":
            return responses
        last_id = requests[-1].participant.participant_id
        if response_mode == "missing_ack":
            responses.pop(last_id)
        elif response_mode == "extra_ack":
            responses["unexpected-participant"] = dict(responses[last_id])
        elif response_mode == "missing_object":
            responses[last_id]["saved_keys"].pop()
        elif response_mode == "duplicate_object":
            keys = responses[last_id]["saved_keys"]
            keys[-1] = keys[0]
        elif response_mode == "wrong_size":
            responses[last_id]["shard"]["payload_bytes"] += 1
        else:
            shard = responses[last_id]["shard"]["shard"]
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


def test_save_write_failure_does_not_publish_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cluster = _source_cluster()
    managers = _managers(cluster, _SOURCE_IDENTITIES)
    participants = [_participant(manager) for manager in managers]
    _wire_participants(monkeypatch, participants)
    checkpoint_dir = _checkpoint_dir(tmp_path)

    def fail_write(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("injected checkpoint write failure")

    monkeypatch.setattr(checkpoint_plugin, "_write_buffer", fail_write)
    with pytest.raises(OSError, match="injected checkpoint write failure"):
        _save_storage_checkpoint(managers[0], str(checkpoint_dir))

    assert all(manager.storage_client._store.registered == {} for manager in managers)
    assert not (checkpoint_dir / "mooncake_storage" / "manifest.json").exists()


@pytest.mark.parametrize("raises", [False, True], ids=["error-code", "exception"])
def test_partial_registration_keeps_the_mapping_alive(
    monkeypatch: pytest.MonkeyPatch,
    quarantined_buffers: list[Any],
    raises: bool,
) -> None:
    store = _FakeStore(_FakeCluster({}, {}), "10.0.0.1:12301")
    buffer = checkpoint_plugin._CheckpointBuffer.allocate(16)

    def partially_register(pointer: int, size: int) -> int:
        store.registered[pointer] = size
        if raises:
            raise RuntimeError("registration failed after the first NIC")
        return -1

    monkeypatch.setattr(store, "register_buffer", partially_register)
    try:
        with pytest.raises(RuntimeError, match="registration failed"):
            with checkpoint_plugin._registered_buffer(
                store, buffer, size=16, label="partial registration"
            ):
                pytest.fail("checkpoint I/O must not run after registration fails")
    finally:
        buffer.close()

    assert quarantined_buffers == [buffer.payload]
    assert not buffer.payload.closed
    assert store.registered == {buffer.pointer: 16}


@pytest.mark.parametrize("raises", [False, True], ids=["error-code", "exception"])
def test_restore_quarantines_a_buffer_when_unregister_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quarantined_buffers: list[Any],
    raises: bool,
) -> None:
    checkpoint_dir, _, _, _ = _save_distributed_checkpoint(monkeypatch, tmp_path)
    cluster = _FakeCluster({}, {})
    managers = _managers(cluster, [("current-a", "10.1.0.1:13301")])
    store = managers[0].storage_client._store
    unregister_calls: list[int] = []

    def unregister_buffer(pointer: int) -> int:
        unregister_calls.append(pointer)
        if raises:
            raise RuntimeError("native unregister failed")
        return -1

    monkeypatch.setattr(store, "unregister_buffer", unregister_buffer)
    participants = [_participant(manager) for manager in managers]
    _wire_participants(monkeypatch, participants)
    with pytest.raises(RuntimeError, match="buffer cleanup failed"):
        _load_storage_checkpoint(managers[0], str(checkpoint_dir))

    assert len(quarantined_buffers) == 1
    assert quarantined_buffers[0].closed is False
    pointer, size = next(iter(store.registered.items()))
    assert unregister_calls == [pointer]
    first = _payloads()["0@router_indices"]
    assert ctypes.string_at(pointer, len(first)) == first


@pytest.mark.parametrize("manifest", [None, []])
def test_load_manifest_rejects_a_non_mapping(
    tmp_path: Path,
    manifest: Any,
) -> None:
    storage_dir = tmp_path / "mooncake_storage"
    storage_dir.mkdir()
    (storage_dir / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="manifest must contain a mapping"):
        checkpoint_plugin._load_manifest(tmp_path)


@pytest.mark.parametrize("location", ["manifest", "index"])
@pytest.mark.parametrize(
    "shard_name",
    [
        "../part-00000.bin",
        "/tmp/part-00000.bin",
        "part-subdir/part-00000.bin",
        "",
        None,
        123,
        "checkpoint.bin",
        "part-00000.tmp",
    ],
)
def test_restore_rejects_invalid_shard_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shard_name: Any,
    location: str,
) -> None:
    checkpoint_dir, _, _, _ = _save_distributed_checkpoint(monkeypatch, tmp_path)
    storage_dir = checkpoint_dir / "mooncake_storage"
    manifest = _manifest(checkpoint_dir)
    if location == "manifest":
        manifest["shards"][0]["shard"] = shard_name
        (storage_dir / "manifest.json").write_text(json.dumps(manifest))
    else:
        index_path = storage_dir / f"{manifest['shards'][0]['shard']}.index.json"
        index = json.loads(index_path.read_text())
        index["objects"][0]["shard"] = shard_name
        index_path.write_text(json.dumps(index))
    cluster = _FakeCluster({}, {})
    manager = _manager(_FakeStore(cluster, "10.1.0.1:13301"))
    _wire_participants(monkeypatch, [_participant(manager)])

    with pytest.raises(ValueError, match="Invalid Mooncake checkpoint shard name"):
        _load_storage_checkpoint(manager, str(checkpoint_dir))

    assert cluster.upsert_calls == []


@pytest.mark.parametrize("damage", ["missing", "truncated", "extended"])
def test_restore_rejects_a_missing_or_wrong_size_owner_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    checkpoint_dir, _, _, _ = _save_distributed_checkpoint(monkeypatch, tmp_path)
    first = _manifest_objects(checkpoint_dir)[0]
    path = checkpoint_dir / "mooncake_storage" / first["shard"]
    if damage == "missing":
        path.unlink()
    elif damage == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    else:
        path.write_bytes(path.read_bytes() + b"extra")
    cluster = _FakeCluster({}, {})
    managers = _managers(cluster, [("current-a", "10.1.0.1:13301")])
    _wire_participants(monkeypatch, [_participant(managers[0])])

    with pytest.raises(
        ValueError, match="Missing or corrupt Mooncake checkpoint shard"
    ):
        _load_storage_checkpoint(managers[0], str(checkpoint_dir))

    assert cluster.upsert_calls == []


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


def test_restore_checks_clean_store_inside_its_single_load_round(
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
    calls = _wire_participants(monkeypatch, [_participant(restore_manager)])

    with pytest.raises(
        RuntimeError,
        match="requires a clean store: key '0@router_indices' already exists",
    ):
        _load_storage_checkpoint(restore_manager, str(checkpoint_dir))

    assert len(calls) == 1
    assert calls[0][0].body["operation"] == "LOAD_SHARDS"
    assert cluster.upsert_calls == []


@pytest.mark.parametrize("error_code", [-600, -900])
def test_restore_reports_existence_query_errors_without_writes_or_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_code: int,
) -> None:
    checkpoint_dir, _, _, _ = _save_distributed_checkpoint(monkeypatch, tmp_path)
    cluster = _FakeCluster({}, {})
    store = _FakeStore(cluster, "10.1.0.1:13301")
    manager = _manager(store)
    _wire_participants(monkeypatch, [_participant(manager)])
    queries: list[list[str]] = []

    def failed_query(keys: list[str]) -> list[int]:
        queries.append(list(keys))
        return [error_code, *([0] * (len(keys) - 1))]

    monkeypatch.setattr(store, "batch_is_exist", failed_query)
    with pytest.raises(RuntimeError, match="Mooncake existence check failed") as error:
        _load_storage_checkpoint(manager, str(checkpoint_dir))

    assert len(queries) == 1
    assert str(error.value) == (
        f"Mooncake existence check failed for key {queries[0][0]!r}: "
        f"error code {error_code}"
    )
    assert cluster.objects == {}
    assert cluster.upsert_calls == []
    assert store.registrations == []


@pytest.mark.parametrize(
    "response",
    [
        None,
        "00",
        {"a": 0, "b": 0},
        [],
        [0],
        [0, 0, 0],
        [False, 0],
        [0, "0"],
        [2, 0],
        [0, 1.0],
    ],
)
def test_clean_store_rejects_malformed_existence_responses(
    monkeypatch: pytest.MonkeyPatch,
    response: Any,
) -> None:
    store = _FakeStore(_FakeCluster({}, {}), "10.1.0.1:13301")
    queries: list[list[str]] = []

    def malformed_query(keys: list[str]) -> Any:
        queries.append(list(keys))
        return response

    monkeypatch.setattr(store, "batch_is_exist", malformed_query)
    with pytest.raises(RuntimeError, match="Malformed Mooncake existence response"):
        checkpoint_plugin._require_clean_store(store, ["a", "b"])

    assert queries == [["a", "b"]]


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


def test_plugin_install_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    from transfer_queue.storage.managers import mooncake_manager
    from transfer_queue.storage.managers.base import StorageManagerFactory

    monkeypatch.setitem(
        StorageManagerFactory._registry,
        "MooncakeStore",
        mooncake_manager.MooncakeStorageManager,
    )
    install_tq_mooncake_checkpoint_plugin()
    installed_manager = StorageManagerFactory._registry["MooncakeStore"]

    install_tq_mooncake_checkpoint_plugin()

    assert StorageManagerFactory._registry["MooncakeStore"] is installed_manager
    assert issubclass(installed_manager, checkpoint_plugin._CheckpointManagerMixin)
    assert issubclass(installed_manager, mooncake_manager.MooncakeStorageManager)


@pytest.mark.parametrize("registration", [None, object(), SimpleNamespace])
def test_plugin_install_rejects_unexpected_registration(
    monkeypatch: pytest.MonkeyPatch, registration: Any
) -> None:
    from transfer_queue.storage.managers.base import StorageManagerFactory

    monkeypatch.setitem(StorageManagerFactory._registry, "MooncakeStore", registration)

    with pytest.raises(RuntimeError, match="Unexpected TQ MooncakeStore"):
        install_tq_mooncake_checkpoint_plugin()

    assert StorageManagerFactory._registry["MooncakeStore"] is registration


def test_configure_and_command_reuse_the_existing_process_local_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transfer_queue as tq
    from transfer_queue import interface as tq_interface

    manager = object.__new__(checkpoint_plugin._CheckpointManagerMixin)
    manager.__dict__.update(
        vars(_manager(_FakeStore(_FakeCluster({}, {}), "10.0.0.1:12301")))
    )
    participant = _participant(manager)
    client = SimpleNamespace(storage_manager=manager)
    monkeypatch.setattr(tq, "get_client", lambda: client)
    monkeypatch.setattr(tq_interface, "_TQ_CLIENT", client)

    def unexpected_init(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("checkpoint RPC created another TQ client")

    monkeypatch.setattr(tq, "init", unexpected_init)
    workers = [object(), object()]
    checkpoint_plugin.configure_checkpoint_workers(workers)
    assert list(manager._checkpoint_workers) == workers
    response = checkpoint_plugin.run_checkpoint_command(
        checkpoint_plugin._request_body(manager, "DESCRIBE")
    )
    assert response["participant"] == asdict(participant.info)


def test_configure_rejects_and_command_ignores_uninstalled_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transfer_queue as tq
    from transfer_queue import interface as tq_interface

    client = SimpleNamespace(storage_manager=SimpleNamespace())
    monkeypatch.setattr(tq, "get_client", lambda: client)
    monkeypatch.setattr(tq_interface, "_TQ_CLIENT", client)

    with pytest.raises(RuntimeError, match="checkpoint manager is not installed"):
        checkpoint_plugin.configure_checkpoint_workers([])
    assert checkpoint_plugin.run_checkpoint_command({"operation": "DESCRIBE"}) is None


def test_command_does_not_initialize_a_client_on_non_owner_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transfer_queue as tq
    from transfer_queue import interface as tq_interface

    monkeypatch.setattr(tq_interface, "_TQ_CLIENT", None)

    def unexpected_client() -> None:
        pytest.fail("checkpoint command initialized a client on a non-owner rank")

    monkeypatch.setattr(tq, "get_client", unexpected_client)
    assert checkpoint_plugin.run_checkpoint_command({"operation": "DESCRIBE"}) is None


@pytest.mark.parametrize(
    ("actor_id", "enabled", "expected_capacity", "owns_segment"),
    [
        (None, True, 0, False),
        ("actor-id", True, 1024, True),
        (None, False, 1024, False),
    ],
)
def test_installed_manager_keeps_non_actor_clients_out_of_the_storage_topology(
    monkeypatch: pytest.MonkeyPatch,
    actor_id: str | None,
    enabled: bool,
    expected_capacity: int,
    owns_segment: bool,
) -> None:
    import ray
    from transfer_queue.storage.managers import mooncake_manager
    from transfer_queue.storage.managers.base import StorageManagerFactory

    endpoint = "10.3.0.7:14321"
    cluster = _FakeCluster({}, {})
    store = _FakeStore(cluster, endpoint)
    manager_closes: list[str] = []
    config = _manager(store).config
    config["global_segment_size"] = 1024
    config["checkpoint"]["enabled"] = enabled

    def base_init(self: Any, controller_info: Any, config: dict[str, Any]) -> None:
        fake = _manager(store, "manager-live")
        self.config = config
        self.storage_client = fake.storage_client
        self.storage_client.global_segment_size = config["global_segment_size"]
        self.storage_manager_id = fake.storage_manager_id
        self.controller_info = controller_info
        self.controller_handshake_socket = None
        self.zmq_context = SimpleNamespace(term=lambda: None)

    monkeypatch.setattr(mooncake_manager.MooncakeStorageManager, "__init__", base_init)
    monkeypatch.setattr(
        mooncake_manager.MooncakeStorageManager,
        "close",
        lambda _self: manager_closes.append("manager"),
    )
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_actor_id=lambda: actor_id),
    )
    monkeypatch.setitem(
        StorageManagerFactory._registry,
        "MooncakeStore",
        mooncake_manager.MooncakeStorageManager,
    )

    install_tq_mooncake_checkpoint_plugin()
    manager_type = StorageManagerFactory._registry["MooncakeStore"]
    manager = manager_type(_manager(store).controller_info, config)
    participant = manager._checkpoint_participant
    assert (participant is not None) is owns_segment
    assert manager.config["global_segment_size"] == expected_capacity
    assert config["global_segment_size"] == 1024
    if enabled:
        assert manager.config is not config
    if participant is not None:
        assert asdict(participant.info) == {
            "participant_id": "manager-live",
            "controller_session": checkpoint_plugin._controller_session(manager),
            "segment_name": endpoint,
            "transport_endpoint": endpoint,
        }

    manager.close()
    assert manager_closes == ["manager"]


@pytest.mark.parametrize("enabled", [False, True])
def test_installed_manager_dispatches_only_enabled_checkpoint_operations(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
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
    manager.config = {"checkpoint": {"enabled": enabled}}
    manager.storage_manager_id = "test-manager"
    manager.controller_handshake_socket = None
    manager.zmq_context = SimpleNamespace(term=lambda: None)
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

    async def base_save(_self: Any, path: str) -> None:
        calls.append(("base-save", path))

    async def base_load(_self: Any, path: str) -> None:
        calls.append(("base-load", path))

    monkeypatch.setattr(
        mooncake_manager.MooncakeStorageManager, "save_checkpoint", base_save
    )
    monkeypatch.setattr(
        mooncake_manager.MooncakeStorageManager, "load_checkpoint", base_load
    )

    asyncio.run(manager.save_checkpoint("/checkpoint-save"))
    asyncio.run(manager.load_checkpoint("/checkpoint-load"))

    prefix = "" if enabled else "base-"
    assert calls == [
        (f"{prefix}save", "/checkpoint-save"),
        (f"{prefix}load", "/checkpoint-load"),
    ]


def test_owner_local_save_avoids_staging_and_load_reuses_one_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checkpoint_plugin, "_BATCH_KEYS", 3)
    monkeypatch.setattr(checkpoint_plugin, "_BATCH_BYTES", 512)
    sizes = [4, 8, 4, 8, 513, 4, 8, 4, 8]
    payloads = {
        f"{index}@value": bytes([index + 1]) * size for index, size in enumerate(sizes)
    }
    endpoint = "127.0.0.1:12301"
    cluster = _FakeCluster(payloads, {key: (endpoint,) for key in payloads})
    store = _FakeStore(cluster, endpoint)
    manager = _manager(store)
    _wire_participants(monkeypatch, [_participant(manager)])
    monkeypatch.setattr(
        checkpoint_plugin, "_controller_keys", lambda _path: sorted(payloads)
    )
    checkpoint_dir = _checkpoint_dir(tmp_path)
    _save_storage_checkpoint(manager, str(checkpoint_dir))
    batches = list(checkpoint_plugin._checkpoint_batches(sizes))
    assert store.get_batches == []
    assert len(batches) > 1
    assert all(len(batch) <= 3 for batch in store.replica_batches)
    assert len(store.replica_batches) == 3
    assert [key for batch in store.replica_batches for key in batch] == sorted(payloads)
    assert store.registrations == []
    assert store.registered == {}
    assert _saved_payloads(checkpoint_dir) == payloads
    assert (
        checkpoint_dir / "mooncake_storage" / "part-00000.bin"
    ).stat().st_size == sum(sizes)
    cluster.objects.clear()
    cluster.owners.clear()
    _load_storage_checkpoint(manager, str(checkpoint_dir))
    assert cluster.objects == payloads
    assert len(store.upsert_batches) == len(batches)
    assert len(store.registrations) == 1
    assert store.registrations[0][1] == 513
    assert store.registered == {}


def test_native_batch_key_limit_and_scalar_alignment() -> None:
    batches = list(checkpoint_plugin._checkpoint_batches([4] * 401))
    assert [(start, stop) for start, stop, _ in batches] == [(0, 400), (400, 401)]
    assert batches[0][2] == list(range(0, 400 * 256, 256))
    assert batches[1][2] == [0]


@pytest.mark.parametrize(
    "bad_result", [None, 0, [], [0], [0, 0, 0], [True, 0], [-1, 0], [4, 7], [0.0, 0]]
)
def test_native_batch_results_require_exact_cardinality_and_values(
    bad_result: Any,
) -> None:
    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        checkpoint_plugin._check_batch_results(bad_result, [0, 0], operation="restore")


def test_partial_native_batch_does_not_ack_and_unregisters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir, cluster, managers, _ = _save_distributed_checkpoint(
        monkeypatch, tmp_path
    )
    cluster.objects.clear()
    cluster.owners.clear()
    store = managers[0].storage_client._store
    original = store.batch_upsert_from
    monkeypatch.setattr(store, "batch_upsert_from", lambda *args: original(*args)[:-1])
    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        _load_storage_checkpoint(managers[0], str(checkpoint_dir))
    assert all(manager.storage_client._store.registered == {} for manager in managers)


@pytest.mark.parametrize("sizes", [[0], [-1], [True], [4, 8]])
def test_save_rejects_invalid_or_disagreeing_replica_sizes(
    monkeypatch: pytest.MonkeyPatch, sizes: list[Any]
) -> None:
    endpoint = "127.0.0.1:12301"
    store = _FakeStore(_FakeCluster({}, {}), endpoint)
    participant = _participant(_manager(store))
    monkeypatch.setattr(
        store,
        "batch_get_replica_desc",
        lambda _keys: {"key": [_FakeMemoryReplica(endpoint, size) for size in sizes]},
    )
    with pytest.raises(RuntimeError, match="invalid or inconsistent"):
        participant._dispatch(
            checkpoint_plugin._request_body(
                participant._manager,
                "DISCOVER_SAVE",
                keys=["key"],
                current_endpoints=[endpoint],
            )
        )


@pytest.mark.parametrize("result", [None, {}, {"unexpected": []}])
def test_replica_batch_requires_exact_key_coverage(
    monkeypatch: pytest.MonkeyPatch, result: Any
) -> None:
    store = _FakeStore(_FakeCluster({}, {}), "127.0.0.1:12301")
    monkeypatch.setattr(store, "batch_get_replica_desc", lambda _keys: result)
    with pytest.raises(RuntimeError, match="malformed data"):
        checkpoint_plugin._batch_replicas(store, ["key"])


@pytest.mark.parametrize(
    "mode",
    [
        "foreign-owner",
        "same-host-peer",
        "foreign-session",
        "zero-address",
        "bool-address",
    ],
)
def test_save_rejects_foreign_or_invalid_owner_memory_before_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    manager = _manager(_FakeStore(_source_cluster(), "10.0.0.1:12301"))
    participant = _participant(manager)
    group = checkpoint_plugin._SaveGroup(
        "foreign" if mode == "foreign-session" else participant.info.controller_session,
        "foreign"
        if mode == "foreign-owner"
        else "10.0.0.1:12302"
        if mode == "same-host-peer"
        else participant.info.transport_endpoint,
        [
            checkpoint_plugin._StoredObject(
                "key",
                4,
                True if mode == "bool-address" else 0 if mode == "zero-address" else 1,
            )
        ],
    )

    def unexpected_write(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("invalid owner memory reached file writing")

    monkeypatch.setattr(checkpoint_plugin, "_write_buffer", unexpected_write)
    (tmp_path / "mooncake_storage").mkdir()
    with pytest.raises(ValueError, match="different owner/session|memory descriptor"):
        participant._dispatch(
            checkpoint_plugin._request_body(
                manager,
                "SAVE_SHARD",
                checkpoint_root=str(tmp_path),
                shard_name="part-00000.bin",
                object_groups=[_FakeObjectRef(group)],
            )
        )


@pytest.mark.parametrize(
    "format_fields",
    [
        {"objects": []},
        {"format_version": 2, "shards": []},
        {"format_version": 1, "shards": []},
        {"format_version": 4, "shards": []},
        {"format_version": "3", "shards": []},
        {"format_version": 3.0, "shards": []},
        {"format_version": True, "shards": []},
        {"format_version": 3, "shards": [], "objects": []},
    ],
)
def test_load_rejects_legacy_or_unsupported_manifest_formats(
    tmp_path: Path, format_fields: dict[str, Any]
) -> None:
    storage_dir = tmp_path / "mooncake_storage"
    storage_dir.mkdir()
    manifest = {
        "storage_layout": {"use_gdr": False, "gdr_staging_buffer_mb": 1024},
        **format_fields,
    }
    (storage_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported.*format version"):
        checkpoint_plugin._load_manifest(tmp_path)


def test_batch_failure_preserves_primary_error_and_quarantines_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quarantined_buffers: list[Any]
) -> None:
    checkpoint_dir, _, _, _ = _save_distributed_checkpoint(monkeypatch, tmp_path)
    cluster = _FakeCluster({}, {})
    managers = _managers(
        cluster,
        [("current-a", "10.1.0.1:13301")],
        store_options={"current-a": {"unregister_result": -1}},
    )
    monkeypatch.setattr(
        managers[0].storage_client._store, "batch_upsert_from", lambda *_args: []
    )
    _wire_participants(monkeypatch, [_participant(manager) for manager in managers])
    with pytest.raises(RuntimeError, match="restore returned incomplete") as error:
        _load_storage_checkpoint(managers[0], str(checkpoint_dir))
    assert any("buffer cleanup failed" in note for note in error.value.__notes__)
    assert len(quarantined_buffers) == 1
    assert not quarantined_buffers[0].closed


@pytest.mark.parametrize(
    ("status_name", "is_memory"), [("PROCESSING", True), ("COMPLETE", False)]
)
def test_save_rejects_incomplete_or_nonmemory_replicas(
    monkeypatch: pytest.MonkeyPatch, status_name: str, is_memory: bool
) -> None:
    endpoint = "127.0.0.1:12301"
    store = _FakeStore(_FakeCluster({}, {}), endpoint)
    participant = _participant(_manager(store))
    replica = _FakeMemoryReplica(endpoint, 4)
    replica.status.name = status_name
    monkeypatch.setattr(replica, "is_memory_replica", lambda: is_memory)
    monkeypatch.setattr(
        replica,
        "get_memory_descriptor",
        lambda: pytest.fail("ineligible replica reached memory descriptor access"),
    )
    monkeypatch.setattr(
        store, "batch_get_replica_desc", lambda _keys: {"key": [replica]}
    )

    with pytest.raises(RuntimeError, match="no COMPLETE memory replica"):
        participant._dispatch(
            checkpoint_plugin._request_body(
                participant._manager,
                "DISCOVER_SAVE",
                keys=["key"],
                current_endpoints=[endpoint],
            )
        )
