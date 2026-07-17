import os

import pytest
from transfer_queue.client import AsyncTransferQueueClient
from transfer_queue.controller import PartitionIndexManager
from transfer_queue.storage.clients import mooncake_client
from transfer_queue.storage.clients.mooncake_client import MooncakeStoreClient

import nemo_rl.data_plane.adapters.transfer_queue as tq_adapter
from nemo_rl.data_plane.adapters.transfer_queue import (
    _MOONCAKE_TQ_ACTOR_SETUP_HOOK,
    _configure_mooncake_client_limits,
    _disable_mooncake_index_reuse,
    _patch_mooncake_clear_semantics,
)


@pytest.fixture(autouse=True)
def reset_mooncake_client_limit_state(monkeypatch):
    original_batch_limit = mooncake_client.BATCH_SIZE_LIMIT
    original_threads = mooncake_client.MAX_WORKER_THREADS
    original_defaults = tq_adapter._MOONCAKE_CLIENT_DEFAULTS
    monkeypatch.setattr(tq_adapter, "_MOONCAKE_CLIENT_DEFAULTS", None)
    yield
    monkeypatch.setattr(mooncake_client, "BATCH_SIZE_LIMIT", original_batch_limit)
    monkeypatch.setattr(mooncake_client, "MAX_WORKER_THREADS", original_threads)
    monkeypatch.setattr(
        tq_adapter,
        "_MOONCAKE_CLIENT_DEFAULTS",
        original_defaults,
    )


def test_mooncake_client_limits_from_config(monkeypatch):
    original_batch_limit = mooncake_client.BATCH_SIZE_LIMIT
    original_threads = mooncake_client.MAX_WORKER_THREADS
    try:
        cfg = {
            "backend": "mooncake_cpu",
            "mooncake_batch_size_limit": 16,
            "mooncake_max_worker_threads": 1,
        }

        _configure_mooncake_client_limits(cfg)  # type: ignore[arg-type]

        assert mooncake_client.BATCH_SIZE_LIMIT == 16
        assert mooncake_client.MAX_WORKER_THREADS == 1
    finally:
        monkeypatch.setattr(mooncake_client, "BATCH_SIZE_LIMIT", original_batch_limit)
        monkeypatch.setattr(mooncake_client, "MAX_WORKER_THREADS", original_threads)


def test_mooncake_client_limits_from_env(monkeypatch):
    original_batch_limit = mooncake_client.BATCH_SIZE_LIMIT
    original_threads = mooncake_client.MAX_WORKER_THREADS
    try:
        monkeypatch.setenv("NRL_TQ_MOONCAKE_BATCH_SIZE_LIMIT", "8")
        monkeypatch.setenv("NRL_TQ_MOONCAKE_MAX_WORKER_THREADS", "2")

        _configure_mooncake_client_limits({"backend": "mooncake_cpu"})  # type: ignore[arg-type]

        assert mooncake_client.BATCH_SIZE_LIMIT == 8
        assert mooncake_client.MAX_WORKER_THREADS == 2
    finally:
        monkeypatch.setattr(mooncake_client, "BATCH_SIZE_LIMIT", original_batch_limit)
        monkeypatch.setattr(mooncake_client, "MAX_WORKER_THREADS", original_threads)


def test_mooncake_client_limits_ignore_non_mooncake(monkeypatch):
    original_batch_limit = mooncake_client.BATCH_SIZE_LIMIT
    try:
        monkeypatch.setenv("NRL_TQ_MOONCAKE_BATCH_SIZE_LIMIT", "8")

        _configure_mooncake_client_limits({"backend": "simple"})  # type: ignore[arg-type]

        assert mooncake_client.BATCH_SIZE_LIMIT == original_batch_limit
    finally:
        monkeypatch.setattr(mooncake_client, "BATCH_SIZE_LIMIT", original_batch_limit)


def test_mooncake_client_limits_restore_defaults(monkeypatch):
    original_batch_limit = mooncake_client.BATCH_SIZE_LIMIT
    original_threads = mooncake_client.MAX_WORKER_THREADS
    try:
        _configure_mooncake_client_limits(
            {
                "backend": "mooncake_cpu",
                "mooncake_batch_size_limit": 16,
                "mooncake_max_worker_threads": 1,
            }  # type: ignore[arg-type]
        )
        _configure_mooncake_client_limits({"backend": "mooncake_cpu"})  # type: ignore[arg-type]

        assert mooncake_client.BATCH_SIZE_LIMIT == original_batch_limit
        assert mooncake_client.MAX_WORKER_THREADS == original_threads
    finally:
        monkeypatch.setattr(mooncake_client, "BATCH_SIZE_LIMIT", original_batch_limit)
        monkeypatch.setattr(mooncake_client, "MAX_WORKER_THREADS", original_threads)


def test_mooncake_client_limits_reject_invalid():
    cfg = {
        "backend": "mooncake_cpu",
        "mooncake_batch_size_limit": 0,
    }

    try:
        _configure_mooncake_client_limits(cfg)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "mooncake_batch_size_limit" in str(exc)
    else:
        raise AssertionError("expected invalid mooncake_batch_size_limit to fail")


def test_mooncake_actor_setup_hook_runtime_env_serializes():
    from ray.runtime_env import RuntimeEnv

    env = RuntimeEnv(worker_process_setup_hook=_MOONCAKE_TQ_ACTOR_SETUP_HOOK)

    assert "worker_process_setup_hook" in env.serialize()


def test_mooncake_actor_runtime_env_upgrades_after_simple_patch(monkeypatch):
    from transfer_queue.controller import TransferQueueController
    from transfer_queue.storage.simple_backend import SimpleStorageUnit

    calls: list[dict] = []

    def options(*args, **kwargs):
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(tq_adapter, "_TQ_RUNTIME_ENV_PATCH_LEVEL", 0)
    monkeypatch.setattr(tq_adapter, "_resolve_tq_pin", lambda: "TransferQueue==test")
    monkeypatch.setenv("PYTHONPATH", "/tmp/current-pythonpath")
    monkeypatch.setattr(SimpleStorageUnit, "options", options)
    monkeypatch.setattr(TransferQueueController, "options", options)

    tq_adapter._patch_tq_actor_runtime_env(patch_mooncake_semantics=False)
    SimpleStorageUnit.options()
    assert calls[-1]["runtime_env"] == {"pip": ["TransferQueue==test"]}

    tq_adapter._patch_tq_actor_runtime_env(patch_mooncake_semantics=True)
    TransferQueueController.options()
    runtime_env = calls[-1]["runtime_env"]

    assert runtime_env["pip"] == ["TransferQueue==test"]
    assert runtime_env["worker_process_setup_hook"] == _MOONCAKE_TQ_ACTOR_SETUP_HOOK
    pythonpath = runtime_env["env_vars"]["PYTHONPATH"].split(os.pathsep)
    assert pythonpath[0] == "/opt/nemo-rl"
    assert "/tmp/current-pythonpath" in pythonpath


def test_mooncake_index_reuse_is_disabled(monkeypatch):
    original_allocate = PartitionIndexManager.allocate_indexes
    original_release_partition = PartitionIndexManager.release_partition
    original_release_indexes = PartitionIndexManager.release_indexes
    original_marker = getattr(PartitionIndexManager, "_nemo_mooncake_no_reuse", None)
    original_global = tq_adapter._MOONCAKE_INDEX_REUSE_PATCHED

    try:
        monkeypatch.delattr(
            PartitionIndexManager,
            "_nemo_mooncake_no_reuse",
            raising=False,
        )
        tq_adapter._MOONCAKE_INDEX_REUSE_PATCHED = False

        _disable_mooncake_index_reuse()

        manager = PartitionIndexManager()
        assert manager.allocate_indexes("train", 3) == [0, 1, 2]
        manager.release_indexes("train", [0, 1, 2])
        assert manager.reusable_indexes == []
        assert manager.allocate_indexes("train", 2) == [3, 4]
    finally:
        monkeypatch.setattr(
            PartitionIndexManager,
            "allocate_indexes",
            original_allocate,
        )
        monkeypatch.setattr(
            PartitionIndexManager,
            "release_partition",
            original_release_partition,
        )
        monkeypatch.setattr(
            PartitionIndexManager,
            "release_indexes",
            original_release_indexes,
        )
        if original_marker is None:
            monkeypatch.delattr(
                PartitionIndexManager,
                "_nemo_mooncake_no_reuse",
                raising=False,
            )
        else:
            monkeypatch.setattr(
                PartitionIndexManager,
                "_nemo_mooncake_no_reuse",
                original_marker,
            )
        tq_adapter._MOONCAKE_INDEX_REUSE_PATCHED = original_global


def test_mooncake_clear_raises_on_remove_failure(monkeypatch):
    original_clear = MooncakeStoreClient.clear
    original_marker = getattr(MooncakeStoreClient, "_nemo_raise_on_clear_fail", None)
    original_clear_partition = AsyncTransferQueueClient.async_clear_partition
    original_clear_samples = AsyncTransferQueueClient.async_clear_samples
    original_clear_marker = getattr(
        AsyncTransferQueueClient,
        "_nemo_storage_first_clear",
        None,
    )
    original_global = tq_adapter._MOONCAKE_CLEAR_SEMANTICS_PATCHED

    class Store:
        def batch_remove(self, keys, force=True):
            assert force is True
            return [0, -800]

    try:
        monkeypatch.delattr(
            MooncakeStoreClient,
            "_nemo_raise_on_clear_fail",
            raising=False,
        )
        tq_adapter._MOONCAKE_CLEAR_SEMANTICS_PATCHED = False

        _patch_mooncake_clear_semantics()

        client = object.__new__(MooncakeStoreClient)
        client._store = Store()
        try:
            client.clear(["ok", "bad"])
        except RuntimeError as exc:
            assert "bad:-800" in str(exc)
        else:
            raise AssertionError("expected failed Mooncake remove to raise")
    finally:
        monkeypatch.setattr(MooncakeStoreClient, "clear", original_clear)
        monkeypatch.setattr(
            AsyncTransferQueueClient,
            "async_clear_partition",
            original_clear_partition,
        )
        monkeypatch.setattr(
            AsyncTransferQueueClient,
            "async_clear_samples",
            original_clear_samples,
        )
        if original_marker is None:
            monkeypatch.delattr(
                MooncakeStoreClient,
                "_nemo_raise_on_clear_fail",
                raising=False,
            )
        else:
            monkeypatch.setattr(
                MooncakeStoreClient,
                "_nemo_raise_on_clear_fail",
                original_marker,
            )
        if original_clear_marker is None:
            monkeypatch.delattr(
                AsyncTransferQueueClient,
                "_nemo_storage_first_clear",
                raising=False,
            )
        else:
            monkeypatch.setattr(
                AsyncTransferQueueClient,
                "_nemo_storage_first_clear",
                original_clear_marker,
            )
        tq_adapter._MOONCAKE_CLEAR_SEMANTICS_PATCHED = original_global


def test_mooncake_clear_samples_clears_storage_before_controller(monkeypatch):
    original_clear = MooncakeStoreClient.clear
    original_clear_marker = getattr(
        MooncakeStoreClient, "_nemo_raise_on_clear_fail", None
    )
    original_clear_partition = AsyncTransferQueueClient.async_clear_partition
    original_clear_samples = AsyncTransferQueueClient.async_clear_samples
    original_marker = getattr(
        AsyncTransferQueueClient, "_nemo_storage_first_clear", None
    )
    original_global = tq_adapter._MOONCAKE_CLEAR_SEMANTICS_PATCHED
    calls: list[str] = []

    class StorageManager:
        async def clear_data(self, metadata):
            calls.append("storage")

    class Metadata:
        size = 1

    async def clear_meta(self, metadata):
        calls.append("controller")

    try:
        monkeypatch.delattr(
            AsyncTransferQueueClient,
            "_nemo_storage_first_clear",
            raising=False,
        )
        tq_adapter._MOONCAKE_CLEAR_SEMANTICS_PATCHED = False

        _patch_mooncake_clear_semantics()

        client = object.__new__(AsyncTransferQueueClient)
        client.client_id = "test"
        client.storage_manager = StorageManager()
        client._controller = object()
        client._clear_meta_in_controller = clear_meta.__get__(client)

        import asyncio

        asyncio.run(client.async_clear_samples(Metadata()))

        assert calls == ["storage", "controller"]
    finally:
        monkeypatch.setattr(MooncakeStoreClient, "clear", original_clear)
        monkeypatch.setattr(
            AsyncTransferQueueClient,
            "async_clear_partition",
            original_clear_partition,
        )
        monkeypatch.setattr(
            AsyncTransferQueueClient,
            "async_clear_samples",
            original_clear_samples,
        )
        if original_clear_marker is None:
            monkeypatch.delattr(
                MooncakeStoreClient,
                "_nemo_raise_on_clear_fail",
                raising=False,
            )
        else:
            monkeypatch.setattr(
                MooncakeStoreClient,
                "_nemo_raise_on_clear_fail",
                original_clear_marker,
            )
        if original_marker is None:
            monkeypatch.delattr(
                AsyncTransferQueueClient,
                "_nemo_storage_first_clear",
                raising=False,
            )
        else:
            monkeypatch.setattr(
                AsyncTransferQueueClient,
                "_nemo_storage_first_clear",
                original_marker,
            )
        tq_adapter._MOONCAKE_CLEAR_SEMANTICS_PATCHED = original_global
