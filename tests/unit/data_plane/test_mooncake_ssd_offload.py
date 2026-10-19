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
"""Mooncake SSD offload: master flags, segment owners, per-process directory."""

from __future__ import annotations

import atexit
import os
from types import SimpleNamespace

import pytest

from nemo_rl.data_plane.adapters import mooncake_ssd_offload as offload
from nemo_rl.data_plane.interfaces import MooncakeCpuConfig

GLOBAL = 1 << 30
UNIT = 1 << 28
_SSD = {"enabled": True, "path": "/local/ssd"}


def _cfg(**mooncake_cpu) -> MooncakeCpuConfig:
    return MooncakeCpuConfig(global_segment_size=GLOBAL, **mooncake_cpu)


def test_master_args_are_empty_when_offload_is_off() -> None:
    assert offload.master_args(_cfg()) == ()


def test_master_args_turn_eviction_into_offload() -> None:
    cfg = _cfg(ssd_offload={**_SSD, "offload_threshold": 0.2})
    assert offload.master_args(cfg) == (
        "--enable_offload=true",
        "--offload_on_evict=true",
        "--offload_force_evict=false",
        "--eviction_high_watermark_ratio=0.2",
        "--eviction_ratio=0.1",
        "--default_kv_lease_ttl=1000",
    )


def test_offload_starts_at_half_full_by_default() -> None:
    # At 0.9 the headroom above the watermark is less than one large put.
    assert _cfg(ssd_offload=_SSD).ssd_offload.offload_threshold == 0.5


@pytest.mark.parametrize(
    ("unit_segment_size", "segment_size", "owned"),
    [
        (0, None, GLOBAL),  # driver or worker: the controller's segment
        (0, 0, 0),  # generation worker: asks for none
        (UNIT, None, 0),  # with units, every client owns nothing
        (UNIT, UNIT, UNIT),  # the storage unit
    ],
)
def test_owned_segment_size(unit_segment_size, segment_size, owned) -> None:
    cfg = _cfg(storage_unit_segment_size=unit_segment_size)
    assert cfg.owned_segment_size(segment_size) == owned


@pytest.fixture
def fake_store(monkeypatch, tmp_path):
    """A fake mooncake store class, fresh offload state, and a clean env."""
    import ray
    from transfer_queue.storage.clients import mooncake_client

    calls = []

    class FakeStore:
        def setup(self, *args, **kwargs):
            calls.append((args, kwargs))
            return 0

    FakeClient = _client_cls([])
    monkeypatch.setattr(mooncake_client, "MooncakeDistributedStore", FakeStore)
    monkeypatch.setattr(mooncake_client, "MooncakeStoreClient", FakeClient)
    monkeypatch.setattr(
        ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_job_id=lambda: "05000000"),
    )
    for var in offload.ENV_DEFAULTS:
        # setenv first so monkeypatch restores the var even if it was unset.
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    monkeypatch.setattr(offload, "_dir", None)
    exit_hooks = []
    monkeypatch.setattr(atexit, "register", exit_hooks.append)
    return SimpleNamespace(
        module=mooncake_client,
        cls=FakeStore,
        client_cls=FakeClient,
        calls=calls,
        exit_hooks=exit_hooks,
    )


@pytest.mark.parametrize(
    ("mooncake_cpu", "segment_size"),
    [
        ({}, None),  # offload off
        ({"ssd_offload": _SSD, "storage_unit_segment_size": UNIT}, None),  # client
        ({"ssd_offload": _SSD}, 0),  # generation worker
    ],
)
def test_install_is_a_noop_unless_this_process_owns_memory(
    fake_store, mooncake_cpu, segment_size
) -> None:
    offload.install(_cfg(**mooncake_cpu), segment_size)
    assert fake_store.module.MooncakeDistributedStore is fake_store.cls
    assert fake_store.exit_hooks == []
    # Every client writes, so every client waits for memory when offload is on.
    waits = getattr(fake_store.client_cls, "_nrl_waits_for_memory", False)
    assert waits == ("ssd_offload" in mooncake_cpu)


def test_install_gives_each_owner_its_own_dir(fake_store, tmp_path) -> None:
    cfg = _cfg(ssd_offload={**_SSD, "path": str(tmp_path)})
    offload.install(cfg, None)
    offload.install(cfg, None)  # idempotent
    fake_store.module.MooncakeDistributedStore().setup("h", "m", 1, 2, "tcp", "", "a")

    (owner_dir,) = (tmp_path / "nemo_rl_mooncake_offload").iterdir()
    assert owner_dir.name.startswith(f"05000000-{os.getpid()}-")
    assert owner_dir.stat().st_mode & 0o777 == 0o700
    assert fake_store.calls == [
        (
            ("h", "m", 1, 2, "tcp", "", "a"),
            {"enable_ssd_offload": True, "ssd_offload_path": str(owner_dir)},
        )
    ]
    # The bucket backend holds objects in memory until a 256 MB bucket fills,
    # and disk watermark eviction would delete the only copy of an object.
    assert {var: os.environ[var] for var in offload.ENV_DEFAULTS} == {
        "MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR": "file_per_key_storage_backend",
        "MOONCAKE_OFFLOAD_ENABLE_DISK_WATERMARK_EVICTION": "false",
    }

    # Removed at a normal exit, or on request (storage units at shutdown).
    assert fake_store.exit_hooks == [offload.remove_dir]
    offload.remove_dir()
    assert not owner_dir.exists()


def test_install_keeps_a_launcher_override(fake_store, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR", "custom")
    offload.install(_cfg(ssd_offload={**_SSD, "path": str(tmp_path)}), None)
    assert os.environ["MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR"] == "custom"


def test_remove_dir_without_install_is_a_noop(fake_store) -> None:
    offload.remove_dir()


FULL = offload.NO_AVAILABLE_HANDLE


def _client_cls(tq_retries):
    """A fresh TQ store-client class (patching is per class) recording TQ's retries."""

    class Client:
        replica_config = None

        def _batch_upsert_with_retry(self, keys, ptrs, sizes):
            tq_retries.append(list(keys))

    return Client


class _Store:
    """Answers each batch_upsert_from with the next scripted result list."""

    def __init__(self, *rounds):
        self.rounds = list(rounds)
        self.calls = []

    def batch_upsert_from(self, keys, ptrs, sizes, config=None):
        self.calls.append(list(keys))
        return self.rounds.pop(0)


def _waiting_client(monkeypatch, *rounds, step_s=0.0):
    """A store client whose puts wait for memory; returns it, TQ's retries, sleeps."""
    tq_retries, sleeps, now = [], [], [0.0]

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += step_s

    monkeypatch.setattr(
        offload, "time", SimpleNamespace(sleep=sleep, monotonic=lambda: now[0])
    )

    Client = _client_cls(tq_retries)
    offload.wait_for_memory_on_put(Client)
    client = Client()
    client._store = _Store(*rounds)
    return client, tq_retries, sleeps


def test_put_waits_for_offload_to_free_memory(monkeypatch) -> None:
    client, tq_retries, sleeps = _waiting_client(monkeypatch, [FULL, 0], [FULL], [0])
    client._batch_upsert_with_retry(["a", "b"], [1, 2], [8, 8])
    # Only the key that found memory full is put again, until it fits.
    assert client._store.calls == [["a", "b"], ["a"], ["a"]]
    assert tq_retries == []
    assert sleeps == [0.05, 0.1]


def test_put_backoff_is_capped(monkeypatch) -> None:
    client, _, sleeps = _waiting_client(monkeypatch, *([[FULL]] * 8), [0])
    client._batch_upsert_with_retry(["a"], [1], [8])
    assert max(sleeps) == offload.PUT_BACKOFF_MAX_S


@pytest.mark.parametrize(
    ("rounds", "warnings"), [([[FULL], [FULL], [0]], 1), ([[0]], 0)]
)
def test_a_waiting_put_warns_once(monkeypatch, caplog, rounds, warnings) -> None:
    client, _, _ = _waiting_client(monkeypatch, *rounds)
    with caplog.at_level("WARNING", logger=offload.LOGGER.name):
        client._batch_upsert_with_retry(["a"], [1], [8])
    assert [
        "waiting for SSD offload to free memory" in r.getMessage()
        for r in caplog.records
    ] == [True] * warnings


def test_other_put_errors_keep_tqs_own_retry(monkeypatch) -> None:
    client, tq_retries, sleeps = _waiting_client(monkeypatch, [-1, 0])
    client._batch_upsert_with_retry(["a", "b"], [1, 2], [8, 8])
    assert tq_retries == [["a"]]
    assert sleeps == []


def test_put_fails_once_memory_stays_full_for_the_whole_wait(monkeypatch) -> None:
    rounds = [[FULL]] * 100
    client, _, _ = _waiting_client(monkeypatch, *rounds, step_s=100.0)
    with pytest.raises(RuntimeError, match="stayed full .* SSD offload.*'a'"):
        client._batch_upsert_with_retry(["a"], [1], [8])
    assert len(client._store.calls) == offload.PUT_WAIT_S / 100 + 1


def test_put_rejects_a_short_result_list(monkeypatch) -> None:
    client, _, _ = _waiting_client(monkeypatch, [0])
    with pytest.raises(RuntimeError, match=r"returned \[0\] for 2 keys"):
        client._batch_upsert_with_retry(["a", "b"], [1, 2], [8, 8])


def test_wait_for_memory_patches_a_class_once() -> None:
    Client = _client_cls([])
    offload.wait_for_memory_on_put(Client)
    patched = Client._batch_upsert_with_retry
    offload.wait_for_memory_on_put(Client)
    assert Client._batch_upsert_with_retry is patched


_METRICS = """\
# HELP master_evict_disk_replica_requests_total Total number of EvictDiskReplica requests received
# TYPE master_evict_disk_replica_requests_total counter
master_evict_disk_replica_requests_total {value}
master_evict_disk_replica_failures_total 0
master_put_start_requests_total 12
"""


@pytest.mark.parametrize(
    ("text", "count"),
    [
        (_METRICS.format(value=0), 0),
        (_METRICS.format(value=3), 3),
        (_METRICS.format(value="2.0"), 2),
        ('master_evict_disk_replica_requests_total{tenant="a"} 4\n', 4),
        ("master_put_start_requests_total 9\n", 0),  # counter not exported yet
    ],
)
def test_disk_evictions_reads_the_master_counter(text, count) -> None:
    assert offload.disk_evictions(text) == count


def _watch(reads, **kwargs):
    """Run the watcher over scripted counter reads until it stops."""
    script = iter(reads)
    calls: list[str] = []

    def read() -> int:
        value = next(script, None)
        if value is None:
            stop.set()
            return 0
        if isinstance(value, Exception):
            raise value
        return value

    def on_eviction() -> None:
        calls.append("evicted")

    stop = offload.watch_disk_evictions(
        read, "/local/ssd", interval_s=0.001, on_eviction=on_eviction, **kwargs
    )
    stop.wait(timeout=10)
    return calls


def test_watcher_stops_the_run_on_the_first_disk_eviction(caplog) -> None:
    calls = _watch([0, 0, 2, 5])

    # Once only: the first deletion already means data may be lost.
    assert calls == ["evicted"]
    assert "deleted SSD copies" in caplog.text
    assert "/local/ssd" in caplog.text


def test_watcher_tolerates_an_unreachable_master() -> None:
    # The master can be briefly unreachable (startup, shutdown); keep watching.
    assert _watch([OSError("refused"), 0, OSError("timeout"), 0]) == []


def test_watcher_stops_when_asked() -> None:
    import time

    reads: list[int] = []

    def read() -> int:
        reads.append(0)
        return 0

    stop = offload.watch_disk_evictions(
        read, "/local/ssd", interval_s=0.001, on_eviction=lambda: None
    )
    time.sleep(0.05)
    stop.set()
    time.sleep(0.05)  # let an in-flight read finish
    settled = len(reads)
    time.sleep(0.1)
    assert reads and len(reads) == settled
