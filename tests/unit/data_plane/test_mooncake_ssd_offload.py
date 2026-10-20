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
import functools
import http.client
import http.server
import importlib.util
import multiprocessing
import os
import signal
import threading
from pathlib import Path
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
    for var in offload.REQUIRED_ENV:
        # setenv first so monkeypatch restores the var even if it was unset.
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    monkeypatch.setattr(offload, "_dir", None)
    monkeypatch.setattr(offload, "_lock_fd", None)
    monkeypatch.setattr(offload, "_offloading_store_set_up", False)
    monkeypatch.setattr(offload, "_sweep_stop", None)
    exit_hooks = []
    monkeypatch.setattr(atexit, "register", exit_hooks.append)
    yield SimpleNamespace(
        module=mooncake_client,
        cls=FakeStore,
        client_cls=FakeClient,
        calls=calls,
        exit_hooks=exit_hooks,
    )
    if offload._sweep_stop is not None:
        offload._sweep_stop.set()  # a fake store has nothing to sweep


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
    assert {var: os.environ[var] for var in offload.REQUIRED_ENV} == {
        "MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR": "file_per_key_storage_backend",
        "MOONCAKE_OFFLOAD_ENABLE_DISK_WATERMARK_EVICTION": "false",
    }

    # Removed at a normal exit, or on request (storage units at shutdown).
    assert fake_store.exit_hooks == [offload.remove_dir]
    offload.remove_dir()
    assert not owner_dir.exists()


@pytest.mark.parametrize(
    "name,value",
    [
        ("MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR", "bucket_storage_backend"),
        ("MOONCAKE_OFFLOAD_ENABLE_DISK_WATERMARK_EVICTION", "true"),
    ],
)
def test_install_rejects_a_launcher_override(
    fake_store, monkeypatch, tmp_path, name, value
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=f"needs {name}="):
        offload.install(_cfg(ssd_offload={**_SSD, "path": str(tmp_path)}), None)
    assert not (tmp_path / "nemo_rl_mooncake_offload").exists()


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
    with pytest.raises(
        RuntimeError, match="(?s)stayed full .* SSD offload.*'a'.*larger than any"
    ):
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
    calls = _watch([0, 0, 1])

    # Once only: the first deletion already means data may be lost.
    assert calls == ["evicted"]
    assert "deleted SSD copies" in caplog.text
    assert "/local/ssd" in caplog.text


def test_watcher_survives_any_read_error(caplog) -> None:
    # A dead watcher would hide data loss, so no read error may end it.
    bad = http.client.IncompleteRead(b"")
    assert _watch([bad, bad, ValueError("garbled"), 0, 1]) == ["evicted"]
    assert [r.getMessage() for r in caplog.records if r.levelname == "WARNING"] == [
        "Cannot read the Mooncake SSD-eviction count; retrying"
    ]


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


@pytest.fixture
def metrics_server():
    """A local stand-in for the master's /metrics; set ``.body`` to change it."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = server.body.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.body = "master_evict_disk_replica_requests_total 0\n"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()


def test_watch_master_disk_evictions_reads_the_real_endpoint(
    monkeypatch, metrics_server
) -> None:
    fired = threading.Event()
    monkeypatch.setattr(offload, "_watch_stop", None)
    monkeypatch.setattr(
        offload,
        "watch_disk_evictions",
        functools.partial(
            offload.watch_disk_evictions, interval_s=0.01, on_eviction=fired.set
        ),
    )
    url = f"http://127.0.0.1:{metrics_server.server_port}/metrics"

    offload.watch_master_disk_evictions(url, "/local/ssd")
    assert not fired.wait(0.2)  # counter at 0: keep watching
    metrics_server.body = "master_evict_disk_replica_requests_total 1\n"
    assert fired.wait(5)


def test_a_reinit_replaces_the_old_watcher(monkeypatch) -> None:
    started = []
    monkeypatch.setattr(offload, "_watch_stop", None)
    monkeypatch.setattr(
        offload,
        "watch_disk_evictions",
        lambda read, path: started.append(threading.Event()) or started[-1],
    )

    offload.watch_master_disk_evictions("http://unused/metrics", "/a")
    offload.watch_master_disk_evictions("http://unused/metrics", "/b")

    assert [stop.is_set() for stop in started] == [True, False]


def test_an_owner_must_build_the_offloading_store(fake_store, tmp_path) -> None:
    # If TQ stops building its store through the swapped class, the owner
    # would silently never offload.
    offload.install(_cfg(ssd_offload={**_SSD, "path": str(tmp_path)}), None)
    with pytest.raises(RuntimeError, match="would never offload"):
        offload.require_offloading_store()

    fake_store.module.MooncakeDistributedStore().setup("h", "m", 1, 2, "tcp", "", "a")
    offload.require_offloading_store()


def test_a_process_that_owns_nothing_needs_no_offloading_store(fake_store) -> None:
    offload.install(_cfg(ssd_offload=_SSD, storage_unit_segment_size=UNIT), None)
    offload.require_offloading_store()


def _own_dir(path, how, ready):
    offload._hold_dir_lock(path)
    ready.set()
    if how == "exit":
        os._exit(0)  # Mooncake's SIGTERM handler ends this way: no atexit
    signal.pause()  # until killed


def test_install_removes_only_dead_owners_dirs(fake_store, tmp_path) -> None:
    parent = tmp_path / "nemo_rl_mooncake_offload"
    parent.mkdir()
    ctx = multiprocessing.get_context("fork")
    owners = {}
    for how in ("exit", "killed", "alive"):
        (parent / how).mkdir()
        ready = ctx.Event()
        owners[how] = ctx.Process(target=_own_dir, args=(str(parent / how), how, ready))
        owners[how].start()
        assert ready.wait(10)
    (parent / "starting").mkdir()  # no lock yet: its owner is still in install
    owners["exit"].join(10)
    os.kill(owners["killed"].pid, signal.SIGKILL)
    owners["killed"].join(10)
    try:
        offload.install(_cfg(ssd_offload={**_SSD, "path": str(tmp_path)}), None)
        left = {p.name for p in parent.iterdir()}
        assert {"alive", "starting"} <= left
        assert not {"exit", "killed"} & left
        assert Path(offload._dir).name in left  # this process's own new dir
    finally:
        owners["alive"].kill()
        owners["alive"].join(10)


def test_copied_mooncake_names_exist_in_the_installed_mooncake() -> None:
    """Names copied from Mooncake's C++ must still be in the shipped binaries.

    Mooncake exports none of them to Python, so they cannot be imported, and a
    rename on a Mooncake bump would be silent: an env var nothing reads, or a
    metric the eviction watcher never finds.
    """
    spec = importlib.util.find_spec("mooncake")
    assert spec is not None, "mooncake is a core dependency on Linux"
    root = Path(spec.submodule_search_locations[0])
    store = (root / "store.so").read_bytes()  # the client reads the env vars
    master = (root / "mooncake_master").read_bytes()  # flags and /metrics

    for name in offload.REQUIRED_ENV:
        assert name.encode() in store, name
    backend = offload.REQUIRED_ENV["MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR"]
    assert backend.encode() in store, backend
    assert offload.EVICTION_COUNTER.encode() in master, offload.EVICTION_COUNTER
    for arg in offload.master_args(_cfg(ssd_offload=_SSD)):
        flag = arg.removeprefix("--").split("=", 1)[0]
        assert flag.encode() in master, flag


def _kv_file(
    path: Path, key: str, value: bytes = b"v" * 300, tenant: str | None = "default"
) -> Path:
    """A file as Mooncake's per-key backend writes it: KVEntry{tenant\\0key, value}."""

    def field(tag: int, data: bytes) -> bytes:
        n, out = len(data), bytearray([tag])
        while True:
            out.append((n & 0x7F) | (0x80 if n > 0x7F else 0))
            n >>= 7
            if not n:
                return bytes(out) + data

    path.parent.mkdir(parents=True, exist_ok=True)
    stored = key if tenant is None else f"{tenant}\0{key}"
    path.write_bytes(field(0x0A, stored.encode()) + field(0x12, value))
    return path


def test_file_key_reads_the_key_stored_in_the_file(tmp_path) -> None:
    # The file name is the key with / \\ : * ? " < > | replaced by _, so only
    # the stored key is trustworthy.
    long_key = "7@" + "f" * 200  # a two-byte length varint
    assert offload._file_key(str(_kv_file(tmp_path / "a", "3@a/b"))) == "3@a/b"
    assert offload._file_key(str(_kv_file(tmp_path / "b", long_key))) == long_key
    (tmp_path / "c").write_bytes(b"not a kv entry")
    (tmp_path / "d").write_bytes(b"\x0a\x09short")  # length past the end
    assert offload._file_key(str(tmp_path / "c")) is None
    assert offload._file_key(str(tmp_path / "d")) is None
    # Only the default tenant's keys are TQ's; an unscoped key is not a key
    # the master knows (querying it would wrongly report it absent).
    assert offload._file_key(str(_kv_file(tmp_path / "e", "1@x", tenant="t2"))) is None
    assert offload._file_key(str(_kv_file(tmp_path / "f", "1@x", tenant=None))) is None


def test_file_key_reads_a_file_as_mooncake_0_3_13_writes_it(tmp_path) -> None:
    # Header bytes of a real file from job 1338878.
    real = tmp_path / "default_98@big"
    real.write_bytes(b"\n\x0edefault\x0098@big\x12\x80\x80\x80\x02" + b"\x00" * 64)
    assert offload._file_key(str(real)) == "98@big"


def test_sweep_deletes_only_the_files_of_cleared_keys(tmp_path) -> None:
    live = _kv_file(tmp_path / "a" / "b" / "1@x", "1@x")
    cleared = _kv_file(tmp_path / "c" / "d" / "2@x", "2@x")
    unknown = _kv_file(tmp_path / "e" / "f" / "3@x", "3@x")
    (tmp_path / ".lock").write_bytes(b"")
    (tmp_path / "foreign").write_bytes(b"not mooncake's")
    answers = {"1@x": 1, "2@x": 0, "3@x": -1}  # exists, absent, RPC error

    removed = offload.sweep_cleared_files(
        str(tmp_path), lambda keys: [answers[k] for k in keys]
    )

    assert removed == 1
    assert not cleared.exists()
    assert live.exists() and unknown.exists()
    assert (tmp_path / ".lock").exists() and (tmp_path / "foreign").exists()


def test_sweep_keeps_a_file_rewritten_after_it_was_listed(tmp_path) -> None:
    # The key was absent at the check, then re-put and offloaded again before
    # the unlink: the file changed since the listing, so it is the live copy.
    rewritten = _kv_file(tmp_path / "2@x", "2@x")
    replaced = _kv_file(tmp_path / "3@x", "3@x")

    def exists(keys):
        _kv_file(rewritten, "2@x", value=b"w" * 500)  # rewritten in place
        tmp = _kv_file(tmp_path / "tmp", "3@x")
        os.replace(tmp, replaced)  # replaced by rename (new inode)
        return [0] * len(keys)

    assert offload.sweep_cleared_files(str(tmp_path), exists) == 0
    assert rewritten.exists() and replaced.exists()


def test_sweep_keeps_everything_on_a_short_result_list(tmp_path) -> None:
    kept = _kv_file(tmp_path / "2@x", "2@x")
    assert offload.sweep_cleared_files(str(tmp_path), lambda keys: []) == 0
    assert kept.exists()


def test_an_owners_store_sweeps_until_its_dir_is_removed(
    fake_store, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(offload, "_sweep_stop", None)
    offload.install(_cfg(ssd_offload={**_SSD, "path": str(tmp_path)}), None)
    assert offload._sweep_stop is None  # the store is not set up yet

    fake_store.module.MooncakeDistributedStore().setup("h", "m", 1, 2, "tcp", "", "a")
    stop = offload._sweep_stop
    assert stop is not None and not stop.is_set()

    offload.remove_dir()
    assert stop.is_set()
