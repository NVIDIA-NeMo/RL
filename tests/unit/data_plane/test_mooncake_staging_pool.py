# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""RDMA buffer-registration bookkeeping for the mooncake staging pool.

Every failure guarded here reaches production as the same symptom — mooncake's
generic ``TRANSFER_FAIL`` (-800) out of ``batch_upsert_from``, carrying no root
cause and often on keys the pool never touched, because the damage is to
address-level registration state shared with the unpatched byte paths. None of
it needs an RDMA device to reproduce: the pool's contract is with
``register_buffer`` / ``unregister_buffer``, so a recording double is enough.
"""

from __future__ import annotations

import threading
import time

import pytest

from nemo_rl.data_plane.adapters import transfer_queue as tq_adapter


class _FakeStore:
    """Recording stand-in for ``MooncakeDistributedStore``.

    Tracks which addresses are currently registered so a test can assert the
    invariant that actually matters: the pool never drops a buffer whose
    memory mooncake still maps.
    """

    def __init__(self, fail_after: int | None = None) -> None:
        # None: never fail. n: fail every registration after the first n.
        self.fail_after = fail_after
        self.registered: dict[int, int] = {}
        self.register_calls = 0
        self.unregistered: list[int] = []

    def register_buffer(self, ptr: int, nbytes: int) -> int:
        self.register_calls += 1
        if self.fail_after is not None and self.register_calls > self.fail_after:
            return -1
        self.registered[ptr] = nbytes
        return 0

    def unregister_buffer(self, ptr: int) -> int:
        self.unregistered.append(ptr)
        self.registered.pop(ptr, None)
        return 0


class _FakeClient:
    """Just the attribute surface ``_StagingPoolRegistry.pool_for`` touches."""

    def __init__(self, store: _FakeStore) -> None:
        self._store = store


# ── register_buffer status checking ──────────────────────────────────────────


@pytest.mark.parametrize("status", [0, None], ids=["zero", "none"])
def test_register_checked_accepts_success_statuses(status) -> None:
    """``None`` must pass: the binding's return type varies across wheels."""
    store = _FakeStore()
    store.register_buffer = lambda ptr, nbytes: status  # type: ignore[method-assign]
    tq_adapter._register_checked(store, 0x1000, 4096)


# Comfortably above every payload these tests stage, so the ceiling only
# matters in the test that sets it explicitly.
_MAX = 1 << 24


def test_register_checked_raises_on_failed_registration() -> None:
    """The whole point: fail at the registration, not three retries later."""
    store = _FakeStore(fail_after=0)
    with pytest.raises(RuntimeError, match="register_buffer.*failed with status -1"):
        tq_adapter._register_checked(store, 0x1000, 4096)


def test_register_all_buffers_patch_checks_upstream_call_site(monkeypatch) -> None:
    """The patched path is the one in the -800 traceback.

    ``_put_bytes_thread_worker`` — non-tensor fields, untouched by the staging
    pool — reaches ``register_buffer`` only through ``_register_all_buffers``,
    which upstream calls for its side effect and never inspects.
    """
    mc = pytest.importorskip("transfer_queue.storage.clients.mooncake_client")
    cls = mc.MooncakeStoreClient

    # Restore both the method and the idempotence flag so the shared module is
    # left exactly as found for other tests in this session.
    monkeypatch.setattr(cls, "_register_all_buffers", cls._register_all_buffers)
    monkeypatch.setattr(cls, "_nrl_register_checked", False, raising=False)

    tq_adapter._patch_mooncake_register_check()

    client = cls.__new__(cls)  # __init__ would build a real mooncake store
    client._store = _FakeStore(fail_after=1)
    with pytest.raises(RuntimeError, match="register_buffer"):
        client._register_all_buffers([0x1000, 0x2000], [4096, 4096])


# ── pool slot bookkeeping ────────────────────────────────────────────────────


def test_growing_a_slot_unregisters_before_dropping_the_old_buffer() -> None:
    store = _FakeStore()
    pool = tq_adapter._StagingPool(store, n_slots=1, max_bytes=_MAX)

    with pool.buffer(1024) as small:
        small_ptr = small.data_ptr()
    assert store.registered == {small_ptr: 1024}

    with pool.buffer(8 * 1024 * 1024) as big:
        big_ptr = big.data_ptr()

    assert small_ptr in store.unregistered
    assert store.registered == {big_ptr: 8 * 1024 * 1024}


def test_failed_growth_leaves_the_slot_empty_not_poisoned() -> None:
    """A slot must never come back holding an unregistered buffer.

    Reusing one is the silent variant of this bug: every later transfer
    through that slot writes into memory the NIC never mapped and returns
    -800, which retrying cannot fix.
    """
    store = _FakeStore(fail_after=0)
    pool = tq_adapter._StagingPool(store, n_slots=1, max_bytes=_MAX)

    with pytest.raises(RuntimeError, match="register_buffer"):
        with pool.buffer(1024):
            pass
    assert store.registered == {}

    store.fail_after = None
    with pool.buffer(1024) as buf:
        assert store.registered == {buf.data_ptr(): 1024}


def test_oversized_transfer_bypasses_the_pool_and_unregisters() -> None:
    """Outliers get a transient registration; it must not outlive the call."""
    store = _FakeStore()
    pool = tq_adapter._StagingPool(store, n_slots=1, max_bytes=4096)

    with pool.buffer(8192) as buf:
        assert store.registered == {buf.data_ptr(): 8192}
        oversized_ptr = buf.data_ptr()

    assert store.unregistered == [oversized_ptr]
    assert store.registered == {}


def test_slot_exhaustion_fails_loudly_instead_of_hanging(monkeypatch) -> None:
    """More concurrent transfers than slots must raise, not block forever."""
    monkeypatch.setattr(tq_adapter, "_STAGING_SLOT_TIMEOUT_S", 0.05)
    pool = tq_adapter._StagingPool(_FakeStore(), n_slots=1, max_bytes=_MAX)

    with pool.buffer(1024):
        with pytest.raises(RuntimeError, match="No mooncake staging slot free"):
            with pool.buffer(1024):
                pass


# ── lazy construction under concurrency ──────────────────────────────────────


def test_pool_is_constructed_once_under_concurrent_first_use(monkeypatch) -> None:
    """The regression: a second pool's buffers are freed while still mapped.

    TQ submits one thread worker per ``BATCH_SIZE_LIMIT`` (400 keys) batch to
    a shared ``ThreadPoolExecutor``, so a rollout step whose tensor fields
    exceed that reaches a cold client from several threads at once. The loser
    of an unsynchronized check-then-set has its pool overwritten and garbage
    collected, freeing registered memory that the allocator immediately hands
    to the next caller — including ``_put_bytes_thread_worker``'s receive
    region, which is where the -800 surfaced.

    Slowing the constructor makes the interleaving deterministic rather than
    relying on winning a race a fixed number of times.
    """
    constructed: list[object] = []
    original_init = tq_adapter._StagingPool.__init__

    def slow_init(self, store, n_slots, max_bytes):  # type: ignore[no-untyped-def]
        time.sleep(0.05)  # widen the check-then-set window
        original_init(self, store, n_slots, max_bytes)
        constructed.append(self)

    monkeypatch.setattr(tq_adapter._StagingPool, "__init__", slow_init)

    # The production registry is a local of _patch_mooncake_staging_buffers, so
    # build one here rather than reaching into the patch closure.
    registry = tq_adapter._StagingPoolRegistry(4, _MAX)
    client = _FakeClient(_FakeStore())
    n_threads = 8
    barrier = threading.Barrier(n_threads)
    seen: list[object] = []

    def worker() -> None:
        barrier.wait()
        seen.append(registry.pool_for(client))

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(constructed) == 1
    assert len(seen) == n_threads
    # Identity, not storage location: every caller must get the one pool that
    # was actually constructed.
    assert seen == [constructed[0]] * n_threads
