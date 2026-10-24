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
"""A transfer that fails without reaching DONE may still land later.

Releasing a NIXL handle does not stop the transfer, so its destination must
never be reused. These tests stub the NIXL agent with one that stays in PROC,
which no real run produces on demand.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from nemo_rl.data_plane.nixl.bufpool import BufferPool, PoolExhausted
from nemo_rl.data_plane.nixl.errors import TransferError
from nemo_rl.data_plane.nixl.nixl_io import NixlEndpoint


class _StuckAgent:
    """NIXL agent stub: transfers stay in PROC (or fail to post)."""

    def __init__(self, post_fails: bool = False) -> None:
        self.post_fails = post_fails
        self.released = 0

    def get_xfer_descs(self, descs, mem):
        return descs

    def initialize_xfer(self, op, ldesc, rdesc, remote_agent):
        if self.post_fails:
            raise RuntimeError("NIXL_ERR_REMOTE_DISCONNECT")
        return object()

    def transfer(self, h):
        return "PROC"

    def check_xfer_state(self, h):
        return "PROC"

    def release_xfer_handle(self, h):
        self.released += 1


def _endpoint(agent) -> NixlEndpoint:
    ep = object.__new__(NixlEndpoint)
    ep.name, ep.agent, ep.unfinished, ep.unfinished_handles = "ep", agent, [], []
    return ep


def test_timed_out_read_is_in_flight_and_its_local_ranges_recorded():
    ep = _endpoint(_StuckAgent())
    with pytest.raises(TransferError) as ei:
        ep.transfer("READ", [(1000, 64)], [(0, 64)], "peer", timeout_s=0.01)
    assert ei.value.in_flight
    # Never released: POSIX would delete the request its pending I/O points at.
    assert ep.agent.released == 0
    assert len(ep.unfinished_handles) == 1
    assert ep.touches_unfinished(1032, 8)
    assert not ep.touches_unfinished(1064, 8)
    assert not ep.touches_unfinished(900, 100)


def test_timed_out_write_records_its_local_source():
    ep = _endpoint(_StuckAgent())
    with pytest.raises(TransferError) as ei:
        ep.transfer("WRITE", [(1000, 64)], [(0, 64)], "peer", timeout_s=0.01)
    assert ei.value.in_flight
    assert ep.agent.released == 0
    assert len(ep.unfinished_handles) == 1
    # The WRITE may still read its source: its memory must outlive it too.
    # (The destination is remote; the unit retires it.)
    assert ep.touches_unfinished(1000, 64)


def test_transfer_that_never_posted_is_not_in_flight():
    ep = _endpoint(_StuckAgent(post_fails=True))
    with pytest.raises(TransferError) as ei:
        ep.transfer("READ", [(1000, 64)], [(0, 64)], "peer", timeout_s=0.01)
    assert not ei.value.in_flight
    assert ep.unfinished == [] and ep.unfinished_handles == []


class _DoneAgent(_StuckAgent):
    def transfer(self, h):
        return "DONE"


def test_done_transfer_releases_its_handle():
    ep = _endpoint(_DoneAgent())
    ep.transfer("READ", [(1000, 64)], [(0, 64)], "peer")
    assert ep.agent.released == 1
    assert ep.unfinished == [] and ep.unfinished_handles == []


def test_close_keeps_nixl_state_while_a_transfer_may_run():
    ep = _endpoint(_StuckAgent())
    ep._remotes, ep._regs = {"peer"}, {1000: ("reg", ["UCX"])}
    with pytest.raises(TransferError):
        ep.transfer("READ", [(1000, 64)], [(0, 64)], "peer", timeout_s=0.01)
    ep.close()
    assert ep.agent is not None and 1000 in ep._regs and "peer" in ep._remotes


class _FakeEp:
    def __init__(self) -> None:
        self.unfinished: list[tuple[int, int]] = []
        self.registered: set[int] = set()

    def register(self, arr):
        a = int(arr.ctypes.data)
        self.registered.add(a)
        return a

    def deregister(self, addr):
        self.registered.discard(addr)

    def touches_unfinished(self, addr, nbytes):
        return NixlEndpoint.touches_unfinished(self, addr, nbytes)


@pytest.mark.parametrize("nbytes", [512, 4096])  # base buffer, oversize buffer
def test_pool_never_reuses_a_buffer_a_failed_read_may_still_fill(nbytes):
    ep = _FakeEp()
    pool = BufferPool(ep, threading.Lock(), base_bytes=1024, base_count=1)
    with pool.acquire(nbytes) as (addr, buf):
        ep.unfinished.append((addr + 8, 16))  # a READ into it timed out
        stuck = buf
    with pool.acquire(nbytes) as (addr2, buf2):
        assert buf2 is not stuck
        assert not ep.touches_unfinished(addr2, buf2.nbytes)
    # The retired buffer stays alive and registered: the late READ lands there.
    assert any(b is stuck for _, b in pool.retired)
    assert addr in ep.registered
    assert pool.stats["retired_bytes"] == stuck.nbytes


def test_a_retired_buffer_counts_whole_against_the_graveyard_limit(monkeypatch):
    from nemo_rl.data_plane.nixl import nixl_io

    g = nixl_io.Graveyard(max_bytes=2000)
    monkeypatch.setattr(nixl_io, "GRAVEYARD", g)
    ep = _FakeEp()
    pool = BufferPool(ep, threading.Lock(), base_bytes=1024, base_count=2)
    with pool.acquire(512) as (addr, _):
        ep.unfinished.append((addr, 8))  # a tiny READ timed out; the buffer is 1024 B
    assert g.bytes == 1024 and g.poisoned is None
    with pool.acquire(512) as (addr, _):
        ep.unfinished.append((addr, 8))
    assert g.bytes == 2048 and g.poisoned is not None  # over the limit: POISONED


def test_pool_reuses_buffers_after_a_clean_op():
    ep = _FakeEp()
    pool = BufferPool(ep, threading.Lock(), base_bytes=1024, base_count=1)
    with pool.acquire(512) as (_, buf):
        first = buf
    with pool.acquire(512) as (_, buf):
        assert buf is first
    assert pool.retired == []


def test_eviction_while_the_only_cached_oversize_buffer_is_in_use():
    ep = _FakeEp()
    pool = BufferPool(ep, threading.Lock(), base_bytes=1024, max_oversize_bytes=8192)
    with pool.acquire(4096):
        pass
    with pool.acquire(4096):  # takes the cached one: its bucket is now empty
        with pool.acquire(8192) as (_, buf):  # over budget: must evict, not crash
            assert buf.nbytes == 8192


def test_failed_replacement_shrinks_the_pool_and_never_hangs():
    ep = _FakeEp()
    pool = BufferPool(ep, threading.Lock(), base_bytes=1024, base_count=1)

    def fail(arr):
        raise MemoryError("register failed")

    with pytest.raises(ValueError, match="caller"):  # caller's error not masked
        with pool.acquire(512) as (addr, _):
            ep.unfinished.append((addr, 8))
            ep.register = fail
            raise ValueError("caller")
    assert pool.stats["replace_failures"] == 1
    with pytest.raises(PoolExhausted, match="no base buffer left"):
        with pool.acquire(512):
            pass


def test_acquire_wait_is_bounded_by_timeout():
    ep = _FakeEp()
    pool = BufferPool(ep, threading.Lock(), base_bytes=1024, base_count=1)
    with pool.acquire(512):
        with pytest.raises(PoolExhausted, match="within"):
            with pool.acquire(512, timeout_s=0.05):
                pass


def test_waiter_gives_up_when_the_last_base_buffer_is_lost():
    ep = _FakeEp()
    pool = BufferPool(ep, threading.Lock(), base_bytes=1024, base_count=1)
    errors = []

    def waiter():
        try:
            with pool.acquire(512):
                pass
        except PoolExhausted as e:
            errors.append(e)

    with pool.acquire(512) as (addr, _):
        t = threading.Thread(target=waiter)
        t.start()
        while pool.stats["base_waits"] == 0:
            pass
        ep.unfinished.append((addr, 8))
        ep.register = lambda arr: (_ for _ in ()).throw(MemoryError())
    t.join(timeout=5)
    assert not t.is_alive() and len(errors) == 1


def _client_with(store_put):
    """A NixlKVClient wired to a fake endpoint and a stub unit store."""
    from nemo_rl.data_plane.adapters.tq_nixl import NixlKVClient

    ep = _FakeEp()
    c = object.__new__(NixlKVClient)
    c.ep, c._lock = ep, threading.RLock()
    c.pool = BufferPool(ep, c._lock, base_bytes=1 << 20, base_count=1)
    c.store = SimpleNamespace(put_segments=store_put)
    c.zero_copy_min, c.zero_copy_max = 1 << 20, 32 << 20
    c.parallel_copy_min = 16 << 20
    c.retained_sources, c.retained_source_bytes = [], 0
    return c, ep


def _plan_with_zero_copy_value():
    import torch

    from nemo_rl.data_plane.nixl import blob_format

    zc = torch.arange(1 << 19, dtype=torch.float32)  # 2 MiB: zero-copy window
    return blob_format.plan(["small", "zc"], [torch.ones(16), zc]), zc


def test_zero_copy_source_of_in_flight_write_stays_registered_and_alive():
    from nemo_rl.data_plane.nixl.errors import UnitFull

    def put_segments(segments, nbytes, n_entries, unit_id, blob_id=None, deadline=None):
        # The WRITE timed out: every local range it used is now unfinished.
        ep.unfinished.extend((a, n) for a, _, n in segments)
        raise UnitFull("no unit could take it")

    c, ep = _client_with(put_segments)
    p, zc = _plan_with_zero_copy_value()
    with pytest.raises(UnitFull):
        c._put_segments(p, "b", time.monotonic() + 60, 0)
    zc_addr = zc.data_ptr()
    assert zc_addr in ep.registered  # never deregistered
    assert [a for a, _, _ in c.retained_sources] == [zc_addr]
    assert c.retained_sources[0][2] is not None  # owner referenced
    assert c.retained_source_bytes == zc.nbytes
    # The packed pool buffer was a WRITE source too: retired, not reused.
    assert len(c.pool.retired) == 1
    zc.zero_()  # overwriting contents is allowed


def test_zero_copy_source_is_deregistered_after_a_clean_put():
    def put_segments(segments, nbytes, n_entries, unit_id, blob_id=None, deadline=None):
        return blob_id

    c, ep = _client_with(put_segments)
    p, zc = _plan_with_zero_copy_value()
    c._put_segments(p, "b", time.monotonic() + 60, 0)
    assert zc.data_ptr() not in ep.registered
    assert c.retained_sources == [] and c.pool.retired == []


def test_put_over_max_blob_bytes_is_split_at_key_boundaries():
    import torch

    blobs = []

    def put_segments(segments, nbytes, n_entries, unit_id, blob_id=None, deadline=None):
        blobs.append((blob_id, nbytes, n_entries))
        return blob_id

    c, _ = _client_with(put_segments)
    c.max_blob_bytes = 10_000  # 4 KiB values: at most 2 per blob
    keys = [f"k{i}" for i in range(5)]
    c._put_split(0, keys, [torch.ones(1024)] * 5, time.monotonic() + 60)
    assert len(blobs) == 3
    assert all(n <= c.max_blob_bytes for _, n, _ in blobs)
    assert sum(k for _, _, k in blobs) == 5


def test_failed_part_of_a_split_put_keeps_the_parts_written():
    # SimpleStorage's policy: committed parts stay published; the put raises,
    # TQ is not notified, and the next overwrite or clear of the rows frees them.
    import torch

    from nemo_rl.data_plane.nixl.errors import UnitFull

    calls = []

    def put_segments(segments, nbytes, n_entries, unit_id, blob_id=None, deadline=None):
        calls.append(blob_id)
        if len(calls) == 2:
            raise UnitFull("no unit could take it")
        return blob_id

    c, _ = _client_with(put_segments)
    c.max_blob_bytes = 5_000  # one 4 KiB value per blob
    with pytest.raises(UnitFull):
        c._put_split(0, ["a", "b"], [torch.ones(1024)] * 2, time.monotonic() + 60)
    assert len(calls) == 2


def _store_with(ep_transfer_error: TransferError):
    from nemo_rl.data_plane.nixl.blobstore import UnitSlabStore

    notes = []
    unit = {
        "unit_id": 0,
        "slab_base": 0,
        "agent_name": "unit-0-abc",
        "zmq": "tcp://x",
    }

    def transfer(*a, **k):
        raise ep_transfer_error

    st = object.__new__(UnitSlabStore)

    def transfer_many(ops, **k):
        return [ep_transfer_error] * len(ops)

    st.ep = SimpleNamespace(
        transfer=transfer, transfer_many=transfer_many, has_remote=lambda n: True
    )
    st.ctl = SimpleNamespace(
        call=lambda addr, op, **kw: 0,
        notify=lambda addr, op, **kw: notes.append((op, kw)),
        timeout_ms=30_000,
    )
    st.timeout_s, st.ssd = 0.01, False
    st._units = {0: unit}
    st._refresh_units = lambda: None
    st._unit_dead = lambda unit_id: False  # a live unit: failures are retryable
    st.stats = {"unit_refresh": 0}
    return st, notes


@pytest.mark.parametrize("in_flight", [True, False])
def test_failed_put_tells_the_unit_whether_the_write_may_still_land(in_flight):
    from nemo_rl.data_plane.nixl.blobstore import TransferFailed

    st, notes = _store_with(TransferError("timed out", in_flight=in_flight))
    # A failed transfer is retryable, never UnitFull (design §6).
    with pytest.raises(TransferFailed) as ei:
        st.put_segments([(1000, 0, 64)], 64, 1, 0, blob_id="b")
    assert ei.value.in_flight is in_flight
    assert notes and all(op == "abort" for op, _ in notes)
    assert all(kw["in_flight"] is in_flight for _, kw in notes)


def _unit_abort():
    from nemo_rl.data_plane.nixl.allocator import SlabAllocator
    from nemo_rl.data_plane.nixl_storage_unit import BlobRec, NixlStorageUnit

    cls = NixlStorageUnit.__ray_metadata__.modified_class
    alloc = SlabAllocator(1 << 20)
    off = alloc.alloc(4096)
    unit = SimpleNamespace(
        _mu=threading.RLock(),
        alloc_=alloc,
        retired_bytes=0,
        ssd_dir=None,
        blobs={"b": BlobRec(off=off, nbytes=4096, refs=1, created_at=0.0)},
    )
    return cls.abort, unit, alloc


def test_unit_abort_of_in_flight_write_never_frees_the_region():
    abort, unit, alloc = _unit_abort()
    used = alloc.used_bytes
    abort(unit, "b", in_flight=True)
    assert "b" not in unit.blobs
    assert alloc.used_bytes == used  # a late WRITE may still land there
    assert unit.retired_bytes == 4096


def test_unit_abort_of_unposted_write_frees_at_once():
    abort, unit, alloc = _unit_abort()
    abort(unit, "b", in_flight=False)
    assert alloc.used_bytes == 0
    assert unit.retired_bytes == 0


# ------------------------------------------------------------------ transfer_many
class _Boom(BaseException):
    """Stands in for KeyboardInterrupt: not an Exception."""


class _ScriptAgent:
    """Handles are numbered in post order; ``polls[h]`` lists the states
    ``check_xfer_state(h)`` returns in turn (then PROC forever)."""

    def __init__(self, polls, init_fails=(), poll_raises=(), init_boom=None):
        self.polls = [list(p) for p in polls]
        self.init_fails, self.poll_raises, self.init_boom = (
            set(init_fails),
            set(poll_raises),
            init_boom,
        )
        self.calls = 0
        self.handles = 0
        self.events: list[str] = []
        self.released: list[int] = []

    def get_xfer_descs(self, descs, mem):
        return descs

    def initialize_xfer(self, op, ldesc, rdesc, remote_agent):
        i, self.calls = self.calls, self.calls + 1
        if i == self.init_boom:
            raise _Boom()
        if i in self.init_fails:
            raise RuntimeError("NIXL_ERR_REMOTE_DISCONNECT")
        self.events.append("post")
        self.handles += 1
        return self.handles - 1

    def transfer(self, h):
        return "PROC"

    def check_xfer_state(self, h):
        self.events.append("check")
        if h in self.poll_raises:
            raise RuntimeError("poll failed")
        return self.polls[h].pop(0) if self.polls[h] else "PROC"

    def release_xfer_handle(self, h):
        self.released.append(h)


def _ops(n):
    return [("READ", [(1000 * (i + 1), 8)], [(0, 8)], f"peer{i}") for i in range(n)]


@pytest.fixture
def graveyard(monkeypatch):
    from nemo_rl.data_plane.nixl import nixl_io

    g = nixl_io.Graveyard(max_bytes=1 << 30)
    monkeypatch.setattr(nixl_io, "GRAVEYARD", g)
    return g


def test_transfer_many_posts_every_op_before_it_waits(graveyard):
    agent = _ScriptAgent([["PROC", "DONE"], ["DONE"]])
    ep = _endpoint(agent)
    assert ep.transfer_many(_ops(2), timeout_s=5) == [None, None]
    assert agent.events[:2] == ["post", "post"]
    assert sorted(agent.released) == [0, 1]
    assert ep.unfinished == [] and graveyard.bytes == 0


def test_transfer_many_timeout_retains_each_unfinished_op_once(graveyard):
    agent = _ScriptAgent([["DONE"], [], []])
    ep = _endpoint(agent)
    errs = ep.transfer_many(_ops(3), timeout_s=0.01)
    assert errs[0] is None and all(e.in_flight for e in errs[1:])
    assert ep.unfinished == [(2000, 8), (3000, 8)]
    assert len(ep.unfinished_handles) == 2 and graveyard.bytes == 16
    assert agent.released == [0]


def test_a_failed_post_does_not_stop_the_other_ops(graveyard):
    agent = _ScriptAgent([["DONE"]], init_fails={0})
    ep = _endpoint(agent)
    errs = ep.transfer_many(_ops(2), timeout_s=5)
    assert errs[0].in_flight is False and errs[1] is None
    assert ep.unfinished == [] and graveyard.bytes == 0  # nothing was posted for op 0


def test_a_poll_error_retains_only_that_transfer(graveyard):
    agent = _ScriptAgent([[], ["DONE"]], poll_raises={0})
    ep = _endpoint(agent)
    errs = ep.transfer_many(_ops(2), timeout_s=5)
    assert errs[0].in_flight is True and errs[1] is None
    assert ep.unfinished == [(1000, 8)]


def test_an_exception_while_waiting_retains_every_posted_transfer(graveyard):
    agent = _ScriptAgent([[], []])
    ep = _endpoint(agent)

    def boom(h):  # the first poll dies with a non-Exception
        raise _Boom()

    agent.check_xfer_state = boom
    with pytest.raises(_Boom):
        ep.transfer_many(_ops(2), timeout_s=5)
    assert ep.unfinished == [(1000, 8), (2000, 8)]
    assert len(ep.unfinished_handles) == 2 and graveyard.bytes == 16


def test_an_exception_while_posting_retains_the_ops_already_posted(graveyard):
    agent = _ScriptAgent([[]], init_boom=1)
    ep = _endpoint(agent)
    with pytest.raises(_Boom):
        ep.transfer_many(_ops(2), timeout_s=5)
    assert ep.unfinished == [(1000, 8)] and len(ep.unfinished_handles) == 1


def test_invalid_ops_raise_before_anything_is_posted(graveyard):
    agent = _ScriptAgent([[]])
    ep = _endpoint(agent)
    bad = [*_ops(1), ("READ", [(1000, 8)], [(0, 16)], "peer")]
    with pytest.raises(ValueError):
        ep.transfer_many(bad)
    assert agent.calls == 0


def test_a_read_posts_every_unit_in_one_batch():
    from nemo_rl.data_plane.nixl.blobstore import TransferFailed
    from tests.unit.data_plane._nixl_helpers import StubCtl, StubEp, stub_store

    batches = []

    class Ep(StubEp):
        def transfer_many(self, ops, *, timeout_s):
            batches.append([o[3] for o in ops])
            return self.results

    ep = Ep()
    ep.results = [None, None]
    s = stub_store(StubCtl({}), ep=ep)
    reads = {0: [(0, 8, 1000)], 1: [(0, 8, 2000)]}
    keys_of = {0: ["1@a"], 1: ["2@b"]}
    s.read_entries(reads, keys_of, None)
    assert batches == [["unit-0-abc", "unit-1-abc"]]
    # A READ that may still land fails the read (retryable: the unit is alive).
    ep.results = [None, TransferError("late", in_flight=True)]
    with pytest.raises(TransferFailed, match="late") as ei:
        s.read_entries(reads, keys_of, None)
    assert ei.value.in_flight


def test_pool_registration_waits_are_counted_by_the_timed_lock():
    from nemo_rl.data_plane.adapters.tq_nixl import TimedLock

    lock = TimedLock()
    pool = BufferPool(_FakeEp(), lock, base_bytes=1024, base_count=1)
    held, release = threading.Event(), threading.Event()

    def hold():
        with lock:
            held.set()
            release.wait(5)

    t = threading.Thread(target=hold)
    t.start()
    held.wait()
    threading.Timer(0.05, release.set).start()
    with pool.acquire(4096):  # oversize: registered under the lock
        pass
    t.join()
    assert lock.waits >= 1 and lock.wait_ms_max >= 20


def test_timed_lock_counts_contention_but_not_reentry():
    from nemo_rl.data_plane.adapters.tq_nixl import TimedLock

    lock = TimedLock()
    with lock:
        with lock:
            pass
    assert lock.snapshot() == (0, 0.0, 0.0)


def test_client_logs_op_time_and_lock_waits_every_30_seconds(capsys):
    from nemo_rl.data_plane.adapters.tq_nixl import NixlKVClient, TimedLock

    c = object.__new__(NixlKVClient)
    c._lock, c._op_mu, c._who = TimedLock(), threading.Lock(), "pid=1"
    c._ops, c._op_ms, c._stats_last = 0, 0.0, None
    c._stats_at = time.monotonic() - 31

    def fail(keys, values):
        raise RuntimeError("put failed")

    c._put = fail
    with pytest.raises(RuntimeError):
        c.put(["k"], [1])  # a failed put still counts
    out = capsys.readouterr().out
    assert "[nvdp-client] pid=1 stats" in out and "'ops': 1" in out
    with pytest.raises(RuntimeError):
        c.put(["k"], [1])
    assert capsys.readouterr().out == ""  # not due again yet
