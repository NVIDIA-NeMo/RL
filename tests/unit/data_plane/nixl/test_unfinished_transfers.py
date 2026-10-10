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


def test_pool_reuses_buffers_after_a_clean_op():
    ep = _FakeEp()
    pool = BufferPool(ep, threading.Lock(), base_bytes=1024, base_count=1)
    with pool.acquire(512) as (_, buf):
        first = buf
    with pool.acquire(512) as (_, buf):
        assert buf is first
    assert pool.retired == []


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

    def put_segments(segments, nbytes, n_entries, blob_id):
        # The WRITE timed out: every local range it used is now unfinished.
        ep.unfinished.extend((a, n) for a, _, n in segments)
        raise UnitFull("no unit could take it")

    c, ep = _client_with(put_segments)
    p, zc = _plan_with_zero_copy_value()
    with pytest.raises(UnitFull):
        c._put_segments(p, "b")
    zc_addr = zc.data_ptr()
    assert zc_addr in ep.registered  # never deregistered
    assert [a for a, _, _ in c.retained_sources] == [zc_addr]
    assert c.retained_sources[0][2] is not None  # owner referenced
    assert c.retained_source_bytes == zc.nbytes
    # The packed pool buffer was a WRITE source too: retired, not reused.
    assert len(c.pool.retired) == 1
    zc.zero_()  # overwriting contents is allowed


def test_zero_copy_source_is_deregistered_after_a_clean_put():
    def put_segments(segments, nbytes, n_entries, blob_id):
        return blob_id, {"u": 0}

    c, ep = _client_with(put_segments)
    p, zc = _plan_with_zero_copy_value()
    c._put_segments(p, "b")
    assert zc.data_ptr() not in ep.registered
    assert c.retained_sources == [] and c.pool.retired == []


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
    st.ep = SimpleNamespace(transfer=transfer, has_remote=lambda n: True)
    st.ctl = SimpleNamespace(
        call=lambda addr, op, **kw: 0,
        notify=lambda addr, op, **kw: notes.append((op, kw)),
    )
    st.placement = SimpleNamespace(order=lambda units, **kw: units)
    st.node_id, st.timeout_s = "n0", 0.01
    st._units, st._blob_unit = {0: unit}, {}
    st._refresh_units = lambda: None
    st.stats = {"fast_reads": 0, "slow_reads": 0, "tag_mismatch": 0}
    return st, notes


@pytest.mark.parametrize("in_flight", [True, False])
def test_failed_put_tells_the_unit_whether_the_write_may_still_land(in_flight):
    from nemo_rl.data_plane.nixl.errors import UnitFull

    st, notes = _store_with(TransferError("timed out", in_flight=in_flight))
    with pytest.raises(UnitFull):
        st.put_segments([(1000, 0, 64)], 64, 1, blob_id="b")
    assert notes and all(op == "abort" for op, _ in notes)
    assert all(kw["in_flight"] is in_flight for _, kw in notes)


def test_in_flight_hinted_read_fails_instead_of_refilling_the_same_buffer():
    from nemo_rl.data_plane.nixl.blobstore import LostBlobs

    st, _ = _store_with(TransferError("timed out", in_flight=True))
    pinned = []
    st._read_pinned = lambda reqs: pinned.append(reqs)
    meta = {"u": 0, "so": 0, "z": 4096, "g": "abc"}
    scratch = (5000, memoryview(bytearray(64)))
    with pytest.raises(LostBlobs):
        st.read({"b": [(0, 64, 1000)]}, {"b": meta}, scratch)
    assert pinned == []  # the late READ could overwrite whatever pinned wrote


def test_failed_hinted_read_that_never_posted_falls_back_to_pinned():
    st, _ = _store_with(TransferError("no peer", in_flight=False))
    pinned = []
    st._read_pinned = lambda reqs: pinned.append(reqs)
    meta = {"u": 0, "so": 0, "z": 4096, "g": "abc"}
    st.read({"b": [(0, 64, 1000)]}, {"b": meta}, (5000, memoryview(bytearray(64))))
    assert pinned == [{"b": [(0, 64, 1000)]}]


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


class _FileEp:
    """Endpoint stub for FileStore: every transfer fails the given way."""

    name = "ep"

    def __init__(self, err: TransferError | None) -> None:
        self.err = err
        self.deregistered: list[int] = []

    def register_file(self, fd, nbytes, *, backend="POSIX"):
        return fd

    def deregister_file(self, fd):
        self.deregistered.append(fd)

    def transfer(self, *a, **k):
        if self.err is not None:
            raise self.err


def _file_store(tmp_path, monkeypatch, err):
    from nemo_rl.data_plane.nixl import blobstore

    monkeypatch.setattr(blobstore.ray, "get", lambda x: x)
    directory = SimpleNamespace(
        put_blobs=SimpleNamespace(remote=lambda items: None),
        release=SimpleNamespace(remote=lambda items: [b for b, _ in items]),
    )
    return blobstore.FileStore(
        _FileEp(err), directory, root=str(tmp_path), timeout_s=0.01
    )


@pytest.mark.parametrize("in_flight", [True, False])
def test_file_store_failed_write_keeps_file_and_fd_only_if_in_flight(
    tmp_path, monkeypatch, in_flight
):
    import os

    from nemo_rl.data_plane.nixl import blob_format

    st = _file_store(tmp_path, monkeypatch, TransferError("t", in_flight=in_flight))
    p = blob_format.plan(["k"], [b"x" * 64])
    with pytest.raises(TransferError):
        st.put(1000, p)
    files = os.listdir(tmp_path)
    if in_flight:
        assert len(files) == 1 and len(st.retained) == 1
        assert st.ep.deregistered == []
        fd = next(iter(st.retained.values()))[0]
        os.fstat(fd)  # still open
    else:
        assert files == [] and st.retained == {}


def test_file_store_failed_read_keeps_fd_and_file_through_release(
    tmp_path, monkeypatch
):
    import os

    from nemo_rl.data_plane.nixl.blobstore import LostBlobs

    st = _file_store(tmp_path, monkeypatch, None)
    with open(st.path("b"), "wb") as f:
        f.write(b"\0" * 64)
    st.ep.err = TransferError("t", in_flight=True)
    with pytest.raises(LostBlobs):
        st.read({"b": [(0, 64, 1000)]}, {"b": {"z": 64}})
    st.release({"b": 1})
    st.close()
    assert os.path.exists(st.path("b"))
    assert st.ep.deregistered == []
    os.fstat(st.retained["b"][0])
