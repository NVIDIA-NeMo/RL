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
"""Error kinds reach NeMo-RL intact, every wait has a deadline, and the
Graveyard poisons the process past its cap (design §2.2, §2.4, §2.5, §6).

Only confirmed loss is DataPlaneLostKeys; a failed transfer or a slow unit
is retryable. CPU-only: stub agents and stores, a real ZMQ socket pair.
"""

import threading
import time

import pytest

from nemo_rl.data_plane.adapters import tq_nixl
from nemo_rl.data_plane.nixl import blobstore, control, nixl_io
from nemo_rl.data_plane.nixl.blobstore import LostBlobs, TransferFailed
from nemo_rl.data_plane.nixl.bufpool import PoolExhausted, PoolTimeout
from nemo_rl.data_plane.nixl.errors import (
    DataPlaneBusy,
    DataPlaneFatal,
    DataPlaneLostKeys,
    DataPlaneTimeout,
    DataPlaneTransferFailed,
    TransferError,
    UnitFull,
)
from nemo_rl.data_plane.nixl.nixl_io import Graveyard, NixlEndpoint
from tests.unit.data_plane._nixl_helpers import StubCtl as _Ctl
from tests.unit.data_plane._nixl_helpers import StubEp as _Ep
from tests.unit.data_plane._nixl_helpers import bare_unit, gidx_of, key_of, write_blob
from tests.unit.data_plane._nixl_helpers import stub_store as _store


# ------------------------------------------------------------------ wire (§6)
@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(control, "node_ip", lambda: "127.0.0.1")

    def busy():
        raise DataPlaneBusy("staging full", reason="staging_full")

    def missing():
        raise LostBlobs(["b1", "b2"], "blob not on unit")

    def full():
        raise UnitFull("no room")

    def broken():
        raise KeyError("oops")

    def slow():
        time.sleep(1.0)
        return 1

    srv = control.UnitServer(
        {
            "busy": busy,
            "missing": missing,
            "full": full,
            "broken": broken,
            "slow": slow,
        },
        threading.Lock(),
    )
    yield srv
    srv.close()


def test_unit_errors_keep_their_kind_over_the_wire(server):
    cli = control.UnitClient(timeout_s=5)
    try:
        with pytest.raises(DataPlaneBusy) as ei:
            cli.call(server.address, "busy")
        assert ei.value.reason == "staging_full"
        with pytest.raises(LostBlobs) as ei:
            cli.call(server.address, "missing")
        assert ei.value.blob_ids == ["b1", "b2"]
        with pytest.raises(UnitFull):
            cli.call(server.address, "full")
        with pytest.raises(TransferError, match="KeyError"):
            cli.call(server.address, "broken")
        with pytest.raises(TransferError, match="ValueError"):
            cli.call(server.address, "no_such_op")
    finally:
        cli.close()


def test_reply_wait_honours_the_callers_budget(server):
    cli = control.UnitClient(timeout_s=30)
    try:
        t0 = time.monotonic()
        with pytest.raises(TransferError, match="did not answer"):
            cli.call(server.address, "slow", timeout_s=0.2)
        assert time.monotonic() - t0 < 0.8
    finally:
        cli.close()


def test_malformed_op_does_not_stop_the_unit_serving(server):
    import zmq

    ctx = zmq.Context()
    bad = ctx.socket(zmq.DEALER)
    bad.setsockopt(zmq.LINGER, 0)
    bad.connect(server.address)
    bad.send(control._enc.encode([[1], "x", {}]))  # unhashable op
    cli = control.UnitClient(timeout_s=5)
    try:
        with pytest.raises(UnitFull):
            cli.call(server.address, "full")
    finally:
        cli.close()
        bad.close(0)
        ctx.term()


def test_pre_structured_error_replies_still_raise():
    with pytest.raises(TransferError, match="KeyError"):
        control.raise_reply("tcp://x", "op", ["KeyError", "oops"])


# ------------------------------------------------------------------ adapter mapping (§6)
def _client():
    return object.__new__(tq_nixl.NixlKVClient)


@pytest.mark.parametrize(
    "raised, expected",
    [
        (LostBlobs(["bB"], "unit 0 is dead"), DataPlaneLostKeys),
        (
            TransferFailed(["bB"], "READ timed out", in_flight=True),
            DataPlaneTransferFailed,
        ),
        (PoolTimeout("no base buffer free within 1s"), DataPlaneTimeout),
        (PoolExhausted("no base buffer left"), DataPlaneFatal),
    ],
)
def test_adapter_maps_store_errors(raised, expected):
    keys = ["k0", "k1", "k2"]
    by_blob = {"bA": [0], "bB": [1, 2]}
    with pytest.raises(expected) as ei:
        with _client()._errors(keys, by_blob):
            raise raised
    if expected in (DataPlaneLostKeys, DataPlaneTransferFailed):
        assert ei.value.keys == ["k1", "k2"]  # only the affected blob's keys
    if expected is DataPlaneTransferFailed:
        assert ei.value.in_flight


def test_no_retryable_error_counts_as_data_loss():
    for cls in (
        DataPlaneTransferFailed,
        DataPlaneTimeout,
        DataPlaneBusy,
        DataPlaneFatal,
    ):
        assert not issubclass(cls, DataPlaneLostKeys)


# ------------------------------------------------------------------ store classification
@pytest.mark.parametrize("dead, expected", [(False, TransferFailed), (True, LostBlobs)])
def test_unanswered_resolve_is_loss_only_if_the_unit_is_dead(dead, expected):
    ctl = _Ctl({"resolve": lambda a: TransferError("did not answer resolve")})
    s = _store(ctl, dead=dead)
    with pytest.raises(expected):
        s._read_pinned({"b": [(0, 8, 1000)]})


def test_failed_pinned_read_on_a_live_unit_is_retryable_and_unpins():
    ctl = _Ctl(
        {"resolve": lambda a: {"pin_id": "p", "offsets": {"b": 0}, "missing": []}}
    )
    s = _store(ctl, ep=_Ep(fail=TransferError("timed out", in_flight=True)))
    with pytest.raises(TransferFailed) as ei:
        s._read_pinned({"b": [(0, 8, 1000)]})
    assert ei.value.in_flight
    assert ("unpin", {"pin_id": "p"}) in ctl.notified


def test_unit_reporting_the_blob_missing_is_loss():
    ctl = _Ctl({"resolve": lambda a: {"pin_id": "p", "offsets": {}, "missing": ["b"]}})
    with pytest.raises(LostBlobs):
        _store(ctl)._read_pinned({"b": [(0, 8, 1000)]})


def test_put_raises_unitfull_only_when_every_unit_is_full():
    s = _store(_Ctl({"alloc": lambda a: UnitFull("full")}))
    with pytest.raises(UnitFull):
        s.put_segments([(1000, 0, 8)], 8, 1, blob_id="b")
    s = _store(_Ctl({"alloc": lambda a: 0}), ep=_Ep(fail=TransferError("x")))
    with pytest.raises(TransferFailed):
        s.put_segments([(1000, 0, 8)], 8, 1, blob_id="b")


def test_put_waits_out_busy_units_and_times_out_at_its_deadline(monkeypatch):
    monkeypatch.setattr(blobstore, "BUSY_RETRY_S", 0.02)
    calls = []

    def busy(addr):
        calls.append(time.monotonic())
        return DataPlaneBusy("pressure", reason="pressure")

    s = _store(_Ctl({"alloc": busy}))
    t0 = time.monotonic()
    with pytest.raises(DataPlaneTimeout):
        s.put_segments([(1000, 0, 8)], 8, 1, blob_id="b", deadline=t0 + 0.3)
    assert time.monotonic() - t0 < 0.6
    assert len(calls) > 2  # retried, never UnitFull


def test_put_succeeds_once_busy_clears(monkeypatch):
    monkeypatch.setattr(blobstore, "BUSY_RETRY_S", 0.01)
    n = {"alloc": 0}

    def alloc(addr):
        n["alloc"] += 1
        return DataPlaneBusy("pressure") if n["alloc"] <= 3 else 64

    s = _store(_Ctl({"alloc": alloc}))
    blob_id, hint = s.put_segments([(1000, 0, 8)], 8, 1, blob_id="b")
    assert hint["so"] == 64


def test_ram_only_unit_is_busy_while_room_is_quarantined():
    u = bare_unit(8192, read_pin_s=0.2)
    write_blob(u.slab, u.alloc("a", 8192, 1), 8192, key_of("a"), 1)
    assert u.commit("a") == "done"
    with pytest.raises(UnitFull):  # nothing will come back
        u.alloc("b", 4096, 1)
    u.release_rows([gidx_of("a")])
    with pytest.raises(DataPlaneBusy):  # comes back after read_pin_s
        u.alloc("b", 4096, 1)
    with pytest.raises(UnitFull):  # bigger than the slab: never fits
        u.alloc("c", 16384, 1)
    time.sleep(0.25)
    assert u.alloc("b", 4096, 1) == 0


def test_deadline_spent_after_alloc_frees_the_region_at_once():
    ctl = _Ctl({"alloc": lambda a: 0})
    s = _store(ctl)
    with pytest.raises(DataPlaneTimeout):
        s.put_segments([(1000, 0, 8)], 8, 1, blob_id="b", deadline=time.monotonic())
    # The deadline check before alloc fires first, or the abort frees it.
    assert all(a.get("in_flight") is False for op, a in ctl.notified if op == "abort")


def test_unanswered_alloc_tells_the_unit_to_free_the_region():
    ctl = _Ctl({"alloc": lambda a: TransferError("did not answer alloc")})
    with pytest.raises(TransferFailed):
        _store(ctl).put_segments([(1000, 0, 8)], 8, 1, blob_id="b")
    aborts = [a for op, a in ctl.notified if op == "abort"]
    assert aborts and all(a == {"blob_id": "b", "in_flight": False} for a in aborts)


def test_unreachable_agent_after_alloc_frees_the_region():
    class _NoAgent(_Ep):
        def has_remote(self, name):
            return False

        def add_remote(self, md):
            raise RuntimeError("NIXL_ERR_REMOTE_DISCONNECT")

    ctl = _Ctl({"alloc": lambda a: 0})
    with pytest.raises(TransferFailed):
        _store(ctl, ep=_NoAgent()).put_segments([(1000, 0, 8)], 8, 1, blob_id="b")
    aborts = [a for op, a in ctl.notified if op == "abort"]
    assert len(aborts) == ctl.calls.count("alloc")
    assert all(a == {"blob_id": "b", "in_flight": False} for a in aborts)


# ------------------------------------------------------------------ config (§3.3)
def test_read_pin_must_cover_a_transfer_plus_margin():
    assert tq_nixl.check_timeouts({}) == (120.0, 150.0, 240.0)
    with pytest.raises(ValueError, match="read_pin_s"):
        tq_nixl.check_timeouts({"read_pin_s": 60.0})  # today's old default
    assert tq_nixl.check_timeouts(
        {"read_pin_s": 4.0, "transfer_timeout_s": 3.0, "lease_margin_s": 1.0}
    ) == (3.0, 4.0, 6.0)


@pytest.mark.parametrize("unset", [None, "absent"])
def test_unset_pin_and_op_deadline_follow_the_transfer_timeout(unset):
    cfg = {"transfer_timeout_s": 300.0}
    if unset != "absent":  # what the schema defaults hand over
        cfg["read_pin_s"] = cfg["op_timeout_s"] = unset
    assert tq_nixl.check_timeouts(cfg) == (300.0, 330.0, 600.0)


# ------------------------------------------------------------------ Graveyard (§2.2, §2.4)
class _StuckAgent:
    def __init__(self):
        self.posted = 0
        self.released = 0

    def get_xfer_descs(self, descs, mem):
        return descs

    def initialize_xfer(self, op, ldesc, rdesc, remote_agent):
        self.posted += 1
        return object()

    def transfer(self, h):
        return "PROC"

    def check_xfer_state(self, h):
        return "PROC"

    def release_xfer_handle(self, h):
        self.released += 1


def _endpoint(agent):
    ep = object.__new__(NixlEndpoint)
    ep.name, ep.agent, ep.unfinished, ep.unfinished_handles = "ep", agent, [], []
    return ep


def test_graveyard_cap_poisons_the_process(monkeypatch):
    g = Graveyard(max_bytes=100)
    monkeypatch.setattr(nixl_io, "GRAVEYARD", g)
    agent = _StuckAgent()
    ep = _endpoint(agent)
    with pytest.raises(TransferError):
        ep.transfer("READ", [(1000, 64)], [(0, 64)], "peer", timeout_s=0.01)
    assert g.poisoned is None and g.bytes == 64
    # The caller catches the error; the process still must not go on.
    with pytest.raises(TransferError):
        ep.transfer("READ", [(2000, 64)], [(0, 64)], "peer", timeout_s=0.01)
    assert g.poisoned is not None
    with pytest.raises(DataPlaneFatal, match="POISONED"):
        ep.transfer("READ", [(3000, 8)], [(0, 8)], "peer", timeout_s=0.01)
    assert agent.posted == 2  # the third op never reached NIXL
    assert agent.released == 0


def test_graveyard_outlives_a_dropped_endpoint(monkeypatch):
    import gc

    g = Graveyard()
    monkeypatch.setattr(nixl_io, "GRAVEYARD", g)
    agent = _StuckAgent()
    ep = _endpoint(agent)
    with pytest.raises(TransferError):
        ep.transfer("WRITE", [(1000, 64)], [(0, 64)], "peer", timeout_s=0.01)
    del ep
    gc.collect()
    (held_agent, h, ldesc, rdesc, local) = g.entries[0]
    assert held_agent is agent and h is not None and local == [(1000, 64)]
    assert agent.released == 0
