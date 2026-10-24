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
"""Shared helpers for the NIXL data-plane tests."""

from __future__ import annotations


def make_tq_conf(
    *,
    num_units: int = 2,
    slab_bytes: int = 64 << 20,
    staging_bytes: int = 32 << 20,
    read_pin_s: float = 4.0,
    namespace: str = "nrl-nixl-test-ns",
):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "controller": {"polling_mode": True},
            "backend": {
                "storage_backend": "NixlStore",
                "NixlStore": {
                    "namespace": namespace,
                    "directory_name": "BlobDirectory",
                    "num_storage_units": num_units,
                    "unit_slab_bytes": slab_bytes,
                    "client_staging_bytes": staging_bytes,
                    "read_pin_s": read_pin_s,
                    # read_pin_s >= transfer_timeout_s + lease_margin_s
                    "transfer_timeout_s": read_pin_s - 1.0,
                    "lease_margin_s": 1.0,
                    "nixl": {"backend_name": "UCX", "backend_init_params": {}},
                },
            },
        }
    )


def dense(t):
    """Pinned TQ returns every non-scalar field as a jagged NestedTensor; tests
    stack uniform rows densely."""
    import torch

    if isinstance(t, torch.Tensor) and t.is_nested:
        return torch.stack(list(t.unbind()))
    return t


# ------------------------------------------------------------------ NIXL stubs
class StubCtl:
    """UnitClient stub: scripted alloc / resolve replies, records notifies."""

    timeout_ms = 30_000

    def __init__(self, replies):
        self.replies = replies
        self.calls = []
        self.notified = []

    def call(self, address, op, *, timeout_s=None, **args):
        self.calls.append(op)
        if op == "commit" and op not in self.replies:
            return "done"  # every put commits (publishes its rows)
        r = self.replies[op](address)
        if isinstance(r, Exception):
            raise r
        return r

    def notify(self, address, op, **args):
        self.notified.append((op, args))


class StubEp:
    def __init__(self, fail=None):
        self.fail = fail

    def has_remote(self, name):
        return True

    def transfer(self, op, local, remote, agent, *, timeout_s):
        if self.fail is not None:
            raise self.fail

    def transfer_many(self, ops, *, timeout_s):
        return [self.fail] * len(ops)


def stub_store(ctl, ep=None, dead=False, monkeypatch=None, ssd=False):
    from nemo_rl.data_plane.nixl.blobstore import UnitSlabStore

    s = object.__new__(UnitSlabStore)
    s.ssd = ssd
    s.ep = ep or StubEp()
    s.ctl = ctl
    s.timeout_s = 5.0
    s.namespace = "ns"
    s._epoch = 1
    s._units = {
        u: {
            "unit_id": u,
            "zmq": f"tcp://u{u}",
            "agent_name": f"unit-{u}-abc",
            "slab_base": 0,
        }
        for u in (0, 1)
    }
    s.stats = {"unit_refresh": 0}
    s._refresh_units = lambda: None
    s._unit_dead = lambda unit_id: dead
    return s


def bare_unit(
    slab_bytes: int,
    *,
    read_pin_s: float = 0.0,
    ssd_dir: str | None = None,
    staging_bytes: int = 0,
    pickup_s: float = 10.0,
):
    """A NixlStorageUnit without Ray, NIXL or ZMQ: allocator, blobs and SSD tier."""
    import threading

    import numpy as np

    from nemo_rl.data_plane import nixl_storage_unit

    cls = nixl_storage_unit.NixlStorageUnit.__ray_metadata__.modified_class
    u = cls.__new__(cls)
    u.unit_id = 0
    u.read_pin_s = read_pin_s
    u.slab = np.zeros(slab_bytes, dtype=np.uint8)
    u.slab_base = 0
    u.blobs = {}
    u.rows = {}
    u._pins = {}
    u._pin_exp = {}
    u.alloc_lease_s = None
    u.retired_bytes = 0
    u._released_held = {}
    u.checkpointing = False
    u.clears_during_save = 0
    u._mu = threading.RLock()
    u._init_ssd(slab_bytes, ssd_dir, staging_bytes, pickup_s)
    u._start_ticker()
    return u


def key_of(name: str, field: str = "x") -> str:
    """A TransferQueue key ``gidx@field`` derived from a test blob name."""
    return f"{int.from_bytes(name.encode(), 'little') % 10**9}@{field}"


def gidx_of(name: str) -> int:
    return int(key_of(name).split("@")[0])


def write_blob(slab, off: int, nbytes: int, key: str, fill: int) -> int:
    """Make ``slab[off:off+nbytes]`` a valid blob of one ``key``.

    The value bytes are ``fill``; the tail holds the blob's index and footer,
    as a client's WRITE would. Returns the value length (bytes that are ``fill``).
    """
    import struct

    import msgspec

    from nemo_rl.data_plane.nixl import blob_format as bf

    slab[off : off + nbytes] = fill
    probe = msgspec.msgpack.encode([[key, 0, nbytes, bf.KIND_NUMPY, "|u1", [nbytes]]])
    io = nbytes - bf.FOOTER_SIZE - len(probe)
    index = msgspec.msgpack.encode([[key, 0, io, bf.KIND_NUMPY, "|u1", [io]]])
    io = nbytes - bf.FOOTER_SIZE - len(index)
    index = msgspec.msgpack.encode([[key, 0, io, bf.KIND_NUMPY, "|u1", [io]]])
    assert io + len(index) + bf.FOOTER_SIZE == nbytes
    slab[off + io : off + io + len(index)] = list(index)
    footer = struct.pack(
        bf._FOOTER_FMT, bf.MAGIC, bf.VERSION, 0, io, len(index), bf.NO_TAG
    )
    slab[off + nbytes - bf.FOOTER_SIZE : off + nbytes] = list(footer)
    return io
