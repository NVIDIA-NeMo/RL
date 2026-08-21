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
"""Register mode: what put publishes, what get moves, what clear releases.

The Transfer Engine is faked — ``register_memory`` records the range and
``batch_transfer_sync_read`` performs the pull with ``memmove``. That is
faithful to what the real binding does to memory (a one-sided READ lands the
producer's bytes in the consumer's registered buffer), so address publication,
base dedupe, endpoint grouping, receive-buffer registration and refcounted
release all run exactly as in production without an RDMA NIC.

The GDR half of the contract — CUDA source registered in place, CUDA
destination — is asserted through the same fake on a real GPU where one is
available, and through the config plumbing everywhere else.
"""

from __future__ import annotations

import ctypes
import time
from typing import Any

import pytest
import torch

from nemo_rl.data_plane.adapters import tq_register_mode
from nemo_rl.data_plane.adapters.tq_register_mode import TransferEngineClient

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GDR paths need a GPU"
)


class FakeEngine:
    """In-process stand-in for ``mooncake.engine.TransferEngine``.

    ``move_bytes`` emulates the one-sided READ with ``memmove``, which is only
    valid for host pointers. The GDR tests turn it off and assert on the
    recorded descriptors instead — the addresses are the thing under test
    there, and a real transfer needs a real NIC.
    """

    _next_port = 20000
    move_bytes = True

    def __init__(self) -> None:
        self.registered: dict[int, int] = {}
        self.read_calls: list[tuple[str, int]] = []
        self.transfers: list[tuple[int, int, int]] = []
        FakeEngine._next_port += 1
        self._port = FakeEngine._next_port

    def initialize(self, local_server_name, metadata_server, protocol, device_name):
        self.local_server_name = local_server_name
        self.protocol = protocol
        self.device_name = device_name
        return 0

    def get_rpc_port(self) -> int:
        return self._port

    def register_memory(self, addr: int, size: int, location: str = "*") -> int:
        assert addr not in self.registered, "overlapping registration"
        self.registered[addr] = size
        return 0

    def unregister_memory(self, addr: int) -> int:
        self.registered.pop(addr, None)
        return 0

    def batch_register_memory(self, addrs, sizes, location: str = "*") -> int:
        for addr, size in zip(addrs, sizes, strict=True):
            self.register_memory(addr, size)
        return 0

    def batch_unregister_memory(self, addrs) -> int:
        for addr in addrs:
            self.unregister_memory(addr)
        return 0

    def batch_transfer_sync_read(
        self,
        target_hostname,
        buffers,
        peer_buffer_addresses,
        lengths,
        transport_hint="",
    ) -> int:
        self.read_calls.append((target_hostname, len(buffers)))
        for dst, src, size in zip(
            buffers, peer_buffer_addresses, lengths, strict=True
        ):
            self.transfers.append((dst, src, size))
            if self.move_bytes:
                ctypes.memmove(dst, src, size)
        return 0


@pytest.fixture(autouse=True)
def _fake_engine(monkeypatch):
    monkeypatch.setattr(tq_register_mode, "TransferEngine", FakeEngine, raising=False)
    monkeypatch.setattr(tq_register_mode, "TRANSFER_ENGINE_IMPORTED", True)


def _make_client(
    *, use_gdr: bool = False, offload_source_to_host: bool = False
) -> TransferEngineClient:
    return TransferEngineClient(
        {
            "local_hostname": "127.0.0.1",
            "rpc_port": 0,
            "protocol": "rdma",
            "device_name": "mlx5_0",
            "metadata_server": "P2PHANDSHAKE",
            "use_gdr": use_gdr,
            "offload_source_to_host": offload_source_to_host,
        }
    )


def _batch_rows(
    n: int = 4, width: int = 8, device: str = "cpu"
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """A batched allocation plus its per-row views — what TQ hands ``put``."""
    flat = torch.arange(n * width, dtype=torch.float32, device=device)
    batch = flat.reshape(n, width)
    return batch, [batch[i] for i in range(n)]


def _get_args(rows: list[torch.Tensor], meta: list[dict]) -> dict[str, Any]:
    return {
        "shapes": [tuple(row.shape) for row in rows],
        "dtypes": [row.dtype for row in rows],
        "custom_backend_meta": meta,
    }


def test_put_publishes_row_addresses_without_copying():
    client = _make_client()
    batch, rows = _batch_rows()
    keys = [f"{i}@input_ids" for i in range(len(rows))]

    meta = client.put(keys, rows)

    # One registration for the shared allocation, not one per row: the engine
    # rejects overlapping regions, and pinning is the expensive part.
    assert list(client._engine.registered) == [batch.untyped_storage().data_ptr()]
    # Every row is published in place — base + offset is the row's own address.
    for row, entry in zip(rows, meta, strict=True):
        assert entry["base"] + entry["offset"] == row.data_ptr()
        assert entry["size"] == row.nbytes
        assert entry["endpoint"] == client.endpoint


def test_get_from_own_endpoint_skips_the_fabric():
    client = _make_client()
    _, rows = _batch_rows()
    keys = [f"{i}@input_ids" for i in range(len(rows))]

    meta = client.put(keys, rows)
    values = client.get(keys, **_get_args(rows, meta))

    for row, value in zip(rows, values, strict=True):
        torch.testing.assert_close(value, row)
    assert client._engine.read_calls == []


def test_get_pulls_from_a_remote_producer():
    producer = _make_client()
    consumer = _make_client()
    _, rows = _batch_rows()
    keys = [f"{i}@input_ids" for i in range(len(rows))]

    meta = producer.put(keys, rows)
    values = consumer.get(keys, **_get_args(rows, meta))

    for row, value in zip(rows, values, strict=True):
        torch.testing.assert_close(value, row)
    assert consumer._engine.read_calls == [(producer.endpoint, len(rows))]
    # Receive regions are registered only for the duration of the read.
    assert consumer._engine.registered == {}


def test_non_tensor_values_roundtrip():
    client = _make_client()
    keys = ["0@extra_info", "1@extra_info"]
    values = [{"task": "math", "score": 0.5}, ["a", "b", "c"]]

    meta = client.put(keys, values)
    decoded = client.get(
        keys, shapes=[[], []], dtypes=[None, None], custom_backend_meta=meta
    )

    assert decoded == values


def test_clear_releases_the_registration_once_every_key_is_gone():
    client = _make_client()
    _, rows = _batch_rows()
    keys = [f"{i}@input_ids" for i in range(len(rows))]

    meta = client.put(keys, rows)
    client.clear(keys[:-1], custom_backend_meta=meta[:-1])
    # Rows share one registration; it survives until the last key is cleared.
    assert len(client._engine.registered) == 1

    client.clear(keys[-1:], custom_backend_meta=meta[-1:])
    assert client._engine.registered == {}


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    """Poll ``predicate`` — releases cross a socket, so they are not synchronous."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_clear_by_a_consumer_releases_the_producers_registration():
    """The leak fix: only the owner can unpin, so ``clear`` must reach it.

    ``clear`` runs on the driver in both trainers, so without this a producer's
    registrations accumulate for the life of the process — every step, forever.
    """
    producer = _make_client()
    consumer = _make_client()
    _, rows = _batch_rows()
    keys = [f"{i}@input_ids" for i in range(len(rows))]

    meta = producer.put(keys, rows)
    assert len(producer._engine.registered) == 1
    assert producer.pinned_keys == len(keys)

    # A different client clears — exactly what the driver does at step end.
    consumer.clear(keys, custom_backend_meta=meta)

    assert _wait_until(lambda: producer._engine.registered == {})
    assert producer.pinned_keys == 0


def test_a_replayed_clear_cannot_release_a_republished_key():
    """The sharp edge: TQ recycles ``<global_idx>@<field>`` keys across steps.

    A stale clear naming keys that have since been republished must not unpin
    the new registration — that would be a use-after-unregister, with consumers
    reading an address the NIC no longer maps. Matching on the key alone does
    not catch it, and neither does matching on the address: the caching
    allocator commonly hands back the same block.
    """
    producer = _make_client()
    consumer = _make_client()
    _, rows = _batch_rows()
    keys = [f"{i}@input_ids" for i in range(len(rows))]

    stale_meta = producer.put(keys, rows)
    consumer.clear(keys, custom_backend_meta=stale_meta)
    assert _wait_until(lambda: producer.pinned_keys == 0)

    # Same keys, same rows — so the same base address comes back.
    fresh_meta = producer.put(keys, rows)
    assert fresh_meta[0]["base"] == stale_meta[0]["base"]
    assert fresh_meta[0]["seq"] != stale_meta[0]["seq"]

    consumer.clear(keys, custom_backend_meta=stale_meta)  # replay
    time.sleep(0.3)
    assert producer.pinned_keys == len(keys), "replay released a live publication"
    assert len(producer._engine.registered) == 1

    consumer.clear(keys, custom_backend_meta=fresh_meta)
    assert _wait_until(lambda: producer._engine.registered == {})


def test_republishing_a_key_releases_the_value_it_replaced():
    """An upsert TQ never cleared must not strand the old registration."""
    producer = _make_client()
    keys = ["0@input_ids"]

    first = torch.arange(8, dtype=torch.float32)
    second = torch.arange(8, dtype=torch.float32) + 100
    producer.put(keys, [first])
    assert len(producer._engine.registered) == 1
    producer.put(keys, [second])

    # One live publication, so one registration — not two.
    assert producer.pinned_keys == 1
    assert len(producer._engine.registered) == 1
    assert list(producer._engine.registered) == [second.untyped_storage().data_ptr()]


def test_release_to_a_departed_owner_does_not_block():
    """A producer that already exited must not stall the caller of clear."""
    producer = _make_client()
    consumer = _make_client()
    _, rows = _batch_rows()
    keys = [f"{i}@input_ids" for i in range(len(rows))]
    meta = producer.put(keys, rows)
    producer.close()

    started = time.monotonic()
    consumer.clear(keys, custom_backend_meta=meta)
    assert time.monotonic() - started < 3.0


def test_get_rejects_a_key_that_was_never_published():
    client = _make_client()
    _, rows = _batch_rows(n=2)
    keys = ["0@input_ids", "1@input_ids"]
    meta = client.put(keys, rows)

    with pytest.raises(ValueError, match="never published"):
        client.get(keys, **_get_args(rows, [meta[0], None]))


def test_get_rejects_a_size_mismatch_between_schema_and_publication():
    client = _make_client()
    _, rows = _batch_rows(n=2)
    keys = ["0@input_ids", "1@input_ids"]
    meta = client.put(keys, rows)

    args = _get_args(rows, meta)
    args["shapes"] = [(rows[0].numel() + 1,), tuple(rows[1].shape)]
    with pytest.raises(ValueError, match="was published as"):
        client.get(keys, **args)


def test_close_unregisters_everything_still_pinned():
    client = _make_client()
    _, rows = _batch_rows()
    engine = client._engine
    client.put([f"{i}@input_ids" for i in range(len(rows))], rows)

    client.close()

    assert engine.registered == {}


def test_receive_device_is_host_without_gdr():
    """A CPU-only client shares the cluster's config and must not claim HBM."""
    assert _make_client(use_gdr=False).receive_device.type == "cpu"


@requires_cuda
def test_gdr_registers_the_cuda_source_in_place():
    """The point of register mode under GDR: no D2H on the put path."""
    torch.cuda.init()
    client = _make_client(use_gdr=True)
    batch, rows = _batch_rows(device="cuda")
    keys = [f"{i}@input_ids" for i in range(len(rows))]

    meta = client.put(keys, rows)

    assert list(client._engine.registered) == [batch.untyped_storage().data_ptr()]
    for row, entry in zip(rows, meta, strict=True):
        # The published address IS the CUDA tensor's address.
        assert entry["base"] + entry["offset"] == row.data_ptr()


@requires_cuda
def test_gdr_get_reads_device_to_device(monkeypatch):
    """Source and destination are both HBM addresses — no host bounce anywhere."""
    torch.cuda.init()
    monkeypatch.setattr(FakeEngine, "move_bytes", False)
    producer = _make_client(use_gdr=True)
    consumer = _make_client(use_gdr=True)
    _, rows = _batch_rows(device="cuda")
    keys = [f"{i}@input_ids" for i in range(len(rows))]

    meta = producer.put(keys, rows)
    values = consumer.get(keys, **_get_args(rows, meta))

    assert consumer.receive_device.type == "cuda"
    assert all(value.is_cuda for value in values)
    transfers = {
        src: (dst, size) for dst, src, size in consumer._engine.transfers
    }
    for row, value in zip(rows, values, strict=True):
        # Pulled straight from the producer's tensor into the consumer's.
        dst, size = transfers[row.data_ptr()]
        assert dst == value.data_ptr()
        assert size == row.nbytes


@requires_cuda
def test_offloading_the_source_registers_host_memory_but_still_receives_in_hbm():
    """CPU-resident source, HBM destination: one hop, no producer HBM held.

    This is the mixed shape — the producer hands its bytes to host memory and
    consumers still pull straight into their own HBM, which is what makes the
    HBM-residency cost avoidable without giving up the single-hop landing.
    """
    torch.cuda.init()
    client = _make_client(use_gdr=True, offload_source_to_host=True)
    _, rows = _batch_rows(device="cuda")
    keys = [f"{i}@input_ids" for i in range(len(rows))]

    meta = client.put(keys, rows)

    # Published addresses are host addresses, not the CUDA tensors'.
    for row, entry in zip(rows, meta, strict=True):
        assert entry["base"] + entry["offset"] != row.data_ptr()
    # ...while reads still land in device memory.
    assert client.receive_device.type == "cuda"
    values = client.get(keys, **_get_args(rows, meta))
    for row, value in zip(rows, values, strict=True):
        assert value.is_cuda
        torch.testing.assert_close(value, row)


@requires_cuda
def test_non_tensor_payloads_stay_on_the_host_under_gdr():
    """msgpack buffers are decoded on the CPU; sending them to HBM helps nobody."""
    torch.cuda.init()
    client = _make_client(use_gdr=True)
    keys = ["0@extra_info"]
    values = [{"task": "math"}]

    meta = client.put(keys, values)
    decoded = client.get(
        keys, shapes=[[]], dtypes=[None], custom_backend_meta=meta
    )

    assert decoded == values
