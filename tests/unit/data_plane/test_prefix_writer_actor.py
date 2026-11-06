# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Prefix writer transport preserves bytes and the durable acknowledgement boundary."""

from __future__ import annotations

import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.prefix_writer_actor import (
    G_DTYPES,
    PrefixPut,
    PrefixWriterService,
    RayPrefixWriterClient,
    pack_columns,
    unpack_columns,
)
from nemo_rl.data_plane.tq_token_sink import TQTokenSink
from nemo_rl.models.generation.vllm.config import GenerationCutControlConfig
from tests.unit.data_plane.test_direct_prefix_batch import MemoryClient, record


def packet(fields: TensorDict) -> PrefixPut:
    return PrefixPut(
        ("a", "b"), "prefix", pack_columns(fields), ({}, {}), "cut", "batch", 0.0
    )


@pytest.mark.parametrize("dtype", list(G_DTYPES.values()))
@pytest.mark.parametrize("jagged", [False, True])
def test_transport_preserves_dtype_shape_and_exact_bytes(
    dtype: torch.dtype, jagged: bool
) -> None:
    data = torch.arange(6, dtype=torch.int64).to(dtype)
    values = data.reshape(2, 3)
    if jagged:
        values = torch.nested.nested_tensor_from_jagged(
            data, offsets=torch.tensor([0, 2, 6])
        )
    fields = TensorDict({"values": values}, batch_size=[2])
    # A transport round trip owns its buffers independently of the sender.
    wire = pickle.loads(pickle.dumps(packet(fields), protocol=5))
    data.fill_(0)
    restored = unpack_columns(wire.columns, 2)["values"]
    actual = restored.values() if jagged else restored.reshape(-1)
    expected = torch.arange(6, dtype=torch.int64).to(dtype)
    assert actual.dtype == dtype
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    if jagged:
        assert restored.offsets().tolist() == [0, 2, 6]


def test_transport_keeps_float_bits_and_readonly_buffers() -> None:
    values = torch.tensor(
        [[-0.0, float("inf")], [float("nan"), -1e-45]], dtype=torch.float32
    )
    fields = TensorDict({"x": values}, batch_size=[2])
    wire = packet(fields)
    wire.columns["x"].data.flags.writeable = False
    with pytest.warns(UserWarning, match="not writable"):
        restored = unpack_columns(wire.columns, 2)
    assert np.array_equal(
        restored["x"].view(torch.uint8).numpy(), values.view(torch.uint8).numpy()
    )


@pytest.mark.parametrize("corruption", ["size", "dtype", "rows", "offsets"])
def test_transport_rejects_invalid_layout(corruption: str) -> None:
    fields = TensorDict({"x": torch.ones(2, 3)}, batch_size=[2])
    wire = packet(fields)
    column = wire.columns["x"]
    if corruption == "size":
        column = replace(column, data=column.data[:-1])
    elif corruption == "dtype":
        column = replace(column, dtype="torch.missing")
    elif corruption == "rows":
        column = replace(column, shape=(3, 2))
    else:
        column = replace(
            column, shape=(6,), offsets=np.array([0, 5, 4], dtype=np.int64)
        )
    with pytest.raises(ValueError):
        unpack_columns({"x": column}, 2)


def test_service_waits_for_durability_and_propagates_failure() -> None:
    entered, release = threading.Event(), threading.Event()

    class BlockingClient:
        def put_samples(self, **kwargs: Any) -> None:
            entered.set()
            assert release.wait(5)
            raise RuntimeError("injected storage failure")

    service = PrefixWriterService(BlockingClient(), partition_id="prefix")
    wire = packet(TensorDict({"x": torch.ones(2, 3)}, batch_size=[2]))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(service.put, wire)
        assert entered.wait(5)
        assert not future.done()
        release.set()
        with pytest.raises(RuntimeError, match="injected storage failure"):
            future.result(timeout=5)


def test_service_rejects_other_partition() -> None:
    service = PrefixWriterService(MemoryClient(), partition_id="other")
    with pytest.raises(ValueError, match="partition"):
        service.put(packet(TensorDict({"x": torch.ones(2, 3)}, batch_size=[2])))


def test_prefix_redirection_does_not_move_terminal_writes_or_cleanup() -> None:
    local, remote = MemoryClient(), MemoryClient()
    sink = TQTokenSink(local, staging_partition="prefix")
    sink.enable_direct_prefix_batch_encoding()
    sink.configure_prefix_writer(remote, staging_partition="prefix")
    row = record(0, 3)
    keys = sink.stage_generation_prefix_batch(
        [row], checkpoint_id="cut", chunk_sequences=[0]
    )
    assert len(remote.puts) == 1 and not local.puts
    assert sink.stage(row).ok
    assert len(local.puts) == 1 and len(remote.puts) == 1
    sink.clear(keys)
    assert local.cleared == keys and not remote.cleared


@pytest.mark.parametrize(
    "options",
    [
        {"prefix_writer_actor": True},
        {"prefix_writer_actor": True, "pipeline": {"enabled": True}},
        {"prefix_writer_actor": True, "direct_prefix_batch_encoding": True},
    ],
)
def test_actor_configuration_rejects_incomplete_setup(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="requires"):
        GenerationCutControlConfig.model_validate(options)


def test_prefix_facade_uses_packed_buffers_and_checks_ack_ids(monkeypatch: Any) -> None:
    import nemo_rl.data_plane.prefix_writer_actor as transport

    backend = MemoryClient()
    service = PrefixWriterService(backend, partition_id="prefix")

    class Call:
        def remote(self, wire: PrefixPut) -> tuple[str, ...]:
            assert all(isinstance(c.data, np.ndarray) for c in wire.columns.values())
            return service.put(wire)

    class Actor:
        put = Call()

    monkeypatch.setattr(
        transport.ray, "put", lambda wire: pickle.loads(pickle.dumps(wire, protocol=5))
    )
    monkeypatch.setattr(transport.ray, "get", lambda value: value)
    client = RayPrefixWriterClient(Actor())
    client.put_samples(
        sample_ids=["a", "b"],
        partition_id="prefix",
        fields=TensorDict({"x": torch.ones(2, 3)}, batch_size=[2]),
        tags=[{}, {}],
    )
    assert backend.puts[0]["sample_ids"] == ["a", "b"]
    monkeypatch.setattr(Call, "remote", lambda self, wire: ("wrong",))
    with pytest.raises(RuntimeError, match="different durable IDs"):
        client.put_samples(
            sample_ids=["a", "b"],
            partition_id="prefix",
            fields=TensorDict({"x": torch.ones(2, 3)}, batch_size=[2]),
            tags=[{}, {}],
        )
