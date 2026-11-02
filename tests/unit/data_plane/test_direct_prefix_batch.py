# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Direct prefix columns preserve custody, wire bytes, and fallback behavior."""

from __future__ import annotations

import importlib.util
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from nemo_gym.token_id_capture.staging.digest import (
    compute_chain_hash,
    compute_extras_digest,
    hash_token_ids,
)
from nemo_gym.token_id_capture.staging.records import StagedCallRecord

from nemo_rl.data_plane.direct_prefix_batch import encode_text_generation_prefix_batch
from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
from nemo_rl.models.generation.vllm.config import GenerationCutControlConfig
from nemo_rl.utils.prefix_cut_diagnostics import (
    current_prefix_cut_probe,
    prefix_cut_probe,
)
from nemo_rl.utils.routed_experts_codec import encode_routed_experts


class MemoryClient:
    def __init__(self, *, fail: bool = False, serial: Any = None) -> None:
        self.puts: list[dict[str, Any]] = []
        self.cleared: list[str] = []
        self.fail = fail
        self.serial = serial

    def put_samples(self, **kwargs: Any) -> None:
        if self.serial is not None:
            # Real TQ codec, with copied frames as on a transport boundary.
            frames = self.serial.encode(kwargs["fields"])
            kwargs["fields"] = self.serial.decode([bytes(frame) for frame in frames])
        self.puts.append(kwargs)
        if self.fail:
            raise RuntimeError("storage failed after partial publication")

    def get_samples(self, **kwargs: Any) -> Any:
        assert len(self.puts) == 1
        assert kwargs["sample_ids"] == self.puts[0]["sample_ids"]
        return self.puts[0]["fields"].select(*kwargs["select_fields"])

    def clear_samples(self, **kwargs: Any) -> None:
        self.cleared.extend(kwargs["sample_ids"])


def record(
    index: int, length: int, *, extras: dict[str, Any] | None = None
) -> StagedCallRecord:
    ids = list(range(100 + index, 100 + index + length))
    if index == 0:
        ids[0] = 2**62 + 13
    child = index % 2 == 1
    prev = 3 if child else 0
    values = [-0.0, -0.1, -1e-45, -1.25]
    return StagedCallRecord.from_components(
        rollout_id=f"rollout-{index}-é" + "x" * index,
        model_call_id=f"call-{index}",
        parent_call_id=f"parent-{index}" if child else None,
        mode="token_in" if child else "text",
        prev_len=prev,
        delta_len=length,
        cum_len=prev + length,
        weight_version=index % 4,
        token_ids_delta=ids,
        token_mask_delta=[1.0] * length,
        generation_log_probs_delta=[values[i % len(values)] for i in range(length)],
        extras=extras,
        extras_digest=compute_extras_digest(extras),
        chain_hash=compute_chain_hash(None, ids),
        cumulative_hash=hash_token_ids([42] * prev + ids),
    )


def stage(
    records: list[StagedCallRecord], *, direct: bool, client: MemoryClient | None = None
) -> tuple[MemoryClient, list[str]]:
    client = MemoryClient() if client is None else client
    sink = TQTokenSink(client, staging_partition="prefix")
    if direct:
        sink.enable_direct_prefix_batch_encoding()
    keys = sink.stage_generation_prefix_batch(
        records,
        checkpoint_id="checkpoint-7",
        chunk_sequences=list(range(10, 10 + len(records))),
    )
    return client, keys


def assert_equal_batches(left: dict[str, Any], right: dict[str, Any]) -> int:
    assert left["sample_ids"] == right["sample_ids"]
    assert left["tags"] == right["tags"]
    assert left["partition_id"] == right["partition_id"]
    a, b = left["fields"], right["fields"]
    assert a.batch_size == b.batch_size and set(a.keys()) == set(b.keys())
    payload_bytes = 0
    for name in a.keys():
        x, y = a[name], b[name]
        assert x.dtype == y.dtype and x.is_nested == y.is_nested, name
        if x.is_nested:
            assert torch.equal(x.offsets(), y.offsets()), name
        for i in range(a.batch_size[0]):
            first, second = x[i].contiguous(), y[i].contiguous()
            assert first.shape == second.shape, name
            raw = first.numpy().tobytes()
            assert raw == second.numpy().tobytes(), (name, i)
            payload_bytes += len(raw)
    return payload_bytes


@pytest.mark.parametrize(
    "lengths", [[1], [4, 4, 4], [1, 3, 17, 5], [1, 2048, 8192, 65536]]
)
def test_matches_general_fields_and_normal_restore_digest(lengths: list[int]) -> None:
    records = [
        record(
            i,
            length,
            extras={"note": "β", "nested": [i, True, None]} if i % 2 else None,
        )
        for i, length in enumerate(lengths)
    ]
    old, old_keys = stage(records, direct=False)
    new, new_keys = stage(records, direct=True)
    assert old_keys == new_keys
    assert_equal_batches(old.puts[0], new.puts[0])
    restored = TQTokenSource(new, staging_partition="prefix").fetch(new_keys)
    assert [r.digest for r in restored] == [r.digest for r in records]


def test_buffers_are_owned_and_retain_float_bits() -> None:
    records = [record(0, 4), record(1, 17)]
    client, keys = stage(records, direct=True)
    original = client.puts[0]["fields"]["token_ids_delta"][0].clone()
    records[0].token_ids_delta[0] = 9
    records[0].generation_log_probs_delta[0] = -99.0
    records[0].token_mask_delta[0] = 0.0
    batch = client.puts[0]["fields"]
    assert torch.equal(batch["token_ids_delta"][0], original)
    assert batch["generation_logprobs_delta"][0][0].signbit().item()
    assert batch["token_mask_delta"][0][0].item() == 1.0
    TQTokenSource(client, staging_partition="prefix").fetch(keys)


def test_default_stays_general_and_opt_in_has_exact_payload_counters() -> None:
    assert GenerationCutControlConfig().direct_prefix_batch_encoding is False
    records = [record(0, 3), record(1, 17)]
    counters = []
    puts = []
    for enabled in (False, True):
        with prefix_cut_probe(
            enabled=True,
            checkpoint_id="cp",
            inventory_digest="d",
            calls=2,
            deadline=None,
        ):
            client, _ = stage(records, direct=enabled)
            puts.append(client.puts[0])
            probe = current_prefix_cut_probe()
            assert probe is not None
            counters.append(probe.summary("completed")["counts"])
    actual_bytes = assert_equal_batches(*puts)
    for item in counters:
        assert (
            item["tensor_payload_bytes"]
            == item["max_batch_tensor_payload_bytes"]
            == actual_bytes
        )
        assert item["tq_batches"] == 1
    assert "direct_batch_encoder_batches" not in counters[0]
    assert counters[1]["direct_batch_encoder_batches"] == 1
    assert counters[1]["direct_batch_encoder_rows"] == 2


@pytest.mark.parametrize("kind", ["routes", "media", "null-route-key"])
def test_unsupported_batches_use_general_encoder(monkeypatch: Any, kind: str) -> None:
    import nemo_rl.data_plane.tq_token_sink as module

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Unsupported batch reached the direct encoder")

    monkeypatch.setattr(module, "encode_text_generation_prefix_batch", unexpected)
    extras = None
    if kind == "routes":
        extras = {
            "routed_experts": encode_routed_experts(
                torch.arange(12, dtype=torch.int16).reshape(3, 2, 2)
            )
        }
    elif kind == "null-route-key":
        extras = {"routed_experts": None, "other": 7}
    records = [record(0, 3, extras=extras)]
    clients = []
    for enabled in (False, True):
        client = MemoryClient()
        sink = TQTokenSink(
            client,
            staging_partition="prefix",
            capture_media=kind == "media",
            media_pixel_dtype=torch.float32 if kind == "media" else None,
        )
        if enabled:
            sink.enable_direct_prefix_batch_encoding()
        sink.stage_generation_prefix_batch(
            records, checkpoint_id="cp", chunk_sequences=[0]
        )
        clients.append(client)
    assert_equal_batches(clients[0].puts[0], clients[1].puts[0])


@pytest.mark.parametrize("enabled", [False, True])
def test_failed_write_clears_every_possibly_published_key(enabled: bool) -> None:
    records = [record(0, 3), record(1, 17)]
    client = MemoryClient(fail=True)
    with pytest.raises(RuntimeError, match="storage failed"):
        stage(records, direct=enabled, client=client)
    assert client.cleared == client.puts[0]["sample_ids"]


def test_rejects_duplicate_keys_mismatched_sequences_and_media_attachments() -> None:
    client = MemoryClient()
    sink = TQTokenSink(client, staging_partition="prefix")
    sink.enable_direct_prefix_batch_encoding()
    item = record(0, 3)
    with pytest.raises(ValueError, match="duplicate staging keys"):
        sink.stage_generation_prefix_batch(
            [item, item], checkpoint_id="cp", chunk_sequences=[0, 0]
        )
    with pytest.raises(ValueError, match="equal lengths"):
        sink.stage_generation_prefix_batch(
            [item], checkpoint_id="cp", chunk_sequences=[]
        )
    with pytest.raises(ValueError, match="media attachments require"):
        sink.stage_generation_prefix_batch(
            [item],
            checkpoint_id="cp",
            chunk_sequences=[0],
            attachments=[{"imgs": torch.zeros(1, 1, 3)}],
        )
    assert (
        sink.stage_generation_prefix_batch([], checkpoint_id="cp", chunk_sequences=[])
        == []
    )
    assert not client.puts


def test_standalone_encoder_rejects_empty_routes_and_bad_lengths() -> None:
    with pytest.raises(ValueError, match="at least one"):
        encode_text_generation_prefix_batch([], mode_codes={"text": 0})
    routed = record(0, 3, extras={"routed_experts": None})
    with pytest.raises(ValueError, match="routed experts"):
        encode_text_generation_prefix_batch([routed], mode_codes={"text": 0})
    broken = record(0, 3)
    broken.token_ids_delta.pop()
    with pytest.raises(ValueError, match="lengths differ"):
        encode_text_generation_prefix_batch([broken], mode_codes={"text": 0})


def test_concurrent_batches_keep_their_own_buffers() -> None:
    def execute(i: int) -> int:
        records = [record(i * 2, 17 + i), record(i * 2 + 1, 31)]
        client, keys = stage(records, direct=True)
        restored = TQTokenSource(client, staging_partition="prefix").fetch(keys)
        assert [x.digest for x in restored] == [x.digest for x in records]
        return len(restored)

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(execute, range(8))) == [2] * 8


def test_actual_pinned_tq_wire_codec_roundtrip(monkeypatch: Any) -> None:
    source = os.environ.get("NRL_TEST_TQ_SERIAL_UTILS_SOURCE")
    if source is None:
        pytest.skip("Set NRL_TEST_TQ_SERIAL_UTILS_SOURCE to the pinned TQ codec")
    path = Path(source)
    # Load the unchanged codec/logger without importing TQ's Ray/backend SDKs.
    # This keeps the transport-byte test usable in the CPU-only test lane.
    for name, directory in (
        ("transfer_queue", path.parent.parent),
        ("transfer_queue.utils", path.parent),
    ):
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        monkeypatch.setitem(sys.modules, name, package)
    logger_name = "transfer_queue.utils.logging_utils"
    logger_spec = importlib.util.spec_from_file_location(
        logger_name, path.with_name("logging_utils.py")
    )
    assert logger_spec is not None and logger_spec.loader is not None
    logger_module = importlib.util.module_from_spec(logger_spec)
    monkeypatch.setitem(sys.modules, logger_name, logger_module)
    logger_spec.loader.exec_module(logger_module)
    spec = importlib.util.spec_from_file_location("_direct_prefix_tq_serial", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = [record(i, n) for i, n in enumerate([1, 3, 17, 2048])]
    old, _ = stage(records, direct=False, client=MemoryClient(serial=module))
    new, keys = stage(records, direct=True, client=MemoryClient(serial=module))
    assert_equal_batches(old.puts[0], new.puts[0])
    restored = TQTokenSource(new, staging_partition="prefix").fetch(keys)
    assert [x.digest for x in restored] == [r.digest for r in records]
