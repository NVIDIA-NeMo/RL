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

import json
import threading
from concurrent.futures import Future
from contextlib import contextmanager
from unittest.mock import Mock, patch

import pytest
import torch

from nemo_rl.utils import weight_transfer_protocol as protocol_mod
from nemo_rl.utils import weight_transfer_sparse_codec as sparse_codec_mod
from nemo_rl.utils.weight_transfer import (
    G_DELTA_UPDATE_KIND,
    G_DENSE_TRANSPORT,
    G_FULL_UPDATE_KIND,
    G_SPARSE_INDICES_TRANSPORT,
    G_TRANSFER_DONE_KIND,
    pack_named_tensors,
    packed_weight_transfer_consumer,
    packed_weight_transfer_producer,
    unpack_named_tensors,
)
from nemo_rl.utils.weight_transfer_delta_tracker import (
    DeltaCompressionTracker,
    _baseline_mmap_pending_bytes,
    _baseline_mmap_write_workers,
    create_vllm_delta_transfer_tracker,
)
from nemo_rl.utils.weight_transfer_protocol import additive_weight_load_context


class RecordingGroup:
    def __init__(self) -> None:
        self.rank = 0
        self.broadcasted_tensors: list[torch.Tensor] = []

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        self.broadcasted_tensors.append(tensor.clone())


class ReplayGroup:
    def __init__(self, broadcasted_tensors: list[torch.Tensor]) -> None:
        self.rank = 1
        self.broadcasted_tensors = broadcasted_tensors
        self.current_index = 0

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        tensor.copy_(self.broadcasted_tensors[self.current_index])
        self.current_index += 1


class FailingGroup:
    def __init__(self) -> None:
        self.rank = 0

    def broadcast(self, _tensor: torch.Tensor, src: int) -> None:
        raise RuntimeError("simulated broadcast failure")


class CountingIterator:
    def __init__(self, items):
        self._items = iter(items)
        self.names = []

    def __iter__(self):
        return self

    def __next__(self):
        name, tensor = next(self._items)
        self.names.append(name)
        return name, tensor


def _recorded_headers(broadcasted_tensors: list[torch.Tensor]) -> list[dict]:
    headers = []
    index = 0
    while index < len(broadcasted_tensors):
        kind, transport, payload_numel, metadata_len = (
            protocol_mod.decode_header_control(broadcasted_tensors[index])
        )
        index += 1
        header = {
            "kind": kind,
            "transport": transport,
            "payload_entries": [],
            "payload_numel": payload_numel,
            "sparse_metadata": [],
        }
        if metadata_len > 0:
            metadata_tensor = broadcasted_tensors[index]
            header.update(json.loads(metadata_tensor.numpy().tobytes().decode("utf-8")))
            index += 1
        headers.append(header)
        if header["kind"] != G_TRANSFER_DONE_KIND and int(header["payload_numel"]) > 0:
            index += 1
    return headers


def _tracker_config(
    full_sync_interval: int = 20,
    sparse_bucket_size_bytes: int = 1024,
    delta_load_batch_size_bytes: int = 1024,
) -> dict[str, object]:
    return {
        "enabled": True,
        "dtype": "float32",
        "full_sync_interval": full_sync_interval,
        "sparse_bucket_size_bytes": sparse_bucket_size_bytes,
        "delta_load_batch_size_bytes": delta_load_batch_size_bytes,
    }


@contextmanager
def _cpu_transfer_env(chunk_size: int = 1024):
    with (
        patch(
            "nemo_rl.utils.weight_transfer.get_target_packed_tensor_size",
            return_value=chunk_size,
        ),
        patch(
            "nemo_rl.utils.weight_transfer.torch.cuda.is_available",
            return_value=False,
        ),
    ):
        yield


def _state_loaders(
    receiver_state: dict[str, torch.Tensor],
    *,
    delta_batches: list[list[str]] | None = None,
    counters: dict[str, int] | None = None,
):
    def load_full(weights):
        if counters is not None:
            counters["full"] += 1
        for name, tensor in weights:
            receiver_state[name] = tensor.clone()

    def load_delta(weights):
        if counters is not None:
            counters["delta"] += 1
        if delta_batches is not None:
            delta_batches.append([name for name, _ in weights])
        for name, tensor in weights:
            receiver_state[name].add_(tensor)

    return load_full, load_delta


def _run_transfer(
    weights,
    *,
    tracker=None,
    load_full=None,
    load_delta=None,
    delta_load_batch_size_bytes: int | None = None,
):
    producer_group = RecordingGroup()
    packed_weight_transfer_producer(
        iterator=iter(weights),
        group=producer_group,
        src=0,
        delta_tracker=tracker,
    )
    if load_full is not None and load_delta is not None:
        packed_weight_transfer_consumer(
            group=ReplayGroup(producer_group.broadcasted_tensors),
            src=0,
            load_full_weights_func=load_full,
            load_delta_weights_func=load_delta,
            device="cpu",
            delta_load_batch_size_bytes=delta_load_batch_size_bytes,
        )
    return producer_group


def _run_peer_transfer(weights, source_group, *, tracker=None):
    iterator = CountingIterator(weights)
    packed_weight_transfer_producer(
        iterator=iterator,
        group=ReplayGroup(source_group.broadcasted_tensors),
        src=0,
        delta_tracker=tracker,
    )
    return iterator


def test_pack_named_tensors_aligns_mixed_dtypes():
    tensors = [
        ("half", torch.tensor([1.0], dtype=torch.float16)),
        ("float", torch.tensor([2.0], dtype=torch.float32)),
    ]

    payload, entries = pack_named_tensors(tensors)
    unpacked = unpack_named_tensors(payload, entries)

    assert torch.equal(unpacked[0][1], tensors[0][1])
    assert torch.equal(unpacked[1][1], tensors[1][1])


def test_pack_named_tensors_roundtrips_float8_dtype_when_available():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("float8_e4m3fn is not available in this PyTorch build")

    tensor = torch.arange(4, dtype=torch.uint8).view(torch.float8_e4m3fn)
    payload, entries = pack_named_tensors([("float8", tensor)])
    unpacked = unpack_named_tensors(payload, entries)

    assert unpacked[0][1].dtype == tensor.dtype
    assert torch.equal(unpacked[0][1].view(torch.uint8), tensor.view(torch.uint8))


def test_additive_weight_load_context_adds_instead_of_overwriting():
    target = torch.tensor([1.0, 2.0])
    source = torch.tensor([0.5, -1.0], dtype=torch.float16)
    temporary = torch.empty_like(target)

    with additive_weight_load_context([target]):
        target.copy_(source)
        temporary.fill_(3.0)
        temporary.copy_(source)

    assert torch.equal(target, torch.tensor([1.5, 1.0]))
    assert torch.equal(temporary, source.to(dtype=temporary.dtype))

    target.copy_(torch.tensor([7.0, 8.0]))
    assert torch.equal(target, torch.tensor([7.0, 8.0]))


def test_additive_weight_load_context_adds_to_parameter_slices():
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])

    with additive_weight_load_context([target]):
        target[:2].copy_(torch.tensor([0.5, 1.0]))
        target[2:].fill_(2.0)

    assert torch.equal(target, torch.tensor([1.5, 3.0, 5.0, 6.0]))


def test_additive_weight_load_context_adds_for_setitem_loaders():
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    temporary = torch.zeros_like(target)

    with additive_weight_load_context([target]):
        target[0] = torch.tensor([0.5, 1.0])
        temporary[0] = torch.tensor([9.0, 8.0])

    assert torch.equal(target, torch.tensor([[1.5, 3.0], [3.0, 4.0]]))
    assert torch.equal(temporary, torch.tensor([[9.0, 8.0], [0.0, 0.0]]))


def test_packed_weight_transfer_full_roundtrip_without_delta_tracker():
    received = {}
    weights = [
        ("linear.weight", torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        ("linear.bias", torch.tensor([0.5, -0.5])),
    ]

    load_full, _ = _state_loaders(received)

    def load_delta(_loaded_weights):
        raise AssertionError("full transfer should not call delta loader")

    with (
        _cpu_transfer_env(),
        patch("nemo_rl.utils.weight_transfer._AsyncWeightLoadQueue") as load_queue_cls,
    ):
        _run_transfer(weights, load_full=load_full, load_delta=load_delta)

    load_queue_cls.assert_not_called()
    for name, tensor in weights:
        assert torch.equal(received[name], tensor)


def test_empty_transfer_header_uses_single_control_broadcast():
    group = RecordingGroup()

    header, refs = protocol_mod.broadcast_header(
        {
            "kind": G_DELTA_UPDATE_KIND,
            "transport": G_DENSE_TRANSPORT,
            "payload_entries": [],
            "payload_numel": 0,
            "sparse_metadata": [],
        },
        group=group,
        src=0,
        device=torch.device("cpu"),
    )

    assert header["payload_numel"] == 0
    assert refs[1] is None
    assert len(group.broadcasted_tensors) == 1


def test_protocol_rejects_invalid_header_dtype_and_env(monkeypatch):
    with pytest.raises(ValueError, match="header control"):
        protocol_mod.decode_header_control(torch.tensor([99, 0, 0, 0]))

    with pytest.raises(ValueError, match="header control"):
        protocol_mod.decode_header_control(torch.tensor([1, 99, 0, 0]))

    with pytest.raises(ValueError, match="Unsupported tensor dtype"):
        protocol_mod.dtype_from_name("complex64")

    monkeypatch.setenv("NRL_REFIT_BASELINE_PREWARM_CHUNK_BYTES", "not-an-int")
    with pytest.raises(ValueError, match="Expected integer value"):
        protocol_mod.env_int("NRL_REFIT_BASELINE_PREWARM_CHUNK_BYTES", default=1)


def test_sparse_indices_encoder_coalesces_small_tensors(monkeypatch):
    monkeypatch.setenv("NRL_REFIT_SPARSE_ENCODE_COALESCE_BYTES", "1024")
    tensors = [
        ("weight_0", torch.tensor([0.0, 1.0, 0.0])),
        ("weight_1", torch.tensor([2.0, 0.0, 3.0])),
    ]

    payload_tensors, _, metadata = sparse_codec_mod.encode_sparse_indices(tensors)
    decoded = dict(
        next(
            sparse_codec_mod.decode_sparse(
                payload_tensors,
                metadata,
                torch.device("cpu"),
                byte_cap=1024,
            )
        )
    )

    assert torch.equal(decoded["weight_0"], tensors[0][1])
    assert torch.equal(decoded["weight_1"], tensors[1][1])
    assert len(metadata) == 2


def test_sparse_indices_encoder_aliases_contiguous_views(monkeypatch):
    monkeypatch.setenv("NRL_REFIT_SPARSE_ENCODE_COALESCE_BYTES", "1024")
    arena = torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
    tensors = [
        ("weight_0", arena[:3]),
        ("weight_1", arena[3:]),
    ]
    infos = [
        sparse_codec_mod.SparseTensorInfo(name, tensor, tensor.view(-1))
        for name, tensor in tensors
    ]

    aliased = sparse_codec_mod.alias_sparse_index_group(infos)

    assert aliased is not None
    assert aliased.untyped_storage().data_ptr() == arena.untyped_storage().data_ptr()
    assert torch.equal(aliased, arena)
    payload_tensors, _, metadata = sparse_codec_mod.encode_sparse_indices(tensors)
    decoded = dict(
        next(
            sparse_codec_mod.decode_sparse(
                payload_tensors,
                metadata,
                torch.device("cpu"),
                byte_cap=1024,
            )
        )
    )

    assert torch.equal(decoded["weight_0"], tensors[0][1])
    assert torch.equal(decoded["weight_1"], tensors[1][1])


def test_sparse_indices_group_oom_splits_to_single_tensors(monkeypatch):
    monkeypatch.setenv("NRL_REFIT_SPARSE_ENCODE_COALESCE_BYTES", "1024")
    tensors = [
        ("weight_0", torch.tensor([0.0, 1.0, 0.0])),
        ("weight_1", torch.tensor([2.0, 0.0, 3.0])),
    ]
    real_sparse_indices_for_group = sparse_codec_mod.sparse_indices_for_group

    def raise_group_oom(group):
        if len(group) > 1:
            raise torch.OutOfMemoryError("simulated grouped sparse OOM")
        return real_sparse_indices_for_group(group)

    monkeypatch.setattr(
        sparse_codec_mod,
        "sparse_indices_for_group",
        raise_group_oom,
    )

    payload_tensors, _, metadata = sparse_codec_mod.encode_sparse_indices(tensors)
    decoded = dict(
        next(
            sparse_codec_mod.decode_sparse(
                payload_tensors,
                metadata,
                torch.device("cpu"),
                byte_cap=1024,
            )
        )
    )

    assert torch.equal(decoded["weight_0"], tensors[0][1])
    assert torch.equal(decoded["weight_1"], tensors[1][1])
    assert len(metadata) == 2


def test_peer_chunk_advance_matches_chunk_boundaries_without_collecting_tensors():
    weights = [
        ("weight_0", torch.ones(4)),
        ("weight_1", torch.ones(4)),
        ("weight_2", torch.ones(4)),
    ]
    iterator = CountingIterator(weights)

    pending_item, exhausted = protocol_mod.advance_chunk(
        iterator,
        byte_cap=20,
    )
    assert not exhausted
    assert pending_item is not None
    assert pending_item[0] == "weight_1"
    assert pending_item[1] is weights[1][1]
    assert iterator.names == ["weight_0", "weight_1"]

    pending_item, exhausted = protocol_mod.advance_chunk(
        iterator,
        byte_cap=20,
        pending_item=pending_item,
    )
    assert not exhausted
    assert pending_item is not None
    assert pending_item[0] == "weight_2"
    assert pending_item[1] is weights[2][1]
    assert iterator.names == ["weight_0", "weight_1", "weight_2"]

    pending_item, exhausted = protocol_mod.advance_chunk(
        iterator,
        byte_cap=20,
        pending_item=pending_item,
    )
    assert exhausted
    assert pending_item is None
    assert iterator.names == ["weight_0", "weight_1", "weight_2"]


def test_full_transfer_non_source_producer_advances_iterator_for_rank_collectives():
    weights = [
        ("linear.weight", torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        ("linear.bias", torch.tensor([0.5, -0.5])),
    ]

    with _cpu_transfer_env():
        source_group = _run_transfer(weights)
        peer_iterator = _run_peer_transfer(weights, source_group)

    assert peer_iterator.names == [name for name, _ in weights]


def test_packed_weight_transfer_full_then_sparse_delta_roundtrip():
    tracker = DeltaCompressionTracker(_tracker_config())
    receiver_state = {}
    delta_batches = []
    load_full, load_delta = _state_loaders(
        receiver_state,
        delta_batches=delta_batches,
    )

    initial = [
        ("linear.weight", torch.arange(16, dtype=torch.float32)),
        ("linear.bias", torch.zeros(8)),
    ]
    updated = [
        ("linear.weight", torch.arange(16, dtype=torch.float32)),
        ("linear.bias", torch.zeros(8)),
    ]
    updated[0][1][3] += 0.5
    updated[1][1][5] = -1.0

    with _cpu_transfer_env():
        _run_transfer(
            initial,
            tracker=tracker,
            load_full=load_full,
            load_delta=load_delta,
            delta_load_batch_size_bytes=16,
        )
        _run_transfer(
            updated,
            tracker=tracker,
            load_full=load_full,
            load_delta=load_delta,
            delta_load_batch_size_bytes=16,
        )

    for name, tensor in updated:
        assert torch.equal(receiver_state[name], tensor)
    assert delta_batches == [["linear.weight"], ["linear.bias"]]


def test_failed_delta_transfer_retries_with_full_update():
    tracker = DeltaCompressionTracker(_tracker_config())
    receiver_state = {}
    counters = {"full": 0, "delta": 0}
    initial = [
        ("linear.weight", torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        ("linear.bias", torch.tensor([0.5, -0.5])),
    ]
    updated = [
        ("linear.weight", torch.tensor([[1.0, 1.5], [3.25, 4.0]])),
        ("linear.bias", torch.tensor([0.5, 0.0])),
    ]

    load_full, load_delta = _state_loaders(receiver_state, counters=counters)

    with _cpu_transfer_env():
        _run_transfer(
            initial,
            tracker=tracker,
            load_full=load_full,
            load_delta=load_delta,
            delta_load_batch_size_bytes=1024,
        )
        with pytest.raises(RuntimeError, match="simulated broadcast failure"):
            packed_weight_transfer_producer(
                iterator=iter(updated),
                group=FailingGroup(),
                src=0,
                delta_tracker=tracker,
            )
        _run_transfer(
            updated,
            tracker=tracker,
            load_full=load_full,
            load_delta=load_delta,
            delta_load_batch_size_bytes=1024,
        )

    assert counters == {"full": 2, "delta": 0}
    for name, tensor in updated:
        assert torch.equal(receiver_state[name], tensor)


def test_packed_weight_transfer_all_zero_sparse_delta_is_noop():
    tracker = DeltaCompressionTracker(_tracker_config())
    receiver_state = {}
    weights = [
        ("linear.weight", torch.tensor([[1.001, 2.003], [3.005, 4.007]])),
        ("linear.bias", torch.tensor([0.123, -0.456])),
    ]

    load_full, _ = _state_loaders(receiver_state)

    def load_delta(_weights):
        raise AssertionError("all-zero sparse deltas should not call load_delta")

    with _cpu_transfer_env():
        _run_transfer(
            weights,
            tracker=tracker,
            load_full=load_full,
            load_delta=load_delta,
            delta_load_batch_size_bytes=1024,
        )
        producer_group = _run_transfer(
            weights,
            tracker=tracker,
        )
        loaded_weights = packed_weight_transfer_consumer(
            group=ReplayGroup(producer_group.broadcasted_tensors),
            src=0,
            load_full_weights_func=load_full,
            load_delta_weights_func=load_delta,
            device="cpu",
            delta_load_batch_size_bytes=1024,
        )

    for name, tensor in weights:
        assert torch.equal(receiver_state[name], tensor)
    assert not loaded_weights.loaded_any
    assert loaded_weights.is_delta_sync


def test_packed_weight_transfer_non_floating_chunk_uses_full_update():
    tracker = DeltaCompressionTracker(_tracker_config())
    receiver_state = {}
    counters = {"full": 0, "delta": 0}
    initial = [
        ("linear.weight", torch.tensor([1.0, 2.0])),
        ("step", torch.tensor([1], dtype=torch.int64)),
    ]
    updated = [
        ("linear.weight", torch.tensor([1.5, 2.0])),
        ("step", torch.tensor([2], dtype=torch.int64)),
    ]

    load_full, load_delta = _state_loaders(receiver_state, counters=counters)

    with _cpu_transfer_env():
        for weights in (initial, updated):
            _run_transfer(
                weights,
                tracker=tracker,
                load_full=load_full,
                load_delta=load_delta,
                delta_load_batch_size_bytes=1024,
            )

    assert counters == {"full": 2, "delta": 0}
    for name, tensor in updated:
        assert torch.equal(receiver_state[name], tensor)


def test_packed_weight_transfer_sparse_encoding_unavailable_uses_full_update(
    monkeypatch,
):
    tracker = DeltaCompressionTracker(_tracker_config())
    receiver_state = {}
    counters = {"full": 0, "delta": 0}
    initial = [("linear.weight", torch.zeros(4))]
    updated = [("linear.weight", torch.ones(4))]

    load_full, load_delta = _state_loaders(receiver_state, counters=counters)

    with _cpu_transfer_env():
        _run_transfer(
            initial,
            tracker=tracker,
            load_full=load_full,
            load_delta=load_delta,
            delta_load_batch_size_bytes=1024,
        )
        monkeypatch.setattr(
            sparse_codec_mod,
            "encode_sparse_indices",
            Mock(side_effect=sparse_codec_mod.SparseEncodingUnavailable),
        )
        _run_transfer(
            updated,
            tracker=tracker,
            load_full=load_full,
            load_delta=load_delta,
            delta_load_batch_size_bytes=1024,
        )

    assert counters == {"full": 2, "delta": 0}
    assert torch.equal(receiver_state["linear.weight"], updated[0][1])


def test_consumer_flushes_async_sparse_decode_before_full_update():
    group = RecordingGroup()
    sparse_delta = torch.zeros(1024)
    sparse_delta[0] = 1.0
    sparse_tensors, transport, sparse_metadata = sparse_codec_mod.encode_sparse_indices(
        [("linear.weight", sparse_delta)]
    )
    sparse_payload, sparse_entries = pack_named_tensors(sparse_tensors)
    sparse_header = {
        "kind": G_DELTA_UPDATE_KIND,
        "transport": transport,
        "payload_entries": sparse_entries,
        "payload_numel": int(sparse_payload.numel()),
        "sparse_metadata": sparse_metadata,
    }
    full_payload, full_entries = pack_named_tensors(
        [("step", torch.tensor([2], dtype=torch.int64))]
    )
    full_header = {
        "kind": G_FULL_UPDATE_KIND,
        "transport": G_DENSE_TRANSPORT,
        "payload_entries": full_entries,
        "payload_numel": int(full_payload.numel()),
        "sparse_metadata": [],
    }

    for header, payload in (
        (sparse_header, sparse_payload),
        (full_header, full_payload),
    ):
        protocol_mod.broadcast_header(
            header,
            group=group,
            src=0,
            device=torch.device("cpu"),
        )
        group.broadcast(payload, src=0)
    protocol_mod.broadcast_header(
        {"kind": G_TRANSFER_DONE_KIND},
        group=group,
        src=0,
        device=torch.device("cpu"),
    )

    decode_started = threading.Event()
    release_decode = threading.Event()
    load_order = []
    errors = []

    def decode_sparse(*_args, **_kwargs):
        decode_started.set()
        release_decode.wait(timeout=5)
        yield [("linear.weight", sparse_delta)]

    def load_full(_weights):
        if not release_decode.is_set():
            raise AssertionError("full update loaded before prior sparse delta")
        load_order.append("full")

    def load_delta(_weights):
        load_order.append("delta")

    def consume():
        try:
            packed_weight_transfer_consumer(
                group=ReplayGroup(group.broadcasted_tensors),
                src=0,
                load_full_weights_func=load_full,
                load_delta_weights_func=load_delta,
                device="cpu",
                delta_load_batch_size_bytes=1024,
            )
        except Exception as error:
            errors.append(error)

    with (
        _cpu_transfer_env(),
        patch(
            "nemo_rl.utils.weight_transfer_sparse_codec.decode_sparse", decode_sparse
        ),
    ):
        thread = threading.Thread(target=consume)
        thread.start()
        assert decode_started.wait(timeout=2)
        release_decode.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []
    assert load_order == ["delta", "full"]


def test_high_density_sparse_delta_uses_sparse_update():
    tracker = DeltaCompressionTracker(_tracker_config())
    receiver_state = {}
    counters = {"full": 0, "delta": 0}
    initial = [
        ("linear.weight", torch.zeros(4)),
    ]
    updated = [
        ("linear.weight", torch.ones(4)),
    ]

    load_full, load_delta = _state_loaders(receiver_state, counters=counters)

    with _cpu_transfer_env():
        for weights in (initial, updated):
            _run_transfer(
                weights,
                tracker=tracker,
                load_full=load_full,
                load_delta=load_delta,
                delta_load_batch_size_bytes=1024,
            )

    assert counters == {"full": 1, "delta": 1}
    assert torch.equal(receiver_state["linear.weight"], updated[0][1])


def test_full_sync_interval_one_does_not_create_baseline():
    tracker = DeltaCompressionTracker(_tracker_config(full_sync_interval=1))
    is_delta, _ = tracker.prepare_chunk([("linear.weight", torch.ones(4))])
    tracker.on_sync_succeeded()

    assert not is_delta
    assert tracker.baseline == {}


def test_delta_tracker_rejects_invalid_config():
    with pytest.raises(ValueError, match="Unsupported delta compression dtype"):
        DeltaCompressionTracker({**_tracker_config(), "dtype": "float64"})

    with pytest.raises(ValueError, match="full_sync_interval"):
        DeltaCompressionTracker(_tracker_config(full_sync_interval=0))

    with pytest.raises(ValueError, match="sparse_bucket_size_bytes"):
        DeltaCompressionTracker(_tracker_config(sparse_bucket_size_bytes=0))


def test_delta_baseline_prewarm_reuses_allocations(monkeypatch):
    monkeypatch.setenv("NRL_REFIT_BASELINE_PREWARM_CHUNK_BYTES", "32")
    tracker = DeltaCompressionTracker(_tracker_config())
    metadata = {
        "linear.weight": (torch.Size([2, 4]), torch.float32),
        "linear.bias": (torch.Size([4]), torch.float32),
        "embed.weight": (torch.Size([2, 2]), torch.float16),
    }

    with _cpu_transfer_env():
        tracker.prewarm_baseline_from_metadata(metadata)

    assert set(tracker.baseline) == set(metadata)
    assert tracker.baseline["linear.weight"].shape == torch.Size([2, 4])
    assert tracker.baseline["linear.bias"].dtype == torch.float32
    assert tracker.baseline["embed.weight"].dtype == torch.float16
    assert tracker._baseline_entries["linear.weight"].arena.dtype == torch.float32
    assert tracker._baseline_entries["embed.weight"].arena.dtype == torch.float16

    tensors = [
        (name, torch.ones(tuple(shape), dtype=dtype))
        for name, (shape, dtype) in metadata.items()
    ]
    existing_baselines = dict(tracker.baseline)
    tracker._snapshot_baseline(tensors)

    assert all(
        tracker.baseline[name] is baseline
        for name, baseline in existing_baselines.items()
    )


def test_delta_baseline_prewarm_can_be_disabled(monkeypatch):
    monkeypatch.setenv("NRL_REFIT_PREWARM_DELTA_BASELINE", "0")
    tracker = DeltaCompressionTracker(_tracker_config())

    with _cpu_transfer_env():
        tracker.prewarm_baseline_from_metadata(
            {"linear.weight": (torch.Size([4]), torch.float32)}
        )

    assert tracker.baseline == {}
    assert tracker._baseline_entries == {}


def test_delta_baseline_prewarm_honors_max_bytes(monkeypatch):
    monkeypatch.setenv("NRL_REFIT_BASELINE_PREWARM_CHUNK_BYTES", "1024")
    monkeypatch.setenv("NRL_REFIT_BASELINE_PREWARM_MAX_BYTES", "16")
    tracker = DeltaCompressionTracker(_tracker_config())
    metadata = {
        "linear.weight": (torch.Size([4]), torch.float32),
        "linear.bias": (torch.Size([4]), torch.float32),
    }

    with _cpu_transfer_env():
        tracker.prewarm_baseline_from_metadata(metadata)

    assert set(tracker.baseline) == {"linear.weight"}
    assert set(tracker._baseline_entries) == {"linear.weight"}


def test_delta_baseline_can_use_mmap_storage(monkeypatch, tmp_path):
    monkeypatch.setenv("NRL_REFIT_BASELINE_MMAP_MIN_BYTES", "1")
    monkeypatch.setenv("NRL_REFIT_BASELINE_MMAP_DIR", str(tmp_path))
    tracker = DeltaCompressionTracker(_tracker_config())
    metadata = {
        "linear.weight": (torch.Size([2]), torch.float32),
    }

    with _cpu_transfer_env():
        tracker.prewarm_baseline_from_metadata(metadata)
        first_is_delta, _ = tracker.prepare_chunk(
            [("linear.weight", torch.tensor([1.0, 2.0]))]
        )
        tracker.on_sync_succeeded()
        is_delta, deltas = tracker.prepare_chunk(
            [("linear.weight", torch.tensor([1.5, 1.0]))]
        )

    assert tracker._mmap_arenas
    assert not first_is_delta
    assert is_delta
    torch.testing.assert_close(deltas[0][1], torch.tensor([0.5, -1.0]))


def test_delta_baseline_flush_waits_for_mmap_write_future():
    tracker = DeltaCompressionTracker(_tracker_config())
    future: Future[None] = Future()
    future.set_result(None)
    tracker._baseline_write_futures["linear.weight"] = future
    tracker._mmap_pending_writes.append((future, 16))
    tracker._mmap_pending_stage_bytes = 16

    tracker.flush_baseline(["linear.weight"])

    assert tracker._baseline_write_futures == {}
    assert tracker._mmap_pending_writes == []
    assert tracker._mmap_pending_stage_bytes == 0


def test_delta_baseline_mmap_stage_settings(monkeypatch):
    monkeypatch.delenv("NRL_REFIT_BASELINE_MMAP_PENDING_BYTES", raising=False)
    monkeypatch.delenv("NRL_REFIT_BASELINE_MMAP_WRITE_WORKERS", raising=False)

    assert _baseline_mmap_pending_bytes(32) == 256
    assert _baseline_mmap_write_workers() == 4

    monkeypatch.setenv("NRL_REFIT_BASELINE_MMAP_PENDING_BYTES", "64")
    monkeypatch.setenv("NRL_REFIT_BASELINE_MMAP_WRITE_WORKERS", "2")

    assert _baseline_mmap_pending_bytes(32) == 64
    assert _baseline_mmap_write_workers() == 2


def test_delta_baseline_keeps_source_dtype_but_emits_delta_dtype():
    tracker = DeltaCompressionTracker(_tracker_config())
    initial = [
        ("linear.weight", torch.tensor([1.0, 2.0], dtype=torch.float16)),
    ]
    updated = [
        ("linear.weight", torch.tensor([1.5, 1.0], dtype=torch.float16)),
    ]

    with _cpu_transfer_env():
        first_is_delta, _ = tracker.prepare_chunk(initial)
        tracker.on_sync_succeeded()
        is_delta, deltas = tracker.prepare_chunk(updated)

    assert not first_is_delta
    assert tracker.baseline["linear.weight"].dtype == torch.float16
    assert is_delta
    assert deltas[0][1].dtype == torch.float32
    torch.testing.assert_close(
        deltas[0][1],
        torch.tensor([0.5, -1.0], dtype=torch.float32),
    )


def test_delta_baseline_records_contiguous_entries(monkeypatch):
    monkeypatch.setenv("NRL_REFIT_BASELINE_PREWARM_CHUNK_BYTES", "1024")
    tracker = DeltaCompressionTracker(_tracker_config())
    metadata = {
        "linear.weight": (torch.Size([2, 4]), torch.float32),
        "linear.bias": (torch.Size([4]), torch.float32),
    }

    with _cpu_transfer_env():
        tracker.prewarm_baseline_from_metadata(metadata)

    weight_entry = tracker._baseline_entries["linear.weight"]
    bias_entry = tracker._baseline_entries["linear.bias"]
    assert weight_entry.arena is bias_entry.arena
    assert weight_entry.offset == 0
    assert bias_entry.offset == weight_entry.numel


def test_baseline_spans_honor_memory_limited_stage_cap(monkeypatch):
    monkeypatch.setenv("NRL_REFIT_BASELINE_PREWARM_CHUNK_BYTES", "1024")
    tracker = DeltaCompressionTracker(_tracker_config())
    metadata = {
        "weight_0": (torch.Size([2]), torch.float32),
        "weight_1": (torch.Size([2]), torch.float32),
        "weight_2": (torch.Size([2]), torch.float32),
    }

    with _cpu_transfer_env():
        tracker.prewarm_baseline_from_metadata(metadata)

    tensors = [
        (name, torch.ones(tuple(shape), dtype=dtype))
        for name, (shape, dtype) in metadata.items()
    ]
    spans = list(
        tracker._iter_baseline_spans(
            tensors,
            itemsize=None,
            max_bytes=8,
        )
    )

    assert [[name for name, _, _ in span] for span, _, _ in spans] == [
        ["weight_0"],
        ["weight_1"],
        ["weight_2"],
    ]


def test_memory_limited_stage_bytes_uses_fraction_of_free_cuda_memory(monkeypatch):
    monkeypatch.setattr(
        protocol_mod.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        protocol_mod.torch.cuda,
        "mem_get_info",
        lambda _device=None: (1024, 4096),
    )

    stage_bytes = protocol_mod.memory_limited_stage_bytes(
        torch.device("cuda", 0),
        requested_bytes=512,
    )

    assert stage_bytes == 128


def test_successful_sync_defers_async_baseline_flush_until_needed():
    tracker = DeltaCompressionTracker(_tracker_config())
    flush_count = 0

    def flush_baseline(_names=None):
        nonlocal flush_count
        flush_count += 1

    tracker.flush_baseline = flush_baseline
    tracker.on_sync_succeeded()

    assert flush_count == 0
    assert tracker.committed_syncs == 1


def test_failed_sync_flushes_async_baseline():
    tracker = DeltaCompressionTracker(_tracker_config())
    flush_count = 0

    def flush_baseline(_names=None):
        nonlocal flush_count
        flush_count += 1

    tracker.flush_baseline = flush_baseline
    tracker.on_sync_failed()

    assert flush_count == 1
    assert tracker.committed_syncs == 0


def test_transfer_done_sync_does_not_synchronize_whole_device():
    stream = Mock()
    with (
        patch(
            "nemo_rl.utils.weight_transfer_protocol.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "nemo_rl.utils.weight_transfer_protocol.torch.cuda.current_stream",
            return_value=stream,
        ) as current_stream,
        patch(
            "nemo_rl.utils.weight_transfer_protocol.torch.cuda.synchronize"
        ) as synchronize,
    ):
        protocol_mod.synchronize_current_transfer_stream(torch.device("cuda", 0))

    current_stream.assert_called_once_with(torch.device("cuda", 0))
    stream.synchronize.assert_called_once_with()
    synchronize.assert_not_called()


def test_non_source_producer_advances_iterator_without_delta_baseline():
    source_tracker = DeltaCompressionTracker(_tracker_config())
    peer_tracker = DeltaCompressionTracker(_tracker_config())
    initial = [
        ("linear.weight", torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        ("linear.bias", torch.tensor([0.5, -0.5])),
    ]
    source_update = [
        ("linear.weight", torch.tensor([[1.0, 2.5], [3.0, 4.0]])),
        ("linear.bias", torch.tensor([0.5, -0.25])),
    ]

    with _cpu_transfer_env():
        source_group = _run_transfer(initial, tracker=source_tracker)
        initial_peer_iterator = _run_peer_transfer(
            initial,
            source_group,
            tracker=peer_tracker,
        )

        source_group = _run_transfer(source_update, tracker=source_tracker)
        update_peer_iterator = _run_peer_transfer(
            source_update,
            source_group,
            tracker=peer_tracker,
        )

    assert initial_peer_iterator.names == [name for name, _ in initial]
    assert update_peer_iterator.names == [name for name, _ in source_update]
    assert peer_tracker.baseline == {}
    assert peer_tracker.committed_syncs == 0


def test_packed_weight_transfer_preserves_tensors_across_chunk_boundaries():
    tracker = DeltaCompressionTracker(_tracker_config())
    received = {}
    weights = [
        ("weight_0", torch.ones(4, dtype=torch.float32)),
        ("weight_1", torch.ones(4, dtype=torch.float32) * 2),
        ("weight_2", torch.ones(4, dtype=torch.float32) * 3),
    ]

    load_full, _ = _state_loaders(received)

    with _cpu_transfer_env(chunk_size=20):
        _run_transfer(
            weights,
            tracker=tracker,
            load_full=load_full,
            load_delta=lambda _: None,
            delta_load_batch_size_bytes=1024,
        )

    assert set(received) == {name for name, _ in weights}
    for name, tensor in weights:
        assert torch.equal(received[name], tensor)


def test_packed_weight_transfer_batches_deltas_across_transfer_chunks():
    tracker = DeltaCompressionTracker(_tracker_config())
    received = {}
    delta_batches = []
    initial = [
        ("weight_0", torch.ones(8, dtype=torch.float32)),
        ("weight_1", torch.ones(8, dtype=torch.float32) * 2),
        ("weight_2", torch.ones(8, dtype=torch.float32) * 3),
    ]
    updated = [
        ("weight_0", torch.ones(8, dtype=torch.float32)),
        ("weight_1", torch.ones(8, dtype=torch.float32) * 2),
        ("weight_2", torch.ones(8, dtype=torch.float32) * 3),
    ]
    updated[0][1][0] += 0.5
    updated[1][1][0] += 0.5
    updated[2][1][0] += 0.5

    load_full, load_delta = _state_loaders(received, delta_batches=delta_batches)

    with _cpu_transfer_env(chunk_size=40):
        for weights in (initial, updated):
            _run_transfer(
                weights,
                tracker=tracker,
                load_full=load_full,
                load_delta=load_delta,
                delta_load_batch_size_bytes=1024,
            )

    assert delta_batches == [["weight_0", "weight_1", "weight_2"]]
    for name, tensor in updated:
        assert torch.equal(received[name], tensor)


def test_sparse_delta_payloads_consolidate_across_transfer_chunks():
    tracker = DeltaCompressionTracker(_tracker_config())
    received = {}
    delta_batches = []
    initial = [(f"weight_{idx}", torch.zeros(64)) for idx in range(3)]
    updated = [(name, tensor.clone()) for name, tensor in initial]
    for idx, (_, tensor) in enumerate(updated):
        tensor[idx] = float(idx + 1)

    load_full, load_delta = _state_loaders(received, delta_batches=delta_batches)

    with _cpu_transfer_env(chunk_size=128):
        _run_transfer(
            initial,
            tracker=tracker,
            load_full=load_full,
            load_delta=load_delta,
            delta_load_batch_size_bytes=1024,
        )
        producer_group = _run_transfer(
            updated,
            tracker=tracker,
            load_full=load_full,
            load_delta=load_delta,
            delta_load_batch_size_bytes=1024,
        )
        headers = _recorded_headers(producer_group.broadcasted_tensors)

    sparse_delta_headers = [
        header
        for header in headers
        if header["kind"] == G_DELTA_UPDATE_KIND
        and header["transport"] == G_SPARSE_INDICES_TRANSPORT
        and header["sparse_metadata"]
    ]
    noop_delta_headers = [
        header
        for header in headers
        if header["kind"] == G_DELTA_UPDATE_KIND and int(header["payload_numel"]) == 0
    ]

    assert len(sparse_delta_headers) == 1
    assert len(sparse_delta_headers[0]["sparse_metadata"]) == len(updated)
    assert len(noop_delta_headers) == len(updated)
    assert delta_batches == [["weight_0", "weight_1", "weight_2"]]
    for name, tensor in updated:
        assert torch.equal(received[name], tensor)


def test_queued_sparse_payloads_wait_for_readiness_events_before_packing():
    tracker = DeltaCompressionTracker(_tracker_config())
    initial = [(f"weight_{idx}", torch.zeros(64)) for idx in range(3)]
    updated = [(name, tensor.clone()) for name, tensor in initial]
    for idx, (_, tensor) in enumerate(updated):
        tensor[idx] = float(idx + 1)

    recorded_events = []
    wait_calls = []

    def record_ready():
        event = f"ready-{len(recorded_events)}"
        recorded_events.append(event)
        return (event,)

    def wait_ready(events):
        if events:
            wait_calls.append(tuple(events))

    with (
        _cpu_transfer_env(chunk_size=128),
        patch(
            "nemo_rl.utils.weight_transfer.record_payload_readiness_events",
            side_effect=record_ready,
        ),
        patch(
            "nemo_rl.utils.weight_transfer.wait_for_payload_events",
            side_effect=wait_ready,
        ),
    ):
        _run_transfer(initial, tracker=tracker)
        recorded_events.clear()
        wait_calls.clear()

        _run_transfer(updated, tracker=tracker)

    assert wait_calls == [
        ("ready-0", "ready-1", "ready-2"),
        ("ready-3",),
    ]


def test_non_source_producer_advances_iterator_with_consolidated_sparse_delta():
    source_tracker = DeltaCompressionTracker(_tracker_config())
    peer_tracker = DeltaCompressionTracker(_tracker_config())
    initial = [(f"weight_{idx}", torch.zeros(64)) for idx in range(3)]
    updated = [(name, tensor.clone()) for name, tensor in initial]
    for idx, (_, tensor) in enumerate(updated):
        tensor[idx] = float(idx + 1)

    with _cpu_transfer_env(chunk_size=128):
        source_group = _run_transfer(initial, tracker=source_tracker)
        initial_peer_iterator = _run_peer_transfer(
            initial,
            source_group,
            tracker=peer_tracker,
        )

        source_group = _run_transfer(updated, tracker=source_tracker)
        update_peer_iterator = _run_peer_transfer(
            updated,
            source_group,
            tracker=peer_tracker,
        )

    assert initial_peer_iterator.names == [name for name, _ in initial]
    assert update_peer_iterator.names == [name for name, _ in updated]
    assert peer_tracker.baseline == {}
    assert peer_tracker.committed_syncs == 0


def test_vllm_delta_transfer_config_rejects_colocated_mode():
    generation_config = {
        "backend": "vllm",
        "vllm_cfg": {"precision": "bfloat16"},
        "colocated": {"enabled": True},
        "delta_compression": _tracker_config(),
    }

    with pytest.raises(ValueError, match="non-colocated"):
        create_vllm_delta_transfer_tracker(generation_config)


def test_vllm_delta_transfer_config_returns_none_when_disabled():
    assert create_vllm_delta_transfer_tracker(None) is None
    assert create_vllm_delta_transfer_tracker({}) is None
    assert (
        create_vllm_delta_transfer_tracker({"delta_compression": {"enabled": False}})
        is None
    )


def test_vllm_delta_transfer_config_rejects_non_vllm_backend():
    generation_config = {
        "backend": "sglang",
        "vllm_cfg": {"precision": "bfloat16"},
        "colocated": {"enabled": False},
        "delta_compression": _tracker_config(),
    }

    with pytest.raises(ValueError, match="vLLM"):
        create_vllm_delta_transfer_tracker(generation_config)


def test_vllm_delta_transfer_config_creates_tracker_when_enabled():
    generation_config = {
        "backend": "vllm",
        "vllm_cfg": {"precision": "bfloat16"},
        "colocated": {"enabled": False},
        "delta_compression": _tracker_config(),
    }

    tracker = create_vllm_delta_transfer_tracker(generation_config)

    assert isinstance(tracker, DeltaCompressionTracker)
    assert tracker.delta_dtype == torch.float32


def test_vllm_delta_transfer_config_rejects_quantized_modelopt_path():
    generation_config = {
        "backend": "vllm",
        "quant_cfg": "NVFP4_DEFAULT_CFG",
        "vllm_cfg": {"precision": "bfloat16"},
        "colocated": {"enabled": False},
        "delta_compression": _tracker_config(),
    }

    with pytest.raises(NotImplementedError, match="ModelOpt"):
        create_vllm_delta_transfer_tracker(generation_config)


def test_vllm_delta_transfer_config_rejects_fp8_precision():
    generation_config = {
        "backend": "vllm",
        "vllm_cfg": {"precision": "fp8"},
        "colocated": {"enabled": False},
        "delta_compression": _tracker_config(),
    }

    with pytest.raises(NotImplementedError, match="FP8"):
        create_vllm_delta_transfer_tracker(generation_config)
