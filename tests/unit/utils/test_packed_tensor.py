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

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch

from nemo_rl.utils.packed_tensor import (
    packed_broadcast_consumer,
    packed_broadcast_producer,
)


@pytest.mark.parametrize("buffer_size_bytes", [0, -1])
def test_producer_rejects_nonpositive_buffer_size(buffer_size_bytes):
    with pytest.raises(ValueError, match="buffer_size_bytes must be > 0"):
        packed_broadcast_producer(
            iterator=iter([]),
            group=None,
            src=0,
            post_iter_func=lambda x: x,
            buffer_size_bytes=buffer_size_bytes,
        )


class MockCommunicationGroup:
    """Mock communication group for testing broadcast operations."""

    def __init__(self):
        self.broadcasted_tensors = []
        self.broadcast_count = 0

    def broadcast(self, tensor, src):
        """Mock broadcast that stores the tensor for later verification."""
        # Store a copy of the tensor
        self.broadcasted_tensors.append(tensor.clone())
        self.broadcast_count += 1


class MockConsumerCommunicationGroup:
    """Mock communication group for consumer that returns pre-stored tensors."""

    def __init__(self, tensors_to_return):
        self.tensors_to_return = tensors_to_return
        self.current_index = 0

    def broadcast(self, tensor, src):
        """Mock broadcast that fills the tensor with pre-stored data."""
        if self.current_index < len(self.tensors_to_return):
            tensor.copy_(self.tensors_to_return[self.current_index])
            self.current_index += 1


def create_mock_model_params():
    """Create mock model parameters for testing."""
    params = [
        ("layer1.weight", torch.randn(10, 20, dtype=torch.float32)),
        ("layer1.bias", torch.randn(10, dtype=torch.float32)),
        ("layer2.weight", torch.randn(20, 30, dtype=torch.float32)),
        ("layer2.bias", torch.randn(20, dtype=torch.float32)),
        ("layer3.weight", torch.randn(30, 40, dtype=torch.float16)),
        ("kv_amax", torch.tensor(42.0, dtype=torch.bfloat16)),
        ("transposed.weight", torch.randn(4, 5, dtype=torch.float32).T),
    ]
    return params


def create_mock_state_dict_info(params):
    """Create state dict info (name -> (shape, dtype)) from params."""
    return {name: (tensor.shape, tensor.dtype) for name, tensor in params}


def test_packed_broadcast_consumer_callback_is_eager(monkeypatch):
    batches = [[("a", "a-value")], [("b", "b-value")]]
    callbacks = []

    monkeypatch.setattr(
        "nemo_rl.utils.packed_tensor._packed_broadcast_consumer_batches",
        lambda _iterator, _group, _src, **_kwargs: iter(batches),
    )

    result = packed_broadcast_consumer(
        iterator=iter(()),
        group=object(),
        src=0,
        post_unpack_func=callbacks.append,
    )

    assert result is None
    assert callbacks == batches


def test_packed_broadcast_consumer_returns_lazy_flattened_iterator(monkeypatch):
    batches_started = []

    def batches(_iterator, _group, _src, **_kwargs):
        batches_started.append(True)
        yield [("a", "a-value")]
        yield [("b", "b-value")]

    monkeypatch.setattr(
        "nemo_rl.utils.packed_tensor._packed_broadcast_consumer_batches", batches
    )

    result = packed_broadcast_consumer(
        iterator=iter(()),
        group=object(),
        src=0,
        post_unpack_func=None,
        return_iterator=True,
    )

    assert batches_started == []
    assert result is not None
    assert list(result) == [("a", "a-value"), ("b", "b-value")]
    assert batches_started == [True]


def test_packed_broadcast_consumer_closes_abandoned_lazy_batches(monkeypatch):
    class CloseableBatches:
        def __init__(self):
            self.batches = iter([[("a", "a-value"), ("b", "b-value")]])
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.batches)

        def close(self):
            self.closed = True

    batches = CloseableBatches()
    monkeypatch.setattr(
        "nemo_rl.utils.packed_tensor._packed_broadcast_consumer_batches",
        lambda _iterator, _group, _src, **_kwargs: batches,
    )

    result = packed_broadcast_consumer(
        iterator=iter(()),
        group=object(),
        src=0,
        post_unpack_func=None,
        return_iterator=True,
    )

    assert result is not None
    assert next(result) == ("a", "a-value")
    assert batches.closed is False

    result.close()

    assert batches.closed is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("return_iterator", [False, True])
@pytest.mark.parametrize(
    ("producer_num_buffers", "consumer_num_buffers"),
    [(None, None), (2, 1)],
)
def test_packed_broadcast_producer_consumer_roundtrip(
    producer_num_buffers, consumer_num_buffers, return_iterator
):
    """Test that producer and consumer work together correctly."""
    # Create mock parameters
    params = create_mock_model_params()

    # Move params to CUDA
    params_cuda = [(name, tensor.cuda()) for name, tensor in params]

    # Create mock communication group for producer
    producer_group = MockCommunicationGroup()

    # Mock the target size to force packing
    target_size = 2000
    with patch(
        "nemo_rl.utils.packed_tensor.get_target_packed_tensor_size",
        return_value=target_size,
    ):
        # Post-iter function that just returns the tensor
        post_iter_func = lambda x: x[1]

        # Run producer
        packed_broadcast_producer(
            iterator=iter(params_cuda),
            group=producer_group,
            src=0,
            post_iter_func=post_iter_func,
            num_buffers=producer_num_buffers,
        )

        # Now test consumer with the broadcasted tensors
        consumer_group = MockConsumerCommunicationGroup(
            producer_group.broadcasted_tensors
        )

        # Create state dict info for consumer
        state_dict_info = create_mock_state_dict_info(params_cuda)
        for name in ("kv_amax", "transposed.weight"):
            shape, dtype = state_dict_info[name]
            state_dict_info[name] = (list(shape), dtype)

        # Store unpacked tensors
        unpacked_tensors = {}

        def post_unpack_func(tensor_list):
            """Store unpacked tensors for verification."""
            for name, tensor in tensor_list:
                unpacked_tensors[name] = tensor

        # Run consumer
        result = packed_broadcast_consumer(
            iterator=iter(state_dict_info.items()),
            group=consumer_group,
            src=0,
            post_unpack_func=None if return_iterator else post_unpack_func,
            return_iterator=return_iterator,
            num_buffers=consumer_num_buffers,
        )
        if return_iterator:
            assert result is not None
            for name, tensor in result:
                unpacked_tensors[name] = tensor

        # Readiness token, a header and payload per batch, then COMPLETE.
        # Target is 2000 B. The two 2400 B tensors are isolated in their own
        # batches: [800+40], [2400], [80], [2400], [2+80].
        assert consumer_group.current_index == 12
        assert producer_group.broadcasted_tensors[0].item() == 0

    # Verify all parameters were unpacked
    assert len(unpacked_tensors) == len(params)

    # Verify each tensor matches the original
    for name, original_tensor in params_cuda:
        assert name in unpacked_tensors
        unpacked = unpacked_tensors[name]

        # Check shape and dtype
        assert unpacked.shape == original_tensor.shape
        assert unpacked.dtype == original_tensor.dtype

        # Check values are close (accounting for floating point precision)
        assert torch.allclose(unpacked, original_tensor, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_packed_broadcast_single_large_tensor():
    """Test with a single tensor larger than target size."""
    # Create a large tensor
    large_tensor = torch.randn(1000, 1000, dtype=torch.float32).cuda()
    params = [("large_weight", large_tensor)]

    # Create mock communication group
    mock_group = MockCommunicationGroup()

    # Small target size to force the tensor to exceed it
    with patch(
        "nemo_rl.utils.packed_tensor.get_target_packed_tensor_size", return_value=100
    ):
        packed_broadcast_producer(
            iterator=iter(params),
            group=mock_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

    # Should still broadcast the tensor
    assert mock_group.broadcast_count == 4
    assert len(mock_group.broadcasted_tensors) == 4
    assert mock_group.broadcasted_tensors[0].item() == 0
    assert mock_group.broadcasted_tensors[1].tolist() == [
        0,
        large_tensor.numel() * large_tensor.element_size(),
        1,
    ]
    assert mock_group.broadcasted_tensors[3].tolist() == [1, 0, 0]

    # Verify the size matches the large tensor
    expected_size = large_tensor.numel() * large_tensor.element_size()
    assert mock_group.broadcasted_tensors[2].numel() == expected_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_packed_broadcast_isolates_large_tensor_after_small_tensor():
    """An oversized tensor must not force a copy of the preceding prefix."""
    params = [
        ("small_weight", torch.randn(16, dtype=torch.float32).cuda()),
        ("large_weight", torch.randn(100, dtype=torch.float32).cuda()),
    ]
    producer_group = MockCommunicationGroup()

    with patch(
        "nemo_rl.utils.packed_tensor.get_target_packed_tensor_size", return_value=100
    ):
        packed_broadcast_producer(
            iterator=iter(params),
            group=producer_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

        consumer_group = MockConsumerCommunicationGroup(
            producer_group.broadcasted_tensors
        )
        unpacked_tensors = {}

        def post_unpack_func(tensor_list):
            unpacked_tensors.update(tensor_list)

        packed_broadcast_consumer(
            iterator=iter(create_mock_state_dict_info(params).items()),
            group=consumer_group,
            src=0,
            post_unpack_func=post_unpack_func,
        )

    # Skip the readiness token and frame headers.
    payloads = [
        tensor
        for tensor in producer_group.broadcasted_tensors
        if tensor.dtype == torch.uint8
    ]
    assert [tensor.numel() for tensor in payloads] == [
        params[0][1].nbytes,
        params[1][1].nbytes,
    ]
    for name, original_tensor in params:
        assert torch.equal(unpacked_tensors[name], original_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_packed_broadcast_legacy_packing_keeps_oversized_tensor_in_prefix_buffer():
    """vLLM's own consumer (Dynamo) packs an oversized tensor with its prefix."""
    params = [
        ("small_weight", torch.randn(16, dtype=torch.float32).cuda()),
        ("large_weight", torch.randn(100, dtype=torch.float32).cuda()),
    ]
    group = MockCommunicationGroup()

    with patch(
        "nemo_rl.utils.packed_tensor.get_target_packed_tensor_size", return_value=100
    ):
        packed_broadcast_producer(
            iterator=iter(params),
            group=group,
            src=0,
            post_iter_func=lambda x: x[1],
            use_legacy_packing=True,
            preflight_checked=True,
        )

    assert [t.numel() for t in group.broadcasted_tensors] == [
        sum(t.nbytes for _, t in params)
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_packed_broadcast_multiple_batches():
    """Test that tensors are properly batched when exceeding target size."""
    # Create many small tensors
    params = [
        (f"weight_{i}", torch.randn(10, 10, dtype=torch.float32).cuda())
        for i in range(20)
    ]

    # Create mock communication group
    mock_group = MockCommunicationGroup()

    # Small target size to force multiple batches
    with patch(
        "nemo_rl.utils.packed_tensor.get_target_packed_tensor_size", return_value=2000
    ):
        packed_broadcast_producer(
            iterator=iter(params),
            group=mock_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

    # Should have multiple broadcasts
    assert mock_group.broadcast_count > 1

    # Total size should match sum of all tensors
    assert mock_group.broadcasted_tensors[0].item() == 0
    assert mock_group.broadcasted_tensors[-1].tolist() == [1, 0, 0]
    total_broadcasted_size = sum(
        tensor.numel()
        for tensor in mock_group.broadcasted_tensors
        if tensor.dtype == torch.uint8
    )
    expected_total_size = sum(t.numel() * t.element_size() for _, t in params)
    assert total_broadcasted_size == expected_total_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_packed_broadcast_preflight_failure_stops_before_payload():
    producer_group = MockCommunicationGroup()
    iterator_was_consumed = False

    def params():
        nonlocal iterator_was_consumed
        iterator_was_consumed = True
        yield "weight", torch.ones(1, device="cuda")

    with pytest.raises(RuntimeError, match="injected producer preflight failure"):
        packed_broadcast_producer(
            iterator=params(),
            group=producer_group,
            src=0,
            post_iter_func=lambda x: x[1],
            preflight_error=RuntimeError("injected producer preflight failure"),
        )

    assert not iterator_was_consumed
    assert producer_group.broadcast_count == 1
    assert producer_group.broadcasted_tensors[0].item() == 1

    consumer_group = MockConsumerCommunicationGroup(producer_group.broadcasted_tensors)
    with pytest.raises(RuntimeError, match="producer preflight failed"):
        packed_broadcast_consumer(
            iterator=iter([("weight", ((1,), torch.float32))]),
            group=consumer_group,
            src=0,
            post_unpack_func=lambda _tensors: pytest.fail(
                "consumer loaded payload after failed preflight"
            ),
        )
    assert consumer_group.current_index == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_packed_broadcast_preflight_checked_skips_readiness_token():
    params = [
        ("layer1.weight", torch.randn(4, 4, device="cuda")),
        ("layer1.bias", torch.randn(4, device="cuda")),
    ]

    default_group = MockCommunicationGroup()
    packed_broadcast_producer(
        iterator=iter(params),
        group=default_group,
        src=0,
        post_iter_func=lambda x: x[1],
    )

    checked_group = MockCommunicationGroup()
    packed_broadcast_producer(
        iterator=iter(params),
        group=checked_group,
        src=0,
        post_iter_func=lambda x: x[1],
        preflight_checked=True,
    )

    # The caller already sent the token, so a second one would shift every frame.
    assert checked_group.broadcast_count == default_group.broadcast_count - 1
    # Frame headers are int64; the payload frames carry the packed uint8 bytes.
    assert any(t.dtype == torch.uint8 for t in checked_group.broadcasted_tensors)

    consumer_group = MockConsumerCommunicationGroup(checked_group.broadcasted_tensors)
    unpacked = {}
    packed_broadcast_consumer(
        iterator=iter(create_mock_state_dict_info(params).items()),
        group=consumer_group,
        src=0,
        post_unpack_func=lambda batch: unpacked.update(batch),
        preflight_checked=True,
    )
    assert consumer_group.current_index == checked_group.broadcast_count
    for name, tensor in params:
        assert torch.equal(unpacked[name], tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    ("producer_checked", "consumer_checked"), [(True, False), (False, True)]
)
def test_packed_broadcast_preflight_checked_mismatch_is_detected(
    producer_checked, consumer_checked
):
    params = [
        ("layer1.weight", torch.randn(4, 4, device="cuda")),
        ("layer1.bias", torch.randn(4, device="cuda")),
    ]
    producer_group = MockCommunicationGroup()
    packed_broadcast_producer(
        iterator=iter(params),
        group=producer_group,
        src=0,
        post_iter_func=lambda x: x[1],
        preflight_checked=producer_checked,
    )

    consumer_group = MockConsumerCommunicationGroup(producer_group.broadcasted_tensors)
    unpacked = {}

    # producer_checked: the consumer reads the first frame header as the
    # readiness token. consumer_checked: the consumer reads the readiness token
    # as a frame header, which the frame protocol rejects before any payload.
    with pytest.raises(
        RuntimeError,
        match=None if producer_checked else "Invalid packed-broadcast DATA frame",
    ):
        packed_broadcast_consumer(
            iterator=iter(create_mock_state_dict_info(params).items()),
            group=consumer_group,
            src=0,
            post_unpack_func=lambda batch: unpacked.update(batch),
            preflight_checked=consumer_checked,
        )
    assert unpacked == {}
    assert consumer_group.current_index < producer_group.broadcast_count


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_packed_broadcast_preflight_checked_error_raises_without_broadcast():
    group = MockCommunicationGroup()
    with pytest.raises(RuntimeError, match="^local draft export failed$"):
        packed_broadcast_producer(
            iterator=iter([("weight", torch.ones(1, device="cuda"))]),
            group=group,
            src=0,
            post_iter_func=lambda x: x[1],
            preflight_error=ValueError("local draft export failed"),
            preflight_checked=True,
        )
    assert group.broadcast_count == 0


def test_packed_broadcast_midstream_failure_reaches_consumer(monkeypatch):
    _force_cpu_packed_broadcast(monkeypatch)

    def params():
        yield "first.weight", torch.tensor([1.0])
        raise RuntimeError("injected mid-stream producer failure")

    producer_group = MockCommunicationGroup()
    with pytest.raises(RuntimeError, match="injected mid-stream producer failure"):
        packed_broadcast_producer(
            iterator=params(),
            group=producer_group,
            src=0,
            post_iter_func=lambda item: item[1],
        )

    loaded_names = []
    consumer_group = MockConsumerCommunicationGroup(producer_group.broadcasted_tensors)
    with pytest.raises(RuntimeError, match="producer failed during payload transfer"):
        packed_broadcast_consumer(
            iterator=iter(
                [
                    ("first.weight", ((1,), torch.float32)),
                    ("second.weight", ((1,), torch.float32)),
                ]
            ),
            group=consumer_group,
            src=0,
            post_unpack_func=lambda tensors: loaded_names.extend(
                name for name, _ in tensors
            ),
        )

    assert loaded_names == ["first.weight"]
    assert consumer_group.current_index == len(producer_group.broadcasted_tensors)


def _force_cpu_packed_broadcast(monkeypatch):
    # Mirrors the producer's stream contract: buffer fences plus the export
    # event that carries one buffer's dependencies to the next (#4720).
    class ImmediateEvent:
        def record(self, stream=None):
            return None

    class ImmediateStream:
        def synchronize(self):
            return None

        def wait_event(self, event):
            return None

    original_empty = torch.empty
    original_tensor = torch.tensor

    def force_cpu(factory):
        def wrapped(*args, **kwargs):
            device = kwargs.get("device")
            if device is not None and torch.device(device).type == "cuda":
                kwargs["device"] = "cpu"
            return factory(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(torch, "empty", force_cpu(original_empty))
    monkeypatch.setattr(torch, "tensor", force_cpu(original_tensor))
    monkeypatch.setattr(torch.cuda, "Stream", ImmediateStream)
    monkeypatch.setattr(torch.cuda, "Event", ImmediateEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        "nemo_rl.utils.packed_tensor.get_target_packed_tensor_size", lambda: 1
    )
    monkeypatch.setattr("nemo_rl.utils.packed_tensor.get_num_buffers", lambda: 1)


def _broadcast_two_single_tensor_frames() -> MockCommunicationGroup:
    producer_group = MockCommunicationGroup()
    packed_broadcast_producer(
        iterator=iter(
            [
                ("first.weight", torch.tensor([1.0])),
                ("second.weight", torch.tensor([2.0])),
            ]
        ),
        group=producer_group,
        src=0,
        post_iter_func=lambda item: item[1],
    )
    return producer_group


_TWO_FRAME_CONSUMER_METADATA = [
    ("first.weight", ((1,), torch.float32)),
    ("second.weight", ((1,), torch.float32)),
]


def test_packed_broadcast_consumer_drains_after_load_failure(monkeypatch):
    _force_cpu_packed_broadcast(monkeypatch)
    producer_group = _broadcast_two_single_tensor_frames()

    callback_error = RuntimeError("injected consumer load failure")
    callback_names = []

    def fail_first_load(tensors):
        callback_names.append([name for name, _ in tensors])
        raise callback_error

    consumer_group = MockConsumerCommunicationGroup(producer_group.broadcasted_tensors)
    with pytest.raises(RuntimeError, match="injected consumer load failure") as error:
        packed_broadcast_consumer(
            iterator=iter(_TWO_FRAME_CONSUMER_METADATA),
            group=consumer_group,
            src=0,
            post_unpack_func=fail_first_load,
        )

    assert error.value is callback_error
    assert callback_names == [["first.weight"]]
    assert consumer_group.current_index == len(producer_group.broadcasted_tensors)


def test_packed_broadcast_lazy_consumer_drains_when_closed_early(monkeypatch):
    _force_cpu_packed_broadcast(monkeypatch)
    producer_group = _broadcast_two_single_tensor_frames()

    consumer_group = MockConsumerCommunicationGroup(producer_group.broadcasted_tensors)
    weights = packed_broadcast_consumer(
        iterator=iter(_TWO_FRAME_CONSUMER_METADATA),
        group=consumer_group,
        src=0,
        post_unpack_func=None,
        return_iterator=True,
    )

    assert weights is not None
    name, tensor = next(weights)
    assert name == "first.weight"
    assert torch.equal(tensor, torch.tensor([1.0]))
    assert consumer_group.current_index < len(producer_group.broadcasted_tensors)

    weights.close()

    assert consumer_group.current_index == len(producer_group.broadcasted_tensors)


def test_packed_broadcast_consumer_drains_after_metadata_failure(monkeypatch):
    _force_cpu_packed_broadcast(monkeypatch)
    producer_group = _broadcast_two_single_tensor_frames()

    metadata_error = RuntimeError("injected consumer metadata failure")

    def metadata():
        yield _TWO_FRAME_CONSUMER_METADATA[0]
        raise metadata_error

    loaded_names = []
    consumer_group = MockConsumerCommunicationGroup(producer_group.broadcasted_tensors)
    with pytest.raises(
        RuntimeError, match="injected consumer metadata failure"
    ) as error:
        packed_broadcast_consumer(
            iterator=metadata(),
            group=consumer_group,
            src=0,
            post_unpack_func=lambda tensors: loaded_names.extend(
                name for name, _ in tensors
            ),
        )

    assert error.value is metadata_error
    assert loaded_names == ["first.weight"]
    assert consumer_group.current_index == len(producer_group.broadcasted_tensors)


def test_packed_broadcast_consumer_drains_after_unpack_failure(monkeypatch):
    _force_cpu_packed_broadcast(monkeypatch)
    producer_group = _broadcast_two_single_tensor_frames()

    # A float dimension keeps the byte count equal but cannot be split.
    metadata = [
        ("first.weight", ((1.0,), torch.float32)),
        _TWO_FRAME_CONSUMER_METADATA[1],
    ]
    loaded_names = []
    consumer_group = MockConsumerCommunicationGroup(producer_group.broadcasted_tensors)
    with pytest.raises(TypeError):
        packed_broadcast_consumer(
            iterator=iter(metadata),
            group=consumer_group,
            src=0,
            post_unpack_func=lambda tensors: loaded_names.extend(
                name for name, _ in tensors
            ),
        )

    assert loaded_names == []
    assert consumer_group.current_index == len(producer_group.broadcasted_tensors)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_packed_broadcast_producer_carries_export_dependency_across_streams(
    monkeypatch,
):
    """Each buffer's stream must wait for the previous buffer's export before packing.

    ``full_tensor()`` registers the completion of its all-gather only on the
    stream that was current when it ran. When the iterator yields several views
    of one gathered tensor (per-expert weights) and the packer switches streams
    between them, the new stream has to inherit that dependency, and it must do
    so through an event recorded *before* the previous broadcast so the two
    buffers still overlap. The old code only synchronized the reused buffer's
    own stream and fails this test.
    """
    # Streams are identified by their CUDA handle: torch.cuda.current_stream()
    # returns a fresh Python wrapper on every call, so object identity would
    # never match the Stream objects the packer holds.
    events: list = []  # ("record", event_id, cuda_stream) / ("wait", cuda_stream, event_id) / ("broadcast", cuda_stream)

    real_event_cls = torch.cuda.Event

    class _Event(real_event_cls):
        def record(self, stream=None):
            stream = stream if stream is not None else torch.cuda.current_stream()
            events.append(("record", id(self), stream.cuda_stream))
            return super().record(stream)

    real_wait_event = torch.cuda.Stream.wait_event

    def wait_event(self, event):
        events.append(("wait", self.cuda_stream, id(event)))
        return real_wait_event(self, event)

    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda.Stream, "wait_event", wait_event)

    params = [(name, t.cuda()) for name, t in create_mock_model_params()]

    class _Group:
        def broadcast(self, tensor, src):
            # Frame headers share their payload's stream; track payloads only.
            if tensor.dtype == torch.uint8:
                events.append(("broadcast", torch.cuda.current_stream().cuda_stream))

    with patch(
        "nemo_rl.utils.packed_tensor.get_target_packed_tensor_size",
        return_value=2000,
    ):
        packed_broadcast_producer(
            iterator=iter(params),
            group=_Group(),
            src=0,
            post_iter_func=lambda x: x[1],
            num_buffers=2,
            # The readiness token goes out on the default stream. Skip it so
            # only payload broadcasts are checked.
            preflight_checked=True,
        )

    broadcasts = [i for i, e in enumerate(events) if e[0] == "broadcast"]
    assert len(broadcasts) >= 3, "need at least one buffer reuse to be meaningful"
    for prev_b, next_b in zip(broadcasts[:-1], broadcasts[1:]):
        prev_stream = events[prev_b][1]
        next_stream = events[next_b][1]
        assert prev_stream != next_stream, "buffers must alternate streams"
        # An event recorded on the previous stream before its broadcast ...
        recorded = {
            e[1] for e in events[:prev_b] if e[0] == "record" and e[2] == prev_stream
        }
        # ... is waited on by the next stream before the next broadcast.
        waited = {
            e[2]
            for e in events[prev_b:next_b]
            if e[0] == "wait" and e[1] == next_stream
        }
        assert recorded & waited, (
            "next buffer's stream did not wait on an export event recorded on the "
            "previous stream before the previous broadcast"
        )
