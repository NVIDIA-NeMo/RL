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

import nemo_rl.utils.packed_tensor as packed_tensor
from nemo_rl.utils.packed_tensor import (
    get_target_packed_tensor_size,
    packed_broadcast_consumer,
    packed_broadcast_producer,
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
    ]
    return params


def create_mock_state_dict_info(params):
    """Create state dict info (name -> (shape, dtype)) from params."""
    return {name: (tensor.shape, tensor.dtype) for name, tensor in params}


def test_explicit_packed_tensor_size_does_not_depend_on_local_gpu(monkeypatch):
    """A fixed size is safe for collectives spanning heterogeneous GPUs."""
    get_target_packed_tensor_size.cache_clear()
    monkeypatch.setenv("NRL_REFIT_BUFFER_SIZE_BYTES", "536870912")
    try:
        assert get_target_packed_tensor_size() == 512 * 1024**2
    finally:
        get_target_packed_tensor_size.cache_clear()


@pytest.mark.parametrize("value", ["not-an-int", "0", "-1", str(5 * 1024**3 + 1)])
def test_explicit_packed_tensor_size_rejects_invalid_values(monkeypatch, value):
    get_target_packed_tensor_size.cache_clear()
    monkeypatch.setenv("NRL_REFIT_BUFFER_SIZE_BYTES", value)
    try:
        with pytest.raises(ValueError, match="NRL_REFIT_BUFFER_SIZE_BYTES"):
            get_target_packed_tensor_size()
    finally:
        get_target_packed_tensor_size.cache_clear()


@pytest.mark.parametrize("role", ["producer", "consumer"])
def test_packed_broadcast_synchronizes_every_stream_before_return(
    monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    class FakeStream:
        def __init__(self) -> None:
            self.synchronize_calls = 0

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    streams: list[FakeStream] = []

    def make_stream() -> FakeStream:
        stream = FakeStream()
        streams.append(stream)
        return stream

    monkeypatch.setattr(packed_tensor, "get_num_buffers", lambda: 2)
    monkeypatch.setattr(packed_tensor, "get_target_packed_tensor_size", lambda: 1)
    monkeypatch.setattr(packed_tensor.torch.cuda, "Stream", make_stream)
    monkeypatch.setattr(
        packed_tensor.torch.cuda, "stream", lambda _stream: nullcontext()
    )
    monkeypatch.setattr(
        packed_tensor.torch, "empty", lambda *_args, **_kwargs: object()
    )

    if role == "producer":
        packed_broadcast_producer(
            iterator=iter(()),
            group=object(),
            src=0,
            post_iter_func=lambda tensor: tensor,
        )
    else:
        packed_broadcast_consumer(
            iterator=iter(()),
            group=object(),
            src=0,
            post_unpack_func=lambda tensors: tensors,
        )

    # The first loop iteration reuses stream 1, so it synchronizes once before
    # work and once at shutdown. Stream 0 is reached only by the shutdown loop.
    assert [stream.synchronize_calls for stream in streams] == [1, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_packed_broadcast_producer_consumer_roundtrip():
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
        )

        # Now test consumer with the broadcasted tensors
        consumer_group = MockConsumerCommunicationGroup(
            producer_group.broadcasted_tensors
        )

        # Create state dict info for consumer
        state_dict_info = create_mock_state_dict_info(params_cuda)

        # Store unpacked tensors
        unpacked_tensors = {}

        def post_unpack_func(tensor_list):
            """Store unpacked tensors for verification."""
            for name, tensor in tensor_list:
                unpacked_tensors[name] = tensor

        # Run consumer
        packed_broadcast_consumer(
            iterator=iter(state_dict_info.items()),
            group=consumer_group,
            src=0,
            post_unpack_func=post_unpack_func,
        )

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
    assert mock_group.broadcast_count == 1
    assert len(mock_group.broadcasted_tensors) == 1

    # Verify the size matches the large tensor
    expected_size = large_tensor.numel() * large_tensor.element_size()
    assert mock_group.broadcasted_tensors[0].numel() == expected_size


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
    total_broadcasted_size = sum(t.numel() for t in mock_group.broadcasted_tensors)
    expected_total_size = sum(t.numel() * t.element_size() for _, t in params)
    assert total_broadcasted_size == expected_total_size
