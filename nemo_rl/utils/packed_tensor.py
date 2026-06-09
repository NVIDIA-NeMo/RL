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

import math
import os
from collections import deque
from functools import lru_cache
from typing import Any, Callable, Deque, Iterator, List, Optional, Tuple

import torch


@lru_cache(maxsize=1)
def get_target_packed_tensor_size():
    memory_ratio = os.getenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0.02")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    total_memory_bytes = props.total_memory
    # max size is 5GB
    target_size = min(int(total_memory_bytes * float(memory_ratio)), 5 * 1024**3)
    return target_size


@lru_cache(maxsize=1)
def get_num_buffers():
    return int(os.getenv("NRL_REFIT_NUM_BUFFERS", "2"))


@lru_cache(maxsize=1)
def get_prefetch_depth():
    return int(os.getenv("NRL_REFIT_PREFETCH_DEPTH", "4"))


# Cached resources. Both producer and consumer are called many times per
# training run (once per refit), and ``packed_broadcast_producer`` /
# ``packed_broadcast_consumer`` historically created fresh CUDA streams and
# ``torch.cat``-allocated fresh packed buffers on every call. On large
# models the per-refit packing broadcasts ~10-20 buffers; the combined
# allocator churn is visible in refit traces. We cache streams and packed
# buffers module-local (per-process; producer and consumer live in
# separate processes so they don't contend).


_cached_streams: dict[str, list[torch.cuda.Stream]] = {} # role -> list[Stream]
_cached_buffers: dict[tuple[str, int], list[torch.Tensor]] = {} # (role, size_bytes) -> list[Tensor]

def _get_cached_streams(role: str, num_buffers: int) -> list[torch.cuda.Stream]:
    """Return ``num_buffers`` CUDA streams, lazily allocated once per role.

    ``role`` must be ``"producer"`` or ``"consumer"`` so a process that plays
    both roles (uncommon but possible in tests) gets independent stream
    pools instead of accidental cross-talk.
    """
    streams = _cached_streams.get(role)
    if streams is None:
        streams = [torch.cuda.Stream() for _ in range(num_buffers)]
        _cached_streams[role] = streams
    return streams


def _get_cached_buffers(
    role: str, num_buffers: int, size_bytes: int
) -> list[torch.Tensor]:
    """Return ``num_buffers`` persistent uint8 CUDA buffers of ``size_bytes``.

    Buffers are reallocated only if ``size_bytes`` grows beyond the cached
    capacity (e.g. a single weight tensor exceeds ``target_packed_tensor_size``
    and forces an oversized broadcast). The returned buffers are always
    sliced by the caller to the actual packed size before ``group.broadcast``
    so correctness is unaffected by reuse at a larger capacity.
    """
    key = (role, size_bytes)
    buffers = _cached_buffers.get(key)
    if buffers is not None:
        return buffers

    # Drop any smaller-capacity cache entries for the same role.
    # We only keep the largest capacity we've ever needed.
    for existing_key in list(_cached_buffers.keys()):
        if existing_key[0] == role:
            del _cached_buffers[existing_key]

    buffers = [
        torch.empty(size_bytes, dtype=torch.uint8, device="cuda")
        for _ in range(num_buffers)
    ]
    _cached_buffers[key] = buffers
    return buffers


class _PrefetchingTensorIterator:
    """Runs ``post_iter_func(next(iterator))`` ahead on a side CUDA stream.

    Keeps up to ``depth`` converted tensors queued with per-item CUDA events
    so the consumer can order its packing stream on the producer stream via
    ``stream.wait_event(event)`` — without any CPU-side synchronize.

    Each produced element is the result of
    ``post_iter_func(next(iterator)).view(torch.uint8).view(-1)``, i.e. a
    linearized uint8 view suitable for direct ``torch.cat`` into the packing
    buffer. Doing the view here (rather than in the consumer) keeps the
    prefetch stream as the sole writer to the underlying storage and
    simplifies ordering.

    When ``depth <= 0`` the prefetcher degrades to an eager no-prefetch path
    that records an event on the default stream (semantically identical to
    the pre-prefetch implementation apart from the extra event), so callers
    can always consume via the same API.
    """

    def __init__(
        self,
        iterator: Iterator[Any],
        post_iter_func: Callable[[Any], torch.Tensor],
        depth: int,
    ) -> None:
        self._iterator = iterator
        self._post_iter_func = post_iter_func
        self._depth = max(depth, 0)
        self._queue: Deque[Tuple[Optional[torch.cuda.Event], torch.Tensor]] = deque()
        self._stream: torch.cuda.Stream = torch.cuda.Stream()
        self._exhausted = False

    def try_prefetch(self) -> None:
        """Top up the internal queue to ``depth`` without blocking.

        Safe to call from any stream context; the prefetch work is issued on
        the dedicated prefetch stream. Silently stops once the underlying
        iterator is exhausted.
        """
        if self._exhausted or self._depth == 0:
            return
        while len(self._queue) < self._depth:
            try:
                with torch.cuda.stream(self._stream):  # type: ignore[arg-type]
                    tensor = (
                        self._post_iter_func(next(self._iterator))
                        .contiguous()
                        .view(torch.uint8)
                        .view(-1)
                    )
                    event = torch.cuda.Event()
                    event.record(self._stream)  # type: ignore[arg-type]
                self._queue.append((event, tensor))
            except StopIteration:
                self._exhausted = True
                return

    def next_on(
        self, consumer_stream: torch.cuda.Stream
    ) -> torch.Tensor:
        """Return the next prefetched tensor, ordered on ``consumer_stream``.

        Raises ``StopIteration`` when the underlying iterator is drained.
        """
        if self._depth == 0:
            # No prefetch: do the conversion inline on the consumer stream.
            tensor = (
                self._post_iter_func(next(self._iterator))
                .contiguous()
                .view(torch.uint8)
                .view(-1)
            )
            return tensor

        if not self._queue:
            self.try_prefetch()
        if not self._queue:
            raise StopIteration
        event, tensor = self._queue.popleft()
        # Ensure the consumer stream observes the prefetch stream's writes.
        consumer_stream.wait_event(event)
        return tensor


def packed_broadcast_producer(iterator, group, src, post_iter_func):
    """Broadcast a list of tensors in a packed manner.

    Args:
        iterator: iterator of model parameters. Returns a tuple of (name, tensor)
        group: process group (vllm PyNcclCommunicator)
        src: source rank (0 in current implementation)
        post_iter_func: function to apply to each tensor before packing, should return a tensor

    Returns:
        None

    Concurrency:
        Uses ``num_buffers`` packing streams (double-buffering by default) to
        overlap the outbound ``group.broadcast`` of packed buffer N with
        packing of buffer N+1, and a dedicated prefetch stream (controlled by
        ``NRL_REFIT_PREFETCH_DEPTH``) so ``post_iter_func(next(iterator))``
        — which for the Megatron path issues TP-gather + PP-broadcast on
        communicators *disjoint* from the inference-cluster ``group`` — runs
        ahead of where the packing stream currently is. This lets task N+1's
        TP/PP collectives execute concurrently with task N's outbound packed
        broadcast on the GPU, closing the serialization gap called out in
        ``investigations/megatron-non-colocated-refit-bottlenecks.md`` §6.
    """
    target_packed_tensor_size = get_target_packed_tensor_size()

    num_buffers = get_num_buffers()
    # Streams and packed buffers are cached across refits to avoid per-call
    # ``torch.cuda.Stream()`` construction and large uint8 buffer churn
    # through the PyTorch caching allocator. ``torch.cat`` below writes into
    # a slice of the persistent buffer via ``out=`` and the broadcast uses
    # the slice, so correctness is unaffected.
    streams = _get_cached_streams("producer", num_buffers)
    buffer_idx = 0

    packing_tensor_list: list[list[torch.Tensor]] = [[] for _ in range(num_buffers)]
    packing_tensor_sizes = [0 for _ in range(num_buffers)]
    # Pre-allocate ``target_packed_tensor_size`` per buffer. ``_pack_and_broadcast``
    # will grow the cache on demand if a single packed buffer overshoots
    # ``target_packed_tensor_size`` (the pack loop only breaks *after* exceeding).
    packed_buffers = _get_cached_buffers(
        "producer", num_buffers, target_packed_tensor_size
    )

    def _pack_and_broadcast(idx: int) -> None:
        nonlocal packed_buffers
        total_size = packing_tensor_sizes[idx]
        if total_size > packed_buffers[idx].numel():
            # Grow the cache for both buffers (keeps them uniformly sized so
            # a subsequent buffer swap doesn't re-allocate). The cache
            # helper drops any smaller-capacity entries for this role, so
            # we don't leak memory across growths.
            packed_buffers = _get_cached_buffers("producer", num_buffers, total_size)
        out_view = packed_buffers[idx].narrow(0, 0, total_size)
        torch.cat(packing_tensor_list[idx], dim=0, out=out_view)
        group.broadcast(out_view, src=src)

    prefetcher = _PrefetchingTensorIterator(
        iterator=iterator,
        post_iter_func=post_iter_func,
        depth=get_prefetch_depth(),
    )
    # Prime the pipeline so the very first buffer's packing stream finds
    # tasks already converted and waiting.
    prefetcher.try_prefetch()

    while True:
        # Move to the next buffer
        buffer_idx = (buffer_idx + 1) % num_buffers
        # Synchronize the current stream
        streams[buffer_idx].synchronize()
        # Start tasks for the new buffer in a new stream
        with torch.cuda.stream(streams[buffer_idx]):  # type: ignore[arg-type]
            try:
                # Initialize the packing tensor list and sizes
                packing_tensor_list[buffer_idx] = []
                packing_tensor_sizes[buffer_idx] = 0
                # Pack the tensors
                while True:
                    # Apply backend specific post processing and then convert to linearized uint8 tensor
                    tensor = prefetcher.next_on(streams[buffer_idx])
                    packing_tensor_list[buffer_idx].append(tensor)
                    packing_tensor_sizes[buffer_idx] += tensor.view(torch.uint8).numel()
                    # Refill prefetch queue as we consume so task N+depth's
                    # TP/PP work can start while we're still packing task N.
                    prefetcher.try_prefetch()
                    if packing_tensor_sizes[buffer_idx] > target_packed_tensor_size:
                        break
                # Pack into the persistent buffer and broadcast.
                _pack_and_broadcast(buffer_idx)
                # Nudge the prefetcher once more so the next buffer's tasks
                # begin enqueuing TP/PP NCCL work on the prefetch stream
                # concurrently with this buffer's outbound group.broadcast.
                prefetcher.try_prefetch()
            except StopIteration:
                # do the last broadcast if there are remaining tensors
                if len(packing_tensor_list[buffer_idx]) > 0:
                    _pack_and_broadcast(buffer_idx)
                break


def packed_broadcast_consumer(iterator, group, src, post_unpack_func):
    """Consume a packed tensor and unpack it into a list of tensors.

    Args:
        iterator: iterator of model parameters. Returns a tuple of (name, tensor)
        group: process group (vllm PyNcclCommunicator)
        src: source rank (0 in current implementation)
        post_unpack_func: function to apply to each tensor after unpacking

    Returns:
        None

    """

    def unpack_tensor(
        packed_tensor: torch.Tensor, meta_data_list: list[Any]
    ) -> List[Tuple[str, torch.Tensor]]:
        """Unpack a single tensor into a list of tensors.

        Args:
            packed_tensor: the packed torch.uint8 tensor to unpack
            meta_data_list: List[(name, shape, dtype, offset, tensor_size)]

        Returns:
            unpacked List[(name, tensor)]
        """
        unpacked_list = []
        # Perform batched split with torch.split_with_sizes
        packed_tensor_sizes = list(map(lambda x: x[4], meta_data_list))
        unpacked_tensor = packed_tensor.split_with_sizes(packed_tensor_sizes)

        # unpacked_list = List[(name, torch.Tensor.view(dtype).view(*shape))]
        unpacked_list = [
            (
                meta_data_list[i][0],
                tensor.view(meta_data_list[i][2]).view(*meta_data_list[i][1]),
            )
            for i, tensor in enumerate(unpacked_tensor)
        ]

        return unpacked_list

    target_packed_tensor_size = get_target_packed_tensor_size()

    num_buffers = get_num_buffers()
    # Same caching rationale as the producer: reuse streams and packed
    # receive buffers across refits. Each iteration slices the persistent
    # buffer to the exact packed size before ``group.broadcast`` so NCCL
    # sees the right receive length.
    streams = _get_cached_streams("consumer", num_buffers)
    buffer_idx = 0

    packing_tensor_meta_data: list[list[Any]] = [[] for _ in range(num_buffers)]
    packing_tensor_sizes = [0 for _ in range(num_buffers)]
    offsets = [0 for _ in range(num_buffers)]
    packed_buffers = _get_cached_buffers(
        "consumer", num_buffers, target_packed_tensor_size
    )

    def _recv_and_unpack(idx: int) -> None:
        nonlocal packed_buffers
        total_size = packing_tensor_sizes[idx]
        if total_size > packed_buffers[idx].numel():
            packed_buffers = _get_cached_buffers("consumer", num_buffers, total_size)
        recv_view = packed_buffers[idx].narrow(0, 0, total_size)
        group.broadcast(recv_view, src=src)
        post_unpack_func(unpack_tensor(recv_view, packing_tensor_meta_data[idx]))

    while True:
        # Move to the next buffer
        buffer_idx = (buffer_idx + 1) % num_buffers
        # Synchronize the current stream
        streams[buffer_idx].synchronize()
        with torch.cuda.stream(streams[buffer_idx]):  # type: ignore[arg-type]
            # Initialize the packing tensor meta data
            packing_tensor_meta_data[buffer_idx] = []
            packing_tensor_sizes[buffer_idx] = 0
            offsets[buffer_idx] = 0
            try:
                # Form a packed tensor
                while True:
                    name, (shape, dtype) = next(iterator)
                    tensor_size = math.prod(shape) * dtype.itemsize
                    packing_tensor_meta_data[buffer_idx].append(
                        (name, shape, dtype, offsets[buffer_idx], tensor_size)
                    )
                    packing_tensor_sizes[buffer_idx] += tensor_size
                    offsets[buffer_idx] += tensor_size
                    if packing_tensor_sizes[buffer_idx] > target_packed_tensor_size:
                        break
                _recv_and_unpack(buffer_idx)
            except StopIteration:
                # do the last broadcast if there are remaining tensors
                if len(packing_tensor_meta_data[buffer_idx]) > 0:
                    _recv_and_unpack(buffer_idx)
                break
