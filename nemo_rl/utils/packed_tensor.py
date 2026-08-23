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
from functools import lru_cache
from typing import Any, List, Tuple

import torch


_PACKED_FRAME_DATA = 0
_PACKED_FRAME_COMPLETE = 1
_PACKED_FRAME_ERROR = 2
_PACKED_FRAME_HEADER_WORDS = 3


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


def _broadcast_preflight_status(group, src: int, *, failed: bool) -> bool:
    """Make failure status the first collective in every packed refit."""
    status = torch.tensor(
        int(failed),
        dtype=torch.int32,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    group.broadcast(status, src=src)
    return bool(status.item())


def packed_broadcast_preflight_producer(
    group,
    src: int,
    error: Exception | None,
) -> None:
    """Publish producer readiness before any shared model-update payload."""
    if not _broadcast_preflight_status(group, src, failed=error is not None):
        return
    if error is not None:
        raise RuntimeError(str(error)) from error
    raise RuntimeError("Packed-broadcast producer preflight failed on source rank")


def packed_broadcast_preflight_consumer(group, src: int) -> None:
    """Receive producer readiness before entering shared payload collectives."""
    if _broadcast_preflight_status(group, src, failed=False):
        raise RuntimeError("Packed-broadcast producer preflight failed")


def _broadcast_packed_frame_header(
    group,
    src: int,
    *,
    status: int,
    num_bytes: int = 0,
    num_tensors: int = 0,
    readback: bool = True,
) -> tuple[int, int, int]:
    """Broadcast one fixed-size payload frame header."""
    header = torch.tensor(
        [status, num_bytes, num_tensors],
        dtype=torch.int64,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    group.broadcast(header, src=src)
    if not readback:
        return status, num_bytes, num_tensors
    received = header.cpu().tolist()
    if len(received) != _PACKED_FRAME_HEADER_WORDS:
        raise RuntimeError("Invalid packed-broadcast frame header")
    return int(received[0]), int(received[1]), int(received[2])


def packed_broadcast_producer(
    iterator,
    group,
    src,
    post_iter_func,
    *,
    buffer_size_bytes: int | None = None,
    num_buffers: int | None = None,
    preflight_error: Exception | None = None,
    preflight_checked: bool = False,
):
    """Broadcast a list of tensors in a packed manner.

    Args:
        iterator: iterator of model parameters. Returns a tuple of (name, tensor)
        group: process group (vllm PyNcclCommunicator)
        src: source rank (0 in current implementation)
        post_iter_func: function to apply to each tensor before packing, should return a tensor
        buffer_size_bytes: packed-buffer target. Uses the NeMo-RL default when unset.
        num_buffers: number of alternating CUDA buffers. Uses the default when unset.
        preflight_error: local producer error synchronized before any payload broadcast.
        preflight_checked: caller already completed the shared readiness collective.

    Returns:
        None

    Note:
        Synchronous Python iterator, export, and packing failures are published
        as an ERROR frame. Stream synchronization, CUDA allocation/kernel/OOM,
        and NCCL/broadcast failures are communicator-fatal and propagate
        directly; this protocol cannot use a failed CUDA context or transport
        to publish a terminal collective. Peer liveness in that case requires
        communicator teardown by the caller.

    """
    if not preflight_checked:
        packed_broadcast_preflight_producer(group, src, preflight_error)
    elif preflight_error is not None:
        raise RuntimeError(str(preflight_error)) from preflight_error

    target_packed_tensor_size = (
        get_target_packed_tensor_size()
        if buffer_size_bytes is None
        else buffer_size_bytes
    )

    num_buffers = get_num_buffers() if num_buffers is None else num_buffers
    streams = [torch.cuda.Stream() for _ in range(num_buffers)]
    buffer_idx = 0

    packing_tensor_list = [[] for _ in range(num_buffers)]
    packing_tensor_sizes = [0 for _ in range(num_buffers)]
    packed_tensors = [
        torch.empty(0, dtype=torch.uint8, device="cuda") for _ in range(num_buffers)
    ]

    iterator_exhausted = False
    while not iterator_exhausted:
        # Move to the next buffer
        buffer_idx = (buffer_idx + 1) % num_buffers
        # CUDA/NCCL failures at a stream fence are communicator-fatal. Do not
        # pretend that another collective on the same context can report them.
        streams[buffer_idx].synchronize()
        try:
            # Prepare a complete frame before publishing its DATA header. Any
            # synchronous Python export failure can then be reported without
            # leaving the consumer waiting for a payload that will never be
            # broadcast.
            with torch.cuda.stream(streams[buffer_idx]):  # type: ignore[arg-type]
                # Initialize the packing tensor list and sizes
                packing_tensor_list[buffer_idx] = []
                packing_tensor_sizes[buffer_idx] = 0
                # Pack the tensors
                try:
                    while True:
                        # Apply backend specific post processing and then convert to linearized uint8 tensor.
                        # contiguous() is required because the upstream iterator may
                        # yield non-contiguous tensors that view(...) cannot handle.
                        tensor = post_iter_func(next(iterator))
                        if tensor.device.type != "cuda":
                            # Everything here is concatenated into one buffer and
                            # broadcast over a CUDA collective, so a single host
                            # tensor anywhere in the stream fails the cat. The
                            # producer owns its buffer's device rather than
                            # trusting every upstream exporter to agree.
                            tensor = tensor.to(torch.cuda.current_device())
                        tensor = tensor.contiguous().reshape(-1).view(torch.uint8)
                        packing_tensor_list[buffer_idx].append(tensor)
                        packing_tensor_sizes[buffer_idx] += tensor.numel()
                        if packing_tensor_sizes[buffer_idx] > target_packed_tensor_size:
                            break
                except StopIteration:
                    iterator_exhausted = True
                if packing_tensor_list[buffer_idx]:
                    packed_tensors[buffer_idx] = torch.cat(
                        packing_tensor_list[buffer_idx], dim=0
                    )
        except BaseException:
            try:
                _broadcast_packed_frame_header(
                    group,
                    src,
                    status=_PACKED_FRAME_ERROR,
                )
            except BaseException:
                pass
            raise

        # Detect deferred device/transport failures before DATA publication,
        # but treat them as communicator-fatal rather than attempting another
        # collective on a potentially failed CUDA context.
        streams[buffer_idx].synchronize()
        with torch.cuda.stream(streams[buffer_idx]):  # type: ignore[arg-type]
            if packing_tensor_list[buffer_idx]:
                _broadcast_packed_frame_header(
                    group,
                    src,
                    status=_PACKED_FRAME_DATA,
                    num_bytes=packing_tensor_sizes[buffer_idx],
                    num_tensors=len(packing_tensor_list[buffer_idx]),
                    readback=False,
                )
                group.broadcast(packed_tensors[buffer_idx], src=src)
            if iterator_exhausted:
                _broadcast_packed_frame_header(
                    group,
                    src,
                    status=_PACKED_FRAME_COMPLETE,
                    readback=False,
                )

    # Join all packing/broadcast side streams before returning. Without this,
    # the caller may mutate or offload the source weights while the final
    # broadcasts are still in flight on the side streams (vLLM >= 0.25's
    # PyNcclCommunicator enqueues on the current stream without blocking).
    for s in streams:
        s.synchronize()


def packed_broadcast_consumer(
    iterator,
    group,
    src,
    post_unpack_func,
    *,
    preflight_checked: bool = False,
):
    """Consume a packed tensor and unpack it into a list of tensors.

    Args:
        iterator: iterator of model parameters. Returns a tuple of (name, tensor)
        group: process group (vllm PyNcclCommunicator)
        src: source rank (0 in current implementation)
        post_unpack_func: function to apply to each tensor after unpacking

    Returns:
        None

    Note:
        Synchronous Python unpack and load-callback failures are preserved while
        remaining frames are drained through the terminal frame. Stream
        synchronization, CUDA allocation/kernel/OOM, and NCCL/broadcast failures
        are communicator-fatal and propagate directly because further
        collectives on that context or transport are not reliable. Peer liveness
        in that case requires communicator teardown by the caller.

    """
    if not preflight_checked:
        packed_broadcast_preflight_consumer(group, src)

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

        def restore_tensor(
            tensor: torch.Tensor, shape: torch.Size | list[int], dtype: torch.dtype
        ) -> torch.Tensor:
            """Restore dtype and shape for a tensor from the packed byte stream.

            Unlike the 512-byte-aligned IPC/ZMQ refit path, packed collective
            refit adds no padding between tensors. Scalar GEMM or K/V amax can
            therefore leave the next mixed-dtype slice unaligned. Cloning moves
            only such slices to offset zero. ``reshape(tuple(shape))`` accepts an
            empty tuple and therefore also restores scalar tensors.
            """
            if tensor.storage_offset() % dtype.itemsize:
                tensor = tensor.clone()
            return tensor.view(dtype).reshape(tuple(shape))

        unpacked_list = [
            (
                meta_data_list[i][0],
                restore_tensor(tensor, meta_data_list[i][1], meta_data_list[i][2]),
            )
            for i, tensor in enumerate(unpacked_tensor)
        ]

        return unpacked_list

    num_buffers = get_num_buffers()
    streams = [torch.cuda.Stream() for _ in range(num_buffers)]
    buffer_idx = 0

    packing_tensor_meta_data = [[] for _ in range(num_buffers)]
    packing_tensor_sizes = [0 for _ in range(num_buffers)]
    offsets = [0 for _ in range(num_buffers)]
    packed_tensors = [
        torch.empty(0, dtype=torch.uint8, device="cuda") for _ in range(num_buffers)
    ]

    protocol_error: str | None = None
    consumer_error: BaseException | None = None
    while True:
        # Move to the next buffer
        buffer_idx = (buffer_idx + 1) % num_buffers
        # A deferred CUDA/NCCL failure invalidates the drain transport itself.
        streams[buffer_idx].synchronize()
        with torch.cuda.stream(streams[buffer_idx]):  # type: ignore[arg-type]
            status, packed_size, tensor_count = _broadcast_packed_frame_header(
                group,
                src,
                status=_PACKED_FRAME_DATA,
            )
            if status == _PACKED_FRAME_ERROR:
                if consumer_error is None:
                    consumer_error = RuntimeError(
                        "Packed-broadcast producer failed during payload transfer"
                    )
                break
            if status == _PACKED_FRAME_COMPLETE:
                try:
                    next(iterator)
                except StopIteration:
                    break
                if protocol_error is None:
                    protocol_error = (
                        "Packed-broadcast producer completed before consumer metadata"
                    )
                break
            if status != _PACKED_FRAME_DATA or packed_size < 0 or tensor_count <= 0:
                raise RuntimeError(
                    "Invalid packed-broadcast DATA frame: "
                    f"status={status}, bytes={packed_size}, tensors={tensor_count}"
                )

            # Initialize the packing tensor meta data
            packing_tensor_meta_data[buffer_idx] = []
            packing_tensor_sizes[buffer_idx] = 0
            offsets[buffer_idx] = 0
            for _ in range(tensor_count):
                try:
                    name, (shape, dtype) = next(iterator)
                except StopIteration:
                    protocol_error = "Packed-broadcast producer sent more tensors than consumer metadata"
                    continue
                if protocol_error is None:
                    tensor_size = math.prod(shape) * dtype.itemsize
                    packing_tensor_meta_data[buffer_idx].append(
                        (name, shape, dtype, offsets[buffer_idx], tensor_size)
                    )
                    packing_tensor_sizes[buffer_idx] += tensor_size
                    offsets[buffer_idx] += tensor_size

            packed_tensors[buffer_idx] = torch.empty(
                packed_size, dtype=torch.uint8, device="cuda"
            )
            group.broadcast(packed_tensors[buffer_idx], src=src)
            if (
                protocol_error is None
                and packing_tensor_sizes[buffer_idx] != packed_size
            ):
                protocol_error = (
                    "Packed-broadcast payload size does not match consumer metadata: "
                    f"producer={packed_size}, consumer={packing_tensor_sizes[buffer_idx]}"
                )
            if protocol_error is None and consumer_error is None:
                try:
                    post_unpack_func(
                        unpack_tensor(
                            packed_tensors[buffer_idx],
                            packing_tensor_meta_data[buffer_idx],
                        )
                    )
                except BaseException as error:
                    consumer_error = error

    # Join all recv/unpack/load side streams before returning. Without this,
    # generation can start reading model weights while the final unpack/load
    # copies are still in flight on the side streams, producing garbage
    # logprobs (vLLM >= 0.25's PyNcclCommunicator enqueues on the current
    # stream without blocking).
    for s in streams:
        s.synchronize()

    if consumer_error is not None:
        raise consumer_error
    if protocol_error is not None:
        raise RuntimeError(protocol_error)
