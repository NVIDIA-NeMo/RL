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

import queue
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from nemo_rl.utils import weight_transfer_sparse_codec as sparse_codec
from nemo_rl.utils.packed_tensor import get_target_packed_tensor_size
from nemo_rl.utils.weight_transfer_delta_tracker import DeltaCompressionTracker
from nemo_rl.utils.weight_transfer_protocol import (
    G_DELTA_UPDATE_KIND,
    G_DENSE_TRANSPORT,
    G_FULL_UPDATE_KIND,
    G_SPARSE_INDICES_TRANSPORT,
    G_TRANSFER_DONE_KIND,
    HeaderRefs,
    NamedTensor,
    QueuedPayload,
    SparseBucketPayload,
    TensorBatch,
    TensorPayload,
    WeightLoadFunc,
    WeightTransferKind,
    advance_chunk,
    broadcast_header,
    cuda_device,
    cuda_streams,
    next_chunk,
    normalize_device,
    pack_named_tensors,
    record_header_stream,
    record_payload_readiness_events,
    record_stream_event,
    record_tensor_stream,
    recv_payload,
    sync_streams,
    synchronize_current_transfer_stream,
    unpack_named_tensors,
    use_stream,
    wait_for_payload_events,
    wire_bytes,
)

__all__ = [
    "G_DELTA_UPDATE_KIND",
    "G_DENSE_TRANSPORT",
    "G_FULL_UPDATE_KIND",
    "G_SPARSE_INDICES_TRANSPORT",
    "G_TRANSFER_DONE_KIND",
    "WeightTransferResult",
    "pack_named_tensors",
    "packed_weight_transfer_consumer",
    "packed_weight_transfer_producer",
    "unpack_named_tensors",
]


@dataclass(frozen=True)
class WeightTransferResult:
    loaded_any: bool
    is_delta_sync: bool


class _PendingWirePayloads:
    """Queues wire payloads while sparse delta buckets are still coalescing."""

    def __init__(
        self,
        *,
        transfer_device: torch.device,
        sparse_bucket_size_bytes: int,
        is_delta_sync: bool,
    ) -> None:
        self._transfer_device = transfer_device
        self._sparse_bucket_size_bytes = sparse_bucket_size_bytes
        self._is_delta_sync = is_delta_sync
        self._queued: list[QueuedPayload] = []
        self._sparse_bucket: list[SparseBucketPayload] = []
        self._sparse_bucket_bytes = 0

    @property
    def queued_count(self) -> int:
        return len(self._queued)

    def queue_empty_delta_payload(self) -> None:
        self._queued.append(
            (
                G_DELTA_UPDATE_KIND,
                ([], G_SPARSE_INDICES_TRANSPORT, []),
                (),
            )
        )

    def queue_payload(self, kind: WeightTransferKind, payload: TensorPayload) -> None:
        if kind != G_DELTA_UPDATE_KIND:
            self.flush_sparse_bucket()
            self._queued.append((kind, payload, record_payload_readiness_events()))
            return

        tensors, _, metadata = payload
        payload_bytes = wire_bytes(tensors)
        if (
            self._sparse_bucket
            and self._sparse_bucket_bytes + payload_bytes
            > self._sparse_bucket_size_bytes
        ):
            self.flush_sparse_bucket()

        if metadata:
            self._sparse_bucket.append((payload, record_payload_readiness_events()))
            self._sparse_bucket_bytes += payload_bytes

        if self._sparse_bucket_bytes >= self._sparse_bucket_size_bytes:
            self.flush_sparse_bucket()

    def flush_sparse_bucket(self) -> None:
        if not self._sparse_bucket:
            return

        payloads = [payload for payload, _ in self._sparse_bucket]
        ready_events = tuple(
            event for _, events in self._sparse_bucket for event in events
        )
        wait_for_payload_events(ready_events)
        self._queued.append(
            (
                G_DELTA_UPDATE_KIND,
                payloads[0]
                if len(payloads) == 1
                else sparse_codec.merge_sparse_payloads(payloads),
                record_payload_readiness_events(),
            )
        )
        self._sparse_bucket = []
        self._sparse_bucket_bytes = 0

    def pop(self) -> tuple[dict[str, Any], torch.Tensor]:
        kind, payload, ready_events = self._queued.pop(0)
        wait_for_payload_events(ready_events)
        tensors, transport, metadata = payload
        if not tensors:
            return (
                {
                    "kind": kind,
                    "transport": transport,
                    "payload_entries": [],
                    "payload_numel": 0,
                    "sparse_metadata": metadata,
                    "is_delta_sync": self._is_delta_sync,
                },
                torch.empty(0, dtype=torch.uint8, device=self._transfer_device),
            )

        packed, entries = pack_named_tensors(tensors)
        return (
            {
                "kind": kind,
                "transport": transport,
                "payload_entries": entries,
                "payload_numel": int(packed.numel()),
                "sparse_metadata": metadata,
                "is_delta_sync": self._is_delta_sync,
            },
            packed,
        )


def packed_weight_transfer_producer(
    iterator: Iterable[NamedTensor],
    *,
    group: Any,
    src: int,
    delta_tracker: DeltaCompressionTracker | None = None,
) -> None:
    """Broadcast full or delta weight chunks with explicit chunk metadata."""
    encode_streams = cuda_streams()
    broadcast_streams = cuda_streams()
    buffer_idx = 0
    transfer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    header_refs: list[HeaderRefs | None] = [None for _ in broadcast_streams]
    payload_refs: list[torch.Tensor | None] = [None for _ in broadcast_streams]
    # The source reads one chunk ahead before sending the current payload. Peer
    # ranks must advance the same amount because Megatron export can issue
    # collectives while yielding source-rank tensors.
    source_lookahead_chunks = 2

    if group.rank != src:
        pending_item = None
        iterator_exhausted = False
        tensor_iterator = iter(iterator)

        def prefetch_non_source_chunk() -> None:
            nonlocal pending_item, iterator_exhausted
            if iterator_exhausted:
                return
            pending_item, iterator_exhausted = advance_chunk(
                tensor_iterator,
                get_target_packed_tensor_size(),
                pending_item=pending_item,
            )

        for _ in range(source_lookahead_chunks):
            prefetch_non_source_chunk()

        while True:
            with use_stream(broadcast_streams, buffer_idx):
                header, header_ref = broadcast_header(
                    {},
                    group=group,
                    src=src,
                    device=transfer_device,
                )
                header_refs[buffer_idx] = header_ref
                record_header_stream(header_ref)
                if header["kind"] == G_TRANSFER_DONE_KIND:
                    break
                payload = recv_payload(
                    int(header["payload_numel"]),
                    group=group,
                    src=src,
                    device=transfer_device,
                )
                payload_refs[buffer_idx] = payload
                record_tensor_stream(payload)
            prefetch_non_source_chunk()
            buffer_idx = (buffer_idx + 1) % len(broadcast_streams)
        sync_streams(broadcast_streams)
        synchronize_current_transfer_stream(transfer_device)
        return

    target_chunk_size = get_target_packed_tensor_size()
    tensor_iterator = iter(iterator)
    pending_item = None
    is_delta_sync = (
        delta_tracker.is_delta_sync() if delta_tracker is not None else False
    )
    pending_payloads = _PendingWirePayloads(
        transfer_device=transfer_device,
        sparse_bucket_size_bytes=(
            delta_tracker.sparse_bucket_size_bytes if delta_tracker is not None else 0
        ),
        is_delta_sync=is_delta_sync,
    )

    def queue_chunk(chunk: TensorBatch) -> None:
        if delta_tracker is None:
            pending_payloads.queue_payload(
                G_FULL_UPDATE_KIND, (chunk, G_DENSE_TRANSPORT, [])
            )
            return

        queued_before = pending_payloads.queued_count
        is_delta, tensors = delta_tracker.prepare_chunk(chunk)
        if not is_delta:
            pending_payloads.queue_payload(
                G_FULL_UPDATE_KIND, (tensors, G_DENSE_TRANSPORT, [])
            )
            return

        try:
            payload = sparse_codec.encode_sparse_indices(tensors)
        except sparse_codec.SparseEncodingUnavailable:
            pending_payloads.queue_payload(
                G_FULL_UPDATE_KIND, (chunk, G_DENSE_TRANSPORT, [])
            )
            return
        pending_payloads.queue_payload(G_DELTA_UPDATE_KIND, payload)
        if pending_payloads.queued_count == queued_before:
            # Keep one control message per source chunk so non-source ranks can
            # advance Megatron export collectives in lockstep with source lookahead.
            pending_payloads.queue_empty_delta_payload()

    def send(
        header: Mapping[str, Any],
        payload: torch.Tensor,
        event: torch.cuda.Event | None,
    ) -> None:
        nonlocal buffer_idx
        with use_stream(broadcast_streams, buffer_idx):
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
            _, header_ref = broadcast_header(
                header,
                group=group,
                src=src,
                device=payload.device,
            )
            header_refs[buffer_idx] = header_ref
            record_header_stream(header_ref)
            payload_refs[buffer_idx] = payload
            if payload.numel() > 0:
                group.broadcast(payload, src=src)
                record_tensor_stream(payload)
        buffer_idx = (buffer_idx + 1) % len(broadcast_streams)

    read_idx = 0
    encode_idx = 0
    pack_idx = 0

    def read_next_chunk() -> TensorBatch:
        nonlocal pending_item, read_idx
        with use_stream(encode_streams, read_idx):
            chunk, pending_item = next_chunk(
                tensor_iterator,
                target_chunk_size,
                pending_item=pending_item,
            )
        read_idx = (read_idx + 1) % len(encode_streams)
        return chunk

    def prepare_chunk_payload(
        chunk: TensorBatch,
    ) -> tuple[
        dict[str, Any],
        torch.Tensor,
        torch.cuda.Event | None,
    ]:
        nonlocal encode_idx, pack_idx
        with use_stream(encode_streams, encode_idx):
            queue_chunk(chunk)
        encode_idx = (encode_idx + 1) % len(encode_streams)
        with use_stream(encode_streams, pack_idx):
            header, payload = pending_payloads.pop()
            event = record_stream_event(encode_streams[pack_idx])
        pack_idx = (pack_idx + 1) % len(encode_streams)
        return header, payload, event

    try:
        chunk = read_next_chunk()
        while chunk:
            header, payload, event = prepare_chunk_payload(chunk)
            next_weight_chunk = read_next_chunk()
            send(header, payload, event)
            chunk = next_weight_chunk

        pending_payloads.flush_sparse_bucket()
        while pending_payloads.queued_count > 0:
            with use_stream(encode_streams, pack_idx):
                header, payload = pending_payloads.pop()
                event = record_stream_event(encode_streams[pack_idx])
            pack_idx = (pack_idx + 1) % len(encode_streams)
            send(header, payload, event)

        sync_streams(encode_streams)
        sync_streams(broadcast_streams)
        _, header_refs[0] = broadcast_header(
            {"kind": G_TRANSFER_DONE_KIND},
            group=group,
            src=src,
            device=transfer_device,
        )
        synchronize_current_transfer_stream(transfer_device)
    except Exception:
        if delta_tracker is not None:
            delta_tracker.on_sync_failed()
        raise

    if delta_tracker is not None:
        delta_tracker.on_sync_succeeded()


def packed_weight_transfer_consumer(
    *,
    group: Any,
    src: int,
    load_full_weights_func: WeightLoadFunc,
    load_delta_weights_func: WeightLoadFunc,
    device: torch.device | int | str,
    delta_load_batch_size_bytes: int | None = None,
) -> WeightTransferResult:
    """Receive full or delta chunks from ``packed_weight_transfer_producer``."""
    if delta_load_batch_size_bytes is not None and delta_load_batch_size_bytes < 1:
        raise ValueError("delta_load_batch_size_bytes must be >= 1 when set.")

    streams = cuda_streams(device)
    buffer_idx = 0
    header_refs: list[HeaderRefs | None] = [None for _ in streams]
    payload_refs: list[torch.Tensor | None] = [None for _ in streams]
    load_queue: _AsyncWeightLoadQueue | None = None
    decode_queue: _AsyncSparseDecodeQueue | None = None
    if delta_load_batch_size_bytes is not None:
        load_queue = _AsyncWeightLoadQueue(
            device=device,
            max_pending_batches=len(streams),
        )
        decode_queue = _AsyncSparseDecodeQueue(
            device=device,
            byte_cap=delta_load_batch_size_bytes,
            load_queue=load_queue,
            load_delta_weights_func=load_delta_weights_func,
            max_pending_payloads=len(streams),
        )

    transfer_done = False
    loaded_any = False
    is_delta_sync = False
    try:
        while True:
            with use_stream(streams, buffer_idx):
                header, header_ref = broadcast_header(
                    {},
                    group=group,
                    src=src,
                    device=device,
                )
                header_refs[buffer_idx] = header_ref
                record_header_stream(header_ref)
                if header["kind"] == G_TRANSFER_DONE_KIND:
                    transfer_done = True
                    break
                is_delta_sync = is_delta_sync or bool(header.get("is_delta_sync"))

                payload_numel = int(header["payload_numel"])
                payload_tensors: TensorBatch = []
                if payload_numel > 0:
                    payload = recv_payload(
                        payload_numel,
                        group=group,
                        src=src,
                        device=device,
                    )
                    payload_refs[buffer_idx] = payload
                    record_tensor_stream(payload)
                    payload_tensors = unpack_named_tensors(
                        payload,
                        entries=header["payload_entries"],
                    )
                else:
                    payload_refs[buffer_idx] = None

                if header["kind"] == G_FULL_UPDATE_KIND:
                    if load_queue is None:
                        load_full_weights_func(payload_tensors)
                    else:
                        decode_queue.flush_pending()
                        event = record_stream_event(streams[buffer_idx])
                        load_queue.enqueue(
                            load_full_weights_func,
                            payload_tensors,
                            ready_events=[] if event is None else [event],
                        )
                    loaded_any = True
                elif payload_numel > 0:
                    event = record_stream_event(streams[buffer_idx])
                    decode_queue.enqueue(
                        payload_tensors,
                        header["sparse_metadata"],
                        ready_events=[] if event is None else [event],
                    )
                    loaded_any = True

            if decode_queue is not None:
                decode_queue.raise_if_failed()
            if load_queue is not None:
                load_queue.raise_if_failed()
            buffer_idx = (buffer_idx + 1) % len(streams)
    finally:
        if decode_queue is not None:
            decode_queue.close()
        if transfer_done:
            sync_streams(streams)
        if load_queue is not None:
            load_queue.close()
    return WeightTransferResult(loaded_any=loaded_any, is_delta_sync=is_delta_sync)


class _AsyncWeightLoadQueue:
    def __init__(
        self,
        *,
        device: torch.device | int | str,
        max_pending_batches: int,
    ) -> None:
        self._device = device
        self._requests: queue.Queue[
            tuple[WeightLoadFunc, TensorBatch, list[torch.cuda.Event]] | None
        ] = queue.Queue(maxsize=max_pending_batches)
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="nemo-weight-load",
            daemon=True,
        )
        self._thread.start()

    def enqueue(
        self,
        load_func: WeightLoadFunc,
        batch: TensorBatch,
        *,
        ready_events: list[torch.cuda.Event],
    ) -> None:
        if batch:
            self.raise_if_failed()
            self._requests.put((load_func, batch, ready_events))
            self.raise_if_failed()

    def close(self) -> None:
        self._requests.put(None)
        self._thread.join()
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def _run(self) -> None:
        with cuda_device(normalize_device(self._device)):
            while True:
                request = self._requests.get()
                try:
                    if request is None:
                        return
                    if self._error is None:
                        load_func, batch, events = request
                        for event in events:
                            torch.cuda.current_stream().wait_event(event)
                        load_func(batch)
                        synchronize_current_transfer_stream(self._device)
                except Exception as error:
                    self._error = error
                finally:
                    self._requests.task_done()


class _AsyncSparseDecodeQueue:
    def __init__(
        self,
        *,
        device: torch.device | int | str,
        byte_cap: int,
        load_queue: _AsyncWeightLoadQueue,
        load_delta_weights_func: WeightLoadFunc,
        max_pending_payloads: int,
    ) -> None:
        self._device = device
        self._byte_cap = byte_cap
        self._load_queue = load_queue
        self._load_delta_weights_func = load_delta_weights_func
        self._requests: queue.Queue[
            tuple[TensorBatch, list[dict[str, Any]], list[torch.cuda.Event]] | None
        ] = queue.Queue(maxsize=max_pending_payloads)
        self._delta_batch: TensorBatch = []
        self._delta_batch_bytes = 0
        self._delta_events: list[torch.cuda.Event] = []
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="nemo-sparse-decode",
            daemon=True,
        )
        self._thread.start()

    def enqueue(
        self,
        payload_tensors: TensorBatch,
        metadata: list[dict[str, Any]],
        *,
        ready_events: list[torch.cuda.Event],
    ) -> None:
        if metadata:
            self.raise_if_failed()
            self._requests.put((payload_tensors, metadata, ready_events))
            self.raise_if_failed()

    def flush_pending(self) -> None:
        self.raise_if_failed()
        self._requests.join()
        self.raise_if_failed()
        self._flush_delta_batch()
        self.raise_if_failed()

    def close(self) -> None:
        self._requests.put(None)
        self._thread.join()
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def _flush_delta_batch(self) -> None:
        if not self._delta_batch:
            return
        self._load_queue.enqueue(
            self._load_delta_weights_func,
            self._delta_batch,
            ready_events=self._delta_events,
        )
        self._delta_batch = []
        self._delta_batch_bytes = 0
        self._delta_events = []

    def _add_delta_batch(
        self,
        batch: TensorBatch,
        stream: torch.cuda.Stream | None,
    ) -> None:
        if not batch:
            return
        batch_bytes = sum(tensor.numel() * tensor.element_size() for _, tensor in batch)
        if self._delta_batch and self._delta_batch_bytes + batch_bytes > self._byte_cap:
            self._flush_delta_batch()
        self._delta_batch.extend(batch)
        self._delta_batch_bytes += batch_bytes
        event = record_stream_event(stream)
        if event is not None:
            self._delta_events.append(event)
        if self._delta_batch_bytes >= self._byte_cap:
            self._flush_delta_batch()

    def _run(self) -> None:
        with cuda_device(normalize_device(self._device)):
            streams = cuda_streams(self._device)
            stream_idx = 0
            while True:
                request = self._requests.get()
                try:
                    if request is None:
                        if self._error is None:
                            self._flush_delta_batch()
                        return
                    if self._error is None:
                        payload_tensors, metadata, events = request
                        stream = streams[stream_idx]
                        with use_stream(streams, stream_idx):
                            for event in events:
                                torch.cuda.current_stream().wait_event(event)
                            for _, tensor in payload_tensors:
                                record_tensor_stream(tensor)
                            for batch in sparse_codec.decode_sparse(
                                payload_tensors,
                                metadata,
                                self._device,
                                self._byte_cap,
                            ):
                                self._add_delta_batch(batch, stream)
                        stream_idx = (stream_idx + 1) % len(streams)
                except Exception as error:
                    self._error = error
                finally:
                    self._requests.task_done()
