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

import contextlib
import itertools
import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Literal, cast

import torch

from nemo_rl.utils.packed_tensor import get_num_buffers

G_PAYLOAD_ALIGNMENT_BYTES = 8
G_PACKED_INDICES_NAME = "__packed_indices__"
G_PACKED_VALUES_NAME = "__packed_values__"
G_INDEX_START_KEY = "index_start"
G_INDEX_END_KEY = "index_end"
G_DEFAULT_SPARSE_ENCODE_COALESCE_BYTES = 256 * 1024**2
G_DEFAULT_BASELINE_PREWARM_CHUNK_BYTES = 256 * 1024**2
G_DEFAULT_BASELINE_PREWARM_MAX_BYTES = 128 * 1024**3
G_DEFAULT_BASELINE_STAGE_COALESCE_BYTES = 256 * 1024**2
G_DEFAULT_BASELINE_MMAP_MIN_BYTES = 128 * 1024**3
G_DEFAULT_BASELINE_MMAP_WRITE_WORKERS = 4
G_BASELINE_STAGE_FREE_MEMORY_FRACTION = 0.125

DeltaCompressionTransport = Literal["dense", "sparse_indices"]
WeightTransferKind = Literal["full", "delta", "done"]

G_DENSE_TRANSPORT: DeltaCompressionTransport = "dense"
G_SPARSE_INDICES_TRANSPORT: DeltaCompressionTransport = "sparse_indices"
G_FULL_UPDATE_KIND: WeightTransferKind = "full"
G_DELTA_UPDATE_KIND: WeightTransferKind = "delta"
G_TRANSFER_DONE_KIND: WeightTransferKind = "done"

G_FLOAT_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

G_TENSOR_DTYPE_MAP = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float64": torch.float64,
}

for _float8_dtype_name in (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
):
    _float8_dtype = getattr(torch, _float8_dtype_name, None)
    if _float8_dtype is not None:
        G_TENSOR_DTYPE_MAP[_float8_dtype_name] = _float8_dtype
del _float8_dtype, _float8_dtype_name

NamedTensor = tuple[str, torch.Tensor]
TensorBatch = list[NamedTensor]
WeightLoadFunc = Callable[[TensorBatch], None]
TensorPayload = tuple[TensorBatch, DeltaCompressionTransport, list[dict[str, Any]]]
PayloadEvents = tuple[torch.cuda.Event, ...]
QueuedPayload = tuple[WeightTransferKind, TensorPayload, PayloadEvents]
SparseBucketPayload = tuple[TensorPayload, PayloadEvents]
HeaderRefs = tuple[torch.Tensor, torch.Tensor | None]
TensorMetadata = Mapping[str, tuple[Iterable[int], torch.dtype]]

G_REFIT_SPARSE_ENCODE_COALESCE_BYTES_ENV = "NRL_REFIT_SPARSE_ENCODE_COALESCE_BYTES"
G_REFIT_PREWARM_DELTA_BASELINE_ENV = "NRL_REFIT_PREWARM_DELTA_BASELINE"
G_REFIT_BASELINE_PREWARM_CHUNK_BYTES_ENV = "NRL_REFIT_BASELINE_PREWARM_CHUNK_BYTES"
G_REFIT_BASELINE_PREWARM_MAX_BYTES_ENV = "NRL_REFIT_BASELINE_PREWARM_MAX_BYTES"
G_REFIT_BASELINE_STAGE_COALESCE_BYTES_ENV = "NRL_REFIT_BASELINE_STAGE_COALESCE_BYTES"
G_REFIT_BASELINE_MMAP_MIN_BYTES_ENV = "NRL_REFIT_BASELINE_MMAP_MIN_BYTES"
G_REFIT_BASELINE_MMAP_DIR_ENV = "NRL_REFIT_BASELINE_MMAP_DIR"
G_REFIT_BASELINE_MMAP_PENDING_BYTES_ENV = "NRL_REFIT_BASELINE_MMAP_PENDING_BYTES"
G_REFIT_BASELINE_MMAP_WRITE_WORKERS_ENV = "NRL_REFIT_BASELINE_MMAP_WRITE_WORKERS"

G_HEADER_KIND_TO_CODE = {
    G_TRANSFER_DONE_KIND: 0,
    G_FULL_UPDATE_KIND: 1,
    G_DELTA_UPDATE_KIND: 2,
}
G_HEADER_CODE_TO_KIND = {code: kind for kind, code in G_HEADER_KIND_TO_CODE.items()}
G_HEADER_TRANSPORT_TO_CODE = {
    G_DENSE_TRANSPORT: 0,
    G_SPARSE_INDICES_TRANSPORT: 1,
}
G_HEADER_CODE_TO_TRANSPORT = {
    code: transport for transport, code in G_HEADER_TRANSPORT_TO_CODE.items()
}


def env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Expected integer value for {name}.") from None


def metadata_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def dtype_itemsize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def memory_limited_stage_bytes(
    device: torch.device | int | str | None,
    requested_bytes: int,
) -> int:
    if requested_bytes <= 0 or not torch.cuda.is_available():
        return requested_bytes

    normalized_device = normalize_device(device)
    if normalized_device is not None and normalized_device.type != "cuda":
        return requested_bytes

    free_bytes, _ = torch.cuda.mem_get_info(normalized_device)
    free_memory_cap = int(free_bytes * G_BASELINE_STAGE_FREE_MEMORY_FRACTION)
    return min(requested_bytes, free_memory_cap)


def pack_named_tensors(tensors: TensorBatch) -> tuple[torch.Tensor, list[dict]]:
    """Pack tensors with mixed dtypes into one uint8 tensor."""
    chunks = []
    entries = []
    for name, tensor in tensors:
        tensor = tensor.contiguous()
        byte_view = tensor.view(torch.uint8).view(-1)
        byte_size = int(byte_view.numel())
        pad = (-byte_size) % G_PAYLOAD_ALIGNMENT_BYTES
        chunks.append(byte_view)
        if pad:
            chunks.append(torch.zeros(pad, dtype=torch.uint8, device=tensor.device))
        entries.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": dtype_to_name(tensor.dtype),
                "byte_size": byte_size,
                "wire_byte_size": byte_size + pad,
            }
        )
    return torch.cat(chunks, dim=0), entries


def unpack_named_tensors(
    payload: torch.Tensor, entries: list[dict[str, Any]]
) -> TensorBatch:
    """Unpack a uint8 transfer payload according to header metadata."""
    byte_views = payload.split_with_sizes(
        [int(entry["wire_byte_size"]) for entry in entries]
    )
    return [
        (
            entry["name"],
            byte_view[: int(entry["byte_size"])]
            .view(dtype_from_name(entry["dtype"]))
            .view(tuple(entry["shape"])),
        )
        for entry, byte_view in zip(entries, byte_views, strict=True)
    ]


@contextlib.contextmanager
def additive_weight_load_context(target_tensors: Iterable[torch.Tensor]):
    """Make weight loaders add into model tensors instead of overwriting them."""
    original_copy = torch.Tensor.copy_
    original_fill = torch.Tensor.fill_
    original_setitem = torch.Tensor.__setitem__
    target_storage_ptrs = {
        tensor.untyped_storage().data_ptr() for tensor in target_tensors
    }

    def should_add(tensor: torch.Tensor) -> bool:
        return (
            tensor.dtype.is_floating_point
            and tensor.untyped_storage().data_ptr() in target_storage_ptrs
        )

    def additive_copy(self, src, non_blocking=False):
        if should_add(self):
            self.add_(src.to(self.device, self.dtype, non_blocking=non_blocking))
            return self
        return original_copy(self, src, non_blocking=non_blocking)

    def additive_fill(self, value, *args, **kwargs):
        if should_add(self):
            self.add_(value)
            return self
        return original_fill(self, value, *args, **kwargs)

    def additive_setitem(self, index, value):
        destination = self[index]
        if should_add(destination):
            if isinstance(value, torch.Tensor):
                value = value.to(device=destination.device, dtype=destination.dtype)
            destination.add_(value)
            return
        return original_setitem(self, index, value)

    torch.Tensor.copy_ = cast(Any, additive_copy)
    torch.Tensor.fill_ = cast(Any, additive_fill)
    torch.Tensor.__setitem__ = cast(Any, additive_setitem)
    try:
        yield
    finally:
        torch.Tensor.copy_ = cast(Any, original_copy)
        torch.Tensor.fill_ = cast(Any, original_fill)
        torch.Tensor.__setitem__ = cast(Any, original_setitem)


def wire_bytes(tensors: TensorBatch) -> int:
    return sum(
        (byte_size := int(tensor.numel() * tensor.element_size()))
        + (-byte_size) % G_PAYLOAD_ALIGNMENT_BYTES
        for _, tensor in tensors
    )


def next_chunk(
    iterator: Iterator[NamedTensor],
    byte_cap: int,
    *,
    pending_item: NamedTensor | None = None,
) -> tuple[TensorBatch, NamedTensor | None]:
    chunk: TensorBatch = []
    chunk_bytes = 0
    items: Iterable[NamedTensor] = iterator
    if pending_item is not None:
        items = itertools.chain((pending_item,), iterator)
    for item in items:
        tensor_bytes = item[1].numel() * item[1].element_size()
        if chunk and chunk_bytes + tensor_bytes > byte_cap:
            return chunk, item
        chunk.append(item)
        chunk_bytes += tensor_bytes
    return chunk, None


def advance_chunk(
    iterator: Iterator[NamedTensor],
    byte_cap: int,
    *,
    pending_item: NamedTensor | None = None,
) -> tuple[NamedTensor | None, bool]:
    chunk_bytes = 0
    consumed_item = False
    items: Iterable[NamedTensor] = iterator
    if pending_item is not None:
        items = itertools.chain((pending_item,), iterator)
    for item in items:
        tensor_bytes = item[1].numel() * item[1].element_size()
        if consumed_item and chunk_bytes + tensor_bytes > byte_cap:
            return item, False
        consumed_item = True
        chunk_bytes += tensor_bytes
    return None, True


def broadcast_header(
    header: Mapping[str, Any],
    *,
    group: Any,
    src: int,
    device: torch.device | int | str,
) -> tuple[dict[str, Any], HeaderRefs]:
    encoded = _encode_header_metadata(header)
    if group.rank == src:
        control_tensor = torch.tensor(
            _header_control_values(header, len(encoded)),
            dtype=torch.int64,
            device=device,
        )
    else:
        control_tensor = torch.empty(4, dtype=torch.int64, device=device)
    group.broadcast(control_tensor, src=src)
    kind, transport, payload_numel, metadata_len = decode_header_control(control_tensor)

    metadata_tensor = None
    metadata: dict[str, Any] = {}
    if metadata_len > 0:
        if group.rank == src:
            metadata_tensor = torch.tensor(
                list(encoded), dtype=torch.uint8, device=device
            )
        else:
            metadata_tensor = torch.empty(
                metadata_len, dtype=torch.uint8, device=device
            )
        group.broadcast(metadata_tensor, src=src)
        if group.rank != src:
            metadata = json.loads(
                metadata_tensor.cpu().numpy().tobytes().decode("utf-8")
            )

    if group.rank == src:
        received_header = dict(header)
    else:
        received_header = {
            "kind": kind,
            "transport": transport,
            "payload_entries": [],
            "payload_numel": payload_numel,
            "sparse_metadata": [],
        }
        received_header.update(metadata)
    return received_header, (control_tensor, metadata_tensor)


def _header_control_values(header: Mapping[str, Any], metadata_len: int) -> list[int]:
    kind = header["kind"]
    if kind not in G_HEADER_KIND_TO_CODE:
        raise ValueError(f"Unsupported weight transfer header kind: {kind}")
    transport = header.get("transport", G_DENSE_TRANSPORT)
    if transport not in G_HEADER_TRANSPORT_TO_CODE:
        raise ValueError(f"Unsupported weight transfer header transport: {transport}")
    return [
        G_HEADER_KIND_TO_CODE[kind],
        G_HEADER_TRANSPORT_TO_CODE[transport],
        int(header.get("payload_numel", 0)),
        metadata_len,
    ]


def decode_header_control(
    control_tensor: torch.Tensor,
) -> tuple[WeightTransferKind, DeltaCompressionTransport, int, int]:
    kind_code, transport_code, payload_numel, metadata_len = [
        int(value) for value in control_tensor.cpu().tolist()
    ]
    try:
        kind = G_HEADER_CODE_TO_KIND[kind_code]
        transport = G_HEADER_CODE_TO_TRANSPORT[transport_code]
    except KeyError:
        raise ValueError(
            f"Unsupported weight transfer header control values: "
            f"kind={kind_code}, transport={transport_code}"
        ) from None
    return (
        cast(WeightTransferKind, kind),
        cast(DeltaCompressionTransport, transport),
        payload_numel,
        metadata_len,
    )


def _encode_header_metadata(header: Mapping[str, Any]) -> bytes:
    metadata = {
        key: header[key]
        for key in ("payload_entries", "sparse_metadata", "is_delta_sync")
        if header.get(key)
    }
    return json.dumps(metadata).encode("utf-8") if metadata else b""


def recv_payload(
    payload_numel: int,
    *,
    group: Any,
    src: int,
    device: torch.device | int | str,
) -> torch.Tensor:
    payload = torch.empty(payload_numel, dtype=torch.uint8, device=device)
    if payload.numel() > 0:
        group.broadcast(payload, src=src)
    return payload


def dtype_to_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def dtype_from_name(name: str) -> torch.dtype:
    try:
        return G_TENSOR_DTYPE_MAP[name]
    except KeyError:
        raise ValueError(
            f"Unsupported tensor dtype in weight transfer: {name}"
        ) from None


def cuda_streams(
    device: torch.device | int | str | None = None,
) -> list[torch.cuda.Stream | None]:
    if not torch.cuda.is_available():
        return [None]
    normalized_device = normalize_device(device)
    if normalized_device is not None and normalized_device.type != "cuda":
        return [None]
    with cuda_device(normalized_device):
        return [torch.cuda.Stream() for _ in range(get_num_buffers())]


@contextlib.contextmanager
def use_stream(
    streams: list[torch.cuda.Stream | None],
    index: int,
):
    stream = streams[index]
    if stream is None:
        yield
        return
    with torch.cuda.stream(stream):
        yield


def record_header_stream(refs: HeaderRefs) -> None:
    control_tensor, metadata_tensor = refs
    record_tensor_stream(control_tensor)
    if metadata_tensor is not None:
        record_tensor_stream(metadata_tensor)


def record_tensor_stream(tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        return
    tensor.record_stream(torch.cuda.current_stream())


def sync_streams(streams: list[torch.cuda.Stream | None]) -> None:
    for stream in streams:
        if stream is not None:
            stream.synchronize()


def synchronize_current_transfer_stream(device: torch.device | int | str) -> None:
    if not torch.cuda.is_available():
        return
    normalized_device = normalize_device(device)
    if normalized_device is None or normalized_device.type == "cuda":
        torch.cuda.current_stream(normalized_device).synchronize()


def record_stream_event(stream: torch.cuda.Stream | None) -> torch.cuda.Event | None:
    if stream is None:
        return None
    return stream.record_event()


def record_payload_readiness_events() -> PayloadEvents:
    if not torch.cuda.is_available():
        return ()
    return (torch.cuda.current_stream().record_event(),)


def wait_for_payload_events(events: Iterable[torch.cuda.Event]) -> None:
    seen_events = set()
    current_stream = None
    for event in events:
        if id(event) in seen_events:
            continue
        if current_stream is None:
            current_stream = torch.cuda.current_stream()
        current_stream.wait_event(event)
        seen_events.add(id(event))


@contextlib.contextmanager
def cuda_device(device: torch.device | None):
    if device is None or device.type != "cuda":
        yield
        return
    with torch.cuda.device(device):
        yield


def normalize_device(device: torch.device | int | str | None) -> torch.device | None:
    if device is None:
        return None
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device)
