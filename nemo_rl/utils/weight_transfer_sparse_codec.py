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

import os
import tempfile
import threading
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import prod
from typing import Any, Literal

import numpy as np
import torch

NamedTensor = tuple[str, torch.Tensor]
TensorBatch = list[NamedTensor]
SparseOperation = Literal["xor", "overwrite"]
SparseInfo = tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, SparseOperation]
TensorPayload = tuple[torch.Tensor, tuple[torch.Tensor, ...], list[dict[str, Any]]]
PreparedTensorPayload = tuple[TensorPayload, int, int]
DecodedSparseItem = tuple[dict[str, Any], torch.Tensor, torch.Tensor]
DecodedSparsePayload = tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, ...],
    list[dict[str, Any]],
]

_INTEGER_DTYPE_BY_SIZE = {
    1: torch.uint8,
    2: torch.int16,
    4: torch.int32,
    8: torch.int64,
}


class _TensorPayloadBuilder:
    def __init__(self) -> None:
        self.locations: list[torch.Tensor] = []
        self.value_parts: list[list[torch.Tensor]] = []
        self.value_group_by_dtype: dict[torch.dtype, int] = {}
        self.value_offsets: list[int] = []
        self.metadata: list[dict[str, Any]] = []
        self.index_offset = 0

    def add_locations(self, locations: torch.Tensor) -> tuple[int, int]:
        start = self.index_offset
        if locations.numel():
            self.locations.append(locations)
            self.index_offset += locations.numel()
        return start, self.index_offset

    def add_values(self, values: torch.Tensor) -> tuple[int, int, int]:
        group = self.value_group_by_dtype.get(values.dtype)
        if group is None:
            group = len(self.value_parts)
            self.value_group_by_dtype[values.dtype] = group
            self.value_parts.append([])
            self.value_offsets.append(0)
        start = self.value_offsets[group]
        self.value_parts[group].append(values)
        self.value_offsets[group] += values.numel()
        return group, start, self.value_offsets[group]

    def finish(self) -> TensorPayload:
        indices = (
            torch.cat(self.locations)
            if self.locations
            else torch.empty(0, dtype=torch.uint8)
        )
        values = tuple(
            torch.cat(parts) if len(parts) > 1 else parts[0]
            for parts in self.value_parts
        )
        return indices, values, self.metadata


@dataclass(frozen=True)
class SparseShardProjection:
    """Map one local tensor shard into a canonical HF tensor."""

    name: str
    global_shape: tuple[int, ...]
    offsets: tuple[int, ...]

    def map_locations(
        self, locations: torch.Tensor, local_shape: tuple[int, ...]
    ) -> torch.Tensor:
        if len(local_shape) != len(self.global_shape) or len(local_shape) != len(
            self.offsets
        ):
            raise ValueError(f"Sparse shard {self.name!r} has inconsistent ranks.")
        if any(
            offset < 0 or offset + local > global_size
            for local, global_size, offset in zip(
                local_shape, self.global_shape, self.offsets, strict=True
            )
        ):
            raise ValueError(f"Sparse shard {self.name!r} exceeds its global shape.")
        if local_shape == self.global_shape and not any(self.offsets):
            return locations

        mapped = torch.zeros_like(locations)
        for dim, (local_size, global_size, offset) in enumerate(
            zip(local_shape, self.global_shape, self.offsets, strict=True)
        ):
            local_stride = prod(local_shape[dim + 1 :])
            global_stride = prod(self.global_shape[dim + 1 :])
            coordinate = torch.div(
                locations, local_stride, rounding_mode="floor"
            ).remainder(local_size)
            mapped.add_((coordinate + offset) * global_stride)
        return mapped


def integer_dtype_for_element_size(element_size: int) -> torch.dtype:
    try:
        return _INTEGER_DTYPE_BY_SIZE[element_size]
    except KeyError as error:
        raise ValueError(f"Unsupported tensor element size {element_size}.") from error


def dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported sparse-refit tensor dtype {name!r}.")
    integer_dtype_for_element_size(dtype.itemsize)
    return dtype


def sparse_operation(value: object) -> SparseOperation:
    if value == "xor" or value == "overwrite":
        return value
    raise ValueError(f"Unsupported sparse-refit operation {value!r}.")


def _integer_view(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(
        integer_dtype_for_element_size(tensor.element_size())
    )


def _bytewise_diff_mask(current: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    if current.shape != baseline.shape or current.dtype != baseline.dtype:
        raise ValueError(
            "Current tensor and baseline must have identical shape and dtype."
        )
    return _integer_view(current) != _integer_view(baseline)


def encode_sparse_infos(
    infos: Iterable[SparseInfo],
) -> TensorPayload:
    payload = _TensorPayloadBuilder()
    for name, tensor, raw_locations, raw_values, operation in infos:
        count = int(raw_values.numel())
        if count == 1 or int(raw_locations[-1] - raw_locations[0] + 1) == count:
            index_start = index_end = payload.index_offset
            location_metadata = {
                "index_encoding": "range",
                "range_start": int(raw_locations[0]),
            }
        else:
            location_tensor = _encode_explicit_locations(raw_locations)
            location_metadata = {"index_encoding": "deltas"}
            index_start, index_end = payload.add_locations(location_tensor)
        value_group, value_start, value_end = payload.add_values(raw_values)
        payload.metadata.append(
            {
                "name": name,
                "shape": tuple(int(dim) for dim in tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "operation": operation,
                "index_start": index_start,
                "index_end": index_end,
                "value_group": value_group,
                "value_start": value_start,
                "value_end": value_end,
                **location_metadata,
            }
        )
    return payload.finish()


def merge_sparse_payloads(payloads: Iterable[TensorPayload]) -> TensorPayload:
    """Combine encoded chunks without materializing dense source tensors."""
    payload = _TensorPayloadBuilder()
    for locations, value_groups, items in payloads:
        group_remap = {}
        group_starts = {}
        for old_group, values in enumerate(value_groups):
            new_group, start, _ = payload.add_values(values)
            group_remap[old_group] = new_group
            group_starts[old_group] = start
        index_offset, _ = payload.add_locations(locations)
        for item in items:
            merged = dict(item)
            merged["index_start"] = int(item["index_start"]) + index_offset
            merged["index_end"] = int(item["index_end"]) + index_offset
            old_group = int(item["value_group"])
            merged["value_group"] = group_remap[old_group]
            merged["value_start"] = int(item["value_start"]) + group_starts[old_group]
            merged["value_end"] = int(item["value_end"]) + group_starts[old_group]
            payload.metadata.append(merged)
    return payload.finish()


def sparse_locations_for_item(
    item: dict[str, Any],
    packed_locations: torch.Tensor,
    *,
    device: torch.device | int | str,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    if dtype not in (torch.int32, torch.int64):
        raise ValueError(f"Unsupported sparse location dtype {dtype}.")
    count = int(item["value_end"]) - int(item["value_start"])
    if item["index_encoding"] == "range":
        start = int(item["range_start"])
        return torch.arange(start, start + count, dtype=dtype, device=device)

    index_start, index_end = int(item["index_start"]), int(item["index_end"])
    raw = (
        packed_locations[index_start:index_end]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8, copy=False)
        .tobytes()
    )
    delta_dtype = {2: np.uint16, 4: np.uint32, 8: np.uint64}[len(raw) // count]
    location_dtype = np.int32 if dtype == torch.int32 else np.int64
    deltas = np.frombuffer(raw, dtype=delta_dtype).astype(location_dtype, copy=False)
    locations = np.cumsum(deltas + 1, dtype=location_dtype) - 1
    return torch.from_numpy(locations).to(device=device)


def _merge_tensor_parts(parts: list[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    if not parts:
        return torch.empty(0, dtype=dtype)
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def decode_sparse_tensor_payload_for_staging(
    payload: TensorPayload,
) -> DecodedSparsePayload:
    """Flatten decoded locations so workers mmap only a few tensor storages."""
    packed_locations, value_groups, source_metadata = payload
    location_parts: tuple[list[torch.Tensor], list[torch.Tensor]] = ([], [])
    location_offsets = [0, 0]
    metadata = []
    for item in source_metadata:
        location_dtype = (
            torch.int32
            if prod(item["shape"]) <= torch.iinfo(torch.int32).max
            else torch.int64
        )
        locations = sparse_locations_for_item(
            item, packed_locations, device="cpu", dtype=location_dtype
        )
        group = 0 if locations.dtype == torch.int32 else 1
        staged_item = dict(item)
        staged_item["decoded_location_group"] = group
        staged_item["decoded_location_start"] = location_offsets[group]
        location_offsets[group] += locations.numel()
        staged_item["decoded_location_end"] = location_offsets[group]
        location_parts[group].append(locations)
        metadata.append(staged_item)
    locations = (
        _merge_tensor_parts(location_parts[0], torch.int32),
        _merge_tensor_parts(location_parts[1], torch.int64),
    )
    return locations, value_groups, metadata


def iter_decoded_sparse_payload(
    payload: DecodedSparsePayload,
) -> Iterable[DecodedSparseItem]:
    location_groups, value_groups, metadata = payload
    for item in metadata:
        location_start = int(item["decoded_location_start"])
        location_end = int(item["decoded_location_end"])
        value_start = int(item["value_start"])
        value_end = int(item["value_end"])
        yield (
            item,
            location_groups[int(item["decoded_location_group"])][
                location_start:location_end
            ],
            value_groups[int(item["value_group"])][value_start:value_end],
        )


def _encode_explicit_locations(
    locations: torch.Tensor,
) -> torch.Tensor:
    indices = locations.detach().cpu().numpy().astype(np.int64, copy=False)
    deltas = np.diff(indices, prepend=-1) - 1
    max_delta = int(deltas.max())
    dtype = (
        np.uint16
        if max_delta <= 0xFFFF
        else np.uint32
        if max_delta <= 0xFFFFFFFF
        else np.uint64
    )
    raw = deltas.astype(dtype, copy=False).tobytes()
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())


class DeltaCompressionTracker:
    """Source-side CPU or mmap baseline for sparse-delta refit."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        projections: Mapping[str, SparseShardProjection] | None = None,
    ) -> None:
        self.sparse_bucket_size_bytes = int(config["sparse_bucket_size_bytes"])
        if self.sparse_bucket_size_bytes < 1:
            raise ValueError("delta_compression.sparse_bucket_size_bytes must be >= 1")
        self.encoding = sparse_operation(config["encoding"])
        self.verification_samples = int(
            os.getenv("NRL_REFIT_VERIFY_SAMPLES_PER_PAYLOAD", "0")
        )
        if self.verification_samples < 0:
            raise ValueError("NRL_REFIT_VERIFY_SAMPLES_PER_PAYLOAD must be >= 0")
        self.baseline_in_memory = os.getenv("NRL_REFIT_BASELINE_IN_MEMORY") == "1"
        self.baseline_mmap_dir = os.getenv("NRL_REFIT_BASELINE_MMAP_DIR")
        self.projections = dict(projections or {})
        self.baseline: dict[str, torch.Tensor] = {}
        self._pending_updates: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._pending_updates_lock = threading.Lock()
        self._baseline_commits: tuple[Any, ...] = ()
        self._baseline_commit_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="nrl-refit-baseline"
        )

    def prepare_sparse_delta_payload(
        self, tensors: TensorBatch
    ) -> PreparedTensorPayload:
        self._wait_for_baseline_commits()
        sparse_infos = []
        pending_updates = {}
        changed_elements = total_elements = 0
        for name, tensor in tensors:
            baseline, current, locations, current_values = self._find_changes(
                name, tensor
            )
            baseline_bits = _integer_view(baseline).view(-1)
            total_elements += current.numel()
            changed_elements += locations.numel()
            if locations.numel():
                values = (
                    current_values.bitwise_xor(baseline_bits[locations])
                    if self.encoding == "xor"
                    else current_values
                )
                payload_name, payload_tensor, payload_locations = (
                    name,
                    current,
                    locations,
                )
                if projection := self.projections.get(name):
                    payload_name = projection.name
                    payload_locations = projection.map_locations(
                        locations, tuple(current.shape)
                    )
                    payload_tensor = torch.empty(
                        projection.global_shape, dtype=current.dtype, device="meta"
                    )
                sparse_infos.append(
                    (
                        payload_name,
                        payload_tensor,
                        payload_locations,
                        values,
                        self.encoding,
                    )
                )
                pending_updates[name] = (locations, current_values)
        with self._pending_updates_lock:
            self._pending_updates.update(pending_updates)
        payload = encode_sparse_infos(sparse_infos)
        if self.verification_samples:
            self._add_verification_samples(payload[2])
        return payload, changed_elements, total_elements

    def prepare_change_summary(
        self, tensors: Iterable[NamedTensor]
    ) -> tuple[set[str], int, int]:
        """Scan local tensors without constructing a wire payload."""
        self._wait_for_baseline_commits()
        changed_names = set()
        pending_updates = {}
        changed_elements = total_elements = 0
        for name, tensor in tensors:
            _, current, locations, current_values = self._find_changes(name, tensor)
            total_elements += current.numel()
            changed_elements += locations.numel()
            if locations.numel():
                changed_names.add(name)
                pending_updates[name] = (locations, current_values)
        with self._pending_updates_lock:
            self._pending_updates.update(pending_updates)
        return changed_names, changed_elements, total_elements

    def _find_changes(
        self, name: str, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        baseline = self.baseline.get(name)
        if baseline is None:
            raise RuntimeError(f"Sparse delta baseline is missing {name!r}.")
        current = tensor.detach().cpu().contiguous()
        locations = _bytewise_diff_mask(current, baseline).view(-1).nonzero().view(-1)
        current_values = _integer_view(current).view(-1)[locations]
        return baseline, current, locations, current_values

    def _add_verification_samples(
        self,
        metadata: list[dict[str, Any]],
    ) -> None:
        sizes = [int(item["value_end"]) - int(item["value_start"]) for item in metadata]
        total = sum(sizes)
        count = min(self.verification_samples, total)
        sample_ranks = [
            ((2 * index + 1) * total) // (2 * count) for index in range(count)
        ]
        sample_index = offset = 0
        for item, size in zip(metadata, sizes, strict=True):
            end = offset + size
            samples = 0
            while sample_index < count and sample_ranks[sample_index] < end:
                samples += 1
                sample_index += 1
            if samples:
                item["verification_samples"] = samples
            offset = end

    def on_sync_succeeded(self) -> None:
        with self._pending_updates_lock:
            pending_updates, self._pending_updates = self._pending_updates, {}
        items = list(pending_updates.items())
        workers = min(4, len(items))
        self._baseline_commits = tuple(
            self._baseline_commit_executor.submit(
                self._commit_baseline_updates, items[worker::workers]
            )
            for worker in range(workers)
        )

    def on_sync_failed(self) -> None:
        with self._pending_updates_lock:
            self._pending_updates.clear()

    def snapshot_baseline(self, tensors: Iterable[NamedTensor]) -> None:
        self._wait_for_baseline_commits()
        for name, tensor in tensors:
            baseline = self._baseline(name, tuple(tensor.shape), tensor.dtype)
            baseline.view(torch.uint8).view(-1).copy_(
                tensor.detach().cpu().contiguous().view(torch.uint8).view(-1)
            )

    def _wait_for_baseline_commits(self) -> None:
        for commit in self._baseline_commits:
            commit.result()
        self._baseline_commits = ()

    def _commit_baseline_updates(
        self,
        updates: Iterable[tuple[str, tuple[torch.Tensor, torch.Tensor]]],
    ) -> None:
        for name, (locations, values) in updates:
            target = _integer_view(self.baseline[name]).view(-1)
            count = locations.numel()
            first = int(locations[0])
            if int(locations[-1]) - first + 1 == count:
                target[first : first + count].copy_(values)
                continue
            target.index_copy_(0, locations, values)

    def _baseline(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if name in self.baseline:
            return self.baseline[name]
        numel = torch.Size(shape).numel()
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        if self.baseline_in_memory:
            storage = torch.empty(nbytes, dtype=torch.uint8)
        else:
            with tempfile.NamedTemporaryFile(
                prefix="nrl-refit-baseline-", dir=self.baseline_mmap_dir
            ) as handle:
                handle.truncate(nbytes)
                storage = torch.from_file(
                    handle.name, shared=True, size=nbytes, dtype=torch.uint8
                )
        baseline = storage.view(dtype).view(shape)
        self.baseline[name] = baseline
        return baseline
