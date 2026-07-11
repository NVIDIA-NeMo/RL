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
from typing import Any, Literal

import numpy as np
import torch

NamedTensor = tuple[str, torch.Tensor]
TensorBatch = list[NamedTensor]
SparseOperation = Literal["xor", "overwrite"]
SparseInfo = tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, SparseOperation]
TensorPayload = tuple[torch.Tensor, tuple[torch.Tensor, ...], list[dict[str, Any]]]
PreparedTensorPayload = tuple[TensorPayload, int, int]

_INTEGER_DTYPE_BY_SIZE = {
    1: torch.uint8,
    2: torch.int16,
    4: torch.int32,
    8: torch.int64,
}
_DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
}


def integer_dtype_for_element_size(element_size: int) -> torch.dtype:
    try:
        return _INTEGER_DTYPE_BY_SIZE[element_size]
    except KeyError as error:
        raise ValueError(f"Unsupported tensor element size {element_size}.") from error


def dtype_from_name(name: str) -> torch.dtype:
    try:
        return _DTYPE_BY_NAME[name]
    except KeyError as error:
        raise ValueError(f"Unsupported sparse-refit tensor dtype {name!r}.") from error


def sparse_operation(value: object) -> SparseOperation:
    if value == "xor" or value == "overwrite":
        return value
    raise ValueError(f"Unsupported sparse-refit operation {value!r}.")


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if name not in _DTYPE_BY_NAME:
        raise ValueError(f"Unsupported sparse-refit tensor dtype {dtype}.")
    return name


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
    packed_locations = []
    value_parts: list[list[torch.Tensor]] = []
    value_group_by_dtype: dict[torch.dtype, int] = {}
    value_offsets: list[int] = []
    metadata: list[dict[str, Any]] = []
    index_offset = 0
    for name, tensor, raw_locations, raw_values, operation in infos:
        count = int(raw_values.numel())
        if count == 1 or int(raw_locations[-1] - raw_locations[0] + 1) == count:
            index_count = 0
            location_metadata = {
                "index_encoding": "range",
                "range_start": int(raw_locations[0]),
            }
        else:
            location_tensor = _encode_explicit_locations(raw_locations)
            location_metadata = {"index_encoding": "deltas"}
            packed_locations.append(location_tensor)
            index_count = int(location_tensor.numel())
        value_group = value_group_by_dtype.get(raw_values.dtype)
        if value_group is None:
            value_group = len(value_parts)
            value_group_by_dtype[raw_values.dtype] = value_group
            value_parts.append([])
            value_offsets.append(0)
        value_start = value_offsets[value_group]
        value_parts[value_group].append(raw_values)
        value_offsets[value_group] += count
        metadata.append(
            {
                "name": name,
                "shape": tuple(int(dim) for dim in tensor.shape),
                "dtype": _dtype_name(tensor.dtype),
                "operation": operation,
                "index_start": index_offset,
                "index_end": index_offset + index_count,
                "value_group": value_group,
                "value_start": value_start,
                "value_end": value_start + count,
                **location_metadata,
            }
        )
        index_offset += index_count
    indices = (
        torch.cat(packed_locations)
        if packed_locations
        else torch.empty(0, dtype=torch.uint8)
    )
    values = tuple(
        torch.cat(parts) if len(parts) > 1 else parts[0] for parts in value_parts
    )
    return indices, values, metadata


def sparse_locations_for_item(
    item: dict[str, Any],
    packed_locations: torch.Tensor,
    *,
    device: torch.device | int | str,
) -> torch.Tensor:
    count = int(item["value_end"]) - int(item["value_start"])
    if item["index_encoding"] == "range":
        start = int(item["range_start"])
        return torch.arange(start, start + count, device=device)

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
    deltas = np.frombuffer(raw, dtype=delta_dtype).astype(np.int64, copy=False)
    locations = np.cumsum(deltas + 1, dtype=np.int64) - 1
    return torch.from_numpy(locations).to(device=device)


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

    def __init__(self, config: Mapping[str, Any]) -> None:
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
        verification_sources = []
        pending_updates = {}
        changed_elements = total_elements = 0
        for name, tensor in tensors:
            baseline = self.baseline.get(name)
            if baseline is None:
                raise RuntimeError(f"Sparse delta baseline is missing {name!r}.")
            current = tensor.detach().cpu().contiguous()
            current_bits = _integer_view(current).view(-1)
            baseline_bits = _integer_view(baseline).view(-1)
            total_elements += current.numel()
            locations = (
                _bytewise_diff_mask(current, baseline).view(-1).nonzero().view(-1)
            )
            changed_elements += locations.numel()
            if locations.numel():
                current_values = current_bits[locations]
                values = (
                    current_values.bitwise_xor(baseline_bits[locations])
                    if self.encoding == "xor"
                    else current_values
                )
                sparse_infos.append(
                    (
                        name,
                        current,
                        locations,
                        values,
                        self.encoding,
                    )
                )
                if self.verification_samples:
                    verification_sources.append((locations, values))
                pending_updates[name] = (locations, current_values)
        with self._pending_updates_lock:
            self._pending_updates.update(pending_updates)
        payload = encode_sparse_infos(sparse_infos)
        if verification_sources:
            self._add_verification_samples(payload[2], verification_sources)
        return payload, changed_elements, total_elements

    def _add_verification_samples(
        self,
        metadata: list[dict[str, Any]],
        sources: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        total = sum(int(locations.numel()) for locations, _ in sources)
        count = min(self.verification_samples, total)
        sample_ranks = [
            ((2 * index + 1) * total) // (2 * count) for index in range(count)
        ]
        sample_index = offset = 0
        for item, (locations, values) in zip(metadata, sources, strict=True):
            end = offset + locations.numel()
            while sample_index < count and sample_ranks[sample_index] < end:
                local_index = sample_ranks[sample_index] - offset
                location = int(locations[local_index])
                item.setdefault("verification_locations", []).append(location)
                item.setdefault("verification_values", []).append(
                    int(values[local_index])
                )
                sample_index += 1
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
            if count > 1:
                first, last = int(locations[0]), int(locations[-1])
                span = last - first
                if span % (count - 1) == 0:
                    step = span // (count - 1)
                    if step == 1 or all(
                        torch.equal(
                            locations[start:end],
                            first
                            + torch.arange(start, end, dtype=locations.dtype) * step,
                        )
                        for start in range(0, count, 1 << 20)
                        for end in (min(start + (1 << 20), count),)
                    ):
                        target[first : last + 1 : step].copy_(values)
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
