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

import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from nemo_rl.utils.weight_transfer_protocol import (
    G_DEFAULT_SPARSE_ENCODE_COALESCE_BYTES,
    G_INDEX_END_KEY,
    G_INDEX_START_KEY,
    G_PACKED_INDICES_NAME,
    G_PACKED_VALUES_NAME,
    G_REFIT_SPARSE_ENCODE_COALESCE_BYTES_ENV,
    G_SPARSE_INDICES_TRANSPORT,
    TensorBatch,
    TensorPayload,
    dtype_from_name,
    dtype_to_name,
    env_int,
)


@dataclass(frozen=True)
class SparseTensorInfo:
    name: str
    tensor: torch.Tensor
    flat: torch.Tensor

    @property
    def numel(self) -> int:
        return int(self.flat.numel())

    @property
    def byte_size(self) -> int:
        return int(self.flat.numel() * self.flat.element_size())


class SparseEncodingUnavailable(Exception):
    """Signal sparse encoding cannot represent the current tensors."""


class _SparsePayloadBuilder:
    """Builds sparse-index payload tensors and metadata."""

    def __init__(
        self,
        *,
        value_dtype: torch.dtype,
    ) -> None:
        self._value_dtype = value_dtype
        self._packed_parts: list[torch.Tensor] = []
        self._value_parts: list[torch.Tensor] = []
        self._metadata: list[dict[str, Any]] = []
        self._packed_offset = 0
        self._value_offset = 0

    def append_group(self, group: list[SparseTensorInfo]) -> None:
        if len(group) == 1:
            self._append_single_tensor(group[0])
            return

        try:
            locations, values, counts = sparse_indices_for_group(group)
        except torch.OutOfMemoryError:
            for info in group:
                self.append_group([info])
            return

        if values.numel() == 0:
            return

        value_start = 0
        packed_locations = locations.to(torch.int32)
        for info, count in zip(group, counts, strict=True):
            if count == 0:
                continue
            value_end = value_start + count
            self._append_tensor(
                info=info,
                packed=packed_locations[value_start:value_end],
                values=values[value_start:value_end],
            )
            value_start = value_end

    def build(self, device: torch.device) -> TensorPayload:
        return (
            [
                (
                    G_PACKED_INDICES_NAME,
                    (
                        torch.cat(self._packed_parts, dim=0)
                        if self._packed_parts
                        else torch.empty(0, dtype=torch.int32, device=device)
                    ),
                ),
                (
                    G_PACKED_VALUES_NAME,
                    (
                        torch.cat(self._value_parts, dim=0)
                        if self._value_parts
                        else torch.empty(0, dtype=self._value_dtype, device=device)
                    ),
                ),
            ],
            G_SPARSE_INDICES_TRANSPORT,
            self._metadata,
        )

    def _append_single_tensor(self, info: SparseTensorInfo) -> None:
        try:
            locations, values = _sparse_indices_for_tensor(info.flat)
        except torch.OutOfMemoryError:
            raise SparseEncodingUnavailable from None
        if values.numel() == 0:
            return
        self._append_tensor(
            info=info,
            packed=locations.to(torch.int32),
            values=values,
        )

    def _append_tensor(
        self,
        *,
        info: SparseTensorInfo,
        packed: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        nnz = int(values.numel())
        self._metadata.append(
            {
                "name": info.name,
                "dtype": dtype_to_name(info.tensor.dtype),
                "shape": list(info.tensor.shape),
                "numel": info.numel,
                G_INDEX_START_KEY: self._packed_offset,
                G_INDEX_END_KEY: self._packed_offset + int(packed.numel()),
                "value_start": self._value_offset,
                "value_end": self._value_offset + nnz,
            }
        )
        self._packed_parts.append(packed)
        self._value_parts.append(values)
        self._packed_offset += int(packed.numel())
        self._value_offset += nnz


def encode_sparse_indices(tensors: TensorBatch) -> TensorPayload:
    builder = _SparsePayloadBuilder(value_dtype=tensors[0][1].dtype)

    for group in _iter_sparse_index_groups(tensors):
        builder.append_group(group)

    return builder.build(tensors[0][1].device)


def _sparse_encode_coalesce_bytes() -> int:
    return env_int(
        G_REFIT_SPARSE_ENCODE_COALESCE_BYTES_ENV,
        default=G_DEFAULT_SPARSE_ENCODE_COALESCE_BYTES,
    )


def _iter_sparse_index_groups(
    tensors: TensorBatch,
) -> Iterator[list[SparseTensorInfo]]:
    coalesce_bytes = _sparse_encode_coalesce_bytes()
    current: list[SparseTensorInfo] = []
    current_bytes = 0

    for name, tensor in tensors:
        flat = tensor.contiguous().view(-1)
        if flat.numel() > torch.iinfo(torch.int32).max:
            raise SparseEncodingUnavailable

        info = SparseTensorInfo(name=name, tensor=tensor, flat=flat)
        can_coalesce = (
            coalesce_bytes > 0
            and current
            and current[0].flat.dtype == flat.dtype
            and current[0].flat.device == flat.device
            and current_bytes + info.byte_size <= coalesce_bytes
        )
        if current and not can_coalesce:
            yield current
            current = []
            current_bytes = 0

        current.append(info)
        current_bytes += info.byte_size

    if current:
        yield current


def _sparse_indices_for_tensor(
    flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    locations = torch.nonzero(flat, as_tuple=True)[0]
    values = flat[locations]
    return locations, values


def sparse_indices_for_group(
    group: list[SparseTensorInfo],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    flat = alias_sparse_index_group(group)
    if flat is None:
        flat = torch.cat([info.flat for info in group], dim=0)
    global_locations, values = _sparse_indices_for_tensor(flat)
    if global_locations.numel() == 0:
        return global_locations, values, [0 for _ in group]

    offsets = list(itertools.accumulate(info.numel for info in group))
    boundaries = torch.tensor(
        offsets[:-1],
        dtype=global_locations.dtype,
        device=global_locations.device,
    )
    segment_ids = torch.bucketize(global_locations, boundaries, right=True)
    starts = torch.tensor(
        [0, *offsets[:-1]],
        dtype=global_locations.dtype,
        device=global_locations.device,
    )
    locations = global_locations - starts[segment_ids]
    counts = torch.bincount(segment_ids, minlength=len(group)).cpu().tolist()
    return locations, values, [int(count) for count in counts]


def alias_sparse_index_group(group: list[SparseTensorInfo]) -> torch.Tensor | None:
    if len(group) < 2:
        return None

    first = group[0].flat
    if not first.is_contiguous():
        return None

    storage_ptr = first.untyped_storage().data_ptr()
    expected_offset = first.storage_offset()
    total_numel = 0
    for info in group:
        flat = info.flat
        if (
            not flat.is_contiguous()
            or flat.dtype != first.dtype
            or flat.device != first.device
            or flat.untyped_storage().data_ptr() != storage_ptr
            or flat.storage_offset() != expected_offset
        ):
            return None
        expected_offset += flat.numel()
        total_numel += flat.numel()

    return torch.as_strided(
        first,
        (total_numel,),
        (1,),
        storage_offset=first.storage_offset(),
    )


def decode_sparse(
    payload_tensors: TensorBatch,
    metadata: list[dict[str, Any]],
    device: torch.device | int | str,
    byte_cap: int,
) -> Iterator[TensorBatch]:
    payload = dict(payload_tensors)
    packed_values = payload[G_PACKED_VALUES_NAME].to(device=device)
    packed_locations = payload[G_PACKED_INDICES_NAME].to(
        device=device,
        dtype=torch.long,
    )
    batch: TensorBatch = []
    batch_bytes = 0
    for item in metadata:
        numel = int(item["numel"])
        dtype = dtype_from_name(item["dtype"])
        values = packed_values[int(item["value_start"]) : int(item["value_end"])].to(
            dtype=dtype
        )
        tensor = torch.zeros(numel, dtype=dtype, device=device)
        tensor.index_copy_(
            0,
            packed_locations[int(item[G_INDEX_START_KEY]) : int(item[G_INDEX_END_KEY])],
            values,
        )
        tensor = tensor.view(tuple(item["shape"]))
        tensor_bytes = tensor.numel() * tensor.element_size()
        if batch and batch_bytes + tensor_bytes > byte_cap:
            yield batch
            batch = []
            batch_bytes = 0
        batch.append((item["name"], tensor))
        batch_bytes += tensor_bytes
    if batch:
        yield batch


def merge_sparse_payloads(payloads: list[TensorPayload]) -> TensorPayload:
    packed_parts = []
    value_parts = []
    metadata = []
    packed_offset = 0
    value_offset = 0
    for tensors, _, sparse_metadata in payloads:
        payload = dict(tensors)
        packed = payload[G_PACKED_INDICES_NAME]
        values = payload[G_PACKED_VALUES_NAME]
        packed_parts.append(packed)
        value_parts.append(values)
        for item in sparse_metadata:
            item = dict(item)
            item[G_INDEX_START_KEY] += packed_offset
            item[G_INDEX_END_KEY] += packed_offset
            item["value_start"] += value_offset
            item["value_end"] += value_offset
            metadata.append(item)
        packed_offset += int(packed.numel())
        value_offset += int(values.numel())
    return (
        [
            (G_PACKED_INDICES_NAME, torch.cat(packed_parts, dim=0)),
            (G_PACKED_VALUES_NAME, torch.cat(value_parts, dim=0)),
        ],
        G_SPARSE_INDICES_TRANSPORT,
        metadata,
    )
