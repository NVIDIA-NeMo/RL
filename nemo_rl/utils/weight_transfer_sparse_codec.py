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
from typing import Any

import numpy as np
import torch

NamedTensor = tuple[str, torch.Tensor]
TensorBatch = list[NamedTensor]
TensorPayload = tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]
PreparedTensorPayload = tuple[TensorPayload, int, int]


def encode_sparse_infos(
    infos: Iterable[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    empty_dtype: torch.dtype,
) -> TensorPayload:
    packed_locations = []
    packed_values = []
    metadata: list[dict[str, Any]] = []
    index_offset = value_offset = 0
    for name, tensor, raw_locations, raw_values in infos:
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
        packed_values.append(raw_values)
        metadata.append(
            {
                "name": name,
                "shape": tuple(int(dim) for dim in tensor.shape),
                "index_start": index_offset,
                "index_end": index_offset + index_count,
                "value_start": value_offset,
                "value_end": value_offset + count,
                **location_metadata,
            }
        )
        index_offset += index_count
        value_offset += count
    indices = (
        torch.cat(packed_locations)
        if packed_locations
        else torch.empty(0, dtype=torch.uint8)
    )
    values = (
        torch.cat(packed_values) if packed_values else torch.empty(0, dtype=empty_dtype)
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
        self.delta_dtype = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }[str(config["dtype"]).lower()]
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
            current = tensor.detach().cpu()
            current_flat, baseline_flat = current.view(-1), baseline.view(-1)
            total_elements += current_flat.numel()
            locations = (current_flat != baseline_flat).nonzero().view(-1)
            changed_elements += locations.numel()
            if locations.numel():
                current_values = current_flat[locations]
                baseline_values = baseline_flat[locations]
                deltas = (current_values - baseline_values).to(self.delta_dtype)
                expected_values = baseline_values + deltas.to(baseline.dtype)
                sparse_infos.append(
                    (
                        name,
                        current,
                        locations,
                        deltas,
                    )
                )
                if self.verification_samples:
                    verification_sources.append((locations, deltas))
                pending_updates[name] = (locations, expected_values)
        with self._pending_updates_lock:
            self._pending_updates.update(pending_updates)
        payload = encode_sparse_infos(sparse_infos, empty_dtype=self.delta_dtype)
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
        for item, (locations, deltas) in zip(metadata, sources, strict=True):
            end = offset + locations.numel()
            while sample_index < count and sample_ranks[sample_index] < end:
                local_index = sample_ranks[sample_index] - offset
                location = int(locations[local_index])
                item.setdefault("verification_locations", []).append(location)
                item.setdefault("verification_deltas", []).append(
                    float(deltas[local_index])
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
            self._baseline(name, tuple(tensor.shape), tensor.dtype).copy_(tensor)

    def _wait_for_baseline_commits(self) -> None:
        for commit in self._baseline_commits:
            commit.result()
        self._baseline_commits = ()

    def _commit_baseline_updates(
        self, updates: Iterable[tuple[str, tuple[torch.Tensor, torch.Tensor]]]
    ) -> None:
        for name, (locations, values) in updates:
            target = self.baseline[name].view(-1)
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
        if self.baseline_in_memory:
            baseline = torch.empty(shape, dtype=dtype)
        else:
            numel = torch.Size(shape).numel()
            with tempfile.NamedTemporaryFile(
                prefix="nrl-refit-baseline-", dir=self.baseline_mmap_dir
            ) as handle:
                handle.truncate(numel * torch.empty((), dtype=dtype).element_size())
                baseline = torch.from_file(
                    handle.name, shared=True, size=numel, dtype=dtype
                ).view(shape)
        self.baseline[name] = baseline
        return baseline
