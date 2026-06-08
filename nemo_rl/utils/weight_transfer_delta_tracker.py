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

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from nemo_rl.utils.packed_tensor import get_target_packed_tensor_size
from nemo_rl.utils.weight_transfer_protocol import (
    G_DEFAULT_BASELINE_PREWARM_CHUNK_BYTES,
    G_DEFAULT_BASELINE_STAGE_COALESCE_BYTES,
    G_FLOAT_DTYPE_MAP,
    G_REFIT_BASELINE_PREWARM_CHUNK_BYTES_ENV,
    G_REFIT_BASELINE_STAGE_COALESCE_BYTES_ENV,
    G_REFIT_PREWARM_DELTA_BASELINE_ENV,
    TensorBatch,
    TensorMetadata,
    dtype_itemsize,
    env_flag,
    env_int,
    memory_limited_stage_bytes,
    metadata_numel,
)


@dataclass(frozen=True)
class _BaselineEntry:
    arena: torch.Tensor
    offset: int
    numel: int


def _baseline_prewarm_enabled() -> bool:
    return env_flag(
        G_REFIT_PREWARM_DELTA_BASELINE_ENV,
        default=True,
    )


def _baseline_prewarm_chunk_bytes() -> int:
    default = (
        get_target_packed_tensor_size()
        if torch.cuda.is_available()
        else G_DEFAULT_BASELINE_PREWARM_CHUNK_BYTES
    )
    return env_int(
        G_REFIT_BASELINE_PREWARM_CHUNK_BYTES_ENV,
        default=default,
    )


def _baseline_stage_coalesce_bytes() -> int:
    return env_int(
        G_REFIT_BASELINE_STAGE_COALESCE_BYTES_ENV,
        default=G_DEFAULT_BASELINE_STAGE_COALESCE_BYTES,
    )


class DeltaCompressionTracker:
    """Tracks source-rank full baselines and prepares additive deltas."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        dtype_name = config["dtype"]
        if dtype_name not in G_FLOAT_DTYPE_MAP:
            raise ValueError(
                f"Unsupported delta compression dtype {dtype_name!r}; "
                f"expected one of {sorted(G_FLOAT_DTYPE_MAP)}."
            )
        self.delta_dtype = G_FLOAT_DTYPE_MAP[dtype_name]
        self.full_sync_interval = int(config["full_sync_interval"])
        self.sparse_bucket_size_bytes = int(config["sparse_bucket_size_bytes"])
        if self.full_sync_interval < 1:
            raise ValueError("delta_compression.full_sync_interval must be >= 1")
        if self.sparse_bucket_size_bytes < 1:
            raise ValueError("delta_compression.sparse_bucket_size_bytes must be >= 1")

        self.baseline: dict[str, torch.Tensor] = {}
        self._baseline_entries: dict[str, _BaselineEntry] = {}
        self.committed_syncs = 0
        self._d2h_stream: torch.cuda.Stream | None = None
        self._baseline_ready_events: dict[str, torch.cuda.Event] = {}

    def _baseline_dtype(self, dtype: torch.dtype) -> torch.dtype:
        return dtype

    def should_prewarm_baseline(self) -> bool:
        return (
            self.full_sync_interval > 1
            and self.committed_syncs == 0
            and _baseline_prewarm_enabled()
        )

    def is_delta_sync(self) -> bool:
        return (
            self.committed_syncs != 0
            and self.committed_syncs % self.full_sync_interval != 0
        )

    def prewarm_baseline_from_metadata(
        self,
        metadata: TensorMetadata,
    ) -> None:
        """Allocate baseline storage before the first full refit snapshot."""
        if not self.should_prewarm_baseline():
            return

        chunk_cap = _baseline_prewarm_chunk_bytes()
        if chunk_cap < 1:
            raise ValueError("baseline prewarm chunk bytes must be >= 1")

        pending: list[tuple[str, tuple[int, ...], torch.dtype]] = []
        pending_bytes = 0

        def flush_pending() -> None:
            nonlocal pending, pending_bytes
            if not pending:
                return
            self._allocate_baseline_views(pending)
            pending = []
            pending_bytes = 0

        for name, (shape, dtype) in metadata.items():
            shape_tuple = tuple(int(dim) for dim in shape)
            if self._has_matching_baseline(name, shape_tuple, dtype):
                continue
            if name in self._baseline_ready_events:
                self.flush_baseline([name])

            baseline_dtype = self._baseline_dtype(dtype)
            tensor_bytes = metadata_numel(shape_tuple) * dtype_itemsize(baseline_dtype)
            if pending and (
                pending[0][2] != baseline_dtype
                or pending_bytes + tensor_bytes > chunk_cap
            ):
                flush_pending()
            pending.append((name, shape_tuple, baseline_dtype))
            pending_bytes += tensor_bytes
            if pending_bytes >= chunk_cap:
                flush_pending()

        flush_pending()

    def prepare_chunk(
        self,
        tensors: TensorBatch,
    ) -> tuple[bool, TensorBatch]:
        if (
            self.committed_syncs == 0
            or self.committed_syncs % self.full_sync_interval == 0
        ):
            if self.full_sync_interval > 1:
                self._snapshot_baseline(tensors)
            return False, tensors

        # Keep this order: gate on prior D2H baseline writes, read the old
        # baseline, then snapshot the new weights for the next successful sync.
        self.flush_baseline(name for name, _ in tensors)
        has_non_float = any(not tensor.dtype.is_floating_point for _, tensor in tensors)
        if has_non_float:
            self._snapshot_baseline(tensors)
            return False, tensors

        deltas = self._make_delta_tensors(tensors)

        self._snapshot_baseline(tensors)
        return True, deltas

    def on_sync_succeeded(self) -> None:
        # The baseline snapshot reads from caller-owned model tensors. Finish
        # those reads before the caller can offload or update the weights.
        self.flush_baseline()
        self.committed_syncs += 1

    def on_sync_failed(self) -> None:
        self.flush_baseline()
        self.committed_syncs = 0

    def flush_baseline(self, names: Iterable[str] | None = None) -> None:
        if not self._baseline_ready_events:
            return
        d2h_stream = self._d2h_stream
        assert d2h_stream is not None
        if names is None:
            d2h_stream.synchronize()
            self._baseline_ready_events.clear()
            return

        names_to_wait = self._baseline_ready_events.keys() & set(names)
        current_stream = torch.cuda.current_stream()
        seen_events = set()
        for name in names_to_wait:
            event = self._baseline_ready_events.pop(name)
            if id(event) in seen_events:
                continue
            current_stream.wait_event(event)
            seen_events.add(id(event))

    def _snapshot_baseline(
        self,
        tensors: TensorBatch,
    ) -> None:
        if tensors[0][1].is_cuda and torch.cuda.is_available():
            self._snapshot_cuda_baseline(tensors)
            return
        for name, tensor in tensors:
            baseline = self.baseline.get(name)
            if (
                baseline is None
                or baseline.shape != tensor.shape
                or baseline.dtype != self._baseline_dtype(tensor.dtype)
            ):
                baseline = torch.empty_like(
                    tensor,
                    dtype=self._baseline_dtype(tensor.dtype),
                    device=torch.device("cpu"),
                )
                self.baseline[name] = baseline
                self._baseline_entries[name] = _BaselineEntry(
                    arena=baseline.view(-1),
                    offset=0,
                    numel=baseline.numel(),
                )
            baseline.copy_(tensor.detach())

    def _snapshot_cuda_baseline(
        self,
        tensors: TensorBatch,
    ) -> None:
        if self._d2h_stream is None:
            self._d2h_stream = torch.cuda.Stream()
        self._ensure_cuda_baseline_buffers(tensors)
        event = torch.cuda.current_stream().record_event()
        with torch.cuda.stream(self._d2h_stream):
            self._d2h_stream.wait_event(event)
            self._snapshot_cuda_baseline_to_host(tensors)
            ready_event = self._d2h_stream.record_event()
        for name, tensor in tensors:
            tensor.record_stream(self._d2h_stream)
            self._baseline_ready_events[name] = ready_event

    def _make_delta_tensors(self, tensors: TensorBatch) -> TensorBatch:
        if tensors[0][1].is_cuda and torch.cuda.is_available():
            return self._make_cuda_delta_tensors(tensors)
        return self._make_per_tensor_deltas(tensors)

    def _make_per_tensor_deltas(self, tensors: TensorBatch) -> TensorBatch:
        deltas = []
        for name, tensor in tensors:
            if name not in self.baseline:
                raise KeyError(
                    f"Delta baseline is missing tensor {name!r}; run a full sync "
                    "before delta sync resumes."
                )
            delta = tensor.detach().to(
                device=tensor.device,
                dtype=self.delta_dtype,
                non_blocking=tensor.is_cuda,
                copy=True,
            )
            baseline = self.baseline[name].to(
                device=tensor.device,
                dtype=self.delta_dtype,
                non_blocking=tensor.is_cuda,
                copy=True,
            )
            torch.sub(delta, baseline, out=delta)
            deltas.append((name, delta))
        return deltas

    def _make_cuda_delta_tensors(self, tensors: TensorBatch) -> TensorBatch:
        if any(name not in self._baseline_entries for name, _ in tensors):
            return self._make_per_tensor_deltas(tensors)

        delta_by_name: dict[str, torch.Tensor] = {}
        max_stage_bytes = memory_limited_stage_bytes(
            tensors[0][1].device,
            _baseline_stage_coalesce_bytes(),
        )

        def make_span_deltas(
            span: list[tuple[str, torch.Tensor, _BaselineEntry]],
            start: int,
            length: int,
        ) -> None:
            _, first_tensor, first_entry = span[0]
            source = first_entry.arena.narrow(0, start, length)
            try:
                delta_arena = source.to(
                    device=first_tensor.device,
                    dtype=self.delta_dtype,
                    non_blocking=True,
                    copy=True,
                )
            except torch.OutOfMemoryError:
                if len(span) == 1:
                    raise
                midpoint = len(span) // 2
                left = span[:midpoint]
                right = span[midpoint:]
                left_start, left_length = self._baseline_span_bounds(left)
                right_start, right_length = self._baseline_span_bounds(right)
                make_span_deltas(left, left_start, left_length)
                make_span_deltas(right, right_start, right_length)
                return

            for name, tensor, entry in span:
                delta = delta_arena[
                    entry.offset - start : entry.offset - start + entry.numel
                ].view(tensor.shape)
                current = tensor.detach()
                if current.dtype != delta.dtype:
                    current = current.to(dtype=delta.dtype)
                torch.sub(current, delta, out=delta)
                delta_by_name[name] = delta

        for span, start, length in self._iter_baseline_spans(
            tensors,
            itemsize=dtype_itemsize(self.delta_dtype),
            max_bytes=max_stage_bytes,
        ):
            make_span_deltas(span, start, length)

        return [(name, delta_by_name[name]) for name, _ in tensors]

    def _snapshot_cuda_baseline_to_host(
        self,
        tensors: TensorBatch,
    ) -> None:
        staging_buffers: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        max_stage_bytes = memory_limited_stage_bytes(
            tensors[0][1].device,
            _baseline_stage_coalesce_bytes(),
        )

        def copy_tensor_to_host(name: str, tensor: torch.Tensor) -> None:
            baseline = self.baseline[name]
            baseline.copy_(tensor.detach(), non_blocking=True)

        def get_staging_buffer(
            tensor: torch.Tensor,
            length: int,
        ) -> torch.Tensor | None:
            dtype = self._baseline_dtype(tensor.dtype)
            key = (tensor.device, dtype)
            buffer = staging_buffers.get(key)
            if buffer is not None and buffer.numel() >= length:
                return buffer.narrow(0, 0, length)

            try:
                buffer = torch.empty(length, dtype=dtype, device=tensor.device)
            except torch.OutOfMemoryError:
                return None

            staging_buffers[key] = buffer
            return buffer

        def copy_span_to_host(
            span: list[tuple[str, torch.Tensor, _BaselineEntry]],
            start: int,
            length: int,
        ) -> None:
            if len(span) == 1:
                name, tensor, _ = span[0]
                copy_tensor_to_host(name, tensor)
                return

            staging = get_staging_buffer(span[0][1], length)
            if staging is None:
                for name, tensor, _ in span:
                    copy_tensor_to_host(name, tensor)
                return

            staging_offset = 0
            for _, tensor, entry in span:
                staging.narrow(0, staging_offset, entry.numel).view(tensor.shape).copy_(
                    tensor.detach(), non_blocking=True
                )
                staging_offset += entry.numel

            first_entry = span[0][2]
            staging.record_stream(torch.cuda.current_stream())
            first_entry.arena.narrow(0, start, length).copy_(
                staging,
                non_blocking=True,
            )

        for span, start, length in self._iter_baseline_spans(
            tensors,
            itemsize=None,
            max_bytes=max_stage_bytes,
        ):
            copy_span_to_host(span, start, length)

    @staticmethod
    def _baseline_span_bounds(
        span: list[tuple[str, torch.Tensor, _BaselineEntry]],
    ) -> tuple[int, int]:
        start = span[0][2].offset
        end = span[-1][2].offset + span[-1][2].numel
        return start, end - start

    def _iter_baseline_spans(
        self,
        tensors: TensorBatch,
        *,
        itemsize: int | None,
        max_bytes: int,
    ) -> Iterator[tuple[list[tuple[str, torch.Tensor, _BaselineEntry]], int, int]]:
        grouped: dict[
            tuple[int, torch.device],
            list[tuple[str, torch.Tensor, _BaselineEntry]],
        ] = {}
        for name, tensor in tensors:
            entry = self._baseline_entries[name]
            key = (id(entry.arena), tensor.device)
            grouped.setdefault(key, []).append((name, tensor, entry))

        for items in grouped.values():
            items.sort(key=lambda item: item[2].offset)
            span: list[tuple[str, torch.Tensor, _BaselineEntry]] = []
            span_start = 0
            span_end = 0
            span_bytes = 0
            for item in items:
                entry = item[2]
                entry_itemsize = (
                    itemsize if itemsize is not None else entry.arena.element_size()
                )
                item_bytes = entry.numel * entry_itemsize
                can_merge = (
                    max_bytes > 0
                    and span
                    and entry.offset == span_end
                    and span_bytes + item_bytes <= max_bytes
                )
                if not span or can_merge:
                    if not span:
                        span_start = entry.offset
                        span_end = entry.offset + entry.numel
                        span_bytes = item_bytes
                    else:
                        span_end = entry.offset + entry.numel
                        span_bytes += item_bytes
                    span.append(item)
                    continue

                yield span, span_start, span_end - span_start
                span = [item]
                span_start = entry.offset
                span_end = entry.offset + entry.numel
                span_bytes = item_bytes

            if span:
                yield span, span_start, span_end - span_start

    def _ensure_cuda_baseline_buffers(
        self,
        tensors: TensorBatch,
    ) -> None:
        missing_by_dtype: dict[torch.dtype, TensorBatch] = {}
        for name, tensor in tensors:
            baseline_dtype = self._baseline_dtype(tensor.dtype)
            if self._has_matching_baseline(name, tuple(tensor.shape), tensor.dtype):
                continue
            if name in self._baseline_ready_events:
                self.flush_baseline([name])
            missing_by_dtype.setdefault(baseline_dtype, []).append((name, tensor))

        for dtype, missing_tensors in missing_by_dtype.items():
            self._allocate_baseline_views(
                [
                    (name, tuple(tensor.shape), dtype)
                    for name, tensor in missing_tensors
                ],
            )

    def _has_matching_baseline(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> bool:
        baseline = self.baseline.get(name)
        dtype = self._baseline_dtype(dtype)
        return (
            baseline is not None
            and tuple(baseline.shape) == shape
            and baseline.dtype == dtype
        )

    def _allocate_baseline_views(
        self,
        items: list[tuple[str, tuple[int, ...], torch.dtype]],
    ) -> None:
        if not items:
            return
        dtype = items[0][2]
        if any(item_dtype != dtype for _, _, item_dtype in items):
            raise ValueError("baseline allocation items must have the same dtype")
        total_numel = sum(metadata_numel(shape) for _, shape, _ in items)
        baseline_arena = torch.empty(
            total_numel,
            dtype=dtype,
            device=torch.device("cpu"),
            pin_memory=torch.cuda.is_available(),
        )
        offset = 0
        for name, shape, _ in items:
            numel = metadata_numel(shape)
            self.baseline[name] = baseline_arena[offset : offset + numel].view(shape)
            self._baseline_entries[name] = _BaselineEntry(
                arena=baseline_arena,
                offset=offset,
                numel=numel,
            )
            offset += numel


def create_vllm_delta_transfer_tracker(
    generation_config: Mapping[str, Any] | None,
) -> DeltaCompressionTracker | None:
    """Create a vLLM delta transfer tracker when config enables it."""
    if generation_config is None:
        return None
    delta_config = generation_config.get("delta_compression")
    if delta_config is None or not delta_config["enabled"]:
        return None
    if generation_config["backend"] != "vllm":
        raise ValueError("Delta compression is currently supported only for vLLM.")
    if generation_config["colocated"]["enabled"]:
        raise ValueError(
            "Delta compression is supported only for non-colocated vLLM refit."
        )
    if generation_config.get("quant_cfg") is not None:
        raise NotImplementedError(
            "Delta compression for vLLM ModelOpt quantized weights is not implemented."
        )
    if generation_config["vllm_cfg"].get("precision") == "fp8":
        raise NotImplementedError(
            "Delta compression for vLLM FP8 model weights is not implemented."
        )
    return DeltaCompressionTracker(delta_config)
