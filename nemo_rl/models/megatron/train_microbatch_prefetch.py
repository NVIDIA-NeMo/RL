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
"""All-field TransferQueue fetch at packed training-microbatch granularity.

NeMo-RL supplies an exact key and packing plan. One rank per data-parallel
replica reads those keys from TQ, materializes every selected train field, and
distributes the CPU payload over a dedicated Gloo group. Every model-parallel
rank publishes the microbatch to a local capacity-one queue only after that
distribution completes.

The dedicated background group is important for pipeline parallelism: pipeline
stages consume their data iterators at different times, so the main thread must
not enter an all-stage collective from ``next()``.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterator, Sequence

import torch
from megatron.core import parallel_state

from nemo_rl.data.llm_message_utils import attach_message_log_view
from nemo_rl.data_plane import DataPlaneClient, KVBatchMeta, materialize
from nemo_rl.data_plane.schema import (
    ELEM_COUNTS_PER_GB,
    GLOBAL_FORWARD_PAD_SEQLEN,
    GLOBAL_VALID_SEQS_PER_GB,
    GLOBAL_VALID_TOKS_PER_GB,
    MICRO_BATCH_INDICES,
    MICRO_BATCH_LENGTHS,
)
from nemo_rl.data_plane.worker_mixin import _broadcast_batched_data_dict
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.r3_trace import trace_tq_prefetch_payload

TQ_SAMPLE_IDS_FIELD = "__tq_sample_ids"

_QUEUE_POLL_SECONDS = 0.1
_CLOSE_TIMEOUT_SECONDS = 10.0
_MAX_ERROR_LENGTH = 4096


class TrainMicrobatchPrefetchError(RuntimeError):
    """Failure raised consistently by consumers of a prefetched microbatch."""


@dataclass(frozen=True)
class RankCoordinates:
    """Global rank and its data/model-parallel coordinates."""

    global_rank: int
    dp_rank: int
    pp_rank: int
    tp_rank: int
    cp_rank: int


@dataclass(frozen=True)
class TrainMicrobatchPrefetchGroup:
    """Dedicated CPU distribution group for one DP replica."""

    replica_group: Any
    replica_ranks: tuple[int, ...]
    source_rank: int
    is_source: bool


@dataclass(frozen=True)
class TrainGlobalBatchPlan:
    """Packed microbatches and normalization totals for one global batch."""

    global_batch_index: int
    microbatch_sample_ids: tuple[tuple[str, ...], ...]
    microbatch_lengths: tuple[int, ...]
    global_valid_seqs: float
    global_valid_toks: float


@dataclass(frozen=True)
class TrainMicrobatchPlan:
    """Complete metadata-only plan for one local DP shard."""

    fields: tuple[str, ...]
    pad_to_seqlen: int
    global_batches: tuple[TrainGlobalBatchPlan, ...]

    @property
    def num_microbatches(self) -> int:
        return sum(len(batch.microbatch_sample_ids) for batch in self.global_batches)


@dataclass(frozen=True)
class PrefetchedMicrobatch:
    """One fully materialized and distributed CPU microbatch."""

    global_batch_index: int
    microbatch_index: int
    sample_ids: tuple[str, ...]
    data: BatchedDataDict[Any]


@dataclass(frozen=True)
class _PrefetchFailure:
    global_batch_index: int
    microbatch_index: int
    message: str


def _validate_timeout_seconds(value: float, *, name: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{name} must be finite and greater than zero, got {value!r}")
    return seconds


def _as_nonempty_sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"invalid {name}: {value!r}")
    return value


def build_train_microbatch_plan(meta: KVBatchMeta) -> TrainMicrobatchPlan:
    """Translate driver packing metadata into exact per-microbatch key lists."""
    if not meta.fields:
        raise ValueError("train microbatch prefetch requires a non-empty field list")

    extra = meta.extra_info or {}
    indices_per_gb = _as_nonempty_sequence(
        extra.get(MICRO_BATCH_INDICES), name=MICRO_BATCH_INDICES
    )
    lengths_per_gb = _as_nonempty_sequence(
        extra.get(MICRO_BATCH_LENGTHS), name=MICRO_BATCH_LENGTHS
    )
    if len(indices_per_gb) != len(lengths_per_gb):
        raise ValueError(
            f"{MICRO_BATCH_INDICES} has {len(indices_per_gb)} global batches but "
            f"{MICRO_BATCH_LENGTHS} has {len(lengths_per_gb)}"
        )

    elem_counts = extra.get(ELEM_COUNTS_PER_GB)
    if elem_counts is None:
        if len(indices_per_gb) != 1:
            raise ValueError(
                f"{ELEM_COUNTS_PER_GB!r} is required for multiple global batches"
            )
        elem_counts = [len(meta.sample_ids)]
    if not isinstance(elem_counts, (list, tuple)) or len(elem_counts) != len(
        indices_per_gb
    ):
        raise ValueError(
            f"{ELEM_COUNTS_PER_GB} must contain one value per global batch"
        )
    counts = [int(count) for count in elem_counts]
    if any(count <= 0 for count in counts) or sum(counts) != len(meta.sample_ids):
        raise ValueError(
            f"invalid {ELEM_COUNTS_PER_GB}={counts}; expected positive counts "
            f"summing to {len(meta.sample_ids)}"
        )

    valid_seqs = _as_nonempty_sequence(
        extra.get(GLOBAL_VALID_SEQS_PER_GB), name=GLOBAL_VALID_SEQS_PER_GB
    )
    valid_toks = _as_nonempty_sequence(
        extra.get(GLOBAL_VALID_TOKS_PER_GB), name=GLOBAL_VALID_TOKS_PER_GB
    )
    if len(valid_seqs) != len(indices_per_gb) or len(valid_toks) != len(indices_per_gb):
        raise ValueError(
            "train normalization metadata must contain one value per global batch"
        )

    pad_to_seqlen = int(extra.get(GLOBAL_FORWARD_PAD_SEQLEN, 0))
    if pad_to_seqlen <= 0:
        raise ValueError(
            f"train microbatch prefetch requires positive {GLOBAL_FORWARD_PAD_SEQLEN}"
        )

    global_batches: list[TrainGlobalBatchPlan] = []
    offset = 0
    for gb_idx, (ranges, lengths, elem_count) in enumerate(
        zip(indices_per_gb, lengths_per_gb, counts, strict=True)
    ):
        ranges = _as_nonempty_sequence(ranges, name=f"{MICRO_BATCH_INDICES}[{gb_idx}]")
        lengths = _as_nonempty_sequence(
            lengths, name=f"{MICRO_BATCH_LENGTHS}[{gb_idx}]"
        )
        if len(ranges) != len(lengths):
            raise ValueError(
                f"global batch {gb_idx} has {len(ranges)} ranges but "
                f"{len(lengths)} packed lengths"
            )

        key_batches: list[tuple[str, ...]] = []
        expected_start = 0
        parsed_lengths: list[int] = []
        for mb_idx, (microbatch_range, length) in enumerate(
            zip(ranges, lengths, strict=True)
        ):
            if (
                not isinstance(microbatch_range, (list, tuple))
                or len(microbatch_range) != 2
            ):
                raise ValueError(
                    f"invalid packed range for global batch {gb_idx}, "
                    f"microbatch {mb_idx}: {microbatch_range!r}"
                )
            start, end = (int(microbatch_range[0]), int(microbatch_range[1]))
            if start != expected_start or end <= start or end > elem_count:
                raise ValueError(
                    "packed ranges must cover each global batch exactly once in "
                    f"row order; global batch {gb_idx} expected start "
                    f"{expected_start}, got [{start}, {end})"
                )
            packed_length = int(length)
            if packed_length <= 0:
                raise ValueError(
                    f"invalid packed length {packed_length} for global batch "
                    f"{gb_idx}, microbatch {mb_idx}"
                )
            key_batches.append(tuple(meta.sample_ids[offset + start : offset + end]))
            parsed_lengths.append(packed_length)
            expected_start = end

        if expected_start != elem_count:
            raise ValueError(
                f"global batch {gb_idx} covers {expected_start} of {elem_count} rows"
            )
        gb_valid_seqs = float(valid_seqs[gb_idx])
        gb_valid_toks = float(valid_toks[gb_idx])
        if gb_valid_seqs < 0 or gb_valid_toks < 0:
            raise ValueError("train normalization totals must be non-negative")
        global_batches.append(
            TrainGlobalBatchPlan(
                global_batch_index=gb_idx,
                microbatch_sample_ids=tuple(key_batches),
                microbatch_lengths=tuple(parsed_lengths),
                global_valid_seqs=gb_valid_seqs,
                global_valid_toks=gb_valid_toks,
            )
        )
        offset += elem_count

    return TrainMicrobatchPlan(
        fields=tuple(meta.fields),
        pad_to_seqlen=pad_to_seqlen,
        global_batches=tuple(global_batches),
    )


def build_replica_topology(
    coordinates: Sequence[RankCoordinates],
) -> dict[int, tuple[RankCoordinates, ...]]:
    """Group every TP/CP/PP sibling by DP rank in deterministic rank order."""
    if not coordinates:
        raise ValueError("train-prefetch topology cannot be empty")

    replicas: dict[int, tuple[RankCoordinates, ...]] = {}
    for dp_rank in sorted({coord.dp_rank for coord in coordinates}):
        members = tuple(
            sorted(
                (coord for coord in coordinates if coord.dp_rank == dp_rank),
                key=lambda coord: coord.global_rank,
            )
        )
        sources = [
            coord
            for coord in members
            if coord.pp_rank == 0 and coord.tp_rank == 0 and coord.cp_rank == 0
        ]
        if len(sources) != 1:
            raise ValueError(
                "train-prefetch topology requires exactly one PP=TP=CP=0 "
                f"source for DP {dp_rank}; got {len(sources)}"
            )
        logical_coords = {
            (coord.pp_rank, coord.tp_rank, coord.cp_rank) for coord in members
        }
        if len(logical_coords) != len(members):
            raise ValueError(f"duplicate model-parallel coordinates in DP {dp_rank}")
        replicas[dp_rank] = members
    return replicas


def initialize_train_microbatch_prefetch_group(
    *, collective_timeout_s: float | None
) -> TrainMicrobatchPrefetchGroup:
    """Create one dedicated Gloo group per DP replica on every world rank."""
    if collective_timeout_s is not None:
        collective_timeout_s = _validate_timeout_seconds(
            collective_timeout_s,
            name="policy.train_microbatch_prefetch.collective_timeout_s",
        )
    if not torch.distributed.is_initialized():
        raise RuntimeError("train microbatch prefetch requires torch.distributed")

    world_size = torch.distributed.get_world_size()
    world_backend = torch.distributed.get_backend()
    coordinate_device: Any = (
        torch.device("cuda", torch.cuda.current_device())
        if world_backend == "nccl"
        else torch.device("cpu")
    )
    local = torch.tensor(
        [
            parallel_state.get_data_parallel_rank(),
            parallel_state.get_pipeline_model_parallel_rank(),
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_context_parallel_rank(),
        ],
        dtype=torch.long,
        device=coordinate_device,
    )
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, local)
    coordinates = [
        RankCoordinates(rank, *[int(value) for value in row.tolist()])
        for rank, row in enumerate(gathered)
    ]
    replicas = build_replica_topology(coordinates)

    groups: dict[int, Any] = {}
    for dp_rank, members in replicas.items():
        group_kwargs: dict[str, Any] = {
            "ranks": [coord.global_rank for coord in members],
            "backend": "gloo",
        }
        if collective_timeout_s is not None:
            group_kwargs["timeout"] = timedelta(seconds=collective_timeout_s)
        groups[dp_rank] = torch.distributed.new_group(**group_kwargs)

    my_dp_rank = parallel_state.get_data_parallel_rank()
    my_members = replicas[my_dp_rank]
    source = next(
        coord
        for coord in my_members
        if coord.pp_rank == 0 and coord.tp_rank == 0 and coord.cp_rank == 0
    )
    my_rank = torch.distributed.get_rank()
    return TrainMicrobatchPrefetchGroup(
        replica_group=groups[my_dp_rank],
        replica_ranks=tuple(coord.global_rank for coord in my_members),
        source_rank=source.global_rank,
        is_source=my_rank == source.global_rank,
    )


def _tensor_bytes(data: BatchedDataDict[Any]) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in data.values()
        if isinstance(value, torch.Tensor)
    )


class TrainMicrobatchPrefetcher(Iterator[PrefetchedMicrobatch]):
    """Fetch and distribute all train fields with depth-zero or depth-one."""

    def __init__(
        self,
        *,
        client: DataPlaneClient,
        meta: KVBatchMeta,
        group: TrainMicrobatchPrefetchGroup,
        pad_value_dict: dict[str, Any],
        depth: int,
        item_ready_timeout_s: float,
    ) -> None:
        if depth not in (0, 1):
            raise ValueError(
                f"train microbatch prefetch depth must be 0 or 1, got {depth}"
            )
        self._client = client
        self._meta = meta
        self._group = group
        self._pad_value_dict = pad_value_dict
        self._depth = depth
        self._item_ready_timeout_s = _validate_timeout_seconds(
            item_ready_timeout_s,
            name="policy.train_microbatch_prefetch.item_ready_timeout_s",
        )
        self.plan = build_train_microbatch_plan(meta)
        self._items = tuple(
            (batch.global_batch_index, mb_idx, sample_ids)
            for batch in self.plan.global_batches
            for mb_idx, sample_ids in enumerate(batch.microbatch_sample_ids)
        )
        self._queue: queue.Queue[PrefetchedMicrobatch | _PrefetchFailure] = queue.Queue(
            maxsize=1
        )
        # One initial permit prepares MB0. Depth one releases after consuming
        # MB N; depth zero releases on the source only when MB N+1 is requested.
        self._producer_permit = threading.Semaphore(1)
        self._stop_event = threading.Event()
        self._consumed = 0
        self._closed = False
        self._terminal_failure: _PrefetchFailure | None = None
        self._started_at = time.perf_counter()
        self._metrics_lock = threading.Lock()
        self._metrics: dict[str, float] = {
            "tq_get_s": 0.0,
            "materialize_s": 0.0,
            "distribute_s": 0.0,
            "consumer_wait_s": 0.0,
            "first_microbatch_ready_s": 0.0,
            "tq_get_calls": 0.0,
            "materialized_payload_bytes": 0.0,
            "queued_payload_peak_bytes": 0.0,
            "ready_count": 0.0,
            "consume_count": 0.0,
        }
        target = self._source_loop if group.is_source else self._receiver_loop
        self._producer_thread = threading.Thread(
            target=target,
            name=f"train-microbatch-prefetch-rank-{torch.distributed.get_rank()}",
            daemon=True,
        )
        self._producer_thread.start()

    def __iter__(self) -> TrainMicrobatchPrefetcher:
        return self

    def _record(self, key: str, value: float) -> None:
        with self._metrics_lock:
            self._metrics[key] += value

    def _record_max(self, key: str, value: float) -> None:
        with self._metrics_lock:
            self._metrics[key] = max(self._metrics[key], value)

    def _publish(self, item: PrefetchedMicrobatch | _PrefetchFailure) -> None:
        if self._stop_event.is_set():
            return
        if self._consumed == 0:
            with self._metrics_lock:
                if self._metrics["first_microbatch_ready_s"] == 0.0:
                    self._metrics["first_microbatch_ready_s"] = (
                        time.perf_counter() - self._started_at
                    )
        if isinstance(item, PrefetchedMicrobatch):
            self._record_max(
                "queued_payload_peak_bytes", float(_tensor_bytes(item.data))
            )
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=_QUEUE_POLL_SECONDS)
                return
            except queue.Full:
                continue

    def _broadcast_envelope(self, envelope: list[Any]) -> tuple[Any, ...]:
        if len(self._group.replica_ranks) > 1:
            torch.distributed.broadcast_object_list(
                envelope,
                src=self._group.source_rank,
                group=self._group.replica_group,
            )
        value = envelope[0]
        if not isinstance(value, tuple):
            raise TrainMicrobatchPrefetchError(
                f"invalid train-prefetch envelope: {value!r}"
            )
        return value

    def _source_loop(self) -> None:
        for gb_idx, mb_idx, sample_ids in self._items:
            self._producer_permit.acquire()
            if self._stop_event.is_set():
                return
            data: BatchedDataDict[Any] | None = None
            try:
                start = time.perf_counter()
                wire = self._client.get_samples(
                    sample_ids=list(sample_ids),
                    partition_id=self._meta.partition_id,
                    select_fields=list(self.plan.fields),
                )
                self._record("tq_get_s", time.perf_counter() - start)
                self._record("tq_get_calls", 1.0)

                start = time.perf_counter()
                data = materialize(
                    wire,
                    layout="padded",
                    pad_value_dict=self._pad_value_dict,
                    pad_to_seqlen=self.plan.pad_to_seqlen,
                )
                del wire
                if data.size != len(sample_ids):
                    raise ValueError(
                        f"materialized {data.size} rows for {len(sample_ids)} keys"
                    )
                missing = [field for field in self.plan.fields if field not in data]
                if missing:
                    raise KeyError(
                        f"materialized microbatch is missing fields {missing}"
                    )
                payload_bytes = float(_tensor_bytes(data))
                self._record("materialize_s", time.perf_counter() - start)
                self._record("materialized_payload_bytes", payload_bytes)
                envelope: list[Any] = [("ok", gb_idx, mb_idx)]
            except Exception as error:
                envelope = [
                    (
                        "error",
                        gb_idx,
                        mb_idx,
                        f"{type(error).__name__}: {error}"[:_MAX_ERROR_LENGTH],
                    )
                ]

            try:
                start = time.perf_counter()
                status = self._broadcast_envelope(envelope)
                if status[0] == "error":
                    self._record("distribute_s", time.perf_counter() - start)
                    self._publish(_PrefetchFailure(gb_idx, mb_idx, str(status[3])))
                    return
                assert data is not None
                if len(self._group.replica_ranks) > 1:
                    data = _broadcast_batched_data_dict(
                        data,
                        is_leader=True,
                        src=self._group.source_rank,
                        group=self._group.replica_group,
                    )
                self._record("distribute_s", time.perf_counter() - start)
                self._publish(PrefetchedMicrobatch(gb_idx, mb_idx, sample_ids, data))
            except Exception as error:
                self._publish(
                    _PrefetchFailure(
                        gb_idx,
                        mb_idx,
                        f"{type(error).__name__}: {error}"[:_MAX_ERROR_LENGTH],
                    )
                )
                return

    def _receiver_loop(self) -> None:
        for expected_gb, expected_mb, sample_ids in self._items:
            self._producer_permit.acquire()
            if self._stop_event.is_set():
                return
            try:
                start = time.perf_counter()
                status = self._broadcast_envelope([None])
                if status[0] == "error":
                    self._record("distribute_s", time.perf_counter() - start)
                    self._publish(
                        _PrefetchFailure(int(status[1]), int(status[2]), str(status[3]))
                    )
                    return
                if status != ("ok", expected_gb, expected_mb):
                    raise TrainMicrobatchPrefetchError(
                        "train-prefetch order mismatch: expected "
                        f"({expected_gb}, {expected_mb}), received {status}"
                    )
                data = _broadcast_batched_data_dict(
                    None,
                    is_leader=False,
                    src=self._group.source_rank,
                    group=self._group.replica_group,
                )
                self._record("distribute_s", time.perf_counter() - start)
                self._publish(
                    PrefetchedMicrobatch(expected_gb, expected_mb, sample_ids, data)
                )
            except Exception as error:
                self._publish(
                    _PrefetchFailure(
                        expected_gb,
                        expected_mb,
                        f"{type(error).__name__}: {error}"[:_MAX_ERROR_LENGTH],
                    )
                )
                return

    def _take(self) -> PrefetchedMicrobatch | _PrefetchFailure:
        start = time.perf_counter()
        try:
            item = self._queue.get_nowait()
            self._record("ready_count", 1.0)
        except queue.Empty:
            deadline = time.monotonic() + self._item_ready_timeout_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    expected_gb, expected_mb, _ = self._items[self._consumed]
                    item = _PrefetchFailure(
                        expected_gb,
                        expected_mb,
                        "microbatch did not become ready within "
                        f"{self._item_ready_timeout_s:g}s; background data-plane "
                        "or collective work may still be running",
                    )
                    self._terminal_failure = item
                    self._stop_event.set()
                    break
                try:
                    item = self._queue.get(timeout=min(_QUEUE_POLL_SECONDS, remaining))
                    break
                except queue.Empty:
                    if not self._producer_thread.is_alive():
                        item = _PrefetchFailure(
                            -1,
                            self._consumed,
                            "prefetch producer exited without publishing a result",
                        )
                        break
        self._record("consumer_wait_s", time.perf_counter() - start)
        return item

    def __next__(self) -> PrefetchedMicrobatch:
        if self._terminal_failure is not None:
            raise TrainMicrobatchPrefetchError(
                "train microbatch prefetch is terminal after failure: "
                f"{self._terminal_failure.message}"
            )
        if self._consumed >= len(self._items):
            raise StopIteration

        expected_gb, expected_mb, expected_ids = self._items[self._consumed]
        if self._depth == 0 and self._group.is_source and self._consumed > 0:
            self._producer_permit.release()

        item = self._take()
        if isinstance(item, _PrefetchFailure):
            self._terminal_failure = item
            self._stop_event.set()
            raise TrainMicrobatchPrefetchError(
                "train microbatch prefetch failed for global batch "
                f"{item.global_batch_index}, microbatch {item.microbatch_index}: "
                f"{item.message}"
            )
        if (
            item.global_batch_index != expected_gb
            or item.microbatch_index != expected_mb
            or item.sample_ids != expected_ids
        ):
            raise TrainMicrobatchPrefetchError(
                "train-prefetch consumer order mismatch: expected "
                f"({expected_gb}, {expected_mb}, {expected_ids}), received "
                f"({item.global_batch_index}, {item.microbatch_index}, "
                f"{item.sample_ids})"
            )
        if item.data.size != len(expected_ids):
            raise TrainMicrobatchPrefetchError(
                f"prefetched microbatch has {item.data.size} rows for "
                f"{len(expected_ids)} keys"
            )

        attach_message_log_view(item.data)
        trace_tq_prefetch_payload(keys=item.sample_ids, data=item.data)
        self._consumed += 1
        self._record("consume_count", 1.0)
        if self._depth == 1 or not self._group.is_source:
            self._producer_permit.release()
        return item

    def iter_global_batch(
        self, global_batch_index: int
    ) -> Iterator[BatchedDataDict[Any]]:
        """Yield exactly the planned raw microbatches for one global batch."""
        batch_plan = self.plan.global_batches[global_batch_index]
        for _ in batch_plan.microbatch_sample_ids:
            item = next(self)
            if item.global_batch_index != global_batch_index:
                raise TrainMicrobatchPrefetchError(
                    f"expected global batch {global_batch_index}, received "
                    f"{item.global_batch_index}"
                )
            yield item.data

    def assert_complete(self) -> None:
        if self._consumed != len(self._items):
            raise TrainMicrobatchPrefetchError(
                "Megatron consumed an unexpected number of microbatches: "
                f"consumed={self._consumed}, planned={len(self._items)}"
            )

    def metrics(self) -> dict[str, float]:
        """Return development metrics for this rank."""
        with self._metrics_lock:
            result = dict(self._metrics)
        consumed = result["consume_count"]
        result["ready_fraction"] = result["ready_count"] / consumed if consumed else 0.0
        return result

    def close(self) -> None:
        """Finish the producer or bound cleanup after a terminal timeout."""
        if self._closed:
            return
        self._closed = True
        deadline = time.monotonic() + _CLOSE_TIMEOUT_SECONDS
        while self._producer_thread.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                self._queue.get(timeout=min(_QUEUE_POLL_SECONDS, remaining))
            except queue.Empty:
                pass
            self._producer_permit.release()
        remaining = max(0.0, deadline - time.monotonic())
        self._producer_thread.join(timeout=remaining)
        if self._producer_thread.is_alive():
            self._stop_event.set()
            self._producer_permit.release()
            raise TrainMicrobatchPrefetchError(
                "train microbatch prefetch producer did not stop within "
                f"{_CLOSE_TIMEOUT_SECONDS:.0f}s"
            )
