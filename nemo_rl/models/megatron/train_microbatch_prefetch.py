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

NeMo-RL supplies an exact key and packing plan. One TP0/CP0 leader for every
fixed data-parallel and pipeline-parallel stage reads those keys from TQ and
materializes every selected train field on a background CPU thread. When that
stage consumes the microbatch, its foreground training thread broadcasts the
payload through the existing TP×CP NCCL group.

No collective spans pipeline stages, and no producer thread launches NCCL.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterator, Sequence, TypeVar

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
from nemo_rl.utils.nsys import nsys_nvtx_range
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
    """Direct-reader and stage-local NCCL topology for this rank."""

    stage_group: Any
    stage_ranks: tuple[int, ...]
    stage_source_rank: int
    is_stage_leader: bool


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
    """One fully materialized microbatch, GPU-resident after stage fanout."""

    global_batch_index: int
    microbatch_index: int
    sample_ids: tuple[str, ...]
    data: BatchedDataDict[Any]


@dataclass(frozen=True)
class _PrefetchFailure:
    message: str
    from_source: bool


class _PrefetchEnd:
    pass


_PREFETCH_END = _PrefetchEnd()
_T = TypeVar("_T")


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
    fields = meta.fields
    if not fields:
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
        fields=tuple(fields),
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
        pp_ranks = {coord.pp_rank for coord in members}
        stage_leaders = [
            coord for coord in members if coord.tp_rank == 0 and coord.cp_rank == 0
        ]
        if len(stage_leaders) != len(pp_ranks):
            raise ValueError(
                "train-prefetch topology requires exactly one TP=CP=0 leader "
                f"per PP stage for DP {dp_rank}; got leaders={stage_leaders}"
            )
        if {leader.pp_rank for leader in stage_leaders} != pp_ranks:
            raise ValueError(
                "train-prefetch topology is missing a TP=CP=0 leader for at "
                f"least one PP stage in DP {dp_rank}"
            )
        logical_coords = {
            (coord.pp_rank, coord.tp_rank, coord.cp_rank) for coord in members
        }
        if len(logical_coords) != len(members):
            raise ValueError(f"duplicate model-parallel coordinates in DP {dp_rank}")
        replicas[dp_rank] = members
    return replicas


def initialize_train_microbatch_prefetch_group() -> TrainMicrobatchPrefetchGroup:
    """Select one direct TQ reader and existing TP×CP NCCL group per PP stage."""
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

    my_dp_rank = parallel_state.get_data_parallel_rank()
    my_members = replicas[my_dp_rank]
    my_pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    my_stage_members = tuple(
        coord for coord in my_members if coord.pp_rank == my_pp_rank
    )
    stage_leaders = [
        coord for coord in my_stage_members if coord.tp_rank == 0 and coord.cp_rank == 0
    ]
    if len(stage_leaders) != 1:
        raise ValueError(
            "train-prefetch topology requires exactly one stage leader for "
            f"DP {my_dp_rank}, PP {my_pp_rank}; got {len(stage_leaders)}"
        )
    stage_leader = stage_leaders[0]

    stage_group = parallel_state.get_tensor_and_context_parallel_group()
    expected_stage_ranks = tuple(
        sorted(coord.global_rank for coord in my_stage_members)
    )
    actual_stage_ranks = tuple(torch.distributed.get_process_group_ranks(stage_group))
    if set(actual_stage_ranks) != set(expected_stage_ranks) or len(
        actual_stage_ranks
    ) != len(expected_stage_ranks):
        raise RuntimeError(
            "Megatron TP×CP group does not match the train-prefetch PP stage: "
            f"expected={expected_stage_ranks}, actual={actual_stage_ranks}"
        )
    if torch.distributed.get_backend(stage_group) != "nccl":
        raise RuntimeError("train-prefetch stage-local group must use NCCL")

    my_rank = torch.distributed.get_rank()
    return TrainMicrobatchPrefetchGroup(
        stage_group=stage_group,
        stage_ranks=expected_stage_ranks,
        stage_source_rank=stage_leader.global_rank,
        is_stage_leader=my_rank == stage_leader.global_rank,
    )


def _tensor_bytes(data: BatchedDataDict[Any]) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in data.values()
        if isinstance(value, torch.Tensor)
    )


def _move_stage_payload_to_cuda(
    data: BatchedDataDict[Any],
) -> BatchedDataDict[Any]:
    """Stage the complete leader payload before peers enter payload collectives."""
    return data.to(torch.cuda.current_device())


class _ThreadPrefetchError(RuntimeError):
    def __init__(self, message: str, *, from_source: bool) -> None:
        super().__init__(message)
        self.from_source = from_source


class _ThreadPrefetchIterator(Iterator[_T], Generic[_T]):
    """Run a finite synchronous iterator ahead on one daemon thread.

    Item zero is prepared eagerly. ``close()`` drains the finite source so that
    distributed producers and receivers finish the same in-flight operations.
    """

    def __init__(
        self,
        source_iterator: Iterator[_T],
        *,
        lookahead: bool,
        item_ready_timeout_s: float,
        thread_name: str,
        item_size: Callable[[_T], int] | None = None,
    ) -> None:
        self._source_iterator = source_iterator
        self._lookahead = lookahead
        self._item_ready_timeout_s = _validate_timeout_seconds(
            item_ready_timeout_s,
            name="item_ready_timeout_s",
        )
        self._item_size = item_size
        self._queue: queue.Queue[_T | _PrefetchFailure | _PrefetchEnd] = queue.Queue(
            maxsize=1
        )
        # The queue bounds completed results; this permit separately prevents
        # the producer from starting another item before lookahead is allowed.
        self._producer_permit = threading.Semaphore(1)
        self._stop_event = threading.Event()
        self._consumed = 0
        self._closed = False
        self._ended = False
        self._terminal_failure: _PrefetchFailure | None = None
        self._started_at = time.perf_counter()
        self._metrics_lock = threading.Lock()
        self._metrics: dict[str, float] = {
            "consumer_wait_s": 0.0,
            "first_item_ready_s": 0.0,
            "queued_payload_peak_bytes": 0.0,
            "ready_count": 0.0,
            "consume_count": 0.0,
        }
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            name=thread_name,
            daemon=True,
        )
        self._producer_thread.start()

    def __iter__(self) -> _ThreadPrefetchIterator[_T]:
        return self

    def _record(self, key: str, value: float) -> None:
        with self._metrics_lock:
            self._metrics[key] += value

    def _record_max(self, key: str, value: float) -> None:
        with self._metrics_lock:
            self._metrics[key] = max(self._metrics[key], value)

    def _publish(self, item: _T | _PrefetchFailure | _PrefetchEnd) -> None:
        if self._stop_event.is_set():
            return
        if self._consumed == 0:
            with self._metrics_lock:
                if self._metrics["first_item_ready_s"] == 0.0:
                    self._metrics["first_item_ready_s"] = (
                        time.perf_counter() - self._started_at
                    )
        if self._item_size is not None and not isinstance(
            item, (_PrefetchFailure, _PrefetchEnd)
        ):
            self._record_max("queued_payload_peak_bytes", float(self._item_size(item)))
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=_QUEUE_POLL_SECONDS)
                return
            except queue.Full:
                continue

    def _producer_loop(self) -> None:
        while not self._stop_event.is_set():
            self._producer_permit.acquire()
            if self._stop_event.is_set():
                return
            try:
                item = next(self._source_iterator)
            except StopIteration:
                self._publish(_PREFETCH_END)
                return
            except Exception as error:
                self._publish(
                    _PrefetchFailure(
                        str(error)[:_MAX_ERROR_LENGTH],
                        from_source=True,
                    )
                )
                return
            try:
                self._publish(item)
            except Exception as error:
                self._publish(
                    _PrefetchFailure(
                        str(error)[:_MAX_ERROR_LENGTH],
                        from_source=False,
                    )
                )
                return

    def _take(self) -> _T | _PrefetchFailure | _PrefetchEnd:
        start = time.perf_counter()
        try:
            item = self._queue.get_nowait()
            self._record("ready_count", 1.0)
        except queue.Empty:
            deadline = time.monotonic() + self._item_ready_timeout_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    item = _PrefetchFailure(
                        "prefetched item did not become ready within "
                        f"{self._item_ready_timeout_s:g}s; background work "
                        "may still be running",
                        from_source=False,
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
                            "prefetch producer exited without publishing a result",
                            from_source=False,
                        )
                        break
        self._record("consumer_wait_s", time.perf_counter() - start)
        return item

    def __next__(self) -> _T:
        if self._terminal_failure is not None:
            raise _ThreadPrefetchError(
                "prefetch iterator is terminal after failure: "
                f"{self._terminal_failure.message}",
                from_source=self._terminal_failure.from_source,
            )
        if self._ended:
            raise StopIteration
        if not self._lookahead and self._consumed > 0:
            self._producer_permit.release()

        item = self._take()
        if isinstance(item, _PrefetchFailure):
            self._terminal_failure = item
            self._stop_event.set()
            raise _ThreadPrefetchError(
                item.message,
                from_source=item.from_source,
            )
        if isinstance(item, _PrefetchEnd):
            self._ended = True
            raise StopIteration

        self._consumed += 1
        self._record("consume_count", 1.0)
        if self._lookahead:
            self._producer_permit.release()
        return item

    def metrics(self) -> dict[str, float]:
        with self._metrics_lock:
            return dict(self._metrics)

    def close(self) -> None:
        """Drain finite work or bound cleanup after a terminal timeout."""
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
            raise _ThreadPrefetchError(
                f"prefetch producer did not stop within {_CLOSE_TIMEOUT_SECONDS:.0f}s",
                from_source=False,
            )


class _TrainMicrobatchLoader(Iterator[PrefetchedMicrobatch]):
    """Synchronously load one planned microbatch from TQ."""

    def __init__(
        self,
        *,
        client: DataPlaneClient,
        meta: KVBatchMeta,
        plan: TrainMicrobatchPlan,
        items: Sequence[tuple[int, int, tuple[str, ...]]],
        pad_value_dict: dict[str, Any],
    ) -> None:
        self._client = client
        self._meta = meta
        self._plan = plan
        self._items = iter(items)
        self._pad_value_dict = pad_value_dict
        self._metrics_lock = threading.Lock()
        self._metrics: dict[str, float] = {
            "tq_get_s": 0.0,
            "materialize_s": 0.0,
            "tq_get_calls": 0.0,
            "materialized_payload_bytes": 0.0,
        }

    def __iter__(self) -> _TrainMicrobatchLoader:
        return self

    def _record(self, key: str, value: float) -> None:
        with self._metrics_lock:
            self._metrics[key] += value

    @staticmethod
    def _batch_error(
        gb_idx: int, mb_idx: int, message: str
    ) -> TrainMicrobatchPrefetchError:
        return TrainMicrobatchPrefetchError(
            "train microbatch prefetch failed for global batch "
            f"{gb_idx}, microbatch {mb_idx}: {message}"
        )

    def _load(
        self, gb_idx: int, mb_idx: int, sample_ids: tuple[str, ...]
    ) -> PrefetchedMicrobatch:
        try:
            start = time.perf_counter()
            with nsys_nvtx_range("train_prefetch/tq_get"):
                wire = self._client.get_samples(
                    sample_ids=list(sample_ids),
                    partition_id=self._meta.partition_id,
                    select_fields=list(self._plan.fields),
                )
            self._record("tq_get_s", time.perf_counter() - start)
            self._record("tq_get_calls", 1.0)

            start = time.perf_counter()
            with nsys_nvtx_range("train_prefetch/materialize"):
                data = materialize(
                    wire,
                    layout="padded",
                    pad_value_dict=self._pad_value_dict,
                    pad_to_seqlen=self._plan.pad_to_seqlen,
                )
            del wire
            if data.size != len(sample_ids):
                raise ValueError(
                    f"materialized {data.size} rows for {len(sample_ids)} keys"
                )
            missing = [field for field in self._plan.fields if field not in data]
            if missing:
                raise KeyError(f"materialized microbatch is missing fields {missing}")
            payload_bytes = float(_tensor_bytes(data))
            self._record("materialize_s", time.perf_counter() - start)
            self._record("materialized_payload_bytes", payload_bytes)
        except Exception as error:
            raise self._batch_error(
                gb_idx,
                mb_idx,
                f"{type(error).__name__}: {error}"[:_MAX_ERROR_LENGTH],
            ) from error
        return PrefetchedMicrobatch(gb_idx, mb_idx, sample_ids, data)

    def __next__(self) -> PrefetchedMicrobatch:
        gb_idx, mb_idx, sample_ids = next(self._items)
        return self._load(gb_idx, mb_idx, sample_ids)

    def metrics(self) -> dict[str, float]:
        with self._metrics_lock:
            return dict(self._metrics)


class TrainMicrobatchPrefetcher(Iterator[PrefetchedMicrobatch]):
    """Prefetch on each stage leader and fan out through foreground NCCL."""

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
        item_ready_timeout_s = _validate_timeout_seconds(
            item_ready_timeout_s,
            name="policy.train_microbatch_prefetch.item_ready_timeout_s",
        )
        self.plan = build_train_microbatch_plan(meta)
        self._items = tuple(
            (batch.global_batch_index, mb_idx, sample_ids)
            for batch in self.plan.global_batches
            for mb_idx, sample_ids in enumerate(batch.microbatch_sample_ids)
        )
        self._group = group
        self._loader: _TrainMicrobatchLoader | None = None
        self._prefetch_iterator: (
            _ThreadPrefetchIterator[PrefetchedMicrobatch] | None
        ) = None
        if group.is_stage_leader:
            self._loader = _TrainMicrobatchLoader(
                client=client,
                meta=meta,
                plan=self.plan,
                items=self._items,
                pad_value_dict=pad_value_dict,
            )
            self._prefetch_iterator = _ThreadPrefetchIterator(
                self._loader,
                lookahead=depth == 1,
                item_ready_timeout_s=item_ready_timeout_s,
                thread_name=(
                    f"train-microbatch-prefetch-rank-{torch.distributed.get_rank()}"
                ),
                item_size=lambda item: _tensor_bytes(item.data),
            )
        self._consumed = 0
        self._metrics_lock = threading.Lock()
        self._foreground_distribute_s = 0.0

    def __iter__(self) -> TrainMicrobatchPrefetcher:
        return self

    @staticmethod
    def _batch_error(
        gb_idx: int, mb_idx: int, message: str
    ) -> TrainMicrobatchPrefetchError:
        return TrainMicrobatchPrefetchError(
            "train microbatch prefetch failed for global batch "
            f"{gb_idx}, microbatch {mb_idx}: {message}"
        )

    def _take_stage_leader_item(
        self,
    ) -> PrefetchedMicrobatch | _PrefetchFailure:
        if self._prefetch_iterator is None:
            return _PrefetchFailure(
                "stage leader has no background TQ prefetch iterator",
                from_source=False,
            )
        try:
            with nsys_nvtx_range("train_prefetch/consumer_wait"):
                return next(self._prefetch_iterator)
        except Exception as error:
            return _PrefetchFailure(
                f"{type(error).__name__}: {error}"[:_MAX_ERROR_LENGTH],
                from_source=(
                    error.from_source
                    if isinstance(error, _ThreadPrefetchError)
                    else False
                ),
            )

    def _stage_fanout(
        self,
        item: PrefetchedMicrobatch | _PrefetchFailure | None,
        *,
        expected_gb: int,
        expected_mb: int,
        expected_ids: tuple[str, ...],
    ) -> PrefetchedMicrobatch:
        """Broadcast one PP-stage payload from TP0/CP0 in the foreground."""
        start = time.perf_counter()
        with nsys_nvtx_range("train_prefetch/foreground_distribute"):
            if self._group.is_stage_leader:
                if isinstance(item, PrefetchedMicrobatch):
                    try:
                        item = PrefetchedMicrobatch(
                            item.global_batch_index,
                            item.microbatch_index,
                            item.sample_ids,
                            _move_stage_payload_to_cuda(item.data),
                        )
                    except Exception as error:
                        item = _PrefetchFailure(
                            "stage-leader GPU staging failed: "
                            f"{type(error).__name__}: {error}"[:_MAX_ERROR_LENGTH],
                            from_source=False,
                        )
                if isinstance(item, _PrefetchFailure):
                    envelope: list[Any] = [
                        (
                            "error",
                            expected_gb,
                            expected_mb,
                            item.message,
                        )
                    ]
                elif isinstance(item, PrefetchedMicrobatch):
                    envelope = [
                        (
                            "ok",
                            item.global_batch_index,
                            item.microbatch_index,
                            item.sample_ids,
                        )
                    ]
                else:
                    envelope = [
                        (
                            "error",
                            expected_gb,
                            expected_mb,
                            "stage leader has no prefetched payload",
                        )
                    ]
            else:
                envelope = [None]

            torch.distributed.broadcast_object_list(
                envelope,
                src=self._group.stage_source_rank,
                group=self._group.stage_group,
            )
            status = envelope[0]
            if not isinstance(status, tuple) or not status:
                raise self._batch_error(
                    expected_gb,
                    expected_mb,
                    f"invalid stage-prefetch envelope: {status!r}",
                )
            if status[0] == "error":
                raise self._batch_error(
                    int(status[1]),
                    int(status[2]),
                    str(status[3]),
                )

            data = _broadcast_batched_data_dict(
                item.data if isinstance(item, PrefetchedMicrobatch) else None,
                is_leader=self._group.is_stage_leader,
                src=self._group.stage_source_rank,
                group=self._group.stage_group,
                keep_on_broadcast_device=True,
            )
            # Complete the shared payload collective before evaluating local
            # metadata. A rank-local early exit would strand its stage peers.
            expected_status = ("ok", expected_gb, expected_mb, expected_ids)
            if status != expected_status:
                raise self._batch_error(
                    expected_gb,
                    expected_mb,
                    "stage-prefetch order mismatch: expected "
                    f"{expected_status}, received {status}",
                )
        with self._metrics_lock:
            self._foreground_distribute_s += time.perf_counter() - start
        return PrefetchedMicrobatch(expected_gb, expected_mb, expected_ids, data)

    def __next__(self) -> PrefetchedMicrobatch:
        if self._consumed >= len(self._items):
            raise StopIteration
        expected_gb, expected_mb, expected_ids = self._items[self._consumed]
        leader_item: PrefetchedMicrobatch | _PrefetchFailure | None = None
        if self._group.is_stage_leader:
            leader_item = self._take_stage_leader_item()
        item = self._stage_fanout(
            leader_item,
            expected_gb=expected_gb,
            expected_mb=expected_mb,
            expected_ids=expected_ids,
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
        result = (
            self._loader.metrics()
            if self._loader is not None
            else {
                "tq_get_s": 0.0,
                "materialize_s": 0.0,
                "tq_get_calls": 0.0,
                "materialized_payload_bytes": 0.0,
            }
        )
        iterator_metrics = (
            self._prefetch_iterator.metrics()
            if self._prefetch_iterator is not None
            else {
                "consumer_wait_s": 0.0,
                "first_item_ready_s": 0.0,
                "queued_payload_peak_bytes": 0.0,
                "ready_count": 0.0,
                "consume_count": 0.0,
            }
        )
        with self._metrics_lock:
            foreground_distribute_s = self._foreground_distribute_s
        result.update(
            {
                "foreground_distribute_s": foreground_distribute_s,
                "consumer_wait_s": iterator_metrics["consumer_wait_s"],
                "first_microbatch_ready_s": iterator_metrics["first_item_ready_s"],
                "queued_payload_peak_bytes": iterator_metrics[
                    "queued_payload_peak_bytes"
                ],
                "ready_count": iterator_metrics["ready_count"],
                "consume_count": iterator_metrics["consume_count"],
            }
        )
        consumed = result["consume_count"]
        result["ready_fraction"] = result["ready_count"] / consumed if consumed else 0.0
        return result

    def close(self) -> None:
        if self._prefetch_iterator is None:
            return
        try:
            self._prefetch_iterator.close()
        except _ThreadPrefetchError as error:
            raise TrainMicrobatchPrefetchError(str(error)) from error
