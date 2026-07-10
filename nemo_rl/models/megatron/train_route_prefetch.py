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
"""Depth-one TransferQueue prefetch for train-time router replay.

The pipeline schedule consumes data iterators at different times on different
pipeline stages. Consequently, route delivery must never use a blocking
TPxCPxPP collective from ``next()``. This module uses two independent hops:

* a background Gloo broadcast between the TP=CP=0 leader of each PP stage;
* a main-thread NCCL broadcast within the consuming stage's TPxCP group.

Only the PP=0 leader reads TransferQueue, so a route is retrieved once per DP
replica. The producer queue has capacity one, which allows the next packed
microbatch to overlap the current microbatch without retaining a full DP
shard of routes.
"""

from __future__ import annotations

import queue
import threading
import time
from datetime import timedelta
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
from megatron.core import parallel_state

from nemo_rl.data_plane import DataPlaneClient, KVBatchMeta, materialize
from nemo_rl.data_plane.schema import (
    ELEM_COUNTS_PER_GB,
    GLOBAL_FORWARD_PAD_SEQLEN,
    MICRO_BATCH_INDICES,
    ROUTED_EXPERTS_FIELD,
)

TQ_SAMPLE_IDS_FIELD = "__tq_sample_ids"

_STATUS_OK = 0
_STATUS_ERROR = 1
_HEADER_LENGTH = 8
_MAX_ERROR_BYTES = 4096
_QUEUE_POLL_SECONDS = 0.1
_COLLECTIVE_TIMEOUT_SECONDS = 120.0
_CLOSE_TIMEOUT_SECONDS = _COLLECTIVE_TIMEOUT_SECONDS + 10.0

_DTYPE_TO_CODE = {
    torch.int8: 0,
    torch.int16: 1,
    torch.int32: 2,
    torch.int64: 3,
}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}


class TrainRoutePrefetchError(RuntimeError):
    """Failure raised consistently by every rank consuming prefetched routes."""


@dataclass(frozen=True)
class RankCoordinates:
    """Global rank and its data/model-parallel coordinates."""

    global_rank: int
    dp_rank: int
    pp_rank: int
    tp_rank: int
    cp_rank: int


@dataclass(frozen=True)
class TrainRoutePrefetchGroups:
    """Process groups and sources used by train-route delivery."""

    pp_leader_group: Any
    pp_leader_ranks: tuple[int, ...]
    pp_source_rank: int
    stage_group: Any
    stage_source_rank: int
    is_stage_leader: bool
    is_pp_source: bool


@dataclass(frozen=True)
class PrefetchedRoutes:
    """One packed microbatch's row identities and padded route tensor."""

    microbatch_index: int
    sample_ids: tuple[str, ...]
    routed_experts: torch.Tensor


@dataclass(frozen=True)
class _PrefetchFailure:
    microbatch_index: int
    message: str


def build_pp_leader_topology(
    coordinates: Sequence[RankCoordinates],
) -> dict[int, tuple[RankCoordinates, ...]]:
    """Return PP-stage leaders grouped by DP rank and ordered by PP rank.

    This is deliberately independent of global-rank layout. In particular,
    the PP=0 source need not be the lowest global rank.
    """
    if not coordinates:
        raise ValueError("route-prefetch topology cannot be empty")

    dp_ranks = sorted({coord.dp_rank for coord in coordinates})
    leaders_by_dp: dict[int, tuple[RankCoordinates, ...]] = {}
    for dp_rank in dp_ranks:
        leaders = tuple(
            sorted(
                (
                    coord
                    for coord in coordinates
                    if coord.dp_rank == dp_rank
                    and coord.tp_rank == 0
                    and coord.cp_rank == 0
                ),
                key=lambda coord: (coord.pp_rank, coord.global_rank),
            )
        )
        pp_ranks = [coord.pp_rank for coord in leaders]
        expected_pp_ranks = list(range(max(pp_ranks, default=-1) + 1))
        if pp_ranks != expected_pp_ranks:
            raise ValueError(
                "route-prefetch topology requires exactly one TP=CP=0 leader "
                f"for every PP rank in DP {dp_rank}; got PP ranks {pp_ranks}"
            )
        leaders_by_dp[dp_rank] = leaders
    return leaders_by_dp


def build_route_key_batches(meta: KVBatchMeta) -> tuple[tuple[str, ...], ...]:
    """Translate driver packing metadata into ordered microbatch key lists."""
    extra = meta.extra_info or {}
    if MICRO_BATCH_INDICES not in extra:
        raise ValueError(
            "train route prefetch currently requires sequence-packing metadata "
            f"{MICRO_BATCH_INDICES!r}"
        )

    indices_per_gb = extra[MICRO_BATCH_INDICES]
    if not isinstance(indices_per_gb, (list, tuple)) or not indices_per_gb:
        raise ValueError(f"invalid {MICRO_BATCH_INDICES}: {indices_per_gb!r}")

    elem_counts = extra.get(ELEM_COUNTS_PER_GB)
    if elem_counts is None:
        if len(indices_per_gb) != 1:
            raise ValueError(
                f"{ELEM_COUNTS_PER_GB!r} is required when packing metadata "
                "contains multiple global batches"
            )
        elem_counts = [len(meta.sample_ids)]
    if len(elem_counts) != len(indices_per_gb):
        raise ValueError(
            f"{ELEM_COUNTS_PER_GB} has {len(elem_counts)} entries but "
            f"{MICRO_BATCH_INDICES} has {len(indices_per_gb)}"
        )

    counts = [int(count) for count in elem_counts]
    if any(count < 0 for count in counts) or sum(counts) != len(meta.sample_ids):
        raise ValueError(
            f"invalid {ELEM_COUNTS_PER_GB}={counts}; expected non-negative "
            f"counts summing to {len(meta.sample_ids)}"
        )

    key_batches: list[tuple[str, ...]] = []
    offset = 0
    for global_batch_index, (microbatch_ranges, elem_count) in enumerate(
        zip(indices_per_gb, counts, strict=True)
    ):
        if not microbatch_ranges:
            raise ValueError(
                f"train route prefetch received no packed microbatches for "
                f"global batch {global_batch_index}"
            )
        expected_start = 0
        for microbatch_range in microbatch_ranges:
            if (
                not isinstance(microbatch_range, (list, tuple))
                or len(microbatch_range) != 2
            ):
                raise ValueError(
                    f"invalid packed range in global batch {global_batch_index}: "
                    f"{microbatch_range!r}"
                )
            start, end = (int(microbatch_range[0]), int(microbatch_range[1]))
            if start < 0 or end <= start or end > elem_count:
                raise ValueError(
                    f"packed range [{start}, {end}) is outside global batch "
                    f"{global_batch_index} with {elem_count} local samples"
                )
            if start != expected_start:
                raise ValueError(
                    "packed ranges must cover each global batch exactly once "
                    f"in row order; global batch {global_batch_index} expected "
                    f"start {expected_start}, got {start}"
                )
            key_batches.append(tuple(meta.sample_ids[offset + start : offset + end]))
            expected_start = end
        if expected_start != elem_count:
            raise ValueError(
                "packed ranges must cover each global batch exactly once; "
                f"global batch {global_batch_index} covers {expected_start} of "
                f"{elem_count} rows"
            )
        offset += elem_count

    if not key_batches:
        raise ValueError("train route prefetch received no packed microbatches")
    return tuple(key_batches)


def initialize_train_route_prefetch_groups() -> TrainRoutePrefetchGroups:
    """Collect topology and create PP-leader Gloo groups on every world rank."""
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "train route prefetch requires initialized torch.distributed"
        )

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
    coordinates = []
    for rank, row in enumerate(gathered):
        dp_rank, pp_rank, tp_rank, cp_rank = [int(value) for value in row.tolist()]
        coordinates.append(
            RankCoordinates(
                global_rank=rank,
                dp_rank=dp_rank,
                pp_rank=pp_rank,
                tp_rank=tp_rank,
                cp_rank=cp_rank,
            )
        )
    leaders_by_dp = build_pp_leader_topology(coordinates)

    my_rank = torch.distributed.get_rank()
    my_dp_rank = parallel_state.get_data_parallel_rank()
    my_pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    my_tp_rank = parallel_state.get_tensor_model_parallel_rank()
    my_cp_rank = parallel_state.get_context_parallel_rank()

    pp_groups: dict[int, Any] = {}
    for dp_rank in sorted(leaders_by_dp):
        leader_ranks = [coord.global_rank for coord in leaders_by_dp[dp_rank]]
        pp_groups[dp_rank] = torch.distributed.new_group(
            ranks=leader_ranks,
            backend="gloo",
            timeout=timedelta(seconds=_COLLECTIVE_TIMEOUT_SECONDS),
        )

    my_leaders = leaders_by_dp[my_dp_rank]
    pp_world_size = parallel_state.get_pipeline_model_parallel_world_size()
    if len(my_leaders) != pp_world_size:
        raise RuntimeError(
            f"expected {pp_world_size} PP-stage leaders for DP {my_dp_rank}, "
            f"got {len(my_leaders)}"
        )
    pp_zero_leaders = [coord for coord in my_leaders if coord.pp_rank == 0]
    if len(pp_zero_leaders) != 1:
        raise RuntimeError(
            f"expected one PP=0 route source for DP {my_dp_rank}, got "
            f"{len(pp_zero_leaders)}"
        )
    pp_source_rank = pp_zero_leaders[0].global_rank

    stage_sources = [
        coord
        for coord in coordinates
        if coord.dp_rank == my_dp_rank
        and coord.pp_rank == my_pp_rank
        and coord.tp_rank == 0
        and coord.cp_rank == 0
    ]
    if len(stage_sources) != 1:
        raise RuntimeError(
            "expected one TP=CP=0 route source in this pipeline stage, got "
            f"{len(stage_sources)}"
        )
    stage_source_rank = stage_sources[0].global_rank
    stage_group = parallel_state.get_tensor_and_context_parallel_group()
    stage_ranks = torch.distributed.get_process_group_ranks(stage_group)
    if stage_source_rank not in stage_ranks:
        raise RuntimeError(
            f"stage route source rank {stage_source_rank} is not in TPxCP group "
            f"{stage_ranks}"
        )

    is_stage_leader = my_tp_rank == 0 and my_cp_rank == 0
    return TrainRoutePrefetchGroups(
        pp_leader_group=pp_groups[my_dp_rank],
        pp_leader_ranks=tuple(coord.global_rank for coord in my_leaders),
        pp_source_rank=pp_source_rank,
        stage_group=stage_group,
        stage_source_rank=stage_source_rank,
        is_stage_leader=is_stage_leader,
        is_pp_source=my_rank == pp_source_rank,
    )


def _error_bytes(message: str, *, device: Any) -> torch.Tensor:
    encoded = message.encode("utf-8", errors="replace")[:_MAX_ERROR_BYTES]
    return torch.tensor(list(encoded), dtype=torch.uint8, device=device)


def _decode_error_bytes(payload: torch.Tensor) -> str:
    return bytes(payload.cpu().tolist()).decode("utf-8", errors="replace")


def _ok_header(
    *, microbatch_index: int, routed_experts: torch.Tensor, device: Any
) -> torch.Tensor:
    if routed_experts.dim() != 4:
        raise ValueError(
            "prefetched routed_experts must have shape [B, S, L, K], got "
            f"{tuple(routed_experts.shape)}"
        )
    if routed_experts.dtype not in _DTYPE_TO_CODE:
        raise TypeError(
            "prefetched routed_experts must use int8, int16, int32, or int64, got "
            f"{routed_experts.dtype}"
        )
    return torch.tensor(
        [
            _STATUS_OK,
            microbatch_index,
            _DTYPE_TO_CODE[routed_experts.dtype],
            *routed_experts.shape,
            0,
        ],
        dtype=torch.long,
        device=device,
    )


def _error_header(
    *, microbatch_index: int, error_length: int, device: Any
) -> torch.Tensor:
    return torch.tensor(
        [_STATUS_ERROR, microbatch_index, 0, 0, 0, 0, 0, error_length],
        dtype=torch.long,
        device=device,
    )


class TrainRoutePrefetcher(Iterator[PrefetchedRoutes]):
    """Fetch and distribute one DP replica's routes with depth-one lookahead."""

    def __init__(
        self,
        *,
        client: DataPlaneClient,
        meta: KVBatchMeta,
        groups: TrainRoutePrefetchGroups,
    ) -> None:
        self._client = client
        self._meta = meta
        self._groups = groups
        self._key_batches = build_route_key_batches(meta)
        stage_backend = torch.distributed.get_backend(groups.stage_group)
        self._stage_device: Any = (
            torch.device("cuda", torch.cuda.current_device())
            if stage_backend == "nccl"
            else torch.device("cpu")
        )
        self._pad_to_seqlen = int(
            (meta.extra_info or {}).get(GLOBAL_FORWARD_PAD_SEQLEN, 0)
        )
        if self._pad_to_seqlen <= 0:
            raise ValueError(
                "train route prefetch requires a positive "
                f"{GLOBAL_FORWARD_PAD_SEQLEN!r}"
            )

        self._queue: queue.Queue[PrefetchedRoutes | _PrefetchFailure] = queue.Queue(
            maxsize=1
        )
        self._prefetch_permit = threading.Semaphore(1)
        self._consumed = 0
        self._closed = False
        self._metrics_lock = threading.Lock()
        self._metrics: dict[str, float] = {
            "tq_get_s": 0.0,
            "materialize_s": 0.0,
            "pp_leader_broadcast_s": 0.0,
            "h2d_s": 0.0,
            "consumer_wait_s": 0.0,
            "stage_broadcast_s": 0.0,
            "tq_get_calls": 0.0,
            "materialized_route_bytes": 0.0,
            "ready_count": 0.0,
            "consume_count": 0.0,
        }
        self._producer_thread: threading.Thread | None = None
        if groups.is_stage_leader:
            target = (
                self._source_producer_loop
                if groups.is_pp_source
                else self._receiver_producer_loop
            )
            self._producer_thread = threading.Thread(
                target=target,
                name=f"train-route-prefetch-rank-{torch.distributed.get_rank()}",
                daemon=True,
            )
            self._producer_thread.start()

    def __iter__(self) -> TrainRoutePrefetcher:
        return self

    def _record(self, key: str, value: float) -> None:
        with self._metrics_lock:
            self._metrics[key] += value

    def _put(self, item: PrefetchedRoutes | _PrefetchFailure) -> None:
        self._queue.put(item)

    def _broadcast_pp_error(self, *, microbatch_index: int, message: str) -> None:
        if len(self._groups.pp_leader_ranks) == 1:
            return
        error_payload = _error_bytes(message, device="cpu")
        header = _error_header(
            microbatch_index=microbatch_index,
            error_length=error_payload.numel(),
            device="cpu",
        )
        torch.distributed.broadcast(
            header,
            src=self._groups.pp_source_rank,
            group=self._groups.pp_leader_group,
        )
        if error_payload.numel() > 0:
            torch.distributed.broadcast(
                error_payload,
                src=self._groups.pp_source_rank,
                group=self._groups.pp_leader_group,
            )

    def _source_producer_loop(self) -> None:
        for microbatch_index, sample_ids in enumerate(self._key_batches):
            self._prefetch_permit.acquire()
            try:
                start = time.perf_counter()
                wire = self._client.get_samples(
                    sample_ids=list(sample_ids),
                    partition_id=self._meta.partition_id,
                    select_fields=[ROUTED_EXPERTS_FIELD],
                )
                self._record("tq_get_s", time.perf_counter() - start)
                self._record("tq_get_calls", 1.0)

                start = time.perf_counter()
                materialized = materialize(
                    wire,
                    layout="padded",
                    pad_to_seqlen=self._pad_to_seqlen,
                )
                routed_experts = materialized[ROUTED_EXPERTS_FIELD]
                if not isinstance(routed_experts, torch.Tensor):
                    raise TypeError(
                        "materialized routed_experts must be a torch.Tensor, got "
                        f"{type(routed_experts).__name__}"
                    )
                routed_experts = routed_experts.contiguous().cpu()
                header = _ok_header(
                    microbatch_index=microbatch_index,
                    routed_experts=routed_experts,
                    device="cpu",
                )
                self._record("materialize_s", time.perf_counter() - start)
                self._record(
                    "materialized_route_bytes",
                    float(routed_experts.numel() * routed_experts.element_size()),
                )
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                try:
                    self._broadcast_pp_error(
                        microbatch_index=microbatch_index,
                        message=message,
                    )
                except Exception as broadcast_error:
                    message += (
                        "; PP-leader error broadcast also failed: "
                        f"{type(broadcast_error).__name__}: {broadcast_error}"
                    )
                self._put(_PrefetchFailure(microbatch_index, message))
                return

            try:
                if len(self._groups.pp_leader_ranks) > 1:
                    start = time.perf_counter()
                    torch.distributed.broadcast(
                        header,
                        src=self._groups.pp_source_rank,
                        group=self._groups.pp_leader_group,
                    )
                    torch.distributed.broadcast(
                        routed_experts,
                        src=self._groups.pp_source_rank,
                        group=self._groups.pp_leader_group,
                    )
                    self._record("pp_leader_broadcast_s", time.perf_counter() - start)
                self._put(
                    PrefetchedRoutes(
                        microbatch_index=microbatch_index,
                        sample_ids=sample_ids,
                        routed_experts=routed_experts,
                    )
                )
            except Exception as error:
                self._put(
                    _PrefetchFailure(
                        microbatch_index,
                        f"{type(error).__name__}: {error}",
                    )
                )
                return

    def _receiver_producer_loop(self) -> None:
        for expected_index, sample_ids in enumerate(self._key_batches):
            self._prefetch_permit.acquire()
            try:
                header = torch.empty(_HEADER_LENGTH, dtype=torch.long, device="cpu")
                start = time.perf_counter()
                torch.distributed.broadcast(
                    header,
                    src=self._groups.pp_source_rank,
                    group=self._groups.pp_leader_group,
                )
                values = [int(value) for value in header.tolist()]
                (
                    status,
                    microbatch_index,
                    dtype_code,
                    b,
                    s,
                    layers,
                    topk,
                    error_len,
                ) = values
                if status == _STATUS_ERROR:
                    error_payload = torch.empty(
                        error_len, dtype=torch.uint8, device="cpu"
                    )
                    if error_len > 0:
                        torch.distributed.broadcast(
                            error_payload,
                            src=self._groups.pp_source_rank,
                            group=self._groups.pp_leader_group,
                        )
                    message = _decode_error_bytes(error_payload)
                    self._record("pp_leader_broadcast_s", time.perf_counter() - start)
                    self._put(_PrefetchFailure(microbatch_index, message))
                    return
                if status != _STATUS_OK or dtype_code not in _CODE_TO_DTYPE:
                    raise TrainRoutePrefetchError(
                        f"invalid PP-leader route header: {values}"
                    )

                routed_experts = torch.empty(
                    (b, s, layers, topk),
                    dtype=_CODE_TO_DTYPE[dtype_code],
                    device="cpu",
                )
                torch.distributed.broadcast(
                    routed_experts,
                    src=self._groups.pp_source_rank,
                    group=self._groups.pp_leader_group,
                )
                self._record("pp_leader_broadcast_s", time.perf_counter() - start)
                if microbatch_index != expected_index:
                    raise TrainRoutePrefetchError(
                        "PP-leader route order mismatch: expected "
                        f"{expected_index}, received {microbatch_index}"
                    )
                self._put(
                    PrefetchedRoutes(
                        microbatch_index=microbatch_index,
                        sample_ids=sample_ids,
                        routed_experts=routed_experts,
                    )
                )
            except Exception as error:
                self._put(
                    _PrefetchFailure(
                        expected_index,
                        f"{type(error).__name__}: {error}",
                    )
                )
                return

    def _take_stage_leader_item(self) -> PrefetchedRoutes | _PrefetchFailure:
        start = time.perf_counter()
        try:
            item = self._queue.get_nowait()
            self._record("ready_count", 1.0)
        except queue.Empty:
            while True:
                try:
                    item = self._queue.get(timeout=_QUEUE_POLL_SECONDS)
                    break
                except queue.Empty:
                    producer = self._producer_thread
                    if producer is not None and not producer.is_alive():
                        item = _PrefetchFailure(
                            self._consumed,
                            "route prefetch producer exited without publishing "
                            "a payload or error",
                        )
                        break
        self._prefetch_permit.release()
        self._record("consumer_wait_s", time.perf_counter() - start)
        return item

    def __next__(self) -> PrefetchedRoutes:
        if self._consumed >= len(self._key_batches):
            raise StopIteration
        expected_index = self._consumed
        expected_sample_ids = self._key_batches[expected_index]
        stage_group_size = torch.distributed.get_world_size(self._groups.stage_group)

        item: PrefetchedRoutes | _PrefetchFailure | None = None
        if self._groups.is_stage_leader:
            item = self._take_stage_leader_item()
            if isinstance(item, _PrefetchFailure):
                error_payload = _error_bytes(item.message, device=self._stage_device)
                header = _error_header(
                    microbatch_index=item.microbatch_index,
                    error_length=error_payload.numel(),
                    device=self._stage_device,
                )
                routed_experts_device = None
            else:
                try:
                    start = time.perf_counter()
                    routed_experts_device = item.routed_experts.to(self._stage_device)
                    self._record("h2d_s", time.perf_counter() - start)
                    header = _ok_header(
                        microbatch_index=item.microbatch_index,
                        routed_experts=routed_experts_device,
                        device=self._stage_device,
                    )
                    error_payload = None
                except Exception as error:
                    message = f"{type(error).__name__}: {error}"
                    error_payload = _error_bytes(message, device=self._stage_device)
                    header = _error_header(
                        microbatch_index=expected_index,
                        error_length=error_payload.numel(),
                        device=self._stage_device,
                    )
                    routed_experts_device = None
        else:
            header = torch.empty(
                _HEADER_LENGTH, dtype=torch.long, device=self._stage_device
            )
            error_payload = None
            routed_experts_device = None

        start = time.perf_counter()
        if stage_group_size > 1:
            torch.distributed.broadcast(
                header,
                src=self._groups.stage_source_rank,
                group=self._groups.stage_group,
            )
        values = [int(value) for value in header.tolist()]
        status, microbatch_index, dtype_code, b, s, layers, topk, error_len = values

        if status == _STATUS_ERROR:
            if not self._groups.is_stage_leader:
                error_payload = torch.empty(
                    error_len, dtype=torch.uint8, device=self._stage_device
                )
            assert error_payload is not None
            if stage_group_size > 1 and error_len > 0:
                torch.distributed.broadcast(
                    error_payload,
                    src=self._groups.stage_source_rank,
                    group=self._groups.stage_group,
                )
            self._record("stage_broadcast_s", time.perf_counter() - start)
            raise TrainRoutePrefetchError(
                "train route prefetch failed for microbatch "
                f"{microbatch_index}: {_decode_error_bytes(error_payload)}"
            )
        if status != _STATUS_OK or dtype_code not in _CODE_TO_DTYPE:
            raise TrainRoutePrefetchError(f"invalid stage route header: {values}")

        if not self._groups.is_stage_leader:
            routed_experts_device = torch.empty(
                (b, s, layers, topk),
                dtype=_CODE_TO_DTYPE[dtype_code],
                device=self._stage_device,
            )
        assert routed_experts_device is not None
        if stage_group_size > 1:
            torch.distributed.broadcast(
                routed_experts_device,
                src=self._groups.stage_source_rank,
                group=self._groups.stage_group,
            )
        self._record("stage_broadcast_s", time.perf_counter() - start)

        if microbatch_index != expected_index:
            raise TrainRoutePrefetchError(
                "stage route order mismatch: expected "
                f"{expected_index}, received {microbatch_index}"
            )
        if routed_experts_device.shape[0] != len(expected_sample_ids):
            raise TrainRoutePrefetchError(
                "prefetched route batch dimension does not match packed "
                f"microbatch {expected_index}: routes={routed_experts_device.shape[0]}, "
                f"samples={len(expected_sample_ids)}"
            )

        self._consumed += 1
        self._record("consume_count", 1.0)
        return PrefetchedRoutes(
            microbatch_index=microbatch_index,
            sample_ids=expected_sample_ids,
            routed_experts=routed_experts_device,
        )

    def assert_complete(self) -> None:
        """Require the Megatron schedule to consume exactly the planned routes."""
        if self._consumed != len(self._key_batches):
            raise TrainRoutePrefetchError(
                "Megatron consumed an unexpected number of route microbatches: "
                f"consumed={self._consumed}, planned={len(self._key_batches)}"
            )

    def metrics(self) -> dict[str, float]:
        """Return step-aggregated development metrics for this rank."""
        with self._metrics_lock:
            result = dict(self._metrics)
        consume_count = result["consume_count"]
        result["ready_fraction"] = (
            result["ready_count"] / consume_count if consume_count else 0.0
        )
        return result

    def close(self) -> None:
        """Drain and join the producer before TQ samples may be cleared."""
        if self._closed:
            return
        self._closed = True
        if self._producer_thread is not None:
            deadline = time.monotonic() + _CLOSE_TIMEOUT_SECONDS
            while self._producer_thread.is_alive() and time.monotonic() < deadline:
                try:
                    self._queue.get(timeout=_QUEUE_POLL_SECONDS)
                    self._prefetch_permit.release()
                except queue.Empty:
                    continue
            remaining = max(0.0, deadline - time.monotonic())
            self._producer_thread.join(timeout=remaining)
            if self._producer_thread.is_alive():
                raise TrainRoutePrefetchError(
                    "train route prefetch producer did not stop within "
                    f"{_CLOSE_TIMEOUT_SECONDS:.0f}s"
                )
