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
"""CPU-only contracts for train-time routed-expert prefetch."""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import (
    ELEM_COUNTS_PER_GB,
    GLOBAL_FORWARD_PAD_SEQLEN,
    MICRO_BATCH_INDICES,
    ROUTED_EXPERTS_FIELD,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

pytestmark = pytest.mark.mcore


@pytest.fixture
def route_prefetch_module():
    from nemo_rl.models.megatron import train_route_prefetch

    return train_route_prefetch


@pytest.fixture
def megatron_data_module():
    from nemo_rl.models.megatron import data

    return data


def _meta(
    sample_ids: list[str],
    *,
    indices: Any,
    elem_counts: list[int] | None = None,
    include_elem_counts: bool = False,
    pad_to_seqlen: int | None = None,
) -> KVBatchMeta:
    extra_info = {MICRO_BATCH_INDICES: indices}
    if include_elem_counts:
        extra_info[ELEM_COUNTS_PER_GB] = elem_counts
    if pad_to_seqlen is not None:
        extra_info[GLOBAL_FORWARD_PAD_SEQLEN] = pad_to_seqlen
    return KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=sample_ids,
        fields=[ROUTED_EXPERTS_FIELD],
        sequence_lengths=[pad_to_seqlen or 1] * len(sample_ids),
        extra_info=extra_info,
    )


def _raw_microbatch(
    sample_ids: list[str],
    *,
    sequence_length: int = 4,
) -> BatchedDataDict:
    from nemo_rl.models.megatron.train_route_prefetch import TQ_SAMPLE_IDS_FIELD

    return BatchedDataDict(
        {
            "input_ids": torch.arange(
                len(sample_ids) * sequence_length,
                dtype=torch.long,
            ).reshape(len(sample_ids), sequence_length),
            TQ_SAMPLE_IDS_FIELD: sample_ids,
        }
    )


def test_build_route_key_batches_single_global_batch(route_prefetch_module) -> None:
    meta = _meta(
        ["A", "B", "C", "D", "E"],
        indices=[[[0, 2], [2, 3], [3, 5]]],
    )

    assert route_prefetch_module.build_route_key_batches(meta) == (
        ("A", "B"),
        ("C",),
        ("D", "E"),
    )


def test_build_route_key_batches_offsets_each_global_batch(
    route_prefetch_module,
) -> None:
    meta = _meta(
        ["A", "B", "C", "D", "E", "F", "G", "H"],
        indices=[
            [[0, 2], [2, 3]],
            [[0, 1], [1, 4], [4, 5]],
        ],
        elem_counts=[3, 5],
        include_elem_counts=True,
    )

    assert route_prefetch_module.build_route_key_batches(meta) == (
        ("A", "B"),
        ("C",),
        ("D",),
        ("E", "F", "G"),
        ("H",),
    )


@pytest.mark.parametrize(
    ("extra_info", "match"),
    [
        pytest.param({}, MICRO_BATCH_INDICES, id="missing-indices"),
        pytest.param(
            {MICRO_BATCH_INDICES: []},
            f"invalid {MICRO_BATCH_INDICES}",
            id="empty-indices",
        ),
        pytest.param(
            {MICRO_BATCH_INDICES: [[[0, 2]], [[0, 3]]]},
            ELEM_COUNTS_PER_GB,
            id="multiple-global-batches-need-counts",
        ),
        pytest.param(
            {
                MICRO_BATCH_INDICES: [[[0, 2]], [[0, 3]]],
                ELEM_COUNTS_PER_GB: [5],
            },
            "has 1 entries",
            id="count-cardinality",
        ),
        pytest.param(
            {
                MICRO_BATCH_INDICES: [[[0, 2]], [[0, 3]]],
                ELEM_COUNTS_PER_GB: [-1, 6],
            },
            f"invalid {ELEM_COUNTS_PER_GB}",
            id="negative-count",
        ),
        pytest.param(
            {
                MICRO_BATCH_INDICES: [[[0, 2]], [[0, 2]]],
                ELEM_COUNTS_PER_GB: [2, 2],
            },
            f"invalid {ELEM_COUNTS_PER_GB}",
            id="counts-do-not-cover-keys",
        ),
        pytest.param(
            {MICRO_BATCH_INDICES: [[[0]]]},
            "invalid packed range",
            id="malformed-range",
        ),
        pytest.param(
            {MICRO_BATCH_INDICES: [[[2, 2]]]},
            "outside global batch",
            id="empty-range",
        ),
        pytest.param(
            {MICRO_BATCH_INDICES: [[[0, 6]]]},
            "outside global batch",
            id="range-past-global-batch",
        ),
        pytest.param(
            {MICRO_BATCH_INDICES: [[[0, 2], [1, 5]]]},
            "expected start 2, got 1",
            id="overlapping-ranges",
        ),
        pytest.param(
            {MICRO_BATCH_INDICES: [[[0, 2], [3, 5]]]},
            "expected start 2, got 3",
            id="gapped-ranges",
        ),
        pytest.param(
            {MICRO_BATCH_INDICES: [[[0, 4]]]},
            "covers 4 of 5 rows",
            id="incomplete-coverage",
        ),
        pytest.param(
            {MICRO_BATCH_INDICES: [[]]},
            "no packed microbatches",
            id="no-microbatches",
        ),
    ],
)
def test_build_route_key_batches_rejects_invalid_metadata(
    route_prefetch_module,
    extra_info: dict[str, Any],
    match: str,
) -> None:
    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=["A", "B", "C", "D", "E"],
        extra_info=extra_info,
    )

    with pytest.raises(ValueError, match=match):
        route_prefetch_module.build_route_key_batches(meta)


def test_build_pp_leader_topology_uses_pp_order_not_global_rank(
    route_prefetch_module,
) -> None:
    RankCoordinates = route_prefetch_module.RankCoordinates
    # In both DP replicas, PP=0 deliberately has a larger global rank than
    # PP=1. Non-leader TP/CP coordinates are interspersed in arbitrary order.
    coordinates = [
        RankCoordinates(global_rank=1, dp_rank=1, pp_rank=1, tp_rank=0, cp_rank=0),
        RankCoordinates(global_rank=7, dp_rank=0, pp_rank=0, tp_rank=1, cp_rank=0),
        RankCoordinates(global_rank=9, dp_rank=0, pp_rank=0, tp_rank=0, cp_rank=0),
        RankCoordinates(global_rank=4, dp_rank=1, pp_rank=0, tp_rank=0, cp_rank=1),
        RankCoordinates(global_rank=2, dp_rank=0, pp_rank=1, tp_rank=0, cp_rank=0),
        RankCoordinates(global_rank=8, dp_rank=1, pp_rank=0, tp_rank=0, cp_rank=0),
        RankCoordinates(global_rank=3, dp_rank=0, pp_rank=1, tp_rank=1, cp_rank=0),
        RankCoordinates(global_rank=6, dp_rank=1, pp_rank=1, tp_rank=0, cp_rank=1),
    ]

    topology = route_prefetch_module.build_pp_leader_topology(coordinates)

    assert list(topology) == [0, 1]
    assert [coord.pp_rank for coord in topology[0]] == [0, 1]
    assert [coord.global_rank for coord in topology[0]] == [9, 2]
    assert [coord.pp_rank for coord in topology[1]] == [0, 1]
    assert [coord.global_rank for coord in topology[1]] == [8, 1]


@pytest.mark.parametrize(
    ("dtype", "dtype_code"),
    [
        pytest.param(torch.int8, 0, id="int8"),
        pytest.param(torch.int16, 1, id="int16"),
        pytest.param(torch.int32, 2, id="int32"),
        pytest.param(torch.int64, 3, id="int64"),
    ],
)
def test_route_header_round_trips_supported_integer_dtypes(
    route_prefetch_module,
    dtype: torch.dtype,
    dtype_code: int,
) -> None:
    routes = torch.zeros((2, 4, 3, 2), dtype=dtype)

    header = route_prefetch_module._ok_header(
        microbatch_index=7,
        routed_experts=routes,
        device="cpu",
    )

    assert header.tolist() == [0, 7, dtype_code, 2, 4, 3, 2, 0]
    assert route_prefetch_module._CODE_TO_DTYPE[dtype_code] is dtype


def test_inject_prefetched_routes_preserves_microbatch_order(
    monkeypatch,
    route_prefetch_module,
    megatron_data_module,
) -> None:
    monkeypatch.setattr(
        megatron_data_module, "trace_tq_prefetch_payload", lambda **_: None
    )
    raw = [
        _raw_microbatch(["A", "B"]),
        _raw_microbatch(["C"]),
    ]
    first_routes = torch.full((2, 4, 2, 1), 11, dtype=torch.int16)
    second_routes = torch.full((1, 4, 2, 1), 22, dtype=torch.int16)
    routes = [
        route_prefetch_module.PrefetchedRoutes(0, ("A", "B"), first_routes),
        route_prefetch_module.PrefetchedRoutes(1, ("C",), second_routes),
    ]

    injected = list(
        megatron_data_module.inject_prefetched_routes(iter(raw), iter(routes))
    )

    assert len(injected) == 2
    assert injected[0][ROUTED_EXPERTS_FIELD] is first_routes
    assert injected[1][ROUTED_EXPERTS_FIELD] is second_routes
    assert route_prefetch_module.TQ_SAMPLE_IDS_FIELD not in injected[0]
    assert route_prefetch_module.TQ_SAMPLE_IDS_FIELD not in injected[1]


def test_prefetch_trace_keeps_one_step_across_microbatches(monkeypatch) -> None:
    from nemo_rl.utils import r3_trace

    context = {
        "active": True,
        "stage": "train",
        "trace_step": 7,
        "microbatch_counts": defaultdict(int),
    }
    records: list[dict[str, Any]] = []
    monkeypatch.setattr(r3_trace, "_current_context", lambda: context)
    monkeypatch.setattr(r3_trace, "_write_record", records.append)
    data = {
        "input_ids": torch.tensor([[1, 2, 0, 0]]),
        "input_lengths": torch.tensor([2]),
        "routed_experts": torch.zeros((1, 4, 1, 1), dtype=torch.int8),
    }

    r3_trace.trace_tq_prefetch_payload(keys=["A"], data=data)
    r3_trace.trace_tq_prefetch_payload(keys=["B"], data=data)

    assert [record["trace_step"] for record in records] == [7, 7]
    assert [record["microbatch_idx"] for record in records] == [1, 2]
    assert [record["key"] for record in records] == ["A", "B"]


def test_inject_prefetched_routes_rejects_sample_order_mismatch(
    route_prefetch_module,
    megatron_data_module,
) -> None:
    raw = _raw_microbatch(["B", "A"])
    routes = route_prefetch_module.PrefetchedRoutes(
        0,
        ("A", "B"),
        torch.zeros((2, 4, 1, 1), dtype=torch.int16),
    )

    with pytest.raises(RuntimeError, match="sample order mismatch"):
        next(megatron_data_module.inject_prefetched_routes(iter([raw]), iter([routes])))


def test_inject_prefetched_routes_rejects_missing_sample_ids(
    route_prefetch_module,
    megatron_data_module,
) -> None:
    raw = BatchedDataDict({"input_ids": torch.zeros((1, 4), dtype=torch.long)})
    routes = route_prefetch_module.PrefetchedRoutes(
        0,
        ("A",),
        torch.zeros((1, 4, 1, 1), dtype=torch.int16),
    )

    with pytest.raises(RuntimeError, match="missing '__tq_sample_ids'"):
        next(megatron_data_module.inject_prefetched_routes(iter([raw]), iter([routes])))


def test_inject_prefetched_routes_rejects_missing_route_batch(
    megatron_data_module,
) -> None:
    raw = _raw_microbatch(["A"])

    with pytest.raises(RuntimeError, match="ended before the raw microbatch"):
        next(megatron_data_module.inject_prefetched_routes(iter([raw]), iter(())))


def test_inject_prefetched_routes_rejects_existing_route_field(
    route_prefetch_module,
    megatron_data_module,
) -> None:
    raw = _raw_microbatch(["A"])
    raw[ROUTED_EXPERTS_FIELD] = torch.ones((1, 4, 1, 1), dtype=torch.int16)
    routes = route_prefetch_module.PrefetchedRoutes(
        0,
        ("A",),
        torch.zeros((1, 4, 1, 1), dtype=torch.int16),
    )

    with pytest.raises(RuntimeError, match="already contains routed_experts"):
        next(megatron_data_module.inject_prefetched_routes(iter([raw]), iter([routes])))


@pytest.mark.parametrize(
    ("route_shape", "match"),
    [
        pytest.param((1, 4, 1, 1), "batch mismatch", id="batch"),
        pytest.param((2, 3, 1, 1), "sequence mismatch", id="sequence"),
    ],
)
def test_inject_prefetched_routes_rejects_incompatible_shape(
    route_prefetch_module,
    megatron_data_module,
    route_shape: tuple[int, ...],
    match: str,
) -> None:
    raw = _raw_microbatch(["A", "B"], sequence_length=4)
    routes = route_prefetch_module.PrefetchedRoutes(
        0,
        ("A", "B"),
        torch.zeros(route_shape, dtype=torch.int16),
    )

    with pytest.raises(RuntimeError, match=match):
        next(megatron_data_module.inject_prefetched_routes(iter([raw]), iter([routes])))


def test_assert_complete_rejects_unconsumed_route_batches(
    route_prefetch_module,
) -> None:
    # The injection generator intentionally does not probe one item past the raw
    # iterator (that probe could start an unwanted collective). The worker calls
    # assert_complete after train to detect this inverse cardinality mismatch.
    prefetcher = object.__new__(route_prefetch_module.TrainRoutePrefetcher)
    prefetcher._consumed = 1
    prefetcher._key_batches = (("A",), ("B",))

    with pytest.raises(
        route_prefetch_module.TrainRoutePrefetchError,
        match="consumed=1, planned=2",
    ):
        prefetcher.assert_complete()


def test_disabled_prefetch_delegates_to_existing_train_path(monkeypatch) -> None:
    from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker._train_route_prefetch_enabled = False
    delegated: dict[str, Any] = {}

    def legacy_train_presharded(self, meta, **kwargs):
        delegated["self"] = self
        delegated["meta"] = meta
        delegated["kwargs"] = kwargs
        return {"legacy": True}

    monkeypatch.setattr(TQWorkerMixin, "train_presharded", legacy_train_presharded)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    meta = _meta(["A"], indices=[[[0, 1]]])

    result = worker.train_presharded(
        meta,
        loss_fn="loss",
        eval_mode=True,
        gbs=1,
        mbs=1,
    )

    assert result == {"legacy": True}
    assert delegated == {
        "self": worker,
        "meta": meta,
        "kwargs": {
            "loss_fn": "loss",
            "eval_mode": True,
            "gbs": 1,
            "mbs": 1,
        },
    }


def test_enabled_prefetch_excludes_only_routes_from_initial_fetch(monkeypatch) -> None:
    from nemo_rl.models.policy.workers import megatron_policy_worker

    worker = object.__new__(megatron_policy_worker.MegatronPolicyWorkerImpl)
    worker._train_route_prefetch_enabled = True
    worker._train_route_prefetch_groups = object()
    client = object()
    calls: dict[str, Any] = {}
    base_data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "input_lengths": torch.tensor([4]),
            "sample_mask": torch.tensor([1.0]),
        }
    )

    def fetch(meta):
        calls["fetch_meta"] = meta
        return base_data

    class FakePrefetcher:
        def __init__(self, *, client, meta, groups):
            calls["prefetch_args"] = (client, meta, groups)

        def assert_complete(self) -> None:
            calls["assert_complete"] = True

        def metrics(self) -> dict[str, float]:
            return {"ready_fraction": 1.0}

        def close(self) -> None:
            calls["closed"] = True

    def train(data, **kwargs):
        calls["train_data"] = data
        calls["train_kwargs"] = kwargs
        return {"rank": 0}

    worker._fetch = fetch
    worker._attach_or_repack_pack_metadata = lambda data, meta: data
    worker._require_dp_client = lambda: client
    worker.train = train
    monkeypatch.setattr(megatron_policy_worker, "TrainRoutePrefetcher", FakePrefetcher)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    meta = _meta(["A"], indices=[[[0, 1]]], pad_to_seqlen=4)
    meta.fields = ["input_ids", ROUTED_EXPERTS_FIELD]

    result = worker.train_presharded(meta, loss_fn="loss", gbs=1, mbs=1)

    assert calls["fetch_meta"].fields == ["input_ids"]
    assert calls["prefetch_args"] == (client, meta, worker._train_route_prefetch_groups)
    assert calls["train_data"]["__tq_sample_ids"] == ["A"]
    assert calls["train_kwargs"]["route_iterator"].__class__ is FakePrefetcher
    assert calls["assert_complete"] is True
    assert calls["closed"] is True
    assert result["train_route_prefetch_source_metrics"] == {"ready_fraction": 1.0}


class _RecordingClient:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._calls: list[tuple[str, ...]] = []

    def get_samples(
        self,
        *,
        sample_ids: list[str],
        partition_id: str,
        select_fields: list[str],
    ) -> torch.Tensor:
        assert partition_id == "train"
        assert select_fields == [ROUTED_EXPERTS_FIELD]
        with self._condition:
            self._calls.append(tuple(sample_ids))
            call_number = len(self._calls)
            self._condition.notify_all()
        return torch.full(
            (len(sample_ids), 4, 2, 1),
            call_number,
            dtype=torch.int16,
        )

    def wait_for_call_count(self, count: int, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while len(self._calls) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._condition.wait(timeout=remaining):
                    raise AssertionError(
                        f"timed out waiting for {count} gets; saw {self._calls}"
                    )

    def calls(self) -> list[tuple[str, ...]]:
        with self._condition:
            return list(self._calls)


def test_source_producer_stays_one_microbatch_ahead_without_cuda(
    monkeypatch,
    route_prefetch_module,
) -> None:
    client = _RecordingClient()
    meta = _meta(
        ["A", "B", "C", "D"],
        indices=[[[0, 2], [2, 3], [3, 4]]],
        pad_to_seqlen=4,
    )
    groups = route_prefetch_module.TrainRoutePrefetchGroups(
        pp_leader_group=object(),
        pp_leader_ranks=(0,),
        pp_source_rank=0,
        stage_group=object(),
        stage_source_rank=0,
        is_stage_leader=True,
        is_pp_source=True,
    )

    def fake_materialize(
        wire: torch.Tensor,
        *,
        layout: str,
        pad_to_seqlen: int,
    ) -> dict[str, torch.Tensor]:
        assert layout == "padded"
        assert pad_to_seqlen == 4
        return {ROUTED_EXPERTS_FIELD: wire}

    monkeypatch.setattr(route_prefetch_module, "materialize", fake_materialize)
    monkeypatch.setattr(
        route_prefetch_module.torch.distributed,
        "get_backend",
        lambda _group: "gloo",
    )
    monkeypatch.setattr(route_prefetch_module.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        route_prefetch_module.torch.distributed,
        "get_world_size",
        lambda _group: 1,
    )

    prefetcher = route_prefetch_module.TrainRoutePrefetcher(
        client=client,
        meta=meta,
        groups=groups,
    )
    try:
        # A permit is acquired before retrieval, so the producer cannot start
        # the second TQ get until the consumer takes the first payload.
        client.wait_for_call_count(1)
        assert client.calls() == [("A", "B")]

        first = next(prefetcher)
        assert first.sample_ids == ("A", "B")
        assert first.routed_experts.device.type == "cpu"

        # Consuming MB0 releases exactly one permit for MB1. MB2 must remain
        # unfetched until MB1 is consumed, which is strict depth-one lookahead.
        client.wait_for_call_count(2)
        assert client.calls() == [("A", "B"), ("C",)]

        second = next(prefetcher)
        assert second.sample_ids == ("C",)
        client.wait_for_call_count(3)
        assert client.calls() == [("A", "B"), ("C",), ("D",)]
    finally:
        prefetcher.close()


class _DistributedRouteClient:
    def __init__(self, *, rank: int, fail_on_call: int | None = None) -> None:
        self.rank = rank
        self.fail_on_call = fail_on_call
        self.calls = 0

    def get_samples(
        self,
        *,
        sample_ids: list[str],
        partition_id: str,
        select_fields: list[str],
    ):
        from tensordict import TensorDict

        assert self.rank == 0, f"non-source rank {self.rank} touched TQ"
        assert partition_id == "train"
        assert select_fields == [ROUTED_EXPERTS_FIELD]
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("injected TQ failure")
        rows = [
            torch.full(
                (4, 2, 1),
                ord(sample_id) - ord("A") + 1,
                dtype=torch.int8,
            )
            for sample_id in sample_ids
        ]
        routes = torch.stack(rows)
        return TensorDict({ROUTED_EXPERTS_FIELD: routes}, batch_size=[len(rows)])


def _distributed_prefetch_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_queue,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from nemo_rl.models.megatron.train_route_prefetch import (
            TrainRoutePrefetchError,
            TrainRoutePrefetchGroups,
            TrainRoutePrefetcher,
        )

        # All ranks create every group in the same order. Ranks 0/2 are the
        # PP-stage leaders; [0,1] and [2,3] are the two TP stages.
        pp_leader_group = dist.new_group(ranks=[0, 2], backend="gloo")
        stage_zero_group = dist.new_group(ranks=[0, 1], backend="gloo")
        stage_one_group = dist.new_group(ranks=[2, 3], backend="gloo")
        pp_rank = 0 if rank < 2 else 1
        groups = TrainRoutePrefetchGroups(
            pp_leader_group=pp_leader_group,
            pp_leader_ranks=(0, 2),
            pp_source_rank=0,
            stage_group=stage_zero_group if pp_rank == 0 else stage_one_group,
            stage_source_rank=0 if pp_rank == 0 else 2,
            is_stage_leader=rank in (0, 2),
            is_pp_source=rank == 0,
        )
        meta = _meta(
            ["A", "B", "C", "D"],
            indices=[[[0, 2], [2, 3], [3, 4]]],
            pad_to_seqlen=4,
        )

        client = _DistributedRouteClient(rank=rank)
        prefetcher = TrainRoutePrefetcher(
            client=client,
            meta=meta,
            groups=groups,
        )
        if pp_rank == 1:
            time.sleep(0.1)
        expected_batches = (("A", "B"), ("C",), ("D",))
        for expected_ids in expected_batches:
            payload = next(prefetcher)
            assert payload.sample_ids == expected_ids
            for row, sample_id in enumerate(expected_ids):
                expected_value = ord(sample_id) - ord("A") + 1
                assert torch.all(payload.routed_experts[row] == expected_value)
        prefetcher.assert_complete()
        prefetcher.close()
        assert client.calls == (3 if rank == 0 else 0)

        # The same topology propagates a source-side TQ error through both
        # PP leaders and then through both independent stage groups.
        dist.barrier()
        failing_client = _DistributedRouteClient(rank=rank, fail_on_call=2)
        failing_prefetcher = TrainRoutePrefetcher(
            client=failing_client,
            meta=meta,
            groups=groups,
        )
        first = next(failing_prefetcher)
        assert first.sample_ids == ("A", "B")
        try:
            next(failing_prefetcher)
        except TrainRoutePrefetchError as error:
            assert "injected TQ failure" in str(error)
        else:
            raise AssertionError("prefetch error did not reach route consumer")
        failing_prefetcher.close()
        assert failing_client.calls == (2 if rank == 0 else 0)

        # Early close drains the remaining fixed collective sequence instead
        # of cancelling one PP leader locally and stranding its peer.
        dist.barrier()
        early_close_client = _DistributedRouteClient(rank=rank)
        early_close_prefetcher = TrainRoutePrefetcher(
            client=early_close_client,
            meta=meta,
            groups=groups,
        )
        assert next(early_close_prefetcher).sample_ids == ("A", "B")
        if pp_rank == 1:
            time.sleep(0.1)
        early_close_prefetcher.close()
        assert early_close_client.calls == (3 if rank == 0 else 0)
        dist.barrier()
        result_queue.put((rank, "ok"))
    except Exception as error:  # pragma: no cover - surface child failures
        result_queue.put((rank, f"{type(error).__name__}: {error}"))
    finally:
        dist.destroy_process_group()


def test_pp2_route_prefetch_uses_independent_stage_consumption(tmp_path) -> None:
    init_file = str(tmp_path / "prefetch-init")
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_distributed_prefetch_worker,
            args=(rank, 4, init_file, result_queue),
        )
        for rank in range(4)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=45)
            assert process.exitcode == 0, f"worker exited with {process.exitcode}"

        results = sorted(result_queue.get(timeout=5) for _ in processes)
        assert results == [(rank, "ok") for rank in range(4)], results
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
