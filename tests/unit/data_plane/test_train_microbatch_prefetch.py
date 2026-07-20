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
"""CPU contracts for all-field packed-microbatch TQ prefetch."""

from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.column_io import kv_first_write
from nemo_rl.data_plane.schema import (
    ELEM_COUNTS_PER_GB,
    GLOBAL_FORWARD_PAD_SEQLEN,
    GLOBAL_VALID_SEQS_PER_GB,
    GLOBAL_VALID_TOKS_PER_GB,
    MICRO_BATCH_INDICES,
    MICRO_BATCH_LENGTHS,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

pytestmark = pytest.mark.mcore

FIELDS = [
    "input_ids",
    "input_lengths",
    "generation_logprobs",
    "prev_logprobs",
    "reference_policy_logprobs",
    "advantages",
    "token_mask",
    "sample_mask",
    "routed_experts",
]


@pytest.fixture
def prefetch_module():
    from nemo_rl.models.megatron import train_microbatch_prefetch

    return train_microbatch_prefetch


def _meta(
    sample_ids: list[str],
    *,
    indices: Any,
    lengths: Any,
    elem_counts: list[int] | None = None,
    valid_seqs: list[float] | None = None,
    valid_toks: list[float] | None = None,
) -> KVBatchMeta:
    num_global_batches = len(indices)
    extra_info = {
        MICRO_BATCH_INDICES: indices,
        MICRO_BATCH_LENGTHS: lengths,
        GLOBAL_FORWARD_PAD_SEQLEN: 8,
        GLOBAL_VALID_SEQS_PER_GB: valid_seqs or [4.0] * num_global_batches,
        GLOBAL_VALID_TOKS_PER_GB: valid_toks or [16.0] * num_global_batches,
    }
    if elem_counts is not None:
        extra_info[ELEM_COUNTS_PER_GB] = elem_counts
    return KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=sample_ids,
        fields=list(FIELDS),
        sequence_lengths=[4] * len(sample_ids),
        extra_info=extra_info,
    )


def _payload(sample_ids: list[str], sequence_length: int = 8) -> BatchedDataDict:
    values = torch.tensor(
        [ord(sample_id) - ord("A") + 1 for sample_id in sample_ids],
        dtype=torch.long,
    )
    batch_size = len(sample_ids)
    tiled = values[:, None].expand(batch_size, sequence_length)
    return BatchedDataDict(
        {
            "input_ids": tiled.clone(),
            "input_lengths": torch.full((batch_size,), 4, dtype=torch.long),
            "generation_logprobs": tiled.float(),
            "prev_logprobs": tiled.float() + 0.1,
            "reference_policy_logprobs": tiled.float() + 0.2,
            "advantages": tiled.float() + 0.3,
            "token_mask": torch.ones(batch_size, sequence_length),
            "sample_mask": torch.ones(batch_size),
            "routed_experts": tiled[:, :, None, None].to(torch.int32),
        }
    )


def test_build_plan_preserves_multi_global_batch_boundaries(prefetch_module) -> None:
    meta = _meta(
        list("ABCDEFGH"),
        indices=[[[0, 2], [2, 3]], [[0, 1], [1, 4], [4, 5]]],
        lengths=[[8, 4], [4, 12, 4]],
        elem_counts=[3, 5],
        valid_seqs=[3.0, 4.0],
        valid_toks=[10.0, 14.0],
    )

    plan = prefetch_module.build_train_microbatch_plan(meta)

    assert plan.fields == tuple(FIELDS)
    assert plan.pad_to_seqlen == 8
    assert plan.num_microbatches == 5
    assert plan.global_batches[0].microbatch_sample_ids == (("A", "B"), ("C",))
    assert plan.global_batches[0].microbatch_lengths == (8, 4)
    assert plan.global_batches[0].global_valid_toks == 10.0
    assert plan.global_batches[1].microbatch_sample_ids == (
        ("D",),
        ("E", "F", "G"),
        ("H",),
    )
    assert plan.global_batches[1].global_valid_seqs == 4.0


@pytest.mark.parametrize(
    ("extra_key", "extra_value", "error"),
    [
        (MICRO_BATCH_INDICES, [[[0, 1], [2, 4]]], "cover each global batch"),
        (MICRO_BATCH_LENGTHS, [[8]], "ranges but"),
        (GLOBAL_VALID_TOKS_PER_GB, [], GLOBAL_VALID_TOKS_PER_GB),
    ],
)
def test_build_plan_rejects_invalid_metadata(
    prefetch_module,
    extra_key: str,
    extra_value: Any,
    error: str,
) -> None:
    meta = _meta(
        list("ABCD"),
        indices=[[[0, 2], [2, 4]]],
        lengths=[[8, 8]],
    )
    meta.extra_info[extra_key] = extra_value
    with pytest.raises(ValueError, match=error):
        prefetch_module.build_train_microbatch_plan(meta)


def test_build_replica_topology_is_independent_of_global_rank_layout(
    prefetch_module,
) -> None:
    coordinates = [
        prefetch_module.RankCoordinates(0, 1, 1, 0, 0),
        prefetch_module.RankCoordinates(1, 0, 1, 0, 0),
        prefetch_module.RankCoordinates(2, 1, 0, 0, 0),
        prefetch_module.RankCoordinates(3, 0, 0, 0, 0),
    ]
    topology = prefetch_module.build_replica_topology(coordinates)
    assert tuple(x.global_rank for x in topology[0]) == (1, 3)
    assert tuple(x.global_rank for x in topology[1]) == (0, 2)


@pytest.mark.parametrize(
    ("timeout_s", "expected_timeout_s"),
    [(None, None), (321.5, 321.5)],
)
def test_prefetch_group_uses_configured_collective_timeout(
    monkeypatch,
    prefetch_module,
    timeout_s: float | None,
    expected_timeout_s: float | None,
) -> None:
    distributed = prefetch_module.torch.distributed
    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(distributed, "get_backend", lambda: "gloo")
    monkeypatch.setattr(distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        distributed,
        "all_gather",
        lambda gathered, local: gathered[0].copy_(local),
    )
    group_calls: list[dict[str, Any]] = []

    def fake_new_group(**kwargs):
        group_calls.append(kwargs)
        return object()

    monkeypatch.setattr(distributed, "new_group", fake_new_group)
    monkeypatch.setattr(
        prefetch_module.parallel_state, "get_data_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        prefetch_module.parallel_state,
        "get_pipeline_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        prefetch_module.parallel_state,
        "get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        prefetch_module.parallel_state,
        "get_context_parallel_rank",
        lambda: 0,
    )

    prefetch_module.initialize_train_microbatch_prefetch_group(
        collective_timeout_s=timeout_s
    )

    assert len(group_calls) == 1
    if expected_timeout_s is None:
        assert "timeout" not in group_calls[0]
    else:
        assert group_calls[0]["timeout"].total_seconds() == expected_timeout_s


@pytest.mark.parametrize("timeout_s", [0, -1, float("nan"), float("inf")])
def test_prefetch_group_rejects_invalid_collective_timeout(
    prefetch_module,
    timeout_s: float,
) -> None:
    with pytest.raises(ValueError, match="collective_timeout_s"):
        prefetch_module.initialize_train_microbatch_prefetch_group(
            collective_timeout_s=timeout_s
        )


@pytest.mark.parametrize("timeout_s", [0, -1, float("nan"), float("inf")])
def test_prefetcher_rejects_invalid_item_ready_timeout(
    prefetch_module,
    timeout_s: float,
) -> None:
    group = prefetch_module.TrainMicrobatchPrefetchGroup(
        replica_group=object(),
        replica_ranks=(0,),
        source_rank=0,
        is_source=True,
    )
    with pytest.raises(ValueError, match="item_ready_timeout_s"):
        prefetch_module.TrainMicrobatchPrefetcher(
            client=object(),
            meta=_meta(["A"], indices=[[[0, 1]]], lengths=[[4]]),
            group=group,
            pad_value_dict={},
            depth=1,
            item_ready_timeout_s=timeout_s,
        )


def test_stamp_train_normalization_matches_current_formula() -> None:
    from nemo_rl.models.policy.tq_policy import _stamp_train_normalization

    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=list("ABCD"),
        fields=list(FIELDS),
        sequence_lengths=[4, 4, 4, 4],
        extra_info={},
    )
    sample_mask = torch.tensor([1.0, 0.0, 1.0, 1.0])
    token_mask = torch.tensor(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 1, 1],
        ],
        dtype=torch.float32,
    )

    _stamp_train_normalization(
        meta,
        sample_mask=sample_mask,
        token_mask=token_mask,
        batch_size=2,
    )

    assert meta.extra_info[GLOBAL_VALID_SEQS_PER_GB] == [1.0, 2.0]
    assert meta.extra_info[GLOBAL_VALID_TOKS_PER_GB] == [2.0, 4.0]


def test_normalization_metadata_survives_dp_packing(prefetch_module) -> None:
    from nemo_rl.data_plane.preshard import shard_meta_for_dp
    from nemo_rl.models.policy.tq_policy import _stamp_train_normalization

    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=list("ABCDEFGH"),
        fields=list(FIELDS),
        sequence_lengths=[2, 3, 4, 5, 2, 3, 4, 5],
        extra_info={GLOBAL_FORWARD_PAD_SEQLEN: 8},
    )
    _stamp_train_normalization(
        meta,
        sample_mask=torch.tensor([1, 1, 0, 1, 1, 0, 1, 1], dtype=torch.float32),
        token_mask=torch.ones(8, 8),
        batch_size=4,
    )
    shards, _ = shard_meta_for_dp(
        meta,
        dp_world=2,
        batch_size=4,
        sequence_packing_args={
            "algorithm": "modified_first_fit_decreasing",
            "input_key": "input_ids",
            "input_lengths_key": "input_lengths",
            "sequence_length_pad_multiple": 1,
            "max_tokens_per_microbatch": 8,
        },
    )

    assert len(shards) == 2
    for shard in shards:
        assert shard.extra_info[GLOBAL_VALID_SEQS_PER_GB] == [3.0, 3.0]
        assert shard.extra_info[GLOBAL_VALID_TOKS_PER_GB] == [21.0, 21.0]
        plan = prefetch_module.build_train_microbatch_plan(shard)
        assert [batch.global_valid_seqs for batch in plan.global_batches] == [
            3.0,
            3.0,
        ]


def test_prefetched_iterator_uses_plan_shape_without_skeleton_payload(
    monkeypatch,
) -> None:
    from nemo_rl.models.megatron import data as data_module

    raw_microbatches = [_payload(["A", "B"]), _payload(["C"])]
    calls: dict[str, Any] = {}

    def fake_pack_parameters(*_args):
        return 8, 16, 32

    def fake_processed_iterator(**kwargs):
        calls.update(kwargs)
        return kwargs["raw_iterator"]

    monkeypatch.setattr(
        data_module,
        "_get_pack_sequence_parameters_for_megatron",
        fake_pack_parameters,
    )
    monkeypatch.setattr(
        data_module,
        "make_processed_microbatch_iterator",
        fake_processed_iterator,
    )
    cfg = {
        "sequence_packing": {"enabled": True},
        "dynamic_batching": {"enabled": False},
        "make_sequence_length_divisible_by": 8,
        "megatron_cfg": {},
    }
    skeleton = BatchedDataDict({"__tq_sample_ids": ["A", "B", "C"]})

    iterator, count, mbs, seq_length, padded_seq_length = (
        data_module.get_microbatch_iterator(
            skeleton,
            cfg,
            mbs=1,
            straggler_timer=object(),
            prefetched_raw_iterator=iter(raw_microbatches),
            prefetched_microbatch_lengths=[8, 4],
            prefetched_forward_pad_seqlen=8,
        )
    )

    received = list(iterator)
    assert received[0] is raw_microbatches[0]
    assert received[1] is raw_microbatches[1]
    assert count == 2
    assert mbs == 1
    assert seq_length == 8
    assert padded_seq_length == 32
    assert calls["seq_length_key"] == "input_lengths"
    assert calls["pad_individual_seqs_to_multiple_of"] == 8
    assert calls["pad_packed_seq_to_multiple_of"] == 16


def test_plan_accepts_r3_off_fields(prefetch_module) -> None:
    meta = _meta(["A", "B"], indices=[[[0, 2]]], lengths=[[8]])
    meta.fields = [field for field in FIELDS if field != "routed_experts"]
    plan = prefetch_module.build_train_microbatch_plan(meta)
    assert "routed_experts" not in plan.fields
    assert plan.global_batches[0].microbatch_sample_ids == (("A", "B"),)


def test_disabled_prefetch_delegates_to_existing_train_path(monkeypatch) -> None:
    from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker._train_microbatch_prefetch_enabled = False
    delegated: dict[str, Any] = {}

    def legacy_train_presharded(self, meta, **kwargs):
        delegated["self"] = self
        delegated["meta"] = meta
        delegated["kwargs"] = kwargs
        return {"legacy": True}

    monkeypatch.setattr(TQWorkerMixin, "train_presharded", legacy_train_presharded)
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    meta = _meta(["A"], indices=[[[0, 1]]], lengths=[[4]])

    result = worker.train_presharded(
        meta,
        loss_fn="loss",
        eval_mode=True,
        gbs=1,
        mbs=1,
    )

    assert result == {"legacy": True}
    assert delegated["self"] is worker
    assert delegated["meta"] is meta


def test_enabled_prefetch_skips_complete_shard_fetch(monkeypatch) -> None:
    from nemo_rl.models.policy.workers import megatron_policy_worker

    worker = object.__new__(megatron_policy_worker.MegatronPolicyWorkerImpl)
    worker._train_microbatch_prefetch_enabled = True
    worker._train_microbatch_prefetch_depth = 1
    worker._train_microbatch_prefetch_item_ready_timeout_s = 30.0
    worker._train_microbatch_prefetch_group = object()
    client = object()
    calls: dict[str, Any] = {}

    class FakePrefetcher:
        def __init__(self, **kwargs):
            calls["prefetch_kwargs"] = kwargs

        def assert_complete(self) -> None:
            calls["assert_complete"] = True

        def close(self) -> None:
            calls["closed"] = True

        def metrics(self) -> dict[str, float]:
            return {"ready_fraction": 1.0}

    def train(data, **kwargs):
        calls["train_data"] = data
        calls["train_kwargs"] = kwargs
        return {"rank": 0}

    def fail_fetch(_meta):
        raise AssertionError("complete-shard _fetch must not run")

    worker._fetch = fail_fetch
    worker._attach_or_repack_pack_metadata = lambda data, meta: data
    worker._require_dp_client = lambda: client
    worker._pad_value_dict = lambda: {"input_ids": 0}
    worker.train = train
    monkeypatch.setattr(
        megatron_policy_worker,
        "TrainMicrobatchPrefetcher",
        FakePrefetcher,
    )
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    meta = _meta(["A"], indices=[[[0, 1]]], lengths=[[4]])

    result = worker.train_presharded(meta, loss_fn="loss", gbs=1, mbs=1)

    assert calls["prefetch_kwargs"] == {
        "client": client,
        "meta": meta,
        "group": worker._train_microbatch_prefetch_group,
        "pad_value_dict": {"input_ids": 0},
        "depth": 1,
        "item_ready_timeout_s": 30.0,
    }
    assert calls["train_data"]["__tq_sample_ids"] == ["A"]
    assert calls["train_kwargs"]["microbatch_prefetcher"].__class__ is FakePrefetcher
    assert calls["assert_complete"] is True
    assert calls["closed"] is True
    assert result["train_microbatch_prefetch_metrics"] == {"ready_fraction": 1.0}


def test_prefetch_plan_matches_legacy_complete_shard_fetch(
    monkeypatch,
    prefetch_module,
    tq_client_backends,
) -> None:
    """Per-MB real TQ reads must equal slices of the legacy full-shard read."""
    sample_ids = list("ABCDE")
    partition_id = "train-prefetch-plan-legacy-eq"
    payload = _payload(sample_ids)
    payload["input_lengths"] = torch.tensor([3, 7, 4, 6, 2])
    extra_info = {
        MICRO_BATCH_INDICES: [[[0, 2], [2, 5]]],
        MICRO_BATCH_LENGTHS: [[8, 8]],
        GLOBAL_FORWARD_PAD_SEQLEN: 8,
        GLOBAL_VALID_SEQS_PER_GB: [5.0],
        GLOBAL_VALID_TOKS_PER_GB: [22.0],
    }
    pad_value_dict = {"input_ids": -1}

    tq_client_backends.register_partition(
        partition_id=partition_id,
        fields=list(FIELDS),
        num_samples=len(sample_ids),
        consumer_tasks=["train"],
    )
    prefetcher = None
    try:
        meta = kv_first_write(
            payload,
            sample_ids=sample_ids,
            dp_client=tq_client_backends,
            partition_id=partition_id,
            extra_info=extra_info,
        )
        full_wire = tq_client_backends.get_samples(
            sample_ids=sample_ids,
            partition_id=partition_id,
            select_fields=list(FIELDS),
        )
        assert full_wire["input_ids"].is_nested
        assert full_wire["routed_experts"].is_nested
        full_shard = prefetch_module.materialize(
            full_wire,
            layout="padded",
            pad_value_dict=pad_value_dict,
            pad_to_seqlen=8,
        )

        group = prefetch_module.TrainMicrobatchPrefetchGroup(
            replica_group=object(),
            replica_ranks=(0,),
            source_rank=0,
            is_source=True,
        )
        monkeypatch.setattr(prefetch_module.torch.distributed, "get_rank", lambda: 0)
        prefetcher = prefetch_module.TrainMicrobatchPrefetcher(
            client=tq_client_backends,
            meta=meta,
            group=group,
            pad_value_dict=pad_value_dict,
            depth=1,
            item_ready_timeout_s=30,
        )

        planned_microbatches = list(prefetcher.iter_global_batch(0))
        assert len(planned_microbatches) == 2
        for actual, (start, end) in zip(
            planned_microbatches,
            ((0, 2), (2, 5)),
            strict=True,
        ):
            expected = full_shard.slice(start, end)
            assert set(actual) == set(expected) == set(FIELDS)
            for field in FIELDS:
                assert actual[field].shape == expected[field].shape
                assert actual[field].dtype == expected[field].dtype
                assert torch.equal(actual[field], expected[field]), field
        prefetcher.assert_complete()
    finally:
        try:
            if prefetcher is not None:
                prefetcher.close()
        finally:
            tq_client_backends.clear_samples(
                sample_ids=None,
                partition_id=partition_id,
            )


def _make_prefetch_train_worker(megatron_policy_worker):
    worker = object.__new__(megatron_policy_worker.MegatronPolicyWorkerImpl)
    worker._train_microbatch_prefetch_enabled = True
    worker.timer = SimpleNamespace(start=lambda _: None)
    worker.model = SimpleNamespace(
        inference_params=None,
        config=SimpleNamespace(mtp_num_layers=0),
        modules=lambda: (),
        eval=lambda: None,
    )
    worker.cfg = {
        "megatron_cfg": {
            "empty_unused_memory_level": 0,
            "use_fused_linear_logprobs": False,
        },
        "sequence_packing": {"enabled": True},
    }
    worker.dp_size = 1
    worker.fp8_cfg = None
    worker.should_disable_forward_pre_hook = False
    worker.mcore_state = SimpleNamespace(straggler_timer=None)
    worker.delegate_pack_to_model = False
    worker.sampling_params = None
    worker.draft_model = None
    worker.defer_fp32_logits = False
    worker._router_replay_enabled = False
    worker._uses_mxfp8_overlap_shared_param_buffer = lambda: True
    worker._set_moe_grad_scale_func = lambda _: None
    worker._set_mtp_grad_scale_func = lambda _: None
    return worker


def _patch_prefetch_train_cpu_prologue(monkeypatch, megatron_policy_worker) -> None:
    original_tensor = torch.tensor

    def tensor_without_cuda(*args, **kwargs):
        kwargs = dict(kwargs)
        if kwargs.get("device") == "cuda":
            kwargs.pop("device")
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(megatron_policy_worker.torch, "tensor", tensor_without_cuda)
    monkeypatch.setattr(
        megatron_policy_worker.torch.distributed,
        "all_reduce",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        megatron_policy_worker.torch.distributed,
        "barrier",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        megatron_policy_worker.torch.cuda,
        "synchronize",
        lambda: None,
    )
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    monkeypatch.setattr(
        megatron_policy_worker.parallel_state,
        "get_data_parallel_group",
        lambda: object(),
    )


def test_train_prefetch_plan_count_mismatch(monkeypatch) -> None:
    from nemo_rl.models.megatron import train_microbatch_prefetch
    from nemo_rl.models.policy.workers import megatron_policy_worker

    worker = _make_prefetch_train_worker(megatron_policy_worker)
    _patch_prefetch_train_cpu_prologue(monkeypatch, megatron_policy_worker)
    plan = train_microbatch_prefetch.build_train_microbatch_plan(
        _meta(
            list("ABCD"),
            indices=[[[0, 2], [2, 4]]],
            lengths=[[8, 8]],
        )
    )
    data = BatchedDataDict(
        {train_microbatch_prefetch.TQ_SAMPLE_IDS_FIELD: list("ABCD")}
    )
    prefetcher = SimpleNamespace(plan=plan)

    with pytest.raises(
        RuntimeError,
        match="plan has 1, training expects 2",
    ):
        worker.train(
            data,
            loss_fn=object(),
            eval_mode=True,
            gbs=2,
            mbs=1,
            microbatch_prefetcher=prefetcher,
        )


def test_train_prefetch_wires_plan_into_megatron_forward_backward(
    monkeypatch,
) -> None:
    from nemo_rl.models.megatron import train_microbatch_prefetch
    from nemo_rl.models.policy.workers import megatron_policy_worker

    class StopAfterForwardBackward(Exception):
        pass

    worker = _make_prefetch_train_worker(megatron_policy_worker)
    _patch_prefetch_train_cpu_prologue(monkeypatch, megatron_policy_worker)
    plan = train_microbatch_prefetch.build_train_microbatch_plan(
        _meta(
            ["A", "B"],
            indices=[[[0, 1], [1, 2]]],
            lengths=[[8, 4]],
            valid_seqs=[3.0],
            valid_toks=[11.0],
        )
    )
    raw_microbatches = [_payload(["A"]), _payload(["B"])]
    calls: dict[str, Any] = {"iter_global_batch": []}

    class FakePrefetcher:
        def __init__(self) -> None:
            self.plan = plan
            self.raw_iterator = None

        def iter_global_batch(self, global_batch_index):
            calls["iter_global_batch"].append(global_batch_index)
            self.raw_iterator = iter(raw_microbatches)
            return self.raw_iterator

    prefetcher = FakePrefetcher()
    processed_iterator = object()

    def fail_legacy_global_batch(*_args, **_kwargs):
        raise AssertionError("prefetch train path must not process the skeleton batch")

    def capture_microbatch_iterator(*_args, **kwargs):
        calls["microbatch_iterator"] = kwargs
        return processed_iterator, 2, 1, 8, 8

    def capture_forward_backward(**kwargs):
        calls["forward_backward"] = kwargs
        raise StopAfterForwardBackward

    monkeypatch.setattr(
        megatron_policy_worker,
        "process_global_batch",
        fail_legacy_global_batch,
    )
    monkeypatch.setattr(
        megatron_policy_worker,
        "get_microbatch_iterator",
        capture_microbatch_iterator,
    )
    monkeypatch.setattr(
        megatron_policy_worker,
        "LossPostProcessor",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        megatron_policy_worker,
        "get_rerun_state_machine",
        lambda: SimpleNamespace(should_run_forward_backward=lambda _: True),
    )
    monkeypatch.setattr(
        megatron_policy_worker,
        "megatron_forward_backward",
        capture_forward_backward,
    )
    data = BatchedDataDict({train_microbatch_prefetch.TQ_SAMPLE_IDS_FIELD: ["A", "B"]})

    with pytest.raises(StopAfterForwardBackward):
        worker.train(
            data,
            loss_fn=object(),
            eval_mode=True,
            gbs=2,
            mbs=1,
            microbatch_prefetcher=prefetcher,
        )

    assert calls["iter_global_batch"] == [0]
    iterator_kwargs = calls["microbatch_iterator"]
    assert iterator_kwargs["prefetched_raw_iterator"] is prefetcher.raw_iterator
    assert iterator_kwargs["prefetched_microbatch_lengths"] == (8, 4)
    assert iterator_kwargs["prefetched_forward_pad_seqlen"] == 8
    forward_kwargs = calls["forward_backward"]
    assert forward_kwargs["data_iterator"] is processed_iterator
    assert forward_kwargs["num_microbatches"] == 2
    assert forward_kwargs["global_valid_seqs"].item() == 3.0
    assert forward_kwargs["global_valid_toks"].item() == 11.0


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
    ) -> BatchedDataDict:
        assert partition_id == "train"
        assert select_fields == FIELDS
        with self._condition:
            self._calls.append(tuple(sample_ids))
            self._condition.notify_all()
        return _payload(sample_ids)

    def wait_for_call_count(self, count: int) -> None:
        deadline = time.monotonic() + 5
        with self._condition:
            while len(self._calls) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"timed out waiting for {count} calls; got {self._calls}"
                    )
                self._condition.wait(timeout=remaining)

    def calls(self) -> list[tuple[str, ...]]:
        with self._condition:
            return list(self._calls)


class _BlockingClient:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def get_samples(
        self,
        *,
        sample_ids: list[str],
        partition_id: str,
        select_fields: list[str],
    ) -> BatchedDataDict:
        assert partition_id == "train"
        assert select_fields == FIELDS
        self.calls += 1
        self.started.set()
        self.release.wait()
        return _payload(sample_ids)


def _single_rank_prefetcher(monkeypatch, prefetch_module, *, depth: int):
    client = _RecordingClient()
    meta = _meta(
        list("ABCD"),
        indices=[[[0, 2], [2, 3], [3, 4]]],
        lengths=[[8, 4, 4]],
    )
    group = prefetch_module.TrainMicrobatchPrefetchGroup(
        replica_group=object(),
        replica_ranks=(0,),
        source_rank=0,
        is_source=True,
    )

    def fake_materialize(
        wire: BatchedDataDict,
        *,
        layout: str,
        pad_value_dict: dict[str, Any],
        pad_to_seqlen: int,
    ) -> BatchedDataDict:
        assert layout == "padded"
        assert pad_value_dict == {"input_ids": 0}
        assert pad_to_seqlen == 8
        return wire

    monkeypatch.setattr(prefetch_module, "materialize", fake_materialize)
    monkeypatch.setattr(prefetch_module.torch.distributed, "get_rank", lambda: 0)
    return client, prefetch_module.TrainMicrobatchPrefetcher(
        client=client,
        meta=meta,
        group=group,
        pad_value_dict={"input_ids": 0},
        depth=depth,
        item_ready_timeout_s=5,
    )


def test_close_is_bounded_after_item_ready_timeout(
    monkeypatch,
    prefetch_module,
) -> None:
    client = _BlockingClient()
    meta = _meta(["A"], indices=[[[0, 1]]], lengths=[[4]])
    group = prefetch_module.TrainMicrobatchPrefetchGroup(
        replica_group=object(),
        replica_ranks=(0,),
        source_rank=0,
        is_source=True,
    )
    monkeypatch.setattr(prefetch_module, "materialize", lambda wire, **_: wire)
    monkeypatch.setattr(prefetch_module.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(prefetch_module, "_CLOSE_TIMEOUT_SECONDS", 0.05)
    prefetcher = prefetch_module.TrainMicrobatchPrefetcher(
        client=client,
        meta=meta,
        group=group,
        pad_value_dict={},
        depth=1,
        item_ready_timeout_s=0.05,
    )
    assert client.started.wait(timeout=1)
    try:
        take_started = time.monotonic()
        with pytest.raises(
            prefetch_module.TrainMicrobatchPrefetchError,
            match="did not become ready within 0.05s",
        ):
            next(prefetcher)
        assert time.monotonic() - take_started < 0.5
        with pytest.raises(
            prefetch_module.TrainMicrobatchPrefetchError,
            match="terminal after failure",
        ):
            next(prefetcher)

        close_started = time.monotonic()
        with pytest.raises(
            prefetch_module.TrainMicrobatchPrefetchError,
            match="producer did not stop within",
        ):
            prefetcher.close()
        assert time.monotonic() - close_started < 0.5
    finally:
        client.release.set()
        prefetcher._prefetch_iterator._producer_thread.join(timeout=1)
    assert not prefetcher._prefetch_iterator._producer_thread.is_alive()
    assert client.calls == 1


@pytest.mark.parametrize("lookahead", [False, True])
def test_thread_prefetch_iterator_controls_lookahead(
    prefetch_module,
    lookahead: bool,
) -> None:
    condition = threading.Condition()
    source_calls: list[int] = []

    def source():
        for value in range(3):
            with condition:
                source_calls.append(value)
                condition.notify_all()
            yield value

    def wait_for_call_count(count: int) -> None:
        deadline = time.monotonic() + 5
        with condition:
            while len(source_calls) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"timed out waiting for {count} calls; got {source_calls}"
                    )
                condition.wait(timeout=remaining)

    prefetcher = prefetch_module._ThreadPrefetchIterator(
        iter(source()),
        lookahead=lookahead,
        item_ready_timeout_s=5,
        thread_name="test-thread-prefetch",
    )
    try:
        wait_for_call_count(1)
        assert source_calls == [0]
        assert next(prefetcher) == 0

        if lookahead:
            wait_for_call_count(2)
            assert source_calls == [0, 1]
        else:
            time.sleep(0.05)
            assert source_calls == [0]

        assert next(prefetcher) == 1
        if lookahead:
            wait_for_call_count(3)
            assert source_calls == [0, 1, 2]
            assert next(prefetcher) == 2
        else:
            time.sleep(0.05)
            assert source_calls == [0, 1]
            assert next(prefetcher) == 2
            assert source_calls == [0, 1, 2]
        with pytest.raises(StopIteration):
            next(prefetcher)
    finally:
        prefetcher.close()


def test_thread_prefetch_iterator_propagates_source_failure(prefetch_module) -> None:
    def source():
        yield 1
        raise RuntimeError("injected loader failure")

    prefetcher = prefetch_module._ThreadPrefetchIterator(
        iter(source()),
        lookahead=True,
        item_ready_timeout_s=5,
        thread_name="test-thread-prefetch-failure",
    )
    try:
        assert next(prefetcher) == 1
        with pytest.raises(
            prefetch_module._ThreadPrefetchError,
            match="injected loader failure",
        ):
            next(prefetcher)
        with pytest.raises(
            prefetch_module._ThreadPrefetchError,
            match="terminal after failure",
        ):
            next(prefetcher)
    finally:
        prefetcher.close()
        prefetcher.close()


def test_depth_one_stays_exactly_one_microbatch_ahead(
    monkeypatch,
    prefetch_module,
) -> None:
    client, prefetcher = _single_rank_prefetcher(monkeypatch, prefetch_module, depth=1)
    try:
        client.wait_for_call_count(1)
        assert client.calls() == [("A", "B")]

        first = next(prefetcher)
        assert first.sample_ids == ("A", "B")
        assert set(first.data) == set(FIELDS)
        client.wait_for_call_count(2)
        assert client.calls() == [("A", "B"), ("C",)]

        second = next(prefetcher)
        assert second.sample_ids == ("C",)
        client.wait_for_call_count(3)
        assert client.calls() == [("A", "B"), ("C",), ("D",)]
        third = next(prefetcher)

        metrics = prefetcher.metrics()
        assert metrics["tq_get_calls"] == 3
        assert metrics["consume_count"] == 3
        assert metrics["materialized_payload_bytes"] > 0
        assert {
            "tq_get_s",
            "materialize_s",
            "distribute_s",
            "consumer_wait_s",
            "first_microbatch_ready_s",
            "ready_fraction",
        } <= metrics.keys()

        reconstructed = BatchedDataDict.from_batches(
            [first.data, second.data, third.data]
        )
        baseline = _payload(list("ABCD"))
        assert set(reconstructed) == set(baseline)
        for field in FIELDS:
            assert reconstructed[field].dtype == baseline[field].dtype
            assert reconstructed[field].shape == baseline[field].shape
            assert torch.equal(reconstructed[field], baseline[field])
    finally:
        prefetcher.close()


def test_depth_zero_waits_for_the_next_consumer_request(
    monkeypatch,
    prefetch_module,
) -> None:
    client, prefetcher = _single_rank_prefetcher(monkeypatch, prefetch_module, depth=0)
    try:
        client.wait_for_call_count(1)
        assert next(prefetcher).sample_ids == ("A", "B")
        time.sleep(0.05)
        assert client.calls() == [("A", "B")]

        assert next(prefetcher).sample_ids == ("C",)
        client.wait_for_call_count(2)
        time.sleep(0.05)
        assert client.calls() == [("A", "B"), ("C",)]
    finally:
        prefetcher.close()


def test_multiple_global_batches_are_consumed_without_crossing_boundaries(
    monkeypatch,
    prefetch_module,
) -> None:
    client = _RecordingClient()
    meta = _meta(
        list("ABCDEF"),
        indices=[[[0, 2], [2, 3]], [[0, 1], [1, 3]]],
        lengths=[[8, 4], [4, 8]],
        elem_counts=[3, 3],
    )
    group = prefetch_module.TrainMicrobatchPrefetchGroup(
        replica_group=object(),
        replica_ranks=(0,),
        source_rank=0,
        is_source=True,
    )
    monkeypatch.setattr(prefetch_module, "materialize", lambda wire, **_: wire)
    monkeypatch.setattr(prefetch_module.torch.distributed, "get_rank", lambda: 0)
    prefetcher = prefetch_module.TrainMicrobatchPrefetcher(
        client=client,
        meta=meta,
        group=group,
        pad_value_dict={},
        depth=1,
        item_ready_timeout_s=5,
    )
    try:
        first = list(prefetcher.iter_global_batch(0))
        second = list(prefetcher.iter_global_batch(1))
        assert [batch.size for batch in first] == [2, 1]
        assert [batch.size for batch in second] == [1, 2]
        assert client.calls() == [("A", "B"), ("C",), ("D",), ("E", "F")]
        prefetcher.assert_complete()
    finally:
        prefetcher.close()


class _DistributedClient:
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
    ) -> BatchedDataDict:
        assert self.rank == 0, f"non-source rank {self.rank} touched TQ"
        assert partition_id == "train"
        assert select_fields == FIELDS
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("injected TQ failure")
        return _payload(sample_ids)


def _distributed_worker(
    rank: int, world_size: int, init_file: str, result_queue
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
        from nemo_rl.models.megatron import train_microbatch_prefetch as module

        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        prefetch_group = module.TrainMicrobatchPrefetchGroup(
            replica_group=group,
            replica_ranks=tuple(range(world_size)),
            source_rank=0,
            is_source=rank == 0,
        )
        meta = _meta(
            list("ABCD"),
            indices=[[[0, 2], [2, 3], [3, 4]]],
            lengths=[[8, 4, 4]],
        )
        module.materialize = lambda wire, **_: wire

        for depth in (0, 1):
            client = _DistributedClient(rank=rank)
            prefetcher = module.TrainMicrobatchPrefetcher(
                client=client,
                meta=meta,
                group=prefetch_group,
                pad_value_dict={},
                depth=depth,
                item_ready_timeout_s=5,
            )
            for step, expected_ids in enumerate((("A", "B"), ("C",), ("D",))):
                if rank >= world_size // 2:
                    time.sleep(0.03 * (step + 1))
                item = next(prefetcher)
                assert item.sample_ids == expected_ids
                for row, sample_id in enumerate(expected_ids):
                    expected = ord(sample_id) - ord("A") + 1
                    assert torch.all(item.data["input_ids"][row] == expected)
                    assert torch.all(item.data["routed_experts"][row] == expected)
            prefetcher.assert_complete()
            prefetcher.close()
            assert client.calls == (3 if rank == 0 else 0)
            dist.barrier()

        failing_client = _DistributedClient(rank=rank, fail_on_call=2)
        failing = module.TrainMicrobatchPrefetcher(
            client=failing_client,
            meta=meta,
            group=prefetch_group,
            pad_value_dict={},
            depth=1,
            item_ready_timeout_s=5,
        )
        assert next(failing).sample_ids == ("A", "B")
        with pytest.raises(module.TrainMicrobatchPrefetchError, match="injected"):
            next(failing)
        failing.close()
        assert failing_client.calls == (2 if rank == 0 else 0)
        dist.barrier()

        early_client = _DistributedClient(rank=rank)
        early = module.TrainMicrobatchPrefetcher(
            client=early_client,
            meta=meta,
            group=prefetch_group,
            pad_value_dict={},
            depth=1,
            item_ready_timeout_s=5,
        )
        assert next(early).sample_ids == ("A", "B")
        early.close()
        assert early_client.calls == (3 if rank == 0 else 0)
        dist.barrier()
        result_queue.put((rank, "ok"))
    except Exception as error:  # pragma: no cover - surface child failures
        result_queue.put((rank, f"{type(error).__name__}: {error}"))
    finally:
        dist.destroy_process_group()


def test_all_field_distribution_survives_uneven_pp_consumption(tmp_path) -> None:
    init_file = str(tmp_path / "prefetch-init")
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_distributed_worker,
            args=(rank, 4, init_file, result_queue),
        )
        for rank in range(4)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=60)
            assert process.exitcode == 0, f"worker exited with {process.exitcode}"
        results = sorted(result_queue.get(timeout=5) for _ in processes)
        assert results == [(rank, "ok") for rank in range(4)], results
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
