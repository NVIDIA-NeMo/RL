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
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.algorithms import distillation_sync
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.column_io import kv_first_write, read_columns, write_columns
from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.data_plane.schema import (
    DISTILLATION_TRAIN_FIELDS,
    GLOBAL_FORWARD_PAD_SEQLEN,
    TEACHER_TOPK_INDICES,
    TEACHER_TOPK_LOGITS,
    TEACHER_TOPK_SEED_FIELDS,
)
from nemo_rl.data_plane.transport_metrics import (
    add_byte_metric_derivatives,
    tensor_nbytes,
    topk_payload_nbytes,
    valid_token_tensor_nbytes,
)
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.tq_policy import TQPolicy


def _distillation_seed_batch(batch_size: int, seq_len: int) -> BatchedDataDict:
    return BatchedDataDict(
        {
            "input_ids": torch.arange(batch_size * seq_len, dtype=torch.long).reshape(
                batch_size, seq_len
            ),
            "input_lengths": torch.tensor([seq_len] * batch_size, dtype=torch.long),
            "token_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
            "sample_mask": torch.ones((batch_size,), dtype=torch.long),
        }
    )


def test_teacher_topk_columns_roundtrip_as_bsk() -> None:
    batch_size, seq_len, k = 3, 5, 4
    client = NoOpDataPlaneClient()
    client.register_partition(
        partition_id="train",
        fields=list(DISTILLATION_TRAIN_FIELDS),
        num_samples=batch_size,
        consumer_tasks=["teacher_topk", "train"],
    )
    meta = kv_first_write(
        _distillation_seed_batch(batch_size, seq_len),
        sample_ids=[f"u{i}" for i in range(batch_size)],
        dp_client=client,
        partition_id="train",
    )
    logits = torch.randn(batch_size, seq_len, k, dtype=torch.bfloat16)
    indices = torch.arange(batch_size * seq_len * k, dtype=torch.long).reshape(
        batch_size, seq_len, k
    )

    write_columns(
        client,
        meta,
        fields={
            TEACHER_TOPK_LOGITS: logits,
            TEACHER_TOPK_INDICES: indices,
        },
    )
    fetched = read_columns(
        client,
        meta,
        select_fields=[TEACHER_TOPK_LOGITS, TEACHER_TOPK_INDICES],
    )

    assert fetched[TEACHER_TOPK_LOGITS].shape == (batch_size, seq_len, k)
    assert fetched[TEACHER_TOPK_INDICES].shape == (batch_size, seq_len, k)
    assert fetched[TEACHER_TOPK_LOGITS].dtype == torch.bfloat16
    assert fetched[TEACHER_TOPK_INDICES].dtype == torch.long
    assert torch.equal(fetched[TEACHER_TOPK_LOGITS], logits)
    assert torch.equal(fetched[TEACHER_TOPK_INDICES], indices)


def test_tq_policy_prepare_step_accepts_custom_fields_and_tasks() -> None:
    calls = []

    class _Client:
        def register_partition(self, **kwargs):
            calls.append(kwargs)

    policy = TQPolicy.__new__(TQPolicy)
    policy.dp_client = _Client()
    policy.tq_partition_id = "train"

    TQPolicy.prepare_step(
        policy,
        num_samples=7,
        fields=["input_ids", TEACHER_TOPK_LOGITS],
        consumer_tasks=["teacher_topk", "train"],
    )

    assert calls == [
        {
            "partition_id": "train",
            "fields": ["input_ids", TEACHER_TOPK_LOGITS],
            "num_samples": 7,
            "consumer_tasks": ["teacher_topk", "train"],
            "grpo_group_size": None,
        }
    ]


def test_tq_policy_train_from_meta_uses_custom_fetch_fields() -> None:
    class _ShardingAnnotations:
        def get_axis_size(self, axis: str) -> int:
            assert axis == "data_parallel"
            return 1

    class _WorkerGroup:
        def __init__(self) -> None:
            self.method = None
            self.submitted_meta = None
            self.cluster = SimpleNamespace(world_size=lambda: 1)

        def run_all_workers_sharded_data(self, method: str, **kwargs):
            self.method = method
            self.submitted_meta = kwargs["meta"]
            return ["future"]

        def get_all_worker_results(self, futures):
            assert futures == ["future"]
            return [
                {
                    "global_loss": torch.tensor(1.0),
                    "grad_norm": torch.tensor(2.0),
                    "all_mb_metrics": {},
                }
            ]

        def shutdown(self, **_) -> None:
            return None

    policy = TQPolicy.__new__(TQPolicy)
    policy.cfg = {"train_global_batch_size": 2, "train_micro_batch_size": 1}
    policy.use_dynamic_batches = False
    policy.use_sequence_packing = False
    policy.sharding_annotations = _ShardingAnnotations()
    policy.worker_group = _WorkerGroup()
    policy.flops_tracker = None
    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=["u0", "u1"],
        fields=list(DISTILLATION_TRAIN_FIELDS),
        sequence_lengths=[4, 4],
    )
    custom_fields = ["input_ids", TEACHER_TOPK_LOGITS, TEACHER_TOPK_INDICES]

    result = TQPolicy.train_from_meta(
        policy,
        meta,
        loss_fn=object(),
        fields=custom_fields,
    )

    assert result["loss"].item() == 1.0
    assert policy.worker_group.method == "train_presharded"
    assert len(policy.worker_group.submitted_meta) == 1
    assert policy.worker_group.submitted_meta[0].fields == custom_fields


class _TopKWorker(TQWorkerMixin):
    tokenizer = SimpleNamespace(pad_token_id=0)
    cfg = {
        "sequence_packing": {"enabled": False},
        "dynamic_batching": {"enabled": False},
    }

    def __init__(self, client: NoOpDataPlaneClient) -> None:
        self._dp_client = client

    def get_topk_logits(self, *, data, k: int, micro_batch_size=None):
        del micro_batch_size
        batch_size, seq_len = data["input_ids"].shape
        logits = torch.arange(batch_size * seq_len * k, dtype=torch.float32).reshape(
            batch_size, seq_len, k
        )
        indices = (
            torch.arange(k, dtype=torch.long)
            .reshape(1, 1, k)
            .expand(batch_size, seq_len, k)
        )
        return BatchedDataDict({"topk_logits": logits, "topk_indices": indices})


def test_teacher_worker_topk_writeback_returns_only_scalar_ack() -> None:
    batch_size, seq_len, k = 2, 6, 3
    client = NoOpDataPlaneClient()
    client.register_partition(
        partition_id="train",
        fields=list(DISTILLATION_TRAIN_FIELDS),
        num_samples=batch_size,
        consumer_tasks=["teacher_topk", "train"],
    )
    meta = kv_first_write(
        _distillation_seed_batch(batch_size, seq_len),
        sample_ids=[f"u{i}" for i in range(batch_size)],
        dp_client=client,
        partition_id="train",
    )
    worker = _TopKWorker(client)

    result = worker.get_topk_logits_presharded(
        replace(meta, fields=list(TEACHER_TOPK_SEED_FIELDS)),
        k=k,
    )

    expected_bytes = (
        batch_size
        * seq_len
        * k
        * (
            torch.tensor([], dtype=torch.float32).element_size()
            + torch.tensor([], dtype=torch.long).element_size()
        )
    )
    assert result["teacher_topk_payload_bytes"] == expected_bytes
    assert result["teacher_topk_valid_payload_bytes"] == expected_bytes
    assert result["tq_teacher_topk_write_bytes"] == expected_bytes
    assert result["tq_teacher_topk_write_num_samples"] == batch_size
    assert result["tq_teacher_topk_write_ms"] >= 0.0
    assert all(not isinstance(v, torch.Tensor) for v in result.values())
    fetched = read_columns(
        client,
        meta,
        select_fields=[TEACHER_TOPK_LOGITS, TEACHER_TOPK_INDICES],
    )
    assert fetched[TEACHER_TOPK_LOGITS].shape == (batch_size, seq_len, k)
    assert fetched[TEACHER_TOPK_INDICES].shape == (batch_size, seq_len, k)


def test_attach_policy_workers_to_data_plane_after_student_bootstrap(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(distillation_sync.ray, "get", lambda value: value)

    class _WorkerGroup:
        def run_all_workers_single_data(self, method: str, cfg: dict):
            calls.append((method, cfg))
            return ["ok"]

    policy = SimpleNamespace(worker_group=_WorkerGroup())
    dp_cfg = {"enabled": True, "impl": "transfer_queue"}

    distillation_sync.attach_policy_workers_to_data_plane(policy, dp_cfg)

    assert calls == [("setup_data_plane", dp_cfg)]


def test_teacher_dispatch_uses_writeback_method_and_not_driver_topk() -> None:
    class _ShardingAnnotations:
        def get_axis_size(self, axis: str) -> int:
            assert axis == "data_parallel"
            return 1

    class _WorkerGroup:
        def __init__(self) -> None:
            self.method = None
            self.meta = None
            self.common_kwargs = None

        def run_all_workers_sharded_data(self, method: str, meta, common_kwargs, **_):
            self.method = method
            self.meta = meta
            self.common_kwargs = common_kwargs
            return ["future"]

        def get_all_worker_results(self, futures):
            assert futures == ["future"]
            return [{}]

    def _driver_topk_should_not_run(*_, **__):
        raise AssertionError("driver get_topk_logits must not be called")

    worker_group = _WorkerGroup()
    policy = SimpleNamespace(
        use_dynamic_batches=False,
        use_sequence_packing=False,
        sharding_annotations=_ShardingAnnotations(),
        worker_group=worker_group,
        get_topk_logits=_driver_topk_should_not_run,
    )
    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=["u0", "u1"],
        fields=list(DISTILLATION_TRAIN_FIELDS),
        sequence_lengths=[5, 5],
        extra_info={"pad_to_multiple": 4},
    )

    metrics = distillation_sync.dispatch_teacher_topk_writeback(
        policy,
        meta,
        fields=list(TEACHER_TOPK_SEED_FIELDS),
        k=8,
    )

    assert worker_group.method == "get_topk_logits_presharded"
    assert worker_group.common_kwargs == {"k": 8}
    assert worker_group.meta[0].fields == list(TEACHER_TOPK_SEED_FIELDS)
    assert GLOBAL_FORWARD_PAD_SEQLEN in meta.extra_info
    assert metrics["teacher_topk_payload_bytes"] == 0
    assert metrics["driver_teacher_topk_bytes_avoided"] == 0


def test_teacher_dispatch_aggregates_transport_metrics() -> None:
    class _ShardingAnnotations:
        def get_axis_size(self, axis: str) -> int:
            assert axis == "data_parallel"
            return 1

    class _WorkerGroup:
        def run_all_workers_sharded_data(self, *_, **__):
            return ["future"]

        def get_all_worker_results(self, futures):
            assert futures == ["future"]
            return [
                {
                    "teacher_topk_payload_bytes": 100,
                    "teacher_topk_valid_payload_bytes": 80,
                    "teacher_topk_padding_overhead_bytes": 20,
                    "tq_teacher_topk_write_bytes": 80,
                    "tq_teacher_topk_write_num_samples": 2,
                    "tq_teacher_topk_write_ms": 1.5,
                },
                {
                    "teacher_topk_payload_bytes": 200,
                    "teacher_topk_valid_payload_bytes": 160,
                    "teacher_topk_padding_overhead_bytes": 40,
                    "tq_teacher_topk_write_bytes": 160,
                    "tq_teacher_topk_write_num_samples": 4,
                    "tq_teacher_topk_write_ms": 2.5,
                },
            ]

    policy = SimpleNamespace(
        use_dynamic_batches=False,
        use_sequence_packing=False,
        sharding_annotations=_ShardingAnnotations(),
        worker_group=_WorkerGroup(),
    )
    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=["u0", "u1"],
        fields=list(DISTILLATION_TRAIN_FIELDS),
        sequence_lengths=[5, 5],
    )

    metrics = distillation_sync.dispatch_teacher_topk_writeback(
        policy,
        meta,
        fields=list(TEACHER_TOPK_SEED_FIELDS),
        k=8,
    )

    assert metrics["teacher_topk_payload_bytes"] == 300
    assert metrics["teacher_topk_valid_payload_bytes"] == 240
    assert metrics["teacher_topk_padding_overhead_bytes"] == 60
    assert metrics["driver_rx_teacher_topk_bytes"] == 0
    assert metrics["driver_tx_teacher_topk_bytes"] == 0
    assert metrics["driver_teacher_topk_bytes"] == 0
    assert metrics["driver_teacher_topk_bytes_avoided"] == 600
    assert metrics["tq_teacher_topk_write_bytes"] == 240
    assert metrics["tq_teacher_topk_write_num_samples"] == 6
    assert metrics["tq_teacher_topk_write_ms_sum"] == 4.0
    assert metrics["tq_teacher_topk_write_ms_max"] == 2.5


def test_topk_payload_metrics_account_for_valid_lengths() -> None:
    logits = torch.zeros((2, 5, 3), dtype=torch.bfloat16)
    indices = torch.zeros((2, 5, 3), dtype=torch.long)

    assert topk_payload_nbytes(logits, indices) == 2 * 5 * 3 * (2 + 8)
    assert topk_payload_nbytes(logits, indices, [5, 2]) == 7 * 3 * (2 + 8)

    metrics: dict[str, float | int] = {"teacher_topk_payload_bytes": 300}
    add_byte_metric_derivatives(metrics, token_count=30)

    assert metrics["teacher_topk_payload_gib"] == 300 / float(1024**3)
    assert metrics["teacher_topk_payload_bytes_per_token"] == 10


def test_dedupe_fields_preserves_first_occurrence_order() -> None:
    assert distillation_sync._dedupe_fields() == []
    assert distillation_sync._dedupe_fields(["a", "a", "b"]) == ["a", "b"]
    # First occurrence across groups wins; tuples and lists interoperate.
    assert distillation_sync._dedupe_fields(["a", "b"], ("b", "c"), ["a", "d"]) == [
        "a",
        "b",
        "c",
        "d",
    ]


def test_as_row_aligned_tensor_branches() -> None:
    rows = 3
    valid = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)

    # Row-aligned 2D tensor is returned unchanged.
    assert distillation_sync._as_row_aligned_tensor(valid, rows) is valid
    # Non-tensors are not TQ-storable.
    assert distillation_sync._as_row_aligned_tensor([1, 2, 3], rows) is None
    # Scalar (0-dim) tensors carry no row axis.
    assert distillation_sync._as_row_aligned_tensor(torch.tensor(1.0), rows) is None
    # Row-count mismatch is rejected.
    assert distillation_sync._as_row_aligned_tensor(valid, rows + 1) is None
    # PackedTensor is unwrapped via as_tensor() before the row check.
    packed = PackedTensor(valid, dim_to_pack=0)
    result = distillation_sync._as_row_aligned_tensor(packed, rows)
    assert result is not None and result.shape == (rows, 4)


def test_aggregate_teacher_topk_transport_results_skips_empty_and_none() -> None:
    metrics = distillation_sync._aggregate_teacher_topk_transport_results([{}, None])
    assert metrics["teacher_topk_payload_bytes"] == 0
    assert metrics["driver_teacher_topk_bytes_avoided"] == 0
    assert metrics["tq_teacher_topk_write_ms_max"] == 0.0


def test_aggregate_teacher_topk_transport_results_sums_and_maxes() -> None:
    metrics = distillation_sync._aggregate_teacher_topk_transport_results(
        [
            {
                "teacher_topk_payload_bytes": 100,
                "tq_teacher_topk_write_num_samples": 2,
                "tq_teacher_topk_write_ms": 1.5,
            },
            {
                "teacher_topk_payload_bytes": 200,
                "tq_teacher_topk_write_num_samples": 4,
                "tq_teacher_topk_write_ms": 0.5,
            },
        ]
    )
    assert metrics["teacher_topk_payload_bytes"] == 300
    assert metrics["tq_teacher_topk_write_num_samples"] == 6
    assert metrics["tq_teacher_topk_write_ms_sum"] == 2.0
    assert metrics["tq_teacher_topk_write_ms_max"] == 1.5
    # Avoided driver bytes are twice the resident payload (RX + TX hop).
    assert metrics["driver_teacher_topk_bytes_avoided"] == 600


def test_aggregate_teacher_topk_transport_results_rejects_non_mapping() -> None:
    with pytest.raises(RuntimeError, match="scalar metric acks"):
        distillation_sync._aggregate_teacher_topk_transport_results([("not", "a map")])


def test_aggregate_teacher_topk_transport_results_rejects_tensor_payloads() -> None:
    with pytest.raises(RuntimeError, match="tensor payloads"):
        distillation_sync._aggregate_teacher_topk_transport_results(
            [{"teacher_topk_payload_bytes": torch.zeros(2)}]
        )


def test_packing_args_for_policy_selects_branch() -> None:
    dynamic_policy = SimpleNamespace(
        use_dynamic_batches=True,
        use_sequence_packing=False,
        dynamic_batching_args={"sequence_length_round": 8},
        cfg={"dynamic_batching": {"logprob_mb_tokens": 512}},
    )
    spa, dba = distillation_sync._packing_args_for_policy(
        dynamic_policy, "logprob_mb_tokens"
    )
    assert spa is None
    assert dba == {"sequence_length_round": 8, "max_tokens_per_microbatch": 512}

    packing_policy = SimpleNamespace(
        use_dynamic_batches=False,
        use_sequence_packing=True,
        sequence_packing_args={"algorithm": "first_fit"},
        cfg={"sequence_packing": {"logprob_mb_tokens": 256}},
    )
    spa, dba = distillation_sync._packing_args_for_policy(
        packing_policy, "logprob_mb_tokens"
    )
    assert dba is None
    assert spa == {"algorithm": "first_fit", "max_tokens_per_microbatch": 256}

    plain_policy = SimpleNamespace(
        use_dynamic_batches=False, use_sequence_packing=False
    )
    assert distillation_sync._packing_args_for_policy(
        plain_policy, "logprob_mb_tokens"
    ) == (None, None)


def test_stamp_policy_pad_seqlen_uses_pad_to_multiple_for_plain_policy() -> None:
    policy = SimpleNamespace(use_dynamic_batches=False, use_sequence_packing=False)
    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=["u0", "u1"],
        fields=list(DISTILLATION_TRAIN_FIELDS),
        sequence_lengths=[5, 7],
        extra_info={"pad_to_multiple": 4},
    )
    distillation_sync._stamp_policy_pad_seqlen(
        policy, meta, mb_tokens_key="logprob_mb_tokens"
    )
    # round_up(max(5, 7), max(4, 1)) == 8
    assert meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] == 8


def test_stamp_policy_pad_seqlen_honors_dynamic_sequence_length_round() -> None:
    policy = SimpleNamespace(
        use_dynamic_batches=True,
        use_sequence_packing=False,
        dynamic_batching_args={"sequence_length_round": 16},
        cfg={"dynamic_batching": {"logprob_mb_tokens": 512}},
    )
    meta = KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=["u0"],
        fields=list(DISTILLATION_TRAIN_FIELDS),
        sequence_lengths=[5],
    )
    distillation_sync._stamp_policy_pad_seqlen(
        policy, meta, mb_tokens_key="logprob_mb_tokens"
    )
    # round_up(5, max(1, 16)) == 16
    assert meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] == 16


def test_tensor_nbytes_counts_only_tensors() -> None:
    assert tensor_nbytes("not a tensor") == 0
    assert tensor_nbytes(torch.zeros((2, 3), dtype=torch.float32)) == 2 * 3 * 4
    assert tensor_nbytes(torch.zeros((4,), dtype=torch.int64)) == 4 * 8


def test_valid_token_tensor_nbytes_accounts_for_valid_prefix() -> None:
    tensor = torch.zeros((2, 5, 3), dtype=torch.float32)
    # No lengths -> full tensor.
    assert valid_token_tensor_nbytes(tensor, None) == 2 * 5 * 3 * 4
    # Lengths clamped to [0, max_seq_len]; trailing dims multiply through.
    assert valid_token_tensor_nbytes(tensor, [5, 9]) == (5 + 5) * 3 * 4
    assert valid_token_tensor_nbytes(tensor, [2, -1]) == (2 + 0) * 3 * 4
    # Row-count mismatch falls back to the full tensor byte count.
    assert valid_token_tensor_nbytes(tensor, [5]) == 2 * 5 * 3 * 4
    # 1D tensors have no sequence axis -> full byte count.
    assert valid_token_tensor_nbytes(torch.zeros((4,)), [2]) == 4 * 4
