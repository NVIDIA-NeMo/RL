import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.lm_policy import Policy


class _FakeShardingAnnotations:
    def __init__(self, sizes=None):
        self._sizes = sizes or {"data_parallel": 2}

    def get_axis_size(self, axis_name: str) -> int:
        return self._sizes[axis_name]


class _RecordingWorkerGroup:
    def __init__(self, results):
        self._results = results
        self.run_kwargs = None

    def run_all_workers_sharded_data(self, method_name: str, **kwargs):
        self.run_kwargs = {"method_name": method_name, **kwargs}
        return "fake-futures"

    def get_all_worker_results(self, futures, return_generators_as_proxies=False):
        assert futures == "fake-futures"
        self.return_generators_as_proxies = return_generators_as_proxies
        return self._results


def test_get_topk_logits_concatenates_one_payload_per_data_parallel_shard():
    policy = object.__new__(Policy)
    policy.sharding_annotations = _FakeShardingAnnotations()
    policy.worker_group = _RecordingWorkerGroup(
        [
            {
                "topk_logits": torch.arange(12, dtype=torch.float32).reshape(1, 4, 3),
                "topk_indices": torch.arange(12, dtype=torch.int64).reshape(1, 4, 3),
            },
            {
                "topk_logits": torch.arange(12, 36, dtype=torch.float32).reshape(
                    2, 4, 3
                ),
                "topk_indices": torch.arange(12, 36, dtype=torch.int64).reshape(
                    2, 4, 3
                ),
            },
        ]
    )
    policy.use_dynamic_batches = False
    policy.use_sequence_packing = False
    policy.cfg = {}

    batch = BatchedDataDict(
        {
            "input_ids": torch.arange(12).reshape(3, 4),
            "input_lengths": torch.tensor([4, 4, 4], dtype=torch.int32),
        }
    )

    outputs = policy.get_topk_logits(batch, k=3)

    assert outputs["topk_logits"].shape == (3, 4, 3)
    assert outputs["topk_indices"].shape == (3, 4, 3)
    torch.testing.assert_close(
        outputs["topk_logits"],
        torch.cat(
            [
                policy.worker_group._results[0]["topk_logits"],
                policy.worker_group._results[1]["topk_logits"],
            ],
            dim=0,
        ),
    )
    torch.testing.assert_close(
        outputs["topk_indices"],
        torch.cat(
            [
                policy.worker_group._results[0]["topk_indices"],
                policy.worker_group._results[1]["topk_indices"],
            ],
            dim=0,
        ),
    )
    assert policy.worker_group.run_kwargs["method_name"] == "get_topk_logits"
    assert policy.worker_group.run_kwargs["replicate_on_axes"] == [
        "context_parallel",
        "tensor_parallel",
        "pipeline_parallel",
    ]
    assert policy.worker_group.run_kwargs["output_is_replicated"] == [
        "context_parallel",
        "tensor_parallel",
        "pipeline_parallel",
    ]
    assert policy.worker_group.run_kwargs["in_sharded_axes"] == ["data_parallel"]
    sharded_inputs = policy.worker_group.run_kwargs["data"]
    assert len(sharded_inputs) == 2
    torch.testing.assert_close(
        sharded_inputs[0]["input_ids"], torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    )
    torch.testing.assert_close(
        sharded_inputs[1]["input_ids"], torch.tensor([[8, 9, 10, 11]])
    )


def test_stage_layout_is_derived_from_sharding_and_policy_config():
    policy = object.__new__(Policy)
    policy.sharding_annotations = _FakeShardingAnnotations(
        {
            "data_parallel": 3,
            "context_parallel": 2,
            "tensor_parallel": 4,
            "pipeline_parallel": 1,
        }
    )
    policy.cfg = {
        "make_sequence_length_divisible_by": 16,
        "sequence_packing": {"enabled": True},
        "dynamic_batching": {"enabled": False},
    }
    policy.supports_multimodal = True

    layout = policy.stage_layout()

    assert layout.dp == 3
    assert layout.cp == 2
    assert layout.tp == 4
    assert layout.pp == 1
    assert layout.pad_multiple == 16
    assert layout.supports_sequence_packing is True
    assert layout.supports_dynamic_batching is False
    assert layout.supports_multimodal is True


def test_annotate_topk_stream_returns_worker_envelope_shards():
    from nemo_rl.algorithms.distillation_streaming import ShardedBatchStream

    policy = object.__new__(Policy)
    policy.worker_group = _RecordingWorkerGroup(["stream-0", "stream-1"])

    manifest = type("_FakeManifest", (), {"batch_id": "batch-0"})()
    token_stream = ShardedBatchStream(
        batch_id="batch-0",
        layout="dp_fullseq",
        dp_size=2,
        manifest=manifest,
        shard_streams=[["chunk-0"], ["chunk-1"]],
    )

    annotated_stream = policy.annotate_topk_stream(
        token_stream,
        k=4,
        micro_batch_size=2,
    )

    assert annotated_stream.batch_id == "batch-0"
    assert annotated_stream.shard_streams == ["stream-0", "stream-1"]
    assert policy.worker_group.return_generators_as_proxies is False
    assert policy.worker_group.run_kwargs["method_name"] == "annotate_topk_stream"
    assert policy.worker_group.run_kwargs["token_chunks"] == [["chunk-0"], ["chunk-1"]]
    assert "remote_call_options" not in policy.worker_group.run_kwargs
    assert policy.worker_group.run_kwargs["common_kwargs"] == {
        "k": 4,
        "micro_batch_size": 2,
        "batch_id": "batch-0",
    }


def test_train_distillation_stream_routes_envelopes_and_aggregates_results():
    from nemo_rl.algorithms.distillation_streaming import (
        AnnotatedTokenChunk,
        ShardedBatchStream,
    )

    policy = object.__new__(Policy)
    policy.cfg = {"train_global_batch_size": 4, "train_micro_batch_size": 1}
    policy.sharding_annotations = _FakeShardingAnnotations({"data_parallel": 2})
    policy.use_dynamic_batches = False
    policy.use_sequence_packing = False
    policy.flops_tracker = None
    policy.worker_group = _RecordingWorkerGroup(
        [
            {
                "global_loss": torch.tensor(1.5),
                "grad_norm": torch.tensor(0.5),
                "all_mb_metrics": {"global_valid_toks": [3]},
            },
            {
                "global_loss": torch.tensor(1.5),
                "grad_norm": torch.tensor(0.5),
                "all_mb_metrics": {"global_valid_toks": [4]},
            },
        ]
    )

    envelope = AnnotatedTokenChunk(
        batch_id="batch-0",
        chunk_id=0,
        token_chunk_ref=object(),
        teacher_topk_ref=object(),
        sample_ids=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        sample_order=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        update_group=torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        global_batch_slot=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
    )
    manifest = type("_FakeManifest", (), {"batch_id": "batch-0"})()
    stream = ShardedBatchStream(
        batch_id="batch-0",
        layout="dp_fullseq",
        dp_size=1,
        manifest=manifest,
        shard_streams=[[envelope]],
    )
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(16).reshape(4, 4),
            "input_lengths": torch.tensor([4, 4, 4, 4], dtype=torch.int32),
            "token_mask": torch.ones((4, 4), dtype=torch.bool),
            "sample_mask": torch.ones(4, dtype=torch.bool),
            "sample_ids": torch.tensor([10, 11, 12, 13], dtype=torch.int64),
            "sample_order": torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            "update_group": torch.tensor([0, 0, 0, 0], dtype=torch.int64),
            "global_batch_slot": torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        }
    )

    results = policy.train_distillation_stream(data, stream, loss_fn="loss-fn")

    assert torch.equal(results["loss"], torch.tensor(1.5))
    assert results["all_mb_metrics"] == {"global_valid_toks": [3, 4]}
    assert policy.worker_group.run_kwargs["method_name"] == "train_distillation_stream"
    assert policy.worker_group.run_kwargs["annotated_chunks"] == [[envelope], [envelope]]
    assert len(policy.worker_group.run_kwargs["data"]) == 2
    assert policy.worker_group.run_kwargs["common_kwargs"]["loss_fn"] == "loss-fn"
    assert policy.worker_group.run_kwargs["common_kwargs"]["student_dp_size"] == 2


def test_train_distillation_stream_routes_by_actual_sharded_sample_ids():
    from nemo_rl.algorithms.distillation_streaming import (
        AnnotatedTokenChunk,
        ShardedBatchStream,
    )

    policy = object.__new__(Policy)
    policy.cfg = {"train_global_batch_size": 4, "train_micro_batch_size": 1}
    policy.sharding_annotations = _FakeShardingAnnotations({"data_parallel": 2})
    policy.use_dynamic_batches = False
    policy.use_sequence_packing = False
    policy.flops_tracker = None
    policy.worker_group = _RecordingWorkerGroup(
        [
            {
                "global_loss": torch.tensor(1.0),
                "grad_norm": torch.tensor(0.0),
                "all_mb_metrics": {"global_valid_toks": [1]},
            },
            {
                "global_loss": torch.tensor(1.0),
                "grad_norm": torch.tensor(0.0),
                "all_mb_metrics": {"global_valid_toks": [1]},
            },
        ]
    )

    envelope = AnnotatedTokenChunk(
        batch_id="batch-0",
        chunk_id=0,
        token_chunk_ref=object(),
        teacher_topk_ref=object(),
        sample_ids=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        sample_order=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        update_group=torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        global_batch_slot=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
    )
    manifest = type("_FakeManifest", (), {"batch_id": "batch-0"})()
    stream = ShardedBatchStream(
        batch_id="batch-0",
        layout="dp_fullseq",
        dp_size=1,
        manifest=manifest,
        shard_streams=[[envelope]],
    )
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(16).reshape(4, 4),
            "input_lengths": torch.tensor([4, 4, 4, 4], dtype=torch.int32),
            "token_mask": torch.ones((4, 4), dtype=torch.bool),
            "sample_mask": torch.ones(4, dtype=torch.bool),
            "sample_ids": torch.tensor([10, 12, 11, 13], dtype=torch.int64),
            "sample_order": torch.tensor([0, 2, 1, 3], dtype=torch.int64),
            "update_group": torch.tensor([0, 0, 0, 0], dtype=torch.int64),
            "global_batch_slot": torch.tensor([0, 2, 1, 3], dtype=torch.int64),
        }
    )

    policy.train_distillation_stream(data, stream, loss_fn="loss-fn")

    sharded_inputs = policy.worker_group.run_kwargs["data"]
    assert sharded_inputs[0]["sample_ids"].tolist() == [10, 12]
    assert sharded_inputs[1]["sample_ids"].tolist() == [11, 13]
    assert policy.worker_group.run_kwargs["annotated_chunks"] == [[envelope], [envelope]]


def test_train_distillation_stream_from_refs_routes_by_global_slot():
    from nemo_rl.algorithms.distillation_streaming import (
        AnnotatedTokenChunk,
        ShardedBatchStream,
    )

    policy = object.__new__(Policy)
    policy.cfg = {"train_global_batch_size": 4, "train_micro_batch_size": 1}
    policy.sharding_annotations = _FakeShardingAnnotations({"data_parallel": 2})
    policy.use_dynamic_batches = False
    policy.use_sequence_packing = False
    policy.flops_tracker = None
    policy.worker_group = _RecordingWorkerGroup(
        [
            {
                "global_loss": torch.tensor(2.0),
                "grad_norm": torch.tensor(0.5),
                "all_mb_metrics": {"global_valid_toks": [2]},
            },
            {
                "global_loss": torch.tensor(2.0),
                "grad_norm": torch.tensor(0.5),
                "all_mb_metrics": {"global_valid_toks": [3]},
            },
        ]
    )

    shard0_envelope = AnnotatedTokenChunk(
        batch_id="batch-0",
        chunk_id=0,
        token_chunk_ref=object(),
        teacher_topk_ref=object(),
        sample_ids=torch.tensor([10, 11], dtype=torch.int64),
        sample_order=torch.tensor([0, 1], dtype=torch.int64),
        update_group=torch.tensor([0, 0], dtype=torch.int64),
        global_batch_slot=torch.tensor([0, 1], dtype=torch.int64),
    )
    shard1_envelope = AnnotatedTokenChunk(
        batch_id="batch-0",
        chunk_id=1,
        token_chunk_ref=object(),
        teacher_topk_ref=object(),
        sample_ids=torch.tensor([12, 13], dtype=torch.int64),
        sample_order=torch.tensor([2, 3], dtype=torch.int64),
        update_group=torch.tensor([0, 0], dtype=torch.int64),
        global_batch_slot=torch.tensor([2, 3], dtype=torch.int64),
    )
    manifest = type("_FakeManifest", (), {"batch_id": "batch-0"})()
    stream = ShardedBatchStream(
        batch_id="batch-0",
        layout="dp_fullseq",
        dp_size=1,
        manifest=manifest,
        shard_streams=[[shard0_envelope, shard1_envelope]],
    )

    results = policy.train_distillation_stream_from_refs(stream, loss_fn="loss-fn")

    assert torch.equal(results["loss"], torch.tensor(2.0))
    assert results["all_mb_metrics"] == {"global_valid_toks": [2, 3]}
    assert (
        policy.worker_group.run_kwargs["method_name"]
        == "train_distillation_stream_from_refs"
    )
    assert policy.worker_group.run_kwargs["annotated_chunks"] == [
        [shard0_envelope],
        [shard1_envelope],
    ]
    assert policy.worker_group.run_kwargs["student_dp_rank"] == [0, 1]
    assert policy.worker_group.run_kwargs["common_kwargs"]["loss_fn"] == "loss-fn"
    assert policy.worker_group.run_kwargs["common_kwargs"]["student_dp_size"] == 2


def test_train_distillation_stream_from_refs_rejects_reordered_student_sharding():
    from nemo_rl.algorithms.distillation_streaming import ShardedBatchStream

    policy = object.__new__(Policy)
    policy.cfg = {"train_global_batch_size": 4, "train_micro_batch_size": 1}
    policy.sharding_annotations = _FakeShardingAnnotations({"data_parallel": 2})
    policy.use_dynamic_batches = True
    policy.use_sequence_packing = False
    policy.worker_group = _RecordingWorkerGroup([])

    stream = ShardedBatchStream(
        batch_id="batch-0",
        layout="dp_fullseq",
        dp_size=1,
        manifest=type("_FakeManifest", (), {"batch_id": "batch-0"})(),
        shard_streams=[[]],
    )

    with pytest.raises(AssertionError, match="dynamic batching"):
        policy.train_distillation_stream_from_refs(stream, loss_fn="loss-fn")

    policy.use_dynamic_batches = False
    policy.use_sequence_packing = True
    with pytest.raises(AssertionError, match="sequence packing"):
        policy.train_distillation_stream_from_refs(stream, loss_fn="loss-fn")


def test_train_distillation_sparse_stream_routes_and_aggregates_results():
    from nemo_rl.algorithms.distillation_streaming import (
        AnnotatedTokenChunk,
        ShardedBatchStream,
    )

    policy = object.__new__(Policy)
    policy.cfg = {"train_global_batch_size": 4, "train_micro_batch_size": 1}
    policy.sharding_annotations = _FakeShardingAnnotations(
        {
            "data_parallel": 2,
            "context_parallel": 1,
        }
    )
    policy.use_dynamic_batches = False
    policy.use_sequence_packing = False
    policy.flops_tracker = None
    policy.worker_group = _RecordingWorkerGroup(
        [
            {
                "global_loss": torch.tensor(3.0),
                "grad_norm": torch.tensor(0.1),
                "all_mb_metrics": {"global_valid_toks": [5]},
            },
            {
                "global_loss": torch.tensor(3.0),
                "grad_norm": torch.tensor(0.1),
                "all_mb_metrics": {"global_valid_toks": [6]},
            },
        ]
    )

    envelope = AnnotatedTokenChunk(
        batch_id="batch-0",
        chunk_id=0,
        token_chunk_ref=object(),
        teacher_topk_ref=object(),
        sample_ids=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        sample_order=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        update_group=torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        global_batch_slot=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
    )
    stream = ShardedBatchStream(
        batch_id="batch-0",
        layout="dp_fullseq",
        dp_size=1,
        manifest=type("_FakeManifest", (), {"batch_id": "batch-0"})(),
        shard_streams=[[envelope]],
    )
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(16).reshape(4, 4),
            "input_lengths": torch.tensor([4, 4, 4, 4], dtype=torch.int32),
            "token_mask": torch.ones((4, 4), dtype=torch.bool),
            "sample_mask": torch.ones(4, dtype=torch.bool),
            "sample_ids": torch.tensor([10, 11, 12, 13], dtype=torch.int64),
            "sample_order": torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            "update_group": torch.tensor([0, 0, 0, 0], dtype=torch.int64),
            "global_batch_slot": torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        }
    )

    result = policy.train_distillation_sparse_stream(data, stream, loss_fn="loss-fn")

    assert torch.equal(result["loss"], torch.tensor(3.0))
    assert result["all_mb_metrics"] == {"global_valid_toks": [5, 6]}
    assert (
        policy.worker_group.run_kwargs["method_name"]
        == "train_distillation_sparse_stream"
    )
    assert policy.worker_group.run_kwargs["annotated_chunks"] == [[envelope], [envelope]]


def test_train_distillation_sparse_stream_rejects_unsupported_layouts():
    from nemo_rl.algorithms.distillation_streaming import ShardedBatchStream

    policy = object.__new__(Policy)
    policy.cfg = {"train_global_batch_size": 4, "train_micro_batch_size": 1}
    policy.sharding_annotations = _FakeShardingAnnotations(
        {
            "data_parallel": 2,
            "context_parallel": 1,
        }
    )
    policy.use_dynamic_batches = True
    policy.use_sequence_packing = False
    policy.worker_group = _RecordingWorkerGroup([])
    stream = ShardedBatchStream(
        batch_id="batch-0",
        layout="dp_fullseq",
        dp_size=1,
        manifest=type("_FakeManifest", (), {"batch_id": "batch-0"})(),
        shard_streams=[[]],
    )
    data = BatchedDataDict({"sample_ids": torch.arange(4)})

    with pytest.raises(AssertionError, match="dynamic batching"):
        policy.train_distillation_sparse_stream(data, stream, loss_fn="loss-fn")

    policy.use_dynamic_batches = False
    policy.use_sequence_packing = True
    with pytest.raises(AssertionError, match="sequence packing"):
        policy.train_distillation_sparse_stream(data, stream, loss_fn="loss-fn")

    policy.use_sequence_packing = False
    policy.sharding_annotations = _FakeShardingAnnotations(
        {
            "data_parallel": 2,
            "context_parallel": 2,
        }
    )
    with pytest.raises(AssertionError, match="CP"):
        policy.train_distillation_sparse_stream(data, stream, loss_fn="loss-fn")
