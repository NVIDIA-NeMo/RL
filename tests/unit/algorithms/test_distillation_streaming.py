# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import threading
from dataclasses import replace

import pytest
import torch

from nemo_rl.algorithms.distillation_streaming import (
    AnnotatedTokenChunk,
    BatchManifest,
    DPReshardPlanner,
    STREAM_METADATA_KEYS,
    ShardedBatchStream,
    SpanTable,
    StepManifest,
    TeacherTopKChunk,
    TokenChunk,
    assemble_dense_distillation_batch_from_annotated_chunks,
    attach_sparse_teacher_topk_to_local_batch_from_annotated_chunks,
    attach_teacher_topk_to_local_batch_from_annotated_chunks,
    attach_step_metadata,
    build_batch_manifest_from_train_data,
    build_conservation_oracle,
    build_repeated_sample_metadata,
    build_step_manifest,
    build_teacher_topk_chunk_from_dense,
    build_token_stream_from_train_data,
    build_token_stream_from_rollout_batch,
    collect_teacher_topk_chunks_to_dense,
    drain_annotated_stream,
    iter_annotated_teacher_topk_chunks,
    loss_spans_from_token_mask,
    parse_data_pipeline_config,
    route_drained_stream_to_student_shards,
    route_drained_stream_to_student_shards_by_sample_ids,
    slice_span_table,
    validate_chunk_budgets,
    validate_streaming_capabilities,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _span_table_single_full_loss(input_lengths: list[int]) -> SpanTable:
    offsets = [0]
    starts = []
    ends = []
    for input_length in input_lengths:
        if input_length > 1:
            starts.append(0)
            ends.append(input_length - 1)
        offsets.append(len(starts))
    return SpanTable(
        offsets=torch.tensor(offsets, dtype=torch.int64),
        starts=torch.tensor(starts, dtype=torch.int32),
        ends=torch.tensor(ends, dtype=torch.int32),
    )


def _batch_manifest(
    *,
    sample_ids: list[int] | None = None,
    sample_order: list[int] | None = None,
    update_group: list[int] | None = None,
    global_batch_slot: list[int] | None = None,
    input_lengths: list[int] | None = None,
    loss_spans: SpanTable | None = None,
) -> BatchManifest:
    if sample_ids is not None:
        batch_size = len(sample_ids)
    elif input_lengths is not None:
        batch_size = len(input_lengths)
    else:
        batch_size = 4

    sample_ids = sample_ids or list(range(100, 100 + batch_size))
    sample_order = sample_order or list(range(batch_size))
    update_group = update_group or [index // 2 for index in range(batch_size)]
    global_batch_slot = global_batch_slot or [index % 2 for index in range(batch_size)]
    input_lengths = input_lengths or [batch_size + 1 - index for index in range(batch_size)]
    loss_spans = loss_spans or _span_table_single_full_loss(input_lengths)

    return BatchManifest(
        batch_id="batch-0",
        step=7,
        sample_ids=torch.tensor(sample_ids, dtype=torch.int64),
        sample_order=torch.tensor(sample_order, dtype=torch.int64),
        update_group=torch.tensor(update_group, dtype=torch.int64),
        global_batch_slot=torch.tensor(global_batch_slot, dtype=torch.int64),
        prompt_ids=torch.arange(batch_size, dtype=torch.int64),
        generation_ids=torch.arange(batch_size, dtype=torch.int32),
        input_lengths=torch.tensor(input_lengths, dtype=torch.int32),
        sample_mask=torch.ones(batch_size, dtype=torch.bool),
        loss_spans=loss_spans,
        max_sequence_length=max(input_lengths),
        tokenizer_name_or_path="unit-test-tokenizer",
        tokenizer_config={},
    )


def _token_chunk(
    manifest: BatchManifest,
    *,
    chunk_id: int = 0,
    sample_indices: list[int] | None = None,
) -> TokenChunk:
    sample_indices = sample_indices or list(range(manifest.batch_size))
    index_tensor = torch.tensor(sample_indices, dtype=torch.int64)
    input_lengths = manifest.input_lengths.index_select(0, index_tensor)
    transport_width = int(input_lengths.max().item())
    input_ids = torch.arange(
        len(sample_indices) * transport_width,
        dtype=torch.int64,
    ).reshape(len(sample_indices), transport_width)
    return TokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=chunk_id,
        sample_ids=manifest.sample_ids.index_select(0, index_tensor),
        sample_order=manifest.sample_order.index_select(0, index_tensor),
        update_group=manifest.update_group.index_select(0, index_tensor),
        global_batch_slot=manifest.global_batch_slot.index_select(0, index_tensor),
        input_ids=input_ids,
        input_lengths=input_lengths,
        sample_mask=manifest.sample_mask.index_select(0, index_tensor),
        loss_spans=slice_span_table(manifest.loss_spans, index_tensor),
    )


def _teacher_chunk(
    manifest: BatchManifest,
    *,
    chunk_id: int = 0,
    sample_indices: list[int] | None = None,
    positions_by_sample: list[list[int]] | None = None,
) -> TeacherTopKChunk:
    sample_indices = sample_indices or list(range(manifest.batch_size))
    index_tensor = torch.tensor(sample_indices, dtype=torch.int64)
    if positions_by_sample is None:
        positions_by_sample = []
        offsets = manifest.loss_spans.offsets.tolist()
        starts = manifest.loss_spans.starts.tolist()
        ends = manifest.loss_spans.ends.tolist()
        for sample_index in sample_indices:
            sample_positions = []
            for span_index in range(offsets[sample_index], offsets[sample_index + 1]):
                sample_positions.extend(range(starts[span_index], ends[span_index]))
            positions_by_sample.append(sample_positions)

    offsets = [0]
    positions = []
    for sample_positions in positions_by_sample:
        positions.extend(sample_positions)
        offsets.append(len(positions))

    topk_width = 2
    row_count = len(positions)
    return TeacherTopKChunk(
        batch_id=manifest.batch_id,
        chunk_id=chunk_id,
        sample_ids=manifest.sample_ids.index_select(0, index_tensor),
        sample_order=manifest.sample_order.index_select(0, index_tensor),
        update_group=manifest.update_group.index_select(0, index_tensor),
        global_batch_slot=manifest.global_batch_slot.index_select(0, index_tensor),
        position_offsets=torch.tensor(offsets, dtype=torch.int64),
        positions=torch.tensor(positions, dtype=torch.int32),
        topk_logits=torch.zeros((row_count, topk_width), dtype=torch.float32),
        topk_indices=torch.zeros((row_count, topk_width), dtype=torch.int64),
    )


def test_step_manifest_rejects_duplicate_sample_ids():
    with pytest.raises(AssertionError, match="sample_ids must be unique"):
        StepManifest(
            batch_id="step-0",
            step=1,
            sample_ids=torch.tensor([10, 10], dtype=torch.int64),
            sample_order=torch.tensor([0, 1], dtype=torch.int64),
            update_group=torch.tensor([0, 0], dtype=torch.int64),
            global_batch_slot=torch.tensor([0, 1], dtype=torch.int64),
            prompt_ids=torch.tensor([0, 1], dtype=torch.int64),
            generation_ids=torch.tensor([0, 0], dtype=torch.int32),
            max_sequence_length=8,
            tokenizer_name_or_path="unit-test-tokenizer",
            tokenizer_config={},
        )


def test_batch_manifest_rejects_out_of_bounds_loss_spans():
    with pytest.raises(AssertionError, match="must satisfy 0 <= start < end"):
        _batch_manifest(
            input_lengths=[4],
            loss_spans=SpanTable(
                offsets=torch.tensor([0, 1], dtype=torch.int64),
                starts=torch.tensor([0], dtype=torch.int32),
                ends=torch.tensor([4], dtype=torch.int32),
            ),
            sample_ids=[5],
            sample_order=[0],
            update_group=[0],
            global_batch_slot=[0],
        )


def test_validate_chunk_budgets_rejects_token_and_byte_limits():
    manifest = _batch_manifest(input_lengths=[5, 4])
    chunk = _token_chunk(manifest)

    token_limited = parse_data_pipeline_config({"max_chunk_tokens": 8})
    with pytest.raises(AssertionError, match="max_chunk_tokens=8"):
        validate_chunk_budgets(chunk, token_limited, context="token_chunk")

    byte_limited = parse_data_pipeline_config(
        {"max_chunk_bytes": 16, "max_chunk_tokens": 64}
    )
    with pytest.raises(AssertionError, match="max_chunk_bytes=16"):
        validate_chunk_budgets(chunk, byte_limited, context="token_chunk")


def test_validate_chunk_budgets_rejects_excess_loss_positions():
    manifest = _batch_manifest(input_lengths=[5, 4])
    token_chunk = _token_chunk(manifest)
    teacher_chunk = _teacher_chunk(manifest)
    loss_limited = parse_data_pipeline_config(
        {
            "max_chunk_tokens": 64,
            "max_chunk_loss_positions": 3,
        }
    )

    with pytest.raises(AssertionError, match="max_chunk_loss_positions=3"):
        validate_chunk_budgets(token_chunk, loss_limited, context="token_chunk")

    with pytest.raises(AssertionError, match="max_chunk_loss_positions=3"):
        validate_chunk_budgets(teacher_chunk, loss_limited, context="teacher_chunk")


def test_loss_spans_from_token_mask_uses_next_token_positions_and_sample_mask():
    spans = loss_spans_from_token_mask(
        token_mask=torch.tensor(
            [
                [0, 1, 1, 0, 1, 1],
                [0, 1, 1, 1, 0, 0],
            ],
            dtype=torch.int64,
        ),
        input_lengths=torch.tensor([5, 4], dtype=torch.int32),
        sample_mask=torch.tensor([1, 0], dtype=torch.float32),
    )

    assert spans.offsets.tolist() == [0, 2, 2]
    assert spans.starts.tolist() == [0, 3]
    assert spans.ends.tolist() == [2, 4]


def test_build_repeated_sample_metadata_matches_shard_grouping():
    metadata = build_repeated_sample_metadata(
        step=3,
        prompt_count=4,
        num_generations_per_prompt=2,
        train_global_batch_size=4,
        step_sample_stride=8,
    )

    assert metadata["sample_ids"].tolist() == [24, 25, 26, 27, 28, 29, 30, 31]
    assert metadata["update_group"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert metadata["global_batch_slot"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]

    batch = BatchedDataDict(
        {
            "sample_order": metadata["sample_order"],
            "update_group": metadata["update_group"],
            "global_batch_slot": metadata["global_batch_slot"],
        }
    )
    shards = batch.shard_by_batch_size(shards=2, batch_size=4)

    assert shards[0]["sample_order"].tolist() == [0, 1, 4, 5]
    assert shards[0]["update_group"].tolist() == [0, 0, 1, 1]
    assert shards[0]["global_batch_slot"].tolist() == [0, 1, 0, 1]
    assert shards[1]["sample_order"].tolist() == [2, 3, 6, 7]
    assert shards[1]["update_group"].tolist() == [0, 0, 1, 1]
    assert shards[1]["global_batch_slot"].tolist() == [2, 3, 2, 3]


def test_manifest_builders_attach_metadata_and_build_loss_spans():
    step_manifest = build_step_manifest(
        batch_id="batch-3",
        step=3,
        prompt_count=2,
        num_generations_per_prompt=2,
        train_global_batch_size=2,
        max_sequence_length=8,
        tokenizer_name_or_path="unit-test-tokenizer",
        tokenizer_config={"padding_side": "right"},
        configured_prompts_per_step=2,
    )
    batch = {}
    attach_step_metadata(batch, step_manifest)

    assert sorted(batch) == sorted(STREAM_METADATA_KEYS)
    assert batch["prompt_ids"].tolist() == [0, 0, 1, 1]
    assert batch["generation_ids"].tolist() == [0, 1, 0, 1]

    train_data = {
        "input_lengths": torch.tensor([4, 3, 4, 2], dtype=torch.int32),
        "sample_mask": torch.ones(4, dtype=torch.bool),
        "token_mask": torch.tensor(
            [
                [0, 1, 1, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 1, 0, 0],
            ],
            dtype=torch.int64,
        ),
    }
    batch_manifest = build_batch_manifest_from_train_data(
        batch_id=step_manifest.batch_id,
        step=step_manifest.step,
        train_data=train_data,
        metadata=batch,
        max_sequence_length=8,
        tokenizer_name_or_path="unit-test-tokenizer",
        tokenizer_config={},
    )

    assert batch_manifest.sample_ids.tolist() == step_manifest.sample_ids.tolist()
    assert batch_manifest.loss_spans.offsets.tolist() == [0, 1, 2, 4, 5]
    assert batch_manifest.loss_spans.starts.tolist() == [0, 1, 0, 2, 0]
    assert batch_manifest.loss_spans.ends.tolist() == [3, 2, 1, 3, 1]


def test_annotated_token_chunk_validates_eos_marker_shape():
    eos_marker = AnnotatedTokenChunk(
        batch_id="batch-0",
        chunk_id=9,
        token_chunk_ref=None,
        teacher_topk_ref=None,
        end_of_stream=True,
    )
    assert eos_marker.end_of_stream is True

    with pytest.raises(AssertionError, match="EOS markers must not carry payload references"):
        AnnotatedTokenChunk(
            batch_id="batch-0",
            chunk_id=10,
            token_chunk_ref=object(),
            teacher_topk_ref=None,
            end_of_stream=True,
        )


def test_dp_reshard_planner_preserves_order_and_update_groups():
    manifest = _batch_manifest(
        sample_ids=[10, 11, 12, 13, 14, 15],
        sample_order=[0, 1, 2, 3, 4, 5],
        update_group=[0, 0, 1, 1, 2, 2],
        global_batch_slot=[0, 1, 0, 1, 0, 1],
        input_lengths=[8, 3, 4, 4, 6, 2],
    )
    planner = DPReshardPlanner(mode="balance_loss_tokens")
    plan = planner.plan(
        manifest=manifest,
        src_dp_size=2,
        dst_dp_size=2,
        max_chunk_tokens=32,
    )

    assignment_by_group = {}
    for dst_shard, assignments in enumerate(plan):
        for assignment in assignments:
            observed_group = int(assignment.update_group)
            assignment_by_group.setdefault(observed_group, set()).add(dst_shard)
            assert assignment.sample_order.tolist() == sorted(assignment.sample_order.tolist())

    assert assignment_by_group == {0: {0}, 1: {1}, 2: {1}}

    flattened = [
        (assignment.update_group, assignment.sample_ids.tolist(), assignment.global_batch_slot.tolist())
        for assignments in plan
        for assignment in assignments
    ]
    assert flattened[0] == (0, [10], [0])
    assert flattened[1] == (0, [11], [1])
    assert flattened[2] == (1, [12], [0])
    assert flattened[3] == (1, [13], [1])
    assert flattened[4] == (2, [14], [0])
    assert flattened[5] == (2, [15], [1])


def test_dp_reshard_planner_splits_update_group_by_source_shard():
    manifest = _batch_manifest(
        sample_ids=[10, 11, 12, 13],
        sample_order=[0, 1, 2, 3],
        update_group=[0, 0, 0, 0],
        global_batch_slot=[0, 1, 2, 3],
        input_lengths=[2, 2, 2, 2],
    )
    planner = DPReshardPlanner(mode="preserve_update_groups")
    plan = planner.plan(
        manifest=manifest,
        src_dp_size=2,
        dst_dp_size=1,
        max_chunk_tokens=8,
    )

    assignments = plan[0]
    assert [assignment.src_shard for assignment in assignments] == [0, 1]
    assert [assignment.sample_ids.tolist() for assignment in assignments] == [
        [10, 11],
        [12, 13],
    ]


def test_dp_reshard_planner_can_preserve_source_shards():
    manifest = _batch_manifest(
        sample_ids=[10, 11, 12, 13],
        sample_order=[0, 1, 2, 3],
        update_group=[0, 0, 0, 0],
        global_batch_slot=[0, 1, 2, 3],
        input_lengths=[2, 2, 2, 2],
    )
    planner = DPReshardPlanner(mode="preserve_source_shards")
    plan = planner.plan(
        manifest=manifest,
        src_dp_size=4,
        dst_dp_size=4,
        max_chunk_tokens=8,
    )

    assert [
        [assignment.sample_ids.tolist() for assignment in assignments]
        for assignments in plan
    ] == [[[10]], [[11]], [[12]], [[13]]]


def test_token_stream_builder_emits_budget_checked_token_chunks():
    manifest = _batch_manifest(
        sample_ids=[10, 11, 12, 13],
        sample_order=[0, 1, 2, 3],
        update_group=[0, 0, 0, 0],
        global_batch_slot=[0, 1, 2, 3],
        input_lengths=[2, 2, 2, 2],
    )
    train_data = {
        "input_ids": torch.arange(16, dtype=torch.int64).reshape(4, 4),
    }

    stream = build_token_stream_from_train_data(
        manifest=manifest,
        train_data=train_data,
        data_pipeline_config=parse_data_pipeline_config({"max_chunk_tokens": 8}),
        src_dp_size=2,
        dst_dp_size=1,
    )

    assert stream.batch_id == manifest.batch_id
    assert stream.dp_size == 1
    assert [chunk.sample_ids.tolist() for chunk in stream.shard_streams[0]] == [
        [10, 11],
        [12, 13],
    ]


def test_teacher_chunk_from_dense_uses_loss_positions_and_collects_for_parity():
    loss_spans = SpanTable(
        offsets=torch.tensor([0, 2, 3], dtype=torch.int64),
        starts=torch.tensor([0, 2, 1], dtype=torch.int32),
        ends=torch.tensor([1, 3, 2], dtype=torch.int32),
    )
    manifest = _batch_manifest(
        sample_ids=[20, 21],
        sample_order=[0, 1],
        update_group=[0, 0],
        global_batch_slot=[0, 1],
        input_lengths=[4, 3],
        loss_spans=loss_spans,
    )
    token_chunk = TokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=0,
        sample_ids=manifest.sample_ids,
        sample_order=manifest.sample_order,
        update_group=manifest.update_group,
        global_batch_slot=manifest.global_batch_slot,
        input_ids=torch.arange(8, dtype=torch.int64).reshape(2, 4),
        input_lengths=manifest.input_lengths,
        sample_mask=manifest.sample_mask,
        loss_spans=manifest.loss_spans,
    )
    topk_logits = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    topk_indices = torch.arange(100, 116, dtype=torch.int64).reshape(2, 4, 2)

    teacher_chunk = build_teacher_topk_chunk_from_dense(
        token_chunk=token_chunk,
        topk_logits=topk_logits,
        topk_indices=topk_indices,
    )

    assert teacher_chunk.positions.tolist() == [0, 2, 1]
    torch.testing.assert_close(
        teacher_chunk.topk_logits,
        torch.stack([topk_logits[0, 0], topk_logits[0, 2], topk_logits[1, 1]]),
    )
    build_conservation_oracle(manifest).validate_teacher_chunks([teacher_chunk])

    dense = collect_teacher_topk_chunks_to_dense(
        manifest=manifest,
        teacher_chunks=[teacher_chunk],
        sequence_length=4,
    )
    torch.testing.assert_close(dense["teacher_topk_logits"][0, 0], topk_logits[0, 0])
    torch.testing.assert_close(dense["teacher_topk_logits"][0, 2], topk_logits[0, 2])
    torch.testing.assert_close(dense["teacher_topk_logits"][1, 1], topk_logits[1, 1])
    torch.testing.assert_close(dense["teacher_topk_indices"][0, 0], topk_indices[0, 0])


def test_iter_annotated_teacher_topk_chunks_yields_refs_and_eos():
    manifest = _batch_manifest(sample_ids=[30, 31], input_lengths=[4, 4])
    token_chunk = _token_chunk(manifest)
    calls = []

    def fake_get_topk_logits(data, k, micro_batch_size):
        calls.append((data, k, micro_batch_size))
        return {
            "topk_logits": torch.ones((2, 4, 2), dtype=torch.float32),
            "topk_indices": torch.zeros((2, 4, 2), dtype=torch.int64),
        }

    outputs = list(
        iter_annotated_teacher_topk_chunks(
            token_chunks=[token_chunk],
            get_topk_logits=fake_get_topk_logits,
            k=2,
            micro_batch_size=8,
            put_fn=lambda value: value,
        )
    )

    assert calls[0][1:] == (2, 2)
    assert outputs[0].token_chunk_ref is token_chunk
    assert isinstance(outputs[0].teacher_topk_ref, TeacherTopKChunk)
    assert outputs[1].end_of_stream is True


def test_iter_annotated_teacher_topk_chunks_uses_batch_id_for_empty_shard_eos():
    outputs = list(
        iter_annotated_teacher_topk_chunks(
            token_chunks=[],
            get_topk_logits=lambda **_kwargs: pytest.fail(
                "empty streams should not call get_topk_logits"
            ),
            k=2,
            batch_id="batch-empty",
            put_fn=lambda value: value,
        )
    )

    assert len(outputs) == 1
    assert outputs[0].batch_id == "batch-empty"
    assert outputs[0].end_of_stream is True


def test_drain_and_route_annotated_stream_to_student_shards():
    manifest = _batch_manifest(
        sample_ids=[40, 41, 42, 43],
        sample_order=[0, 1, 2, 3],
        update_group=[0, 0, 0, 0],
        global_batch_slot=[0, 1, 2, 3],
        input_lengths=[4, 4, 4, 4],
    )
    token_chunk = _token_chunk(manifest)
    teacher_chunk = _teacher_chunk(manifest)
    envelope = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=0,
        token_chunk_ref=token_chunk,
        teacher_topk_ref=teacher_chunk,
        sample_ids=token_chunk.sample_ids,
        sample_order=token_chunk.sample_order,
        update_group=token_chunk.update_group,
        global_batch_slot=token_chunk.global_batch_slot,
    )
    eos = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=1,
        token_chunk_ref=None,
        teacher_topk_ref=None,
        end_of_stream=True,
    )
    stream = ShardedBatchStream(
        batch_id=manifest.batch_id,
        layout="dp_fullseq",
        dp_size=1,
        manifest=manifest,
        shard_streams=[[envelope, eos]],
    )

    drained = drain_annotated_stream(stream, get_fn=lambda value: value)
    routed = route_drained_stream_to_student_shards(
        drained,
        train_global_batch_size=4,
        student_dp_size=2,
    )

    assert routed.dp_size == 2
    assert routed.shard_streams[0] == [envelope]
    assert routed.shard_streams[1] == [envelope]

    routed_by_samples = route_drained_stream_to_student_shards_by_sample_ids(
        drained,
        sharded_sample_ids=[
            torch.tensor([40, 42], dtype=torch.int64),
            torch.tensor([41, 43], dtype=torch.int64),
        ],
    )
    assert routed_by_samples.shard_streams[0] == [envelope]
    assert routed_by_samples.shard_streams[1] == [envelope]


def test_drain_annotated_stream_consumes_shards_concurrently():
    manifest = _batch_manifest(
        sample_ids=[40, 41],
        sample_order=[0, 1],
        update_group=[0, 0],
        global_batch_slot=[0, 1],
        input_lengths=[4, 4],
    )
    first_token_chunk = _token_chunk(manifest, chunk_id=0, sample_indices=[0])
    second_token_chunk = _token_chunk(manifest, chunk_id=1, sample_indices=[1])
    first_teacher_chunk = _teacher_chunk(manifest, chunk_id=0, sample_indices=[0])
    second_teacher_chunk = _teacher_chunk(manifest, chunk_id=1, sample_indices=[1])
    first_envelope = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=0,
        token_chunk_ref=first_token_chunk,
        teacher_topk_ref=first_teacher_chunk,
        sample_ids=first_token_chunk.sample_ids,
        sample_order=first_token_chunk.sample_order,
        update_group=first_token_chunk.update_group,
        global_batch_slot=first_token_chunk.global_batch_slot,
    )
    second_envelope = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=1,
        token_chunk_ref=second_token_chunk,
        teacher_topk_ref=second_teacher_chunk,
        sample_ids=second_token_chunk.sample_ids,
        sample_order=second_token_chunk.sample_order,
        update_group=second_token_chunk.update_group,
        global_batch_slot=second_token_chunk.global_batch_slot,
    )
    first_eos = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=2,
        token_chunk_ref=None,
        teacher_topk_ref=None,
        end_of_stream=True,
    )
    second_eos = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=2,
        token_chunk_ref=None,
        teacher_topk_ref=None,
        end_of_stream=True,
    )
    barrier = threading.Barrier(2, timeout=1.0)

    def blocking_stream(envelope, eos):
        barrier.wait()
        yield envelope
        yield eos

    stream = ShardedBatchStream(
        batch_id=manifest.batch_id,
        layout="dp_fullseq",
        dp_size=2,
        manifest=manifest,
        shard_streams=[
            blocking_stream(first_envelope, first_eos),
            blocking_stream(second_envelope, second_eos),
        ],
    )

    drained = drain_annotated_stream(stream, get_fn=lambda value: value)

    assert drained.shard_streams[0][0] is first_envelope
    assert drained.shard_streams[1][0] is second_envelope


def test_drain_annotated_stream_rejects_stale_batch_id():
    manifest = _batch_manifest(input_lengths=[4])
    token_chunk = _token_chunk(manifest)
    teacher_chunk = _teacher_chunk(manifest)
    stale_envelope = AnnotatedTokenChunk(
        batch_id="stale-batch",
        chunk_id=0,
        token_chunk_ref=token_chunk,
        teacher_topk_ref=teacher_chunk,
        sample_ids=token_chunk.sample_ids,
        sample_order=token_chunk.sample_order,
        update_group=token_chunk.update_group,
        global_batch_slot=token_chunk.global_batch_slot,
    )
    stream = ShardedBatchStream(
        batch_id=manifest.batch_id,
        layout="dp_fullseq",
        dp_size=1,
        manifest=manifest,
        shard_streams=[[stale_envelope]],
    )

    with pytest.raises(AssertionError, match="batch_id must match stream batch_id"):
        drain_annotated_stream(stream, get_fn=lambda value: value)


def test_drain_annotated_stream_requires_terminal_eos_and_no_trailing_payload():
    manifest = _batch_manifest(sample_ids=[40], input_lengths=[4])
    token_chunk = _token_chunk(manifest)
    teacher_chunk = _teacher_chunk(manifest)
    envelope = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=0,
        token_chunk_ref=token_chunk,
        teacher_topk_ref=teacher_chunk,
        sample_ids=token_chunk.sample_ids,
        sample_order=token_chunk.sample_order,
        update_group=token_chunk.update_group,
        global_batch_slot=token_chunk.global_batch_slot,
    )
    eos = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=1,
        token_chunk_ref=None,
        teacher_topk_ref=None,
        end_of_stream=True,
    )
    stream_without_eos = ShardedBatchStream(
        batch_id=manifest.batch_id,
        layout="dp_fullseq",
        dp_size=1,
        manifest=manifest,
        shard_streams=[[envelope]],
    )
    with pytest.raises(AssertionError, match="without an EOS marker"):
        drain_annotated_stream(stream_without_eos, get_fn=lambda value: value)

    stream_with_payload_after_eos = ShardedBatchStream(
        batch_id=manifest.batch_id,
        layout="dp_fullseq",
        dp_size=1,
        manifest=manifest,
        shard_streams=[[eos, envelope]],
    )
    with pytest.raises(AssertionError, match="payload after EOS marker"):
        drain_annotated_stream(stream_with_payload_after_eos, get_fn=lambda value: value)


def test_stream_routing_rejects_invalid_boundaries():
    manifest = _batch_manifest(
        sample_ids=[40, 41],
        sample_order=[0, 1],
        update_group=[0, 0],
        global_batch_slot=[0, 1],
        input_lengths=[4, 4],
    )
    token_chunk = _token_chunk(manifest)
    teacher_chunk = _teacher_chunk(manifest)
    envelope = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=0,
        token_chunk_ref=token_chunk,
        teacher_topk_ref=teacher_chunk,
        sample_ids=token_chunk.sample_ids,
        sample_order=token_chunk.sample_order,
        update_group=token_chunk.update_group,
        global_batch_slot=token_chunk.global_batch_slot,
    )
    drained = ShardedBatchStream(
        batch_id=manifest.batch_id,
        layout="dp_fullseq",
        dp_size=1,
        manifest=manifest,
        shard_streams=[[envelope]],
    )

    with pytest.raises(AssertionError, match="must be divisible"):
        route_drained_stream_to_student_shards(
            drained,
            train_global_batch_size=3,
            student_dp_size=2,
        )

    with pytest.raises(AssertionError, match="multiple student shards"):
        route_drained_stream_to_student_shards_by_sample_ids(
            drained,
            sharded_sample_ids=[
                torch.tensor([40], dtype=torch.int64),
                torch.tensor([40], dtype=torch.int64),
            ],
        )

    with pytest.raises(AssertionError, match="missing from student shards"):
        route_drained_stream_to_student_shards_by_sample_ids(
            drained,
            sharded_sample_ids=[
                torch.tensor([140], dtype=torch.int64),
                torch.tensor([141], dtype=torch.int64),
            ],
        )

    with pytest.raises(AssertionError, match="missing from student shards"):
        route_drained_stream_to_student_shards_by_sample_ids(
            drained,
            sharded_sample_ids=[
                torch.tensor([40], dtype=torch.int64),
                torch.tensor([141], dtype=torch.int64),
            ],
        )


def test_stream_routing_rejects_eos_chunks():
    manifest = _batch_manifest(sample_ids=[40], input_lengths=[4])
    eos = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=0,
        token_chunk_ref=None,
        teacher_topk_ref=None,
        end_of_stream=True,
    )
    drained = ShardedBatchStream(
        batch_id=manifest.batch_id,
        layout="dp_fullseq",
        dp_size=1,
        manifest=manifest,
        shard_streams=[[eos]],
    )

    with pytest.raises(AssertionError, match="EOS chunks"):
        route_drained_stream_to_student_shards(
            drained,
            train_global_batch_size=1,
            student_dp_size=1,
        )

    with pytest.raises(AssertionError, match="EOS chunks"):
        route_drained_stream_to_student_shards_by_sample_ids(
            drained,
            sharded_sample_ids=[torch.tensor([40], dtype=torch.int64)],
        )


def test_assemble_dense_distillation_batch_filters_student_shard_rows():
    manifest = _batch_manifest(
        sample_ids=[50, 51, 52, 53],
        sample_order=[0, 1, 2, 3],
        update_group=[0, 0, 0, 0],
        global_batch_slot=[0, 1, 2, 3],
        input_lengths=[4, 4, 4, 4],
    )
    token_chunk = _token_chunk(manifest)
    topk_logits = torch.arange(32, dtype=torch.float32).reshape(4, 4, 2)
    topk_indices = torch.arange(100, 132, dtype=torch.int64).reshape(4, 4, 2)
    teacher_chunk = build_teacher_topk_chunk_from_dense(
        token_chunk=token_chunk,
        topk_logits=topk_logits,
        topk_indices=topk_indices,
    )
    envelope = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=0,
        token_chunk_ref=token_chunk,
        teacher_topk_ref=teacher_chunk,
        sample_ids=token_chunk.sample_ids,
        sample_order=token_chunk.sample_order,
        update_group=token_chunk.update_group,
        global_batch_slot=token_chunk.global_batch_slot,
    )

    dense = assemble_dense_distillation_batch_from_annotated_chunks(
        [envelope],
        train_global_batch_size=4,
        student_dp_size=2,
        student_dp_rank=1,
        get_fn=lambda value: value,
    )

    assert dense["input_ids"].shape == (2, 4)
    torch.testing.assert_close(dense["input_ids"], token_chunk.input_ids[2:4])
    torch.testing.assert_close(
        dense["teacher_topk_logits"][0, 0],
        topk_logits[2, 0],
    )
    torch.testing.assert_close(
        dense["teacher_topk_indices"][1, 2],
        topk_indices[3, 2],
    )
    assert dense["token_mask"][:, 0].tolist() == [False, False]
    assert dense["token_mask"][:, 1:].all().item()


def test_assemble_dense_distillation_batch_rejects_stale_resolved_refs():
    manifest = _batch_manifest(sample_ids=[50], input_lengths=[4])
    token_chunk = _token_chunk(manifest)
    stale_token_chunk = replace(token_chunk, batch_id="stale-batch")
    teacher_chunk = _teacher_chunk(manifest)
    envelope = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=0,
        token_chunk_ref=stale_token_chunk,
        teacher_topk_ref=teacher_chunk,
        sample_ids=token_chunk.sample_ids,
        sample_order=token_chunk.sample_order,
        update_group=token_chunk.update_group,
        global_batch_slot=token_chunk.global_batch_slot,
    )

    with pytest.raises(AssertionError, match="token_chunk batch_id"):
        assemble_dense_distillation_batch_from_annotated_chunks(
            [envelope],
            train_global_batch_size=1,
            student_dp_size=1,
            student_dp_rank=0,
            get_fn=lambda value: value,
        )

    stale_teacher_chunk = replace(teacher_chunk, batch_id="stale-batch")
    envelope = replace(
        envelope,
        token_chunk_ref=token_chunk,
        teacher_topk_ref=stale_teacher_chunk,
    )
    with pytest.raises(AssertionError, match="teacher_topk chunk batch_id"):
        assemble_dense_distillation_batch_from_annotated_chunks(
            [envelope],
            train_global_batch_size=1,
            student_dp_size=1,
            student_dp_rank=0,
            get_fn=lambda value: value,
        )


def test_assemble_dense_distillation_batch_rejects_wrong_same_batch_refs():
    manifest = _batch_manifest(sample_ids=[50, 51], input_lengths=[4, 4])
    token_chunk = _token_chunk(manifest, chunk_id=0, sample_indices=[0])
    teacher_chunk = _teacher_chunk(manifest, chunk_id=0, sample_indices=[0])
    wrong_chunk_id_teacher = _teacher_chunk(
        manifest,
        chunk_id=1,
        sample_indices=[0],
    )
    envelope = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=0,
        token_chunk_ref=token_chunk,
        teacher_topk_ref=wrong_chunk_id_teacher,
        sample_ids=token_chunk.sample_ids,
        sample_order=token_chunk.sample_order,
        update_group=token_chunk.update_group,
        global_batch_slot=token_chunk.global_batch_slot,
    )

    with pytest.raises(AssertionError, match="chunk_id"):
        assemble_dense_distillation_batch_from_annotated_chunks(
            [envelope],
            train_global_batch_size=2,
            student_dp_size=1,
            student_dp_rank=0,
            get_fn=lambda value: value,
        )

    wrong_metadata_teacher = replace(
        teacher_chunk,
        sample_ids=torch.tensor([999], dtype=torch.int64),
    )
    envelope = replace(envelope, teacher_topk_ref=wrong_metadata_teacher)
    with pytest.raises(AssertionError, match="sample_ids"):
        assemble_dense_distillation_batch_from_annotated_chunks(
            [envelope],
            train_global_batch_size=2,
            student_dp_size=1,
            student_dp_rank=0,
            get_fn=lambda value: value,
        )


def test_assemble_dense_distillation_batch_reorders_multichunk_rows():
    manifest = _batch_manifest(
        sample_ids=[70, 71, 72, 73],
        sample_order=[0, 1, 2, 3],
        update_group=[0, 0, 0, 0],
        global_batch_slot=[0, 1, 2, 3],
        input_lengths=[4, 4, 4, 4],
    )
    chunks = [
        _token_chunk(manifest, chunk_id=0, sample_indices=[2, 3]),
        _token_chunk(manifest, chunk_id=1, sample_indices=[0, 1]),
    ]
    envelopes = []
    for chunk in chunks:
        teacher_chunk = build_teacher_topk_chunk_from_dense(
            token_chunk=chunk,
            topk_logits=torch.ones((2, 4, 2), dtype=torch.float32) * chunk.chunk_id,
            topk_indices=torch.ones((2, 4, 2), dtype=torch.int64) * chunk.chunk_id,
        )
        envelopes.append(
            AnnotatedTokenChunk(
                batch_id=manifest.batch_id,
                chunk_id=chunk.chunk_id,
                token_chunk_ref=chunk,
                teacher_topk_ref=teacher_chunk,
                sample_ids=chunk.sample_ids,
                sample_order=chunk.sample_order,
                update_group=chunk.update_group,
                global_batch_slot=chunk.global_batch_slot,
            )
        )

    dense = assemble_dense_distillation_batch_from_annotated_chunks(
        envelopes,
        train_global_batch_size=4,
        student_dp_size=2,
        student_dp_rank=0,
        get_fn=lambda value: value,
    )

    assert dense["input_ids"].tolist() == chunks[1].input_ids.tolist()
    assert dense["teacher_topk_logits"][0, 1, 0].item() == 1.0


def test_assemble_dense_distillation_batch_rejects_eos_chunk():
    manifest = _batch_manifest(sample_ids=[50], input_lengths=[4])
    eos = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=0,
        token_chunk_ref=None,
        teacher_topk_ref=None,
        end_of_stream=True,
    )

    with pytest.raises(AssertionError, match="EOS chunks"):
        assemble_dense_distillation_batch_from_annotated_chunks(
            [eos],
            train_global_batch_size=1,
            student_dp_size=1,
            student_dp_rank=0,
            get_fn=lambda value: value,
        )


def test_attach_teacher_topk_to_local_batch_preserves_local_row_order():
    manifest = _batch_manifest(
        sample_ids=[60, 61],
        sample_order=[0, 1],
        update_group=[0, 0],
        global_batch_slot=[0, 1],
        input_lengths=[4, 4],
    )
    token_chunk = _token_chunk(manifest)
    topk_logits = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    topk_indices = torch.arange(200, 216, dtype=torch.int64).reshape(2, 4, 2)
    teacher_chunk = build_teacher_topk_chunk_from_dense(
        token_chunk=token_chunk,
        topk_logits=topk_logits,
        topk_indices=topk_indices,
    )
    envelope = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=0,
        token_chunk_ref=token_chunk,
        teacher_topk_ref=teacher_chunk,
        sample_ids=token_chunk.sample_ids,
        sample_order=token_chunk.sample_order,
        update_group=token_chunk.update_group,
        global_batch_slot=token_chunk.global_batch_slot,
    )
    local_batch = BatchedDataDict(
        {
            "sample_ids": torch.tensor([61, 60], dtype=torch.int64),
            "input_ids": token_chunk.input_ids.flip(0),
            "input_lengths": token_chunk.input_lengths.flip(0),
            "token_mask": torch.ones((2, 4), dtype=torch.bool),
            "sample_mask": torch.ones(2, dtype=torch.bool),
        }
    )

    attached = attach_teacher_topk_to_local_batch_from_annotated_chunks(
        local_batch,
        [envelope],
        get_fn=lambda value: value,
    )

    torch.testing.assert_close(
        attached["teacher_topk_logits"][0, 0],
        topk_logits[1, 0],
    )
    torch.testing.assert_close(
        attached["teacher_topk_logits"][1, 2],
        topk_logits[0, 2],
    )


def test_attach_teacher_topk_rejects_stale_and_eos_chunks():
    manifest = _batch_manifest(sample_ids=[60], input_lengths=[4])
    token_chunk = _token_chunk(manifest)
    teacher_chunk = _teacher_chunk(manifest)
    local_batch = BatchedDataDict(
        {
            "sample_ids": torch.tensor([60], dtype=torch.int64),
            "input_ids": token_chunk.input_ids,
            "input_lengths": token_chunk.input_lengths,
            "token_mask": torch.ones((1, 4), dtype=torch.bool),
            "sample_mask": torch.ones(1, dtype=torch.bool),
        }
    )
    envelope = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=0,
        token_chunk_ref=token_chunk,
        teacher_topk_ref=replace(teacher_chunk, batch_id="stale-batch"),
        sample_ids=token_chunk.sample_ids,
        sample_order=token_chunk.sample_order,
        update_group=token_chunk.update_group,
        global_batch_slot=token_chunk.global_batch_slot,
    )

    with pytest.raises(AssertionError, match="teacher_topk chunk batch_id"):
        attach_teacher_topk_to_local_batch_from_annotated_chunks(
            local_batch.copy(),
            [envelope],
            get_fn=lambda value: value,
        )

    envelope = replace(
        envelope,
        teacher_topk_ref=replace(
            teacher_chunk,
            sample_ids=torch.tensor([999], dtype=torch.int64),
        ),
    )
    with pytest.raises(AssertionError, match="sample_ids"):
        attach_teacher_topk_to_local_batch_from_annotated_chunks(
            local_batch.copy(),
            [envelope],
            get_fn=lambda value: value,
        )

    eos = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=1,
        token_chunk_ref=None,
        teacher_topk_ref=None,
        end_of_stream=True,
    )
    with pytest.raises(AssertionError, match="EOS chunks"):
        attach_teacher_topk_to_local_batch_from_annotated_chunks(
            local_batch.copy(),
            [eos],
            get_fn=lambda value: value,
        )


def test_attach_sparse_teacher_topk_to_local_batch_uses_active_positions_only():
    manifest = _batch_manifest(
        sample_ids=[80, 81],
        sample_order=[0, 1],
        update_group=[0, 0],
        global_batch_slot=[0, 1],
        input_lengths=[4, 4],
    )
    token_chunk = _token_chunk(manifest)
    topk_logits = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    topk_indices = torch.arange(300, 316, dtype=torch.int64).reshape(2, 4, 2)
    teacher_chunk = build_teacher_topk_chunk_from_dense(
        token_chunk=token_chunk,
        topk_logits=topk_logits,
        topk_indices=topk_indices,
    )
    envelope = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=0,
        token_chunk_ref=token_chunk,
        teacher_topk_ref=teacher_chunk,
        sample_ids=token_chunk.sample_ids,
        sample_order=token_chunk.sample_order,
        update_group=token_chunk.update_group,
        global_batch_slot=token_chunk.global_batch_slot,
    )
    local_batch = BatchedDataDict(
        {
            "sample_ids": torch.tensor([81, 80], dtype=torch.int64),
            "input_ids": token_chunk.input_ids.flip(0),
            "input_lengths": token_chunk.input_lengths.flip(0),
            "token_mask": torch.ones((2, 4), dtype=torch.bool),
            "sample_mask": torch.ones(2, dtype=torch.bool),
        }
    )

    attached = attach_sparse_teacher_topk_to_local_batch_from_annotated_chunks(
        local_batch,
        [envelope],
        get_fn=lambda value: value,
    )

    assert "teacher_topk_logits" not in attached
    assert attached["teacher_topk_sparse_logits"].shape == (2, 3, 2)
    assert attached["teacher_topk_sparse_positions"].tolist() == [
        [0, 1, 2],
        [0, 1, 2],
    ]
    assert attached["teacher_topk_sparse_mask"].all().item()
    torch.testing.assert_close(
        attached["teacher_topk_sparse_logits"][0, 1],
        topk_logits[1, 1],
    )
    torch.testing.assert_close(
        attached["teacher_topk_sparse_indices"][1, 2],
        topk_indices[0, 2],
    )


def test_attach_sparse_teacher_topk_handles_no_loss_sample_padding():
    loss_spans = SpanTable(
        offsets=torch.tensor([0, 1, 1], dtype=torch.int64),
        starts=torch.tensor([1], dtype=torch.int32),
        ends=torch.tensor([3], dtype=torch.int32),
    )
    manifest = _batch_manifest(
        sample_ids=[90, 91],
        sample_order=[0, 1],
        update_group=[0, 0],
        global_batch_slot=[0, 1],
        input_lengths=[4, 4],
        loss_spans=loss_spans,
    )
    token_chunk = _token_chunk(manifest)
    topk_logits = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    topk_indices = torch.arange(400, 416, dtype=torch.int64).reshape(2, 4, 2)
    teacher_chunk = build_teacher_topk_chunk_from_dense(
        token_chunk=token_chunk,
        topk_logits=topk_logits,
        topk_indices=topk_indices,
    )
    envelope = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=0,
        token_chunk_ref=token_chunk,
        teacher_topk_ref=teacher_chunk,
        sample_ids=token_chunk.sample_ids,
        sample_order=token_chunk.sample_order,
        update_group=token_chunk.update_group,
        global_batch_slot=token_chunk.global_batch_slot,
    )
    local_batch = BatchedDataDict(
        {
            "sample_ids": torch.tensor([90, 91], dtype=torch.int64),
            "input_ids": token_chunk.input_ids,
            "input_lengths": token_chunk.input_lengths,
            "token_mask": torch.tensor(
                [[False, False, True, True], [False, False, False, False]]
            ),
            "sample_mask": torch.tensor([True, True]),
        }
    )

    attached = attach_sparse_teacher_topk_to_local_batch_from_annotated_chunks(
        local_batch,
        [envelope],
        get_fn=lambda value: value,
    )

    assert attached["teacher_topk_sparse_mask"].tolist() == [
        [True, True],
        [False, False],
    ]
    assert attached["teacher_topk_sparse_positions"].tolist() == [[1, 2], [0, 0]]
    torch.testing.assert_close(
        attached["teacher_topk_sparse_logits"][0, 0],
        topk_logits[0, 1],
    )
    torch.testing.assert_close(
        attached["teacher_topk_sparse_indices"][0, 1],
        topk_indices[0, 2],
    )


def test_attach_sparse_teacher_topk_rejects_stale_and_eos_chunks():
    manifest = _batch_manifest(sample_ids=[80], input_lengths=[4])
    token_chunk = _token_chunk(manifest)
    teacher_chunk = _teacher_chunk(manifest)
    local_batch = BatchedDataDict(
        {
            "sample_ids": torch.tensor([80], dtype=torch.int64),
            "input_ids": token_chunk.input_ids,
            "input_lengths": token_chunk.input_lengths,
            "token_mask": torch.ones((1, 4), dtype=torch.bool),
            "sample_mask": torch.ones(1, dtype=torch.bool),
        }
    )
    envelope = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=0,
        token_chunk_ref=token_chunk,
        teacher_topk_ref=replace(teacher_chunk, batch_id="stale-batch"),
        sample_ids=token_chunk.sample_ids,
        sample_order=token_chunk.sample_order,
        update_group=token_chunk.update_group,
        global_batch_slot=token_chunk.global_batch_slot,
    )

    with pytest.raises(AssertionError, match="teacher_topk chunk batch_id"):
        attach_sparse_teacher_topk_to_local_batch_from_annotated_chunks(
            local_batch.copy(),
            [envelope],
            get_fn=lambda value: value,
        )

    envelope = replace(
        envelope,
        teacher_topk_ref=replace(
            teacher_chunk,
            sample_ids=torch.tensor([999], dtype=torch.int64),
        ),
    )
    with pytest.raises(AssertionError, match="sample_ids"):
        attach_sparse_teacher_topk_to_local_batch_from_annotated_chunks(
            local_batch.copy(),
            [envelope],
            get_fn=lambda value: value,
        )

    eos = AnnotatedTokenChunk(
        batch_id=manifest.batch_id,
        chunk_id=1,
        token_chunk_ref=None,
        teacher_topk_ref=None,
        end_of_stream=True,
    )
    with pytest.raises(AssertionError, match="EOS chunks"):
        attach_sparse_teacher_topk_to_local_batch_from_annotated_chunks(
            local_batch.copy(),
            [eos],
            get_fn=lambda value: value,
        )


@pytest.mark.parametrize(
    ("chunks", "expected_message"),
    [
        (
            [
                _token_chunk(_batch_manifest(sample_ids=[20, 21, 22]), chunk_id=0, sample_indices=[0, 1]),
                _token_chunk(_batch_manifest(sample_ids=[20, 21, 22]), chunk_id=1, sample_indices=[1]),
            ],
            r"duplicate=\[21\]",
        ),
        (
            [
                _token_chunk(_batch_manifest(sample_ids=[20, 21, 22]), chunk_id=0, sample_indices=[0, 1]),
            ],
            r"missing=\[22\]",
        ),
    ],
)
def test_conservation_oracle_detects_duplicate_and_missing_samples(
    chunks, expected_message
):
    manifest = _batch_manifest(sample_ids=[20, 21, 22], input_lengths=[5, 4, 3])
    oracle = build_conservation_oracle(manifest)

    with pytest.raises(AssertionError, match=expected_message):
        oracle.validate_token_chunks(chunks)


def test_conservation_oracle_detects_position_and_metadata_mismatch():
    manifest = _batch_manifest(sample_ids=[20, 21], input_lengths=[4, 3])
    oracle = build_conservation_oracle(manifest)

    missing_position = _teacher_chunk(
        manifest,
        positions_by_sample=[[0, 1], [0, 1]],
    )
    with pytest.raises(AssertionError, match="missing=\\[\\(20, 2\\)\\]"):
        oracle.validate_teacher_chunks([missing_position])

    wrong_group = _teacher_chunk(manifest)
    wrong_group_update = wrong_group.update_group.clone()
    wrong_group_update[0] += 1
    wrong_group = TeacherTopKChunk(
        batch_id=wrong_group.batch_id,
        chunk_id=wrong_group.chunk_id,
        sample_ids=wrong_group.sample_ids,
        sample_order=wrong_group.sample_order,
        update_group=wrong_group_update,
        global_batch_slot=wrong_group.global_batch_slot,
        position_offsets=wrong_group.position_offsets,
        positions=wrong_group.positions,
        topk_logits=wrong_group.topk_logits,
        topk_indices=wrong_group.topk_indices,
    )
    with pytest.raises(AssertionError, match="metadata mismatch"):
        oracle.validate_teacher_chunks([wrong_group])

    with pytest.raises(AssertionError, match="metadata mismatch"):
        oracle.validate_student_boundary(
            manifest.sample_ids,
            manifest.sample_order,
            manifest.update_group + 1,
            manifest.global_batch_slot,
        )

    with pytest.raises(AssertionError, match="canonical sample_order"):
        oracle.validate_student_boundary(
            manifest.sample_ids.flip(0),
            manifest.sample_order.flip(0),
            manifest.update_group.flip(0),
            manifest.global_batch_slot.flip(0),
        )


def test_conservation_oracle_rejects_duplicate_chunk_ids():
    manifest = _batch_manifest(sample_ids=[20, 21], input_lengths=[5, 4])
    oracle = build_conservation_oracle(manifest)

    with pytest.raises(AssertionError, match="duplicate chunk_id"):
        oracle.validate_token_chunks(
            [
                _token_chunk(manifest, chunk_id=0, sample_indices=[0]),
                _token_chunk(manifest, chunk_id=0, sample_indices=[1]),
            ]
        )

    with pytest.raises(AssertionError, match="duplicate chunk_id"):
        oracle.validate_teacher_chunks(
            [
                _teacher_chunk(manifest, chunk_id=0, sample_indices=[0]),
                _teacher_chunk(manifest, chunk_id=0, sample_indices=[1]),
            ]
        )


def test_data_pipeline_config_defaults_and_validation():
    assert parse_data_pipeline_config(None).mode == "legacy"
    parsed = parse_data_pipeline_config(
        {
            "mode": "stream_teacher",
            "max_chunk_tokens": 1024,
            "max_chunk_bytes": 4096,
            "max_chunk_loss_positions": 512,
            "log_memory_metrics": False,
        }
    )
    assert parsed.mode == "stream_teacher"
    assert parsed.max_chunk_tokens == 1024
    assert parsed.max_chunk_bytes == 4096
    assert parsed.max_chunk_loss_positions == 512
    assert parsed.log_memory_metrics is False

    with pytest.raises(AssertionError, match="unsupported data pipeline mode"):
        parse_data_pipeline_config({"mode": "unknown"})

    with pytest.raises(AssertionError, match="max_chunk_tokens"):
        parse_data_pipeline_config({"max_chunk_tokens": 0})

    with pytest.raises(AssertionError, match="unknown distillation.data_pipeline keys"):
        parse_data_pipeline_config({"mode": "legacy", "surprise": True})

    with pytest.raises(AssertionError, match="unknown distillation.data_pipeline keys"):
        parse_data_pipeline_config({"mode": "legacy", "max_inflight_chunks": 2})


def test_validate_streaming_capabilities_rejects_unsupported_v1_combinations():
    base_policy = {
        "tokenizer": {"use_processor": False},
        "dynamic_batching": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "dtensor_cfg": {
            "enabled": True,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
        },
    }
    stream_config = parse_data_pipeline_config({"mode": "stream_teacher"})
    validate_streaming_capabilities(base_policy, base_policy, stream_config)

    multimodal_policy = {
        **base_policy,
        "tokenizer": {"use_processor": True},
    }
    with pytest.raises(AssertionError, match="multimodal OPD streaming"):
        validate_streaming_capabilities(multimodal_policy, base_policy, stream_config)

    with pytest.raises(AssertionError, match="multimodal OPD streaming"):
        validate_streaming_capabilities(
            base_policy,
            base_policy,
            stream_config,
            processor_present=True,
        )

    dynamic_megatron_policy = {
        **base_policy,
        "dtensor_cfg": {"enabled": False},
        "dynamic_batching": {"enabled": True},
        "megatron_cfg": {
            "enabled": True,
            "pipeline_model_parallel_size": 2,
        },
    }
    with pytest.raises(AssertionError, match="dynamic batching"):
        validate_streaming_capabilities(
            dynamic_megatron_policy,
            base_policy,
            stream_config,
        )

    mixed_batching_policy = {
        **base_policy,
        "dynamic_batching": {"enabled": True},
        "sequence_packing": {"enabled": True},
    }
    with pytest.raises(AssertionError, match="mutually exclusive"):
        validate_streaming_capabilities(
            mixed_batching_policy,
            base_policy,
            stream_config,
        )

    with pytest.raises(AssertionError, match="teacher dynamic batching"):
        validate_streaming_capabilities(
            base_policy,
            {**base_policy, "dynamic_batching": {"enabled": True}},
            stream_config,
        )

    validate_streaming_capabilities(
        base_policy,
        {**base_policy, "sequence_packing": {"enabled": True}},
        stream_config,
    )

    dtensor_cp_sequence_packing_policy = {
        **base_policy,
        "dtensor_cfg": {
            "enabled": True,
            "tensor_parallel_size": 1,
            "context_parallel_size": 2,
        },
        "sequence_packing": {"enabled": True},
    }
    with pytest.raises(AssertionError, match="DTensor context parallel"):
        validate_streaming_capabilities(
            dtensor_cp_sequence_packing_policy,
            base_policy,
            stream_config,
        )

    megatron_cp_without_sequence_packing_policy = {
        **base_policy,
        "dtensor_cfg": {"enabled": False},
        "megatron_cfg": {
            "enabled": True,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 2,
        },
        "sequence_packing": {"enabled": False},
    }
    with pytest.raises(AssertionError, match="Megatron context parallel"):
        validate_streaming_capabilities(
            megatron_cp_without_sequence_packing_policy,
            base_policy,
            stream_config,
        )

    megatron_cp_with_sequence_packing_policy = {
        **base_policy,
        "dtensor_cfg": {"enabled": False},
        "megatron_cfg": {
            "enabled": True,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 2,
        },
        "sequence_packing": {"enabled": True},
    }
    validate_streaming_capabilities(
        megatron_cp_with_sequence_packing_policy,
        megatron_cp_with_sequence_packing_policy,
        stream_config,
    )

    legacy_config = parse_data_pipeline_config(None)
    validate_streaming_capabilities(
        dynamic_megatron_policy,
        multimodal_policy,
        legacy_config,
    )


def test_validate_stream_rollout_rejects_reordered_student_batching():
    base_policy = {
        "tokenizer": {"use_processor": False},
        "dynamic_batching": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "dtensor_cfg": {
            "enabled": True,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
        },
    }
    stream_rollout_config = parse_data_pipeline_config({"mode": "stream_rollout"})
    validate_streaming_capabilities(
        base_policy,
        base_policy,
        stream_rollout_config,
    )

    with pytest.raises(AssertionError, match="dynamic batching"):
        validate_streaming_capabilities(
            {**base_policy, "dynamic_batching": {"enabled": True}},
            base_policy,
            stream_rollout_config,
        )

    with pytest.raises(AssertionError, match="sequence packing"):
        validate_streaming_capabilities(
            {**base_policy, "sequence_packing": {"enabled": True}},
            base_policy,
            stream_rollout_config,
        )


def test_validate_sparse_loss_rejects_unsupported_layouts():
    base_policy = {
        "tokenizer": {"use_processor": False},
        "dynamic_batching": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "dtensor_cfg": {
            "enabled": True,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
        },
    }
    sparse_config = parse_data_pipeline_config({"mode": "sparse_loss"})
    validate_streaming_capabilities(base_policy, base_policy, sparse_config)

    with pytest.raises(AssertionError, match="dynamic batching"):
        validate_streaming_capabilities(
            {**base_policy, "dynamic_batching": {"enabled": True}},
            base_policy,
            sparse_config,
        )

    with pytest.raises(AssertionError, match="sequence packing"):
        validate_streaming_capabilities(
            {**base_policy, "sequence_packing": {"enabled": True}},
            base_policy,
            sparse_config,
        )

    with pytest.raises(AssertionError, match="context parallel"):
        validate_streaming_capabilities(
            {
                **base_policy,
                "dtensor_cfg": {
                    "enabled": True,
                    "tensor_parallel_size": 1,
                    "context_parallel_size": 2,
                },
            },
            base_policy,
            sparse_config,
        )


def test_build_token_stream_from_rollout_batch_matches_dense_manifest():
    step_manifest = build_step_manifest(
        batch_id="rollout-step-0",
        step=3,
        prompt_count=2,
        num_generations_per_prompt=1,
        train_global_batch_size=2,
        max_sequence_length=8,
        tokenizer_name_or_path="unit-test-tokenizer",
        tokenizer_config={},
    )
    rollout_batch = BatchedDataDict(
        {
            "message_log": [
                [
                    {
                        "token_ids": torch.tensor([1, 2], dtype=torch.int64),
                        "role": "user",
                    },
                    {
                        "token_ids": torch.tensor([3, 4], dtype=torch.int64),
                        "role": "assistant",
                    },
                ],
                [
                    {
                        "token_ids": torch.tensor([5], dtype=torch.int64),
                        "role": "user",
                    },
                    {
                        "token_ids": torch.tensor([6, 7], dtype=torch.int64),
                        "role": "assistant",
                    },
                ],
            ],
            "loss_multiplier": torch.ones(2, dtype=torch.float32),
        }
    )
    attach_step_metadata(rollout_batch, step_manifest)

    manifest, stream = build_token_stream_from_rollout_batch(
        batch_id=step_manifest.batch_id,
        step=step_manifest.step,
        rollout_batch=rollout_batch,
        tokenizer_pad_token_id=0,
        make_sequence_length_divisible_by=1,
        max_sequence_length=8,
        tokenizer_name_or_path="unit-test-tokenizer",
        tokenizer_config={},
        data_pipeline_config=parse_data_pipeline_config(
            {"mode": "stream_rollout", "max_chunk_tokens": 16}
        ),
        src_dp_size=1,
        dst_dp_size=1,
    )

    chunks = [chunk for shard in stream.shard_streams for chunk in shard]
    assert manifest.batch_id == "rollout-step-0"
    assert manifest.input_lengths.tolist() == [4, 3]
    assert manifest.sample_ids.tolist() == step_manifest.sample_ids.tolist()
    assert len(chunks) == 1
    assert chunks[0].input_ids.tolist() == [[1, 2, 3, 4], [5, 6, 7, 0]]
    assert chunks[0].loss_spans.starts.tolist() == [1, 0]
    assert chunks[0].loss_spans.ends.tolist() == [3, 2]

    oracle = build_conservation_oracle(manifest)
    oracle.validate_token_chunks(chunks)


def test_build_token_stream_from_rollout_batch_can_preserve_source_shards():
    step_manifest = build_step_manifest(
        batch_id="rollout-step-source-shards",
        step=4,
        prompt_count=4,
        num_generations_per_prompt=1,
        train_global_batch_size=4,
        max_sequence_length=8,
        tokenizer_name_or_path="unit-test-tokenizer",
        tokenizer_config={},
    )
    rollout_batch = BatchedDataDict(
        {
            "message_log": [
                [
                    {
                        "token_ids": torch.tensor([10 + index], dtype=torch.int64),
                        "role": "user",
                    },
                    {
                        "token_ids": torch.tensor([20 + index], dtype=torch.int64),
                        "role": "assistant",
                    },
                ]
                for index in range(4)
            ],
            "loss_multiplier": torch.ones(4, dtype=torch.float32),
        }
    )
    attach_step_metadata(rollout_batch, step_manifest)

    manifest, stream = build_token_stream_from_rollout_batch(
        batch_id=step_manifest.batch_id,
        step=step_manifest.step,
        rollout_batch=rollout_batch,
        tokenizer_pad_token_id=0,
        make_sequence_length_divisible_by=1,
        max_sequence_length=8,
        tokenizer_name_or_path="unit-test-tokenizer",
        tokenizer_config={},
        data_pipeline_config=parse_data_pipeline_config(
            {"mode": "stream_rollout", "max_chunk_tokens": 16}
        ),
        src_dp_size=4,
        dst_dp_size=4,
        planner_mode="preserve_source_shards",
    )

    assert manifest.update_group.tolist() == [0, 0, 0, 0]
    assert [
        [chunk.sample_ids.tolist() for chunk in shard_chunks]
        for shard_chunks in stream.shard_streams
    ] == [
        [[step_manifest.sample_ids[0].item()]],
        [[step_manifest.sample_ids[1].item()]],
        [[step_manifest.sample_ids[2].item()]],
        [[step_manifest.sample_ids[3].item()]],
    ]


def test_build_token_stream_from_rollout_batch_rejects_multimodal_payload():
    from nemo_rl.data.multimodal_utils import PackedTensor

    step_manifest = build_step_manifest(
        batch_id="rollout-step-mm",
        step=0,
        prompt_count=1,
        num_generations_per_prompt=1,
        train_global_batch_size=1,
        max_sequence_length=8,
        tokenizer_name_or_path="unit-test-tokenizer",
        tokenizer_config={},
    )
    rollout_batch = BatchedDataDict(
        {
            "message_log": [
                [
                    {
                        "token_ids": torch.tensor([1], dtype=torch.int64),
                        "role": "user",
                        "images": PackedTensor(
                            torch.zeros((1, 3, 2, 2), dtype=torch.float32),
                            dim_to_pack=0,
                        ),
                    },
                    {
                        "token_ids": torch.tensor([2], dtype=torch.int64),
                        "role": "assistant",
                    },
                ],
            ],
            "loss_multiplier": torch.ones(1, dtype=torch.float32),
        }
    )
    attach_step_metadata(rollout_batch, step_manifest)

    with pytest.raises(AssertionError, match="multimodal"):
        build_token_stream_from_rollout_batch(
            batch_id=step_manifest.batch_id,
            step=step_manifest.step,
            rollout_batch=rollout_batch,
            tokenizer_pad_token_id=0,
            make_sequence_length_divisible_by=1,
            max_sequence_length=8,
            tokenizer_name_or_path="unit-test-tokenizer",
            tokenizer_config={},
            data_pipeline_config=parse_data_pipeline_config({"mode": "stream_rollout"}),
            src_dp_size=1,
            dst_dp_size=1,
        )
