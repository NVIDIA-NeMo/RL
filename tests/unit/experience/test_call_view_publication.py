# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original call inputs survive publication, cleanup and logical GRPO on CPU.

Storage is an in-memory transport. The real sink/source, verifier, finalizer,
actor handler, wire packing and controller advantage stage execute. This does
not qualify native Megatron forwards, gradients, GPU memory or resumed training.
"""

import asyncio
from dataclasses import replace

import pytest
import torch

from nemo_rl.data_plane.schema import INVALID_TOOL_CALL_MASK, MALFORMED_THINKING_MASK
from nemo_rl.experience.rollout_reassembler import (
    ActionOutputFlags,
    FinalizedGroup,
    RolloutReassembler,
    RolloutSelection,
)
from nemo_rl.experience.rollout_reassembler_actor import (
    ReassemblyRequest,
    RolloutReassemblerActor,
    assert_metadata_only,
)
from tests.unit.experience.test_finalized_call_views import (
    _Case,
    _case,
    _RecordingDataPlane,
)
from tests.unit.experience.test_cc_routes import _RecordingSink, _segment
from tests.unit.models.generation.test_vllm_token_capture_hosting import (
    _worker_with_capture,
)
from tests.unit.single_controller.test_logical_advantage import _controller

pytestmark = pytest.mark.nemo_gym


def _request(
    cases: list[_Case], *, group: str = "group", padding: int = 1
) -> ReassemblyRequest:
    return ReassemblyRequest(
        group_id=group,
        rollout_ids=tuple(case.receipt["rollout_id"] for case in cases),
        canonical_sample_ids=tuple(f"{group}_g{i}" for i in range(len(cases))),
        receipts=tuple(case.receipt for case in cases),
        rewards=tuple(float(i) for i in range(len(cases))),
        mask_sample=(False,) * len(cases),
        fallback_weight_version=13,
        prompt_idx=7,
        logical_selections=tuple(
            RolloutSelection(
                response_ids=tuple(r["response_id"] for r in case.receipt["manifest"]),
                action_flags=case.flags,
                session_ids=(f"session-{i}",) * len(case.flags),
                session_roles=("parent",) * len(case.flags),
            )
            for i, case in enumerate(cases)
        ),
        execution_row_multiple=padding,
        capture_call_views=True,
    )


def _publish(
    finalizer: RolloutReassembler, request: ReassemblyRequest
) -> FinalizedGroup:
    actor_cls = RolloutReassemblerActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_cls)
    actor._finalizer = finalizer
    return actor.finalize(request)


@pytest.mark.parametrize("padding", [1, 3])
def test_call_inputs_and_media_publish_once_with_exact_cleanup_plan(
    padding: int,
) -> None:
    plane = _RecordingDataPlane()
    cases = [
        _case(media=True, rollout_id=f"owner{i}", data_plane=plane) for i in range(2)
    ]
    request = _request(cases, padding=padding)
    planned = request.cleanup_sample_ids
    assert_metadata_only(request)
    result = _publish(cases[0].finalizer, request)
    assert result.meta is not None and not result.dropped
    assert_metadata_only(result)
    assert tuple(result.meta.sample_ids) == planned
    assert result.valid_row_count == result.total_row_count == 8
    assert result.canonical_output_tokens == 8
    assert result.metrics["finalize/valid_logical_owner_count"] == 2
    assert result.metrics["finalize/context_segment_count"] == 2
    assert (result.group_min_wv, result.group_max_wv) == (12, 13)
    assert plane.events[-1][0:2] == ("clear", "staged")
    assert len(plane.events[-1][2]) == 8
    assert not any(partition == "staged" for partition, _ in plane.rows)
    assert sum(event[0:2] == ("put", "canonical") for event in plane.events) == 1
    covered = set()
    for physical, (sample, tag) in enumerate(
        zip(result.meta.sample_ids, result.meta.tags, strict=True)
    ):
        stored = plane.rows["canonical", sample]
        if tag["is_execution_padding"]:
            assert stored["sample_mask"].item() == 0
            assert not stored["token_mask"].any()
            assert tag["logical_rollout_id"] is None
            continue
        slot, call_index = tag["logical_slot"], tag["call_index"]
        case = cases[slot]
        start = case.records[call_index].prev_len + 1
        end = case.records[call_index].cum_len
        assert tag["context_segment_index"] == 0
        assert tag["context_segment_count"] == 1
        assert tag["segment_index"] == call_index and tag["segment_count"] == 4
        assert tag["model_call_id"] == f"c{call_index}"
        assert tag["generation_weight_version"] == (12 if call_index < 2 else 13)
        assert (
            tag["session_id"] == f"session-{slot}" and tag["session_role"] == "parent"
        )
        assert (
            tag["call_generation_start"] == start and tag["call_generation_end"] == end
        )
        assert stored["input_ids"][0].tolist() == [
            token
            for record in case.records[: call_index + 1]
            for token in record.token_ids_delta
        ]
        expected_routes = case.routes[call_index].clone()
        if call_index:
            expected_routes[case.records[call_index].prev_len - 1] = 900 + call_index
        assert torch.equal(stored["routed_experts"][0], expected_routes)
        assert (
            stored["generation_logprobs"][0, start].item()
            == case.records[call_index].generation_log_probs_delta[-1]
        )
        assert stored["token_mask"][0].tolist() == [0.0] * start + [1.0]
        identity = (slot, start)
        assert identity not in covered
        covered.add(identity)
        for field, value in (
            (INVALID_TOOL_CALL_MASK, case.flags[call_index].invalid_tool_call),
            (MALFORMED_THINKING_MASK, case.flags[call_index].malformed_thinking),
        ):
            assert stored[field][0].tolist() == [False] * start + [value]
        expected_pixels = ([], [1.0] * 12, [1.0] * 12, [1.0] * 12 + [3.0] * 12)[
            call_index
        ]
        assert stored["pixel_values"][0].tolist() == expected_pixels
        assert stored["total_reward"].item() == float(slot)
        assert tag["num_assistant_messages"] == 1
    assert len(covered) == 8
    assert sum(t["num_invalid_tool_calls"] for t in result.meta.tags) == 4
    assert sum(t["num_malformed_thinking"] for t in result.meta.tags) == 2


@pytest.mark.parametrize("failure", ["corrupt_last_route", "row_length", "poisoned"])
def test_invalid_owner_keeps_intent_cardinality_but_trains_no_partial_calls(
    failure: str,
) -> None:
    plane = _RecordingDataPlane()
    cases = [
        _case(media=False, rollout_id=f"owner{i}", data_plane=plane) for i in range(2)
    ]
    request = _request(cases, padding=3)
    if failure == "corrupt_last_route":
        plane.rows["staged", "owner0/c3"]["routed_experts"][0, 0, 0, 0] += 1
    elif failure == "row_length":
        # The second owner selects its first three calls. Its unselected last
        # call stays cleanup-owned; its six-token context fits the real cap.
        receipt = dict(request.receipts[1], terminal_model_call_id="c2")
        selection = request.logical_selections[1]
        selection = replace(
            selection,
            response_ids=selection.response_ids[:3],
            action_flags=selection.action_flags[:3],
            session_ids=selection.session_ids[:3],
            session_roles=selection.session_roles[:3],
        )
        request = replace(
            request,
            receipts=(request.receipts[0], receipt),
            logical_selections=(request.logical_selections[0], selection),
        )
        limited = RolloutReassembler(
            plane,
            partition_id="canonical",
            staging_partition="staged",
            pad_token_id=0,
            max_seq_len=1024,
            router_replay_enabled=True,
            max_train_row_tokens=7,
        )
        cases[0] = replace(cases[0], finalizer=limited)
    else:
        receipt = dict(request.receipts[0], capture_poisoned=True)
        selections = (RolloutSelection((), ()), request.logical_selections[1])
        request = replace(
            request,
            receipts=(receipt, request.receipts[1]),
            logical_selections=selections,
        )
    planned = request.cleanup_sample_ids
    result = _publish(cases[0].finalizer, request)
    assert result.meta is not None and tuple(result.meta.sample_ids) == planned
    assert result.metrics["finalize/valid_logical_owner_count"] == 1
    owner_rows = [
        i for i, tag in enumerate(result.meta.tags) if tag["logical_slot"] == 0
    ]
    assert len(owner_rows) == (1 if failure == "poisoned" else 4)
    for row in owner_rows:
        stored = plane.rows["canonical", result.meta.sample_ids[row]]
        assert stored["sample_mask"].item() == 0
        assert not stored["token_mask"].any()
    assert result.canonical_output_tokens == (3 if failure == "row_length" else 4)
    assert not any(partition == "staged" for partition, _ in plane.rows)


def test_lost_publication_ack_preserves_all_staging_and_exact_orphan_cleanup_plan() -> (
    None
):
    case = _case(media=True)
    request = _request([case], padding=3)
    planned = request.cleanup_sample_ids
    case.data_plane.lose_ack = True
    with pytest.raises(OSError, match="acknowledgement lost"):
        _publish(case.finalizer, request)
    assert case.data_plane.delete_count == 0
    assert {
        sample for partition, sample in case.data_plane.rows if partition == "canonical"
    } == set(planned)
    assert {
        sample for partition, sample in case.data_plane.rows if partition == "staged"
    } == {f"owner/c{i}" for i in range(4)}


def test_all32_logical_owners_keep_one_reward_vote_and_every_target_in_controller() -> (
    None
):
    plane = _RecordingDataPlane()
    results = []
    for prompt in range(8):
        cases = [
            _case(media=False, rollout_id=f"p{prompt}-owner{i}", data_plane=plane)
            for i in range(4)
        ]
        request = _request(cases, group=f"group{prompt}")
        results.append(_publish(cases[0].finalizer, request))
    meta = results[0].meta.concat(*(result.meta for result in results[1:]))
    assert meta.size == 128
    assert len({(t["dispatch_group_id"], t["logical_slot"]) for t in meta.tags}) == 32
    assert sum(result.canonical_output_tokens for result in results) == 128
    data = plane.get_samples(
        sample_ids=meta.sample_ids,
        partition_id=meta.partition_id,
        select_fields=meta.fields,
    )
    controller = _controller(meta, data)
    _, valid = asyncio.run(controller._advantage_stage(meta))
    assert valid
    assert controller._step_log_dict["rewards"][0].numel() == 32
    assert controller._step_log_dict["sample_masks"][0].numel() == 32
    for index, tag in enumerate(meta.tags):
        expected = float(tag["logical_slot"]) - 1.5
        torch.testing.assert_close(
            data["advantages"][index],
            torch.full_like(data["advantages"][index], expected),
        )
    assert sum(int(mask.sum()) for mask in data["token_mask"].unbind()) == 128


def test_call_publication_without_logical_selection_is_rejected_before_io() -> None:
    case = _case(media=False)
    request = replace(_request([case]), logical_selections=None)
    with pytest.raises(ValueError, match="logical owner selections"):
        _ = request.cleanup_sample_ids
    with pytest.raises(ValueError, match="logical owner selections"):
        _publish(case.finalizer, request)
    assert case.data_plane.delete_count == 0
    assert case.data_plane.write_count == 4


def test_parent_child_contexts_keep_session_ownership_and_one_episode_reward() -> None:
    plane = _RecordingDataPlane()
    sink = _RecordingSink(plane)
    worker = _worker_with_capture(sink)
    finalizer = RolloutReassembler(
        plane,
        partition_id="canonical",
        staging_partition="staged",
        pad_token_id=0,
        max_seq_len=1024,
        router_replay_enabled=True,
    )
    parent = _segment(worker, sink, "attempt0")
    child = _segment(worker, sink, "attempt0", child=False, base=30)
    sibling = _segment(worker, sink, "attempt1", child=False, base=50)
    request = ReassemblyRequest(
        group_id="group",
        rollout_ids=("attempt0", "attempt1"),
        canonical_sample_ids=("group_g0", "group_g1"),
        receipts=(child["receipt"], sibling["receipt"]),
        rewards=(0.0, 1.0),
        mask_sample=(False, False),
        fallback_weight_version=0,
        prompt_idx=4,
        logical_selections=(
            RolloutSelection(
                tuple(parent["ids"] + child["ids"]),
                (ActionOutputFlags(False, False),) * 3,
                session_ids=("parent0", "parent0", "child0"),
                session_roles=("parent", "parent", "child"),
            ),
            RolloutSelection(
                tuple(sibling["ids"]),
                (ActionOutputFlags(False, False),),
                session_ids=("parent1",),
                session_roles=("parent",),
            ),
        ),
        capture_call_views=True,
    )
    planned = request.cleanup_sample_ids
    result = _publish(finalizer, request)
    meta = result.meta
    assert meta is not None and tuple(meta.sample_ids) == planned
    assert meta.sequence_lengths == [3, 6, 3, 3]
    assert [tag["context_segment_index"] for tag in meta.tags] == [0, 0, 1, 0]
    assert [tag["session_role"] for tag in meta.tags] == [
        "parent",
        "parent",
        "child",
        "parent",
    ]
    assert [tag["segment_count"] for tag in meta.tags] == [3, 3, 3, 1]
    data = plane.get_samples(
        sample_ids=meta.sample_ids,
        partition_id=meta.partition_id,
        select_fields=meta.fields,
    )
    controller = _controller(meta, data)
    _, valid = asyncio.run(controller._advantage_stage(meta))
    assert valid and controller._step_log_dict["rewards"][0].numel() == 2
    for index, expected in enumerate([-0.5, -0.5, -0.5, 0.5]):
        torch.testing.assert_close(
            data["advantages"][index],
            torch.full_like(data["advantages"][index], expected),
        )
    for index, expected in enumerate(
        [[11, 12, -1], [900, 901, 17, 13, 14, -1], [31, 32, -1], [51, 52, -1]]
    ):
        assert data["routed_experts"][index, :, 0, 0].tolist() == expected
    assert result.canonical_output_tokens == 4
