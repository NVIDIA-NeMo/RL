# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RL-owned row planning through real HTTP custody and TQ codecs on CPU."""

from typing import Any

import pytest
from responses_api_models.vllm_model.tests import test_framework_context as gym_harness
from tensordict import TensorDict

from nemo_rl.data_plane.codec import stack_or_nest
from nemo_rl.experience.rollout_reassembler import ActionOutputFlags, RolloutSelection
from nemo_rl.models.generation.capture_context import decide_capture_input


class MemoryDataPlane:
    """Store the sink's actual TensorDict rows and serve explicitly selected fields."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], TensorDict] = {}
        self.lose_ack = False
        self.write_count = 0
        self.delete_count = 0

    def put_samples(
        self,
        *,
        sample_ids: list[str],
        partition_id: str,
        fields: TensorDict,
        tags: list[dict],
    ) -> None:
        for index, key in enumerate(sample_ids):
            self.rows[partition_id, key] = fields[index : index + 1].clone()
            self.write_count += 1
        if self.lose_ack:
            raise OSError("write completed but acknowledgement lost")

    def get_samples(
        self, *, sample_ids: list[str], partition_id: str, select_fields: list[str]
    ) -> TensorDict:
        return TensorDict(
            {
                name: stack_or_nest(
                    [self.rows[partition_id, key][name][0] for key in sample_ids]
                )
                for name in select_fields
            },
            batch_size=[len(sample_ids)],
        )

    def clear_samples(self, **kwargs: Any) -> None:
        self.delete_count += 1
        raise AssertionError("finalize_rollout must retain staged evidence")


class PublicationDataPlane(MemoryDataPlane):
    """Record actual published rows and exact cleanup ordering."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, str, list[str]]] = []

    def put_samples(
        self,
        *,
        sample_ids: list[str],
        partition_id: str,
        fields: TensorDict,
        tags: list[dict],
    ) -> None:
        self.events.append(("put", partition_id, list(sample_ids)))
        # Integer indexing works for both dense staging and jagged canonical
        # fields; batch slicing of a jagged tensor is not implemented by Torch.
        for index, key in enumerate(sample_ids):
            self.rows[partition_id, key] = TensorDict(
                {
                    name: value[index].unsqueeze(0).clone()
                    for name, value in fields.items()
                },
                batch_size=[1],
            )
            self.write_count += 1
        if self.lose_ack:
            raise OSError("write completed but acknowledgement lost")

    def clear_samples(self, *, sample_ids: list[str], partition_id: str) -> None:
        self.events.append(("clear", partition_id, list(sample_ids)))
        self.delete_count += 1
        for key in sample_ids:
            del self.rows[partition_id, key]


def capture_segment(harness, scope, *, child=False):
    first = gym_harness.assert_clean(harness.post(scope, gym_harness.HISTORY))
    ids = [first["id"]]
    output = list(first["output"])
    if child:
        second = gym_harness.assert_clean(
            harness.post(
                scope,
                gym_harness.HISTORY
                + first["output"]
                + [{"role": "user", "content": "next observation"}],
            )
        )
        ids.append(second["id"])
        output.extend(second["output"])
    manifest = harness.manifest(scope)
    return {"scope": scope, "manifest": manifest, "ids": ids, "output": output}


def finalize(finalizer, owners, *, execution_row_multiple=1, mask_sample=None):
    receipts, selections = [], []
    for chunks in owners:
        manifest = chunks[-1]["manifest"]
        ids = tuple(item for chunk in chunks for item in chunk["ids"])
        last = next(
            record.model_call_id
            for record in manifest.records
            if record.response_id == ids[-1]
        )
        receipts.append(
            dict(
                rollout_id=chunks[-1]["scope"],
                manifest=[record.model_dump() for record in manifest.records],
                terminal_model_call_id=last,
                terminal_selection="declared",
                attempted_call_ids=manifest.attempted_call_ids,
                pending_call_ids=manifest.pending_call_ids,
            )
        )
        selections.append(
            RolloutSelection(ids, tuple(ActionOutputFlags(False, False) for _ in ids))
        )
    return finalizer.finalize_group(
        "group",
        [chunks[-1]["scope"] for chunks in owners],
        receipts,
        [float(i + 1) for i in range(len(owners))],
        mask_sample=mask_sample or [False] * len(owners),
        fallback_weight_version=7,
        prompt_idx=99,
        canonical_sample_ids=[f"group_g{i}" for i in range(len(owners))],
        logical_selections=selections,
        execution_row_multiple=execution_row_multiple,
    )


@pytest.mark.parametrize("padding", [1, 4])
def test_selected_roots_publish_in_order_and_cleanup_exact_attempt(
    monkeypatch, tmp_path, padding
):
    from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
    from nemo_rl.experience.rollout_reassembler import RolloutReassembler

    plane = PublicationDataPlane()
    source = TQTokenSource(plane, staging_partition="staged")
    harness = gym_harness.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(plane, staging_partition="staged"),
        fetch_prefix=source.fetch_prefix_token_ids,
        root_prompt=[10, 11],
        decide_input=decide_capture_input,
    )
    first = capture_segment(harness, "attempt0", child=True)
    sibling = capture_segment(harness, "attempt1")
    staged = {key for partition, key in plane.rows if partition == "staged"}
    result = finalize(
        RolloutReassembler(
            plane,
            partition_id="canonical",
            staging_partition="staged",
            pad_token_id=0,
            max_seq_len=1024,
        ),
        [[first], [sibling]],
        execution_row_multiple=padding,
    )
    assert result.valid_row_count == 2
    assert result.meta.sample_ids[:2] == ["group_g0_s0", "group_g1_s0"]
    assert len(result.meta.sample_ids) == (4 if padding == 4 else 2)
    assert plane.rows["canonical", "group_g0_s0"]["input_ids"].tolist() == [
        [10, 11, 1001, 20, 21, 1002]
    ]
    assert plane.rows["canonical", "group_g0_s0"]["token_mask"].tolist() == [
        [0, 0, 1, 0, 0, 1]
    ]
    assert not any(partition == "staged" for partition, key in plane.rows)
    assert (
        plane.events[-1] == ("clear", "staged", list(staged))
        or set(plane.events[-1][2]) == staged
    )


def test_lost_publication_ack_preserves_staging_and_exact_cleanup_plan(
    monkeypatch, tmp_path
):
    from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
    from nemo_rl.experience.rollout_reassembler import RolloutReassembler
    from nemo_rl.experience.rollout_reassembler_actor import ReassemblyRequest

    plane = PublicationDataPlane()
    source = TQTokenSource(plane, staging_partition="staged")
    harness = gym_harness.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(plane, staging_partition="staged"),
        fetch_prefix=source.fetch_prefix_token_ids,
        root_prompt=[10, 11],
        decide_input=decide_capture_input,
    )
    selected = capture_segment(harness, "attempt", child=True)
    manifest = selected["manifest"]
    receipt = dict(
        rollout_id="attempt",
        manifest=[record.model_dump() for record in manifest.records],
        terminal_model_call_id=manifest.records[-1].model_call_id,
        terminal_selection="declared",
        attempted_call_ids=manifest.attempted_call_ids,
    )
    request = ReassemblyRequest(
        group_id="group",
        rollout_ids=("attempt",),
        canonical_sample_ids=("group_g0",),
        receipts=(receipt,),
        rewards=(1.0,),
        fallback_weight_version=7,
        prompt_idx=99,
        mask_sample=(False,),
        logical_selections=(
            RolloutSelection(
                tuple(selected["ids"]), (ActionOutputFlags(False, False),) * 2
            ),
        ),
        execution_row_multiple=4,
    )
    planned = request.cleanup_sample_ids
    assert planned == ("group_g0_s0", "group_pad0", "group_pad1", "group_pad2")
    plane.lose_ack = True
    with pytest.raises(OSError, match="acknowledgement lost"):
        finalize(
            RolloutReassembler(
                plane,
                partition_id="canonical",
                staging_partition="staged",
                pad_token_id=0,
                max_seq_len=1024,
            ),
            [[selected]],
            execution_row_multiple=4,
        )
    assert set(key for partition, key in plane.rows if partition == "canonical") == set(
        planned
    )
    assert len([key for partition, key in plane.rows if partition == "staged"]) == 2
    assert not any(event[0] == "clear" for event in plane.events)


def test_unknown_capture_ack_is_not_cleanup_permission():
    from nemo_rl.experience.rollout_recovery import _receipt_staging_keys

    receipt = dict(
        rollout_id="attempt",
        manifest=[],
        attempted_call_ids=["c1"],
        pending_call_ids=["c1"],
    )
    with pytest.raises(ValueError, match="unresolved"):
        _receipt_staging_keys(receipt)
    receipt["pending_call_ids"] = []
    assert _receipt_staging_keys(receipt) == ["attempt/c1"]


@pytest.mark.parametrize(
    "invalid", [None, "duplicate", "reordered", "missing", "terminal", "foreign"]
)
def test_selection_validates_before_publication_and_unselected_roots_are_cleaned(
    monkeypatch, tmp_path, invalid
):
    from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
    from nemo_rl.experience.rollout_reassembler import RolloutReassembler

    plane = PublicationDataPlane()
    source = TQTokenSource(plane, staging_partition="staged")
    harness = gym_harness.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(plane, staging_partition="staged"),
        fetch_prefix=source.fetch_prefix_token_ids,
        root_prompt=[10, 11],
        decide_input=decide_capture_input,
    )
    calls = [capture_segment(harness, "attempt") for _ in range(3)]
    manifest = calls[-1]["manifest"]
    records = [record.model_dump() for record in manifest.records]
    ids = (calls[0]["ids"][0], calls[2]["ids"][0])
    if invalid == "duplicate":
        ids = (ids[0], ids[0])
    elif invalid == "reordered":
        ids = tuple(reversed(ids))
    elif invalid == "missing":
        ids = (ids[0], "absent")
    elif invalid == "terminal":
        ids = (ids[0],)
    elif invalid == "foreign":
        records[0]["staging_key"] = "another-attempt/call"
    receipt = dict(
        rollout_id="attempt",
        manifest=records,
        terminal_model_call_id=manifest.records[-1].model_call_id,
        terminal_selection="declared",
        attempted_call_ids=manifest.attempted_call_ids,
    )
    selection = RolloutSelection(
        ids, tuple(ActionOutputFlags(False, False) for _ in ids)
    )
    reassembler = RolloutReassembler(
        plane,
        partition_id="canonical",
        staging_partition="staged",
        pad_token_id=0,
        max_seq_len=1024,
    )

    def run():
        return reassembler.finalize_group(
            "group",
            ["attempt"],
            [receipt],
            [1.0],
            mask_sample=[False],
            fallback_weight_version=7,
            prompt_idx=0,
            canonical_sample_ids=["group_g0"],
            logical_selections=[selection],
        )

    if invalid:
        with pytest.raises(ValueError):
            run()
        assert all(
            kind == "put" and partition == "staged"
            for kind, partition, _ in plane.events
        )
        assert len(plane.rows) == 3
    else:
        result = run()
        assert result.valid_row_count == 2
        assert len(plane.rows) == 2
        assert set(plane.events[-1][2]) == {
            record.staging_key for record in manifest.records
        }


def test_gym_adapter_returns_one_real_receipt_and_rl_selection(monkeypatch, tmp_path):
    import asyncio
    from unittest.mock import AsyncMock

    from nemo_rl.environments.nemo_gym import NemoGym
    from nemo_rl.experience.rollout_reassembler_actor import assert_metadata_only

    harness = gym_harness.make_capture_harness(
        monkeypatch, tmp_path, root_prompt=[10, 11]
    )
    captured = capture_segment(harness, "attempt", child=True)
    env = object.__new__(NemoGym.__ray_metadata__.modified_class)
    env._context_compaction = True
    env.cfg = {}

    async def control(*args):
        return await harness.ledger.manifest("attempt")

    env._control = AsyncMock(side_effect=control)
    result = dict(
        reward=1.0,
        response={"id": captured["ids"][-1], "output": captured["output"]},
        instance_config={"mask_sample": True},
        context_compaction_result=dict(
            logical_rollout_id="attempt",
            selected_actions=[
                dict(response_id=identity, finish_reason="stop", last_output_item=None)
                for identity in captured["ids"]
            ],
            outcome="completed",
        ),
    )
    processed = asyncio.run(
        env._postprocess_receipt_mode({"_ng_rollout_id": "attempt"}, result)
    )
    assert env._control.await_count == 1
    assert processed["receipt"]["rollout_id"] == "attempt"
    assert len(processed["receipt"]["manifest"]) == 2
    assert (
        processed["receipt"]["attempted_call_ids"]
        == captured["manifest"].attempted_call_ids
    )
    assert processed["logical_selection"].response_ids == tuple(captured["ids"])
    assert processed["full_result"]["instance_config"]["mask_sample"] is True
    assert "logical_segments" not in processed
    assert_metadata_only(processed["logical_selection"])


@pytest.mark.parametrize("compact_after", [None, 10])
def test_twenty_calls_have_one_trace_per_context(monkeypatch, tmp_path, compact_after):
    import asyncio

    from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
    from nemo_rl.experience.rollout_reassembler import RolloutReassembler
    from nemo_rl.experience.rollout_reassembler_actor import ReassemblyRequest

    plane = PublicationDataPlane()
    source = TQTokenSource(plane, staging_partition="staged")
    harness = gym_harness.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(plane, staging_partition="staged"),
        fetch_prefix=source.fetch_prefix_token_ids,
        root_prompt=[10, 11],
        decide_input=decide_capture_input,
    )
    expected_segments = 1 if compact_after is None else 2
    try:
        messages = list(gym_harness.HISTORY)
        ids = []
        for turn in range(20):
            if turn == compact_after:
                messages = [{"role": "user", "content": "summary of previous work"}]
            response = gym_harness.assert_clean(harness.post("attempt", messages))
            ids.append(response["id"])
            messages = (
                messages
                + response["output"]
                + [{"role": "user", "content": f"observation {turn + 1}"}]
            )
        manifest = harness.manifest("attempt")
        records = manifest.records
        assert len(records) == 20
        assert (
            sum(record.parent_call_id is None for record in records)
            == expected_segments
        )
        for index, record in enumerate(records):
            expected_parent = (
                None
                if index in (0, compact_after)
                else records[index - 1].model_call_id
            )
            assert record.parent_call_id == expected_parent
        # Twenty small delta records, never twenty growing full contexts.
        staged = source.fetch_for_finalization(
            [record.staging_key for record in records], include_route_fragments=False
        )
        assert [len(call.snapshot.token_ids_delta) for call in staged] == [3] * 20
        receipt = dict(
            rollout_id="attempt",
            manifest=[record.model_dump() for record in records],
            terminal_model_call_id=records[-1].model_call_id,
            terminal_selection="declared",
            attempted_call_ids=manifest.attempted_call_ids,
        )
        request = ReassemblyRequest(
            group_id="group",
            rollout_ids=("attempt",),
            canonical_sample_ids=("group_g0",),
            receipts=(receipt,),
            rewards=(1.0,),
            fallback_weight_version=7,
            prompt_idx=99,
            mask_sample=(False,),
            logical_selections=(
                RolloutSelection(
                    tuple(ids),
                    (ActionOutputFlags(False, False),) * 20,
                ),
            ),
        )
        result = finalize(
            RolloutReassembler(
                plane,
                partition_id="canonical",
                staging_partition="staged",
                pad_token_id=0,
                max_seq_len=1024,
            ),
            [[{"scope": "attempt", "manifest": manifest, "ids": ids}]],
        )
        assert result.valid_row_count == expected_segments
        assert result.total_row_count == expected_segments
        assert tuple(result.meta.sample_ids) == request.cleanup_sample_ids
        assert result.meta.sequence_lengths == (
            [60] if compact_after is None else [30, 30]
        )
        actions = []
        for key in result.meta.sample_ids:
            row = plane.rows["canonical", key]
            actions.extend(
                token
                for token, mask in zip(
                    row["input_ids"][0].tolist(),
                    row["token_mask"][0].tolist(),
                    strict=True,
                )
                if mask
            )
        assert actions == list(range(1001, 1021))
        assert result.canonical_output_tokens == 20
        assert not any(partition == "staged" for partition, _ in plane.rows)
    finally:
        harness.client.close()
        asyncio.run(harness.ledger.close())


def test_existing_detectors_mask_only_selected_generation_spans(monkeypatch, tmp_path):
    import asyncio

    import torch

    from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
    from nemo_rl.experience.rollout_reassembler import RolloutReassembler
    from nemo_rl.experience.rollout_reassembler_actor import assert_metadata_only
    from tests.unit.experience.test_cc_dispatch import _env

    plane = PublicationDataPlane()
    source = TQTokenSource(plane, staging_partition="staged")
    harness = gym_harness.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(plane, staging_partition="staged"),
        fetch_prefix=source.fetch_prefix_token_ids,
        root_prompt=[10, 11],
        decide_input=decide_capture_input,
    )
    try:
        first = capture_segment(harness, "attempt", child=True)
        rewritten = capture_segment(harness, "attempt")
        ids = first["ids"] + rewritten["ids"]
        actions = [
            dict(response_id=identity, finish_reason="stop", last_output_item=None)
            for identity in ids
        ]
        actions[0]["last_output_item"] = {
            "type": "message",
            "id": "message-0",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "[[BAD]] <badthink>", "annotations": []}
            ],
        }
        actions[2]["last_output_item"] = {
            "type": "reasoning",
            "id": "reasoning-1",
            "summary": [{"type": "summary_text", "text": "<badthink><badthink>"}],
        }
        env = _env(harness)
        env.cfg = {
            "invalid_tool_call_patterns": ["[[BAD]]"],
            "thinking_tags": ["<badthink>"],
        }

        # Detector fixture metadata now comes from shared capture, rather than
        # an agent's private result. Token and row fixtures remain unchanged.
        async def captured_metadata(*args):
            manifest = await harness.ledger.manifest("attempt")
            for captured_record, action in zip(
                manifest["records"], actions, strict=True
            ):
                if action["last_output_item"] is not None:
                    captured_record["last_output_item"] = action["last_output_item"]
            return manifest

        env._control.side_effect = captured_metadata
        processed = asyncio.run(
            env._postprocess_receipt_mode(
                {"_ng_rollout_id": "attempt"},
                dict(
                    reward=1.0,
                    response={
                        "id": ids[-1],
                        "output": first["output"] + rewritten["output"],
                    },
                    context_compaction_result=dict(
                        logical_rollout_id="attempt",
                        selected_actions=actions,
                        outcome="completed",
                    ),
                ),
            )
        )
        assert_metadata_only(processed)
        result = RolloutReassembler(
            plane,
            partition_id="canonical",
            staging_partition="staged",
            pad_token_id=0,
            max_seq_len=1024,
        ).finalize_group(
            "group",
            ["attempt"],
            [processed["receipt"]],
            [1.0],
            mask_sample=[False],
            fallback_weight_version=7,
            prompt_idx=99,
            canonical_sample_ids=["group_g0"],
            logical_selections=[processed["logical_selection"]],
        )
        rows = [plane.rows["canonical", key] for key in result.meta.sample_ids]
        assert rows[0]["invalid_tool_call_mask"][0].tolist() == [
            False,
            False,
            True,
            False,
            False,
            False,
        ]
        assert rows[0]["malformed_thinking_mask"][0].tolist() == [
            False,
            False,
            True,
            False,
            False,
            False,
        ]
        assert rows[1]["invalid_tool_call_mask"][0].tolist() == [False, False, False]
        assert rows[1]["malformed_thinking_mask"][0].tolist() == [False, False, True]
        for row in rows:
            assert not torch.any(
                row["malformed_thinking_mask"] & ~row["token_mask"].bool()
            )
        assert sum(tag["num_invalid_tool_calls"] for tag in result.meta.tags) == 1
        assert sum(tag["num_malformed_thinking"] for tag in result.meta.tags) == 2
        assert sum(tag["num_assistant_messages"] for tag in result.meta.tags) == 3
    finally:
        harness.client.close()
        asyncio.run(harness.ledger.close())
