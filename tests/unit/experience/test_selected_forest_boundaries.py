# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selected forests, with real capture and synthetic model outputs.

Strict xfails document deferred provider shapes, not required forest behavior.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any

import pytest
import torch

pytest.importorskip("nemo_gym", reason="requires the paired Gym checkout")

from responses_api_models.vllm_model.tests import test_framework_context as gym

from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
from nemo_rl.environments.gym_selection import select_captured_calls
from nemo_rl.experience.rollout_reassembler import (
    ActionOutputFlags,
    RolloutReassembler,
    RolloutSelection,
    SegmentReceipt,
)
from nemo_rl.models.generation.capture_context import decide_capture_input
from tests.unit.experience.test_logical_owner_finalization import PublicationDataPlane

pytestmark = pytest.mark.nemo_gym


@dataclass
class ForestCapture:
    harness: Any
    plane: PublicationDataPlane
    finalizer: RolloutReassembler

    def capture(self, parents: list[int | None]) -> tuple[list, list[dict]]:
        contexts, replies = [], []
        for index, parent in enumerate(parents):
            prompt = (
                [{"role": "user", "content": f"independent root {index}"}]
                if parent is None
                else contexts[parent]
                + replies[parent]["output"]
                + [{"role": "user", "content": f"observation {index}"}]
            )
            contexts.append(prompt)
            replies.append(gym.assert_clean(self.harness.post("attempt", prompt)))
        records = list(self.harness.manifest("attempt").records)
        # Parent edges must come from production inference, never injected hints.
        assert [r.parent_call_id for r in records] == [
            records[parent].model_call_id if parent is not None else None
            for parent in parents
        ]
        return records, replies


@pytest.fixture
def forest(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> ForestCapture:
    plane = PublicationDataPlane()
    source = TQTokenSource(plane, staging_partition="staged")
    harness = gym.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(plane, staging_partition="staged"),
        fetch_prefix=source.fetch_prefix_token_ids,
        decide_input=decide_capture_input,
    )
    return ForestCapture(
        harness,
        plane,
        RolloutReassembler(
            plane,
            partition_id="canonical",
            staging_partition="staged",
            pad_token_id=0,
            max_seq_len=1024,
        ),
    )


def ordinary(replies: list[dict], *, terminal: str | None = None) -> dict:
    return {
        "response": {
            "id": terminal or replies[-1]["id"],
            "output": [item for reply in replies for item in reply["output"]],
        }
    }


def receipt(records: list, *, terminal: Any) -> dict:
    return {
        "rollout_id": "attempt",
        "manifest": [r.model_dump() for r in records],
        "terminal_model_call_id": terminal.model_call_id,
        "terminal_selection": "declared",
    }


def selection(records: list) -> RolloutSelection:
    return RolloutSelection(
        tuple(r.response_id for r in records),
        tuple(ActionOutputFlags(False, False) for _ in records),
    )


def publish(forest: ForestCapture, records: list, selected: list) -> Any:
    return forest.finalizer.finalize_group(
        "group",
        ["attempt"],
        [receipt(records, terminal=selected[-1])],
        [1.0],
        mask_sample=[False],
        fallback_weight_version=7,
        prompt_idx=99,
        canonical_sample_ids=["group_g0"],
        logical_selections=[selection(selected)],
    )


def assert_loss_once(forest: ForestCapture, selected: list, all_records: list) -> None:
    counts = Counter()
    for (partition, _), fields in forest.plane.rows.items():
        if partition == "canonical":
            tokens = fields["input_ids"].tolist()[0]
            mask = fields["token_mask"].tolist()[0]
            counts.update(
                token for token, active in zip(tokens, mask, strict=True) if active
            )
    expected = Counter(1001 + all_records.index(r) for r in selected)
    assert counts == expected


@pytest.mark.parametrize("compact", [False, True])
def test_twenty_calls_keep_baseline_rows(forest: ForestCapture, compact: bool) -> None:
    parents = [None] + list(range(19))
    if compact:
        parents[10] = None
    records, replies = forest.capture(parents)
    selected = select_captured_calls(records, ordinary(replies))
    result = publish(forest, records, selected)
    assert result.valid_row_count == (2 if compact else 1)
    assert_loss_once(forest, selected, records)
    assert not any(partition == "staged" for partition, _ in forest.plane.rows)


@pytest.mark.parametrize(
    "parents,expected_rows",
    [
        ([None, None, 0, 1], 2),
        ([None, 0, 0], 2),
        ([None, None, 0, None, 1, 3], 3),
        ([None, None, None, 1, 2, 0], 3),
        ([None, None, 0, None, 1, 3, 2], 3),
        ([None, None, 0, None, 3, None, 4, 5], 4),
        ([None, 0, 0, 1, 1, 2], 3),
    ],
    ids=[
        "interleaved",
        "shared-fork",
        "interleaved-with-compaction",
        "two-workers-parent-resumes",
        "worker-compacts-parent-resumes",
        "both-compact",
        "nested-forks",
    ],
)
def test_selected_forest_publishes_once(
    forest: ForestCapture, parents: list[int | None], expected_rows: int
) -> None:
    records, replies = forest.capture(parents)
    selected = select_captured_calls(records, ordinary(replies))
    assert selected == records
    result = publish(forest, records, selected)
    assert result.valid_row_count == expected_rows
    assert_loss_once(forest, selected, records)


def test_shared_action_masks_counts_and_prompt_are_owned_once(
    forest: ForestCapture,
) -> None:
    records, replies = forest.capture([None, 0, 0])
    selected = select_captured_calls(records, ordinary(replies))
    flags = (
        ActionOutputFlags(True, True),
        ActionOutputFlags(False, False),
        ActionOutputFlags(False, True),
    )
    result = forest.finalizer.finalize_group(
        "group",
        ["attempt"],
        [receipt(records, terminal=records[-1])],
        [1.0],
        mask_sample=[False],
        fallback_weight_version=7,
        prompt_idx=99,
        canonical_sample_ids=["group_g0"],
        logical_selections=[
            RolloutSelection(tuple(r.response_id for r in selected), flags)
        ],
    )
    assert result.valid_row_count == 2
    rows = [forest.plane.rows["canonical", key] for key in result.meta.sample_ids]
    assert rows[0]["token_mask"].tolist() == [[0, 0, 1, 0, 0, 1]]
    assert rows[1]["token_mask"].tolist() == [[0, 0, 0, 0, 0, 1]]
    assert rows[0]["invalid_tool_call_mask"].tolist() == [
        [False, False, True, False, False, False]
    ]
    assert not rows[1]["invalid_tool_call_mask"].any()
    assert rows[0]["malformed_thinking_mask"].sum() == 1
    assert rows[1]["malformed_thinking_mask"].tolist() == [
        [False, False, False, False, False, True]
    ]
    assert sum(tag["num_invalid_tool_calls"] for tag in result.meta.tags) == 1
    assert sum(tag["num_malformed_thinking"] for tag in result.meta.tags) == 2
    assert [tag["num_assistant_messages"] for tag in result.meta.tags] == [2, 1]
    assert {tag["logical_rollout_id"] for tag in result.meta.tags} == {"group_g0"}
    assert_loss_once(forest, selected, records)
    assert not any(partition == "staged" for partition, _ in forest.plane.rows)


def test_shared_forest_loss_and_gradient_match_independent_calls(
    forest: ForestCapture,
) -> None:
    records, replies = forest.capture([None, 0, 0])
    selected = select_captured_calls(records, ordinary(replies))
    reference = [
        forest.finalizer.finalize_rollout(
            "attempt",
            receipt([records[i] for i in indices], terminal=records[indices[-1]]),
            reward=1.0,
            context_compaction=True,
        )
        for indices in ([0], [0, 1], [0, 2])
    ]
    result = publish(forest, records, selected)
    # A small causal scorer: each position sees only the tokens up to that point.
    # The reference scores each call's final generated token independently.
    embedding = (
        torch.sin(torch.arange(2048 * 4, dtype=torch.float64))
        .reshape(2048, 4)
        .requires_grad_()
    )
    head = (
        torch.cos(torch.arange(4 * 2048, dtype=torch.float64))
        .reshape(4, 2048)
        .requires_grad_()
    )

    def losses(tokens: list[int]) -> torch.Tensor:
        ids = torch.tensor(tokens)
        logits = torch.tanh(embedding[ids].cumsum(0)) @ head
        return torch.nn.functional.cross_entropy(logits[:-1], ids[1:], reduction="none")

    expected = sum(losses(row.token_ids)[-1] for row in reference)
    actual = sum(
        (
            losses(forest.plane.rows["canonical", key]["input_ids"][0].tolist())
            * forest.plane.rows["canonical", key]["token_mask"][0, 1:]
        ).sum()
        for key in result.meta.sample_ids
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for got, want in zip(
        torch.autograd.grad(actual, (embedding, head)),
        torch.autograd.grad(expected, (embedding, head)),
        strict=True,
    ):
        torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)


def test_omitted_ancestor_stays_rejected(forest: ForestCapture) -> None:
    records, replies = forest.capture([None, 0, 1])
    selected = select_captured_calls(records, ordinary([replies[0], replies[2]]))
    with pytest.raises(ValueError, match="selected predecessor"):
        publish(forest, records, selected)
    assert not any(partition == "canonical" for partition, _ in forest.plane.rows)


def test_excluded_leaf_preserves_selected_ancestor(forest: ForestCapture) -> None:
    records, replies = forest.capture([None, 0, None])
    selected = select_captured_calls(records, ordinary([replies[0], replies[2]]))
    result = publish(forest, records, selected)
    assert result.valid_row_count == 2
    assert_loss_once(forest, selected, records)
    assert not any(partition == "staged" for partition, _ in forest.plane.rows)


@pytest.mark.xfail(
    strict=True, raises=ValueError, reason="Selector requires global capture order"
)
def test_per_chain_envelopes_preserve_membership(forest: ForestCapture) -> None:
    records, replies = forest.capture([None, None, 0, 1])
    result = {
        "responses": [
            ordinary([replies[0], replies[2]])["response"],
            ordinary([replies[1], replies[3]])["response"],
        ]
    }
    selected = select_captured_calls(records, result)
    assert {r.model_call_id for r in selected} == {r.model_call_id for r in records}


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="Scored terminal must be the last selected call",
)
def test_explicit_scored_parent_can_precede_worker(forest: ForestCapture) -> None:
    records, replies = forest.capture([None, None, 0, 1])
    result = ordinary(replies)
    result["terminal_response_id"] = records[2].response_id
    # Synthetic aggregate envelope ID leaves explicit terminal as authority.
    result["response"]["id"] = "aggregate"
    selected = select_captured_calls(records, result)
    assert selected == records
    plan = forest.finalizer._plan_selected_calls(
        ["attempt"], [receipt(records, terminal=records[2])], [selection(selected)]
    )
    assert len(plan[0]) == 2


def test_shared_paths_need_loss_ownership_beyond_planning(
    forest: ForestCapture,
) -> None:
    records, _ = forest.capture([None, 0, 0])
    segments = []
    for indices in ([0, 1], [0, 2]):
        path = [records[i] for i in indices]
        row_receipt = receipt(path, terminal=path[-1])
        flags = selection(path).action_flags
        # Isolate the downstream boundary with explicit path receipts. This is
        # NOT a provider-derived selection or a proposed parallel planner.
        row = forest.finalizer.finalize_rollout(
            "attempt",
            row_receipt,
            reward=1.0,
            context_compaction=True,
            action_flags=flags,
        )
        assert row.valid
        assert row.token_ids == [
            10,
            11,
            1001,
            10 * (indices[-1] + 1),
            10 * (indices[-1] + 1) + 1,
            1001 + indices[-1],
        ]
        segments.append(
            SegmentReceipt(
                "attempt",
                row_receipt,
                tuple(r.response_id for r in path),
                action_flags=flags,
            )
        )
    prepared = forest.finalizer._finalize_logical_rows(
        "group",
        ["attempt"],
        [segments],
        [1.0],
        [False],
        canonical_sample_ids=["group_g0"],
    )
    assert prepared.valid_owner_count == 1
    assert all(row.valid for row in prepared.rows)
    loss_tokens = Counter(
        token
        for row in prepared.rows
        for token, active in zip(row.token_ids, row.token_mask, strict=True)
        if active
    )
    assert loss_tokens == Counter([1001, 1002, 1003])


@pytest.mark.parametrize("bad_parent", ["missing", "self", "future"])
def test_invalid_selected_edges_fail_before_publication(
    forest: ForestCapture, bad_parent: str
) -> None:
    records, _ = forest.capture([None, 0, 0])
    parent = {
        "missing": "not-captured",
        "self": records[1].model_call_id,
        "future": records[2].model_call_id,
    }[bad_parent]
    records[1] = records[1].model_copy(update={"parent_call_id": parent})
    with pytest.raises(ValueError, match="selected predecessor"):
        publish(forest, records, records)
    assert not any(partition == "canonical" for partition, _ in forest.plane.rows)


@pytest.mark.parametrize("broken_index", [0, 1, 2])
def test_one_corrupt_shared_path_masks_entire_owner(
    forest: ForestCapture, broken_index: int
) -> None:
    records, _ = forest.capture([None, 0, 0])
    # Alter captured tensor bytes without changing the authenticated record.
    key = records[broken_index].staging_key
    staged = forest.plane.rows["staged", key]
    staged["token_ids_delta"][0, -1] += 1
    if broken_index == 0:
        with pytest.raises(ValueError, match="unverifiable initial prompt"):
            publish(forest, records, records)
    else:
        result = publish(forest, records, records)
        assert result.valid_row_count == 0
        assert result.meta is None
        assert not any(partition == "staged" for partition, _ in forest.plane.rows)
    assert not any(partition == "canonical" for partition, _ in forest.plane.rows)


def test_shared_plan_and_masks_survive_selection_serialization(
    forest: ForestCapture,
) -> None:
    import json
    from dataclasses import asdict

    records, _ = forest.capture([None, 0, 0, None, 3])
    selected = selection(records)
    saved = json.loads(json.dumps(asdict(selected)))
    restored = RolloutSelection(
        tuple(saved["response_ids"]),
        tuple(ActionOutputFlags(**flags) for flags in saved["action_flags"]),
        truncated=saved["truncated"],
    )
    wire = receipt(records, terminal=records[-1])
    plans = [
        forest.finalizer._plan_selected_calls(["attempt"], [wire], [value])
        for value in (selected, restored)
    ]
    assert plans[0] == plans[1]
    prepared = [
        forest.finalizer._finalize_logical_rows(
            "group",
            ["attempt"],
            plan,
            [1.0],
            [False],
            canonical_sample_ids=["group_g0"],
        )
        for plan in plans
    ]
    assert prepared[0] == prepared[1]
    assert prepared[0].valid_owner_count == 1
