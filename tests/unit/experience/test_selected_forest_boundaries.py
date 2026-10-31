# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selected forests, with real capture and synthetic model outputs.

Strict xfails document deferred provider shapes, not required forest behavior.
"""

import asyncio
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import torch

pytest.importorskip("nemo_gym", reason="requires the paired Gym checkout")

from responses_api_models.vllm_model.tests import test_framework_context as gym

from nemo_rl.data_plane.schema import ROLLOUT_METRICS
from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
from nemo_rl.environments.call_selection import select_captured_calls
from nemo_rl.experience.reward_penalties import RewardChecks, RewardLogContext
from nemo_rl.experience.rollout_reassembler import (
    ActionOutputFlags,
    RolloutReassembler,
    RolloutSelection,
    SegmentReceipt,
)
from nemo_rl.experience.rollout_reassembler_actor import (
    ReassemblyRequest,
    RolloutReassemblerActor,
)
from nemo_rl.experience.rollouts import EffortLevelsConfig
from nemo_rl.models.generation.capture_context import decide_capture_input
from tests.unit.experience.test_forest_dispatch import _env
from tests.unit.experience.test_logical_owner_finalization import PublicationDataPlane

pytestmark = pytest.mark.nemo_gym


@dataclass
class ForestCapture:
    harness: Any
    plane: PublicationDataPlane
    finalizer: RolloutReassembler

    def capture(
        self, parents: list[int | None], *, owner: str = "attempt"
    ) -> tuple[list, list[dict]]:
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
            replies.append(gym.assert_clean(self.harness.post(owner, prompt)))
        records = list(self.harness.manifest(owner).records)
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
        fetch_prefix=lambda keys: source.fetch_prefix_chains(keys).expanded,
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
        "capture_key": "attempt",
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


@pytest.mark.parametrize("policy", ["returned_history", "all_captured"])
@pytest.mark.parametrize("reverse_manifest", [False, True])
def test_selection_policy_reuses_forest_publication(forest, policy, reverse_manifest):
    # A -> abandoned B; A -> scored C; independent D -> E.
    records, replies = forest.capture([None, 0, 0, None, 3])
    accepted = [records[i] for i in (0, 2, 3, 4)]
    env = _env(forest.harness)
    env._selection_policy = policy
    original = env._drained_manifest

    async def drained(owner):
        manifest = await original(owner)
        if reverse_manifest:
            manifest["records"].reverse()
        return manifest

    env._drained_manifest = drained
    output = ordinary([replies[i] for i in (0, 2, 3, 4)], terminal=replies[2]["id"])
    output["reward"] = 1.0
    processed = asyncio.run(
        env._postprocess_receipt_mode({"_ng_rollout_id": "attempt"}, output)
    )
    assert processed["receipt"]["terminal_model_call_id"] == records[2].model_call_id
    selected = records if policy == "all_captured" else accepted
    assert set(processed["logical_selection"].response_ids) == {
        r.response_id for r in selected
    }
    result = forest.finalizer.finalize_group(
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
    assert result.meta is not None
    assert_loss_once(forest, selected, records)


@pytest.mark.parametrize("policy", ["returned_history", "all_captured"])
@pytest.mark.parametrize("interleaved", [False, True])
def test_forest_restart_retains_selection_terminal_and_once_only_loss(
    forest: ForestCapture, tmp_path: Path, policy: str, interleaved: bool
) -> None:
    from nemo_rl.experience.rollout_recovery import RolloutRecoveryLedger
    from tests.unit.experience.test_rollout_recovery import _mutate, _prompt, _reserve

    ledger = RolloutRecoveryLedger()
    group = _reserve(
        ledger,
        group_id="group",
        prompt_id="7",
        prompt_payload=_prompt(),
        expected_generations=1,
        target_step=0,
        start_weight_version=7,
    )
    owner = group.gate_rollout_id(0)
    _mutate(lambda cut: ledger.mark_group_dispatched(cut, "group"))
    # Shared ancestor, omitted retry, and an independent post-compaction context.
    parents = [None, 0, 0, None, 3]
    accepted_indices, scored_index = (0, 2, 3, 4), 2
    expected_rows = 3 if policy == "all_captured" else 2
    if interleaved:
        # Main A -> retry C / accepted E; worker B -> D / H;
        # the main agent compacts into independent F -> scored G.
        parents = [None, None, 0, 1, 0, None, 5, 1]
        accepted_indices, scored_index = (0, 1, 3, 4, 5, 6, 7), 6
        expected_rows = 5 if policy == "all_captured" else 4
    records, replies = forest.capture(parents, owner=owner)
    env = _env(forest.harness)
    env._selection_policy = policy
    output = ordinary(
        [replies[i] for i in accepted_indices], terminal=replies[scored_index]["id"]
    )
    output["reward"] = 1.0
    processed = asyncio.run(
        env._postprocess_receipt_mode({"_ng_rollout_id": owner}, output)
    )
    processed["receipt"]["manifest"].reverse()
    checks = RewardChecks(False, False, False)
    context = RewardLogContext("agent", '{"answer":"done"}')
    _mutate(
        lambda cut: ledger.mark_sibling_sealed(
            cut,
            "group",
            generation_index=0,
            gate_rollout_id=owner,
            receipt=processed["receipt"],
            reward=1.0,
            mask_sample=False,
            logical_selection=processed["logical_selection"],
            reward_checks=checks,
            reward_log_context=context,
        )
    )
    path = tmp_path / "forest-recovery.pt"
    torch.save(ledger.state_dict(), path)
    restored = RolloutRecoveryLedger.from_state_dict(
        torch.load(path, weights_only=True)
    )
    _mutate(lambda cut: restored.prepare_for_restart(cut))
    attempt = restored.get_group("group").siblings[0].current_attempt
    assert attempt == ledger.get_group("group").siblings[0].current_attempt
    assert (
        attempt.receipt["terminal_model_call_id"] == records[scored_index].model_call_id
    )
    owners, sample_ids, receipts, rewards, masks, reward_checks, log_contexts = (
        restored.finalization_inputs("group")
    )
    assert reward_checks == [checks] and log_contexts == [context]
    result = forest.finalizer.finalize_group(
        "group",
        owners,
        receipts,
        rewards,
        mask_sample=masks,
        fallback_weight_version=7,
        prompt_idx=7,
        canonical_sample_ids=sample_ids,
        logical_selections=[attempt.logical_selection],
        execution_row_multiple=4,
        reward_checks=reward_checks,
        reward_log_contexts=log_contexts,
    )
    assert result.meta is not None
    assert result.meta.size == ((expected_rows + 3) // 4) * 4
    assert result.total_row_count == result.valid_row_count == expected_rows
    selected = (
        records if policy == "all_captured" else [records[i] for i in accepted_indices]
    )
    assert_loss_once(forest, selected, records)
    assert not any(partition == "staged" for partition, _ in forest.plane.rows)


@pytest.mark.parametrize("policy", ["returned_history", "all_captured"])
@pytest.mark.parametrize("reverse_manifest", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        "none",
        "failed_leaf",
        "unknown_leaf",
        "filtered_leaf",
        "length_leaf",
        "max_tokens_leaf",
        "max_output_tokens_leaf",
        "failed_parent",
        "failed_terminal",
        "uncommitted",
        "capture_lost",
        "pending",
        "cancel_after_commit",
        "selected_parent_capture_failure",
        "selected_terminal_capture_failure",
        "uncommitted_without_record",
        "settled_capture_lost",
        "refused",
    ],
)
def test_subagent_failures_reach_final_training_rows(
    forest: ForestCapture,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
    reverse_manifest: bool,
    failure: str,
) -> None:
    from nemo_gym.token_id_capture import UNCOMMITTED_CALL_REASON

    # Main A -> B / retry C; worker D -> E / H;
    # main compacts into F -> scored G. H completes after G.
    records, replies = forest.capture([None, 0, 0, None, 3, None, 5, 3])
    completion_failure = {
        "failed_leaf": (2, "failed", "stop"),
        "unknown_leaf": (2, "unknown", None),
        "filtered_leaf": (2, "incomplete", "content_filter"),
        "length_leaf": (2, "incomplete", "length"),
        "max_tokens_leaf": (2, "incomplete", "max_tokens"),
        "max_output_tokens_leaf": (2, "incomplete", "max_output_tokens"),
        "failed_parent": (3, "failed", "stop"),
        "failed_terminal": (6, "failed", "stop"),
    }.get(failure)
    pending = failure in ("uncommitted", "capture_lost", "pending")
    if pending:
        asyncio.run(forest.harness.ledger.begin_call("attempt", "lost"))
        if failure != "pending":
            asyncio.run(
                forest.harness.ledger.record_failure(
                    "attempt",
                    "lost",
                    "capture_lost"
                    if failure == "capture_lost"
                    else UNCOMMITTED_CALL_REASON,
                )
            )
        # A failure does not acknowledge the worker's possible late write.
        assert forest.harness.manifest("attempt").pending_call_ids == ["lost"]
        forest.plane.rows["staged", "attempt/lost"] = forest.plane.rows[
            "staged", records[0].staging_key
        ].clone()
    if failure == "cancel_after_commit":
        record_commit = forest.harness.ledger.record

        async def commit_then_cancel(commit: Any) -> None:
            # Cancel after the durable write, before the request observes success.
            await record_commit(commit)
            raise asyncio.CancelledError

        monkeypatch.setattr(forest.harness.ledger, "record", commit_then_cancel)
        assert (
            forest.harness.post(
                "attempt", [{"role": "user", "content": "cancelled auxiliary task"}]
            ).status_code
            == 500
        )
        settled = forest.harness.manifest("attempt")
        records = list(settled.records)
        assert len(records) == 9 and not settled.pending_call_ids
        assert settled.failures and all(
            f.model_call_id == records[-1].model_call_id
            and f.reason == UNCOMMITTED_CALL_REASON
            for f in settled.failures
        )
    elif failure in (
        "selected_parent_capture_failure",
        "selected_terminal_capture_failure",
        "uncommitted_without_record",
        "settled_capture_lost",
    ):
        failed_call = {
            "selected_parent_capture_failure": records[3].model_call_id,
            "selected_terminal_capture_failure": records[6].model_call_id,
            "uncommitted_without_record": "untracked",
            "settled_capture_lost": records[2].model_call_id,
        }[failure]
        asyncio.run(
            forest.harness.ledger.record_failure(
                "attempt",
                failed_call,
                "capture_lost"
                if failure == "settled_capture_lost"
                else UNCOMMITTED_CALL_REASON,
            )
        )
        assert not forest.harness.manifest("attempt").pending_call_ids
    elif failure == "refused":
        asyncio.run(forest.harness.ledger.begin_call("attempt", "refused"))
        asyncio.run(forest.harness.ledger.refuse_call("attempt", "refused"))
        assert not forest.harness.manifest("attempt").pending_call_ids
    if failure == "pending":
        monkeypatch.setattr("nemo_rl.environments.nemo_gym._CAPTURE_DRAIN_TIMEOUT_S", 0)
    env = _env(forest.harness)
    env._selection_policy = policy
    control = env._control

    async def capture_control(method: str, path: str) -> dict:
        manifest = await control(method, path)
        if completion_failure is not None:
            index, status, reason = completion_failure
            for record in manifest["records"]:
                if record["model_call_id"] == records[index].model_call_id:
                    record.update(response_status=status, finish_reason=reason)
                    if failure in ("unknown_leaf", "filtered_leaf"):
                        record["last_output_item"] = None
        if reverse_manifest:
            manifest["records"].reverse()
        return manifest

    env._control = AsyncMock(side_effect=capture_control)
    accepted_indices = (0, 1, 3, 4, 5, 6, 7)
    output = ordinary([replies[i] for i in accepted_indices], terminal=replies[6]["id"])
    output["reward"] = 1.0
    processed = asyncio.run(
        env._postprocess_receipt_mode({"_ng_rollout_id": "attempt"}, output)
    )
    masked = (
        pending
        or failure
        in (
            "failed_parent",
            "failed_terminal",
            "selected_parent_capture_failure",
            "selected_terminal_capture_failure",
            "uncommitted_without_record",
            "settled_capture_lost",
        )
        or (failure == "cancel_after_commit" and policy == "all_captured")
    )
    assert processed["receipt"]["capture_poisoned"] is masked
    truncated = policy == "all_captured" and failure in (
        "length_leaf",
        "max_tokens_leaf",
        "max_output_tokens_leaf",
    )
    assert processed["truncated"] is truncated
    result = forest.finalizer.finalize_group(
        "group",
        ["attempt"],
        [processed["receipt"]],
        [1.0],
        mask_sample=[False],
        fallback_weight_version=7,
        prompt_idx=99,
        canonical_sample_ids=["group_g0"],
        logical_selections=[processed["logical_selection"]],
        execution_row_multiple=4,
    )
    selected = (
        records
        if policy == "all_captured"
        and failure not in ("failed_leaf", "unknown_leaf", "filtered_leaf")
        else [records[i] for i in accepted_indices]
    )
    assert_loss_once(forest, [] if masked else selected, records)
    assert result.valid_row_count == (0 if masked else (5 if len(selected) == 8 else 4))
    # Termination rates count the rollout once, not once per segment row.
    assert result.metrics["finalize/truncated_count"] == float(truncated and not masked)
    if result.meta is not None:
        for key, tag in zip(result.meta.sample_ids, result.meta.tags, strict=True):
            row = forest.plane.rows["canonical", key]
            assert row["truncated"].item() == (
                truncated and not tag["is_execution_padding"]
            )
    assert {key for partition, key in forest.plane.rows if partition == "staged"} == (
        {"attempt/lost"} if pending else set()
    )


def test_all_captured_does_not_require_returned_execution_history(forest):
    records, replies = forest.capture([None, 0, None])
    env = _env(forest.harness)
    env._selection_policy = "all_captured"
    # Explicit scored identity suffices; no output history is supplied.
    result = {"terminal_response_id": replies[1]["id"], "reward": 1.0}
    processed = asyncio.run(
        env._postprocess_receipt_mode({"_ng_rollout_id": "attempt"}, result)
    )
    assert set(processed["logical_selection"].response_ids) == {
        r.response_id for r in records
    }
    assert processed["receipt"]["terminal_model_call_id"] == records[1].model_call_id


@pytest.mark.parametrize("policy", ["returned_history", "all_captured"])
@pytest.mark.parametrize(
    "status,reason,empty_output",
    [
        ("failed", "stop", False),
        ("failed", "stop", True),
        ("unknown", None, False),
        ("unknown", None, True),
        ("incomplete", "content_filter", False),
        ("incomplete", "content_filter", True),
    ],
)
@pytest.mark.parametrize("failure_index", [0, -1])
def test_unsuccessful_selected_completion_masks_the_owner(
    forest, policy, status, reason, empty_output, failure_index
):
    records, replies = forest.capture([None, 0, None, 2])
    env = _env(forest.harness)
    env._selection_policy = policy
    manifest = forest.harness.manifest("attempt").model_dump(mode="json")
    manifest["records"][failure_index]["response_status"] = status
    manifest["records"][failure_index]["finish_reason"] = reason
    if empty_output:
        manifest["records"][failure_index]["last_output_item"] = None
    env._control = AsyncMock(return_value=manifest)
    output = ordinary(replies)
    output["reward"] = 1.0
    processed = asyncio.run(
        env._postprocess_receipt_mode({"_ng_rollout_id": "attempt"}, output)
    )
    assert processed["receipt"]["capture_poisoned"] is True
    assert processed["receipt"]["failure_reason"] == "logical_execution_failure"
    result = forest.finalizer.finalize_group(
        "group",
        ["attempt"],
        [processed["receipt"]],
        [1.0],
        mask_sample=[False],
        fallback_weight_version=7,
        prompt_idx=99,
        canonical_sample_ids=["group_g0"],
        logical_selections=[processed["logical_selection"]],
        execution_row_multiple=4,
    )
    assert result.valid_row_count == 0
    assert_loss_once(forest, [], records)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("response_status", None, "completion metadata"),
        ("last_output_item", None, "output validation evidence"),
        ("response_status", "incomplete", "finish reason"),
        ("finish_reason", "", "finish reason"),
    ],
)
def test_all_captured_does_not_hide_missing_leaf_completion_evidence(
    forest: ForestCapture, field: str, value: Any, error: str
) -> None:
    _, replies = forest.capture([None, 0, None])
    manifest = forest.harness.manifest("attempt").model_dump(mode="json")
    leaf = manifest["records"][-1]
    leaf[field] = value
    if field == "finish_reason":
        leaf["response_status"] = "incomplete"
    if field == "response_status" and value == "incomplete":
        leaf["finish_reason"] = None
    env = _env(forest.harness)
    env._selection_policy = "all_captured"
    env._control = AsyncMock(return_value=manifest)
    with pytest.raises(ValueError, match=error):
        asyncio.run(
            env._postprocess_receipt_mode(
                {"_ng_rollout_id": "attempt"}, ordinary(replies[:2]) | {"reward": 1.0}
            )
        )


def test_ordinary_capture_reconstructs_without_framework_output_metadata(
    monkeypatch, tmp_path
):
    from nemo_gym.token_id_capture.staging.rebuild import verify_and_linearize
    from nemo_gym.token_id_capture.staging.records import (
        RolloutReceipt,
        StagedCallBaseSnapshot,
    )

    harness = gym.make_capture_harness(
        monkeypatch,
        tmp_path,
        decide_input=decide_capture_input,
        parent_resolution_mode="fingerprint",
    )
    history = [{"role": "user", "content": "task"}]
    first = gym.assert_clean(harness.post("attempt", history))
    second = gym.assert_clean(
        harness.post(
            "attempt",
            history + first["output"] + [{"role": "user", "content": "continue"}],
        )
    )
    manifest = harness.manifest("attempt")
    assert len(manifest.records) == 2
    assert manifest.records[1].parent_call_id == manifest.records[0].model_call_id
    assert all(
        r.output_items is None
        and r.response_status is None
        and r.last_output_item is None
        for r in manifest.records
    )
    env = _env(harness)
    env._forest_training = False
    processed = asyncio.run(
        env._postprocess_receipt_mode(
            {"_ng_rollout_id": "attempt"}, {"response": second, "reward": 1.0}
        )
    )
    captured_receipt = RolloutReceipt.model_validate(processed["receipt"])
    assert not captured_receipt.capture_poisoned
    snapshots = [
        StagedCallBaseSnapshot.model_validate(r.model_dump(exclude={"extras"}))
        for r in harness.sink.records.values()
    ]
    row = verify_and_linearize(captured_receipt, snapshots)
    assert row.token_ids == [10, 11, 1001, 20, 21, 1002]
    assert row.token_mask == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize("policy", ["returned_history", "all_captured"])
@pytest.mark.parametrize("continuation", [False, True])
def test_calls_after_lost_ack_remain_masked_and_preserve_uncertain_staging(
    forest, policy, continuation
):
    _, replies = forest.capture([None])
    forest.harness.omit_coords = True
    assert (
        forest.harness.post(
            "attempt", [{"role": "user", "content": "subagent task"}]
        ).status_code
        == 500
    )
    failed = forest.harness.manifest("attempt")
    assert len(failed.records) == 1 and len(failed.pending_call_ids) == 1
    uncertain_key = f"attempt/{failed.pending_call_ids[0]}"
    assert ("staged", uncertain_key) in forest.plane.rows

    forest.harness.omit_coords = False
    history = (
        [{"role": "user", "content": "independent root 0"}]
        + replies[0]["output"]
        + [{"role": "user", "content": "continue healthy branch"}]
        if continuation
        else [{"role": "user", "content": "another independent task"}]
    )
    replies.append(gym.assert_clean(forest.harness.post("attempt", history)))
    manifest = forest.harness.manifest("attempt")
    assert len(manifest.records) == 2
    assert manifest.pending_call_ids == failed.pending_call_ids
    assert manifest.failures == failed.failures
    assert manifest.records[1].parent_call_id == (
        manifest.records[0].model_call_id if continuation else None
    )

    env = _env(forest.harness)
    env._selection_policy = policy
    output = ordinary(replies)
    output["reward"] = 1.0
    processed = asyncio.run(
        env._postprocess_receipt_mode({"_ng_rollout_id": "attempt"}, output)
    )
    assert processed["receipt"]["capture_poisoned"]
    assert processed["receipt"]["pending_call_ids"] == failed.pending_call_ids
    result = forest.finalizer.finalize_group(
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
    assert result.valid_row_count == 0
    assert_loss_once(forest, [], manifest.records)
    assert {key for partition, key in forest.plane.rows if partition == "staged"} == {
        uncertain_key
    }


@pytest.mark.parametrize("response_id", [None, "exporter-local-response"])
@pytest.mark.parametrize("cumulative", [False, True])
@pytest.mark.parametrize("unique_output", [False, True])
def test_all_captured_attributes_exported_content(
    forest, monkeypatch, response_id, cumulative, unique_output
):
    # The scored conversation ends before an independent worker completes.
    # Exporter-local identities must not force selection of that later leaf.
    if unique_output:
        worker = gym.NeMoGymAsyncOpenAI.create_chat_completion

        async def distinct_worker(client, **body):
            payload = await worker(client, **body)
            payload["choices"][0]["message"]["content"] += payload["id"]
            return payload

        monkeypatch.setattr(
            gym.NeMoGymAsyncOpenAI, "create_chat_completion", distinct_worker
        )
    records, replies = forest.capture([None, 0, None])
    env = _env(forest.harness)
    env._selection_policy = "all_captured"
    result = ordinary(replies[:2] if cumulative else [replies[1]])
    result["response"]["id"] = response_id
    result["response"]["output"] = [
        {**item, "id": f"exporter-local-item-{index}"}
        for index, item in enumerate(result["response"]["output"])
    ]
    result["reward"] = 1.0
    if not cumulative and not unique_output:
        # This fixture otherwise returns the same "answer" on every call.
        # Its final text alone cannot identify which call was scored.
        with pytest.raises(ValueError, match="content_ambiguous"):
            asyncio.run(
                env._postprocess_receipt_mode({"_ng_rollout_id": "attempt"}, result)
            )
        assert not any(partition == "canonical" for partition, _ in forest.plane.rows)
        return
    processed = asyncio.run(
        env._postprocess_receipt_mode({"_ng_rollout_id": "attempt"}, result)
    )
    assert processed["receipt"]["terminal_model_call_id"] == records[1].model_call_id
    assert set(processed["logical_selection"].response_ids) == {
        record.response_id for record in records
    }
    finalized = forest.finalizer.finalize_group(
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
    assert finalized.valid_row_count == 2
    assert_loss_once(forest, records, records)


@pytest.mark.parametrize("contradictory", [False, True])
def test_all_captured_never_guesses_scored_terminal(forest, contradictory):
    records, replies = forest.capture([None, None])
    env = _env(forest.harness)
    env._selection_policy = "all_captured"
    result = {"reward": 1.0}
    if contradictory:
        result.update(response=replies[0], terminal_response_id=replies[1]["id"])
    with pytest.raises(ValueError, match="identifiable scored terminal"):
        asyncio.run(
            env._postprocess_receipt_mode({"_ng_rollout_id": "attempt"}, result)
        )


@pytest.mark.parametrize("compact", [False, True])
def test_twenty_calls_keep_baseline_rows(forest: ForestCapture, compact: bool) -> None:
    parents = [None] + list(range(19))
    if compact:
        parents[10] = None
    records, replies = forest.capture(parents)
    selected = select_captured_calls(records, ordinary(replies))[0]
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
    selected = select_captured_calls(records, ordinary(replies))[0]
    assert selected == records
    result = publish(forest, records, selected)
    assert result.valid_row_count == expected_rows
    assert_loss_once(forest, selected, records)


def test_shared_action_masks_counts_and_prompt_are_owned_once(
    forest: ForestCapture,
) -> None:
    records, replies = forest.capture([None, 0, 0])
    selected = select_captured_calls(records, ordinary(replies))[0]
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
    selected = select_captured_calls(records, ordinary(replies))[0]
    reference = [
        forest.finalizer.finalize_rollout(
            "attempt",
            receipt([records[i] for i in indices], terminal=records[indices[-1]]),
            reward=1.0,
            forest_training=True,
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
    selected = select_captured_calls(records, ordinary([replies[0], replies[2]]))[0]
    with pytest.raises(ValueError, match="selected predecessor"):
        publish(forest, records, selected)
    assert not any(partition == "canonical" for partition, _ in forest.plane.rows)


@pytest.mark.parametrize("indices", [(2,), (0, 2), (0, 1, 2)])
@pytest.mark.parametrize("custom_policy", [False, True])
def test_incomplete_history_masks_only_its_owner(
    forest: ForestCapture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    indices: tuple[int, ...],
    custom_policy: bool,
) -> None:
    records, replies = forest.capture([None, 0, 1])
    healthy = [gym.assert_clean(forest.harness.post("healthy", gym.HISTORY))]
    healthy.append(
        gym.assert_clean(
            forest.harness.post(
                "healthy",
                gym.HISTORY
                + healthy[0]["output"]
                + [{"role": "user", "content": "next observation"}],
            )
        )
    )
    healthy_records = list(forest.harness.manifest("healthy").records)
    env = _env(forest.harness)
    if custom_policy:
        (tmp_path / "closure_selection.py").write_text(
            "def choose(records, result):\n"
            "    ids = result['selected_response_ids']\n"
            "    return [r for r in records if r.response_id in ids], result['response']['id']\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        env._selection_policy = "closure_selection:choose"
    processed = []
    for owner, history, expected, reward in (
        ("attempt", [replies[i] for i in indices], [records[i] for i in indices], 1.0),
        ("healthy", healthy, healthy_records, 0.75),
    ):
        output = ordinary(history) | {
            "reward": reward,
            "selected_response_ids": [r.response_id for r in expected],
        }
        item = asyncio.run(
            env._postprocess_receipt_mode({"_ng_rollout_id": owner}, output)
        )
        assert item["logical_selection"].response_ids == tuple(
            r.response_id for r in expected
        )
        processed.append(item)
    incomplete = len(indices) < len(records)
    assert processed[0]["receipt"]["capture_poisoned"] is incomplete
    if incomplete:
        assert processed[0]["receipt"]["failure_reason"] == "selection_not_closed"
    assert not processed[1]["receipt"]["capture_poisoned"]
    reference = forest.finalizer.finalize_rollout(
        "healthy", processed[1]["receipt"], reward=0.75, forest_training=True
    )
    assert reference.valid
    request = ReassemblyRequest(
        group_id="group",
        rollout_ids=("attempt", "healthy"),
        canonical_sample_ids=("group_g0", "group_g1"),
        receipts=tuple(item["receipt"] for item in processed),
        rewards=(1.0, 0.75),
        fallback_weight_version=7,
        prompt_idx=99,
        mask_sample=(False, False),
        logical_selections=tuple(item["logical_selection"] for item in processed),
        execution_row_multiple=4,
    )
    cleanup_ids = request.cleanup_sample_ids
    actor = object.__new__(RolloutReassemblerActor.__ray_metadata__.modified_class)
    actor._finalizer = forest.finalizer
    result = actor.finalize(request)
    assert result.meta is not None and not result.dropped
    assert result.meta.size == 4
    assert tuple(result.meta.sample_ids) == cleanup_ids
    assert result.valid_row_count == (1 if incomplete else 2)
    assert result.metrics["finalize/selection_not_closed_rollouts"] == int(incomplete)
    assert result.metrics["finalize/logical_owner_count"] == 2
    assert result.metrics["finalize/valid_logical_owner_count"] == (
        1 if incomplete else 2
    )
    for key, tag in zip(result.meta.sample_ids, result.meta.tags, strict=True):
        row = forest.plane.rows["canonical", key]
        if tag["logical_slot"] == 1:
            length = row["input_lengths"].item()
            assert row["input_ids"][0, :length].tolist() == reference.token_ids
            assert row["token_mask"][0, :length].tolist() == reference.token_mask
            assert row["generation_logprobs"][0, :length].tolist() == reference.logprobs
            assert row["total_reward"].item() == 0.75
            assert row["sample_mask"].item() == 1
        elif incomplete or tag["is_execution_padding"]:
            assert row["sample_mask"].item() == 0
            assert row["token_mask"].count_nonzero().item() == 0
    assert_loss_once(
        forest,
        healthy_records if incomplete else records + healthy_records,
        records + healthy_records,
    )
    assert not any(partition == "staged" for partition, _ in forest.plane.rows)


def test_excluded_leaf_preserves_selected_ancestor(forest: ForestCapture) -> None:
    records, replies = forest.capture([None, 0, None])
    selected = select_captured_calls(records, ordinary([replies[0], replies[2]]))[0]
    result = publish(forest, records, selected)
    assert result.valid_row_count == 2
    assert_loss_once(forest, selected, records)
    assert not any(partition == "staged" for partition, _ in forest.plane.rows)


def test_per_chain_envelopes_preserve_membership(forest: ForestCapture) -> None:
    records, replies = forest.capture([None, None, 0, 1])
    result = {
        "responses": [
            ordinary([replies[0], replies[2]])["response"],
            ordinary([replies[1], replies[3]])["response"],
        ]
    }
    selected = select_captured_calls(records, result)[0]
    assert {r.model_call_id for r in selected} == {r.model_call_id for r in records}


def test_explicit_scored_parent_can_precede_worker(forest: ForestCapture) -> None:
    records, replies = forest.capture([None, None, 0, 1])
    result = ordinary(replies)
    result["terminal_response_id"] = records[2].response_id
    # Synthetic aggregate envelope ID leaves explicit terminal as authority.
    result["response"]["id"] = "aggregate"
    selected = select_captured_calls(records, result)[0]
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
            forest_training=True,
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
        receipts=[receipt(records, terminal=records[-1])],
        reward_checks=(None,),
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
    if bad_parent == "future":
        # Completion order is no longer proof against this edge. The capture
        # digests still reject the forged parent and mask the owner.
        assert publish(forest, records, records).valid_row_count == 0
    else:
        with pytest.raises(ValueError, match="selected predecessor|cycle"):
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
            receipts=[wire],
            reward_checks=(None,),
        )
        for plan in plans
    ]
    assert prepared[0] == prepared[1]
    assert prepared[0].valid_owner_count == 1


@pytest.mark.parametrize("parents", [[None, 0, 0], [None, 0, None], [None, None, 0, 1]])
@pytest.mark.parametrize("reverse", [False, True])
def test_completion_order_does_not_change_selected_forest(
    forest: ForestCapture, parents: list[int | None], reverse: bool
) -> None:
    records, replies = forest.capture(parents)
    manifest = list(reversed(records)) if reverse else records
    # Deliberately exercise a stronger permutation than real completion order:
    # storage order is not an ancestry or accepted-history contract.
    selected = select_captured_calls(manifest, ordinary(replies))[0]
    result = publish(forest, manifest, selected)
    assert result.valid_row_count == 2
    assert_loss_once(forest, selected, records)
    first = forest.plane.rows["canonical", result.meta.sample_ids[0]]
    assert first["input_ids"].tolist()[0][:2] == [10, 11]


def test_all_captured_membership_uses_existing_planner_boundary(
    forest: ForestCapture,
) -> None:
    records, _ = forest.capture([None, 0, 0, None, 3])
    # A provider supplies membership directly; ordinary-history matching is
    # not required by the shared planner. Terminal and membership are separate.
    selected = [records[i] for i in (1, 2, 0, 4, 3)]
    result = forest.finalizer.finalize_group(
        "group",
        ["attempt"],
        [receipt(list(reversed(records)), terminal=records[1])],
        [1.0],
        mask_sample=[False],
        fallback_weight_version=7,
        prompt_idx=99,
        canonical_sample_ids=["group_g0"],
        logical_selections=[selection(selected)],
    )
    assert result.valid_row_count == 3
    assert_loss_once(forest, selected, records)


@pytest.mark.parametrize("scored_index", [0, 1, 2, 3])
def test_forest_shaping_and_observations_are_per_owner(
    forest: ForestCapture, scored_index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A shared root with two leaves plus a new root (compaction). The scored
    # call can precede either endpoint, and must shape this owner's reward once.
    complete = gym.RolloutTokenCapture.complete_call

    def variable_completion(capture: Any, active: Any, **kwargs: Any) -> Any:
        generated = kwargs["generated_token_ids"]
        generated *= generated[0] - 1000
        kwargs["generated_logprobs"] = [-0.25] * len(generated)
        return complete(capture, active, **kwargs)

    monkeypatch.setattr(gym.RolloutTokenCapture, "complete_call", variable_completion)
    records, _ = forest.capture([None, 0, 0, None])
    forest.finalizer._effort_config = EffortLevelsConfig(
        low_weight=1, low_penalty=1, low_ub=10, low_string="budget"
    )
    result = forest.finalizer.finalize_group(
        "group",
        ["attempt"],
        [receipt(records, terminal=records[scored_index])],
        [1.0],
        mask_sample=[False],
        fallback_weight_version=7,
        prompt_idx=99,
        canonical_sample_ids=["group_g0"],
        logical_selections=[selection(records)],
        execution_row_multiple=4,
        reward_checks=[RewardChecks(False, False, True)],
        reward_log_contexts=[RewardLogContext("test-agent", None)],
    )
    assert result.valid_row_count == 3
    assert len(result.meta.sample_ids) == 4
    assert len(result.reward_observations) == 1
    observation = result.reward_observations[0]
    assert observation.sample_id == "group_g0"
    assert observation.rollout_id == "attempt"
    assert observation.reward == pytest.approx(2.0 - (scored_index + 1) / 10)
    assert observation.log_context.agent_name == "test-agent"
    assert result.metrics[f"finalize/effort/low/{scored_index + 1}"] == 1
    for key in result.meta.sample_ids[:3]:
        assert forest.plane.rows["canonical", key][
            "total_reward"
        ].item() == pytest.approx(2.0 - (scored_index + 1) / 10)
    assert (
        forest.plane.rows["canonical", result.meta.sample_ids[-1]][
            "total_reward"
        ].item()
        == 0
    )


def test_publication_identity_drift_preserves_staging(forest, monkeypatch):
    records, _ = forest.capture([None, 0, None])
    original = forest.finalizer._finalize_logical_rows

    def corrupt_ids(*args, **kwargs):
        prepared = original(*args, **kwargs)
        prepared.rows[0] = replace(prepared.rows[0], rollout_id="wrong")
        return prepared

    monkeypatch.setattr(forest.finalizer, "_finalize_logical_rows", corrupt_ids)
    before = set(forest.plane.rows)
    with pytest.raises(ValueError, match="publication IDs differ"):
        publish(forest, records, records)
    assert set(forest.plane.rows) == before


@pytest.mark.parametrize(
    "parents,omitted",
    [
        ([None, 0, 1, None, 3, 4], None),
        ([None, 0, 0, None, 3], None),
        ([None, 0, 0, None, 3], 1),
    ],
)
@pytest.mark.parametrize("masked", [False, True])
def test_capture_lengths_count_unique_generations_before_padding(
    forest: ForestCapture, parents: list[int | None], omitted: int | None, masked: bool
) -> None:
    records, _ = forest.capture(parents)
    selected = [record for index, record in enumerate(records) if index != omitted]
    result = forest.finalizer.finalize_group(
        "group",
        ["attempt"],
        [receipt(records, terminal=records[-1])],
        [1.0],
        mask_sample=[masked],
        fallback_weight_version=7,
        prompt_idx=0,
        canonical_sample_ids=["group_g0"],
        logical_selections=[selection(selected)],
        execution_row_multiple=4,
        reward_log_contexts=[RewardLogContext("agent", None)],
    )
    assert result.meta is not None and result.meta.size == 4
    metrics = result.meta.extra_info[ROLLOUT_METRICS][0]
    # This harness generates exactly one token per model call. Contexts and
    # duplicated ancestors in sibling rows must not change that population.
    assert metrics["gen_tokens_per_sample/histogram"] == [len(selected)]
    assert metrics["agent/gen_tokens_per_sample/mean"] == len(selected)
    assert metrics["max_gen_tokens_per_turn/max"] == 1
    assert metrics["total_tokens_per_sample/max"] == max(r.cum_len for r in selected)
    assert result.canonical_output_tokens == len(selected)
