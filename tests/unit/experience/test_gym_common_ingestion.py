# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Production common ingestion, served metadata persistence and Chat custody."""

import asyncio
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("nemo_gym", reason="requires the paired Gym checkout")

from nemo_gym.token_id_capture.completion import (
    completion_metadata,
    output_item_fingerprint,
    output_validation_item,
)
from nemo_gym.token_id_capture.lineage import FileLineageStore, InMemoryLineageStore
from nemo_gym.token_id_capture.staging.records import (
    CallRecord,
    CaptureLedgerCommit,
    OutputItemEvidence,
    RolloutManifest,
)

from nemo_rl.environments.call_selection import select_captured_calls
from nemo_rl.environments.nemo_gym import _normalize_gym_validity
from tests.unit.experience.test_agent_output_compatibility import captured  # noqa: F401
from tests.unit.experience.test_common_output_comparison import message, record
from tests.unit.experience.test_forest_dispatch import _env

pytestmark = pytest.mark.nemo_gym


def test_accepted_earlier_call_after_rejected_attempt_needs_no_parent_hint(
    captured: Any,  # noqa: F811
) -> None:  # noqa: F811
    captured.queue.extend(
        {"role": "assistant", "content": text, "reasoning": None}
        for text in ("accepted", "rejected", "continued")
    )
    prompt = [{"role": "user", "content": "task"}]
    first = captured.harness.post("owner", prompt).json()
    second = captured.harness.post("owner", prompt).json()
    third = captured.harness.post(
        "owner", prompt + first["output"] + [{"role": "user", "content": "next"}]
    ).json()
    assert "output" in third, third
    manifest = captured.harness.manifest("owner")
    assert manifest.records[2].parent_call_id == manifest.records[0].model_call_id
    result = asyncio.run(
        _env(captured.harness)._postprocess_receipt_mode(
            {"_ng_rollout_id": "owner"},
            {
                "response": third | {"output": first["output"] + third["output"]},
                "reward": 1.0,
            },
        )
    )
    assert result["logical_selection"].response_ids == (first["id"], third["id"])
    assert second["id"] not in result["logical_selection"].response_ids


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("response_status", None, "completion metadata"),
        ("last_output_item", None, "output validation evidence"),
        ("response_status", "incomplete", "finish reason"),
    ],
)
@pytest.mark.parametrize(
    "policy",
    [
        "returned_history",
        "all_captured",
        "nemo_rl.environments.call_selection:select_all_captured",
    ],
)
def test_missing_capture_evidence_is_rejected(
    captured: Any,  # noqa: F811
    field: str,
    value: Any,
    error: str,  # noqa: F811
    policy: str,
) -> None:  # noqa: F811
    captured.queue.append({"role": "assistant", "content": "answer", "reasoning": None})
    served = captured.harness.post(
        "owner", [{"role": "user", "content": "task"}]
    ).json()
    manifest = captured.harness.manifest("owner").model_dump(mode="json")
    manifest["records"][0][field] = value
    if field == "response_status" and value == "incomplete":
        manifest["records"][0]["finish_reason"] = None
    env = _env(captured.harness)
    env._selection_policy = policy
    env._control = AsyncMock(return_value=manifest)
    with pytest.raises(ValueError, match=error):
        asyncio.run(
            env._postprocess_receipt_mode(
                {"_ng_rollout_id": "owner"}, {"response": served, "reward": 1.0}
            )
        )


def call(index: int, items: list[dict], *, parent: int | None = None) -> CallRecord:
    return record(index, items, parent=parent).model_copy(
        update={
            "output_items": [
                OutputItemEvidence(
                    id=item.get("id"), fingerprint=output_item_fingerprint(item)
                )
                for item in items
            ],
            "response_status": "completed",
            "last_output_item": output_validation_item(items[-1]),
        }
    )


@pytest.mark.parametrize(
    "item_type",
    [
        "custom_tool_call",
        "computer_call",
        "local_shell_call",
        "apply_patch_call",
        "shell_call",
        "tool_search_call",
        "web_search_call",
        "file_search_call",
        "code_interpreter_call",
        "image_generation_call",
        "mcp_call",
        "mcp_approval_request",
        "mcp_list_tools",
    ],
)
@pytest.mark.parametrize("transitions", [False, True])
@pytest.mark.parametrize("references", [False, True])
def test_unsupported_generated_item_cannot_be_hidden_by_matching_calls(
    item_type: str, transitions: bool, references: bool
) -> None:
    final = message("final") | {"id": "final"}
    records = [call(1, [final])]
    output = [{"type": item_type, "id": "unsupported"}, final]
    result = {
        "response": {
            "id": "response-1",
            "output": [output] if transitions else output,
            "contains_transitions": transitions,
        }
    }
    if references:
        result["ng_trajectory"] = {
            "turns": [{"model_calls": [{"model_call_id": records[0].model_call_id}]}]
        }
    with pytest.raises(
        ValueError, match=f"Unsupported generated output item.*{item_type}"
    ):
        select_captured_calls(records, result)[0]


def test_refusal_and_tool_results_keep_supported_selection() -> None:
    refusal = message("unused") | {
        "id": "refusal",
        "content": [{"type": "refusal", "refusal": "I cannot do that."}],
    }
    tool_call = {
        "type": "function_call",
        "id": "function",
        "call_id": "tool",
        "name": "step",
        "arguments": "{}",
    }
    records = [call(1, [tool_call]), call(2, [refusal], parent=1)]
    result = {
        "response": {
            "id": "response-2",
            "output": [
                {"role": "user", "content": "task"},
                tool_call,
                {"type": "function_call_output", "call_id": "tool", "output": "ok"},
                refusal,
            ],
        }
    }
    assert select_captured_calls(records, result)[0] == records


def test_exact_ids_distinguish_equal_text_across_reset_and_ignore_old_contract() -> (
    None
):
    a, b, c = [
        message(text) | {"id": str(i)}
        for i, text in enumerate(("same", "same", "last"))
    ]
    records = [call(1, [a]), call(2, [b]), call(3, [c])]
    result = {
        "response": {"id": "response-3", "output": [b, c]},
        "context_compaction_result": {
            "logical_rollout_id": "wrong",
            "selected_actions": [],
        },
    }
    assert [r.response_id for r in select_captured_calls(records, result)[0]] == [
        "response-2",
        "response-3",
    ]
    result["response"]["output"][0] = b | {"content": message("edited")["content"]}
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        select_captured_calls(records, result)[0]


def test_reasoning_only_identity_and_whole_call_requirement() -> None:
    reasoning = {
        "id": "reason",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "think"}],
    }
    final = message("final") | {"id": "final"}
    records = [call(1, [reasoning]), call(2, [final])]
    assert (
        len(
            select_captured_calls(
                records, {"response": {"output": [reasoning, final]}}
            )[0]
        )
        == 2
    )
    assert (
        len(select_captured_calls(records, {"response": {"output": [final]}})[0]) == 1
    )
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        select_captured_calls(
            [call(3, [reasoning, final])], {"response": {"output": [final]}}
        )[0]


@pytest.mark.parametrize("accepted", [("response-2",), ("response-1", "response-2")])
def test_response_collection_native_ids_disambiguate_equal_outputs(
    accepted: tuple[str, ...],
) -> None:
    same, last = message("same"), message("last")
    records = [call(1, [same]), call(2, [same]), call(3, [last])]
    result = {
        "responses": [
            *({"id": identity, "output": [same]} for identity in accepted),
            {"id": "response-3", "output": [last]},
        ],
        "response": {"id": "aggregate", "output": [same, last]},
    }
    assert tuple(r.response_id for r in select_captured_calls(records, result)[0]) == (
        *accepted,
        "response-3",
    )


def test_response_collection_native_id_cannot_contradict_output() -> None:
    first, second = message("first"), message("second")
    records = [call(1, [first]), call(2, [second])]
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        select_captured_calls(
            records,
            {"responses": [{"id": "response-1", "output": [second]}]},
        )[0]


@pytest.mark.parametrize("collection", [False, True])
def test_native_envelope_accepts_scored_call_before_last_history_item(
    collection: bool,
) -> None:
    first, scored, worker = [
        message("same") | {"id": name} for name in ("first", "scored", "worker")
    ]
    records = [call(1, [first]), call(2, [worker]), call(3, [scored], parent=1)]
    envelope = {"id": "response-3", "output": [first, scored, worker]}
    result = {"responses": [envelope]} if collection else {"response": envelope}
    assert select_captured_calls(records, result)[0] == [
        records[0],
        records[2],
        records[1],
    ]


def test_native_envelope_cannot_borrow_identity_from_previous_envelope() -> None:
    first, second = message("first"), message("second")
    records = [call(1, [first]), call(2, [second])]
    result = {
        "responses": [
            {"id": "export", "output": [first]},
            {"id": "response-1", "output": [second]},
        ]
    }
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        select_captured_calls(records, result)[0]


def test_response_collection_native_id_identifies_cumulative_terminal() -> None:
    first, last = message("first"), message("last")
    records = [call(1, [first]), call(2, [last], parent=1)]
    selected = select_captured_calls(
        records,
        {"responses": [{"id": "response-2", "output": [first, last]}]},
    )[0]
    assert [r.response_id for r in selected] == ["response-1", "response-2"]


@pytest.mark.parametrize("contradict", [False, True])
def test_empty_synthetic_envelopes_do_not_erase_native_identity(
    contradict: bool,
) -> None:
    first, second = message("first"), message("second")
    records = [call(1, [first]), call(2, [second])]
    result = {
        "responses": [
            {"id": "empty-before", "output": []},
            {"id": "response-1", "output": [second if contradict else first]},
            {"id": "empty-after", "output": []},
        ]
    }
    if contradict:
        with pytest.raises(ValueError, match="Missing or ambiguous"):
            select_captured_calls(records, result)[0]
    else:
        assert [r.response_id for r in select_captured_calls(records, result)[0]] == [
            "response-1"
        ]


def test_empty_native_envelope_cannot_claim_a_captured_generation() -> None:
    first, last = message("first"), message("last")
    with pytest.raises(ValueError, match="No accepted captured generations"):
        select_captured_calls(
            [call(1, [first]), call(2, [last])],
            {
                "responses": [
                    {"id": "response-1", "output": []},
                    {"id": "response-2", "output": [last]},
                ]
            },
        )[0]


def test_existing_trajectory_references_exclude_noise_and_validate_both_ids() -> None:
    records = [call(i, [message(str(i))]) for i in range(1, 5)]
    result = {
        "response": {"id": "synthetic", "output": [message("1"), message("4")]},
        "ng_trajectory": {
            "turns": [
                {
                    "model_calls": [
                        {"response_id": "response-1", "model_call_id": "call-1"}
                    ]
                },
                {"model_calls": [{"response_id": "response-4"}]},
            ]
        },
    }
    assert [r.response_id for r in select_captured_calls(records, result)[0]] == [
        "response-1",
        "response-4",
    ]
    result["response"]["output"] = [message("copied prompt")]
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        select_captured_calls(records, result)[0]
    result["ng_trajectory"]["turns"][0]["model_calls"][0]["model_call_id"] = "call-2"
    with pytest.raises(ValueError, match="contradictory"):
        select_captured_calls(records, result)[0]


@pytest.mark.parametrize("with_references", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        "unchanged",
        "edited_text",
        "synthetic",
        "missing_reasoning",
        "missing_tool",
        "tool_arguments",
    ],
)
def test_whole_output_validation_is_shared(with_references: bool, change: str) -> None:
    items = [
        {
            "id": "reason",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "think"}],
        },
        message("answer") | {"id": "answer"},
        {
            "id": "tool-item",
            "type": "function_call",
            "call_id": "tool-call",
            "name": "check",
            "arguments": '{"x":1}',
        },
    ]
    records = [call(1, items)]
    output = deepcopy(items)
    if change == "edited_text":
        output[1]["content"][0]["text"] = "edited answer"
    elif change == "synthetic":
        output.append(message("harness recovery") | {"id": "synthetic"})
    elif change == "missing_reasoning":
        output.pop(0)
    elif change == "missing_tool":
        output.pop()
    elif change == "tool_arguments":
        output[-1]["arguments"] = '{"x":2}'
    result = {"response": {"id": "export", "output": output}}
    if with_references:
        result["ng_trajectory"] = {
            "turns": [{"model_calls": [{"response_id": "response-1"}]}]
        }
    if change == "unchanged":
        assert select_captured_calls(records, result)[0] == records
    else:
        with pytest.raises(ValueError, match="Missing or ambiguous") as failure:
            select_captured_calls(records, result)[0]
        assert "authored_output_index=" in str(failure.value)
        assert "item_id=" in str(failure.value)
        assert "no whole captured call matches" in str(failure.value)


def test_references_disambiguate_equal_content_without_bypassing_validation() -> None:
    item = message("same")
    records = [call(1, [item]), call(2, [item])]
    result = {"response": {"id": "export", "output": [item]}}
    with pytest.raises(ValueError, match="multiple captured calls match"):
        select_captured_calls(records, result)[0]
    result["ng_trajectory"] = {
        "turns": [{"model_calls": [{"response_id": "response-2"}]}]
    }
    assert select_captured_calls(records, result)[0] == [records[1]]


@pytest.mark.parametrize("output", [[], [message("first")]])
def test_output_must_cover_all_explicitly_referenced_calls(output: list[dict]) -> None:
    records = [call(1, [message("first")]), call(2, [message("second")])]
    result = {
        "response": {"id": "export", "output": output},
        "ng_trajectory": {
            "turns": [
                {"model_calls": [{"response_id": r.response_id} for r in records]}
            ]
        },
    }
    with pytest.raises(ValueError, match="missing referenced generations") as failure:
        select_captured_calls(records, result)[0]
    assert "call-2" in str(failure.value)


def test_references_cannot_select_different_calls_from_matching_output() -> None:
    first, second = message("first"), message("second")
    records = [call(1, [first]), call(2, [second])]
    result = {
        "response": {"id": "export", "output": [second]},
        "ng_trajectory": {"turns": [{"model_calls": [{"model_call_id": "call-1"}]}]},
    }
    with pytest.raises(ValueError, match="expected_call='call-1'"):
        select_captured_calls(records, result)[0]


def test_reference_only_output_keeps_existing_identity_checks() -> None:
    records = [call(1, [message("first")])]
    result = {
        "ng_trajectory": {"turns": [{"model_calls": [{"model_call_id": "call-1"}]}]}
    }
    assert select_captured_calls(records, result)[0] == records
    result["ng_trajectory"]["turns"][0]["model_calls"][0]["model_call_id"] = "unknown"
    with pytest.raises(ValueError, match="outside this capture") as failure:
        select_captured_calls(records, result)[0]
    assert "model_call_id='unknown'" in str(failure.value)
    assert "turn=0, reference=0" in str(failure.value)


@pytest.mark.parametrize("with_references", [False, True])
def test_ingestion_reports_rollout_and_offending_item_before_finalization(
    captured: Any,  # noqa: F811
    with_references: bool,
) -> None:
    captured.queue.append(
        {"role": "assistant", "content": "original", "reasoning": None}
    )
    served = captured.harness.post(
        "owner", [{"role": "user", "content": "task"}]
    ).json()
    result = {"response": deepcopy(served), "reward": 1.0}
    result["response"]["output"][0]["content"][0]["text"] = "harness edit"
    if with_references:
        result["ng_trajectory"] = {
            "turns": [{"model_calls": [{"response_id": served["id"]}]}]
        }
    with pytest.raises(ValueError, match="Rollout 'owner'") as failure:
        asyncio.run(
            _env(captured.harness)._postprocess_receipt_mode(
                {"_ng_rollout_id": "owner"}, result
            )
        )
    detail = str(failure.value)
    assert "authored_output_index=0" in detail
    assert served["output"][0]["id"] in detail
    assert captured.harness.manifest("owner").records[0].model_call_id in detail
    assert not any(partition == "canonical" for partition, _ in captured.plane.rows)


@pytest.mark.parametrize(
    "top,nested", [(False, False), (True, False), (False, True), (True, True)]
)
def test_common_validity_preserves_zero_reward_and_combines_masks(
    top: bool, nested: bool
) -> None:
    original = {
        "reward": 0.0,
        "mask_sample": top,
        "instance_config": {"mask_sample": nested},
        "failure_kind": "unsolved",
    }
    normalized = _normalize_gym_validity(original)
    assert normalized["instance_config"]["mask_sample"] == (top or nested)
    assert normalized["reward"] == 0.0 and normalized["failure_kind"] == "unsolved"
    assert original["instance_config"]["mask_sample"] == nested


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({}, ("unknown", None)),
        ({"status": "cancelled"}, ("failed", None)),
        ({"status": "queued"}, ("unknown", None)),
        ({"choices": [{"finish_reason": "abort"}]}, ("failed", "abort")),
        ({"status": "completed"}, ("completed", None)),
        ({"choices": [{"finish_reason": "stop"}]}, ("completed", "stop")),
        ({"choices": [{"finish_reason": "length"}]}, ("incomplete", "length")),
        ({"status": "failed"}, ("failed", None)),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
            },
            ("incomplete", "content_filter"),
        ),
    ],
)
def test_completion_evidence_does_not_fabricate_success(
    payload: dict, expected: tuple
) -> None:
    assert completion_metadata(payload) == expected


@pytest.mark.parametrize("file", [False, True])
def test_capture_evidence_roundtrip_restart_and_conflict(
    tmp_path: Path, file: bool
) -> None:
    async def run() -> None:
        store = FileLineageStore(tmp_path) if file else InMemoryLineageStore()
        item = message("original") | {"id": "native"}
        saved = call(1, [item])
        commit = CaptureLedgerCommit(
            rollout_id="owner", record=saved, request_items=[], response_items=[item]
        )
        await store.record(commit)
        if file:
            await store.close()
            store = FileLineageStore(tmp_path)
        try:
            assert RolloutManifest.model_validate(
                await store.manifest("owner")
            ).records == [saved]
            for field, value in (
                ("response_status", "failed"),
                ("finish_reason", "length"),
                ("output_items", []),
                ("last_output_item", {}),
            ):
                changed = commit.model_copy(
                    update={"record": saved.model_copy(update={field: value})}
                )
                with pytest.raises(ValueError, match="conflicting"):
                    await store.record(changed)
        finally:
            await store.close()

    asyncio.run(run())


def test_identical_retry_is_resolved_by_the_terminal_parent_chain() -> None:
    a, b = message("A"), message("B")
    records = [call(1, [a]), call(2, [a]), call(3, [b], parent=2)]
    selected, terminal = select_captured_calls(
        records, {"response": {"id": "response-3", "output": [a, b]}}
    )
    assert [r.response_id for r in selected] == ["response-2", "response-3"]
    assert terminal == "response-3"


@pytest.mark.parametrize(
    "mode,error",
    [
        ("valid", None),
        ("duplicate", "unique captured calls"),
        ("empty", "unique captured calls"),
        ("foreign", "unchanged records"),
        ("edited", "unchanged records"),
        ("missing_terminal", "identify its scored terminal"),
        ("unselected_terminal", "identify its scored terminal"),
    ],
)
def test_external_selector_uses_shared_ingestion(
    captured: Any,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    error: str | None,
) -> None:
    (tmp_path / "integration_selection.py").write_text(
        "def choose(records, result):\n"
        "    mode = result['selector_mode']\n"
        "    chosen = records[0]\n"
        "    if mode == 'empty': return [], None\n"
        "    if mode == 'duplicate': return [chosen, chosen], chosen.response_id\n"
        "    if mode == 'foreign': chosen.model_call_id = 'outside-capture'\n"
        "    if mode == 'edited': chosen.cum_len += 1\n"
        "    if mode == 'missing_terminal': return [chosen], None\n"
        "    if mode == 'unselected_terminal': return [chosen], records[1].response_id\n"
        "    return [chosen], chosen.response_id\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    captured.queue.extend(
        {"role": "assistant", "content": text, "reasoning": None}
        for text in ("accepted", "rejected")
    )
    first, _ = [
        captured.harness.post("owner", [{"role": "user", "content": "task"}]).json()
        for _ in range(2)
    ]
    env = _env(captured.harness)
    env._selection_policy = "integration_selection:choose"
    run = env._postprocess_receipt_mode(
        {"_ng_rollout_id": "owner"},
        {"response": first, "reward": 1.0, "selector_mode": mode},
    )
    if error:
        with pytest.raises(ValueError, match=error):
            asyncio.run(run)
    else:
        result = asyncio.run(run)
        assert result["logical_selection"].response_ids == (first["id"],)
        assert not result["receipt"]["capture_poisoned"]
