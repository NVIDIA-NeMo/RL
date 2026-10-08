# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Production common ingestion, served metadata persistence and Chat custody."""

import asyncio
import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("nemo_gym", reason="requires the paired Gym checkout")

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.token_id_capture.completion import (
    completion_metadata,
    output_item_fingerprint,
)
from nemo_gym.token_id_capture.lineage import FileLineageStore, InMemoryLineageStore
from nemo_gym.token_id_capture.staging.records import (
    CallRecord,
    CaptureLedgerCommit,
    OutputItemEvidence,
    RolloutManifest,
)

from nemo_rl.environments.gym_selection import select_captured_calls
from nemo_rl.environments.nemo_gym import _normalize_gym_validity
from nemo_rl.experience.rollout_reassembler import RolloutReassembler
from tests.unit.experience.test_agent_output_compatibility import captured  # noqa: F401
from tests.unit.experience.test_cc_dispatch import _env
from tests.unit.experience.test_common_output_comparison import message, record
from tests.unit.experience.test_external_exporter_compatibility import (
    ForkInstanceConfig,
    fork_exporter,  # noqa: F401
    response,
)

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
def test_missing_capture_evidence_is_rejected(
    captured: Any,  # noqa: F811
    field: str,
    value: Any,
    error: str,  # noqa: F811
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
            "last_output_item": items[-1],
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
        select_captured_calls(records, result)


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
    assert select_captured_calls(records, result) == records


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
    assert [r.response_id for r in select_captured_calls(records, result)] == [
        "response-2",
        "response-3",
    ]
    result["response"]["output"][0] = b | {"content": message("edited")["content"]}
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        select_captured_calls(records, result)


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
            select_captured_calls(records, {"response": {"output": [reasoning, final]}})
        )
        == 2
    )
    assert len(select_captured_calls(records, {"response": {"output": [final]}})) == 1
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        select_captured_calls(
            [call(3, [reasoning, final])], {"response": {"output": [final]}}
        )


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
    assert tuple(r.response_id for r in select_captured_calls(records, result)) == (
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
        )


def test_response_collection_native_id_identifies_cumulative_terminal() -> None:
    first, last = message("first"), message("last")
    records = [call(1, [first]), call(2, [last], parent=1)]
    selected = select_captured_calls(
        records,
        {"responses": [{"id": "response-2", "output": [first, last]}]},
    )
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
            select_captured_calls(records, result)
    else:
        assert [r.response_id for r in select_captured_calls(records, result)] == [
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
        )


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
    assert [r.response_id for r in select_captured_calls(records, result)] == [
        "response-1",
        "response-4",
    ]
    result["response"]["output"] = [message("copied prompt")]
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        select_captured_calls(records, result)
    result["ng_trajectory"]["turns"][0]["model_calls"][0]["model_call_id"] = "call-2"
    with pytest.raises(ValueError, match="contradictory"):
        select_captured_calls(records, result)


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
        assert select_captured_calls(records, result) == records
    else:
        with pytest.raises(ValueError, match="Missing or ambiguous") as failure:
            select_captured_calls(records, result)
        assert "authored_output_index=" in str(failure.value)
        assert "item_id=" in str(failure.value)
        assert "no whole captured call matches" in str(failure.value)


def test_references_disambiguate_equal_content_without_bypassing_validation() -> None:
    item = message("same")
    records = [call(1, [item]), call(2, [item])]
    result = {"response": {"id": "export", "output": [item]}}
    with pytest.raises(ValueError, match="multiple captured calls match"):
        select_captured_calls(records, result)
    result["ng_trajectory"] = {
        "turns": [{"model_calls": [{"response_id": "response-2"}]}]
    }
    assert select_captured_calls(records, result) == [records[1]]


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
        select_captured_calls(records, result)
    assert "call-2" in str(failure.value)


def test_references_cannot_select_different_calls_from_matching_output() -> None:
    first, second = message("first"), message("second")
    records = [call(1, [first]), call(2, [second])]
    result = {
        "response": {"id": "export", "output": [second]},
        "ng_trajectory": {"turns": [{"model_calls": [{"model_call_id": "call-1"}]}]},
    }
    with pytest.raises(ValueError, match="expected_call='call-1'"):
        select_captured_calls(records, result)


def test_reference_only_output_keeps_existing_identity_checks() -> None:
    records = [call(1, [message("first")])]
    result = {
        "ng_trajectory": {"turns": [{"model_calls": [{"model_call_id": "call-1"}]}]}
    }
    assert select_captured_calls(records, result) == records
    result["ng_trajectory"]["turns"][0]["model_calls"][0]["model_call_id"] = "unknown"
    with pytest.raises(ValueError, match="outside this capture") as failure:
        select_captured_calls(records, result)
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
        ({}, (None, None)),
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


@pytest.mark.parametrize("compact", [False, True])
def test_actual_custom_exporter_chat_capture_to_training_rows(
    captured: Any,  # noqa: F811
    fork_exporter: Any,  # noqa: F811
    tmp_path: Path,
    compact: bool,  # noqa: F811
) -> None:  # noqa: F811
    calls = []
    prompt = [{"role": "user", "content": "task"}]
    texts = ["first", "second", "summary", "last"] if compact else ["first", "second"]
    captured.queue.extend(
        {"role": "assistant", "content": text, "reasoning": None} for text in texts
    )
    for index, text in enumerate(texts):
        segment = max(0, index - 1)
        if index == 2:
            prompt = [{"role": "system", "content": "summarize"}, *prompt]
        elif index == 3:
            prompt = [{"role": "user", "content": "summary"}]
        served = captured.harness.client.post(
            "/ng-rollout/owner/training-token-capture/v1/chat/completions",
            json={"model": "test-model", "messages": prompt},
        )
        assert served.status_code == 200, served.text
        raw = served.json()
        calls.append(
            {
                "turn": index + 1,
                "session_id": "main",
                "segment_index": segment,
                "messages": deepcopy(prompt),
                "response": raw,
            }
        )
        prompt = [
            *prompt,
            {
                key: value
                for key, value in raw["choices"][0]["message"].items()
                if value is not None
            },
            {"role": "user", "content": "continue"},
        ]
    manifest = captured.harness.manifest("owner")
    assert not manifest.failures
    assert sum(r.parent_call_id is None for r in manifest.records) == (
        3 if compact else 1
    )
    setup = tmp_path / "setup"
    dumps = setup / "opencode/eval/run/llm_completions/main"
    dumps.mkdir(parents=True)
    for data in calls:
        path = dumps / f"turn-{data['turn']:04d}.json"
        path.write_text(json.dumps(data))
        os.utime(path, (data["turn"], data["turn"]))
    collector = fork_exporter.source.RunOpenHandsAgent()
    collector.config = SimpleNamespace(
        problem_info={"instance_id": "task"},
        eval_dir_in_openhands="eval",
        agent_framework="opencode",
        openhands_config_file_path=tmp_path / "config",
        opencode_setup_dir=setup,
        trajectories_root=tmp_path / "trajectories/task",
    )
    collector._openhands_dir_copy_from_host(None)
    wrapper = fork_exporter.wrapper
    wrapper.responses = AsyncMock(
        return_value=response(
            len(calls),
            "last",
            metadata={
                "input": json.dumps(calls[-1]["messages"]),
                "metrics": '{"resolved": true}',
                "instance_config": ForkInstanceConfig(
                    persistent_dir=tmp_path, instance_id="task"
                ).model_dump_json(),
            },
        )
    )
    exported = asyncio.run(
        wrapper.run(BaseRunRequest(responses_create_params={"input": "task"}))
    ).model_dump(mode="json")
    result = asyncio.run(
        _env(captured.harness)._postprocess_receipt_mode(
            {"_ng_rollout_id": "owner"}, exported
        )
    )
    selection = result["logical_selection"]
    assert selection.response_ids == tuple(
        f"chatcmpl-{i + 1}" for i in range(len(calls))
    )
    finalizer = RolloutReassembler(
        captured.plane,
        partition_id="canonical",
        staging_partition="staged",
        pad_token_id=0,
        max_seq_len=1024,
    )
    finalized = finalizer.finalize_group(
        "group",
        ["owner"],
        [result["receipt"]],
        [1.0],
        mask_sample=[False],
        fallback_weight_version=7,
        prompt_idx=0,
        canonical_sample_ids=["group_g0"],
        logical_selections=[selection],
    )
    assert finalized.valid_row_count == (3 if compact else 1)
    data = captured.plane.get_samples(
        sample_ids=finalized.meta.sample_ids,
        partition_id="canonical",
        select_fields=finalized.meta.fields,
    )
    tokens = [
        token
        for ids, mask in zip(
            data["input_ids"].unbind(), data["token_mask"].unbind(), strict=True
        )
        for token in ids[mask.bool()].tolist()
    ]
    assert tokens == list(range(1001, 1001 + len(calls)))
    assert not any(partition == "staged" for partition, _ in captured.plane.rows)
