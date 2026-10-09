# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from nemo_rl.environments.gym_sessions import (
    session_trace_metadata,
    session_training_overrides,
)


def traces():
    return [
        {"session_id": "P", "session_role": "parent", "parent_session_id": None, "spawn_call_id": None, "response_ids": ["r0", "r1", "r4"]},
        {"session_id": "A", "session_role": "child", "parent_session_id": "P", "spawn_call_id": "call_a", "response_ids": ["r2"]},
        {"session_id": "B", "session_role": "child", "parent_session_id": "P", "spawn_call_id": "call_b", "response_ids": ["r3"]},
    ]


def test_subagent_traces_bind_every_selected_response_to_one_session():
    tags, sessions, roles = session_trace_metadata(
        traces(), selected_response_ids=("r0", "r1", "r2", "r3", "r4"), training_mode="subagent"
    )
    assert [t["session_id"] for t in tags] == ["P", "A", "B"]
    assert [t["trace_index"] for t in tags] == [0, 1, 2]
    assert tags[0]["request_ids"] == ["r0", "r1", "r4"] and tags[0]["root_session_id"] == "P"
    assert tags[1]["spawn_call_id"] == "call_a" and tags[1]["parent_session_id"] == "P"
    assert sessions == ("P", "P", "A", "B", "P")
    assert roles == ("parent", "parent", "child", "child", "parent")


def test_child_without_selected_responses_is_dropped_but_parent_must_remain():
    tags, _, _ = session_trace_metadata(traces(), selected_response_ids=("r0", "r1", "r2", "r4"), training_mode="subagent")
    assert [t["session_id"] for t in tags] == ["P", "A"]
    with pytest.raises(ValueError, match="parent session owns no selected"):
        session_trace_metadata(traces(), selected_response_ids=("r2",), training_mode="subagent")


def test_unowned_or_doubly_owned_responses_are_rejected():
    with pytest.raises(ValueError, match="without a session"):
        session_trace_metadata(traces(), selected_response_ids=("r0", "r9"), training_mode="subagent")
    doubled = traces()
    doubled[1]["response_ids"].append("r0")
    with pytest.raises(ValueError, match="owned by two sessions"):
        session_trace_metadata(doubled, selected_response_ids=("r0",), training_mode="subagent")


def test_duplicate_spawn_ids_fail_the_vendored_validator():
    dup = traces()
    dup[2]["spawn_call_id"] = "call_a"
    with pytest.raises(ValueError, match="unique spawn_call_id"):
        session_trace_metadata(dup, selected_response_ids=("r0", "r2", "r3"), training_mode="subagent")


def test_main_session_modes_reject_children_and_compaction():
    with pytest.raises(ValueError, match="admits only the main session"):
        session_trace_metadata(traces(), selected_response_ids=("r0", "r2"), training_mode="plain")
    only_parent = traces()[:1]
    tags, _, _ = session_trace_metadata(only_parent, selected_response_ids=("r0", "r1"), training_mode="plain")
    assert len(tags) == 1 and tags[0]["session_role"] == "parent"
    with pytest.raises(ValueError, match="forbids OpenCode compaction"):
        session_trace_metadata(only_parent, selected_response_ids=("r0",), training_mode="subagent", compaction_summaries=2)
    tags, _, _ = session_trace_metadata(only_parent, selected_response_ids=("r0",), training_mode="compaction", compaction_summaries=2)
    assert tags[0]["training_mode"] == "compaction"


def test_missing_traces_are_an_error_when_a_mode_is_configured():
    with pytest.raises(ValueError, match="requires session traces"):
        session_trace_metadata(None, selected_response_ids=("r0",), training_mode="subagent")


def test_training_overrides_match_the_harness_contract():
    assert session_training_overrides(None) == {}
    assert session_training_overrides("plain") == {"opencode_training_mode": "plain", "opencode_subagents_enabled": False}
    sub = session_training_overrides("subagent")
    assert sub["opencode_subagents_enabled"] is True and sub["opencode_subagent_max_depth"] == 1
    assert sub["opencode_compaction_auto"] is False and sub["opencode_subagent_max_active"] is None
