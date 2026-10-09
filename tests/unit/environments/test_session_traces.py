# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
import hashlib
import json

import pytest

from nemo_rl.environments.session_traces import (
    bind_session_generation_spans,
    session_training_semantics,
    validate_session_trace_metadata,
    validate_session_training_config,
    validate_tagged_session_rollout,
)


def metadata() -> list[dict]:
    return [
        {
            "training_mode": "subagent",
            "session_id": session,
            "root_session_id": "root",
            "parent_session_id": "" if session == "root" else "root",
            "session_role": "parent" if session == "root" else "child",
            "spawn_call_id": "" if session == "root" else f"task-{session}",
            "segment_index": 0,
            "segment_boundary_reason": "",
            "request_ids": [f"request-{session}"],
        }
        for session in ("root", "child-one", "child-two")
    ]


def config() -> dict:
    return {
        "nemo_gym": {
            "training_mode": "subagent",
            "results_dir": "/first-allocation",
            "swe": {
                "responses_api_agents": {
                    "swe_agents": {
                        "opencode_training_mode": "subagent",
                        "opencode_subagents_enabled": True,
                        "opencode_subagent_max_depth": 1,
                        "opencode_subagent_max_active": 1,
                    }
                }
            },
        }
    }


def test_sequential_children_and_independent_request_ownership() -> None:
    traces = metadata()
    validate_session_trace_metadata(traces, training_mode="subagent")
    validate_tagged_session_rollout(traces)
    traces[1]["request_ids"] = '["request-child-one"]'
    validate_tagged_session_rollout(traces)


@pytest.mark.parametrize(
    "change",
    [
        {"training_mode": "compaction"},
        {"session_id": "root"},
        {"root_session_id": "other"},
        {"parent_session_id": "child-two"},
        {"parent_session_id": "unknown"},
        {"spawn_call_id": ""},
        {"spawn_call_id": "task-child-two"},
        {"segment_index": 1},
        {"segment_index": False},
        {"segment_boundary_reason": "compaction"},
        {"is_compaction_summary": True},
        {"session_role": "parent"},
        {"request_ids": ["request-root"]},
        {"request_ids": ["duplicate", "duplicate"]},
        {"request_ids": []},
    ],
)
def test_invalid_child_ownership_and_compaction_are_rejected(change: dict) -> None:
    traces = metadata()
    traces[1].update(change)
    with pytest.raises(ValueError):
        validate_tagged_session_rollout(traces)


def test_missing_root_and_mixed_legacy_tags_are_rejected() -> None:
    with pytest.raises(ValueError, match="common parent/root"):
        validate_tagged_session_rollout(metadata()[1:])
    with pytest.raises(ValueError, match="mixed"):
        validate_tagged_session_rollout([metadata()[0], {}])
    validate_tagged_session_rollout([{"segment_index": 0}])


def test_explicit_plain_mode_cannot_receive_children() -> None:
    traces = metadata()
    for trace in traces:
        trace["training_mode"] = "plain"
    with pytest.raises(ValueError, match="only valid"):
        validate_tagged_session_rollout(traces)
    validate_tagged_session_rollout(traces[:1])


def test_mode_and_limits_are_explicit_and_operational_paths_are_excluded() -> None:
    original = config()
    resumed = deepcopy(original)
    resumed["nemo_gym"]["results_dir"] = "/resumed-allocation"
    assert validate_session_training_config(original) == "subagent"
    assert session_training_semantics(original) == session_training_semantics(resumed)


@pytest.mark.parametrize(
    "change",
    [
        {"opencode_training_mode": None},
        {"opencode_training_mode": "compaction"},
        {"opencode_subagents_enabled": False},
        {"opencode_subagent_max_depth": 2},
        {"opencode_subagent_max_active": 2},
        {"opencode_subagent_max_active": True},
        {"opencode_compaction_auto": True},
        {"opencode_pruning_enabled": True},
    ],
)
def test_incompatible_runtime_mode_and_limits_fail_before_launch(change: dict) -> None:
    current = config()
    current["nemo_gym"]["swe"]["responses_api_agents"]["swe_agents"].update(change)
    with pytest.raises(ValueError):
        validate_session_training_config(current)


def test_subagents_cannot_be_enabled_through_legacy_mode() -> None:
    current = config()
    current["nemo_gym"].pop("training_mode")
    with pytest.raises(ValueError):
        validate_session_training_config(current)
    assert validate_session_training_config({}) is None
    assert validate_session_training_config({"nemo_gym": {}}) is None


def test_generation_manifest_binds_exact_echo_and_keeps_zero_token_calls() -> None:
    call = {
        "request_id": "generated",
        "prompt_token_count": 4,
        "generation_token_count": 2,
        "generation_token_ids_sha256": hashlib.sha256(b"[20,21]").hexdigest(),
    }
    empty = {
        "request_id": "empty",
        "prompt_token_count": 2,
        "generation_token_count": 0,
        "generation_token_ids_sha256": hashlib.sha256(b"[]").hexdigest(),
    }
    tag = {
        "request_ids": ["empty", "generated"],
        "generation_calls": json.dumps([empty, call]),
    }
    output = {"prompt_token_ids": [1, 2, 3, 4], "generation_token_ids": [20, 21]}
    spans = bind_session_generation_spans(tag, [output])
    assert spans == [
        {
            "output_index": 0,
            "request_id": "generated",
            "target_start": 4,
            "target_end": 6,
            "generation_token_ids_sha256": call["generation_token_ids_sha256"],
        }
    ]
    for change in (
        {"generation_token_ids": [20, 22]},
        {"prompt_token_ids": [1, 2, 3]},
        {"generation_token_ids": [20]},
    ):
        with pytest.raises(ValueError, match="differs from"):
            bind_session_generation_spans(tag, [{**output, **change}])
    with pytest.raises(ValueError, match="missing or duplicates"):
        bind_session_generation_spans(tag, [output, output])
    with pytest.raises(ValueError, match="request ownership"):
        bind_session_generation_spans(
            {**tag, "request_ids": ["generated", "empty"]}, [output]
        )
