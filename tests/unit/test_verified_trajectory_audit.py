# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import json
from copy import deepcopy

import pytest

from examples.swe_bench.verified_trajectory_audit import (
    SWE_BENCH_VERIFIED,
    AuditError,
    build_no_timeout_manifest,
    build_subset_manifest,
    canonical_json_sha256,
    canonical_trajectory_sha256,
    compare_arms,
    overlay_result_rows,
    read_result_jsonl,
    validate_manifest,
)


def manifest_row(instance_id: str) -> dict:
    return {
        "instance_id": instance_id,
        "base_commit": f"commit-{instance_id}",
        "agent_ref": {"type": "responses_api_agents", "name": "swe_agents_val"},
        "responses_create_params": {
            "metadata": {
                "instance_id": instance_id,
                "base_commit": f"commit-{instance_id}",
                "dataset_name": SWE_BENCH_VERIFIED,
            }
        },
    }


def result_row(
    instance_id: str,
    reward: int,
    *,
    trajectory_text: str | None = None,
    model_patch: str | None = None,
) -> dict:
    trajectory_text = trajectory_text or f"trajectory-{instance_id}"
    model_patch = model_patch or f"patch-{instance_id}"
    row = {
        "response": {
            "output": [
                {
                    "type": "message",
                    "id": f"response-{instance_id}",
                    "role": "assistant",
                    "status": "completed",
                    "content": trajectory_text,
                }
            ]
        },
        "instance_config": {},
        "responses_create_params": {
            "model": "test-model",
            "temperature": 0.0,
            "top_p": 1.0,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": f"problem-{instance_id}",
                }
            ],
            "metadata": {
                "instance_id": instance_id,
                "base_commit": f"commit-{instance_id}",
                "dataset_name": SWE_BENCH_VERIFIED,
            },
        },
        "reward": reward,
        "resolved": reward == 1,
        "model_patch": model_patch,
        "total_model_call_time": 2.0,
    }
    row["trajectory_sha256"] = canonical_trajectory_sha256(row)
    row["model_patch_available"] = True
    row["model_patch_nonempty"] = bool(model_patch)
    row["model_patch_sha256"] = canonical_json_sha256(model_patch)
    return row


def test_validate_manifest_rejects_duplicates() -> None:
    row = manifest_row("owner__repo-1")
    with pytest.raises(AuditError, match="duplicate instance_id"):
        validate_manifest([row, row], expected_count=2)


def test_compare_arms_uses_strict_manifest_denominator() -> None:
    manifest = [manifest_row("owner__repo-1"), manifest_row("owner__repo-2")]
    report, outcomes = compare_arms(
        manifest,
        [result_row("owner__repo-1", 1)],
        [result_row("owner__repo-2", 1)],
        expected_count=2,
    )

    assert report["complete"] is False
    assert report["streaming_off"]["strict_accuracy"] == 0.5
    assert report["streaming_on"]["strict_accuracy"] == 0.5
    assert report["paired"]["streaming_off_only"] == 1
    assert report["paired"]["streaming_on_only"] == 1
    assert len(outcomes) == 2


def test_compare_arms_rejects_sampling() -> None:
    manifest = [manifest_row("owner__repo-1")]
    sampled = result_row("owner__repo-1", 0)
    sampled["responses_create_params"]["temperature"] = 0.7

    with pytest.raises(AuditError, match="temperature=0.7"):
        compare_arms(manifest, [sampled], [], expected_count=1)


def test_read_result_jsonl_projects_large_trajectory_fields(tmp_path) -> None:
    row = result_row("owner__repo-1", 0)
    row["response"]["output"] = ["Temporary failure in name resolution"]
    row["num_tool_calls"] = 0
    path = tmp_path / "trajectories.jsonl"
    path.write_text(json.dumps(row) + "\n")

    projected_params = {
        key: value
        for key, value in row["responses_create_params"].items()
        if key != "input"
    }
    assert read_result_jsonl(path) == [
        {
            "responses_create_params": projected_params,
            "reward": 0,
            "resolved": False,
            "infrastructure_error": True,
            "response_error": False,
            "trajectory_sha256": canonical_trajectory_sha256(row),
            "model_patch_available": True,
            "model_patch_nonempty": True,
            "model_patch_sha256": canonical_json_sha256(row["model_patch"]),
            "response": None,
            "instance_config": None,
            **{
                field: row.get(field)
                for field in (
                    "total_model_call_time",
                    "openhands_run_time",
                    "total_command_exec_time",
                    "final_eval_time",
                    "total_prompt_tokens",
                    "total_completion_tokens",
                    "num_turns",
                    "num_tool_calls",
                    "streaming_tool_call_eligible_actions",
                    "streaming_tool_call_sessions_started",
                    "streaming_tool_call_prefill_requests",
                    "streaming_tool_call_prefill_tokens",
                    "streaming_tool_call_overhead_seconds",
                )
            },
        }
    ]


def test_read_result_jsonl_counts_function_calls_when_metric_is_missing(
    tmp_path,
) -> None:
    row = result_row("owner__repo-1", 0)
    row["response"]["output"] = [
        {"type": "function_call", "call_id": "call-1"},
        {"type": "function_call_output", "call_id": "call-1"},
        {"type": "function_call", "call_id": "call-2"},
    ]
    path = tmp_path / "trajectories.jsonl"
    path.write_text(json.dumps(row) + "\n")

    result = read_result_jsonl(path)

    assert result[0]["num_tool_calls"] == 2


def test_overlay_result_rows_replaces_retried_instance(tmp_path) -> None:
    retry = result_row("owner__repo-1", 1)
    retry_path = tmp_path / "retry.jsonl"
    retry_path.write_text(json.dumps(retry) + "\n")

    overlaid = overlay_result_rows([result_row("owner__repo-1", 0)], [retry_path])

    assert len(overlaid) == 1
    assert overlaid[0]["reward"] == 1


def test_build_no_timeout_manifest_uses_union_and_preserves_order(tmp_path) -> None:
    manifest = [
        manifest_row("owner__repo-2"),
        manifest_row("owner__repo-1"),
        manifest_row("owner__repo-3"),
    ]
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in manifest))

    trajectory_paths = []
    for trajectory_number, timeout_id in enumerate(
        ("owner__repo-1", "owner__repo-2"), start=1
    ):
        rows = []
        for row in manifest:
            result = result_row(row["instance_id"], 0)
            result["openhands_run_time"] = (
                1800.0 if row["instance_id"] == timeout_id else 100.0
            )
            rows.append(result)
        trajectory_path = tmp_path / f"trajectory-{trajectory_number}.jsonl"
        trajectory_path.write_text(
            "".join(json.dumps(row) + "\n" for row in reversed(rows))
        )
        trajectory_paths.append(trajectory_path)

    output = tmp_path / "no-timeout.jsonl"
    metadata = build_no_timeout_manifest(
        manifest_path,
        trajectory_paths,
        output,
        expected_count=3,
        expected_output_count=1,
        timeout_seconds=1800.0,
    )

    output_rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["instance_id"] for row in output_rows] == ["owner__repo-3"]
    assert metadata["excluded_instance_ids"] == [
        "owner__repo-1",
        "owner__repo-2",
    ]
    assert [trajectory["timeout_count"] for trajectory in metadata["trajectories"]] == [
        1,
        1,
    ]


def test_build_subset_manifest_is_hash_ranked_and_order_independent(tmp_path) -> None:
    rows = [
        manifest_row("owner__repo-3"),
        manifest_row("owner__repo-1"),
        manifest_row("owner__repo-2"),
    ]
    first_manifest = tmp_path / "first.jsonl"
    second_manifest = tmp_path / "second.jsonl"
    first_manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    second_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(rows))
    )

    first_output = tmp_path / "first-subset.jsonl"
    second_output = tmp_path / "second-subset.jsonl"
    first_metadata = build_subset_manifest(
        first_manifest,
        first_output,
        expected_count=3,
        subset_count=2,
        selection_seed="test-seed",
    )
    second_metadata = build_subset_manifest(
        second_manifest,
        second_output,
        expected_count=3,
        subset_count=2,
        selection_seed="test-seed",
    )

    assert first_output.read_text() == second_output.read_text()
    assert first_metadata["selection"] == second_metadata["selection"]
    assert first_metadata["selection"]["method"] == "sha256_ranked_instance_id"
    assert first_metadata["rows"] == 2


def test_canonical_trajectory_hash_ignores_request_local_ids() -> None:
    left = result_row("owner__repo-1", 1)
    left["response"]["output"] = [
        {
            "type": "function_call",
            "id": "response-a",
            "call_id": "call-a",
            "name": "execute_bash",
            "arguments": '{"command":"pytest"}',
            "generation_str": "run tests",
            "prompt_str": "provider prompt a",
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "id": "output-a",
            "call_id": "call-a",
            "output": "1 passed",
            "status": "completed",
        },
    ]
    right = deepcopy(left)
    right["response"]["output"][0].update(
        id="response-b", call_id="call-b", prompt_str="provider prompt b"
    )
    right["response"]["output"][1].update(id="output-b", call_id="call-b")

    assert canonical_trajectory_sha256(left) == canonical_trajectory_sha256(right)

    right["response"]["output"][1]["output"] = "1 failed"
    assert canonical_trajectory_sha256(left) != canonical_trajectory_sha256(right)


def test_compare_arms_reports_trajectory_and_patch_parity() -> None:
    manifest = [manifest_row("owner__repo-1"), manifest_row("owner__repo-2")]
    off_rows = [
        result_row("owner__repo-1", 1, trajectory_text="same"),
        result_row("owner__repo-2", 0, trajectory_text="off", model_patch="same"),
    ]
    on_rows = [
        result_row("owner__repo-1", 0, trajectory_text="same"),
        result_row("owner__repo-2", 0, trajectory_text="on", model_patch="same"),
    ]

    report, outcomes = compare_arms(manifest, off_rows, on_rows, expected_count=2)

    assert report["paired"]["trajectory"]["exact_match_count"] == 1
    assert report["paired"]["trajectory"]["mismatch_count"] == 1
    assert report["paired"]["trajectory"][
        "exact_match_reward_mismatch_instance_ids"
    ] == ["owner__repo-1"]
    assert report["paired"]["model_patch"]["exact_match_count"] == 2
    assert report["paired"]["model_patch"]["exact_nonempty_match_count"] == 2
    assert report["paired"]["model_patch"]["both_empty_count"] == 0
    assert outcomes[0]["trajectory_exact_match"] is True
    assert outcomes[1]["trajectory_exact_match"] is False
