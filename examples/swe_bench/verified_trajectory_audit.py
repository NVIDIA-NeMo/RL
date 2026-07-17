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

"""Build and audit strict paired SWE-bench Verified trajectory evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Iterable

SWE_BENCH_VERIFIED = "princeton-nlp/SWE-bench_Verified"
PERF_FIELDS = (
    "total_model_call_time",
    "total_model_call_count",
    "openhands_run_time",
    "total_command_exec_time",
    "total_controller_action_count",
    "total_controller_action_time",
    "total_controller_cmd_action_count",
    "total_controller_cmd_action_time",
    "create_runtime_time",
    "connect_to_runtime_time",
    "initialize_runtime_time",
    "run_controller_time",
    "complete_runtime_time",
    "runtime_close_time",
    "post_controller_processing_time",
    "process_instance_time",
    "openhands_model_api_call_count",
    "openhands_model_api_call_seconds",
    "openhands_controller_observation_count",
    "openhands_controller_observation_seconds",
    "openhands_action_request_count",
    "openhands_action_request_seconds",
    "openhands_action_server_seconds",
    "openhands_runtime_setup_action_count",
    "openhands_runtime_setup_action_request_seconds",
    "openhands_runtime_setup_action_server_seconds",
    "openhands_runtime_initialize_action_count",
    "openhands_runtime_initialize_action_request_seconds",
    "openhands_runtime_initialize_action_server_seconds",
    "openhands_controller_action_count",
    "openhands_controller_action_request_seconds",
    "openhands_controller_action_server_seconds",
    "openhands_controller_cmd_action_count",
    "openhands_controller_cmd_action_request_seconds",
    "openhands_controller_cmd_action_server_seconds",
    "openhands_runtime_completion_action_count",
    "openhands_runtime_completion_action_request_seconds",
    "openhands_runtime_completion_action_server_seconds",
    "final_eval_time",
    "total_prompt_tokens",
    "total_completion_tokens",
    "num_turns",
    "num_tool_calls",
    "streaming_tool_call_eligible_actions",
    "streaming_tool_call_skipped_no_stable_output",
    "streaming_tool_call_skipped_no_output",
    "streaming_tool_call_skipped_single_snapshot",
    "streaming_tool_call_skipped_below_min_chunk_chars",
    "streaming_tool_call_skipped_completed_before_admission",
    "streaming_tool_call_snapshot_polls",
    "streaming_tool_call_snapshot_revisions",
    "streaming_tool_call_nonempty_snapshots",
    "streaming_tool_call_snapshots_at_or_above_min_chunk_chars",
    "streaming_tool_call_snapshots_at_or_above_initial_chunk_chars",
    "streaming_tool_call_snapshot_wait_cancellations",
    "streaming_tool_call_sessions_started",
    "streaming_tool_call_prefill_requests",
    "streaming_tool_call_prefill_control_plane_requests",
    "streaming_tool_call_prefill_tokens",
    "streaming_tool_call_valid_prefill_actions",
    "streaming_tool_call_tokenizer_only_actions",
    "streaming_tool_call_valid_tokenizer_actions",
    "streaming_tool_call_prompt_reuse_candidates",
    "streaming_tool_call_prompt_reuse_requests",
    "streaming_tool_call_prompt_reuse_matches",
    "streaming_tool_call_prompt_reuse_exact_matches",
    "streaming_tool_call_prompt_reuse_token_equivalent_matches",
    "streaming_tool_call_prompt_reuse_mismatches",
    "streaming_tool_call_prompt_reuse_missing",
    "streaming_tool_call_prefill_reuse_model_call_count",
    "streaming_tool_call_prefill_reuse_model_call_seconds",
    "streaming_tool_call_prefill_reuse_model_call_prompt_tokens",
    "streaming_tool_call_prefill_reuse_model_call_completion_tokens",
    "streaming_tool_call_prefill_reuse_model_call_reused_prompt_tokens",
    "streaming_tool_call_prefill_reuse_model_call_cached_prompt_tokens",
    "streaming_tool_call_prefill_reuse_model_call_required_prefix_tokens",
    "streaming_tool_call_prefill_reuse_model_call_observed_prefill_cached_tokens",
    "streaming_tool_call_prefill_reuse_model_call_observed_prefill_actions",
    "streaming_tool_call_tokenizer_reuse_model_call_count",
    "streaming_tool_call_tokenizer_reuse_model_call_seconds",
    "streaming_tool_call_tokenizer_reuse_model_call_prompt_tokens",
    "streaming_tool_call_tokenizer_reuse_model_call_completion_tokens",
    "streaming_tool_call_tokenizer_reuse_model_call_cached_prompt_tokens",
    "streaming_tool_call_prompt_reuse_miss_model_call_count",
    "streaming_tool_call_prompt_reuse_miss_model_call_seconds",
    "streaming_tool_call_prompt_reuse_miss_model_call_prompt_tokens",
    "streaming_tool_call_prompt_reuse_miss_model_call_completion_tokens",
    "streaming_tool_call_prompt_reuse_miss_model_call_cached_prompt_tokens",
    "streaming_tool_call_incremental_tokenizer_requests",
    "streaming_tool_call_incremental_tokenizer_tokens",
    "streaming_tool_call_exact_incremental_tokenizer_actions",
    "streaming_tool_call_final_only_incremental_tokenizer_actions",
    "streaming_tool_call_valid_final_only_incremental_tokenizer_actions",
    "streaming_tool_call_final_only_prefill_actions",
    "streaming_tool_call_valid_final_only_prefill_actions",
    "streaming_tool_call_final_only_prefill_failures",
    "streaming_tool_call_final_only_prefill_seconds",
    "streaming_tool_call_final_only_prefill_reused_tokens",
    "streaming_tool_call_prefill_after_admission_actions",
    "streaming_tool_call_valid_prefill_after_admission_actions",
    "streaming_tool_call_effective_prefill_after_admission_actions",
    "streaming_tool_call_prefill_after_admission_requests",
    "streaming_tool_call_prefill_after_admission_tokens",
    "streaming_tool_call_prefill_after_admission_request_seconds",
    "streaming_tool_call_prefill_committed_tokens",
    "streaming_tool_call_prefill_dynamic_tokens",
    "streaming_tool_call_prefill_effective_requests",
    "streaming_tool_call_prefill_background_scheduled_chunks",
    "streaming_tool_call_prefill_background_scheduled_tokens",
    "streaming_tool_call_prefill_background_completed_chunks",
    "streaming_tool_call_prefill_background_completed_tokens",
    "streaming_tool_call_prefill_background_completed_dummy_tokens",
    "streaming_tool_call_prefill_background_effective_chunks",
    "streaming_tool_call_prefill_background_dynamic_tokens",
    "streaming_tool_call_prefill_background_cancelled_chunks",
    "streaming_tool_call_prefill_background_cancelled_tokens",
    "streaming_tool_call_prefill_background_failed_chunks",
    "streaming_tool_call_prefill_background_failed_tokens",
    "streaming_tool_call_prefill_background_enqueue_seconds",
    "streaming_tool_call_prefill_background_completion_seconds",
    "streaming_tool_call_deferred_prefill_admissions",
    "streaming_tool_call_final_prefix_tokenizations",
    "streaming_tool_call_prefill_race_attempts",
    "streaming_tool_call_prefill_race_prefill_first",
    "streaming_tool_call_prefill_race_command_first",
    "streaming_tool_call_prefill_race_grace_admissions",
    "streaming_tool_call_prefill_race_cancellations",
    "streaming_tool_call_prefill_race_abort_requests",
    "streaming_tool_call_prefill_race_abort_existing_sessions",
    "streaming_tool_call_prefill_race_abort_failures",
    "streaming_tool_call_prefill_race_abort_seconds",
    "streaming_tool_call_prefill_race_cancelled_request_seconds",
    "streaming_tool_call_prefix_seed_attempts",
    "streaming_tool_call_prefix_seed_successes",
    "streaming_tool_call_prefix_seed_fallbacks",
    "streaming_tool_call_prefix_seed_seconds",
    "streaming_tool_call_stable_first_snapshot_prefill_attempts",
    "streaming_tool_call_stable_first_snapshot_prefill_successes",
    "streaming_tool_call_stable_first_snapshot_prefill_fallbacks",
    "streaming_tool_call_stable_first_snapshot_prefill_seconds",
    "streaming_tool_call_stable_first_snapshot_prefill_stable_tokens",
    "streaming_tool_call_stable_first_snapshot_prefill_committable_tokens",
    "streaming_tool_call_stable_first_snapshot_prefill_dynamic_tokens",
    "streaming_tool_call_fallbacks",
    "streaming_tool_call_fallback_snapshot_errors",
    "streaming_tool_call_fallback_request_errors",
    "streaming_tool_call_fallback_output_prefix_changes",
    "streaming_tool_call_fallback_other_errors",
    "streaming_tool_call_incremental_tokenizer_encoded_chars",
    "streaming_tool_call_incremental_tokenizer_encoded_tokens",
    "streaming_tool_call_incremental_tokenizer_reused_tokens",
    "streaming_tool_call_incremental_tokenizer_rollback_tokens",
    "streaming_tool_call_incremental_tokenizer_checkpoints",
    "streaming_tool_call_incremental_tokenizer_checkpoint_tokens",
    "streaming_tool_call_incremental_tokenizer_checkpoint_mismatches",
    "streaming_tool_call_incremental_tokenizer_seconds",
    "streaming_tool_call_incremental_tokenizer_start_seconds",
    "streaming_tool_call_incremental_tokenizer_final_seconds",
    "streaming_tool_call_server_materialize_seconds",
    "streaming_tool_call_server_render_seconds",
    "streaming_tool_call_server_prefix_render_seconds",
    "streaming_tool_call_server_incremental_tokenizer_seconds",
    "streaming_tool_call_server_authoritative_tokenizer_seconds",
    "streaming_tool_call_server_request_handler_seconds",
    "streaming_tool_call_client_payload_build_seconds",
    "streaming_tool_call_client_payload_serialize_seconds",
    "streaming_tool_call_client_request_bytes",
    "streaming_tool_call_client_compact_context_requests",
    "streaming_tool_call_client_http_round_trip_seconds",
    "streaming_tool_call_client_response_json_seconds",
    "streaming_tool_call_gym_compact_context_registrations",
    "streaming_tool_call_gym_compact_context_hits",
    "streaming_tool_call_gym_compact_context_rebuild_seconds",
    "streaming_tool_call_gym_compact_context_registration_seconds",
    "streaming_tool_call_gym_preprocess_seconds",
    "streaming_tool_call_gym_vllm_request_seconds",
    "streaming_tool_call_gym_request_handler_seconds",
    "streaming_tool_call_server_compact_context_registrations",
    "streaming_tool_call_server_compact_context_hits",
    "streaming_tool_call_server_compact_context_rebuild_seconds",
    "streaming_tool_call_server_compact_context_registration_seconds",
    "streaming_tool_call_counterfactual_full_tokenizer_requests",
    "streaming_tool_call_counterfactual_full_tokenizer_seconds",
    "streaming_tool_call_counterfactual_full_tokenizer_tokens",
    "streaming_tool_call_counterfactual_full_tokenizer_mismatches",
    "streaming_tool_call_counterfactual_full_tokenizer_failures",
    "streaming_tool_call_command_request_seconds",
    "streaming_tool_call_post_command_tail_seconds",
    "streaming_tool_call_snapshot_request_seconds",
    "streaming_tool_call_overhead_seconds",
)
TRAJECTORY_DISTRIBUTION_FIELDS = (
    "openhands_run_time",
    "openhands_model_api_call_seconds",
    "openhands_model_api_call_count",
    "total_model_call_time",
    "total_model_call_count",
    "total_command_exec_time",
    "num_tool_calls",
)
INFRASTRUCTURE_ERROR_MARKERS = (
    "NameResolutionError",
    "Temporary failure in name resolution",
    "requests.exceptions.ConnectionError",
    "Container not found",
)
TRANSIENT_TRAJECTORY_FIELDS = {
    "encrypted_content",
    "id",
    "prompt_str",
}


class AuditError(ValueError):
    """Raised when an evaluation artifact violates the strict protocol."""


def canonical_json_sha256(value: Any) -> str:
    """Hash a value using a deterministic JSON representation."""
    serialized = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def canonical_trajectory_sha256(result: dict[str, Any]) -> str | None:
    """Hash the semantic Gym transcript without request-local identifiers."""
    params = result.get("responses_create_params")
    response = result.get("response")
    if not isinstance(params, dict) or not isinstance(response, dict):
        return None

    input_items = params.get("input")
    output_items = response.get("output")
    if not isinstance(input_items, list) or not isinstance(output_items, list):
        return None

    call_ids: dict[str, str] = {}

    def normalize_call_id(call_id: Any) -> Any:
        if not isinstance(call_id, str):
            return call_id
        if call_id not in call_ids:
            call_ids[call_id] = f"call_{len(call_ids)}"
        return call_ids[call_id]

    def normalize_item(item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        normalized = {
            key: value
            for key, value in item.items()
            if key not in TRANSIENT_TRAJECTORY_FIELDS
        }
        if "call_id" in normalized:
            normalized["call_id"] = normalize_call_id(normalized["call_id"])
        return normalized

    canonical_trajectory = {
        "input": [normalize_item(item) for item in input_items],
        "output": [normalize_item(item) for item in output_items],
    }
    return canonical_json_sha256(canonical_trajectory)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and report the exact malformed line on failure."""
    rows = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AuditError(f"{path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise AuditError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def count_function_calls(response: Any) -> int | None:
    """Return the number of function calls recorded in a Responses API result."""
    if not isinstance(response, dict):
        return None
    output = response.get("output")
    if not isinstance(output, list):
        return None
    return sum(
        isinstance(item, dict) and item.get("type") == "function_call"
        for item in output
    )


def is_infrastructure_error(full_result: dict[str, Any]) -> bool:
    """Detect harness failures without scanning normal agent tool output."""
    if full_result.get("infrastructure_error") is True:
        return True
    response = full_result.get("response")
    if not isinstance(response, dict) or response.get("error") is None:
        return False
    error_text = json.dumps(response["error"], ensure_ascii=False)
    return any(marker in error_text for marker in INFRASTRUCTURE_ERROR_MARKERS)


def read_result_jsonl(path: Path) -> list[dict[str, Any]]:
    """Stream a large trajectory JSONL while retaining only audit fields."""
    rows = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                full_result = json.loads(line)
            except json.JSONDecodeError as error:
                raise AuditError(f"{path}:{line_number}: {error}") from error
            if not isinstance(full_result, dict):
                raise AuditError(f"{path}:{line_number}: expected a JSON object")

            params = full_result.get("responses_create_params", {})
            metadata = params.get("metadata", {})
            response = full_result.get("response")
            num_tool_calls = full_result.get("num_tool_calls")
            if not isinstance(num_tool_calls, (int, float)) or isinstance(
                num_tool_calls, bool
            ):
                num_tool_calls = count_function_calls(response)
            result = {
                "responses_create_params": {
                    "model": params.get("model"),
                    "temperature": params.get("temperature"),
                    "top_p": params.get("top_p"),
                    "metadata": {
                        "instance_id": metadata.get("instance_id"),
                        "dataset_name": metadata.get("dataset_name"),
                        "base_commit": metadata.get("base_commit"),
                    },
                },
                "reward": full_result.get("reward"),
                "resolved": full_result.get("resolved"),
                "infrastructure_error": is_infrastructure_error(full_result),
                "response_error": isinstance(response, dict)
                and response.get("error") is not None,
                "trajectory_sha256": canonical_trajectory_sha256(full_result),
                "model_patch_available": "model_patch" in full_result,
                "model_patch_nonempty": isinstance(full_result.get("model_patch"), str)
                and bool(full_result["model_patch"]),
                "model_patch_sha256": (
                    canonical_json_sha256(full_result["model_patch"])
                    if "model_patch" in full_result
                    else None
                ),
            }
            for required_field in ("response", "instance_config"):
                if required_field in full_result:
                    result[required_field] = None
            for field in PERF_FIELDS:
                result[field] = (
                    num_tool_calls
                    if field == "num_tool_calls"
                    else full_result.get(field)
                )
            rows.append(result)
    return rows


def get_instance_id(row: dict[str, Any]) -> str:
    """Extract an instance ID from either an input row or a Gym result."""
    instance_id = row.get("instance_id")
    if instance_id is None:
        instance_id = (
            row.get("responses_create_params", {})
            .get("metadata", {})
            .get("instance_id")
        )
    if not isinstance(instance_id, str) or not instance_id:
        raise AuditError("row is missing responses_create_params.metadata.instance_id")
    return instance_id


def get_dataset_name(row: dict[str, Any]) -> str | None:
    """Return the source dataset name when present."""
    return row.get("responses_create_params", {}).get("metadata", {}).get(
        "dataset_name"
    ) or row.get("dataset_name")


def index_unique_rows(
    rows: Iterable[dict[str, Any]], artifact_name: str
) -> dict[str, dict[str, Any]]:
    """Index rows by instance ID, rejecting duplicate trajectories."""
    indexed = {}
    for row in rows:
        instance_id = get_instance_id(row)
        if instance_id in indexed:
            raise AuditError(f"{artifact_name}: duplicate instance_id={instance_id}")
        indexed[instance_id] = row
    return indexed


def validate_manifest(
    rows: list[dict[str, Any]], expected_count: int
) -> dict[str, dict[str, Any]]:
    """Validate exact Verified membership and routing for a manifest."""
    indexed = index_unique_rows(rows, "manifest")
    if len(indexed) != expected_count:
        raise AuditError(
            f"manifest: expected {expected_count} unique instances, got {len(indexed)}"
        )
    for instance_id, row in indexed.items():
        if get_dataset_name(row) != SWE_BENCH_VERIFIED:
            raise AuditError(f"manifest: {instance_id} is not SWE-bench Verified")
        if row.get("agent_ref") != {
            "type": "responses_api_agents",
            "name": "swe_agents_val",
        }:
            raise AuditError(f"manifest: {instance_id} is not routed to swe_agents_val")
    return indexed


def file_sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically write a formatted JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        temporary_path = Path(f.name)
    os.replace(temporary_path, path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomically write deterministic JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        temporary_path = Path(f.name)
    os.replace(temporary_path, path)


def build_manifest(
    inputs: list[Path], output: Path, expected_count: int
) -> dict[str, Any]:
    """Merge input shards into a sorted, immutable evaluation manifest."""
    rows = []
    sources = []
    for path in sorted(inputs):
        source_rows = read_jsonl(path)
        rows.extend(source_rows)
        sources.append(
            {
                "path": str(path.resolve()),
                "rows": len(source_rows),
                "sha256": file_sha256(path),
            }
        )

    indexed = validate_manifest(rows, expected_count)
    atomic_write_jsonl(output, (indexed[key] for key in sorted(indexed)))
    metadata = {
        "dataset": SWE_BENCH_VERIFIED,
        "rows": len(indexed),
        "sha256": file_sha256(output),
        "manifest": str(output.resolve()),
        "sources": sources,
    }
    atomic_write_json(output.with_suffix(".metadata.json"), metadata)
    return metadata


def build_infrastructure_retry_manifest(
    manifest_path: Path,
    comparison_report_path: Path,
    output: Path,
    expected_count: int,
) -> dict[str, Any]:
    """Build a paired retry manifest from either arm's infrastructure failures."""
    manifest = validate_manifest(read_jsonl(manifest_path), expected_count)
    with comparison_report_path.open() as f:
        comparison_report = json.load(f)
    retry_ids = sorted(
        set(comparison_report["streaming_off"]["infrastructure_error_instance_ids"])
        | set(comparison_report["streaming_on"]["infrastructure_error_instance_ids"])
    )
    unknown_ids = sorted(set(retry_ids) - set(manifest))
    if unknown_ids:
        raise AuditError(
            f"retry report contains unknown instance IDs: {unknown_ids[:10]}"
        )
    if not retry_ids:
        raise AuditError("retry report does not contain infrastructure failures")

    atomic_write_jsonl(output, (manifest[instance_id] for instance_id in retry_ids))
    metadata = {
        "dataset": SWE_BENCH_VERIFIED,
        "rows": len(retry_ids),
        "sha256": file_sha256(output),
        "manifest": str(output.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": file_sha256(manifest_path),
        "comparison_report": str(comparison_report_path.resolve()),
    }
    atomic_write_json(output.with_suffix(".metadata.json"), metadata)
    return metadata


def build_no_timeout_manifest(
    manifest_path: Path,
    trajectory_paths: list[Path],
    output: Path,
    expected_count: int,
    expected_output_count: int | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Exclude instances that reached the agent timeout in any trajectory file."""
    if timeout_seconds <= 0:
        raise AuditError("timeout seconds must be positive")

    manifest_rows = read_jsonl(manifest_path)
    manifest = validate_manifest(manifest_rows, expected_count)
    excluded_ids = set()
    trajectory_summaries = []
    for trajectory_path in trajectory_paths:
        rows = read_result_jsonl(trajectory_path)
        indexed = index_unique_rows(rows, f"trajectory {trajectory_path}")
        missing_ids = sorted(set(manifest) - set(indexed))
        unexpected_ids = sorted(set(indexed) - set(manifest))
        if missing_ids or unexpected_ids:
            raise AuditError(
                f"trajectory {trajectory_path}: membership mismatch; "
                f"missing={missing_ids[:10]}, unexpected={unexpected_ids[:10]}"
            )

        timeout_ids = []
        for instance_id, row in indexed.items():
            runtime = row.get("openhands_run_time")
            if not isinstance(runtime, (int, float)) or isinstance(runtime, bool):
                raise AuditError(
                    f"trajectory {trajectory_path}: {instance_id} is missing numeric "
                    "openhands_run_time"
                )
            if runtime >= timeout_seconds:
                timeout_ids.append(instance_id)
                excluded_ids.add(instance_id)
        trajectory_summaries.append(
            {
                "path": str(trajectory_path.resolve()),
                "rows": len(rows),
                "sha256": file_sha256(trajectory_path),
                "timeout_count": len(timeout_ids),
                "timeout_instance_ids": sorted(timeout_ids),
            }
        )

    retained_rows = [
        row for row in manifest_rows if get_instance_id(row) not in excluded_ids
    ]
    if (
        expected_output_count is not None
        and len(retained_rows) != expected_output_count
    ):
        raise AuditError(
            f"no-timeout manifest: expected {expected_output_count} retained instances, "
            f"got {len(retained_rows)}"
        )

    atomic_write_jsonl(output, retained_rows)
    metadata = {
        "dataset": SWE_BENCH_VERIFIED,
        "rows": len(retained_rows),
        "sha256": file_sha256(output),
        "manifest": str(output.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_rows": len(manifest),
        "source_manifest_sha256": file_sha256(manifest_path),
        "filter": {
            "metric": "openhands_run_time",
            "operator": ">=",
            "timeout_seconds": timeout_seconds,
            "semantics": "exclude if observed in any supplied trajectory",
        },
        "excluded_count": len(excluded_ids),
        "excluded_instance_ids": sorted(excluded_ids),
        "trajectories": trajectory_summaries,
    }
    atomic_write_json(output.with_suffix(".metadata.json"), metadata)
    return metadata


def build_subset_manifest(
    manifest_path: Path,
    output: Path,
    expected_count: int,
    subset_count: int,
    selection_seed: str,
) -> dict[str, Any]:
    """Build a deterministic hash-ranked subset of a validated manifest."""
    if subset_count <= 0:
        raise AuditError("subset count must be positive")
    if not selection_seed:
        raise AuditError("selection seed must be non-empty")

    manifest_rows = read_jsonl(manifest_path)
    manifest = validate_manifest(manifest_rows, expected_count)
    if subset_count > len(manifest):
        raise AuditError(
            f"subset count {subset_count} exceeds manifest size {len(manifest)}"
        )

    ranked_ids = sorted(
        manifest,
        key=lambda instance_id: hashlib.sha256(
            f"{selection_seed}\0{instance_id}".encode()
        ).hexdigest(),
    )
    selected_ids = ranked_ids[:subset_count]
    atomic_write_jsonl(output, (manifest[instance_id] for instance_id in selected_ids))
    metadata = {
        "dataset": SWE_BENCH_VERIFIED,
        "rows": len(selected_ids),
        "sha256": file_sha256(output),
        "manifest": str(output.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_rows": len(manifest),
        "source_manifest_sha256": file_sha256(manifest_path),
        "selection": {
            "method": "sha256_ranked_instance_id",
            "seed": selection_seed,
            "selected_instance_ids_sha256": canonical_json_sha256(selected_ids),
        },
    }
    atomic_write_json(output.with_suffix(".metadata.json"), metadata)
    return metadata


def build_prefix_manifest(
    manifest_path: Path,
    output: Path,
    expected_count: int,
    prefix_count: int,
) -> dict[str, Any]:
    """Build a manifest from the first rows in source order."""
    if prefix_count <= 0:
        raise AuditError("prefix count must be positive")

    manifest_rows = read_jsonl(manifest_path)
    manifest = validate_manifest(manifest_rows, expected_count)
    if prefix_count > len(manifest_rows):
        raise AuditError(
            f"prefix count {prefix_count} exceeds manifest size {len(manifest_rows)}"
        )

    selected_rows = manifest_rows[:prefix_count]
    selected_ids = [get_instance_id(row) for row in selected_rows]
    atomic_write_jsonl(output, selected_rows)
    metadata = {
        "dataset": SWE_BENCH_VERIFIED,
        "rows": len(selected_rows),
        "sha256": file_sha256(output),
        "manifest": str(output.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_rows": len(manifest),
        "source_manifest_sha256": file_sha256(manifest_path),
        "selection": {
            "method": "source_order_prefix",
            "prefix_count": prefix_count,
            "selected_instance_ids_sha256": canonical_json_sha256(selected_ids),
        },
    }
    atomic_write_json(output.with_suffix(".metadata.json"), metadata)
    return metadata


def overlay_result_rows(
    base_rows: list[dict[str, Any]], retry_paths: list[Path]
) -> list[dict[str, Any]]:
    """Overlay retry results by instance ID, with later retries taking precedence."""
    indexed = index_unique_rows(base_rows, "base results")
    for retry_path in retry_paths:
        retry_rows = index_unique_rows(
            read_result_jsonl(retry_path), f"retry results {retry_path}"
        )
        indexed.update(retry_rows)
    return list(indexed.values())


def metric_summary(values: list[float]) -> dict[str, float | int | None]:
    """Summarize a numeric per-trajectory metric without dropping its count."""
    if not values:
        return {"count": 0, "sum": None, "mean": None, "median": None, "p95": None}
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
    }


def percentile_distribution(values: list[float]) -> dict[str, float | int | None]:
    """Summarize per-trajectory values with nearest-rank percentiles."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    ordered = sorted(values)

    def nearest_rank(fraction: float) -> float:
        index = max(0, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]

    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "p25": nearest_rank(0.25),
        "p50": nearest_rank(0.50),
        "p75": nearest_rank(0.75),
        "p90": nearest_rank(0.90),
        "p95": nearest_rank(0.95),
        "p99": nearest_rank(0.99),
        "max": ordered[-1],
    }


def build_timeout_excluded_distribution(
    manifest_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    expected_count: int,
    timeout_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare per-trajectory distributions after excluding timeout union."""
    if timeout_seconds <= 0:
        raise AuditError("timeout seconds must be positive")

    manifest = validate_manifest(manifest_rows, expected_count)
    baseline = index_unique_rows(baseline_rows, "baseline")
    candidate = index_unique_rows(candidate_rows, "candidate")
    expected_ids = set(manifest)
    for name, indexed in (("baseline", baseline), ("candidate", candidate)):
        missing_ids = sorted(expected_ids - set(indexed))
        unexpected_ids = sorted(set(indexed) - expected_ids)
        if missing_ids or unexpected_ids:
            raise AuditError(
                f"{name}: membership mismatch; missing={missing_ids[:10]}, "
                f"unexpected={unexpected_ids[:10]}"
            )

    timeout_ids = {"baseline": [], "candidate": []}
    paired_rows = []
    retained_values = {
        "baseline": {field: [] for field in TRAJECTORY_DISTRIBUTION_FIELDS},
        "candidate": {field: [] for field in TRAJECTORY_DISTRIBUTION_FIELDS},
        "candidate_minus_baseline": {
            field: [] for field in TRAJECTORY_DISTRIBUTION_FIELDS
        },
    }
    independent_values = {
        "baseline": {field: [] for field in TRAJECTORY_DISTRIBUTION_FIELDS},
        "candidate": {field: [] for field in TRAJECTORY_DISTRIBUTION_FIELDS},
    }
    for instance_id in sorted(manifest):
        arm_rows = {
            "baseline": baseline[instance_id],
            "candidate": candidate[instance_id],
        }
        arm_timeouts = {}
        for name, row in arm_rows.items():
            runtime = row.get("openhands_run_time")
            if not isinstance(runtime, (int, float)) or isinstance(runtime, bool):
                raise AuditError(
                    f"{name}: {instance_id} is missing numeric openhands_run_time"
                )
            arm_timeouts[name] = float(runtime) >= timeout_seconds
            if arm_timeouts[name]:
                timeout_ids[name].append(instance_id)

        excluded = arm_timeouts["baseline"] or arm_timeouts["candidate"]
        arm_values = {}
        missing_metric_fields = {}
        for name, row in arm_rows.items():
            metrics = {}
            missing_fields = []
            for field in TRAJECTORY_DISTRIBUTION_FIELDS:
                value = row.get(field)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    if not excluded:
                        raise AuditError(
                            f"{name}: {instance_id} is missing numeric {field}"
                        )
                    metrics[field] = None
                    missing_fields.append(field)
                else:
                    metrics[field] = float(value)
            arm_values[name] = metrics
            missing_metric_fields[name] = missing_fields

        deltas = {
            field: (
                arm_values["candidate"][field] - arm_values["baseline"][field]
                if arm_values["candidate"][field] is not None
                and arm_values["baseline"][field] is not None
                else None
            )
            for field in TRAJECTORY_DISTRIBUTION_FIELDS
        }
        paired_rows.append(
            {
                "instance_id": instance_id,
                "excluded_as_timeout": excluded,
                "baseline_timeout": arm_timeouts["baseline"],
                "candidate_timeout": arm_timeouts["candidate"],
                "baseline_missing_metric_fields": missing_metric_fields["baseline"],
                "candidate_missing_metric_fields": missing_metric_fields["candidate"],
                "baseline": arm_values["baseline"],
                "candidate": arm_values["candidate"],
                "candidate_minus_baseline": deltas,
            }
        )
        for name in ("baseline", "candidate"):
            if arm_timeouts[name]:
                continue
            for field in TRAJECTORY_DISTRIBUTION_FIELDS:
                value = arm_values[name][field]
                if value is None:
                    raise AuditError(
                        f"{name}: {instance_id} is missing numeric {field}"
                    )
                independent_values[name][field].append(value)
        if excluded:
            continue
        for field in TRAJECTORY_DISTRIBUTION_FIELDS:
            retained_values["baseline"][field].append(arm_values["baseline"][field])
            retained_values["candidate"][field].append(arm_values["candidate"][field])
            retained_values["candidate_minus_baseline"][field].append(deltas[field])

    excluded_ids = sorted(set(timeout_ids["baseline"]) | set(timeout_ids["candidate"]))
    report = {
        "protocol": {
            "expected_instances": expected_count,
            "timeout_seconds": timeout_seconds,
            "timeout_rule": "openhands_run_time >= timeout_seconds",
            "cohort_rule": "exclude union of baseline and candidate timeouts",
            "percentile_method": "nearest-rank",
        },
        "timeout": {
            "baseline_count": len(timeout_ids["baseline"]),
            "baseline_instance_ids": timeout_ids["baseline"],
            "candidate_count": len(timeout_ids["candidate"]),
            "candidate_instance_ids": timeout_ids["candidate"],
            "excluded_union_count": len(excluded_ids),
            "excluded_union_instance_ids": excluded_ids,
            "retained_count": expected_count - len(excluded_ids),
        },
        "metrics": {
            cohort: {
                field: percentile_distribution(values)
                for field, values in cohort_values.items()
            }
            for cohort, cohort_values in retained_values.items()
        },
        "independent_non_timeout": {
            "cohort_rule": "exclude each arm's own timeouts independently",
            "baseline_retained_count": expected_count - len(timeout_ids["baseline"]),
            "candidate_retained_count": expected_count - len(timeout_ids["candidate"]),
            "metrics": {
                cohort: {
                    field: percentile_distribution(values)
                    for field, values in cohort_values.items()
                }
                for cohort, cohort_values in independent_values.items()
            },
        },
    }
    return report, paired_rows


def summarize_arm(
    name: str,
    rows: list[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    expected_temperature: float,
    expected_top_p: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Audit one arm and compute accuracy with the full manifest denominator."""
    indexed = index_unique_rows(rows, name)
    unexpected = sorted(set(indexed) - set(manifest))
    if unexpected:
        raise AuditError(f"{name}: unexpected instance IDs: {unexpected[:10]}")

    resolved_count = 0
    infrastructure_error_ids = []
    response_error_ids = []
    models = set()
    metrics: dict[str, list[float]] = {field: [] for field in PERF_FIELDS}
    for instance_id, row in indexed.items():
        for required_field in ("response", "instance_config"):
            if required_field not in row:
                raise AuditError(
                    f"{name}: {instance_id} is missing full trajectory field "
                    f"{required_field}"
                )
        params = row.get("responses_create_params", {})
        metadata = params.get("metadata", {})
        if metadata.get("dataset_name") != SWE_BENCH_VERIFIED:
            raise AuditError(f"{name}: {instance_id} is not labeled SWE-bench Verified")
        if params.get("temperature") != expected_temperature:
            raise AuditError(
                f"{name}: {instance_id} temperature={params.get('temperature')}, "
                f"expected {expected_temperature}"
            )
        if params.get("top_p") != expected_top_p:
            raise AuditError(
                f"{name}: {instance_id} top_p={params.get('top_p')}, "
                f"expected {expected_top_p}"
            )
        expected_base_commit = manifest[instance_id].get("base_commit")
        if metadata.get("base_commit") != expected_base_commit:
            raise AuditError(f"{name}: {instance_id} has a mismatched base commit")
        models.add(params.get("model"))

        reward = row.get("reward")
        resolved = row.get("resolved")
        if reward not in (0, 0.0, 1, 1.0):
            raise AuditError(f"{name}: {instance_id} has invalid reward={reward}")
        if (reward == 1) != (resolved is True):
            raise AuditError(
                f"{name}: {instance_id} has inconsistent reward/resolved values"
            )
        resolved_count += int(reward == 1)
        if row.get("infrastructure_error"):
            infrastructure_error_ids.append(instance_id)
        if row.get("response_error"):
            response_error_ids.append(instance_id)

        for field in PERF_FIELDS:
            value = row.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[field].append(float(value))

    missing = sorted(set(manifest) - set(indexed))
    denominator = len(manifest)
    if indexed and (None in models or len(models) != 1):
        raise AuditError(f"{name}: expected one non-null model, got {models}")
    return (
        {
            "trajectory_rows": len(rows),
            "unique_present": len(indexed),
            "missing_count": len(missing),
            "missing_instance_ids": missing,
            "resolved": resolved_count,
            "infrastructure_error_count": len(infrastructure_error_ids),
            "infrastructure_error_instance_ids": sorted(infrastructure_error_ids),
            "response_error_count": len(response_error_ids),
            "response_error_instance_ids": sorted(response_error_ids),
            "strict_denominator": denominator,
            "strict_accuracy": resolved_count / denominator,
            "models": sorted(str(model) for model in models),
            "metrics": {
                field: metric_summary(values) for field, values in metrics.items()
            },
        },
        indexed,
    )


def compare_arms(
    manifest_rows: list[dict[str, Any]],
    off_rows: list[dict[str, Any]],
    on_rows: list[dict[str, Any]],
    expected_count: int = 500,
    expected_temperature: float = 0.0,
    expected_top_p: float = 1.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare paired arms over the exact manifest, treating missing rows as wrong."""
    manifest = validate_manifest(manifest_rows, expected_count)
    off_summary, off = summarize_arm(
        "streaming_off", off_rows, manifest, expected_temperature, expected_top_p
    )
    on_summary, on = summarize_arm(
        "streaming_on", on_rows, manifest, expected_temperature, expected_top_p
    )
    if (
        off_summary["models"]
        and on_summary["models"]
        and off_summary["models"] != on_summary["models"]
    ):
        raise AuditError(
            "streaming arms used different models: "
            f"off={off_summary['models']}, on={on_summary['models']}"
        )

    outcomes = []
    off_only = 0
    on_only = 0
    both_resolved = 0
    neither_resolved = 0
    trajectory_exact_match_ids = []
    trajectory_mismatch_ids = []
    trajectory_unavailable_ids = []
    patch_exact_match_ids = []
    patch_exact_nonempty_match_ids = []
    patch_both_empty_ids = []
    patch_mismatch_ids = []
    patch_unavailable_ids = []
    exact_trajectory_reward_mismatch_ids = []
    for instance_id in sorted(manifest):
        off_row = off.get(instance_id, {})
        on_row = on.get(instance_id, {})
        off_reward = int(off_row.get("reward", 0) == 1)
        on_reward = int(on_row.get("reward", 0) == 1)
        off_only += int(off_reward == 1 and on_reward == 0)
        on_only += int(off_reward == 0 and on_reward == 1)
        both_resolved += int(off_reward == 1 and on_reward == 1)
        neither_resolved += int(off_reward == 0 and on_reward == 0)

        off_trajectory_hash = off_row.get("trajectory_sha256")
        on_trajectory_hash = on_row.get("trajectory_sha256")
        trajectory_exact_match = None
        if off_trajectory_hash is not None and on_trajectory_hash is not None:
            trajectory_exact_match = off_trajectory_hash == on_trajectory_hash
            if trajectory_exact_match:
                trajectory_exact_match_ids.append(instance_id)
                if off_reward != on_reward:
                    exact_trajectory_reward_mismatch_ids.append(instance_id)
            else:
                trajectory_mismatch_ids.append(instance_id)
        else:
            trajectory_unavailable_ids.append(instance_id)

        off_patch_hash = off_row.get("model_patch_sha256")
        on_patch_hash = on_row.get("model_patch_sha256")
        patch_exact_match = None
        if off_row.get("model_patch_available") and on_row.get("model_patch_available"):
            patch_exact_match = off_patch_hash == on_patch_hash
            if patch_exact_match:
                patch_exact_match_ids.append(instance_id)
                if off_row.get("model_patch_nonempty") and on_row.get(
                    "model_patch_nonempty"
                ):
                    patch_exact_nonempty_match_ids.append(instance_id)
                elif not off_row.get("model_patch_nonempty") and not on_row.get(
                    "model_patch_nonempty"
                ):
                    patch_both_empty_ids.append(instance_id)
            else:
                patch_mismatch_ids.append(instance_id)
        else:
            patch_unavailable_ids.append(instance_id)

        outcomes.append(
            {
                "instance_id": instance_id,
                "streaming_off_present": instance_id in off,
                "streaming_off_reward": off_reward,
                "streaming_off_trajectory_sha256": off_trajectory_hash,
                "streaming_off_model_patch_sha256": off_patch_hash,
                "streaming_off_model_patch_nonempty": off_row.get(
                    "model_patch_nonempty"
                ),
                "streaming_on_present": instance_id in on,
                "streaming_on_reward": on_reward,
                "streaming_on_trajectory_sha256": on_trajectory_hash,
                "streaming_on_model_patch_sha256": on_patch_hash,
                "streaming_on_model_patch_nonempty": on_row.get("model_patch_nonempty"),
                "trajectory_exact_match": trajectory_exact_match,
                "model_patch_exact_match": patch_exact_match,
            }
        )

    report = {
        "protocol": {
            "dataset": SWE_BENCH_VERIFIED,
            "expected_instances": expected_count,
            "temperature": expected_temperature,
            "top_p": expected_top_p,
            "missing_trajectories_count_as_incorrect": True,
        },
        "streaming_off": off_summary,
        "streaming_on": on_summary,
        "paired": {
            "both_resolved": both_resolved,
            "streaming_off_only": off_only,
            "streaming_on_only": on_only,
            "neither_resolved": neither_resolved,
            "accuracy_delta_on_minus_off": (
                on_summary["strict_accuracy"] - off_summary["strict_accuracy"]
            ),
            "trajectory": {
                "exact_match_count": len(trajectory_exact_match_ids),
                "mismatch_count": len(trajectory_mismatch_ids),
                "unavailable_count": len(trajectory_unavailable_ids),
                "exact_match_instance_ids": trajectory_exact_match_ids,
                "mismatch_instance_ids": trajectory_mismatch_ids,
                "unavailable_instance_ids": trajectory_unavailable_ids,
                "exact_match_reward_mismatch_count": len(
                    exact_trajectory_reward_mismatch_ids
                ),
                "exact_match_reward_mismatch_instance_ids": (
                    exact_trajectory_reward_mismatch_ids
                ),
            },
            "model_patch": {
                "exact_match_count": len(patch_exact_match_ids),
                "exact_nonempty_match_count": len(patch_exact_nonempty_match_ids),
                "both_empty_count": len(patch_both_empty_ids),
                "mismatch_count": len(patch_mismatch_ids),
                "unavailable_count": len(patch_unavailable_ids),
                "exact_match_instance_ids": patch_exact_match_ids,
                "exact_nonempty_match_instance_ids": (patch_exact_nonempty_match_ids),
                "both_empty_instance_ids": patch_both_empty_ids,
                "mismatch_instance_ids": patch_mismatch_ids,
                "unavailable_instance_ids": patch_unavailable_ids,
            },
        },
        "complete": not off_summary["missing_count"]
        and not on_summary["missing_count"],
    }
    return report, outcomes


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge = subparsers.add_parser("merge", help="merge and pin Verified shards")
    merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--expected-count", type=int, default=500)

    verify = subparsers.add_parser("verify", help="verify an evaluation manifest")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--expected-count", type=int, default=500)

    retry_manifest = subparsers.add_parser(
        "retry-manifest", help="build a paired manifest for infrastructure failures"
    )
    retry_manifest.add_argument("--manifest", type=Path, required=True)
    retry_manifest.add_argument("--comparison-report", type=Path, required=True)
    retry_manifest.add_argument("--output", type=Path, required=True)
    retry_manifest.add_argument("--expected-count", type=int, default=500)

    no_timeout_manifest = subparsers.add_parser(
        "no-timeout-manifest",
        help="exclude instances that timed out in any supplied trajectory",
    )
    no_timeout_manifest.add_argument("--manifest", type=Path, required=True)
    no_timeout_manifest.add_argument(
        "--trajectories", type=Path, nargs="+", required=True
    )
    no_timeout_manifest.add_argument("--output", type=Path, required=True)
    no_timeout_manifest.add_argument("--expected-count", type=int, default=500)
    no_timeout_manifest.add_argument("--expected-output-count", type=int)
    no_timeout_manifest.add_argument("--timeout-seconds", type=float, default=1800.0)

    subset_manifest = subparsers.add_parser(
        "subset-manifest", help="build a deterministic hash-ranked manifest subset"
    )
    subset_manifest.add_argument("--manifest", type=Path, required=True)
    subset_manifest.add_argument("--output", type=Path, required=True)
    subset_manifest.add_argument("--expected-count", type=int, default=500)
    subset_manifest.add_argument("--subset-count", type=int, required=True)
    subset_manifest.add_argument(
        "--selection-seed", default="streaming-tool-call-subset-v1"
    )

    prefix_manifest = subparsers.add_parser(
        "prefix-manifest", help="build a manifest from the first rows in source order"
    )
    prefix_manifest.add_argument("--manifest", type=Path, required=True)
    prefix_manifest.add_argument("--output", type=Path, required=True)
    prefix_manifest.add_argument("--expected-count", type=int, default=500)
    prefix_manifest.add_argument("--prefix-count", type=int, required=True)

    compare = subparsers.add_parser("compare", help="strictly compare two result files")
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--streaming-off", type=Path, required=True)
    compare.add_argument("--streaming-on", type=Path, required=True)
    compare.add_argument(
        "--streaming-off-retry", type=Path, action="append", default=[]
    )
    compare.add_argument("--streaming-on-retry", type=Path, action="append", default=[])
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--paired-output", type=Path, required=True)
    compare.add_argument("--expected-count", type=int, default=500)
    compare.add_argument("--expected-temperature", type=float, default=0.0)
    compare.add_argument("--expected-top-p", type=float, default=1.0)

    distribution = subparsers.add_parser(
        "distribution",
        help="compare per-trajectory metrics after excluding timeout union",
    )
    distribution.add_argument("--manifest", type=Path, required=True)
    distribution.add_argument("--baseline", type=Path, required=True)
    distribution.add_argument("--candidate", type=Path, required=True)
    distribution.add_argument("--output", type=Path, required=True)
    distribution.add_argument("--paired-output", type=Path, required=True)
    distribution.add_argument("--expected-count", type=int, default=500)
    distribution.add_argument("--timeout-seconds", type=float, default=1800.0)
    return parser.parse_args()


def main() -> int:
    """Run the selected audit operation."""
    args = parse_args()
    try:
        if args.command == "merge":
            summary = build_manifest(args.inputs, args.output, args.expected_count)
        elif args.command == "verify":
            rows = read_jsonl(args.manifest)
            indexed = validate_manifest(rows, args.expected_count)
            summary = {
                "dataset": SWE_BENCH_VERIFIED,
                "rows": len(indexed),
                "sha256": file_sha256(args.manifest),
                "manifest": str(args.manifest.resolve()),
            }
        elif args.command == "retry-manifest":
            summary = build_infrastructure_retry_manifest(
                args.manifest,
                args.comparison_report,
                args.output,
                args.expected_count,
            )
        elif args.command == "no-timeout-manifest":
            summary = build_no_timeout_manifest(
                args.manifest,
                args.trajectories,
                args.output,
                args.expected_count,
                args.expected_output_count,
                args.timeout_seconds,
            )
        elif args.command == "subset-manifest":
            summary = build_subset_manifest(
                args.manifest,
                args.output,
                args.expected_count,
                args.subset_count,
                args.selection_seed,
            )
        elif args.command == "prefix-manifest":
            summary = build_prefix_manifest(
                args.manifest,
                args.output,
                args.expected_count,
                args.prefix_count,
            )
        elif args.command == "distribution":
            summary, paired_rows = build_timeout_excluded_distribution(
                read_jsonl(args.manifest),
                read_result_jsonl(args.baseline),
                read_result_jsonl(args.candidate),
                expected_count=args.expected_count,
                timeout_seconds=args.timeout_seconds,
            )
            summary["protocol"]["manifest_sha256"] = file_sha256(args.manifest)
            summary["artifacts"] = {
                "manifest": str(args.manifest.resolve()),
                "baseline": str(args.baseline.resolve()),
                "candidate": str(args.candidate.resolve()),
                "paired_trajectories": str(args.paired_output.resolve()),
            }
            atomic_write_json(args.output, summary)
            atomic_write_jsonl(args.paired_output, paired_rows)
        else:
            report, outcomes = compare_arms(
                read_jsonl(args.manifest),
                overlay_result_rows(
                    read_result_jsonl(args.streaming_off), args.streaming_off_retry
                ),
                overlay_result_rows(
                    read_result_jsonl(args.streaming_on), args.streaming_on_retry
                ),
                args.expected_count,
                args.expected_temperature,
                args.expected_top_p,
            )
            report["protocol"]["manifest_sha256"] = file_sha256(args.manifest)
            report["artifacts"] = {
                "manifest": str(args.manifest.resolve()),
                "streaming_off": str(args.streaming_off.resolve()),
                "streaming_on": str(args.streaming_on.resolve()),
                "streaming_off_retries": [
                    str(path.resolve()) for path in args.streaming_off_retry
                ],
                "streaming_on_retries": [
                    str(path.resolve()) for path in args.streaming_on_retry
                ],
                "paired_outcomes": str(args.paired_output.resolve()),
            }
            atomic_write_json(args.output, report)
            atomic_write_jsonl(args.paired_output, outcomes)
            summary = report
    except AuditError as error:
        print(json.dumps({"error": str(error)}, indent=2))
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.command == "compare" and not summary["complete"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
