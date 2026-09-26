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

"""Inspect the coordinated Gym/TQ snapshot used by the turn-recovery test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any


_PROFILES = ("counter", "workplace", "genrm", "sharded")
_WORKPLACE_EVENT = {
    "event_name": "NeMo RL checkpoint recovery sentinel",
    "participant_email": "checkpoint-recovery@example.com",
    "event_start": "2025-01-15 10:00:00",
    "duration": "30",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected an object in {path}")
    return value


def _participant(checkpoint: dict[str, Any], component: str) -> dict[str, Any]:
    matches = [
        item
        for item in checkpoint.get("participants", [])
        if item.get("participant", {}).get("component") == component
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {component!r} checkpoint participant, got {len(matches)}"
        )
    return matches[0]


def _participants(checkpoint: dict[str, Any], component: str) -> list[dict[str, Any]]:
    return [
        item
        for item in checkpoint.get("participants", [])
        if item.get("participant", {}).get("component") == component
    ]


def _validate_participant_manifest(snapshot: Path, participant: dict[str, Any]) -> Path:
    reference = participant["manifest"]
    path = (snapshot / reference["relative_path"]).resolve()
    path.relative_to(snapshot.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _digest(path)
    if actual != reference["manifest_digest"]:
        raise AssertionError(
            f"participant manifest digest mismatch for {path}: "
            f"expected={reference['manifest_digest']}, actual={actual}"
        )
    return path


def _read_artifact(snapshot: Path, reference: dict[str, Any]) -> list[dict[str, Any]]:
    path = (snapshot / reference["relative_path"]).resolve()
    path.relative_to(snapshot.resolve())
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise AssertionError(f"artifact digest mismatch for {path}")
    if len(payload) != reference["bytes"]:
        raise AssertionError(f"artifact byte count mismatch for {path}")
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if len(records) != reference["records"]:
        raise AssertionError(f"artifact record count mismatch for {path}")
    if not all(isinstance(record, dict) for record in records):
        raise TypeError(f"artifact rows must be objects in {path}")
    return records


def _read_agent_records(snapshot: Path, manifest_path: Path) -> list[dict[str, Any]]:
    """Validate and read the archive-only Gym agent checkpoint format."""
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 2:
        raise AssertionError(
            "functional recovery requires the archive-only Gym agent checkpoint schema"
        )
    indexed = _read_artifact(snapshot, manifest["record_index"])
    archives = manifest.get("archives")
    if not isinstance(archives, list):
        raise TypeError("agent checkpoint archives must be a list")
    if manifest.get("records") != len(indexed):
        raise AssertionError("agent checkpoint record count does not match its index")

    archive_names = [item.get("name") for item in archives if isinstance(item, dict)]
    indexed_archive_names = {item.get("archive") for item in indexed}
    if len(archive_names) != len(archives) or len(set(archive_names)) != len(
        archive_names
    ):
        raise AssertionError(
            "agent checkpoint manifest contains invalid or duplicate archives"
        )
    if set(archive_names) != indexed_archive_names:
        raise AssertionError(
            "agent checkpoint archive inventory does not match its index"
        )

    records: list[dict[str, Any]] = []
    for archive_reference in archives:
        archive_path = (manifest_path.parent / archive_reference["name"]).resolve()
        archive_path.relative_to(snapshot.resolve())
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        if archive_path.stat().st_size != archive_reference["bytes"]:
            raise AssertionError(
                f"agent archive byte count mismatch for {archive_path}"
            )
        if _digest(archive_path) != archive_reference["sha256"]:
            raise AssertionError(f"agent archive digest mismatch for {archive_path}")
        expected = [
            item for item in indexed if item["archive"] == archive_reference["name"]
        ]
        if len(expected) != archive_reference["members"]:
            raise AssertionError(
                f"agent archive member count mismatch for {archive_path}"
            )
        try:
            with tarfile.open(archive_path, mode="r:") as archive:
                infos = archive.getmembers()
                if [info.name for info in infos] != [
                    item["member"] for item in expected
                ]:
                    raise AssertionError(
                        f"agent archive inventory mismatch for {archive_path}"
                    )
                for info, member in zip(infos, expected, strict=True):
                    if not info.isfile():
                        raise AssertionError(
                            f"agent archive member is not a regular file: {info.name}"
                        )
                    extracted = archive.extractfile(info)
                    if extracted is None:
                        raise AssertionError(
                            f"agent archive member cannot be read: {info.name}"
                        )
                    payload = extracted.read()
                    if (
                        len(payload) != member["bytes"]
                        or hashlib.sha256(payload).hexdigest() != member["sha256"]
                    ):
                        raise AssertionError(
                            f"agent archive member is corrupted: {info.name}"
                        )
                    record = json.loads(payload)
                    if not isinstance(record, dict):
                        raise TypeError(
                            f"agent archive member must be an object: {info.name}"
                        )
                    if (record.get("rollout_id"), record.get("attempt_index")) != (
                        member["rollout_id"],
                        member["attempt_index"],
                    ):
                        raise AssertionError(
                            f"agent archive member identity mismatch: {info.name}"
                        )
                    records.append(record)
        except tarfile.TarError as error:
            raise AssertionError(
                f"agent archive cannot be read: {archive_path}"
            ) from error
    return records


def _matching_recovery_attempt(
    recovery: dict[str, Any], rollout_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for group in recovery.get("groups", []):
        for sibling in group.get("siblings", []):
            logical_rollout_id = f"{group['group_id']}_g{sibling['generation_index']}"
            if logical_rollout_id == rollout_id:
                matches.append((group, sibling["attempts"][-1]))
    if len(matches) != 1:
        raise AssertionError(
            f"Gym boundary {rollout_id!r} did not map to exactly one RL sibling"
        )
    return matches[0]


def _inspect_genrm_snapshot(
    snapshot: Path,
    dataset_rows: list[dict[str, Any]],
    gym_checkpoint: dict[str, Any],
    model: dict[str, Any],
    agent: dict[str, Any],
    agent_manifest_path: Path,
) -> dict[str, Any]:
    """Select a complete two-sibling GenRM cohort parked before verification."""
    import torch

    if model["payload"]["rows"] < 2:
        raise AssertionError(
            "GenRM checkpoint has fewer than two committed policy calls"
        )
    if agent["payload"]["records"] < 2:
        raise AssertionError("GenRM checkpoint has fewer than two parked siblings")

    continuation_rows = _read_artifact(
        snapshot,
        agent["payload"]["continuation_index"],
    )
    storage_reference_rows = _read_artifact(
        snapshot,
        model["payload"]["storage_reference_index"],
    )
    agent_records = _read_agent_records(snapshot, agent_manifest_path)
    recovery = torch.load(snapshot / "rollout_recovery.pt", weights_only=True)
    replay = torch.load(snapshot / "replay_buffer_metadata.pt", weights_only=False)
    if not isinstance(recovery, dict) or not isinstance(replay, dict):
        raise TypeError("GenRM rollout checkpoint sidecars must be mappings")
    if recovery.get("pending_completed_execution_acknowledgements"):
        raise AssertionError(
            "completed Gym executions were not acknowledged before checkpoint commit"
        )

    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in agent_records:
        if (
            record.get("boundary_kind") != "turn_complete"
            or record.get("boundary_index", 0) < 1
            or not record.get("last_committed_model_call_id")
            or record.get("pending_model") is not None
        ):
            continue
        group, attempt = _matching_recovery_attempt(recovery, record["rollout_id"])
        if attempt["attempt_index"] != record["attempt_index"]:
            raise AssertionError(
                "GenRM agent boundary and RL ledger disagree about the attempt"
            )
        if attempt["status"] != "dispatched":
            raise AssertionError("a parked GenRM sibling must remain dispatched")
        candidates.append((record, group))

    group_ids = {group["group_id"] for _, group in candidates}
    if len(group_ids) != 1:
        raise AssertionError(
            "GenRM checkpoint must contain one fully parked sibling cohort"
        )
    group = candidates[0][1]
    expected_rollout_ids = {
        f"{group['group_id']}_g{sibling['generation_index']}"
        for sibling in group["siblings"]
    }
    selected_records = {
        record["rollout_id"]: record
        for record, candidate_group in candidates
        if candidate_group["group_id"] == group["group_id"]
    }
    if len(expected_rollout_ids) != 2 or set(selected_records) != expected_rollout_ids:
        raise AssertionError(
            "GenRM checkpoint must park both siblings from one two-member cohort"
        )

    continuation_capture_keys = {row["capture_key"] for row in continuation_rows}
    expected_capture_keys = {
        (
            rollout_id
            if record["attempt_index"] == 0
            else f"{rollout_id}-a{record['attempt_index']}"
        )
        for rollout_id, record in selected_records.items()
    }
    if not expected_capture_keys.issubset(continuation_capture_keys):
        raise AssertionError("GenRM checkpoint is missing a sibling continuation root")
    storage_capture_keys = {row["capture_key"] for row in storage_reference_rows}
    if not storage_capture_keys.issubset(continuation_capture_keys):
        raise AssertionError(
            "GenRM storage references contain a rollout without a continuation"
        )

    try:
        prompt_index = int(group["prompt_ref"]["sample_id"])
        dataset_rows[prompt_index]
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise AssertionError(
            "selected GenRM cohort does not resolve to its dataset row"
        ) from error

    return {
        "snapshot_path": str(snapshot.resolve()),
        "checkpoint_id": gym_checkpoint["checkpoint_id"],
        "profile": "genrm",
        "group_id": group["group_id"],
        "prompt_index": prompt_index,
        "completed_group_ids": sorted(item["group_id"] for item in replay["groups"]),
        "rollouts": [
            {
                "rollout_id": rollout_id,
                "source_attempt_index": record["attempt_index"],
                "restored_attempt_index": record["attempt_index"] + 1,
                "boundary_index": record["boundary_index"],
                "last_committed_model_call_id": record["last_committed_model_call_id"],
            }
            for rollout_id, record in sorted(selected_records.items())
        ],
    }


def _checkpoint_shard_name(snapshot: Path, manifest_path: Path) -> str:
    relative = manifest_path.relative_to(snapshot)
    if len(relative.parts) < 3 or relative.parts[0] != "gym-shards":
        raise AssertionError(
            f"shard-local Gym artifact is not under gym-shards/<name>: {relative}"
        )
    return relative.parts[1]


def _inspect_sharded_snapshot(
    snapshot: Path,
    dataset_rows: list[dict[str, Any]],
    gym_checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Validate one two-shard cut and select one continuation per shard."""
    import torch

    models = _participants(gym_checkpoint, "responses_api_models")
    agents = _participants(gym_checkpoint, "responses_api_agents")
    resources = _participants(gym_checkpoint, "resources_servers")
    if len(models) != 1 or len(agents) != 2 or len(resources) != 2:
        raise AssertionError(
            "sharded recovery requires one shared model plus two shard-local "
            f"agent/resource participants; got models={len(models)}, "
            f"agents={len(agents)}, resources={len(resources)}"
        )

    model = models[0]
    _validate_participant_manifest(snapshot, model)
    agent_manifests = {
        _checkpoint_shard_name(
            snapshot,
            manifest_path := _validate_participant_manifest(snapshot, participant),
        ): (participant, manifest_path)
        for participant in agents
    }
    resource_manifests = {
        _checkpoint_shard_name(
            snapshot,
            manifest_path := _validate_participant_manifest(snapshot, participant),
        ): (participant, manifest_path)
        for participant in resources
    }
    if set(agent_manifests) != set(resource_manifests) or len(agent_manifests) != 2:
        raise AssertionError(
            "agent and resource checkpoint artifacts do not cover the same two "
            f"shards: agents={sorted(agent_manifests)!r}, "
            f"resources={sorted(resource_manifests)!r}"
        )

    continuation_rows: list[dict[str, Any]] = []
    agent_records_by_shard: dict[str, list[dict[str, Any]]] = {}
    for shard_name, (agent, manifest_path) in agent_manifests.items():
        if agent["payload"]["records"] < 1:
            raise AssertionError(f"Gym shard {shard_name!r} has no parked boundary")
        continuation_rows.extend(
            _read_artifact(snapshot, agent["payload"]["continuation_index"])
        )
        agent_records_by_shard[shard_name] = _read_agent_records(
            snapshot, manifest_path
        )
    if model["payload"]["rows"] < 2:
        raise AssertionError(
            "shared Gym model ledger has fewer than two committed turns"
        )
    storage_reference_rows = _read_artifact(
        snapshot,
        model["payload"]["storage_reference_index"],
    )
    continuation_capture_keys = {row["capture_key"] for row in continuation_rows}
    storage_capture_keys = {row["capture_key"] for row in storage_reference_rows}
    if not storage_capture_keys.issubset(continuation_capture_keys):
        raise AssertionError(
            "shared model storage references contain a rollout without a "
            "shard-local parked continuation"
        )

    recovery = torch.load(snapshot / "rollout_recovery.pt", weights_only=True)
    replay = torch.load(snapshot / "replay_buffer_metadata.pt", weights_only=False)
    if not isinstance(recovery, dict):
        raise TypeError("rollout recovery sidecar is not a mapping")
    if not isinstance(replay, dict) or not replay.get("groups"):
        raise AssertionError(
            "sharded snapshot has no completed canonical group alongside its "
            "unfinished continuations"
        )
    if recovery.get("pending_completed_execution_acknowledgements"):
        raise AssertionError(
            "completed Gym executions were not acknowledged before checkpoint commit"
        )

    resource_by_name = {
        participant["participant"]["participant_name"]: (
            shard_name,
            participant,
            path,
        )
        for shard_name, (participant, path) in resource_manifests.items()
    }
    selected_rollouts: list[dict[str, Any]] = []
    selected_capture_keys: set[str] = set()
    for shard_name, records in sorted(agent_records_by_shard.items()):
        candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for record in records:
            pending_model = record.get("pending_model")
            if (
                record.get("boundary_index", 0) < 1
                or not record.get("last_committed_model_call_id")
                or not record.get("resource_state_revisions")
                or not isinstance(pending_model, dict)
                or pending_model.get("pending_action_cursor", 0) < 1
            ):
                continue
            group, attempt = _matching_recovery_attempt(recovery, record["rollout_id"])
            if attempt["attempt_index"] != record["attempt_index"]:
                raise AssertionError(
                    "Gym boundary and RL ledger disagree about the physical attempt"
                )
            if attempt["status"] != "dispatched":
                raise AssertionError(
                    "a parked, unfinished Gym execution must remain dispatched in RL"
                )
            candidates.append((record, group))
        if len(candidates) != 1:
            raise AssertionError(
                f"expected one recoverable continuation on shard {shard_name!r}, "
                f"got {len(candidates)}"
            )
        boundary, group = candidates[0]

        try:
            prompt_index = int(group["prompt_ref"]["sample_id"])
            dataset_row = dataset_rows[prompt_index]
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise AssertionError(
                "selected sharded recovery group does not resolve to its dataset row"
            ) from error

        revisions = boundary["resource_state_revisions"]
        matching_resources = [
            resource_by_name[name] for name in revisions if name in resource_by_name
        ]
        if len(matching_resources) != 1:
            raise AssertionError(
                "shard continuation does not resolve to exactly one checkpointed "
                f"resource: shard={shard_name!r}, revisions={revisions!r}"
            )
        resource_shard, resource, resource_manifest_path = matching_resources[0]
        if resource_shard != shard_name:
            raise AssertionError(
                "agent continuation and resource state were committed under "
                f"different shards: agent={shard_name!r}, resource={resource_shard!r}"
            )
        if resource["payload"]["sessions"] < 1:
            raise AssertionError(f"Gym shard {shard_name!r} saved no environment")

        resource_manifest = _read_json(resource_manifest_path)
        resource_snapshots = []
        for name, expected_digest in resource_manifest.get("files", {}).items():
            resource_path = resource_manifest_path.parent / name
            if _digest(resource_path) != expected_digest:
                raise AssertionError(
                    f"resources state digest mismatch for {resource_path}"
                )
            resource_record = _read_json(resource_path)
            if (
                resource_record.get("rollout_id") == boundary["rollout_id"]
                and resource_record.get("attempt_index") == boundary["attempt_index"]
            ):
                resource_snapshots.append(resource_record)
        if len(resource_snapshots) != 1:
            raise AssertionError(
                f"shard {shard_name!r} continuation did not map to exactly one "
                "resources snapshot"
            )
        resource_snapshot = resource_snapshots[0]
        resource_name = resource["participant"]["participant_name"]
        expected_revision = revisions.get(resource_name)
        if resource_snapshot.get("state_revision") != expected_revision:
            raise AssertionError(
                "agent boundary and resources snapshot disagree about state revision"
            )
        if not isinstance(expected_revision, int) or expected_revision < 2:
            raise AssertionError(
                "checkpointed resource state has no committed mutation"
            )

        task_source = group.get("task_source")
        state = resource_snapshot.get("state") or {}
        rollout = {
            "shard": shard_name,
            "task_source": task_source,
            "rollout_id": boundary["rollout_id"],
            "source_attempt_index": boundary["attempt_index"],
            "restored_attempt_index": boundary["attempt_index"] + 1,
            "boundary_index": boundary["boundary_index"],
            "last_committed_model_call_id": boundary["last_committed_model_call_id"],
            "resource_state_revisions": revisions,
            "group_id": group["group_id"],
            "prompt_index": prompt_index,
        }
        if task_source == "example_session_state_mgmt_simple_agent":
            initial_count = dataset_row.get("initial_count")
            expected_count = dataset_row.get("expected_count")
            checkpoint_counter = state.get("counter")
            if (
                isinstance(initial_count, bool)
                or not isinstance(initial_count, int)
                or isinstance(expected_count, bool)
                or not isinstance(expected_count, int)
                or isinstance(checkpoint_counter, bool)
                or not isinstance(checkpoint_counter, int)
                or not initial_count < checkpoint_counter <= expected_count
            ):
                raise AssertionError("sharded counter state is not recoverable")
            rollout.update(
                initial_count=initial_count,
                checkpoint_counter=checkpoint_counter,
                expected_count=expected_count,
            )
        elif task_source == "workplace_assistant_checkpoint_test_agent":
            sentinel_count = _workplace_sentinel_count(state)
            if sentinel_count != 1:
                raise AssertionError(
                    "sharded Workplace checkpoint must contain exactly one "
                    f"sentinel event, got {sentinel_count}"
                )
            rollout["checkpoint_sentinel_count"] = sentinel_count
        else:
            raise AssertionError(
                f"unexpected task source in sharded recovery cut: {task_source!r}"
            )

        capture_key = (
            boundary["rollout_id"]
            if boundary["attempt_index"] == 0
            else f"{boundary['rollout_id']}-a{boundary['attempt_index']}"
        )
        selected_capture_keys.add(capture_key)
        selected_rollouts.append(rollout)

    expected_task_sources = {
        "example_session_state_mgmt_simple_agent",
        "workplace_assistant_checkpoint_test_agent",
    }
    if {
        rollout["task_source"] for rollout in selected_rollouts
    } != expected_task_sources:
        raise AssertionError(
            "sharded checkpoint did not capture one continuation from each test "
            f"agent: rollouts={selected_rollouts!r}"
        )
    if not selected_capture_keys.issubset(continuation_capture_keys):
        raise AssertionError(
            "sharded agent continuation roots are missing from their indexes"
        )

    return {
        "snapshot_path": str(snapshot.resolve()),
        "checkpoint_id": gym_checkpoint["checkpoint_id"],
        "profile": "sharded",
        "shards": sorted(agent_manifests),
        "completed_group_ids": sorted(item["group_id"] for item in replay["groups"]),
        "rollouts": sorted(selected_rollouts, key=lambda item: item["shard"]),
    }


def inspect_snapshot(
    snapshot: Path,
    dataset_rows: list[dict[str, Any]],
    *,
    profile: str = "counter",
) -> dict[str, Any]:
    """Validate one published cross-system snapshot and select a continuation."""
    import torch

    manifest = _read_json(snapshot / "manifest.json")
    if manifest.get("base_train_step") != 0:
        raise AssertionError("the fault-injection cut must use the bootstrap anchor")
    gym_checkpoint = manifest.get("gym_checkpoint")
    if not isinstance(gym_checkpoint, dict):
        raise AssertionError("snapshot has no Gym participant checkpoint")
    if not isinstance(manifest.get("gym_topology_fingerprint"), str):
        raise AssertionError("snapshot has no Gym topology fingerprint")

    for required in (
        snapshot / "data_plane",
        snapshot / "train_dataloader.pt",
        snapshot / "replay_buffer_metadata.pt",
        snapshot / "rollout_recovery.pt",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    if profile == "sharded":
        return _inspect_sharded_snapshot(snapshot, dataset_rows, gym_checkpoint)

    model = _participant(gym_checkpoint, "responses_api_models")
    agent = _participant(gym_checkpoint, "responses_api_agents")
    _validate_participant_manifest(snapshot, model)
    agent_manifest_path = _validate_participant_manifest(snapshot, agent)
    if profile == "genrm":
        return _inspect_genrm_snapshot(
            snapshot,
            dataset_rows,
            gym_checkpoint,
            model,
            agent,
            agent_manifest_path,
        )

    resources = _participant(gym_checkpoint, "resources_servers")
    resources_manifest_path = _validate_participant_manifest(snapshot, resources)

    if model["payload"]["rows"] < 1:
        raise AssertionError("Gym model ledger has no committed turn")
    if agent["payload"]["records"] < 1:
        raise AssertionError("Gym agent has no parked turn boundary")
    if resources["payload"]["sessions"] < 1:
        raise AssertionError("Gym resources participant has no saved environment")

    continuation_rows = _read_artifact(
        snapshot,
        agent["payload"]["continuation_index"],
    )
    storage_reference_rows = _read_artifact(
        snapshot,
        model["payload"]["storage_reference_index"],
    )
    if not continuation_rows:
        raise AssertionError("Gym checkpoint has no active continuation roots")
    if not storage_reference_rows:
        raise AssertionError("Gym checkpoint has no referenced TQ staging rows")
    continuation_capture_keys = {row["capture_key"] for row in continuation_rows}
    storage_capture_keys = {row["capture_key"] for row in storage_reference_rows}
    if not storage_capture_keys.issubset(continuation_capture_keys):
        raise AssertionError(
            "Gym storage references contain a rollout with no parked continuation"
        )

    agent_records = _read_agent_records(snapshot, agent_manifest_path)
    recovery = torch.load(snapshot / "rollout_recovery.pt", weights_only=True)
    if not isinstance(recovery, dict):
        raise TypeError("rollout recovery sidecar is not a mapping")
    replay = torch.load(snapshot / "replay_buffer_metadata.pt", weights_only=False)
    if not isinstance(replay, dict) or not replay.get("groups"):
        raise AssertionError(
            "snapshot has no completed canonical group alongside the unfinished "
            "Gym continuation"
        )
    if recovery.get("pending_completed_execution_acknowledgements"):
        raise AssertionError(
            "completed Gym executions were not acknowledged before checkpoint commit"
        )

    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in agent_records:
        pending_model = record.get("pending_model")
        if (
            record.get("boundary_index", 0) < 1
            or not record.get("last_committed_model_call_id")
            or not record.get("resource_state_revisions")
            or not isinstance(pending_model, dict)
            or pending_model.get("pending_action_cursor", 0) < 1
        ):
            continue
        group, attempt = _matching_recovery_attempt(recovery, record["rollout_id"])
        if attempt["attempt_index"] != record["attempt_index"]:
            raise AssertionError(
                "Gym boundary and RL ledger disagree about the physical attempt"
            )
        if attempt["status"] != "dispatched":
            raise AssertionError(
                "a parked, unfinished Gym execution must remain dispatched in RL"
            )
        candidates.append((record, group))

    if not candidates:
        raise AssertionError(
            "snapshot has no post-mutation Gym turn boundary tied to an unfinished "
            "RL sibling"
        )
    boundary, group = sorted(
        candidates,
        key=lambda item: (item[0]["rollout_id"], item[0]["attempt_index"]),
    )[0]

    try:
        prompt_index = int(group["prompt_ref"]["sample_id"])
        dataset_row = dataset_rows[prompt_index]
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise AssertionError(
            "selected recovery group does not resolve to its deterministic "
            f"{profile} dataset row"
        ) from error
    if profile == "counter":
        initial_count = dataset_row.get("initial_count")
        expected_count = dataset_row.get("expected_count")
        if (
            isinstance(initial_count, bool)
            or not isinstance(initial_count, int)
            or isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count <= initial_count
        ):
            raise AssertionError("counter test row has invalid initial/expected values")

    resources_manifest = _read_json(resources_manifest_path)
    resource_snapshots: list[dict[str, Any]] = []
    for name, expected_digest in resources_manifest.get("files", {}).items():
        resource_path = resources_manifest_path.parent / name
        if _digest(resource_path) != expected_digest:
            raise AssertionError(f"resources state digest mismatch for {resource_path}")
        resource_record = _read_json(resource_path)
        if (
            resource_record.get("rollout_id") == boundary["rollout_id"]
            and resource_record.get("attempt_index") == boundary["attempt_index"]
        ):
            resource_snapshots.append(resource_record)
    if len(resource_snapshots) != 1:
        raise AssertionError(
            "selected Gym continuation did not map to exactly one resources snapshot"
        )
    resource_snapshot = resource_snapshots[0]
    resource_name = resources["participant"]["participant_name"]
    expected_revision = boundary["resource_state_revisions"].get(resource_name)
    if resource_snapshot.get("state_revision") != expected_revision:
        raise AssertionError(
            "agent boundary and resources snapshot disagree about state revision"
        )
    if not isinstance(expected_revision, int) or expected_revision < 2:
        raise AssertionError("checkpointed resource state has no committed mutation")

    selected = {
        "snapshot_path": str(snapshot.resolve()),
        "checkpoint_id": gym_checkpoint["checkpoint_id"],
        "profile": profile,
        "rollout_id": boundary["rollout_id"],
        "source_attempt_index": boundary["attempt_index"],
        "restored_attempt_index": boundary["attempt_index"] + 1,
        "boundary_index": boundary["boundary_index"],
        "last_committed_model_call_id": boundary["last_committed_model_call_id"],
        "resource_state_revisions": boundary["resource_state_revisions"],
        "group_id": group["group_id"],
        "prompt_index": prompt_index,
        "completed_group_ids": sorted(item["group_id"] for item in replay["groups"]),
    }
    state = resource_snapshot.get("state") or {}
    if profile == "counter":
        checkpoint_counter = state.get("counter")
        if (
            isinstance(checkpoint_counter, bool)
            or not isinstance(checkpoint_counter, int)
            or not initial_count < checkpoint_counter <= expected_count
        ):
            raise AssertionError(
                "checkpoint must contain a counter mutation strictly after the "
                "initial state and no later than the expected terminal state"
            )
        selected.update(
            initial_count=initial_count,
            checkpoint_counter=checkpoint_counter,
            expected_count=expected_count,
        )
    else:
        sentinel_count = _workplace_sentinel_count(state)
        if sentinel_count != 1:
            raise AssertionError(
                "Workplace checkpoint must contain exactly one sentinel calendar "
                f"event, got {sentinel_count}"
            )
        selected["checkpoint_sentinel_count"] = sentinel_count
    return selected


def _workplace_sentinel_count(state: dict[str, Any]) -> int:
    try:
        payload = state["containers"]["calendar"]["_calendar_events"]
        frame = json.loads(payload)
        columns = frame["columns"]
        rows = frame["data"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise AssertionError(
            "Workplace checkpoint has no serialized calendar state"
        ) from error
    return sum(
        all(
            str(row[columns.index(field)]) == value
            for field, value in _WORKPLACE_EVENT.items()
        )
        for row in rows
    )


def _published_bootstrap_snapshots(checkpoint_dir: Path) -> list[Path]:
    root = checkpoint_dir / "bootstrap" / "rollout_snapshots"
    if not root.is_dir():
        return []
    return sorted(root.glob("snapshot_[0-9]*"), reverse=True)


def select_snapshot(args: argparse.Namespace) -> None:
    dataset_rows = [
        json.loads(line)
        for line in args.dataset.read_text().splitlines()
        if line.strip()
    ]
    if not dataset_rows or not all(isinstance(row, dict) for row in dataset_rows):
        raise TypeError("counter dataset must contain JSON objects")
    deadline = time.monotonic() + args.timeout_s
    last_error = "no published bootstrap snapshot"
    while time.monotonic() < deadline:
        for snapshot in _published_bootstrap_snapshots(args.checkpoint_dir):
            try:
                selected = inspect_snapshot(
                    snapshot,
                    dataset_rows,
                    profile=args.profile,
                )
            except (
                AssertionError,
                FileNotFoundError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                last_error = f"{snapshot}: {type(error).__name__}: {error}"
                print(f"snapshot candidate rejected: {last_error}", flush=True)
                continue
            args.selection.parent.mkdir(parents=True, exist_ok=True)
            args.selection.write_text(
                json.dumps(selected, sort_keys=True, indent=2) + "\n"
            )
            return

        try:
            os.kill(args.pid, 0)
        except ProcessLookupError as error:
            tail = ""
            if args.run_log.is_file():
                tail = "\n".join(
                    args.run_log.read_text(errors="replace").splitlines()[-60:]
                )
            raise RuntimeError(
                "phase one exited before publishing a coordinated turn "
                f"checkpoint; last candidate: {last_error}\n{tail}"
            ) from error
        time.sleep(0.1)
    raise TimeoutError(
        "no coordinated Gym/TQ bootstrap snapshot was published before the "
        f"deadline; last candidate: {last_error}"
    )


def verify_restore(args: argparse.Namespace) -> None:
    selected = _read_json(args.selection)
    events = [
        json.loads(line)
        for line in args.events.read_text().splitlines()
        if line.strip()
    ]
    if args.profile == "genrm":
        _verify_genrm_restore(selected, events, args.audit_events)
        return
    if args.profile == "sharded":
        _verify_sharded_restore(selected, events, args.audit_events)
        return
    expected_capture_key = (
        f"{selected['rollout_id']}-a{selected['restored_attempt_index']}"
    )
    matching_dispatches = [
        event
        for event in events
        if event.get("event") == "dispatch"
        and expected_capture_key in event.get("rollout_ids", [])
    ]
    if len(matching_dispatches) != 1:
        raise AssertionError(
            "restored Gym boundary was not redispatched exactly once with its "
            f"next attempt identity: expected={expected_capture_key!r}, "
            f"matches={matching_dispatches!r}"
        )
    stale_dispatches = [
        event
        for event in events
        if event.get("event") == "dispatch"
        and selected["rollout_id"] in event.get("rollout_ids", [])
    ]
    if stale_dispatches:
        raise AssertionError(
            "restore reused the tombstoned source attempt instead of incrementing it"
        )
    matching_completions = [
        event
        for event in events
        if event.get("event") == "completion_forwarded"
        and event.get("rollout_id") == expected_capture_key
    ]
    if len(matching_completions) != 1:
        raise AssertionError(
            "restored counter continuation did not complete exactly once: "
            f"matches={matching_completions!r}"
        )
    completion = matching_completions[0]
    if completion.get("reward") != 1.0:
        raise AssertionError(
            "restored counter continuation received a failed verifier reward; "
            "the pre-checkpoint tool mutation may have been replayed: "
            f"completion={completion!r}"
        )
    regenerated_completed_groups = [
        event
        for event in events
        if event.get("event") == "dispatch"
        and event.get("group_id") in selected["completed_group_ids"]
    ]
    if regenerated_completed_groups:
        raise AssertionError(
            "a completed, checkpointed group was regenerated after restore: "
            f"events={regenerated_completed_groups!r}"
        )
    if args.profile == "workplace":
        _verify_workplace_audit(selected, args.audit_events)


def _verify_sharded_restore(
    selected: dict[str, Any],
    events: list[dict[str, Any]],
    audit_path: Path | None,
) -> None:
    if len(selected["rollouts"]) != 2:
        raise AssertionError("sharded recovery selection must contain two rollouts")
    for rollout in selected["rollouts"]:
        expected_capture_key = (
            f"{rollout['rollout_id']}-a{rollout['restored_attempt_index']}"
        )
        dispatches = [
            event
            for event in events
            if event.get("event") == "dispatch"
            and expected_capture_key in event.get("rollout_ids", [])
        ]
        if len(dispatches) != 1:
            raise AssertionError(
                "restored shard continuation was not redispatched exactly once: "
                f"capture_key={expected_capture_key!r}, matches={dispatches!r}"
            )
        stale_dispatches = [
            event
            for event in events
            if event.get("event") == "dispatch"
            and rollout["rollout_id"] in event.get("rollout_ids", [])
        ]
        if stale_dispatches:
            raise AssertionError(
                "sharded restore reused a tombstoned source attempt: "
                f"events={stale_dispatches!r}"
            )
        completions = [
            event
            for event in events
            if event.get("event") == "completion_forwarded"
            and event.get("rollout_id") == expected_capture_key
        ]
        if len(completions) != 1 or completions[0].get("reward") != 1.0:
            raise AssertionError(
                "restored shard continuation did not complete exactly once with "
                f"reward 1: capture_key={expected_capture_key!r}, "
                f"events={completions!r}"
            )
        if rollout["task_source"] == "workplace_assistant_checkpoint_test_agent":
            _verify_workplace_audit(rollout, audit_path)

    regenerated_completed_groups = [
        event
        for event in events
        if event.get("event") == "dispatch"
        and event.get("group_id") in selected["completed_group_ids"]
    ]
    if regenerated_completed_groups:
        raise AssertionError(
            "a completed, checkpointed sharded group was regenerated after restore: "
            f"events={regenerated_completed_groups!r}"
        )


def _verify_genrm_restore(
    selected: dict[str, Any],
    events: list[dict[str, Any]],
    audit_path: Path | None,
) -> None:
    expected_capture_keys = {
        f"{rollout['rollout_id']}-a{rollout['restored_attempt_index']}"
        for rollout in selected["rollouts"]
    }
    matching_dispatches = [
        event
        for event in events
        if event.get("event") == "dispatch"
        and expected_capture_keys.issubset(set(event.get("rollout_ids", [])))
    ]
    if len(matching_dispatches) != 1:
        raise AssertionError(
            "restored GenRM cohort was not redispatched exactly once: "
            f"matches={matching_dispatches!r}"
        )
    stale_rollout_ids = {rollout["rollout_id"] for rollout in selected["rollouts"]}
    stale_dispatches = [
        event
        for event in events
        if event.get("event") == "dispatch"
        and stale_rollout_ids.intersection(event.get("rollout_ids", []))
    ]
    if stale_dispatches:
        raise AssertionError("GenRM recovery reused a tombstoned source attempt")

    for capture_key in expected_capture_keys:
        completions = [
            event
            for event in events
            if event.get("event") == "completion_forwarded"
            and event.get("rollout_id") == capture_key
        ]
        if len(completions) != 1 or completions[0].get("reward") != 1.0:
            raise AssertionError(
                "restored GenRM sibling did not complete exactly once with its "
                f"cohort reward: capture_key={capture_key!r}, events={completions!r}"
            )

    if audit_path is None or not audit_path.is_file():
        raise AssertionError("GenRM recovery produced no durable verifier audit")
    audit = [
        json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()
    ]

    expected_phase1_counts = {
        ("phase1", "verify_entered"): 1,
        ("phase1", "verify_waiting"): 1,
        ("phase1", "reward_computed"): 0,
        ("phase1", "verify_returned"): 0,
    }
    actual_phase1_counts = {
        key: sum(
            item.get("phase") == key[0] and item.get("event") == key[1]
            for item in audit
        )
        for key in expected_phase1_counts
    }
    if actual_phase1_counts != expected_phase1_counts:
        raise AssertionError(
            "GenRM cohort did not freeze exactly once before the crash: "
            f"expected={expected_phase1_counts!r}, "
            f"actual={actual_phase1_counts!r}, audit={audit!r}"
        )

    phase2 = [item for item in audit if item.get("phase") == "phase2"]
    entered = [item for item in phase2 if item.get("event") == "verify_entered"]
    returned = [item for item in phase2 if item.get("event") == "verify_returned"]

    def capture_id(item: dict[str, Any]) -> str:
        value = item.get("capture_rollout_id")
        if not isinstance(value, str) or not value:
            raise AssertionError(
                f"GenRM audit event has no capture rollout identity: event={item!r}"
            )
        return value

    entered_counts = Counter(capture_id(item) for item in entered)
    returned_counts = Counter(capture_id(item) for item in returned)
    if entered_counts != returned_counts or any(
        count != 1 for count in entered_counts.values()
    ):
        raise AssertionError(
            "a phase-two GenRM verification was duplicated or left unfinished: "
            f"entered={entered_counts!r}, returned={returned_counts!r}, audit={audit!r}"
        )

    cohort_size = len(expected_capture_keys)

    def cohort_members(item: dict[str, Any]) -> frozenset[str]:
        raw_members = item.get("capture_rollout_ids")
        if not isinstance(raw_members, list) or any(
            not isinstance(member, str) or not member for member in raw_members
        ):
            raise AssertionError(
                f"GenRM cohort audit event has no member identities: event={item!r}"
            )
        members = frozenset(raw_members)
        if len(members) != len(raw_members):
            raise AssertionError(
                f"GenRM cohort audit event repeats a member identity: event={item!r}"
            )
        return members

    reward_events = [item for item in phase2 if item.get("event") == "reward_computed"]
    reward_cohorts = [cohort_members(item) for item in reward_events]
    if any(
        len(members) != cohort_size or event.get("cohort_size") != cohort_size
        for members, event in zip(reward_cohorts, reward_events, strict=True)
    ):
        raise AssertionError(
            f"phase-two GenRM reward used an incomplete cohort: events={reward_events!r}"
        )
    if len(set(reward_cohorts)) != len(reward_cohorts):
        raise AssertionError(
            f"a phase-two GenRM cohort was rewarded more than once: events={reward_events!r}"
        )

    rewarded_counts = Counter(member for cohort in reward_cohorts for member in cohort)
    if rewarded_counts != entered_counts:
        raise AssertionError(
            "phase-two GenRM requests and rewarded cohort membership disagree: "
            f"entered={entered_counts!r}, rewarded={rewarded_counts!r}, audit={audit!r}"
        )

    waiting_events = [item for item in phase2 if item.get("event") == "verify_waiting"]
    waiting_cohorts = [cohort_members(item) for item in waiting_events]
    if any(
        len(members) != cohort_size - 1 or event.get("cohort_size") != cohort_size - 1
        for members, event in zip(waiting_cohorts, waiting_events, strict=True)
    ):
        raise AssertionError(
            f"phase-two GenRM wait has invalid cohort membership: events={waiting_events!r}"
        )
    for rewarded in reward_cohorts:
        matching_waits = [waiting for waiting in waiting_cohorts if waiting < rewarded]
        if len(matching_waits) != 1:
            raise AssertionError(
                "phase-two GenRM cohort did not wait and resolve exactly once: "
                f"rewarded={sorted(rewarded)!r}, waits={matching_waits!r}, audit={audit!r}"
            )

    selected_cohort = frozenset(expected_capture_keys)
    if selected_cohort not in reward_cohorts:
        raise AssertionError(
            "the selected restored GenRM siblings were not rewarded as one cohort: "
            f"selected={sorted(selected_cohort)!r}, rewarded={reward_cohorts!r}"
        )


def _verify_workplace_audit(
    selected: dict[str, Any],
    audit_path: Path | None,
) -> None:
    if audit_path is None or not audit_path.is_file():
        raise AssertionError("Workplace recovery produced no durable audit events")
    events = [
        json.loads(line) for line in audit_path.read_text().splitlines() if line.strip()
    ]
    rollout_id = selected["rollout_id"]
    source_attempt = selected["source_attempt_index"]
    restored_attempt = selected["restored_attempt_index"]
    mutations = [
        event
        for event in events
        if event.get("event") == "mutation_applied"
        and event.get("rollout_id") == rollout_id
    ]
    expected_mutation = [
        event
        for event in mutations
        if event.get("attempt_index") == source_attempt
        and event.get("sentinel_count") == 1
    ]
    if len(expected_mutation) != 1 or len(mutations) != 1:
        raise AssertionError(
            "Workplace sentinel mutation did not execute exactly once before the "
            f"crash: events={mutations!r}"
        )
    for event_name in ("state_restored", "state_verified"):
        matches = [
            event
            for event in events
            if event.get("event") == event_name
            and event.get("rollout_id") == rollout_id
            and event.get("attempt_index") == restored_attempt
            and event.get("sentinel_count") == 1
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"Workplace restored state was not observed exactly once at "
                f"{event_name}: events={matches!r}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("checkpoint_dir", type=Path)
    select.add_argument("selection", type=Path)
    select.add_argument("pid", type=int)
    select.add_argument("run_log", type=Path)
    select.add_argument("dataset", type=Path)
    select.add_argument("timeout_s", type=float)
    select.add_argument("--profile", choices=_PROFILES, default="counter")

    verify = subparsers.add_parser("verify-restore")
    verify.add_argument("selection", type=Path)
    verify.add_argument("events", type=Path)
    verify.add_argument("--profile", choices=_PROFILES, default="counter")
    verify.add_argument("--audit-events", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "select":
        select_snapshot(args)
    else:
        verify_restore(args)


if __name__ == "__main__":
    main()
