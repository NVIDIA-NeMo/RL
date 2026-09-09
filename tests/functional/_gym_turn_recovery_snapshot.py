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
import time
from pathlib import Path
from typing import Any

import torch


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


def inspect_snapshot(snapshot: Path) -> dict[str, Any]:
    """Validate one published cross-system snapshot and select a continuation."""
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

    model = _participant(gym_checkpoint, "responses_api_models")
    agent = _participant(gym_checkpoint, "responses_api_agents")
    resources = _participant(gym_checkpoint, "resources_servers")
    for participant in (model, agent, resources):
        _validate_participant_manifest(snapshot, participant)

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
    if model["payload"].get("excluded_inactive", 0) < 1:
        raise AssertionError(
            "Gym checkpoint did not exclude any completed/acknowledged lineage; "
            "the test requires one inactive rollout alongside the continuation"
        )
    continuation_capture_keys = {row["capture_key"] for row in continuation_rows}
    storage_capture_keys = {row["capture_key"] for row in storage_reference_rows}
    if not storage_capture_keys.issubset(continuation_capture_keys):
        raise AssertionError(
            "Gym storage references contain a rollout with no parked continuation"
        )

    agent_manifest_path = _validate_participant_manifest(snapshot, agent)
    agent_manifest = _read_json(agent_manifest_path)
    agent_dir = agent_manifest_path.parent
    recovery = torch.load(snapshot / "rollout_recovery.pt", weights_only=True)
    if not isinstance(recovery, dict):
        raise TypeError("rollout recovery sidecar is not a mapping")
    replay = torch.load(snapshot / "replay_buffer_metadata.pt", weights_only=True)
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
    for name, expected_digest in agent_manifest.get("files", {}).items():
        record_path = agent_dir / name
        if _digest(record_path) != expected_digest:
            raise AssertionError(f"agent boundary digest mismatch for {record_path}")
        record = _read_json(record_path)
        if (
            record.get("boundary_index", 0) < 1
            or not record.get("last_committed_model_call_id")
            or not record.get("resource_state_revisions")
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
            "snapshot has no Gym turn boundary tied to an unfinished RL sibling"
        )
    boundary, group = sorted(
        candidates,
        key=lambda item: (item[0]["rollout_id"], item[0]["attempt_index"]),
    )[0]
    return {
        "snapshot_path": str(snapshot.resolve()),
        "checkpoint_id": gym_checkpoint["checkpoint_id"],
        "rollout_id": boundary["rollout_id"],
        "source_attempt_index": boundary["attempt_index"],
        "restored_attempt_index": boundary["attempt_index"] + 1,
        "boundary_index": boundary["boundary_index"],
        "last_committed_model_call_id": boundary["last_committed_model_call_id"],
        "resource_state_revisions": boundary["resource_state_revisions"],
        "group_id": group["group_id"],
        "completed_group_ids": sorted(item["group_id"] for item in replay["groups"]),
    }


def _published_bootstrap_snapshots(checkpoint_dir: Path) -> list[Path]:
    root = checkpoint_dir / "bootstrap" / "rollout_snapshots"
    if not root.is_dir():
        return []
    return sorted(root.glob("snapshot_[0-9]*"), reverse=True)


def select_snapshot(args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.timeout_s
    last_error = "no published bootstrap snapshot"
    while time.monotonic() < deadline:
        for snapshot in _published_bootstrap_snapshots(args.checkpoint_dir):
            try:
                selected = inspect_snapshot(snapshot)
            except (
                AssertionError,
                FileNotFoundError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                last_error = f"{snapshot}: {type(error).__name__}: {error}"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("checkpoint_dir", type=Path)
    select.add_argument("selection", type=Path)
    select.add_argument("pid", type=int)
    select.add_argument("run_log", type=Path)
    select.add_argument("timeout_s", type=float)

    verify = subparsers.add_parser("verify-restore")
    verify.add_argument("selection", type=Path)
    verify.add_argument("events", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "select":
        select_snapshot(args)
    else:
        verify_restore(args)


if __name__ == "__main__":
    main()
