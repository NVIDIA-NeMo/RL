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

"""Select and verify the active-generation snapshot used by the prefix test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
import time
from pathlib import Path
from typing import Any


_PROFILES = ("basic", "workplace")
_WORKPLACE_EVENT = {
    "event_name": "NeMo RL checkpoint recovery sentinel",
    "participant_email": "checkpoint-recovery@example.com",
    "event_start": "2025-01-15 10:00:00",
    "duration": "30",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected an object in {path}")
    return value


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _participant_manifest(snapshot: Path, participant: dict[str, Any]) -> Path:
    reference = participant["manifest"]
    path = (snapshot / reference["relative_path"]).resolve()
    path.relative_to(snapshot.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    if _digest(path) != reference["manifest_digest"]:
        raise AssertionError(f"participant manifest digest mismatch for {path}")
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


def _worker_proofs(proof: dict[str, Any]) -> list[dict[str, Any]]:
    workers = proof.get("workers")
    if workers is None:
        return [proof]
    if not isinstance(workers, list) or not all(
        isinstance(worker, dict) for worker in workers
    ):
        raise TypeError("generation-cut coordinator proof has invalid workers")
    return workers


def _legacy_active_prefixes(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    prefixes: list[dict[str, Any]] = []
    proofs = manifest.get("gym_generation_cut_proofs", [])
    if not isinstance(proofs, list):
        raise TypeError("snapshot generation-cut proofs must be a list")
    for proof in proofs:
        if not isinstance(proof, dict):
            raise TypeError("snapshot generation-cut proof must be an object")
        for worker in _worker_proofs(proof):
            receipt = worker.get("generation_cut_receipt")
            if receipt is None:
                continue
            if not isinstance(receipt, dict):
                raise TypeError("generation-cut receipt must be an object")
            receipt_prefixes = receipt.get("prefixes", [])
            if not isinstance(receipt_prefixes, list):
                raise TypeError("generation-cut receipt prefixes must be a list")
            prefixes.extend(
                prefix
                for prefix in receipt_prefixes
                if isinstance(prefix, dict)
                and prefix.get("disposition") == "durable_prefix"
                and prefix.get("cut_kind") == "active_prefix"
                and isinstance(prefix.get("staging_keys"), list)
                and bool(prefix["staging_keys"])
                and isinstance(prefix.get("prefix_token_count"), int)
                and prefix["prefix_token_count"] > 0
            )
    return prefixes


def _lineage_active_prefixes(
    snapshot: Path,
    gym_checkpoint: dict[str, Any],
) -> list[dict[str, Any]]:
    model = _participant(gym_checkpoint, "responses_api_models")
    manifest_path = _participant_manifest(snapshot, model)
    ledger_manifest = _read_json(manifest_path)
    indexed = _read_artifact(snapshot, ledger_manifest["lineage_index"])
    rows_by_capture_key: dict[str, list[dict[str, Any]]] = {}
    for archive_reference in ledger_manifest.get("archives", []):
        archive_path = (manifest_path.parent / archive_reference["name"]).resolve()
        archive_path.relative_to(snapshot.resolve())
        if archive_path.stat().st_size != archive_reference["bytes"]:
            raise AssertionError(
                f"model lineage archive byte count mismatch for {archive_path}"
            )
        if _digest(archive_path) != archive_reference["sha256"]:
            raise AssertionError(
                f"model lineage archive digest mismatch for {archive_path}"
            )
        expected = {
            item["member"]: item
            for item in indexed
            if item["archive"] == archive_reference["name"]
        }
        with tarfile.open(archive_path, mode="r:") as archive:
            for info in archive.getmembers():
                member = expected.get(info.name)
                if member is None or not info.isfile():
                    raise AssertionError(
                        f"unexpected model lineage archive member: {archive_path}/{info.name}"
                    )
                extracted = archive.extractfile(info)
                if extracted is None:
                    raise AssertionError(
                        f"model lineage archive member cannot be read: {info.name}"
                    )
                payload = extracted.read()
                if (
                    len(payload) != member["bytes"]
                    or hashlib.sha256(payload).hexdigest() != member["sha256"]
                ):
                    raise AssertionError(
                        f"model lineage archive member is corrupted: {info.name}"
                    )
                rows_by_capture_key[member["capture_key"]] = [
                    json.loads(line) for line in payload.splitlines() if line.strip()
                ]

    prefixes: list[dict[str, Any]] = []
    for rows in rows_by_capture_key.values():
        committed_calls = {
            row.get("model_call_id")
            for row in rows
            if row.get("event") != "generation_cut"
            and row.get("failure_reason") is None
            and isinstance(row.get("staging_key"), str)
        }
        prefixes.extend(
            row
            for row in rows
            if row.get("event") == "generation_cut"
            and row.get("checkpoint_id") == gym_checkpoint["checkpoint_id"]
            and row.get("model_call_id") not in committed_calls
            and row.get("disposition") == "durable_prefix"
            and row.get("cut_kind") == "active_prefix"
            and isinstance(row.get("staging_keys"), list)
            and bool(row["staging_keys"])
            and isinstance(row.get("prefix_token_count"), int)
            and row["prefix_token_count"] > 0
        )
    return prefixes


def _active_prefixes(
    snapshot: Path,
    manifest: dict[str, Any],
    gym_checkpoint: dict[str, Any],
) -> list[dict[str, Any]]:
    legacy = _legacy_active_prefixes(manifest)
    if legacy:
        return legacy
    return _lineage_active_prefixes(snapshot, gym_checkpoint)


def _matching_attempt(
    recovery: dict[str, Any], rollout_id: str, attempt_index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for group in recovery.get("groups", []):
        for sibling in group.get("siblings", []):
            logical_id = f"{group['group_id']}_g{sibling['generation_index']}"
            if logical_id != rollout_id:
                continue
            attempts = [
                attempt
                for attempt in sibling.get("attempts", [])
                if attempt.get("attempt_index") == attempt_index
            ]
            matches.extend((group, attempt) for attempt in attempts)
    if len(matches) != 1:
        raise AssertionError(
            "active generation cut did not map to exactly one RL rollout attempt"
        )
    return matches[0]


def _agent_records(snapshot: Path, participant: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_path = _participant_manifest(snapshot, participant)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 2:
        raise AssertionError(
            "prefix recovery requires the archive-only Gym agent checkpoint schema"
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
                    if (
                        record.get("rollout_id"),
                        record.get("attempt_index"),
                    ) != (member["rollout_id"], member["attempt_index"]):
                        raise AssertionError(
                            f"agent archive member identity mismatch: {info.name}"
                        )
                    records.append(record)
        except tarfile.TarError as error:
            raise AssertionError(
                f"agent archive cannot be read: {archive_path}"
            ) from error
    return records


def inspect_snapshot(
    snapshot: Path,
    *,
    max_generation_tokens: int,
    profile: str = "basic",
) -> dict[str, Any]:
    """Validate a published bootstrap snapshot containing a non-empty live cut."""
    import torch

    manifest = _read_json(snapshot / "manifest.json")
    if manifest.get("base_train_step") != 0:
        raise AssertionError("the prefix fault injection must use a bootstrap snapshot")
    gym_checkpoint = manifest.get("gym_checkpoint")
    if not isinstance(gym_checkpoint, dict):
        raise AssertionError("snapshot has no Gym participant checkpoint")
    for required in (
        snapshot / "data_plane",
        snapshot / "train_dataloader.pt",
        snapshot / "replay_buffer_metadata.pt",
        snapshot / "rollout_recovery.pt",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    prefixes = _active_prefixes(snapshot, manifest, gym_checkpoint)
    if not prefixes:
        raise AssertionError("snapshot has no non-empty active generation prefix")
    prefixes = [
        prefix
        for prefix in prefixes
        if prefix["prefix_token_count"] < max_generation_tokens
    ]
    if not prefixes:
        raise AssertionError(
            "active generation prefixes already exhaust the request output limit"
        )
    prefix = sorted(
        prefixes,
        key=lambda item: (
            -item["prefix_token_count"],
            item["rollout_id"],
            item["attempt_index"],
        ),
    )[0]

    agent = _participant(gym_checkpoint, "responses_api_agents")
    resources = _participant(gym_checkpoint, "resources_servers")
    model = _participant(gym_checkpoint, "responses_api_models")
    _participant_manifest(snapshot, model)
    resources_manifest_path = _participant_manifest(snapshot, resources)
    if agent.get("payload", {}).get("records", 0) < 1:
        raise AssertionError("Gym agent has no saved continuation boundary")
    if resources.get("payload", {}).get("sessions", 0) < 1:
        raise AssertionError("Gym resources participant has no saved environment")
    matching_boundaries = [
        record
        for record in _agent_records(snapshot, agent)
        if record.get("rollout_id") == prefix["rollout_id"]
        and record.get("attempt_index") == prefix["attempt_index"]
    ]
    if len(matching_boundaries) != 1:
        raise AssertionError(
            "active generation cut has no unique saved agent continuation boundary"
        )
    boundary = matching_boundaries[0]
    resource_revisions = boundary.get("resource_state_revisions")
    if not isinstance(resource_revisions, dict) or not resource_revisions:
        raise AssertionError("saved agent boundary has no resource state revision")

    resources_manifest = _read_json(resources_manifest_path)
    matching_resource_states: list[dict[str, Any]] = []
    for name, expected_digest in resources_manifest.get("files", {}).items():
        resource_path = resources_manifest_path.parent / name
        if _digest(resource_path) != expected_digest:
            raise AssertionError(f"resources state digest mismatch for {resource_path}")
        resource_state = _read_json(resource_path)
        if (
            resource_state.get("rollout_id") == prefix["rollout_id"]
            and resource_state.get("attempt_index") == prefix["attempt_index"]
        ):
            matching_resource_states.append(resource_state)
    if len(matching_resource_states) != 1:
        raise AssertionError(
            "active generation cut has no unique saved resources state"
        )
    resource_name = resources["participant"]["participant_name"]
    expected_resource_revision = resource_revisions.get(resource_name)
    resource_state = matching_resource_states[0]
    if (
        not isinstance(expected_resource_revision, int)
        or expected_resource_revision < 1
        or resource_state.get("state_revision") != expected_resource_revision
    ):
        raise AssertionError(
            "agent boundary and resources snapshot disagree about state revision"
        )

    recovery = torch.load(snapshot / "rollout_recovery.pt", weights_only=True)
    if not isinstance(recovery, dict):
        raise TypeError("rollout recovery sidecar is not a mapping")
    group, attempt = _matching_attempt(
        recovery,
        prefix["rollout_id"],
        prefix["attempt_index"],
    )
    if attempt.get("status") != "dispatched":
        raise AssertionError("a cut generation must remain dispatched in the RL ledger")

    selected = {
        "snapshot_path": str(snapshot.resolve()),
        "checkpoint_id": gym_checkpoint["checkpoint_id"],
        "rollout_id": prefix["rollout_id"],
        "source_attempt_index": prefix["attempt_index"],
        "restored_attempt_index": prefix["attempt_index"] + 1,
        "source_model_call_id": prefix["model_call_id"],
        "staging_keys": prefix["staging_keys"],
        "prefix_token_count": prefix["prefix_token_count"],
        "prefix_digest": prefix["prefix_digest"],
        "boundary_index": boundary["boundary_index"],
        "resource_state_revisions": resource_revisions,
        "group_id": group["group_id"],
        "profile": profile,
    }
    if profile == "workplace":
        if model.get("payload", {}).get("rows", 0) < 1:
            raise AssertionError(
                "combined Workplace checkpoint has no committed first model call"
            )
        if (
            boundary.get("boundary_kind") != "turn_complete"
            or boundary.get("boundary_index", 0) < 3
            or not boundary.get("last_committed_model_call_id")
            or boundary.get("pending_model") is not None
        ):
            raise AssertionError(
                "combined Workplace checkpoint is not parked after its completed first turn"
            )
        output_types = {
            item.get("type")
            for item in boundary.get("output_items", [])
            if isinstance(item, dict)
        }
        if not {"function_call", "function_call_output"}.issubset(output_types):
            raise AssertionError(
                "combined Workplace checkpoint is missing its tool-call transcript"
            )
        if prefix["model_call_id"] == boundary["last_committed_model_call_id"]:
            raise AssertionError(
                "generation prefix must belong to the model call after the committed turn"
            )
        if expected_resource_revision < 2:
            raise AssertionError(
                "combined Workplace checkpoint has no committed environment mutation"
            )
        sentinel_count = _workplace_sentinel_count(resource_state.get("state") or {})
        if sentinel_count != 1:
            raise AssertionError(
                "combined Workplace checkpoint must contain exactly one sentinel "
                f"calendar event, got {sentinel_count}"
            )
        selected.update(
            last_committed_model_call_id=boundary["last_committed_model_call_id"],
            checkpoint_sentinel_count=sentinel_count,
        )
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
    deadline = time.monotonic() + args.timeout_s
    last_error = "no published bootstrap snapshot"
    while time.monotonic() < deadline:
        newest_error = None
        for snapshot in _published_bootstrap_snapshots(args.checkpoint_dir):
            try:
                selected = inspect_snapshot(
                    snapshot,
                    max_generation_tokens=args.max_generation_tokens,
                    profile=args.profile,
                )
            except (
                AssertionError,
                FileNotFoundError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                if newest_error is None:
                    newest_error = f"{snapshot}: {type(error).__name__}: {error}"
                continue
            args.selection.parent.mkdir(parents=True, exist_ok=True)
            args.selection.write_text(
                json.dumps(selected, sort_keys=True, indent=2) + "\n"
            )
            print(
                "selected active generation prefix: "
                f"tokens={selected['prefix_token_count']} "
                f"rollout={selected['rollout_id']}",
                flush=True,
            )
            return
        if newest_error is not None:
            last_error = newest_error
        try:
            os.kill(args.pid, 0)
        except ProcessLookupError as error:
            tail = ""
            if args.run_log.is_file():
                tail = "\n".join(
                    args.run_log.read_text(errors="replace").splitlines()[-80:]
                )
            raise RuntimeError(
                "phase one exited before publishing an active-prefix snapshot; "
                f"last candidate: {last_error}\n{tail}"
            ) from error
        time.sleep(0.1)
    raise TimeoutError(
        "no active-prefix bootstrap snapshot was published before the deadline; "
        f"last candidate: {last_error}"
    )


def _capture_key(rollout_id: str, attempt_index: int) -> str:
    return rollout_id if attempt_index == 0 else f"{rollout_id}-a{attempt_index}"


def verify_restore(args: argparse.Namespace) -> None:
    selected = _read_json(args.selection)
    events = [
        json.loads(line)
        for line in args.events.read_text().splitlines()
        if line.strip()
    ]
    restored_key = _capture_key(
        selected["rollout_id"], selected["restored_attempt_index"]
    )
    source_key = _capture_key(selected["rollout_id"], selected["source_attempt_index"])
    source_dispatches = [
        event
        for event in events
        if event.get("event") == "dispatch"
        and source_key in event.get("rollout_ids", [])
    ]
    if source_dispatches:
        raise AssertionError(
            f"source rollout attempt {source_key!r} was redispatched after restore"
        )
    dispatches = [
        event
        for event in events
        if event.get("event") == "dispatch"
        and restored_key in event.get("rollout_ids", [])
    ]
    if len(dispatches) != 1:
        raise AssertionError(
            f"expected one replacement dispatch for {restored_key!r}, got {len(dispatches)}"
        )
    completions = [
        event
        for event in events
        if event.get("event") == "completion_forwarded"
        and event.get("rollout_id") == restored_key
    ]
    if len(completions) != 1:
        raise AssertionError(
            f"replacement rollout {restored_key!r} did not complete exactly once"
        )

    log = args.run_log.read_text(errors="replace")
    restored_pattern = re.compile(
        r"generation prefix restored: rollout_id=(\S+) model_call_id=(\S+) "
        r"source_model_call_id=(\S+) prefix_tokens=(\d+) prefix_digest=([0-9a-f]{64})"
    )
    restored = [
        match
        for match in restored_pattern.finditer(log)
        if match.group(1) == restored_key
        and match.group(3) == selected["source_model_call_id"]
    ]
    if len(restored) != 1:
        raise AssertionError(
            "replacement request did not fetch the selected prefix once"
        )
    if int(restored[0].group(4)) != selected["prefix_token_count"]:
        raise AssertionError("replacement request restored the wrong prefix length")
    if restored[0].group(5) != selected["prefix_digest"]:
        raise AssertionError("replacement request restored the wrong prefix digest")

    completed_pattern = re.compile(
        r"generation prefix completed: rollout_id=(\S+) model_call_id=(\S+) "
        r"source_model_call_id=(\S+) prefix_tokens=(\d+) tail_tokens=(\d+) "
        r"total_generation_tokens=(\d+)"
    )
    completed = [
        match
        for match in completed_pattern.finditer(log)
        if match.group(1) == restored_key
        and match.group(3) == selected["source_model_call_id"]
    ]
    if len(completed) != 1:
        raise AssertionError(
            "replacement request did not seal one combined terminal row"
        )
    prefix_tokens, tail_tokens, total_tokens = map(int, completed[0].group(4, 5, 6))
    if prefix_tokens != selected["prefix_token_count"] or tail_tokens <= 0:
        raise AssertionError("combined terminal row has an invalid prefix/tail split")
    if total_tokens != prefix_tokens + tail_tokens:
        raise AssertionError(
            "combined terminal row duplicated or omitted generation tokens"
        )
    if args.profile == "workplace":
        _verify_workplace_audit(selected, args.audit_events)


def _verify_workplace_audit(
    selected: dict[str, Any],
    audit_path: Path | None,
) -> None:
    if audit_path is None or not audit_path.is_file():
        raise AssertionError("Workplace prefix recovery produced no audit events")
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
    expected_mutations = [
        event
        for event in mutations
        if event.get("attempt_index") == source_attempt
        and event.get("sentinel_count") == 1
    ]
    if len(expected_mutations) != 1 or len(mutations) != 1:
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
                f"Workplace state was not observed exactly once at {event_name}: "
                f"events={matches!r}"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("checkpoint_dir", type=Path)
    select.add_argument("selection", type=Path)
    select.add_argument("pid", type=int)
    select.add_argument("run_log", type=Path)
    select.add_argument("timeout_s", type=float)
    select.add_argument("max_generation_tokens", type=int)
    select.add_argument("--profile", choices=_PROFILES, default="basic")

    verify = subparsers.add_parser("verify-restore")
    verify.add_argument("selection", type=Path)
    verify.add_argument("events", type=Path)
    verify.add_argument("run_log", type=Path)
    verify.add_argument("--profile", choices=_PROFILES, default="basic")
    verify.add_argument("--audit-events", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "select":
        select_snapshot(args)
    else:
        verify_restore(args)


if __name__ == "__main__":
    main()
