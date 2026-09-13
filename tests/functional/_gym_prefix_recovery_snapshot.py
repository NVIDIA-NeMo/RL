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
import time
from pathlib import Path
from typing import Any


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


def _worker_proofs(proof: dict[str, Any]) -> list[dict[str, Any]]:
    workers = proof.get("workers")
    if workers is None:
        return [proof]
    if not isinstance(workers, list) or not all(
        isinstance(worker, dict) for worker in workers
    ):
        raise TypeError("generation-cut coordinator proof has invalid workers")
    return workers


def _active_prefixes(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    prefixes: list[dict[str, Any]] = []
    proofs = manifest.get("gym_generation_cut_proofs")
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
                and str(prefix.get("frozen_buffer_id", "")).startswith("active/")
                and isinstance(prefix.get("prefix_token_count"), int)
                and prefix["prefix_token_count"] > 0
            )
    return prefixes


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
    records: list[dict[str, Any]] = []
    for name, expected_digest in manifest.get("files", {}).items():
        path = manifest_path.parent / name
        if _digest(path) != expected_digest:
            raise AssertionError(f"agent boundary digest mismatch for {path}")
        records.append(_read_json(path))
    return records


def inspect_snapshot(
    snapshot: Path,
    *,
    max_generation_tokens: int,
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

    prefixes = _active_prefixes(manifest)
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
    _participant_manifest(
        snapshot, _participant(gym_checkpoint, "responses_api_models")
    )
    _participant_manifest(snapshot, resources)
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

    return {
        "snapshot_path": str(snapshot.resolve()),
        "checkpoint_id": gym_checkpoint["checkpoint_id"],
        "rollout_id": prefix["rollout_id"],
        "source_attempt_index": prefix["attempt_index"],
        "restored_attempt_index": prefix["attempt_index"] + 1,
        "source_model_call_id": prefix["model_call_id"],
        "staging_key": prefix["staging_key"],
        "prefix_token_count": prefix["prefix_token_count"],
        "prefix_digest": prefix["prefix_digest"],
        "boundary_index": boundary["boundary_index"],
        "resource_state_revisions": boundary.get("resource_state_revisions", {}),
        "group_id": group["group_id"],
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
                selected = inspect_snapshot(
                    snapshot,
                    max_generation_tokens=args.max_generation_tokens,
                )
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
            print(
                "selected active generation prefix: "
                f"tokens={selected['prefix_token_count']} "
                f"rollout={selected['rollout_id']}",
                flush=True,
            )
            return
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
        r"source_model_call_id=(\S+) prefix_tokens=(\d+)"
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

    verify = subparsers.add_parser("verify-restore")
    verify.add_argument("selection", type=Path)
    verify.add_argument("events", type=Path)
    verify.add_argument("run_log", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "select":
        select_snapshot(args)
    else:
        verify_restore(args)


if __name__ == "__main__":
    main()
