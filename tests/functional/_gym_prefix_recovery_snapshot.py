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

"""Select and verify the Gym-v2 snapshot used by the prefix recovery test."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _ledger_episode_index(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    import torch

    state = torch.load(path, weights_only=True)
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for group in state["groups"]:
        for sibling in group["siblings"]:
            attempt = sibling["attempts"][-1]
            generation_index = sibling["generation_index"]
            rollout_id = (
                f"{group['group_id']}_g{generation_index}_a"
                f"{uuid.UUID(bytes=attempt['attempt_uuid']).hex}"
            )
            result[(rollout_id, attempt["gym_attempt"])] = {
                "group_id": group["group_id"],
                "generation_index": generation_index,
                "target_step": group["target_step"],
                "restore_level": group["restore_level"],
                "status": attempt["status"],
                "gym_instance_id": attempt["gym_instance_id"],
            }
    return result


def _participant_records(
    snapshot: Path, kind: str
) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    pattern = f"gym-instances/*/replica-*/gym/{kind}/*/manifest.json"
    for manifest_path in snapshot.glob(pattern):
        manifest = _read_json(manifest_path)
        if manifest.get("kind") != kind or manifest.get("record_count", 0) <= 0:
            continue
        records_path = manifest_path.parent / manifest["records_file"]
        for line in records_path.read_text().splitlines():
            if line:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError(
                        f"participant row must be an object in {records_path}"
                    )
                records.append((manifest_path, record))
    return records


def _episode_key(record: dict[str, Any]) -> tuple[str, int]:
    episode = record["episode_id"]
    return episode["rollout_id"], episode["attempt"]


def _calendar_sentinel_count(state: dict[str, Any], event_name: str) -> int:
    """Count the selected calendar event in a checkpointed Workplace state."""

    columns = state["calendar"]["_calendar_events"]["columns"]
    names = next(
        column["values"] for column in columns if column["name"] == "event_name"
    )
    return sum(str(name).lower() == event_name.lower() for name in names)


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _log_tail(path: Path, lines: int = 80) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def _recoverable_cut(record: dict[str, Any]) -> dict[str, Any] | None:
    for cut in record.get("generation_cuts", []):
        continuation = cut.get("continuation")
        if not isinstance(continuation, dict):
            continue
        token_count = continuation.get("prefix_token_count")
        output_limit = continuation.get("effective_output_limit")
        staging_keys = continuation.get("staging_keys")
        if (
            isinstance(token_count, int)
            and isinstance(output_limit, int)
            and 0 < token_count < output_limit
            and isinstance(staging_keys, list)
            and bool(staging_keys)
            and continuation.get("terminal_finish_reason") is None
            and continuation.get("terminal_stop_reason") is None
        ):
            return {
                "model_call_id": cut["model_call_id"],
                "request_digest": cut["request_digest"],
                **continuation,
            }
    return None


def _require_cut_coverage(
    candidates: list[dict[str, Any]], *, min_cuts: int, min_cut_groups: int
) -> None:
    """Reject a snapshot whose recoverable prefixes cover too little work.

    Gym refuses a dispatch that lands while a checkpoint holds admission, and
    the refused rollout starts later, so requiring every sibling to be mid-decode
    at once is a race. Coverage per prompt group still restores every group.
    """
    groups = {candidate["group_id"] for candidate in candidates}
    if len(candidates) < min_cuts or len(groups) < min_cut_groups:
        raise AssertionError(
            f"snapshot has {len(candidates)} recoverable prefixes in "
            f"{len(groups)} prompt groups, need {min_cuts} in {min_cut_groups}"
        )


def inspect_snapshot(
    snapshot: Path,
    sentinel_event: str | None,
    *,
    min_cuts: int = 1,
    min_cut_groups: int = 1,
) -> dict[str, Any]:
    """Return the post-mutation nonterminal prefixes in ``snapshot``.

    The top-level fields describe the longest cut; ``cuts`` lists every one,
    and the snapshot qualifies only if it holds at least ``min_cuts`` of them
    spread across at least ``min_cut_groups`` prompt groups. A ``None``
    ``sentinel_event`` selects a stateless resources server (e.g. CLEVR's
    string_match), whose episodes export no resources state to check.
    """

    required = [
        snapshot / "manifest.json",
        snapshot / "gym_checkpoint.json",
        snapshot / "rollout_recovery.pt",
        snapshot / "replay_buffer_metadata.pt",
    ]
    if (
        not all(path.is_file() for path in required)
        or not (snapshot / "data_plane").is_dir()
    ):
        raise FileNotFoundError("snapshot is not fully published")
    manifest = _read_json(required[0])
    if manifest.get("base_train_step") != 0 or manifest.get("trainer_version") != 0:
        raise AssertionError("prefix fault injection must use a bootstrap snapshot")

    gym = _read_json(required[1])
    ledger = _ledger_episode_index(required[2])
    agent_keys = {
        _episode_key(record) for _, record in _participant_records(snapshot, "agent")
    }
    resource_index = {
        _episode_key(record): (manifest_path, record["state"])
        for manifest_path, record in _participant_records(snapshot, "resources")
        if record.get("state")
    }
    model_records = _participant_records(snapshot, "model")

    candidates: list[dict[str, Any]] = []
    for model_manifest, record in model_records:
        key = _episode_key(record)
        recovery = ledger.get(key)
        cut = _recoverable_cut(record)
        if recovery is None or cut is None:
            continue
        resources_manifest = None
        if sentinel_event is not None:
            resource_entry = resource_index.get(key)
            if resource_entry is None:
                continue
            manifest_path, resources_state = resource_entry
            try:
                sentinel_count = _calendar_sentinel_count(
                    resources_state, sentinel_event
                )
            except (KeyError, StopIteration, TypeError):
                continue
            if sentinel_count != 1:
                continue
            resources_manifest = str(manifest_path.relative_to(snapshot))
        instance_id = recovery["gym_instance_id"]
        checkpointed_episodes = {
            (episode["rollout_id"], episode["attempt"])
            for episode in gym.get("instances", {}).get(instance_id, [])
        }
        retained_keys = set(gym.get("staging_keys", {}).get(instance_id, []))
        if (
            key not in checkpointed_episodes
            or key not in agent_keys
            or recovery["restore_level"] != "prefix"
            or recovery["status"] != "dispatched"
            or not set(cut["staging_keys"]).issubset(retained_keys)
        ):
            continue
        candidates.append(
            {
                "snapshot": str(snapshot),
                "snapshot_name": snapshot.name,
                "gym_instance_id": instance_id,
                "gym_rollout_id": key[0],
                "gym_attempt": key[1],
                "group_id": recovery["group_id"],
                "generation_index": recovery["generation_index"],
                "target_step": recovery["target_step"],
                "model_manifest": str(model_manifest.relative_to(snapshot)),
                "resources_manifest": resources_manifest,
                "sentinel_event": sentinel_event,
                **cut,
            }
        )
    if not candidates:
        raise AssertionError(
            "snapshot has no recoverable nonterminal active generation prefix"
        )
    _require_cut_coverage(candidates, min_cuts=min_cuts, min_cut_groups=min_cut_groups)
    cuts = sorted(
        candidates, key=lambda item: (item["group_id"], item["generation_index"])
    )
    longest = max(candidates, key=lambda item: item["prefix_token_count"])
    return {**longest, "cuts": cuts}


def select_snapshot(args: argparse.Namespace) -> None:
    root = Path(args.snapshot_root)
    selection_path = Path(args.selection_file)
    backup_path = Path(args.backup_path)
    phase_log = Path(args.phase_log)
    deadline = time.monotonic() + args.timeout_s
    last_error = "no snapshot observed"

    while time.monotonic() < deadline:
        for snapshot in sorted(root.glob("snapshot_*"), reverse=True):
            try:
                selection = inspect_snapshot(
                    snapshot,
                    None if args.stateless_resources else args.sentinel_event,
                    min_cuts=args.min_cuts,
                    min_cut_groups=args.min_cut_groups,
                )
            except (AssertionError, FileNotFoundError, json.JSONDecodeError) as error:
                last_error = f"{snapshot}: {error}"
                continue
            try:
                shutil.copytree(snapshot, backup_path)
            except FileExistsError:
                shutil.rmtree(backup_path)
                continue
            except FileNotFoundError:
                shutil.rmtree(backup_path, ignore_errors=True)
                continue
            selection_path.write_text(
                json.dumps(selection, sort_keys=True, indent=2) + "\n"
            )
            return
        if not _process_alive(args.phase_pid):
            raise RuntimeError(
                "phase one exited before publishing a recoverable generation prefix "
                f"(last rejection: {last_error}):\n" + _log_tail(phase_log)
            )
        time.sleep(0.2)
    raise TimeoutError(f"no recoverable prefix snapshot found ({last_error})")


def _events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _matching_siblings(
    events: list[dict[str, Any]], selection: dict[str, Any], event_name: str
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for event in events:
        if event["event"] != event_name or event["group_id"] != selection["group_id"]:
            continue
        siblings = event.get("siblings") or [event.get("sibling")]
        found.extend(
            sibling
            for sibling in siblings
            if sibling is not None
            and sibling["gym_rollout_id"] == selection["gym_rollout_id"]
        )
    return found


def _verify_restored_cut(
    cut: dict[str, Any],
    *,
    phase1: list[dict[str, Any]],
    phase2: list[dict[str, Any]],
    restored_prefixes: list[tuple[str, ...]],
) -> None:
    """Require one cut to return to its Gym owner and resume its exact prefix."""

    phase1_dispatch = _matching_siblings(phase1, cut, "dispatch")
    if not phase1_dispatch:
        raise AssertionError(
            f"cut Gym episode {cut['gym_rollout_id']!r} was not observed in "
            "phase-one dispatches"
        )
    if not all(
        sibling["gym_instance_id"] == cut["gym_instance_id"]
        and sibling["gym_attempt"] == cut["gym_attempt"]
        for sibling in phase1_dispatch
    ):
        raise AssertionError(
            f"phase-one dispatch identity mismatch: {phase1_dispatch!r}"
        )

    phase2_dispatch = _matching_siblings(phase2, cut, "dispatch")
    phase2_refused = _matching_siblings(phase2, cut, "refused")
    if len(phase2_dispatch) - len(phase2_refused) != 1:
        raise AssertionError(
            f"expected one accepted restored dispatch: {phase2_dispatch!r}, {phase2_refused!r}"
        )
    restored = phase2_dispatch[-1]
    expected_identity = (
        cut["gym_instance_id"],
        cut["gym_attempt"] + 1,
        cut["generation_index"],
    )
    actual_identity = (
        restored["gym_instance_id"],
        restored["gym_attempt"],
        restored["generation_index"],
    )
    if actual_identity != expected_identity:
        raise AssertionError((actual_identity, expected_identity))

    completions = _matching_siblings(phase2, cut, "completion")
    if not any(
        sibling["gym_instance_id"] == cut["gym_instance_id"]
        and sibling["gym_attempt"] == cut["gym_attempt"] + 1
        and sibling["status"] == "sealed"
        for sibling in completions
    ):
        raise AssertionError(f"restored Gym episode did not seal: {completions!r}")

    expected_prefix = (
        cut["model_call_id"],
        cut["prefix_token_count"],
        cut["prefix_digest"],
    )
    if not any(
        source_model_call_id == expected_prefix[0]
        and int(prefix_tokens) == expected_prefix[1]
        and prefix_digest == expected_prefix[2]
        for _, _, source_model_call_id, prefix_tokens, prefix_digest in restored_prefixes
    ):
        raise AssertionError(
            f"cut prefix was not restored: expected={expected_prefix!r}, "
            f"observed={restored_prefixes!r}"
        )


def verify_restore(args: argparse.Namespace) -> None:
    selection = _read_json(Path(args.selection_file))
    phase1 = _events(Path(args.phase1_events))
    phase2 = _events(Path(args.phase2_events))
    log = Path(args.phase2_log).read_text()
    training_info = _read_json(Path(args.training_info))

    restored_prefixes = re.findall(
        r"generation prefix restored: rollout_id=(\S+) model_call_id=(\S+) "
        r"source_model_call_id=(\S+) prefix_tokens=(\d+) "
        r"prefix_digest=([0-9a-f]{64})",
        log,
    )
    for cut in selection.get("cuts", [selection]):
        _verify_restored_cut(
            cut,
            phase1=phase1,
            phase2=phase2,
            restored_prefixes=restored_prefixes,
        )

    if training_info["current_step"] != args.max_steps:
        raise AssertionError(training_info)
    if training_info["trainer_version"] != args.max_steps:
        raise AssertionError(training_info)
    for step in range(1, args.max_steps + 1):
        matches = re.findall(rf"train step {step}/{args.max_steps}(?:\s|$)", log)
        if len(matches) != 1:
            raise AssertionError(f"train step {step} appeared {len(matches)} times")


def _parity_helpers() -> ModuleType:
    """Load the sibling parity module by path.

    Unit tests load this module by path, without its directory on sys.path, so
    a plain import would only work from the CLI.
    """
    path = Path(__file__).with_name("_gym_prefix_recovery_parity.py")
    spec = importlib.util.spec_from_file_location("_gym_prefix_recovery_parity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_restored_prefixes_completed(
    selection: Mapping[str, Any],
    recovery_log: str,
    recovery_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Require each restored call to finish once and train on its restored prefix.

    Needs no uninterrupted reference run. The cut call is its row's closing
    call, so its tokens end the row: counting back ``total_generation_tokens``
    from the end finds where the call starts. Unlike the last unmasked segment,
    that is not fooled by positions the trainer masked inside the call (e.g.
    top-p mismatches). The restored prefix must sit there, hash-identical to
    what the worker logged when it restored the call.
    """
    parity = _parity_helpers()
    parity._assert_selected_prefix_was_trained(selection, recovery_records)
    rows = {
        (row["group_id"], int(row["generation_index"])): row
        for batch in recovery_records
        for row in batch["rows"]
    }
    restored = parity._restored_prefix_hashes(recovery_log)
    completed = re.findall(
        r"generation prefix completed: rollout_id=\S+ model_call_id=\S+ "
        r"source_model_call_id=(\S+) prefix_tokens=(\d+) tail_tokens=(\d+) "
        r"total_generation_tokens=(\d+)",
        recovery_log,
    )
    report = []
    for cut in parity._selection_cuts(selection):
        source = cut["model_call_id"]
        prefix_tokens = int(cut["prefix_token_count"])
        matches = [m for m in completed if m[0] == source]
        if len(matches) != 1:
            raise AssertionError(
                f"restored call {source!r} did not complete exactly once: {matches!r}"
            )
        logged_prefix, tail, total = (int(value) for value in matches[0][1:])
        if logged_prefix != prefix_tokens or tail <= 0 or logged_prefix + tail != total:
            raise AssertionError(
                f"restored call {source!r} token accounting is wrong: selected "
                f"prefix={prefix_tokens}, logged prefix={logged_prefix}, "
                f"tail={tail}, total={total}"
            )
        if source not in restored:
            raise AssertionError(f"restored call {source!r} logged no prefix hash")
        row = rows[(cut["group_id"], int(cut["generation_index"]))]
        token_ids = row["token_ids"]
        mask = row["token_loss_mask"]
        first_generated = next((i for i, keep in enumerate(mask) if keep), None)
        start = len(token_ids) - total
        if first_generated is None or start < first_generated:
            raise AssertionError(
                f"trained row for {source!r} is shorter than its generation: "
                f"row={len(token_ids)}, generated={total}, "
                f"first_generated={first_generated}"
            )
        prefix_hash = parity._token_ids_sha256(token_ids[start : start + prefix_tokens])
        if prefix_hash != restored[source]:
            raise AssertionError(
                f"trained row for {source!r} does not begin its call with the "
                "restored prefix"
            )
        report.append(
            {"model_call_id": source, "prefix_tokens": prefix_tokens, "total": total}
        )
    return report


def verify_completion(args: argparse.Namespace) -> None:
    """Without a reference run, require every restored prefix to finish and train."""
    parity = _parity_helpers()
    selection = _read_json(Path(args.selection_file))
    records = parity._read_jsonl(Path(args.recovery_training))
    report = assert_restored_prefixes_completed(
        selection, Path(args.phase2_log).read_text(), records
    )
    parity._assert_retired_prefix_keys(
        selection, records, Path(args.successor_checkpoint)
    )
    print(f"restored prefixes completed and trained: {report}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select")
    select.add_argument("snapshot_root")
    select.add_argument("selection_file")
    select.add_argument("phase_pid", type=int)
    select.add_argument("phase_log")
    select.add_argument("timeout_s", type=float)
    select.add_argument("backup_path")
    select.add_argument("sentinel_event", nargs="?")
    # Explicit opt-out, so a Workplace caller that drops the sentinel fails
    # instead of silently skipping the post-mutation resources check.
    select.add_argument("--stateless-resources", action="store_true")
    select.add_argument("--min-cuts", type=int, default=1)
    select.add_argument("--min-cut-groups", type=int, default=1)
    select.set_defaults(func=select_snapshot)

    verify = commands.add_parser("verify-restore")
    verify.add_argument("selection_file")
    verify.add_argument("phase1_events")
    verify.add_argument("phase2_events")
    verify.add_argument("phase2_log")
    verify.add_argument("training_info")
    verify.add_argument("max_steps", type=int)
    verify.set_defaults(func=verify_restore)

    completion = commands.add_parser("verify-completion")
    completion.add_argument("selection_file")
    completion.add_argument("recovery_training")
    completion.add_argument("phase2_log")
    completion.add_argument("successor_checkpoint")
    completion.set_defaults(func=verify_completion)
    return parser


if __name__ == "__main__":
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.command == "select" and (
        (arguments.sentinel_event is None) != arguments.stateless_resources
    ):
        parser.error(
            "select needs exactly one of sentinel_event or --stateless-resources"
        )
    arguments.func(arguments)
