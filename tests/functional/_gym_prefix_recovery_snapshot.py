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
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from _gym_prefix_recovery_parity import (
    _assert_prefix_boundary,
    _assert_retired_prefix_keys,
    _rows_by_identity,
    load_training_records,
)
from _gym_recovery_snapshot_helpers import (
    calendar_sentinel_count,
    ledger_episode_index,
    matching_siblings,
    participant_records,
    read_json,
    read_jsonl,
    restored_prefixes,
)


def _episode_key(record: dict[str, Any]) -> tuple[str, int]:
    episode = record["episode_id"]
    return episode["rollout_id"], episode["attempt"]


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
    manifest = read_json(required[0])
    if manifest.get("base_train_step") != 0 or manifest.get("trainer_version") != 0:
        raise AssertionError("prefix fault injection must use a bootstrap snapshot")

    gym = read_json(required[1])
    ledger = ledger_episode_index(required[2])
    agent_keys = {
        _episode_key(record) for _, record in participant_records(snapshot, "agent")
    }
    resource_index = {
        _episode_key(record): (manifest_path, record["state"])
        for manifest_path, record in participant_records(snapshot, "resources")
        if record.get("state")
    }
    model_records = participant_records(snapshot, "model")

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
            if calendar_sentinel_count(resources_state, sentinel_event) != 1:
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


def _verify_restored_cut(
    cut: dict[str, Any],
    *,
    phase1: list[dict[str, Any]],
    phase2: list[dict[str, Any]],
    restored: list[dict[str, Any]],
) -> None:
    """Require one cut to return to its Gym owner and resume its exact prefix."""

    phase1_dispatch = matching_siblings(phase1, cut, "dispatch")
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

    phase2_dispatch = matching_siblings(phase2, cut, "dispatch")
    phase2_refused = matching_siblings(phase2, cut, "refused")
    if len(phase2_dispatch) - len(phase2_refused) != 1:
        raise AssertionError(
            f"expected one accepted restored dispatch: {phase2_dispatch!r}, {phase2_refused!r}"
        )
    restored_dispatch = phase2_dispatch[-1]
    expected_identity = (
        cut["gym_instance_id"],
        cut["gym_attempt"] + 1,
        cut["generation_index"],
    )
    actual_identity = (
        restored_dispatch["gym_instance_id"],
        restored_dispatch["gym_attempt"],
        restored_dispatch["generation_index"],
    )
    if actual_identity != expected_identity:
        raise AssertionError((actual_identity, expected_identity))

    completions = matching_siblings(phase2, cut, "completion")
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
        (
            line["source_model_call_id"],
            line["prefix_tokens"],
            line["prefix_digest"],
        )
        == expected_prefix
        for line in restored
    ):
        raise AssertionError(
            f"cut prefix was not restored: expected={expected_prefix!r}, "
            f"observed={restored!r}"
        )


def verify_restore(args: argparse.Namespace) -> None:
    selection = read_json(Path(args.selection_file))
    phase1 = read_jsonl(Path(args.phase1_events))
    phase2 = read_jsonl(Path(args.phase2_events))
    log = Path(args.phase2_log).read_text()
    training_info = read_json(Path(args.training_info))

    restored = restored_prefixes(log)
    for cut in selection["cuts"]:
        _verify_restored_cut(cut, phase1=phase1, phase2=phase2, restored=restored)

    if training_info["current_step"] != args.max_steps:
        raise AssertionError(training_info)
    if training_info["trainer_version"] != args.max_steps:
        raise AssertionError(training_info)
    for step in range(1, args.max_steps + 1):
        matches = re.findall(rf"train step {step}/{args.max_steps}(?:\s|$)", log)
        if len(matches) != 1:
            raise AssertionError(f"train step {step} appeared {len(matches)} times")


# Router replay: the worker hashes a restored call's kept prefix routes once
# when it rebuilds them from TQ and again from the row it stages for training.
_PREFIX_ROUTES_LINE = re.compile(
    r"generation prefix (?P<event>restored|completed): .*?"
    r"source_model_call_id=(?P<source_model_call_id>\S+) "
    r"prefix_tokens=(?P<prefix_tokens>\d+) .*?"
    r"prefix_routes_sha256=(?P<routes_sha256>[0-9a-f]{64}|none)"
)


def _assert_restored_routes_staged(
    selection: dict[str, Any], recovery_log: str
) -> None:
    """Require each restored call to stage, for training, the routes it restored.

    A call that dropped its routes, or replaced its prefix's routes with ones
    recomputed after the restart, logs a different hash at completion.
    """
    logged: dict[tuple[str, str, int], list[str]] = {}
    for match in _PREFIX_ROUTES_LINE.finditer(recovery_log):
        key = (
            match["event"],
            match["source_model_call_id"],
            int(match["prefix_tokens"]),
        )
        logged.setdefault(key, []).append(match["routes_sha256"])
    for cut in selection["cuts"]:
        source = cut["model_call_id"]
        prefix_tokens = int(cut["prefix_token_count"])
        restored = logged.get(("restored", source, prefix_tokens), [])
        completed = logged.get(("completed", source, prefix_tokens), [])
        if len(restored) != 1 or len(completed) != 1:
            raise AssertionError(
                f"restored call {source!r} logged {len(restored)} restored and "
                f"{len(completed)} completed route hashes, expected one each"
            )
        if "none" in (restored[0], completed[0]):
            raise AssertionError(f"restored call {source!r} carried no routes")
        if restored != completed:
            raise AssertionError(
                f"row staged for {source!r} does not carry its restored prefix routes"
            )


def _generation_kl_by_span(
    selection: dict[str, Any],
    records: list[dict[str, Any]],
    starts: dict[tuple[str, int], int],
    *,
    kl_type: str,
) -> dict[str, dict[str, float]]:
    """Report the trainer's ``gen_kl_error``, split by where the tokens came from.

    Restored prefix tokens were sampled before the crash and are trained with
    their restored routes; a resumed call's tail and every other row are
    sampled and trained as usual. Wrong prefix routes would make the training
    logprobs of the restored prefix disagree with the ones it was sampled
    with.
    """
    # Deferred: loads the trainer's loss stack, which only router-replay
    # verification needs.
    import torch

    from nemo_rl.algorithms.utils import calculate_kl

    prefix_tokens = {
        (cut["group_id"], int(cut["generation_index"])): int(cut["prefix_token_count"])
        for cut in selection["cuts"]
    }
    spans: dict[str, list[float]] = {
        "restored_prefix": [],
        "resumed_tail": [],
        "other_rows": [],
    }
    for batch in records:
        for row in batch["rows"]:
            if "prev_logprobs" not in row:
                raise AssertionError("the trainer dump has no prev_logprobs")
            # The dump writes non-finite values as null.
            prev = [math.nan if v is None else v for v in row["prev_logprobs"]]
            generated = [
                math.nan if v is None else v for v in row["generation_logprobs"]
            ]
            kl = calculate_kl(
                logprobs=torch.tensor(generated),
                logprobs_reference=torch.tensor(prev),
                kl_type=kl_type,
                input_clamp_value=None,
                output_clamp_value=None,
            ).tolist()
            key = (row["group_id"], int(row["generation_index"]))
            start = starts.get(key)
            for position, keep in enumerate(row["token_loss_mask"]):
                if not keep:
                    continue
                if start is None:
                    span = "other_rows"
                elif position < start + prefix_tokens[key]:
                    span = "restored_prefix"
                else:
                    span = "resumed_tail"
                spans[span].append(kl[position])
    report = {
        name: {
            "tokens": len(values),
            "mean": sum(values) / len(values) if values else float("nan"),
            "max": max(values, default=float("nan")),
        }
        for name, values in spans.items()
    }
    if not spans["restored_prefix"]:
        raise AssertionError("no trained restored-prefix tokens to report")
    if not all(math.isfinite(value) for values in spans.values() for value in values):
        raise AssertionError(f"gen_kl_error is not finite: {report!r}")
    return report


def verify_completion(args: argparse.Namespace) -> None:
    """Without a reference run, require every restored prefix to finish and train.

    Each restored call must complete once, close its trained row, and start
    with the exact prefix the worker restored (``_assert_prefix_boundary``); no
    row may be trained twice, and no restored prefix row may still be live when
    training reads the step.
    """
    selection = read_json(Path(args.selection_file))
    records = load_training_records(
        Path(args.recovery_log_dir), staging_keys=Path(args.recovery_staging_keys)
    )
    _rows_by_identity(records)
    starts = _assert_prefix_boundary(
        selection, Path(args.phase2_log).read_text(), records
    )
    _assert_retired_prefix_keys(selection, records)
    print(f"restored prefixes completed and trained: {sorted(starts)!r}", flush=True)
    if args.require_routes:
        recovery_log = Path(args.phase2_log).read_text()
        _assert_restored_routes_staged(selection, recovery_log)
        report = _generation_kl_by_span(
            selection, records, starts, kl_type=args.kl_type
        )
        print(
            f"restored prefix routes staged; gen_kl_error: {json.dumps(report)}",
            flush=True,
        )


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
    completion.add_argument("recovery_log_dir")
    completion.add_argument("recovery_staging_keys")
    completion.add_argument("phase2_log")
    # Router replay: also require the restored routes and report gen_kl_error.
    completion.add_argument("--require-routes", action="store_true")
    completion.add_argument("--kl-type", default="k3")
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
