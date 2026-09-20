# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Select recovery cuts and compare uninterrupted/restarted functional runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

from _gym_prefix_recovery_snapshot import _active_prefixes, _read_json


_SEMANTIC_METRICS = (
    "train/reward",
    "train/loss",
    "train/gen_kl_error",
    "train/advantages/mean",
    "train/advantages/max",
    "train/advantages/min",
    "train/global_valid_seqs",
    "train/global_valid_toks",
    "train/mean_prompt_length",
)
_EXACT_TRAIN_FIELDS = (
    "content",
    "rewards",
    "input_lengths",
    "token_ids",
    "token_loss_mask",
    "sample_loss_mask",
)
_APPROXIMATE_TRAIN_FIELDS = (
    "advantages",
    "generation_logprobs",
    "prev_logprobs",
)


def _published_snapshots(checkpoint_dir: Path) -> list[Path]:
    # Snapshot counters restart per anchor directory, so order by publication
    # time. The live retention pass can remove a candidate during this scan.
    ranked: list[tuple[int, str, Path]] = []
    for path in checkpoint_dir.glob("**/rollout_snapshots/snapshot_[0-9]*"):
        try:
            if not (path / "manifest.json").is_file():
                continue
            ranked.append((path.stat().st_mtime_ns, str(path), path))
        except OSError:
            continue
    ranked.sort(reverse=True)
    return [path for _, _, path in ranked]


def _matching_group(
    recovery: dict[str, Any],
    *,
    rollout_id: str,
    attempt_index: int,
) -> dict[str, Any] | None:
    for group in recovery.get("groups", []):
        group_id = group.get("group_id")
        for sibling in group.get("siblings", []):
            logical_id = f"{group_id}_g{sibling.get('generation_index')}"
            if logical_id != rollout_id:
                continue
            if any(
                attempt.get("attempt_index") == attempt_index
                for attempt in sibling.get("attempts", [])
            ):
                return group
    return None


def inspect_cut_candidate(
    snapshot: Path,
    *,
    min_train_step: int,
    task_source: str,
    max_generation_tokens: int,
) -> dict[str, Any]:
    """Return one active cut for ``task_source`` or reject this snapshot."""
    # The parity comparator does not need the heavyweight training dependency.
    import torch

    manifest = _read_json(snapshot / "manifest.json")
    base_train_step = manifest.get("base_train_step")
    if not isinstance(base_train_step, int) or base_train_step < min_train_step:
        raise AssertionError(
            f"snapshot train step {base_train_step!r} precedes {min_train_step}"
        )
    gym_checkpoint = manifest.get("gym_checkpoint")
    if not isinstance(gym_checkpoint, dict):
        raise AssertionError("snapshot has no Gym participant checkpoint")
    recovery = torch.load(snapshot / "rollout_recovery.pt", weights_only=True)
    if not isinstance(recovery, dict):
        raise TypeError("rollout recovery sidecar is not a mapping")

    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for prefix in _active_prefixes(snapshot, manifest, gym_checkpoint):
        token_count = prefix.get("prefix_token_count")
        if (
            not isinstance(token_count, int)
            or token_count <= 0
            or token_count >= max_generation_tokens
        ):
            continue
        group = _matching_group(
            recovery,
            rollout_id=prefix["rollout_id"],
            attempt_index=prefix["attempt_index"],
        )
        if group is not None and group.get("task_source") == task_source:
            matches.append((prefix, group))
    if not matches:
        raise AssertionError(
            f"snapshot has no non-empty active prefix for task_source={task_source!r}"
        )

    prefix, group = max(matches, key=lambda pair: pair[0]["prefix_token_count"])
    return {
        "snapshot_path": str(snapshot.resolve()),
        "checkpoint_id": gym_checkpoint["checkpoint_id"],
        "base_train_step": base_train_step,
        "task_source": task_source,
        "group_id": group["group_id"],
        "rollout_id": prefix["rollout_id"],
        "attempt_index": prefix["attempt_index"],
        "model_call_id": prefix["model_call_id"],
        "prefix_token_count": prefix["prefix_token_count"],
        "prefix_digest": prefix["prefix_digest"],
        "staging_keys": prefix["staging_keys"],
    }


def select_cut(args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.timeout_s
    last_error = "no published rollout snapshot"
    while time.monotonic() < deadline:
        for snapshot in _published_snapshots(args.checkpoint_dir):
            try:
                selection = inspect_cut_candidate(
                    snapshot,
                    min_train_step=args.min_train_step,
                    task_source=args.task_source,
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
                json.dumps(selection, indent=2, sort_keys=True) + "\n"
            )
            print(
                "selected recovery-parity cut: "
                f"step={selection['base_train_step']} "
                f"task_source={selection['task_source']} "
                f"tokens={selection['prefix_token_count']} "
                f"rollout={selection['rollout_id']}",
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
                "training exited before publishing the requested recovery cut; "
                f"last candidate: {last_error}\n{tail}"
            ) from error
        time.sleep(0.1)
    raise TimeoutError(
        "no matching recovery cut was published before the deadline; "
        f"last candidate: {last_error}"
    )


def prune_to_selection(args: argparse.Namespace) -> None:
    """Discard progress newer than the crash point selected by ``select-cut``."""
    selection = _read_json(args.selection)
    checkpoint_dir = args.checkpoint_dir.resolve()
    selected = Path(selection["snapshot_path"]).resolve()
    selected.relative_to(checkpoint_dir)
    if not selected.is_dir():
        raise FileNotFoundError(selected)
    base_train_step = selection["base_train_step"]

    for step_dir in checkpoint_dir.glob("step_[0-9]*"):
        try:
            step = int(step_dir.name.removeprefix("step_"))
        except ValueError:
            continue
        if step > base_train_step:
            shutil.rmtree(step_dir)

    selected_root = selected.parent
    for candidate in selected_root.glob("snapshot_[0-9]*"):
        if candidate.resolve() != selected:
            shutil.rmtree(candidate)

    if base_train_step == 0:
        for step_dir in checkpoint_dir.glob("step_[0-9]*"):
            shutil.rmtree(step_dir)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not all(isinstance(record, dict) for record in records):
        raise TypeError(f"JSONL records must be objects: {path}")
    return records


def _logical_dispatches(events: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    ordered: list[tuple[Any, ...]] = []
    seen: set[tuple[Any, ...]] = set()
    for event in events:
        if event.get("event") != "dispatch":
            continue
        identity = (
            event.get("target_step"),
            event.get("prompt_idx"),
            event.get("task_source"),
        )
        if identity not in seen:
            ordered.append(identity)
            seen.add(identity)
    return ordered


def _logical_completions(
    events: list[dict[str, Any]],
) -> tuple[list[tuple[Any, ...]], dict[tuple[Any, ...], float]]:
    ordered: list[tuple[Any, ...]] = []
    completed: dict[tuple[Any, ...], float] = {}
    for event in events:
        if event.get("event") != "completion_forwarded":
            continue
        identity = (
            event.get("target_step"),
            event.get("prompt_idx"),
            event.get("task_source"),
            event.get("generation_index"),
        )
        if identity in completed:
            raise AssertionError(f"logical completion was forwarded twice: {identity}")
        ordered.append(identity)
        completed[identity] = float(event["reward"])
    return ordered, completed


def _effective_completions(
    stages: list[list[dict[str, Any]]],
) -> tuple[list[tuple[Any, ...]], dict[tuple[Any, ...], float]]:
    """Select each logical completion's last stage after checkpoint pruning.

    A crashed process can forward a result after the selected snapshot was
    published. That result is absent from the restored ledger and legitimately
    appears again in a later stage. Duplicate forwarding within one stage is
    still an error; a later stage supersedes an earlier discarded attempt.
    """
    effective: dict[tuple[Any, ...], tuple[int, int, float]] = {}
    for stage_index, events in enumerate(stages):
        ordered, completed = _logical_completions(events)
        for position, identity in enumerate(ordered):
            effective[identity] = (stage_index, position, completed[identity])
    ordered = sorted(
        effective,
        key=lambda identity: (effective[identity][0], effective[identity][1]),
    )
    return ordered, {identity: effective[identity][2] for identity in ordered}


def _trained_step_events(
    events: list[dict[str, Any]], *, steps: int
) -> list[dict[str, Any]]:
    """Keep hook events belonging to steps the trainer is expected to consume."""
    return [
        event
        for event in events
        if isinstance(event.get("target_step"), int)
        and 0 <= event["target_step"] < steps
    ]


def _rollout_timeline(
    stages: list[list[dict[str, Any]]],
    *,
    effective_order: list[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    timestamped = [
        event
        for events in stages
        for event in events
        if "timestamp_ns" in event
    ]
    if not timestamped:
        return []
    origin_ns = min(int(event["timestamp_ns"]) for event in timestamped)
    entries: dict[tuple[Any, ...], dict[str, Any]] = {}
    effective_ranks = {
        identity: rank for rank, identity in enumerate(effective_order)
    }

    def entry_for(identity: tuple[Any, ...]) -> dict[str, Any]:
        return entries.setdefault(
            identity,
            {
                "target_step": identity[0],
                "prompt_idx": identity[1],
                "task_source": identity[2],
                "generation_index": identity[3],
                "dispatches": [],
                "arrivals": [],
                "forwarded_completions": [],
                "effective_completion_rank": effective_ranks.get(identity),
            },
        )

    for stage_index, events in enumerate(stages):
        for event in events:
            if "timestamp_ns" not in event:
                continue
            elapsed_s = (int(event["timestamp_ns"]) - origin_ns) / 1_000_000_000
            if event.get("event") == "dispatch":
                generation_indices = event.get("generation_indices", [])
                rollout_ids = event.get("rollout_ids", [])
                if len(generation_indices) != len(rollout_ids):
                    raise AssertionError(
                        "dispatch event has mismatched generation indices and "
                        "rollout IDs"
                    )
                for generation_index, rollout_id in zip(
                    generation_indices, rollout_ids, strict=True
                ):
                    identity = (
                        event.get("target_step"),
                        event.get("prompt_idx"),
                        event.get("task_source"),
                        generation_index,
                    )
                    entry_for(identity)["dispatches"].append(
                        {
                            "stage": stage_index,
                            "rollout_id": rollout_id,
                            "elapsed_s": elapsed_s,
                        }
                    )
                continue
            if event.get("event") not in (
                "completion_arrived",
                "completion_forwarded",
            ):
                continue
            identity = (
                event.get("target_step"),
                event.get("prompt_idx"),
                event.get("task_source"),
                event.get("generation_index"),
            )
            field = (
                "arrivals"
                if event["event"] == "completion_arrived"
                else "forwarded_completions"
            )
            entry_for(identity)[field].append(
                {
                    "stage": stage_index,
                    "rollout_id": event.get("rollout_id"),
                    "elapsed_s": elapsed_s,
                    "reward": event.get("reward"),
                }
            )

    for identity, entry in entries.items():
        if identity not in effective_ranks:
            continue
        forwarded = entry["forwarded_completions"]
        if forwarded:
            entry["effective_completion"] = forwarded[-1]

    return sorted(
        entries.values(),
        key=lambda entry: (
            entry["effective_completion_rank"] is None,
            entry["effective_completion_rank"]
            if entry["effective_completion_rank"] is not None
            else 0,
            entry["target_step"],
            entry["prompt_idx"],
            entry["generation_index"],
        ),
    )


def _write_rollout_timeline(
    path: Path,
    *,
    baseline_stages: list[list[dict[str, Any]]],
    recovery_stages: list[list[dict[str, Any]]],
    baseline_order: list[tuple[Any, ...]],
    recovery_order: list[tuple[Any, ...]],
    completion_order_matches: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "completion_order_matches": completion_order_matches,
                "baseline_completion_order": baseline_order,
                "recovery_completion_order": recovery_order,
                "baseline": _rollout_timeline(
                    baseline_stages, effective_order=baseline_order
                ),
                "recovery": _rollout_timeline(
                    recovery_stages, effective_order=recovery_order
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _assert_nested_close(
    baseline: Any,
    recovery: Any,
    *,
    path: str,
    rtol: float,
    atol: float,
) -> None:
    if isinstance(baseline, list) and isinstance(recovery, list):
        if len(baseline) != len(recovery):
            raise AssertionError(
                f"{path} length differs: {len(baseline)} != {len(recovery)}"
            )
        for index, (left, right) in enumerate(zip(baseline, recovery, strict=True)):
            _assert_nested_close(
                left,
                right,
                path=f"{path}[{index}]",
                rtol=rtol,
                atol=atol,
            )
        return
    if isinstance(baseline, (int, float)) and isinstance(recovery, (int, float)):
        if not math.isclose(
            float(baseline),
            float(recovery),
            rel_tol=rtol,
            abs_tol=atol,
        ):
            raise AssertionError(f"{path} differs: {baseline!r} != {recovery!r}")
        return
    if baseline != recovery:
        raise AssertionError(f"{path} differs: {baseline!r} != {recovery!r}")


def _step_files(log_dir: Path, *, allow_restarts: bool = False) -> dict[int, Path]:
    candidates: dict[int, list[Path]] = {}
    for path in log_dir.glob("**/train_data_step*.jsonl"):
        step = int(path.stem.removeprefix("train_data_step"))
        candidates.setdefault(step, []).append(path)
    result: dict[int, Path] = {}
    for step, paths in candidates.items():
        if len(paths) > 1 and not allow_restarts:
            raise AssertionError(
                f"multiple training payloads found for step {step}: "
                f"{sorted(paths)!r}"
            )
        result[step] = max(
            paths,
            key=lambda path: (path.stat().st_mtime_ns, str(path)),
        )
    return result


def _compare_training_payloads(
    baseline_dir: Path,
    recovery_dir: Path,
    *,
    steps: int,
    rtol: float,
    atol: float,
) -> None:
    baseline_files = _step_files(baseline_dir)
    recovery_files = _step_files(recovery_dir, allow_restarts=True)
    expected_steps = set(range(1, steps + 1))
    if set(baseline_files) != expected_steps:
        raise AssertionError(
            f"baseline training payload steps are {sorted(baseline_files)}, "
            f"expected {sorted(expected_steps)}"
        )
    if set(recovery_files) != expected_steps:
        raise AssertionError(
            f"recovery training payload steps are {sorted(recovery_files)}, "
            f"expected {sorted(expected_steps)}"
        )

    for step in sorted(expected_steps):
        baseline = _read_jsonl(baseline_files[step])
        recovery = _read_jsonl(recovery_files[step])
        if len(baseline) != len(recovery):
            raise AssertionError(
                f"step {step} trained sample count differs: "
                f"{len(baseline)} != {len(recovery)}"
            )
        for row, (left, right) in enumerate(zip(baseline, recovery, strict=True)):
            for field in _EXACT_TRAIN_FIELDS:
                if left.get(field) != right.get(field):
                    raise AssertionError(
                        f"step {step} row {row} field {field!r} differs"
                    )
            for field in _APPROXIMATE_TRAIN_FIELDS:
                _assert_nested_close(
                    left.get(field),
                    right.get(field),
                    path=f"step {step} row {row} {field}",
                    rtol=rtol,
                    atol=atol,
                )


def _compare_metrics(
    baseline_path: Path,
    recovery_path: Path,
    *,
    steps: int,
    rtol: float,
    atol: float,
) -> None:
    baseline = _read_json(baseline_path)
    recovery = _read_json(recovery_path)
    expected_steps = {str(step) for step in range(1, steps + 1)}
    required = _SEMANTIC_METRICS[:3]
    for metric in required:
        if metric not in baseline or metric not in recovery:
            raise AssertionError(f"required semantic metric is missing: {metric}")
    compared = 0
    for metric in _SEMANTIC_METRICS:
        if metric not in baseline and metric not in recovery:
            continue
        if metric not in baseline or metric not in recovery:
            raise AssertionError(f"metric presence differs: {metric}")
        if set(baseline[metric]) != expected_steps:
            raise AssertionError(
                f"baseline metric {metric} has steps {sorted(baseline[metric])}"
            )
        if set(recovery[metric]) != expected_steps:
            raise AssertionError(
                f"recovery metric {metric} has steps {sorted(recovery[metric])}"
            )
        for step in sorted(expected_steps, key=int):
            _assert_nested_close(
                baseline[metric][step],
                recovery[metric][step],
                path=f"metric {metric} step {step}",
                rtol=rtol,
                atol=atol,
            )
        compared += 1
    if compared < len(required):
        raise AssertionError("too few semantic metrics were compared")


def _verify_workplace_audit(
    baseline_path: Path,
    recovery_paths: list[Path],
    *,
    expected_mutations: int,
) -> None:
    baseline = _read_jsonl(baseline_path)
    recovery = [record for path in recovery_paths for record in _read_jsonl(path)]
    baseline_mutations = [
        event for event in baseline if event.get("event") == "mutation_applied"
    ]
    recovery_mutations = [
        event for event in recovery if event.get("event") == "mutation_applied"
    ]
    if len(baseline_mutations) != expected_mutations:
        raise AssertionError(
            f"baseline applied {len(baseline_mutations)} Workplace mutations; "
            f"expected {expected_mutations}"
        )
    # Discarded post-snapshot executions can legitimately mutate an isolated
    # resource instance before the process is killed. The restored execution
    # uses checkpointed state, so require every trained mutation plus the
    # per-instance exactly-once sentinel invariant below.
    if len(recovery_mutations) < expected_mutations:
        raise AssertionError(
            f"recovery applied {len(recovery_mutations)} Workplace mutations; "
            f"expected at least {expected_mutations}"
        )
    if any(event.get("sentinel_count") != 1 for event in recovery_mutations):
        raise AssertionError("a recovered Workplace mutation executed more than once")
    restored = [event for event in recovery if event.get("event") == "state_restored"]
    if not restored or any(event.get("sentinel_count") != 1 for event in restored):
        raise AssertionError("Workplace state was not restored with one mutation")


def compare_runs(args: argparse.Namespace) -> None:
    baseline_events = _trained_step_events(
        _read_jsonl(args.baseline_events), steps=args.steps
    )
    recovery_stages = [
        _trained_step_events(_read_jsonl(path), steps=args.steps)
        for path in args.recovery_events
    ]
    recovery_events = [event for stage in recovery_stages for event in stage]
    baseline_dispatches = _logical_dispatches(baseline_events)
    recovery_dispatches = _logical_dispatches(recovery_events)
    if baseline_dispatches != recovery_dispatches:
        raise AssertionError(
            "logical prompt order differs between uninterrupted and recovery runs: "
            f"baseline={baseline_dispatches!r}, recovery={recovery_dispatches!r}"
        )
    expected_groups = args.steps * args.prompts_per_step
    if len(baseline_dispatches) != expected_groups:
        raise AssertionError(
            f"observed {len(baseline_dispatches)} logical prompt groups, "
            f"expected {expected_groups}"
        )

    baseline_completion_order, baseline_completions = _effective_completions(
        [baseline_events]
    )
    recovery_completion_order, recovery_completions = _effective_completions(
        recovery_stages
    )
    completion_order_matches = baseline_completion_order == recovery_completion_order
    if args.timeline_output is not None:
        _write_rollout_timeline(
            args.timeline_output,
            baseline_stages=[baseline_events],
            recovery_stages=recovery_stages,
            baseline_order=baseline_completion_order,
            recovery_order=recovery_completion_order,
            completion_order_matches=completion_order_matches,
        )
    if args.require_completion_order_match and not completion_order_matches:
        raise AssertionError(
            "logical completion order differs between uninterrupted and recovery runs: "
            f"baseline={baseline_completion_order!r}, "
            f"recovery={recovery_completion_order!r}"
        )
    print(
        "logical rollout completion order "
        + ("matches" if completion_order_matches else "differs")
        + " between uninterrupted and recovery runs",
        flush=True,
    )
    if baseline_completions != recovery_completions:
        raise AssertionError(
            "logical completion rewards differ between uninterrupted and recovery runs"
        )
    expected_completions = expected_groups * args.generations_per_prompt
    if len(baseline_completions) != expected_completions:
        raise AssertionError(
            f"observed {len(baseline_completions)} logical completions, "
            f"expected {expected_completions}"
        )

    retried_sources = {
        event.get("task_source")
        for event in recovery_events
        if event.get("event") == "dispatch"
        and any("-a" in rollout_id for rollout_id in event.get("rollout_ids", []))
    }
    expected_sources = set(args.required_retried_task_source)
    if not expected_sources.issubset(retried_sources):
        raise AssertionError(
            "recovery did not redispatch every required agent type: "
            f"required={sorted(expected_sources)}, observed={sorted(retried_sources)}"
        )

    _compare_training_payloads(
        args.baseline_log_dir,
        args.recovery_log_dir,
        steps=args.steps,
        rtol=args.rtol,
        atol=args.atol,
    )
    _compare_metrics(
        args.baseline_metrics,
        args.recovery_metrics,
        steps=args.steps,
        rtol=args.rtol,
        atol=args.atol,
    )
    _verify_workplace_audit(
        args.baseline_audit,
        args.recovery_audit,
        expected_mutations=args.steps * args.generations_per_prompt,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select-cut")
    select.add_argument("checkpoint_dir", type=Path)
    select.add_argument("selection", type=Path)
    select.add_argument("pid", type=int)
    select.add_argument("run_log", type=Path)
    select.add_argument("timeout_s", type=float)
    select.add_argument("min_train_step", type=int)
    select.add_argument("task_source")
    select.add_argument("max_generation_tokens", type=int)
    select.set_defaults(func=select_cut)

    prune = commands.add_parser("prune-to-selection")
    prune.add_argument("checkpoint_dir", type=Path)
    prune.add_argument("selection", type=Path)
    prune.set_defaults(func=prune_to_selection)

    compare = commands.add_parser("compare")
    compare.add_argument("--baseline-events", type=Path, required=True)
    compare.add_argument(
        "--recovery-events", type=Path, action="append", required=True
    )
    compare.add_argument("--baseline-log-dir", type=Path, required=True)
    compare.add_argument("--recovery-log-dir", type=Path, required=True)
    compare.add_argument("--baseline-metrics", type=Path, required=True)
    compare.add_argument("--recovery-metrics", type=Path, required=True)
    compare.add_argument("--baseline-audit", type=Path, required=True)
    compare.add_argument(
        "--recovery-audit", type=Path, action="append", required=True
    )
    compare.add_argument("--timeline-output", type=Path)
    compare.add_argument("--require-completion-order-match", action="store_true")
    compare.add_argument("--steps", type=int, required=True)
    compare.add_argument("--prompts-per-step", type=int, required=True)
    compare.add_argument("--generations-per-prompt", type=int, required=True)
    compare.add_argument(
        "--required-retried-task-source", action="append", default=[]
    )
    compare.add_argument("--rtol", type=float, default=1e-5)
    compare.add_argument("--atol", type=float, default=1e-6)
    compare.set_defaults(func=compare_runs)
    return parser


if __name__ == "__main__":
    arguments = _parser().parse_args()
    arguments.func(arguments)
