#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _nested_get(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = row
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _reward(row: dict[str, Any]) -> float:
    value = row.get("reward")
    if value is None:
        value = _nested_get(row, ("result", "reward_info", "reward"))
    if value is None:
        return 0.0
    return float(value)


def _domain(row: dict[str, Any]) -> str | None:
    value = _nested_get(row, ("config", "domain"))
    if value is None:
        value = _nested_get(row, ("task", "user_scenario", "instructions", "domain"))
    if value is None:
        value = _nested_get(row, ("info", "user_scenario", "instructions", "domain"))
    return str(value) if value is not None else None


def _task_key(row: dict[str, Any]) -> str:
    for path in (
        ("task", "id"),
        ("result", "task_id"),
        ("info", "id"),
    ):
        value = _nested_get(row, path)
        if value is not None:
            return str(value)
    if row.get("_ng_task_index") is not None:
        return str(row["_ng_task_index"])
    if row.get("task_idx") is not None:
        return str(row["task_idx"])
    raise ValueError("Could not infer task id from rollout row")


def _num_steps(row: dict[str, Any]) -> int:
    for path in (("num_steps",), ("result", "num_steps")):
        value = _nested_get(row, path)
        if value is not None:
            return int(value)
    return len(_actual_tool_calls(row))


def _parse_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _actual_tool_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for item in _nested_get(row, ("response", "output")) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        name = item.get("name")
        if not name:
            continue
        calls.append(
            {
                "name": name,
                "arguments": _parse_arguments(item.get("arguments", {})),
            }
        )
    return calls


def _expected_tool_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    checks = _nested_get(row, ("result", "reward_info", "action_checks")) or []
    calls = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        action = check.get("action")
        if not isinstance(action, dict):
            continue
        if action.get("requestor") not in (None, "assistant"):
            continue
        name = action.get("name")
        if not name:
            continue
        calls.append(
            {
                "name": name,
                "arguments": action.get("arguments", {}),
            }
        )
    if calls:
        return calls

    actions = _nested_get(row, ("task", "evaluation_criteria", "actions")) or []
    if not actions:
        actions = _nested_get(row, ("info", "evaluation_criteria", "actions")) or []
    for action in actions:
        if not isinstance(action, dict):
            continue
        if action.get("requestor") not in (None, "assistant"):
            continue
        name = action.get("name")
        if name:
            calls.append(
                {
                    "name": name,
                    "arguments": action.get("arguments", {}),
                }
            )
    return calls


def _format_call(call: dict[str, Any], include_arguments: bool) -> str:
    name = str(call["name"])
    if not include_arguments:
        return name
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        rendered_args = arguments
    else:
        rendered_args = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    return f"{name}({rendered_args})"


def _format_privileged_information(
    calls: list[dict[str, Any]],
    pi_format: str,
) -> str:
    if pi_format == "tool_calls_args":
        lines = [_format_call(call, include_arguments=True) for call in calls]
        title = "Successful tool calls with arguments:"
    elif pi_format == "tool_calls":
        lines = [_format_call(call, include_arguments=False) for call in calls]
        title = "Successful tool calls:"
    elif pi_format == "hints":
        rendered = ", then ".join(
            _format_call(call, include_arguments=True) for call in calls
        )
        return f"Use the successful trajectory as a guide: {rendered}."
    else:
        raise ValueError(f"Unsupported PI format: {pi_format}")
    numbered = "\n".join(f"{idx + 1}) {line}" for idx, line in enumerate(lines))
    return f"{title}\n{numbered}"


def _select_best_rows(
    rows: list[dict[str, Any]],
    min_reward: float,
    domain: str | None,
    one_per_task: bool,
) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        if _reward(row) < min_reward:
            continue
        if domain is not None and _domain(row) != domain:
            continue
        candidates.append(row)

    if not one_per_task:
        return sorted(candidates, key=lambda row: (_task_key(row), _num_steps(row)))

    best_by_task: dict[str, dict[str, Any]] = {}
    for row in candidates:
        key = _task_key(row)
        prev = best_by_task.get(key)
        if prev is None:
            best_by_task[key] = row
            continue
        if (_num_steps(row), -_reward(row)) < (_num_steps(prev), -_reward(prev)):
            best_by_task[key] = row
    return [best_by_task[key] for key in sorted(best_by_task)]


def _build_training_row(
    rollout: dict[str, Any],
    calls: list[dict[str, Any]],
    pi_format: str,
    agent_name: str | None,
    source_rollout_path: Path | None,
) -> dict[str, Any]:
    if "responses_create_params" not in rollout:
        raise ValueError(f"Task {_task_key(rollout)!r} is missing responses_create_params")
    row = {
        "responses_create_params": rollout["responses_create_params"],
        "privileged_information": _format_privileged_information(calls, pi_format),
        "privileged_information_type": pi_format,
        "source_task_id": _task_key(rollout),
        "source_reward": _reward(rollout),
        "source_num_steps": _num_steps(rollout),
    }
    for key in ("task_idx", "example_id", "reward_profile", "info"):
        if key in rollout:
            row[key] = rollout[key]
    if source_rollout_path is not None:
        row["source_rollout_path"] = str(source_rollout_path)
        row["source_rollout_name"] = source_rollout_path.parent.name
    agent_ref = dict(rollout.get("agent_ref") or {})
    if agent_name is not None:
        agent_ref["name"] = agent_name
    if agent_ref:
        row["agent_ref"] = agent_ref
    return row


def build_pi_dataset(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for rollout_path in args.rollouts:
        for row in _read_jsonl(rollout_path):
            row["_pi_source_rollout_path"] = rollout_path
            rows.append(row)
    selected = _select_best_rows(
        rows,
        min_reward=args.min_reward,
        domain=args.domain,
        one_per_task=args.one_per_task,
    )
    output = []
    skipped_no_calls = 0
    for row in selected:
        calls = _actual_tool_calls(row) if args.source == "actual" else _expected_tool_calls(row)
        if not calls:
            skipped_no_calls += 1
            continue
        output.append(
            _build_training_row(
                row,
                calls,
                args.pi_format,
                args.agent_name,
                row.get("_pi_source_rollout_path"),
            )
        )
    if skipped_no_calls:
        print(f"Skipped {skipped_no_calls} selected rows with no tool calls")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pi-format",
        choices=("tool_calls_args", "tool_calls", "hints"),
        default="tool_calls_args",
    )
    parser.add_argument("--source", choices=("actual", "expected"), default="actual")
    parser.add_argument("--domain")
    parser.add_argument("--min-reward", type=float, default=1.0)
    parser.add_argument("--one-per-task", action="store_true", default=True)
    parser.add_argument("--all-successes", dest="one_per_task", action="store_false")
    parser.add_argument("--agent-name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_pi_dataset(args)
    _write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
