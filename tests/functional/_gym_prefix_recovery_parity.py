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

"""Capture and compare finalized training rows for prefix-recovery parity."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


_EXACT_FIELDS = (
    "input_length",
    "token_ids",
    "token_loss_mask",
    "sample_loss_mask",
    "reward",
)
_APPROXIMATE_FIELDS = (
    "advantages",
    "generation_logprobs",
    "prev_logprobs",
)
_TOKEN_FIELDS = {
    "input_ids": "token_ids",
    "token_mask": "token_loss_mask",
    "advantages": "advantages",
    "generation_logprobs": "generation_logprobs",
    "prev_logprobs": "prev_logprobs",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _generation_identity(sample_id: str) -> tuple[str, int]:
    try:
        group_id, generation_index = sample_id.rsplit("_g", 1)
    except ValueError as error:
        raise ValueError(
            f"sample ID has no generation suffix: {sample_id!r}"
        ) from error
    if not group_id or not generation_index.isdigit():
        raise ValueError(f"invalid generated sample ID: {sample_id!r}")
    return group_id, int(generation_index)


def _tensor(value: Any, field: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"training field {field!r} is not a tensor")
    return value.detach().cpu()


def serialize_training_batch(
    *,
    train_step: int,
    chunk_index: int,
    sample_ids: Sequence[str],
    tags: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any],
    staging_sample_ids: Sequence[str],
) -> dict[str, Any]:
    """Convert one materialized TQ training batch to a stable JSON record."""

    if len(sample_ids) != len(tags):
        raise ValueError("training sample IDs and tags must have the same length")
    lengths = _tensor(data["input_lengths"], "input_lengths").tolist()
    if len(lengths) != len(sample_ids):
        raise ValueError("training input lengths do not match sample IDs")

    tensors = {
        field: _tensor(data[field], field)
        for field in (*_TOKEN_FIELDS, "sample_mask", "total_reward")
    }
    rows: list[dict[str, Any]] = []
    for row_index, (sample_id, tag, raw_length) in enumerate(
        zip(sample_ids, tags, lengths, strict=True)
    ):
        group_id, generation_index = _generation_identity(sample_id)
        prompt_idx = tag.get("prompt_idx")
        if not isinstance(prompt_idx, int):
            raise TypeError(f"sample {sample_id!r} has no integer prompt_idx tag")
        length = int(raw_length)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "group_id": group_id,
            "prompt_idx": prompt_idx,
            "generation_index": generation_index,
            "input_length": length,
            "sample_loss_mask": float(tensors["sample_mask"][row_index].item()),
            "reward": float(tensors["total_reward"][row_index].item()),
        }
        for source, destination in _TOKEN_FIELDS.items():
            row[destination] = tensors[source][row_index, :length].tolist()
        rows.append(row)

    return {
        "train_step": train_step,
        "chunk_index": chunk_index,
        "staging_sample_ids": sorted(staging_sample_ids),
        "rows": rows,
    }


def append_training_batch(path: Path, record: Mapping[str, Any]) -> None:
    """Append one complete batch record with a single filesystem write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload)


def _rows_by_identity(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int, int], Mapping[str, Any]]:
    result: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    groups: dict[tuple[int, int], str] = {}
    for batch in records:
        train_step = int(batch["train_step"])
        for row in batch["rows"]:
            identity = (
                train_step,
                int(row["prompt_idx"]),
                int(row["generation_index"]),
            )
            if identity in result:
                raise AssertionError(f"training row was consumed twice: {identity!r}")
            logical_group = identity[:2]
            group_id = str(row["group_id"])
            prior_group_id = groups.setdefault(logical_group, group_id)
            if prior_group_id != group_id:
                raise AssertionError(
                    "one logical prompt was split across physical groups: "
                    f"identity={logical_group!r}, groups="
                    f"{[prior_group_id, group_id]!r}"
                )
            result[identity] = row
    return result


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
            float(baseline), float(recovery), rel_tol=rtol, abs_tol=atol
        ):
            raise AssertionError(f"{path} differs: {baseline!r} != {recovery!r}")
        return
    if baseline != recovery:
        raise AssertionError(f"{path} differs: {baseline!r} != {recovery!r}")


def _compare_training_rows(
    baseline_records: Sequence[Mapping[str, Any]],
    recovery_records: Sequence[Mapping[str, Any]],
    *,
    steps: int,
    prompts_per_step: int,
    generations_per_prompt: int,
    rtol: float,
    atol: float,
) -> None:
    baseline = _rows_by_identity(baseline_records)
    recovery = _rows_by_identity(recovery_records)
    expected_count = steps * prompts_per_step * generations_per_prompt
    if len(baseline) != expected_count:
        raise AssertionError(
            f"baseline trained {len(baseline)} rows, expected {expected_count}"
        )
    expected_generations = set(range(generations_per_prompt))
    for train_step in range(1, steps + 1):
        prompts = {prompt_idx for step, prompt_idx, _ in baseline if step == train_step}
        if len(prompts) != prompts_per_step:
            raise AssertionError(
                f"step {train_step} trained {len(prompts)} prompts, "
                f"expected {prompts_per_step}"
            )
        for prompt_idx in prompts:
            generations = {
                generation_index
                for step, candidate_prompt_idx, generation_index in baseline
                if step == train_step and candidate_prompt_idx == prompt_idx
            }
            if generations != expected_generations:
                raise AssertionError(
                    "logical prompt has missing or duplicate generation indices: "
                    f"step={train_step}, prompt={prompt_idx}, "
                    f"generations={sorted(generations)!r}"
                )
    if set(baseline) != set(recovery):
        raise AssertionError(
            "logical prompt/generation step membership differs: "
            f"missing={sorted(set(baseline) - set(recovery))!r}, "
            f"unexpected={sorted(set(recovery) - set(baseline))!r}"
        )

    for identity in sorted(baseline):
        left, right = baseline[identity], recovery[identity]
        for field in _EXACT_FIELDS:
            if left.get(field) != right.get(field):
                raise AssertionError(f"row {identity!r} field {field!r} differs")
        for field in _APPROXIMATE_FIELDS:
            _assert_nested_close(
                left.get(field),
                right.get(field),
                path=f"row {identity!r} {field}",
                rtol=rtol,
                atol=atol,
            )


def _metric_at_step(metrics: Mapping[str, Any], name: str, step: int) -> Any:
    if name not in metrics:
        raise AssertionError(f"required metric is missing: {name}")
    values = metrics[name]
    if not isinstance(values, Mapping) or str(step) not in values:
        raise AssertionError(f"metric {name!r} has no value for step {step}")
    return values[str(step)]


def _compare_metrics(
    baseline: Mapping[str, Any],
    recovery: Mapping[str, Any],
    *,
    steps: int,
    rtol: float,
    atol: float,
) -> None:
    for name in ("train/loss", "train/reward"):
        for step in range(1, steps + 1):
            _assert_nested_close(
                _metric_at_step(baseline, name, step),
                _metric_at_step(recovery, name, step),
                path=f"metric {name} step {step}",
                rtol=rtol,
                atol=atol,
            )


def _assert_prefix_boundary(selection: Mapping[str, Any], recovery_log: str) -> None:
    completed = re.findall(
        r"generation prefix completed: rollout_id=(\S+) model_call_id=(\S+) "
        r"source_model_call_id=(\S+) prefix_tokens=(\d+) tail_tokens=(\d+) "
        r"total_generation_tokens=(\d+)",
        recovery_log,
    )
    matches = [
        match
        for match in completed
        if match[2] == selection["model_call_id"]
        and int(match[3]) == selection["generation_token_count"]
    ]
    if len(matches) != 1:
        raise AssertionError(
            "selected prefix did not complete exactly once: "
            f"source={selection['model_call_id']!r}, matches={matches!r}"
        )
    _, _, _, prefix_tokens, tail_tokens, total_tokens = matches[0]
    if int(tail_tokens) <= 0:
        raise AssertionError("restored generation produced no suffix tokens")
    if int(prefix_tokens) + int(tail_tokens) != int(total_tokens):
        raise AssertionError(
            "restored generation token accounting has a gap or overlap: "
            f"prefix={prefix_tokens}, tail={tail_tokens}, total={total_tokens}"
        )


def _assert_selected_prefix_was_trained(
    selection: Mapping[str, Any],
    recovery_records: Sequence[Mapping[str, Any]],
) -> None:
    matches = [
        row
        for batch in recovery_records
        for row in batch["rows"]
        if row["group_id"] == selection["group_id"]
        and row["generation_index"] == selection["generation_index"]
    ]
    if len(matches) != 1:
        raise AssertionError(
            "selected recovered prefix was not trained exactly once: "
            f"group={selection['group_id']!r}, "
            f"generation={selection['generation_index']!r}, matches={len(matches)}"
        )


def _assert_retired_prefix_keys(
    selection: Mapping[str, Any],
    recovery_records: Sequence[Mapping[str, Any]],
    successor_checkpoint: Path,
) -> None:
    """Require old prefix rows to be absent before the successor TQ snapshot."""

    old_keys = set(selection.get("staging_keys", []))
    if not old_keys:
        raise AssertionError("selected prefix has no staging keys")
    for batch in recovery_records:
        leaked = old_keys.intersection(batch.get("staging_sample_ids", []))
        if leaked:
            raise AssertionError(
                "restored prefix rows were still live when training began: "
                f"{sorted(leaked)!r}"
            )

    manifest_path = successor_checkpoint / "gym_checkpoint.json"
    manifest = _read_json(manifest_path)
    referenced = {
        key for keys in manifest.get("staging_keys", {}).values() for key in keys
    }
    leaked = old_keys & referenced
    if leaked:
        raise AssertionError(
            "successor Gym checkpoint retained obsolete prefix keys: "
            f"{sorted(leaked)!r}"
        )


def compare(args: argparse.Namespace) -> None:
    baseline_records = _read_jsonl(args.baseline_training)
    recovery_records = _read_jsonl(args.recovery_training)
    selection = _read_json(args.selection)
    _compare_training_rows(
        baseline_records,
        recovery_records,
        steps=args.steps,
        prompts_per_step=args.prompts_per_step,
        generations_per_prompt=args.generations_per_prompt,
        rtol=args.rtol,
        atol=args.atol,
    )
    _compare_metrics(
        _read_json(args.baseline_metrics),
        _read_json(args.recovery_metrics),
        steps=args.steps,
        rtol=args.rtol,
        atol=args.atol,
    )
    _assert_prefix_boundary(selection, args.recovery_log.read_text())
    _assert_selected_prefix_was_trained(selection, recovery_records)
    _assert_retired_prefix_keys(selection, recovery_records, args.successor_checkpoint)
    print(
        "validated uninterrupted-vs-prefix-recovery parity: "
        f"rows={args.steps * args.prompts_per_step * args.generations_per_prompt}",
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-training", type=Path, required=True)
    parser.add_argument("--recovery-training", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--recovery-metrics", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--recovery-log", type=Path, required=True)
    parser.add_argument("--successor-checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--prompts-per-step", type=int, required=True)
    parser.add_argument("--generations-per-prompt", type=int, required=True)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser


if __name__ == "__main__":
    compare(_parser().parse_args())
