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
import hashlib
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
_EXCUSED_ROW_EXACT_FIELDS = ("reward", "sample_loss_mask")
_APPROXIMATE_FIELDS = (
    "advantages",
    "prev_logprobs",
)
# vLLM sampling logprobs are not batch-invariant: two runs that choose the same
# tokens still differ by bf16 rounding on a few of them. They are gated in
# aggregate (see _assert_generation_logprob_agreement), not elementwise.
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
    excused: frozenset[tuple[int, int, int]] = frozenset(),
) -> None:
    """Compare rows by logical identity.

    ``excused`` rows diverged inside a restored prefix: the restore replayed
    tokens phase one generated before the crash, so their difference from the
    baseline is run-to-run sampling noise, not a recovery error.
    """
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
        if identity in excused:
            # Tokens differ by phase-one sampling noise, but the verifier and
            # the row's admission must still agree.
            for field in _EXCUSED_ROW_EXACT_FIELDS:
                if left.get(field) != right.get(field):
                    raise AssertionError(f"row {identity!r} field {field!r} differs")
            continue
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


def _selection_cuts(selection: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    # A selection written before multi-cut support describes exactly one cut.
    return list(selection["cuts"]) if "cuts" in selection else [selection]


def _assert_prefix_boundary(
    selection: Mapping[str, Any],
    recovery_log: str,
    recovery_records: Sequence[Mapping[str, Any]],
) -> None:
    """Require each restored call to complete once with its tokens all trained.

    The worker logs ``total = prefix + tail`` by construction, so the total is
    checked against the generated tokens of the trained row's closing segment:
    a dropped or duplicated token changes that count.
    """
    rows = {
        (row["group_id"], int(row["generation_index"])): row
        for batch in recovery_records
        for row in batch["rows"]
    }
    completed = re.findall(
        r"generation prefix completed: rollout_id=(\S+) model_call_id=(\S+) "
        r"source_model_call_id=(\S+) prefix_tokens=(\d+) tail_tokens=(\d+) "
        r"total_generation_tokens=(\d+)",
        recovery_log,
    )
    for cut in _selection_cuts(selection):
        matches = [
            match
            for match in completed
            if match[2] == cut["model_call_id"]
            and int(match[3]) == cut["generation_token_count"]
        ]
        if len(matches) != 1:
            raise AssertionError(
                "restored prefix did not complete exactly once: "
                f"source={cut['model_call_id']!r}, matches={matches!r}"
            )
        _, _, _, prefix_tokens, tail_tokens, total_tokens = matches[0]
        if int(tail_tokens) <= 0:
            raise AssertionError("restored generation produced no suffix tokens")
        if int(prefix_tokens) + int(tail_tokens) != int(total_tokens):
            raise AssertionError(
                "restored generation token accounting has a gap or overlap: "
                f"prefix={prefix_tokens}, tail={tail_tokens}, total={total_tokens}"
            )
        row = rows.get((cut["group_id"], int(cut["generation_index"])))
        if row is None:
            raise AssertionError(
                f"restored call {cut['model_call_id']!r} has no trained row"
            )
        mask = row["token_loss_mask"]
        segment_start = _last_generated_segment_start(mask)
        trained = 0 if segment_start is None else int(sum(mask[segment_start:]))
        if trained != int(total_tokens):
            raise AssertionError(
                "restored call's trained tokens differ from its generation: "
                f"model_call_id={cut['model_call_id']!r} "
                f"trained={trained} generated={total_tokens}"
            )


def _assert_selected_prefix_was_trained(
    selection: Mapping[str, Any],
    recovery_records: Sequence[Mapping[str, Any]],
) -> None:
    for cut in _selection_cuts(selection):
        matches = [
            row
            for batch in recovery_records
            for row in batch["rows"]
            if row["group_id"] == cut["group_id"]
            and row["generation_index"] == cut["generation_index"]
        ]
        if len(matches) != 1:
            raise AssertionError(
                "recovered prefix was not trained exactly once: "
                f"group={cut['group_id']!r}, "
                f"generation={cut['generation_index']!r}, matches={len(matches)}"
            )


def _assert_retired_prefix_keys(
    selection: Mapping[str, Any],
    recovery_records: Sequence[Mapping[str, Any]],
    successor_checkpoint: Path,
) -> None:
    """Require old prefix rows to be absent before the successor TQ snapshot."""

    old_keys: set[str] = set()
    for cut in _selection_cuts(selection):
        keys = set(cut.get("staging_keys", []))
        if not keys:
            raise AssertionError(
                f"restored prefix {cut['model_call_id']!r} has no staging keys"
            )
        old_keys |= keys
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


def _last_generated_segment_start(token_loss_mask: Sequence[float]) -> int | None:
    start = None
    for index, keep in enumerate(token_loss_mask):
        if keep and (index == 0 or not token_loss_mask[index - 1]):
            start = index
    return start


def _restored_prefix_hashes(recovery_log: str) -> dict[str, str]:
    """Map each restored source call to the hash of its generated prefix IDs."""
    return dict(
        re.findall(
            r"generation prefix restored: .*?source_model_call_id=(\S+) "
            r".*?prefix_ids_sha256=([0-9a-f]{64})",
            recovery_log,
        )
    )


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(",".join(map(str, token_ids)).encode()).hexdigest()


def _classify_token_divergence(
    baseline_records: Sequence[Mapping[str, Any]],
    recovery_records: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    recovery_log: str,
) -> dict[tuple[int, int, int], dict[str, Any]]:
    """Locate each row's first token difference relative to its restored prefix.

    A cut is always the closing model call, the last generated segment of the
    row. Kinds:

    - ``pre_generation``: the prompt differs, which no sampling noise explains.
    - ``pre_cut``: the difference lies inside a restored prefix whose trained
      tokens hash-match what the worker restored, i.e. phase one sampled
      differently before the crash and the restore replayed it faithfully.
    - ``prefix_mismatch``: inside the restored prefix, but the trained tokens
      are not the restored ones (dropped, duplicated, or substituted).
    - ``post_cut``: generated by the restored continuation.
    - ``unrestored``: a row without a cut.

    Only ``pre_cut`` is excusable.
    """

    baseline = _rows_by_identity(baseline_records)
    recovery = _rows_by_identity(recovery_records)
    restored_hashes = _restored_prefix_hashes(recovery_log)
    cuts = {
        (cut["group_id"], int(cut["generation_index"])): cut
        for cut in _selection_cuts(selection)
    }
    result: dict[tuple[int, int, int], dict[str, Any]] = {}
    for identity in sorted(set(baseline) & set(recovery)):
        left = baseline[identity]["token_ids"]
        right = recovery[identity]["token_ids"]
        first_diff = next(
            (i for i, (x, y) in enumerate(zip(left, right)) if x != y),
            None if len(left) == len(right) else min(len(left), len(right)),
        )
        row = recovery[identity]
        mask = row["token_loss_mask"]
        first_generated = next((i for i, keep in enumerate(mask) if keep), len(mask))
        cut = cuts.get((row["group_id"], int(row["generation_index"])))
        prefix_end = None
        prefix_hash_matches = None
        if cut is not None:
            segment_start = _last_generated_segment_start(mask)
            if segment_start is not None:
                prefix_end = segment_start + int(cut["generation_token_count"])
                logged = restored_hashes.get(cut["model_call_id"])
                prefix_hash_matches = logged is not None and logged == (
                    _token_ids_sha256(right[segment_start:prefix_end])
                )
        if first_diff is None:
            kind = "match"
        elif first_diff < first_generated:
            kind = "pre_generation"
        elif prefix_end is None:
            kind = "unrestored"
        elif first_diff >= prefix_end:
            kind = "post_cut"
        else:
            kind = "pre_cut" if prefix_hash_matches else "prefix_mismatch"
        result[identity] = {
            "kind": kind,
            "first_diff": first_diff,
            "restored_prefix_end": prefix_end,
            "restored_prefix_hash_matches": prefix_hash_matches,
        }
    return result


def _assert_reward_signal(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require rewards that can expose a mismatch, and report advantage support."""

    rows = _rows_by_identity(records)
    rewards = {identity: float(row["reward"]) for identity, row in rows.items()}
    if len(set(rewards.values())) < 2:
        raise AssertionError(
            "baseline rewards are constant, so reward parity cannot detect a "
            f"verifier divergence: rewards={sorted(set(rewards.values()))!r}"
        )
    nonzero_advantages = sum(
        any(value != 0 for value in row["advantages"]) for row in rows.values()
    )
    return {
        "rewards": {
            repr(identity): reward for identity, reward in sorted(rewards.items())
        },
        "rows_with_nonzero_advantages": nonzero_advantages,
        "rows": len(rows),
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            ranks[order[position]] = (start + end) / 2.0
        start = end + 1
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    count = len(left)
    if count < 2:
        return None
    left_mean = sum(left) / count
    right_mean = sum(right) / count
    covariance = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_norm == 0 or right_norm == 0:
        return None
    return covariance / (left_norm * right_norm)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1) + 0.5))]


def _generation_logprob_report(
    baseline_records: Sequence[Mapping[str, Any]],
    recovery_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize generated-token logprob agreement over token-matched rows.

    A row whose tokens diverged compares logprobs of different tokens after the
    divergence, so it is listed in ``excluded_rows`` instead of pooled.
    """

    baseline = _rows_by_identity(baseline_records)
    recovery = _rows_by_identity(recovery_records)
    per_row: dict[str, Any] = {}
    excluded: list[str] = []
    pooled_left: list[float] = []
    pooled_right: list[float] = []
    for identity in sorted(set(baseline) & set(recovery)):
        if baseline[identity]["token_ids"] != recovery[identity]["token_ids"]:
            excluded.append(repr(identity))
            continue
        mask = baseline[identity]["token_loss_mask"]
        left = [
            float(value)
            for value, keep in zip(
                baseline[identity]["generation_logprobs"], mask, strict=True
            )
            if keep
        ]
        right = [
            float(value)
            for value, keep in zip(
                recovery[identity]["generation_logprobs"], mask, strict=True
            )
            if keep
        ]
        if len(left) != len(right) or not left:
            continue
        absolute = [abs(x - y) for x, y in zip(left, right, strict=True)]
        relative = [
            100.0 * diff / max(abs(x), 1e-12)
            for x, diff in zip(left, absolute, strict=True)
        ]
        per_row[repr(identity)] = {
            "generated_tokens": len(left),
            "spearman": _pearson(_average_ranks(left), _average_ranks(right)),
            "pearson": _pearson(left, right),
            "bit_identical_fraction": sum(d == 0 for d in absolute) / len(left),
            "max_abs_diff": max(absolute),
            "median_pct_diff": _percentile(relative, 0.5),
            "p95_pct_diff": _percentile(relative, 0.95),
            "sequence_logprob_baseline": sum(left),
            "sequence_logprob_recovery": sum(right),
        }
        pooled_left.extend(left)
        pooled_right.extend(right)
    pooled: dict[str, Any] = {"generated_tokens": len(pooled_left)}
    if pooled_left:
        absolute = [abs(x - y) for x, y in zip(pooled_left, pooled_right, strict=True)]
        uncertain = [
            100.0 * diff / abs(x)
            for x, diff in zip(pooled_left, absolute, strict=True)
            if abs(x) > 0.1
        ]
        pooled.update(
            spearman=_pearson(
                _average_ranks(pooled_left), _average_ranks(pooled_right)
            ),
            pearson=_pearson(pooled_left, pooled_right),
            bit_identical_fraction=sum(d == 0 for d in absolute) / len(absolute),
            max_abs_diff=max(absolute),
            median_abs_diff=_percentile(absolute, 0.5),
            p95_abs_diff=_percentile(absolute, 0.95),
            uncertain_tokens=len(uncertain),
            uncertain_median_pct_diff=_percentile(uncertain, 0.5)
            if uncertain
            else None,
            uncertain_p95_pct_diff=_percentile(uncertain, 0.95) if uncertain else None,
        )
    return {"pooled": pooled, "per_row": per_row, "excluded_rows": excluded}


def _assert_generation_logprob_agreement(
    pooled: Mapping[str, Any],
    *,
    min_bit_identical: float,
    min_spearman: float,
    max_abs_diff: float,
) -> None:
    """Gate vLLM sampling logprobs on aggregate agreement over matched tokens."""

    if not pooled.get("generated_tokens"):
        raise AssertionError("no token-matched rows to compare generation logprobs")
    failures = []
    if pooled["bit_identical_fraction"] < min_bit_identical:
        failures.append(
            f"bit_identical_fraction={pooled['bit_identical_fraction']} < {min_bit_identical}"
        )
    spearman = pooled["spearman"]
    if spearman is None or spearman < min_spearman:
        failures.append(f"spearman={spearman} < {min_spearman}")
    if pooled["max_abs_diff"] > max_abs_diff:
        failures.append(f"max_abs_diff={pooled['max_abs_diff']} > {max_abs_diff}")
    if failures:
        raise AssertionError(
            "generation_logprobs disagree beyond bf16 noise: " + ", ".join(failures)
        )


def _completion_order(
    events: Sequence[Mapping[str, Any]], *, event_name: str, steps: int
) -> list[tuple[int, int, int]]:
    """Return ``(target_step, prompt_idx, generation_index)`` in event order."""

    order: list[tuple[int, int, int]] = []
    for event in events:
        if event.get("event") != event_name:
            continue
        target_step = event.get("target_step")
        if not isinstance(target_step, int) or not 0 <= target_step < steps:
            continue
        generation_index = (
            event["sibling"]["generation_index"]
            if event_name == "completion"
            else event["generation_index"]
        )
        order.append((target_step, int(event["prompt_idx"]), int(generation_index)))
    return order


def _effective_recovery_order(
    phase1_order: Sequence[tuple[int, int, int]],
    phase2_order: Sequence[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """Merge the two recovery processes into the order rows became available.

    A phase-one completion that phase two did not repeat survived the crash in
    the restored snapshot and was available before any phase-two completion. A
    repeated identity was discarded with the crash, so its phase-two arrival is
    the one training used.
    """

    redone = set(phase2_order)
    retained = [identity for identity in phase1_order if identity not in redone]
    return [*dict.fromkeys(retained), *phase2_order]


def _prompt_group_ready_order(
    completion_order: Sequence[tuple[int, int, int]],
) -> list[tuple[int, int]]:
    """Order prompt groups by the position of their last completed sibling."""

    last_rank: dict[tuple[int, int], int] = {}
    for rank, identity in enumerate(completion_order):
        last_rank[identity[:2]] = rank
    return sorted(last_rank, key=last_rank.__getitem__)


def _within_step_rank_metrics(
    baseline_order: Sequence[tuple[int, ...]],
    recovery_order: Sequence[tuple[int, ...]],
) -> dict[str, Any]:
    """Kendall tau and inversions over pairs that share a target step."""

    if set(baseline_order) != set(recovery_order):
        raise AssertionError(
            "rank parity requires identical logical identities: "
            f"missing={sorted(set(baseline_order) - set(recovery_order))!r}, "
            f"unexpected={sorted(set(recovery_order) - set(baseline_order))!r}"
        )
    recovery_rank = {identity: rank for rank, identity in enumerate(recovery_order)}
    concordant = discordant = 0
    per_step = []
    for target_step in sorted({identity[0] for identity in baseline_order}):
        baseline_step = [i for i in baseline_order if i[0] == target_step]
        recovery_step = [i for i in recovery_order if i[0] == target_step]
        step_discordant = 0
        for index, left in enumerate(baseline_step):
            for right in baseline_step[index + 1 :]:
                if recovery_rank[left] < recovery_rank[right]:
                    concordant += 1
                else:
                    discordant += 1
                    step_discordant += 1
        per_step.append(
            {
                "target_step": target_step,
                "baseline_order": baseline_step,
                "recovery_order": recovery_step,
                "exact_match": baseline_step == recovery_step,
                "discordant_pairs": step_discordant,
            }
        )
    comparable = concordant + discordant
    return {
        "exact_match": all(step["exact_match"] for step in per_step),
        "kendall_tau": (concordant - discordant) / comparable if comparable else None,
        "inversion_rate": discordant / comparable if comparable else None,
        "discordant_pairs": discordant,
        "comparable_pairs": comparable,
        "per_step": per_step,
    }


def _rollout_timeline(
    events: Sequence[Mapping[str, Any]], *, steps: int
) -> dict[str, dict[str, float]]:
    """Seconds since the run's first dispatch at which each rollout moved."""

    timed = [event for event in events if isinstance(event.get("time_s"), (int, float))]
    if not timed:
        return {}
    origin = min(event["time_s"] for event in timed)
    timeline: dict[str, dict[str, float]] = {}
    for event in timed:
        target_step = event.get("target_step")
        if not isinstance(target_step, int) or not 0 <= target_step < steps:
            continue
        name = event["event"]
        if name == "completion":
            indices = [event["sibling"]["generation_index"]]
        elif name == "completion_arrived":
            indices = [event["generation_index"]]
        else:
            indices = event.get("generation_indices", [])
        for generation_index in indices:
            key = repr((target_step, int(event["prompt_idx"]), int(generation_index)))
            # Keep the latest occurrence: a re-dispatch supersedes the original.
            timeline.setdefault(key, {})[name] = round(event["time_s"] - origin, 3)
    return dict(sorted(timeline.items()))


def _ordering_report(
    *,
    baseline_events: Sequence[Mapping[str, Any]],
    phase1_events: Sequence[Mapping[str, Any]],
    recovery_events: Sequence[Mapping[str, Any]],
    steps: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for label, event_name in (
        ("sealed", "completion"),
        ("arrived", "completion_arrived"),
    ):
        baseline = _completion_order(
            baseline_events, event_name=event_name, steps=steps
        )
        recovery = _effective_recovery_order(
            _completion_order(phase1_events, event_name=event_name, steps=steps),
            _completion_order(recovery_events, event_name=event_name, steps=steps),
        )
        report[label] = {
            "individual": _within_step_rank_metrics(baseline, recovery),
            "prompt_group_ready": _within_step_rank_metrics(
                _prompt_group_ready_order(baseline),
                _prompt_group_ready_order(recovery),
            ),
        }
    return report


def _restored_target_steps(
    selection: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> set[int]:
    """Target steps whose prompt groups held a restored generation prefix."""

    group_steps = {
        event["group_id"]: event["target_step"]
        for event in events
        if "group_id" in event and isinstance(event.get("target_step"), int)
    }
    steps = set()
    for cut in _selection_cuts(selection):
        if cut["group_id"] not in group_steps:
            raise AssertionError(
                f"restored group {cut['group_id']!r} has no recorded target step"
            )
        steps.add(group_steps[cut["group_id"]])
    return steps


def _assert_restored_prompt_group_order(
    ordering: Mapping[str, Any], restored_steps: set[int]
) -> None:
    """Require baseline prompt-group readiness order in restored steps.

    Later steps are generated wholly after the restore. Their order follows
    when Gym's checkpoint admission happened to refuse and re-send each
    sibling, which differs between any two runs; they stay reported only.
    """

    for label, metrics in ordering.items():
        differing = [
            step
            for step in metrics["prompt_group_ready"]["per_step"]
            if step["target_step"] in restored_steps and not step["exact_match"]
        ]
        if differing:
            raise AssertionError(
                f"{label} prompt-group readiness order differs in a restored "
                f"step: {differing!r}"
            )


def compare(args: argparse.Namespace) -> None:
    baseline_records = _read_jsonl(args.baseline_training)
    recovery_records = _read_jsonl(args.recovery_training)
    selection = _read_json(args.selection)

    # Write every diagnostic before the strict checks so a failing run still
    # shows where ordering, timing, and logprobs landed.
    baseline_events = _read_jsonl(args.baseline_events)
    phase1_events = _read_jsonl(args.phase1_events)
    recovery_events = _read_jsonl(args.recovery_events)
    recovery_log = args.recovery_log.read_text()
    divergence = _classify_token_divergence(
        baseline_records, recovery_records, selection, recovery_log
    )
    ordering = _ordering_report(
        baseline_events=baseline_events,
        phase1_events=phase1_events,
        recovery_events=recovery_events,
        steps=args.steps,
    )
    report = {
        "ordering": ordering,
        "timeline": {
            "baseline": _rollout_timeline(baseline_events, steps=args.steps),
            "phase1": _rollout_timeline(phase1_events, steps=args.steps),
            "recovery": _rollout_timeline(recovery_events, steps=args.steps),
        },
        "generation_logprobs": _generation_logprob_report(
            baseline_records, recovery_records
        ),
        "restored_prefixes": [
            {
                key: cut[key]
                for key in ("group_id", "generation_index", "generation_token_count")
            }
            for cut in _selection_cuts(selection)
        ],
        "token_divergence": {
            repr(identity): classification
            for identity, classification in divergence.items()
        },
    }
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for label, metrics in ordering.items():
        for scope in ("individual", "prompt_group_ready"):
            values = metrics[scope]
            print(
                f"{label} {scope} order: exact_match={values['exact_match']} "
                f"kendall_tau={values['kendall_tau']} "
                f"inversions={values['discordant_pairs']}/{values['comparable_pairs']}",
                flush=True,
            )
    pooled = report["generation_logprobs"]["pooled"]
    print(
        "generation_logprobs: "
        + " ".join(f"{key}={value}" for key, value in sorted(pooled.items())),
        flush=True,
    )
    print(
        f"restored prefixes: {len(report['restored_prefixes'])} "
        + str(
            sorted(cut["generation_token_count"] for cut in report["restored_prefixes"])
        ),
        flush=True,
    )
    for identity, classification in divergence.items():
        if classification["kind"] != "match":
            print(f"token divergence {identity!r}: {classification}", flush=True)
    print(f"parity report written to {args.report_output}", flush=True)

    reward_signal = _assert_reward_signal(baseline_records)
    print(
        "reward signal: rewards per row="
        f"{reward_signal['rewards']}, rows with nonzero advantages="
        f"{reward_signal['rows_with_nonzero_advantages']}/{reward_signal['rows']}",
        flush=True,
    )
    if args.require_prompt_group_order:
        _assert_restored_prompt_group_order(
            ordering,
            _restored_target_steps(selection, [*phase1_events, *recovery_events]),
        )

    _compare_training_rows(
        baseline_records,
        recovery_records,
        steps=args.steps,
        prompts_per_step=args.prompts_per_step,
        generations_per_prompt=args.generations_per_prompt,
        rtol=args.rtol,
        atol=args.atol,
        excused=frozenset(
            identity
            for identity, classification in divergence.items()
            if classification["kind"] == "pre_cut"
        ),
    )
    _assert_generation_logprob_agreement(
        report["generation_logprobs"]["pooled"],
        min_bit_identical=args.logprob_min_bit_identical,
        min_spearman=args.logprob_min_spearman,
        max_abs_diff=args.logprob_max_abs_diff,
    )
    _compare_metrics(
        _read_json(args.baseline_metrics),
        _read_json(args.recovery_metrics),
        steps=args.steps,
        rtol=args.rtol,
        atol=args.atol,
    )
    _assert_prefix_boundary(selection, recovery_log, recovery_records)
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
    parser.add_argument("--baseline-events", type=Path, required=True)
    parser.add_argument("--phase1-events", type=Path, required=True)
    parser.add_argument("--recovery-events", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    # Prompt groups in this workload have schema-fixed lengths far apart, so
    # their readiness order in a restored step is deterministic; sibling order
    # within a group, and group order in later steps, are not.
    parser.add_argument("--require-prompt-group-order", action="store_true")
    # Calibrated on four Qwen3-1.7B runs of this workload (~28k generated
    # tokens): 99.2-99.3% bit-identical, max |diff| 0.25-0.30, spearman
    # 0.982-0.996. Spearman ranks only the ~0.7% of tokens that differ, so it
    # is the noisiest of the three and gets the widest margin.
    parser.add_argument("--logprob-min-bit-identical", type=float, default=0.99)
    parser.add_argument("--logprob-min-spearman", type=float, default=0.97)
    parser.add_argument("--logprob-max-abs-diff", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser


if __name__ == "__main__":
    compare(_parser().parse_args())
