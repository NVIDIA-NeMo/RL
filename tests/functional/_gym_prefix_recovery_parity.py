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
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from _gym_recovery_snapshot_helpers import read_json, read_jsonl, restored_prefixes

_EXACT_FIELDS = (
    "input_length",
    "token_ids",
    "token_loss_mask",
    "sample_loss_mask",
    "reward",
)
# prev_logprobs are not compared: training applies the run's top_p, which
# keeps one token, so they are 0 in both runs.
_APPROXIMATE_FIELDS = ("advantages",)
# A restored row whose phase-1 prefix already differed from the reference keeps
# only these: its prompt verdict and admission, not its tokens.
_DIVERGED_ROW_EXACT_FIELDS = ("reward", "sample_loss_mask")
# vLLM sampling logprobs are not batch-invariant: two runs that choose the same
# tokens still differ by bf16 rounding on a few of them. They are gated in
# aggregate (see _assert_generation_logprob_agreement), not elementwise.


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


# Per-token columns the trainer's dump writes for each row.
_DUMPED_SEQUENCE_FIELDS = (
    "token_ids",
    "token_loss_mask",
    "advantages",
    "generation_logprobs",
    "prev_logprobs",
)


def _training_row(dumped: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one ``train_data_step*.jsonl`` row to the compared row shape."""
    (sample_id,) = dumped["sample_id"]
    (length,) = dumped["input_lengths"]
    (tags,) = dumped["metadata"]
    group_id, generation_index = _generation_identity(sample_id)
    prompt_idx = tags.get("prompt_idx")
    if not isinstance(prompt_idx, int):
        raise TypeError(f"sample {sample_id!r} has no integer prompt_idx tag")
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "group_id": group_id,
        "prompt_idx": prompt_idx,
        "generation_index": generation_index,
        "input_length": int(length),
        "sample_loss_mask": float(dumped["sample_loss_mask"][0]),
        "reward": float(dumped["rewards"][0]),
    }
    for field in _DUMPED_SEQUENCE_FIELDS:
        if field in dumped:
            row[field] = dumped[field][0]
    return row


def load_training_records(
    log_dir: Path, staging_keys: Path | None = None
) -> list[dict[str, Any]]:
    """Load the rows a run trained on from its ``async_rl.log_full_train_data`` dump.

    ``log_dir`` is the ``logger.log_dir`` the run was given; the entrypoint
    writes into a new ``exp_NNN`` directory under it, and each run here gets its
    own ``log_dir``, so there must be exactly one.

    Returns one record per train step. ``staging_keys``, written by the test
    hook, adds the token-capture staging keys that were live before each
    training chunk of that step.
    """
    live_keys: dict[int, set[str]] = {}
    if staging_keys is not None:
        for probe in read_jsonl(staging_keys):
            live_keys.setdefault(int(probe["train_step"]), set()).update(
                probe["staging_sample_ids"]
            )
    run_dirs = sorted(log_dir.glob("exp_*"))
    if len(run_dirs) != 1:
        raise AssertionError(
            f"expected one experiment directory under {log_dir}, found "
            f"{[run_dir.name for run_dir in run_dirs]!r}"
        )
    (run_dir,) = run_dirs
    records = []
    for path in sorted(run_dir.glob("train_data_step*.jsonl")):
        dumped_rows = read_jsonl(path)
        steps = {int(row["step"]) for row in dumped_rows}
        if len(steps) != 1:
            raise AssertionError(f"{path} mixes train steps {sorted(steps)!r}")
        (train_step,) = steps
        records.append(
            {
                "train_step": train_step,
                "rows": [_training_row(row) for row in dumped_rows],
                "staging_sample_ids": sorted(live_keys.get(train_step, ())),
            }
        )
    if not records:
        raise AssertionError(f"{run_dir} holds no train_data_step*.jsonl dump")
    return sorted(records, key=lambda record: record["train_step"])


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
    diverged_before_cut: frozenset[tuple[int, int, int]] = frozenset(),
) -> None:
    """Compare rows by logical identity; every row must match exactly.

    The only exception is ``diverged_before_cut`` (see
    ``_rows_diverged_before_cut``): those rows keep their reward and admission
    exact, since the reference does not predict their tokens.
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
        if identity in diverged_before_cut:
            for field in _DIVERGED_ROW_EXACT_FIELDS:
                if left.get(field) != right.get(field):
                    raise AssertionError(f"row {identity!r} field {field!r} differs")
            continue
        for field in _EXACT_FIELDS:
            a, b = left.get(field), right.get(field)
            if a != b:
                where = ""
                if isinstance(a, list) and isinstance(b, list):
                    first = next(
                        (i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                        min(len(a), len(b)),
                    )
                    where = f" (first difference at index {first})"
                raise AssertionError(f"row {identity!r} field {field!r} differs{where}")
        for field in _APPROXIMATE_FIELDS:
            _assert_nested_close(
                left.get(field),
                right.get(field),
                path=f"row {identity!r} {field}",
                rtol=rtol,
                atol=atol,
            )


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
    for cut in selection["cuts"]:
        matches = [
            match
            for match in completed
            if match[2] == cut["model_call_id"]
            and int(match[3]) == cut["prefix_token_count"]
        ]
        if len(matches) != 1:
            raise AssertionError(
                "restored prefix did not complete exactly once: "
                f"source={cut['model_call_id']!r}, matches={matches!r}"
            )
        _, _, _, _, tail_tokens, total_tokens = matches[0]
        if int(tail_tokens) <= 0:
            raise AssertionError("restored generation produced no suffix tokens")
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


def _assert_retired_prefix_keys(
    selection: Mapping[str, Any],
    recovery_records: Sequence[Mapping[str, Any]],
) -> None:
    """Require old prefix rows to be absent from every batch the trainer consumed."""

    old_keys: set[str] = set()
    for cut in selection["cuts"]:
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


def _last_generated_segment_start(token_loss_mask: Sequence[float]) -> int | None:
    start = None
    for index, keep in enumerate(token_loss_mask):
        if keep and (index == 0 or not token_loss_mask[index - 1]):
            start = index
    return start


def _closing_prefix_sha256(row: Mapping[str, Any], prefix_tokens: int) -> str:
    """Hash a row's first closing-call tokens the way the restore log does."""
    start = _last_generated_segment_start(row["token_loss_mask"])
    generated = [] if start is None else row["token_ids"][start : start + prefix_tokens]
    return hashlib.sha256(",".join(map(str, generated)).encode()).hexdigest()


def _rows_diverged_before_cut(
    selection: Mapping[str, Any],
    recovery_log: str,
    baseline_records: Sequence[Mapping[str, Any]],
    recovery_records: Sequence[Mapping[str, Any]],
) -> frozenset[tuple[int, int, int]]:
    """Return restored rows whose phase-1 prefix already differed from the reference.

    vLLM sampling is not batch-invariant, so phase 1 can take the other side of
    a near-tie before the cut. Recovery must then continue phase 1's tokens, and
    the reference no longer predicts that row. The restore logs a hash of the
    exact prefix it restored: every restored row must start with those tokens,
    and a row is set aside only when the reference's opening tokens hash
    differently. Everything before the closing call must still match, and at
    least one restored row must match the reference outright, so a capture that
    recorded the wrong tokens for every cut cannot pass.
    """
    restored = restored_prefixes(recovery_log)
    baseline = _rows_by_identity(baseline_records)
    recovery = _rows_by_identity(recovery_records)
    by_call = {
        (row["group_id"], int(row["generation_index"])): identity
        for identity, row in recovery.items()
    }
    diverged: set[tuple[int, int, int]] = set()
    for cut in selection["cuts"]:
        prefix_tokens = int(cut["prefix_token_count"])
        hashes = [
            line["prefix_ids_sha256"]
            for line in restored
            if line["source_model_call_id"] == cut["model_call_id"]
            and line["prefix_tokens"] == prefix_tokens
        ]
        if len(hashes) != 1:
            raise AssertionError(
                f"restored prefix {cut['model_call_id']!r} was logged "
                f"{len(hashes)} times, expected once"
            )
        identity = by_call.get((cut["group_id"], int(cut["generation_index"])))
        if identity is None:
            raise AssertionError(
                f"restored call {cut['model_call_id']!r} has no trained row"
            )
        right = recovery[identity]
        if _closing_prefix_sha256(right, prefix_tokens) != hashes[0]:
            raise AssertionError(
                f"row {identity!r} does not start with its {prefix_tokens} "
                "restored prefix tokens"
            )
        left = baseline.get(identity)
        if left is None or _closing_prefix_sha256(left, prefix_tokens) == hashes[0]:
            continue
        start = _last_generated_segment_start(left["token_loss_mask"])
        if (
            start != _last_generated_segment_start(right["token_loss_mask"])
            or left["token_ids"][:start] != right["token_ids"][:start]
            or left["token_loss_mask"][:start] != right["token_loss_mask"][:start]
        ):
            raise AssertionError(
                f"row {identity!r} differs before its restored closing call"
            )
        diverged.add(identity)
    if diverged and len(diverged) == len(selection["cuts"]):
        raise AssertionError(
            "every restored prefix differed from the reference, so no restored "
            "row checks the regenerated tail"
        )
    return frozenset(diverged)


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


def _assert_generation_logprob_agreement(
    baseline_records: Sequence[Mapping[str, Any]],
    recovery_records: Sequence[Mapping[str, Any]],
    *,
    min_bit_identical: float,
    max_abs_diff: float,
) -> None:
    """Gate vLLM sampling logprobs over generated tokens of token-equal rows.

    Two numbers make the gate: how many tokens disagree at all, and by how much
    the worst one does.
    """
    baseline = _rows_by_identity(baseline_records)
    recovery = _rows_by_identity(recovery_records)
    diffs: list[float] = []
    for identity, left in baseline.items():
        right = recovery[identity]
        if left["token_ids"] != right["token_ids"]:
            continue
        diffs.extend(
            abs(a - b)
            for a, b, keep in zip(
                left["generation_logprobs"],
                right["generation_logprobs"],
                left["token_loss_mask"],
                strict=True,
            )
            if keep
        )
    if not diffs:
        raise AssertionError("no token-matched rows to compare generation logprobs")
    identical = sum(diff == 0 for diff in diffs) / len(diffs)
    worst = max(diffs)
    summary = (
        f"bit_identical={identical:.4f}, max_abs_diff={worst:.3f}, tokens={len(diffs)}"
    )
    print(f"generation_logprobs: {summary}", flush=True)
    if identical < min_bit_identical or worst > max_abs_diff:
        raise AssertionError(
            f"generation_logprobs disagree beyond bf16 noise: {summary}"
        )


def compare(args: argparse.Namespace) -> None:
    baseline_records = load_training_records(args.baseline_log_dir)
    recovery_records = load_training_records(
        args.recovery_log_dir, staging_keys=args.recovery_staging_keys
    )
    selection = read_json(args.selection)

    recovery_log = args.recovery_log.read_text()
    cuts = selection["cuts"]
    print(
        f"restored prefixes: {len(cuts)} "
        + str(sorted(int(cut["prefix_token_count"]) for cut in cuts)),
        flush=True,
    )

    reward_signal = _assert_reward_signal(baseline_records)
    print(
        "reward signal: rewards per row="
        f"{reward_signal['rewards']}, rows with nonzero advantages="
        f"{reward_signal['rows_with_nonzero_advantages']}/{reward_signal['rows']}",
        flush=True,
    )
    diverged = _rows_diverged_before_cut(
        selection, recovery_log, baseline_records, recovery_records
    )
    print(
        "restored rows whose phase-1 prefix differed from the reference: "
        f"{sorted(diverged)!r}",
        flush=True,
    )
    _compare_training_rows(
        baseline_records,
        recovery_records,
        steps=args.steps,
        prompts_per_step=args.prompts_per_step,
        generations_per_prompt=args.generations_per_prompt,
        rtol=args.rtol,
        atol=args.atol,
        diverged_before_cut=diverged,
    )
    _assert_generation_logprob_agreement(
        baseline_records,
        recovery_records,
        min_bit_identical=args.logprob_min_bit_identical,
        max_abs_diff=args.logprob_max_abs_diff,
    )
    _assert_prefix_boundary(selection, recovery_log, recovery_records)
    _assert_retired_prefix_keys(selection, recovery_records)
    print(
        "validated uninterrupted-vs-prefix-recovery parity: "
        f"rows={args.steps * args.prompts_per_step * args.generations_per_prompt}",
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log-dir", type=Path, required=True)
    parser.add_argument("--recovery-log-dir", type=Path, required=True)
    parser.add_argument("--recovery-staging-keys", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--recovery-log", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--prompts-per-step", type=int, required=True)
    parser.add_argument("--generations-per-prompt", type=int, required=True)
    # Calibrated on four Qwen3-1.7B runs of this workload (~28k generated
    # tokens): 99.2-99.3% bit-identical, max |diff| 0.25-0.30.
    parser.add_argument("--logprob-min-bit-identical", type=float, default=0.99)
    parser.add_argument("--logprob-max-abs-diff", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser


if __name__ == "__main__":
    compare(_parser().parse_args())
