#!/usr/bin/env python3
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

"""Compare repeated paired direct/cache-aware/consistent-hash Phase 2 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIRED_ARMS = ("direct", "cache_aware", "consistent_hash")


@dataclass(frozen=True)
class RunEvidence:
    directory: Path
    run_id: str
    repeat_id: str
    policy: str
    workload_sha256: str
    workload_seed: str
    warmup_workload_sha256: str
    warmup_requests: int
    engine_launch_id: str
    comparison_compatibility_sha256: str
    summary: Mapping[str, Any]
    outcomes: Mapping[tuple[int, int], float]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _comparison_compatibility_sha256(invariants: object) -> str:
    """Hash matrix invariants after removing uv's non-semantic check output."""
    normalized = json.loads(json.dumps(invariants))
    if not isinstance(normalized, dict):
        raise TypeError("matrix comparison invariants must be a JSON object")
    runtime_verification = (
        normalized.get("experiment", {})
        .get("software", {})
        .get("runtime_verification", {})
    )
    if isinstance(runtime_verification, dict):
        for verification in runtime_verification.values():
            if not isinstance(verification, dict):
                continue
            # The verification artifact is still checksum-audited by each source
            # report. Its digest changes when uv prints a different elapsed time.
            verification.pop("artifact_sha256", None)
            sync_check = verification.get("uv_sync_check")
            if isinstance(sync_check, dict):
                sync_check.pop("stdout", None)
                sync_check.pop("stderr", None)
    return _canonical_sha256(normalized)


def _read_outcomes(path: Path) -> dict[tuple[int, int], float]:
    outcomes: dict[tuple[int, int], float] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number}: expected an object")
            key = (int(record["prompt_index"]), int(record["generation_index"]))
            if key in outcomes:
                raise ValueError(f"{path}: duplicate outcome {key!r}")
            reward = float(record["reward"])
            if reward not in {0.0, 1.0}:
                raise ValueError(f"{path}: paired accuracy requires binary rewards")
            outcomes[key] = reward
    if not outcomes:
        raise ValueError(f"{path}: no outcomes")
    return outcomes


def _verify_source_checksums(directory: Path) -> set[str]:
    checksum_path = directory / "artifact_checksums.sha256"
    expected: dict[str, str] = {}
    with checksum_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            digest, separator, relative = line.rstrip("\n").partition("  ")
            relative_path = Path(relative)
            if (
                not separator
                or len(digest) != 64
                or relative_path.is_absolute()
                or ".." in relative_path.parts
            ):
                raise ValueError(f"{checksum_path}:{line_number}: invalid checksum row")
            expected[relative_path.as_posix()] = digest
    required = {"manifest.json", "summary.json", "evaluation/outcomes.jsonl"}
    if not required.issubset(expected):
        raise ValueError(f"{checksum_path}: required source hashes are missing")
    for relative, digest in expected.items():
        path = directory / relative
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"{directory}: checksum mismatch for {relative}")
    return set(expected)


def load_run(directory: Path) -> RunEvidence:
    """Load one report directory and its audit identity."""
    audited_artifacts = _verify_source_checksums(directory)
    manifest = _read_json(directory / "manifest.json")
    declared_artifacts = manifest.get("artifacts")
    if not isinstance(declared_artifacts, list) or set(declared_artifacts) != {
        *audited_artifacts,
        "artifact_checksums.sha256",
    }:
        raise ValueError(
            f"{directory}: manifest artifact inventory differs from checksums"
        )
    summary = _read_json(directory / "summary.json")
    if manifest.get("run_id") != summary.get("run_id"):
        raise ValueError(f"{directory}: run_id differs between manifest and summary")
    policy = str(manifest["routing_policy"])
    if policy not in REQUIRED_ARMS:
        raise ValueError(f"{directory}: unsupported Phase 2 arm {policy!r}")
    input_data = manifest.get("input_data")
    if not isinstance(input_data, dict):
        raise ValueError(f"{directory}: manifest has no input_data")
    repeat_id = str(manifest.get("repeat_id") or "")
    workload_hash = str(input_data.get("workload_sha256") or "")
    workload_seed = str(input_data.get("workload_seed") or "")
    warmup_hash = str(input_data.get("warmup_workload_sha256") or "")
    warmup_requests = input_data.get("warmup_requests")
    launch_id = str(manifest.get("engine_launch_id") or "")
    invariants_hash = str(manifest.get("comparison_invariants_sha256") or "")
    if (
        not repeat_id
        or not workload_hash
        or not workload_seed
        or not warmup_hash
        or isinstance(warmup_requests, bool)
        or not isinstance(warmup_requests, int)
        or warmup_requests <= 0
        or not launch_id
        or len(invariants_hash) != 64
    ):
        raise ValueError(
            f"{directory}: formal workload, warmup, engine, and invariant identity are required"
        )
    experiment = manifest.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError(f"{directory}: manifest has no experiment metadata")
    derived = experiment.get("derived")
    engine = experiment.get("engine")
    if (
        not isinstance(derived, dict)
        or derived.get("matrix_comparison_invariants_sha256") != invariants_hash
        or _canonical_sha256(derived.get("matrix_comparison_invariants"))
        != invariants_hash
        or not isinstance(engine, dict)
        or engine.get("fresh") is not True
        or engine.get("launch_id") != launch_id
        or input_data.get("workload_replay_faithful") is not True
    ):
        raise ValueError(f"{directory}: inconsistent formal experiment metadata")
    return RunEvidence(
        directory=directory,
        run_id=str(manifest["run_id"]),
        repeat_id=repeat_id,
        policy=policy,
        workload_sha256=workload_hash,
        workload_seed=workload_seed,
        warmup_workload_sha256=warmup_hash,
        warmup_requests=warmup_requests,
        engine_launch_id=launch_id,
        comparison_compatibility_sha256=_comparison_compatibility_sha256(
            derived["matrix_comparison_invariants"]
        ),
        summary=summary,
        outcomes=_read_outcomes(directory / "evaluation" / "outcomes.jsonl"),
    )


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _metric_distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"runs": 0, "mean": None, "min": None, "max": None}
    return {
        "runs": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def _optional_values(runs: Sequence[RunEvidence], path: Sequence[str]) -> list[float]:
    values = []
    for run in runs:
        value: Any = run.summary
        for key in path:
            value = value[key]
        if value is not None:
            values.append(float(value))
    return values


def _arm_summary(runs: Sequence[RunEvidence]) -> dict[str, Any]:
    correct = sum(int(run.summary["accuracy"]["correct"]) for run in runs)
    evaluated = sum(int(run.summary["accuracy"]["evaluated"]) for run in runs)
    backend_hits = sum(
        float(run.summary["cache"]["backend_prefix_cache"]["hits"]) for run in runs
    )
    backend_queries = sum(
        float(run.summary["cache"]["backend_prefix_cache"]["queries"]) for run in runs
    )
    router_hits = sum(
        float(run.summary["cache"]["router_routing_cache"]["hits"] or 0) for run in runs
    )
    router_misses = sum(
        float(run.summary["cache"]["router_routing_cache"]["misses"] or 0)
        for run in runs
    )
    return {
        "runs": len(runs),
        "run_ids": [run.run_id for run in runs],
        "accuracy": {
            "correct": correct,
            "evaluated": evaluated,
            "value": _ratio(correct, evaluated),
        },
        "backend_prefix_cache": {
            "hits": backend_hits,
            "queries": backend_queries,
            "hit_rate": _ratio(backend_hits, backend_queries),
        },
        "router_routing_cache": {
            "hits": router_hits,
            "misses": router_misses,
            "hit_rate": _ratio(router_hits, router_hits + router_misses),
        },
        "request_p99_s": _metric_distribution(
            _optional_values(runs, ("request_timing", "p99_s"))
        ),
        "session_p99_s": _metric_distribution(
            _optional_values(runs, ("session_timing", "p99_s"))
        ),
        "makespan_s": _metric_distribution(_optional_values(runs, ("makespan_s",))),
        "ttft_p99_s": _metric_distribution(
            _optional_values(runs, ("backend", "latency_seconds", "backend_ttft_p99"))
        ),
        "itl_p99_s": _metric_distribution(
            _optional_values(runs, ("backend", "latency_seconds", "backend_itl_p99"))
        ),
        "output_tokens_per_s": _metric_distribution(
            _optional_values(runs, ("throughput", "output_tokens_per_s"))
        ),
        "request_count_cv": _metric_distribution(
            _optional_values(
                runs,
                ("backend", "request_distribution", "coefficient_of_variation"),
            )
        ),
        "generated_token_cv": _metric_distribution(
            _optional_values(
                runs,
                (
                    "backend",
                    "generated_token_distribution",
                    "coefficient_of_variation",
                ),
            )
        ),
    }


def _exact_mcnemar_p_value(direct_only: int, candidate_only: int) -> float:
    discordant = direct_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = min(direct_only, candidate_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2 * probability)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _paired_bootstrap_ci(
    differences: Sequence[float], *, samples: int = 10_000
) -> tuple[float, float]:
    if not differences:
        raise ValueError("cannot bootstrap empty paired differences")
    generator = random.Random(0)
    estimates = [
        statistics.fmean(generator.choice(differences) for _ in differences)
        for _ in range(samples)
    ]
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _paired_comparison(
    direct_runs: Mapping[str, RunEvidence],
    candidate_runs: Mapping[str, RunEvidence],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = []
    direct_only = 0
    candidate_only = 0
    differences = []
    for repeat_id in sorted(direct_runs):
        direct = direct_runs[repeat_id]
        candidate = candidate_runs[repeat_id]
        if set(direct.outcomes) != set(candidate.outcomes):
            raise ValueError(
                f"{repeat_id}: {candidate.policy} outcome coverage differs from direct"
            )
        for prompt_index, generation_index in sorted(direct.outcomes):
            direct_reward = direct.outcomes[(prompt_index, generation_index)]
            candidate_reward = candidate.outcomes[(prompt_index, generation_index)]
            difference = candidate_reward - direct_reward
            differences.append(difference)
            direct_only += direct_reward == 1 and candidate_reward == 0
            candidate_only += direct_reward == 0 and candidate_reward == 1
            records.append(
                {
                    "repeat_id": repeat_id,
                    "candidate_policy": candidate.policy,
                    "prompt_index": prompt_index,
                    "generation_index": generation_index,
                    "direct_reward": direct_reward,
                    "candidate_reward": candidate_reward,
                    "difference": difference,
                }
            )
    lower, upper = _paired_bootstrap_ci(differences)
    mean_difference = statistics.fmean(differences)
    return (
        {
            "common_coverage": len(differences),
            "direct_only_correct": direct_only,
            "candidate_only_correct": candidate_only,
            "absolute_accuracy_difference": mean_difference,
            "absolute_percentage_point_difference": mean_difference * 100,
            "paired_bootstrap_95_ci": {
                "lower": lower,
                "upper": upper,
                "seed": 0,
                "resamples": 10_000,
            },
            "exact_mcnemar_p_value": _exact_mcnemar_p_value(
                direct_only, candidate_only
            ),
        },
        records,
    )


def _nested_metric(run: RunEvidence, path: Sequence[str]) -> float:
    value: Any = run.summary
    for key in path:
        value = value[key]
    if value is None:
        raise ValueError(
            f"{run.directory}: required paired metric {'.'.join(path)} is unavailable"
        )
    return float(value)


def _paired_operational_metric(
    direct_runs: Mapping[str, RunEvidence],
    candidate_runs: Mapping[str, RunEvidence],
    path: Sequence[str],
) -> dict[str, Any]:
    records = []
    deltas = []
    ratios = []
    for repeat_id in sorted(direct_runs):
        direct = _nested_metric(direct_runs[repeat_id], path)
        candidate = _nested_metric(candidate_runs[repeat_id], path)
        delta = candidate - direct
        ratio = candidate / direct if direct != 0 else None
        deltas.append(delta)
        if ratio is not None:
            ratios.append(ratio)
        records.append(
            {
                "repeat_id": repeat_id,
                "direct": direct,
                "candidate": candidate,
                "candidate_minus_direct": delta,
                "candidate_over_direct": ratio,
            }
        )
    return {
        "records": records,
        "candidate_minus_direct": _metric_distribution(deltas),
        "candidate_over_direct": _metric_distribution(ratios),
    }


def build_comparison(
    runs: Sequence[RunEvidence], *, min_repeats: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate the Phase 2 matrix and build paired aggregate evidence."""
    by_policy: dict[str, dict[str, RunEvidence]] = {
        policy: {} for policy in REQUIRED_ARMS
    }
    duplicate_keys = []
    for run in runs:
        if run.repeat_id in by_policy[run.policy]:
            duplicate_keys.append((run.policy, run.repeat_id))
        by_policy[run.policy][run.repeat_id] = run
    if duplicate_keys:
        raise ValueError(f"duplicate policy/repeat runs: {duplicate_keys!r}")

    repeat_sets = {policy: set(values) for policy, values in by_policy.items()}
    all_arms_present = all(by_policy[policy] for policy in REQUIRED_ARMS)
    if not all_arms_present:
        raise ValueError(
            "Phase 2 comparison requires direct, cache_aware, and consistent_hash"
        )
    common_repeats = set.intersection(*repeat_sets.values())
    same_repeat_matrix = all(
        values == common_repeats for values in repeat_sets.values()
    )
    enough_repeats = len(common_repeats) >= min_repeats
    workload_matched = True
    warmup_matched = True
    comparison_invariants_matched = True
    outcome_coverage_matched = True
    for repeat_id in common_repeats:
        repeat_runs = [by_policy[policy][repeat_id] for policy in REQUIRED_ARMS]
        workload_matched &= (
            len({(run.workload_sha256, run.workload_seed) for run in repeat_runs}) == 1
        )
        warmup_matched &= (
            len(
                {
                    (run.warmup_workload_sha256, run.warmup_requests)
                    for run in repeat_runs
                }
            )
            == 1
        )
        comparison_invariants_matched &= (
            len({run.comparison_compatibility_sha256 for run in repeat_runs}) == 1
        )
        outcome_coverage_matched &= (
            len({frozenset(run.outcomes) for run in repeat_runs}) == 1
        )
    all_run_gates_passed = all(
        run.summary["gates"].get("passed") is True for run in runs
    )
    run_ids_unique = len({run.run_id for run in runs}) == len(runs)
    fresh_engine_launch_ids_unique = len({run.engine_launch_id for run in runs}) == len(
        runs
    )

    gates = {
        "all_three_arms_present": all_arms_present,
        "repeat_matrix_exactly_matched": same_repeat_matrix,
        "minimum_repeats_met": enough_repeats,
        "workload_hash_and_seed_matched_within_repeat": workload_matched,
        "warmup_matched_within_repeat": warmup_matched,
        "comparison_invariants_matched_within_repeat": (comparison_invariants_matched),
        "paired_outcome_coverage_matched": outcome_coverage_matched,
        "all_single_run_gates_passed": all_run_gates_passed,
        "run_ids_unique": run_ids_unique,
        "fresh_engine_launch_ids_unique": fresh_engine_launch_ids_unique,
    }
    if not same_repeat_matrix:
        missing = {
            policy: sorted(common_repeats.symmetric_difference(repeats))
            for policy, repeats in repeat_sets.items()
        }
        raise ValueError(f"Phase 2 repeat matrix is incomplete: {missing!r}")
    if (
        not workload_matched
        or not warmup_matched
        or not comparison_invariants_matched
        or not outcome_coverage_matched
    ):
        raise ValueError("paired Phase 2 inputs or outcome coverage do not match")

    paired = {}
    paired_operational = {}
    paired_records = []
    for policy in ("cache_aware", "consistent_hash"):
        comparison, records = _paired_comparison(by_policy["direct"], by_policy[policy])
        paired[f"{policy}_vs_direct"] = comparison
        paired_records.extend(records)
        paired_operational[f"{policy}_vs_direct"] = {
            name: _paired_operational_metric(
                by_policy["direct"], by_policy[policy], path
            )
            for name, path in {
                "request_p99_s": ("request_timing", "p99_s"),
                "session_p99_s": ("session_timing", "p99_s"),
                "makespan_s": ("makespan_s",),
                "ttft_p99_s": (
                    "backend",
                    "latency_seconds",
                    "backend_ttft_p99",
                ),
                "itl_p99_s": (
                    "backend",
                    "latency_seconds",
                    "backend_itl_p99",
                ),
                "output_tokens_per_s": ("throughput", "output_tokens_per_s"),
                "backend_prefix_cache_hit_rate": (
                    "cache",
                    "backend_prefix_cache",
                    "hit_rate",
                ),
                "request_count_cv": (
                    "backend",
                    "request_distribution",
                    "coefficient_of_variation",
                ),
                "generated_token_cv": (
                    "backend",
                    "generated_token_distribution",
                    "coefficient_of_variation",
                ),
            }.items()
        }
    summary = {
        "schema_version": 1,
        "arms": {
            policy: _arm_summary(
                [by_policy[policy][repeat_id] for repeat_id in sorted(common_repeats)]
            )
            for policy in REQUIRED_ARMS
        },
        "repeat_ids": sorted(common_repeats),
        "gates": {**gates, "passed": all(gates.values())},
        "paired_accuracy": paired,
        "paired_operational": paired_operational,
        "interpretation_constraint": (
            "Accuracy differences are paired correctness evidence; cache or Router "
            "causality requires the accompanying cache, load, latency, and error evidence."
        ),
    }
    return summary, paired_records


def _format_rate(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.2%}"


def _format_number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def render_comparison(summary: Mapping[str, Any]) -> str:
    lines = [
        "# NeMo Gym Router Phase 2 paired comparison",
        "",
        f"- Repeats: {', '.join(summary['repeat_ids'])}",
        f"- Phase 2 matrix gates: **{'PASS' if summary['gates']['passed'] else 'FAIL'}**",
        "",
        "## Arms",
        "",
        "|Arm|Runs|Accuracy|Backend cache hit|Router cache hit|Request p99 mean|TTFT p99 mean|ITL p99 mean|Makespan mean|Output tok/s mean|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in REQUIRED_ARMS:
        arm = summary["arms"][policy]
        lines.append(
            f"|{policy}|{arm['runs']}|{_format_rate(arm['accuracy']['value'])}|"
            f"{_format_rate(arm['backend_prefix_cache']['hit_rate'])}|"
            f"{_format_rate(arm['router_routing_cache']['hit_rate'])}|"
            f"{_format_number(arm['request_p99_s']['mean'])}|"
            f"{_format_number(arm['ttft_p99_s']['mean'])}|"
            f"{_format_number(arm['itl_p99_s']['mean'])}|"
            f"{_format_number(arm['makespan_s']['mean'])}|"
            f"{_format_number(arm['output_tokens_per_s']['mean'])}|"
        )
    lines.extend(
        [
            "",
            "## Paired accuracy",
            "",
            "|Comparison|Common pairs|Difference (pp)|95% paired bootstrap CI|Exact McNemar p|",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, comparison in summary["paired_accuracy"].items():
        interval = comparison["paired_bootstrap_95_ci"]
        lines.append(
            f"|{name}|{comparison['common_coverage']}|"
            f"{comparison['absolute_percentage_point_difference']:.3f}|"
            f"[{interval['lower'] * 100:.3f}, {interval['upper'] * 100:.3f}] pp|"
            f"{comparison['exact_mcnemar_p_value']:.6f}|"
        )
    lines.extend(
        [
            "",
            "## Paired operational changes",
            "",
            "All values are the mean within-repeat candidate minus Direct change.",
            "",
            "|Comparison|Request p99 (s)|Session p99 (s)|TTFT p99 (s)|ITL p99 (s)|Makespan (s)|Output tok/s|Backend cache (pp)|Request CV|Generated-token CV|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, metrics in summary["paired_operational"].items():

        def delta(metric: str) -> float | None:
            return metrics[metric]["candidate_minus_direct"]["mean"]

        cache_delta = delta("backend_prefix_cache_hit_rate")
        lines.append(
            f"|{name}|{_format_number(delta('request_p99_s'))}|"
            f"{_format_number(delta('session_p99_s'))}|"
            f"{_format_number(delta('ttft_p99_s'))}|"
            f"{_format_number(delta('itl_p99_s'))}|"
            f"{_format_number(delta('makespan_s'))}|"
            f"{_format_number(delta('output_tokens_per_s'))}|"
            f"{_format_number(cache_delta * 100 if cache_delta is not None else None)}|"
            f"{_format_number(delta('request_count_cv'))}|"
            f"{_format_number(delta('generated_token_cv'))}|"
        )
    lines.extend(["", "## Gates", ""])
    for name, passed in summary["gates"].items():
        if name != "passed":
            lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    lines.extend(["", summary["interpretation_constraint"], ""])
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def write_comparison(
    output_dir: Path,
    *,
    runs: Sequence[RunEvidence],
    summary: Mapping[str, Any],
    paired_records: Sequence[Mapping[str, Any]],
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"comparison output directory must be empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "comparison-report.md").write_text(
        render_comparison(summary), encoding="utf-8"
    )
    with (output_dir / "paired-outcomes.jsonl").open("w", encoding="utf-8") as output:
        for record in paired_records:
            output.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
    manifest = {
        "schema_version": 1,
        "runs": [
            {
                "directory": str(run.directory),
                "run_id": run.run_id,
                "repeat_id": run.repeat_id,
                "routing_policy": run.policy,
                "summary_sha256": _sha256(run.directory / "summary.json"),
                "outcomes_sha256": _sha256(
                    run.directory / "evaluation" / "outcomes.jsonl"
                ),
            }
            for run in runs
        ],
    }
    (output_dir / "comparison-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = sorted(path for path in output_dir.iterdir() if path.is_file())
    (output_dir / "artifact_checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--min-repeats", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_repeats < 2:
        raise ValueError("formal Phase 2 comparison requires at least two repeats")
    runs = [load_run(directory) for directory in args.run]
    summary, records = build_comparison(runs, min_repeats=args.min_repeats)
    write_comparison(
        args.output_dir, runs=runs, summary=summary, paired_records=records
    )
    print(args.output_dir / "comparison-report.md")
    if summary["gates"]["passed"] is not True:
        raise SystemExit("Phase 2 comparison gates failed; see comparison-report.md")


if __name__ == "__main__":
    main()
