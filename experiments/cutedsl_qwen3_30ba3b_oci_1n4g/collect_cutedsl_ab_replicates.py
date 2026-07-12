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

"""Collect deterministic paired CuTeDSL ON/OFF replicate statistics."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricSpec:
    """A measured raw-timing field included in the paired aggregate."""

    name: str
    category: str
    field: str


METRIC_SPECS = (
    MetricSpec("e2e_duration", "duration", "total_step_seconds"),
    MetricSpec("generation_duration", "duration", "generation_seconds"),
    MetricSpec(
        "generation_finalize_duration", "duration", "generation_finalize_seconds"
    ),
    MetricSpec("logprob_duration", "duration", "logprob_seconds"),
    MetricSpec("policy_training_duration", "duration", "policy_training_seconds"),
    MetricSpec("refit_duration", "duration", "refit_transfer_update_seconds"),
    MetricSpec("e2e_throughput", "throughput", "e2e_tokens_per_sec_per_gpu"),
    MetricSpec(
        "generation_throughput",
        "throughput",
        "generation_tokens_per_sec_per_gpu",
    ),
    MetricSpec(
        "logprob_throughput",
        "throughput",
        "policy_and_reference_logprobs_tokens_per_sec_per_gpu",
    ),
    MetricSpec(
        "policy_training_throughput",
        "throughput",
        "policy_training_tokens_per_sec_per_gpu",
    ),
    MetricSpec(
        "refit_effective_throughput",
        "throughput",
        "refit_effective_tokens_per_sec_per_gpu",
    ),
)
REQUIRED_CANONICAL_METRICS = frozenset(
    {
        "timing/train/total_step_time",
        "timing/train/generation",
        "timing/train/generation_finalize",
        "timing/train/get_logprobs",
        "timing/train/policy_training",
        "timing/train/prepare_for_generation/transfer_and_update_weights",
        "performance/tokens_per_sec_per_gpu",
        "performance/generation_tokens_per_sec_per_gpu",
        "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu",
        "performance/policy_training_tokens_per_sec_per_gpu",
        "train/total_num_tokens",
        "train/global_valid_toks",
    }
)
VALID_ORDERS = frozenset({"on,off", "off,on"})
ORDERED_TIMING_ORDERS = ("on,off", "off,on")
RATIO_DEFINITION = "median(on measured steps) / median(off measured steps)"
CSV_FIELDS = (
    "scope",
    "metric",
    "category",
    "ratio_definition",
    "replicate_count",
    "replicate_index",
    "job_id",
    "timing_order",
    "ratio",
    "median_ratio",
    "replicate_median_cv_percent",
    "ci95_lower",
    "ci95_upper",
    "extend_to_six",
    "recommendation_reasons",
)


class CollectorError(ValueError):
    """Raised when submitted benchmark evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class Replicate:
    """Validated inputs and paired ratios for one submitted job."""

    replicate_index: int
    job_id: str
    run_id: str
    result_dir: Path
    timing_order: str
    profile_enabled: bool
    source_identity: str
    image_identity: str
    workload_identity: str
    metric_identity: str
    measured_workload_identity: str
    ratios: dict[str, float]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CollectorError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CollectorError(f"{label} {path} must contain a JSON object")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CollectorError(f"{label} must be a nonempty string")
    return value


def _require_nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CollectorError(f"{label} must be a non-negative integer")
    return value


def _require_profile_flag(value: Any, label: str) -> bool:
    if value not in (0, 1, False, True):
        raise CollectorError(f"{label} must be 0 or 1")
    return bool(value)


def _load_submission(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise CollectorError(f"cannot read submission JSONL {path}: {error}") from error
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise CollectorError(
                f"submission JSONL {path}:{line_number} is invalid: {error}"
            ) from error
        if not isinstance(record, dict):
            raise CollectorError(
                f"submission JSONL {path}:{line_number} must contain an object"
            )
        records.append(record)
    return records


def _contained_file(root: Path, path: Path, label: str) -> Path:
    root = root.resolve()
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        raise CollectorError(f"{label} escapes benchmark result root")
    if not resolved.is_file():
        raise CollectorError(f"{label} does not exist as a regular file: {path}")
    return resolved


def _safe_artifact(root: Path, job_dir: Path, relative_path: Any, label: str) -> Path:
    relative = Path(_require_string(relative_path, label))
    if relative.is_absolute():
        raise CollectorError(f"{label} must be relative to job result directory")
    job_root = job_dir.resolve()
    path = _contained_file(root, job_dir / relative, label)
    if path == job_root or job_root not in path.parents:
        raise CollectorError(f"{label} escapes job result directory: {relative}")
    return path


def _require_sha(value: Any, *, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value) is None
    ):
        raise CollectorError(f"{label} must be a {length}-character hexadecimal SHA")
    return value


def _validate_manifest_identity(manifest: dict[str, Any], job_id: str) -> None:
    _require_sha(manifest.get("source_sha"), length=40, label="source_sha")
    _require_string(manifest.get("upstream_ref"), "upstream_ref")
    _require_sha(manifest.get("upstream_sha"), length=40, label="upstream_sha")
    _require_string(manifest.get("image"), "image")
    _require_sha(manifest.get("image_sha256"), length=64, label="image_sha256")
    _require_string(manifest.get("recipe"), "recipe")
    warmup_updates = _require_nonnegative_integer(
        manifest.get("warmup_updates"), f"job {job_id} warmup_updates"
    )
    measured_updates = _require_nonnegative_integer(
        manifest.get("measured_updates"), f"job {job_id} measured_updates"
    )
    total_updates = _require_nonnegative_integer(
        manifest.get("total_updates"), f"job {job_id} total_updates"
    )
    if warmup_updates == 0 or measured_updates == 0:
        raise CollectorError(
            f"job {job_id} warmup and measured updates must be positive"
        )
    if total_updates != warmup_updates + measured_updates:
        raise CollectorError(
            f"job {job_id} total_updates must equal warmup_updates + measured_updates"
        )
    topology = manifest.get("topology")
    if not isinstance(topology, dict) or not topology:
        raise CollectorError("topology must be a nonempty object")
    fixed_config = manifest.get("fixed_config_evidence")
    if (
        not isinstance(fixed_config, dict)
        or set(fixed_config) != {"on", "off"}
        or any(
            not isinstance(fixed_config[arm], dict) or not fixed_config[arm]
            for arm in ("on", "off")
        )
    ):
        raise CollectorError("fixed_config_evidence must contain ON/OFF objects")


def _validate_manifest_metrics(
    manifest: dict[str, Any], job_id: str
) -> dict[str, dict[str, str]]:
    metrics = manifest.get("resolved_metric_names")
    if not isinstance(metrics, dict) or set(metrics) != {"on", "off"}:
        raise CollectorError(
            f"job {job_id} manifest must contain ON/OFF resolved metric names"
        )
    validated = {}
    for arm in ("on", "off"):
        mapping = metrics[arm]
        if not isinstance(mapping, dict) or set(mapping) != REQUIRED_CANONICAL_METRICS:
            raise CollectorError(
                f"job {job_id} {arm.upper()} resolved metric names must contain "
                "the exact canonical metric set"
            )
        for canonical_name, source_name in mapping.items():
            if not isinstance(source_name, str) or not source_name:
                raise CollectorError(
                    f"job {job_id} resolved metric name for {canonical_name!r} "
                    "must be a nonempty string"
                )
        validated[arm] = mapping
    if validated["on"] != validated["off"]:
        raise CollectorError(f"job {job_id} ON/OFF resolved metric names must match")
    return validated


def _numeric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CollectorError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise CollectorError(f"{label} must be finite and positive")
    return result


def _validate_metric_names(
    raw: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    resolved = raw.get("resolved_metric_names")
    if not isinstance(resolved, dict):
        raise CollectorError(f"{label} lacks resolved_metric_names")
    missing = sorted(REQUIRED_CANONICAL_METRICS - resolved.keys())
    if missing:
        raise CollectorError(f"{label} is missing resolved metric names: {missing}")
    if resolved != expected:
        raise CollectorError(f"{label} resolved metric names differ from manifest")


def _load_raw_timing(
    root: Path,
    job_dir: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    job_id: str,
    run_id: str,
) -> dict[str, dict[str, Any]]:
    raw_files = summary.get("raw_timing_files")
    if not isinstance(raw_files, list) or len(raw_files) != 2:
        raise CollectorError(
            f"job {job_id} must reference exactly two raw timing files"
        )
    manifest_metrics = _validate_manifest_metrics(manifest, job_id)

    by_arm = {}
    order_indices = set()
    for raw_file in raw_files:
        path = _safe_artifact(root, job_dir, raw_file, f"job {job_id} raw timing file")
        raw = _read_json(path, f"job {job_id} raw timing")
        arm = raw.get("arm")
        if arm not in ("on", "off") or arm in by_arm:
            raise CollectorError(
                f"job {job_id} raw timing arms must be exactly ON and OFF"
            )
        if raw.get("run_id") != run_id:
            raise CollectorError(f"job {job_id} raw timing run_id does not match run")
        order_index = _require_nonnegative_integer(
            raw.get("order_index"), f"job {job_id} {arm.upper()} raw order_index"
        )
        timing_order = summary["timing_order"]
        if (
            order_index not in (0, 1)
            or order_index in order_indices
            or timing_order[order_index] != arm
        ):
            raise CollectorError(
                f"job {job_id} raw timing order_index does not evidence timing order"
            )
        order_indices.add(order_index)
        _validate_metric_names(
            raw,
            manifest_metrics[arm],
            f"job {job_id} {arm.upper()} raw timing",
        )
        workload = raw.get("measured_step_workload")
        if not isinstance(workload, list) or not workload:
            raise CollectorError(
                f"job {job_id} {arm.upper()} measured_step_workload must be nonempty"
            )
        by_arm[arm] = raw
    if set(by_arm) != {"on", "off"}:
        raise CollectorError(f"job {job_id} raw timing arms must be exactly ON and OFF")
    if order_indices != {0, 1}:
        raise CollectorError(
            f"job {job_id} raw timing order_index does not evidence timing order"
        )
    return by_arm


def _measured_workload_identity(by_arm: dict[str, dict[str, Any]], job_id: str) -> str:
    identities = {}
    for arm in ("on", "off"):
        rows = by_arm[arm]["measured_step_workload"]
        identity_rows = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise CollectorError(
                    f"job {job_id} {arm.upper()} workload row {row_index} must be an object"
                )
            step = _require_nonnegative_integer(
                row.get("step"), f"job {job_id} {arm.upper()} workload step"
            )
            total_tokens = _numeric(
                row.get("total_num_tokens"),
                f"job {job_id} {arm.upper()} total_num_tokens step {step}",
            )
            valid_tokens = _numeric(
                row.get("global_valid_toks"),
                f"job {job_id} {arm.upper()} global_valid_toks step {step}",
            )
            identity_rows.append((step, total_tokens, valid_tokens))
        identities[arm] = identity_rows
    if identities["on"] != identities["off"]:
        raise CollectorError(
            f"job {job_id} measured workload differs across replicates or ON/OFF arms"
        )
    return _canonical_json(identities["on"])


def _paired_ratios(by_arm: dict[str, dict[str, Any]], job_id: str) -> dict[str, float]:
    ratios = {}
    for spec in METRIC_SPECS:
        medians = {}
        for arm in ("on", "off"):
            rows = by_arm[arm]["measured_step_workload"]
            values = [
                _numeric(
                    row.get(spec.field),
                    f"job {job_id} {arm.upper()} {spec.field} row {row_index}",
                )
                for row_index, row in enumerate(rows)
            ]
            medians[arm] = statistics.median(values)
        ratios[spec.name] = medians["on"] / medians["off"]
    return ratios


def _validate_profile_attribution(root: Path, job_dir: Path, job_id: str) -> None:
    attribution_paths = sorted(job_dir.rglob("kernel_attribution.json"))
    if len(attribution_paths) != 1:
        raise CollectorError(
            f"designated profile job {job_id} expected exactly one "
            f"kernel_attribution.json, found {len(attribution_paths)}"
        )
    attribution_path = _contained_file(
        root,
        attribution_paths[0],
        f"designated profile job {job_id} kernel attribution",
    )
    attribution = _read_json(
        attribution_path, f"designated profile job {job_id} kernel attribution"
    )
    if attribution.get("passed") is not True:
        raise CollectorError(
            f"designated profile job {job_id} kernel attribution did not pass"
        )
    arms = attribution.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"on", "off"}:
        raise CollectorError(
            f"designated profile job {job_id} attribution must contain ON/OFF arms"
        )
    for arm in ("on", "off"):
        result = arms[arm]
        if not isinstance(result, dict):
            raise CollectorError(
                f"designated profile job {job_id} {arm.upper()} attribution is invalid"
            )
        _safe_artifact(
            root,
            job_dir,
            result.get("kernel_evidence"),
            f"designated profile job {job_id} {arm.upper()} kernel evidence",
        )
        if (
            _require_nonnegative_integer(
                result.get("grouped_gemm_match_count"),
                f"designated profile job {job_id} {arm.upper()} grouped GEMM match count",
            )
            == 0
        ):
            raise CollectorError(
                f"designated profile job {job_id} {arm.upper()} lacks grouped GEMM attribution"
            )
    for field in ("fused_glu_match_count", "fused_dglu_match_count"):
        on_count = _require_nonnegative_integer(
            arms["on"].get(field),
            f"designated profile job {job_id} ON {field}",
        )
        off_count = _require_nonnegative_integer(
            arms["off"].get(field),
            f"designated profile job {job_id} OFF {field}",
        )
        if on_count == 0 or off_count != 0:
            raise CollectorError(
                f"designated profile job {job_id} has invalid {field} attribution"
            )

    profile_paths = sorted(job_dir.glob("profiles/*/profile_summary.json"))
    if len(profile_paths) != 2:
        raise CollectorError(
            f"designated profile job {job_id} expected two profile summaries, "
            f"found {len(profile_paths)}"
        )
    profile_arms = set()
    for path in profile_paths:
        path = _contained_file(
            root, path, f"designated profile job {job_id} profile summary"
        )
        profile = _read_json(path, f"designated profile job {job_id} profile summary")
        arm = profile.get("arm")
        if arm not in ("on", "off") or arm in profile_arms:
            raise CollectorError(
                f"designated profile job {job_id} profile arms must be ON and OFF"
            )
        profile_arms.add(arm)
        if (
            _require_nonnegative_integer(
                profile.get("nsight_report_count"),
                f"designated profile job {job_id} {arm.upper()} Nsight report count",
            )
            == 0
        ):
            raise CollectorError(
                f"designated profile job {job_id} {arm.upper()} has no Nsight report"
            )
        _safe_artifact(
            root,
            path.parent,
            profile.get("kernel_evidence"),
            f"designated profile job {job_id} {arm.upper()} profile kernel evidence",
        )


def _find_completed_run(root: Path, job_id: str) -> tuple[Path, str]:
    if (
        Path(job_id).is_absolute()
        or Path(job_id).name != job_id
        or job_id in (".", "..")
        or "/" in job_id
        or "\\" in job_id
    ):
        raise CollectorError("submission job_id must be a safe single path component")
    root = root.resolve()
    if not root.is_dir():
        raise CollectorError(f"benchmark result root is not a directory: {root}")
    candidate_pattern = re.compile(rf"{re.escape(job_id)}(?:-r[1-9][0-9]*)?")
    candidates = sorted(
        path for path in root.iterdir() if candidate_pattern.fullmatch(path.name)
    )
    if not candidates:
        raise CollectorError(
            f"submitted job {job_id} result directory is missing under {root}"
        )

    successful = []
    failed_exit_codes = []
    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if (
            resolved_candidate == root
            or root not in resolved_candidate.parents
            or not resolved_candidate.is_dir()
        ):
            raise CollectorError(
                f"job {job_id} result directory escapes benchmark root"
            )
        status_path = _contained_file(
            root, candidate / "status.json", f"job {job_id} status"
        )
        status = _read_json(status_path, f"job {job_id} status")
        if status.get("job_id") != job_id or status.get("run_id") != candidate.name:
            raise CollectorError(
                f"job {job_id} status identity does not match submission"
            )
        exit_code = status.get("exit_code")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise CollectorError(f"job {job_id} status exit_code must be an integer")
        if exit_code == 0:
            successful.append((resolved_candidate, candidate.name))
        else:
            failed_exit_codes.append(exit_code)
    if len(successful) > 1:
        raise CollectorError(
            f"job {job_id} has multiple successfully completed run directories"
        )
    if not successful:
        detail = (
            failed_exit_codes[0] if len(failed_exit_codes) == 1 else failed_exit_codes
        )
        raise CollectorError(
            f"job {job_id} is not completed successfully: exit_code={detail!r}"
        )
    return successful[0]


def _load_replicate(root: Path, record: dict[str, Any]) -> Replicate:
    replicate_index = _require_nonnegative_integer(
        record.get("replicate_index"), "submission replicate_index"
    )
    job_id = _require_string(record.get("job_id"), "submission job_id")
    timing_order = _require_string(
        record.get("timing_order"), f"submission timing order for job {job_id}"
    )
    expected_order = "on,off" if replicate_index % 2 == 0 else "off,on"
    if timing_order != expected_order:
        raise CollectorError(
            f"submission timing order for replicate {replicate_index} must be "
            f"{expected_order}, found {timing_order}"
        )
    profile_enabled = _require_profile_flag(
        record.get("profile_enabled"), f"submission profile_enabled for job {job_id}"
    )
    job_dir, run_id = _find_completed_run(root, job_id)

    timing_paths = sorted(job_dir.rglob("timing_summary.json"))
    if len(timing_paths) != 1:
        raise CollectorError(
            f"job {job_id} expected exactly one timing_summary.json, "
            f"found {len(timing_paths)}"
        )
    summary_path = _contained_file(
        root, timing_paths[0], f"job {job_id} timing summary"
    )
    summary = _read_json(summary_path, f"job {job_id} timing summary")
    if summary.get("run_id") != run_id:
        raise CollectorError(f"job {job_id} timing summary run_id does not match")
    if summary.get("timing_order") != timing_order.split(","):
        raise CollectorError(
            f"job {job_id} timing summary order differs from submission"
        )
    if summary.get("workload_equality_observed") is not True:
        raise CollectorError(f"job {job_id} requires workload_equality_observed=true")
    if summary.get("workload_equality_required") is not True:
        raise CollectorError(f"job {job_id} requires workload_equality_required=true")
    if summary.get("workload_metric") != "train/total_num_tokens":
        raise CollectorError(
            f"job {job_id} workload metric must be train/total_num_tokens"
        )

    manifest_path = job_dir / "benchmark_manifest.json"
    manifest_path = _contained_file(
        root, manifest_path, f"job {job_id} benchmark manifest"
    )
    manifest = _read_json(manifest_path, f"job {job_id} benchmark manifest")
    manifest_contracts = {
        "run_id": run_id,
        "replicate_index": replicate_index,
        "timing_order": timing_order.split(","),
        "profile_enabled": profile_enabled,
        "submission_group": record.get("submission_group"),
    }
    for field, expected in manifest_contracts.items():
        if manifest.get(field) != expected:
            raise CollectorError(
                f"job {job_id} manifest {field} differs from submission: "
                f"{manifest.get(field)!r} != {expected!r}"
            )

    _validate_manifest_identity(manifest, job_id)
    by_arm = _load_raw_timing(root, job_dir, summary, manifest, job_id, run_id)
    measured_workload_identity = _measured_workload_identity(by_arm, job_id)
    ratios = _paired_ratios(by_arm, job_id)
    source_identity = _canonical_json(
        {
            field: manifest.get(field)
            for field in ("source_sha", "upstream_ref", "upstream_sha")
        }
    )
    image_identity = _canonical_json(
        {field: manifest.get(field) for field in ("image", "image_sha256")}
    )
    workload_identity = _canonical_json(
        {
            field: manifest.get(field)
            for field in (
                "recipe",
                "warmup_updates",
                "measured_updates",
                "total_updates",
                "topology",
                "fixed_config_evidence",
            )
        }
    )
    metric_identity = _canonical_json(manifest.get("resolved_metric_names"))
    return Replicate(
        replicate_index=replicate_index,
        job_id=job_id,
        run_id=run_id,
        result_dir=job_dir,
        timing_order=timing_order,
        profile_enabled=profile_enabled,
        source_identity=source_identity,
        image_identity=image_identity,
        workload_identity=workload_identity,
        metric_identity=metric_identity,
        measured_workload_identity=measured_workload_identity,
        ratios=ratios,
    )


def _clean_float(value: float) -> float:
    return round(value, 12)


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return sorted_values[lower_index] + fraction * (
        sorted_values[upper_index] - sorted_values[lower_index]
    )


def _bootstrap_ci(
    ratios: list[float], *, samples: int, seed: int, stream_name: str
) -> dict[str, float]:
    digest = hashlib.sha256(f"{seed}:{stream_name}".encode()).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    count = len(ratios)
    estimates = sorted(
        statistics.median(generator.choices(ratios, k=count)) for _ in range(samples)
    )
    return {
        "lower": _clean_float(_percentile(estimates, 0.025)),
        "upper": _clean_float(_percentile(estimates, 0.975)),
    }


def _summarize_ratios(
    ratios: list[float], *, samples: int, seed: int, stream_name: str
) -> dict[str, Any]:
    median_ratio = _clean_float(statistics.median(ratios))
    cv_percent = None
    if len(ratios) > 1:
        cv_percent = _clean_float(
            statistics.stdev(ratios) / statistics.mean(ratios) * 100.0
        )
    ci = _bootstrap_ci(ratios, samples=samples, seed=seed, stream_name=stream_name)
    reasons = []
    if cv_percent is not None and cv_percent > 5.0:
        reasons.append("CV exceeds 5%")
    if ci["lower"] <= 1.0 <= ci["upper"]:
        reasons.append("CI crosses 1")
    return {
        "replicate_count": len(ratios),
        "median_ratio": median_ratio,
        "replicate_median_cv_percent": cv_percent,
        "paired_bootstrap_ci95": ci,
        "recommendation": {
            "extend_to_six": bool(reasons),
            "reasons": reasons,
        },
    }


def _validate_replicates(replicates: list[Replicate]) -> Replicate:
    job_ids = [replicate.job_id for replicate in replicates]
    replicate_indices = [replicate.replicate_index for replicate in replicates]
    if len(replicates) < 3 or len(set(job_ids)) != len(replicates):
        raise CollectorError(
            "at least 3 distinct completed replicate jobs are required"
        )
    if len(set(replicate_indices)) != len(replicates):
        raise CollectorError("replicate indices must be distinct")
    orders = {replicate.timing_order for replicate in replicates}
    if orders != VALID_ORDERS:
        raise CollectorError(
            f"alternating timing orders must both be represented; found {sorted(orders)}"
        )
    profiles = [replicate for replicate in replicates if replicate.profile_enabled]
    if len(profiles) != 1:
        raise CollectorError(
            f"exactly one designated profile replicate is required; found {len(profiles)}"
        )
    identity_contracts = (
        ("source identity differs", "source_identity"),
        ("image identity differs", "image_identity"),
        ("workload identity differs", "workload_identity"),
        ("resolved metric names differ", "metric_identity"),
        ("measured workload differs", "measured_workload_identity"),
    )
    for failure, attribute in identity_contracts:
        values = {getattr(replicate, attribute) for replicate in replicates}
        if len(values) != 1:
            raise CollectorError(f"{failure} across replicates")
    return profiles[0]


def collect(
    submission_path: Path,
    result_root: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate replicate artifacts and return JSON and long-form CSV records."""
    records = _load_submission(submission_path)
    replicates = sorted(
        (_load_replicate(result_root, record) for record in records),
        key=lambda replicate: replicate.replicate_index,
    )
    profile_replicate = _validate_replicates(replicates)
    _validate_profile_attribution(
        result_root,
        profile_replicate.result_dir,
        profile_replicate.job_id,
    )

    metrics = {}
    csv_rows = []
    extension_metrics = []
    for spec in METRIC_SPECS:
        replicate_values = [replicate.ratios[spec.name] for replicate in replicates]
        summary = _summarize_ratios(
            replicate_values,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            stream_name=f"aggregate:{spec.name}",
        )
        replicate_rows = [
            {
                "replicate_index": replicate.replicate_index,
                "job_id": replicate.job_id,
                "timing_order": replicate.timing_order,
                "ratio": _clean_float(replicate.ratios[spec.name]),
            }
            for replicate in replicates
        ]
        order_summaries = {}
        for order in ORDERED_TIMING_ORDERS:
            order_values = [
                replicate.ratios[spec.name]
                for replicate in replicates
                if replicate.timing_order == order
            ]
            order_summaries[order] = _summarize_ratios(
                order_values,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
                stream_name=f"order:{order}:{spec.name}",
            )
        metrics[spec.name] = {
            "category": spec.category,
            "measured_field": spec.field,
            "replicates": replicate_rows,
            **summary,
            "order_stratified": order_summaries,
        }
        if summary["recommendation"]["extend_to_six"]:
            extension_metrics.append(spec.name)

        for replicate_row in replicate_rows:
            csv_rows.append(
                {
                    "scope": "replicate",
                    "metric": spec.name,
                    "category": spec.category,
                    "ratio_definition": RATIO_DEFINITION,
                    "replicate_count": 1,
                    **replicate_row,
                }
            )
        csv_rows.append(
            _summary_csv_row(
                scope="aggregate",
                metric=spec.name,
                category=spec.category,
                timing_order="",
                summary=summary,
            )
        )
        for order, order_summary in order_summaries.items():
            csv_rows.append(
                _summary_csv_row(
                    scope="order",
                    metric=spec.name,
                    category=spec.category,
                    timing_order=order,
                    summary=order_summary,
                )
            )

    first = replicates[0]
    aggregate = {
        "schema_version": 1,
        "submission_jsonl": str(submission_path),
        "benchmark_result_root": str(result_root),
        "ratio_definition": RATIO_DEFINITION,
        "duration_ratio_interpretation": "values below 1 favor ON",
        "throughput_ratio_interpretation": "values above 1 favor ON",
        "replicate_count": len(replicates),
        "replicate_indices": [replicate.replicate_index for replicate in replicates],
        "job_ids": [replicate.job_id for replicate in replicates],
        "run_ids": [replicate.run_id for replicate in replicates],
        "timing_orders": list(ORDERED_TIMING_ORDERS),
        "profile_replicate": {
            "replicate_index": profile_replicate.replicate_index,
            "job_id": profile_replicate.job_id,
            "run_id": profile_replicate.run_id,
        },
        "source": json.loads(first.source_identity),
        "image": json.loads(first.image_identity),
        "workload": json.loads(first.workload_identity),
        "resolved_metric_names": json.loads(first.metric_identity),
        "bootstrap": {
            "method": "paired replicate resampling of median ratios",
            "confidence_level": 0.95,
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "recommendation": {
            "extend_to_six": bool(extension_metrics),
            "metrics": extension_metrics,
            "rule": "extend when replicate-median CV exceeds 5% or paired CI crosses 1",
        },
        "metrics": metrics,
    }
    return aggregate, csv_rows


def _summary_csv_row(
    *,
    scope: str,
    metric: str,
    category: str,
    timing_order: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    recommendation = summary["recommendation"]
    return {
        "scope": scope,
        "metric": metric,
        "category": category,
        "ratio_definition": RATIO_DEFINITION,
        "replicate_count": summary["replicate_count"],
        "replicate_index": "",
        "job_id": "",
        "timing_order": timing_order,
        "ratio": "",
        "median_ratio": summary["median_ratio"],
        "replicate_median_cv_percent": (
            ""
            if summary["replicate_median_cv_percent"] is None
            else summary["replicate_median_cv_percent"]
        ),
        "ci95_lower": summary["paired_bootstrap_ci95"]["lower"],
        "ci95_upper": summary["paired_bootstrap_ci95"]["upper"],
        "extend_to_six": recommendation["extend_to_six"],
        "recommendation_reasons": "; ".join(recommendation["reasons"]),
    }


def _render_csv(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write(content)
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_jsonl", type=Path)
    parser.add_argument("benchmark_result_root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2606)
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    if args.bootstrap_seed < 0:
        parser.error("--bootstrap-seed must be non-negative")
    if args.output_json is None:
        args.output_json = args.submission_jsonl.with_suffix(".aggregate.json")
    if args.output_csv is None:
        args.output_csv = args.submission_jsonl.with_suffix(".aggregate.csv")
    if args.output_json.resolve() == args.output_csv.resolve():
        parser.error("--output-json and --output-csv must differ")
    if args.submission_jsonl.resolve() in {
        args.output_json.resolve(),
        args.output_csv.resolve(),
    }:
        parser.error("output paths must not overwrite the submission JSONL")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the replicate collector command-line interface."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        aggregate, csv_rows = collect(
            args.submission_jsonl,
            args.benchmark_result_root,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        _atomic_write(
            args.output_json,
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        )
        _atomic_write(args.output_csv, _render_csv(csv_rows))
    except (CollectorError, OSError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    print(f"[INFO] Wrote aggregate JSON: {args.output_json}")
    print(f"[INFO] Wrote aggregate CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
