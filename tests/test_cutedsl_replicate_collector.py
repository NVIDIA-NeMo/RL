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

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

EXPERIMENT_DIR = Path(__file__).parents[1] / "experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
COLLECTOR = EXPERIMENT_DIR / "collect_cutedsl_ab_replicates.py"

CANONICAL_METRICS = {
    "timing/train/total_step_time": "timing/train/total_step_time",
    "timing/train/generation": "timing/train/generation",
    "timing/train/generation_finalize": "timing/train/generation_finalize",
    "timing/train/get_logprobs": "timing/train/policy_and_reference_logprobs",
    "timing/train/policy_training": "timing/train/policy_training",
    "timing/train/prepare_for_generation/transfer_and_update_weights": (
        "timing/train/prepare_for_generation/transfer_and_update_weights"
    ),
    "performance/tokens_per_sec_per_gpu": ("performance/tokens_per_sec_per_gpu"),
    "performance/generation_tokens_per_sec_per_gpu": (
        "performance/generation_tokens_per_sec_per_gpu"
    ),
    "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu": (
        "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu"
    ),
    "performance/policy_training_tokens_per_sec_per_gpu": (
        "performance/policy_training_tokens_per_sec_per_gpu"
    ),
    "train/total_num_tokens": "train/total_num_tokens",
    "train/global_valid_toks": "train/global_valid_toks",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _raw_timing(
    *,
    job_id: str,
    arm: str,
    order_index: int,
    duration_ratio: float,
    throughput_ratio: float,
) -> dict[str, Any]:
    arm_duration_scale = duration_ratio if arm == "on" else 1.0
    arm_throughput_scale = throughput_ratio if arm == "on" else 1.0
    measured_step_workload = []
    for offset, step in enumerate((6, 7, 8)):
        token_count = 1000.0 + offset * 100.0
        measured_step_workload.append(
            {
                "step": step,
                "total_step_seconds": (10.0 + offset) * arm_duration_scale,
                "generation_seconds": (4.0 + offset) * arm_duration_scale,
                "generation_finalize_seconds": (0.4 + offset / 10) * arm_duration_scale,
                "logprob_seconds": (1.0 + offset / 10) * arm_duration_scale,
                "policy_training_seconds": (3.0 + offset / 10) * arm_duration_scale,
                "refit_transfer_update_seconds": (0.8 + offset / 10)
                * arm_duration_scale,
                "total_num_tokens": token_count,
                "global_valid_toks": token_count - 10.0,
                "e2e_tokens_per_sec_per_gpu": (25.0 + offset) * arm_throughput_scale,
                "generation_tokens_per_sec_per_gpu": (50.0 + offset)
                * arm_throughput_scale,
                "policy_and_reference_logprobs_tokens_per_sec_per_gpu": (80.0 + offset)
                * arm_throughput_scale,
                "policy_training_tokens_per_sec_per_gpu": (90.0 + offset)
                * arm_throughput_scale,
                "refit_effective_tokens_per_sec_per_gpu": (300.0 + offset)
                * arm_throughput_scale,
            }
        )
    return {
        "run_id": job_id,
        "arm": arm,
        "order_index": order_index,
        "warmup_updates": 5,
        "measured_updates": 3,
        "resolved_metric_names": CANONICAL_METRICS,
        "measured_step_workload": measured_step_workload,
    }


def _create_job(
    root: Path,
    *,
    job_id: str,
    replicate_index: int,
    timing_order: str,
    profile_enabled: bool,
    duration_ratio: float,
    throughput_ratio: float,
) -> Path:
    job_dir = root / job_id
    order = timing_order.split(",")
    raw_paths = []
    raw_by_arm = {}
    for order_index, arm in enumerate(order):
        raw_path = Path("timing") / f"{order_index}-{arm}" / "raw_timing.json"
        raw = _raw_timing(
            job_id=job_id,
            arm=arm,
            order_index=order_index,
            duration_ratio=duration_ratio,
            throughput_ratio=throughput_ratio,
        )
        _write_json(job_dir / raw_path, raw)
        raw_paths.append(str(raw_path))
        raw_by_arm[arm] = raw

    _write_json(
        job_dir / "status.json",
        {"run_id": job_id, "job_id": job_id, "exit_code": 0},
    )
    _write_json(
        job_dir / "benchmark_manifest.json",
        {
            "run_id": job_id,
            "replicate_index": replicate_index,
            "submission_group": "group-a",
            "timing_order": order,
            "warmup_updates": 5,
            "measured_updates": 3,
            "total_updates": 8,
            "profile_enabled": profile_enabled,
            "source_sha": "a" * 40,
            "upstream_ref": "origin/feature",
            "upstream_sha": "a" * 40,
            "image": "/images/nemo.sqsh",
            "image_sha256": "b" * 64,
            "recipe": "recipes/cutedsl.yaml",
            "topology": {"num_nodes": 1, "gpus_per_node": 4},
            "fixed_config_evidence": {
                "on": {"moe_grouped_gemm": True},
                "off": {"moe_grouped_gemm": True},
            },
            "resolved_metric_names": {
                "on": CANONICAL_METRICS,
                "off": CANONICAL_METRICS,
            },
        },
    )
    _write_json(
        job_dir / "timing_summary.json",
        {
            "run_id": job_id,
            "timing_order": order,
            "raw_timing_files": raw_paths,
            "workload_metric": "train/total_num_tokens",
            "workload_equality_required": True,
            "workload_equality_observed": True,
            "measured_total_num_tokens": {
                arm: [
                    item["total_num_tokens"]
                    for item in raw_by_arm[arm]["measured_step_workload"]
                ]
                for arm in ("on", "off")
            },
        },
    )
    if profile_enabled:
        _write_json(
            job_dir / "kernel_attribution.json",
            {
                "passed": True,
                "arms": {
                    "on": {
                        "kernel_evidence": "profiles/0-on/kernel_evidence.txt",
                        "fused_glu_match_count": 3,
                        "fused_dglu_match_count": 2,
                        "grouped_gemm_match_count": 4,
                    },
                    "off": {
                        "kernel_evidence": "profiles/1-off/kernel_evidence.txt",
                        "fused_glu_match_count": 0,
                        "fused_dglu_match_count": 0,
                        "grouped_gemm_match_count": 2,
                    },
                },
                "failures": [],
            },
        )
        for order_index, arm in enumerate(order):
            profile_dir = job_dir / "profiles" / f"{order_index}-{arm}"
            _write_json(
                profile_dir / "profile_summary.json",
                {
                    "arm": arm,
                    "order_index": order_index,
                    "nsight_report_count": 1,
                    "kernel_evidence": "kernel_evidence.txt",
                },
            )
            (profile_dir / "kernel_evidence.txt").write_text(
                f"{arm} grouped gemm evidence\n"
            )
    return job_dir


def _create_valid_inputs(tmp_path: Path) -> tuple[Path, Path]:
    result_root = tmp_path / "benchmark-results"
    submission = tmp_path / "submission.jsonl"
    records = []
    for replicate_index, (order, duration_ratio, throughput_ratio) in enumerate(
        (
            ("on,off", 0.9, 1.1),
            ("off,on", 1.0, 1.2),
            ("on,off", 1.1, 1.3),
        )
    ):
        job_id = str(100 + replicate_index)
        profile_enabled = replicate_index == 0
        _create_job(
            result_root,
            job_id=job_id,
            replicate_index=replicate_index,
            timing_order=order,
            profile_enabled=profile_enabled,
            duration_ratio=duration_ratio,
            throughput_ratio=throughput_ratio,
        )
        records.append(
            {
                "replicate_index": replicate_index,
                "timing_order": order,
                "profile_enabled": int(profile_enabled),
                "job_id": job_id,
                "submission_group": "group-a",
            }
        )
    submission.write_text("".join(json.dumps(record) + "\n" for record in records))
    return submission, result_root


def _run_collector(
    submission: Path,
    result_root: Path,
    output_dir: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(COLLECTOR),
            str(submission),
            str(result_root),
            "--output-json",
            str(output_dir / "aggregate.json"),
            "--output-csv",
            str(output_dir / "aggregate.csv"),
            "--bootstrap-samples",
            "1000",
            "--bootstrap-seed",
            "2606",
        ],
        capture_output=True,
        text=True,
    )


def test_collector_writes_deterministic_paired_aggregate_json_and_csv(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    output_dir = tmp_path / "output"
    result = _run_collector(submission, result_root, output_dir)
    assert result.returncode == 0, result.stderr

    json_path = output_dir / "aggregate.json"
    csv_path = output_dir / "aggregate.csv"
    first_json = json_path.read_bytes()
    first_csv = csv_path.read_bytes()
    aggregate = json.loads(first_json)
    assert aggregate["schema_version"] == 1
    assert aggregate["replicate_count"] == 3
    assert aggregate["replicate_indices"] == [0, 1, 2]
    assert aggregate["timing_orders"] == ["on,off", "off,on"]
    assert aggregate["profile_replicate"] == {
        "job_id": "100",
        "replicate_index": 0,
        "run_id": "100",
    }
    assert (
        aggregate["ratio_definition"]
        == "median(on measured steps) / median(off measured steps)"
    )
    assert set(aggregate["metrics"]) == {
        "e2e_duration",
        "generation_duration",
        "generation_finalize_duration",
        "logprob_duration",
        "policy_training_duration",
        "refit_duration",
        "e2e_throughput",
        "generation_throughput",
        "logprob_throughput",
        "policy_training_throughput",
        "refit_effective_throughput",
    }
    e2e_duration = aggregate["metrics"]["e2e_duration"]
    assert [item["ratio"] for item in e2e_duration["replicates"]] == [
        0.9,
        1.0,
        1.1,
    ]
    assert e2e_duration["median_ratio"] == 1.0
    assert e2e_duration["replicate_median_cv_percent"] == 10.0
    assert e2e_duration["paired_bootstrap_ci95"]["lower"] <= 1.0
    assert e2e_duration["paired_bootstrap_ci95"]["upper"] >= 1.0
    assert e2e_duration["recommendation"]["extend_to_six"] is True
    assert "CI crosses 1" in e2e_duration["recommendation"]["reasons"]
    assert set(e2e_duration["order_stratified"]) == {"off,on", "on,off"}
    assert e2e_duration["order_stratified"]["on,off"]["replicate_count"] == 2
    assert aggregate["metrics"]["e2e_throughput"]["median_ratio"] == 1.2

    with csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert {row["scope"] for row in rows} == {"aggregate", "order", "replicate"}
    assert len([row for row in rows if row["scope"] == "replicate"]) == 33
    assert len([row for row in rows if row["scope"] == "aggregate"]) == 11

    rerun = _run_collector(submission, result_root, output_dir)
    assert rerun.returncode == 0, rerun.stderr
    assert json_path.read_bytes() == first_json
    assert csv_path.read_bytes() == first_csv


def test_collector_requires_three_distinct_completed_jobs(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    records = submission.read_text().splitlines()
    submission.write_text("\n".join(records[:2]) + "\n")
    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "at least 3 distinct completed replicate jobs" in result.stderr

    submission.write_text("\n".join(records) + "\n")
    status_path = result_root / "101/status.json"
    status = json.loads(status_path.read_text())
    status["exit_code"] = 7
    _write_json(status_path, status)
    incomplete = _run_collector(submission, result_root, tmp_path / "output")
    assert incomplete.returncode != 0
    assert "job 101 is not completed successfully" in incomplete.stderr


def test_collector_requires_exactly_one_timing_summary_per_job(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    original = json.loads((result_root / "100/timing_summary.json").read_text())
    _write_json(result_root / "100/duplicate/timing_summary.json", original)
    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "job 100 expected exactly one timing_summary.json, found 2" in result.stderr


def test_collector_requires_both_alternating_orders(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    records = [json.loads(line) for line in submission.read_text().splitlines()]
    records[1]["timing_order"] = "on,off"
    submission.write_text("".join(json.dumps(record) + "\n" for record in records))
    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "submission timing order" in result.stderr


def test_collector_rejects_inconsistent_source_image_workload_and_metrics(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    manifest_path = result_root / "102/benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha"] = "c" * 40
    _write_json(manifest_path, manifest)
    source_result = _run_collector(submission, result_root, tmp_path / "output")
    assert source_result.returncode != 0
    assert "source identity differs across replicates" in source_result.stderr

    manifest["source_sha"] = "a" * 40
    manifest["image_sha256"] = "d" * 64
    _write_json(manifest_path, manifest)
    image_result = _run_collector(submission, result_root, tmp_path / "output")
    assert image_result.returncode != 0
    assert "image identity differs across replicates" in image_result.stderr

    manifest["image_sha256"] = "b" * 64
    manifest["topology"]["gpus_per_node"] = 8
    _write_json(manifest_path, manifest)
    workload_result = _run_collector(submission, result_root, tmp_path / "output")
    assert workload_result.returncode != 0
    assert "workload identity differs across replicates" in workload_result.stderr

    manifest["topology"]["gpus_per_node"] = 4
    manifest["resolved_metric_names"]["on"]["timing/train/get_logprobs"] = (
        "timing/train/get_logprobs"
    )
    manifest["resolved_metric_names"]["off"]["timing/train/get_logprobs"] = (
        "timing/train/get_logprobs"
    )
    _write_json(manifest_path, manifest)
    for raw_path in (result_root / "102/timing").glob("*/raw_timing.json"):
        raw = json.loads(raw_path.read_text())
        raw["resolved_metric_names"]["timing/train/get_logprobs"] = (
            "timing/train/get_logprobs"
        )
        _write_json(raw_path, raw)
    metrics_result = _run_collector(submission, result_root, tmp_path / "output")
    assert metrics_result.returncode != 0
    assert "resolved metric names differ across replicates" in metrics_result.stderr


def test_collector_rejects_unobserved_or_cross_replicate_workload_mismatch(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    summary_path = result_root / "101/timing_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["workload_equality_observed"] = False
    _write_json(summary_path, summary)
    equality_result = _run_collector(submission, result_root, tmp_path / "output")
    assert equality_result.returncode != 0
    assert "workload_equality_observed=true" in equality_result.stderr

    summary["workload_equality_observed"] = True
    _write_json(summary_path, summary)
    raw_path = result_root / "102/timing/0-on/raw_timing.json"
    raw = json.loads(raw_path.read_text())
    raw["measured_step_workload"][0]["total_num_tokens"] += 1
    _write_json(raw_path, raw)
    workload_result = _run_collector(submission, result_root, tmp_path / "output")
    assert workload_result.returncode != 0
    assert "measured workload differs across replicates" in workload_result.stderr


def test_collector_requires_profile_and_kernel_attribution_for_designated_job(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    (result_root / "100/kernel_attribution.json").unlink()
    missing_result = _run_collector(submission, result_root, tmp_path / "output")
    assert missing_result.returncode != 0
    assert "designated profile job 100" in missing_result.stderr
    assert "kernel_attribution.json" in missing_result.stderr

    _write_json(
        result_root / "100/kernel_attribution.json",
        {"passed": False, "arms": {}, "failures": ["missing kernels"]},
    )
    failed_result = _run_collector(submission, result_root, tmp_path / "output")
    assert failed_result.returncode != 0
    assert "kernel attribution did not pass" in failed_result.stderr


@pytest.mark.parametrize("unsafe_job_id", ("../outside", "/tmp/outside", "nested/job"))
def test_collector_rejects_job_ids_outside_benchmark_root(
    tmp_path: Path, unsafe_job_id: str
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    records = [json.loads(line) for line in submission.read_text().splitlines()]
    records[0]["job_id"] = unsafe_job_id
    submission.write_text("".join(json.dumps(record) + "\n" for record in records))
    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "job_id must be a safe single path component" in result.stderr


def test_collector_rejects_artifact_symlink_outside_benchmark_root(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    status_path = result_root / "100/status.json"
    outside_status = tmp_path / "outside-status.json"
    outside_status.write_text(status_path.read_text())
    status_path.unlink()
    status_path.symlink_to(outside_status)
    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "job 100 status escapes benchmark result root" in result.stderr


def test_collector_requires_identical_nonempty_on_off_metric_names(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    for job_dir in result_root.iterdir():
        manifest_path = job_dir / "benchmark_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["resolved_metric_names"]["on"]["timing/train/get_logprobs"] = (
            "timing/train/get_logprobs"
        )
        _write_json(manifest_path, manifest)
        on_path = next(job_dir.glob("timing/*-on/raw_timing.json"))
        raw = json.loads(on_path.read_text())
        raw["resolved_metric_names"]["timing/train/get_logprobs"] = (
            "timing/train/get_logprobs"
        )
        _write_json(on_path, raw)
    asymmetric = _run_collector(submission, result_root, tmp_path / "output")
    assert asymmetric.returncode != 0
    assert "ON/OFF resolved metric names must match" in asymmetric.stderr

    for job_dir in result_root.iterdir():
        manifest_path = job_dir / "benchmark_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for arm in ("on", "off"):
            manifest["resolved_metric_names"][arm]["timing/train/get_logprobs"] = None
        _write_json(manifest_path, manifest)
        for raw_path in job_dir.glob("timing/*/raw_timing.json"):
            raw = json.loads(raw_path.read_text())
            raw["resolved_metric_names"]["timing/train/get_logprobs"] = None
            _write_json(raw_path, raw)
    empty = _run_collector(submission, result_root, tmp_path / "output")
    assert empty.returncode != 0
    assert "resolved metric name" in empty.stderr
    assert "nonempty string" in empty.stderr


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected_error"),
    (
        ("source_sha", None, "source_sha must be a 40-character hexadecimal SHA"),
        ("image_sha256", "bad", "image_sha256 must be a 64-character hexadecimal SHA"),
        ("recipe", "", "recipe must be a nonempty string"),
        ("topology", {}, "topology must be a nonempty object"),
        (
            "fixed_config_evidence",
            {},
            "fixed_config_evidence must contain ON/OFF objects",
        ),
    ),
)
def test_collector_rejects_missing_or_malformed_identity_fields(
    tmp_path: Path, field: str, invalid_value: Any, expected_error: str
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    for manifest_path in result_root.glob("*/benchmark_manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        manifest[field] = invalid_value
        _write_json(manifest_path, manifest)
    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert expected_error in result.stderr


def test_collector_requires_raw_artifacts_to_evidence_timing_order(
    tmp_path: Path,
) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    raw_path = result_root / "100/timing/0-on/raw_timing.json"
    raw = json.loads(raw_path.read_text())
    raw["order_index"] = 1
    _write_json(raw_path, raw)
    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode != 0
    assert "raw timing order_index does not evidence timing order" in result.stderr


def test_collector_resolves_unique_successful_restarted_job_run(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    restarted_dir = result_root / "100-r1"
    shutil.move(result_root / "100", restarted_dir)
    status_path = restarted_dir / "status.json"
    status = json.loads(status_path.read_text())
    status["run_id"] = "100-r1"
    _write_json(status_path, status)
    for artifact in ("benchmark_manifest.json", "timing_summary.json"):
        artifact_path = restarted_dir / artifact
        value = json.loads(artifact_path.read_text())
        value["run_id"] = "100-r1"
        _write_json(artifact_path, value)
    for raw_path in restarted_dir.glob("timing/*/raw_timing.json"):
        raw = json.loads(raw_path.read_text())
        raw["run_id"] = "100-r1"
        _write_json(raw_path, raw)

    result = _run_collector(submission, result_root, tmp_path / "output")
    assert result.returncode == 0, result.stderr
    aggregate = json.loads((tmp_path / "output/aggregate.json").read_text())
    assert aggregate["profile_replicate"] == {
        "job_id": "100",
        "replicate_index": 0,
        "run_id": "100-r1",
    }


def test_collector_refuses_to_overwrite_submission_jsonl(tmp_path: Path) -> None:
    submission, result_root = _create_valid_inputs(tmp_path)
    original = submission.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            str(COLLECTOR),
            str(submission),
            str(result_root),
            "--output-json",
            str(submission),
            "--output-csv",
            str(tmp_path / "aggregate.csv"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "output paths must not overwrite the submission JSONL" in result.stderr
    assert submission.read_bytes() == original
