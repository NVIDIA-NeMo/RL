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

"""Tests for the TQ checkpoint benchmark matrix and report generator."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from tools.run_tq_checkpoint_benchmark_matrix import (
    BenchmarkCase,
    _case_signature,
    _new_state,
    _reconcile_completed_runs,
    _validate_existing_state,
    aggregate_rows,
    build_cases,
    flatten_result,
    metric_stats,
    render_markdown,
)


def _result() -> dict:
    return {
        "run_dir": "/tmp/suite/runs/baseline-r01",
        "save": {
            "config": {
                "num_rows": 8192,
                "min_seq_len": 4096,
                "max_seq_len": 4096,
                "payload_profile": "train-ready",
                "num_storage_units": 4,
                "producer_mode": "quiescent",
                "num_producers": 4,
            },
            "base_fill": {"put_logical_gib_per_s": 1.5},
            "checkpoint": {"duration_s": 2.0, "disk_bytes": 2 * 1024**3},
            "producers": {
                "before": {"rows_per_s": 0},
                "during": {"rows_per_s": 0, "put_latency_p95_ms": None},
                "after": {"rows_per_s": 0},
                "overlap_observed": False,
            },
        },
        "load": {
            "checkpoint": {"load_duration_s": 1.0},
            "restored": {
                "logical_tensor_bytes": 1024**3,
                "lengths": {"total_valid_tokens": 8192 * 4096},
            },
            "verification": {"status": "pass"},
        },
        "summary": {
            "effective_save_gib_per_s": 0.5,
            "effective_load_gib_per_s": 1.0,
            "checkpoint_to_logical_ratio": 2.0,
        },
    }


def test_core_matrix_reuses_one_baseline() -> None:
    cases = build_cases("core")
    case_ids = [case.case_id for case in cases]

    assert len(case_ids) == len(set(case_ids)) == 12
    baseline = next(case for case in cases if case.case_id == "baseline")
    assert set(baseline.series) == {
        "payload",
        "storage",
        "topology",
        "row_cardinality",
        "raggedness",
    }
    assert sum(case.case_id == "baseline" for case in cases) == 1


def test_fixed_token_row_cases_have_equal_token_volume() -> None:
    cases = build_cases("core")
    row_cases = [case for case in cases if "row_cardinality" in case.series]

    assert {
        case.num_rows * case.min_seq_len
        for case in row_cases
        if case.min_seq_len == case.max_seq_len
    } == {8192 * 4096}


def test_production_matrix_has_131k_rows_and_expected_profiles() -> None:
    cases = build_cases("production")

    assert len(cases) == 12
    assert {case.num_rows for case in cases} == {131072}
    assert {case.min_seq_len for case in cases} == {1024, 4096, 8192, 32768, 65536, 131072}
    assert {case.payload_profile for case in cases} == {
        "generation",
        "train-ready",
    }


def test_production_grid_is_train_ready_cross_product() -> None:
    cases = build_cases("production-grid")
    expected_sizes = {8192, 16384, 32768, 65536, 131072}

    assert len(cases) == 25
    assert {case.num_rows for case in cases} == expected_sizes
    assert {case.min_seq_len for case in cases} == expected_sizes
    assert {case.max_seq_len for case in cases} == expected_sizes
    assert {case.payload_profile for case in cases} == {"train-ready"}
    assert {case.num_storage_units for case in cases} == {8}
    assert {
        case.min_seq_len: case.batch_rows
        for case in cases
        if case.num_rows == 8192
    } == {
        8192: 256,
        16384: 128,
        32768: 64,
        65536: 32,
        131072: 16,
    }


def test_metric_stats() -> None:
    assert metric_stats([]) == {
        "mean": None,
        "stdev": None,
        "min": None,
        "max": None,
    }
    assert metric_stats([2.0]) == {
        "mean": 2.0,
        "stdev": 0.0,
        "min": 2.0,
        "max": 2.0,
    }
    stats = metric_stats([1.0, 2.0, 3.0])
    assert stats["mean"] == 2.0
    assert stats["stdev"] == pytest.approx(1.0)


def test_flatten_and_aggregate_success(tmp_path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_result()))
    case = next(case for case in build_cases("core") if case.case_id == "baseline")
    run = {
        "case_id": "baseline",
        "repetition": 1,
        "run_name": "baseline-r01",
        "status": "success",
        "result_path": str(result_path),
        "log_path": "/tmp/baseline.log",
        "error": None,
    }

    row = flatten_result(case=case, repetition=1, run_record=run)
    aggregate = aggregate_rows([row], [case])[0]

    assert row["logical_gib"] == 1.0
    assert row["checkpoint_gib"] == 2.0
    assert row["verification_status"] == "pass"
    assert aggregate["successful"] == 1
    assert aggregate["save_s"]["mean"] == 2.0
    assert aggregate["effective_load_gib_s"]["mean"] == 1.0


def test_markdown_contains_series_and_failures(tmp_path) -> None:
    cases = [build_cases("core")[0]]
    state = _new_state(
        suite="core",
        suite_name="test-suite",
        repetitions=1,
        cases=cases,
    )
    failed_row = {
        "case_id": "baseline",
        "repetition": 1,
        "run_name": "baseline-r01",
        "status": "failed",
        "error": "simulated failure",
        "log_path": "/tmp/failure.log",
    }
    aggregate = aggregate_rows([failed_row], cases)

    markdown = render_markdown(
        state=state,
        cases=cases,
        rows=[failed_row],
        aggregates=aggregate,
    )

    assert "# TransferQueue checkpoint benchmark report" in markdown
    assert "Payload profile" in markdown
    assert "`baseline-r01`" in markdown
    assert "simulated failure" in markdown


def test_case_state_is_json_serializable() -> None:
    case = BenchmarkCase(
        case_id="test",
        description="test",
        series=("smoke",),
        num_rows=1,
        min_seq_len=1,
        max_seq_len=1,
        payload_profile="generation",
        num_storage_units=1,
    )

    assert json.loads(json.dumps(asdict(case)))["series"] == ["smoke"]


def test_case_signature_survives_json_round_trip() -> None:
    cases = build_cases("smoke")
    state = _new_state(
        suite="smoke",
        suite_name="smoke-suite",
        repetitions=1,
        cases=cases,
    )
    restored = json.loads(json.dumps(state))

    assert restored["cases"] == _case_signature(cases)
    _validate_existing_state(
        restored,
        suite="smoke",
        repetitions=1,
        cases=cases,
    )


def test_reconcile_completed_run_after_runner_interruption(tmp_path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_result()))
    state = {
        "runs": [
            {
                "status": "running",
                "result_path": str(result_path),
                "finished_at": None,
                "returncode": None,
                "error": None,
            }
        ]
    }

    assert _reconcile_completed_runs(state)
    assert state["runs"][0]["status"] == "success"
    assert state["runs"][0]["returncode"] == 0


def test_reconcile_marks_incomplete_run_interrupted(tmp_path) -> None:
    state = {
        "runs": [
            {
                "status": "running",
                "result_path": str(tmp_path / "missing-result.json"),
                "finished_at": None,
                "returncode": None,
                "error": None,
            }
        ]
    }

    assert _reconcile_completed_runs(state)
    assert state["runs"][0]["status"] == "interrupted"
    assert "before producing" in state["runs"][0]["error"]
