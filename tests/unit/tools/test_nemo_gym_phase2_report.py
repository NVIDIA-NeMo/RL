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

import hashlib
import json
from pathlib import Path

import pytest

from tools.nemo_gym_phase2_report import (
    RESULT_MARKER,
    ROUTER_NATIVE_ACTIVITY_METRICS,
    ROUTER_NATIVE_METRICS_AUDITED,
    TIMING_MARKER,
    WARMUP_RESULT_MARKER,
    build_query_definitions,
    build_range_query_definitions,
    build_summary,
    parse_range_samples,
    render_markdown,
    summarize_cache_metrics,
    summarize_driver_log,
    summarize_eval_results,
    summarize_router_logs,
    validate_experiment_metadata,
    write_report_artifacts,
)


def _vector_response(samples: list[tuple[dict[str, str], float]]) -> dict:
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {"metric": labels, "value": [1000.0, str(value)]}
                for labels, value in samples
            ],
        },
    }


def _matrix_response(
    series: list[tuple[dict[str, str], list[tuple[float, float]]]],
) -> dict:
    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {
                    "metric": labels,
                    "values": [[timestamp, str(value)] for timestamp, value in values],
                }
                for labels, values in series
            ],
        },
    }


def _query_archive() -> dict:
    archive = {
        "schema_version": 1,
        "run_id": "run-7",
        "start_time": 900.0,
        "end_time": 1000.0,
        "window_seconds": 100,
        "range_step_seconds": 1,
        "queries": {},
        "range_queries": {},
    }
    for definition in build_query_definitions("run-7", 100):
        archive["queries"][definition.name] = {
            "promql": definition.promql,
            "response": _vector_response([]),
        }
    for definition in build_range_query_definitions("run-7"):
        archive["range_queries"][definition.name] = {
            "promql": definition.promql,
            "response": _matrix_response([]),
        }
    return archive


def _set_samples(
    archive: dict,
    name: str,
    samples: list[tuple[dict[str, str], float]],
) -> None:
    archive["queries"][name]["response"] = _vector_response(samples)


def _set_scalar(archive: dict, name: str, value: float) -> None:
    _set_samples(archive, name, [({}, value)])


def _set_range(
    archive: dict,
    name: str,
    series: list[tuple[dict[str, str], list[tuple[float, float]]]],
) -> None:
    archive["range_queries"][name]["response"] = _matrix_response(series)


def _eval_rows() -> list[dict]:
    rows = []
    for prompt_index in range(2):
        for generation_index in range(2):
            sample_index = prompt_index * 2 + generation_index
            started_at = 920.0 + sample_index * 4
            duration_s = 1.0 + sample_index
            reward = 0 if sample_index == 3 else 1
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "generation_index": generation_index,
                    "num_generations_per_prompt": 2,
                    "reward": reward,
                    "full_result": {
                        "response": {"status": "completed", "error": None},
                        "ng_trajectory": {
                            "schema_version": "1.0",
                            "task_id": f"task-{prompt_index}",
                            "rollout_id": f"{prompt_index}-{generation_index}",
                            "gaps": [
                                {"code": "turns_unavailable"},
                                {"code": "conversation_unavailable"},
                            ],
                            "model_calls": [
                                {
                                    "model_call_id": f"call-{sample_index}",
                                    "started_at": started_at,
                                    "completed_at": started_at + duration_s,
                                    "duration_ms": duration_s * 1000,
                                    "response_metadata": {
                                        "status_code": 200,
                                        "response_status": "completed",
                                        "finish_reason": "stop",
                                        "error_category": None,
                                    },
                                    "token_stats": {
                                        "prompt_tokens": 100,
                                        "completion_tokens": 25,
                                        "cached_tokens": 50,
                                    },
                                }
                            ],
                        },
                    },
                }
            )
    return rows


def _write_eval(path: Path, rows: list[dict] | None = None) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in (rows or _eval_rows())),
        encoding="utf-8",
    )


def _populate_formal_metrics(archive: dict) -> None:
    _set_samples(
        archive,
        "target_up_min_by_target",
        [
            ({"component": "vllm_router", "replica": "router"}, 1),
            ({"component": "vllm_backend", "replica": "0"}, 1),
            ({"component": "vllm_backend", "replica": "1"}, 1),
        ],
    )
    _set_scalar(archive, "router_cache_hits", 75)
    _set_scalar(archive, "router_cache_misses", 25)
    _set_samples(
        archive,
        "router_cache_metrics_provenance",
        [({"source": "debug_log_compat"}, 1)],
    )
    _set_samples(
        archive,
        "router_metrics_adapter_info",
        [({"source": "native_aggregate_compat", "policy": "cache_aware"}, 1)],
    )
    _set_samples(
        archive,
        "router_native_metric_present_by_metric",
        [
            (
                {"metric": metric},
                (
                    1
                    if metric
                    in ROUTER_NATIVE_ACTIVITY_METRICS | {"vllm_router_running_requests"}
                    else 0
                ),
            )
            for metric in sorted(ROUTER_NATIVE_METRICS_AUDITED)
        ],
    )
    _set_samples(
        archive,
        "router_worker_health_source",
        [({"source": "adapter_backend_health_probe"}, 1)],
    )
    _set_scalar(archive, "router_cache_threshold", 0.3)
    _set_scalar(archive, "router_cache_log_observations", 100)
    per_replica_metrics = {
        "backend_prefix_cache_hits_by_replica": [80, 70],
        "backend_prefix_cache_queries_by_replica": [100, 100],
        "backend_request_success_by_replica": [8, 8],
        "backend_prompt_tokens_by_replica": [800, 800],
        "backend_generation_tokens_by_replica": [600, 400],
        "backend_prefix_cache_hits_resets_by_replica": [0, 0],
        "backend_prefix_cache_queries_resets_by_replica": [0, 0],
        "backend_request_success_resets_by_replica": [0, 0],
        "backend_prompt_tokens_resets_by_replica": [0, 0],
        "backend_generation_tokens_resets_by_replica": [0, 0],
        "backend_preemptions_by_replica": [0, 0],
        "backend_running_mean_by_replica": [2, 2],
        "backend_running_p95_by_replica": [4, 4],
        "backend_running_max_by_replica": [6, 6],
        "backend_running_request_seconds_by_replica": [200, 200],
        "backend_waiting_mean_by_replica": [1, 1],
        "backend_waiting_p95_by_replica": [2, 2],
        "backend_waiting_max_by_replica": [3, 3],
        "backend_waiting_request_seconds_by_replica": [100, 100],
        "backend_kv_usage_max_by_replica": [0.5, 0.4],
    }
    for name, values in per_replica_metrics.items():
        _set_samples(
            archive,
            name,
            [({"replica": str(index)}, value) for index, value in enumerate(values)],
        )
    for metric in ("ttft", "itl", "e2e", "queue"):
        for percentile in ("p50", "p90", "p99"):
            _set_scalar(archive, f"backend_{metric}_{percentile}", 0.1)
    for name, value in {
        "router_requests": 16,
        "router_request_errors": 0,
        "router_retries": 0,
        "router_retries_exhausted": 0,
        "router_requests_resets": 0,
        "router_cache_hits_resets": 0,
        "router_cache_misses_resets": 0,
        "router_load_balancing_events": 1,
        "router_cb_transitions": 0,
        "router_cb_failures": 0,
        "router_cb_successes": 16,
        "router_active_workers_min": 2,
        "router_request_duration_p50": 0.1,
        "router_request_duration_p90": 0.2,
        "router_request_duration_p99": 0.3,
    }.items():
        _set_scalar(archive, name, value)
    _set_samples(
        archive,
        "router_worker_health_min_by_worker",
        [
            ({"worker": "http://worker-0:8000"}, 1),
            ({"worker": "http://worker-1:8000"}, 1),
        ],
    )
    for name, values in {
        "router_policy_decisions_by_worker": [8, 8],
        "router_processed_requests_by_worker": [8, 8],
        "router_cb_state_max_by_worker": [0, 0],
        "router_cb_transitions_by_worker": [0, 0],
        "router_cb_successes_by_worker": [8, 8],
        "router_cb_failures_by_worker": [0, 0],
    }.items():
        _set_samples(
            archive,
            name,
            [
                ({"worker": f"http://worker-{index}:8000"}, value)
                for index, value in enumerate(values)
            ],
        )
    _set_range(
        archive,
        "target_up_by_target",
        [
            (labels, [(900, 1), (1000, 1)])
            for labels in (
                {"component": "vllm_router", "replica": "router"},
                {"component": "vllm_backend", "replica": "0"},
                {"component": "vllm_backend", "replica": "1"},
            )
        ],
    )
    _set_range(
        archive,
        "target_scrape_age_seconds_by_target",
        [
            (labels, [(900, 1), (1000, 1)])
            for labels in (
                {"component": "vllm_router", "replica": "router"},
                {"component": "vllm_backend", "replica": "0"},
                {"component": "vllm_backend", "replica": "1"},
            )
        ],
    )
    for name in ("backend_running_by_replica", "backend_waiting_by_replica"):
        _set_range(
            archive,
            name,
            [
                (
                    {"replica": "0"},
                    [(900, 0), (920, 2), (930, 4), (940, 0), (1000, 0)],
                ),
                (
                    {"replica": "1"},
                    [(900, 0), (920, 3), (930, 5), (940, 0), (1000, 0)],
                ),
            ],
        )


def _manifest(router_logs: list[str] | None = None) -> dict:
    targets = [
        {
            "address": address,
            "metrics_url": f"http://{address}/metrics",
            "ready_at_registration": True,
            "labels": {
                "routing_policy": "cache_aware",
                "component": component,
                "replica": replica,
                "run_id": "run-7",
                "model": "model-1",
            },
        }
        for component, replica, address in (
            ("vllm_router", "router", "router:9000"),
            ("vllm_backend", "0", "worker-0:8000"),
            ("vllm_backend", "1", "worker-1:8000"),
        )
    ]
    return {
        "run_id": "run-7",
        "targets": targets,
        "registration": {"status": "registered"},
        "router_log_paths": router_logs or [],
        "monitoring_config": {
            "scrape_interval_s": 1,
            "initial_scrape_wait_s": 12,
            "final_scrape_wait_s": 12,
            "target_lifecycle": "dedicated",
        },
        "model_call_capture_dir": "/tmp/model-calls",
    }


def _driver_summary() -> dict:
    return {
        "benchmark_result": {
            "average_score": 0.75,
            "num_samples": 2,
            "rollout_metrics": [{"timing/rollout/total": 10.0}],
        },
        "warmup_result": {
            "status": "completed",
            "source": "measurement_workload_prefix",
            "requests": 1,
            "results": 1,
            "workload_sha256": "a" * 64,
            "model_call_capture_reset": True,
            "settle_seconds": 14,
        },
        "legacy_request_timing": None,
    }


def _router_log_summary() -> dict:
    return {
        "available": True,
        "session_affinity": {
            "available": False,
            "repeated_keys": 0,
            "violations": {},
            "passed": False,
        },
        "router_queue": {
            "status": "not_exposed_by_router_version",
            "router_version": "0.1.15",
            "log_event_counts": {},
        },
        "error_signal_counts": {},
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _experiment_metadata(workload: Path, warmup: Path) -> dict:
    return {
        "schema_version": 1,
        "engine": {"fresh": True, "launch_id": "slurm-123-engine-launch"},
        "workload_replay": {
            "faithful": True,
            "workload_sha256": _sha256(workload),
            "seed": "7",
            "num_prompts": 2,
            "num_generations_per_prompt": 2,
        },
        "warmup": {
            "completed": True,
            "source": "measurement_workload_prefix",
            "workload_sha256": _sha256(warmup),
            "requests": 1,
        },
        "software": {
            "nemo_rl_commit": "a" * 40,
            "container_digest": "sha256:" + "b" * 64,
        },
        "model": {
            "name": "model-1",
            "revision": "revision-1",
            "tokenizer": "model-1",
            "tokenizer_revision": "revision-1",
            "chat_template_sha256": "c" * 64,
        },
        "topology": {
            "tensor_parallel_size": 1,
            "data_parallel_size": 2,
            "num_nodes": 1,
            "gpus_per_node": 2,
        },
        "generation": {
            "sampling_parameters": {"temperature": 0, "top_p": 1},
            "concurrency": 16,
            "max_context_tokens": 8192,
            "max_output_tokens": 256,
        },
        "backend": {
            "prefix_caching_enabled": True,
            "scheduler_parameters": {"policy": "fcfs"},
            "batching_parameters": {"max_num_seqs": 256},
        },
        "router": {
            "enabled": True,
            "policy": "cache_aware",
            "session_affinity_header": "X-Session-ID",
            "cache_metrics_mode": "debug_log_compat",
            "cache_threshold": 0.3,
        },
    }


def test_queries_are_per_replica_and_tolerate_prefix_counter_suffix() -> None:
    definitions = {
        definition.name: definition.promql
        for definition in build_query_definitions('run"7', 120)
    }

    hits_query = definitions["backend_prefix_cache_hits_by_replica"]
    assert "sum by (replica)" in hits_query
    assert "prefix_cache_hits(_total)?" in hits_query
    assert 'run_id="run\\"7"' in hits_query
    assert "[120s]" in hits_query
    assert (
        "nemo_rl_vllm_router_request_errors_total"
        in definitions["router_request_errors"]
    )
    assert "or vector(0)" not in definitions["router_request_errors"]
    assert "or vector(0)" not in definitions["router_cache_hits"]
    assert "or vector(0)" not in definitions["router_cache_misses"]
    assert (
        "nemo_rl_vllm_router_cache_metrics_info"
        in definitions["router_cache_metrics_provenance"]
    )
    assert (
        "nemo_rl_vllm_router_metrics_adapter_info"
        in definitions["router_metrics_adapter_info"]
    )
    assert "vllm:num_requests_running" in definitions["backend_running_mean_by_replica"]
    assert (
        "nemo_rl_vllm_router_cb_outcomes_total"
        in definitions["router_cb_successes_by_worker"]
    )


def test_cache_summary_uses_global_counter_ratio_not_mean_of_replica_rates() -> None:
    archive = _query_archive()
    _set_scalar(archive, "router_cache_hits", 80)
    _set_scalar(archive, "router_cache_misses", 20)
    _set_samples(
        archive,
        "router_cache_metrics_provenance",
        [({"source": "debug_log_compat"}, 1)],
    )
    _set_scalar(archive, "router_cache_threshold", 0.3)
    _set_scalar(archive, "router_cache_log_observations", 100)
    _set_samples(
        archive,
        "backend_prefix_cache_hits_by_replica",
        [({"replica": "0"}, 900), ({"replica": "1"}, 1)],
    )
    _set_samples(
        archive,
        "backend_prefix_cache_queries_by_replica",
        [({"replica": "0"}, 1000), ({"replica": "1"}, 10)],
    )

    cache = summarize_cache_metrics(archive)

    assert cache["router_routing_cache"]["hit_rate"] == pytest.approx(0.8)
    assert cache["router_routing_cache"]["source"] == "debug_log_compat"
    assert cache["router_routing_cache"]["debug_log_observations"] == 100
    assert cache["backend_prefix_cache"]["hit_rate"] == pytest.approx(901 / 1010)
    assert cache["backend_prefix_cache"]["per_replica"]["0"][
        "hit_rate"
    ] == pytest.approx(0.9)
    assert cache["backend_prefix_cache"]["per_replica"]["1"][
        "hit_rate"
    ] == pytest.approx(0.1)


def test_eval_summary_uses_gym_model_call_capture_and_nonblocking_gaps(
    tmp_path: Path,
) -> None:
    path = tmp_path / "eval.jsonl"
    _write_eval(path)

    summary = summarize_eval_results(path)

    assert summary["correct"] == 3
    assert summary["accuracy"] == 0.75
    assert summary["coverage_pairs"] == 4
    assert summary["request_timing"]["samples"] == 4
    assert summary["request_timing"]["p50_s"] == pytest.approx(2.5)
    assert summary["request_timing"]["p95_s"] == pytest.approx(3.85)
    assert summary["request_timing"]["p99_over_p50"] == pytest.approx(1.588)
    assert summary["session_timing"]["samples"] == 4
    assert summary["session_timing"]["p99_s"] == pytest.approx(3.97)
    assert summary["session_timing"]["max_s"] == 4.0
    assert summary["makespan_s"] == 16.0
    assert summary["model_call_observability"]["complete"] is True
    assert summary["model_call_observability"]["blocking_gap_counts"] == {}
    assert summary["api_reported_cached_tokens"]["cached_token_share"] == 0.5
    assert summary["natural_termination_rate"] == 1.0


def test_eval_summary_rejects_duplicate_coverage(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    rows = _eval_rows()
    _write_eval(path, [rows[0], rows[0]])

    with pytest.raises(ValueError, match="duplicate"):
        summarize_eval_results(path)


def test_driver_log_summary_uses_final_marker_and_keeps_legacy_timing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "driver.log"
    path.write_text(
        "prefix "
        + TIMING_MARKER
        + json.dumps({"elapsed_s": 1.0, "status": "ok"})
        + "\nprefix "
        + TIMING_MARKER
        + json.dumps({"elapsed_s": 3.0, "status": "ok"})
        + "\nprefix "
        + WARMUP_RESULT_MARKER
        + json.dumps(
            {
                "status": "completed",
                "requests": 1,
                "results": 1,
                "workload_sha256": "a" * 64,
                "model_call_capture_reset": True,
                "settle_seconds": 14,
            }
        )
        + "\nprefix "
        + RESULT_MARKER
        + json.dumps({"average_score": 0.5, "num_samples": 1, "rollout_metrics": []})
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_driver_log(path)

    assert summary["legacy_request_timing"]["p50_s"] == 2.0
    assert summary["legacy_request_timing"]["p99_s"] == pytest.approx(2.98)
    assert summary["benchmark_result"]["average_score"] == 0.5
    assert summary["warmup_result"]["requests"] == 1


def test_router_log_summary_hashes_keys_and_detects_affinity_violation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "router.stdout.log"
    path.write_text(
        "Consistent hash routing: key='private-session' -> worker='worker-0'\n"
        "Consistent hash routing: key='private-session' -> worker='worker-1'\n"
        "request queue is full\n",
        encoding="utf-8",
    )

    summary = summarize_router_logs([path, tmp_path / "missing.log"])

    affinity = summary["session_affinity"]
    assert affinity["repeated_keys"] == 1
    assert affinity["passed"] is False
    assert "private-session" not in json.dumps(summary)
    assert len(next(iter(affinity["violations"]))) == 16
    assert summary["router_queue"]["status"] == "not_exposed_by_router_version"
    assert summary["router_queue"]["log_event_counts"]["queue_full"] == 1


def test_parse_range_samples_preserves_replica_series() -> None:
    archive = _query_archive()
    _set_range(
        archive,
        "backend_running_by_replica",
        [({"replica": "0"}, [(900, 0), (910, 2)])],
    )

    parsed = parse_range_samples(archive, "backend_running_by_replica")

    assert parsed[(("replica", "0"),)] == [(900.0, 0.0), (910.0, 2.0)]


def test_build_summary_combines_accuracy_cache_skew_and_hard_gates(
    tmp_path: Path,
) -> None:
    archive = _query_archive()
    _populate_formal_metrics(archive)
    eval_path = tmp_path / "eval.jsonl"
    _write_eval(eval_path)

    summary = build_summary(
        manifest=_manifest(),
        query_archive=archive,
        driver_summary=_driver_summary(),
        eval_summary=summarize_eval_results(eval_path),
        router_log_summary=_router_log_summary(),
    )

    assert summary["gates"]["passed"] is True
    assert summary["cache"]["backend_prefix_cache"]["hit_rate"] == 0.75
    assert summary["backend"]["generated_token_distribution"][
        "coefficient_of_variation"
    ] == pytest.approx(0.2)
    assert (
        summary["backend"]["generated_token_distribution"]["largest_replica_share"]
        == 0.6
    )
    assert summary["accuracy"]["value"] == 0.75
    assert summary["timing_calculation"] == {
        "duration_unit": "seconds",
        "percentile_method": "linear_interpolation",
        "implementation": "tools.nemo_gym_phase2_report.percentile",
    }
    assert summary["measurement"]["start_time"] == 920.0
    assert summary["measurement"]["end_time"] == 936.0
    assert summary["backend"]["running_by_replica"]["request_seconds"][
        "0"
    ] == pytest.approx(44.0)
    assert summary["backend"]["running_by_replica"]["mean"]["0"] == pytest.approx(2.75)
    report = render_markdown(summary)
    assert "Accuracy|75.00%" in report
    assert "Model-call samples|4" in report
    assert "Model-call p95|3.8500 s" in report
    assert "Tail amplification (model-call p99 / p50)|1.5880" in report
    assert "Session samples|4" in report
    assert "Session max|4.0000 s" in report
    assert "`linear_interpolation`" in report
    assert "Worker-load source: `backend_prometheus_num_requests_running`" in report
    assert "|http://worker-0:8000|8|8|1.0000|2.7500|4.0000|4.0000|" in report
    assert "Backend prefix cache|150|200|75.00%" in report
    assert "not_exposed_by_router_version" in report


def test_consistent_hash_reset_gate_ignores_inapplicable_cache_counters(
    tmp_path: Path,
) -> None:
    archive = _query_archive()
    _populate_formal_metrics(archive)
    _set_samples(
        archive,
        "router_metrics_adapter_info",
        [({"source": "native_aggregate_compat", "policy": "consistent_hash"}, 1)],
    )
    _set_samples(archive, "router_cache_hits_resets", [])
    _set_samples(archive, "router_cache_misses_resets", [])
    manifest = _manifest()
    for target in manifest["targets"]:
        target["labels"]["routing_policy"] = "consistent_hash"
    router_logs = _router_log_summary()
    router_logs["session_affinity"] = {
        "available": True,
        "repeated_keys": 1,
        "violations": {},
        "passed": True,
    }
    eval_path = tmp_path / "eval.jsonl"
    _write_eval(eval_path)

    summary = build_summary(
        manifest=manifest,
        query_archive=archive,
        driver_summary=_driver_summary(),
        eval_summary=summarize_eval_results(eval_path),
        router_log_summary=router_logs,
    )

    assert summary["router"]["counter_resets"]["router_cache_hits_resets"] is None
    assert summary["gates"]["router_counter_resets_zero"] is True
    assert summary["gates"]["passed"] is True


@pytest.mark.parametrize(
    ("failure_mode", "failed_gate"),
    [
        ("missing_provenance", "router_cache_metrics_provenance_available"),
        (
            "observation_mismatch",
            "router_cache_debug_log_observations_match",
        ),
    ],
)
def test_build_summary_rejects_unverifiable_router_cache_metrics(
    tmp_path: Path,
    failure_mode: str,
    failed_gate: str,
) -> None:
    archive = _query_archive()
    _populate_formal_metrics(archive)
    if failure_mode == "missing_provenance":
        _set_samples(archive, "router_cache_metrics_provenance", [])
    else:
        _set_scalar(archive, "router_cache_log_observations", 99)
    eval_path = tmp_path / "eval.jsonl"
    _write_eval(eval_path)

    summary = build_summary(
        manifest=_manifest(),
        query_archive=archive,
        driver_summary=_driver_summary(),
        eval_summary=summarize_eval_results(eval_path),
        router_log_summary=_router_log_summary(),
    )

    assert summary["gates"][failed_gate] is False
    assert summary["gates"]["passed"] is False


def test_build_summary_reports_scrape_staleness_and_counter_reset(
    tmp_path: Path,
) -> None:
    archive = _query_archive()
    _populate_formal_metrics(archive)
    _set_samples(
        archive,
        "backend_prompt_tokens_resets_by_replica",
        [({"replica": "0"}, 1), ({"replica": "1"}, 0)],
    )
    _set_range(
        archive,
        "target_scrape_age_seconds_by_target",
        [
            (labels, [(900, 1), (1000, 30 if replica == "0" else 1)])
            for labels, replica in (
                ({"component": "vllm_router", "replica": "router"}, "router"),
                ({"component": "vllm_backend", "replica": "0"}, "0"),
                ({"component": "vllm_backend", "replica": "1"}, "1"),
            )
        ],
    )
    eval_path = tmp_path / "eval.jsonl"
    _write_eval(eval_path)

    summary = build_summary(
        manifest=_manifest(),
        query_archive=archive,
        driver_summary=_driver_summary(),
        eval_summary=summarize_eval_results(eval_path),
        router_log_summary=_router_log_summary(),
    )

    assert summary["backend"]["counter_resets_total"] == 1
    assert summary["monitoring"]["max_observed_scrape_age_s"] == 30
    assert summary["gates"]["backend_counter_resets_zero"] is False
    assert summary["gates"]["prometheus_scrape_freshness_complete"] is False
    assert summary["gates"]["passed"] is False


def test_experiment_metadata_validates_replay_and_reproduction_identity(
    tmp_path: Path,
) -> None:
    workload = tmp_path / "workload.jsonl"
    workload.write_text('{"id":0}\n{"id":1}\n', encoding="utf-8")
    warmup = tmp_path / "warmup.jsonl"
    warmup.write_text('{"id":0}\n', encoding="utf-8")
    eval_path = tmp_path / "eval.jsonl"
    _write_eval(eval_path)
    metadata = _experiment_metadata(workload, warmup)

    validated = validate_experiment_metadata(
        metadata,
        target_manifest=_manifest(),
        eval_summary=summarize_eval_results(eval_path),
        workload_path=workload,
        warmup_workload_path=warmup,
        workload_seed="7",
    )

    assert validated["derived"]["engine_launch_id"] == "slurm-123-engine-launch"
    assert len(validated["derived"]["comparison_invariants_sha256"]) == 64
    metadata["workload_replay"]["faithful"] = False
    with pytest.raises(ValueError, match="faithful"):
        validate_experiment_metadata(
            metadata,
            target_manifest=_manifest(),
            eval_summary=summarize_eval_results(eval_path),
            workload_path=workload,
            warmup_workload_path=warmup,
            workload_seed="7",
        )


def test_experiment_metadata_rejects_warmup_not_executed_by_hook(
    tmp_path: Path,
) -> None:
    workload = tmp_path / "workload.jsonl"
    workload.write_text('{"id":0}\n{"id":1}\n', encoding="utf-8")
    warmup = tmp_path / "warmup.jsonl"
    warmup.write_text('{"id":"different"}\n', encoding="utf-8")
    eval_path = tmp_path / "eval.jsonl"
    _write_eval(eval_path)
    metadata = _experiment_metadata(workload, warmup)

    with pytest.raises(ValueError, match="measured workload prefix"):
        validate_experiment_metadata(
            metadata,
            target_manifest=_manifest(),
            eval_summary=summarize_eval_results(eval_path),
            workload_path=workload,
            warmup_workload_path=warmup,
            workload_seed="7",
        )


def test_write_report_artifacts_emits_contract_tree_and_checksums(
    tmp_path: Path,
) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    eval_path = sources / "eval.jsonl"
    _write_eval(eval_path)
    driver_path = sources / "driver.log"
    command_path = sources / "command.txt"
    command_path.write_text("python benchmark.py\n", encoding="utf-8")
    workload_path = sources / "workload.jsonl"
    workload_path.write_text(
        '{"prompt":"fixed-0"}\n{"prompt":"fixed-1"}\n', encoding="utf-8"
    )
    warmup_path = sources / "warmup.jsonl"
    warmup_path.write_text('{"prompt":"fixed-0"}\n', encoding="utf-8")
    driver_path.write_text(
        WARMUP_RESULT_MARKER
        + json.dumps(
            {
                "status": "completed",
                "source": "measurement_workload_prefix",
                "requests": 1,
                "results": 1,
                "workload_sha256": _sha256(warmup_path),
                "model_call_capture_reset": True,
                "settle_seconds": 14,
            }
        )
        + "\n"
        + RESULT_MARKER
        + json.dumps({"average_score": 0.75, "num_samples": 2, "rollout_metrics": []})
        + "\n",
        encoding="utf-8",
    )
    experiment_metadata = _experiment_metadata(workload_path, warmup_path)
    experiment_metadata_path = sources / "experiment.json"
    experiment_metadata_path.write_text(
        json.dumps(experiment_metadata), encoding="utf-8"
    )
    config_path = sources / "recipe.yaml"
    config_path.write_text("seed: 7\n", encoding="utf-8")
    router_stdout = sources / "router.stdout.log"
    router_stderr = sources / "router.stderr.log"
    router_stdout.write_text("router started\n", encoding="utf-8")
    router_stderr.write_text("", encoding="utf-8")
    backend_logs = {}
    for replica in ("0", "1"):
        path = sources / f"backend-{replica}.log"
        path.write_text(f"replica={replica} ready=true\n", encoding="utf-8")
        backend_logs[replica] = str(path)

    archive = _query_archive()
    _populate_formal_metrics(archive)
    target_manifest = _manifest([str(router_stdout), str(router_stderr)])
    target_manifest["backend_log_paths"] = backend_logs
    target_manifest_path = sources / "prometheus-targets.json"
    target_manifest_path.write_text(json.dumps(target_manifest), encoding="utf-8")
    eval_summary = summarize_eval_results(eval_path)
    summary = build_summary(
        manifest=target_manifest,
        query_archive=archive,
        driver_summary=summarize_driver_log(driver_path),
        eval_summary=eval_summary,
        router_log_summary=_router_log_summary(),
    )
    output = tmp_path / "report"

    finalized = write_report_artifacts(
        output,
        target_manifest_path=target_manifest_path,
        target_manifest=target_manifest,
        query_archive=archive,
        driver_log_path=driver_path,
        eval_results_path=eval_path,
        workload_path=workload_path,
        warmup_workload_path=warmup_path,
        workload_seed="7",
        repeat_id="repeat-01",
        command_path=command_path,
        experiment_metadata_path=experiment_metadata_path,
        experiment_metadata=experiment_metadata,
        config_paths=[config_path],
        versions={
            "python": "3.13.13",
            "nemo_rl": "0.4.0",
            "nemo_gym": "0.5.0",
            "vllm": "0.20.0",
            "vllm_router": "0.1.15",
            "uv": "0.11.28",
            "rl_insight": "0.2.1",
        },
        backend_log_paths=backend_logs,
        eval_summary=eval_summary,
        summary=summary,
    )

    required = {
        "command.txt",
        "manifest.json",
        "prometheus-targets.json",
        "config/00-recipe.yaml",
        "logs/router.stdout.log",
        "logs/router.stderr.log",
        "logs/backends/replica-0.log",
        "logs/backends/replica-1.log",
        "metrics/prometheus-query-results.json",
        "metrics/backend-per-replica.csv",
        "requests/per-request.jsonl",
        "evaluation/results.jsonl",
        "evaluation/workload.jsonl",
        "evaluation/warmup-workload.jsonl",
        "evaluation/outcomes.jsonl",
        "experiment/input-metadata.json",
        "experiment/metadata.json",
        "summary.json",
        "report.md",
        "artifact_checksums.sha256",
        "figures/completion-ecdf.svg",
        "figures/concurrency-drain.svg",
        "figures/instance-concurrency-heatmap.svg",
    }
    assert not [relative for relative in required if not (output / relative).is_file()]
    assert finalized["gates"]["required_artifact_inputs_complete"] is True
    checksums = (output / "artifact_checksums.sha256").read_text(encoding="utf-8")
    assert "  manifest.json\n" in checksums
    assert "  summary.json\n" in checksums
    assert "private-session" not in (output / "summary.json").read_text(
        encoding="utf-8"
    )
