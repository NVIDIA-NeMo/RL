# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.vllm.vllm_step_trace import (
    VLLM_STEP_TRACE_AGGREGATION_VERSION,
    VLLM_STEP_TRACE_MODEL_STEP_DISCRIMINATOR,
    VLLM_STEP_TRACE_PHASE_CLASSIFIER,
    VLLM_STEP_TRACE_SCHEMA_VERSION,
    VLLM_STEP_TRACE_TIMELINE_COLUMN_ENCODING,
    VllmStepTraceBuffer,
    VllmStepTraceContractError,
    build_vllm_step_trace_record,
    summarize_exact_model_step_range,
)


def make_scheduler_stats(*, unpadded_tokens: int = 6) -> SimpleNamespace:
    debug_stats = SimpleNamespace(
        calc_duration=0.001,
        num_prefill_requests=1,
        num_decode_requests=2,
        context_breakdown={
            "num_prefill_requests": 1,
            "prefill_num_tokens": 4,
            "prefill_context_len": 8,
            "prefill_token_context_product": 32,
            "num_decode_requests": 2,
            "decode_num_tokens": 2,
            "decode_context_len": 21,
            "decode_token_context_product": 21,
        },
        num_flops_per_gpu_breakdown={"attention": 7, "mlp": 5},
        num_read_bytes_per_gpu_breakdown={"weights": 10, "kv": 3},
        num_write_bytes_per_gpu_breakdown={"kv": 4},
    )
    perf_stats = SimpleNamespace(
        num_flops_per_gpu=12,
        num_read_bytes_per_gpu=13,
        num_write_bytes_per_gpu=4,
        debug_stats=debug_stats,
    )
    return SimpleNamespace(
        num_running_reqs=3,
        num_waiting_reqs=5,
        kv_cache_usage=0.25,
        perf_stats=perf_stats,
        cudagraph_stats=SimpleNamespace(
            num_unpadded_tokens=unpadded_tokens,
            num_padded_tokens=8,
            num_paddings=2,
            runtime_mode="FULL",
        ),
    )


def make_iteration_stats() -> SimpleNamespace:
    return SimpleNamespace(
        num_generation_tokens=2,
        prompt_token_stats=SimpleNamespace(total=4, computed=4),
        num_preempted_reqs=0,
        finished_requests=[object()],
    )


def make_zero_work_cleanup_stats() -> SimpleNamespace:
    stats = make_scheduler_stats(unpadded_tokens=0)
    debug_stats = stats.perf_stats.debug_stats
    debug_stats.num_prefill_requests = 0
    debug_stats.num_decode_requests = 0
    debug_stats.context_breakdown = {
        field: 0 for field in debug_stats.context_breakdown
    }
    debug_stats.num_flops_per_gpu_breakdown = {}
    debug_stats.num_read_bytes_per_gpu_breakdown = {}
    debug_stats.num_write_bytes_per_gpu_breakdown = {}
    stats.perf_stats.num_flops_per_gpu = 0
    stats.perf_stats.num_read_bytes_per_gpu = 0
    stats.perf_stats.num_write_bytes_per_gpu = 0
    stats.cudagraph_stats = None
    stats.num_running_reqs = 0
    stats.num_waiting_reqs = 0
    stats.kv_cache_usage = 0.0
    return stats


def make_zero_context_stats(*, keep_cudagraph: bool) -> SimpleNamespace:
    stats = make_zero_work_cleanup_stats()
    if keep_cudagraph:
        stats.cudagraph_stats = SimpleNamespace(
            num_unpadded_tokens=0,
            num_padded_tokens=0,
            num_paddings=0,
            runtime_mode="FULL",
        )
    return stats


def test_adapter_records_actual_scheduled_composition_and_cadence() -> None:
    record = build_vllm_step_trace_record(
        make_scheduler_stats(),
        make_iteration_stats(),
        engine_idx=2,
        step_index=7,
        recorded_at_monotonic_ns=1_250,
        trace_window_started_monotonic_ns=1_000,
        previous_recorded_at_monotonic_ns=1_200,
    )

    assert record["phase_classifier"] == VLLM_STEP_TRACE_PHASE_CLASSIFIER
    assert record["schema_version"] == VLLM_STEP_TRACE_SCHEMA_VERSION == 4
    assert record["engine_idx"] == 2
    assert record["step_index"] == 7
    assert record["since_trace_window_start_ns"] == 250
    assert record["frontend_observed_cadence_ns"] == 50
    assert record["post_step_scheduler_state"] == {
        "running_requests": 3,
        "waiting_requests": 5,
        "kv_cache_usage": 0.25,
    }
    assert record["scheduled"]["request_count"] == 3
    assert record["scheduled"]["token_count"] == 6
    assert record["scheduled"]["token_context_product"] == 53
    assert record["scheduled"]["prefill"] == {
        "request_count": 1,
        "token_count": 4,
        "context_len_sum": 8,
        "mean_context_len": 8.0,
        "token_context_product": 32,
    }
    assert record["scheduled"]["decode"] == {
        "request_count": 2,
        "token_count": 2,
        "context_len_sum": 21,
        "mean_context_len": 10.5,
        "token_context_product": 21,
    }
    assert record["cudagraph"] == {
        "unpadded_tokens": 6,
        "padded_tokens": 8,
        "padding_tokens": 2,
        "runtime_mode": "FULL",
        "unpadded_matches_mfu_scheduled_tokens": True,
        "padded_equals_unpadded_plus_padding": True,
    }
    assert record["frontend_output"] == {
        "present": True,
        "emitted_generation_tokens": 2,
        "emitted_prompt_tokens": 4,
        "computed_prompt_tokens": 4,
        "preempted_requests": 0,
        "finished_requests": 1,
    }


def test_adapter_rejects_missing_mfu_debug_stats() -> None:
    stats = make_scheduler_stats()
    stats.perf_stats.debug_stats = None

    with pytest.raises(VllmStepTraceContractError, match="debug_stats is None"):
        build_vllm_step_trace_record(
            stats,
            None,
            engine_idx=0,
            step_index=0,
            recorded_at_monotonic_ns=10,
            trace_window_started_monotonic_ns=0,
            previous_recorded_at_monotonic_ns=None,
        )


def test_adapter_reports_cudagraph_and_mfu_token_disagreement() -> None:
    record = build_vllm_step_trace_record(
        make_scheduler_stats(unpadded_tokens=7),
        None,
        engine_idx=0,
        step_index=0,
        recorded_at_monotonic_ns=10,
        trace_window_started_monotonic_ns=0,
        previous_recorded_at_monotonic_ns=None,
    )

    assert not record["cudagraph"]["unpadded_matches_mfu_scheduled_tokens"]


def test_buffer_is_bounded_and_reports_errors_without_raising() -> None:
    timestamps = iter([100, 120, 160, 200])
    buffer = VllmStepTraceBuffer(
        [0],
        max_records_per_engine=1,
        monotonic_ns=lambda: next(timestamps),
    )

    buffer.record(make_scheduler_stats(), make_iteration_stats(), engine_idx=0)
    buffer.record(make_scheduler_stats(), None, engine_idx=0)
    invalid_stats = make_scheduler_stats()
    invalid_stats.perf_stats.debug_stats = None
    buffer.record(invalid_stats, None, engine_idx=0)
    snapshot = buffer.snapshot()

    assert "records_by_engine" not in snapshot
    aggregate = snapshot["aggregates_by_engine"][0]
    assert aggregate["model_step_count"] == 1
    assert aggregate["first_records"][0][
        "frontend_observed_cadence_ns"
    ] is None
    assert snapshot["dropped_records_by_engine"] == {0: 1}
    assert snapshot["error_count"] == 1
    assert snapshot["dropped_error_count"] == 0
    assert snapshot["errors"][0]["error_type"] == "VllmStepTraceContractError"
    assert snapshot["errors"][0]["step_index"] == 2


def test_buffer_ignores_non_model_control_callbacks_and_clear_resets_window() -> None:
    timestamps = iter([100, 120, 140, 200])
    buffer = VllmStepTraceBuffer([0], monotonic_ns=lambda: next(timestamps))

    buffer.record(SimpleNamespace(perf_stats=None), None, engine_idx=0)
    buffer.record(make_zero_work_cleanup_stats(), None, engine_idx=0)
    before_clear = buffer.snapshot()
    assert before_clear["ignored_non_step_callbacks_by_engine"] == {0: 2}
    assert before_clear[
        "ignored_non_step_callbacks_by_reason_by_engine"
    ] == {
        0: {
            "finished_request_cleanup_zero_scheduled_work": 1,
            "perf_stats_missing": 1,
        }
    }
    assert before_clear[
        "ignored_non_step_callbacks_with_frontend_token_activity_by_engine"
    ] == {0: 0}
    assert before_clear[
        "ignored_non_step_callbacks_with_unverifiable_frontend_output_by_engine"
    ] == {0: 0}

    buffer.clear()
    snapshot = buffer.snapshot()
    assert snapshot["trace_window_started_monotonic_ns"] == 200
    assert snapshot["ignored_non_step_callbacks_by_engine"] == {0: 0}
    assert snapshot[
        "ignored_non_step_callbacks_by_reason_by_engine"
    ] == {0: {}}
    assert "records_by_engine" not in snapshot
    assert snapshot["aggregates_by_engine"][0]["model_step_count"] == 0
    assert isinstance(snapshot["capture_hostname"], str)
    assert snapshot["capture_hostname"]


def test_buffer_flags_ignored_callback_with_frontend_token_activity() -> None:
    timestamps = iter([100, 120])
    buffer = VllmStepTraceBuffer([0], monotonic_ns=lambda: next(timestamps))

    buffer.record(
        make_zero_work_cleanup_stats(),
        make_iteration_stats(),
        engine_idx=0,
    )
    snapshot = buffer.snapshot()

    assert snapshot["error_count"] == 0
    assert snapshot["aggregates_by_engine"][0]["model_step_count"] == 0
    assert snapshot[
        "ignored_non_step_callbacks_with_frontend_token_activity_by_engine"
    ] == {0: 1}


def test_buffer_flags_ignored_callback_with_prompt_only_activity() -> None:
    timestamps = iter([100, 120])
    buffer = VllmStepTraceBuffer([0], monotonic_ns=lambda: next(timestamps))
    iteration_stats = make_iteration_stats()
    iteration_stats.num_generation_tokens = 0
    iteration_stats.prompt_token_stats = SimpleNamespace(total=1, computed=0)
    iteration_stats.finished_requests = []

    buffer.record(
        make_zero_work_cleanup_stats(),
        iteration_stats,
        engine_idx=0,
    )

    assert buffer.snapshot()[
        "ignored_non_step_callbacks_with_frontend_token_activity_by_engine"
    ] == {0: 1}


def test_buffer_accepts_terminal_static_read_estimate_as_control_callback() -> None:
    timestamps = iter([100, 120])
    buffer = VllmStepTraceBuffer([0], monotonic_ns=lambda: next(timestamps))
    stats = make_zero_work_cleanup_stats()
    stats.perf_stats.num_read_bytes_per_gpu = 13
    stats.perf_stats.debug_stats.num_read_bytes_per_gpu_breakdown = {
        "weights": 10,
        "unembed": 3,
    }

    buffer.record(stats, None, engine_idx=0)
    snapshot = buffer.snapshot()

    assert snapshot["error_count"] == 0
    assert snapshot["aggregates_by_engine"][0]["model_step_count"] == 0
    assert snapshot[
        "ignored_non_step_callbacks_by_reason_by_engine"
    ] == {
        0: {
            (
                "finished_request_cleanup_zero_scheduled_work_"
                "with_static_read_estimate"
            ): 1
        }
    }


def test_buffer_rejects_zero_work_callback_with_nonempty_scheduler() -> None:
    timestamps = iter([100, 120])
    buffer = VllmStepTraceBuffer([0], monotonic_ns=lambda: next(timestamps))
    stats = make_zero_work_cleanup_stats()
    stats.num_running_reqs = 1

    buffer.record(stats, None, engine_idx=0)

    snapshot = buffer.snapshot()
    assert snapshot["ignored_non_step_callbacks_by_engine"] == {0: 0}
    assert snapshot["error_count"] == 1
    assert "nonempty scheduler state" in snapshot["errors"][0]["error"]


def test_buffer_rejects_unbalanced_terminal_static_read_estimate() -> None:
    timestamps = iter([100, 120])
    buffer = VllmStepTraceBuffer([0], monotonic_ns=lambda: next(timestamps))
    stats = make_zero_work_cleanup_stats()
    stats.perf_stats.num_read_bytes_per_gpu = 13
    stats.perf_stats.debug_stats.num_read_bytes_per_gpu_breakdown = {
        "weights": 12
    }

    buffer.record(stats, None, engine_idx=0)

    snapshot = buffer.snapshot()
    assert snapshot["ignored_non_step_callbacks_by_engine"] == {0: 0}
    assert snapshot["error_count"] == 1
    assert "totals disagree with breakdowns" in snapshot["errors"][0]["error"]


def zero_work_with_cudagraph(stats: SimpleNamespace) -> None:
    zero = make_zero_context_stats(keep_cudagraph=True)
    stats.perf_stats = zero.perf_stats
    stats.cudagraph_stats = zero.cudagraph_stats


def zero_totals_with_nonzero_breakdown(stats: SimpleNamespace) -> None:
    debug_stats = stats.perf_stats.debug_stats
    debug_stats.num_prefill_requests = 0
    debug_stats.num_decode_requests = 0
    debug_stats.context_breakdown = {
        field: 0 for field in debug_stats.context_breakdown
    }
    stats.perf_stats.num_flops_per_gpu = 0
    stats.perf_stats.num_read_bytes_per_gpu = 0
    stats.perf_stats.num_write_bytes_per_gpu = 0
    stats.cudagraph_stats = None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda stats: setattr(stats, "cudagraph_stats", None),
        lambda stats: (
            setattr(stats.perf_stats.debug_stats, "num_prefill_requests", 0),
            setattr(stats.perf_stats.debug_stats, "num_decode_requests", 0),
            stats.perf_stats.debug_stats.context_breakdown.update(
                {
                    "num_prefill_requests": 0,
                    "num_decode_requests": 0,
                }
            ),
            setattr(stats, "cudagraph_stats", None),
        ),
        zero_work_with_cudagraph,
        lambda stats: (
            setattr(stats.perf_stats.debug_stats, "num_prefill_requests", 0),
            setattr(stats.perf_stats.debug_stats, "num_decode_requests", 0),
            stats.perf_stats.debug_stats.context_breakdown.update(
                {
                    field: 0
                    for field in stats.perf_stats.debug_stats.context_breakdown
                }
            ),
            setattr(stats, "cudagraph_stats", None),
        ),
        zero_totals_with_nonzero_breakdown,
    ],
    ids=[
        "positive-work-without-cudagraph",
        "tokens-without-requests",
        "zero-work-with-cudagraph",
        "zero-context-with-nonzero-estimated-work",
        "zero-totals-with-nonzero-breakdown",
    ],
)
def test_buffer_rejects_ambiguous_or_inconsistent_work_callbacks(
    mutate,
) -> None:
    timestamps = iter([100, 120])
    buffer = VllmStepTraceBuffer([0], monotonic_ns=lambda: next(timestamps))
    stats = make_scheduler_stats()
    mutate(stats)

    buffer.record(stats, None, engine_idx=0)
    snapshot = buffer.snapshot()

    assert snapshot["ignored_non_step_callbacks_by_engine"] == {0: 0}
    assert snapshot["aggregates_by_engine"][0]["model_step_count"] == 0
    assert snapshot["error_count"] == 1
    assert snapshot["errors"][0]["error_type"] == "VllmStepTraceContractError"


def test_buffer_online_aggregate_is_exact_and_retains_only_samples() -> None:
    timestamps = iter(range(100, 108))
    buffer = VllmStepTraceBuffer(
        [0],
        max_records_per_engine=20,
        monotonic_ns=lambda: next(timestamps),
    )

    for _ in range(7):
        buffer.record(
            make_scheduler_stats(),
            make_iteration_stats(),
            engine_idx=0,
        )
    snapshot = buffer.snapshot()
    aggregate = snapshot["aggregates_by_engine"][0]

    assert snapshot["schema_version"] == VLLM_STEP_TRACE_SCHEMA_VERSION == 4
    assert (
        snapshot["capture_contract"]["aggregation_version"]
        == VLLM_STEP_TRACE_AGGREGATION_VERSION
        == 1
    )
    assert (
        snapshot["capture_contract"]["timeline_column_encoding"]
        == VLLM_STEP_TRACE_TIMELINE_COLUMN_ENCODING
        == "python-array-q-v1"
    )
    assert (
        snapshot["capture_contract"]["model_step_discriminator"]
        == VLLM_STEP_TRACE_MODEL_STEP_DISCRIMINATOR
        == "mfu_scheduled_work_then_cudagraph_required-v2"
    )
    assert snapshot["capture_contract"]["retained_record_samples_per_edge"] == 3
    assert "records_by_engine" not in snapshot
    assert aggregate["model_step_count"] == 7
    assert len(aggregate["first_records"]) == 3
    assert len(aggregate["last_records"]) == 3
    assert aggregate["first_records"][0]["step_index"] == 0
    assert aggregate["last_records"][-1]["step_index"] == 6
    assert all(
        len(column) == 7
        for column in aggregate["timeline_columns"].values()
    )
    assert aggregate["runtime_mode_timeline"] == ["FULL"] * 7
    assert aggregate["distributions"]["scheduled_request_count"] == {3: 7}
    assert aggregate["distributions"]["scheduled_token_count"] == {6: 7}
    assert aggregate["scheduled"]["decode"] == {
        "request_count_sum": 14,
        "token_count_sum": 14,
        "context_len_sum": 147,
        "token_context_product_sum": 147,
    }
    estimated = aggregate["estimated_work_per_gpu"]
    assert estimated["flops_sum"] == 84
    assert estimated["flops_breakdown_component_sums"] == {
        "attention": 49,
        "mlp": 35,
    }
    assert estimated["read_bytes_breakdown_component_sums"] == {
        "weights": 70,
        "kv": 21,
    }
    assert aggregate["cudagraph"]["runtime_mode_model_step_counts"] == {
        "FULL": 7
    }
    assert aggregate["frontend_output"]["emitted_generation_tokens"] == 14


def test_exact_model_step_range_selects_contiguous_work_and_timing() -> None:
    timestamps = iter(range(100, 108))
    buffer = VllmStepTraceBuffer(
        [0],
        max_records_per_engine=20,
        monotonic_ns=lambda: next(timestamps),
    )
    for _ in range(7):
        buffer.record(
            make_scheduler_stats(),
            make_iteration_stats(),
            engine_idx=0,
        )

    selected = summarize_exact_model_step_range(
        {0: buffer.snapshot()},
        start_step_index=2,
        stop_step_index=6,
    )

    assert selected["schema"] == "nrl-vllm-exact-model-step-range-v1"
    assert selected["selected_model_step_count_per_engine"] == 4
    assert selected["engine_count"] == 1
    engine = selected["per_engine"][0]
    assert engine["available_model_step_count"] == 7
    assert engine["selected_model_step_count"] == 4
    assert engine["boundary_after_previous_step_monotonic_ns"] == 102
    assert engine["boundary_after_final_selected_step_monotonic_ns"] == 106
    assert engine["frontend_observed_elapsed_ns"] == 4
    assert engine["frontend_observed_model_steps_per_s"] == 1_000_000_000
    assert engine["minimum_scheduled_requests_per_step"] == 3
    assert engine["maximum_prefill_requests_per_step"] == 1
    assert engine["mean_decode_requests_per_step"] == 2
    assert engine["request_weighted_mean_decode_context_len"] == 10.5
    assert engine["aggregate_sums"]["estimated_flops_per_gpu"] == 48
    assert engine["runtime_mode_model_step_counts"] == {"FULL": 4}
    assert len(engine["selected_runtime_mode_sha256"]) == 64
    assert len(engine["selected_work_columns_sha256"]) == 64
    assert len(selected["all_selected_work_columns_sha256"]) == 64


@pytest.mark.parametrize(
    ("start_step_index", "stop_step_index"),
    ((0, 2), (3, 3), (4, 3)),
)
def test_exact_model_step_range_rejects_invalid_boundaries(
    start_step_index: int,
    stop_step_index: int,
) -> None:
    with pytest.raises(
        VllmStepTraceContractError,
        match="positive and nonempty",
    ):
        summarize_exact_model_step_range(
            {0: {}},
            start_step_index=start_step_index,
            stop_step_index=stop_step_index,
        )


def test_exact_model_step_range_fails_closed_on_dropped_records() -> None:
    timestamps = iter(range(100, 105))
    buffer = VllmStepTraceBuffer(
        [0],
        max_records_per_engine=20,
        monotonic_ns=lambda: next(timestamps),
    )
    for _ in range(4):
        buffer.record(
            make_scheduler_stats(),
            make_iteration_stats(),
            engine_idx=0,
        )
    snapshot = buffer.snapshot()
    snapshot["dropped_records_by_engine"][0] = 1

    with pytest.raises(
        VllmStepTraceContractError,
        match="dropped model steps",
    ):
        summarize_exact_model_step_range(
            {0: snapshot},
            start_step_index=1,
            stop_step_index=4,
        )
