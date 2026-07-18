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

import argparse
from types import SimpleNamespace

import pytest

from examples.swe_bench.benchmark_streaming_final_decode import (
    FinalGenerationMeasurement,
    _extract_completion_delta,
    _max_abs_logprob_delta,
    _modal_control_contract,
    _output_histogram,
    _paired_summary,
    _parity,
    _parse_positive_ints,
    _stable_streamed_prefix,
)
from examples.swe_bench.benchmark_streaming_tool_call_prefill import PromptTrace


def _measurement(
    *,
    token_ids: tuple[int, ...] = (7, 8),
    logprobs: tuple[float, ...] = (-0.1, -0.2),
    output_text: str = "ok",
    decoded_text: str = "ok",
    ttft: float = 0.2,
    total: float = 0.4,
) -> FinalGenerationMeasurement:
    return FinalGenerationMeasurement(
        output_kind="delta",
        cached_tokens=16,
        prompt_tokens=32,
        output_token_ids=token_ids,
        output_logprobs=logprobs,
        output_text=output_text,
        decoded_text=decoded_text,
        time_to_first_output_seconds=ttft,
        total_seconds=total,
    )


def test_parse_positive_ints_rejects_duplicates_and_non_positive_values() -> None:
    assert _parse_positive_ints("256, 512") == (256, 512)
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        _parse_positive_ints("0")
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        _parse_positive_ints("256,256")


def test_extract_completion_delta_uses_sampled_token_logprobs() -> None:
    completion = SimpleNamespace(
        token_ids=[7, 8],
        logprobs=[
            {7: SimpleNamespace(logprob=-0.1), 9: SimpleNamespace(logprob=-3.0)},
            {8: SimpleNamespace(logprob=-0.2)},
        ],
        text="ok",
    )
    request_output = SimpleNamespace(outputs=[completion])

    assert _extract_completion_delta(request_output) == (
        (7, 8),
        (-0.1, -0.2),
        "ok",
    )


def test_extract_completion_delta_rejects_missing_sampled_token() -> None:
    completion = SimpleNamespace(
        token_ids=[7],
        logprobs=[{8: SimpleNamespace(logprob=-0.2)}],
        text="",
    )
    with pytest.raises(RuntimeError, match="omitted sampled output token"):
        _extract_completion_delta(SimpleNamespace(outputs=[completion]))


def test_extract_completion_delta_allows_missing_optional_logprobs() -> None:
    completion = SimpleNamespace(token_ids=[7], logprobs=None, text="ok")

    assert _extract_completion_delta(
        SimpleNamespace(outputs=[completion]), require_logprobs=False
    ) == ((7,), (), "ok")


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_extract_completion_delta_rejects_nonfinite_logprobs(nonfinite: float) -> None:
    completion = SimpleNamespace(
        token_ids=[7],
        logprobs=[{7: SimpleNamespace(logprob=nonfinite)}],
        text="",
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        _extract_completion_delta(SimpleNamespace(outputs=[completion]))


def test_output_histogram_counts_semantic_outputs() -> None:
    first = _measurement()
    second = _measurement(token_ids=(9,), output_text="other", decoded_text="other")

    histogram = _output_histogram((first, second, first))

    assert [item["count"] for item in histogram] == [2, 1]
    assert histogram[0]["output_token_ids"] == (7, 8)


def test_parity_reports_each_exact_output_contract() -> None:
    reference = _measurement()
    assert _parity(reference, reference, logprob_atol=0.0)["all"]

    changed = _measurement(logprobs=(-0.1, -0.25))
    details = _parity(reference, changed, logprob_atol=0.04)
    assert details["output_token_ids"]
    assert not details["output_logprobs_exact"]
    assert not details["output_logprobs_within_atol"]
    assert details["max_abs_logprob_delta"] == pytest.approx(0.05)
    assert details["logprob_atol"] == 0.04
    assert details["output_text"]
    assert details["decoded_text"]
    assert not details["all"]

    assert _parity(reference, changed, logprob_atol=0.05)["all"]


def test_max_abs_logprob_delta_rejects_incompatible_lengths() -> None:
    assert _max_abs_logprob_delta(
        _measurement(), _measurement(logprobs=(-0.1,))
    ) == float("inf")


def test_modal_control_contract_accepts_candidate_matching_normal_majority() -> None:
    modal_a = _measurement(logprobs=(-0.1, -0.2))
    modal_b = _measurement(logprobs=(-0.11, -0.25))
    ordinary_outlier = _measurement(
        token_ids=(7, 9),
        output_text="other",
        decoded_text="other",
    )
    candidate = _measurement(logprobs=(-0.105, -0.23))

    details = _modal_control_contract((ordinary_outlier, modal_a, modal_b), candidate)

    assert details["modal_available"]
    assert details["modal_count"] == 2
    assert details["candidate_matches_modal"]
    assert details["normal_modal_logprob_drift"] == pytest.approx(0.05)
    assert details["candidate_modal_logprob_drift"] == pytest.approx(0.02)
    assert details["logprobs_within_modal_control_drift"]
    assert details["all"]


def test_modal_control_contract_rejects_three_way_control_split() -> None:
    controls = (
        _measurement(token_ids=(1,), logprobs=(-0.1,), output_text="a"),
        _measurement(token_ids=(2,), logprobs=(-0.1,), output_text="b"),
        _measurement(token_ids=(3,), logprobs=(-0.1,), output_text="c"),
    )

    details = _modal_control_contract(controls, controls[0])

    assert not details["modal_available"]
    assert not details["all"]


def test_modal_control_contract_keeps_logprob_drift_diagnostic() -> None:
    controls = (
        _measurement(logprobs=(-0.1, -0.2)),
        _measurement(logprobs=(-0.11, -0.21)),
        _measurement(logprobs=(-0.12, -0.22)),
    )
    candidate = _measurement(logprobs=(-0.5, -0.6))

    details = _modal_control_contract(controls, candidate)

    assert details["candidate_matches_modal"]
    assert not details["logprobs_within_modal_control_drift"]
    assert details["all"]


def test_stable_streamed_prefix_drops_mutable_suffix_and_margin() -> None:
    trace = PromptTrace(
        immutable_prefix_tokens=3,
        tool_output_tokens=5,
        mutable_suffix_tokens=2,
        authoritative_prefix=[1, 2, 3],
        initial=[1, 2, 3, 90, 91],
        snapshots=([1, 2, 3, 4, 5, 6, 90, 91],),
        final=[1, 2, 3, 4, 5, 6, 7, 8, 90, 91],
    )

    assert _stable_streamed_prefix(trace, stability_margin_tokens=1) == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_paired_summary_uses_control_minus_candidate_savings() -> None:
    controls = [_measurement(ttft=0.3, total=0.6)]
    candidates = [_measurement(ttft=0.2, total=0.45)]

    summary = _paired_summary(controls, candidates)

    assert summary["paired_time_to_first_output_savings_seconds"][
        "mean"
    ] == pytest.approx(0.1)
    assert summary["paired_total_savings_seconds"]["mean"] == pytest.approx(0.15)
    assert summary["paired_total_win_fraction"] == 1.0
