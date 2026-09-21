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

"""Tests for per-token generation log-prob extraction in the vLLM workers."""

import math
import re

import pytest

from nemo_rl.models.generation.vllm.utils import extract_sampled_logprobs


class FakeLogprob:
    """Stand-in for ``vllm.sequence.Logprob``."""

    def __init__(self, logprob):
        self.logprob = logprob


def _extract(token_ids, logprobs):
    return extract_sampled_logprobs(token_ids, logprobs, sample_label="request_idx=0")


def test_aligned_logprobs_are_extracted_unchanged():
    """Valid samples must be byte-identical to the previous behavior."""
    result = _extract(
        [11, 22, 33],
        [
            {11: FakeLogprob(-0.5)},
            {22: FakeLogprob(-1.25), 99: FakeLogprob(-4.0)},
            {33: FakeLogprob(-2.0)},
        ],
    )
    assert result.valid
    assert result.values == [-0.5, -1.25, -2.0]


def test_zero_logprob_is_a_legitimate_value():
    """0.0 means a certain token, not a missing entry."""
    result = _extract([11, 22], [{11: FakeLogprob(0.0)}, {22: FakeLogprob(-3.0)}])
    assert result.valid
    assert result.values == [0.0, -3.0]


def test_very_negative_logprobs_are_not_clamped():
    result = _extract([11], [{11: FakeLogprob(-99999.0)}])
    assert result.valid
    assert result.values == [-99999.0]


def test_no_generated_tokens_is_not_a_failure():
    result = _extract([], None)
    assert result.valid
    assert result.values == []


@pytest.mark.parametrize(
    ("token_ids", "logprobs", "expected_error"),
    [
        pytest.param(
            [11, 22, 33],
            [{11: FakeLogprob(-0.5)}, {22: FakeLogprob(-1.25)}],
            "token_count=3, logprob_count=2",
            id="short_list",
        ),
        pytest.param(
            [11, 22],
            [
                {11: FakeLogprob(-0.5)},
                {22: FakeLogprob(-1.25)},
                {33: FakeLogprob(-2.0)},
            ],
            "token_count=2, logprob_count=3",
            id="long_list",
        ),
        pytest.param(
            [11, 22, 33],
            [
                {11: FakeLogprob(-0.5)},
                {99: FakeLogprob(-1.25)},
                {33: FakeLogprob(-2.0)},
            ],
            "position=1, token_id=22",
            id="missing_sampled_token",
        ),
        pytest.param(
            [11, 22],
            [{11: FakeLogprob(-0.5)}, {}],
            "position=1, token_id=22",
            id="empty_position_mapping",
        ),
        pytest.param([11, 22], None, "no generation log-probs", id="absent_logprobs"),
        pytest.param(
            [11, 22],
            [{11: FakeLogprob(-0.5)}, [FakeLogprob(-1.0)]],
            "is a list, expected a mapping",
            id="position_is_not_a_mapping",
        ),
        pytest.param(
            [11, 22],
            [{11: FakeLogprob(-0.5)}, {22: object()}],
            "expected a vLLM Logprob carrying a float",
            id="entry_is_not_a_logprob",
        ),
        pytest.param(
            [11, 22],
            [{11: FakeLogprob(-0.5)}, {22: FakeLogprob("not-a-float")}],
            "expected a vLLM Logprob carrying a float",
            id="logprob_is_a_string",
        ),
        pytest.param(
            [11, 22],
            [{11: FakeLogprob(-0.5)}, {22: FakeLogprob(True)}],
            "expected a vLLM Logprob carrying a float",
            id="logprob_is_a_bool",
        ),
        pytest.param(
            [11, 22],
            [{11: FakeLogprob(-0.5)}, {22: FakeLogprob(math.nan)}],
            "log-prob is nan at position=1",
            id="nan",
        ),
        pytest.param(
            [11, 22],
            [{11: FakeLogprob(-0.5)}, {22: FakeLogprob(math.inf)}],
            "log-prob is inf at position=1",
            id="positive_inf",
        ),
        pytest.param(
            [11, 22],
            [{11: FakeLogprob(-0.5)}, {22: FakeLogprob(-math.inf)}],
            "log-prob is -inf at position=1",
            id="negative_inf",
        ),
        pytest.param(
            [11, 22],
            [{11: FakeLogprob(-0.5)}, {22: FakeLogprob(0.7)}],
            "log-prob is positive",
            id="positive_logprob",
        ),
    ],
)
def test_invalid_samples_are_reported(token_ids, logprobs, expected_error):
    result = _extract(token_ids, logprobs)

    assert not result.valid
    assert re.search(expected_error, result.error or ""), result.error
    # The values list is always the right length and always finite, so the
    # masking arithmetic downstream stays well-defined.
    assert len(result.values) == len(token_ids)
    assert all(math.isfinite(value) for value in result.values)


@pytest.mark.parametrize(
    "bad_value", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "neg_inf"]
)
def test_non_finite_values_become_zero_not_a_floor(bad_value):
    """NaN * 0 would be NaN, so an unusable value must not survive as one."""
    result = _extract([11, 22], [{11: FakeLogprob(-0.5)}, {22: FakeLogprob(bad_value)}])

    assert not result.valid
    assert result.values == [-0.5, 0.0]


def test_only_the_failed_positions_are_zeroed():
    result = _extract(
        [11, 22, 33],
        [
            {11: FakeLogprob(-0.5)},
            {99: FakeLogprob(-1.25)},
            {33: FakeLogprob(-2.0)},
        ],
    )

    assert not result.valid
    assert result.values == [-0.5, 0.0, -2.0]


def test_the_first_failure_is_the_reported_one():
    result = _extract(
        [11, 22, 33],
        [{11: FakeLogprob(math.nan)}, {99: FakeLogprob(-1.0)}, {33: FakeLogprob(-2.0)}],
    )

    assert "position=0" in (result.error or "")
