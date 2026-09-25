# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import pytest

from tests.unit.models.generation.chat_template_parity_common import (
    inclusive_token_span,
    prompt_suffix_after_turn,
    token_edit_similarity,
)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([], [], 1.0),
        ([1, 2, 3], [1, 2, 3], 1.0),
        ([1, 2, 3], [1, 9, 3], 2 / 3),
        ([1, 2], [1, 2, 3], 2 / 3),
        ([1, 2, 3], [1, 2], 2 / 3),
        ([1, 2], [3, 4], 0.0),
        ([], [1, 2], 0.0),
    ],
)
def test_token_edit_similarity(
    left: list[int], right: list[int], expected: float
) -> None:
    assert token_edit_similarity(left, right) == pytest.approx(expected)


def test_token_edit_similarity_is_symmetric() -> None:
    left = [1, 2, 3]
    right = [1, 9, 2, 3]
    assert token_edit_similarity(left, right) == token_edit_similarity(right, left)


def test_token_edit_similarity_090_threshold_allows_two_edits_in_twenty_tokens() -> (
    None
):
    left = list(range(20))
    two_edits = [999, 998] + list(range(2, 20))
    assert token_edit_similarity(left, two_edits) == pytest.approx(0.9)

    three_edits = [999, 998, 997] + list(range(3, 20))
    assert token_edit_similarity(left, three_edits) < 0.9


def test_inclusive_token_span_returns_markers_inclusive() -> None:
    tokens = [0, 7, 8, 42, 43, 9, 10, 99]
    assert inclusive_token_span(tokens, [7, 8], [9, 10]) == [7, 8, 42, 43, 9, 10]


def test_inclusive_token_span_returns_first_matching_span() -> None:
    tokens = [7, 1, 9, 7, 2, 9]
    assert inclusive_token_span(tokens, [7], [9]) == [7, 1, 9]


def test_inclusive_token_span_ignores_end_marker_before_start() -> None:
    tokens = [9, 0, 7, 1, 9]
    assert inclusive_token_span(tokens, [7], [9]) == [7, 1, 9]


def test_inclusive_token_span_rejects_missing_start_marker() -> None:
    with pytest.raises(AssertionError, match="start marker .* not found"):
        inclusive_token_span([1, 2, 3], [7], [3])


def test_inclusive_token_span_rejects_missing_end_marker() -> None:
    with pytest.raises(AssertionError, match="end marker .* not found"):
        inclusive_token_span([1, 2, 3], [1], [9])


@pytest.mark.parametrize(
    ("start_marker", "end_marker", "message"),
    [
        ([], [2], "start marker must not be empty"),
        ([1], [], "end marker must not be empty"),
    ],
)
def test_inclusive_token_span_rejects_empty_markers(
    start_marker: list[int], end_marker: list[int], message: str
) -> None:
    with pytest.raises(AssertionError, match=message):
        inclusive_token_span([1, 2], start_marker, end_marker)


def test_prompt_suffix_after_turn_returns_appended_tokens() -> None:
    turns = [
        {"prompt_token_ids": [1, 2], "generation_token_ids": [3, 4]},
        {"prompt_token_ids": [1, 2, 3, 4, 5, 6], "generation_token_ids": [7]},
    ]
    assert prompt_suffix_after_turn(turns, 0) == [5, 6]


def test_prompt_suffix_after_turn_can_be_empty() -> None:
    turns = [
        {"prompt_token_ids": [1], "generation_token_ids": [2]},
        {"prompt_token_ids": [1, 2], "generation_token_ids": [3]},
    ]
    assert prompt_suffix_after_turn(turns, 0) == []


def test_prompt_suffix_after_turn_rejects_changed_prefix() -> None:
    turns = [
        {"prompt_token_ids": [1], "generation_token_ids": [2]},
        {"prompt_token_ids": [1, 9, 3], "generation_token_ids": [4]},
    ]
    with pytest.raises(AssertionError):
        prompt_suffix_after_turn(turns, 0)
