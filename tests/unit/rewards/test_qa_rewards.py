# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from numpy.testing import assert_allclose

from nemo_rl.environments.rewards import (
    normalize_qa_answer,
    qa_exact_match_reward,
    qa_f1_reward,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("The Eiffel Tower", "eiffel tower"),
        ("  an   apple, a day!  ", "apple day"),
        ("U.S.A.", "usa"),
        ("Theater", "theater"),  # "the" inside a word is not an article
        ("...", ""),
    ],
)
def test_normalize_qa_answer(raw, expected):
    assert normalize_qa_answer(raw) == expected


class TestQAExactMatchReward:
    def test_match_after_normalization(self):
        response = "<think>...</think><answer>the Eiffel Tower.</answer>"
        assert qa_exact_match_reward("Eiffel Tower", response) == (1.0, True)

    def test_any_of_multiple_golds(self):
        response = "<answer>NYC</answer>"
        assert qa_exact_match_reward(["New York City", "NYC"], response) == (
            1.0,
            True,
        )

    def test_mismatch(self):
        response = "<answer>Paris</answer>"
        assert qa_exact_match_reward("Eiffel Tower", response) == (0.0, False)

    def test_missing_answer_block(self):
        assert qa_exact_match_reward("Paris", "The answer is Paris") == (0.0, False)

    def test_uses_last_answer_block(self):
        response = (
            "I should reply in <answer>Lyon</answer> form. "
            "<search>capital of France</search> ... <answer>Paris</answer>"
        )
        assert qa_exact_match_reward("Paris", response) == (1.0, True)

    def test_empty_normalized_answer_never_matches(self):
        # Both sides normalize to "" — must not count as a match.
        assert qa_exact_match_reward("...", "<answer>!!</answer>") == (0.0, False)

    def test_custom_tag(self):
        response = "<final>Paris</final>"
        assert qa_exact_match_reward("Paris", response, answer_tag="final") == (
            1.0,
            True,
        )


class TestQAF1Reward:
    def test_exact_match_scores_one(self):
        reward, is_correct = qa_f1_reward(
            "Eiffel Tower", "<answer>The Eiffel Tower</answer>"
        )
        assert_allclose(reward, 1.0)
        assert is_correct is True

    def test_partial_overlap(self):
        # pred tokens {barack, obama}, gold tokens {barack, hussein, obama}
        # precision 1, recall 2/3 -> F1 0.8
        reward, is_correct = qa_f1_reward(
            "Barack Hussein Obama", "<answer>Barack Obama</answer>"
        )
        assert_allclose(reward, 0.8)
        assert is_correct is False

    def test_best_over_golds(self):
        reward, _ = qa_f1_reward(
            ["Abraham Lincoln", "Barack Obama"], "<answer>Obama</answer>"
        )
        assert_allclose(reward, 2 / 3)

    def test_no_overlap_without_floor(self):
        assert qa_f1_reward("Paris", "<answer>London</answer>") == (0.0, False)

    def test_format_floor_applies_only_to_wellformed_answers(self):
        assert qa_f1_reward("Paris", "<answer>London</answer>", format_floor=0.1) == (
            0.1,
            False,
        )
        # no answer block -> no floor
        assert qa_f1_reward("Paris", "London", format_floor=0.1) == (0.0, False)

    def test_format_floor_does_not_cap_real_credit(self):
        reward, _ = qa_f1_reward(
            "Barack Hussein Obama", "<answer>Barack Obama</answer>", format_floor=0.1
        )
        assert_allclose(reward, 0.8)

    def test_empty_answer_scores_floor_not_match(self):
        assert qa_f1_reward("...", "<answer>!!</answer>", format_floor=0.1) == (
            0.1,
            False,
        )

    def test_invalid_floor(self):
        with pytest.raises(ValueError):
            qa_f1_reward("Paris", "<answer>Paris</answer>", format_floor=1.5)
