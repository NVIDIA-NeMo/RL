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

from numpy.testing import assert_allclose

from nemo_rl.environments.rewards import verl_geo3k_reward


def test_correct_answer_with_format():
    """Correct answer + proper </think> and \\boxed{} format → reward ~1.0."""
    response = "<think>The angle is 32 degrees so answer is A</think>\n\\boxed{A}"
    reward, is_correct = verl_geo3k_reward("A", response)
    assert_allclose(reward, 1.0, atol=1e-6)
    assert is_correct is True


def test_correct_answer_without_think_tag():
    """Correct answer but no </think> tag → accuracy only (0.9 with default format_score=0.1)."""
    response = "\\boxed{51}"
    reward, is_correct = verl_geo3k_reward("51", response)
    assert_allclose(reward, 0.9, atol=1e-6)
    assert is_correct is True


def test_wrong_answer_with_format():
    """Wrong answer but correct format → format score only (0.1 with default format_score=0.1)."""
    response = "<think>I think B</think>\n\\boxed{B}"
    reward, is_correct = verl_geo3k_reward("A", response)
    assert_allclose(reward, 0.1, atol=1e-6)
    assert is_correct is False


def test_no_boxed_at_all():
    """No \\boxed{} in response → reward 0.0."""
    response = "The answer is A"
    reward, is_correct = verl_geo3k_reward("A", response)
    assert_allclose(reward, 0.0, atol=1e-6)
    assert is_correct is False


def test_custom_format_score():
    """Custom format_score changes the weighting."""
    response = "<think>Thinking</think>\n\\boxed{A}"
    reward, _ = verl_geo3k_reward("A", response, format_score=0.2)
    assert_allclose(reward, 1.0, atol=1e-6)

    # Wrong answer with format, format_score=0.2 → 0.2
    reward, _ = verl_geo3k_reward("B", response, format_score=0.2)
    assert_allclose(reward, 0.2, atol=1e-6)


def test_numeric_answer_grading():
    """Numeric answers should be graded correctly by mathruler."""
    response = "<think>Counting circles: 4</think>\n\\boxed{4}"
    reward, is_correct = verl_geo3k_reward("4", response)
    assert_allclose(reward, 1.0, atol=1e-6)
    assert is_correct is True
