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
import torch

from nemo_rl.evals.eval import (
    eval_cons_k,
    eval_pass_k,
)


def test_eval_pass_k_basic():
    """Test basic pass@k evaluation."""
    # Test case: 3 samples, 2 correct, k=1
    rewards = torch.tensor([1.0, 0.0, 1.0])
    result = eval_pass_k(rewards, num_tests_per_prompt=3, k=1)
    expected = 2 / 3
    assert isinstance(result, float)
    assert result == pytest.approx(expected, rel=1e-6)


def test_eval_pass_k_all_correct():
    """Test pass@k when all samples are correct."""
    rewards = torch.tensor([1.0, 1.0, 1.0])
    result = eval_pass_k(rewards, num_tests_per_prompt=3, k=1)
    expected = 1.0
    assert isinstance(result, float)
    assert result == pytest.approx(expected, rel=1e-6)


def test_eval_pass_k_none_correct():
    """Test pass@k when no samples are correct."""
    rewards = torch.tensor([0.0, 0.0, 0.0])
    result = eval_pass_k(rewards, num_tests_per_prompt=3, k=1)
    expected = 0.0
    assert isinstance(result, float)
    assert result == pytest.approx(expected, rel=1e-6)


def test_eval_pass_k_multiple_groups():
    """Test pass@k with multiple groups."""
    # Two groups: [1,0,1] and [0,1,0]
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    result = eval_pass_k(rewards, num_tests_per_prompt=3, k=1)
    expected = 1.0
    assert isinstance(result, float)
    assert result == pytest.approx(expected, rel=1e-6)


def test_eval_pass_k_edge_cases():
    """Test pass@k edge cases."""
    # k > num_tests_per_prompt
    rewards = torch.tensor([1.0, 0.0])
    result = eval_pass_k(rewards, num_tests_per_prompt=2, k=3)
    expected = 1.0
    assert isinstance(result, float)
    assert result == pytest.approx(expected, rel=1e-6)


def test_eval_cons_k_basic():
    """Test basic cons@k evaluation."""
    rewards = torch.tensor([1.0, 0.0, 1.0])
    extracted_answers = ["A", "B", "A"]
    result = eval_cons_k(
        rewards, num_tests_per_prompt=3, k=1, extracted_answers=extracted_answers
    )
    expected = 2 / 3
    assert isinstance(result, float)
    assert result == pytest.approx(expected, rel=1e-6)


def test_eval_cons_k_multiple_groups():
    """Test cons@k with multiple groups."""
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    extracted_answers = [
        "Correct",
        "Wrong1",
        "Correct",
        "Wrong2",
        "Correct",
        "Wrong3",
        "Correct",
        "Wrong4",
        "Correct",
        "Wrong4",
    ]
    result = eval_cons_k(
        rewards, num_tests_per_prompt=5, k=3, extracted_answers=extracted_answers
    )
    expected = 11 / 10
    assert isinstance(result, float)
    assert result == pytest.approx(expected, rel=1e-6)
