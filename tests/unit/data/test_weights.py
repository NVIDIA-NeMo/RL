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

import pytest

from nemo_rl.data.weights import (
    TaskWeightSpec,
    compute_quota,
    distribute_counts,
    normalize_weights,
)


def spec(task_name, weight=None, evaluation_only=False):
    return TaskWeightSpec(
        task_name=task_name, weight=weight, evaluation_only=evaluation_only
    )


class TestNormalizeWeights:
    def test_no_weights_declared_returns_empty(self):
        """No weights anywhere means the legacy unweighted path."""
        assert normalize_weights([spec("a"), spec("b")]) == {}

    def test_normalizes_to_one(self):
        weights = normalize_weights([spec("a", 3.0), spec("b", 1.0)])
        assert weights == {"a": 0.75, "b": 0.25}
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_already_normalized_weights_pass_through(self):
        assert normalize_weights([spec("a", 0.25), spec("b", 0.75)]) == {
            "a": 0.25,
            "b": 0.75,
        }

    def test_evaluation_only_excluded_from_denominator(self):
        """Eval-only datasets get weight 0 and do not dilute the others."""
        weights = normalize_weights(
            [spec("a", 3.0), spec("b", 1.0), spec("eval", 0.0, evaluation_only=True)]
        )
        assert weights == {"a": 0.75, "b": 0.25, "eval": 0.0}

    def test_evaluation_only_without_weight_is_allowed(self):
        """Eval-only entries need no weight since they never train."""
        weights = normalize_weights([spec("a", 1.0), spec("eval", None, True)])
        assert weights == {"a": 1.0, "eval": 0.0}

    def test_partial_weights_raise(self):
        """Weights are all-or-nothing; a silent default would skew the mixture."""
        with pytest.raises(ValueError, match="all-or-nothing"):
            normalize_weights([spec("a", 3.0), spec("b")])

    def test_partial_weights_error_names_offenders(self):
        with pytest.raises(ValueError, match="'b'"):
            normalize_weights([spec("a", 3.0), spec("b"), spec("c", 1.0)])

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            normalize_weights([spec("a", 3.0), spec("b", -1.0)])

    def test_all_zero_weights_raise(self):
        with pytest.raises(ValueError, match="must be positive"):
            normalize_weights([spec("a", 0.0), spec("b", 0.0)])

    def test_all_entries_evaluation_only_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            normalize_weights([spec("a", 1.0, True), spec("b", 1.0, True)])


class TestDistributeCounts:
    def test_exact_split(self):
        assert distribute_counts(32, [0.75, 0.25]) == [24, 8]

    def test_remainder_goes_to_largest_fractional_parts(self):
        # 10 * [1/3, 1/3, 1/3] = [3.33, 3.33, 3.33] -> one extra to the first.
        counts = distribute_counts(10, [1 / 3, 1 / 3, 1 / 3])
        assert sum(counts) == 10
        assert sorted(counts) == [3, 3, 4]

    def test_always_sums_to_total(self):
        for total in range(1, 65):
            counts = distribute_counts(total, [0.5, 0.3, 0.2])
            assert sum(counts) == total

    def test_distribute_remainder_false_floors(self):
        """Used where only whole slots matter; may sum to less than total."""
        counts = distribute_counts(
            10, [1 / 3, 1 / 3, 1 / 3], distribute_remainder=False
        )
        assert counts == [3, 3, 3]

    def test_deterministic(self):
        """Same weights must always yield the same split, step after step."""
        first = distribute_counts(17, [0.5, 0.3, 0.2])
        for _ in range(10):
            assert distribute_counts(17, [0.5, 0.3, 0.2]) == first


class TestComputeQuota:
    def test_quota_from_weights(self):
        weights = normalize_weights([spec("a", 3.0), spec("b", 1.0)])
        assert compute_quota(32, weights) == {"a": 24, "b": 8}

    def test_quota_sums_to_step_size(self):
        weights = normalize_weights([spec("a", 5.0), spec("b", 3.0), spec("c", 1.0)])
        quota = compute_quota(64, weights)
        assert sum(quota.values()) == 64

    def test_evaluation_only_omitted_from_quota(self):
        """Zero-weight tasks are absent, not mapped to 0, so callers can iterate
        the quota to get exactly the training tasks."""
        weights = normalize_weights([spec("a", 1.0), spec("eval", 0.0, True)])
        quota = compute_quota(8, weights)
        assert quota == {"a": 8}
        assert "eval" not in quota

    def test_small_step_size_starves_low_weight_task(self):
        """Documents the condition grpo.setup rejects: too few prompts per step
        for the configured weights leaves a task with no slots."""
        weights = normalize_weights([spec("a", 100.0), spec("b", 1.0)])
        quota = compute_quota(4, weights)
        assert quota["b"] == 0
