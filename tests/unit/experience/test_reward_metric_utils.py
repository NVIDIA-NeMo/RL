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

"""CPU-only contracts for reward spread definitions shared by both paths."""

import math
import statistics

import pytest

from nemo_rl.experience.metric_utils import calculate_reward_stddev_metrics


def test_between_prompt_variance_is_not_within_prompt_variance():
    metrics = calculate_reward_stddev_metrics([[0.0, 0.0], [4.0, 4.0]])
    assert metrics["total_reward/mean_within_prompt_stddev"] == 0.0
    assert metrics["total_reward/batch_stddev"] == pytest.approx((16 / 3) ** 0.5)


def test_prompt_mean_is_unweighted_with_unequal_valid_group_sizes():
    metrics = calculate_reward_stddev_metrics([[0.0, 2.0], [], [4.0, 4.0, 4.0]])
    assert metrics["total_reward/mean_within_prompt_stddev"] == pytest.approx(
        math.sqrt(2) / 2
    )
    assert metrics["total_reward/batch_stddev"] == pytest.approx(
        statistics.stdev([0.0, 2.0, 4.0, 4.0, 4.0])
    )


@pytest.mark.parametrize("groups", [[], [[]], [[], []]])
def test_empty_groups_do_not_emit_fake_zero_spread(groups):
    assert calculate_reward_stddev_metrics(groups) == {}


def test_singleton_retains_legacy_undefined_sample_stddev():
    metrics = calculate_reward_stddev_metrics([[2.0]])
    assert math.isnan(metrics["total_reward/batch_stddev"])
    assert math.isnan(metrics["total_reward/mean_within_prompt_stddev"])


def test_singleton_does_not_prevent_pooled_stddev():
    metrics = calculate_reward_stddev_metrics([[0.0], [2.0, 4.0]])
    assert metrics["total_reward/batch_stddev"] == 2.0
    assert math.isnan(metrics["total_reward/mean_within_prompt_stddev"])
