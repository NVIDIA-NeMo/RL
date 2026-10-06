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

"""CPU-only checks for prompt-group realized staleness statistics."""

import math
from collections import Counter

import pytest

from nemo_rl.experience.metric_utils import (
    calculate_staleness_metrics,
    is_histogram_metric,
    resolve_rollout_category,
)


def _assert_counts(metrics: dict[str, float], prefix: str, values: list[int]) -> None:
    expected = Counter(f"count_{lag}" if lag <= 40 else "count_ge_41" for lag in values)
    actual = {
        key.removeprefix(f"{prefix}/"): value
        for key, value in metrics.items()
        if key.startswith(f"{prefix}/count_")
    }
    assert {key: value for key, value in actual.items() if value} == expected
    assert len(actual) == 42
    assert sum(actual.values()) == len(values)
    assert f"{prefix}/distribution" not in metrics


def test_counts_groups_once_despite_unequal_sibling_counts():
    metrics = calculate_staleness_metrics(
        [("fresh_g0", 5), ("fresh_g1", 5), ("fresh_g2", 5), ("old_g0", 3)],
        train_weight_version=5,
    )

    assert metrics["staleness/total/mean"] == 1
    assert metrics["staleness/total/variance"] == 1
    assert metrics["staleness/total/std"] == 1
    assert metrics["staleness/total/num_groups"] == 2
    _assert_counts(metrics, "staleness/total", [0, 2])
    assert metrics["staleness/total/count_0"] == 1
    assert metrics["staleness/total/count_1"] == 0
    assert metrics["staleness/total/count_2"] == 1
    assert metrics["staleness/total/frac_zero"] == 0.5


def test_pools_chunks_instead_of_averaging_chunk_statistics():
    chunks = [[("a_g0", 10)], [("b_g0", 8), ("c_g0", 8), ("d_g0", 6)]]
    metrics = calculate_staleness_metrics(
        (sample for chunk in chunks for sample in chunk), train_weight_version=10
    )

    _assert_counts(metrics, "staleness/total", [0, 2, 2, 4])
    assert metrics["staleness/total/mean"] == 2
    assert metrics["staleness/total/variance"] == 2
    assert metrics["staleness/total/std"] == pytest.approx(math.sqrt(2))
    assert metrics["staleness/total/p50"] == 2
    assert metrics["staleness/total/p90"] == pytest.approx(3.4)
    assert metrics["staleness/total/p99"] == pytest.approx(3.94)
    assert metrics["staleness/total/max"] == 4


def test_uses_oldest_group_version_and_canonical_id_suffix():
    metrics = calculate_staleness_metrics(
        [("task_g12_g0", 7), ("task_g12_g1", 5), ("task_gx", 6)],
        train_weight_version=8,
    )

    _assert_counts(metrics, "staleness/total", [3, 2])
    assert metrics["staleness/total/num_groups"] == 2


def test_scalar_counts_have_fixed_buckets_and_overflow():
    metrics = calculate_staleness_metrics(
        [("a_g0", 60), ("b_g0", 20), ("c_g0", 19), ("d_g0", 0)],
        train_weight_version=60,
    )

    _assert_counts(metrics, "staleness/total", [0, 40, 41, 60])
    assert metrics["staleness/total/count_40"] == 1
    assert metrics["staleness/total/count_ge_41"] == 2
    counts = {key: value for key, value in metrics.items() if "/count_" in key}
    assert len(counts) == 42
    assert sum(counts.values()) == metrics["staleness/total/num_groups"]


@pytest.mark.parametrize("lag", [0, 2])
def test_single_group_has_finite_zero_variance(lag):
    metrics = calculate_staleness_metrics(
        [("group_g0", 4)], train_weight_version=4 + lag
    )

    assert metrics["staleness/total/variance"] == 0
    assert metrics["staleness/total/std"] == 0
    assert metrics["staleness/total/mean"] == lag
    assert metrics["staleness/total/p99"] == lag


def test_empty_population_does_not_fabricate_zero_lag():
    assert calculate_staleness_metrics([], train_weight_version=4) == {}


@pytest.mark.parametrize("version", [-1, 6, True, 1.5, "1"])
def test_invalid_sample_versions_fail_instead_of_clamping(version):
    with pytest.raises(ValueError, match="Invalid weight version"):
        calculate_staleness_metrics([("group_g0", version)], train_weight_version=5)


@pytest.mark.parametrize("version", [-1, True, 1.5])
def test_invalid_trainer_version_fails(version):
    with pytest.raises(ValueError, match="train_weight_version"):
        calculate_staleness_metrics([], train_weight_version=version)


def test_histogram_routing_does_not_treat_scalar_buckets_as_histograms():
    assert not is_histogram_metric("staleness/total/distribution")
    assert is_histogram_metric("agent/reward/histogram")
    assert is_histogram_metric("histogram/gen_tokens_length")
    assert not is_histogram_metric("staleness/total/count_0")
    assert not is_histogram_metric("staleness/total/variance")
    assert not is_histogram_metric("unrelated/distribution")


@pytest.mark.parametrize(
    ("extra_env_info", "task_name", "expected"),
    [
        ({"task_source": "ifbench", "agent_ref": {"name": "agent"}}, "gym", "ifbench"),
        ({"task_source": "  ", "agent_ref": {"name": "agent"}}, "gym", "agent"),
        ({"agent_ref": {"name": "agent"}}, "gym", "agent"),
        ({"agent_ref": {"name": ""}}, "math", "math"),
        (None, "math", "math"),
        ({}, None, "unknown"),
        (None, "", "unknown"),
        ({"task_source": " ifbench "}, None, "ifbench"),
    ],
)
def test_resolve_rollout_category(extra_env_info, task_name, expected):
    assert (
        resolve_rollout_category(extra_env_info=extra_env_info, task_name=task_name)
        == expected
    )


@pytest.mark.parametrize(
    ("extra_env_info", "task_name"),
    [({"task_source": 4}, "gym"), ({"agent_ref": {"name": []}}, "gym"), (None, 3)],
)
def test_invalid_category_labels_fail(extra_env_info, task_name):
    with pytest.raises(ValueError, match="category must be a string"):
        resolve_rollout_category(extra_env_info=extra_env_info, task_name=task_name)


def test_category_statistics_pool_groups_across_chunks_without_sibling_weighting():
    chunks = [
        [("a_g0", 10), ("a_g1", 10), ("b_g0", 8)],
        [("a_g2", 10), ("c_g0", 6), ("d_g0", 0)],
    ]
    metrics = calculate_staleness_metrics(
        (sample for chunk in chunks for sample in chunk),
        train_weight_version=10,
        sample_categories={
            "a_g0": "ifbench",
            "a_g1": "ifbench",
            "a_g2": "ifbench",
            "b_g0": "code",
            "c_g0": "ifbench",
            "d_g0": "code",
        },
    )
    _assert_counts(metrics, "staleness/total", [0, 2, 4, 10])
    assert metrics["staleness/total/mean"] == 4
    assert metrics["staleness/total/variance"] == 14
    for category, lags, mean, variance in [
        ("ifbench", [0, 4], 2, 4),
        ("code", [2, 10], 6, 16),
    ]:
        prefix = f"staleness/category/{category}"
        _assert_counts(metrics, f"{prefix}", lags)
        assert metrics[f"{prefix}/num_groups"] == 2
        assert metrics[f"{prefix}/mean"] == mean
        assert metrics[f"{prefix}/variance"] == variance
        assert metrics[f"{prefix}/std"] == math.sqrt(variance)
        assert sum(metrics[f"{prefix}/count_{lag}"] for lag in lags) == 2
    assert metrics["staleness/category/ifbench/frac_zero"] == 0.5


def test_missing_category_is_counted_under_unknown_for_old_checkpoint_tags():
    metrics = calculate_staleness_metrics(
        [("old_g0", 2), ("old_g1", 2), ("new_g0", 3)],
        train_weight_version=3,
        sample_categories={"new_g0": "math"},
    )
    _assert_counts(metrics, "staleness/category/unknown", [1])
    _assert_counts(metrics, "staleness/category/math", [0])
    assert metrics["staleness/total/num_groups"] == 2


def test_category_labels_are_escaped_without_collisions_or_overwriting_total():
    metrics = calculate_staleness_metrics(
        [("a", 0), ("b", 1), ("c", 2), ("d", 3)],
        train_weight_version=3,
        sample_categories={"a": "a/b", "b": "a%2Fb", "c": "a_b", "d": "total"},
    )
    _assert_counts(metrics, "staleness/category/a/b/total", [3])
    _assert_counts(metrics, "staleness/category/a%252Fb", [2])
    _assert_counts(metrics, "staleness/category/a_b", [1])
    _assert_counts(metrics, "staleness/category/total", [0])
    _assert_counts(metrics, "staleness/total", [3, 2, 1, 0])


def test_hierarchical_categories_do_not_overwrite_parent_statistics():
    metrics = calculate_staleness_metrics(
        [("a", 1), ("b", 2), ("c", 3)],
        train_weight_version=4,
        sample_categories={"a": "math", "b": "math/total", "c": "math/dapo-math-17k"},
        sample_ready_versions={"a": 2, "b": 3, "c": 4},
    )
    _assert_counts(metrics, "staleness/category/math/total", [3])
    _assert_counts(metrics, "staleness/category/math/total/total", [2])
    _assert_counts(metrics, "staleness/category/math/dapo-math-17k/total", [1])
    assert metrics["staleness/category/math/dapo-math-17k/pre_queue/mean"] == 1
    assert metrics["staleness/category/math/dapo-math-17k/in_queue/mean"] == 0
    assert metrics["staleness/total/num_groups"] == 3


def test_category_uses_oldest_sibling_version_and_retains_overflow():
    metrics = calculate_staleness_metrics(
        [("group_g0", 49), ("group_g1", 0)],
        train_weight_version=50,
        sample_categories={"group_g0": "math", "group_g1": "math"},
    )
    _assert_counts(metrics, "staleness/category/math", [50])
    assert metrics["staleness/category/math/count_ge_41"] == 1
    assert metrics["staleness/category/math/variance"] == 0


def test_conflicting_sibling_categories_fail_instead_of_splitting_a_group():
    with pytest.raises(ValueError, match="Inconsistent rollout categories"):
        calculate_staleness_metrics(
            [("group_g0", 0), ("group_g1", 0)],
            train_weight_version=1,
            sample_categories={"group_g0": "math", "group_g1": "code"},
        )


def test_category_distributions_are_not_histograms_and_counts_are_scalars():
    assert not is_histogram_metric("staleness/category/ifbench/distribution")
    assert not is_histogram_metric("staleness/category/a%2Fb/distribution")
    assert not is_histogram_metric("staleness/category/ifbench/count_0")
    assert not is_histogram_metric("staleness/category/ifbench/variance")


def test_empty_category_population_has_no_fabricated_statistics():
    assert (
        calculate_staleness_metrics(
            [], train_weight_version=0, sample_categories={"unused": "math"}
        )
        == {}
    )


def test_queue_decomposition_uses_group_weighting_and_preserves_total() -> None:
    metrics = calculate_staleness_metrics(
        [("a_g0", 10), ("a_g1", 11), ("b_g0", 12), ("c_g0", 15)],
        train_weight_version=15,
        sample_categories={
            "a_g0": "math",
            "a_g1": "math",
            "b_g0": "math",
            "c_g0": "code",
        },
        sample_ready_versions={"a_g0": 12, "a_g1": 12, "b_g0": 15, "c_g0": 15},
    )
    _assert_counts(metrics, "staleness/total", [5, 3, 0])
    _assert_counts(metrics, "staleness/pre_queue", [2, 3, 0])
    _assert_counts(metrics, "staleness/in_queue", [3, 0, 0])
    assert metrics["staleness/total/mean"] == pytest.approx(
        metrics["staleness/pre_queue/mean"] + metrics["staleness/in_queue/mean"]
    )
    for phase, values, mean, variance in [
        ("total", [5, 3], 4, 1),
        ("pre_queue", [2, 3], 2.5, 0.25),
        ("in_queue", [3, 0], 1.5, 2.25),
    ]:
        prefix = f"staleness/category/math/{phase}"
        _assert_counts(metrics, f"{prefix}", values)
        assert metrics[f"{prefix}/mean"] == mean
        assert metrics[f"{prefix}/variance"] == variance
        assert metrics[f"{prefix}/num_groups"] == 2
        _assert_counts(metrics, f"staleness/category/code/{phase}", [0])
    _assert_counts(metrics, "staleness/category/math", [5, 3])
    assert metrics["staleness/decomposition/num_groups"] == 3
    assert metrics["staleness/decomposition/missing_num_groups"] == 0
    assert metrics["staleness/decomposition/frac_known"] == 1


def test_old_checkpoint_groups_keep_total_without_fabricated_queue_lags() -> None:
    metrics = calculate_staleness_metrics(
        [("old_g0", 0), ("old_g1", 0), ("new_g0", 1)],
        train_weight_version=4,
        sample_categories={"new_g0": "math"},
        sample_ready_versions={"new_g0": 2},
    )
    _assert_counts(metrics, "staleness/total", [4, 3])
    _assert_counts(metrics, "staleness/pre_queue", [1])
    _assert_counts(metrics, "staleness/in_queue", [2])
    assert metrics["staleness/decomposition/num_groups"] == 1
    assert metrics["staleness/decomposition/missing_num_groups"] == 1
    assert metrics["staleness/decomposition/frac_known"] == 0.5
    _assert_counts(metrics, "staleness/category/unknown/total", [4])
    assert metrics["staleness/category/unknown/decomposition/frac_known"] == 0
    assert "staleness/category/unknown/pre_queue/mean" not in metrics
    assert "staleness/category/unknown/in_queue/mean" not in metrics


def test_decomposition_with_no_known_ready_versions_reports_only_coverage() -> None:
    metrics = calculate_staleness_metrics(
        [("old_g0", 0)], train_weight_version=1, sample_ready_versions={}
    )
    assert metrics["staleness/total/mean"] == 1
    assert metrics["staleness/decomposition/missing_num_groups"] == 1
    assert not any("/pre_queue/" in key or "/in_queue/" in key for key in metrics)


@pytest.mark.parametrize("ready", [-1, 1, 6, True, 2.5, "3"])
def test_invalid_ready_versions_fail(ready: object) -> None:
    with pytest.raises(ValueError, match="Invalid ready weight version"):
        calculate_staleness_metrics(
            [("group_g0", 2)],
            train_weight_version=5,
            sample_ready_versions={"group_g0": ready},
        )


@pytest.mark.parametrize("second_ready", [None, 3])
def test_inconsistent_sibling_ready_versions_fail(second_ready: int | None) -> None:
    with pytest.raises(ValueError, match="Inconsistent ready versions"):
        calculate_staleness_metrics(
            [("group_g0", 1), ("group_g1", 1)],
            train_weight_version=5,
            sample_ready_versions={"group_g0": 2, "group_g1": second_ready},
        )


@pytest.mark.parametrize("phase", ["total", "pre_queue", "in_queue"])
def test_all_phase_counts_are_scalars(phase: str) -> None:
    for prefix in ["staleness", "staleness/category/math"]:
        assert not is_histogram_metric(f"{prefix}/{phase}/distribution")
        assert not is_histogram_metric(f"{prefix}/{phase}/count_0")


def test_decomposition_counts_overflow() -> None:
    metrics = calculate_staleness_metrics(
        [("group_g0", 0)],
        train_weight_version=90,
        sample_ready_versions={"group_g0": 45},
    )
    for phase, lag in [("total", 90), ("pre_queue", 45), ("in_queue", 45)]:
        _assert_counts(metrics, f"staleness/{phase}", [lag])
        assert metrics[f"staleness/{phase}/count_ge_41"] == 1
