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
"""Producer-declared reductions, independent of rollout or training backends."""

import copy
import math
import statistics

import pytest

from nemo_rl.experience.metric_utils import (
    Metric,
    RolloutTelemetry,
    distribution,
    metrics_from_plain,
    optional_distributions,
    reduce_step,
)


@pytest.mark.parametrize("name", ["new_metric", "environment/math/new_metric", "a/b/c"])
def test_new_distribution_needs_no_reducer_name_registration(name: str) -> None:
    groups = [distribution(name, [10]), distribution(name, [20, 90])]
    before = copy.deepcopy(groups)
    result = reduce_step(groups)
    assert result[f"{name}/median"] == 20
    assert result[f"{name}/p50"] == 20
    assert result[f"{name}/mean"] == 40
    assert result[f"{name}/stddev"] == pytest.approx(statistics.stdev([10, 20, 90]))
    assert result[f"{name}/p95"] == 90
    assert result[f"{name}/p99"] == 90
    assert result[f"{name}/histogram"] == [10, 20, 90]
    assert groups == before
    assert reduce_step(groups) == result


@pytest.mark.parametrize("level", ["low", "high"])
def test_effort_alias_pools_raw_lengths(level: str) -> None:
    groups = [
        {
            f"median_length_{level}": Metric([length], "median"),
            f"mean_length_{level}": Metric([length], "mean"),
        }
        for length in [10, 20, 90]
    ]
    assert reduce_step(groups) == {
        f"median_length_{level}": 20,
        f"mean_length_{level}": 40,
    }


def test_optional_mean_uses_rows_but_other_statistics_use_present_values() -> None:
    result = reduce_step(
        [
            distribution("agent/score", [6, 9], rows=3),
            distribution("agent/score", [], rows=3),
        ]
    )
    assert result["agent/score/mean"] == 2.5
    assert result["agent/score/median"] == 7.5
    assert result["agent/score/min"] == 6
    assert result["agent/score/histogram"] == [6, 9]


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [("sum", 120), ("mean", 40), ("min", 10), ("max", 90), ("median", 20)],
)
def test_declared_reduction_wins_over_name(reduction, expected) -> None:
    assert reduce_step(
        [{"max_not_actually_a_max": Metric([10, 20, 90], reduction)}]
    ) == {"max_not_actually_a_max": expected}


def test_empty_and_singleton_distributions() -> None:
    assert reduce_step([distribution("empty", [])]) == {}
    result = reduce_step([distribution("one", [4])])
    assert math.isnan(result["one/stddev"])
    assert result["one/median"] == 4
    assert reduce_step([{"raw": Metric([], "concat")}]) == {"raw": []}


@pytest.mark.parametrize("rows", [-1, 1])
def test_distribution_rejects_inconsistent_population(rows: int) -> None:
    with pytest.raises(ValueError, match="cover all observed"):
        distribution("x", [1, 2], rows=rows)


def test_conflicting_reducers_fail_loudly() -> None:
    with pytest.raises(ValueError, match="disagree"):
        reduce_step([{"x": Metric([1], "mean")}, {"x": Metric([2], "sum")}])


def test_unknown_reducer_fails_even_for_empty_population() -> None:
    with pytest.raises(ValueError, match="Unknown metric reduction"):
        reduce_step([{"x": Metric([], "typo")}])  # type: ignore[arg-type]


def test_legacy_plain_metrics_are_not_silently_coerced() -> None:
    with pytest.raises(TypeError, match="no declared reduction"):
        reduce_step([{"x": 1}])  # type: ignore[dict-item]


def test_conflicting_output_declarations_fail() -> None:
    with pytest.raises(ValueError, match="overlap"):
        reduce_step([{**distribution("x", [1, 2]), "x/median": Metric([99], "mean")}])


def test_non_scalar_reductions_are_explicit() -> None:
    first_table, second_table = object(), object()
    result = reduce_step(
        [
            {
                "workers": Metric([{0: 3, 1: 5}], "sum_by_key"),
                "tables": Metric([first_table], "concat"),
                "raw": Metric([1, 2], "concat"),
            },
            {
                "workers": Metric([{0: 7, 2: 4}], "sum_by_key"),
                "tables": Metric([second_table], "concat"),
                "raw": Metric([3], "concat"),
            },
        ]
    )
    assert result["workers"] == {0: 10, 1: 5, 2: 4}
    assert result["tables"] == [first_table, second_table]
    assert result["raw"] == [1, 2, 3]


def test_native_adapter_retains_old_scalar_rules_and_pools_distributions() -> None:
    groups = [
        metrics_from_plain(
            {
                "x/histogram": values,
                "x/mean": statistics.fmean(values),
                "x/median": statistics.median(values),
                "min_reward": min(values),
                "max_reward": max(values),
                "total_turns": len(values),
                "max_turns_reached_rate": rate,
                "histogram/native": values,
                "unrelated_payload": object(),
            }
        )
        for values, rate in [([10], 0.0), ([20, 90], 1.0)]
    ]
    result = reduce_step(groups)
    assert result["x/mean"] == 32.5  # Native adapter preserves group means.
    assert result["x/median"] == 20
    assert result["min_reward"] == 10
    assert result["max_reward"] == 90
    assert result["total_turns"] == 3
    assert result["max_turns_reached_rate"] == 0.5
    assert result["histogram/native"] == [10, 20, 90]
    assert "unrelated_payload" not in result


@pytest.mark.parametrize("name", ["histogram/native", "native/histogram"])
def test_native_adapter_rejects_scalar_histograms(name: str) -> None:
    with pytest.raises(TypeError, match="must contain a list"):
        metrics_from_plain({name: 3.0})


def test_optional_fields_include_empty_groups_without_other_environments() -> None:
    groups = [
        optional_distributions("swe", [{"score": 6}, {"score": 9}, {}]),
        optional_distributions("swe", [{}, {}, {}]),
        optional_distributions("math", [{"score": 100}] * 4),
    ]
    before = copy.deepcopy(groups)
    result = reduce_step(groups)
    assert result["swe/score/mean"] == 2.5
    assert result["swe/score/median"] == 7.5
    assert result["swe/score/histogram"] == [6, 9]
    assert result["math/score/mean"] == 100
    assert groups == before


def test_colliding_optional_aliases_pool_declared_populations() -> None:
    result = reduce_step(
        [
            optional_distributions("a", [{"b/c": 0}, {}, {}]),
            optional_distributions("a/b", [{"c": 10}, {}]),
        ]
    )
    assert result["a/b/c/mean"] == 2
    assert result["a/b/c/median"] == 5
    assert result["a/b/c/stddev"] == pytest.approx(50**0.5)


def test_optional_distributions_are_independent_of_group_partition() -> None:
    rows = [{}, {"score": 1}, {"score": 2}, {}, {"score": 40}]
    split = reduce_step([optional_distributions("agent", [row]) for row in rows])
    assert split == reduce_step([optional_distributions("agent", rows)])
    assert split["agent/score/mean"] == 43 / 5
    assert split["agent/score/median"] == 2


def test_capture_roundtrip_retains_declarations_without_tables_or_alias_guessing() -> (
    None
):
    declared = {
        **distribution("arbitrary", [3]),
        **optional_distributions("swe", [{"score": 2}, {}]),
        "median_length_low": Metric([10, 20, 90], "median"),
        "table": Metric([object()], "concat"),
    }
    snapshot = RolloutTelemetry.from_metrics("swe", declared)
    assert "table" not in snapshot.metrics
    state = snapshot.to_state()
    restored = RolloutTelemetry.from_state(state)
    assert restored == snapshot
    assert type(state["metrics"]["arbitrary"]) is list
    state["metrics"]["swe/"][0][0]["swe/score"] = 100
    declared["arbitrary"].values[0] = 100
    assert restored.metrics["arbitrary"].values == [3]
    assert reduce_step([restored.to_metrics()])["swe/score/mean"] == 1
    assert reduce_step([restored.to_metrics()])["median_length_low"] == 20


@pytest.mark.parametrize(
    "pair",
    [
        ([1], "typo"),
        ([1], []),
        ([object()], "mean"),
        ([{"x": object()}], "field_dist"),
        ([1], "field_dist"),
    ],
)
def test_capture_rejects_invalid_declarations(pair) -> None:
    with pytest.raises(ValueError, match="telemetry"):
        RolloutTelemetry.from_state({"environment": "swe", "metrics": {"x": pair}})
