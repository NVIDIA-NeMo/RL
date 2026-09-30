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

import math

import numpy as np
import pytest

from nemo_rl.algorithms.critic_metrics import (
    average_precision,
    binary_roc_auc,
    compute_critic_evaluation,
    compute_critic_evaluation_suites,
)


def test_binary_ranking_metrics_handle_ties_and_perfect_ordering():
    labels = np.asarray([False, True, False, True])

    assert binary_roc_auc(labels, np.asarray([0.1, 0.8, 0.2, 0.9])) == 1.0
    assert average_precision(labels, np.asarray([0.1, 0.8, 0.2, 0.9])) == 1.0
    assert binary_roc_auc(labels, np.ones(4)) == 0.5


def test_evaluation_separates_full_controls_from_observed_continuations():
    records = [
        {
            "prediction": -0.1,
            "target": 0.0,
            "pivot_id": "a",
            "group_metadata": {"label_source": "assumed_deterministic_source_task"},
        },
        {
            "prediction": 1.1,
            "target": 1.0,
            "pivot_id": "b",
            "group_metadata": {"label_source": "assumed_deterministic_source_task"},
        },
        {
            "prediction": 0.2,
            "target": 0.25,
            "pivot_id": "c",
            "pass_count": 1,
            "rollout_count": 4,
            "group_metadata": {"label_source": "observed_continuation"},
        },
        {
            "prediction": 0.8,
            "target": 0.75,
            "pivot_id": "d",
            "pass_count": 3,
            "rollout_count": 4,
            "group_metadata": {"label_source": "observed_continuation"},
        },
    ]

    metrics, details = compute_critic_evaluation(records)

    assert metrics["num_valid_samples"] == 4
    assert metrics["count_provenance_rate"] == 0.5
    assert metrics["roc_auc_exact_success"] == 1.0
    assert metrics["roc_auc_exact_failure"] == 1.0
    assert metrics["out_of_range_rate"] == 0.5
    assert details["observed_continuation"]["count"] == 2
    assert details["by_group"]["label_source"]["observed_continuation"]["count"] == 2
    assert math.isnan(metrics["observed/roc_auc_all_success"])


def test_explained_variance_is_distinct_from_calibration_sensitive_r2():
    records = [
        {"prediction": target + 0.25, "target": target, "group_metadata": {}}
        for target in (0.0, 0.25, 0.75, 1.0)
    ]

    metrics, _ = compute_critic_evaluation(records)

    assert metrics["target_variance"] == pytest.approx(0.15625)
    assert metrics["residual_variance"] == pytest.approx(0.0)
    assert metrics["explained_variance"] == pytest.approx(1.0)
    assert metrics["r2_vs_mean"] == pytest.approx(0.6)


def test_evaluation_suites_do_not_mix_dense_and_anchor_targets():
    records = [
        {
            "prediction": 0.0,
            "target": 1.0,
            "evaluation_suite": "dense_exp",
            "group_metadata": {},
        },
        {
            "prediction": 0.8,
            "target": 1.0,
            "evaluation_suite": "anchor_raw",
            "anchor_kind": "pivot",
            "pass_count": 5,
            "rollout_count": 5,
            "group_metadata": {},
        },
        {
            "prediction": 0.2,
            "target": 0.0,
            "evaluation_suite": "anchor_raw",
            "anchor_kind": "pivot",
            "pass_count": 0,
            "rollout_count": 5,
            "group_metadata": {},
        },
        {
            "prediction": 0.4,
            "target": 0.5,
            "evaluation_suite": "anchor_exp",
            "anchor_kind": "root",
            "group_metadata": {},
        },
    ]

    metrics, details = compute_critic_evaluation_suites(records)

    assert metrics["mse"] == 1.0
    assert metrics["dense_exp/mse"] == 1.0
    assert metrics["anchor_raw/pivot/mse"] == pytest.approx(0.04)
    assert metrics["anchor_raw/pivot/explained_variance"] == pytest.approx(0.84)
    assert metrics["anchor_raw/pivot/r2_vs_mean"] == pytest.approx(0.84)
    assert metrics["anchor_raw/pivot/roc_auc_exact_success"] == 1.0
    assert metrics["anchor_exp/root/mse"] == pytest.approx(0.01)
    assert (
        details["evaluation_suites"]["anchor_raw"]["by_anchor_kind"]["pivot"][
            "observed_continuation"
        ]["count"]
        == 2
    )
