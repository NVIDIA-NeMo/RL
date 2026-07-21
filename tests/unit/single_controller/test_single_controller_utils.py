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

from __future__ import annotations

import math

import pytest
import torch

from nemo_rl.algorithms.single_controller_utils.utils import (
    aggregate_step_metrics,
    aggregate_step_metrics_multi_minibatch,
    reduce_advantage_pump_metrics,
)


def test_aggregate_step_metrics_single_result_list_is_compatible():
    result = {
        "loss": torch.tensor([1.0, 2.0]),
        "grad_norm": torch.tensor([3.0, 5.0]),
        "total_flops": 10.0,
        "num_ranks": 8,
        "moe_metrics": {"load_balancing_loss": 0.25},
        "mtp_metrics": {"mtp_1_loss": 0.5},
        "all_mb_metrics": {
            "probs_ratio_min": [math.inf, 0.8],
            "probs_ratio_max": [1.1, math.inf],
            "reward": [1.0, 3.0],
            "lr": [1.0e-4, 1.0e-4],
            "wd": [0.01, 0.01],
            "global_valid_seqs": [4.0, 4.0],
            "global_valid_toks": [100.0, 100.0],
            "custom_sum": [2.0, 3.0],
        },
    }

    expected = aggregate_step_metrics(result)

    assert aggregate_step_metrics_multi_minibatch([result]) == expected
    assert expected["loss"] == pytest.approx(1.5)
    assert expected["grad_norm"] == pytest.approx(4.0)
    assert expected["probs_ratio_min"] == pytest.approx(0.8)
    assert expected["probs_ratio_max"] == pytest.approx(1.1)
    assert expected["reward"] == pytest.approx(2.0)
    assert expected["custom_sum"] == pytest.approx(5.0)


def test_aggregate_step_metrics_combines_all_optimizer_results():
    results = [
        {
            "loss": torch.tensor([2.0]),
            "grad_norm": torch.tensor([4.0]),
            "total_flops": 10.0,
            "num_ranks": 8,
            "moe_metrics": {"load_balancing_loss": 0.1},
            "mtp_metrics": {"mtp_1_loss": 0.4},
            "all_mb_metrics": {
                "probs_ratio_min": [math.inf, 0.8],
                "probs_ratio_max": [1.2, math.inf],
                "probs_ratio_clamped_min": [math.inf],
                "probs_ratio_clamped_max": [math.inf],
                "reward": [1.0, 3.0],
                "lr": [1.0e-4, 1.0e-4],
                "wd": [0.01, 0.01],
                "global_valid_seqs": [2.0, 2.0],
                "global_valid_toks": [100.0, 100.0],
                "custom_sum": [1.0, 2.0],
            },
        },
        {
            "loss": 6.0,
            "grad_norm": 8.0,
            "total_flops": 20.0,
            "num_ranks": 8,
            "moe_metrics": {"load_balancing_loss": 0.2},
            "mtp_metrics": {"mtp_1_loss": 0.6},
            "all_mb_metrics": {
                "probs_ratio_min": [0.7],
                "probs_ratio_max": [1.5],
                "probs_ratio_clamped_min": [math.inf],
                "probs_ratio_clamped_max": [math.inf],
                "reward": [5.0],
                "lr": [2.0e-4],
                "wd": [0.02],
                "global_valid_seqs": [3.0],
                "global_valid_toks": [200.0],
                "custom_sum": [4.0],
            },
        },
    ]

    metrics = aggregate_step_metrics_multi_minibatch(results)

    assert metrics["loss"] == pytest.approx(4.0)
    assert metrics["grad_norm"] == pytest.approx(6.0)
    assert metrics["total_flops"] == pytest.approx(30.0)
    assert metrics["num_ranks"] == 8
    assert metrics["probs_ratio_min"] == pytest.approx(0.7)
    assert metrics["probs_ratio_max"] == pytest.approx(1.5)
    assert metrics["probs_ratio_clamped_min"] == -1.0
    assert metrics["probs_ratio_clamped_max"] == -1.0
    assert metrics["reward"] == pytest.approx(3.0)
    # Summed keys are normalized within one finish result (a result's list
    # sums to that minibatch's mean), so the step value is the mean of the
    # per-result sums — not the raw concatenated sum, which would inflate
    # these metrics by the minibatch count relative to legacy GRPO.
    assert metrics["custom_sum"] == pytest.approx((1.0 + 2.0 + 4.0) / 2)
    assert metrics["moe/load_balancing_loss"] == pytest.approx((0.1 + 0.2) / 2)
    assert metrics["mtp/mtp_1_loss"] == pytest.approx((0.4 + 0.6) / 2)

    # Counts are repeated within one finish result, then summed across the two
    # optimizer steps. A raw mean over the concatenated values would be wrong.
    assert metrics["global_valid_seqs"] == pytest.approx(5.0)
    assert metrics["global_valid_toks"] == pytest.approx(300.0)

    # The last finish result reports the LR/WD used by the final optimizer step.
    assert metrics["lr"] == pytest.approx(2.0e-4)
    assert metrics["wd"] == pytest.approx(0.02)


def test_aggregate_step_metrics_rejects_empty_result_list():
    with pytest.raises(ValueError, match="at least one train result"):
        aggregate_step_metrics_multi_minibatch([])


def test_reduce_advantage_pump_metrics_staleness_stats():
    out = reduce_advantage_pump_metrics(
        rewards=[torch.tensor([1.0, 0.0])],
        masked_advantages=[torch.tensor([0.5, -0.5])],
        sequence_lengths=[3, 4],
        staleness=[0, 1, 3],
    )

    assert out["staleness/mean"] == pytest.approx(4.0 / 3.0)
    assert out["staleness/max"] == pytest.approx(3.0)
    assert out["staleness/min"] == pytest.approx(0.0)


def test_reduce_advantage_pump_metrics_omits_staleness_when_absent():
    out = reduce_advantage_pump_metrics(
        rewards=[torch.tensor([1.0])],
        masked_advantages=[torch.tensor([0.5])],
        sequence_lengths=[2],
    )

    assert not any(key.startswith("staleness/") for key in out)
