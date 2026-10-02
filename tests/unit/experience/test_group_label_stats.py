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
"""Per-harness and per-agent metrics from per-group counts summed over a step."""

from __future__ import annotations

import pytest
import torch

from nemo_rl.algorithms.single_controller_utils.rollout_stats import (
    per_sample_rollout_stats,
)
from nemo_rl.experience.group_label_stats import (
    AGENT_SCOPE,
    GROUP_STATS_PREFIX,
    HARNESS_SCOPE,
    generated_runs,
    group_label_stats,
    label_segment,
    numeric_result_fields,
    reduce_group_label_stats,
)

CLAUDE = "claude_code_sandboxed_agent"
OPENCODE = "opencode_sandboxed_agent"


def _group(labels, rewards, **per_rollout):
    count = len(rewards)
    return group_label_stats(
        labels,
        rewards=rewards,
        valid=per_rollout.get("valid", [True] * count),
        mask_sample=per_rollout.get("mask_sample", [False] * count),
        gen_tokens=per_rollout.get("gen_tokens", [0] * count),
        turns=per_rollout.get("turns", [0] * count),
        seq_lens=per_rollout.get("seq_lens", [0] * count),
        truncated=per_rollout.get("truncated", [False] * count),
        calls=per_rollout.get("calls", [0] * count),
        result_stats=per_rollout.get("result_stats", ()),
    )


def _summed(*groups):
    """What the controller does over a step's groups before reducing."""
    sums: dict[str, float] = {}
    for group in groups:
        for key, value in group.items():
            sums[key] = sums.get(key, 0.0) + value
    return sums


def test_each_harness_reports_only_its_own_trained_rollouts():
    claude = _group(
        [(HARNESS_SCOPE, CLAUDE), (AGENT_SCOPE, "swe_rebench_claude_code_sandboxed_agent")],
        [1.0, 0.0, 1.0, 0.0],
        valid=[True, True, True, False],  # rollout 3: capture placeholder
        mask_sample=[False, True, False, False],  # rollout 1: masked by a rule
        gen_tokens=[10, 20, 30, 0],
        turns=[2, 3, 4, 0],
        seq_lens=[100, 200, 300, 0],
        truncated=[False, False, True, False],
        calls=[5, 6, 7, 1],
    )
    opencode = _group(
        [(HARNESS_SCOPE, OPENCODE), (AGENT_SCOPE, "swe_rebench_opencode_sandboxed_agent")],
        [0.0, 0.0],
        gen_tokens=[4, 6],
        turns=[1, 1],
        seq_lens=[50, 70],
        calls=[1, 1],
    )

    metrics = reduce_group_label_stats(_summed(claude, opencode))

    cc = f"{HARNESS_SCOPE}/{CLAUDE}/"
    assert metrics[cc + "groups"] == 1.0
    assert metrics[cc + "rollouts"] == 4.0
    # Trained rollouts are 0 and 2, both passing.
    assert metrics[cc + "reward"] == 1.0
    assert metrics[cc + "pass_frac"] == 1.0
    assert metrics[cc + "reward_all_rollouts"] == 0.5
    assert metrics[cc + "trained_frac"] == 0.5
    assert metrics[cc + "placeholder_frac"] == 0.25
    assert metrics[cc + "mask_sample_frac"] == 0.25
    assert metrics[cc + "gen_tokens_mean"] == 20.0
    assert metrics[cc + "gen_tokens_mean_pass"] == 20.0
    assert cc + "gen_tokens_mean_fail" not in metrics
    assert metrics[cc + "turns_mean"] == 3.0
    assert metrics[cc + "seq_len_mean"] == 200.0
    assert metrics[cc + "truncated_frac"] == 0.5
    assert metrics[cc + "calls_per_rollout"] == 6.0
    assert metrics[cc + "all_pass_group_frac"] == 1.0
    assert metrics[cc + "mixed_group_frac"] == 0.0

    oc = f"{HARNESS_SCOPE}/{OPENCODE}/"
    assert metrics[oc + "reward"] == 0.0
    assert metrics[oc + "gen_tokens_mean_fail"] == 5.0
    assert metrics[oc + "all_fail_group_frac"] == 1.0

    assert metrics["agent/swe_rebench_claude_code_sandboxed_agent/reward"] == 1.0
    assert metrics["agent/swe_rebench_opencode_sandboxed_agent/rollouts"] == 2.0
    assert not any(key.startswith(GROUP_STATS_PREFIX) for key in metrics)


def test_groups_of_one_harness_pool_their_rollouts():
    label = [(HARNESS_SCOPE, CLAUDE)]
    mixed = _group(label, [1.0, 1.0, 1.0, 0.0])
    all_fail = _group(label, [0.0, 0.0])

    metrics = reduce_group_label_stats(_summed(mixed, all_fail))

    prefix = f"{HARNESS_SCOPE}/{CLAUDE}/"
    assert metrics[prefix + "groups"] == 2.0
    assert metrics[prefix + "rollouts"] == 6.0
    # 3 passes over 6 rollouts, not the mean of the groups' 0.75 and 0.0.
    assert metrics[prefix + "reward"] == 0.5
    assert metrics[prefix + "mixed_group_frac"] == 0.5
    assert metrics[prefix + "all_pass_group_frac"] == 0.0
    assert metrics[prefix + "all_fail_group_frac"] == 0.5


def test_result_fields_are_means_over_reporting_rollouts_for_the_harness_scope():
    group = _group(
        [(HARNESS_SCOPE, CLAUDE), (AGENT_SCOPE, "swe_next_claude_code_sandboxed_agent")],
        [1.0, 0.0],
        result_stats=[("claude_code_finished", 1.0, 1), ("harness_finished", 1.0, 2)],
    )

    metrics = reduce_group_label_stats(group)

    assert metrics[f"{HARNESS_SCOPE}/{CLAUDE}/result/harness_finished"] == 0.5
    assert metrics[f"{HARNESS_SCOPE}/{CLAUDE}/result/claude_code_finished"] == 1.0
    assert not any(
        "/result/" in key for key in metrics if key.startswith(f"{AGENT_SCOPE}/")
    )


def test_rates_without_a_denominator_are_omitted_not_zero():
    group = _group([(HARNESS_SCOPE, CLAUDE)], [1.0, 0.0], valid=[False, False])

    metrics = reduce_group_label_stats(group)

    prefix = f"{HARNESS_SCOPE}/{CLAUDE}/"
    assert metrics[prefix + "placeholder_frac"] == 1.0
    assert metrics[prefix + "reward_all_rollouts"] == 0.5
    for missing in ("reward", "pass_frac", "gen_tokens_mean", "mixed_group_frac"):
        assert prefix + missing not in metrics


def test_unlabelled_groups_and_other_metrics_add_nothing():
    assert _group((), [1.0]) == {}
    assert reduce_group_label_stats({"finalize/invalid_row_rate": 0.5}) == {}


def test_parallel_inputs_must_cover_every_rollout():
    with pytest.raises(ValueError, match="gen_tokens"):
        group_label_stats(
            [(HARNESS_SCOPE, CLAUDE)],
            rewards=[1.0, 0.0],
            valid=[True, True],
            mask_sample=[False, False],
            gen_tokens=[1],
            turns=[1, 1],
            seq_lens=[1, 1],
            truncated=[False, False],
            calls=[1, 1],
        )


def test_a_label_with_a_slash_stays_one_key_segment():
    assert label_segment("task-source:a/b") == "task-source:a_b"
    group = _group([(AGENT_SCOPE, "task-source:a/b")], [1.0])
    assert reduce_group_label_stats(group)["agent/task-source:a_b/reward"] == 1.0


def test_only_numeric_and_boolean_result_fields_are_kept():
    assert numeric_result_fields(
        {
            "reward": 1,
            "harness_finished": True,
            "rollout_id": "r",
            "instance_config": {"mask_sample": False},
            "bad": float("nan"),
        }
    ) == {"reward": 1.0, "harness_finished": 1.0}


def test_generated_runs_match_rollout_stats():
    rows = [
        [0, 0, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
    ]
    expected = per_sample_rollout_stats(torch.tensor(rows, dtype=torch.float32))
    for index, row in enumerate(rows):
        assert generated_runs(row) == (
            int(expected["gen_tokens"][index]),
            int(expected["turns"][index]),
        )
    assert generated_runs([]) == (0, 0)
