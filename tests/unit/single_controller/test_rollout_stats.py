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

import torch

from nemo_rl.algorithms.single_controller_utils.rollout_stats import (
    accumulate_rollout_stats,
    new_rollout_stats_accumulator,
    per_sample_rollout_stats,
    reduce_rollout_stats,
)


def test_per_sample_stats_count_generated_tokens_and_turn_runs():
    token_mask = torch.tensor(
        [
            [0, 0, 1, 1, 0, 1, 0, 0],  # two runs, 3 tokens
            [1, 1, 0, 0, 0, 0, 0, 0],  # run starting at position 0
            [0, 0, 0, 0, 0, 0, 0, 0],  # nothing generated
            [0, 1, 0, 1, 0, 1, 0, 1],  # four single-token turns
        ],
        dtype=torch.float32,
    )
    stats = per_sample_rollout_stats(token_mask)
    assert stats["gen_tokens"].tolist() == [3.0, 2.0, 0.0, 4.0]
    assert stats["turns"].tolist() == [2.0, 1.0, 0.0, 4.0]
    assert stats["gen_tokens"].device.type == "cpu"


def _fill(
    acc, *, prompt_ids, rewards, sample_mask, gen_tokens, truncated=None, seq_lens=None
):
    """Build a chunk whose token_mask yields exactly ``gen_tokens`` per row."""
    width = int(max(gen_tokens)) + 2
    token_mask = torch.zeros(len(gen_tokens), width)
    for row, count in enumerate(gen_tokens):
        token_mask[row, 1 : 1 + count] = 1.0
    accumulate_rollout_stats(
        acc,
        prompt_ids=torch.tensor(prompt_ids),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        sample_mask=torch.tensor(sample_mask, dtype=torch.float32),
        token_mask=token_mask,
        truncated=None if truncated is None else torch.tensor(truncated),
        seq_lens=None if seq_lens is None else torch.tensor(seq_lens),
    )


def test_group_metrics_split_mixed_all_pass_all_fail():
    acc = new_rollout_stats_accumulator()
    # Three groups of four: all pass, all fail, mixed. Group ids as 1-D ints.
    _fill(
        acc,
        prompt_ids=[7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9],
        rewards=[1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        sample_mask=[1] * 12,
        gen_tokens=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        truncated=[False] * 11 + [True],
        seq_lens=[100] * 12,
    )
    out = reduce_rollout_stats(acc, max_seq_len=100)
    assert out["groups/count"] == 3
    assert out["groups/mixed_count"] == 1
    assert math.isclose(out["groups/mixed_frac"], 1 / 3, rel_tol=1e-6)
    assert math.isclose(out["groups/all_pass_frac"], 1 / 3, rel_tol=1e-6)
    assert math.isclose(out["groups/all_fail_frac"], 1 / 3, rel_tol=1e-6)
    assert math.isclose(out["groups/reward_std_mean"], 0.5 / 3, rel_tol=1e-6)
    assert math.isclose(out["groups/zero_advantage_sample_frac"], 8 / 12, rel_tol=1e-6)
    assert math.isclose(out["reward/pass_frac"], 6 / 12, rel_tol=1e-6)
    assert math.isclose(out["reward/std"], 0.5)
    assert math.isclose(out["truncated_frac"], 1 / 12, rel_tol=1e-6)
    # gen_tokens 10..120 step 10: torch.quantile interpolates linearly.
    assert math.isclose(out["gen_tokens/mean"], 65.0)
    assert math.isclose(out["gen_tokens/p50"], 65.0)
    assert math.isclose(out["gen_tokens/p90"], 109.0, rel_tol=1e-6)
    assert out["gen_tokens/max"] == 120.0
    assert math.isclose(
        out["gen_tokens/mean_pass"], (10 + 20 + 30 + 40 + 90 + 110) / 6, rel_tol=1e-6
    )
    assert math.isclose(
        out["gen_tokens/mean_fail"], (50 + 60 + 70 + 80 + 100 + 120) / 6, rel_tol=1e-6
    )
    assert out["turns/mean"] == 1.0 and out["turns/max"] == 1.0
    assert out["seq_len/mean"] == 100.0 and out["seq_len/max"] == 100.0
    # 100 > 0.9 * 100 for every row.
    assert out["seq_len/frac_above_90pct_ctx"] == 1.0


def test_invalid_rows_are_excluded_and_2d_prompt_ids_group_correctly():
    acc = new_rollout_stats_accumulator()
    # Prompt token ids (B, P): rows 0-2 share a prompt, row 3 is a lone prompt.
    _fill(
        acc,
        prompt_ids=[[1, 2, 3], [1, 2, 3], [1, 2, 3], [4, 5, 6]],
        rewards=[1, 0, 1, 1],
        sample_mask=[1, 1, 0, 1],  # row 2 masked out
        gen_tokens=[5, 6, 1000, 7],
    )
    out = reduce_rollout_stats(acc)
    # Only the first group has >= 2 valid rows; it is mixed (1, 0).
    assert out["groups/count"] == 1
    assert out["groups/mixed_frac"] == 1.0
    assert out["gen_tokens/max"] == 7.0  # the masked 1000-token row is ignored
    assert math.isclose(out["reward/pass_frac"], 2 / 3, rel_tol=1e-6)
    # The lone row 3 has no group spread -> zero advantage; rows 0, 1 are mixed.
    assert math.isclose(out["groups/zero_advantage_sample_frac"], 1 / 3, rel_tol=1e-6)
    # No truncated / seq_lens were accumulated -> those metrics are absent.
    assert "truncated_frac" not in out and "seq_len/mean" not in out


def test_chunks_accumulate_across_calls_and_empty_is_empty():
    assert reduce_rollout_stats(new_rollout_stats_accumulator()) == {}
    acc = new_rollout_stats_accumulator()
    _fill(acc, prompt_ids=[1, 1], rewards=[1, 0], sample_mask=[1, 1], gen_tokens=[2, 4])
    _fill(acc, prompt_ids=[2, 2], rewards=[0, 0], sample_mask=[1, 1], gen_tokens=[6, 8])
    out = reduce_rollout_stats(acc)
    assert out["groups/count"] == 2 and out["groups/mixed_count"] == 1
    assert math.isclose(out["gen_tokens/mean"], 5.0)
    # Chunks carry padded prompt token ids of different widths; the same token
    # prefix in two chunks is still two different groups (grouping is per chunk).
    acc = new_rollout_stats_accumulator()
    _fill(
        acc,
        prompt_ids=[[1, 2, 3, 0], [1, 2, 3, 0]],
        rewards=[1, 0],
        sample_mask=[1, 1],
        gen_tokens=[2, 4],
    )
    _fill(
        acc,
        prompt_ids=[[1, 2], [1, 2], [5, 6], [5, 6]],
        rewards=[1, 1, 0, 1],
        sample_mask=[1] * 4,
        gen_tokens=[1, 1, 1, 1],
    )
    out = reduce_rollout_stats(acc)
    assert out["groups/count"] == 3 and out["groups/mixed_count"] == 2
    assert math.isclose(out["groups/zero_advantage_sample_frac"], 2 / 6, rel_tol=1e-6)
    # An all-masked accumulator reduces to nothing.
    empty = new_rollout_stats_accumulator()
    _fill(
        empty, prompt_ids=[1, 1], rewards=[1, 0], sample_mask=[0, 0], gen_tokens=[2, 4]
    )
    assert reduce_rollout_stats(empty) == {}
