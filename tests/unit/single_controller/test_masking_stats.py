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

from nemo_rl.algorithms.single_controller_utils.masking_stats import (
    accumulate_masking_stats,
    new_masking_stats_accumulator,
    reduce_masking_stats,
)


def test_breakdown_counts_every_mask_source_once():
    acc = new_masking_stats_accumulator()
    # Two groups of 4. Row roles:
    #  0 clean pass  1 clean fail  2 env-flagged (fail)  3 placeholder
    #  4 truncated (fail, overlong on)  5 seq-logprob dropped (pass)  6 clean pass  7 clean fail
    prompt_ids = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2])
    rewards = torch.tensor([1, 0, 0, 0, 0, 1, 1, 0], dtype=torch.float32)
    sample_mask = torch.tensor([1, 1, 1, 0, 1, 1, 1, 1], dtype=torch.float32)
    mask_sample = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.bool)
    truncated = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.bool)
    final = torch.tensor(
        [1, 1, 0, 0, 0, 0, 1, 1], dtype=torch.float32
    )  # 5 dropped by seq-logprob
    baseline = torch.tensor(
        [1, 1, 1, 0, 1, 0, 1, 1], dtype=torch.float32
    )  # rows 2, 4 reinstated
    accumulate_masking_stats(
        acc,
        prompt_ids=prompt_ids,
        rewards=rewards,
        sample_mask=sample_mask,
        mask_sample=mask_sample,
        truncated=truncated,
        overlong_filtering=True,
        final_sample_mask=final,
        baseline_mask=baseline,
    )
    out = reduce_masking_stats(acc)
    assert out["masking/rows"] == 8 and out["masking/placeholder_rows"] == 1
    assert out["masking/env_flag_rows"] == 1 and math.isclose(
        out["masking/env_flag_frac"], 1 / 7, rel_tol=1e-6
    )
    assert (
        out["masking/truncated_rows"] == 1 and out["masking/truncated_masked_rows"] == 1
    )
    assert out["masking/seq_logprob_error_rows"] == 1
    assert out["masking/masked_rows"] == 3 and math.isclose(
        out["masking/masked_frac"], 3 / 7, rel_tol=1e-6
    )
    assert out["masking/trained_rows"] == 4 and math.isclose(
        out["masking/trained_frac"], 4 / 7, rel_tol=1e-6
    )
    assert out["masking/baseline_rows"] == 6 and out["masking/reinstated_rows"] == 2
    assert out["masking/reward_mean_trained"] == 0.5  # rows 0,1,6,7 -> 1,0,1,0
    assert out["masking/reward_mean_env_flag"] == 0.0
    assert out["masking/reward_mean_truncated"] == 0.0
    assert math.isclose(
        out["masking/reward_mean_masked"], 1 / 3, rel_tol=1e-6
    )  # rows 2,4,5 -> 0,0,1
    assert out["masking/groups"] == 2 and out["masking/groups_lt2_trained_frac"] == 0.0


def test_truncation_only_counts_as_masked_when_filtering_is_on():
    acc = new_masking_stats_accumulator()
    common = dict(
        prompt_ids=torch.tensor([1, 1]),
        rewards=torch.tensor([1.0, 0.0]),
        sample_mask=torch.ones(2),
        mask_sample=torch.zeros(2, dtype=torch.bool),
        truncated=torch.tensor([True, False]),
    )
    accumulate_masking_stats(
        acc,
        overlong_filtering=False,
        final_sample_mask=torch.ones(2),
        baseline_mask=torch.ones(2),
        **common,
    )
    out = reduce_masking_stats(acc)
    assert (
        out["masking/truncated_rows"] == 1 and out["masking/truncated_masked_rows"] == 0
    )
    assert out["masking/masked_rows"] == 0 and out["masking/reinstated_rows"] == 0
    assert "masking/reward_mean_masked" not in out


def test_groups_with_fewer_than_two_trained_rows_and_multi_chunk_ids():
    acc = new_masking_stats_accumulator()
    for pid, final in (
        (torch.tensor([[7, 0]] * 3), torch.tensor([1.0, 0.0, 0.0])),
        (torch.tensor([[9]] * 3), torch.ones(3)),
    ):
        accumulate_masking_stats(
            acc,
            prompt_ids=pid,
            rewards=torch.zeros(3),
            sample_mask=torch.ones(3),
            mask_sample=torch.tensor([False, True, True]),
            truncated=torch.zeros(3, dtype=torch.bool),
            overlong_filtering=False,
            final_sample_mask=final,
            baseline_mask=final,
        )
    out = reduce_masking_stats(acc)
    # Two chunks of different prompt widths -> two groups; the first has one trained row.
    assert out["masking/groups"] == 2 and out["masking/groups_lt2_trained_frac"] == 0.5
    assert reduce_masking_stats(new_masking_stats_accumulator()) == {}
