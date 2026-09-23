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

import pytest
import torch

from nemo_rl.algorithms.single_controller_utils.sample_masks import baseline_valid_mask

# Six rows: [clean, env-flagged, truncated (overlong on), placeholder, seq-logprob-masked, clean w/ weight 0.5]
SAMPLE_MASK = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.5])
INCOMPLETE = torch.tensor([False, True, True, False, False, False])
# What trains: env-flagged, truncated, placeholder and seq-logprob rows are out.
FINAL = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.5])


def test_switch_off_returns_the_training_mask_object():
    out = baseline_valid_mask(
        sample_mask=SAMPLE_MASK,
        final_sample_mask=FINAL,
        incomplete=INCOMPLETE,
        keep_incomplete_rewards=False,
    )
    assert out is FINAL


def test_switch_on_reinstates_only_incomplete_rows_with_their_weight():
    out = baseline_valid_mask(
        sample_mask=SAMPLE_MASK,
        final_sample_mask=FINAL,
        incomplete=INCOMPLETE,
        keep_incomplete_rewards=True,
    )
    # env-flagged and truncated rows come back for the baseline; placeholder
    # (sample_mask 0) and seq-logprob-masked rows stay out; clean rows unchanged.
    assert out.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.5]
    # The training mask is untouched.
    assert FINAL.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.5]


def test_incomplete_placeholder_stays_out():
    out = baseline_valid_mask(
        sample_mask=torch.tensor([0.0, 1.0]),
        final_sample_mask=torch.tensor([0.0, 0.0]),
        incomplete=torch.tensor([True, True]),
        keep_incomplete_rewards=True,
    )
    assert out.tolist() == [0.0, 1.0]


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        baseline_valid_mask(
            sample_mask=torch.ones(3),
            final_sample_mask=torch.ones(3),
            incomplete=torch.zeros(2, dtype=torch.bool),
            keep_incomplete_rewards=True,
        )
