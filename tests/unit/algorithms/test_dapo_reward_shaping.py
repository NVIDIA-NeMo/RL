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
import torch

from nemo_rl.algorithms.reward_functions import (
    RewardShapingConfig,
    apply_reward_shaping,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.uint8])
def test_dapo_reward_shaping_matches_scalar_assignment(dtype: torch.dtype) -> None:
    rewards = torch.tensor([1, 1, 0, 1] * 16, dtype=dtype)
    lengths = torch.tensor([4, 5, 6, 106] * 16)
    batch = BatchedDataDict(
        {"total_reward": rewards, "response_token_lengths": lengths}
    )
    config = RewardShapingConfig(
        enabled=True,
        overlong_buffer_length=5,
        overlong_buffer_penalty=0.8,
        max_response_length=10,
    )
    # Lengths 4 and 5 are within the budget. Lengths 6 and 106 incur penalties.
    expected = torch.empty_like(rewards)
    for i, penalty in enumerate([0.0, 0.0, -0.16, -16.16]):
        expected[i] = rewards[i] + penalty

    result = apply_reward_shaping(batch, config)
    torch.testing.assert_close(
        result["total_reward"], expected[:4].repeat(16), rtol=0, atol=0
    )
    torch.testing.assert_close(result["unshaped_total_reward"], rewards, rtol=0, atol=0)
    result["total_reward"].zero_()
    torch.testing.assert_close(result["unshaped_total_reward"], rewards, rtol=0, atol=0)
