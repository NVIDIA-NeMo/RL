# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from tests.unit.algorithms.utils import create_mock_batch_with_responses


def test_reward_shaping_disabled():
    """Test that when reward shaping is disabled, rewards remain unchanged."""
    # Create batch with various response lengths
    batch = create_mock_batch_with_responses(
        num_samples=3, response_lengths=[10, 20, 30], initial_rewards=[1.0, 0.5, 0.8]
    )

    original_rewards = batch["total_reward"].clone()

    # Disabled reward shaping config
    config = RewardShapingConfig(
        enabled=False,
        overlong_buffer_length=5,
        overlong_buffer_penalty=0.1,
        max_response_length=25,
    )

    # Apply reward shaping
    result_batch = apply_reward_shaping(batch, config)

    # Rewards should remain unchanged
    assert torch.allclose(result_batch["total_reward"], original_rewards)
    assert result_batch is batch  # Should return the same batch object


def test_reward_shaping_no_penalties():
    """Test reward shaping when all responses are within acceptable length."""
    # Create batch where all responses are shorter than expected length
    batch = create_mock_batch_with_responses(
        num_samples=3,
        response_lengths=[10, 15, 18],  # All <= 20 (expected_response_length)
        initial_rewards=[1.0, 0.5, 0.8],
    )

    original_rewards = batch["total_reward"].clone()

    # Config: max_response_length=25, overlong_buffer_length=5 -> expected_response_length=20
    config = RewardShapingConfig(
        enabled=True,
        overlong_buffer_length=5,
        overlong_buffer_penalty=1.0,
        max_response_length=25,
    )

    # Apply reward shaping
    result_batch = apply_reward_shaping(batch, config)

    # Since no responses exceed expected length, rewards should remain unchanged
    assert torch.allclose(result_batch["total_reward"], original_rewards)


def test_reward_shaping_with_penalties():
    """Test reward shaping when responses exceed expected length and receive penalties."""
    # Create batch with responses of varying lengths
    batch = create_mock_batch_with_responses(
        num_samples=4,
        response_lengths=[10, 22, 25, 30],  # expected_response_length = 20
        initial_rewards=[1.0, 0.8, 0.6, 0.4],
    )

    # Config: max_response_length=25, overlong_buffer_length=5 -> expected_response_length=20
    config = RewardShapingConfig(
        enabled=True,
        overlong_buffer_length=5,
        overlong_buffer_penalty=0.5,
        max_response_length=25,
    )

    # Apply reward shaping
    result_batch = apply_reward_shaping(batch, config)

    # Calculate expected rewards manually
    # Response 0: length=10, exceed_length=10-20=-10 (no penalty, reward stays 1.0)
    # Response 1: length=22, exceed_length=22-20=2, penalty=min(-2/5*0.5, 0)=-0.2, reward=0.8-0.2=0.6
    # Response 2: length=25, exceed_length=25-20=5, penalty=min(-5/5*0.5, 0)=-0.5, reward=0.6-0.5=0.1
    # Response 3: length=30, exceed_length=30-20=10, penalty=min(-10/5*0.5, 0)=-1.0, reward=0.4-1.0=-0.6

    expected_rewards = torch.tensor([1.0, 0.6, 0.1, -0.6])
    assert torch.allclose(result_batch["total_reward"], expected_rewards, atol=1e-6)


def test_reward_shaping_missing_config_values():
    """Test that missing required config values raise ValueError."""
    batch = create_mock_batch_with_responses(
        num_samples=1, response_lengths=[20], initial_rewards=[1.0]
    )

    # Test missing overlong_buffer_length
    config = RewardShapingConfig(
        enabled=True,
        overlong_buffer_length=None,
        overlong_buffer_penalty=0.1,
        max_response_length=25,
    )

    with pytest.raises(ValueError, match="DAPO reward shaping is currently supported"):
        apply_reward_shaping(batch, config)

    # Test missing overlong_buffer_penalty
    config["overlong_buffer_length"] = 5
    config["overlong_buffer_penalty"] = None

    with pytest.raises(ValueError, match="DAPO reward shaping is currently supported"):
        apply_reward_shaping(batch, config)

    # Test missing max_response_length
    config["overlong_buffer_penalty"] = 0.1
    config["max_response_length"] = None

    with pytest.raises(ValueError, match="DAPO reward shaping is currently supported"):
        apply_reward_shaping(batch, config)


def test_reward_shaping_missing_assistant_response():
    """Test that missing assistant response raises assertion error."""
    # Create a batch with only user messages (no assistant responses)
    message_logs = [
        [{"role": "user", "content": "Question", "token_ids": torch.tensor([1, 2, 3])}]
    ]

    batch = BatchedDataDict[DatumSpec](
        {
            "task_name": ["math"],
            "message_log": message_logs,
            "extra_env_info": [{}],
            "loss_multiplier": torch.ones(1),
            "total_reward": torch.tensor([1.0]),
        }
    )

    config = RewardShapingConfig(
        enabled=True,
        overlong_buffer_length=5,
        overlong_buffer_penalty=0.1,
        max_response_length=25,
    )

    with pytest.raises(
        AssertionError, match="Assistant response not found during reward shaping"
    ):
        apply_reward_shaping(batch, config)


def test_reward_shaping_mismatched_lengths():
    """Test that mismatched message_log and rewards lengths raise assertion error."""
    # Create batch with mismatched lengths
    batch = create_mock_batch_with_responses(
        num_samples=2, response_lengths=[10, 20], initial_rewards=[1.0, 0.5]
    )

    # Manually add an extra reward to create mismatch
    batch["total_reward"] = torch.tensor(
        [1.0, 0.5, 0.3]
    )  # 3 rewards but 2 message_logs

    config = RewardShapingConfig(
        enabled=True,
        overlong_buffer_length=5,
        overlong_buffer_penalty=0.1,
        max_response_length=25,
    )

    with pytest.raises(
        AssertionError,
        match="The number of messages in the batch must match the number of rewards",
    ):
        apply_reward_shaping(batch, config)
