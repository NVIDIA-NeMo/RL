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

from nemo_rl.environments.reward_model_environment import (
    RewardModelEnvironment,
    RewardModelEnvironmentConfig,
)

# Model configuration constants
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
REWARD_MODEL_NAME = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
MAX_MODEL_LEN = 1024

# Basic reward model environment configuration
# This config sets up a reward model environment for testing reward computation
basic_env_config: RewardModelEnvironmentConfig = {
    "enabled": True,
    "model_name": REWARD_MODEL_NAME,
    "tokenizer": {"name": REWARD_MODEL_NAME},
    "precision": "bfloat16",
    "batch_size": 32,
    "checkpoint_path": None,
    "max_model_len": MAX_MODEL_LEN,
    "resources": {"gpus_per_node": 1, "num_nodes": 1},
    "reward_model_cfg": {
        "enabled": True,
        "reward_model_type": "bradley_terry",
    },
    "dtensor_cfg": {
        "enabled": True,
        "cpu_offload": False,
        "sequence_parallel": False,
        "activation_checkpointing": False,
        "tensor_parallel_size": 1,
        "context_parallel_size": 1,
        "custom_parallel_plan": None,
    },
    "dynamic_batching": {"enabled": False},
    "sequence_packing": {"enabled": False},
    "max_grad_norm": None,
}


@pytest.fixture(scope="function")
def reward_model_env():
    """
    Create a reward model environment for testing.

    This fixture creates a RewardModelEnvironment instance with the basic
    configuration and ensures proper cleanup after each test.

    Yields:
        RewardModelEnvironment: A configured reward model environment instance.
    """
    env_actor = None
    try:
        env_actor = RewardModelEnvironment(basic_env_config)
        yield env_actor
    finally:
        if env_actor:
            env_actor.shutdown()


class TestRewardModelEnvironment:
    """Test suite for RewardModelEnvironment functionality."""

    def test_reward_model_environment_initialization(self, reward_model_env):
        """
        Test that the reward model environment initializes correctly.

        This test verifies that the environment is properly configured
        and ready for use.

        Args:
            reward_model_env: The reward model environment fixture.
        """
        # Verify the environment is properly initialized
        assert reward_model_env is not None
        assert reward_model_env.config is not None
        assert reward_model_env.virtual_cluster is not None
        assert reward_model_env.tokenizer is not None
        assert reward_model_env.reward_model_policy is not None

    def test_reward_model_environment_preprocess_data(self, reward_model_env):
        """
        Test the reward model environment's ability to preprocess data.

        This test verifies that the environment can preprocess data correctly.
        """
        message_log_batch = [
            [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of Brazil is Brasilia."},
            ],
        ]
        output = reward_model_env.preprocess_data(message_log_batch)
        target_length = 29
        assert output is not None
        assert output["input_ids"] is not None
        assert output["input_lengths"] is not None

        assert output["input_ids"].shape == (1, target_length)
        assert output["input_lengths"].shape == (1,)
        assert output["input_lengths"][0] == target_length

    def test_reward_model_environment_generate_rewards(self, reward_model_env):
        """
        Test the reward model environment's ability to generate responses and compute rewards.

        This test verifies that:
        1. The environment can process message logs
        2. Rewards are computed correctly
        3. The reward values are reasonable (incorrect answer gets lower reward)
        4. The output format is correct

        Args:
            reward_model_env: The reward model environment fixture.
        """
        # Test data: Two conversation pairs with correct and incorrect answers
        message_log_batch = [
            [
                {"role": "user", "content": "What is the capital of France?"},
                {
                    "role": "assistant",
                    "content": "The capital of Brazil is Brasilia.",
                },  # Incorrect answer
            ],
            [
                {"role": "user", "content": "What is the capital of France?"},
                {
                    "role": "assistant",
                    "content": "The capital of France is Paris.",
                },  # Correct answer
            ],
        ]

        # Execute the environment step
        output = reward_model_env.step(message_log_batch, [])

        # Verify the reward model name
        assert REWARD_MODEL_NAME == "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
        # Verify output structure and properties
        assert output.rewards is not None
        assert output.rewards.shape == (2,)
        assert output.rewards.dtype == torch.float32
        # Verify expected reward values (with tolerance for floating point precision)
        expected_rewards = torch.tensor([-5.3750, 2.6250])
        assert torch.allclose(output.rewards, expected_rewards, atol=1e-4)
