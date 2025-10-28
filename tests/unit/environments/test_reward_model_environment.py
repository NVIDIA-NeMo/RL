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

import os

import pytest
import ray
import torch

from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.environments.reward_model_environment import (
    RewardModelEnvironment,
    RewardModelEnvironmentConfig,
)

# Model configuration constants for testing
REWARD_MODEL_NAME = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
MAX_MODEL_LEN = 1024

# Basic reward model environment configuration for testing
# This config sets up a minimal reward model environment for unit testing
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
        "_v2": True,
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


@pytest.fixture(scope="class")
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
        assert ray.is_initialized()
        reward_model_py_executable_class = (
            "nemo_rl.models.policy.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"
            if basic_env_config["dtensor_cfg"]["_v2"]
            else "nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker"
        )
        env_actor = RewardModelEnvironment.options(  # type: ignore # it's wrapped with ray.remote
            runtime_env={
                "py_executable": get_actor_python_env(reward_model_py_executable_class),
                "env_vars": dict(
                    os.environ
                ),  # Pass thru all user environment variables
            }
        ).remote(basic_env_config)
        yield env_actor
    except Exception as e:
        print(f"Error creating reward model environment: {e}")
        raise
    finally:
        if env_actor:
            try:
                env_actor.shutdown.remote()
            except Exception as e:
                print(f"Warning: Error during actor shutdown: {e}")


class TestRewardModelEnvironment:
    """
    Test suite for RewardModelEnvironment functionality.

    This test class contains all unit tests for the RewardModelEnvironment,
    covering initialization, data processing, reward computation, and resource
    management. Each test method focuses on a specific aspect of the environment's
    functionality.
    """

    def test_reward_model_environment_initialization(self, reward_model_env):
        """
        Test that the reward model environment initializes correctly.

        This test verifies that the environment is properly configured
        and ready for use. It checks that all required components are
        initialized and accessible.

        Args:
            reward_model_env: The reward model environment fixture.
        """
        # Verify the environment is properly initialized
        assert reward_model_env is not None
        assert hasattr(reward_model_env, "shutdown")

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_reward_model_environment_preprocess_data(
        self, reward_model_env, batch_size
    ):
        """
        Test the reward model environment's ability to preprocess data with different batch sizes.

        This test verifies that the environment can preprocess conversation
        data correctly, including tokenization, formatting, and batching.
        It ensures that the output format is compatible with the reward model
        and works correctly with different batch sizes, including edge cases like batch_size=1.

        Args:
            reward_model_env: The reward model environment fixture.
            batch_size: The batch size to test (1, 2, 4, 8).
        """
        # Create message log batch with the specified batch size
        message_log_batch = [
            [
                {
                    "role": "user",
                    "content": f"What is the capital of France? (test {i})",
                },
                {
                    "role": "assistant",
                    "content": f"The capital of Brazil is Brasilia. (response {i})",
                },
            ]
            for i in range(batch_size)
        ]

        # Use remote call for Ray Actor
        future = reward_model_env.preprocess_data.remote(message_log_batch)
        output = ray.get(future)

        target_length = 39
        assert output is not None
        assert output["input_ids"] is not None
        assert output["input_lengths"] is not None

        # Verify the output shapes match the batch size
        assert output["input_ids"].shape == (batch_size, target_length)
        assert output["input_lengths"].shape == (batch_size,)
        assert all(length == target_length for length in output["input_lengths"])

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
        future = reward_model_env.step.remote(message_log_batch, [])
        output = ray.get(future)

        # Verify the reward model name
        assert REWARD_MODEL_NAME == "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
        # Verify output structure and properties
        assert output.rewards is not None
        assert output.rewards.shape == (2,)
        assert output.rewards.dtype == torch.float32
        # Verify expected reward values (with tolerance for floating point precision)
        expected_rewards = torch.tensor([-5.3750, 2.6250])
        assert torch.allclose(output.rewards, expected_rewards, atol=1e-1)
