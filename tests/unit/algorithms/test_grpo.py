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
import ray
import torch

from nemo_rl.algorithms.grpo import dynamic_sampling
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.experience.rollouts import calculate_rewards
from nemo_rl.utils.timer import Timer
from tests.unit.algorithms.utils import (
    create_mock_batch,
)


@ray.remote(num_cpus=0)
class MockEnvironment(EnvironmentInterface):
    def __init__(self, rewards: list[float]):
        self.rewards = rewards
        self._calls = 0

    def step(
        self, messages: list[LLMMessageLogType], env_info: list[dict]
    ) -> EnvironmentReturn:
        self._calls += 1
        return (
            [{"role": "environment", "content": "observation"}] * len(messages),
            [{}] * len(messages),
            [[]] * len(messages),
            self.rewards,
            [True] * len(messages),
            [None] * len(messages),
        )

    def get_calls(self):
        return self._calls

    def reset_calls(self):
        self._calls = 0
        return True

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        return batch, {}


@pytest.fixture(scope="module")
def mock_env():
    """Create a mock environment for single task tests."""
    env = MockEnvironment.remote(rewards=[1.0, 2.0])
    yield env
    ray.kill(env)


@pytest.fixture(scope="module")
def mock_envs():
    """Create mock environments for multiple task tests."""
    math_env = MockEnvironment.remote(rewards=[1.0, 2.0])
    code_env = MockEnvironment.remote(rewards=[3.0, 4.0])
    yield {"math": math_env, "code": code_env}
    ray.kill(math_env)
    ray.kill(code_env)


@pytest.fixture(autouse=True)
def reset_env_calls(mock_env, mock_envs):
    """Reset call counters before each test."""
    ray.get(mock_env.reset_calls.remote())
    ray.get(mock_envs["math"].reset_calls.remote())
    ray.get(mock_envs["code"].reset_calls.remote())
    yield


def test_calculate_rewards_single_task(mock_env):
    """Test reward calculation with a single task type."""
    task_to_env = {"math": mock_env}

    # Create test data
    task_names = ["math", "math"]
    message_logs = [
        [{"role": "user", "content": "1+1"}, {"role": "assistant", "content": "2"}],
        [{"role": "user", "content": "2+2"}, {"role": "assistant", "content": "4"}],
    ]
    batch = create_mock_batch(2, task_names, message_logs)

    # Calculate rewards
    env_observations, metadata, next_stop_strings, rewards, terminateds, answers = (
        calculate_rewards(batch, task_to_env)
    )

    # Verify results
    assert torch.allclose(rewards, torch.tensor([1.0, 2.0]))
    assert len(env_observations) == 2
    assert len(terminateds) == 2
    assert len(next_stop_strings) == 2
    assert len(metadata) == 2
    assert len(answers) == 2
    assert torch.allclose(rewards, torch.tensor([1.0, 2.0]))
    assert (
        ray.get(mock_env.get_calls.remote()) == 1
    )  # Should only call once for all samples of same task


def test_calculate_rewards_multiple_tasks(mock_envs):
    """Test reward calculation with multiple task types."""
    # Create test data
    task_names = ["math", "math", "code", "code"]
    message_logs = [
        [{"role": "user", "content": "1+1"}, {"role": "assistant", "content": "2"}],
        [{"role": "user", "content": "2+2"}, {"role": "assistant", "content": "4"}],
        [
            {"role": "user", "content": "print('hello')"},
            {"role": "assistant", "content": "hello"},
        ],
        [
            {"role": "user", "content": "print('world')"},
            {"role": "assistant", "content": "world"},
        ],
    ]
    batch = create_mock_batch(4, task_names, message_logs)

    # Calculate rewards
    env_observations, metadata, next_stop_strings, rewards, terminateds, answers = (
        calculate_rewards(batch, mock_envs)
    )

    # Verify results
    assert torch.allclose(rewards, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert len(env_observations) == 4
    assert len(terminateds) == 4
    assert len(next_stop_strings) == 4
    assert len(metadata) == 4
    assert len(answers) == 4
    assert torch.allclose(rewards, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert (
        ray.get(mock_envs["math"].get_calls.remote()) == 1
    )  # One call for all math samples
    assert (
        ray.get(mock_envs["code"].get_calls.remote()) == 1
    )  # One call for all code samples


def test_calculate_rewards_empty_batch(mock_env):
    """Test reward calculation with an empty batch."""
    task_to_env = {"math": mock_env}

    # Create empty test data
    batch = create_mock_batch(0, [], [])

    # Calculate rewards
    env_observations, metadata, next_stop_strings, rewards, terminateds, answers = (
        calculate_rewards(batch, task_to_env)
    )

    # Verify results
    assert len(rewards) == 0
    assert len(env_observations) == 0
    assert len(terminateds) == 0
    assert len(next_stop_strings) == 0
    assert len(metadata) == 0
    assert len(answers) == 0
    assert (
        ray.get(mock_env.get_calls.remote()) == 0
    )  # Should not call environment for empty batch


def test_calculate_rewards_missing_environment():
    """Test reward calculation with a missing environment."""
    # Create test data with unknown task
    task_names = ["unknown_task"]
    message_logs = [[{"role": "user", "content": "test"}]]
    batch = create_mock_batch(1, task_names, message_logs)

    # Try to calculate rewards with missing environment
    task_to_env = {}  # Empty dict means no environments available
    with pytest.raises(
        ValueError, match="No environment found for task type: unknown_task"
    ):
        calculate_rewards(batch, task_to_env)


def test_dapo_dynamic_sampling_filters_nonzero_std():
    """Test that DAPO dynamic sampling only selects prompts with non-zero standard deviation."""
    # Create mock batch data with 6 prompts (2 prompts * 3 generations each)
    batch_size = 6
    message_logs = [
        [
            {"role": "user", "content": f"prompt_{i // 3}"},
            {"role": "assistant", "content": f"response_{i}"},
        ]
        for i in range(batch_size)
    ]
    task_names = ["math"] * batch_size

    # Create batch with some prompts having zero std and others non-zero std
    repeated_batch = create_mock_batch(batch_size, task_names, message_logs)
    repeated_batch["total_reward"] = torch.tensor([1.0, 0.0, 1.0, 0.5, 0.5, 0.0])

    # Mock prompts tensor (2 unique prompts, each repeated 3 times)
    prompts = torch.tensor(
        [
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
        ]
    )

    # First prompt group has std=0.5 (rewards: 1.0, 0.0, 1.0 -> std ≠ 0)
    # Second prompt group has std=0.25 (rewards: 0.5, 0.5, 0.0 -> std ≠ 0)
    std = torch.tensor(
        [0.5, 0.5, 0.5, 0.25, 0.25, 0.25]
    )  # Both prompts have non-zero std
    baseline = torch.tensor([0.67, 0.67, 0.67, 0.33, 0.33, 0.33])  # Mock baselines

    # Configuration for dynamic sampling
    master_config = {
        "grpo": {
            "use_dynamic_sampling": True,
            "num_prompts_per_step": 2,  # Want 2 prompts
            "num_generations_per_prompt": 3,  # Each with 3 generations
            "max_num_gen_batches": 5,
        }
    }

    timer = Timer()
    num_gen_batches = 1

    # Test dynamic sampling
    result_batch, is_batch_complete, batch_cache = dynamic_sampling(
        repeated_batch, prompts, std, baseline, num_gen_batches, master_config, timer
    )

    # Since both prompts have non-zero std, all 6 samples should be selected
    assert result_batch.size == 6
    assert is_batch_complete == True
    assert torch.allclose(result_batch["std"], std)
    assert torch.allclose(result_batch["baseline"], baseline)


def test_dapo_dynamic_sampling_filters_zero_std():
    """Test that DAPO dynamic sampling filters out prompts with zero standard deviation."""
    # Create mock batch data
    batch_size = 6
    message_logs = [
        [
            {"role": "user", "content": f"prompt_{i // 3}"},
            {"role": "assistant", "content": f"response_{i}"},
        ]
        for i in range(batch_size)
    ]
    task_names = ["math"] * batch_size

    repeated_batch = create_mock_batch(batch_size, task_names, message_logs)
    repeated_batch["total_reward"] = torch.tensor(
        [1.0, 1.0, 1.0, 0.5, 0.5, 0.0]
    )  # First prompt has same rewards (std=0)

    # Mock prompts tensor
    prompts = torch.tensor(
        [
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
            [4, 5, 6],  # prompt 1
        ]
    )

    # First prompt has zero std (all rewards are 1.0)
    # Second prompt has non-zero std (rewards: 0.5, 0.5, 0.0)
    std = torch.tensor(
        [0.0, 0.0, 0.0, 0.25, 0.25, 0.25]
    )  # First prompt has zero std, second has non-zero
    baseline = torch.tensor([1.0, 1.0, 1.0, 0.33, 0.33, 0.33])

    master_config = {
        "grpo": {
            "use_dynamic_sampling": True,
            "num_prompts_per_step": 1,  # Want 1 prompt only
            "num_generations_per_prompt": 3,
        }
    }

    timer = Timer()
    num_gen_batches = 1

    # Test dynamic sampling
    result_batch, is_batch_complete, batch_cache = dynamic_sampling(
        repeated_batch, prompts, std, baseline, num_gen_batches, master_config, timer
    )

    # Only the second prompt (indices 3,4,5) should be selected since first has zero std
    assert result_batch.size == 3  # Only 3 samples from the second prompt
    assert is_batch_complete == True
    assert torch.allclose(
        result_batch["std"], torch.tensor([0.25, 0.25, 0.25])
    )  # Only non-zero std
    assert torch.allclose(result_batch["baseline"], torch.tensor([0.33, 0.33, 0.33]))

    ## verify that only prompt_1 is selected
    prompts = [
        result_batch["message_log"][i][0]["content"] for i in range(result_batch.size)
    ]
    assert prompts == ["prompt_1", "prompt_1", "prompt_1"]

    # Verify that filtered rewards are correct
    expected_filtered_rewards = torch.tensor(
        [
            0.5,
            0.5,
            0.0,
        ]
    )
    assert torch.allclose(result_batch["filtered_reward"], expected_filtered_rewards)


def test_dapo_dynamic_sampling_batch_caching():
    """Test that DAPO dynamic sampling uses batch caching when insufficient non-zero std prompts are found."""
    # Create mock batch with only 1 prompt having non-zero std, but we need 2
    batch_size = 3
    message_logs = [
        [
            {"role": "user", "content": "prompt_0"},
            {"role": "assistant", "content": f"response_{i}"},
        ]
        for i in range(batch_size)
    ]
    task_names = ["math"] * batch_size

    repeated_batch = create_mock_batch(batch_size, task_names, message_logs)
    repeated_batch["total_reward"] = torch.tensor([1.0, 0.0, 0.5])  # Non-zero std

    prompts = torch.tensor(
        [
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
            [1, 2, 3],  # prompt 0
        ]
    )

    std = torch.tensor([0.4, 0.4, 0.4])  # Only one prompt with non-zero std
    baseline = torch.tensor([0.5, 0.5, 0.5])

    master_config = {
        "grpo": {
            "use_dynamic_sampling": True,
            "num_prompts_per_step": 2,  # Need 2 prompts but only have 1
            "num_generations_per_prompt": 3,
            "max_num_gen_batches": 5,
        }
    }

    timer = Timer()
    num_gen_batches = 1

    # Test dynamic sampling - should indicate batch is not complete
    result_batch, is_batch_complete, batch_cache = dynamic_sampling(
        repeated_batch, prompts, std, baseline, num_gen_batches, master_config, timer
    )

    # Should have cached the batch but marked as incomplete
    assert (
        result_batch.size == 3
    )  # All samples from the single prompt with non-zero std
    assert is_batch_complete == False  # Not enough prompts, need to continue sampling
    assert batch_cache is not None
    assert batch_cache == result_batch

    # Run dynamic sampling again with the cached batch
    result_batch, is_batch_complete, batch_cache = dynamic_sampling(
        repeated_batch,
        prompts,
        std,
        baseline,
        num_gen_batches,
        master_config,
        timer,
        batch_cache,
    )

    # After running dynamic sampling again, the batch should be complete
    assert (
        result_batch.size == 6
    )  # All samples from the single prompt with non-zero std
    assert is_batch_complete == True
    assert batch_cache is not None


def test_dapo_dynamic_sampling_disabled():
    """Test that when dynamic sampling is disabled, all prompts are kept regardless of std."""
    batch_size = 6
    message_logs = [
        [
            {"role": "user", "content": f"prompt_{i // 3}"},
            {"role": "assistant", "content": f"response_{i}"},
        ]
        for i in range(batch_size)
    ]
    task_names = ["math"] * batch_size

    repeated_batch = create_mock_batch(batch_size, task_names, message_logs)
    repeated_batch["total_reward"] = torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 0.0])

    prompts = torch.tensor(
        [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
            [4, 5, 6],
            [4, 5, 6],
        ]
    )

    # Mix of zero and non-zero std
    std = torch.tensor([0.0, 0.0, 0.0, 0.25, 0.25, 0.25])
    baseline = torch.tensor([1.0, 1.0, 1.0, 0.33, 0.33, 0.33])

    # Disable dynamic sampling
    master_config = {
        "grpo": {
            "use_dynamic_sampling": False,
            "num_prompts_per_step": 2,
            "num_generations_per_prompt": 3,
        }
    }

    timer = Timer()
    num_gen_batches = 1

    # Test that dynamic sampling is bypassed
    result_batch, is_batch_complete, batch_cache = dynamic_sampling(
        repeated_batch, prompts, std, baseline, num_gen_batches, master_config, timer
    )

    # All samples should be kept when dynamic sampling is disabled
    assert result_batch.size == 6
    assert is_batch_complete == True
    assert batch_cache is None  # No caching when disabled


def test_noncolocated_inference_requires_explicit_gpus_per_node_single_node():
    """Test that non-colocated inference requires explicit gpus_per_node when policy_nodes=1."""
    from unittest.mock import MagicMock, patch

    from nemo_rl.algorithms.grpo import setup

    # Create minimal config - only what's needed before the validation we're testing
    master_config = {
        "policy": {
            "generation": {
                "backend": "vllm",
                "colocated": {
                    "enabled": False,  # Non-colocated
                    "resources": {
                        "gpus_per_node": None,  # This should trigger error
                        "num_nodes": None,
                    },
                },
            },
        },
        "loss_fn": {},  # Config extraction requires this key
        "env": {},  # Config extraction requires this key
        "grpo": {
            "seed": 42,
            "num_prompts_per_step": 1,
            "val_period": 0,
            "val_at_start": False,
            "use_dynamic_sampling": False,
        },
        "data": {"shuffle": False},
        "logger": {},  # Config extraction requires this key
        "checkpointing": {},  # Config extraction requires this key
        "cluster": {
            "num_nodes": 1,  # Single node, so policy_nodes=1
            "gpus_per_node": 8,
        },
    }

    tokenizer = MagicMock()
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)

    # Mock everything we don't need to test
    with (
        patch("nemo_rl.algorithms.grpo.Logger") as mock_logger,
        patch("nemo_rl.algorithms.grpo.CheckpointManager") as mock_checkpointer,
        patch("nemo_rl.algorithms.grpo.StatefulDataLoader"),
        pytest.raises(
            AssertionError,
            match="policy.generation.colocated.resources.gpus_per_node must be explicitly set",
        ),
    ):
        # Configure mocks to skip checkpoint loading
        mock_checkpointer.return_value.get_latest_checkpoint_path.return_value = None
        setup(master_config, tokenizer, dataset, None)


def test_noncolocated_inference_requires_explicit_gpus_per_node_multi_node():
    """Test that non-colocated inference requires explicit gpus_per_node when policy_nodes>1."""
    from unittest.mock import MagicMock, patch

    from nemo_rl.algorithms.grpo import setup

    # Create minimal config - only what's needed before the validation we're testing
    master_config = {
        "policy": {
            "generation": {
                "backend": "vllm",
                "colocated": {
                    "enabled": False,  # Non-colocated
                    "resources": {
                        "gpus_per_node": None,  # This should trigger error
                        "num_nodes": 1,  # Use 1 node for inference
                    },
                },
            },
        },
        "loss_fn": {},  # Config extraction requires this key
        "env": {},  # Config extraction requires this key
        "grpo": {
            "seed": 42,
            "num_prompts_per_step": 1,
            "val_period": 0,
            "val_at_start": False,
            "use_dynamic_sampling": False,
        },
        "data": {"shuffle": False},
        "logger": {},  # Config extraction requires this key
        "checkpointing": {},  # Config extraction requires this key
        "cluster": {
            "num_nodes": 2,  # Multi-node, so policy_nodes=1 after subtracting inference
            "gpus_per_node": 8,
        },
    }

    tokenizer = MagicMock()
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)

    # Mock everything we don't need to test
    with (
        patch("nemo_rl.algorithms.grpo.Logger") as mock_logger,
        patch("nemo_rl.algorithms.grpo.CheckpointManager") as mock_checkpointer,
        patch("nemo_rl.algorithms.grpo.StatefulDataLoader"),
        pytest.raises(
            AssertionError,
            match="policy.generation.colocated.resources.gpus_per_node must be explicitly set",
        ),
    ):
        # Configure mocks to skip checkpoint loading
        mock_checkpointer.return_value.get_latest_checkpoint_path.return_value = None
        setup(master_config, tokenizer, dataset, None)
