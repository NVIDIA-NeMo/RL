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

from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.experience.rollouts import calculate_rewards


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


def create_mock_batch(
    num_samples: int,
    task_names: list[str],
    message_logs: list[LLMMessageLogType],
    extra_env_info: list[dict] = None,
) -> BatchedDataDict[DatumSpec]:
    """Helper function to create a mock batch for testing."""
    if extra_env_info is None:
        extra_env_info = [{} for _ in range(num_samples)]

    return BatchedDataDict[DatumSpec](
        {
            "task_name": task_names,
            "message_log": message_logs,
            "extra_env_info": extra_env_info,
            "loss_multiplier": torch.ones(num_samples),
        }
    )


@pytest.fixture(scope="module")
def ray_init():
    """Initialize Ray for testing."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(scope="module")
def mock_env(ray_init):
    """Create a mock environment for single task tests."""
    env = MockEnvironment.remote(rewards=[1.0, 2.0])
    yield env
    ray.kill(env)


@pytest.fixture(scope="module")
def mock_envs(ray_init):
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
    env_observations, metadata, next_stop_strings, rewards, terminateds = (
        calculate_rewards(batch, task_to_env)
    )

    # Verify results
    assert torch.allclose(rewards, torch.tensor([1.0, 2.0]))
    assert len(env_observations) == 2
    assert len(terminateds) == 2
    assert len(next_stop_strings) == 2
    assert len(metadata) == 2
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
    env_observations, metadata, next_stop_strings, rewards, terminateds = (
        calculate_rewards(batch, mock_envs)
    )

    # Verify results
    assert torch.allclose(rewards, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert len(env_observations) == 4
    assert len(terminateds) == 4
    assert len(next_stop_strings) == 4
    assert len(metadata) == 4
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
    env_observations, metadata, next_stop_strings, rewards, terminateds = (
        calculate_rewards(batch, task_to_env)
    )

    # Verify results
    assert len(rewards) == 0
    assert len(env_observations) == 0
    assert len(terminateds) == 0
    assert len(next_stop_strings) == 0
    assert len(metadata) == 0
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
