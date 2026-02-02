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

from unittest.mock import MagicMock, patch

import pytest
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.grpo import (
    _default_grpo_save_state,
    async_grpo_train,
    dynamic_sampling,
    grpo_train,
    normalize_advantages_with_epsilon,
    validate,
)
from nemo_rl.algorithms.loss_functions import ClippedPGLossFn
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
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

# ============================================================================
# Stub classes for async GRPO testing (non-Ray versions for easy mocking)
# ============================================================================


class StubReplayBuffer:
    """Non-Ray stub of ReplayBuffer for unit testing

    Each method returns a MagicMock with a 'remote' attribute that can be called.
    """

    def __init__(self, initial_size=10, mock_batch=None, mock_rollout_metrics=None):
        self._size = initial_size
        self._trajectories = []
        self._mock_batch = mock_batch
        self._mock_rollout_metrics = mock_rollout_metrics or {}

    @property
    def size(self):
        """Return a mock that returns buffer size when .remote() is called"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=self._size)  # ray.get will extract this
        return mock

    @property
    def sample(self):
        """Return a mock that returns sample result when .remote() is called"""

        def _sample(num_prompt_groups, current_weight_version, max_age_steps):
            # Return proper trajectory structure expected by async GRPO
            trajectories = [
                {
                    "batch": self._mock_batch,
                    "rollout_metrics": self._mock_rollout_metrics,
                }
                for _ in range(num_prompt_groups)
            ]
            return {
                "trajectories": trajectories,
                "avg_trajectory_age": 0.5,
            }

        mock = MagicMock()
        mock.remote = MagicMock(
            side_effect=lambda *args, **kwargs: _sample(*args, **kwargs)
        )
        return mock

    @property
    def get_debug_info(self):
        """Return a mock that returns debug info when .remote() is called"""
        mock = MagicMock()
        mock.remote = MagicMock(
            return_value={
                "total_trajectories": self._size,
                "trajectory_versions": [0],
                "target_weight_versions": [0],
                "max_size": 100,
            }
        )
        return mock


class StubAsyncTrajectoryCollector:
    """Non-Ray stub of AsyncTrajectoryCollector for unit testing

    Each method is a property that returns a MagicMock with a 'remote' attribute.
    """

    @property
    def start_collection(self):
        """Start collection - returns a remote-callable mock"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=MagicMock())  # Returns a fake ObjectRef
        return mock

    @property
    def set_weight_version(self):
        """Set weight version - returns a remote-callable mock"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=MagicMock())
        return mock

    @property
    def pause(self):
        """Pause collection - returns a remote-callable mock"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=MagicMock())
        return mock

    @property
    def resume(self):
        """Resume collection - returns a remote-callable mock"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=MagicMock())
        return mock

    @property
    def stop(self):
        """Stop collection - returns a remote-callable mock"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=MagicMock())
        return mock

    @property
    def wait_for_stop(self):
        """Wait for stop - returns a remote-callable mock"""
        mock = MagicMock()
        mock.remote = MagicMock(return_value=MagicMock())
        return mock


def mock_async_grpo_infrastructure(mock_batch, mock_rollout_metrics):
    """
    Context manager that mocks all async GRPO infrastructure (Ray actors, venv, etc).

    Returns a dict of patches that can be used as a context manager stack.
    """
    from contextlib import ExitStack

    stack = ExitStack()

    # Create stub instances with mock data
    stub_buffer = StubReplayBuffer(
        initial_size=10,
        mock_batch=mock_batch,
        mock_rollout_metrics=mock_rollout_metrics,
    )
    stub_collector = StubAsyncTrajectoryCollector()

    # Patch venv creation
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.create_local_venv_on_each_node",
            return_value="/fake/venv",
        )
    )
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.get_actor_python_env", return_value="/fake/python"
        )
    )

    # Patch Ray actor classes to return our stubs
    mock_buffer_cls = MagicMock()
    mock_buffer_cls.options.return_value.remote.return_value = stub_buffer
    stack.enter_context(
        patch("nemo_rl.algorithms.async_utils.ReplayBuffer", mock_buffer_cls)
    )

    mock_collector_cls = MagicMock()
    mock_collector_cls.options.return_value.remote.return_value = stub_collector
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector",
            mock_collector_cls,
        )
    )

    # Patch ray.get to return values from our stubs (not remote refs)
    def mock_ray_get(ref):
        # If it's already a plain value (from our stubs), return it
        if isinstance(ref, (int, str, dict, list)):
            return ref
        # If it's a MagicMock, return a default response
        return None

    stack.enter_context(patch("ray.get", side_effect=mock_ray_get))
    stack.enter_context(
        patch("ray.wait", side_effect=lambda refs, **kwargs: (refs, []))
    )
    stack.enter_context(
        patch("ray.kill", return_value=None)
    )  # Mock ray.kill for cleanup

    # Patch the rollout functions used inside async_grpo_train
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.run_multi_turn_rollout",
            return_value=(mock_batch, mock_rollout_metrics),
        )
    )
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.run_async_multi_turn_rollout",
            return_value=(mock_batch, mock_rollout_metrics),
        )
    )

    # Patch refit and validate functions
    stack.enter_context(
        patch("nemo_rl.algorithms.grpo.refit_policy_generation", return_value=None)
    )
    stack.enter_context(
        patch("nemo_rl.algorithms.grpo.validate", return_value=({}, {}))
    )

    # Mock print_performance_metrics to avoid needing real timing metrics
    stack.enter_context(
        patch("nemo_rl.algorithms.grpo.print_performance_metrics", return_value={})
    )

    return stack


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
            "dynamic_sampling_max_gen_batches": 5,
        }
    }

    timer = Timer()
    dynamic_sampling_num_gen_batches = 1

    # Test dynamic sampling
    result_batch, is_batch_complete, batch_cache, _ = dynamic_sampling(
        repeated_batch,
        std,
        baseline,
        dynamic_sampling_num_gen_batches,
        master_config,
        timer,
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
            "dynamic_sampling_max_gen_batches": 5,
        }
    }

    timer = Timer()
    dynamic_sampling_num_gen_batches = 1

    # Test dynamic sampling
    result_batch, is_batch_complete, batch_cache, _ = dynamic_sampling(
        repeated_batch,
        std,
        baseline,
        dynamic_sampling_num_gen_batches,
        master_config,
        timer,
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
            "dynamic_sampling_max_gen_batches": 5,
        }
    }

    timer = Timer()
    dynamic_sampling_num_gen_batches = 1

    # Test dynamic sampling - should indicate batch is not complete
    result_batch, is_batch_complete, batch_cache, _ = dynamic_sampling(
        repeated_batch,
        std,
        baseline,
        dynamic_sampling_num_gen_batches,
        master_config,
        timer,
    )

    # Should have cached the batch but marked as incomplete
    assert (
        result_batch.size == 3
    )  # All samples from the single prompt with non-zero std
    assert is_batch_complete == False  # Not enough prompts, need to continue sampling
    assert batch_cache is not None
    assert batch_cache == result_batch

    # Run dynamic sampling again with the cached batch
    result_batch, is_batch_complete, batch_cache, _ = dynamic_sampling(
        repeated_batch,
        std,
        baseline,
        dynamic_sampling_num_gen_batches,
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
            "dynamic_sampling_max_gen_batches": 5,
        }
    }

    timer = Timer()
    dynamic_sampling_num_gen_batches = 1

    # Test that dynamic sampling is bypassed
    result_batch, is_batch_complete, batch_cache, _ = dynamic_sampling(
        repeated_batch,
        std,
        baseline,
        dynamic_sampling_num_gen_batches,
        master_config,
        timer,
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
            "batch_multiplier": 1,
        },
        "data": {"shuffle": False, "num_workers": 1, "env_name": None},
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
            "batch_multiplier": 1,
        },
        "data": {"shuffle": False, "num_workers": 1, "env_name": None},
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


@pytest.mark.parametrize(
    "colocated_inference, expected_parallel",
    [(True, 0.0), (False, True)],
)
def test_setup_sglang_sets_model_path_and_parallel_flag(
    monkeypatch, colocated_inference, expected_parallel
):
    from nemo_rl.algorithms import grpo as grpo_mod

    logged = {}

    class DummyLogger:
        def log_hyperparams(self, *_args, **_kwargs):
            pass

        def log_metrics(self, metrics, *_args, **_kwargs):
            logged["metrics"] = metrics

    class DummyCheckpointer:
        def get_latest_checkpoint_path(self):
            return None

        def load_training_info(self, _path):
            return None

    class DummyLoader:
        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return 1

        def load_state_dict(self, _state):
            pass

    class DummyCluster:
        def __init__(self, *_args, **_kwargs):
            pass

        def world_size(self):
            return 1

        def get_master_address_and_port(self):
            return "127.0.0.1", 1234

    class DummyPolicy:
        def print_node_ip_and_gpu_id(self):
            pass

        def init_collective(self, *_args, **_kwargs):
            return []

        def prepare_refit_info(self):
            return {}

    class DummySGLangGeneration:
        def finish_generation(self):
            pass

        def prepare_refit_info(self, _state):
            pass

        def init_collective(self, *_args, **_kwargs):
            return []

    monkeypatch.setattr(grpo_mod, "Logger", lambda *_args, **_kwargs: DummyLogger())
    monkeypatch.setattr(
        grpo_mod, "CheckpointManager", lambda *_args, **_kwargs: DummyCheckpointer()
    )
    monkeypatch.setattr(
        grpo_mod, "ClippedPGLossFn", lambda *_args, **_kwargs: MagicMock()
    )
    monkeypatch.setattr(grpo_mod, "StatefulDataLoader", DummyLoader)
    monkeypatch.setattr(grpo_mod, "RayVirtualCluster", DummyCluster)
    monkeypatch.setattr(grpo_mod, "Policy", lambda *_args, **_kwargs: DummyPolicy())
    monkeypatch.setattr(
        grpo_mod,
        "SGLangGeneration",
        lambda *_args, **_kwargs: DummySGLangGeneration(),
    )
    monkeypatch.setattr(grpo_mod.ray, "get", lambda x: x)

    generation_resources = {
        "gpus_per_node": 1,
        "num_nodes": 1,
    }
    if colocated_inference:
        generation_resources = {"gpus_per_node": None, "num_nodes": None}

    master_config = {
        "policy": {
            "model_name": "fake-model",
            "train_global_batch_size": 1,
            "train_micro_batch_size": 1,
            "max_total_sequence_length": 8,
            "make_sequence_length_divisible_by": 1,
            "dtensor_cfg": {"enabled": False},
            "megatron_cfg": {"enabled": False, "pipeline_model_parallel_size": 1},
            "generation": {
                "backend": "sglang",
                "colocated": {
                    "enabled": colocated_inference,
                    "resources": generation_resources,
                },
                "sglang_cfg": {
                    "gpus_per_server": 1,
                    "dp_size": 1,
                    "pp_size": 1,
                    "ep_size": 1,
                },
            },
        },
        "loss_fn": {
            "force_on_policy_ratio": False,
            "use_importance_sampling_correction": False,
        },
        "env": {},
        "grpo": {
            "seed": 1,
            "num_prompts_per_step": 1,
            "num_generations_per_prompt": 1,
            "max_num_steps": 1,
            "max_num_epochs": 1,
            "val_period": 0,
            "val_batch_size": 1,
            "val_at_start": False,
            "max_val_samples": 1,
            "use_dynamic_sampling": False,
            "batch_multiplier": 1,
            "normalize_rewards": False,
            "use_leave_one_out_baseline": False,
            "reward_scaling": {"enabled": False},
            "reward_shaping": {"enabled": False},
            "overlong_filtering": False,
        },
        "data": {"shuffle": False, "num_workers": 0, "env_name": None},
        "logger": {"num_val_samples_to_print": 0},
        "checkpointing": {"enabled": False},
        "cluster": {"num_nodes": 1, "gpus_per_node": 4},
    }

    tokenizer = MagicMock()
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=1)

    grpo_mod.setup(master_config, tokenizer, dataset, None)

    assert (
        master_config["policy"]["generation"]["sglang_cfg"]["model_path"]
        == master_config["policy"]["model_name"]
    )
    assert logged["metrics"]["parallel_init_enabled"] == expected_parallel


def test_refit_policy_generation_sglang_colocated_http(monkeypatch):
    from nemo_rl.algorithms import grpo as grpo_mod

    calls = {
        "prepare_for_generation_tags": [],
        "invalidate_kv_cache": 0,
        "stream_weights_via_http": [],
        "offload_before_refit": 0,
        "offload_after_refit": 0,
    }

    class DummySGLangGeneration:
        def prepare_for_generation(self, tags=None):
            calls["prepare_for_generation_tags"].append(tags)

        def get_sglang_url_to_gpu_uuids(self):
            return {"http://localhost:12345": ["gpu-uuid-0"]}

        def invalidate_kv_cache(self):
            calls["invalidate_kv_cache"] += 1
            return True

    class DummyPolicy:
        def offload_before_refit(self):
            calls["offload_before_refit"] += 1

        def offload_after_refit(self):
            calls["offload_after_refit"] += 1

        def get_free_memory_bytes(self):
            return 1024 * 1024 * 1024

        def stream_weights_via_http(self, sglang_url_to_gpu_uuids):
            calls["stream_weights_via_http"].append(sglang_url_to_gpu_uuids)
            return ["ok"]

    monkeypatch.setattr(grpo_mod, "SGLangGeneration", DummySGLangGeneration)
    monkeypatch.setattr(grpo_mod.ray, "get", lambda x: x)

    grpo_mod.refit_policy_generation(
        policy=DummyPolicy(),
        policy_generation=DummySGLangGeneration(),
        colocated_inference=True,
    )

    assert calls["offload_before_refit"] == 1
    assert calls["offload_after_refit"] == 1
    assert calls["invalidate_kv_cache"] == 1
    assert calls["stream_weights_via_http"] == [
        {"http://localhost:12345": ["gpu-uuid-0"]}
    ]
    assert calls["prepare_for_generation_tags"] == [["weights"], ["kv_cache"]]


def test_refit_policy_generation_sglang_non_colocated_raises(monkeypatch):
    from nemo_rl.algorithms import grpo as grpo_mod

    class DummySGLangGeneration:
        pass

    monkeypatch.setattr(grpo_mod, "SGLangGeneration", DummySGLangGeneration)

    with pytest.raises(NotImplementedError):
        grpo_mod.refit_policy_generation(
            policy=object(),
            policy_generation=DummySGLangGeneration(),
            colocated_inference=False,
        )


def test_grpo_train_collects_generation_logger_metrics(
    monkeypatch, mock_grpo_components
):
    from nemo_rl.algorithms import grpo as grpo_mod

    policy_generation = MagicMock()
    policy_generation.clear_logger_metrics = MagicMock()
    policy_generation.get_logger_metrics = MagicMock(
        return_value={"pending_requests": 1}
    )
    policy_generation.prepare_for_generation = MagicMock()
    policy_generation.finish_generation = MagicMock()

    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))
    mock_rollout_metrics = {"gen_kl_error": 0.0, "mean_gen_tokens_per_sample": 2.0}

    def fake_batched_message_log_to_flat_message(*_args, **_kwargs):
        flat = BatchedDataDict(
            {
                "token_ids": torch.tensor([[1, 2]]),
                "advantages": torch.tensor([[0.5, 0.5]]),
                "generation_logprobs": torch.tensor([[0.0, 0.0]]),
                "token_loss_mask": torch.tensor([[1, 1]]),
                "content": ["ok"],
            }
        )
        return flat, torch.tensor([2])

    monkeypatch.setattr(
        grpo_mod,
        "batched_message_log_to_flat_message",
        fake_batched_message_log_to_flat_message,
    )
    monkeypatch.setattr(
        grpo_mod, "_should_use_async_rollouts", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        grpo_mod,
        "run_async_multi_turn_rollout",
        lambda *_args, **_kwargs: (mock_batch, mock_rollout_metrics),
    )
    monkeypatch.setattr(
        grpo_mod,
        "run_multi_turn_rollout",
        lambda *_args, **_kwargs: (mock_batch, mock_rollout_metrics),
    )
    monkeypatch.setattr(
        grpo_mod,
        "calculate_baseline_and_std_per_prompt",
        lambda *_args, **_kwargs: (torch.tensor([0.1]), torch.tensor([1.0])),
    )
    monkeypatch.setattr(
        grpo_mod, "refit_policy_generation", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        grpo_mod, "print_performance_metrics", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        grpo_mod, "maybe_gpu_profile_step", lambda *_args, **_kwargs: None
    )

    master_config = mock_grpo_components["master_config"]
    master_config["grpo"]["max_num_steps"] = 1
    master_config["grpo"]["max_num_epochs"] = 1
    master_config["grpo"]["val_period"] = 0
    master_config["grpo"]["val_at_start"] = False
    master_config["grpo"]["use_dynamic_sampling"] = False

    grpo_mod.grpo_train(
        mock_grpo_components["policy"],
        policy_generation,
        mock_grpo_components["train_dataloader"],
        mock_grpo_components["val_dataloader"],
        mock_grpo_components["tokenizer"],
        mock_grpo_components["loss_fn"],
        mock_grpo_components["task_to_env"],
        mock_grpo_components["val_task_to_env"],
        mock_grpo_components["logger"],
        mock_grpo_components["checkpointer"],
        _default_grpo_save_state(),
        master_config,
    )

    assert policy_generation.clear_logger_metrics.called
    assert policy_generation.get_logger_metrics.called
    assert any(
        "generation_logger_metrics" in call.args[0]
        for call in mock_grpo_components["logger"].log_metrics.call_args_list
    )


@pytest.fixture
def mock_grpo_components():
    # Create mock components
    policy = MagicMock()
    policy.train.return_value = {
        "loss": torch.tensor(0.5),
        "grad_norm": torch.tensor(1.0),
        "all_mb_metrics": {
            "loss": [0.5],
            "policy_gradient_loss": [0.3],
            "value_loss": [0.2],
            "global_valid_toks": [10],
            "token_mult_prob_error": [
                1.0
            ],  # Must be <= 1.05 to avoid logging extra plots
            "gen_kl_error": [0.0001],
        },
    }
    policy.generate.return_value = {
        "output_ids": torch.randint(0, 100, (2, 20)),
        "generation_lengths": torch.tensor([10, 15]),
        "unpadded_sequence_lengths": torch.tensor([12, 18]),
        "logprobs": torch.randn(2, 20),
    }
    policy.prepare_for_training.return_value = None
    # Mock sharding annotations for async GRPO
    policy.sharding_annotations.get_axis_size.return_value = 1  # data_parallel size

    # Create mock batch with proper structure
    mock_batch = BatchedDataDict[DatumSpec](
        {
            "message_log": [
                [
                    {
                        "role": "user",
                        "content": "test",
                        "token_ids": torch.tensor([1, 2, 3]),
                    },
                ]
            ],
            "task_name": ["math"],
            "extra_env_info": [{}],
            "loss_multiplier": torch.tensor([1.0]),
            "idx": torch.tensor([0]),
            "length": torch.tensor([3]),  # Add length field for GRPO
            "total_reward": torch.tensor(
                [1.0]
            ),  # Add total_reward for rollout processing
        }
    )

    # Create mock dataloader with 10 batches
    train_dataloader = MagicMock(spec=StatefulDataLoader)

    def train_iter(self):
        return iter([mock_batch] * 10)

    train_dataloader.__iter__ = train_iter
    train_dataloader.__len__ = MagicMock(return_value=10)

    val_dataloader = MagicMock(spec=StatefulDataLoader)

    def val_iter(self):
        return iter([mock_batch] * 10)

    val_dataloader.__iter__ = val_iter
    val_dataloader.__len__ = MagicMock(return_value=10)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    loss_fn = ClippedPGLossFn(
        {
            "reference_policy_kl_penalty": 0.01,
            "reference_policy_kl_type": "k3",
            "kl_input_clamp_value": 20.0,
            "kl_output_clamp_value": 10.0,
            "ratio_clip_min": 0.8,
            "ratio_clip_max": 1.2,
            "ratio_clip_c": 1.0,
            "use_on_policy_kl_approximation": False,
            "use_importance_sampling_correction": False,
            "truncated_importance_sampling_ratio": None,
            "sequence_level_importance_ratios": False,
            "token_level_loss": True,
            "force_on_policy_ratio": False,
        }
    )
    logger = MagicMock()
    checkpointer = MagicMock()

    # Create mock environment
    task_to_env = {"math": MagicMock()}
    val_task_to_env = {"math": MagicMock()}

    # Mock environment return values
    for env in [task_to_env["math"], val_task_to_env["math"]]:
        env.step.return_value = (
            [{"role": "environment", "content": "correct"}],  # observations
            [{}],  # metadata
            [[]],  # next_stop_strings
            [1.0],  # rewards
            [True],  # terminateds
            [None],  # answers
        )
        env.global_post_process_and_metrics.return_value = (mock_batch, {})

    # Create mock master config
    master_config = {
        "grpo": {
            "max_num_steps": 5,
            "max_num_epochs": 2,
            "num_prompts_per_step": 1,
            "num_generations_per_prompt": 1,
            "max_rollout_turns": 1,
            "val_period": 100,
            "val_batch_size": 1,
            "val_at_start": False,
            "max_val_samples": 10,
            "seed": 42,
            "advantage_normalization": "global",
            "use_leave_one_out_baseline": False,
            "normalize_rewards": False,
            "overlong_filtering": False,
            "reward_scaling": {"enabled": False},
            "reward_shaping": {"enabled": False},
            "use_dynamic_sampling": False,
            "async_grpo": {
                "enabled": False,
                "max_trajectory_age_steps": 1,
            },
        },
        "policy": {
            "train_global_batch_size": 1,
            "train_micro_batch_size": 1,
            "max_total_sequence_length": 2048,
            "make_sequence_length_divisible_by": 1,
            "generation": {
                "backend": "vllm",
                "colocated": {"enabled": True},
                "vllm_cfg": {"async_engine": True},  # Support async mode
            },
        },
        "loss_fn": {
            "use_importance_sampling_correction": True,  # Required for async mode
        },
        "checkpointing": {
            "enabled": False,
            "checkpoint_must_save_by": None,
            "save_period": 10,
        },
        "cluster": {
            "num_nodes": 1,
            "gpus_per_node": 2,
        },
        "logger": {
            "num_val_samples_to_print": 5,
        },
    }

    return {
        "policy": policy,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "tokenizer": tokenizer,
        "loss_fn": loss_fn,
        "logger": logger,
        "checkpointer": checkpointer,
        "task_to_env": task_to_env,
        "val_task_to_env": val_task_to_env,
        "master_config": master_config,
    }


@pytest.mark.parametrize("train_func", [grpo_train, async_grpo_train])
def test_grpo_exit_on_max_steps(mock_grpo_components, train_func):
    """Test that GRPO training loop exits when max_num_steps is reached"""
    # Set max steps to 12
    mock_grpo_components["master_config"]["grpo"]["max_num_steps"] = 12
    grpo_save_state = _default_grpo_save_state()

    # Async GRPO requires non-colocated inference
    if train_func == async_grpo_train:
        mock_grpo_components["master_config"]["policy"]["generation"]["colocated"][
            "enabled"
        ] = False

    # Prepare mock data
    mock_rollout_metrics = {
        "mean_gen_tokens_per_sample": 10.0,
        "max_gen_tokens": 20,
        "min_gen_tokens": 5,
    }
    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))

    # Use our helper to mock async infrastructure if needed
    if train_func == async_grpo_train:
        with mock_async_grpo_infrastructure(mock_batch, mock_rollout_metrics):
            train_func(
                mock_grpo_components["policy"],
                None,  # policy_generation
                mock_grpo_components["train_dataloader"],
                mock_grpo_components["val_dataloader"],
                mock_grpo_components["tokenizer"],
                mock_grpo_components["loss_fn"],
                mock_grpo_components["task_to_env"],
                mock_grpo_components["val_task_to_env"],
                mock_grpo_components["logger"],
                mock_grpo_components["checkpointer"],
                grpo_save_state,
                mock_grpo_components["master_config"],
            )
    else:
        # For sync grpo_train, just mock the rollout functions
        with patch(
            "nemo_rl.algorithms.grpo.run_multi_turn_rollout",
            return_value=(mock_batch, mock_rollout_metrics),
        ):
            with patch(
                "nemo_rl.algorithms.grpo.run_async_multi_turn_rollout",
                return_value=(mock_batch, mock_rollout_metrics),
            ):
                train_func(
                    mock_grpo_components["policy"],
                    None,  # policy_generation
                    mock_grpo_components["train_dataloader"],
                    mock_grpo_components["val_dataloader"],
                    mock_grpo_components["tokenizer"],
                    mock_grpo_components["loss_fn"],
                    mock_grpo_components["task_to_env"],
                    mock_grpo_components["val_task_to_env"],
                    mock_grpo_components["logger"],
                    mock_grpo_components["checkpointer"],
                    grpo_save_state,
                    mock_grpo_components["master_config"],
                )

    # Verify we trained for exactly 12 steps
    assert mock_grpo_components["policy"].train.call_count == 12


@pytest.mark.parametrize(
    "train_func", [grpo_train]
)  # Only test sync version for epochs (async uses steps)
def test_grpo_exit_on_max_epochs(mock_grpo_components, train_func):
    """Test that GRPO training loop exits when max_num_epochs is reached"""
    # Set max epochs to 2 and max steps to a large number
    mock_grpo_components["master_config"]["grpo"]["max_num_epochs"] = 2
    mock_grpo_components["master_config"]["grpo"]["max_num_steps"] = 100

    grpo_save_state = _default_grpo_save_state()

    # Mock rollout functions to return proper metrics
    mock_rollout_metrics = {
        "mean_gen_tokens_per_sample": 10.0,
        "max_gen_tokens": 20,
        "min_gen_tokens": 5,
    }

    # Get a mock batch to return
    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))

    with patch("nemo_rl.algorithms.grpo.run_multi_turn_rollout") as mock_rollout:
        mock_rollout.return_value = (mock_batch, mock_rollout_metrics)

        with patch(
            "nemo_rl.algorithms.grpo.run_async_multi_turn_rollout"
        ) as mock_async_rollout:
            mock_async_rollout.return_value = (mock_batch, mock_rollout_metrics)

            # Run training
            train_func(
                mock_grpo_components["policy"],
                None,  # policy_generation
                mock_grpo_components["train_dataloader"],
                mock_grpo_components["val_dataloader"],
                mock_grpo_components["tokenizer"],
                mock_grpo_components["loss_fn"],
                mock_grpo_components["task_to_env"],
                mock_grpo_components["val_task_to_env"],
                mock_grpo_components["logger"],
                mock_grpo_components["checkpointer"],
                grpo_save_state,
                mock_grpo_components["master_config"],
            )

    # Verify we trained for exactly two epochs (20 batches)
    assert mock_grpo_components["policy"].train.call_count == 20


@pytest.mark.parametrize("train_func", [grpo_train, async_grpo_train])
def test_grpo_exit_on_timeout(mock_grpo_components, train_func, capsys):
    """Test that GRPO training loop exits when timeout is reached"""
    # Set max steps and epochs to large numbers
    mock_grpo_components["master_config"]["grpo"]["max_num_steps"] = 100
    mock_grpo_components["master_config"]["grpo"]["max_num_epochs"] = 10
    grpo_save_state = _default_grpo_save_state()

    # Async GRPO requires non-colocated inference
    if train_func == async_grpo_train:
        mock_grpo_components["master_config"]["policy"]["generation"]["colocated"][
            "enabled"
        ] = False

    # Prepare mock data
    mock_rollout_metrics = {
        "mean_gen_tokens_per_sample": 10.0,
        "max_gen_tokens": 20,
        "min_gen_tokens": 5,
    }
    mock_batch = next(iter(mock_grpo_components["train_dataloader"]))

    # Mock TimeoutChecker to return False for first 7 checks, then True (timeout)
    with patch("nemo_rl.algorithms.grpo.TimeoutChecker") as mock_timeout_class:
        mock_timeout_instance = MagicMock()
        check_results = [False] * 7 + [True]
        mock_timeout_instance.check_save.side_effect = check_results
        mock_timeout_class.return_value = mock_timeout_instance

        # Use our helper for async, or simple mocking for sync
        if train_func == async_grpo_train:
            with mock_async_grpo_infrastructure(mock_batch, mock_rollout_metrics):
                train_func(
                    mock_grpo_components["policy"],
                    None,  # policy_generation
                    mock_grpo_components["train_dataloader"],
                    mock_grpo_components["val_dataloader"],
                    mock_grpo_components["tokenizer"],
                    mock_grpo_components["loss_fn"],
                    mock_grpo_components["task_to_env"],
                    mock_grpo_components["val_task_to_env"],
                    mock_grpo_components["logger"],
                    mock_grpo_components["checkpointer"],
                    grpo_save_state,
                    mock_grpo_components["master_config"],
                )
        else:
            with patch(
                "nemo_rl.algorithms.grpo.run_multi_turn_rollout",
                return_value=(mock_batch, mock_rollout_metrics),
            ):
                with patch(
                    "nemo_rl.algorithms.grpo.run_async_multi_turn_rollout",
                    return_value=(mock_batch, mock_rollout_metrics),
                ):
                    train_func(
                        mock_grpo_components["policy"],
                        None,  # policy_generation
                        mock_grpo_components["train_dataloader"],
                        mock_grpo_components["val_dataloader"],
                        mock_grpo_components["tokenizer"],
                        mock_grpo_components["loss_fn"],
                        mock_grpo_components["task_to_env"],
                        mock_grpo_components["val_task_to_env"],
                        mock_grpo_components["logger"],
                        mock_grpo_components["checkpointer"],
                        grpo_save_state,
                        mock_grpo_components["master_config"],
                    )

        # Verify training stopped at 8 steps (when check_save returned True)
        assert mock_grpo_components["policy"].train.call_count == 8

        # Verify the timeout message was printed and training actually stopped
        captured = capsys.readouterr()
        output_lines = captured.out.strip().split("\n")

        # Find the timeout message
        timeout_line_idx = None
        for i, line in enumerate(output_lines):
            if "Timeout has been reached, stopping training early" in line:
                timeout_line_idx = i
                break

        assert timeout_line_idx is not None, "Timeout message not found in output"

        # Check what comes after the timeout message
        remaining_lines = output_lines[timeout_line_idx + 1 :]

        # For async_grpo_train, we expect cleanup messages in the finally block
        if train_func.__name__ == "async_grpo_train":
            cleanup_found = any(
                "Stopping trajectory collection" in line
                or "Async GRPO training complete" in line
                for line in remaining_lines
            )
            assert cleanup_found, (
                "Expected cleanup messages after timeout in async mode"
            )

        # Verify no new epoch/step started after timeout
        for line in remaining_lines:
            assert "Epoch" not in line or "Epoch 1/10" in line, (
                f"Training continued to next epoch after timeout: {line}"
            )
            assert not (line.startswith("Step ") and "Step 9" in line), (
                f"Training continued to next step after timeout: {line}"
            )


# ============================================================================
# Tests for normalize_advantages_with_epsilon function
# ============================================================================


def test_normalize_advantages_with_epsilon_basic():
    """Test basic functionality of normalize_advantages_with_epsilon."""
    # Test case with normal values
    advantages = torch.tensor([[2.0], [4.0], [6.0]])
    std = torch.tensor([1.0, 2.0, 3.0])
    epsilon = 1e-6

    result = normalize_advantages_with_epsilon(advantages, std, epsilon)

    expected = torch.tensor([[2.0], [2.0], [2.0]])
    assert torch.allclose(result, expected, rtol=1e-5)


def test_normalize_advantages_with_epsilon_zero_std():
    """Test normalize_advantages_with_epsilon when std contains zeros."""
    advantages = torch.tensor([[1.0], [2.0], [3.0]])
    std = torch.tensor([0.0, 1.0, 0.0])  # Zero std for indices 0 and 2
    epsilon = 1e-6

    result = normalize_advantages_with_epsilon(advantages, std, epsilon)

    # When std=0 AND advantage!=0, normalization is skipped (advantages unchanged)
    # When std>0, normal normalization occurs
    expected = torch.tensor(
        [[1.0], [2.0], [3.0]]
    )  # Samples 0,2 unchanged; sample 1 normalized
    assert torch.allclose(result, expected, rtol=1e-5)


def test_normalize_advantages_with_epsilon_all_zero_std():
    """Test normalize_advantages_with_epsilon when all std values are zero."""
    advantages = torch.tensor([[1.5], [2.5], [3.5]])
    std = torch.tensor([0.0, 0.0, 0.0])
    epsilon = 1e-8

    # Save expected values BEFORE calling function (since it modifies in-place)
    expected = advantages.clone()

    result = normalize_advantages_with_epsilon(advantages, std, epsilon)

    # When std=0 AND advantage!=0, normalization is skipped (all unchanged)
    assert torch.allclose(result, expected, rtol=1e-5)


def test_normalize_advantages_with_epsilon_tensor_shapes():
    """Test normalize_advantages_with_epsilon with different tensor shapes."""
    # Test with batch size 1
    advantages = torch.tensor([[5.0]])
    std = torch.tensor([2.0])
    result = normalize_advantages_with_epsilon(advantages, std)
    expected = torch.tensor([[2.5]])
    assert torch.allclose(result, expected, rtol=1e-5)

    # Test with larger batch
    batch_size = 10
    advantages = torch.ones(batch_size, 1) * 3.0
    std = torch.ones(batch_size) * 1.5
    result = normalize_advantages_with_epsilon(advantages, std)
    expected = torch.ones(batch_size, 1) * 2.0
    assert torch.allclose(result, expected, rtol=1e-5)


def test_normalize_advantages_with_epsilon_negative_advantages():
    """Test normalize_advantages_with_epsilon with negative advantages."""
    advantages = torch.tensor([[-2.0], [3.0], [-1.5]])
    std = torch.tensor([1.0, 1.5, 0.5])

    result = normalize_advantages_with_epsilon(advantages, std)

    expected = torch.tensor([[-2.0], [2.0], [-3.0]])
    assert torch.allclose(result, expected, rtol=1e-5)


def test_normalize_advantages_with_zero_std_from_leave_one_out():
    """Test that zero std (from leave-one-out baseline) is handled gracefully by skipping normalization."""
    # Simulate the leave-one-out case: rewards [1.0, 0.0, 0.0, 0.0]
    # Sample 0 has baseline from [0, 0, 0] -> std=0, advantage=1.0
    # Samples 1-3 have baseline from [1, 0, 0] -> std≈0.577, advantage≈-0.333
    advantages = torch.tensor([[1.0], [-0.333], [-0.333], [-0.333]])
    std = torch.tensor([0.0, 0.577, 0.577, 0.577])
    epsilon = 1e-6

    # Compute expected values BEFORE calling function (since it modifies in-place)
    expected_sample_0 = advantages[0].clone()
    expected_normalized = advantages[1:].clone() / (std[1:].unsqueeze(-1) + epsilon)

    result = normalize_advantages_with_epsilon(advantages, std, epsilon)

    # Sample 0: std=0 -> advantage unchanged (skip normalization)
    assert torch.allclose(result[0], expected_sample_0, rtol=1e-5)

    # Samples 1-3: std>0 -> normalized with epsilon
    assert torch.allclose(result[1:], expected_normalized, rtol=1e-5)


def test_normalize_advantages_with_zero_std_and_zero_advantage():
    """Test that zero std with zero advantage is left unchanged."""
    advantages = torch.tensor([[0.0], [1.0], [0.0]])
    std = torch.tensor([0.0, 0.0, 1.0])
    epsilon = 1e-6

    # Compute expected values BEFORE calling function (since it modifies in-place)
    expected_sample_0 = advantages[0].clone()
    expected_sample_1 = advantages[1].clone()
    expected_sample_2 = advantages[2].clone() / (std[2] + epsilon)

    result = normalize_advantages_with_epsilon(advantages, std, epsilon)

    # Sample 0: std=0, advantage=0 -> unchanged (skip normalization)
    assert torch.allclose(result[0], expected_sample_0, rtol=1e-5)

    # Sample 1: std=0, advantage!=0 -> unchanged (skip normalization)
    assert torch.allclose(result[1], expected_sample_1, rtol=1e-5)

    # Sample 2: std>0 -> normalize with epsilon
    assert torch.allclose(result[2], expected_sample_2, rtol=1e-5)


def test_normalize_advantages_with_small_nonzero_std():
    """Test that small but non-zero std values still get normalized (no threshold)."""
    advantages = torch.tensor([[2.0], [3.0], [-1.0]])
    std = torch.tensor([0.001, 0.01, 0.0001])  # All small but non-zero

    # Compute expected values BEFORE calling function (since it modifies in-place)
    expected = advantages.clone() / (std.unsqueeze(-1) + 1e-6)

    result = normalize_advantages_with_epsilon(advantages, std)

    # All should be normalized since std > 0
    assert torch.allclose(result, expected, rtol=1e-5)


# ============================================================================
# Tests for validate function
# ============================================================================


class TestValidateFunction:
    """Tests for the validate() function."""

    def test_validate_logs_data_when_logger_provided(self, tmp_path):
        """Test that validation data is logged to JSONL when logger is provided."""

        # Create mock components
        mock_policy_gen = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        # Create mock batch
        mock_batch = BatchedDataDict[DatumSpec](
            {
                "message_log": [
                    [
                        {
                            "role": "user",
                            "content": "test1",
                            "token_ids": torch.tensor([1, 2, 3]),
                        },
                        {
                            "role": "assistant",
                            "content": "response1",
                            "token_ids": torch.tensor([4, 5, 6]),
                        },
                    ],
                    [
                        {
                            "role": "user",
                            "content": "test2",
                            "token_ids": torch.tensor([7, 8, 9]),
                        },
                        {
                            "role": "assistant",
                            "content": "response2",
                            "token_ids": torch.tensor([10, 11, 12]),
                        },
                    ],
                ],
                "task_name": ["math", "math"],
                "extra_env_info": [{}, {}],
                "loss_multiplier": torch.tensor([1.0, 1.0]),
                "idx": torch.tensor([0, 1]),
                "length": torch.tensor([6, 6]),
                "total_reward": torch.tensor([1.0, 0.5]),
            }
        )

        # Create mock dataloader that yields mock_batch
        mock_dataloader = MagicMock(spec=StatefulDataLoader)
        mock_dataloader.__iter__ = MagicMock(return_value=iter([mock_batch]))

        # Create mock environment
        mock_env = MagicMock(spec=EnvironmentInterface)
        mock_env.global_post_process_and_metrics.return_value = (mock_batch, {})

        # Create mock logger that captures calls
        mock_logger = MagicMock()
        logged_data = {}

        def capture_log(data, filename):
            logged_data["data"] = data
            logged_data["filename"] = filename

        mock_logger.log_batched_dict_as_jsonl = MagicMock(side_effect=capture_log)

        # Mock config
        mock_config = {
            "grpo": {
                "max_val_samples": 10,
                "val_batch_size": 2,
                "max_rollout_turns": 1,
            },
            "policy": {
                "max_total_sequence_length": 2048,
                "generation": {
                    "backend": "vllm",
                    "colocated": {"enabled": True},
                    "vllm_cfg": {"async_engine": False},
                },
            },
            "logger": {
                "num_val_samples_to_print": 2,
            },
        }

        mock_rollout_metrics = {"mean_gen_tokens_per_sample": 10.0}

        with patch("nemo_rl.algorithms.grpo.run_multi_turn_rollout") as mock_rollout:
            mock_rollout.return_value = (mock_batch, mock_rollout_metrics)
            with patch(
                "nemo_rl.algorithms.grpo._should_use_nemo_gym", return_value=False
            ):
                with patch(
                    "nemo_rl.algorithms.grpo._should_use_async_rollouts",
                    return_value=False,
                ):
                    with patch("nemo_rl.algorithms.grpo.print_message_log_samples"):
                        val_metrics, timing = validate(
                            mock_policy_gen,
                            mock_dataloader,
                            mock_tokenizer,
                            {"math": mock_env},
                            step=5,
                            master_config=mock_config,
                            logger=mock_logger,
                        )

        # Verify log_batched_dict_as_jsonl was called
        mock_logger.log_batched_dict_as_jsonl.assert_called_once()

        # Verify the filename
        assert logged_data["filename"] == "val_data_step5.jsonl"

        # Verify the data structure
        assert "content" in logged_data["data"]
        assert "rewards" in logged_data["data"]

    def test_validate_works_without_logger(self):
        """Test that validation works when logger is None (backward compat)."""
        # Create mock components
        mock_policy_gen = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        # Create mock batch
        mock_batch = BatchedDataDict[DatumSpec](
            {
                "message_log": [
                    [
                        {
                            "role": "user",
                            "content": "test1",
                            "token_ids": torch.tensor([1, 2, 3]),
                        },
                        {
                            "role": "assistant",
                            "content": "response1",
                            "token_ids": torch.tensor([4, 5, 6]),
                        },
                    ],
                ],
                "task_name": ["math"],
                "extra_env_info": [{}],
                "loss_multiplier": torch.tensor([1.0]),
                "idx": torch.tensor([0]),
                "length": torch.tensor([6]),
                "total_reward": torch.tensor([1.0]),
            }
        )

        # Create mock dataloader
        mock_dataloader = MagicMock(spec=StatefulDataLoader)
        mock_dataloader.__iter__ = MagicMock(return_value=iter([mock_batch]))

        # Create mock environment
        mock_env = MagicMock(spec=EnvironmentInterface)
        mock_env.global_post_process_and_metrics.return_value = (mock_batch, {})

        # Mock config
        mock_config = {
            "grpo": {
                "max_val_samples": 10,
                "val_batch_size": 1,
                "max_rollout_turns": 1,
            },
            "policy": {
                "max_total_sequence_length": 2048,
                "generation": {
                    "backend": "vllm",
                    "colocated": {"enabled": True},
                    "vllm_cfg": {"async_engine": False},
                },
            },
            "logger": {
                "num_val_samples_to_print": 1,
            },
        }

        mock_rollout_metrics = {"mean_gen_tokens_per_sample": 10.0}

        with patch("nemo_rl.algorithms.grpo.run_multi_turn_rollout") as mock_rollout:
            mock_rollout.return_value = (mock_batch, mock_rollout_metrics)
            with patch(
                "nemo_rl.algorithms.grpo._should_use_nemo_gym", return_value=False
            ):
                with patch(
                    "nemo_rl.algorithms.grpo._should_use_async_rollouts",
                    return_value=False,
                ):
                    with patch("nemo_rl.algorithms.grpo.print_message_log_samples"):
                        # Call validate without logger (should not raise exception)
                        val_metrics, timing = validate(
                            mock_policy_gen,
                            mock_dataloader,
                            mock_tokenizer,
                            {"math": mock_env},
                            step=5,
                            master_config=mock_config,
                            logger=None,
                        )

        # Verify metrics are returned correctly
        assert "accuracy" in val_metrics
        assert "avg_length" in val_metrics

    def test_validate_returns_empty_when_no_dataloader(self):
        """Test that validate returns empty dicts when no dataloader is provided."""
        mock_policy_gen = MagicMock()
        mock_tokenizer = MagicMock()

        mock_config = {
            "dpo": {"val_period": 0},  # Required for the assertion
        }

        val_metrics, timing = validate(
            mock_policy_gen,
            None,  # No dataloader
            mock_tokenizer,
            None,
            step=0,
            master_config=mock_config,
            logger=None,
        )

        assert val_metrics == {}
        assert timing == {}
