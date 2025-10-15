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
    grpo_train,
)
from nemo_rl.algorithms.loss_functions import ClippedPGLossFn
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
        },
        "data": {"shuffle": False, "num_workers": 1},
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
        },
        "data": {"shuffle": False, "num_workers": 1},
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
        },
    }
    policy.generate.return_value = {
        "output_ids": torch.randint(0, 100, (2, 20)),
        "generation_lengths": torch.tensor([10, 15]),
        "unpadded_sequence_lengths": torch.tensor([12, 18]),
        "logprobs": torch.randn(2, 20),
    }
    policy.prepare_for_training.return_value = None

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
            "ratio_clip_min": 0.8,
            "ratio_clip_max": 1.2,
            "ratio_clip_c": 1.0,
            "use_on_policy_kl_approximation": False,
            "use_importance_sampling_correction": False,
            "token_level_loss": True,
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

    # Mock TimeoutChecker to return False for first 7 checks, then True (timeout)
    with patch("nemo_rl.algorithms.grpo.TimeoutChecker") as mock_timeout_class:
        mock_timeout_instance = MagicMock()
        # Create a side_effect that returns False 7 times, then True
        check_results = [False] * 7 + [True]
        mock_timeout_instance.check_save.side_effect = check_results
        mock_timeout_class.return_value = mock_timeout_instance

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
            # Verify we see cleanup messages (🛑 Stopping trajectory collection...)
            cleanup_found = any(
                "Stopping trajectory collection" in line
                or "Async GRPO training complete" in line
                for line in remaining_lines
            )
            assert cleanup_found, (
                "Expected cleanup messages after timeout in async mode"
            )

        # Verify no new epoch/step started after timeout (sync GRPO has epochs)
        for line in remaining_lines:
            assert "Epoch" not in line or "Epoch 1/10" in line, (
                f"Training continued to next epoch after timeout: {line}"
            )
            # Also check we don't start a new training step
            assert not (line.startswith("Step ") and "Step 9" in line), (
                f"Training continued to next step after timeout: {line}"
            )
