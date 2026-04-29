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
import tempfile
import threading
import unittest.mock as mock

import pytest
import ray
import torch

# Set up Ray temp directory before any Ray operations
# Try multiple approaches to ensure Ray uses a writable directory
_temp_dir = tempfile.mkdtemp(prefix="ray_async_test_")
os.environ["RAY_TEMP_DIR"] = _temp_dir
os.environ["RAY_TMPDIR"] = _temp_dir  # Alternative env var
os.environ["TMPDIR"] = _temp_dir  # System temp dir

from nemo_rl.algorithms.async_utils import (
    AsyncTrajectoryCollector,
    ReplayBuffer,
)
from nemo_rl.algorithms.async_utils.replay_buffer import ReplayBufferNew
from nemo_rl.algorithms.grpo import (
    MasterConfig,
    add_grpo_token_loss_masks_and_generation_logprobs,
    extract_initial_prompt_messages,
)
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)


@ray.remote(num_cpus=0)
class MockEnvironment(EnvironmentInterface):
    """Mock environment for testing async utilities."""

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


class MockGenerationInterface:
    """Mock generation interface for testing."""

    def __init__(self):
        self.prepare_calls = 0
        self.finish_calls = 0

    def prepare_for_generation(self, **kwargs):
        self.prepare_calls += 1

    def finish_generation(self):
        self.finish_calls += 1


class TestReplayBuffer:
    """Test cases for ReplayBuffer."""

    def test_replay_buffer_initialization(self):
        """Test ReplayBuffer initialization."""
        buffer = ReplayBuffer.remote(max_size=10)
        size = ray.get(buffer.size.remote())
        assert size == 0

        debug_info = ray.get(buffer.get_debug_info.remote())
        assert debug_info["total_trajectories"] == 0
        assert debug_info["max_size"] == 10
        assert debug_info["trajectory_versions"] == []
        assert debug_info["target_weight_versions"] == []

        ray.kill(buffer)

    def test_replay_buffer_push_and_size(self):
        """Test pushing trajectories to buffer."""
        buffer = ReplayBuffer.remote(max_size=3)

        # Create mock trajectories
        trajectory1 = {"batch": {"data": "test1"}, "rollout_metrics": {"reward": 1.0}}
        trajectory2 = {"batch": {"data": "test2"}, "rollout_metrics": {"reward": 2.0}}

        # Push trajectories
        status1 = ray.get(
            buffer.add.remote(trajectory1, weight_version=0, target_weight_version=1)
        )
        assert status1 == "success"

        status2 = ray.get(
            buffer.add.remote(trajectory2, weight_version=1, target_weight_version=2)
        )
        assert status2 == "success"

        # Check size
        size = ray.get(buffer.size.remote())
        assert size == 2

        # Check debug info
        debug_info = ray.get(buffer.get_debug_info.remote())
        assert debug_info["total_trajectories"] == 2
        assert debug_info["trajectory_versions"] == [0, 1]
        assert debug_info["target_weight_versions"] == [1, 2]

        ray.kill(buffer)

    def test_replay_buffer_max_size_limit(self):
        """Test that buffer respects max size limit."""
        buffer = ReplayBuffer.remote(max_size=2)

        # Fill buffer to capacity
        trajectory1 = {"batch": {"data": "test1"}, "rollout_metrics": {"reward": 1.0}}
        trajectory2 = {"batch": {"data": "test2"}, "rollout_metrics": {"reward": 2.0}}
        trajectory3 = {"batch": {"data": "test3"}, "rollout_metrics": {"reward": 3.0}}

        # Push first two trajectories
        status1 = ray.get(
            buffer.add.remote(trajectory1, weight_version=0, target_weight_version=1)
        )
        status2 = ray.get(
            buffer.add.remote(trajectory2, weight_version=1, target_weight_version=2)
        )
        assert status1 == "success"
        assert status2 == "success"

        # Try to push third trajectory (should return "full")
        status3 = ray.get(
            buffer.add.remote(trajectory3, weight_version=2, target_weight_version=3)
        )
        assert status3 == "full"

        # Size should still be 2
        size = ray.get(buffer.size.remote())
        assert size == 2

        ray.kill(buffer)

    def test_replay_buffer_sampling_basic(self):
        """Test basic trajectory sampling."""
        buffer = ReplayBuffer.remote(max_size=10)

        # Push trajectories with different weight versions
        trajectories = []
        for i in range(3):
            trajectory = {
                "batch": {"data": f"test{i}"},
                "rollout_metrics": {"reward": float(i)},
            }
            trajectories.append(trajectory)
            ray.get(
                buffer.add.remote(
                    trajectory, weight_version=i, target_weight_version=i + 1
                )
            )

        # Sample trajectories intended for current step 2
        sample_result = ray.get(
            buffer.sample.remote(
                num_prompt_groups=1,
                current_weight_version=2,
                max_age_steps=2,
            )
        )

        assert sample_result is not None
        assert len(sample_result["trajectories"]) == 1
        assert "avg_trajectory_age" in sample_result

        # The trajectory should be intended for step 2 (target_weight_version=2)
        # But we pushed with target_weight_version=i+1, so trajectory at i=1 has target=2
        sampled_trajectory = sample_result["trajectories"][0]
        assert sampled_trajectory["batch"]["data"] == "test1"

        ray.kill(buffer)

    def test_replay_buffer_sampling_insufficient_trajectories(self):
        """Test sampling when insufficient trajectories are available."""
        buffer = ReplayBuffer.remote(max_size=10)

        # Push only one trajectory
        trajectory = {"batch": {"data": "test"}, "rollout_metrics": {"reward": 1.0}}
        ray.get(
            buffer.add.remote(trajectory, weight_version=0, target_weight_version=1)
        )

        # Try to sample more trajectories than available for current step
        sample_result = ray.get(
            buffer.sample.remote(
                num_prompt_groups=2,  # Request 2 but only 1 available
                current_weight_version=1,
                max_age_steps=1,
            )
        )

        assert sample_result is None  # Should return None when insufficient

        ray.kill(buffer)

    def test_replay_buffer_age_filtering(self):
        """Test that old trajectories are filtered out."""
        buffer = ReplayBuffer.remote(max_size=10)

        # Push trajectories with different ages
        old_trajectory = {"batch": {"data": "old"}, "rollout_metrics": {"reward": 1.0}}
        recent_trajectory = {
            "batch": {"data": "recent"},
            "rollout_metrics": {"reward": 2.0},
        }

        ray.get(
            buffer.add.remote(old_trajectory, weight_version=0, target_weight_version=1)
        )
        ray.get(
            buffer.add.remote(
                recent_trajectory, weight_version=2, target_weight_version=3
            )
        )

        # Sample with current_weight_version=3 and max_age_steps=1
        # This should filter out the trajectory with weight_version=0 (too old)
        with pytest.raises(
            ValueError, match="Found .* trajectories older than min_valid_version"
        ):
            ray.get(
                buffer.sample.remote(
                    num_prompt_groups=1,
                    current_weight_version=3,
                    max_age_steps=1,
                )
            )

        ray.kill(buffer)

    def test_replay_buffer_target_weight_matching(self):
        """Test that sampling only returns trajectories intended for current step."""
        buffer = ReplayBuffer.remote(max_size=10)

        # Push trajectories intended for different target steps
        trajectory1 = {
            "batch": {"data": "for_step_1"},
            "rollout_metrics": {"reward": 1.0},
        }
        trajectory2 = {
            "batch": {"data": "for_step_2"},
            "rollout_metrics": {"reward": 2.0},
        }

        ray.get(
            buffer.add.remote(trajectory1, weight_version=0, target_weight_version=1)
        )
        ray.get(
            buffer.add.remote(trajectory2, weight_version=1, target_weight_version=2)
        )

        # Sample for current step 1 - should only get trajectory intended for step 1
        sample_result = ray.get(
            buffer.sample.remote(
                num_prompt_groups=1,
                current_weight_version=1,
                max_age_steps=2,
            )
        )

        assert sample_result is not None
        assert len(sample_result["trajectories"]) == 1
        assert sample_result["trajectories"][0]["batch"]["data"] == "for_step_1"

        ray.kill(buffer)

    def test_replay_buffer_get_existing_target_weights(self):
        """Test getting existing target weight versions."""
        buffer = ReplayBuffer.remote(max_size=10)

        # Initially empty
        existing_weights = ray.get(buffer.get_existing_target_weights.remote())
        assert existing_weights == set()

        # Push trajectories with different target weights
        trajectory1 = {"batch": {"data": "test1"}, "rollout_metrics": {"reward": 1.0}}
        trajectory2 = {"batch": {"data": "test2"}, "rollout_metrics": {"reward": 2.0}}

        ray.get(
            buffer.add.remote(trajectory1, weight_version=0, target_weight_version=1)
        )
        ray.get(
            buffer.add.remote(trajectory2, weight_version=1, target_weight_version=3)
        )

        existing_weights = ray.get(buffer.get_existing_target_weights.remote())
        assert existing_weights == {1, 3}

        ray.kill(buffer)

    def test_replay_buffer_get_target_weight_counts(self):
        """Test counting buffered trajectory groups by target weight."""
        buffer = ReplayBuffer.remote(max_size=10)

        trajectories = [
            ({"batch": {"data": "test1"}, "rollout_metrics": {"reward": 1.0}}, 0, 2),
            ({"batch": {"data": "test2"}, "rollout_metrics": {"reward": 2.0}}, 1, 2),
            ({"batch": {"data": "test3"}, "rollout_metrics": {"reward": 3.0}}, 1, 3),
        ]
        for trajectory, weight_version, target_weight_version in trajectories:
            ray.get(
                buffer.add.remote(
                    trajectory,
                    weight_version=weight_version,
                    target_weight_version=target_weight_version,
                )
            )

        target_counts = ray.get(buffer.get_target_weight_counts.remote())
        assert target_counts == {2: 2, 3: 1}

        ray.kill(buffer)

    def test_replay_buffer_clear(self):
        """Test clearing the buffer."""
        buffer = ReplayBuffer.remote(max_size=10)

        # Push some trajectories
        trajectory = {"batch": {"data": "test"}, "rollout_metrics": {"reward": 1.0}}
        ray.get(
            buffer.add.remote(trajectory, weight_version=0, target_weight_version=1)
        )

        # Verify buffer has content
        size = ray.get(buffer.size.remote())
        assert size == 1

        # Clear buffer
        ray.get(buffer.clear.remote())

        # Verify buffer is empty
        size = ray.get(buffer.size.remote())
        assert size == 0

        debug_info = ray.get(buffer.get_debug_info.remote())
        assert debug_info["total_trajectories"] == 0
        assert debug_info["trajectory_versions"] == []
        assert debug_info["target_weight_versions"] == []

        ray.kill(buffer)


class TestReplayBufferNew:
    """Tests for ReplayBufferNew: staleness-window sampling via _evict + sample."""

    def _make_traj(self, label: str) -> dict:
        return {"batch": {"data": label}, "rollout_metrics": {}}

    def _add(self, buf, label: str, weight_version: int):
        return ray.get(
            buf.add.remote(
                self._make_traj(label),
                weight_version=weight_version,
                target_weight_version=0,  # unused in ReplayBufferNew
            )
        )

    def _sample(self, buf, num_groups: int, trainer_version: int):
        return ray.get(
            buf.sample.remote(
                num_prompt_groups=num_groups,
                current_weight_version=trainer_version,
                max_age_steps=0,  # unused in ReplayBufferNew
            )
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def test_invalid_max_staleness_raises(self):
        with pytest.raises(Exception):
            buf = ReplayBufferNew.remote(max_size=10, max_staleness=-1)
            ray.get(buf.size.remote())

    # ------------------------------------------------------------------
    # _evict (via sample)
    # ------------------------------------------------------------------

    def test_stale_rows_evicted_before_sampling(self):
        """Rows with age > max_staleness are removed before sample() selects."""
        buf = ReplayBufferNew.remote(max_size=10, max_staleness=2)
        # age at trainer=4: gen_v=1 → 3 > 2 (stale), gen_v=3 → 1 ≤ 2 (valid)
        self._add(buf, "stale", weight_version=1)
        self._add(buf, "fresh", weight_version=3)

        result = self._sample(buf, num_groups=1, trainer_version=4)

        assert result is not None
        assert result["trajectories"][0]["batch"]["data"] == "fresh"
        assert ray.get(buf.size.remote()) == 0  # stale row also gone
        ray.kill(buf)

    def test_all_stale_returns_none(self):
        """sample() returns None when all rows are evicted as stale."""
        buf = ReplayBufferNew.remote(max_size=10, max_staleness=1)
        self._add(buf, "a", weight_version=0)
        self._add(buf, "b", weight_version=1)

        # trainer=5: both ages > 1
        result = self._sample(buf, num_groups=1, trainer_version=5)

        assert result is None
        assert ray.get(buf.size.remote()) == 0
        ray.kill(buf)

    def test_eviction_frees_capacity(self):
        """Evicting a stale row allows a subsequent add() to succeed."""
        buf = ReplayBufferNew.remote(max_size=1, max_staleness=1)
        self._add(buf, "x", weight_version=1)
        assert self._add(buf, "x", weight_version=1) == "full"

        # sample() at trainer=5 evicts the stale row (age 4 > 1)
        self._sample(buf, num_groups=1, trainer_version=5)

        assert self._add(buf, "y", weight_version=4) == "success"
        ray.kill(buf)

    def test_within_window_not_evicted(self):
        """Rows whose age is within max_staleness are not evicted."""
        buf = ReplayBufferNew.remote(max_size=10, max_staleness=3)
        self._add(buf, "x", weight_version=4)

        # trainer=6: age = 6 - 4 = 2 ≤ 3 → should survive
        # should return None since there is only 1 row
        result = self._sample(buf, num_groups=2, trainer_version=6)
        assert result is None

        # this sample should still be there
        assert ray.get(buf.size.remote()) == 1
        ray.kill(buf)

    # ------------------------------------------------------------------
    # sample()
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("sample_freshest_first", [True, False])
    def test_sample_freshest_first(self, sample_freshest_first):
        """sample() returns the freshest trajectories first."""
        buf = ReplayBufferNew.remote(
            max_size=10, max_staleness=5, sample_freshest_first=sample_freshest_first
        )
        for gen_v in [3, 4, 5]:
            self._add(buf, f"v{gen_v}", weight_version=gen_v)

        result = self._sample(buf, num_groups=2, trainer_version=6)

        assert result is not None
        data = [t["batch"]["data"] for t in result["trajectories"]]
        if sample_freshest_first:
            assert data == ["v5", "v4"]
        else:
            assert data == ["v3", "v4"]
        ray.kill(buf)

    def test_sample_returns_none_when_insufficient(self):
        """sample() returns None when fewer rows than requested remain after eviction."""
        buf = ReplayBufferNew.remote(max_size=10, max_staleness=5)
        self._add(buf, "only", weight_version=1)

        result = self._sample(buf, num_groups=3, trainer_version=2)

        assert result is None
        ray.kill(buf)

    def test_sample_returns_none_on_empty_buffer(self):
        buf = ReplayBufferNew.remote(max_size=10, max_staleness=5)
        result = self._sample(buf, num_groups=1, trainer_version=1)
        assert result is None
        ray.kill(buf)

    def test_sample_avg_trajectory_age(self):
        """avg_trajectory_age is computed from the sampled generation versions."""
        buf = ReplayBufferNew.remote(max_size=10, max_staleness=5)
        # freshest first: gen 8 (age 2), gen 6 (age 4) → avg = 3.0
        for gen_v in [6, 8]:
            self._add(buf, f"v{gen_v}", weight_version=gen_v)

        result = self._sample(buf, num_groups=2, trainer_version=10)

        assert result is not None
        assert abs(result["avg_trajectory_age"] - 3.0) < 1e-6
        ray.kill(buf)

    def test_sample_consumes_selected_rows(self):
        """Rows returned by sample() are removed from the buffer."""
        buf = ReplayBufferNew.remote(max_size=10, max_staleness=5)
        for gen_v in [1, 2, 3]:
            self._add(buf, f"v{gen_v}", weight_version=gen_v)

        self._sample(buf, num_groups=2, trainer_version=4)

        assert ray.get(buf.size.remote()) == 1
        ray.kill(buf)


class TestAsyncTrajectoryCollector:
    """Test cases for AsyncTrajectoryCollector."""

    def create_mock_config(self) -> MasterConfig:
        """Create a mock master config for testing."""
        config = {
            "grpo": {
                "num_prompts_per_step": 2,
                "num_generations_per_prompt": 3,
                "max_rollout_turns": 1,
                "async_grpo": {"max_trajectory_age_steps": 2},
            },
            "policy": {"max_total_sequence_length": 512},
        }
        return MasterConfig.model_construct(**config)

    def create_mock_batch(self, size: int = 2) -> BatchedDataDict[DatumSpec]:
        """Create a mock batch for testing."""
        message_logs = []
        for i in range(size):
            message_logs.append(
                [
                    {"role": "user", "content": f"Test prompt {i}"},
                ]
            )

        return BatchedDataDict[DatumSpec](
            {
                "task_name": ["test"] * size,
                "message_log": message_logs,
                "extra_env_info": [{}] * size,
                "loss_multiplier": torch.ones(size),
            }
        )

    def test_async_trajectory_collector_initialization(self):
        """Test AsyncTrajectoryCollector initialization."""
        buffer = ReplayBuffer.remote(max_size=10)
        mock_generation = MockGenerationInterface()
        mock_tokenizer = mock.MagicMock()
        mock_env = MockEnvironment.remote(rewards=[1.0, 2.0])
        task_to_env = {"test": mock_env}
        master_config = self.create_mock_config()

        collector = AsyncTrajectoryCollector.remote(
            policy_generation=mock_generation,
            tokenizer=mock_tokenizer,
            task_to_env=task_to_env,
            master_config=master_config,
            replay_buffer=buffer,
            start_step=0,
        )

        # Test basic functionality
        weight_version = ray.get(collector.get_weight_version.remote())
        assert weight_version == 0

        ray.kill(collector)
        ray.kill(buffer)
        ray.kill(mock_env)

    def test_async_trajectory_collector_weight_version_updates(self):
        """Test weight version updates in trajectory collector."""
        buffer = ReplayBuffer.remote(max_size=10)
        mock_generation = MockGenerationInterface()
        mock_tokenizer = mock.MagicMock()
        mock_env = MockEnvironment.remote(rewards=[1.0, 2.0])
        task_to_env = {"test": mock_env}
        master_config = self.create_mock_config()

        collector = AsyncTrajectoryCollector.remote(
            policy_generation=mock_generation,
            tokenizer=mock_tokenizer,
            task_to_env=task_to_env,
            master_config=master_config,
            replay_buffer=buffer,
            start_step=0,
        )

        # Update weight version
        ray.get(collector.set_weight_version.remote(5))
        weight_version = ray.get(collector.get_weight_version.remote())
        assert weight_version == 5

        ray.kill(collector)
        ray.kill(buffer)
        ray.kill(mock_env)

    def test_async_trajectory_collector_pause_resume(self):
        """Test pause and resume functionality."""
        buffer = ReplayBuffer.remote(max_size=10)
        mock_generation = MockGenerationInterface()
        mock_tokenizer = mock.MagicMock()
        mock_env = MockEnvironment.remote(rewards=[1.0, 2.0])
        task_to_env = {"test": mock_env}
        master_config = self.create_mock_config()

        collector = AsyncTrajectoryCollector.remote(
            policy_generation=mock_generation,
            tokenizer=mock_tokenizer,
            task_to_env=task_to_env,
            master_config=master_config,
            replay_buffer=buffer,
            start_step=0,
        )

        # Test pause and resume (these should not raise errors)
        ray.get(collector.pause.remote())
        ray.get(collector.resume.remote())

        ray.kill(collector)
        ray.kill(buffer)
        ray.kill(mock_env)

    def test_async_trajectory_collector_prepare_for_refit(self):
        """Test prepare for refit functionality."""
        buffer = ReplayBuffer.remote(max_size=10)
        mock_generation = MockGenerationInterface()
        mock_tokenizer = mock.MagicMock()
        mock_env = MockEnvironment.remote(rewards=[1.0, 2.0])
        task_to_env = {"test": mock_env}
        master_config = self.create_mock_config()

        collector = AsyncTrajectoryCollector.remote(
            policy_generation=mock_generation,
            tokenizer=mock_tokenizer,
            task_to_env=task_to_env,
            master_config=master_config,
            replay_buffer=buffer,
            start_step=0,
        )

        # Test prepare for refit (should complete without hanging)
        ray.get(collector.prepare_for_refit.remote())
        ray.get(collector.resume_after_refit.remote())

        ray.kill(collector)
        ray.kill(buffer)
        ray.kill(mock_env)

    def test_calculate_target_weights(self):
        """Test target weight calculation logic."""
        buffer = ReplayBuffer.remote(max_size=10)
        mock_generation = MockGenerationInterface()
        mock_tokenizer = mock.MagicMock()
        mock_env = MockEnvironment.remote(rewards=[1.0, 2.0])
        task_to_env = {"test": mock_env}
        master_config = self.create_mock_config()

        collector = AsyncTrajectoryCollector.remote(
            policy_generation=mock_generation,
            tokenizer=mock_tokenizer,
            task_to_env=task_to_env,
            master_config=master_config,
            replay_buffer=buffer,
            start_step=0,
        )

        # Test target weight calculation with different scenarios
        # Note: We can't directly test the private method, but we can test its effects
        # through the public interface behavior

        ray.kill(collector)
        ray.kill(buffer)
        ray.kill(mock_env)

    def test_dataloader_state_retrieval(self):
        """Test getting dataloader state for checkpointing."""
        buffer = ReplayBuffer.remote(max_size=10)
        mock_generation = MockGenerationInterface()
        mock_tokenizer = mock.MagicMock()
        mock_env = MockEnvironment.remote(rewards=[1.0, 2.0])
        task_to_env = {"test": mock_env}
        master_config = self.create_mock_config()

        collector = AsyncTrajectoryCollector.remote(
            policy_generation=mock_generation,
            tokenizer=mock_tokenizer,
            task_to_env=task_to_env,
            master_config=master_config,
            replay_buffer=buffer,
            start_step=0,
        )

        # Test getting dataloader state (should return empty dict when no dataloader)
        state = ray.get(collector.get_dataloader_state.remote())
        assert isinstance(state, dict)

        ray.kill(collector)
        ray.kill(buffer)
        ray.kill(mock_env)


class TestAsyncUtilsIntegration:
    """Integration tests for async utilities working together."""

    def create_mock_config(self) -> MasterConfig:
        """Create a mock master config for testing."""
        config = {
            "grpo": {
                "num_prompts_per_step": 2,
                "num_generations_per_prompt": 2,
                "max_rollout_turns": 1,
                "async_grpo": {"max_trajectory_age_steps": 1},
            },
            "policy": {"max_total_sequence_length": 512},
        }
        return MasterConfig.model_construct(**config)

    def create_mock_batch(self, size: int = 2) -> BatchedDataDict[DatumSpec]:
        """Create a mock batch for testing."""
        message_logs = []
        for i in range(size):
            message_logs.append(
                [
                    {"role": "user", "content": f"Test prompt {i}"},
                ]
            )

        return BatchedDataDict[DatumSpec](
            {
                "task_name": ["test"] * size,
                "message_log": message_logs,
                "extra_env_info": [{}] * size,
                "loss_multiplier": torch.ones(size),
            }
        )

    def test_buffer_and_collector_integration(self):
        """Test that buffer and collector work together correctly."""
        buffer = ReplayBuffer.remote(max_size=10)
        mock_generation = MockGenerationInterface()
        mock_tokenizer = mock.MagicMock()
        mock_env = MockEnvironment.remote(rewards=[1.0, 2.0])
        task_to_env = {"test": mock_env}
        master_config = self.create_mock_config()

        collector = AsyncTrajectoryCollector.remote(
            policy_generation=mock_generation,
            tokenizer=mock_tokenizer,
            task_to_env=task_to_env,
            master_config=master_config,
            replay_buffer=buffer,
            start_step=0,
        )

        # Verify initial state
        buffer_size = ray.get(buffer.size.remote())
        assert buffer_size == 0

        weight_version = ray.get(collector.get_weight_version.remote())
        assert weight_version == 0

        # Test weight version synchronization
        ray.get(collector.set_weight_version.remote(3))
        updated_version = ray.get(collector.get_weight_version.remote())
        assert updated_version == 3

        ray.kill(collector)
        ray.kill(buffer)
        ray.kill(mock_env)

    def test_concurrent_operations(self):
        """Test that concurrent operations don't cause race conditions."""
        buffer = ReplayBuffer.remote(max_size=5)

        # Push trajectories concurrently from multiple threads
        def push_trajectory(buffer, trajectory_id):
            trajectory = {
                "batch": {"data": f"test{trajectory_id}"},
                "rollout_metrics": {"reward": float(trajectory_id)},
            }
            return ray.get(
                buffer.add.remote(
                    trajectory,
                    weight_version=trajectory_id,
                    target_weight_version=trajectory_id + 1,
                )
            )

        # Use threading to simulate concurrent pushes
        threads = []
        results = []

        def worker(traj_id):
            result = push_trajectory(buffer, traj_id)
            results.append(result)

        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All pushes should succeed
        assert all(result == "success" for result in results)

        # Buffer should have correct size
        final_size = ray.get(buffer.size.remote())
        assert final_size == 3

        ray.kill(buffer)

    def test_error_handling(self):
        """Test error handling in async utilities."""
        # Test with invalid buffer size
        with pytest.raises(Exception):
            buffer = ReplayBuffer.remote(max_size=-1)
            ray.get(buffer.size.remote())

        # Test buffer operations
        buffer = ReplayBuffer.remote(max_size=1)

        # Test sampling from empty buffer
        sample_result = ray.get(
            buffer.sample.remote(
                num_prompt_groups=1,
                current_weight_version=0,
                max_age_steps=1,
            )
        )
        assert sample_result is None

        ray.kill(buffer)


class TestPromptExtraction:
    """Test cases for prompt extraction logic used in async GRPO advantage calculation.

    These tests verify that the length-based prompt extraction correctly handles
    multi-turn conversation prompts where the original prompt itself contains
    assistant messages (conversation history).
    """

    def test_prompt_extraction_with_multi_turn_history(self):
        """Test that prompt extraction correctly handles prompts containing assistant messages.

        This tests the fix for multi-turn conversation prompts where the original prompt
        from the dataset itself contains assistant messages (conversation history).
        The extraction should use the length field to identify original prompt messages,
        not break at the first assistant message.
        """
        # Create a multi-turn prompt with assistant messages in the history
        # Original prompt: user -> assistant -> user (3 messages, 15 tokens total)
        original_prompt_messages = [
            {
                "role": "user",
                "content": "What is 2+2?",
                "token_ids": torch.tensor([1, 2, 3, 4, 5]),
            },
            {
                "role": "assistant",
                "content": "4",
                "token_ids": torch.tensor([6, 7, 8, 9, 10]),
            },
            {
                "role": "user",
                "content": "Now what is 3+3?",
                "token_ids": torch.tensor([11, 12, 13, 14, 15]),
            },
        ]

        # Generated response (added after original prompt)
        generated_message = {
            "role": "assistant",
            "content": "6",
            "token_ids": torch.tensor([16, 17, 18]),
        }

        # Full message_log after generation
        full_message_log = original_prompt_messages + [generated_message]

        # Original prompt length = sum of token_ids in original prompt
        original_prompt_length = sum(
            len(m["token_ids"]) for m in original_prompt_messages
        )  # 15

        message_logs = [full_message_log]
        original_prompt_lengths = torch.tensor([original_prompt_length])

        result = extract_initial_prompt_messages(message_logs, original_prompt_lengths)
        initial_prompt_log = result[0]

        # Should extract all 3 original messages, NOT break at first assistant
        assert len(initial_prompt_log) == 3, (
            f"Expected 3 messages (user, assistant, user), got {len(initial_prompt_log)}. "
            "The extraction should NOT break at the first assistant message when it's part of the original prompt."
        )

        assert initial_prompt_log[0]["role"] == "user"
        assert initial_prompt_log[1]["role"] == "assistant"
        assert initial_prompt_log[2]["role"] == "user"
        assert generated_message not in initial_prompt_log

    def test_prompt_extraction_with_single_turn(self):
        """Test that prompt extraction works correctly for single-turn prompts (regression test)."""
        original_prompt_messages = [
            {
                "role": "user",
                "content": "What is 2+2?",
                "token_ids": torch.tensor([1, 2, 3, 4, 5]),
            },
        ]

        generated_message = {
            "role": "assistant",
            "content": "4",
            "token_ids": torch.tensor([6, 7, 8]),
        }

        full_message_log = original_prompt_messages + [generated_message]
        original_prompt_length = sum(
            len(m["token_ids"]) for m in original_prompt_messages
        )

        result = extract_initial_prompt_messages(
            [full_message_log], torch.tensor([original_prompt_length])
        )
        initial_prompt_log = result[0]

        assert len(initial_prompt_log) == 1
        assert initial_prompt_log[0]["role"] == "user"
        assert generated_message not in initial_prompt_log

    def test_prompt_extraction_with_system_message(self):
        """Test prompt extraction with system message included."""
        original_prompt_messages = [
            {
                "role": "system",
                "content": "You are a math tutor.",
                "token_ids": torch.tensor([1, 2, 3]),
            },
            {
                "role": "user",
                "content": "What is 2+2?",
                "token_ids": torch.tensor([4, 5, 6, 7]),
            },
        ]

        generated_message = {
            "role": "assistant",
            "content": "4",
            "token_ids": torch.tensor([8, 9]),
        }

        full_message_log = original_prompt_messages + [generated_message]
        original_prompt_length = sum(
            len(m["token_ids"]) for m in original_prompt_messages
        )

        result = extract_initial_prompt_messages(
            [full_message_log], torch.tensor([original_prompt_length])
        )
        initial_prompt_log = result[0]

        assert len(initial_prompt_log) == 2
        assert initial_prompt_log[0]["role"] == "system"
        assert initial_prompt_log[1]["role"] == "user"
        assert generated_message not in initial_prompt_log

    def test_prompt_extraction_complex_multi_turn(self):
        """Test prompt extraction with complex multi-turn history (multiple assistant turns)."""
        original_prompt_messages = [
            {
                "role": "system",
                "content": "Math tutor",
                "token_ids": torch.tensor([1, 2]),
            },
            {"role": "user", "content": "2+2?", "token_ids": torch.tensor([3, 4])},
            {"role": "assistant", "content": "4", "token_ids": torch.tensor([5, 6])},
            {"role": "user", "content": "3+3?", "token_ids": torch.tensor([7, 8])},
            {"role": "assistant", "content": "6", "token_ids": torch.tensor([9, 10])},
            {"role": "user", "content": "4+4?", "token_ids": torch.tensor([11, 12])},
        ]

        generated_message = {
            "role": "assistant",
            "content": "8",
            "token_ids": torch.tensor([13, 14]),
        }

        full_message_log = original_prompt_messages + [generated_message]
        original_prompt_length = sum(
            len(m["token_ids"]) for m in original_prompt_messages
        )

        result = extract_initial_prompt_messages(
            [full_message_log], torch.tensor([original_prompt_length])
        )
        initial_prompt_log = result[0]

        assert len(initial_prompt_log) == 6, (
            f"Expected 6 messages, got {len(initial_prompt_log)}. "
            "All history messages should be included in the prompt."
        )

        expected_roles = ["system", "user", "assistant", "user", "assistant", "user"]
        actual_roles = [m["role"] for m in initial_prompt_log]
        assert actual_roles == expected_roles
        assert generated_message not in initial_prompt_log

    def test_grpo_loss_mask_excludes_assistant_prompt_history(self):
        """Test that assistant messages in the original prompt are not trained on."""
        original_prompt_messages = [
            {
                "role": "user",
                "content": "What is 2+2?",
                "token_ids": torch.tensor([1, 2]),
            },
            {
                "role": "assistant",
                "content": "4",
                "token_ids": torch.tensor([3, 4]),
            },
            {
                "role": "user",
                "content": "Now what is 3+3?",
                "token_ids": torch.tensor([5, 6]),
            },
        ]
        generated_logprobs = torch.tensor([0.1, 0.2])
        generated_message = {
            "role": "assistant",
            "content": "6",
            "token_ids": torch.tensor([7, 8]),
            "generation_logprobs": generated_logprobs,
        }
        full_message_log = original_prompt_messages + [generated_message]

        add_grpo_token_loss_masks_and_generation_logprobs([full_message_log])

        assert torch.equal(full_message_log[0]["token_loss_mask"], torch.tensor([0, 0]))
        assert torch.equal(full_message_log[1]["token_loss_mask"], torch.tensor([0, 0]))
        assert torch.equal(full_message_log[2]["token_loss_mask"], torch.tensor([0, 0]))
        assert torch.equal(full_message_log[3]["token_loss_mask"], torch.tensor([1, 1]))
        assert torch.equal(
            full_message_log[3]["generation_logprobs"], generated_logprobs
        )

    def test_grpo_loss_mask_uses_generation_logprobs_marker(self):
        """Test that only assistant messages with generation logprobs are trainable."""
        message_log = [
            {
                "role": "assistant",
                "content": "prompt history",
                "token_ids": torch.tensor([1, 2]),
            },
            {
                "role": "user",
                "content": "next question",
                "token_ids": torch.tensor([3, 4]),
                "generation_logprobs": torch.tensor([0.3, 0.4]),
            },
            {
                "role": "assistant",
                "content": "generated response",
                "token_ids": torch.tensor([5, 6]),
                "generation_logprobs": torch.tensor([0.5, 0.6]),
            },
        ]

        add_grpo_token_loss_masks_and_generation_logprobs([message_log])

        assert torch.equal(message_log[0]["token_loss_mask"], torch.tensor([0, 0]))
        assert torch.equal(
            message_log[0]["generation_logprobs"], torch.tensor([0.0, 0.0])
        )
        assert torch.equal(message_log[1]["token_loss_mask"], torch.tensor([0, 0]))
        assert torch.equal(message_log[2]["token_loss_mask"], torch.tensor([1, 1]))
