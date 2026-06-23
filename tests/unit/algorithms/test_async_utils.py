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

import nemo_rl.algorithms.async_utils.trajectory_collector as trajectory_collector_mod
from nemo_rl.algorithms.async_utils import (
    AsyncTrajectoryCollector,
    ReplayBuffer,
)
from nemo_rl.algorithms.async_utils.replay_buffer import (
    ReplayBufferImpl,
    ReplayBufferNew,
)
from nemo_rl.algorithms.grpo import (
    MasterConfig,
    add_grpo_token_loss_masks_and_generation_logprobs,
    get_idx_grouping,
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


class TestReplayBufferImplCheckpointing:
    """Direct implementation tests for checkpoint coverage.

    Ray actor execution is not reliably attributed to source coverage, so these
    tests cover the checkpoint/restore helpers on the local implementation class.
    """

    def _state(
        self,
        trajectory_versions: list[int],
        target_weight_versions: list[int],
        last_target_weight_already_generated: int,
        max_size: int = 10,
    ) -> dict:
        return {
            "trajectories": [
                {"batch": {"data": f"traj_{idx}"}, "rollout_metrics": {}}
                for idx in range(len(trajectory_versions))
            ],
            "trajectory_versions": trajectory_versions,
            "target_weight_versions": target_weight_versions,
            "last_target_weight_already_generated": (
                last_target_weight_already_generated
            ),
            "max_size": max_size,
        }

    def test_local_restore_prepares_current_step_for_gap_fill(self):
        buffer = ReplayBufferImpl(max_size=10)
        state = self._state(
            trajectory_versions=[0, 1, 1, 2],
            target_weight_versions=[1, 2, 2, 3],
            last_target_weight_already_generated=3,
        )

        buffer.load_state_dict(
            state,
            num_prompts_per_step=2,
            current_training_step=2,
        )

        assert buffer.get_debug_info()["trajectory_versions"] == [1, 1, 2]
        assert buffer.get_debug_info()["target_weight_versions"] == [2, 2, 3]
        assert buffer.get_last_target_weight_already_generated() == 1
        assert buffer.has_complete_batch(2, 2)
        assert not buffer.has_complete_batch(3, 2)
        assert buffer.get_trajectories_needed(2, 2) == 0
        assert buffer.get_trajectories_needed(3, 2) == 1

    def test_local_restore_empty_state_resets_generation_watermark(self):
        buffer = ReplayBufferImpl(max_size=10)
        state = self._state(
            trajectory_versions=[],
            target_weight_versions=[],
            last_target_weight_already_generated=7,
        )

        buffer.load_state_dict(
            state,
            num_prompts_per_step=2,
            current_training_step=5,
        )

        assert buffer.size() == 0
        assert buffer.get_last_target_weight_already_generated() == 4
        assert buffer.get_trajectories_needed(5, 2) == 2

    def test_local_restore_removes_stale_trajectories(self):
        buffer = ReplayBufferImpl(max_size=10)
        state = self._state(
            trajectory_versions=[0, 1, 4],
            target_weight_versions=[5, 5, 5],
            last_target_weight_already_generated=5,
        )

        buffer.load_state_dict(
            state,
            num_prompts_per_step=2,
            current_training_step=5,
            max_age_steps=1,
        )

        assert buffer.get_debug_info()["trajectory_versions"] == [4]
        assert buffer.get_debug_info()["target_weight_versions"] == [5]
        assert not buffer.has_complete_batch(5, 2)
        assert buffer.get_trajectories_needed(5, 2) == 1

    def test_local_restore_truncates_after_resume_cleanup(self):
        buffer = ReplayBufferImpl(max_size=2)
        state = self._state(
            trajectory_versions=[0, 1, 2, 3],
            target_weight_versions=[1, 2, 3, 4],
            last_target_weight_already_generated=4,
            max_size=4,
        )

        buffer.load_state_dict(
            state,
            num_prompts_per_step=1,
            current_training_step=2,
        )

        assert buffer.get_debug_info()["trajectory_versions"] == [1, 2]
        assert buffer.get_debug_info()["target_weight_versions"] == [2, 3]

    def test_local_restore_without_current_step_rechecks_after_stale_removal(self):
        buffer = ReplayBufferImpl(max_size=10)
        state = self._state(
            trajectory_versions=[0, 4, 4],
            target_weight_versions=[5, 5, 6],
            last_target_weight_already_generated=6,
        )

        buffer.load_state_dict(
            state,
            num_prompts_per_step=2,
            max_age_steps=1,
        )

        assert buffer.size() == 0
        assert buffer.get_last_target_weight_already_generated() == -1

    def test_local_sample_evicts_stale_restored_trajectories(self):
        buffer = ReplayBufferImpl(max_size=10)
        assert (
            buffer.add(
                {"batch": {"data": "stale"}, "rollout_metrics": {}},
                weight_version=0,
                target_weight_version=5,
            )
            == "success"
        )
        assert (
            buffer.add(
                {"batch": {"data": "valid"}, "rollout_metrics": {}},
                weight_version=4,
                target_weight_version=5,
            )
            == "success"
        )

        sample_result = buffer.sample(
            num_prompt_groups=1,
            current_weight_version=5,
            max_age_steps=1,
        )

        assert sample_result is not None
        assert sample_result["trajectories"][0]["batch"]["data"] == "valid"
        assert buffer.size() == 0

    def test_local_load_state_dict_validates_checkpoint_shape(self):
        buffer = ReplayBufferImpl(max_size=10)

        with pytest.raises(ValueError, match="Checkpoint missing required keys"):
            buffer.load_state_dict(
                {
                    "trajectories": [],
                    "trajectory_versions": [],
                }
            )

        with pytest.raises(ValueError, match="inconsistent replay buffer lengths"):
            buffer.load_state_dict(
                {
                    "trajectories": [{"batch": {"data": "test"}}],
                    "trajectory_versions": [0, 1],
                    "target_weight_versions": [1],
                    "last_target_weight_already_generated": 1,
                }
            )


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
        assert ray.get(buffer.get_last_target_weight_already_generated.remote()) == 2

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
        assert ray.get(buffer.get_last_target_weight_already_generated.remote()) == -1

        ray.kill(buffer)

    def test_replay_buffer_watermark_advances_only_after_consumption(self):
        """Test buffering alone does not mark a target as consumed."""
        buffer = ReplayBuffer.remote(max_size=10)

        trajectory1 = {"batch": {"data": "test1"}, "rollout_metrics": {}}
        trajectory2 = {"batch": {"data": "test2"}, "rollout_metrics": {}}

        ray.get(
            buffer.add.remote(trajectory1, weight_version=4, target_weight_version=5)
        )
        assert ray.get(buffer.get_last_target_weight_already_generated.remote()) == -1

        sample_result = ray.get(
            buffer.sample.remote(
                num_prompt_groups=2,
                current_weight_version=5,
                max_age_steps=1,
            )
        )
        assert sample_result is None
        assert ray.get(buffer.get_last_target_weight_already_generated.remote()) == -1

        ray.get(
            buffer.add.remote(trajectory2, weight_version=4, target_weight_version=5)
        )
        sample_result = ray.get(
            buffer.sample.remote(
                num_prompt_groups=2,
                current_weight_version=5,
                max_age_steps=1,
            )
        )

        assert sample_result is not None
        assert ray.get(buffer.get_last_target_weight_already_generated.remote()) == 5

        ray.kill(buffer)

    def test_replay_buffer_age_filtering(self):
        """Test that old trajectories are evicted."""
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

        assert ray.get(buffer.size.remote()) == 2

        sample_result = ray.get(
            buffer.sample.remote(
                num_prompt_groups=1,
                current_weight_version=3,
                max_age_steps=1,
            )
        )

        assert sample_result is not None
        assert len(sample_result["trajectories"]) == 1
        assert sample_result["trajectories"][0]["batch"]["data"] == "recent"
        assert ray.get(buffer.size.remote()) == 0

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

    def test_replay_buffer_state_dict(self):
        """Test state_dict serialization for checkpointing."""
        buffer = ReplayBuffer.remote(max_size=10)

        trajectory1 = {"batch": {"data": "test1"}, "rollout_metrics": {"reward": 1.0}}
        trajectory2 = {"batch": {"data": "test2"}, "rollout_metrics": {"reward": 2.0}}

        ray.get(
            buffer.add.remote(trajectory1, weight_version=5, target_weight_version=6)
        )
        ray.get(
            buffer.add.remote(trajectory2, weight_version=6, target_weight_version=7)
        )

        state = ray.get(buffer.state_dict.remote())

        assert state["trajectories"][0]["batch"]["data"] == "test1"
        assert state["trajectories"][1]["batch"]["data"] == "test2"
        assert state["trajectory_versions"] == [5, 6]
        assert state["target_weight_versions"] == [6, 7]
        assert state["last_target_weight_already_generated"] == -1
        assert state["max_size"] == 10

        ray.kill(buffer)

    def test_replay_buffer_load_state_dict(self):
        """Test load_state_dict restoration from checkpoint."""
        buffer1 = ReplayBuffer.remote(max_size=10)

        trajectory1 = {"batch": {"data": "test1"}, "rollout_metrics": {"reward": 1.0}}
        trajectory2 = {"batch": {"data": "test2"}, "rollout_metrics": {"reward": 2.0}}

        ray.get(
            buffer1.add.remote(trajectory1, weight_version=5, target_weight_version=6)
        )
        ray.get(
            buffer1.add.remote(trajectory2, weight_version=6, target_weight_version=7)
        )

        state = ray.get(buffer1.state_dict.remote())
        ray.kill(buffer1)

        buffer2 = ReplayBuffer.remote(max_size=10)
        assert ray.get(buffer2.size.remote()) == 0

        ray.get(buffer2.load_state_dict.remote(state))

        assert ray.get(buffer2.size.remote()) == 2
        debug_info = ray.get(buffer2.get_debug_info.remote())
        assert debug_info["trajectory_versions"] == [5, 6]
        assert debug_info["target_weight_versions"] == [6, 7]
        assert ray.get(buffer2.get_last_target_weight_already_generated.remote()) == -1

        ray.kill(buffer2)

    def test_replay_buffer_state_dict_round_trip_sampling(self):
        """Test save/restore preserves sampling behavior."""
        buffer1 = ReplayBuffer.remote(max_size=10)

        for i in range(3):
            trajectory = {
                "batch": {"data": f"test{i}"},
                "rollout_metrics": {"reward": float(i)},
            }
            ray.get(
                buffer1.add.remote(
                    trajectory, weight_version=i, target_weight_version=i + 1
                )
            )

        state = ray.get(buffer1.state_dict.remote())
        ray.kill(buffer1)

        buffer2 = ReplayBuffer.remote(max_size=10)
        ray.get(buffer2.load_state_dict.remote(state))

        sample_result = ray.get(
            buffer2.sample.remote(
                num_prompt_groups=1,
                current_weight_version=2,
                max_age_steps=2,
            )
        )

        assert sample_result is not None
        assert sample_result["trajectories"][0]["batch"]["data"] == "test1"

        ray.kill(buffer2)

    def test_replay_buffer_load_state_dict_max_size_change(self):
        """Test load_state_dict truncates after resume cleanup."""
        buffer1 = ReplayBuffer.remote(max_size=5)

        for i in range(4):
            trajectory = {
                "batch": {"data": f"test{i}"},
                "rollout_metrics": {"reward": float(i)},
            }
            ray.get(
                buffer1.add.remote(
                    trajectory, weight_version=i, target_weight_version=i + 1
                )
            )

        state = ray.get(buffer1.state_dict.remote())
        ray.kill(buffer1)

        buffer2 = ReplayBuffer.remote(max_size=2)
        ray.get(
            buffer2.load_state_dict.remote(
                state,
                num_prompts_per_step=1,
                current_training_step=2,
            )
        )

        assert ray.get(buffer2.size.remote()) == 2
        debug_info = ray.get(buffer2.get_debug_info.remote())
        assert debug_info["trajectory_versions"] == [1, 2]
        assert debug_info["target_weight_versions"] == [2, 3]

        ray.kill(buffer2)

    def test_replay_buffer_load_empty_state_resets_generation_watermark(self):
        """Test empty restore can generate from the current step."""
        buffer = ReplayBuffer.remote(max_size=10)

        state = {
            "trajectories": [],
            "trajectory_versions": [],
            "target_weight_versions": [],
            "last_target_weight_already_generated": 7,
            "max_size": 10,
        }

        ray.get(
            buffer.load_state_dict.remote(
                state,
                num_prompts_per_step=2,
                current_training_step=5,
            )
        )

        assert ray.get(buffer.size.remote()) == 0
        assert ray.get(buffer.get_last_target_weight_already_generated.remote()) == 4
        assert ray.get(buffer.get_trajectories_needed.remote(5, 2)) == 2

        ray.kill(buffer)

    def test_replay_buffer_restore_removes_stale_trajectories(self):
        """Test stale restored trajectories do not make a step look complete."""
        buffer = ReplayBuffer.remote(max_size=10)

        state = {
            "trajectories": [
                {"batch": {"data": "stale_a"}},
                {"batch": {"data": "stale_b"}},
                {"batch": {"data": "valid_a"}},
            ],
            "trajectory_versions": [0, 1, 4],
            "target_weight_versions": [5, 5, 5],
            "last_target_weight_already_generated": 5,
            "max_size": 10,
        }

        ray.get(
            buffer.load_state_dict.remote(
                state,
                num_prompts_per_step=2,
                current_training_step=5,
                max_age_steps=1,
            )
        )

        debug_info = ray.get(buffer.get_debug_info.remote())
        assert debug_info["trajectory_versions"] == [4]
        assert debug_info["target_weight_versions"] == [5]
        assert not ray.get(buffer.has_complete_batch.remote(5, 2))
        assert ray.get(buffer.get_trajectories_needed.remote(5, 2)) == 1

        ray.kill(buffer)

    def test_replay_buffer_readiness_ignores_stale_trajectories(self):
        """Test readiness helpers match sample's age-window filtering."""
        buffer = ReplayBuffer.remote(max_size=10)

        for version in [0, 1, 4]:
            ray.get(
                buffer.add.remote(
                    {"batch": {"data": f"version_{version}"}},
                    weight_version=version,
                    target_weight_version=5,
                )
            )

        assert ray.get(buffer.has_complete_batch.remote(5, 3))
        assert ray.get(buffer.get_trajectories_needed.remote(5, 3)) == 0

        assert not ray.get(buffer.has_complete_batch.remote(5, 3, 1))
        assert ray.get(buffer.get_trajectories_needed.remote(5, 3, 1)) == 2

        assert (
            ray.get(
                buffer.sample.remote(
                    num_prompt_groups=3,
                    current_weight_version=5,
                    max_age_steps=1,
                )
            )
            is None
        )

        ray.kill(buffer)

    def test_replay_buffer_load_state_dict_missing_keys(self):
        """Test load_state_dict raises for missing required keys."""
        buffer = ReplayBuffer.remote(max_size=10)

        incomplete_state = {
            "trajectories": [],
            "trajectory_versions": [],
        }

        with pytest.raises(ValueError, match="Checkpoint missing required keys"):
            ray.get(buffer.load_state_dict.remote(incomplete_state))

        ray.kill(buffer)

    def test_replay_buffer_load_state_dict_inconsistent_lengths(self):
        """Test load_state_dict raises for inconsistent parallel lists."""
        buffer = ReplayBuffer.remote(max_size=10)

        bad_state = {
            "trajectories": [{"batch": {"data": "test"}}],
            "trajectory_versions": [0, 1],
            "target_weight_versions": [1],
            "last_target_weight_already_generated": 1,
        }

        with pytest.raises(ValueError, match="inconsistent replay buffer lengths"):
            ray.get(buffer.load_state_dict.remote(bad_state))

        ray.kill(buffer)

    def test_replay_buffer_restore_for_training_step_gap_fill_accounting(self):
        """Test resume cleanup keeps incomplete future targets for gap filling."""
        buffer = ReplayBuffer.remote(max_size=10)

        state = {
            "trajectories": [
                {"batch": {"data": "past"}},
                {"batch": {"data": "step2_a"}},
                {"batch": {"data": "step2_b"}},
                {"batch": {"data": "step3_a"}},
            ],
            "trajectory_versions": [0, 1, 1, 2],
            "target_weight_versions": [1, 2, 2, 3],
            "last_target_weight_already_generated": 3,
            "max_size": 10,
        }

        ray.get(
            buffer.load_state_dict.remote(
                state,
                num_prompts_per_step=2,
                current_training_step=2,
            )
        )

        debug_info = ray.get(buffer.get_debug_info.remote())
        assert debug_info["trajectory_versions"] == [1, 1, 2]
        assert debug_info["target_weight_versions"] == [2, 2, 3]
        assert ray.get(buffer.has_complete_batch.remote(2, 2))
        assert not ray.get(buffer.has_complete_batch.remote(3, 2))
        assert ray.get(buffer.get_trajectories_needed.remote(2, 2)) == 0
        assert ray.get(buffer.get_trajectories_needed.remote(3, 2)) == 1
        assert ray.get(buffer.get_last_target_weight_already_generated.remote()) == 1

        ray.kill(buffer)

    def test_replay_buffer_remove_incomplete_resets_watermark_before_first_remaining_target(
        self,
    ):
        """Test fallback cleanup does not skip gaps after removing partial targets."""
        buffer = ReplayBuffer.remote(max_size=10)

        state = {
            "trajectories": [
                {"batch": {"data": "step5_a"}},
                {"batch": {"data": "step5_b"}},
                {"batch": {"data": "step6_a"}},
                {"batch": {"data": "step8_a"}},
                {"batch": {"data": "step8_b"}},
            ],
            "trajectory_versions": [4, 4, 5, 7, 7],
            "target_weight_versions": [5, 5, 6, 8, 8],
            "last_target_weight_already_generated": 8,
            "max_size": 10,
        }

        ray.get(buffer.load_state_dict.remote(state, num_prompts_per_step=2))

        debug_info = ray.get(buffer.get_debug_info.remote())
        assert debug_info["target_weight_versions"] == [5, 5, 8, 8]
        assert ray.get(buffer.get_last_target_weight_already_generated.remote()) == 4

        ray.kill(buffer)

    def test_replay_buffer_checkpoint_with_torch_save(self):
        """Test that state_dict can be saved and loaded with torch.save/load."""
        buffer1 = ReplayBuffer.remote(max_size=10)

        trajectory = {
            "batch": {
                "token_ids": torch.tensor([1, 2, 3]),
                "rewards": torch.tensor([0.5]),
            },
            "rollout_metrics": {"reward": 1.0, "length": 10},
            "timestamp": 12345.0,
        }
        ray.get(
            buffer1.add.remote(trajectory, weight_version=5, target_weight_version=6)
        )

        state = ray.get(buffer1.state_dict.remote())
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state, f.name)
            checkpoint_path = f.name

        ray.kill(buffer1)

        loaded_state = torch.load(checkpoint_path, weights_only=False)
        buffer2 = ReplayBuffer.remote(max_size=10)
        ray.get(buffer2.load_state_dict.remote(loaded_state))

        assert ray.get(buffer2.size.remote()) == 1
        debug_info = ray.get(buffer2.get_debug_info.remote())
        assert debug_info["trajectory_versions"] == [5]
        assert debug_info["target_weight_versions"] == [6]

        os.unlink(checkpoint_path)
        ray.kill(buffer2)


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

    def create_local_collector(self, replay_buffer=None):
        """Create a non-Ray collector instance for unit-testing local state."""
        collector_cls = AsyncTrajectoryCollector.__ray_metadata__.modified_class
        mock_generation = MockGenerationInterface()
        mock_tokenizer = mock.MagicMock()
        task_to_env = {}
        master_config = self.create_mock_config()
        if replay_buffer is None:
            replay_buffer = mock.MagicMock()

        return collector_cls(
            policy_generation=mock_generation,
            tokenizer=mock_tokenizer,
            task_to_env=task_to_env,
            master_config=master_config,
            replay_buffer=replay_buffer,
            start_step=0,
        )

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

    def test_maybe_release_target_waits_for_spawning_to_close(self):
        """Test fast workers do not release a target while spawning is open."""
        collector = self.create_local_collector()
        target_weight = 5

        collector._generating_targets.add(target_weight)
        collector._spawning_targets.add(target_weight)
        collector._spawned_per_target[target_weight] = 1
        collector._completed_per_target[target_weight] = 1
        collector._buffered_per_target[target_weight] = 1

        collector._maybe_release_target(target_weight)

        assert target_weight in collector._generating_targets
        assert collector._spawned_per_target[target_weight] == 1
        assert collector._completed_per_target[target_weight] == 1
        assert collector._buffered_per_target[target_weight] == 1

        collector._spawning_targets.remove(target_weight)
        collector._maybe_release_target(target_weight)

        assert target_weight not in collector._generating_targets
        assert target_weight not in collector._spawned_per_target
        assert target_weight not in collector._completed_per_target
        assert target_weight not in collector._buffered_per_target

    def test_process_batch_releases_target_when_worker_start_fails(self, monkeypatch):
        """Test start failures do not leave a target reserved forever."""

        class RemoteMethod:
            def __init__(self, value):
                self.value = value

            def remote(self, *args, **kwargs):
                return self.value

        class FakeReplayBuffer:
            def __init__(self):
                self.get_trajectories_needed = RemoteMethod(1)

        class FakeBatch:
            size = 1

            def slice(self, start, end):
                return self

            def repeat_interleave(self, repeats):
                return self

        class FailingThread:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                raise RuntimeError("thread start failed")

            def is_alive(self):
                return False

        target_weight = 5
        collector = self.create_local_collector(replay_buffer=FakeReplayBuffer())
        collector.running = True

        def reserve_target(generation_weight_version):
            collector._generating_targets.add(target_weight)
            return target_weight

        collector._get_next_target_for_generation = reserve_target
        monkeypatch.setattr(trajectory_collector_mod.ray, "get", lambda value: value)
        monkeypatch.setattr(
            trajectory_collector_mod._threading,
            "Thread",
            FailingThread,
        )

        collector._process_batch(FakeBatch())

        assert target_weight not in collector._generating_targets
        assert target_weight not in collector._spawning_targets
        assert target_weight not in collector._spawned_per_target
        assert target_weight not in collector._completed_per_target
        assert target_weight not in collector._buffered_per_target

    def test_process_batch_gap_fill_spawns_only_needed(self, monkeypatch):
        """Gap-fill generates only the trajectories still needed for a target."""

        class RemoteMethod:
            def __init__(self, value):
                self.value = value

            def remote(self, *args, **kwargs):
                return self.value

        class FakeReplayBuffer:
            def __init__(self):
                # Batch has 2 prompts, but only 1 more trajectory is needed.
                self.get_trajectories_needed = RemoteMethod(1)

        class FakeBatch:
            size = 2

            def slice(self, start, end):
                return self

            def repeat_interleave(self, repeats):
                return self

        started = []

        class RecordingThread:
            def __init__(self, *args, **kwargs):
                self._args = kwargs.get("args", ())

            def start(self):
                started.append(self._args)
                # Simulate the worker finishing and recording completion.
                target = self._args[2]
                with collector._counter_lock:
                    collector._completed_per_target[target] = (
                        collector._completed_per_target.get(target, 0) + 1
                    )

            def is_alive(self):
                return False

        target_weight = 7
        collector = self.create_local_collector(replay_buffer=FakeReplayBuffer())
        collector.running = True

        def reserve_target(generation_weight_version):
            collector._generating_targets.add(target_weight)
            return target_weight

        collector._get_next_target_for_generation = reserve_target
        monkeypatch.setattr(trajectory_collector_mod.ray, "get", lambda value: value)
        monkeypatch.setattr(
            trajectory_collector_mod._threading, "Thread", RecordingThread
        )

        collector._process_batch(FakeBatch())

        # Only one worker spawned even though the batch holds 2 prompts.
        assert len(started) == 1
        # Reservation released after spawning closed and the worker completed.
        assert target_weight not in collector._generating_targets
        assert target_weight not in collector._spawning_targets
        assert target_weight not in collector._spawned_per_target
        assert target_weight not in collector._completed_per_target

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


class TestPromptGrouping:
    """Test cases for the ``(task_name, idx)`` grouping key used in GRPO
    advantage estimation, plus the GRPO loss-mask helper.

    ``get_idx_grouping`` replaces the older prompt-token-sequence grouping,
    which is broken for multi-turn environments (all tasks share the same
    initial prompt tokens) and for multi-dataset batches (idx values collide
    across datasets). task_name disambiguates both cases.
    """

    def test_single_dataset_groups_by_idx(self):
        """With a single task_name, responses group by idx alone."""
        batch = {
            "idx": [0, 0, 1, 1],
            "task_name": ["math", "math", "math", "math"],
        }
        grouping = get_idx_grouping(batch)

        assert grouping.shape == (4, 2)
        assert grouping.dtype == torch.long
        # Same idx -> same group; different idx -> different group.
        assert torch.equal(grouping[0], grouping[1])
        assert torch.equal(grouping[2], grouping[3])
        assert not torch.equal(grouping[0], grouping[2])

    def test_multi_dataset_idx_collision_disambiguated_by_task_name(self):
        """Same idx from different datasets must not group together."""
        batch = {
            "idx": [0, 0, 0, 0],
            "task_name": ["math", "math", "code", "code"],
        }
        grouping = get_idx_grouping(batch)

        # Identical idx, but different task_name -> different group.
        assert torch.equal(grouping[0], grouping[1])
        assert torch.equal(grouping[2], grouping[3])
        assert not torch.equal(grouping[0], grouping[2])

    def test_none_task_name_is_normalised(self):
        """task_name=None (e.g. the math processor) normalises to a sortable key."""
        batch = {
            "idx": [0, 1, 0],
            "task_name": [None, None, None],
        }
        grouping = get_idx_grouping(batch)

        assert grouping.shape == (3, 2)
        # All None -> single task id, so grouping reduces to idx.
        assert torch.equal(grouping[0], grouping[2])
        assert not torch.equal(grouping[0], grouping[1])


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
