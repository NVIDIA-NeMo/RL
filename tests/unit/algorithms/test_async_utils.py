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

import asyncio
import os
import tempfile
import threading
import unittest.mock as mock
from types import SimpleNamespace

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
import nemo_rl.algorithms.grpo as grpo_mod
from nemo_rl.algorithms.async_utils import (
    AsyncTrajectoryCollector,
    ReplayBuffer,
)
from nemo_rl.algorithms.async_utils.replay_buffer import ReplayBufferImpl
from nemo_rl.algorithms.grpo import (
    MasterConfig,
    _get_next_nemo_gym_task_index,
    add_grpo_token_loss_masks_and_generation_logprobs,
    get_idx_grouping,
)
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)


@pytest.mark.parametrize(
    ("rollouts_state", "replay_buffer_state", "expected"),
    [
        (None, None, 0),
        ({"next_ng_task_index": 20}, None, 20),
        (
            {"next_ng_task_index": 8},
            {
                "trajectories": [
                    {"_ng_task_index": 4},
                    {"_ng_task_index": 12},
                    {},
                ]
            },
            13,
        ),
        (
            {"next_ng_task_index": 20},
            {"trajectories": [{"_ng_task_index": 12}]},
            20,
        ),
    ],
)
def test_get_next_nemo_gym_task_index(rollouts_state, replay_buffer_state, expected):
    assert (
        _get_next_nemo_gym_task_index(rollouts_state, replay_buffer_state) == expected
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

    def test_local_debug_info_reports_starvation_diagnostics(self):
        buffer = ReplayBufferImpl(max_size=10)
        assert (
            buffer.add(
                {
                    "batch": {"data": "a"},
                    "rollout_metrics": {
                        "trajectory_duration_s": 10.0,
                        "max_gen_tokens_per_turn/max": 100,
                        "turns_per_sample/max": 3.0,
                    },
                },
                weight_version=0,
                target_weight_version=1,
            )
            == "success"
        )
        assert (
            buffer.add(
                {
                    "batch": {"data": "b"},
                    "rollout_metrics": {
                        "trajectory_duration_s": 20.0,
                        "max_gen_tokens_per_turn": 200,
                        "turns_per_sample/mean": 4.0,
                    },
                },
                weight_version=0,
                target_weight_version=1,
            )
            == "success"
        )

        diagnostics = buffer.get_debug_info()["starvation_diagnostics"]

        duration = diagnostics["trajectory_duration_s"]
        assert duration["mean"] == 15.0
        assert duration["median"] == 15.0
        assert duration["max"] == 20.0
        assert duration["p95"] == 20.0

        gen = diagnostics["max_gen_tokens_per_turn_in_buffer"]
        assert gen["mean"] == 150.0
        assert gen["median"] == 150.0
        assert gen["max"] == 200
        assert gen["p95"] == 200.0

        turns = diagnostics["turns_per_sample_in_buffer"]
        assert turns["mean"] == 3.5
        assert turns["median"] == 3.5
        assert turns["max"] == 4.0
        assert turns["p95"] == 4.0
        assert diagnostics["num_trajectories_sampled"] == 2

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

    def test_replay_buffer_starvation_diagnostics_nemo_gym_turn_keys(self):
        """NeMo Gym uses turns_per_sample/* in rollout_metrics; diagnostics must read them."""
        buffer = ReplayBuffer.remote(max_size=10)
        t1 = {
            "batch": {"data": "a"},
            "rollout_metrics": {
                "trajectory_duration_s": 10.0,
                "max_gen_tokens_per_turn/max": 100,
                "turns_per_sample/max": 3.0,
                "turns_per_sample/mean": 2.5,
            },
        }
        t2 = {
            "batch": {"data": "b"},
            "rollout_metrics": {
                "trajectory_duration_s": 20.0,
                "max_gen_tokens_per_turn/max": 200,
                "turns_per_sample/max": 5.0,
                "turns_per_sample/mean": 4.0,
            },
        }
        ray.get(buffer.add.remote(t1, weight_version=0, target_weight_version=1))
        ray.get(buffer.add.remote(t2, weight_version=0, target_weight_version=1))
        debug_info = ray.get(buffer.get_debug_info.remote())
        diag = debug_info["starvation_diagnostics"]["turns_per_sample_in_buffer"]
        assert diag["max"] == 5.0
        assert diag["mean"] == 4.0
        assert diag["median"] == 4.0
        assert diag["p95"] == 5.0
        gen = debug_info["starvation_diagnostics"]["max_gen_tokens_per_turn_in_buffer"]
        assert gen["max"] == 200
        assert gen["mean"] == 150
        assert gen["median"] == 150
        assert gen["p95"] == 200
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

    def test_resume_deadlock_precondition_detectable(self):
        """Regression: restored buffer can expose the async-GRPO resume deadlock.

        After PR #2651 introduced replay-buffer checkpointing, resuming from a
        checkpoint where target N is complete but target N+1 is absent caused an
        async-GRPO deadlock:

          1. Startup wait sees has_complete_batch(N) == True and breaks immediately.
          2. Training consumes all target-N trajectories and triggers a refit.
          3. Collector's post-refit target window becomes [N+2, ...] (skipping N+1).
          4. Training waits for target N+1, which nobody generates — stall forever.

        The fix is a startup pipeline barrier: before breaking, also require
        has_complete_batch(N+1) to be True (or N+1 >= max_steps).  This test
        constructs the exact precondition state — current step complete, lookahead
        absent — to ensure it remains detectable and to document the expected
        buffer readiness values that the barrier logic branches on.
        """
        num_prompts = 8
        resume_step = 30
        max_age = 1

        # Build a pre-checkpoint buffer: 8 trajectories for target 30, none for 31.
        buffer1 = ReplayBuffer.remote(max_size=20)
        for _ in range(num_prompts):
            ray.get(
                buffer1.add.remote(
                    {"batch": {"data": "x"}, "rollout_metrics": {}},
                    weight_version=resume_step - 1,
                    target_weight_version=resume_step,
                )
            )

        state = ray.get(buffer1.state_dict.remote())
        ray.kill(buffer1)

        # Restore at step 30, simulating a checkpoint resume.
        buffer2 = ReplayBuffer.remote(max_size=20)
        ray.get(
            buffer2.load_state_dict.remote(
                state,
                num_prompts_per_step=num_prompts,
                current_training_step=resume_step,
                max_age_steps=max_age,
            )
        )

        # Step 30 is complete — this is what makes the broken startup return early.
        assert ray.get(
            buffer2.has_complete_batch.remote(resume_step, num_prompts, max_age)
        ), "target step must be complete after restore"

        # Step 31 is absent — this is the deadlock precondition.
        # The startup pipeline barrier must detect this and continue waiting
        # instead of breaking, giving the collector time to generate step 31.
        assert not ray.get(
            buffer2.has_complete_batch.remote(resume_step + 1, num_prompts, max_age)
        ), "lookahead step must be absent; barrier should block here"

        ray.kill(buffer2)


class TestAsyncTrajectoryCollector:
    """Test cases for AsyncTrajectoryCollector."""

    def create_local_collector(
        self, replay_buffer=None, next_nemo_gym_task_index: int = 0
    ):
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
            next_nemo_gym_task_index=next_nemo_gym_task_index,
        )

    def _prime_collection_loop(self, collector):
        """Unblock every wait-event so _collection_loop() runs to completion."""
        for attr in (
            "_manual_pause_cleared",
            "_refit_pause_cleared",
            "_generation_limit_cleared",
        ):
            ev = threading.Event()
            ev.set()
            setattr(collector, attr, ev)
        collector._should_pause_for_generation_limits = lambda: False
        collector.running = True

    def test_collection_loop_marks_data_exhausted_on_natural_completion(self):
        """for...else path: iterator drains cleanly -> data_exhausted, not errored."""
        collector = self.create_local_collector()
        self._prime_collection_loop(collector)
        processed = []
        collector._process_batch = lambda batch: processed.append(batch)
        collector.dataloader = [{"b": 0}, {"b": 1}]

        collector._collection_loop()

        assert processed == [{"b": 0}, {"b": 1}]
        assert collector.data_exhausted is True
        assert collector.collection_failed is False
        status = collector.get_status()
        assert status["data_exhausted"] is True
        assert status["errored"] is False
        assert status["running"] is False

    def test_collection_loop_marks_errored_on_crash(self):
        """A crash sets errored (not data_exhausted) so driver guards fail fast."""
        collector = self.create_local_collector()
        self._prime_collection_loop(collector)

        def _boom(batch):
            raise RuntimeError("collection blew up")

        collector._process_batch = _boom
        collector.dataloader = [{"b": 0}]

        collector._collection_loop()

        assert collector.collection_failed is True
        assert collector.data_exhausted is False
        status = collector.get_status()
        assert status["errored"] is True
        assert status["data_exhausted"] is False
        assert status["running"] is False

    def test_collection_loop_no_exhaustion_on_manual_stop(self):
        """Breaking out (running=False) must not set data_exhausted/errored."""
        collector = self.create_local_collector()
        self._prime_collection_loop(collector)

        def _stop_after_first(batch):
            collector.running = False

        collector._process_batch = _stop_after_first
        collector.dataloader = [{"b": 0}, {"b": 1}, {"b": 2}]

        collector._collection_loop()

        assert collector.data_exhausted is False
        assert collector.collection_failed is False
        status = collector.get_status()
        assert status["data_exhausted"] is False
        assert status["errored"] is False

    def create_mock_config(self) -> MasterConfig:
        """Create a mock master config for testing."""
        config = {
            "grpo": {
                "num_prompts_per_step": 2,
                "num_generations_per_prompt": 3,
                "max_rollout_turns": 1,
                "async_grpo": {"max_trajectory_age_steps": 2},
            },
            "policy": {
                "max_total_sequence_length": 512,
                "make_sequence_length_divisible_by": 1,
            },
            "env": {"should_use_nemo_gym": False},
            "logger": {
                "wandb_enabled": False,
                "wandb": {"log_nemo_gym_full_result_tables": False},
            },
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

    def test_release_target_is_idempotent(self):
        """A batch worker can safely release its target exactly once."""
        collector = self.create_local_collector()
        target_weight = 5

        collector._generating_targets.add(target_weight)
        collector._release_target(target_weight)
        collector._release_target(target_weight)

        assert target_weight not in collector._generating_targets

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

    def test_process_batch_gap_fill_spawns_only_needed(self, monkeypatch):
        """Gap-fill sends only the needed prompt groups to one batch worker."""

        class RemoteMethod:
            def __init__(self, value):
                self.value = value

            def remote(self, *args, **kwargs):
                return self.value

        class FakeReplayBuffer:
            def __init__(self):
                # Batch has 2 prompts, but only 1 more trajectory is needed.
                self.get_trajectories_needed = RemoteMethod(1)

        started = []

        class RecordingThread:
            def __init__(self, *, target, daemon):
                assert daemon
                self.target = target

            def start(self):
                started.append(self)
                self.target()

            def is_alive(self):
                return False

        target_weight = 7
        collector = self.create_local_collector(replay_buffer=FakeReplayBuffer())
        collector.running = True

        def reserve_target(generation_weight_version):
            collector._generating_targets.add(target_weight)
            return target_weight

        captured = {}

        async def capture_batch(**kwargs):
            captured.update(kwargs)
            collector._release_target(target_weight)

        collector._get_next_target_for_generation = reserve_target
        collector._run_rollout_batch_worker = capture_batch
        monkeypatch.setattr(trajectory_collector_mod.ray, "get", lambda value: value)
        monkeypatch.setattr(
            trajectory_collector_mod._threading, "Thread", RecordingThread
        )

        collector._process_batch(self.create_mock_batch(size=2))

        # Only one worker spawned even though the batch holds 2 prompts.
        assert len(started) == 1
        assert captured["repeated_batch"].size == 3
        assert captured["use_nemo_gym"] is False
        assert target_weight not in collector._generating_targets

    def test_process_batch_batches_gym_and_stamps_task_indices(self, monkeypatch):
        """Gym uses one worker while preserving one task index per prompt group."""

        class RemoteMethod:
            def remote(self, *args, **kwargs):
                return 2

        class FakeReplayBuffer:
            get_trajectories_needed = RemoteMethod()

        started_threads = []

        class RecordingThread:
            def __init__(self, *, target, daemon):
                assert daemon
                self.target = target

            def start(self):
                started_threads.append(self)
                self.target()

            def is_alive(self):
                return False

        target_weight = 9
        collector = self.create_local_collector(
            replay_buffer=FakeReplayBuffer(), next_nemo_gym_task_index=37
        )
        collector.running = True
        captured = {}

        def reserve_target(generation_weight_version):
            collector._generating_targets.add(target_weight)
            return target_weight

        async def capture_batch(**kwargs):
            captured.update(kwargs)
            collector._release_target(target_weight)

        collector._get_next_target_for_generation = reserve_target
        collector._run_rollout_batch_worker = capture_batch
        monkeypatch.setattr(trajectory_collector_mod.ray, "get", lambda value: value)
        monkeypatch.setattr(grpo_mod, "_should_use_nemo_gym", lambda config: True)
        monkeypatch.setattr(
            trajectory_collector_mod._threading, "Thread", RecordingThread
        )

        collector._process_batch(self.create_mock_batch(size=2))

        assert len(started_threads) == 1
        assert captured["use_nemo_gym"] is True
        assert captured["num_generations"] == 3
        assert captured["repeated_batch"].size == 6
        assert [
            row["_ng_task_index"]
            for row in captured["repeated_batch"]["extra_env_info"]
        ] == [37, 37, 37, 38, 38, 38]
        assert collector.get_rollouts_state() == {"next_ng_task_index": 39}
        assert target_weight not in collector._generating_targets

    def test_process_batch_non_gym_uses_one_batched_worker(self, monkeypatch):
        """Native collection repeats all prompts into one batch worker."""

        class RemoteMethod:
            def remote(self, *args, **kwargs):
                return 2

        class FakeReplayBuffer:
            get_trajectories_needed = RemoteMethod()

        started_threads = []

        class RecordingThread:
            def __init__(self, *, target, daemon):
                assert daemon
                self.target = target

            def start(self):
                started_threads.append(self)
                self.target()

            def is_alive(self):
                return False

        target_weight = 11
        collector = self.create_local_collector(replay_buffer=FakeReplayBuffer())
        collector.running = True

        def reserve_target(generation_weight_version):
            collector._generating_targets.add(target_weight)
            return target_weight

        captured = {}

        async def capture_batch(**kwargs):
            captured.update(kwargs)
            collector._release_target(target_weight)

        collector._get_next_target_for_generation = reserve_target
        collector._run_rollout_batch_worker = capture_batch
        monkeypatch.setattr(trajectory_collector_mod.ray, "get", lambda value: value)
        monkeypatch.setattr(grpo_mod, "_should_use_nemo_gym", lambda config: False)
        monkeypatch.setattr(
            trajectory_collector_mod._threading, "Thread", RecordingThread
        )

        collector._process_batch(self.create_mock_batch(size=2))

        assert len(started_threads) == 1
        assert captured["repeated_batch"].size == 6
        assert captured["num_generations"] == 3
        assert captured["use_nemo_gym"] is False
        assert all(
            "_ng_task_index" not in row
            for row in captured["repeated_batch"]["extra_env_info"]
        )
        assert collector.get_rollouts_state() == {"next_ng_task_index": 0}
        assert target_weight not in collector._generating_targets

    def test_native_batch_worker_enqueues_each_group(self, monkeypatch):
        """The common worker enqueues every native group without Gym metadata."""

        class _ReadyResult:
            def __init__(self, value):
                self.value = value

            def __await__(self):
                async def _resolve():
                    return self.value

                return _resolve().__await__()

        class RemoteMethod:
            def __init__(self):
                self.calls = []

            def remote(self, *args):
                self.calls.append(args)
                return _ReadyResult("success")

        class FakeReplayBuffer:
            def __init__(self):
                self.add = RemoteMethod()

        replay_buffer = FakeReplayBuffer()
        collector = self.create_local_collector(replay_buffer=replay_buffer)
        collector.running = True
        target_weight = 13
        collector._generating_targets.add(target_weight)
        batches = [
            BatchedDataDict({"value": torch.tensor([1, 2, 3])}),
            BatchedDataDict({"value": torch.tensor([4, 5, 6])}),
        ]
        rollout_metrics = [{"metric": 7.0}, {"metric": 8.0}]

        async def fake_rollouts(**kwargs):
            assert kwargs["input_batch"].size == 6
            assert kwargs["num_generations"] == 3
            for group_index in range(2):
                yield trajectory_collector_mod.RolloutGroupResult(
                    group_index=group_index,
                    final_batch=batches[group_index],
                    rollout_metrics=rollout_metrics[group_index],
                )

        monkeypatch.setattr(
            trajectory_collector_mod,
            "run_async_multi_turn_rollout_groups",
            fake_rollouts,
        )

        asyncio.run(
            collector._run_rollout_batch_worker(
                repeated_batch=self.create_mock_batch(size=2).repeat_interleave(3),
                generation_weight_version=2,
                target_weight_version=target_weight,
                num_generations=3,
                use_nemo_gym=False,
            )
        )

        assert len(replay_buffer.add.calls) == 2
        for group_index, call in enumerate(replay_buffer.add.calls):
            trajectory_group, generation_weight, target = call
            assert trajectory_group["batch"] is batches[group_index]
            assert (
                trajectory_group["rollout_metrics"]["metric"]
                == rollout_metrics[group_index]["metric"]
            )
            assert trajectory_group["rollout_metrics"]["trajectory_duration_s"] >= 0
            assert "_ng_task_index" not in trajectory_group
            assert generation_weight == 2
            assert target == target_weight
        assert target_weight not in collector._generating_targets

    def test_unexpected_replay_buffer_status_fails_batch(self, monkeypatch):
        """Unknown replay-buffer statuses fail instead of polling forever."""

        class _ReadyResult:
            def __await__(self):
                async def _resolve():
                    return "unexpected"

                return _resolve().__await__()

        class _AddRemote:
            def remote(self, *args):
                return _ReadyResult()

        class _ReplayBuffer:
            add = _AddRemote()

        async def fake_rollouts(**kwargs):
            yield trajectory_collector_mod.RolloutGroupResult(
                group_index=0,
                final_batch=BatchedDataDict({"value": torch.tensor([1, 2, 3])}),
                rollout_metrics={},
            )

        collector = self.create_local_collector(replay_buffer=_ReplayBuffer())
        collector.running = True
        monkeypatch.setattr(
            trajectory_collector_mod,
            "run_async_multi_turn_rollout_groups",
            fake_rollouts,
        )

        with pytest.raises(RuntimeError, match="failed to buffer prompt groups") as exc:
            asyncio.run(
                collector._collect_rollout_batch(
                    repeated_batch=self.create_mock_batch(size=1).repeat_interleave(3),
                    generation_weight_version=2,
                    target_weight_version=3,
                    num_generations=3,
                    use_nemo_gym=False,
                )
            )

        assert exc.value.__cause__ is not None
        assert "unexpected add status" in str(exc.value.__cause__)

    def test_nemo_gym_batch_retry_does_not_duplicate_buffered_groups(self, monkeypatch):
        """A partial stream retry only enqueues prompt groups not already buffered."""

        class _ReadyResult:
            def __init__(self, value):
                self.value = value

            def __await__(self):
                async def _resolve():
                    return self.value

                return _resolve().__await__()

        class RemoteMethod:
            def __init__(self):
                self.task_indices = []

            def remote(self, trajectory_group, *args):
                self.task_indices.append(trajectory_group["_ng_task_index"])
                return _ReadyResult("success")

        class FakeReplayBuffer:
            def __init__(self):
                self.add = RemoteMethod()

        replay_buffer = FakeReplayBuffer()
        collector = self.create_local_collector(replay_buffer=replay_buffer)
        collector.running = True
        collector.master_config.policy["generation"] = {
            "stop_token_ids": [1],
            "stop_strings": ["stop"],
        }
        target_weight = 15
        collector._generating_targets.add(target_weight)
        repeated_batch = BatchedDataDict(
            {
                "extra_env_info": [
                    {"_ng_task_index": 7},
                    {"_ng_task_index": 7},
                    {"_ng_task_index": 8},
                    {"_ng_task_index": 8},
                ],
                "loss_multiplier": torch.ones(4),
            }
        )
        rollout_calls = 0

        def _rollout_result(task_index):
            return SimpleNamespace(
                task_index=task_index,
                final_batch=BatchedDataDict({"loss_multiplier": torch.ones(2)}),
                rollout_metrics={"task_index": task_index},
            )

        async def fake_rollouts(**kwargs):
            nonlocal rollout_calls
            assert kwargs["generation_config"]["stop_token_ids"] is None
            assert kwargs["generation_config"]["stop_strings"] is None
            assert kwargs["log_full_result_tables"] is False
            rollout_calls += 1
            yield _rollout_result(7)
            if rollout_calls == 1:
                raise RuntimeError("transient stream failure")
            yield _rollout_result(8)

        async def no_sleep(delay):
            return None

        import nemo_rl.experience.rollouts as rollouts_mod

        monkeypatch.setattr(rollouts_mod, "run_async_nemo_gym_rollout", fake_rollouts)
        monkeypatch.setattr(trajectory_collector_mod.asyncio, "sleep", no_sleep)

        asyncio.run(
            collector._run_rollout_batch_worker(
                repeated_batch=repeated_batch,
                generation_weight_version=3,
                target_weight_version=target_weight,
                num_generations=2,
                use_nemo_gym=True,
            )
        )

        assert rollout_calls == 2
        assert replay_buffer.add.task_indices == [7, 8]
        assert target_weight not in collector._generating_targets

    def test_invalid_gym_batch_releases_target(self):
        """Validation errors cannot leave a target reservation stuck."""
        collector = self.create_local_collector()
        collector.running = True
        target_weight = 16
        collector._generating_targets.add(target_weight)
        repeated_batch = BatchedDataDict(
            {
                "extra_env_info": [{"_ng_task_index": 7}, {}],
                "loss_multiplier": torch.ones(2),
            }
        )

        asyncio.run(
            collector._run_rollout_batch_worker(
                repeated_batch=repeated_batch,
                generation_weight_version=3,
                target_weight_version=target_weight,
                num_generations=2,
                use_nemo_gym=True,
            )
        )

        assert target_weight not in collector._generating_targets

    def test_rollouts_state_retrieval(self):
        collector = self.create_local_collector(next_nemo_gym_task_index=123)

        assert collector.get_rollouts_state() == {"next_ng_task_index": 123}

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
            "policy": {
                "max_total_sequence_length": 512,
                "make_sequence_length_divisible_by": 1,
            },
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


def test_turn_count_fallback_priority():
    """_rollout_metrics_turn_count_for_diagnostics honors the documented key priority."""
    f = ReplayBufferImpl._rollout_metrics_turn_count_for_diagnostics
    assert (
        f(
            {
                "max_turns_per_sample": 7,
                "avg_turns_per_sample": 1,
                "turns_per_sample/max": 2,
                "turns_per_sample/mean": 3,
            }
        )
        == 7.0
    )
    assert f({"avg_turns_per_sample": 4, "turns_per_sample/max": 2}) == 4.0
    assert f({"turns_per_sample/max": 5, "turns_per_sample/mean": 3}) == 5.0
    assert f({"turns_per_sample/mean": 6}) == 6.0
    assert f({"reward": 1.0}) is None
