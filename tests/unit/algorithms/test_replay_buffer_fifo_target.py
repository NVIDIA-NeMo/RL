# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""FIFO target assignment in the async replay buffer (no Ray actor needed)."""

import pytest

from nemo_rl.algorithms.async_utils.replay_buffer import ReplayBufferImpl


def _group(name: str) -> dict:
    return {"batch": {"data": name}, "rollout_metrics": {"reward": 1.0}}


def _buffer(fifo: bool = True, num_prompts_per_step: int = 2) -> ReplayBufferImpl:
    return ReplayBufferImpl(
        max_size=16,
        drop_incomplete_targets_on_restore=False,
        fifo_target_assignment=fifo,
        num_prompts_per_step=num_prompts_per_step if fifo else None,
        max_age_steps=2 if fifo else None,
    )


def test_fifo_requires_num_prompts_per_step_and_age_window():
    with pytest.raises(ValueError, match="num_prompts_per_step"):
        ReplayBufferImpl(
            max_size=4,
            drop_incomplete_targets_on_restore=False,
            fifo_target_assignment=True,
            max_age_steps=2,
        )
    with pytest.raises(ValueError, match="max_age_steps"):
        ReplayBufferImpl(
            max_size=4,
            drop_incomplete_targets_on_restore=False,
            fifo_target_assignment=True,
            num_prompts_per_step=2,
        )


def test_disabled_keeps_reserved_targets():
    buffer = _buffer(fifo=False)
    buffer.add(_group("a"), weight_version=0, target_weight_version=1)
    buffer.add(_group("b"), weight_version=0, target_weight_version=1)
    assert buffer.target_weight_versions == [1, 1]
    assert not buffer.has_complete_batch(0, num_prompts_per_step=2)


def test_arriving_groups_fill_the_earliest_incomplete_step_first():
    """Groups reserved for step 1 complete step 0 while step 0's batch is still out."""
    buffer = _buffer()
    buffer.add(_group("late-batch-0"), weight_version=0, target_weight_version=1)
    buffer.add(_group("late-batch-1"), weight_version=0, target_weight_version=1)
    assert buffer.target_weight_versions == [0, 0]
    assert buffer.has_complete_batch(0, num_prompts_per_step=2)
    assert not buffer.has_complete_batch(1, num_prompts_per_step=2)

    # Step 0 is full, so the next arrivals keep their reservation.
    buffer.add(_group("c"), weight_version=0, target_weight_version=1)
    assert buffer.target_weight_versions == [0, 0, 1]
    # Step 0's own batch finally lands and tops up step 1 instead.
    buffer.add(_group("straggler"), weight_version=0, target_weight_version=0)
    assert buffer.target_weight_versions == [0, 0, 1, 1]
    assert buffer.has_complete_batch(1, num_prompts_per_step=2)


def test_never_assigns_a_step_earlier_than_the_generation_version():
    """A group generated with weights v2 cannot train step 1 (age window: v <= target)."""
    buffer = _buffer()
    buffer.add(_group("v0-for-1"), weight_version=0, target_weight_version=1)
    assert buffer.target_weight_versions == [0]  # step 0 still open
    buffer.add(_group("v2-for-3"), weight_version=2, target_weight_version=3)
    # Step 0 and 1 lack groups but are older than the generating weights.
    assert buffer.target_weight_versions == [0, 2]
    assert buffer._is_valid_for_target(2, 2, max_age_steps=2)


def test_never_assigns_a_consumed_step():
    buffer = _buffer()
    buffer.add(_group("a"), weight_version=0, target_weight_version=0)
    buffer.add(_group("b"), weight_version=0, target_weight_version=0)
    sampled = buffer.sample(
        num_prompt_groups=2, current_weight_version=0, max_age_steps=2
    )
    assert sampled is not None and len(sampled["trajectories"]) == 2
    assert buffer.last_target_weight_already_generated == 0

    # Reserved for step 2 while step 1 is still empty -> fills step 1, not 0.
    buffer.add(_group("c"), weight_version=1, target_weight_version=2)
    assert buffer.target_weight_versions == [1]


def test_sampling_and_age_window_unchanged_for_restamped_groups():
    buffer = _buffer()
    for name in ("a", "b"):
        buffer.add(_group(name), weight_version=0, target_weight_version=1)
    assert buffer.target_weight_versions == [0, 0]
    sampled = buffer.sample(
        num_prompt_groups=2, current_weight_version=0, max_age_steps=1
    )
    assert sampled is not None
    assert [t["batch"]["data"] for t in sampled["trajectories"]] == ["a", "b"]
    assert sampled["avg_trajectory_age"] == 0.0
    assert buffer.size() == 0


def test_restamping_never_leaves_the_age_window():
    """Reviewer scenario: with an age window of 1 a group may not be pushed to a step it is stale for."""
    buffer = ReplayBufferImpl(
        max_size=16,
        drop_incomplete_targets_on_restore=False,
        fifo_target_assignment=True,
        num_prompts_per_step=2,
        max_age_steps=1,
    )
    for name in ("a", "b"):
        buffer.add(_group(name), weight_version=0, target_weight_version=0)
    for name in ("c", "d"):
        buffer.add(_group(name), weight_version=0, target_weight_version=1)
    assert buffer.target_weight_versions == [0, 0, 1, 1]
    # A straggler generated with v0 for step 0: steps 0 and 1 are full and step 2
    # is outside its age window, so it keeps its reservation instead of being
    # stamped onto a step that could never train on it.
    buffer.add(_group("straggler"), weight_version=0, target_weight_version=0)
    assert buffer.target_weight_versions[-1] == 0
    for target, version in zip(
        buffer.target_weight_versions, buffer.trajectory_versions
    ):
        assert buffer._is_valid_for_target(version, target, max_age_steps=1)
    # Every restamped step is complete by the same age-aware count sample() uses.
    assert buffer.has_complete_batch(0, num_prompts_per_step=2, max_age_steps=1)
    assert buffer.has_complete_batch(1, num_prompts_per_step=2, max_age_steps=1)


def test_counts_use_age_valid_groups_only():
    """Stale groups for a step do not make the step look full to FIFO stamping."""
    buffer = ReplayBufferImpl(
        max_size=16,
        drop_incomplete_targets_on_restore=False,
        fifo_target_assignment=True,
        num_prompts_per_step=2,
        max_age_steps=1,
    )
    # Two groups stamped for step 2 but generated with v0 are stale for step 2
    # (age window 1) and must not count toward its batch.
    buffer.trajectories.extend([_group("stale-a"), _group("stale-b")])
    buffer.trajectory_versions.extend([0, 0])
    buffer.target_weight_versions.extend([2, 2])
    buffer.last_target_weight_already_generated = 1
    buffer.add(_group("fresh"), weight_version=2, target_weight_version=3)
    # Step 2 lacks age-valid groups, so the fresh group fills it.
    assert buffer.target_weight_versions[-1] == 2


def test_stale_reservation_for_a_consumed_step_moves_to_the_next_open_step():
    buffer = ReplayBufferImpl(
        max_size=16,
        drop_incomplete_targets_on_restore=False,
        fifo_target_assignment=True,
        num_prompts_per_step=2,
        max_age_steps=2,
    )
    buffer.last_target_weight_already_generated = 9
    buffer.add(_group("late"), weight_version=9, target_weight_version=3)
    assert buffer.target_weight_versions == [10]
