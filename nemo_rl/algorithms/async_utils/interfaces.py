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

from typing import Any, Optional, Protocol, TypedDict

from nemo_rl.data.weights import TaskDeficits, TaskName, TaskQuota


class SampledBatch(TypedDict):
    """Payload returned by ``ReplayBuffer.sample``."""

    trajectories: list[dict[str, Any]]
    avg_trajectory_age: float


class ReplayBufferState(TypedDict):
    """Checkpoint payload for ``ReplayBuffer.state_dict``.

    ``task_names`` is parallel to ``trajectories`` and must survive the
    round-trip: without it the per-task release gate silently degrades to
    first-come-first-served after a resume.
    """

    trajectories: list[dict[str, Any]]
    trajectory_versions: list[int]
    target_weight_versions: list[int]
    task_names: list[TaskName]
    last_target_weight_already_generated: int
    max_size: int


class ReplayBufferProtocol(Protocol):  # pragma: no cover
    """Interface for the replay buffer used in async RL training."""

    def add(
        self,
        trajectory: dict[str, Any],
        weight_version: int,
        target_weight_version: int,
        task_name: TaskName = ...,
    ) -> str:
        """Add a per-prompt trajectory group with metadata.

        Args:
            trajectory: data dict
            weight_version: version of the model weights used for generation
            target_weight_version: version of the model weights this trajectory is intended for training
            task_name: source dataset task; only consulted when a quota is in use
        """
        ...

    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
        quota: TaskQuota | None = None,
    ) -> Optional[SampledBatch]:
        """Sample per-prompt trajectory groups intended for the current training step.

        Only returns trajectories with target_weight_version == current_weight_version.
        If insufficient trajectories are available, returns None to stall training
        until the remaining trajectories are generated. This ensures no trajectory
        loses its last chance to be used for its intended training step.

        Args:
            num_prompt_groups: Groups needed for one training step.
            current_weight_version: Training step being assembled.
            max_age_steps: Age window for usable trajectories.
            quota: Per-task group counts; when given, every task must fill its
                slots before the batch is released.

        Returns:
            The sampled batch, or None if insufficient data.
        """
        ...

    def size(self) -> int:
        """Return current buffer size."""
        ...

    def clear(self) -> None:
        """Clear the buffer."""
        ...

    def state_dict(self) -> ReplayBufferState:
        """Return serializable state for checkpointing."""
        ...

    def load_state_dict(
        self,
        state: dict[str, Any],
        num_prompts_per_step: int | None = None,
        current_training_step: int | None = None,
        max_age_steps: int | None = None,
    ) -> None:
        """Restore state produced by ``state_dict``."""
        ...

    def save_to_path(self, path: str) -> int:
        """Serialize state directly from the replay actor."""
        ...

    def load_from_path(
        self,
        path: str,
        num_prompts_per_step: int | None = None,
        current_training_step: int | None = None,
        max_age_steps: int | None = None,
    ) -> dict[str, int]:
        """Restore state directly in the replay actor."""
        ...

    def get_trajectories_needed(
        self,
        target_step: int,
        num_prompts_per_step: int,
        max_age_steps: int | None = None,
    ) -> int:
        """Return additional trajectories needed for ``target_step``."""
        ...

    def get_task_deficits(
        self,
        target_step: int,
        quota: TaskQuota,
        max_age_steps: int | None = None,
    ) -> TaskDeficits:
        """Return the per-task shortfall for ``target_step``."""
        ...

    def has_complete_batch(
        self,
        target_step: int,
        num_prompts_per_step: int,
        max_age_steps: int | None = None,
        quota: TaskQuota | None = None,
    ) -> bool:
        """Return whether ``target_step`` has enough trajectories to train.

        With a quota, every task must have filled its slots.
        """
        ...
