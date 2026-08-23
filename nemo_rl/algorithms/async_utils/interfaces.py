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

from typing import Any, Optional, Protocol


class ReplayBufferProtocol(Protocol):  # pragma: no cover
    """Interface for the replay buffer used in async RL training."""

    def add(
        self,
        trajectory: dict[str, Any],
        weight_version: int,
        target_weight_version: int | None = None,
        *,
        reserved: bool = False,
    ) -> str:
        """Commit one complete prompt group to the FIFO tail."""
        ...

    def reserve(self, num_prompt_groups: int) -> int:
        """Reserve bounded queue capacity for admitted rollout work."""
        ...

    def release_reserved(self, num_prompt_groups: int) -> None:
        """Release reservations for rollout work that did not complete."""
        ...

    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        """Consume one complete prompt-group batch in FIFO order.

        Returns:
            Dictionary with 'trajectories' and 'avg_trajectory_age' keys, or None if insufficient data
        """
        ...

    def size(self) -> int:
        """Return current buffer size."""
        ...

    def clear(self) -> None:
        """Clear the buffer."""
        ...

    def state_dict(self) -> dict[str, Any]:
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

    def get_trajectories_needed(
        self,
        target_step: int,
        num_prompts_per_step: int,
        max_age_steps: int | None = None,
    ) -> int:
        """Return additional trajectories needed for ``target_step``."""
        ...

    def has_complete_batch(
        self,
        target_step: int,
        num_prompts_per_step: int,
        max_age_steps: int | None = None,
    ) -> bool:
        """Return whether ``target_step`` has enough trajectories to train."""
        ...
