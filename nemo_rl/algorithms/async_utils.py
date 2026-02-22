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

import threading as _threading
import time
from typing import Any, Optional

import ray
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface

TokenizerType = PreTrainedTokenizerBase


@ray.remote  # pragma: no cover
class ReplayBuffer:
    """Replay buffer storing per-prompt groups.

    A single entry corresponds to 1 prompt repeated by
    grpo.num_generations_per_prompt (required to compute per-prompt advantages).
    """

    def __init__(self, max_size: int):
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self.max_size = max_size
        self.trajectories = []  # List[dict[str, Any]]
        # If trajectory_version is 1 and target_weight_version is 4 it means that weight version 1 was used for generating a trajectory and this trajectory will be used for training when weight version is 4.
        self.trajectory_versions = []  # it is the weight-version used for generation of a trajectory
        self.target_weight_versions = []  # it is the weight-version of the trainer where this trajectory will be used.

        self.last_target_weight_already_generated = -1
        self._lock = _threading.Lock()

    def push_with_wait_signal(
        self,
        trajectory: dict[str, Any],
        weight_version: int,
        target_weight_version: int,
    ) -> str:
        """Add a per-prompt trajectory group with metadata.

        Args:
            trajectory: data dict
            weight_version: version of the model weights used for generation
            target_weight_version: version of the model weights this trajectory is intended for training
        """
        with self._lock:
            if len(self.trajectories) >= self.max_size:
                return "full"

            print("🔍 ReplayBuffer.push_with_wait_signal: Adding trajectory")
            self.trajectories.append(trajectory)
            self.trajectory_versions.append(weight_version)
            self.target_weight_versions.append(target_weight_version)
            # NOTE: We intentionally do NOT advance last_target_weight_already_generated here.
            # Advancing it on buffering can jump ahead due to out-of-order arrivals and skip
            # earlier incomplete targets. During normal training it advances when training
            # CONSUMES a batch (in sample()); checkpoint restore may reset it.
            print(
                f"ReplayBuffer state: {len(self.trajectories)} groups, versions={self.trajectory_versions}, targets={self.target_weight_versions}, last_target_weight_already_generated={self.last_target_weight_already_generated}"
            )
            return "success"

    def get_debug_info(self) -> dict:
        """Get debug information about buffer state."""
        return {
            "total_trajectories": len(self.trajectories),
            "trajectory_versions": self.trajectory_versions,
            "target_weight_versions": self.target_weight_versions,
            "max_size": self.max_size,
        }

    def get_last_target_weight_already_generated(self) -> int:
        with self._lock:
            return self.last_target_weight_already_generated

    def get_existing_target_weights(self) -> set[int]:
        """Get set of target weight versions that already have trajectories."""
        with self._lock:
            return set(self.target_weight_versions)

    def sample(
        self,
        num_prompt_groups: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[dict[str, Any]]:
        """Sample per-prompt trajectory groups intended for the current training step.

        Only returns trajectories with target_weight_version == current_weight_version.
        If insufficient trajectories are available, returns None to stall training
        until the remaining trajectories are generated. This ensures no trajectory
        loses its last chance to be used for its intended training step.

        Returns:
            Dictionary with 'trajectories' and 'avg_trajectory_age' keys, or None if insufficient data
        """
        with self._lock:
            if not self.trajectories:
                return None

            total_trajectories = len(self.trajectories)
            print("🔍 ReplayBuffer sampling debug:")
            print(f"   {current_weight_version=}, {max_age_steps=}")
            print(f"   {self.trajectory_versions=}")

            # For debugging: check for unexpected old trajectories
            from collections import Counter

            version_counts = Counter(self.trajectory_versions)
            print(f"   {version_counts=}")

            # Compute minimum valid version based on age window
            # max_age_steps=1 means trajectories from the last 1 step are valid
            min_valid_version = max(0, current_weight_version - max_age_steps)
            print(f"   {min_valid_version=}")

            # Evict old trajectories that are beyond the age window
            # This can happen after checkpoint restore when old trajectories remain
            old_indices = [
                i for i, v in enumerate(self.trajectory_versions)
                if v < min_valid_version
            ]
            if old_indices:
                print(
                    f"   🗑️ Evicting {len(old_indices)} stale trajectories "
                    f"(version < {min_valid_version})"
                )
                # Keep only non-old trajectories
                indices_to_keep = [
                    i for i in range(len(self.trajectories))
                    if i not in set(old_indices)
                ]
                self.trajectories = [self.trajectories[i] for i in indices_to_keep]
                self.trajectory_versions = [self.trajectory_versions[i] for i in indices_to_keep]
                self.target_weight_versions = [self.target_weight_versions[i] for i in indices_to_keep]

            # Filter for valid trajectories without modifying the buffer
            valid_indices = [
                i
                for i, v in enumerate(self.trajectory_versions)
                if min_valid_version <= v <= current_weight_version
            ]
            print(
                f"   valid_indices: {len(valid_indices)}/{total_trajectories} trajectories within age window"
            )
            if not valid_indices:
                print("No trajectories available for sampling.")
                return None

            # Enforce exact number of groups if available; otherwise, signal to wait
            if len(valid_indices) < num_prompt_groups:
                print(
                    f"Insufficient valid groups: have {len(valid_indices)}, need {num_prompt_groups}. Waiting for buffer to fill."
                )
                return None

            # Only select trajectories intended for the current training step
            # This ensures no trajectory loses its "last chance" to be used for its intended step
            intended_indices = [
                i
                for i in valid_indices
                if self.target_weight_versions[i] == current_weight_version
            ]

            print(
                f"   🎯 Found {len(intended_indices)} trajectories intended for current step {current_weight_version}"
            )

            # Stall training if we don't have enough trajectories intended for this step
            if len(intended_indices) < num_prompt_groups:
                print(
                    f"   ⏸️ STALLING: Need {num_prompt_groups} trajectories for step {current_weight_version}, but only {len(intended_indices)} are ready"
                )
                print(
                    f"   ⏸️ Training will wait for remaining {num_prompt_groups - len(intended_indices)} trajectories to be generated"
                )
                return None

            # Select exactly the trajectories intended for this step (FIFO within same target)
            selected: list[int] = intended_indices[:num_prompt_groups]
            print(
                f"   ✅ Selected {len(selected)} trajectories all intended for step {current_weight_version}"
            )

            from collections import Counter

            sampled_weights = [self.trajectory_versions[i] for i in selected]
            avg_trajectory_age = current_weight_version - sum(sampled_weights) / len(
                sampled_weights
            )
            print(
                f"✅ Selected counts by generation weight-version: {Counter(sampled_weights)}"
            )
            print(f"📊 Average trajectory age: {avg_trajectory_age:.2f} steps")
            print(
                f"🎯 All selected trajectories target step {current_weight_version} (100% target match)"
            )

            sampled_items = [self.trajectories[i] for i in selected]

            # Remove selected items in reverse order to maintain correct indices
            for idx in sorted(selected, reverse=True):
                self.trajectory_versions.pop(idx)
                self.target_weight_versions.pop(idx)
                self.trajectories.pop(idx)

            # Advance last_target_weight_already_generated ONLY when training consumes a batch.
            # This prevents the collector from generating more trajectories for this target,
            # avoiding over-generation when slow workers arrive after consumption.
            old_last_target = self.last_target_weight_already_generated
            self.last_target_weight_already_generated = max(
                self.last_target_weight_already_generated, current_weight_version
            )
            if self.last_target_weight_already_generated > old_last_target:
                print(
                    f"🔒 Advanced last_target_weight_already_generated: {old_last_target} → {self.last_target_weight_already_generated} (training consumed batch for step {current_weight_version})"
                )

            print(
                f"🗑️ Consumed and removed {len(selected)} groups from buffer, old buffer size: {total_trajectories}, new buffer size: {len(self.trajectories)}, new target weight versions {self.target_weight_versions}"
            )

            return {
                "trajectories": sampled_items,
                "avg_trajectory_age": avg_trajectory_age,
            }

    def size(self) -> int:
        """Return current buffer size."""
        with self._lock:
            return len(self.trajectories)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self.trajectories.clear()
            self.trajectory_versions.clear()
            self.target_weight_versions.clear()

    def state_dict(self) -> dict:
        """Get serializable state for checkpointing.

        Returns a dictionary containing all state needed to restore the buffer,
        enabling resume from checkpoint without losing buffered trajectories.
        """
        with self._lock:
            return {
                "trajectories": list(self.trajectories),  # Copy to avoid mutation
                "trajectory_versions": list(self.trajectory_versions),
                "target_weight_versions": list(self.target_weight_versions),
                "last_target_weight_already_generated": self.last_target_weight_already_generated,
                "max_size": self.max_size,
            }

    def load_state_dict(
        self,
        state: dict,
        num_prompts_per_step: int | None = None,
        current_training_step: int | None = None,
    ) -> None:
        """Restore state from checkpoint.

        Args:
            state: Dictionary from state_dict() containing buffer state.
            num_prompts_per_step: If provided, validates that each target step has
                complete batches. Incomplete target steps will be removed.
            current_training_step: The training step we're resuming from. If provided,
                ensures the buffer is ready for this step by:
                - Removing trajectories for steps we've already passed
                - Removing incomplete batches for the current step
                - Adjusting last_target_weight_already_generated so collector
                  will regenerate any missing trajectories for current step

        Note:
            The max_size in the checkpoint is stored for validation but the
            current instance's max_size takes precedence (allows config changes).
        """
        with self._lock:
            # Validate checkpoint has expected keys
            required_keys = {
                "trajectories",
                "trajectory_versions",
                "target_weight_versions",
                "last_target_weight_already_generated",
            }
            missing_keys = required_keys - set(state.keys())
            if missing_keys:
                raise ValueError(f"Checkpoint missing required keys: {missing_keys}")

            # Log if max_size differs (user changed config)
            if "max_size" in state and state["max_size"] != self.max_size:
                print(
                    f"⚠️ ReplayBuffer max_size changed: checkpoint={state['max_size']}, "
                    f"current={self.max_size}. Using current config value."
                )

            # Restore state
            self.trajectories = list(state["trajectories"])
            self.trajectory_versions = list(state["trajectory_versions"])
            self.target_weight_versions = list(state["target_weight_versions"])
            self.last_target_weight_already_generated = state[
                "last_target_weight_already_generated"
            ]

            # Truncate if restored state exceeds current max_size
            if len(self.trajectories) > self.max_size:
                print(
                    f"⚠️ Truncating restored buffer from {len(self.trajectories)} "
                    f"to max_size={self.max_size}"
                )
                self.trajectories = self.trajectories[: self.max_size]
                self.trajectory_versions = self.trajectory_versions[: self.max_size]
                self.target_weight_versions = self.target_weight_versions[: self.max_size]

            # Prepare buffer for resume at current_training_step
            if (
                current_training_step is not None
                and num_prompts_per_step is not None
                and len(self.trajectories) > 0
            ):
                self._prepare_for_training_step(
                    current_training_step, num_prompts_per_step
                )
            elif num_prompts_per_step is not None and len(self.trajectories) > 0:
                # Fallback to just removing incomplete steps
                self._remove_incomplete_target_steps(num_prompts_per_step)

            print(
                f"✅ ReplayBuffer restored: {len(self.trajectories)} trajectories, "
                f"last_target_weight_already_generated={self.last_target_weight_already_generated}"
            )

    def _prepare_for_training_step(
        self, current_step: int, num_prompts_per_step: int
    ) -> None:
        """Prepare buffer for resuming training at a specific step.

        This method ensures training can resume at current_step by:
        1. Removing trajectories for steps we've already passed (target < current_step)
        2. Identifying incomplete steps and setting up for gap-filling
        3. Adjusting last_target_weight_already_generated so collector fills gaps

        Incomplete steps are kept, the collector will fill in missing trajectories.
        """
        from collections import Counter

        print(f"   🔧 Preparing buffer for training step {current_step}...")

        # Step 1: Remove trajectories for steps we've already passed
        original_count = len(self.trajectories)
        indices_to_keep = [
            i for i, t in enumerate(self.target_weight_versions)
            if t >= current_step
        ]

        if len(indices_to_keep) < original_count:
            removed_past = original_count - len(indices_to_keep)
            self.trajectories = [self.trajectories[i] for i in indices_to_keep]
            self.trajectory_versions = [self.trajectory_versions[i] for i in indices_to_keep]
            self.target_weight_versions = [self.target_weight_versions[i] for i in indices_to_keep]
            print(f"   🗑️ Removed {removed_past} trajectories for past steps (target < {current_step})")

        if not self.trajectories:
            print(f"   ⚠️ No trajectories for step {current_step} or later")
            # Reset counter so collector will generate for current step
            self.last_target_weight_already_generated = current_step - 1
            print(
                f"   🔄 Reset last_target_weight_already_generated to {current_step - 1} "
                f"(collector will generate for step {current_step})"
            )
            return

        # Step 2: Identify complete and incomplete steps
        target_counts = Counter(self.target_weight_versions)
        complete_targets = {t for t, c in target_counts.items() if c >= num_prompts_per_step}
        incomplete_targets = {t for t, c in target_counts.items() if c < num_prompts_per_step}

        print(f"   📊 Buffer state after removing past steps:")
        print(f"      Complete targets: {sorted(complete_targets) if complete_targets else 'none'}")
        if incomplete_targets:
            for t in sorted(incomplete_targets):
                print(f"      Incomplete target {t}: {target_counts[t]}/{num_prompts_per_step}")

        # Step 3: Set last_target_weight_already_generated to allow filling gaps
        # We set it to current_step - 1 so collector will evaluate all targets from current_step onwards
        # The collector will then check get_trajectories_needed() to see which targets need filling
        self.last_target_weight_already_generated = current_step - 1
        print(
            f"   🔄 Set last_target_weight_already_generated to {current_step - 1} "
            f"(collector will fill gaps starting from step {current_step})"
        )

    def get_trajectories_needed(self, target_step: int, num_prompts_per_step: int) -> int:
        """Return number of additional trajectories needed for a target step.

        Args:
            target_step: The target weight version to check
            num_prompts_per_step: Required number of trajectories per step

        Returns:
            Number of additional trajectories needed (0 if complete or over-complete)
        """
        with self._lock:
            current_count = sum(
                1 for t in self.target_weight_versions if t == target_step
            )
            return max(0, num_prompts_per_step - current_count)

    def has_complete_batch(self, target_step: int, num_prompts_per_step: int) -> bool:
        """Check if a target step has a complete batch of trajectories.

        Args:
            target_step: The target weight version to check
            num_prompts_per_step: Required number of trajectories per step

        Returns:
            True if target_step has >= num_prompts_per_step trajectories
        """
        with self._lock:
            current_count = sum(
                1 for t in self.target_weight_versions if t == target_step
            )
            return current_count >= num_prompts_per_step

    def _remove_incomplete_target_steps(self, num_prompts_per_step: int) -> None:
        """Remove trajectories for target steps that don't have complete batches.

        This prevents training stalls on resume when a checkpoint was saved
        mid-batch (e.g., 127/128 trajectories for a target step).

        Must be called while holding self._lock.
        """
        from collections import Counter

        # Count trajectories per target step
        target_counts = Counter(self.target_weight_versions)

        # Find incomplete target steps
        incomplete_targets = {
            target for target, count in target_counts.items()
            if count < num_prompts_per_step
        }

        if not incomplete_targets:
            print(
                f"   ✓ All target steps have complete batches ({num_prompts_per_step} each)"
            )
            return

        print(
            f"   ⚠️ Found {len(incomplete_targets)} incomplete target steps: "
            f"{sorted(incomplete_targets)}"
        )
        for target in sorted(incomplete_targets):
            print(
                f"      - Target {target}: {target_counts[target]}/{num_prompts_per_step} trajectories"
            )

        # Remove incomplete target step trajectories
        original_count = len(self.trajectories)
        indices_to_keep = [
            i for i, t in enumerate(self.target_weight_versions)
            if t not in incomplete_targets
        ]

        self.trajectories = [self.trajectories[i] for i in indices_to_keep]
        self.trajectory_versions = [self.trajectory_versions[i] for i in indices_to_keep]
        self.target_weight_versions = [self.target_weight_versions[i] for i in indices_to_keep]

        removed_count = original_count - len(self.trajectories)
        print(
            f"   🗑️ Removed {removed_count} trajectories from incomplete target steps"
        )

        # Adjust last_target_weight_already_generated to highest COMPLETE target
        if self.target_weight_versions:
            complete_targets = set(self.target_weight_versions)
            max_complete = max(complete_targets)
            if max_complete < self.last_target_weight_already_generated:
                print(
                    f"   🔄 Adjusting last_target_weight_already_generated: "
                    f"{self.last_target_weight_already_generated} → {max_complete}"
                )
                self.last_target_weight_already_generated = max_complete
        else:
            # No complete targets left, reset counter
            print(
                f"   🔄 No complete targets left, resetting last_target_weight_already_generated to -1"
            )
            self.last_target_weight_already_generated = -1


@ray.remote  # pragma: no cover
class AsyncTrajectoryCollector:
    """Collects trajectories asynchronously and adds them to replay buffer."""

    def __init__(
        self,
        policy_generation: GenerationInterface,
        tokenizer: TokenizerType,
        task_to_env: dict[str, EnvironmentInterface],
        master_config: MasterConfig,
        replay_buffer: Any,
        start_step: int = 0,
    ):
        self.policy_generation = policy_generation
        self.tokenizer = tokenizer
        self.task_to_env = task_to_env
        self.master_config = master_config
        self.replay_buffer = replay_buffer
        self.running = False

        self._pg_lock: _threading.Lock = _threading.Lock()

        # Event for manual pause/resume control
        self._manual_pause_cleared = _threading.Event()
        self._manual_pause_cleared.set()

        self._refit_pause_cleared = _threading.Event()
        self._refit_pause_cleared.set()  # Start in cleared state

        self.current_weight_version: int = start_step
        self.initial_weight_version: int = start_step

        # Track when generation limits cause collection to pause
        self._last_limit_warning_version = None

        # Event to signal when generation limits are cleared (more efficient than polling)
        self._generation_limit_cleared = _threading.Event()
        self._generation_limit_cleared.set()  # Start in cleared state

        # Track threads
        self._inflight_threads: set[_threading.Thread] = set()
        self._threads_lock: _threading.Lock = _threading.Lock()

        # Limit in-flight generator requests to num_prompts_per_step * max_trajectory_age_steps
        # This value limits the parallelism of the generation requests.
        max_inflight = (
            int(self.master_config["grpo"]["num_prompts_per_step"])
            * int(self.master_config["grpo"]["async_grpo"]["max_trajectory_age_steps"])
        ) or 1
        self._inflight_sema = _threading.Semaphore(max_inflight)

        # Simple lock to prevent race conditions when checking/spawning workers
        self._generation_check_lock: _threading.Lock = _threading.Lock()
        # Track which target weights are currently being generated (globally)
        self._generating_targets: set[int] = set()

        # Debug counters: track spawned vs successfully buffered vs completed per target weight
        self._spawned_per_target: dict[int, int] = {}
        self._buffered_per_target: dict[int, int] = {}
        self._completed_per_target: dict[int, int] = {}  # Prompt-group completions (counts once per prompt_idx; success + failure)
        self._counter_lock: _threading.Lock = _threading.Lock()

    def _calculate_target_weights(self, generation_weight_version: int) -> list[int]:
        """Calculate target weight versions for given generation weight version.

        The list of versions returned enumerate the possible version a generation
        server can target. These versions are looped over to see what training
        step they can target. If all target versions are exhausted, this generation
        server will remain idle until the next weight update.

        Example:
        generation_weight_version = 10
        max_trajectory_age_steps = 4

        Returns:
            [11, 12, 13, 14]  # Meaning this generation server can create trajectories for training step 11, 12, 13, 14
        """
        # Read async config strictly from grpo.async_grpo
        async_cfg = self.master_config.get("grpo", {}).get("async_grpo", {})
        max_trajectory_age = async_cfg["max_trajectory_age_steps"]
        if generation_weight_version == self.initial_weight_version:
            return [
                i
                for i in range(
                    self.initial_weight_version,
                    self.initial_weight_version + max_trajectory_age + 1,
                )
            ]

        return [generation_weight_version + i for i in range(1, max_trajectory_age + 1)]

    def _get_next_target_for_generation(
        self, generation_weight_version: int
    ) -> Optional[int]:
        """Get the next target weight that needs generation (if any).

        Checks all targets in the valid range (current weight version + 1 to max_age ahead)
        and returns the first one that needs trajectories and isn't already being generated.

        This approach checks:
        1. last_target_weight_already_generated: Skip targets that training has already consumed
           (prevents over-generation for consumed targets with slow workers still arriving)
        2. _generating_targets: Prevents duplicate concurrent generation for the same target
        3. get_trajectories_needed: Allows gap-filling for incomplete targets after checkpoint resume

        Does the following:
        - Normal generation: new targets need full batches
        - Gap-filling after resume: incomplete targets need partial batches
        - Skipping consumed targets: last_target_weight_already_generated check
        - Skipping complete targets: get_trajectories_needed returns 0

        Note:
            During normal training, last_target_weight_already_generated advances when training
            CONSUMES a batch (in ReplayBuffer.sample()); checkpoint restore may reset it. This
            prevents the race condition where a single trajectory for a higher target could skip
            earlier incomplete targets, while still preventing over-generation for consumed targets.
        """
        num_prompts = int(self.master_config["grpo"]["num_prompts_per_step"])
        max_trajectory_age = int(
            self.master_config["grpo"]["async_grpo"]["max_trajectory_age_steps"]
        )

        # Special handling for initial step: include current step as a target
        if generation_weight_version == self.initial_weight_version:
            target_start = generation_weight_version
            target_end = generation_weight_version + max_trajectory_age + 1
        else:
            # Check all targets from current+1 to current+max_age
            target_start = generation_weight_version + 1
            target_end = generation_weight_version + max_trajectory_age + 1

        # Get the last target that training has consumed - skip anything at or below this
        last_consumed_target = ray.get(
            self.replay_buffer.get_last_target_weight_already_generated.remote()
        )

        with self._generation_check_lock:
            for target_weight in range(target_start, target_end):
                # Skip if training has already consumed this target's batch
                if target_weight <= last_consumed_target:
                    continue

                # Skip if target is already being generated
                if target_weight in self._generating_targets:
                    continue

                # Check if this target needs more trajectories (for gap-filling scenarios)
                trajectories_needed = ray.get(
                    self.replay_buffer.get_trajectories_needed.remote(
                        target_weight, num_prompts
                    )
                )

                if trajectories_needed > 0:
                    self._generating_targets.add(target_weight)
                    if trajectories_needed < num_prompts:
                        print(
                            f"🎯 Reserved target weight {target_weight} for gap-filling "
                            f"(need {trajectories_needed}/{num_prompts} more trajectories)"
                        )
                    else:
                        print(f"🎯 Reserved target weight {target_weight} for generation")
                    return target_weight

        return None

    def set_weight_version(self, version: int) -> None:
        self.current_weight_version = version

        # Resume collection if it was paused due to generation limits
        was_paused = not self._generation_limit_cleared.is_set()
        if was_paused:
            self._generation_limit_cleared.set()  # Signal that collection can resume
            print(f"🔄 Updated weight version to {version}, resuming collection")
        else:
            print(f"🔄 Updated weight version to {version}")

    def _should_pause_for_generation_limits(self) -> bool:
        """Check if collection should be paused due to generation limits.

        Pauses when all targets in the valid range either:
        - Have already been consumed by training (target <= last_target_weight_already_generated)
        - Are already being generated (in _generating_targets)
        - Don't need more trajectories (get_trajectories_needed returns 0)
        """
        try:
            num_prompts = int(self.master_config["grpo"]["num_prompts_per_step"])
            max_trajectory_age = int(
                self.master_config["grpo"]["async_grpo"]["max_trajectory_age_steps"]
            )

            # Special handling for initial step: include current step as a target
            if self.current_weight_version == self.initial_weight_version:
                target_start = self.current_weight_version
                target_end = self.current_weight_version + max_trajectory_age + 1
            else:
                # Check all targets from current+1 to current+max_age
                target_start = self.current_weight_version + 1
                target_end = self.current_weight_version + max_trajectory_age + 1

            # Get the last target that training has consumed
            last_consumed_target = ray.get(
                self.replay_buffer.get_last_target_weight_already_generated.remote()
            )

            with self._generation_check_lock:
                for target_weight in range(target_start, target_end):
                    # Skip if training has already consumed this target
                    if target_weight <= last_consumed_target:
                        continue

                    # Skip if already generating
                    if target_weight in self._generating_targets:
                        continue

                    # Check if this target needs more trajectories
                    trajectories_needed = ray.get(
                        self.replay_buffer.get_trajectories_needed.remote(
                            target_weight, num_prompts
                        )
                    )
                    if trajectories_needed > 0:
                        return False  # Found a target that needs generation

            print(
                f"⏸️ All targets [{target_start}, {target_end}) complete or in progress, pausing"
            )
            return True
        except Exception:
            return False

    def start_collection(self, dataloader: StatefulDataLoader) -> None:
        """Start collecting trajectories from dataloader."""
        self.running = True
        self.dataloader = dataloader

        print("Started continuous trajectory collection")

        self.collection_thread = _threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()

        print("Collection thread started, start_collection returning")

    def _collection_loop(self):
        """Run the collection loop in background thread."""
        try:
            for batch in self.dataloader:
                if not self.running:
                    break

                # Check if manually paused and wait
                if not self._manual_pause_cleared.is_set() and self.running:
                    self._manual_pause_cleared.wait()

                # Check if refit is in progress and wait
                if not self._refit_pause_cleared.is_set() and self.running:
                    print("⏸️ Pausing collection for refit...")
                    self._refit_pause_cleared.wait()
                    print("▶️ Refit completed, resuming collection")

                # Check if generation limits require pausing collection
                if self._should_pause_for_generation_limits() and self.running:
                    # Only log warning once per weight version
                    if self._last_limit_warning_version != self.current_weight_version:
                        async_cfg = self.master_config.get("grpo", {}).get(
                            "async_grpo", {}
                        )
                        max_trajectory_age = async_cfg["max_trajectory_age_steps"]

                        # Show the actual targets being checked (consistent with logic above)
                        if self.current_weight_version == self.initial_weight_version:
                            target_weights = list(range(
                                self.current_weight_version,
                                self.current_weight_version + max_trajectory_age + 1
                            ))
                        else:
                            target_weights = list(range(
                                self.current_weight_version + 1,
                                self.current_weight_version + max_trajectory_age + 1
                            ))

                        print(
                            f"⏸️ Pausing collection: all target weights {target_weights} for weight version {self.current_weight_version} "
                            f"already exist in buffer. Waiting for weight update..."
                        )
                        self._last_limit_warning_version = self.current_weight_version

                        self._generation_limit_cleared.clear()  # Clear the event to pause

                    # Efficiently wait for generation limits to be cleared (no polling!)
                    self._generation_limit_cleared.wait()

                    # Double-check we're still running after being woken up
                    if not self.running:
                        break

                if not self.running:
                    break

                self._process_batch(batch)

        except Exception as e:
            print(f"❌ Error in trajectory collection: {e}")
            import traceback

            traceback.print_exc()
        finally:
            if self._inflight_threads:
                print(
                    f"⏳ Waiting for {len(self._inflight_threads)} pending gap-fill workers to complete..."
                )
                self.wait_for_pending_generations()
            self.running = False
            print("🛑 Trajectory collection stopped")

    def _process_batch(self, batch: BatchedDataDict[DatumSpec]) -> None:
        """Process a single batch and generate for one target weight."""
        try:
            generation_weight_version = self.current_weight_version
            num_generations = self.master_config["grpo"]["num_generations_per_prompt"]
            num_prompts_in_batch = batch.size
            num_prompts_per_step = int(self.master_config["grpo"]["num_prompts_per_step"])

            # Get the next target weight that needs generation
            target_weight = self._get_next_target_for_generation(
                generation_weight_version
            )

            if target_weight is None:
                print(
                    f"🔄 No targets need generation for weight {generation_weight_version}"
                )
                return

            # Check how many trajectories are actually needed for this target
            trajectories_needed = ray.get(
                self.replay_buffer.get_trajectories_needed.remote(
                    target_weight, num_prompts_per_step
                )
            )

            # Limit generation to what's needed (for gap-filling scenarios)
            num_prompts_to_generate = min(num_prompts_in_batch, trajectories_needed)

            if num_prompts_to_generate == 0:
                print(
                    f"🔄 Target {target_weight} already has enough trajectories, skipping"
                )
                with self._generation_check_lock:
                    self._generating_targets.discard(target_weight)
                return

            if num_prompts_to_generate < num_prompts_in_batch:
                print(
                    f"🎯 Gap-filling for target weight {target_weight}: "
                    f"generating {num_prompts_to_generate}/{num_prompts_in_batch} prompts "
                    f"(need {trajectories_needed} more trajectories)"
                )
            else:
                print(
                    f"🎯 Generating for target weight {target_weight} from generation_weight_version {generation_weight_version}"
                )

            # Generate for needed prompts in this batch for the target weight
            for prompt_idx in range(num_prompts_to_generate):
                # Wait for refit to complete if in progress
                if not self._refit_pause_cleared.is_set() and self.running:
                    with self._threads_lock:
                        active_threads = len(self._inflight_threads)
                    print(
                        f"⏸️ Waiting for refit to complete before starting new generation ({active_threads} threads still active)"
                    )
                    print(
                        "   Note: With vLLM V1 async engine, active threads can complete during weight update"
                    )
                    self._refit_pause_cleared.wait()

                    # After refit finishes if weight version has updated, reflect that in the new trajectories
                    generation_weight_version = self.current_weight_version

                single_prompt_batch = batch.slice(prompt_idx, prompt_idx + 1)
                repeated_batch = single_prompt_batch.repeat_interleave(num_generations)

                self._inflight_sema.acquire()
                worker = _threading.Thread(
                    target=self._run_prompt_group_worker,
                    args=(
                        repeated_batch,
                        generation_weight_version,
                        target_weight,
                        prompt_idx,
                    ),
                    daemon=True,
                )
                with self._threads_lock:
                    self._inflight_threads.add(worker)
                worker.start()

                # Debug: track spawned workers per target weight
                with self._counter_lock:
                    self._spawned_per_target[target_weight] = (
                        self._spawned_per_target.get(target_weight, 0) + 1
                    )
                    if prompt_idx == num_prompts_to_generate - 1:  # Last prompt being generated
                        print(
                            f"📊 DEBUG: Spawned {self._spawned_per_target.get(target_weight, 0)} workers for target_weight={target_weight}"
                        )

            self._cleanup_finished_threads()

        except Exception as e:
            print(f"❌ Error processing batch: {e}")
            import traceback

            traceback.print_exc()

    def get_weight_version(self) -> int:
        return self.current_weight_version

    def pause(self) -> None:
        """Pause trajectory collection."""
        self._manual_pause_cleared.clear()  # Signal collection to pause
        print("Trajectory collection paused")

    def resume(self) -> None:
        """Resume trajectory collection."""
        self._manual_pause_cleared.set()  # Signal collection to resume
        print("Trajectory collection resumed")

    def prepare_for_refit(self) -> None:
        """Pause new generation starts and optionally wait for pending generations.

        For vLLM V1 async engine, leverages in-flight weight updates via collective_rpc,
        allowing ongoing generations to continue with their current KV caches while
        weights are updated. This significantly improves async performance.

        For non-async engines, waits for all pending generations to complete before refit.
        """
        start_time = time.time()
        print("🔄 Preparing for refit: pausing new generations...")

        # Pause new generation starts
        self._refit_pause_cleared.clear()
        print("⏸️ New generation starts paused")

        # Check if we're using vLLM async engine
        vllm_cfg = (
            self.master_config.get("policy", {})
            .get("generation", {})
            .get("vllm_cfg", {})
        )
        is_async_engine = vllm_cfg.get("async_engine", False)
        in_flight_weight_updates = (
            self.master_config.get("grpo", {})
            .get("async_grpo", {})
            .get("in_flight_weight_updates", False)
        )

        if is_async_engine and in_flight_weight_updates:
            # vLLM V1 async engine supports in-flight weight updates
            # Ongoing generations will continue with their current KV caches
            # New generations (after weight update) will use the updated weights
            print(
                "🚀 Using vLLM V1 in-flight weight update - skipping wait for pending generations"
            )
            print(
                f"   {len(self._inflight_threads)} ongoing generations will complete with current weights"
            )
        else:
            # For non-async engines, wait for all pending generations to complete
            print(
                "⏸️ Non-async engine: waiting for all pending generations to complete..."
            )
            self.wait_for_pending_generations()

        elapsed = time.time() - start_time
        print(f"✅ Ready for refit (took {elapsed:.2f}s)")

    def resume_after_refit(self) -> None:
        """Resume new generation starts after refit is complete."""
        print("🔄 Resuming generation starts after refit")

        # Invalidate&recompute vLLM caches after the in-flight weight updates if
        # recompute_kv_cache_after_weight_updates is True (AREAL-style implementation).
        # Otherwise, keep using the stale KV caches (Magistral-style implementation).
        async_cfg = self.master_config.get("grpo", {}).get("async_grpo", {})
        if async_cfg.get("in_flight_weight_updates", False) and async_cfg.get(
            "recompute_kv_cache_after_weight_updates", False
        ):
            try:
                print("🔄 Invalidating vLLM prefix/KV caches after weight update")
                invalidated = self.policy_generation.invalidate_kv_cache()
                if invalidated:
                    print("✅ Invalidated vLLM prefix/KV caches after weight update")
                else:
                    print(
                        "⚠️ vLLM cache invalidation reported partial/unsuccessful on some workers"
                    )
            except Exception as e:
                print(f"⚠️ Failed to invalidate vLLM caches: {e}")

        self._refit_pause_cleared.set()

    def wait_for_pending_generations(self) -> None:
        """Wait for all in-flight generation threads to complete."""
        start_time = time.time()

        while True:
            with self._threads_lock:
                finished = {t for t in self._inflight_threads if not t.is_alive()}
                for t in finished:
                    self._inflight_threads.remove(t)

                pending_count = len(self._inflight_threads)

            if pending_count == 0:
                print("✅ All generation threads completed")
                break

            elapsed = time.time() - start_time
            print(
                f"⏳ Waiting for {pending_count} pending generation threads... ({elapsed:.1f}s elapsed)"
            )
            time.sleep(0.5)

    def get_dataloader_state(self) -> dict:
        """Get the current dataloader state for checkpointing."""
        if hasattr(self, "dataloader") and hasattr(self.dataloader, "state_dict"):
            return self.dataloader.state_dict()
        return {}

    def _cleanup_finished_threads(self) -> None:
        with self._threads_lock:
            finished = {t for t in self._inflight_threads if not t.is_alive()}
            for t in finished:
                self._inflight_threads.remove(t)

    def _run_prompt_group_worker(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        generation_weight_version: int,
        target_weight_version: int,
        prompt_idx: int,
        retry_count: int = 0,
    ) -> None:
        MAX_RETRIES = 3
        RETRY_DELAY_BASE = 1.0  # seconds
        _retry_spawned = False  # Flag to skip finally cleanup when retry is spawned

        try:
            # Import here to avoid circular dependency
            from nemo_rl.algorithms.grpo import _should_use_nemo_gym
            from nemo_rl.experience.rollouts import run_async_nemo_gym_rollout

            # Run rollout for this prompt group
            # Async engine supports concurrent generation; avoid locking
            # Check if we should use nemo_gym (similar to synchronous GRPO)
            if _should_use_nemo_gym(self.master_config):
                generation_config = self.master_config["policy"]["generation"]
                env_cfg = self.master_config.get("env") or {}
                nemo_gym_rollout_result = run_async_nemo_gym_rollout(
                    policy_generation=self.policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=self.tokenizer,
                    task_to_env=self.task_to_env,
                    max_seq_len=None,
                    generation_config=generation_config,
                    max_rollout_turns=None,
                    greedy=False,
                    master_config=self.master_config,
                )
                final_batch = nemo_gym_rollout_result.final_batch
                rollout_metrics = nemo_gym_rollout_result.rollout_metrics
            else:
                final_batch, rollout_metrics = run_async_multi_turn_rollout(
                    policy_generation=self.policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=self.tokenizer,
                    task_to_env=self.task_to_env,
                    max_seq_len=self.master_config["policy"][
                        "max_total_sequence_length"
                    ],
                    max_rollout_turns=self.master_config["grpo"]["max_rollout_turns"],
                    greedy=False,
                )

            # Move to CPU and push to buffer (avoid blocking on GC/push)
            final_batch_cpu = final_batch.to("cpu")
            del final_batch

            trajectory_group = {
                "batch": final_batch_cpu,
                "rollout_metrics": rollout_metrics,
                "timestamp": time.time(),
            }

            # Use exponential backoff when buffer is full
            try:
                backoff_delay = 0.01
                while self.running:
                    status = ray.get(
                        self.replay_buffer.push_with_wait_signal.remote(
                            trajectory_group,
                            generation_weight_version,
                            target_weight_version,
                        )
                    )
                    if status == "success":
                        # Debug: track successfully buffered workers per target weight
                        with self._counter_lock:
                            self._buffered_per_target[target_weight_version] = (
                                self._buffered_per_target.get(target_weight_version, 0) + 1
                            )
                            buffered_count = self._buffered_per_target[target_weight_version]
                            spawned_count = self._spawned_per_target.get(target_weight_version, 0)
                        print(
                            f"📦 Buffered per-prompt group (prompt_idx {prompt_idx}, target_weight {target_weight_version}) "
                            f"[{buffered_count}/{spawned_count} buffered for this target]"
                        )

                        # Reservation release is handled in finally block when ALL workers complete
                        break
                    elif status == "full":
                        # Exponential backoff up to 0.5 second
                        time.sleep(min(backoff_delay, 0.5))
                        backoff_delay *= 1.5
                    else:
                        # Unexpected status, wait briefly
                        time.sleep(0.01)
            except Exception as e:
                print(
                    f"❌ Failed to enqueue per-prompt group to buffer (prompt_idx={prompt_idx}, target_weight={target_weight_version}): {e}"
                )
                print(
                    f"   ⚠️ This trajectory will NOT be buffered - may cause stall if training expects it!"
                )
                import traceback

                traceback.print_exc()
        except Exception as e:
            print(
                f"❌ Error in prompt group worker (prompt_idx={prompt_idx}, target_weight={target_weight_version}, retry={retry_count}): {e}"
            )

            # Retry logic for transient errors (e.g., HTTP 500 from Penguin server)
            if retry_count < MAX_RETRIES and self.running:
                retry_delay = RETRY_DELAY_BASE * (2 ** retry_count)  # Exponential backoff
                print(
                    f"   🔄 Retrying in {retry_delay:.1f}s (attempt {retry_count + 1}/{MAX_RETRIES})..."
                )
                time.sleep(retry_delay)

                # Recursive retry - note: this reuses the current thread
                # Release semaphore before retry since we'll acquire it again
                try:
                    self._inflight_sema.release()
                except Exception:
                    pass

                # Remove from thread tracking before recursive call
                with self._threads_lock:
                    current = _threading.current_thread()
                    if current in self._inflight_threads:
                        self._inflight_threads.remove(current)

                # Retry by spawning a new worker thread
                self._inflight_sema.acquire()
                retry_worker = _threading.Thread(
                    target=self._run_prompt_group_worker,
                    args=(
                        repeated_batch,
                        generation_weight_version,
                        target_weight_version,
                        prompt_idx,
                        retry_count + 1,
                    ),
                    daemon=True,
                )
                with self._threads_lock:
                    self._inflight_threads.add(retry_worker)
                retry_worker.start()
                _retry_spawned = True  # Skip finally cleanup - retry worker handles it
                return  # Exit this worker, retry worker will handle cleanup
            else:
                print(
                    f"   ⚠️ Max retries ({MAX_RETRIES}) exceeded - trajectory will NOT be buffered!"
                )
                print(
                    f"   ⚠️ This may cause training to stall if it expects this trajectory."
                )
                import traceback
                traceback.print_exc()
        finally:
            # Skip cleanup if we spawned a retry worker (it will handle cleanup)
            if _retry_spawned:
                return

            # Track completed prompt groups (success or failure) and release reservation
            # only when ALL spawned prompt groups for this target have completed.
            # This prevents gap-filling from spawning duplicate workers while the
            # initial batch is still in-flight.
            with self._counter_lock:
                self._completed_per_target[target_weight_version] = (
                    self._completed_per_target.get(target_weight_version, 0) + 1
                )
                completed_count = self._completed_per_target[target_weight_version]
                spawned_count = self._spawned_per_target.get(target_weight_version, 0)
                buffered_count = self._buffered_per_target.get(target_weight_version, 0)
                should_release = (completed_count >= spawned_count)

            if should_release:
                with self._generation_check_lock:
                    if target_weight_version in self._generating_targets:
                        self._generating_targets.discard(target_weight_version)
                        print(
                            f"🧹 Released reservation for target weight {target_weight_version} "
                            f"(all {spawned_count} workers completed, {buffered_count} buffered)"
                        )

            # Detach thread record when finished
            with self._threads_lock:
                current = _threading.current_thread()
                if current in self._inflight_threads:
                    self._inflight_threads.remove(current)
            try:
                self._inflight_sema.release()
            except Exception:
                import traceback

                traceback.print_exc()
