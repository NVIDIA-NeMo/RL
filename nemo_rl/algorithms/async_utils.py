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
from collections import Counter
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

ASYNC_DEBUG_PREFIX = "[ASYNC_DEBUG]"


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
        self.failed_prompt_groups: list[dict[str, Any]] = []

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
            self.last_target_weight_already_generated = max(
                self.last_target_weight_already_generated, target_weight_version
            )
            print(
                f"ReplayBuffer state: {len(self.trajectories)} groups, versions={self.trajectory_versions}, targets={self.target_weight_versions}, last_target_weight_already_generated={self.last_target_weight_already_generated}"
            )
            return "success"

    def record_failed_prompt_group(
        self,
        weight_version: int,
        target_weight_version: int,
        prompt_idx: int,
        worker_id: int,
        reason: str,
    ) -> str:
        """Record a prompt group that failed before it produced a trainable batch."""
        with self._lock:
            failure_record = {
                "weight_version": weight_version,
                "target_weight_version": target_weight_version,
                "prompt_idx": prompt_idx,
                "worker_id": worker_id,
                "reason": reason,
                "timestamp": time.time(),
            }
            self.failed_prompt_groups.append(failure_record)
            self.last_target_weight_already_generated = max(
                self.last_target_weight_already_generated, target_weight_version
            )
            print(
                f"{ASYNC_DEBUG_PREFIX} recorded_failed_prompt_group "
                f"target={target_weight_version} generation_weight={weight_version} "
                f"prompt_idx={prompt_idx} worker_id={worker_id} reason={reason} "
                f"failed_target_counts={dict(Counter(r['target_weight_version'] for r in self.failed_prompt_groups))}"
            )
            return "success"

    def get_debug_info(self) -> dict:
        """Get debug information about buffer state."""
        with self._lock:
            failed_target_versions = [
                r["target_weight_version"] for r in self.failed_prompt_groups
            ]
            failed_generation_versions = [
                r["weight_version"] for r in self.failed_prompt_groups
            ]
            return {
                "total_trajectories": len(self.trajectories),
                "trajectory_versions": list(self.trajectory_versions),
                "target_weight_versions": list(self.target_weight_versions),
                "trajectory_version_counts": dict(Counter(self.trajectory_versions)),
                "target_weight_counts": dict(Counter(self.target_weight_versions)),
                "failed_prompt_groups": list(self.failed_prompt_groups),
                "failed_prompt_group_count": len(self.failed_prompt_groups),
                "failed_generation_version_counts": dict(
                    Counter(failed_generation_versions)
                ),
                "failed_target_weight_counts": dict(Counter(failed_target_versions)),
                "last_target_weight_already_generated": self.last_target_weight_already_generated,
                "max_size": self.max_size,
            }

    def get_last_target_weight_already_generated(self) -> int:
        with self._lock:
            return self.last_target_weight_already_generated

    def get_existing_target_weights(self) -> set[int]:
        """Get set of target weight versions that already have trajectories."""
        with self._lock:
            return set(self.target_weight_versions)

    def get_target_weight_counts(self) -> dict[int, int]:
        """Get counts of buffered trajectory groups per target weight."""
        with self._lock:
            return dict(Counter(self.target_weight_versions))

    def get_failed_target_weight_counts(self) -> dict[int, int]:
        """Get counts of failed prompt groups per target weight."""
        with self._lock:
            return dict(
                Counter(r["target_weight_version"] for r in self.failed_prompt_groups)
            )

    def get_target_completion_counts(self) -> dict[int, dict[str, int]]:
        """Get buffered and failed prompt-group counts by target weight."""
        with self._lock:
            buffered_counts = Counter(self.target_weight_versions)
            failed_counts = Counter(
                r["target_weight_version"] for r in self.failed_prompt_groups
            )
            target_weights = set(buffered_counts) | set(failed_counts)
            return {
                target_weight: {
                    "buffered": buffered_counts.get(target_weight, 0),
                    "failed": failed_counts.get(target_weight, 0),
                    "accounted": buffered_counts.get(target_weight, 0)
                    + failed_counts.get(target_weight, 0),
                }
                for target_weight in target_weights
            }

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
            failed_for_current = [
                r
                for r in self.failed_prompt_groups
                if r["target_weight_version"] == current_weight_version
            ]
            if not self.trajectories and not failed_for_current:
                return None

            total_trajectories = len(self.trajectories)
            print("🔍 ReplayBuffer sampling debug:")
            print(f"   {current_weight_version=}, {max_age_steps=}")
            print(f"   {self.trajectory_versions=}")

            version_counts = Counter(self.trajectory_versions)
            target_counts = Counter(self.target_weight_versions)
            failed_target_counts = Counter(
                r["target_weight_version"] for r in self.failed_prompt_groups
            )
            print(f"   {version_counts=}")
            print(f"   {target_counts=}")
            print(f"   {failed_target_counts=}")

            # Compute minimum valid version based on age window
            # max_age_steps=1 means trajectories from the last 1 step are valid
            min_valid_version = max(0, current_weight_version - max_age_steps)
            print(f"   {min_valid_version=}")

            # Check for unexpected old trajectories
            old_trajectories = [
                v for v in self.trajectory_versions if v < min_valid_version
            ]
            if old_trajectories:
                raise ValueError(
                    f"Found {len(old_trajectories)} trajectories older than min_valid_version {min_valid_version}"
                )

            # Filter for valid trajectories without modifying the buffer
            valid_indices = [
                i
                for i, v in enumerate(self.trajectory_versions)
                if min_valid_version <= v <= current_weight_version
            ]
            print(
                f"   valid_indices: {len(valid_indices)}/{total_trajectories} trajectories within age window"
            )
            if not valid_indices and not failed_for_current:
                print("No trajectories available for sampling.")
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

            accounted_for_current = len(intended_indices) + len(failed_for_current)

            # Stall training if we don't have enough successful or explicitly failed
            # prompt groups intended for this step.
            if accounted_for_current < num_prompt_groups:
                print(
                    f"   ⏸️ STALLING: Need {num_prompt_groups} trajectories for step {current_weight_version}, but only {len(intended_indices)} are ready"
                )
                print(
                    f"   ⏸️ Training will wait for remaining {num_prompt_groups - accounted_for_current} trajectories to be generated or fail"
                )
                print(
                    f"{ASYNC_DEBUG_PREFIX} replay_buffer_stall current_weight={current_weight_version} "
                    f"needed={num_prompt_groups} intended_ready={len(intended_indices)} "
                    f"failed_for_current={len(failed_for_current)} "
                    f"target_counts={dict(target_counts)} failed_target_counts={dict(failed_target_counts)} "
                    f"version_counts={dict(version_counts)}"
                )
                return None

            if not intended_indices:
                raise RuntimeError(
                    f"All {num_prompt_groups} prompt groups for step {current_weight_version} failed before buffering; no trainable trajectories are available."
                )

            # Select exactly the trajectories intended for this step (FIFO within same target)
            selected: list[int] = intended_indices[:num_prompt_groups]
            skipped_failed_prompt_groups = min(
                len(failed_for_current), num_prompt_groups - len(selected)
            )
            print(
                f"   ✅ Selected {len(selected)} trajectories for step {current_weight_version}; "
                f"skipping {skipped_failed_prompt_groups} failed prompt groups"
            )

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
            consumed_failures = failed_for_current[:skipped_failed_prompt_groups]
            consumed_failure_ids = {id(r) for r in consumed_failures}

            # Remove selected items in reverse order to maintain correct indices
            for idx in sorted(selected, reverse=True):
                self.trajectory_versions.pop(idx)
                self.target_weight_versions.pop(idx)
                self.trajectories.pop(idx)

            # The current training step has accounted for these failed prompt groups.
            # Remove all failure records for the step so they cannot affect later debug
            # state or target-completion accounting.
            self.failed_prompt_groups = [
                r for r in self.failed_prompt_groups if id(r) not in consumed_failure_ids
            ]
            print(
                f"🗑️ Consumed and removed {len(selected)} groups from buffer, old buffer size: {total_trajectories}, new buffer size: {len(self.trajectories)}, new target weight versions {self.target_weight_versions}"
            )
            if skipped_failed_prompt_groups:
                print(
                    f"{ASYNC_DEBUG_PREFIX} replay_buffer_skipped_failed_prompt_groups "
                    f"current_weight={current_weight_version} skipped={skipped_failed_prompt_groups} "
                    f"selected={len(selected)} requested={num_prompt_groups} "
                    f"failures={consumed_failures}"
                )

            return {
                "trajectories": sampled_items,
                "avg_trajectory_age": avg_trajectory_age,
                "requested_prompt_groups": num_prompt_groups,
                "sampled_prompt_groups": len(sampled_items),
                "skipped_failed_prompt_groups": skipped_failed_prompt_groups,
                "failed_prompt_groups": consumed_failures,
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
            self.failed_prompt_groups.clear()


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
        self._max_inflight = max_inflight
        self._inflight_sema = _threading.Semaphore(max_inflight)

        # Simple lock to prevent race conditions when checking/spawning workers
        self._generation_check_lock: _threading.Lock = _threading.Lock()
        # Track how many prompt-group workers are still in flight per target weight.
        self._generating_target_counts: dict[int, int] = {}
        self._worker_id = 0

        print(
            f"{ASYNC_DEBUG_PREFIX} collector_init current_weight={self.current_weight_version} "
            f"initial_weight={self.initial_weight_version} max_inflight={self._max_inflight} "
            f"num_prompts_per_step={self.master_config['grpo']['num_prompts_per_step']} "
            f"num_generations_per_prompt={self.master_config['grpo']['num_generations_per_prompt']} "
            f"max_age={self.master_config['grpo']['async_grpo']['max_trajectory_age_steps']}"
        )

    def _active_thread_count_unlocked(self) -> int:
        return sum(1 for thread in self._inflight_threads if thread.is_alive())

    def _next_worker_id(self) -> int:
        with self._generation_check_lock:
            self._worker_id += 1
            return self._worker_id

    def _generation_state_for_log(
        self,
        target_weights: Optional[list[int]] = None,
        target_completion_counts: Optional[dict[int, dict[str, int]]] = None,
    ) -> dict[str, Any]:
        if target_weights is None:
            target_weights = self._calculate_target_weights(self.current_weight_version)
        if target_completion_counts is None:
            target_completion_counts = ray.get(
                self.replay_buffer.get_target_completion_counts.remote()
            )

        required_groups = int(self.master_config["grpo"]["num_prompts_per_step"])
        with self._generation_check_lock:
            inflight_counts = dict(self._generating_target_counts)
        with self._threads_lock:
            active_threads = self._active_thread_count_unlocked()

        per_target = {}
        for target_weight in target_weights:
            completion_counts = target_completion_counts.get(target_weight, {})
            buffered = completion_counts.get("buffered", 0)
            failed = completion_counts.get("failed", 0)
            inflight = inflight_counts.get(target_weight, 0)
            per_target[target_weight] = {
                "buffered": buffered,
                "failed": failed,
                "inflight": inflight,
                "missing": required_groups - buffered - failed - inflight,
            }

        return {
            "current_weight_version": self.current_weight_version,
            "required_groups": required_groups,
            "max_inflight": self._max_inflight,
            "active_threads": active_threads,
            "inflight_counts": inflight_counts,
            "target_completion_counts": dict(target_completion_counts),
            "candidate_targets": target_weights,
            "per_candidate_target": per_target,
        }

    def get_debug_state(self) -> dict[str, Any]:
        """Return collector and replay-buffer state useful for diagnosing stalls."""
        state = self._generation_state_for_log()
        state["replay_buffer"] = ray.get(self.replay_buffer.get_debug_info.remote())
        return state

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
        self, generation_weight_version: int, max_prompt_groups: int
    ) -> Optional[tuple[int, int]]:
        """Get the next target weight that needs generation (if any)."""
        target_weights = self._calculate_target_weights(generation_weight_version)
        target_completion_counts = ray.get(
            self.replay_buffer.get_target_completion_counts.remote()
        )
        required_groups = int(self.master_config["grpo"]["num_prompts_per_step"])

        with self._generation_check_lock:
            candidate_states = []
            for target_weight in target_weights:
                completion_counts = target_completion_counts.get(target_weight, {})
                buffered_groups = completion_counts.get("buffered", 0)
                failed_groups = completion_counts.get("failed", 0)
                inflight_groups = self._generating_target_counts.get(target_weight, 0)
                missing_groups = (
                    required_groups
                    - buffered_groups
                    - failed_groups
                    - inflight_groups
                )
                candidate_states.append(
                    {
                        "target": target_weight,
                        "buffered": buffered_groups,
                        "failed": failed_groups,
                        "inflight": inflight_groups,
                        "missing": missing_groups,
                    }
                )
                if missing_groups > 0:
                    prompt_groups_to_launch = min(missing_groups, max_prompt_groups)
                    self._generating_target_counts[target_weight] = (
                        inflight_groups + prompt_groups_to_launch
                    )
                    print(
                        f"🎯 Reserved {prompt_groups_to_launch} prompt groups for target weight {target_weight}"
                    )
                    print(
                        f"{ASYNC_DEBUG_PREFIX} reserve target={target_weight} "
                        f"launch={prompt_groups_to_launch} buffered={buffered_groups} "
                        f"failed={failed_groups} "
                        f"previous_inflight={inflight_groups} "
                        f"new_inflight={self._generating_target_counts[target_weight]} "
                        f"required={required_groups} candidates={candidate_states}"
                    )
                    return target_weight, prompt_groups_to_launch

            print(
                f"{ASYNC_DEBUG_PREFIX} no_reservation generation_weight={generation_weight_version} "
                f"targets={target_weights} required={required_groups} "
                f"target_completion_counts={target_completion_counts} "
                f"inflight_counts={dict(self._generating_target_counts)} "
                f"candidates={candidate_states}"
            )

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
        """Check if collection should be paused due to generation limits."""
        try:
            target_weights = self._calculate_target_weights(self.current_weight_version)
            target_completion_counts = ray.get(
                self.replay_buffer.get_target_completion_counts.remote()
            )
            required_groups = int(self.master_config["grpo"]["num_prompts_per_step"])

            # Check if any target weight in our range needs generation
            with self._generation_check_lock:
                for target_weight in target_weights:
                    completion_counts = target_completion_counts.get(target_weight, {})
                    buffered_groups = completion_counts.get("buffered", 0)
                    failed_groups = completion_counts.get("failed", 0)
                    inflight_groups = self._generating_target_counts.get(
                        target_weight, 0
                    )
                    if (
                        buffered_groups + failed_groups + inflight_groups
                        < required_groups
                    ):
                        return False  # Found a target that needs generation

            print(
                f"⏸️ All target weights {target_weights} are full or in progress, pausing"
            )
            print(
                f"{ASYNC_DEBUG_PREFIX} pause_decision "
                f"{self._generation_state_for_log(target_weights, target_completion_counts)}"
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
                        target_weights = self._calculate_target_weights(
                            self.current_weight_version
                        )

                        print(
                            f"⏸️ Pausing collection: all target weights {target_weights} for weight version {self.current_weight_version} "
                            f"are full or already in progress. Waiting for weight update..."
                        )
                        self._last_limit_warning_version = self.current_weight_version

                        self._generation_limit_cleared.clear()  # Clear the event to pause

                    # Efficiently wait for generation limits to be cleared, with a
                    # periodic diagnostic so a stuck run shows whether workers remain.
                    while self.running and not self._generation_limit_cleared.wait(30):
                        print(
                            f"{ASYNC_DEBUG_PREFIX} collection_paused_waiting "
                            f"{self._generation_state_for_log()}"
                        )

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
            self.running = False
            print("🛑 Trajectory collection stopped")

    def _process_batch(self, batch: BatchedDataDict[DatumSpec]) -> None:
        """Process a single batch and generate for one target weight."""
        try:
            generation_weight_version = self.current_weight_version
            num_generations = self.master_config["grpo"]["num_generations_per_prompt"]
            num_prompts = batch.size
            print(
                f"{ASYNC_DEBUG_PREFIX} process_batch generation_weight={generation_weight_version} "
                f"batch_size={num_prompts} num_generations={num_generations} "
                f"state={self._generation_state_for_log()}"
            )

            # Get the next target weight that needs generation
            target_reservation = self._get_next_target_for_generation(
                generation_weight_version, num_prompts
            )

            if target_reservation is None:
                print(
                    f"🔄 No targets need generation for weight {generation_weight_version}"
                )
                return

            target_weight, num_prompt_groups_to_launch = target_reservation
            print(
                f"🎯 Generating {num_prompt_groups_to_launch} prompt groups for target weight {target_weight} from generation_weight_version {generation_weight_version}"
            )

            # Generate only the missing prompt groups for this target.
            for prompt_idx in range(num_prompt_groups_to_launch):
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

                worker_id = self._next_worker_id()
                acquire_start = time.time()
                while not self._inflight_sema.acquire(timeout=30):
                    print(
                        f"{ASYNC_DEBUG_PREFIX} worker_waiting_for_slot "
                        f"worker_id={worker_id} target={target_weight} prompt_idx={prompt_idx} "
                        f"wait_s={time.time() - acquire_start:.1f} "
                        f"state={self._generation_state_for_log()}"
                    )
                acquire_elapsed = time.time() - acquire_start
                if acquire_elapsed > 1:
                    print(
                        f"{ASYNC_DEBUG_PREFIX} worker_acquired_slot_after_wait "
                        f"worker_id={worker_id} target={target_weight} prompt_idx={prompt_idx} "
                        f"wait_s={acquire_elapsed:.1f}"
                    )

                worker = _threading.Thread(
                    target=self._run_prompt_group_worker,
                    args=(
                        worker_id,
                        repeated_batch,
                        generation_weight_version,
                        target_weight,
                        prompt_idx,
                    ),
                    daemon=True,
                )
                with self._threads_lock:
                    self._inflight_threads.add(worker)
                    active_threads = self._active_thread_count_unlocked()
                print(
                    f"{ASYNC_DEBUG_PREFIX} worker_thread_started worker_id={worker_id} "
                    f"target={target_weight} prompt_idx={prompt_idx} "
                    f"generation_weight={generation_weight_version} active_threads={active_threads}"
                )
                worker.start()

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
        worker_id: int,
        repeated_batch: BatchedDataDict[DatumSpec],
        generation_weight_version: int,
        target_weight_version: int,
        prompt_idx: int,
    ) -> None:
        worker_start = time.time()
        outcome = "unknown"
        failure_reason: Optional[str] = None
        print(
            f"{ASYNC_DEBUG_PREFIX} worker_start worker_id={worker_id} "
            f"target={target_weight_version} prompt_idx={prompt_idx} "
            f"generation_weight={generation_weight_version} repeated_batch_size={repeated_batch.size}"
        )
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
                    max_seq_len=self.master_config["policy"][
                        "max_total_sequence_length"
                    ],
                    generation_config=generation_config,
                    max_rollout_turns=None,
                    greedy=False,
                )
                final_batch = nemo_gym_rollout_result.final_batch
                rollout_metrics = nemo_gym_rollout_result.rollout_metrics
                print(
                    f"{ASYNC_DEBUG_PREFIX} worker_rollout_done worker_id={worker_id} "
                    f"target={target_weight_version} prompt_idx={prompt_idx} "
                    f"elapsed_s={time.time() - worker_start:.1f} "
                    f"metric_keys={sorted(rollout_metrics.keys())[:16]}"
                )
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
                print(
                    f"{ASYNC_DEBUG_PREFIX} worker_rollout_done worker_id={worker_id} "
                    f"target={target_weight_version} prompt_idx={prompt_idx} "
                    f"elapsed_s={time.time() - worker_start:.1f} "
                    f"metric_keys={sorted(rollout_metrics.keys())[:16]}"
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
                last_full_log = 0.0
                while self.running:
                    status = ray.get(
                        self.replay_buffer.push_with_wait_signal.remote(
                            trajectory_group,
                            generation_weight_version,
                            target_weight_version,
                        )
                    )
                    if status == "success":
                        outcome = "buffered"
                        print(
                            f"📦 Buffered per-prompt group (prompt_idx {prompt_idx}, target_weight {target_weight_version})"
                        )
                        print(
                            f"{ASYNC_DEBUG_PREFIX} worker_buffered worker_id={worker_id} "
                            f"target={target_weight_version} prompt_idx={prompt_idx} "
                            f"generation_weight={generation_weight_version} "
                            f"elapsed_s={time.time() - worker_start:.1f}"
                        )

                        break
                    elif status == "full":
                        now = time.time()
                        if last_full_log == 0.0 or now - last_full_log >= 30:
                            last_full_log = now
                            print(
                                f"{ASYNC_DEBUG_PREFIX} worker_buffer_full worker_id={worker_id} "
                                f"target={target_weight_version} prompt_idx={prompt_idx} "
                                f"elapsed_s={time.time() - worker_start:.1f}"
                            )
                        # Exponential backoff up to 0.5 second
                        time.sleep(min(backoff_delay, 0.5))
                        backoff_delay *= 1.5
                    else:
                        # Unexpected status, wait briefly
                        time.sleep(0.01)
                if outcome == "unknown":
                    outcome = "stopped_before_buffer"
            except Exception as e:
                outcome = "buffer_enqueue_error"
                failure_reason = f"{type(e).__name__}: {e}"
                print(f"❌ Failed to enqueue per-prompt group to buffer: {e}")
                import traceback

                traceback.print_exc()
        except Exception as e:
            outcome = "worker_error"
            failure_reason = f"{type(e).__name__}: {e}"
            print(f"❌ Error in prompt group worker: {e}")
            import traceback

            traceback.print_exc()
        finally:
            if outcome in {"worker_error", "buffer_enqueue_error"}:
                try:
                    ray.get(
                        self.replay_buffer.record_failed_prompt_group.remote(
                            generation_weight_version,
                            target_weight_version,
                            prompt_idx,
                            worker_id,
                            (failure_reason or outcome)[:1024],
                        )
                    )
                    print(
                        f"{ASYNC_DEBUG_PREFIX} worker_failure_recorded "
                        f"worker_id={worker_id} target={target_weight_version} "
                        f"prompt_idx={prompt_idx} outcome={outcome}"
                    )
                except Exception as e:
                    print(
                        f"{ASYNC_DEBUG_PREFIX} worker_failure_record_failed "
                        f"worker_id={worker_id} target={target_weight_version} "
                        f"prompt_idx={prompt_idx} outcome={outcome} error={e}"
                    )

            # Track worker completion for this target. When all workers launched for a target
            # have finished (success or error), the target becomes eligible for rescheduling
            # if it is still incomplete in the replay buffer.
            with self._generation_check_lock:
                remaining = self._generating_target_counts.get(target_weight_version, 0)
                if remaining > 1:
                    self._generating_target_counts[target_weight_version] = remaining - 1
                    print(
                        f"{ASYNC_DEBUG_PREFIX} worker_release_partial worker_id={worker_id} "
                        f"target={target_weight_version} remaining_inflight={remaining - 1}"
                    )
                elif remaining == 1:
                    self._generating_target_counts.pop(target_weight_version, None)
                    print(
                        f"🧹 Released reservation for target weight {target_weight_version}"
                    )
                    print(
                        f"{ASYNC_DEBUG_PREFIX} worker_release_final worker_id={worker_id} "
                        f"target={target_weight_version}"
                    )
                else:
                    print(
                        f"{ASYNC_DEBUG_PREFIX} worker_release_missing_reservation "
                        f"worker_id={worker_id} target={target_weight_version}"
                    )

            # Detach thread record when finished
            with self._threads_lock:
                current = _threading.current_thread()
                if current in self._inflight_threads:
                    self._inflight_threads.remove(current)
                active_threads = self._active_thread_count_unlocked()
            try:
                self._inflight_sema.release()
            except Exception:
                import traceback

                traceback.print_exc()
            print(
                f"{ASYNC_DEBUG_PREFIX} worker_done worker_id={worker_id} "
                f"target={target_weight_version} prompt_idx={prompt_idx} "
                f"outcome={outcome} elapsed_s={time.time() - worker_start:.1f} "
                f"active_threads={active_threads}"
            )
