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
import random
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


@ray.remote
class ReplayBuffer:
    """Replay buffer storing per-prompt groups.

    A single entry corresponds to 1 prompt repeated by
    grpo.num_generations_per_prompt (required to compute per-prompt advantages).
    """

    def __init__(self, max_size: int, master_config: MasterConfig):
        self.max_size = max_size
        self.trajectories = []
        # If trajectory_version is 1 and target_weight_version is 4 it means that weight version 1 was used for generating a trajectory and this trajectory will be used for training when weight version is 4.
        self.trajectory_versions = []  # it is the weight-version used for generation of a trajectory
        self.target_weight_versions = []  # it is the weight-version of the trainer where this trajectory will be used.

        self.last_target_weight_already_generated = -1
        self._lock = _threading.Lock()

        self.master_config = master_config # We are passing it twice once in ReplayBuffer and once in AsyncTrajectoryCollector
        self.sampler_fns = {
            "fifo": self.fifo_sample,
            "mixed": self.mixed_sample,
            "reward_min_max": self.reward_sample_min_max,
        }

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



    def fifo_sample(self, valid_indices: list[int], num_prompt_groups: int, current_weight_version: int,  min_valid_version: int, **kwargs) -> Optional[dict[str, Any]]:


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

        return {"selected": selected, "removed": None}

    def mixed_sample(self, valid_indices: list[int], num_prompt_groups: int, current_weight_version: int,  min_valid_version: int,**kwargs) -> Optional[dict[str, Any]]:

        #! This is the old logic. There is one issue is that we might have older versions than even min_valid_version, so we need to take all of them up

        #! we don't use valid_indices here because we can directly use self.trajectory_versions

        last_chance_indices = [
                i
                for i, v in enumerate(self.trajectory_versions)
                if v <= min_valid_version
        ]

        #! These indices have to be selected because its their last chance. Now we have remaining num_prompt_groups - len(intended_indices) left to sample from the remaining valid_indices of trajectories those we sample randomly from the remaining

        print(f"   🎯 Found {len(last_chance_indices)} trajectories intended for current step {current_weight_version}")

        # If enough intended are available, just take those (FIFO within intended)
        selected: list[int] = []
        if len(last_chance_indices) >= num_prompt_groups:

            #! This happens when we have more last_chance_indices than we can pick in one go (num_prompt_groups). Ideally we can prevent this but practically it happens rarely and doesn't affect performance. Commit: f4ddc639b6a8ea7c11e20368ac10d3ad0110eb4f

            selected: list[int] = last_chance_indices[:num_prompt_groups] # When we have more last_chance_indices than we can pick, always pick them in FIFO order rather than random sampling
            # selected: list[int] = random.sample(intended_indices, num_prompt_groups)
            print(f"🔴 Selected {len(selected)} trajectories all intended for step {current_weight_version}")
            return selected


        remaining_slots = num_prompt_groups - len(selected)
        
        #! Find the remaining indices not in selected
        remaining_pool = [
            i for i, _ in enumerate(self.trajectory_versions)
            if i not in selected
        ]

        print(f"Filling remaining {remaining_slots} from {len(remaining_pool)} non-intended valid trajectories (random)")

        if remaining_slots > 0:
            if len(remaining_pool) <= remaining_slots:
                selected.extend(remaining_pool)
            else:
                selected.extend(random.sample(remaining_pool, remaining_slots))

        print(f"   ✅ Mixed selection done: {len(selected)} total (intended={len(last_chance_indices)}, mixed_in={len(selected) - len(last_chance_indices)})")

        return {"selected": selected, "removed": None}


    def reward_sample_min_max(self, valid_indices: list[int], num_prompt_groups: int, current_weight_version: int,  min_valid_version: int) -> Optional[dict[str, Any]]:

        #! In FIFO manner find the indices that are eligible based on the min_reward and max_reward

        async_cfg = self.master_config.get("grpo", {}).get(
                            "async_grpo", {}
        )
        min_reward_sample = async_cfg.get("min_reward_sample", 0.0)
        max_reward_sample = async_cfg.get("max_reward_sample", 1.0)

        #! First find all eligible indices
        intended_indices = [
            i
            for i, v in enumerate(self.trajectory_versions) if self.trajectories[i]["avg_reward"] >= min_reward_sample and self.trajectories[i]["avg_reward"] <= max_reward_sample
        ]

        print(
            f"   🎯 Found {len(intended_indices)} trajectories eligible for current step {current_weight_version}"
        )

        # Stall training if we don't have enough trajectories intended for this step
        if len(intended_indices) < num_prompt_groups:
            print(
                f"   ⏸️ STALLING: Need {num_prompt_groups} trajectories eligible for step {current_weight_version}, but only {len(intended_indices)} eligible trajectories are ready"
            )
            print(
                f"   ⏸️ Training will wait for remaining {num_prompt_groups - len(intended_indices)} eligible trajectories to be generated"
            )
            return None

        # Select exactly the trajectories intended for this step (FIFO within same target)
        selected: list[int] = intended_indices[:num_prompt_groups]
        print(
            f"   ✅ Selected {len(selected)} trajectories all intended for step {current_weight_version}"
        )

        #! We need to remove all the indices that we went over but were not eligible
        max_index = max(selected)

        print(f"max_index = {max_index}")

        remove_indices = [i for i in range(max_index) if i not in selected]

        #! In both fifo & mixed, we never threw out any trajectories. So we could just return the selected indices and remove them in the sample(...) function. But here we throw out both selected + any trajectories that were before it

        print(
            f"🗑️ Removing {len(remove_indices)} trajectories that were not eligible: {remove_indices}"
        )

        return {"selected": selected, "removed": remove_indices}
    

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
            print(f"min_valid_version = {min_valid_version=}")

            # Check for unexpected old trajectories
            old_trajectories = [
                v for v in self.trajectory_versions if v < min_valid_version
            ]
            if old_trajectories:
                # raise ValueError(
                #     f"Found {len(old_trajectories)} trajectories older than min_valid_version {min_valid_version}"
                # )
                print(f"Found {len(old_trajectories)} trajectories older than min_valid_version {min_valid_version}")

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
            async_cfg = self.master_config.get("grpo", {}).get("async_grpo", {})
            sampler_type = async_cfg.get("sampler_type", "fifo")
            
            sample_result = self.sampler_fns[sampler_type](valid_indices, num_prompt_groups, current_weight_version, min_valid_version)

            if sample_result is None:
                return None

            selected = sample_result["selected"]
            removed = sample_result["removed"]

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
            if removed is not None and len(removed) > 0:
                for idx in sorted(removed, reverse=True):
                    self.trajectory_versions.pop(idx)
                    self.target_weight_versions.pop(idx)
                    self.trajectories.pop(idx)
            else:
                for idx in sorted(selected, reverse=True):
                    self.trajectory_versions.pop(idx)
                    self.target_weight_versions.pop(idx)
                    self.trajectories.pop(idx)

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


@ray.remote
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

        # Limit in-flight generator requests to num_prompts_per_step
        max_inflight = int(self.master_config["grpo"]["num_prompts_per_step"]) or 1
        self._inflight_sema = _threading.Semaphore(max_inflight)

        # Simple lock to prevent race conditions when checking/spawning workers
        self._generation_check_lock: _threading.Lock = _threading.Lock()
        # Track which target weights are currently being generated (globally)
        self._generating_targets: set[int] = set()

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
        """Get the next target weight that needs generation (if any)."""
        target_weights = self._calculate_target_weights(generation_weight_version)
        last_target_weight_already_generated = ray.get(
            self.replay_buffer.get_last_target_weight_already_generated.remote()
        )

        with self._generation_check_lock:
            for target_weight in target_weights:
                if (
                    target_weight > last_target_weight_already_generated
                    and target_weight not in self._generating_targets
                ):
                    self._generating_targets.add(target_weight)
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
        """Check if collection should be paused due to generation limits."""
        try:
            target_weights = self._calculate_target_weights(self.current_weight_version)
            last_target_weight_already_generated = ray.get(
                self.replay_buffer.get_last_target_weight_already_generated.remote()
            )

            # Check if any target weight in our range needs generation
            with self._generation_check_lock:
                for target_weight in target_weights:
                    if (
                        target_weight > last_target_weight_already_generated
                        and target_weight not in self._generating_targets
                    ):
                        return False  # Found a target that needs generation

            print(
                f"⏸️ All target weights {target_weights} already generated or in progress, pausing"
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
                        target_weights = [
                            self.current_weight_version + i
                            for i in range(max_trajectory_age)
                        ]

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
            self.running = False
            print("🛑 Trajectory collection stopped")

    def _process_batch(self, batch: BatchedDataDict[DatumSpec]) -> None:
        """Process a single batch and generate for one target weight."""
        try:
            generation_weight_version = self.current_weight_version
            num_generations = self.master_config["grpo"]["num_generations_per_prompt"]
            num_prompts = batch.size

            # Get the next target weight that needs generation
            target_weight = self._get_next_target_for_generation(
                generation_weight_version
            )

            if target_weight is None:
                print(
                    f"🔄 No targets need generation for weight {generation_weight_version}"
                )
                return

            print(
                f"🎯 Generating for target weight {target_weight} from generation_weight_version {generation_weight_version}"
            )

            # Generate for all prompts in this batch for the target weight
            for prompt_idx in range(num_prompts):
                # Wait for refit to complete if in progress
                if not self._refit_pause_cleared.is_set() and self.running:
                    with self._threads_lock:
                        active_threads = len(self._inflight_threads)
                    print(
                        f"⏸️ Waiting for refit to complete before starting new generation ({active_threads} threads still active)"
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
        """Pause new generation starts and wait for pending generations to complete before refit."""
        start_time = time.time()
        print("🔄 Preparing for refit: pausing new generations...")

        # Pause new generation starts
        self._refit_pause_cleared.clear()
        print("⏸️ New generation starts paused")

        # Wait for all pending generations to complete
        # Note that is suboptimal for async performance and will be fixed in a follow-up PR where two more options will be added:
        # 1. Pause the generations at their current decoding step, update the weights and continue with decoding.
        # 2. Stop the current generations, store in a buffer and resume them in next iteration with new weights.
        self.wait_for_pending_generations()

        elapsed = time.time() - start_time
        print(
            f"✅ All pending generations completed, ready for refit (took {elapsed:.2f}s)"
        )

    def resume_after_refit(self) -> None:
        """Resume new generation starts after refit is complete."""
        print("🔄 Resuming generation starts after refit")
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
    ) -> None:
        try:
            # Run rollout for this prompt group
            # Async engine supports concurrent generation; avoid locking
            final_batch, rollout_metrics = run_async_multi_turn_rollout(
                policy_generation=self.policy_generation,
                input_batch=repeated_batch,
                tokenizer=self.tokenizer,
                task_to_env=self.task_to_env,
                max_seq_len=self.master_config["policy"]["max_total_sequence_length"],
                max_rollout_turns=self.master_config["grpo"]["max_rollout_turns"],
                greedy=False,
            )

            # Move to CPU and push to buffer (avoid blocking on GC/push)
            final_batch_cpu = final_batch.to("cpu")
            del final_batch

            try:
                group_rewards = final_batch_cpu["total_reward"]
                avg_reward = float(group_rewards.float().mean().item())
            except Exception:
                print(f"🔍 Error in calculating avg_reward: {e}")
                # Be conservative if structure is unexpected
                avg_reward = 0.0

            trajectory_group = {
                "batch": final_batch_cpu,
                "rollout_metrics": rollout_metrics,
                "avg_reward": avg_reward,
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
                        print(
                            f"📦 Buffered per-prompt group (prompt_idx {prompt_idx}, target_weight {target_weight_version})"
                        )

                        # Release reservation when FIRST prompt group for this target is successfully buffered
                        if prompt_idx == 0:
                            with self._generation_check_lock:
                                if target_weight_version in self._generating_targets:
                                    self._generating_targets.discard(
                                        target_weight_version
                                    )
                                    print(
                                        f"🧹 Released reservation for target weight {target_weight_version} (first prompt buffered)"
                                    )
                        break
                    elif status == "full":
                        # Exponential backoff up to 1 second
                        time.sleep(min(backoff_delay, 1.0))
                        backoff_delay *= 1.5
                    else:
                        # Unexpected status, wait briefly
                        time.sleep(0.01)
            except Exception as e:
                print(f"❌ Failed to enqueue per-prompt group to buffer: {e}")
                import traceback

                traceback.print_exc()
        except Exception as e:
            print(f"❌ Error in prompt group worker: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Clean up reservation in case of error (if not already cleaned up)
            with self._generation_check_lock:
                if target_weight_version in self._generating_targets:
                    self._generating_targets.discard(target_weight_version)
                    print(
                        f"🧹 Emergency cleanup: Released reservation for target weight {target_weight_version}"
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
