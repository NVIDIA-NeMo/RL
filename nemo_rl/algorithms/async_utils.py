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

import gc
import random
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
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface

TokenizerType = PreTrainedTokenizerBase


@ray.remote
class ReplayBuffer:
    """Simple replay buffer for storing trajectories."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.trajectories = []
        # Auxiliary metadata for each stored trajectory
        self.trajectory_steps = []  # collector step when generated
        self.trajectory_versions = []  # weight-version used for generation

    def push(self, trajectory: dict[str, Any], step: int, weight_version: int) -> None:
        """Add a trajectory with metadata.

        Args:
            trajectory: data dict
            step:       collector local step
            weight_version: monotonic counter of the model weights used to generate
        """
        print(f"🔍 ReplayBuffer.push: Adding trajectory for step {step}")
        self.trajectories.append(trajectory)
        self.trajectory_steps.append(step)
        self.trajectory_versions.append(weight_version)

        # Remove oldest if buffer is full
        if len(self.trajectories) > self.max_size:
            removed_step = self.trajectory_steps.pop(0)
            self.trajectory_versions.pop(0)
            self.trajectories.pop(0)
            print(f"ReplayBuffer: Removed oldest trajectory (step {removed_step})")

        print(
            f"ReplayBuffer state: {len(self.trajectories)} trajectories, steps={self.trajectory_steps}"
        )

    def get_debug_info(self) -> dict:
        """Get debug information about buffer state."""
        return {
            "total_trajectories": len(self.trajectories),
            "trajectory_steps": self.trajectory_steps,
            "trajectory_versions": self.trajectory_versions,
            "max_size": self.max_size,
        }

    def clean_old_trajectories(
        self, current_weight_version: int, max_age_steps: int
    ) -> int:
        """Remove trajectories that are too old to be useful.

        Returns:
            Number of trajectories removed
        """
        if not self.trajectories:
            return 0

        # Find trajectories to remove
        indices_to_remove = []
        for i, traj_version in enumerate(self.trajectory_versions):
            age = current_weight_version - traj_version
            if age > max_age_steps:
                indices_to_remove.append(i)

        # Remove in reverse order to maintain indices
        removed_count = 0
        for i in reversed(indices_to_remove):
            self.trajectory_steps.pop(i)
            self.trajectory_versions.pop(i)
            self.trajectories.pop(i)
            removed_count += 1

        if removed_count > 0:
            print(f"Cleaned {removed_count} old trajectories from buffer")

        return removed_count

    def sample(
        self,
        batch_size: int,
        current_weight_version: int,
        max_age_steps: int,
    ) -> Optional[list]:
        """Sample trajectories that are not too old."""
        cleaned = self.clean_old_trajectories(current_weight_version, max_age_steps)

        if not self.trajectories:
            return None

        # Filter trajectories by age
        valid_indices = []
        total_trajectories = len(self.trajectories)

        print("🔍 ReplayBuffer sampling debug:")
        print(
            f"   current_weight_version={current_weight_version}, max_age_steps={max_age_steps}"
        )
        print(f"   trajectory_versions={self.trajectory_versions}")
        print(f"   cleaned_old_trajectories={cleaned}")

        for i, traj_version in enumerate(self.trajectory_versions):
            age = current_weight_version - traj_version

            valid = age <= max_age_steps

            print(
                (
                    f"   trajectory[{i}]: weight_version={traj_version}, age={age}, "
                    f"window={max_age_steps}, valid={valid}"
                )
            )

            if valid:
                valid_indices.append(i)

        valid_count = len(valid_indices)
        filtered_count = total_trajectories - valid_count

        if not valid_indices:
            print(
                f"No trajectories within age limit ({max_age_steps} steps). Total: {total_trajectories}, Filtered: {filtered_count}"
            )
            return None

        if filtered_count > 0:
            print(
                f"Filtered {filtered_count}/{total_trajectories} trajectories outside ±{max_age_steps} step window"
            )

        sampled_indices = random.sample(
            valid_indices, min(batch_size, len(valid_indices))
        )
        print(f"✅ Sampled trajectory indices: {sampled_indices}")
        return [self.trajectories[i] for i in sampled_indices]

    def size(self) -> int:
        """Return current buffer size."""
        return len(self.trajectories)

    def clear(self) -> None:
        """Clear the buffer."""
        self.trajectories.clear()
        self.trajectory_steps.clear()
        self.trajectory_versions.clear()


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
        self.current_step = start_step
        self.running = False
        self.paused = False

        self._pg_lock: _threading.Lock = _threading.Lock()
        self._pause_lock: _threading.Lock = _threading.Lock()

        self.current_weight_version: int = start_step

        # Check if we should use async rollouts
        self._use_async_rollouts = False
        if (
            hasattr(policy_generation, "cfg")
            and "vllm_cfg" in policy_generation.cfg
            and policy_generation.cfg["vllm_cfg"].get("async_engine", False)
        ):
            self._use_async_rollouts = True
            print(
                "Trajectory collector: Detected vLLM async engine; enabling async rollouts in collector"
            )

    def set_weight_version(self, version: int) -> None:
        self.current_weight_version = version

    def start_collection(self, dataloader: StatefulDataLoader) -> None:
        """Start collecting trajectories from dataloader."""
        self.running = True
        self.dataloader = dataloader
        print("Started continuous trajectory collection")

        import threading

        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()

        print("Collection thread started, start_collection returning")

    def _collection_loop(self):
        """Run the collection loop in background thread."""
        try:
            for batch in self.dataloader:
                if not self.running:
                    break

                # Check if paused and wait
                while self.paused and self.running:
                    import time

                    time.sleep(0.1)

                if not self.running:
                    break

                self._process_batch(batch)
                self.current_step += 1

        except Exception as e:
            print(f"❌ Error in trajectory collection: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.running = False
            print("🛑 Trajectory collection stopped")

    def _process_batch(self, batch: BatchedDataDict[DatumSpec]) -> None:
        """Process a single batch and add trajectories to replay buffer."""
        try:
            generation_weight_version = self.current_weight_version

            repeated_batch = batch.repeat_interleave(
                self.master_config["grpo"]["num_generations_per_prompt"]
            )

            if self._use_async_rollouts:
                with self._pg_lock:
                    final_batch, rollout_metrics = run_async_multi_turn_rollout(
                        policy_generation=self.policy_generation,
                        input_batch=repeated_batch,
                        tokenizer=self.tokenizer,
                        task_to_env=self.task_to_env,
                        max_seq_len=self.master_config["policy"][
                            "max_total_sequence_length"
                        ],
                        max_rollout_turns=self.master_config["grpo"][
                            "max_rollout_turns"
                        ],
                        greedy=False,
                    )
            else:
                # Fallback to sync rollout
                with self._pg_lock:
                    final_batch, rollout_metrics = run_multi_turn_rollout(
                        policy_generation=self.policy_generation,
                        input_batch=repeated_batch,
                        tokenizer=self.tokenizer,
                        task_to_env=self.task_to_env,
                        max_seq_len=self.master_config["policy"][
                            "max_total_sequence_length"
                        ],
                        max_rollout_turns=self.master_config["grpo"][
                            "max_rollout_turns"
                        ],
                        greedy=False,
                    )

            # Trajectory here is the complete batch required for training.
            # TODO: in future we can see if trajectory is just a prompt * num_generations_per_prompt

            # Move batch to CPU to avoid consuming GPU memory in replay buffer
            final_batch_cpu = final_batch.to("cpu")

            # Explicit cleanup of GPU tensors
            del final_batch
            gc.collect()

            trajectory = {
                "batch": final_batch_cpu,
                "rollout_metrics": rollout_metrics,
                "timestamp": time.time(),
                "collector_step": self.current_step,
            }

            # Add to replay buffer with the weight version that was used for generation
            try:
                ray.get(
                    self.replay_buffer.push.remote(
                        trajectory, self.current_step, generation_weight_version
                    )
                )
                print(
                    f"Successfully added trajectory to buffer (step {self.current_step}, weight_version {generation_weight_version})"
                )
            except Exception as e:
                print(f"❌ Failed to add trajectory to buffer: {e}")
                import traceback

                traceback.print_exc()
                return

            print(
                f"📦 Added trajectory batch (size: {final_batch_cpu.size}) to replay buffer (step {self.current_step})"
            )
            print(
                f"   Trajectory rewards: min={final_batch_cpu['total_reward'].min():.3f}, max={final_batch_cpu['total_reward'].max():.3f}, mean={final_batch_cpu['total_reward'].mean():.3f}"
            )

            try:
                buffer_size_after_push = ray.get(self.replay_buffer.size.remote())
                print(f"   Buffer size after push: {buffer_size_after_push}")
            except Exception as e:
                print(f"❌ Failed to check buffer size after push: {e}")

        except Exception as e:
            print(f"❌ Error processing batch: {e}")
            import traceback

            traceback.print_exc()

    def get_current_step(self) -> int:
        """Return current step for debugging."""
        return self.current_step

    def get_weight_version(self) -> int:
        return self.current_weight_version

    def pause(self) -> None:
        """Pause trajectory collection."""
        with self._pause_lock:
            self.paused = True
        print("Trajectory collection paused")

    def resume(self) -> None:
        """Resume trajectory collection."""
        with self._pause_lock:
            self.paused = False
        print("Trajectory collection resumed")

    def stop(self) -> None:
        """Stop trajectory collection."""
        self.running = False
