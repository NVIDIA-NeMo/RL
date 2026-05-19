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
            int(self.master_config.grpo["num_prompts_per_step"])
            * int(self.master_config.grpo["async_grpo"]["max_trajectory_age_steps"])
        ) or 1
        self._inflight_sema = _threading.Semaphore(max_inflight)

        # Simple lock to prevent race conditions when checking/spawning workers
        self._generation_check_lock: _threading.Lock = _threading.Lock()
        # Track how many prompt-group workers are still in flight per target weight.
        self._generating_target_counts: dict[int, int] = {}

        # Failure tracking for generation workers. _failure_lock guards both
        # _failure_count and _fatal_error and is acquired only inside
        # _run_prompt_group_worker's except path and inside check_health; it
        # is never held while taking _generation_check_lock.
        self._failure_lock: _threading.Lock = _threading.Lock()
        self._failure_count: int = 0
        self._fatal_error: Optional[BaseException] = None
        self._max_generation_failures: int = int(
            self.master_config.grpo["async_grpo"]["max_generation_failures"]
        )

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
        async_cfg = self.master_config.grpo.get("async_grpo", {})
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
        """Get the next target weight that needs generation (if any).

        Returns ``(target_weight, num_prompt_groups_to_launch)`` or ``None`` when
        every target in the window already has enough buffered + in-flight prompt
        groups to satisfy ``num_prompts_per_step``.

        The buffer count is read while holding ``_generation_check_lock`` so it
        is sequenced with the in-flight decrement performed by
        ``_run_prompt_group_worker``'s ``finally`` block. Reading it outside the
        lock would let a stale snapshot of buffered groups combine with the
        already-decremented in-flight count, producing a positive ``missing``
        when the target is actually full and over-launching by one prompt group
        — the extras then linger in the buffer and trip the age-window
        assertion in ``ReplayBuffer.sample`` a few steps later.
        """
        target_weights = self._calculate_target_weights(generation_weight_version)
        required_groups = int(self.master_config.grpo["num_prompts_per_step"])

        with self._generation_check_lock:
            target_weight_counts = ray.get(
                self.replay_buffer.get_target_weight_counts.remote()
            )
            for target_weight in target_weights:
                buffered_groups = target_weight_counts.get(target_weight, 0)
                inflight_groups = self._generating_target_counts.get(target_weight, 0)
                missing_groups = required_groups - buffered_groups - inflight_groups
                if missing_groups > 0:
                    prompt_groups_to_launch = min(missing_groups, max_prompt_groups)
                    self._generating_target_counts[target_weight] = (
                        inflight_groups + prompt_groups_to_launch
                    )
                    print(
                        f"🎯 Reserved {prompt_groups_to_launch} prompt groups for target weight {target_weight}"
                    )
                    return target_weight, prompt_groups_to_launch

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
            required_groups = int(self.master_config.grpo["num_prompts_per_step"])

            # Read the buffer count inside the lock for the same reason as in
            # _get_next_target_for_generation: keeps the buffered/in-flight
            # snapshot consistent with the decrement in the worker's finally.
            with self._generation_check_lock:
                target_weight_counts = ray.get(
                    self.replay_buffer.get_target_weight_counts.remote()
                )
                for target_weight in target_weights:
                    buffered_groups = target_weight_counts.get(target_weight, 0)
                    inflight_groups = self._generating_target_counts.get(
                        target_weight, 0
                    )
                    if buffered_groups + inflight_groups < required_groups:
                        return False  # Found a target that needs generation

            print(
                f"⏸️ All target weights {target_weights} are full or in progress, pausing"
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
        """Process a single batch, distributing prompts across targets that need generation.

        A single reservation may only cover part of the batch (e.g. when one
        target needs fewer prompt groups than ``batch.size``). To avoid
        discarding the remaining prompts, keep reserving against successive
        targets in the age window until the batch is exhausted or no target
        still needs generation.
        """
        try:
            generation_weight_version = self.current_weight_version
            num_generations = self.master_config.grpo["num_generations_per_prompt"]
            num_prompts = batch.size

            prompt_idx = 0
            while prompt_idx < num_prompts:
                remaining_prompts = num_prompts - prompt_idx

                # Get the next target weight that needs generation, along with
                # the number of prompt groups still missing for it (capped by
                # the unused portion of this batch).
                target_reservation = self._get_next_target_for_generation(
                    generation_weight_version, remaining_prompts
                )

                if target_reservation is None:
                    if prompt_idx == 0:
                        print(
                            f"🔄 No targets need generation for weight {generation_weight_version}"
                        )
                    else:
                        print(
                            f"🔄 No more targets need generation; {remaining_prompts} prompt(s) left unused in this batch"
                        )
                    break

                target_weight, num_prompt_groups_to_launch = target_reservation
                print(
                    f"🎯 Generating {num_prompt_groups_to_launch} prompt groups for target weight {target_weight} from generation_weight_version {generation_weight_version}"
                )

                # Launch only the missing prompt groups for this target. Each
                # worker decrements ``_generating_target_counts[target_weight]``
                # by 1 in its ``finally``; if we fail to actually start a worker
                # (sema or thread error) we have to release that slot ourselves,
                # otherwise the reservation leaks.
                workers_started = 0
                try:
                    for _ in range(num_prompt_groups_to_launch):
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
                        repeated_batch = single_prompt_batch.repeat_interleave(
                            num_generations
                        )

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
                        workers_started += 1
                        prompt_idx += 1
                finally:
                    unspawned = num_prompt_groups_to_launch - workers_started
                    if unspawned > 0:
                        with self._generation_check_lock:
                            remaining = self._generating_target_counts.get(
                                target_weight, 0
                            )
                            new_remaining = remaining - unspawned
                            if new_remaining > 0:
                                self._generating_target_counts[target_weight] = (
                                    new_remaining
                                )
                            else:
                                self._generating_target_counts.pop(target_weight, None)
                        print(
                            f"🧹 Released {unspawned} unspawned reservation(s) for target weight {target_weight}"
                        )

            self._cleanup_finished_threads()

        except Exception as e:
            print(f"❌ Error processing batch: {e}")
            import traceback

            traceback.print_exc()

    def get_weight_version(self) -> int:
        return self.current_weight_version

    def check_health(self) -> None:
        """Raise the stored fatal worker error, if any.

        Called by the trainer between sampling iterations. When a generation
        worker has recorded a fatal failure (cumulative count exceeded
        max_generation_failures), this re-raises it so the training job dies
        instead of stalling on an empty replay buffer. Safe to call
        repeatedly: returns silently when no fatal error is set, and raises
        every time once one is.
        """
        with self._failure_lock:
            err = self._fatal_error
        if err is not None:
            raise err

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
        vllm_cfg = self.master_config.policy.get("generation", {}).get("vllm_cfg", {})
        is_async_engine = vllm_cfg.get("async_engine", False)
        in_flight_weight_updates = self.master_config.grpo.get("async_grpo", {}).get(
            "in_flight_weight_updates", False
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
        async_cfg = self.master_config.grpo.get("async_grpo", {})
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
    ) -> None:
        try:
            # Import here to avoid circular dependency
            from nemo_rl.algorithms.grpo import _should_use_nemo_gym
            from nemo_rl.experience.rollouts import run_async_nemo_gym_rollout

            # Run rollout for this prompt group
            # Async engine supports concurrent generation; avoid locking
            # Check if we should use nemo_gym (similar to synchronous GRPO)
            if _should_use_nemo_gym(self.master_config):
                generation_config = self.master_config.policy["generation"]
                nemo_gym_rollout_result = run_async_nemo_gym_rollout(
                    policy_generation=self.policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=self.tokenizer,
                    task_to_env=self.task_to_env,
                    max_seq_len=self.master_config.policy["max_total_sequence_length"],
                    generation_config=generation_config,
                    max_rollout_turns=None,
                    greedy=False,
                )
                final_batch = nemo_gym_rollout_result.final_batch
                rollout_metrics = nemo_gym_rollout_result.rollout_metrics
            else:
                final_batch, rollout_metrics = run_async_multi_turn_rollout(
                    policy_generation=self.policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=self.tokenizer,
                    task_to_env=self.task_to_env,
                    max_seq_len=self.master_config.policy["max_total_sequence_length"],
                    max_rollout_turns=self.master_config.grpo["max_rollout_turns"],
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
            backoff_delay = 0.01
            while self.running:
                status = ray.get(
                    self.replay_buffer.add.remote(
                        trajectory_group,
                        generation_weight_version,
                        target_weight_version,
                    )
                )
                if status == "success":
                    print(
                        f"📦 Buffered per-prompt group (prompt_idx {prompt_idx}, target_weight {target_weight_version})"
                    )
                    break
                elif status == "full":
                    # Exponential backoff up to 0.5 second
                    time.sleep(min(backoff_delay, 0.5))
                    backoff_delay *= 1.5
                else:
                    # Unexpected status, wait briefly
                    time.sleep(0.01)
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            with self._failure_lock:
                self._failure_count += 1
                count = self._failure_count
                threshold = self._max_generation_failures
                fatal = count > threshold
                if fatal and self._fatal_error is None:
                    self._fatal_error = RuntimeError(
                        f"AsyncTrajectoryCollector aborting: "
                        f"{count} generation-worker failure(s) exceeded "
                        f"max_generation_failures={threshold}. "
                        f"Last failure on prompt_idx={prompt_idx}, "
                        f"target_weight={target_weight_version}: {e!r}"
                    )
            print(
                f"[AsyncTrajectoryCollector] generation worker FAILED "
                f"(failure {count}, tolerating {threshold}) "
                f"prompt_idx={prompt_idx} "
                f"target_weight={target_weight_version} "
                f"generation_weight={generation_weight_version}\n{tb}",
                flush=True,
            )
            if fatal:
                print(
                    f"[AsyncTrajectoryCollector] FATAL: failure count {count} "
                    f"exceeds threshold {threshold}; trainer will be notified on "
                    f"next check_health() call.",
                    flush=True,
                )
        finally:
            # Track worker completion. Decrementing on every worker exit (success
            # or error) means a target with N missing prompt groups is freed up
            # for re-reservation only after all N workers have finished, while
            # still allowing the next _process_batch call to top up groups that
            # didn't make it into the buffer.
            with self._generation_check_lock:
                remaining = self._generating_target_counts.get(target_weight_version, 0)
                if remaining > 1:
                    self._generating_target_counts[target_weight_version] = (
                        remaining - 1
                    )
                elif remaining == 1:
                    self._generating_target_counts.pop(target_weight_version, None)
                    print(
                        f"🧹 Released reservation for target weight {target_weight_version}"
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
