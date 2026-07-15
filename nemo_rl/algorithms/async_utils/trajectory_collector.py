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

from __future__ import annotations

import concurrent.futures
import threading as _threading
import time
from collections import defaultdict
from typing import Any, Optional

import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.algorithms.opd import resolve_reference_aliases, teacher_seq_pad_multiple
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.utils.timer import ThreadSafeTimer

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
        teacher_worker_groups: Optional[dict[str, Any]] = None,
        alias_to_group_alias: Optional[dict[str, str]] = None,
        on_policy_distillation_cfg: Optional[dict[str, Any]] = None,
    ):
        self.policy_generation = policy_generation
        self.tokenizer = tokenizer
        self.task_to_env = task_to_env
        self.master_config = master_config
        self.replay_buffer = replay_buffer
        self.teacher_worker_groups = teacher_worker_groups or {}
        self.alias_to_group_alias = alias_to_group_alias or {}
        self.on_policy_distillation_cfg = on_policy_distillation_cfg or {}
        self._has_distillation_teachers = bool(self.teacher_worker_groups)
        self._teacher_seq_pad_multiple = teacher_seq_pad_multiple(
            self.teacher_worker_groups,
            self.master_config.policy["make_sequence_length_divisible_by"],
        )
        # Per-teacher locks to serialize get_logprobs calls. Concurrent calls
        # to the same teacher cause NCCL collective desync across workers
        # (different workers may receive requests in different order → SeqNum
        # mismatch → 600s timeout → crash). Different teachers can still run
        # in parallel since they use separate NCCL groups on separate nodes.
        self._teacher_locks: dict[str, _threading.Lock] = {
            k: _threading.Lock() for k in self.teacher_worker_groups
        }
        self.running = False
        self.data_exhausted = False
        self.collection_failed = False

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
        # Track which target weights are currently being generated (globally)
        self._generating_targets: set[int] = set()
        # Track spawned, buffered, and completed prompt-group workers per target.
        self._spawned_per_target: dict[int, int] = {}
        self._buffered_per_target: dict[int, int] = {}
        self._completed_per_target: dict[int, int] = {}
        self._spawning_targets: set[int] = set()
        self._counter_lock: _threading.Lock = _threading.Lock()

        # Timer for efficiency metrics
        self._efficiency_timer = ThreadSafeTimer(context={"worker": "collector"})

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
        self, generation_weight_version: int
    ) -> Optional[int]:
        """Get the next target weight that needs generation (if any)."""
        target_weights = self._calculate_target_weights(generation_weight_version)
        num_prompts = int(self.master_config.grpo["num_prompts_per_step"])
        max_age_steps = int(
            self.master_config.grpo["async_grpo"]["max_trajectory_age_steps"]
        )
        last_consumed_target = ray.get(
            self.replay_buffer.get_last_target_weight_already_generated.remote()
        )

        with self._generation_check_lock:
            for target_weight in target_weights:
                if target_weight <= last_consumed_target:
                    continue
                if target_weight in self._generating_targets:
                    continue

                trajectories_needed = ray.get(
                    self.replay_buffer.get_trajectories_needed.remote(
                        target_weight, num_prompts, max_age_steps
                    )
                )
                if trajectories_needed <= 0:
                    continue

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
        """Check if collection should be paused due to generation limits."""
        try:
            target_weights = self._calculate_target_weights(self.current_weight_version)
            num_prompts = int(self.master_config.grpo["num_prompts_per_step"])
            max_age_steps = int(
                self.master_config.grpo["async_grpo"]["max_trajectory_age_steps"]
            )
            last_consumed_target = ray.get(
                self.replay_buffer.get_last_target_weight_already_generated.remote()
            )

            # Check if any target weight in our range needs generation
            with self._generation_check_lock:
                for target_weight in target_weights:
                    if target_weight <= last_consumed_target:
                        continue
                    if target_weight in self._generating_targets:
                        continue
                    trajectories_needed = ray.get(
                        self.replay_buffer.get_trajectories_needed.remote(
                            target_weight, num_prompts, max_age_steps
                        )
                    )
                    if trajectories_needed > 0:
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

    def is_data_exhausted(self) -> bool:
        """Check if collection stopped because the dataloader ran out of data."""
        return self.data_exhausted

    def get_status(self) -> dict:
        """Return a snapshot of the collector's internal state for driver-side diagnostics."""
        with self._threads_lock:
            inflight_workers = len(self._inflight_threads)
        return {
            "running": self.running,
            "data_exhausted": self.data_exhausted,
            "errored": self.collection_failed,
            "inflight_workers": inflight_workers,
        }

    def _collection_loop(self):
        """Run the collection loop in background thread."""
        dataloader_exhausted = False
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
                    with self._efficiency_timer.time("idle/refit_event_wait"):
                        self._refit_pause_cleared.wait()
                    print("▶️ Refit completed, resuming collection")

                # Check if generation limits require pausing collection
                if self._should_pause_for_generation_limits() and self.running:
                    # Only log warning once per weight version
                    if self._last_limit_warning_version != self.current_weight_version:
                        async_cfg = self.master_config.grpo.get("async_grpo", {})
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
                    with self._efficiency_timer.time("idle/generation_limit_pause"):
                        self._generation_limit_cleared.wait()

                    # Double-check we're still running after being woken up
                    if not self.running:
                        break

                if not self.running:
                    break

                self._process_batch(batch)
            else:
                # for-loop completed without break → dataloader iterator exhausted
                dataloader_exhausted = True

        except Exception as e:
            print(f"❌ Error in trajectory collection: {e}")
            import traceback

            traceback.print_exc()
            self.collection_failed = True
        finally:
            self.running = False
            if dataloader_exhausted:
                self.data_exhausted = True
                print(
                    "❌ Trajectory collection stopped: dataloader exhausted "
                    "(max_num_epochs reached). No more data available for generation. "
                    "Increase max_num_epochs or use a larger dataset."
                )
            else:
                print("🛑 Trajectory collection stopped")

    def _process_batch(self, batch: BatchedDataDict[DatumSpec]) -> None:
        """Process a single batch and generate for one target weight."""
        target_weight: Optional[int] = None
        try:
            generation_weight_version = self.current_weight_version
            num_generations = self.master_config.grpo["num_generations_per_prompt"]
            num_prompts_in_batch = batch.size
            num_prompts_per_step = int(self.master_config.grpo["num_prompts_per_step"])
            max_age_steps = int(
                self.master_config.grpo["async_grpo"]["max_trajectory_age_steps"]
            )

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

            trajectories_needed = ray.get(
                self.replay_buffer.get_trajectories_needed.remote(
                    target_weight, num_prompts_per_step, max_age_steps
                )
            )
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
                    f"generating {num_prompts_to_generate}/{num_prompts_in_batch} "
                    f"prompts (need {trajectories_needed} more trajectories)"
                )

            # Generate only the prompt groups needed for this target. While the
            # spawn loop is open, workers may finish before later workers start,
            # so reservation release is deferred until spawning closes.
            started = 0
            with self._counter_lock:
                self._spawning_targets.add(target_weight)
            try:
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
                        with self._efficiency_timer.time("idle/refit_event_wait"):
                            self._refit_pause_cleared.wait()

                        # After refit finishes if weight version has updated, reflect that in the new trajectories
                        generation_weight_version = self.current_weight_version

                    single_prompt_batch = batch.slice(prompt_idx, prompt_idx + 1)
                    repeated_batch = single_prompt_batch.repeat_interleave(
                        num_generations
                    )

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
                    self._inflight_sema.acquire()
                    registered = False
                    try:
                        with self._threads_lock:
                            self._inflight_threads.add(worker)
                        with self._counter_lock:
                            self._spawned_per_target[target_weight] = (
                                self._spawned_per_target.get(target_weight, 0) + 1
                            )
                            spawned_count = self._spawned_per_target[target_weight]
                        registered = True
                        worker.start()
                    except Exception:
                        # The worker never ran, so it won't release its slot or
                        # run its finally block; undo the bookkeeping here.
                        with self._threads_lock:
                            self._inflight_threads.discard(worker)
                        if registered:
                            with self._counter_lock:
                                updated_count = (
                                    self._spawned_per_target.get(target_weight, 0) - 1
                                )
                                if updated_count > 0:
                                    self._spawned_per_target[target_weight] = (
                                        updated_count
                                    )
                                else:
                                    self._spawned_per_target.pop(target_weight, None)
                        self._inflight_sema.release()
                        raise
                    started += 1
                    print(
                        f"📊 Started worker {started}/{num_prompts_to_generate} for "
                        f"target_weight={target_weight} ({spawned_count} total)"
                    )
            finally:
                if started < num_prompts_to_generate:
                    print(
                        f"⚠️ Only {started}/{num_prompts_to_generate} workers "
                        f"started for target_weight={target_weight}"
                    )
                with self._counter_lock:
                    self._spawning_targets.discard(target_weight)
                self._maybe_release_target(target_weight)

            self._cleanup_finished_threads()

        except Exception as e:
            if target_weight is not None:
                with self._counter_lock:
                    self._spawning_targets.discard(target_weight)
                self._maybe_release_target(target_weight)
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

        For backends with an async engine in-flight weight updates allows ongoing generations
        to continue with their current KV caches while weights are updated.
        This significantly improves async performance.

        For non-async engines, waits for all pending generations to complete before refit.
        """
        start_time = time.time()
        print("🔄 Preparing for refit: pausing new generations...")

        # Pause new generation starts
        self._refit_pause_cleared.clear()
        print("⏸️ New generation starts paused")

        # Check if we're using async engine
        generation_cfg = self.master_config.policy.get("generation", {})
        backend = generation_cfg.get("backend", "")
        if backend == "vllm":
            is_async_engine = generation_cfg.get("vllm_cfg", {}).get(
                "async_engine", False
            )
        elif backend == "megatron":
            is_async_engine = generation_cfg.get("mcore_generation_config", {}).get(
                "async_engine", False
            )
        elif backend == "trtllm":
            assert generation_cfg["trtllm_cfg"]["async_engine"], (
                "TRT-LLM backend requires trtllm_cfg.async_engine=true; the "
                "synchronous engine path (async_engine=false) is no longer supported."
            )
            is_async_engine = True
        else:
            is_async_engine = False
        in_flight_weight_updates = self.master_config.grpo.get("async_grpo", {}).get(
            "in_flight_weight_updates", False
        )

        if is_async_engine and in_flight_weight_updates:
            # async engines support in-flight weight updates
            # Ongoing generations will continue with their current KV caches
            # New generations (after weight update) will use the updated weights
            print(
                f"🚀 Using {backend} in-flight weight update - skipping wait for pending generations"
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

    def get_efficiency_metrics(self) -> dict[str, float]:
        """Return accumulated efficiency metrics (sum of durations per category).

        Called by the driver process each step to merge collector-side metrics.
        """
        return self._efficiency_timer.get_timing_metrics(reduction_op="sum")

    def _cleanup_finished_threads(self) -> None:
        with self._threads_lock:
            finished = {t for t in self._inflight_threads if not t.is_alive()}
            for t in finished:
                self._inflight_threads.remove(t)

    def _maybe_release_target(self, target_weight_version: int) -> None:
        """Release a target's reservation once all its workers have completed.

        A worker counts as "completed" whether or not it managed to buffer a
        trajectory. The reservation is released exactly when the number of
        completed workers reaches the number spawned for the target and no more
        workers are being spawned for the same target. Safe to call repeatedly:
        the reservation is discarded at most once and the per-target counters
        are dropped on release (so a later re-reservation starts from a clean
        slate and the dicts don't grow unbounded).
        """
        with self._counter_lock:
            if target_weight_version in self._spawning_targets:
                return
            completed = self._completed_per_target.get(target_weight_version, 0)
            spawned = self._spawned_per_target.get(target_weight_version, 0)
            if completed < spawned:
                return
            buffered = self._buffered_per_target.get(target_weight_version, 0)
            self._spawned_per_target.pop(target_weight_version, None)
            self._buffered_per_target.pop(target_weight_version, None)
            self._completed_per_target.pop(target_weight_version, None)
            self._spawning_targets.discard(target_weight_version)

        with self._generation_check_lock:
            if target_weight_version in self._generating_targets:
                self._generating_targets.discard(target_weight_version)
                print(
                    "🧹 Released reservation for target weight "
                    f"{target_weight_version} ({spawned} workers completed, "
                    f"{buffered} buffered)"
                )

    def _compute_teacher_logprobs(
        self,
        input_ids: torch.Tensor,
        agent_refs: list[dict[str, Any]],
        input_lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, float]:
        """Compute teacher logprobs for non-colocated teachers.

        Groups samples by teacher, fans out in parallel, stitches results.

        Args:
            input_ids: [B, S] tokenized input tensor
            agent_refs: list of B agent reference dicts
            input_lengths: [B] per-sample lengths (required for sequence packing)

        Returns:
            ([B, S] teacher logprobs tensor, total_time_seconds)
        """
        opd_cfg = self.on_policy_distillation_cfg
        teacher_model_by_agent_name = opd_cfg.get("teacher_model_by_agent_name", {})
        default_teacher_alias = opd_cfg.get("default_teacher_alias")
        strict = opd_cfg.get("strict_agent_name_match", False)

        # Resolve each sample's agent -> the teacher alias it should be distilled
        # from: the agent name is looked up in teacher_model_by_agent_name; unmapped
        # agents fall back to default_teacher_alias (or raise if strict_agent_name_match).
        # Returns one alias per sample, index-aligned with agent_refs.
        reference_aliases = resolve_reference_aliases(
            agent_refs,
            teacher_model_by_agent_name,
            default_teacher_alias=default_teacher_alias,
            strict_agent_name_match=strict,
        )

        # Map aliases to actual group keys via deduplication mapping
        group_keys = [self.alias_to_group_alias.get(a, a) for a in reference_aliases]

        # Group sample indices by teacher group
        group_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, gk in enumerate(group_keys):
            group_to_indices[gk].append(i)

        B, S = input_ids.shape
        result = torch.zeros(B, S, dtype=torch.float32)
        if (
            not group_to_indices
        ):  # 0-sample batch: nothing to route (avoid max_workers=0)
            return result, 0.0

        def _get_logprobs_for_group(group_key, indices):
            twg = self.teacher_worker_groups[group_key]
            sub_input_ids = input_ids[indices]
            sub_lengths = input_lengths[indices] if input_lengths is not None else None

            # Pad batch to multiple of dp_size (required for DP sharding)
            dp_size = twg.sharding_annotations.get_axis_size("data_parallel")
            actual_batch_size = sub_input_ids.shape[0]
            remainder = actual_batch_size % dp_size
            if remainder != 0:
                pad_count = dp_size - remainder
                # Repeat last row to fill — can't slice [:pad_count] when
                # actual_batch_size < pad_count (e.g., 1 sample, dp_size=4)
                pad_rows = sub_input_ids[-1:].expand(pad_count, -1)
                sub_input_ids = torch.cat([sub_input_ids, pad_rows], dim=0)
                if sub_lengths is not None:
                    sub_lengths = torch.cat(
                        [sub_lengths, sub_lengths[-1:].expand(pad_count)], dim=0
                    )

            sub_data = BatchedDataDict({"input_ids": sub_input_ids})
            if sub_lengths is not None:
                sub_data["input_lengths"] = sub_lengths

            # Serialize calls per teacher to prevent NCCL collective desync
            t_lock_start = time.time()
            with self._teacher_locks[group_key]:
                t_inference_start = time.time()
                logprobs_result = twg.get_logprobs(sub_data)
            t_done = time.time()
            lock_wait = t_inference_start - t_lock_start
            inference_time = t_done - t_inference_start
            print(
                f"[teacher_logprob] group={group_key} samples={actual_batch_size} "
                f"lock_wait={lock_wait:.2f}s inference={inference_time:.2f}s"
            )
            logprobs = logprobs_result["reference_logprobs"]

            # Trim DP padding
            logprobs = logprobs[:actual_batch_size]

            return indices, logprobs

        # Fan out to teachers in parallel
        t_total_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(group_to_indices)
        ) as executor:
            futures = {
                executor.submit(_get_logprobs_for_group, gk, idxs): gk
                for gk, idxs in group_to_indices.items()
            }
            for future in concurrent.futures.as_completed(futures):
                indices, logprobs = future.result()
                result[indices] = logprobs
        total_time = time.time() - t_total_start
        print(
            f"[teacher_logprob] total={total_time:.2f}s for {B} samples across {len(group_to_indices)} teacher(s)"
        )

        return result, total_time

    def _run_prompt_group_worker(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        generation_weight_version: int,
        target_weight_version: int,
        prompt_idx: int,
    ) -> None:
        worker_start = time.perf_counter()
        try:
            # Import here to avoid circular dependency
            from nemo_rl.algorithms.grpo import _should_use_nemo_gym
            from nemo_rl.experience.rollouts import (
                get_nemo_gym_thinking_tags,
                run_async_nemo_gym_rollout,
            )

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
                    reward_penalty_config=self.master_config.reward_penalties,
                    thinking_tags=get_nemo_gym_thinking_tags(self.master_config.env),
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

            # Compute teacher logprobs at collection time (overlapped with async rollouts)
            if self._has_distillation_teachers and "agent_ref" in final_batch_cpu:
                agent_refs = final_batch_cpu["agent_ref"]
                if isinstance(agent_refs, list):
                    from nemo_rl.data.llm_message_utils import (
                        batched_message_log_to_flat_message,
                    )

                    flat_for_teacher, teacher_input_lengths = (
                        batched_message_log_to_flat_message(
                            final_batch_cpu["message_log"],
                            pad_value_dict={"token_ids": self.tokenizer.pad_token_id},
                            make_sequence_length_divisible_by=self._teacher_seq_pad_multiple,
                        )
                    )
                    teacher_logprobs, teacher_logprob_time = (
                        self._compute_teacher_logprobs(
                            flat_for_teacher["token_ids"],
                            agent_refs,
                            input_lengths=teacher_input_lengths,
                        )
                    )
                    # Store inside batch dict so from_batches handles
                    # variable-length padding across prompt groups
                    final_batch_cpu["teacher_reference_logprobs"] = teacher_logprobs
                    rollout_metrics = dict(rollout_metrics)
                    rollout_metrics["teacher_logprob_time"] = teacher_logprob_time

            # Record per-trajectory wall-clock duration for buffer starvation diagnostics
            trajectory_duration_s = time.perf_counter() - worker_start
            rollout_metrics = dict(rollout_metrics)
            rollout_metrics["trajectory_duration_s"] = trajectory_duration_s

            trajectory_group = {
                "batch": final_batch_cpu,
                "rollout_metrics": rollout_metrics,
                "timestamp": time.time(),
            }

            # Use exponential backoff when buffer is full
            try:
                backoff_delay = 0.01
                backoff_start = None
                while self.running:
                    status = ray.get(
                        self.replay_buffer.add.remote(
                            trajectory_group,
                            generation_weight_version,
                            target_weight_version,
                        )
                    )
                    if status == "success":
                        with self._counter_lock:
                            self._buffered_per_target[target_weight_version] = (
                                self._buffered_per_target.get(target_weight_version, 0)
                                + 1
                            )
                            buffered_count = self._buffered_per_target[
                                target_weight_version
                            ]
                            spawned_count = self._spawned_per_target.get(
                                target_weight_version, 0
                            )
                        if backoff_start is not None:
                            self._efficiency_timer.record(
                                "idle/buffer_full_backoff",
                                time.perf_counter() - backoff_start,
                            )
                        print(
                            f"📦 Buffered per-prompt group (prompt_idx {prompt_idx}, "
                            f"target_weight {target_weight_version}) "
                            f"[{buffered_count}/{spawned_count} buffered]"
                        )
                        break
                    elif status == "full":
                        if backoff_start is None:
                            backoff_start = time.perf_counter()
                        # Exponential backoff up to 0.5 second
                        time.sleep(min(backoff_delay, 0.5))
                        backoff_delay *= 1.5
                    else:
                        # Unexpected status, wait briefly
                        time.sleep(0.01)
            except Exception as e:
                print(f"❌ Failed to enqueue per-prompt group to buffer: {e}")
                if backoff_start is not None:
                    self._efficiency_timer.record(
                        "idle/buffer_full_backoff",
                        time.perf_counter() - backoff_start,
                    )
                import traceback

                traceback.print_exc()
        except Exception as e:
            print(f"❌ Error in prompt group worker: {e}")
            self._efficiency_timer.record(
                "wasted/failed_trajectory", time.perf_counter() - worker_start
            )
            import traceback

            traceback.print_exc()
        finally:
            with self._counter_lock:
                self._completed_per_target[target_weight_version] = (
                    self._completed_per_target.get(target_weight_version, 0) + 1
                )
            self._maybe_release_target(target_weight_version)

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
