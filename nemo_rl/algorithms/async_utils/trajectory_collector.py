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

import asyncio
import concurrent.futures
import threading as _threading
import time
from collections import defaultdict
from collections.abc import AsyncGenerator
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
from nemo_rl.experience.interfaces import (
    NEXT_NG_TASK_INDEX_KEY,
    NG_TASK_INDEX_KEY,
)
from nemo_rl.experience.rollouts import (
    RolloutGroupResult,
    run_async_multi_turn_rollout_groups,
)
from nemo_rl.models.generation.interfaces import GenerationInterface

TokenizerType = PreTrainedTokenizerBase
_MAX_NEMO_GYM_ROLLOUT_RETRIES = 3
_NEMO_GYM_RETRY_DELAY_BASE_SECONDS = 1.0


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
        next_ng_task_index: int = 0,
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

        # Simple lock to prevent race conditions when checking/spawning workers
        self._generation_check_lock: _threading.Lock = _threading.Lock()
        # Track which target weights are currently being generated (globally)
        self._generating_targets: set[int] = set()
        self._next_ng_task_index = next_ng_task_index

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
        target_weight: Optional[int] = None
        worker_started = False
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

            # Generate all prompt groups needed for this target in one batched worker.
            from nemo_rl.algorithms.grpo import _should_use_nemo_gym

            use_nemo_gym = _should_use_nemo_gym(self.master_config)

            if not self._refit_pause_cleared.is_set() and self.running:
                with self._threads_lock:
                    active_threads = len(self._inflight_threads)
                print(
                    "⏸️ Waiting for refit to complete before starting new "
                    f"generation ({active_threads} threads still active)"
                )
                self._refit_pause_cleared.wait()
                generation_weight_version = self.current_weight_version

            rollout_batch = batch.slice(0, num_prompts_to_generate)
            if use_nemo_gym:
                stamped_extra_env_info = []
                for offset, row in enumerate(rollout_batch["extra_env_info"]):
                    if not isinstance(row, dict):
                        raise TypeError(
                            "Expected NeMo-Gym extra_env_info row to be a dict, "
                            f"got {type(row)}"
                        )
                    stamped_row = dict(row)
                    stamped_row[NG_TASK_INDEX_KEY] = self._next_ng_task_index + offset
                    stamped_extra_env_info.append(stamped_row)
                rollout_batch["extra_env_info"] = stamped_extra_env_info
                self._next_ng_task_index += num_prompts_to_generate
            repeated_batch = rollout_batch.repeat_interleave(num_generations)

            def _run_rollout_batch() -> None:
                assert target_weight is not None
                asyncio.run(
                    self._run_rollout_batch_worker(
                        repeated_batch=repeated_batch,
                        generation_weight_version=generation_weight_version,
                        target_weight_version=target_weight,
                        num_generations=num_generations,
                        use_nemo_gym=use_nemo_gym,
                    )
                )

            worker = _threading.Thread(target=_run_rollout_batch, daemon=True)
            try:
                with self._threads_lock:
                    self._inflight_threads.add(worker)
                worker.start()
                worker_started = True
            except Exception:
                with self._threads_lock:
                    self._inflight_threads.discard(worker)
                self._release_target(target_weight)
                raise

            backend = "NeMo-Gym" if use_nemo_gym else "native"
            print(
                f"📊 Started one {backend} batch worker for "
                f"{num_prompts_to_generate} prompt groups at "
                f"target_weight={target_weight}"
            )

            self._cleanup_finished_threads()

        except Exception as e:
            if target_weight is not None and not worker_started:
                self._release_target(target_weight)
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

    def get_rollouts_state(self) -> dict[str, int]:
        """Get collector-side rollout state for checkpointing."""
        return {NEXT_NG_TASK_INDEX_KEY: self._next_ng_task_index}

    def _cleanup_finished_threads(self) -> None:
        with self._threads_lock:
            finished = {t for t in self._inflight_threads if not t.is_alive()}
            for t in finished:
                self._inflight_threads.remove(t)

    def _release_target(self, target_weight_version: int) -> None:
        """Release the reservation owned by a completed batch worker."""
        with self._generation_check_lock:
            if target_weight_version in self._generating_targets:
                self._generating_targets.discard(target_weight_version)
                print(
                    f"🧹 Released reservation for target weight {target_weight_version}"
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

    async def _iter_rollout_groups(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        num_generations: int,
        use_nemo_gym: bool,
        task_index_to_group_index: dict[int, int],
    ) -> AsyncGenerator[RolloutGroupResult, None]:
        """Yield prompt groups from either backend through one result type."""
        if use_nemo_gym:
            # Import here to keep the NeMo-Gym dependency local to its backend.
            from nemo_rl.experience.rollouts import (
                get_nemo_gym_thinking_tags,
                run_async_nemo_gym_rollout,
            )

            async for rollout_result in run_async_nemo_gym_rollout(
                policy_generation=self.policy_generation,
                input_batch=repeated_batch,
                tokenizer=self.tokenizer,
                task_to_env=self.task_to_env,
                max_seq_len=self.master_config.policy["max_total_sequence_length"],
                generation_config=self.master_config.policy["generation"],
                num_generations=num_generations,
                max_rollout_turns=None,
                greedy=False,
                reward_penalty_config=self.master_config.reward_penalties,
                thinking_tags=get_nemo_gym_thinking_tags(self.master_config.env),
            ):
                task_index = rollout_result.ng_task_index
                if task_index is None:
                    raise ValueError("NeMo-Gym prompt group is missing _ng_task_index")
                task_index = int(task_index)
                if task_index not in task_index_to_group_index:
                    raise ValueError(f"Unexpected _ng_task_index {task_index}")
                yield RolloutGroupResult(
                    group_index=task_index_to_group_index[task_index],
                    final_batch=rollout_result.final_batch,
                    rollout_metrics=rollout_result.rollout_metrics,
                    task_index=task_index,
                )
            return

        async for rollout_result in run_async_multi_turn_rollout_groups(
            policy_generation=self.policy_generation,
            input_batch=repeated_batch,
            tokenizer=self.tokenizer,
            task_to_env=self.task_to_env,
            max_seq_len=self.master_config.policy["max_total_sequence_length"],
            num_generations=num_generations,
            max_rollout_turns=self.master_config.grpo["max_rollout_turns"],
            greedy=False,
        ):
            yield rollout_result

    async def _run_rollout_batch_worker(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        generation_weight_version: int,
        target_weight_version: int,
        num_generations: int,
        use_nemo_gym: bool,
    ) -> None:
        """Own one target reservation while collecting its rollout batch."""
        try:
            await self._collect_rollout_batch(
                repeated_batch=repeated_batch,
                generation_weight_version=generation_weight_version,
                target_weight_version=target_weight_version,
                num_generations=num_generations,
                use_nemo_gym=use_nemo_gym,
            )
        except Exception as error:
            backend = "NeMo-Gym" if use_nemo_gym else "native"
            print(
                f"❌ Error in {backend} batch worker "
                f"(target_weight={target_weight_version}): {error}"
            )
            import traceback

            traceback.print_exc()
        finally:
            self._release_target(target_weight_version)
            with self._threads_lock:
                self._inflight_threads.discard(_threading.current_thread())

    async def _collect_rollout_batch(
        self,
        repeated_batch: BatchedDataDict[DatumSpec],
        generation_weight_version: int,
        target_weight_version: int,
        num_generations: int,
        use_nemo_gym: bool,
    ) -> None:
        """Run one backend batch and enqueue every completed prompt group."""
        if num_generations <= 0 or repeated_batch.size % num_generations != 0:
            raise ValueError(
                "Rollout batch size must be divisible by a positive num_generations"
            )
        expected_prompt_groups = repeated_batch.size // num_generations
        expected_group_indices = set(range(expected_prompt_groups))
        task_index_to_group_index: dict[int, int] = {}
        if use_nemo_gym:
            for group_index in range(expected_prompt_groups):
                start = group_index * num_generations
                rows = repeated_batch["extra_env_info"][start : start + num_generations]
                raw_task_indices = [row.get(NG_TASK_INDEX_KEY) for row in rows]
                if any(task_index is None for task_index in raw_task_indices):
                    raise ValueError(
                        "Every NeMo-Gym row must include _ng_task_index, got "
                        f"{raw_task_indices} for group {group_index}"
                    )
                task_indices = {int(task_index) for task_index in raw_task_indices}
                if len(task_indices) != 1:
                    raise ValueError(
                        "Expected one _ng_task_index per repeated prompt group, got "
                        f"{sorted(task_indices)} for group {group_index}"
                    )
                task_index = task_indices.pop()
                if task_index in task_index_to_group_index:
                    raise ValueError(f"Duplicate _ng_task_index {task_index}")
                task_index_to_group_index[task_index] = group_index

        buffered_group_indices: set[int] = set()

        async def _enqueue_rollout(rollout_result: RolloutGroupResult) -> None:
            group_index = rollout_result.group_index
            if group_index not in expected_group_indices:
                raise ValueError(f"Unexpected prompt group index {group_index}")

            final_batch_cpu = rollout_result.final_batch.to("cpu")
            rollout_metrics = rollout_result.rollout_metrics
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
                    teacher_logprobs, teacher_logprob_time = await asyncio.to_thread(
                        self._compute_teacher_logprobs,
                        flat_for_teacher["token_ids"],
                        agent_refs,
                        input_lengths=teacher_input_lengths,
                    )
                    final_batch_cpu["teacher_reference_logprobs"] = teacher_logprobs
                    rollout_metrics = dict(rollout_metrics)
                    rollout_metrics["teacher_logprob_time"] = teacher_logprob_time

            trajectory_group = {
                "batch": final_batch_cpu,
                "rollout_metrics": rollout_metrics,
                "timestamp": time.time(),
            }
            if rollout_result.task_index is not None:
                trajectory_group[NG_TASK_INDEX_KEY] = rollout_result.task_index

            backoff_delay = 0.01
            while self.running:
                status = await self.replay_buffer.add.remote(
                    trajectory_group,
                    generation_weight_version,
                    target_weight_version,
                )
                if status == "success":
                    buffered_group_indices.add(group_index)
                    group_description = f"group_index={group_index}"
                    if rollout_result.task_index is not None:
                        group_description = (
                            f"_ng_task_index={rollout_result.task_index}"
                        )
                    print(
                        "📦 Buffered prompt group "
                        f"({group_description}, "
                        f"target_weight={target_weight_version}) "
                        f"[{len(buffered_group_indices)}/"
                        f"{expected_prompt_groups} buffered]"
                    )
                    return
                if status == "full":
                    await asyncio.sleep(min(backoff_delay, 0.5))
                    backoff_delay *= 1.5
                else:
                    await asyncio.sleep(0.01)

            raise RuntimeError("Trajectory collection stopped before enqueue completed")

        last_error: Exception | None = None
        max_retries = _MAX_NEMO_GYM_ROLLOUT_RETRIES if use_nemo_gym else 0
        for retry_count in range(max_retries + 1):
            push_tasks: list[asyncio.Task[None]] = []
            stream_error = None
            try:
                async for rollout_result in self._iter_rollout_groups(
                    repeated_batch=repeated_batch,
                    num_generations=num_generations,
                    use_nemo_gym=use_nemo_gym,
                    task_index_to_group_index=task_index_to_group_index,
                ):
                    if rollout_result.group_index in buffered_group_indices:
                        continue
                    push_tasks.append(
                        asyncio.create_task(_enqueue_rollout(rollout_result))
                    )
            except Exception as error:
                stream_error = error

            push_results = await asyncio.gather(*push_tasks, return_exceptions=True)
            push_errors = [
                result for result in push_results if isinstance(result, Exception)
            ]
            pending_group_indices = expected_group_indices - buffered_group_indices
            if not pending_group_indices:
                return

            last_error = stream_error or (push_errors[0] if push_errors else None)
            if retry_count >= max_retries or not self.running:
                break

            retry_delay = _NEMO_GYM_RETRY_DELAY_BASE_SECONDS * (2**retry_count)
            print(
                "❌ NeMo-Gym batch did not complete prompt groups "
                f"{sorted(pending_group_indices)}; retrying in "
                f"{retry_delay:.1f}s "
                f"(attempt {retry_count + 1}/{max_retries})"
            )
            await asyncio.sleep(retry_delay)

        batch_error = RuntimeError(
            "Rollout batch failed to buffer prompt groups "
            f"{sorted(expected_group_indices - buffered_group_indices)}"
        )
        if last_error is not None:
            raise batch_error from last_error
        raise batch_error
