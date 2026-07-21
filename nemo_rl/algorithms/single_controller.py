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

"""SingleController: asyncio orchestrator for the RL training loop.

CPU-only Ray actor that runs two concurrent pumps and coordinates the
other actors via lightweight RPCs. SC sends control signals and reads
metadata only — model tensors still move through DataPlane or NCCL.

Data flow:
  _rollout_pump  → rollout_manager.generate_and_push(prompt)
                     → TQReplayBuffer.reserve claims a slot at dispatch time;
                       run_rollout runs; TQReplayBuffer.commit tensorizes the
                       record, writes N training rows to TQ, and marks ready.
  _train_pump    → sampler.evict → buffer.remove (stale groups, with DP clear).
                 → sampler.select → drops chosen groups from buffer, returns
                   KVBatchMeta of K groups (or None); meta is already trainable.
                 → _advantage_pump (get → compute → put).
                 → trainer.train_on_meta.
                 → dp_client.clear_samples (trained groups; buffer already dropped).
  _sync_weights  → drain _inflight_rollouts → WeightSynchronizer.sync_weights.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
import warnings
from typing import Any, Optional, Union, cast

import ray
import torch

from nemo_rl.algorithms.async_utils.staleness_sampler import StalenessSampler
from nemo_rl.algorithms.grpo import GRPOSaveState
from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig,
    MasterConfig,
    WeightSyncConfig,
)
from nemo_rl.algorithms.single_controller_utils.setup import SingleControllerBundle
from nemo_rl.algorithms.single_controller_utils.utils import (
    aggregate_step_metrics,
    fields_for_put,
    reduce_advantage_pump_metrics,
    squeeze_trailing_unit_dim,
    tensor_field,
)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import get_keys_from_message_log
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.experience.interfaces import PromptGroupRecord
from nemo_rl.experience.rollout_checkpoint import (
    RolloutWorkItem,
    compute_rollout_fingerprint,
)
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy.tq_policy import TQPolicy
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.logger import Logger, print_message_log_samples
from nemo_rl.utils.timer import TimeoutChecker, Timer

Generation = Union[VllmGeneration, SGLangGeneration]


@ray.remote(num_cpus=1, num_gpus=0)  # pragma: no cover
class SingleControllerActor:
    """CPU-only Ray actor that orchestrates the RL training loop.

    Owns two concurrent asyncio tasks:
      - _rollout_pump: dispatches prompts to GenerationWorkerActor
      - _train_pump:   claims DataPlane meta, trains, clears consumed rows,
                       then runs _sync_weights (drain gate + weight
                       synchronization) inline after each optimizer step

    All other actors are passive — they expose methods and wait to be called.
    """

    def __init__(
        self,
        master_config: MasterConfig,
        bundle: SingleControllerBundle,
    ) -> None:
        """Initialize the SingleController actor.

        Args:
            master_config: SC MasterConfig.
            bundle: Pre-built bundle from setup_single_controller. Tests can
                construct a bundle by hand (or with fakes) to bypass the real factories.
        """
        self._advantage_cfg = AdvantageConfig()
        self._weight_sync_cfg = WeightSyncConfig()
        self._partition_id: str = bundle.partition_id
        self._diagnostics: bool = False

        self._master_config = master_config
        self._async_cfg = master_config.async_rl
        self._dp_client = bundle.dp_client
        self._gen: Generation = bundle.gen_handle
        self._trainer: TQPolicy = bundle.trainer_handle
        self._dataloader = bundle.dataloader
        self._val_dataloader = bundle.val_dataloader
        self._weight_synchronizer = bundle.weight_synchronizer
        self._advantage_estimator = bundle.advantage_estimator
        self._loss_fn = bundle.loss_fn
        self._buffer = bundle.tq_buffer
        self._rollout_manager = bundle.rollout_manager
        # Rebind so writer and sampler share one buffer instance even
        # when Ray deserializes rollout_manager and tq_buffer separately.
        self._rollout_manager._tq_buffer = self._buffer
        self._rollout_checkpoint = bundle.rollout_checkpoint
        checkpointing_enabled = master_config.rollout_checkpointing.enabled
        if checkpointing_enabled != (self._rollout_checkpoint is not None):
            raise ValueError(
                "SingleController rollout checkpoint configuration and runtime "
                "bundle do not agree"
            )

        # Built here, not on the driver: Logger backends (wandb/tb/...) hold
        # _thread.lock that Ray can't cloudpickle into the actor.
        self._logger = Logger(master_config.logger)  # type: ignore
        self._timer = Timer()

        # Built here, not on the driver: TimeoutChecker must capture wall-clock
        # start times inside the actor, not at driver setup time. The bundle
        # only carries the driver-side restore products (save_state,
        # last_checkpoint_path).
        self._checkpointer = CheckpointManager(master_config.checkpointing)
        self._timeout = TimeoutChecker(
            timeout=master_config.checkpointing["checkpoint_must_save_by"],
            fit_last_save_time=True,
        )
        self._timeout.start_iterations()

        # Loaded (or default) GRPOSaveState; keys SC does not own
        # (val_reward, ...) pass through to saved checkpoints untouched.
        self._save_state: GRPOSaveState = bundle.save_state
        self._last_checkpoint_path: Optional[str] = bundle.last_checkpoint_path
        self._consumed_samples: int = bundle.save_state["consumed_samples"]
        self._current_epoch: int = bundle.save_state["current_epoch"]
        self._total_valid_tokens: int = bundle.save_state.get(
            "total_valid_tokens", 0
        )  # Default to 0 for backward compatibility with older checkpoints

        # Pin clusters so RayVirtualCluster.__del__ doesn't remove the PGs.
        self._train_cluster = bundle.train_cluster
        self._inference_cluster = bundle.inference_cluster

        num_prompts_per_step = self._master_config.grpo["num_prompts_per_step"]
        if num_prompts_per_step < self._async_cfg.min_prompt_groups_per_batch:
            raise ValueError(
                f"grpo.num_prompts_per_step ({num_prompts_per_step}) "
                f"must be >= async_rl.min_prompt_groups_per_batch "
                f"({self._async_cfg.min_prompt_groups_per_batch})"
            )

        if self._async_cfg.batch_selection_strategy == "strict_on_policy":
            self._async_cfg.max_weight_staleness_versions = 0
            self._async_cfg.over_sampling = False
            print(
                "Using strict_on_policy, auto setting max_weight_staleness_versions to 0 and over_sampling to False.",
                flush=True,
            )

        if not self._async_cfg.over_sampling:
            expected_buffer = num_prompts_per_step * (
                self._async_cfg.max_weight_staleness_versions + 1
            )
            if self._async_cfg.max_buffered_rollouts != expected_buffer:
                raise ValueError(
                    f"over_sampling=False requires max_buffered_rollouts "
                    f"({self._async_cfg.max_buffered_rollouts}) == "
                    f"num_prompts_per_step * (max_weight_staleness_versions + 1) "
                    f"({expected_buffer})"
                )

        if self._async_cfg.force_in_order and self._async_cfg.over_sampling:
            raise ValueError(
                "force_in_order=True requires over_sampling=False so that each "
                "dispatched batch corresponds to exactly one target training step."
            )

        # SC split path does one optimizer.step per RL step.
        # TODO: support multi-mini-step (legacy train() does gbs-sized
        # mini-steps with shared prev_logprobs).
        rl_step_samples = (
            num_prompts_per_step
            * self._master_config.grpo["num_generations_per_prompt"]
        )
        train_gbs = self._master_config.policy["train_global_batch_size"]
        if rl_step_samples != train_gbs:
            raise ValueError(
                f"num_prompts_per_step * num_generations_per_prompt "
                f"({rl_step_samples}) must equal policy.train_global_batch_size "
                f"({train_gbs}) so that one RL step maps to exactly one "
                f"optimizer.step. Multi-mini-step inside a single RL step is "
                f"not supported on the SC split path."
            )

        self._sampler = StalenessSampler(
            self._buffer,
            max_staleness_versions=self._async_cfg.max_weight_staleness_versions,
            require_order=not self._async_cfg.over_sampling,
            force_in_order=self._async_cfg.force_in_order,
        )

        # ── asyncio state ──────────────────────────────────────────────────
        # Gate: cleared during weight sync or validation-only rollout draining,
        # set when generation may proceed.
        self._rollout_permitted: asyncio.Event = asyncio.Event()
        self._rollout_permitted.set()

        # Count of in-flight generate_and_push calls
        self._inflight_rollouts: int = 0

        # Cancellation handles for in-flight rollout dispatches.
        self._dispatched_rollouts: set[asyncio.Task[None]] = set()

        # Metadata-only work ledger. Failed entries remain available for the
        # later fence/replace/redispatch recovery slice.
        self._checkpoint_inflight_work: dict[str, RolloutWorkItem] = {}
        self._next_rollout_dispatch_sequence: int = 0

        # over_sampling=False batch gate: farthest trainer_version covered by
        # already-dispatched batches. Restored as current_step - 1 to preserve
        # the fresh-start invariant _max_rollout_version == _trainer_version - 1
        # (the quota gate and force_in_order target_step matching rely on it).
        self._max_rollout_version: int = bundle.save_state["current_step"] - 1

        # Backpressure valve: max unconsumed rollout groups allowed in DataPlane.
        # Acquired before each rollout dispatch; released when the buffer
        # drops a group (sampler.evict or post-train buffer.remove).
        self._buffer_capacity: asyncio.Semaphore = asyncio.Semaphore(
            self._async_cfg.max_buffered_rollouts
        )

        self._trainer_version: int = bundle.save_state["current_step"]
        self._train_steps: int = bundle.save_state["current_step"]
        self._step_log_dict: dict[str, list] = {
            "rewards": [],
            "masked_advantages": [],
            "sequence_lengths": [],
        }

        print(
            f"SingleControllerActor: "
            f"staleness_cap={self._async_cfg.max_weight_staleness_versions} "
            f"buffer={self._async_cfg.max_buffered_rollouts} "
            f"inflight={self._async_cfg.max_inflight_prompts} "
            f"over_sampling={self._async_cfg.over_sampling} "
            f"transport={self._weight_sync_cfg.transport}",
            flush=True,
        )

    # ── public API ─────────────────────────────────────────────────────────

    async def run(self) -> dict[str, Any]:
        """Main entry point. Runs until max_train_steps is reached."""
        # Synchronize weights before starting the pumps
        await self._sync_weights()

        if self._master_config.grpo["val_at_start"] and self._train_steps == 0:
            await self._run_validation(step=0)

        # Restore committed rollout groups from the previous run. Only when
        # over_sampling=True — with the strict quota the restored window can
        # never be completed (see the checkpointing plan's design note).
        if self._async_cfg.over_sampling and self._last_checkpoint_path is not None:
            buffer_path = os.path.join(self._last_checkpoint_path, "replay_buffer.pt")
            if os.path.exists(buffer_path):
                print(f"📦 Restoring replay buffer from checkpoint: {buffer_path}")
                # weights_only=False: groups hold pickled KVBatchMeta/TensorDicts,
                # not plain tensors. The checkpoint is a trusted same-job artifact.
                buffer_state = await asyncio.to_thread(
                    torch.load, buffer_path, weights_only=False
                )
                restored = await self._buffer.load_state_dict(
                    buffer_state,
                    max_groups=self._async_cfg.max_buffered_rollouts,
                    expected_partition_id=self._partition_id,
                    expected_group_size=self._master_config.grpo[
                        "num_generations_per_prompt"
                    ],
                )
                # Each buffered group holds one _buffer_capacity permit. The
                # load truncation guarantees restored <= capacity, so these
                # acquisitions never wait — blocking here would hang run()
                # forever (no pump is running yet to release permits). Any
                # restored group now outside the staleness window is dropped
                # by the train pump's first sampler.evict, which releases its
                # permit — load does not stale-filter.
                assert restored <= self._async_cfg.max_buffered_rollouts
                for _ in range(restored):
                    await self._buffer_capacity.acquire()
            else:
                print(
                    f"⚠️ No replay buffer checkpoint found at {buffer_path}. "
                    "Starting with an empty replay buffer."
                )

        # Start both pumps and supervise either one failing. Waiting only for
        # the train pump would hang if rollout generation stopped producing
        # consumable groups.
        rollout_task = asyncio.create_task(self._rollout_pump())
        train_task = asyncio.create_task(self._train_pump())
        try:
            done, _ = await asyncio.wait(
                {rollout_task, train_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if rollout_task in done:
                # A finite rollout pump may complete normally before the train
                # pump consumes its final groups. Only an exception is fatal.
                await rollout_task
                await train_task
            else:
                await train_task
        finally:
            for task in (rollout_task, train_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(rollout_task, train_task, return_exceptions=True)
            self._logger.finish()

        return {
            "train_steps": self._train_steps,
            "trainer_version": self._trainer_version,
        }

    async def ping(self) -> dict[str, Any]:
        """Liveness check — returns immediately if event loop is running."""
        return {
            "alive": True,
            "trainer_version": self._trainer_version,
            "train_steps": self._train_steps,
            "inflight_rollouts": self._inflight_rollouts,
            "rollout_permitted": self._rollout_permitted.is_set(),
            "checkpoint_inflight_groups": len(self._checkpoint_inflight_work),
            "epoch": self._current_epoch,
        }

    # ── internal helpers ───────────────────────────────────────────────────

    async def _ray_get(self, obj_ref: Any) -> Any:
        """Await a Ray ObjectRef without blocking the asyncio event loop."""
        return await obj_ref

    async def _call_dp(self, method_name: str, **kwargs) -> Any:
        """Call a DataPlaneClient method or a Ray actor exposing that method."""
        method = getattr(self._dp_client, method_name)
        remote = getattr(method, "remote", None)
        if remote is not None:
            return await self._ray_get(remote(**kwargs))
        result = method(**kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _run_validation(self, *, step: int) -> dict[str, Any]:
        """Run one validation pass against the synchronized generator."""
        if self._val_dataloader is None:
            raise RuntimeError(
                "SingleController validation resources were not initialized."
            )

        max_val_samples = cast(
            Optional[int], self._master_config.grpo["max_val_samples"]
        )

        timer = Timer()
        rewards: list[float] = []
        lengths: list[float] = []
        message_logs: list[Any] = []

        print(f"▶ Starting validation at step {step}...", flush=True)
        with timer.time("total_validation_time"):
            for val_batch in self._val_dataloader:
                if max_val_samples is None:
                    selected_rows = val_batch.size
                else:
                    remaining = max_val_samples - len(rewards)
                    if remaining <= 0:
                        break

                    selected_rows = min(val_batch.size, remaining)
                prompts: list[DatumSpec] = []
                for row_idx in range(selected_rows):
                    prompt: DatumSpec = {  # type: ignore
                        key: value[row_idx] for key, value in val_batch.items()
                    }
                    prompts.append(prompt)
                results = await asyncio.gather(
                    *(
                        self._rollout_manager.run_rollout(
                            prompt,
                            num_generations_per_prompt=1,
                            is_validation=True,
                        )
                        for prompt in prompts
                    ),
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, BaseException):
                        raise result

                records = cast(list[PromptGroupRecord], results)
                for record in records:
                    if len(record.completions) != 1:
                        raise ValueError(
                            "Validation rollouts must return exactly one completion "
                            "per prompt."
                        )
                    completion = record.completions[0]
                    rewards.append(float(completion.reward))
                    lengths.append(
                        float(
                            record.rollout_metrics["mean_gen_tokens_per_sample"]
                        )
                    )
                    message_logs.append(
                        get_keys_from_message_log(
                            completion.message_log,
                            ["role", "content"],
                        )
                    )

        num_samples = len(rewards)
        metrics: dict[str, Any] = {
            "accuracy": sum(rewards) / num_samples if num_samples else 0.0,
            "avg_length": sum(lengths) / num_samples if num_samples else 0.0,
            "num_samples": num_samples,
        }
        timing_metrics = timer.get_timing_metrics(reduction_op="sum")
        validation_time = float(timing_metrics.get("total_validation_time", 0.0))

        print("\n📊 Validation Results:")
        print(f"    • Accuracy: {metrics['accuracy']:.4f}")
        print(f"    • Average response length: {metrics['avg_length']:.1f} tokens")
        print(f"    • Samples processed: {num_samples}")
        print(f"    • Total validation time: {validation_time:.2f}s", flush=True)

        try:
            print_message_log_samples(
                message_logs,
                rewards,
                num_samples=min(
                    self._master_config.logger["num_val_samples_to_print"],
                    num_samples,
                ),
                step=step,
            )
        except Exception as e:
            print(f"\n  ⚠️ Error displaying message samples: {str(e)}")
            print("  ⚠️ Continuing validation without displaying samples...", flush=True)

        self._logger.log_batched_dict_as_jsonl(
            {"content": message_logs, "rewards": rewards},
            f"val_data_step{step}.jsonl",
        )
        self._logger.log_metrics(metrics, step=step, prefix="validation")
        self._logger.log_metrics(
            timing_metrics,
            step=step,
            prefix="timing/validation",
        )
        return metrics

    def _build_rollout_work(
        self,
        prompt: DatumSpec,
        target_step: Optional[int] = None,
    ) -> Optional[RolloutWorkItem]:
        """Assign stable checkpoint identity before dispatching one prompt group."""
        if self._rollout_checkpoint is None:
            return None

        dispatch_sequence = self._next_rollout_dispatch_sequence
        self._next_rollout_dispatch_sequence += 1
        prompt_index = int(prompt["idx"])
        task_name = prompt.get("task_name")
        prompt_fingerprint = compute_rollout_fingerprint(
            {
                "idx": prompt_index,
                "task_name": task_name,
                "message_log": prompt["message_log"],
                "extra_env_info": prompt["extra_env_info"],
                "loss_multiplier": prompt["loss_multiplier"],
            }
        )
        return RolloutWorkItem(
            run_id=self._rollout_checkpoint.run_id,
            group_id=f"g{dispatch_sequence:016x}-{uuid.uuid4().hex}",
            prompt_id=f"{task_name or 'prompt'}:{prompt_index}",
            dispatch_sequence=dispatch_sequence,
            target_step=target_step,
            attempt_id=0,
            policy_version=self._trainer_version,
            prompt_fingerprint=prompt_fingerprint,
            sampling_fingerprint=self._rollout_checkpoint.sampling_fingerprint,
            tokenizer_fingerprint=self._rollout_checkpoint.tokenizer_fingerprint,
            num_generations=self._master_config.grpo["num_generations_per_prompt"],
            prompt_ref={"idx": prompt_index, "task_name": task_name},
        )

    # ── the three pumps + the inline advantage stage ───────────────────────

    async def _rollout_pump(self) -> None:
        """Continuously dispatch rollout tasks until cancellation.

        Per batch (over_sampling=False):
          0. Wait while _max_rollout_version >= trainer_version + max_staleness,
             then claim the next step by incrementing _max_rollout_version.

        Per prompt:
          1. Acquire _buffer_capacity slot (backpressure)
          2. Acquire sem (cap concurrent in-flight rollouts)
          3. Wait for _rollout_permitted (paused during weight sync or validation)
          4. Call rollout_manager.generate_and_push(prompt) — local async
             RolloutManager reserves a slot, runs the rollout, then commits the
             group via TQReplayBuffer (→ dp_client.put_samples + mark ready)
          5. Decrement _inflight_rollouts
        """
        sem = asyncio.Semaphore(self._async_cfg.max_inflight_prompts)
        over_sampling = self._async_cfg.over_sampling
        max_staleness = self._async_cfg.max_weight_staleness_versions
        force_in_order = self._async_cfg.force_in_order
        num_generations_per_prompt = self._master_config.grpo[
            "num_generations_per_prompt"
        ]
        checkpoint_runtime = self._rollout_checkpoint
        print("rollout_pump: starting", flush=True)

        async def _dispatch_one_prompt(
            prompt: DatumSpec,
            target_step: Optional[int],
            checkpoint_work: Optional[RolloutWorkItem],
        ) -> None:
            committed = False
            try:
                if checkpoint_work is None:
                    await self._rollout_manager.generate_and_push(
                        prompt,
                        num_generations_per_prompt=num_generations_per_prompt,
                        target_step=target_step,
                    )
                else:
                    if checkpoint_runtime is None:
                        raise RuntimeError(
                            "checkpoint work was assigned without a checkpoint runtime"
                        )
                    await self._rollout_manager.generate_and_push(
                        prompt,
                        num_generations_per_prompt=num_generations_per_prompt,
                        target_step=target_step,
                        checkpoint_work=checkpoint_work,
                        checkpoint_writer=checkpoint_runtime.writer,
                    )
                    self._checkpoint_inflight_work.pop(checkpoint_work.group_id, None)
                committed = True
                if self._diagnostics:
                    content = ""
                    for i in range(len(prompt["message_log"])):
                        if prompt["message_log"][i]["role"] == "user":
                            content = prompt["message_log"][i]["content"]
                            break
                    print(f"  rollout done for prompt='{content[:20]}...'", flush=True)
            finally:
                if not committed:
                    # A failed rollout never creates a consumable TQ group, so
                    # the train pump cannot release this dispatch's capacity.
                    self._buffer_capacity.release()
                self._inflight_rollouts -= 1
                sem.release()

        max_epochs = self._master_config.grpo["max_num_epochs"]
        async with asyncio.TaskGroup() as rollout_tasks:
            while max_epochs is None or self._current_epoch < max_epochs:
                for prompt_batch in self._dataloader:
                    # over_sampling=False: batch-level gate on max_rollout_version.
                    if not over_sampling:
                        while (
                            self._max_rollout_version
                            >= self._trainer_version + max_staleness
                        ):
                            await asyncio.sleep(0.005)
                        self._max_rollout_version += 1

                    # target_step = batch dispatch index when force_in_order is on.
                    target_step = self._max_rollout_version if force_in_order else None

                    for prompt_idx in range(prompt_batch.size):
                        prompt: DatumSpec = {  # type: ignore
                            k: v[prompt_idx] for k, v in prompt_batch.items()
                        }
                        capacity_acquired = False
                        inflight_acquired = False
                        try:
                            await self._buffer_capacity.acquire()
                            capacity_acquired = True
                            await sem.acquire()
                            inflight_acquired = True
                            await self._rollout_permitted.wait()

                            checkpoint_work = self._build_rollout_work(
                                prompt, target_step
                            )
                            if checkpoint_work is not None:
                                self._checkpoint_inflight_work[
                                    checkpoint_work.group_id
                                ] = checkpoint_work

                            self._inflight_rollouts += 1
                            task = rollout_tasks.create_task(
                                _dispatch_one_prompt(
                                    prompt, target_step, checkpoint_work
                                )
                            )
                            self._dispatched_rollouts.add(task)
                            task.add_done_callback(self._dispatched_rollouts.discard)
                            # The child now owns both permits.
                            capacity_acquired = False
                            inflight_acquired = False
                        finally:
                            if inflight_acquired:
                                sem.release()
                            if capacity_acquired:
                                self._buffer_capacity.release()
                self._current_epoch += 1

        print(f"rollout_pump: completed {self._current_epoch} epoch(s)", flush=True)

    async def _train_pump(self) -> None:
        """Per-prompt-group streaming train loop.

        Per step:
          1. sampler.evict drops stale groups from the buffer and clears their TQ rows.
          2. sampler.select returns K prompt groups (or None) and drops them from the
             buffer; DP rows survive so the trainer can read them. Already trainable —
             buffer wrote training-shaped rows at rollout time.
          3. _advantage_pump(train_meta).
          4. trainer.train_microbatch_from_meta + finish_train_step.
          5. dp_client.clear_samples on consumed sample_ids; release _buffer_capacity
             per dropped group, then sync.
        """
        adv_cfg = self._advantage_cfg
        grpo_cfg = self._master_config.grpo

        # TODO: fix the prev_logprobs_required and reference_logprobs_required logic
        prev_logprobs_required = adv_cfg.policy_logprobs_field is not None
        reference_logprobs_required = adv_cfg.reference_logprobs_field is not None

        while self._train_steps < grpo_cfg["max_num_steps"]:
            groups_dispatched = 0
            min_sample_version = None
            step_open = False

            with self._timer.time("total_step_time"):
                while groups_dispatched < grpo_cfg["num_prompts_per_step"]:
                    # Wait for a selectable batch
                    with self._timer.time("exposed_generation"):
                        await asyncio.sleep(0)

                        # Evict stale groups
                        evicted = await self._sampler.evict(
                            current_train_weight=self._trainer_version,
                        )
                        if evicted:
                            print(
                                f"  evicted {evicted} stale prompt group(s)",
                                flush=True,
                            )
                            for _ in range(evicted):
                                self._buffer_capacity.release()

                        # Select a batch
                        max_prompt_groups = (
                            grpo_cfg["num_prompts_per_step"] - groups_dispatched
                        )
                        min_prompt_groups = min(
                            self._async_cfg.min_prompt_groups_per_batch,
                            max_prompt_groups,
                        )
                        train_meta, num_groups = await self._sampler.select(
                            current_train_weight=self._trainer_version,
                            min_prompt_groups=min_prompt_groups,
                            max_prompt_groups=max_prompt_groups,
                        )

                        # If no batch is selectable, sleep and retry
                        if train_meta is None:
                            await asyncio.sleep(0.05)
                            continue

                        # Release buffer capacity
                        for _ in range(num_groups):
                            self._buffer_capacity.release()

                    # Compute prev_logprobs / ref_logprobs
                    with self._timer.time("logprob_inference_prep"):
                        await asyncio.to_thread(self._trainer.prepare_for_lp_inference)
                    with self._timer.time("policy_and_reference_logprobs"):
                        if prev_logprobs_required:
                            await asyncio.to_thread(
                                self._trainer.get_logprobs_from_meta, train_meta
                            )
                        if reference_logprobs_required:
                            await asyncio.to_thread(
                                self._trainer.get_reference_policy_logprobs_from_meta,
                                train_meta,
                            )

                    # Compute advantages
                    with self._timer.time("advantage_calculation"):
                        train_meta = await self._advantage_stage(train_meta)

                    # Train
                    with self._timer.time("training_prep"):
                        await asyncio.to_thread(self._trainer.prepare_for_training)
                    with self._timer.time("policy_training"):
                        if not step_open:
                            await asyncio.to_thread(
                                self._trainer.begin_train_step,
                                self._loss_fn,
                            )
                            step_open = True
                        await asyncio.to_thread(
                            self._trainer.train_microbatches_from_meta,
                            train_meta,
                        )

                    if train_meta.sequence_lengths:
                        self._step_log_dict["sequence_lengths"].extend(
                            int(s) for s in train_meta.sequence_lengths
                        )

                    # Refresh min_sample_version
                    curr_min_sample_version = min(
                        t["weight_version"]
                        for t in train_meta.tags  # type: ignore
                    )
                    if min_sample_version is not None:
                        min_sample_version = min(
                            min_sample_version, curr_min_sample_version
                        )
                    else:
                        min_sample_version = curr_min_sample_version

                    # Remove consumed sample_ids from the buffer
                    await self._call_dp(
                        "clear_samples",
                        sample_ids=list(train_meta.sample_ids),
                        partition_id=self._partition_id,
                    )

                    groups_dispatched += num_groups

                if not step_open:
                    print(
                        "train_pump: rollout exhausted before any group ready",
                        flush=True,
                    )
                    break

                with self._timer.time("policy_training"):
                    result = await asyncio.to_thread(self._trainer.finish_train_step)

                step_metrics = aggregate_step_metrics(result)
                step_metrics.update(
                    reduce_advantage_pump_metrics(**self._step_log_dict)
                )
                self._step_log_dict = {k: [] for k in self._step_log_dict}

                self._trainer_version += 1
                self._train_steps += 1
                val_metrics: Optional[dict[str, Any]] = None
                is_last_step = self._train_steps >= grpo_cfg["max_num_steps"]
                should_validate = (
                    grpo_cfg["val_period"] > 0
                    and self._train_steps % grpo_cfg["val_period"] == 0
                ) or (grpo_cfg["val_at_end"] and is_last_step)

                if should_validate:
                    await self._pause_and_drain_rollouts()
                    with self._timer.time("weight_sync"):
                        await self._sync_weights(reopen_rollouts=False)
                    val_metrics = await self._run_validation(step=self._train_steps)
                    self._rollout_permitted.set()
                else:
                    with self._timer.time("weight_sync"):
                        await self._sync_weights()

                # Checkpointing (ported from the legacy async loop).
                self._consumed_samples += grpo_cfg["num_prompts_per_step"]
                self._total_valid_tokens += step_metrics.get("global_valid_toks", 0)
                self._timeout.mark_iteration()

                save_period = self._master_config.checkpointing["save_period"]
                # _train_steps was already incremented above, so it equals the
                # legacy loop's 1-indexed `step + 1`.
                should_save_by_step = (
                    is_last_step or self._train_steps % save_period == 0
                )
                should_save_by_timeout = self._timeout.check_save()

                if self._master_config.checkpointing["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    save_state = self._save_state
                    save_state["current_step"] = self._train_steps
                    save_state["total_steps"] = self._train_steps
                    save_state["current_epoch"] = self._current_epoch
                    save_state["consumed_samples"] = self._consumed_samples
                    save_state["total_valid_tokens"] = self._total_valid_tokens
                    # Snapshot synchronously — no await between the save
                    # decision and here — so it cannot interleave with
                    # _rollout_pump iterating this same dataloader (asyncio
                    # single-threading makes the snapshot race-free); the
                    # slow torch.save happens off-loop below.
                    dataloader_state = self._dataloader.state_dict()
                    if val_metrics is not None:
                        save_state["val_reward"] = val_metrics["accuracy"]
                    elif "val_reward" in save_state:
                        del save_state["val_reward"]

                    full_metric_name = self._master_config.checkpointing["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:',\n"
                            f'followed by the corresponding name in the "val" or "train" metrics dictionary.'
                            f"  If you are using an old config, please updated checkpointing.metric_name to the new format, "
                            f" e.g. 'val_reward --> 'val:accuracy'"
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = (
                            step_metrics if prefix == "train" else val_metrics
                        )
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. "
                                "This checkpoint will not be saved as top-k.",
                                stacklevel=2,
                            )
                            if full_metric_name in save_state:
                                del save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            save_state[full_metric_name] = metrics_source[metric_name]

                    with self._timer.time("checkpointing"):
                        print(f"Saving checkpoint for step {self._train_steps}...")
                        checkpoint_path = self._checkpointer.init_tmp_checkpoint(
                            self._train_steps, save_state, self._master_config
                        )
                        await asyncio.to_thread(
                            self._trainer.save_checkpoint,
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            )
                            if self._checkpointer.save_optimizer
                            else None,
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=self._master_config.checkpointing,
                        )
                        await asyncio.to_thread(
                            torch.save,
                            dataloader_state,
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        if self._async_cfg.over_sampling:
                            buffer_state = await self._buffer.state_dict(
                                saved_capacity=self._async_cfg.max_buffered_rollouts
                            )
                            await asyncio.to_thread(
                                torch.save,
                                buffer_state,
                                os.path.join(checkpoint_path, "replay_buffer.pt"),
                            )
                        await asyncio.to_thread(
                            self._checkpointer.finalize_checkpoint, checkpoint_path
                        )

            timing_metrics: dict[str, float] = self._timer.get_timing_metrics(
                reduction_op="sum"
            )  # type: ignore

            total_time = timing_metrics.get("total_step_time", 0.0)
            cluster_cfg = self._master_config.cluster
            total_num_gpus = cluster_cfg["num_nodes"] * cluster_cfg["gpus_per_node"]
            if total_time > 0 and "global_valid_toks" in step_metrics:
                timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                    step_metrics["global_valid_toks"] / total_time / total_num_gpus
                )

            print("\n⏱️  Timing:")
            print(f"  • Total step time: {total_time:.2f}s")
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k == "total_step_time":
                    continue
                percent = (v / total_time * 100) if total_time > 0 else 0.0
                print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

            # TODO: per-step train_data jsonl dump, vllm metrics logger,
            #   histogram log, rollout_metrics, seq_logprob_error_metrics,
            #   pretty-print "Training Results" block, print_performance_metrics.
            print(f"step_metrics={step_metrics}", flush=True)
            self._logger.log_metrics(
                step_metrics, step=self._train_steps, prefix="train"
            )
            self._logger.log_metrics(
                timing_metrics, step=self._train_steps, prefix="timing/train"
            )
            self._timer.reset()

            # min sample version refers to the version each consumed sample was
            # generated with; lag = current trainer version - oldest sample version.
            lag = self._trainer_version - min_sample_version  # type: ignore
            print(
                f"train step {self._train_steps}/{grpo_cfg['max_num_steps']}  "
                f"trainer_v={self._trainer_version}  "
                f"lag={lag}  ",
                flush=True,
            )

            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                break

    async def _pause_and_drain_rollouts(self) -> None:
        """Pause admission and wait for all already-dispatched rollouts.

        All tasks in the snapshot are allowed to settle before the first task
        error is propagated. The admission gate remains closed on return and
        on error so the caller can safely proceed to validation or abort.
        """
        self._rollout_permitted.clear()
        dispatched_rollouts = tuple(self._dispatched_rollouts)
        results = await asyncio.gather(
            *dispatched_rollouts,
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def _sync_weights(self, *, reopen_rollouts: bool = True) -> None:
        """Synchronize weights, draining checkpointed rollouts first.

        SC owns the drain gate (when to sync); WeightSynchronizer owns how.

        Flow:
          1. _rollout_permitted.clear()  — no new dispatches
          2. when checkpointing is enabled, drain _inflight_rollouts → 0
          3. weight_synchronizer.sync_weights(trainer_version)
          4. _rollout_permitted.set()   — resume

        Pass ``reopen_rollouts=False`` to keep the admission gate closed
        through validation.
        """
        self._rollout_permitted.clear()

        if self._rollout_checkpoint is not None:
            # Checkpoint work pins one policy version. Do not update the
            # generation backend until every dispatched group has either
            # reached TQ or failed into the controller recovery ledger.
            drain_start = time.monotonic()
            while self._inflight_rollouts > 0:
                await asyncio.sleep(0.005)

            if self._checkpoint_inflight_work:
                raise RuntimeError(
                    "cannot synchronize generation weights while failed rollout "
                    "checkpoint work awaits recovery"
                )

            drain_elapsed = time.monotonic() - drain_start
            print(
                f"  _sync_weights: drained in {drain_elapsed:.3f}s, "
                f"syncing weights v{self._trainer_version}",
                flush=True,
            )

        t0 = time.monotonic()
        await asyncio.to_thread(self._weight_synchronizer.sync_weights)
        elapsed = time.monotonic() - t0

        print(f"  _sync_weights: sync done in {elapsed:.3f}s", flush=True)
        self._rollout_manager.set_weight_version(self._trainer_version)
        if reopen_rollouts:
            self._rollout_permitted.set()

    async def _advantage_stage(self, meta: KVBatchMeta) -> KVBatchMeta:
        """Fetch advantage inputs, compute advantages, and write them back.

        SC owns the prompt-group-scoped advantage stage because the selected
        ``KVBatchMeta`` still contains complete prompt groups before trainer
        DP sharding. Tensor payloads still move through DataPlane: SC fetches
        only the configured advantage input columns and writes the computed
        ``advantages`` column back under the same ``sample_ids``.
        """
        if self._advantage_estimator is None:
            return meta
        adv_cfg = self._advantage_cfg

        data = await self._call_dp(
            "get_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=self._advantage_input_fields(),
        )

        prompt_ids = tensor_field(data, adv_cfg.prompt_ids_field)
        rewards = squeeze_trailing_unit_dim(
            tensor_field(data, adv_cfg.reward_field)
        ).float()
        self._step_log_dict["rewards"].append(rewards.detach())
        token_mask = tensor_field(data, adv_cfg.token_mask_field).float()
        sample_mask = squeeze_trailing_unit_dim(
            tensor_field(data, adv_cfg.sample_mask_field)
        ).float()
        mask = token_mask * sample_mask.unsqueeze(-1)

        repeated_batch: dict[str, torch.Tensor] = {
            "total_reward": rewards,
        }
        for field_name in adv_cfg.repeated_batch_fields:
            repeated_batch[field_name] = squeeze_trailing_unit_dim(
                tensor_field(data, field_name)
            )

        kwargs: dict[str, torch.Tensor] = {}
        if adv_cfg.policy_logprobs_field is not None:
            kwargs["logprobs_policy"] = tensor_field(
                data,
                adv_cfg.policy_logprobs_field,
            )
        if adv_cfg.reference_logprobs_field is not None:
            kwargs["logprobs_reference"] = tensor_field(
                data,
                adv_cfg.reference_logprobs_field,
            )

        advantages = self._advantage_estimator.compute_advantage(
            prompt_ids=prompt_ids,
            rewards=rewards,
            mask=mask,
            repeated_batch=repeated_batch,
            **kwargs,
        )
        self._step_log_dict["masked_advantages"].append(
            torch.masked_select(advantages.detach(), mask.bool())
        )

        await self._call_dp(
            "put_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            fields=fields_for_put(
                meta,
                {adv_cfg.output_field: advantages},
            ),
        )
        return meta.with_fields([adv_cfg.output_field])

    # ── utility helpers ────────────────────────────────────────────────────

    def _advantage_input_fields(self) -> list[str]:
        adv_cfg = self._advantage_cfg
        fields = [
            adv_cfg.prompt_ids_field,
            adv_cfg.reward_field,
            adv_cfg.token_mask_field,
            adv_cfg.sample_mask_field,
            *adv_cfg.repeated_batch_fields,
        ]
        if adv_cfg.policy_logprobs_field is not None:
            fields.append(adv_cfg.policy_logprobs_field)
        if adv_cfg.reference_logprobs_field is not None:
            fields.append(adv_cfg.reference_logprobs_field)
        return list(dict.fromkeys(fields))
