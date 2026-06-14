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

CPU-only Ray actor that runs three concurrent pumps and coordinates the
other actors via lightweight RPCs. SC sends control signals and reads
metadata only — model tensors still move through DataPlane or NCCL.

Data flow:
  _rollout_pump  → rollout_manager.generate_and_push(prompt)
                     → TQReplayBuffer.add tensorizes the record and writes
                       N training rows to TQ as one prompt-group.
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
import logging
import time
from typing import Any, Optional

import ray
import torch
from tensordict import TensorDict

from nemo_rl.algorithms.async_utils.staleness_sampler import StalenessSampler
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.environments.interfaces import EnvironmentInterface

log = logging.getLogger(__name__)


@ray.remote(num_cpus=1, num_gpus=0)  # pragma: no cover
class SingleControllerActor:
    """CPU-only Ray actor that orchestrates the RL training loop.

    Owns three concurrent asyncio tasks:
      - _rollout_pump: dispatches prompts via RolloutManager → TQReplayBuffer.add
      - _train_pump:   evicts stale groups, samples a batch, trains, drops it
      - _sync_weights: drain gate + weight synchronization

    All other actors are passive — they expose methods and wait to be called.
    """

    def __init__(
        self,
        master_config: MasterConfig,
        *,
        dp_client: Any,
        gen_handle: Any,
        trainer_handle: Any,
        env_handles: dict[str, EnvironmentInterface],
        train_cluster: Any,
        inference_cluster: Any,
        dataset: Any,
        components: Optional[tuple] = None,
    ) -> None:
        """Initialize the SingleController actor.

        Args:
            master_config: SC MasterConfig.
            dp_client: DataPlane client handle.
            gen_handle: Generation backend.
            trainer_handle: Trainer Ray actor handle.
            env_handles: ``task_name -> EnvironmentInterface`` mapping.
            train_cluster: Training Ray cluster (weight-sync rendezvous).
            inference_cluster: Inference Ray cluster.
            dataset: Train dataset wrapped into a StatefulDataLoader here.
            components: Test-only escape hatch — a tuple
                ``(dataloader, weight_synchronizer, advantage_estimator,
                rollout_manager, tq_buffer)`` that bypasses the in-actor
                setup. Production callers leave this ``None``.
        """
        import logging as _logging

        _logging.basicConfig(
            level=_logging.INFO,
            format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
        )

        # Build tokenizer + components inside the actor so the heavy
        # Python objects never ride through Ray cloudpickle. The
        # ``components`` arg is a test-only escape hatch that injects
        # fakes built CPU-side.
        if components is None:
            from nemo_rl.algorithms.utils import get_tokenizer
            from nemo_rl.algorithms.single_controller_utils.setup import (
                setup_single_controller_component,
            )

            tokenizer = get_tokenizer(master_config.policy["tokenizer"])
            components = setup_single_controller_component(
                master_config,
                tokenizer,
                dp_client=dp_client,
                gen_handle=gen_handle,
                trainer_handle=trainer_handle,
                env_handles=env_handles,
                train_cluster=train_cluster,
                inference_cluster=inference_cluster,
                dataset=dataset,
                partition_id=master_config.partition_id,
            )

        (
            dataloader,
            weight_synchronizer,
            advantage_estimator,
            rollout_manager,
            tq_buffer,
        ) = components

        self._mc = master_config
        self._dp_client = dp_client
        self._gen = gen_handle
        self._trainer = trainer_handle
        self._dataloader = dataloader
        self._weight_synchronizer = weight_synchronizer
        self._advantage_estimator = advantage_estimator
        self._buffer = tq_buffer
        self._rollout_manager = rollout_manager
        # When tests pass rollout_manager + tq_buffer as separate
        # cloudpickle blobs, Ray deserializes them into distinct buffer
        # instances. Rebind so the writer and the sampler share one.
        self._rollout_manager._tq_buffer = self._buffer

        adv_cfg = master_config.advantage
        stale_cfg = master_config.staleness
        conc_cfg = master_config.concurrency
        ws_cfg = master_config.weight_sync

        if adv_cfg.enabled and self._advantage_estimator is None:
            raise ValueError(
                "advantage.enabled=True requires an advantage_estimator instance"
            )

        if stale_cfg.target_prompt_groups_per_step is None:
            stale_cfg.target_prompt_groups_per_step = stale_cfg.min_prompt_groups_per_batch
        if stale_cfg.target_prompt_groups_per_step < stale_cfg.min_prompt_groups_per_batch:
            raise ValueError(
                f"target_prompt_groups_per_step ({stale_cfg.target_prompt_groups_per_step}) "
                f"must be >= min_prompt_groups_per_batch ({stale_cfg.min_prompt_groups_per_batch})"
            )

        if stale_cfg.batch_selection_strategy == "strict_on_policy":
            stale_cfg.max_weight_staleness_versions = 0
            print(
                "Using strict_on_policy, auto setting max_weight_staleness_versions to 0."
            )

        self._sampler = StalenessSampler(
            self._buffer,
            max_staleness_versions=stale_cfg.max_weight_staleness_versions,
        )

        # ── asyncio state ──────────────────────────────────────────────────
        # Gate: cleared during _sync_weights, set when generation may proceed
        self._rollout_permitted: asyncio.Event = asyncio.Event()
        self._rollout_permitted.set()

        # Count of in-flight generate_and_push calls
        self._inflight_rollouts: int = 0

        # Backpressure valve: max unconsumed rollout groups allowed in DataPlane.
        # Acquired before each rollout dispatch; released when the buffer
        # drops a group (sampler.evict or post-train buffer.remove).
        self._buffer_capacity: asyncio.Semaphore = asyncio.Semaphore(
            conc_cfg.max_buffered_rollouts
        )

        self._trainer_version: int = 0
        self._train_steps: int = 0
        self._step_consumed_sample_ids: list[str] = []

        log.info(
            "SingleControllerActor: staleness_cap=%d buffer=%d inflight=%d transport=%s",
            stale_cfg.max_weight_staleness_versions,
            conc_cfg.max_buffered_rollouts,
            conc_cfg.max_inflight_prompts,
            ws_cfg.transport,
        )

    # ── public API ─────────────────────────────────────────────────────────

    async def run(self) -> dict[str, Any]:
        """Main entry point. Runs until max_train_steps is reached."""
        rollout_task = asyncio.create_task(self._rollout_pump())
        train_task = asyncio.create_task(self._train_pump())

        await train_task

        rollout_task.cancel()
        try:
            await rollout_task
        except asyncio.CancelledError:
            pass

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
        }

    # ── internal helpers ───────────────────────────────────────────────────

    async def _ray_get(self, obj_ref: Any) -> Any:
        """Await a Ray ObjectRef without blocking the asyncio event loop."""
        return await obj_ref

    async def _reap_in_flight_nonblocking(
        self, refs: list[ray.ObjectRef]
    ) -> list[ray.ObjectRef]:
        """Drain completed refs without blocking; return still-pending refs.

        Uses ``asyncio.wait`` with ``timeout=0`` so Ray ObjectRefs are checked
        through their awaitable interface (which is accurate in async actors).
        ``ray.wait(timeout=0)`` does not always reflect cross-process ref
        readiness from an async actor, so we avoid it here.
        """
        if not refs:
            return []
        ref_to_task = {ref: asyncio.ensure_future(ref) for ref in refs}
        await asyncio.wait(ref_to_task.values(), timeout=0.05)
        pending: list[ray.ObjectRef] = []
        for ref, task in ref_to_task.items():
            if task.done():
                task.result()  # surface exceptions; payload ignored
            else:
                task.cancel()
                pending.append(ref)
        return pending

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

    # ── the three pumps + advantage helper ────────────────────────────────

    async def _rollout_pump(self) -> None:
        """Continuously dispatch rollout tasks until cancellation.

        Flow per prompt:
          1. Acquire _buffer_capacity slot (backpressure)
          2. Acquire sem (cap concurrent in-flight rollouts)
          3. Wait for _rollout_permitted (paused during weight sync)
          4. Call rollout_manager.generate_and_push(prompt) — local async
             RolloutManager runs the rollout and writes the group via
             TQReplayBuffer.add (→ dp_client.put_samples + meta append)
          5. Decrement _inflight_rollouts
        """
        sem = asyncio.Semaphore(self._mc.concurrency.max_inflight_prompts)
        log.info("rollout_pump: starting")

        async def _dispatch_one_prompt(prompt: DatumSpec) -> None:
            self._inflight_rollouts += 1
            try:
                await self._rollout_manager.generate_and_push(prompt)
                if self._mc.diagnostics:
                    content = ""
                    for i in range(len(prompt["message_log"])):
                        if prompt["message_log"][i]["role"] == "user":
                            content = prompt["message_log"][i]["content"]
                            break
                    log.info("  rollout done for prompt='%s...'", content[:20])
            finally:
                self._inflight_rollouts -= 1
                sem.release()

        # TODO: limit max_train_steps to max_num_epochs * len(dataloader) when setup
        max_epochs = self._mc.training.max_num_epochs
        epoch = 0
        while max_epochs is None or epoch < max_epochs:
            for prompt in self._dataloader:
                # check if buffer is full
                await self._buffer_capacity.acquire()
                # check if inflight rollouts is full
                await sem.acquire()
                # wait for rollout to be permitted
                await self._rollout_permitted.wait()

                # dispatch rollout
                asyncio.create_task(_dispatch_one_prompt(prompt))
            epoch += 1

        log.info("rollout_pump: completed %d epoch(s)", epoch)

    async def _train_pump(self) -> None:
        """Drain stale groups, sample, train, drop.

        Per step:
          1. sampler.evict drops stale groups from the buffer and clears their TQ rows.
          2. sampler.select returns K prompt groups (or None) and drops them from the
             buffer; DP rows survive so the trainer can read them. Already trainable —
             buffer wrote training-shaped rows at rollout time.
          3. _advantage_pump(train_meta).
          4. trainer.train_on_meta(train_meta, dp_client).
          5. dp_client.clear_samples on consumed sample_ids; release _buffer_capacity
             per dropped group, then sync.
        """
        adv_cfg = self._mc.advantage
        stale_cfg = self._mc.staleness
        train_cfg = self._mc.training

        logprobs_required = (
            adv_cfg.policy_logprobs_field is not None
            or adv_cfg.reference_logprobs_field is not None
        )

        while self._train_steps < train_cfg.max_train_steps:
            step_id = f"sc-step-{self._train_steps:06d}"
            # __init__ coerces None → min_prompt_groups_per_batch (int);
            # the assert narrows the Optional[int] type for pyrefly.
            assert stale_cfg.target_prompt_groups_per_step is not None
            target_groups: int = stale_cfg.target_prompt_groups_per_step
            groups_dispatched = 0
            in_flight: list[ray.ObjectRef] = []
            step_open = False

            evicted = await self._sampler.evict(
                current_train_weight=self._trainer_version,
            )
            if evicted:
                log.info("  evicted %d stale prompt group(s)", evicted)
                for _ in range(evicted):
                    self._buffer_capacity.release()

            while groups_dispatched < target_groups:
                await asyncio.sleep(0)

                # TODO @yukih: wait train pump merged, now always return min_prompt_groups_per_batch
                # need to add a max_prompt_groups_per_batch
                train_meta, num_groups = await self._sampler.select(
                    current_train_weight=self._trainer_version,
                    min_prompt_groups=stale_cfg.min_prompt_groups_per_batch,
                )

                if train_meta is None:
                    await asyncio.sleep(0.05)
                    continue

                for _ in range(num_groups):
                    self._buffer_capacity.release()

                if logprobs_required:
                    await self._ray_get(
                        self._trainer.prepare_logprobs_from_meta.remote(train_meta)
                    )

                train_meta = await self._advantage_pump(train_meta)

                if not step_open:
                    await self._ray_get(self._trainer.begin_train_step.remote(step_id))
                    step_open = True

                future = self._trainer.train_microbatch_from_meta.remote(
                    step_id, train_meta
                )
                in_flight.append(future)
                groups_dispatched += num_groups
                self._step_consumed_sample_ids.extend(train_meta.sample_ids)

                in_flight = await self._reap_in_flight_nonblocking(in_flight)

            for fut in in_flight:
                await self._ray_get(fut)

            if not step_open:
                log.info("train_pump: rollout exhausted before any group ready")
                break

            result = await self._ray_get(
                self._trainer.finish_train_step.remote(step_id)
            )
            consumed_ids = list(self._step_consumed_sample_ids)
            self._step_consumed_sample_ids = []
            await self._call_dp(
                "clear_samples",
                sample_ids=list(consumed_ids),
                partition_id=self._mc.partition_id,
            )

            self._trainer_version = result["trainer_version"]
            min_sample_version = min(t["weight_version"] for t in train_meta.tags)  # type: ignore
            lag = self._trainer_version - min_sample_version
            log.info(
                "train step %d/%d  trainer_v=%d  lag=%d  batch_size=%d",
                self._train_steps + 1,
                train_cfg.max_train_steps,
                self._trainer_version,
                lag,
                len(consumed_ids),
            )

            await self._sync_weights()
            self._train_steps += 1

    async def _sync_weights(self) -> None:
        """Drain in-flight rollouts then synchronize weights.

        SC owns the drain gate (when to sync); WeightSynchronizer owns how.

        Flow:
          1. _rollout_permitted.clear()  — no new dispatches
          2. drain _inflight_rollouts → 0  (5ms poll)
          3. weight_synchronizer.sync_weights(trainer_version)
          4. _rollout_permitted.set()   — resume
        """
        self._rollout_permitted.clear()

        # Drain: wait for all in-flight rollouts to complete before NCCL
        # Critical: if GenWorker has queued calls when NCCL init is dispatched,
        # the init sits behind them — trainer blocks in rendezvous → deadlock
        drain_start = time.monotonic()
        while self._inflight_rollouts > 0:
            await asyncio.sleep(0.005)

        drain_elapsed = time.monotonic() - drain_start
        log.info(
            "  _sync_weights: drained in %.3fs, syncing weights v%d",
            drain_elapsed,
            self._trainer_version,
        )

        t0 = time.monotonic()
        await self._weight_synchronizer.sync_weights(self._trainer_version)
        elapsed = time.monotonic() - t0

        log.info("  _sync_weights: sync done in %.3fs", elapsed)
        self._rollout_manager.set_weight_version(self._trainer_version)
        self._rollout_permitted.set()

    async def _advantage_pump(self, meta: KVBatchMeta) -> KVBatchMeta:
        """Fetch advantage inputs, compute advantages, and write them back.

        SC owns the prompt-group-scoped advantage stage because the selected
        ``KVBatchMeta`` still contains complete prompt groups before trainer
        DP sharding. Tensor payloads still move through DataPlane: SC fetches
        only the configured advantage input columns and writes the computed
        ``advantages`` column back under the same ``sample_ids``.
        """
        adv_cfg = self._mc.advantage
        if not adv_cfg.enabled:
            return meta
        assert self._advantage_estimator is not None

        data = await self._call_dp(
            "get_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=self._advantage_input_fields(),
        )

        prompt_ids = _tensor_field(data, adv_cfg.prompt_ids_field)
        rewards = _squeeze_trailing_unit_dim(
            _tensor_field(data, adv_cfg.reward_field)
        ).float()
        token_mask = _tensor_field(data, adv_cfg.token_mask_field).float()
        sample_mask = _squeeze_trailing_unit_dim(
            _tensor_field(data, adv_cfg.sample_mask_field)
        ).float()
        mask = token_mask * sample_mask.unsqueeze(-1)

        repeated_batch: dict[str, torch.Tensor] = {
            "total_reward": rewards,
        }
        for field_name in adv_cfg.repeated_batch_fields:
            repeated_batch[field_name] = _squeeze_trailing_unit_dim(
                _tensor_field(data, field_name)
            )

        kwargs: dict[str, torch.Tensor] = {}
        if adv_cfg.policy_logprobs_field is not None:
            kwargs["logprobs_policy"] = _tensor_field(
                data,
                adv_cfg.policy_logprobs_field,
            )
        if adv_cfg.reference_logprobs_field is not None:
            kwargs["logprobs_reference"] = _tensor_field(
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

        await self._call_dp(
            "put_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            fields=_fields_for_put(
                meta,
                {adv_cfg.output_field: advantages},
            ),
        )
        return meta.with_fields([adv_cfg.output_field])

    # ── utility helpers ────────────────────────────────────────────────────

    def _advantage_input_fields(self) -> list[str]:
        adv_cfg = self._mc.advantage
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


def _tensor_field(data: TensorDict, field_name: str) -> torch.Tensor:
    value = data[field_name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"advantage_pump expected tensor field {field_name!r}; got {type(value)}"
        )
    if value.is_nested:
        return torch.nested.to_padded_tensor(value, padding=0)
    return value


def _squeeze_trailing_unit_dim(value: torch.Tensor) -> torch.Tensor:
    if value.dim() >= 2 and value.shape[-1] == 1:
        return value.squeeze(-1)
    return value


def _fields_for_put(meta: KVBatchMeta, fields: dict[str, torch.Tensor]) -> TensorDict:
    packed: dict[str, torch.Tensor] = {}
    if meta.sequence_lengths is None:
        for field_name, value in fields.items():
            packed[field_name] = value.detach().contiguous()
        # pyrefly: ignore[bad-argument-type]
        return TensorDict(packed, batch_size=[meta.size])

    lengths = torch.tensor(meta.sequence_lengths, dtype=torch.long)
    for field_name, value in fields.items():
        if value.dim() >= 2 and value.shape[1] == int(lengths.max().item()):
            rows = [
                value[i, : int(lengths[i].item())].detach().contiguous()
                for i in range(meta.size)
            ]
            packed[field_name] = torch.nested.as_nested_tensor(
                rows,
                layout=torch.jagged,
            )
        else:
            packed[field_name] = value.detach().contiguous()
    # pyrefly: ignore[bad-argument-type]
    return TensorDict(packed, batch_size=[meta.size])
