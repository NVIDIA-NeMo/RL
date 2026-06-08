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
  _train_pump    → sampler.evict → buffer.remove (stale groups).
                 → sampler.select → KVBatchMeta of K groups (or None);
                   meta is already trainable.
                 → _advantage_pump (get → compute → put).
                 → trainer.train_on_meta.
                 → buffer.remove (trained groups).
  _sync_weights  → drain _inflight_rollouts → WeightSynchronizer.sync_weights.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import ray
import torch
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.staleness_sampler import StalenessSampler
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollout_manager import RolloutManager

log = logging.getLogger(__name__)


@dataclass
class SingleControllerConfig:
    """Configuration for SingleController."""

    # Staleness
    max_weight_staleness_versions: int = 1
    min_prompt_groups_per_batch: int = 2
    target_prompt_groups_per_step: Optional[int] = None
    generations_per_prompt: int = 4
    batch_selection_strategy: Literal[
        "strict_on_policy",
        "staleness_window",
    ] = "strict_on_policy"

    # Concurrency limits
    max_inflight_prompts: int = 8
    max_buffered_rollouts: int = 8  # _buffer_capacity semaphore size

    # Training
    max_train_steps: int = 10

    # DataPlane partition
    partition_id: str = "rollout_data"

    # Advantage calculation
    advantage_enabled: bool = False
    advantage_output_field: str = "advantages"
    advantage_prompt_ids_field: str = "prompt_ids_for_adv"
    advantage_reward_field: str = "total_reward"
    advantage_token_mask_field: str = "token_mask"
    advantage_sample_mask_field: str = "sample_mask"
    advantage_repeated_batch_fields: list[str] = field(default_factory=list)
    advantage_policy_logprobs_field: str | None = None
    advantage_reference_logprobs_field: str | None = None

    # Diagnostics
    diagnostics: bool = False

    # Weight transport backend ("stub" for dry-run, "nccl" for production)
    weight_transport: str = "stub"
    weight_nccl_addr: str = "127.0.0.1"
    weight_nccl_port: Optional[int] = None

    # Rollout config. Read only when SC builds RolloutManager itself.
    rollout_max_seq_len: int = 1024
    rollout_max_turns: Optional[int] = None
    use_nemo_gym: bool = False

    # Extra fields passed through to avoid TypedDict issues
    extra: dict = field(default_factory=dict)


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
        cfg: SingleControllerConfig,
        dp_client: Any,
        gen_handle: Any,
        trainer_handle: Any,
        env_handles: dict[str, EnvironmentInterface],
        # TODO: move into SC's setup phase
        dataloader: StatefulDataLoader,
        weight_synchronizer: Any,
        advantage_estimator: Any | None = None,
        tokenizer: Any | None = None,
        # TODO: remove the rollout_manager / tq_buffer overrides once SC's
        # setup phase owns construction; today they let the dry-run test
        # share one buffer + manager instance with SC.
        rollout_manager: RolloutManager | None = None,
        tq_buffer: TQReplayBuffer | None = None,
    ) -> None:
        import logging as _logging

        _logging.basicConfig(
            level=_logging.INFO,
            format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
        )

        self._cfg = cfg
        self._dp_client = dp_client
        self._gen = gen_handle
        self._trainer = trainer_handle
        self._dataloader = dataloader
        self._weight_synchronizer = weight_synchronizer
        self._advantage_estimator = advantage_estimator

        if cfg.advantage_enabled and self._advantage_estimator is None:
            raise ValueError(
                "advantage_enabled=True requires an advantage_estimator instance"
            )

        if cfg.target_prompt_groups_per_step is None:
            cfg.target_prompt_groups_per_step = cfg.min_prompt_groups_per_batch
        if cfg.target_prompt_groups_per_step < cfg.min_prompt_groups_per_batch:
            raise ValueError(
                f"target_prompt_groups_per_step ({cfg.target_prompt_groups_per_step}) "
                f"must be >= min_prompt_groups_per_batch ({cfg.min_prompt_groups_per_batch})"
            )

        pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        self._buffer = (
            tq_buffer
            if tq_buffer is not None
            else TQReplayBuffer(
                dp_client,
                partition_id=cfg.partition_id,
                pad_value_dict={"token_ids": pad_id},
            )
        )

        if rollout_manager is None:
            self._rollout_manager = RolloutManager(
                tokenizer=tokenizer,
                env_handles=env_handles,
                num_generations_per_prompt=cfg.generations_per_prompt,
                max_seq_len=cfg.rollout_max_seq_len,
                max_rollout_turns=cfg.rollout_max_turns,
                use_nemo_gym=cfg.use_nemo_gym,
                policy_generation=gen_handle,
                tq_buffer=self._buffer,
            )
        else:
            self._rollout_manager = rollout_manager

        # Initialize sampler
        assert cfg.batch_selection_strategy in [
            "strict_on_policy",
            "staleness_window",
        ], f"Unknown batch_selection_strategy: {cfg.batch_selection_strategy}"

        if cfg.batch_selection_strategy == "strict_on_policy":
            cfg.max_weight_staleness_versions = 0
            print(
                "Using strict_on_policy, auto setting max_weight_staleness_versions to 0."
            )

        self._sampler = StalenessSampler(
            self._buffer,
            max_staleness_versions=cfg.max_weight_staleness_versions,
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
            cfg.max_buffered_rollouts
        )

        self._trainer_version: int = 0
        self._train_steps: int = 0
        self._step_consumed_sample_ids: list[str] = []

        log.info(
            "SingleControllerActor: staleness_cap=%d buffer=%d inflight=%d transport=%s",
            cfg.max_weight_staleness_versions,
            cfg.max_buffered_rollouts,
            cfg.max_inflight_prompts,
            cfg.weight_transport,
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
        sem = asyncio.Semaphore(self._cfg.max_inflight_prompts)
        log.info("rollout_pump: starting")

        async def _dispatch_one_prompt(prompt: DatumSpec) -> None:
            self._inflight_rollouts += 1
            try:
                await self._rollout_manager.generate_and_push(prompt)
                if self._cfg.diagnostics:
                    content = ""
                    for i in range(len(prompt["message_log"])):
                        if prompt["message_log"][i]["role"] == "user":
                            content = prompt["message_log"][i]["content"]
                            break
                    log.info("  rollout done for prompt='%s...'", content[:20])
            finally:
                self._inflight_rollouts -= 1
                sem.release()

        # TODO: add max_num_epochs and limit max_train_steps to max_num_epochs * len(dataloader) when setup
        while True:
            for prompt in self._dataloader:
                # check if buffer is full
                await self._buffer_capacity.acquire()
                # check if inflight rollouts is full
                await sem.acquire()
                # wait for rollout to be permitted
                await self._rollout_permitted.wait()

                # dispatch rollout
                asyncio.create_task(_dispatch_one_prompt(prompt))

    async def _train_pump(self) -> None:
        """Drain stale groups, sample, train, drop.

        Per step:
          1. sampler.evict → buffer.remove stale prompt-group entries.
          2. sampler.select → train_meta covering K prompt groups (or None).
             Already trainable — buffer wrote training-shaped rows at rollout time.
          3. _advantage_pump(train_meta).
          4. trainer.train_on_meta(train_meta, dp_client).
          5. buffer.remove(train_meta.sample_ids) drops trained groups and clears
             their TQ rows; release _buffer_capacity per dropped group, then sync.
        """
        logprobs_required = (
            self._cfg.advantage_policy_logprobs_field is not None
            or self._cfg.advantage_reference_logprobs_field is not None
        )

        while self._train_steps < self._cfg.max_train_steps:
            step_id = f"sc-step-{self._train_steps:06d}"
            # __init__ coerces None → min_prompt_groups_per_batch (int);
            # the assert narrows the Optional[int] type for pyrefly.
            assert self._cfg.target_prompt_groups_per_step is not None
            target_groups: int = self._cfg.target_prompt_groups_per_step
            groups_dispatched = 0
            in_flight: list[ray.ObjectRef] = []
            step_open = False
            step_min_weight_version: int | None = None

            evicted = await self._sampler.evict(
                current_train_weight=self._trainer_version,
            )
            if evicted:
                log.info("  evicted %d stale prompt group(s)", evicted)
                for _ in range(evicted):
                    self._buffer_capacity.release()

            while groups_dispatched < target_groups:
                await asyncio.sleep(0)

                train_meta = self._sampler.select(
                    current_train_weight=self._trainer_version,
                    min_prompt_groups=self._cfg.min_prompt_groups_per_batch,
                )

                if train_meta is None:
                    await asyncio.sleep(0.05)
                    continue

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
                groups_dispatched += 1
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
            dropped = await self._buffer.remove(list(consumed_ids))
            for _ in range(dropped):
                self._buffer_capacity.release()
            self._step_consumed_sample_ids = []

            self._trainer_version = result["trainer_version"]
            lag = self._trainer_version - min(
                t["weight_version"] for t in train_meta.tags
            )
            log.info(
                "train step %d/%d  trainer_v=%d  lag=%d  batch_size=%d",
                self._train_steps + 1,
                self._cfg.max_train_steps,
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
        self._rollout_permitted.set()

    async def _advantage_pump(self, meta: KVBatchMeta) -> KVBatchMeta:
        """Fetch advantage inputs, compute advantages, and write them back.

        SC owns the prompt-group-scoped advantage stage because the selected
        ``KVBatchMeta`` still contains complete prompt groups before trainer
        DP sharding. Tensor payloads still move through DataPlane: SC fetches
        only the configured advantage input columns and writes the computed
        ``advantages`` column back under the same ``sample_ids``.
        """
        if not self._cfg.advantage_enabled:
            return meta
        assert self._advantage_estimator is not None

        data = await self._call_dp(
            "get_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=self._advantage_input_fields(),
        )

        prompt_ids = _tensor_field(data, self._cfg.advantage_prompt_ids_field)
        rewards = _squeeze_trailing_unit_dim(
            _tensor_field(data, self._cfg.advantage_reward_field)
        ).float()
        token_mask = _tensor_field(data, self._cfg.advantage_token_mask_field).float()
        sample_mask = _squeeze_trailing_unit_dim(
            _tensor_field(data, self._cfg.advantage_sample_mask_field)
        ).float()
        mask = token_mask * sample_mask.unsqueeze(-1)

        repeated_batch: dict[str, torch.Tensor] = {
            "total_reward": rewards,
        }
        for field_name in self._cfg.advantage_repeated_batch_fields:
            repeated_batch[field_name] = _squeeze_trailing_unit_dim(
                _tensor_field(data, field_name)
            )

        kwargs: dict[str, torch.Tensor] = {}
        if self._cfg.advantage_policy_logprobs_field is not None:
            kwargs["logprobs_policy"] = _tensor_field(
                data,
                self._cfg.advantage_policy_logprobs_field,
            )
        if self._cfg.advantage_reference_logprobs_field is not None:
            kwargs["logprobs_reference"] = _tensor_field(
                data,
                self._cfg.advantage_reference_logprobs_field,
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
                {self._cfg.advantage_output_field: advantages},
            ),
        )
        return meta.with_fields([self._cfg.advantage_output_field])

    # ── utility helpers ────────────────────────────────────────────────────

    def _advantage_input_fields(self) -> list[str]:
        fields = [
            self._cfg.advantage_prompt_ids_field,
            self._cfg.advantage_reward_field,
            self._cfg.advantage_token_mask_field,
            self._cfg.advantage_sample_mask_field,
            *self._cfg.advantage_repeated_batch_fields,
        ]
        if self._cfg.advantage_policy_logprobs_field is not None:
            fields.append(self._cfg.advantage_policy_logprobs_field)
        if self._cfg.advantage_reference_logprobs_field is not None:
            fields.append(self._cfg.advantage_reference_logprobs_field)
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
