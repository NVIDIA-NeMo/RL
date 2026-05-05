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
"""TQ-mediated Policy: drop-in replacement for ``Policy`` whose
``train`` / ``get_logprobs`` / ``get_reference_policy_logprobs`` route
their per-step bulk tensors through a TransferQueue partition instead
of Ray's in-memory object store.

Same method names and return shapes as ``Policy``. Only the transport
between driver and DP workers changes — workers fetch their slice from
TQ via ``self._fetch(meta)`` (already wired on
:class:`AbstractPolicyWorker`) and return data via Ray, just as the
legacy path does.

Method bodies mirror :class:`Policy` line-for-line on the structural
pieces (shard, dispatch, aggregate, reorder, FLOPs annotation). The
deltas are isolated and clearly marked: ``fan_out_per_rank_metas`` to
seed the partition, ``meta=metas`` instead of ``data=sharded`` on the
worker call, and the worker method name (``*_presharded`` vs the
legacy worker entrypoints).

Long-term retirement: when the legacy in-memory path is removed,
``Policy``'s method bodies get replaced with the bodies here and this
file goes away.
"""

from __future__ import annotations

import warnings
from contextlib import nullcontext
from typing import Any, Optional

import ray

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.preshard import (
    DP_SEED_FIELDS,
    LP_SEED_FIELDS,
    fan_out_per_rank_metas,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import GenerationDatumSpec
from nemo_rl.models.policy.interfaces import (
    LogprobOutputSpec,
    ReferenceLogprobOutputSpec,
)
from nemo_rl.models.policy.lm_policy import (
    Policy,
    _aggregate_logprob_results,
    _aggregate_reference_logprob_results,
    _aggregate_train_results,
)
from nemo_rl.utils.flops_tracker import get_theoretical_tflops
from nemo_rl.utils.timer import Timer


class TQPolicy(Policy):
    """TQ-mediated counterpart to :class:`Policy`.

    Constructor accepts an additional ``dp_cfg`` (the
    ``master_config["data_plane"]`` dict). Bootstraps the controller on
    the driver and forwards ``setup_data_plane(dp_cfg)`` to every worker
    so they can attach as clients (``bootstrap=False``).

    The partition lifecycle (``register_partition`` / ``kv_clear``) is
    the trainer's responsibility — this class assumes the partition
    named ``self._tq_partition_id`` (default ``"train"``) is open with a
    schema that includes the seed fields written by ``fan_out_per_rank_metas``.
    """

    def __init__(
        self,
        *args: Any,
        dp_cfg: dict[str, Any],
        tq_partition_id: str = "train",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Lazy import — keeps ``Policy``-only call sites from importing
        # the data-plane stack at module load.
        from nemo_rl.data_plane import build_data_plane_client

        # Driver-side controller bootstrap. Workers attach with
        # ``bootstrap=False`` via the ``setup_data_plane`` forwarded below.
        # ``dp_cfg`` is public so async ReplayBuffer / AsyncTrajectoryCollector
        # can read it off the policy without referencing master_config.
        self.dp_cfg = dp_cfg
        self._dp_client = build_data_plane_client(dp_cfg, bootstrap=True)
        self._tq_partition_id = tq_partition_id

        # Per-step monotonic counter for key namespacing — every fan-out
        # call within a step needs a distinct prefix or keys collide
        # in the partition. The trainer's per-step ``kv_clear`` resets
        # the partition each step; the counter doesn't reset, but the
        # combination ``f"{tag}_{idx}"`` stays unique within a partition
        # life cycle.
        self._tq_call_idx = 0

        # Forward to workers (replaces ``Policy.setup_data_plane`` call
        # site in the trainer — TQPolicy bundles bootstrap + worker
        # attach into construction so the trainer just instantiates
        # ``TQPolicy(...)`` and is done).
        ray.get(
            [
                getattr(w, "setup_data_plane").remote(cfg=dp_cfg)
                for w in self.worker_group._workers
            ]
        )

    # ── lifecycle ──────────────────────────────────────────────────────

    def shutdown(self) -> bool:  # type: ignore[override]
        """Close the TQ client before shutting down the worker group."""
        try:
            self._dp_client.close()
        except Exception as e:
            warnings.warn(f"Error closing data-plane client: {e}")
        return super().shutdown()

    def prepare_step(
        self,
        num_samples: int,
        group_size: Optional[int] = None,
    ) -> None:
        """Register the per-step TQ partition.

        Sync trainers call this at the start of each step (verl-style:
        static partition id ``"train"`` cleared and reused). The schema
        is the union of all consumer fields — producers write only the
        subset they have, consumers fetch via ``select_fields``.
        """
        self._dp_client.register_partition(
            partition_id=self._tq_partition_id,
            fields=list(DP_SEED_FIELDS),
            num_samples=num_samples,
            consumer_tasks=["prev_lp", "ref_lp", "train"],
            grpo_group_size=group_size,
        )

    # ── helpers ────────────────────────────────────────────────────────

    def _next_key_prefix(self, tag: str) -> str:
        """Monotonic per-instance prefix for fan-out keys."""
        self._tq_call_idx += 1
        return f"{tag}_{self._tq_call_idx}"

    def _fan_out_logprob_metas(
        self,
        sharded_data: list,
        task_name: str,
        prefix_tag: str,
    ) -> list[KVBatchMeta]:
        """Stage logprob inputs into the TQ partition."""
        return fan_out_per_rank_metas(
            sharded_data,
            dp_client=self._dp_client,
            partition_id=self._tq_partition_id,
            task_name=task_name,
            key_prefix=self._next_key_prefix(prefix_tag),
            seed_fields=LP_SEED_FIELDS,
        )

    def _fan_out_train_metas(
        self,
        sharded_data: list,
        prefix_tag: str = "step",
    ) -> list[KVBatchMeta]:
        """Stage training inputs into the TQ partition."""
        return fan_out_per_rank_metas(
            sharded_data,
            dp_client=self._dp_client,
            partition_id=self._tq_partition_id,
            task_name="train",
            key_prefix=self._next_key_prefix(prefix_tag),
            seed_fields=DP_SEED_FIELDS,
        )

    # ── overrides — mirror Policy's structure, swap transport ──────────

    def get_logprobs(  # type: ignore[override]
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        timer: Optional[Timer] = None,
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """TQ-mediated counterpart to ``Policy.get_logprobs``.

        Body mirrors the legacy ``get_logprobs`` post-Phase-1 line-for-line:
        ``_shard_for_logprob`` → dispatch → aggregate → reorder. The only
        deltas are the fan-out step (TQ pre-stage) and the worker call
        signature (``meta=metas``, worker method ``*_presharded``).
        """
        with timer.time("get_logprobs/shard_data") if timer else nullcontext():
            sharded_data, unsorted_data_indices = self._shard_for_logprob(data)
            metas = self._fan_out_logprob_metas(
                sharded_data, task_name="prev_lp", prefix_tag="lp",
            )

        with (
            timer.time("get_logprobs/submit_logprob_futures")
            if timer
            else nullcontext()
        ):
            futures = self.worker_group.run_all_workers_sharded_data(
                "get_logprobs_presharded",
                meta=metas,
                in_sharded_axes=["data_parallel"],
                replicate_on_axes=[
                    "context_parallel",
                    "tensor_parallel",
                    "pipeline_parallel",
                ],
                output_is_replicated=[
                    "context_parallel",
                    "tensor_parallel",
                    "pipeline_parallel",
                ],
            )
        logprobs: BatchedDataDict[LogprobOutputSpec] = _aggregate_logprob_results(
            self.worker_group.get_all_worker_results(futures)
        )

        if unsorted_data_indices is not None:
            logprobs.reorder_data(unsorted_data_indices)

        return logprobs

    def get_reference_policy_logprobs(  # type: ignore[override]
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        micro_batch_size: Optional[int] = None,
        timer: Optional[Timer] = None,
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """TQ-mediated counterpart to ``Policy.get_reference_policy_logprobs``.

        Same shape as :meth:`get_logprobs`, just routed to the
        reference-policy worker method and aggregator.
        """
        with (
            timer.time("get_reference_policy_logprobs/shard_data")
            if timer
            else nullcontext()
        ):
            sharded_data, unsorted_data_indices = self._shard_for_logprob(data)
            metas = self._fan_out_logprob_metas(
                sharded_data, task_name="ref_lp", prefix_tag="reflp",
            )

        with (
            timer.time(
                "get_reference_policy_logprobs/submit_reference_policy_logprob_futures"
            )
            if timer
            else nullcontext()
        ):
            futures = self.worker_group.run_all_workers_sharded_data(
                "get_reference_policy_logprobs_presharded",
                meta=metas,
                in_sharded_axes=["data_parallel"],
                replicate_on_axes=[
                    "context_parallel",
                    "tensor_parallel",
                    "pipeline_parallel",
                ],
                output_is_replicated=[
                    "context_parallel",
                    "tensor_parallel",
                    "pipeline_parallel",
                ],
                common_kwargs={"micro_batch_size": micro_batch_size},
            )
        logprobs: BatchedDataDict[ReferenceLogprobOutputSpec] = (
            _aggregate_reference_logprob_results(
                self.worker_group.get_all_worker_results(futures)
            )
        )

        if unsorted_data_indices is not None:
            logprobs.reorder_data(unsorted_data_indices)

        return logprobs

    def train(  # type: ignore[override]
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
        timer: Optional[Timer] = None,
    ) -> dict[str, Any]:
        """TQ-mediated counterpart to ``Policy.train``.

        Body mirrors the legacy ``train`` body post-Phase-1: shard,
        FLOPs accumulation, dispatch, aggregate, FLOPs annotation. The
        deltas are the fan-out step (TQ pre-stage) and the worker call
        signature (``meta=dp_metas``, ``train_presharded``). The
        ``bin_count_multiple=DP_world`` invariant from
        ``a085559c`` is provided by ``self._shard_for_train`` (inherited
        from ``Policy``); ``train_presharded`` reattaches the per-shard
        packing metadata from ``meta.extra_info`` so the worker's local
        ``shards=1`` re-pack doesn't desync Megatron's collectives.
        """
        batch_size = gbs or self.cfg["train_global_batch_size"]
        micro_batch_size = mbs or self.cfg["train_micro_batch_size"]

        with timer.time("policy_training/sharding_data") if timer else nullcontext():
            sharded_data = self._shard_for_train(data, batch_size)
            dp_metas = self._fan_out_train_metas(sharded_data, prefix_tag="step")

        # Drain in finally so a worker exception doesn't leak staged tensors
        # into the next step. Per-instance ``tq_call_idx`` keeps keys unique
        # across calls so we never collide pre-drain, but unbounded growth
        # is wasteful and would eventually evict good data.
        try:
            if self.flops_tracker is not None:
                self.flops_tracker.reset()
                for shard in sharded_data:
                    input_lengths = shard["input_lengths"]
                    self.flops_tracker.track_batch(input_lengths.tolist())

            with (
                timer.time("policy_training/submit_training_futures")
                if timer
                else nullcontext()
            ):
                futures = self.worker_group.run_all_workers_sharded_data(
                    "train_presharded",
                    meta=dp_metas,
                    in_sharded_axes=["data_parallel"],
                    replicate_on_axes=[
                        "context_parallel",
                        "tensor_parallel",
                        "pipeline_parallel",
                    ],
                    output_is_replicated=[
                        "context_parallel",
                        "tensor_parallel",
                        "pipeline_parallel",
                    ],
                    common_kwargs={
                        "loss_fn": loss_fn,
                        "eval_mode": eval_mode,
                        "gbs": batch_size,
                        "mbs": micro_batch_size,
                    },
                )
            results = self.worker_group.get_all_worker_results(futures)
            aggregated_results = _aggregate_train_results(results)

            if self.flops_tracker is not None:
                aggregated_results["total_flops"] = self.flops_tracker.total_flops
                aggregated_results["num_ranks"] = self.worker_group.cluster.world_size()
                gpus_per_worker = self.worker_group.cluster.world_size() / max(
                    len(results), 1
                )
                try:
                    aggregated_results["theoretical_tflops"] = gpus_per_worker * sum(
                        get_theoretical_tflops(r["gpu_name"], r["model_dtype"])
                        for r in results
                    )
                except Exception as e:
                    warnings.warn(f"Error getting theoretical flops: {e}")

            return aggregated_results
        finally:
            try:
                self._dp_client.kv_clear(
                    keys=None, partition_id=self._tq_partition_id,
                )
            except Exception as e:
                warnings.warn(f"Error draining TQ partition after train: {e}")
