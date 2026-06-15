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
"""TQ-mediated Policy: meta-driven 1-hop counterpart to ``Policy``.

Exposes ``train_from_meta`` / ``get_logprobs_from_meta`` /
``get_reference_policy_logprobs_from_meta`` — same return shapes as
``Policy.{train, get_logprobs, get_reference_policy_logprobs}`` but
accepting a ``KVBatchMeta`` instead of a ``BatchedDataDict``. The meta
names per-sample TQ keys minted once at rollout
(:class:`nemo_rl.experience.sync_rollout_actor.SyncRolloutActor`); each
dispatch slices the key list per DP rank via
:func:`nemo_rl.data_plane.preshard.shard_meta_for_dp` (no re-fan-out,
no key minting). Workers fetch their slice from TQ via
``self._fetch(meta)`` and write deltas back via
``self._write_back_result_field(...)``. See
``nemo_rl/data_plane/README.md`` for the full design.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Optional

import ray

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.data_plane import KVBatchMeta, build_data_plane_client
from nemo_rl.data_plane.column_io import read_columns, round_up, write_columns
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.data_plane.schema import (
    DP_TRAIN_FIELDS,
    GLOBAL_FORWARD_PAD_SEQLEN,
    LP_SEED_FIELDS,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.flops_tracker import get_theoretical_tflops
from nemo_rl.utils.timer import Timer

# ──────────────────────────────────────────────────────────────────────────
# Per-stage aggregators that assemble per-rank worker results into the
# shape each Policy method returns. Used by the TQ-mediated overrides
# below; kept out of ``lm_policy.Policy`` since the legacy in-memory
# path doesn't fan out per-rank and never calls these.
# ──────────────────────────────────────────────────────────────────────────


def _aggregate_train_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "loss": results[0]["global_loss"],
        "grad_norm": results[0]["grad_norm"],
    }
    if "moe_metrics" in results[0]:
        out["moe_metrics"] = results[0]["moe_metrics"]
    all_mb_metrics: dict[str, list[Any]] = defaultdict(list)
    for r in results:
        for k, v in r["all_mb_metrics"].items():
            all_mb_metrics[k].extend(v)
    out["all_mb_metrics"] = dict(all_mb_metrics)
    return out


# Logprob results land in TQ directly via the worker-side
# ``_write_back_result_field`` leader path; the per-rank Ray return is
# always None (see :meth:`TQWorkerMixin.get_logprobs_presharded`). The
# dispatcher only waits for completion — no aggregation needed.


class TQPolicy(Policy):
    """TQ-mediated counterpart to :class:`Policy`.

    Constructor accepts an additional ``dp_cfg`` (the
    ``master_config["data_plane"]`` dict). Bootstraps the controller on
    the driver and forwards ``setup_data_plane(dp_cfg)`` to every worker
    so they can attach as clients (``bootstrap=False``).

    The partition lifecycle (``register_partition`` / ``clear_samples``) is
    the trainer's responsibility — this class assumes the partition
    named ``self.tq_partition_id`` (default ``"train"``) is open with a
    schema covering ``DP_TRAIN_FIELDS`` (the bulk schema written by the
    rollout actor at first put + driver-/worker-written deltas).
    """

    def __init__(
        self,
        *args: Any,
        dp_cfg: dict[str, Any],
        tq_partition_id: str = "train",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Validate the topology the data plane fan-out (`shard_meta_for_dp`)
        # depends on. Failing here surfaces a clear error at policy
        # construction; the same condition is re-checked inside
        # `shard_meta_for_dp` as a defensive invariant.
        dp_world = self.sharding_annotations.get_axis_size("data_parallel")
        if dp_world <= 0:
            raise ValueError(
                f"TQPolicy requires data_parallel axis size > 0, got {dp_world}. "
                f"Check cluster config (gpus_per_node * num_nodes) vs. "
                f"TP/PP/CP/EP sizes."
            )
        self.dp_cfg = dp_cfg
        self.dp_client = build_data_plane_client(dp_cfg, bootstrap=True)
        self.tq_partition_id = tq_partition_id

        # Forward to workers (replaces ``Policy.setup_data_plane`` call
        # site in the trainer — TQPolicy bundles bootstrap + worker
        # attach into construction so the trainer just instantiates
        # ``TQPolicy(...)`` and is done).
        ray.get(
            self.worker_group.run_all_workers_single_data(
                "setup_data_plane", cfg=dp_cfg
            )
        )

    # ── lifecycle ──────────────────────────────────────────────────────

    def shutdown(self) -> bool:  # type: ignore[override]
        """Close the TQ client before shutting down the worker group."""
        try:
            self.dp_client.close()
        except Exception as e:
            warnings.warn(f"Error closing data-plane client: {e}")
        return super().shutdown()

    def prepare_step(
        self,
        num_samples: int,
        group_size: Optional[int] = None,
    ) -> None:
        """Register the per-step TQ partition.

        Sync trainers call this at the start of each step. The static
        partition id ``"train"`` is cleared and reused across steps. The
        schema is the union of all consumer fields — producers write
        only the subset they have, consumers fetch via ``select_fields``.

        Args:
            num_samples: Expected total samples this step.
            group_size: GRPO group size for balanced sampling; ``None`` disables grouping.
        """
        self.dp_client.register_partition(
            partition_id=self.tq_partition_id,
            fields=list(DP_TRAIN_FIELDS),
            num_samples=num_samples,
            consumer_tasks=["prev_lp", "ref_lp", "train"],
            grpo_group_size=group_size,
        )

    def prepare_val_partition(
        self, num_samples: int, *, partition_id: str = "val"
    ) -> None:
        """Register a per-batch val partition (single consumer, no GRPO grouping).

        Sync val trainers call this at the start of each val batch.
        Distinct from :meth:`prepare_step` because val has its own
        partition id and a single consumer task.
        """
        self.dp_client.register_partition(
            partition_id=partition_id,
            fields=list(DP_TRAIN_FIELDS),
            num_samples=num_samples,
            consumer_tasks=[partition_id],
            grpo_group_size=None,
        )

    def discard_samples(self, sample_ids: list[str], partition_id: str) -> None:
        """Drop a set of uids from TQ.

        Used both for step-end teardown (via :meth:`finish_step`) and
        mid-step filtering (e.g. dynamic sampling).
        """
        self.dp_client.clear_samples(sample_ids=sample_ids, partition_id=partition_id)

    def finish_step(self, meta: KVBatchMeta) -> None:
        """Drop this step's bulk from TQ. Mirror of :meth:`prepare_step`."""
        self.discard_samples(meta.sample_ids, meta.partition_id)

    def _stamp_pad_seqlen(self, meta: KVBatchMeta) -> None:
        """Mint ``GLOBAL_FORWARD_PAD_SEQLEN`` onto ``meta.extra_info`` (idempotent).

        Cross-DP forward pad target. Preshard shards inherit it via
        ``dict(meta.extra_info)`` propagation.
        """
        if not meta.sequence_lengths:
            return
        if GLOBAL_FORWARD_PAD_SEQLEN in meta.extra_info:
            return
        _, dba = self._packing_args("train_mb_tokens")
        seq_round = int(dba["sequence_length_round"]) if dba is not None else 1
        pad_mult = int(meta.extra_info.get("pad_to_multiple", 1))
        meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] = round_up(
            max(meta.sequence_lengths), max(pad_mult, seq_round)
        )

    def read_from_dataplane(
        self,
        meta: KVBatchMeta,
        *,
        select_fields: list[str],
        pad_value_dict: Optional[dict[str, Any]] = None,
    ) -> BatchedDataDict[Any]:
        """Fetch + materialize columns from the data plane (TQ).

        ``read_columns`` pads to ``meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN]``
        — the same value workers pad to in their forward pass. Driver
        and workers thus return columns at one identical seq dim, with
        no driver-side knowledge of ``sequence_length_round``.
        """
        self._stamp_pad_seqlen(meta)
        return read_columns(
            self.dp_client,
            meta,
            select_fields=select_fields,
            pad_value_dict=pad_value_dict,
        )

    def write_to_dataplane(self, meta: KVBatchMeta, fields: dict[str, Any]) -> None:
        """Write driver-computed columns to the data plane (TQ)."""
        write_columns(self.dp_client, meta, fields=fields)

    # ── 1-hop entrypoints (KVBatchMeta in, no re-fan-out) ──────────────────

    def _packing_args(
        self,
        mb_tokens_key: str,
    ) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        """Resolve (sequence_packing_args, dynamic_batching_args) for a given stage.

        The stage is identified by ``mb_tokens_key`` (``"logprob_mb_tokens"`` or
        ``"train_mb_tokens"``).
        """
        if getattr(self, "use_dynamic_batches", False):
            args = dict(self.dynamic_batching_args)
            args["max_tokens_per_microbatch"] = self.cfg["dynamic_batching"][
                mb_tokens_key
            ]
            return None, args
        if getattr(self, "use_sequence_packing", False):
            args = dict(self.sequence_packing_args)
            args["max_tokens_per_microbatch"] = self.cfg["sequence_packing"][
                mb_tokens_key
            ]
            return args, None
        return None, None

    def _logprob_dispatch(
        self,
        meta: KVBatchMeta,
        *,
        task_name: str,
        worker_method: str,
        timer_prefix: str,
        timer: Optional[Timer],
        common_kwargs: dict[str, Any],
    ) -> None:
        """Shared body of get_logprobs_from_meta / get_reference_policy_logprobs_from_meta.

        Logprob workers need only LP_SEED_FIELDS — narrow the meta's
        field list so ``_fetch`` doesn't pull rollout-only payload (e.g.
        multimodal). The same shape is used for both prev_lp and ref_lp.
        Workers compute the per-token tensor and commit it to TQ via the
        leader-rank ``_write_back_result_field``; the Ray return is
        always None, so this dispatcher just waits for completion.
        """
        self._stamp_pad_seqlen(meta)
        spa, dba = self._packing_args("logprob_mb_tokens")
        lp_meta = replace(meta, fields=list(LP_SEED_FIELDS), task_name=task_name)
        with timer.time(f"{timer_prefix}/shard_meta") if timer else nullcontext():
            metas, _ = shard_meta_for_dp(
                lp_meta,
                dp_world=self.sharding_annotations.get_axis_size("data_parallel"),
                batch_size=None,
                sequence_packing_args=spa,
                dynamic_batching_args=dba,
            )
        with timer.time(f"{timer_prefix}/submit_futures") if timer else nullcontext():
            futures = self.worker_group.run_all_workers_sharded_data(
                worker_method,
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
                common_kwargs=common_kwargs,
            )
        # Wait for completion; per-rank returns are None.
        self.worker_group.get_all_worker_results(futures)

    def get_logprobs_from_meta(
        self,
        meta: KVBatchMeta,
        micro_batch_size: Optional[int] = None,
        timer: Optional[Timer] = None,
    ) -> None:
        self._logprob_dispatch(
            meta,
            task_name="prev_lp",
            worker_method="get_logprobs_presharded",
            timer_prefix="get_logprobs",
            timer=timer,
            common_kwargs={"micro_batch_size": micro_batch_size},
        )

    def get_reference_policy_logprobs_from_meta(
        self,
        meta: KVBatchMeta,
        micro_batch_size: Optional[int] = None,
        timer: Optional[Timer] = None,
    ) -> None:
        self._logprob_dispatch(
            meta,
            task_name="ref_lp",
            worker_method="get_reference_policy_logprobs_presharded",
            timer_prefix="get_reference_policy_logprobs",
            timer=timer,
            common_kwargs={"micro_batch_size": micro_batch_size},
        )

    def train_from_meta(
        self,
        meta: KVBatchMeta,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
        timer: Optional[Timer] = None,
    ) -> dict[str, Any]:
        """1-hop counterpart to :meth:`train`.

        ``meta`` names per-sample keys; columns written by the rollout
        actor + worker logprob deltas + driver-side advantage delta have
        all landed under the same keys at this point. Workers fetch the
        union via ``train_presharded`` → ``self._fetch(meta)``. No
        partition drain here — sync 1-hop's trainer calls ``clear_samples``
        once at end of step.

        Args:
            meta: Full-step ``KVBatchMeta`` (consumed by all DP ranks).
            gbs: Global batch size; defaults to ``cfg["train_global_batch_size"]``.
            mbs: Micro batch size; defaults to ``cfg["train_micro_batch_size"]``.
            timer: Optional timer for nested ``policy_training/*`` measurements.

        Returns:
            Aggregated training-step output dict.
        """
        batch_size = gbs or self.cfg["train_global_batch_size"]
        micro_batch_size = mbs or self.cfg["train_micro_batch_size"]

        self._stamp_pad_seqlen(meta)
        spa, dba = self._packing_args("train_mb_tokens")
        # Train workers fetch the full DP_TRAIN_FIELDS schema (rollout +
        # logprob deltas + advantages + sample_mask). Caller is responsible
        # for ensuring those columns have been written to TQ before this
        # call (workers + driver delta-writes).
        train_meta = replace(
            meta,
            fields=list(DP_TRAIN_FIELDS),
            task_name="train",
        )
        with timer.time("policy_training/shard_meta") if timer else nullcontext():
            dp_metas, _ = shard_meta_for_dp(
                train_meta,
                dp_world=self.sharding_annotations.get_axis_size("data_parallel"),
                batch_size=batch_size,
                sequence_packing_args=spa,
                dynamic_batching_args=dba,
            )

        if self.flops_tracker is not None:
            self.flops_tracker.reset()
            for m in dp_metas:
                self.flops_tracker.track_batch(list(m.sequence_lengths or []))

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

    # ── split-API fanout (SC async path) ───────────────────────────────────
    #
    # Counterpart to :meth:`train_from_meta`, exposed to ``PolicyTrainerActor``
    # so :class:`SingleControllerActor` can stream microbatches without
    # forcing a full-step optimizer.step on every dispatch.
    #
    # Lifecycle:
    #   begin_train_step                  — open step; broadcast loss_fn/gbs/mbs
    #   train_microbatch_from_meta (N×)   — DP-sharded fwd/bwd, grads accumulate
    #   finish_train_step                 — all_reduce + opt.step + sched.step
    #   abort_train_step                  — drop accumulators, no opt.step
    #
    # ``train_from_meta`` is unchanged and remains the sync entrypoint.

    def begin_train_step(
        self,
        step_id: str,
        loss_fn: LossFunction,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> None:
        """Open a logical train step on every worker."""
        batch_size = gbs or self.cfg["train_global_batch_size"]
        micro_batch_size = mbs or self.cfg["train_micro_batch_size"]
        if self.flops_tracker is not None:
            self.flops_tracker.reset()
        futures = self.worker_group.run_all_workers_single_data(
            "begin_train_step_presharded",
            step_id=step_id,
            loss_fn=loss_fn,
            gbs=batch_size,
            mbs=micro_batch_size,
        )
        ray.get(futures)

    def train_microbatch_from_meta(
        self,
        step_id: str,
        meta: KVBatchMeta,
        timer: Optional[Timer] = None,
    ) -> dict[str, Any]:
        """Dispatch one microbatch (DP-sharded) into an open train step.

        Mirrors the sharding logic of :meth:`train_from_meta` but without
        a logical-batch sizing constraint: this routes ``meta`` to DP
        ranks and runs forward+backward; gradients accumulate in
        ``.grad``. The optimizer step happens at :meth:`finish_train_step`.
        """
        self._stamp_pad_seqlen(meta)
        spa, dba = self._packing_args("train_mb_tokens")
        train_meta = replace(
            meta,
            fields=list(DP_TRAIN_FIELDS),
            task_name="train",
        )
        with timer.time("policy_training/shard_meta") if timer else nullcontext():
            dp_metas, _ = shard_meta_for_dp(
                train_meta,
                dp_world=self.sharding_annotations.get_axis_size("data_parallel"),
                batch_size=None,
                sequence_packing_args=spa,
                dynamic_batching_args=dba,
            )

        if self.flops_tracker is not None:
            for m in dp_metas:
                self.flops_tracker.track_batch(list(m.sequence_lengths or []))

        with (
            timer.time("policy_training/submit_microbatch_futures")
            if timer
            else nullcontext()
        ):
            futures = self.worker_group.run_all_workers_sharded_data(
                "train_microbatch_presharded",
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
                common_kwargs={"step_id": step_id},
            )
        results = self.worker_group.get_all_worker_results(futures)
        # Per-microbatch metrics: pass through DP-rank-0 by convention,
        # backend may aggregate later if needed. Surface as-is for now.
        return results[0] if results else {}

    def finish_train_step(self, step_id: str) -> dict[str, Any]:
        """Close an open train step: all_reduce, rescale, optimizer.step.

        Aggregates per-rank step results into the same shape as
        :meth:`train_from_meta` so callers don't have to special-case
        the split path.
        """
        futures = self.worker_group.run_all_workers_single_data(
            "finish_train_step_presharded",
            step_id=step_id,
        )
        results = ray.get(futures)
        aggregated_results = _aggregate_train_results(results)

        if self.flops_tracker is not None:
            aggregated_results["total_flops"] = self.flops_tracker.total_flops
            aggregated_results["num_ranks"] = self.worker_group.cluster.world_size()

        return aggregated_results

    def abort_train_step(self, step_id: str) -> None:
        """Drop partial step state on every worker. No optimizer.step."""
        futures = self.worker_group.run_all_workers_single_data(
            "abort_train_step_presharded",
            step_id=step_id,
        )
        ray.get(futures)

        if self.flops_tracker is not None:
            self.flops_tracker.reset()
