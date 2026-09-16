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

import json
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import ray
import torch

from nemo_rl.algorithms.loss.interfaces import LossFunction, LossType
from nemo_rl.data.packed_rollouts import (
    PACKED_ATTENTION_SEGMENT_LENGTHS,
    PACKED_ATTENTION_SELECTED_SEGMENTS,
    TREE_ATTENTION_EDGE_LENGTHS,
    TREE_ATTENTION_EDGE_SOURCE_INDICES,
    TREE_ATTENTION_EDGE_TARGET_IDS,
    TREE_ATTENTION_LAYOUTS,
    TreeAttentionFragmentBatchLayout,
    reassemble_packed_attention_segments,
    reassemble_tree_attention_edge_values,
)
from nemo_rl.data_plane import KVBatchMeta, build_data_plane_client
from nemo_rl.data_plane.column_io import round_up
from nemo_rl.data_plane.driver_mixin import TQDriverMixin
from nemo_rl.data_plane.interfaces import DataPlaneRuntimeConfig
from nemo_rl.data_plane.preshard import (
    expand_meta_for_tree_attention,
    shard_meta_for_dp,
)
from nemo_rl.data_plane.schema import (
    DP_TRAIN_FIELDS,
    ELEM_COUNTS_PER_GB,
    GLOBAL_FORWARD_PAD_SEQLEN,
    LP_SEED_FIELDS,
    MICRO_BATCH_INDICES,
    ROUTE_PASSTHROUGH_FLAG,
    ROUTE_PLAN_TAG,
    fields_with_optional_opd_full,
    fields_with_optional_routed_experts,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.flops_tracker import get_theoretical_tflops
from nemo_rl.utils.timer import Timer

TREE_ATTENTION_TENSOR_FIELDS = (
    TREE_ATTENTION_EDGE_SOURCE_INDICES,
    TREE_ATTENTION_EDGE_TARGET_IDS,
    TREE_ATTENTION_EDGE_LENGTHS,
)


def _with_tree_attention_fields(
    fields: tuple[str, ...] | list[str], meta: KVBatchMeta
) -> list[str]:
    out = list(fields)
    if TREE_ATTENTION_LAYOUTS in (meta.extra_info or {}):
        out.extend(field for field in TREE_ATTENTION_TENSOR_FIELDS if field not in out)
    return out


def _validate_tree_rows(
    meta: KVBatchMeta,
    sequence_packing_args: Optional[dict[str, Any]],
    *,
    max_context_length: int,
) -> None:
    if TREE_ATTENTION_LAYOUTS not in (meta.extra_info or {}):
        return
    if sequence_packing_args is None:
        raise ValueError("tree rollouts require policy.sequence_packing.enabled=true")
    if not meta.sequence_lengths:
        raise ValueError("tree rollout metadata requires physical sequence lengths")
    for layout in meta.extra_info[TREE_ATTENTION_LAYOUTS]:
        if layout.max_path_length > max_context_length:
            raise ValueError(
                "tree rollout logical path exceeds configured context: "
                f"{layout.max_path_length} > {max_context_length}"
            )


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
    if "mtp_metrics" in results[0]:
        out["mtp_metrics"] = results[0]["mtp_metrics"]
    all_mb_metrics: dict[str, list[Any]] = defaultdict(list)
    for r in results:
        for k, v in r["all_mb_metrics"].items():
            all_mb_metrics[k].extend(v)
    out["all_mb_metrics"] = dict(all_mb_metrics)
    # Only the replica leader ever populates this (see
    # TQWorkerMixin._maybe_assemble_routed_experts), so non-leader entries
    # are always empty and summing every result is safe without an
    # is_replica_leader filter; each DP replica leader covers disjoint data.
    route_fallback_counts: Counter[str] = Counter()
    for r in results:
        route_fallback_counts.update(r.get("route_fallback_counts") or {})
    if route_fallback_counts:
        out["route_fallback_counts"] = dict(route_fallback_counts)
    return out


def _model_sequence_lengths(meta: KVBatchMeta) -> list[int]:
    """Return physical model-call lengths, or logical lengths when unpacked."""
    segment_lengths = (meta.extra_info or {}).get(PACKED_ATTENTION_SEGMENT_LENGTHS)
    if segment_lengths is not None:
        return [length for row in segment_lengths for length in row]
    return list(meta.sequence_lengths or [])


def _streamed_packed_broadcast_slots(meta: KVBatchMeta) -> list[int]:
    """Dense token slots per streamed packed bin for one DP-rank meta."""
    extra = meta.extra_info or {}
    segment_lengths = extra.get(PACKED_ATTENTION_SEGMENT_LENGTHS)
    selected_segments = extra.get(PACKED_ATTENTION_SELECTED_SEGMENTS)
    micro_batch_indices = extra.get(MICRO_BATCH_INDICES)
    elem_counts = extra.get(ELEM_COUNTS_PER_GB)
    if (
        segment_lengths is None
        or selected_segments is None
        or micro_batch_indices is None
    ):
        return []

    slots: list[int] = []
    chunk_offset = 0
    for chunk_index, chunk_ranges in enumerate(micro_batch_indices):
        chunk_count = (
            int(elem_counts[chunk_index])
            if elem_counts is not None
            else (int(chunk_ranges[-1][1]) if chunk_ranges else 0)
        )
        for start, stop in chunk_ranges:
            calls = selected_segments[
                chunk_offset + int(start) : chunk_offset + int(stop)
            ]
            if not calls:
                continue
            width = max(
                int(segment_lengths[parent][segment]) for parent, segment in calls
            )
            slots.append(len(calls) * width)
        chunk_offset += chunk_count
    return slots


def _concatenate_packed_logprob_results(
    results: list[Any],
    *,
    result_key: str,
) -> BatchedDataDict[Any]:
    """Pad rank-local call rows to one width and concatenate in dispatch order."""
    tensors: list[torch.Tensor] = []
    for result in results:
        if not isinstance(result, Mapping) or result_key not in result:
            raise RuntimeError(
                "packed logprob worker result must contain "
                f"{result_key!r}, got {type(result).__name__}"
            )
        tensor = result[result_key]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
            raise TypeError(
                f"packed logprob result {result_key!r} must be a rank-2+ tensor"
            )
        tensors.append(tensor)
    if not tensors:
        raise RuntimeError("packed logprob dispatch returned no worker results")

    max_sequence_length = max(tensor.shape[1] for tensor in tensors)
    padded: list[torch.Tensor] = []
    for tensor in tensors:
        if tensor.shape[1] == max_sequence_length:
            padded.append(tensor)
            continue
        output = tensor.new_zeros(
            (tensor.shape[0], max_sequence_length, *tensor.shape[2:])
        )
        output[:, : tensor.shape[1]] = tensor
        padded.append(output)
    return BatchedDataDict[Any]({result_key: torch.cat(padded, dim=0)})


# Unpacked logprob results land in TQ directly from each worker leader. Packed
# results return per-call tensors to the driver so calls split across DP ranks
# can be reassembled into one logical rollout row before the TQ write.


class TQPolicy(TQDriverMixin, Policy):
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
        dp_cfg: DataPlaneRuntimeConfig,
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
        self._router_replay_enabled = bool(
            (self.cfg.get("router_replay") or {}).get("enabled", False)
        )
        # Per-token teacher payload column read by the full-vocabulary MOPD loss.
        # Resolved by the driver in setup; absent means the feature is off and
        # the column must stay out of every fetch.
        self._opd_full_field: Optional[str] = (
            self.cfg.get("on_policy_distillation_full") or {}
        ).get("payload_field")
        self._open_train_loss_type: LossType | None = None

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

    def load_data_plane_checkpoint(self, checkpoint_dir: str | Path) -> dict[str, Any]:
        """Restore TQ through the clean bootstrap client during SC setup."""
        return self.dp_client.load_checkpoint(checkpoint_dir)

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
            fields=fields_with_optional_opd_full(
                fields_with_optional_routed_experts(
                    [*DP_TRAIN_FIELDS, *TREE_ATTENTION_TENSOR_FIELDS],
                    enabled=self._router_replay_enabled,
                ),
                field=self._opd_full_field,
            ),
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
            fields=fields_with_optional_opd_full(
                fields_with_optional_routed_experts(
                    [*DP_TRAIN_FIELDS, *TREE_ATTENTION_TENSOR_FIELDS],
                    enabled=self._router_replay_enabled,
                ),
                field=self._opd_full_field,
            ),
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

        Cross-DP forward pad target. Packed rollouts use the longest physical
        call, matching legacy's expand-before-shard batch width; unpacked
        batches use the longest logical row. Preshard shards inherit it via
        ``dict(meta.extra_info)`` propagation.
        """
        if not meta.sequence_lengths:
            return
        if GLOBAL_FORWARD_PAD_SEQLEN in meta.extra_info:
            return
        _, dba = self._packing_args("train_mb_tokens")
        seq_round = int(dba["sequence_length_round"]) if dba is not None else 1
        pad_mult = int(meta.extra_info.get("pad_to_multiple", 1))
        model_sequence_lengths = _model_sequence_lengths(meta)
        segment_lengths = (meta.extra_info or {}).get(PACKED_ATTENTION_SEGMENT_LENGTHS)
        meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] = (
            max(model_sequence_lengths)
            if segment_lengths is not None
            else round_up(max(model_sequence_lengths), max(pad_mult, seq_round))
        )

    def read_from_dataplane(
        self,
        meta: KVBatchMeta,
        *,
        select_fields: list[str],
        pad_value_dict: Optional[dict[str, Any]] = None,
    ) -> BatchedDataDict[Any]:
        """Materialize columns with their packed-call or tree layout."""
        data = super().read_from_dataplane(
            meta, select_fields=select_fields, pad_value_dict=pad_value_dict
        )
        for key in (PACKED_ATTENTION_SEGMENT_LENGTHS, TREE_ATTENTION_LAYOUTS):
            if key in meta.extra_info:
                data[key] = meta.extra_info[key]
        return data

    def _emit_packed_padding_comparison(
        self,
        *,
        stage: str,
        meta: KVBatchMeta,
        dp_metas: list[KVBatchMeta],
    ) -> None:
        """Report old logical-row vs expanded-call replica-broadcast padding."""
        observability = (getattr(self, "dp_cfg", {}) or {}).get("observability") or {}
        if not observability.get("packing_memory_enabled", False):
            return
        segment_lengths = (meta.extra_info or {}).get(PACKED_ATTENTION_SEGMENT_LENGTHS)
        if segment_lengths is None or not meta.sequence_lengths:
            return

        _, dba = self._packing_args("train_mb_tokens")
        seq_round = int(dba["sequence_length_round"]) if dba is not None else 1
        pad_mult = int(meta.extra_info.get("pad_to_multiple", 1))
        logical_pad = round_up(max(meta.sequence_lengths), max(pad_mult, seq_round))
        physical_pad = int(meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN])
        previous_logical_slots = sum(
            len(rank_meta.sample_ids) * logical_pad for rank_meta in dp_metas
        )
        expanded_call_slots = sum(
            len(rank_meta.extra_info.get(PACKED_ATTENTION_SELECTED_SEGMENTS, []))
            * physical_pad
            for rank_meta in dp_metas
        )
        valid_call_tokens = sum(
            length for row_lengths in segment_lengths for length in row_lengths
        )
        streamed_slots_by_microbatch = [
            slots
            for rank_meta in dp_metas
            for slots in _streamed_packed_broadcast_slots(rank_meta)
        ]
        streamed_call_slots = sum(streamed_slots_by_microbatch)
        event = {
            "stage": stage,
            "logical_rows": len(meta.sample_ids),
            "physical_calls": sum(len(row) for row in segment_lengths),
            "valid_call_tokens": valid_call_tokens,
            "previous_logical_broadcast_slots": previous_logical_slots,
            "expanded_call_broadcast_slots": expanded_call_slots,
            "previous_padding_fraction": (
                1.0 - valid_call_tokens / previous_logical_slots
                if previous_logical_slots
                else 0.0
            ),
            "expanded_padding_fraction": (
                1.0 - valid_call_tokens / expanded_call_slots
                if expanded_call_slots
                else 0.0
            ),
            "streamed_microbatch_broadcast_slots": streamed_call_slots,
            "streamed_padding_fraction": (
                1.0 - valid_call_tokens / streamed_call_slots
                if streamed_call_slots
                else 0.0
            ),
            "peak_streamed_microbatch_broadcast_slots": max(
                streamed_slots_by_microbatch, default=0
            ),
            "broadcast_slot_reduction_fraction": (
                1.0 - expanded_call_slots / previous_logical_slots
                if previous_logical_slots
                else 0.0
            ),
        }
        print(f"tq_packed_padding: {json.dumps(event, sort_keys=True)}", flush=True)

    def _emit_tree_fragmentation(
        self,
        *,
        stage: str,
        meta: KVBatchMeta,
        layout: TreeAttentionFragmentBatchLayout | None,
    ) -> None:
        """Report the physical-token cost and bound of a fragmented tree batch."""
        if layout is None:
            return
        original_layouts = meta.extra_info[TREE_ATTENTION_LAYOUTS]
        real_fragments = [
            fragment for fragment in layout.fragments if not fragment.is_padding
        ]
        event = {
            "stage": stage,
            "logical_rows": layout.original_batch_size,
            "fragments": len(real_fragments),
            "padding_fragments": len(layout.fragments) - len(real_fragments),
            "original_physical_tokens": sum(
                item.unique_token_count for item in original_layouts
            ),
            "fragment_physical_tokens": sum(
                fragment.layout.unique_token_count for fragment in real_fragments
            ),
            "max_original_physical_tokens": max(
                item.unique_token_count for item in original_layouts
            ),
            "max_fragment_physical_tokens": max(
                fragment.layout.unique_token_count for fragment in real_fragments
            ),
            "owned_edges": sum(
                len(fragment.edge_indices) for fragment in real_fragments
            ),
        }
        print(f"tq_tree_fragmentation: {json.dumps(event, sort_keys=True)}", flush=True)

    # ── 1-hop entrypoints (KVBatchMeta in, no re-fan-out) ──────────────────

    def _with_route_fields(
        self,
        meta: KVBatchMeta,
        base_fields: tuple[str, ...] | list[str],
        *,
        task_name: str,
        want_routes: bool,
    ) -> KVBatchMeta:
        """Resolve direct versus deferred route storage for one worker request.

        Delegates to :meth:`TQDriverMixin._isolated_meta` so the narrowed
        meta also gets its per-dispatch forward-pad target minted.
        """
        want = self._router_replay_enabled and want_routes
        plan_presence = [ROUTE_PLAN_TAG in tag for tag in (meta.tags or [])]
        if want and any(plan_presence) and not all(plan_presence):
            raise RuntimeError(
                "router replay does not support mixed direct/deferred route "
                "storage in one worker fetch"
            )
        passthrough = bool(want and plan_presence and all(plan_presence))
        extra_info = dict(meta.extra_info or {})
        if passthrough:
            extra_info[ROUTE_PASSTHROUGH_FLAG] = True
        else:
            extra_info.pop(ROUTE_PASSTHROUGH_FLAG, None)
        return self._isolated_meta(
            replace(meta, extra_info=extra_info),
            fields=fields_with_optional_routed_experts(
                base_fields,
                enabled=want and not passthrough,
            ),
            task_name=task_name,
        )

    def _logprob_dispatch(
        self,
        meta: KVBatchMeta,
        *,
        task_name: str,
        worker_method: str,
        timer_prefix: str,
        timer: Optional[Timer],
        common_kwargs: dict[str, Any],
        include_router_replay: bool = False,
        result_key: str,
        tq_field: str,
    ) -> None:
        """Shared body of get_logprobs_from_meta / get_reference_policy_logprobs_from_meta.

        Logprob workers fetch ``LP_SEED_FIELDS`` plus the multimodal
        columns ``_isolated_meta`` unions in, so prev/ref logprobs see the
        same model inputs as the training forward, which is narrowed through
        the same helper. Narrowing the
        meta's field list still keeps rollout-only payload (message-log
        bulk, ``content``) in TQ. The same shape is used for both prev_lp
        and ref_lp. Workers compute the per-token tensor and commit it to
        TQ via the leader-rank ``_write_back_result_field``; the Ray
        return is always None, so this dispatcher just waits for
        completion.
        """
        spa, dba = self._packing_args("logprob_mb_tokens")
        if TREE_ATTENTION_LAYOUTS in (meta.extra_info or {}):
            self._validate_tree_execution()
            _validate_tree_rows(
                meta, spa, max_context_length=self.cfg["max_total_sequence_length"]
            )
        tree_fragment_layout = (
            expand_meta_for_tree_attention(
                meta,
                max_physical_tokens=int(spa["max_tokens_per_microbatch"]),
                fragment_count_multiple=self.sharding_annotations.get_axis_size(
                    "data_parallel"
                ),
            )
            if spa is not None
            else None
        )
        self._emit_tree_fragmentation(
            stage=task_name,
            meta=meta,
            layout=tree_fragment_layout,
        )
        # Narrow the fetch to LP_SEED_FIELDS + optional routed_experts under
        # R3 replay. ``_isolated_meta`` unions in the multimodal columns the
        # rollout wrote, for this dispatch and the training one alike, so the
        # prev/ref logprobs and the training forward see identical model inputs.
        lp_meta = self._with_route_fields(
            meta,
            _with_tree_attention_fields(LP_SEED_FIELDS, meta),
            task_name=task_name,
            want_routes=include_router_replay,
        )
        with timer.time(f"{timer_prefix}/shard_meta") if timer else nullcontext():
            metas, unsorted_indices = shard_meta_for_dp(
                lp_meta,
                dp_world=self.sharding_annotations.get_axis_size("data_parallel"),
                batch_size=None,
                sequence_packing_args=spa,
                dynamic_batching_args=dba,
            )
        self._emit_packed_padding_comparison(
            stage=task_name,
            meta=lp_meta,
            dp_metas=metas,
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
        worker_results = self.worker_group.get_all_worker_results(futures)
        segment_lengths = (meta.extra_info or {}).get(PACKED_ATTENTION_SEGMENT_LENGTHS)
        if segment_lengths is None and tree_fragment_layout is None:
            return

        packed_results = _concatenate_packed_logprob_results(
            worker_results,
            result_key=result_key,
        )
        if unsorted_indices is not None:
            packed_results.reorder_data(unsorted_indices)
        if tree_fragment_layout is not None:
            reassembled = reassemble_tree_attention_edge_values(
                packed_results[result_key], tree_fragment_layout
            )
        else:
            if not meta.sequence_lengths:
                raise ValueError(
                    "packed attention logprob dispatch requires sequence_lengths"
                )
            reassembled = reassemble_packed_attention_segments(
                packed_results[result_key],
                segment_lengths,
                output_sequence_length=max(meta.sequence_lengths),
            )
        self.write_to_dataplane(meta, fields={tq_field: reassembled})

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
            include_router_replay=True,
            result_key="logprobs",
            tq_field="prev_logprobs",
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
            result_key="reference_logprobs",
            tq_field="reference_policy_logprobs",
        )

    def train_from_meta(
        self,
        meta: KVBatchMeta,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
        timer: Optional[Timer] = None,
        train_fields: tuple[str, ...] = DP_TRAIN_FIELDS,
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
            train_fields: TQ columns workers fetch this step; defaults to the
                full ``DP_TRAIN_FIELDS`` schema. Caller may narrow it to drop
                columns it skipped writing (e.g. ``prev_logprobs`` when
                ``force_on_policy_ratio=True``).

        Returns:
            Aggregated training-step output dict.
        """
        batch_size = gbs or self.cfg["train_global_batch_size"]
        micro_batch_size = mbs or self.cfg["train_micro_batch_size"]
        segment_lengths = (meta.extra_info or {}).get(PACKED_ATTENTION_SEGMENT_LENGTHS)
        worker_batch_size = (
            sum(len(row) for row in segment_lengths)
            if segment_lengths is not None
            else batch_size
        )
        if segment_lengths is not None and not self.cfg.get("megatron_cfg", {}).get(
            "enabled", False
        ):
            raise NotImplementedError(
                "independent model-call packing currently requires the Megatron "
                "policy backend"
            )

        spa, dba = self._packing_args("train_mb_tokens")
        if TREE_ATTENTION_LAYOUTS in (meta.extra_info or {}):
            self._validate_tree_execution()
            _validate_tree_rows(
                meta, spa, max_context_length=self.cfg["max_total_sequence_length"]
            )
        if (
            spa is not None
            and expand_meta_for_tree_attention(
                meta,
                max_physical_tokens=int(spa["max_tokens_per_microbatch"]),
                fragment_count_multiple=self.sharding_annotations.get_axis_size(
                    "data_parallel"
                ),
            )
            is not None
        ):
            raise NotImplementedError(
                "bounded tree fragmentation is supported by the async split TQ "
                "training path; train_from_meta cannot preserve one optimizer step "
                "across streamed fragments"
            )
        # ``train_fields`` (rollout + logprob deltas + advantages + sample_mask;
        # default ``DP_TRAIN_FIELDS``) must be in TQ before this call — written
        # by workers + driver delta-writes. Caller may narrow to drop columns
        # skipped this step (e.g. ``prev_logprobs`` under force_on_policy_ratio).
        # The multimodal columns are per-batch, not part of the static schema,
        # so ``_isolated_meta`` unions them in — without them a VLM training
        # forward would run image-blind while the logprob forwards saw images.
        train_meta = self._with_route_fields(
            meta,
            tuple(
                fields_with_optional_opd_full(
                    _with_tree_attention_fields(train_fields, meta),
                    field=self._opd_full_field,
                )
            ),
            task_name="train",
            want_routes=True,
        )
        with timer.time("policy_training/shard_meta") if timer else nullcontext():
            dp_metas, _ = shard_meta_for_dp(
                train_meta,
                dp_world=self.sharding_annotations.get_axis_size("data_parallel"),
                batch_size=batch_size,
                sequence_packing_args=spa,
                dynamic_batching_args=dba,
            )
        self._emit_packed_padding_comparison(
            stage="train",
            meta=train_meta,
            dp_metas=dp_metas,
        )

        if self.flops_tracker is not None:
            self.flops_tracker.reset()
            self.flops_tracker.track_batch(_model_sequence_lengths(meta))

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
                    "gbs": worker_batch_size,
                    "mbs": micro_batch_size,
                    "scheduler_step_increment": batch_size
                    if segment_lengths is not None
                    else None,
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
    # Counterpart to :meth:`train_from_meta`, consumed directly by
    # :class:`SingleControllerActor` so it can stream microbatches without
    # forcing a full-step optimizer.step on every dispatch.
    #
    # Lifecycle (one step open at a time — workers raise on a second
    # ``begin``, so no step identifier is threaded through the API):
    #   begin_train_step                    — open step; broadcast loss_fn/gbs/mbs
    #   train_microbatches_from_meta (N×)   — DP-sharded fwd/bwd, grads accumulate
    #   finish_train_step                   — all_reduce + opt.step + sched.step
    #   abort_train_step                    — drop accumulators, no opt.step
    #
    # ``train_from_meta`` is unchanged and remains the sync entrypoint.

    def begin_train_step(
        self,
        loss_fn: LossFunction,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> None:
        """Open a logical train step on every worker."""
        batch_size = gbs or self.cfg["train_global_batch_size"]
        micro_batch_size = mbs or self.cfg["train_micro_batch_size"]
        if self.flops_tracker is not None:
            self.flops_tracker.reset()
        # run_all_workers_single_data returns plain ObjectRefs (one per
        # GPU), not a MultiWorkerFuture — consume with ray.get, matching
        # every other single-data fan-out in lm_policy.
        futures = self.worker_group.run_all_workers_single_data(
            "begin_train_step_presharded",
            loss_fn=loss_fn,
            gbs=batch_size,
            mbs=micro_batch_size,
        )
        ray.get(futures)
        self._open_train_loss_type = loss_fn.loss_type

    def train_microbatches_from_meta(
        self,
        meta: KVBatchMeta,
        timer: Optional[Timer] = None,
        train_fields: tuple[str, ...] = DP_TRAIN_FIELDS,
    ) -> None:
        """Dispatch one meta slice (DP-sharded) into an open train step.

        Named plural because one call fans out to every DP rank and the
        backend then iterates its own internal (pipeline/packed)
        microbatches — with a 2x packing ratio a group of G generations is
        G/2 backend microbatches inside this single call, not G/2 calls.

        Mirrors the sharding logic of :meth:`train_from_meta` but without
        a logical-batch sizing constraint: this routes ``meta`` to DP
        ranks and runs forward+backward; gradients accumulate in
        ``.grad``. Returns nothing — per-microbatch metrics accumulate in
        the workers' open-step state and surface once via
        :meth:`finish_train_step`.

        Args:
            meta: Data-plane metadata for the samples in this chunk.
            timer: Optional timer for nested policy-training measurements.
            train_fields: Columns produced for this step and fetched by workers.
        """
        segment_lengths = (meta.extra_info or {}).get(PACKED_ATTENTION_SEGMENT_LENGTHS)
        if segment_lengths is not None and not self.cfg.get("megatron_cfg", {}).get(
            "enabled", False
        ):
            raise NotImplementedError(
                "independent model-call packing currently requires the Megatron "
                "policy backend"
            )
        self._stamp_pad_seqlen(meta)
        spa, dba = self._packing_args("train_mb_tokens")
        if TREE_ATTENTION_LAYOUTS in (meta.extra_info or {}):
            self._validate_tree_execution()
            _validate_tree_rows(
                meta, spa, max_context_length=self.cfg["max_total_sequence_length"]
            )
        tree_fragment_layout = (
            expand_meta_for_tree_attention(
                meta,
                max_physical_tokens=int(spa["max_tokens_per_microbatch"]),
                fragment_count_multiple=self.sharding_annotations.get_axis_size(
                    "data_parallel"
                ),
            )
            if spa is not None
            else None
        )
        if (
            tree_fragment_layout is not None
            and self._open_train_loss_type != LossType.TOKEN_LEVEL
        ):
            raise ValueError(
                "bounded tree fragmentation currently supports token-level losses only"
            )
        self._emit_tree_fragmentation(
            stage="train_microbatch",
            meta=meta,
            layout=tree_fragment_layout,
        )
        train_meta = self._with_route_fields(
            meta,
            # Raw fields, not pre-wrapped in fields_with_optional_routed_experts:
            # _with_route_fields applies that wrapper itself, gated on both
            # router replay and route-plan passthrough. The opd_full payload
            # column has no such gate, so it is appended here.
            tuple(
                fields_with_optional_opd_full(
                    _with_tree_attention_fields(train_fields, meta),
                    field=self._opd_full_field,
                )
            ),
            task_name="train",
            want_routes=True,
        )
        with timer.time("policy_training/shard_meta") if timer else nullcontext():
            dp_metas, _ = shard_meta_for_dp(
                train_meta,
                dp_world=self.sharding_annotations.get_axis_size("data_parallel"),
                batch_size=None,
                sequence_packing_args=spa,
                dynamic_batching_args=dba,
            )
        self._emit_packed_padding_comparison(
            stage="train_microbatch",
            meta=train_meta,
            dp_metas=dp_metas,
        )

        self._dispatch_train_microbatches(dp_metas, timer=timer)

    def train_placed_microbatches(
        self,
        dp_metas: list[KVBatchMeta],
        timer: Optional[Timer] = None,
    ) -> None:
        """Dispatch one producer-assigned metadata batch per logical DP rank.

        The input order is the logical DP-rank order. Producer field lists
        remain unchanged because an SFT loader can provide a narrower schema
        than the rollout training path.
        """
        dp_world = self.sharding_annotations.get_axis_size("data_parallel")
        if len(dp_metas) != dp_world:
            raise ValueError(
                "Placed metadata must contain exactly one batch per DP rank: "
                f"got {len(dp_metas)} batches for dp_world={dp_world}."
            )
        spa, dba = self._packing_args("train_mb_tokens")
        if spa is not None or dba is not None:
            raise ValueError(
                "Placed metadata supports fixed batches only. Disable NeMo-RL "
                "sequence packing and dynamic batching."
            )
        train_metas = [
            replace(meta, task_name="train")
            for meta in self._stamp_placed_pad_seqlen(dp_metas)
        ]
        self._dispatch_train_microbatches(train_metas, timer=timer)

    def _stamp_placed_pad_seqlen(
        self, dp_metas: list[KVBatchMeta]
    ) -> list[KVBatchMeta]:
        """Mint one fresh forward padding target across all placed DP batches.

        Returns new metadata rather than mutating the caller's, and ignores any
        inherited target: reusing one would let it ratchet upward across steps
        and pad every later step to a historical maximum. This mirrors
        ``TQDriverMixin._isolated_meta``, which pops the key for the same reason.
        """
        sequence_lengths = [
            length for meta in dp_metas for length in (meta.sequence_lengths or [])
        ]
        if not sequence_lengths:
            return list(dp_metas)
        _, dynamic_args = self._packing_args("train_mb_tokens")
        sequence_round = (
            int(dynamic_args["sequence_length_round"])
            if dynamic_args is not None
            else 1
        )
        pad_multiple = max(
            [int(meta.extra_info.get("pad_to_multiple", 1)) for meta in dp_metas]
        )
        target = round_up(max(sequence_lengths), max(pad_multiple, sequence_round))
        return [
            replace(
                meta,
                extra_info={**meta.extra_info, GLOBAL_FORWARD_PAD_SEQLEN: target},
            )
            for meta in dp_metas
        ]

    def _dispatch_train_microbatches(
        self,
        dp_metas: list[KVBatchMeta],
        *,
        timer: Optional[Timer],
    ) -> None:
        """Send prepared per-DP metadata into an open train step."""
        if self.flops_tracker is not None:
            for rank_meta in dp_metas:
                self.flops_tracker.track_batch(list(rank_meta.sequence_lengths or []))

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
            )
        # Wait for completion only — workers return None (metrics
        # accumulate in their open-step state until finish_train_step).
        self.worker_group.get_all_worker_results(futures)

    def finish_train_step(self) -> dict[str, Any]:
        """Close an open train step: all_reduce, rescale, optimizer.step.

        Aggregates per-rank step results into the same shape as
        :meth:`train_from_meta` so callers don't have to special-case
        the split path.
        """
        futures = self.worker_group.run_all_workers_single_data(
            "finish_train_step_presharded",
        )
        results = ray.get(futures)
        # Filter to DP-replica leaders only. ``run_all_workers_single_data``
        # returns one result per GPU (TP×CP×PP×DP), but TP/CP/non-last-PP
        # twins hold identical copies of their DP shard's metric list.
        # Aggregating without dedup inflates every per-token metric by
        # TP*CP*(1 if PP==1 else PP_last_stage_count). ``train_from_meta``
        # gets this for free via ``output_is_replicated`` on its sharded
        # dispatch; finish has no data to shard, so we dedupe here.
        leader_results = [r for r in results if r.get("is_replica_leader", True)]
        aggregated_results = _aggregate_train_results(leader_results)

        if self.flops_tracker is not None:
            aggregated_results["total_flops"] = self.flops_tracker.total_flops
            aggregated_results["num_ranks"] = self.worker_group.cluster.world_size()

        self._open_train_loss_type = None
        return aggregated_results

    def abort_train_step(self) -> None:
        """Drop partial step state on every worker. No optimizer.step."""
        futures = self.worker_group.run_all_workers_single_data(
            "abort_train_step_presharded",
        )
        ray.get(futures)
        self._open_train_loss_type = None

        if self.flops_tracker is not None:
            self.flops_tracker.reset()
