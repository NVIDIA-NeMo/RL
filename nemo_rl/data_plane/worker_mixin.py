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
"""TransferQueue awareness for policy workers, isolated from the base class.

Mix into a worker class to add per-rank TQ-mediated entrypoints
(:meth:`train_presharded`, :meth:`get_logprobs_presharded`,
:meth:`get_reference_policy_logprobs_presharded`, and the frozen-teacher
variant) without touching
``BasePolicyWorker``. Subclasses that don't need TQ keep their bare
inheritance and stay zero-cost.

Subclasses must implement :meth:`_get_replica_group` (returns the
NCCL group of TP×CP×PP siblings within this DP rank, or ``None`` for
TP=CP=PP=1) and inherit ``train`` / ``get_logprobs`` /
``get_reference_policy_logprobs`` from the worker base.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

import torch

FetchPolicy = Literal["auto", "independent", "leader_broadcast"]

from nemo_rl.data.llm_message_utils import attach_message_log_view
from nemo_rl.data_plane.schema import (
    ELEM_COUNTS_PER_GB,
    GLOBAL_FORWARD_PAD_SEQLEN,
    MICRO_BATCH_INDICES,
    MICRO_BATCH_LENGTHS,
    ROUTE_PASSTHROUGH_FLAG,
    ROUTE_PLAN_TAG,
    ROUTED_EXPERTS_ENCODING_FIELD,
    ROUTED_EXPERTS_FIELD,
    ROUTED_EXTRAS_METADATA_FIELD,
    Layout,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict, SequencePackingArgs
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.r3_trace import trace_tq_fetch_payload

if TYPE_CHECKING:
    from nemo_rl.data_plane import DataPlaneConfig, KVBatchMeta
    from nemo_rl.data_plane.interfaces import DataPlaneClient


@dataclass(frozen=True)
class _DeferredRouteFragment:
    routes: torch.Tensor
    encoding: int
    extras_metadata_json: torch.Tensor


def _broadcast_batched_data_dict(
    data: Optional[BatchedDataDict[Any]],
    *,
    is_leader: bool,
    src: int,
    group: Any,
) -> BatchedDataDict[Any]:
    """Broadcast a BatchedDataDict from ``src`` to all ranks in ``group``.

    Two-phase to avoid pickling tensor payloads on the hot path: a small
    descriptor (per-key dtype/shape) ships via ``broadcast_object_list``
    first, then each tensor's data ships via ``broadcast`` on its
    current device. The leader supplies ``data``; non-leaders pass
    ``None`` and get an empty BatchedDataDict filled in-place.
    """
    # NCCL groups can only broadcast CUDA tensors; pick the broadcast
    # device from the group backend so CPU TQ outputs are moved to GPU
    # before NCCL broadcast.
    backend = torch.distributed.get_backend(group)
    bcast_device: Any = torch.cuda.current_device() if backend == "nccl" else "cpu"

    if is_leader:
        assert data is not None, "leader must provide non-None data"
        descriptor: list[Any] = []
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                descriptor.append(
                    (k, "tensor", str(v.dtype), tuple(v.shape), str(v.device))
                )
            else:
                descriptor.append((k, "raw", v))
        payload: list[Any] = [descriptor]
    else:
        payload = [None]

    torch.distributed.broadcast_object_list(payload, src=src, group=group)
    descriptor = payload[0]
    assert descriptor is not None

    # pyrefly: ignore  # bad-assignment
    out: BatchedDataDict[Any] = data if is_leader else BatchedDataDict()
    for entry in descriptor:
        key = entry[0]
        kind = entry[1]
        if kind == "tensor":
            dtype_str, shape, src_device = entry[2], entry[3], entry[4]
            if is_leader:
                tensor = out[key]
                if tensor.device.type != torch.device(bcast_device).type:
                    tensor = tensor.to(bcast_device)
                    out[key] = tensor
            else:
                dtype = getattr(torch, dtype_str.split(".")[-1])
                tensor = torch.empty(shape, dtype=dtype, device=bcast_device)
                out[key] = tensor
            # NCCL has no int16 ("Short") type; ship as int32 and narrow back
            # (routed_experts rides TQ as int16).
            if tensor.dtype == torch.int16:
                wire = tensor.to(torch.int32)
                torch.distributed.broadcast(wire, src=src, group=group)
                tensor = wire.to(torch.int16)
                out[key] = tensor
            else:
                torch.distributed.broadcast(tensor, src=src, group=group)
            # Restore non-leader tensors to the leader's source device
            # so downstream code sees the same layout pre-broadcast.
            if (
                not is_leader
                and torch.device(src_device).type != torch.device(bcast_device).type
            ):
                out[key] = tensor.to(src_device)
        else:
            if not is_leader:
                out[key] = entry[2]
    return out


class TQWorkerMixin:
    """Adds TransferQueue per-rank fetch/write-back to a policy worker.

    The driver-side ``TQPolicy`` fans out per-rank ``KVBatchMeta``;
    each worker calls ``self._fetch(meta, ...)`` to pull its slice from
    TQ and runs the existing per-rank method body.
    """

    _dp_client: Optional[DataPlaneClient] = None
    _route_fallback_counts: Counter[str] = Counter()

    def setup_data_plane(self, cfg: DataPlaneConfig) -> None:
        """Connect this worker process's client to the existing TQ controller.

        Called once by the driver after worker construction. Idempotent.
        """
        if getattr(self, "model_slices_context_parallel_inputs", False):
            raise NotImplementedError(
                "TransferQueue/SingleController does not yet support models that "
                "insert media before context-parallel input selection. Use the "
                "synchronous NeMo-RL policy path for Nemotron Omni."
            )
        if self._dp_client is not None:
            return
        self._route_fallback_counts = Counter()
        from nemo_rl.data_plane import build_data_plane_client

        # bootstrap=False — the driver already created the named
        # controller actor; this process attaches as a client.
        self._dp_client = build_data_plane_client(cfg, bootstrap=False)

    def _require_dp_client(self) -> DataPlaneClient:
        if self._dp_client is None:
            raise RuntimeError(
                "Data-plane client not initialised on worker. The driver "
                "must call setup_data_plane(cfg) before invoking any "
                "*_presharded entrypoint."
            )
        return self._dp_client

    def _get_replica_group(self) -> Optional[Any]:
        """NCCL group of TP×CP×PP siblings within this DP rank.

        ``None`` means "no siblings" (TP=CP=PP=1). Subclasses must
        override using their parallelism state (DTensor ``device_mesh``,
        Megatron ``parallel_state``). Returning ``None`` makes
        :meth:`_fetch` use independent fetch; returning a group makes
        it use leader-fetch + NCCL broadcast.
        """
        return None

    def _routed_experts_dimensions(self) -> tuple[int, int]:
        """Return model-owned ``(num_moe_layers, top_k)`` route dimensions."""
        raise NotImplementedError(
            "the router-replay policy worker must provide route dimensions"
        )

    def _pad_value_dict(self) -> dict[str, Any]:
        """Per-field pad value used by :func:`materialize` to detile the jagged wire format.

        Token-id fields use the tokenizer pad id.
        """
        pad_id = getattr(getattr(self, "tokenizer", None), "pad_token_id", None)
        if pad_id is None:
            return {}
        return {"input_ids": pad_id, "prompt_ids_for_adv": pad_id}

    def _forward_pad_seqlen(self, meta: "KVBatchMeta") -> int:
        """Cross-DP forward pad target, minted by :meth:`TQPolicy._stamp_pad_seqlen`."""
        return int((meta.extra_info or {}).get(GLOBAL_FORWARD_PAD_SEQLEN, 0))

    def _fetch(
        self,
        meta: "KVBatchMeta",
        *,
        layout: Layout = "padded",
        fetch_policy: FetchPolicy = "auto",
        preprocess: Optional[Any] = None,
        dp_aligned_seq_len: bool = True,
    ) -> BatchedDataDict[Any]:
        """Fetch this rank's slice from TQ and return a BatchedDataDict.

        Args:
            meta: Per-rank ``KVBatchMeta`` from :func:`shard_meta_for_dp`.
                Forward-pass pad target is read from
                ``meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN]`` minted by
                :meth:`TQPolicy._stamp_pad_seqlen`.
            layout: Materialization layout (``"padded"`` or ``"jagged"``).
            fetch_policy: ``"auto"`` uses leader-fetch + NCCL broadcast when
                :meth:`_get_replica_group` returns a group, else independent
                fetch (cheapest for TP=CP=PP=1). ``"independent"`` forces
                every sibling to fetch. ``"leader_broadcast"`` forces the
                broadcast path and asserts a replica group exists.
            preprocess: Optional ``(worker, td) -> td`` applied between
                materialize and return.
            dp_aligned_seq_len: When True (default), right-pad the seq
                dim for the forward pass. Disabled in tests that want
                to observe per-rank local-pad behavior.

        Returns:
            ``BatchedDataDict`` of this rank's slice.
        """
        if fetch_policy not in {"auto", "independent", "leader_broadcast"}:
            raise ValueError(f"unknown fetch_policy: {fetch_policy!r}")

        from nemo_rl.data_plane import materialize

        pad_value_dict = self._pad_value_dict()
        replica_group = (
            self._get_replica_group()
            if fetch_policy in {"auto", "leader_broadcast"}
            else None
        )
        if fetch_policy == "leader_broadcast" and replica_group is None:
            raise RuntimeError(
                "_fetch(fetch_policy='leader_broadcast') requires a "
                "replica group, but _get_replica_group() returned None."
            )

        pad_to_seqlen = self._forward_pad_seqlen(meta) if dp_aligned_seq_len else 0

        if replica_group is not None and replica_group.size() > 1:
            is_leader = self._is_replica_leader()
            leader = torch.distributed.get_global_rank(replica_group, 0)
            if is_leader:
                td = self._require_dp_client().get_samples(
                    sample_ids=meta.sample_ids,
                    partition_id=meta.partition_id,
                    select_fields=list(meta.fields),  # type: ignore[no-matching-overload]
                )
                data = materialize(
                    td,
                    layout=layout,
                    pad_value_dict=pad_value_dict,
                    pad_to_seqlen=pad_to_seqlen,
                )
                data = self._maybe_assemble_routed_experts(meta, data)
            else:
                data = None
            data = _broadcast_batched_data_dict(
                data,
                is_leader=is_leader,
                src=leader,
                group=replica_group,
            )
            # Reconstruct message_log after broadcast so the views alias
            # the per-rank local ``input_ids`` rather than the leader's.
            attach_message_log_view(data)
            trace_tq_fetch_payload(
                stage=meta.task_name or "unknown",
                keys=meta.sample_ids,
                data=data,
            )
            if preprocess is not None:
                data = preprocess(self, data)
            return data

        td = self._require_dp_client().get_samples(
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=list(meta.fields),  # type: ignore[no-matching-overload]
        )
        data = materialize(
            td,
            layout=layout,
            pad_value_dict=pad_value_dict,
            pad_to_seqlen=pad_to_seqlen,
        )
        data = self._maybe_assemble_routed_experts(meta, data)
        attach_message_log_view(data)
        trace_tq_fetch_payload(
            stage=meta.task_name or "unknown",
            keys=meta.sample_ids,
            data=data,
        )
        if preprocess is not None:
            data = preprocess(self, data)
        return data

    def _fetch_route_fragments(
        self,
        *,
        keys: list[str],
        partition_id: str,
    ) -> dict[str, _DeferredRouteFragment]:
        """Fetch a unique key set in one request and preserve request identity."""
        if not keys:
            return {}
        rows = self._require_dp_client().get_samples(
            sample_ids=keys,
            partition_id=partition_id,
            select_fields=[
                ROUTED_EXPERTS_FIELD,
                ROUTED_EXPERTS_ENCODING_FIELD,
                ROUTED_EXTRAS_METADATA_FIELD,
            ],
        )
        n_rows = int(rows.batch_size[0]) if len(rows.batch_size) else 0
        if n_rows != len(keys):
            raise KeyError(f"requested {len(keys)} route rows, got {n_rows}")
        route_column = rows.get(ROUTED_EXPERTS_FIELD)
        encoding_column = rows.get(ROUTED_EXPERTS_ENCODING_FIELD)
        metadata_column = rows.get(ROUTED_EXTRAS_METADATA_FIELD)
        if route_column is None or encoding_column is None or metadata_column is None:
            raise KeyError("deferred route row is missing integrity metadata")
        return {
            key: _DeferredRouteFragment(
                routes=route_column[index],
                encoding=int(encoding_column[index].reshape(-1)[0].item()),
                extras_metadata_json=metadata_column[index].reshape(-1),
            )
            for index, key in enumerate(keys)
        }

    @staticmethod
    def _verify_route_fragment_integrity(
        fragment: _DeferredRouteFragment,
        *,
        extras_digest_version: int,
        expected_extras_digest: str,
    ) -> bool:
        """Rebuild deferred extras and verify the receipt-bound Gym digest."""
        from nemo_gym.token_id_capture.staging.digest import (
            EXTRAS_DIGEST_VERSION,
            compute_extras_digest,
        )
        from nemo_rl.utils.routed_experts_codec import encode_routed_experts

        if extras_digest_version != EXTRAS_DIGEST_VERSION:
            return False
        try:
            raw_metadata = bytes(
                int(value) for value in fragment.extras_metadata_json.tolist()
            )
            decoded = json.loads(raw_metadata.decode("utf-8"))
            if decoded is None:
                extras: dict[str, Any] = {}
            elif isinstance(decoded, dict):
                extras = decoded
            else:
                return False
            if fragment.encoding == 1:
                extras[ROUTED_EXPERTS_FIELD] = encode_routed_experts(fragment.routes)
            elif fragment.encoding == 2:
                extras[ROUTED_EXPERTS_FIELD] = fragment.routes.tolist()
            else:
                return False
            return compute_extras_digest(extras) == expected_extras_digest
        except (TypeError, ValueError):
            return False

    def _route_fragments_by_row(
        self,
        plans: list[Any],
    ) -> tuple[list[dict[str, _DeferredRouteFragment]], int, float]:
        """Use one normal-path batch read; isolate error retries per rollout."""
        from nemo_rl.experience.route_plan import decode_route_plan

        decoded = [decode_route_plan(plan) for plan in plans]
        partitions = {plan.staging_partition for plan in decoded}
        if len(partitions) != 1:
            raise RuntimeError(
                f"deferred route plans use mixed staging partitions: {partitions}"
            )
        partition_id = next(iter(partitions))
        keys = list(
            dict.fromkeys(
                span.staging_key
                for plan in decoded
                for span in plan.spans
                if span.staged_route_len > 0
            )
        )
        fetch_start = time.perf_counter()
        try:
            fragments = self._fetch_route_fragments(
                keys=keys,
                partition_id=partition_id,
            )
        except Exception as batch_error:  # noqa: BLE001 - isolate fallback by rollout
            logging.getLogger(__name__).warning(
                "deferred route batch fetch failed; isolating by rollout: %s",
                batch_error,
            )
            per_row: list[dict[str, _DeferredRouteFragment]] = []
            for plan in decoded:
                row_keys = list(
                    dict.fromkeys(
                        span.staging_key
                        for span in plan.spans
                        if span.staged_route_len > 0
                    )
                )
                try:
                    per_row.append(
                        self._fetch_route_fragments(
                            keys=row_keys,
                            partition_id=partition_id,
                        )
                    )
                except Exception:  # noqa: BLE001 - this rollout becomes sentinel
                    per_row.append({})
            return (
                per_row,
                len(keys),
                (time.perf_counter() - fetch_start) * 1000.0,
            )
        return (
            [fragments for _ in decoded],
            len(keys),
            (time.perf_counter() - fetch_start) * 1000.0,
        )

    def _maybe_assemble_routed_experts(
        self,
        meta: "KVBatchMeta",
        data: BatchedDataDict[Any],
    ) -> BatchedDataDict[Any]:
        """Materialize deferred routes at the policy worker consumption boundary."""
        if not (meta.extra_info or {}).get(ROUTE_PASSTHROUGH_FLAG):
            return data

        from nemo_rl.experience.route_plan import (
            classify_route_span,
            decode_route_plan,
        )
        from nemo_rl.models.generation.interfaces import (
            ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL,
        )

        tags = meta.tags or []
        if len(tags) != len(meta.sample_ids):
            raise RuntimeError(
                "deferred route tags must align with sample_ids: "
                f"{len(tags)} tags for {len(meta.sample_ids)} rows"
            )
        encoded_plans = []
        for index, tag in enumerate(tags):
            if ROUTE_PLAN_TAG not in tag:
                raise RuntimeError(
                    f"deferred route plan missing for row {meta.sample_ids[index]!r}"
                )
            encoded_plans.append(tag[ROUTE_PLAN_TAG])
        plans = [decode_route_plan(plan) for plan in encoded_plans]
        fragments_by_row, _, _ = self._route_fragments_by_row(encoded_plans)

        num_moe_layers, top_k = self._routed_experts_dimensions()
        input_ids = data["input_ids"]
        input_lengths = data["input_lengths"].reshape(-1)
        routed = torch.full(
            (
                len(meta.sample_ids),
                int(input_ids.shape[1]),
                num_moe_layers,
                top_k,
            ),
            ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL,
            dtype=torch.int16,
        )
        request_fallbacks: Counter[str] = Counter()
        for row_index, (plan, fragments) in enumerate(zip(plans, fragments_by_row)):
            reason: Optional[str] = None
            canonical_len = int(input_lengths[row_index].item())
            if canonical_len != plan.expected_token_length:
                reason = "canonical_length_mismatch"
            position = 0
            if reason is None:
                for span in plan.spans:
                    contribution = span.carry_len + span.generation_len
                    mode = classify_route_span(span)
                    if mode != "sentinel":
                        fragment = fragments.get(span.staging_key)
                        if fragment is None:
                            reason = "missing_fragment"
                            break
                        if not self._verify_route_fragment_integrity(
                            fragment,
                            extras_digest_version=span.extras_digest_version,
                            expected_extras_digest=span.extras_digest,
                        ):
                            reason = "fragment_integrity"
                            break
                        routes = fragment.routes
                        if routes.dim() != 3:
                            reason = "fragment_rank"
                            break
                        if int(routes.shape[0]) != span.staged_route_len:
                            reason = "fragment_length"
                            break
                        if tuple(routes.shape[1:]) != (num_moe_layers, top_k):
                            reason = "fragment_model_shape"
                            break
                        if mode == "full":
                            routed[row_index, position : position + contribution] = (
                                routes.to(torch.int16)
                            )
                        else:
                            tail_start = position + span.carry_len
                            routed[row_index, tail_start : position + contribution] = (
                                routes[-span.generation_len :].to(torch.int16)
                            )
                    position += contribution
            if reason is None and plan.spans and position != canonical_len:
                reason = "assembled_length_mismatch"
            if reason is not None:
                routed[row_index].fill_(ROUTED_EXPERTS_MISSING_ROUTE_SENTINEL)
                request_fallbacks[reason] += 1

        self._route_fallback_counts.update(request_fallbacks)
        if request_fallbacks:
            logging.getLogger(__name__).warning(
                "deferred route fallback for %d/%d rollouts: %s",
                sum(request_fallbacks.values()),
                len(plans),
                dict(request_fallbacks),
            )
        data[ROUTED_EXPERTS_FIELD] = routed
        return data

    def _apply_packing_prep(self, data: BatchedDataDict[Any]) -> BatchedDataDict[Any]:
        """Re-derive ``micro_batch_indices`` / ``micro_batch_lengths`` on the local slice.

        Uses ``shard_by_batch_size(shards=1, ...)``. The legacy DP path computes those
        as a side effect of the DP-shard call; the TQ presharded path receives a
        per-rank slice without them set, so we recompute here using ``self.cfg``.
        """
        cfg = getattr(self, "cfg", None)
        if not isinstance(cfg, dict):
            return data
        seqpack = cfg.get("sequence_packing", {}) or {}
        dynbatch = cfg.get("dynamic_batching", {}) or {}

        if seqpack.get("enabled", False):
            spa: SequencePackingArgs = {
                "algorithm": seqpack["algorithm"],
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_pad_multiple": cfg[
                    "make_sequence_length_divisible_by"
                ],
                "max_tokens_per_microbatch": seqpack["train_mb_tokens"],
            }
            microbatch_order = seqpack.get("microbatch_order")
            if microbatch_order is not None:
                spa["microbatch_order"] = microbatch_order
            packed, _ = data.shard_by_batch_size(
                shards=1,
                batch_size=None,
                sequence_packing_args=spa,
            )
            return packed[0]

        if dynbatch.get("enabled", False):
            dba = {
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_round": dynbatch["sequence_length_round"],
                "max_tokens_per_microbatch": dynbatch["train_mb_tokens"],
            }
            sharded, _ = data.shard_by_batch_size(
                shards=1,
                batch_size=None,
                # pyrefly: ignore  # bad-argument-type
                dynamic_batching_args=dba,
            )
            return sharded[0]

        return data

    def _attach_or_repack_pack_metadata(
        self,
        data: BatchedDataDict[Any],
        meta: "KVBatchMeta",
    ) -> BatchedDataDict[Any]:
        """Trust driver-supplied packing metadata or re-derive locally.

        When the driver pre-balanced packing across DP ranks it ships
        ``micro_batch_indices`` / ``micro_batch_lengths`` (and optionally
        ``elem_counts_per_gb``) in ``meta.extra_info``. Locally
        re-packing produces variable bin counts across DP groups and
        desyncs Megatron's per-microbatch collectives — trust the driver
        when it provided the metadata.
        """
        extra = meta.extra_info or {}
        if MICRO_BATCH_INDICES in extra and MICRO_BATCH_LENGTHS in extra:
            data.micro_batch_indices = extra[MICRO_BATCH_INDICES]
            data.micro_batch_lengths = extra[MICRO_BATCH_LENGTHS]
            if ELEM_COUNTS_PER_GB in extra:
                data.elem_counts_per_gb = extra[ELEM_COUNTS_PER_GB]
            return data
        return self._apply_packing_prep(data)

    def _local_coords(self) -> dict[str, int]:
        """This worker's (axis -> local-rank) mapping.

        Subclasses MUST override: DTensor reads ``device_mesh``,
        Megatron reads ``parallel_state``. There's no honest default —
        a missing impl would silently make every rank a writeback
        leader and re-create the ``-601 ILLEGAL_CLIENT`` duplicate-write
        bug.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _local_coords() to gate TQ writeback. "
            "Return (axis -> local rank) from the worker's parallelism state."
        )

    def _is_replica_leader(self) -> bool:
        """True iff this rank should perform per-DP-rank-unique side-effects.

        Examples include TQ write-back. Shares the same predicate the
        driver uses to gate dispatch (:meth:`NamedSharding.is_axis_zero`)
        — fed by per-worker :meth:`_local_coords` instead of
        ``NamedSharding.get_worker_coords``; same answer either way.
        """
        from nemo_rl.distributed.named_sharding import REPLICATED_AXES, NamedSharding

        return NamedSharding.is_axis_zero(self._local_coords(), REPLICATED_AXES)

    def _write_back(
        self,
        meta: "KVBatchMeta",
        fields: dict[str, torch.Tensor],
    ) -> None:
        """Leader-only ``put_samples(meta.sample_ids, fields=...)``.

        Per-token fields are jagged-packed via :func:`pack_per_token_field`
        so they land with the same row lengths as the initial put;
        without this a worker write-back (rectangular ``[N, S]``) would
        mismatch the jagged ``input_ids`` on the next read.

        Args:
            meta: Per-rank ``KVBatchMeta`` for this slice.
            fields: Map of field name to tensor to write back.
        """
        if not self._is_replica_leader() or not fields:
            return
        from nemo_rl.data_plane.column_io import write_columns

        write_columns(self._require_dp_client(), meta, fields)

    def _write_back_result_field(
        self,
        meta: "KVBatchMeta",
        result: Any,
        *,
        result_key: str,
        tq_field: str,
    ) -> None:
        """Single chokepoint for ``*_presharded`` write-backs.

        ``result`` is checked via the ``Mapping`` ABC because
        ``BatchedDataDict`` is a ``UserDict`` (not ``dict``).

        Args:
            meta: Per-rank ``KVBatchMeta`` for this slice.
            result: Worker output containing ``result_key``.
            result_key: Key into ``result`` for the tensor to write back.
            tq_field: Field name on the TQ side.
        """
        if self._dp_client is None:
            return
        from collections.abc import Mapping

        if not isinstance(result, Mapping) or result_key not in result:
            raise RuntimeError(
                f"_write_back_result_field: result type {type(result).__name__} "
                f"missing key {result_key!r}; cannot write back."
            )
        val = result[result_key]
        if not isinstance(val, torch.Tensor):
            raise TypeError(
                f"_write_back_result_field: result[{result_key!r}] is "
                f"{type(val).__name__}, expected torch.Tensor."
            )
        if val.shape[0] != len(meta.sample_ids):
            raise ValueError(
                f"_write_back_result_field: shape mismatch — "
                f"result[{result_key!r}] has batch dim {val.shape[0]} "
                f"but meta.sample_ids has {len(meta.sample_ids)}."
            )
        self._write_back(meta, {tq_field: val.detach().to("cpu")})

    @wrap_with_nvtx_name("policy_worker/train_presharded")
    def train_presharded(
        self,
        meta: "KVBatchMeta",
        loss_fn: Any,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Per-rank training entrypoint. Fetch → packing prep → delegate."""
        data = self._fetch(meta)
        data = self._attach_or_repack_pack_metadata(data, meta)
        return self.train(  # type: ignore[attr-defined]
            data,
            loss_fn=loss_fn,
            eval_mode=eval_mode,
            gbs=gbs,
            mbs=mbs,
        )

    @wrap_with_nvtx_name("policy_worker/get_logprobs_presharded")
    def get_logprobs_presharded(
        self,
        meta: "KVBatchMeta",
        micro_batch_size: Optional[int] = None,
    ) -> None:
        """Per-rank logprob entrypoint. Fetch → packing prep → run → write back.

        Returns ``None`` — the per-token tensor is committed to TQ via
        :meth:`_write_back_result_field` under ``prev_logprobs``.
        Callers fetch it through :meth:`TQPolicy.read_from_dataplane` —
        skipping the Ray plasma roundtrip on the (B, S) tensor.
        ``del result`` drops the local reference before returning so the
        worker doesn't carry the tensor into the next dispatch.
        """
        data = self._fetch(meta)
        data = self._attach_or_repack_pack_metadata(data, meta)
        result: BatchedDataDict[Any] = self.get_logprobs(  # type: ignore[attr-defined]
            data=data,
            micro_batch_size=micro_batch_size,
        )
        self._write_back_result_field(
            meta,
            result,
            result_key="logprobs",
            tq_field="prev_logprobs",
        )
        del result

    @wrap_with_nvtx_name("policy_worker/get_reference_policy_logprobs_presharded")
    def get_reference_policy_logprobs_presharded(
        self,
        meta: "KVBatchMeta",
        micro_batch_size: Optional[int] = None,
    ) -> None:
        """Per-rank reference-policy logprob entrypoint.

        See :meth:`get_logprobs_presharded` for the contract. Tensor
        lives in TQ under ``reference_policy_logprobs``.
        """
        data = self._fetch(meta)
        data = self._attach_or_repack_pack_metadata(data, meta)
        result: BatchedDataDict[Any] = self.get_reference_policy_logprobs(  # type: ignore[attr-defined]
            data=data,
            micro_batch_size=micro_batch_size,
        )
        self._write_back_result_field(
            meta,
            result,
            result_key="reference_logprobs",
            tq_field="reference_policy_logprobs",
        )
        del result

    @wrap_with_nvtx_name("policy_worker/get_teacher_logprobs_presharded")
    def get_teacher_logprobs_presharded(
        self,
        meta: "KVBatchMeta",
        micro_batch_size: Optional[int] = None,
    ) -> None:
        """Per-rank frozen-teacher logprob entrypoint for SingleController MOPD."""
        data = self._fetch(meta)
        cfg = getattr(self, "cfg", {})
        batching_enabled = bool(
            cfg.get("sequence_packing", {}).get("enabled", False)
            or cfg.get("dynamic_batching", {}).get("enabled", False)
        )
        extra = meta.extra_info or {}
        if batching_enabled and not (
            MICRO_BATCH_INDICES in extra and MICRO_BATCH_LENGTHS in extra
        ):
            raise RuntimeError(
                "SingleController teacher batching requires driver-provided global "
                "micro_batch_indices and micro_batch_lengths; local worker planning "
                "can desynchronize data-parallel collectives."
            )
        data = self._attach_or_repack_pack_metadata(data, meta)
        result: BatchedDataDict[Any] = self.get_logprobs(  # type: ignore[attr-defined]
            data=data,
            micro_batch_size=micro_batch_size,
        )
        self._write_back_result_field(
            meta,
            result,
            result_key="logprobs",
            tq_field="teacher_reference_logprobs",
        )
        del result

    @wrap_with_nvtx_name("value_worker/get_values_presharded")
    def get_values_presharded(
        self,
        meta: "KVBatchMeta",
        micro_batch_size: Optional[int] = None,
    ) -> None:
        """Per-rank value-forward entrypoint. Fetch → packing prep → run → write back.

        Same contract as get_logprobs_presharded, and only the value workers
        mix it in: only the PPO critic implements get_values.
        """
        data = self._fetch(meta)
        data = self._attach_or_repack_pack_metadata(data, meta)
        result: BatchedDataDict[Any] = self.get_values(  # type: ignore[attr-defined]
            data=data,
            micro_batch_size=micro_batch_size,
        )
        self._write_back_result_field(
            meta,
            result,
            result_key="values",
            tq_field="values",
        )
        del result

    # ── split-API entrypoints (SC async path) ──────────────────────────────
    #
    # The split path lets SingleController drive forward/backward per
    # microbatch (or per pipeline-batch on Megatron) without stepping the
    # optimizer until a full logical batch has accumulated. Backend
    # methods (``begin_train_step``, ``train_microbatch``,
    # ``finish_train_step``, ``abort_train_step``) own the train-step
    # state machine; this mixin just gates them on TQ-presharded data.

    @wrap_with_nvtx_name("policy_worker/begin_train_step_presharded")
    def begin_train_step_presharded(
        self,
        loss_fn: Any,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> None:
        """Open a logical train step. No fetch — pure lifecycle.

        The backend stores ``loss_fn`` / ``gbs`` / ``mbs``, clears
        gradients, and initialises accumulators for ``local_valid_seqs``
        / ``local_valid_toks`` and any per-microbatch metrics. Only one
        step can be open at a time — the backend raises on a second
        ``begin`` — so no step identifier is needed. Optimizer state is
        untouched here.
        """
        self.begin_train_step(  # type: ignore[attr-defined]
            loss_fn=loss_fn,
            gbs=gbs,
            mbs=mbs,
        )

    @wrap_with_nvtx_name("policy_worker/train_microbatch_presharded")
    def train_microbatch_presharded(
        self,
        meta: "KVBatchMeta",
    ) -> None:
        """Per-rank microbatch entrypoint. Fetch → packing prep → forward+backward.

        Gradients accumulate into ``.grad`` across calls; no
        ``optimizer.step`` here. Returns nothing — per-microbatch metrics
        accumulate in the backend's open-step state and surface once via
        ``finish_train_step_presharded``.
        """
        data = self._fetch(meta)
        data = self._attach_or_repack_pack_metadata(data, meta)
        self.train_microbatch(  # type: ignore[attr-defined]
            data=data,
        )

    @wrap_with_nvtx_name("policy_worker/finish_train_step_presharded")
    def finish_train_step_presharded(self) -> dict[str, Any]:
        """Close a logical train step. No fetch — pure lifecycle.

        Backend all-reduces accumulated ``local_valid_seqs/toks``,
        rescales gradients to the final global normalization, runs grad
        clip, steps the optimizer + scheduler, then zeros gradients.
        Returns the aggregated step result (``loss``, ``grad_norm``,
        ``all_mb_metrics``, …).

        Tags the result with ``is_replica_leader`` so the driver-side
        aggregator can dedupe TP/CP/non-last-PP-stage twins that hold
        identical copies of this DP shard's metrics. Without it the
        driver's ``run_all_workers_single_data`` returns one dict per
        GPU and the metric list ends up TP×CP×PP times too long, which
        inflates every per-token aggregate (gen_kl_error, probs_ratio,
        etc.) by that same factor.
        """
        result = self.finish_train_step()  # type: ignore[attr-defined]
        result["is_replica_leader"] = bool(self._is_replica_leader())
        return result

    @wrap_with_nvtx_name("policy_worker/abort_train_step_presharded")
    def abort_train_step_presharded(self) -> None:
        """Discard partial train-step state without stepping the optimizer.

        Used when SC decides the logical batch will not complete (e.g.
        weight-sync triggered mid-step). Backend drops accumulators and
        zeros gradients.
        """
        self.abort_train_step()  # type: ignore[attr-defined]
