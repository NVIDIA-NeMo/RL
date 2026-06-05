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
:meth:`get_reference_policy_logprobs_presharded`) without touching
``BasePolicyWorker``. Subclasses that don't need TQ keep their bare
inheritance and stay zero-cost.

Subclasses must implement :meth:`_get_replica_group` (returns the
NCCL group of TP×CP×PP siblings within this DP rank, or ``None`` for
TP=CP=PP=1) and inherit ``train`` / ``get_logprobs`` /
``get_reference_policy_logprobs`` from the worker base.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional

import torch

FetchPolicy = Literal["auto", "independent", "leader_broadcast"]

from nemo_rl.data.llm_message_utils import attach_message_log_view
from nemo_rl.data_plane.schema import (
    ELEM_COUNTS_PER_GB,
    GLOBAL_FORWARD_PAD_SEQLEN,
    MICRO_BATCH_INDICES,
    MICRO_BATCH_LENGTHS,
    Layout,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.r3_trace import trace_tq_fetch_payload

if TYPE_CHECKING:
    from nemo_rl.data_plane import DataPlaneConfig, KVBatchMeta
    from nemo_rl.data_plane.interfaces import DataPlaneClient


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

    def setup_data_plane(self, cfg: DataPlaneConfig) -> None:
        """Connect this worker process's client to the existing TQ controller.

        Called once by the driver after worker construction. Idempotent.
        """
        if self._dp_client is not None:
            return
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
        attach_message_log_view(data)
        trace_tq_fetch_payload(
            stage=meta.task_name or "unknown",
            keys=meta.sample_ids,
            data=data,
        )
        if preprocess is not None:
            data = preprocess(self, data)
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
            spa = {
                "algorithm": seqpack["algorithm"],
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_pad_multiple": cfg[
                    "make_sequence_length_divisible_by"
                ],
                "max_tokens_per_microbatch": seqpack["train_mb_tokens"],
            }
            packed, _ = data.shard_by_batch_size(
                shards=1,
                batch_size=None,
                # pyrefly: ignore  # bad-argument-type
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
