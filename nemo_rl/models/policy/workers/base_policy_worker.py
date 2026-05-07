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
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import ray
import torch
import zmq

# Type-only imports — runtime imports of nemo_rl.data_plane are lazy
# inside the data-plane method bodies. This keeps `base_policy_worker`
# importable in worker venvs that don't ship the data-plane extra
# (e.g. the mcore worker venv when data-plane isn't engaged).
if TYPE_CHECKING:
    from nemo_rl.data_plane import DataPlaneConfig, KVBatchMeta
    from nemo_rl.data_plane.interfaces import DataPlaneClient
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.interfaces import ReferenceLogprobOutputSpec
from nemo_rl.utils.nsys import wrap_with_nvtx_name


def _broadcast_batched_data_dict(
    data: Optional[BatchedDataDict[Any]],
    *,
    src: int,
    group: Any,
) -> BatchedDataDict[Any]:
    """Broadcast a BatchedDataDict from ``src`` to all ranks in ``group``.

    Two-phase to avoid pickling tensor payloads on the hot path:
      1. ``broadcast_object_list`` ships a tiny shape descriptor
         (per-key dtype + shape for tensors, raw value for non-tensors).
      2. ``broadcast`` ships each tensor's data on its current device.

    The leader's ``data`` argument supplies the source. Non-leaders pass
    ``None``; an empty :class:`BatchedDataDict` is returned with tensor
    fields filled in-place. Tensors are placed on the current CUDA
    device — callers that want CPU tensors must ``.to("cpu")`` after.
    """
    is_leader = torch.distributed.get_rank() == src
    # NCCL groups can only broadcast CUDA tensors; gloo can do either.
    # Pick the broadcast device from the group backend so CPU-side TQ
    # outputs (input_ids, masks, etc.) are moved to GPU before NCCL
    # broadcast. Non-leaders allocate buffers on the same device.
    backend = torch.distributed.get_backend(group)
    bcast_device: Any = (
        torch.cuda.current_device() if backend == "nccl" else "cpu"
    )

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
            # Restore non-leader tensors to the leader's original device
            # so downstream code sees the same layout it had pre-broadcast.
            if not is_leader and torch.device(src_device).type != torch.device(bcast_device).type:
                out[key] = tensor.to(src_device)
        else:
            if not is_leader:
                out[key] = entry[2]
    return out


class AbstractPolicyWorker:
    """Base class for policy workers with shared functionality."""

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> None:
        """Initialize the collective communication.

        Args:
            ip: IP address for the process group
            port: Port for the process group
            world_size: Total world size (train_world_size + inference_world_size)
            train_world_size: Number of training workers (used in inference cluster)
        """
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        self.model_update_group = StatelessProcessGroup(
            master_address=ip, port=port, rank=self.rank, world_size=world_size
        )
        device = torch.cuda.current_device()
        self.model_update_group.init_nccl_communicator(device=device)

    def is_alive(self) -> bool:
        """Check if the worker is alive."""
        return True

    def reset_peak_memory_stats(self) -> None:
        """Reset peak memory statistics."""
        torch.cuda.reset_peak_memory_stats()

    def get_gpu_info(self) -> dict[str, Any]:
        """Return information about the GPU being used by this worker."""
        from nemo_rl.models.policy.utils import get_gpu_info

        return get_gpu_info(self.model)

    def report_device_id(self) -> str:
        """Report the UUID of the current CUDA device using NVML.

        Returns:
            str: UUID of the device in the format "GPU-xxxxx"
        """
        from nemo_rl.utils.nvml import get_device_uuid

        # Get current device index from torch
        device_idx = torch.cuda.current_device()
        # Get device UUID using NVML
        return get_device_uuid(device_idx)

    def get_zmq_address(self) -> str:
        """Get the ZMQ address for the current device."""
        return f"ipc:///tmp/{self.report_device_id()}.sock"

    def maybe_init_zmq(self) -> None:
        """Initialize the ZMQ socket if it doesn't exist."""
        if not hasattr(self, "zmq_socket"):
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.REQ)
            self.zmq_socket.setsockopt(
                zmq.SNDTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(
                zmq.RCVTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.bind(self.get_zmq_address())

    def get_free_memory_bytes(self) -> int:
        """Get the available free memory."""
        from nemo_rl.utils.nvml import get_free_memory_bytes

        device_idx = torch.cuda.current_device()
        return get_free_memory_bytes(device_idx)

    def shutdown(self) -> bool:
        """Shutdown the policy."""
        try:
            # Clean up extension resources like ZMQ sockets
            if hasattr(self, "zmq_socket"):
                self.zmq_socket.close()
                self.zmq_context.term()
            return True
        except Exception:
            return False

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()

    def report_node_ip_and_gpu_id(self) -> tuple[str, int]:
        """Report the node IP and GPU ID of the current worker."""
        ip = ray._private.services.get_node_ip_address()
        gpu_id = ray.get_gpu_ids()[0]
        return (ip, gpu_id)

    # Temporary fix, 'data' is a kwarg due to some sort of ray bug
    @wrap_with_nvtx_name("policy_worker/get_reference_policy_logprobs")
    def get_reference_policy_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Get the logprobs from the reference policy for a batch of data.

        If micro_batch_size is provided, it will be used instead of the configured
        logprob_batch_size.

        Returns:
          a BatchedDataDict with key "reference_logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        with self.use_reference_model():
            reference_logprobs = self.get_logprobs(
                data=data, micro_batch_size=micro_batch_size
            )

        return_data = BatchedDataDict[ReferenceLogprobOutputSpec]()
        return_data["reference_logprobs"] = reference_logprobs["logprobs"].cpu()
        return return_data

    def finish_training(self, *args: Any, **kwargs: Any) -> None:
        # Placeholder implementation
        pass

    # ──────────────────────────────────────────────────────────────────
    # Data-plane (TransferQueue) integration — per-rank fetch.
    #
    # Driver-side ``TQPolicy`` fans out per-rank ``KVBatchMeta``; each
    # worker calls ``self._fetch(meta, ...)`` to pull its slice from TQ
    # and then runs the existing per-rank method body.
    # ──────────────────────────────────────────────────────────────────

    _dp_client: Optional[DataPlaneClient] = None

    def setup_data_plane(self, cfg: DataPlaneConfig) -> None:
        """Connect this worker process's client to the existing TQ controller.

        Called once by the driver after worker construction (when
        ``data_plane.enabled=True``). Idempotent — second call is a no-op.
        """
        if self._dp_client is not None:
            return
        # Lazy import — keeps the data-plane stack out of the worker
        # module-load path. tensordict + TransferQueue are base deps of
        # nemo-rl now, so they'll always be installed; the lazy import is
        # belt-and-braces against future dep-pruning regressions.
        from nemo_rl.data_plane import build_data_plane_client

        # bootstrap=False — the driver already created the named controller
        # actor; this process attaches as a client.
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

        ``None`` means "no siblings" — TP=CP=PP=1. Backend subclasses
        override (DTensor uses ``device_mesh``, Megatron composes from
        ``parallel_state``). Returning ``None`` makes ``_fetch`` use the
        cheap independent-fetch path; returning a real group makes it
        use leader-fetch + NCCL broadcast.
        """
        return None

    def _pad_value_dict(self) -> dict[str, Any]:
        """Per-field pad value used by :func:`materialize` to detile the
        jagged wire format. Token-id fields use the tokenizer pad id;
        masks / logprobs default to 0 (set by codec)."""
        pad_id = getattr(getattr(self, "tokenizer", None), "pad_token_id", None)
        if pad_id is None:
            return {}
        return {"input_ids": pad_id, "prompt_ids_for_adv": pad_id}

    def _fetch(
        self,
        meta: "KVBatchMeta",
        *,
        layout: str = "padded",
        fetch_policy: str = "auto",
        preprocess: Optional[Any] = None,
    ) -> BatchedDataDict[Any]:
        """Fetch this rank's slice from TQ and return a BatchedDataDict.

        Args:
            meta: per-DP-rank shard produced by the driver's
                :func:`nemo_rl.data_plane.preshard.shard_meta_for_dp`.
            layout: codec layout. ``"padded"`` (default) bridges the
                jagged wire format back to rectangular tensors via
                :func:`torch.nested.to_padded_tensor`, using
                :meth:`_pad_value_dict` for the per-field pad value.
                ``"jagged"`` returns nested tensors as-is.
            fetch_policy: how the rank obtains its slice when TP/CP/PP
                siblings share the same ``meta``:
                  * ``"auto"`` (default) — leader-fetch + NCCL broadcast
                    when ``_get_replica_group()`` returns a group
                    (TP/CP/PP > 1); otherwise every rank fetches
                    independently from TQ (TP=CP=PP=1, the cheapest
                    path).
                  * ``"independent"`` — force every sibling to fetch
                    from TQ. Useful when TQ is local-RAM and broadcast
                    overhead would exceed the duplicated read.
                  * ``"leader_broadcast"`` — force the broadcast path.
                    Asserts a replica group exists. Mostly for testing.
                CP slicing of the fetched/broadcast data happens later
                in the worker's forward prep — ``_fetch`` stays
                parallelism-agnostic.
            preprocess: optional ``(worker, td) -> td`` callable applied
                between fetch+materialize and the user method. Useful for
                per-step transforms that need worker state (config,
                tokenizer). Default ``None`` (identity).
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
                "replica group, but _get_replica_group() returned None. "
                "Either configure TP/CP/PP > 1 or use fetch_policy='auto'."
            )

        pad_to_multiple = int((meta.extra_info or {}).get("pad_to_multiple", 1))

        if replica_group is not None:
            leader = torch.distributed.get_global_rank(replica_group, 0)
            is_leader = torch.distributed.get_rank() == leader
            if is_leader:
                td = self._require_dp_client().kv_batch_get(
                    keys=meta.keys,
                    partition_id=meta.partition_id,
                    select_fields=list(meta.fields) if meta.fields else None,
                )
                data = materialize(
                    td, layout=layout,
                    pad_value_dict=pad_value_dict,
                    pad_to_multiple=pad_to_multiple,
                )
            else:
                data = None
            data = _broadcast_batched_data_dict(
                data, src=leader, group=replica_group,
            )
            if preprocess is not None:
                data = preprocess(self, data)
            return data

        client = self._require_dp_client()
        td = client.kv_batch_get(
            keys=meta.keys,
            partition_id=meta.partition_id,
            select_fields=list(meta.fields) if meta.fields else None,
        )
        data = materialize(
            td, layout=layout,
            pad_value_dict=pad_value_dict,
            pad_to_multiple=pad_to_multiple,
        )
        if preprocess is not None:
            data = preprocess(self, data)
        return data

    def _apply_packing_prep(self, data: BatchedDataDict[Any]) -> BatchedDataDict[Any]:
        """Run the sequence-packing or dynamic-batching pre-pass on a
        per-rank ``BatchedDataDict``.

        The legacy DP path computes ``micro_batch_indices`` /
        ``micro_batch_lengths`` as a *side effect* of
        ``shard_by_batch_size(shards=dp, ..., sequence_packing_args=...)``.
        The TQ presharded path receives a per-rank ``BatchedDataDict``
        without those attrs set; without re-deriving them the worker's
        ``train`` body crashes on ``micro_batch_indices[0]`` (NoneType
        not subscriptable).

        Re-run ``shard_by_batch_size`` with ``shards=1`` on the local
        slice to compute the packing/batching metadata without further
        DP-splitting. Reads packing config from ``self.cfg``.
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
                "sequence_length_pad_multiple": cfg["make_sequence_length_divisible_by"],
                "max_tokens_per_microbatch": seqpack["train_mb_tokens"],
            }
            packed, _ = data.shard_by_batch_size(
                shards=1, batch_size=None, sequence_packing_args=spa,
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
                shards=1, batch_size=None, dynamic_batching_args=dba,
            )
            return sharded[0]

        return data

    def _attach_or_repack_pack_metadata(
        self,
        data: BatchedDataDict[Any],
        meta: "KVBatchMeta",
    ) -> BatchedDataDict[Any]:
        """Reattach driver-side packing metadata or re-derive locally.

        When the driver pre-balanced packing across DP ranks it ships
        per-shard ``micro_batch_indices``/``micro_batch_lengths`` (and
        optionally ``elem_counts_per_gb``) in ``meta.extra_info``.  Trust
        those instead of re-packing locally — local
        ``shard_by_batch_size(shards=1, ...)`` produces variable bin counts
        across DP groups and desyncs Megatron's per-microbatch collectives.

        Falls back to :meth:`_apply_packing_prep` when the driver did not
        populate ``extra_info`` (e.g. legacy in-memory tests).
        """
        extra = meta.extra_info or {}
        if "micro_batch_indices" in extra and "micro_batch_lengths" in extra:
            data.micro_batch_indices = extra["micro_batch_indices"]
            data.micro_batch_lengths = extra["micro_batch_lengths"]
            if "elem_counts_per_gb" in extra:
                data.elem_counts_per_gb = extra["elem_counts_per_gb"]
            return data
        return self._apply_packing_prep(data)

    @wrap_with_nvtx_name("policy_worker/train_presharded")
    def train_presharded(
        self,
        meta: KVBatchMeta,
        loss_fn: Any,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Per-rank training entrypoint. Fetch → packing prep → delegate."""
        data = self._fetch(meta)
        data = self._attach_or_repack_pack_metadata(data, meta)
        return self.train(  # type: ignore[attr-defined]
            data, loss_fn=loss_fn, eval_mode=eval_mode, gbs=gbs, mbs=mbs,
        )

    def _is_replica_leader(self) -> bool:
        """True iff this rank should perform per-DP-rank-unique side-effects
        (e.g. TQ write-back). Returns ``True`` for non-replicated configs."""
        replica_group = self._get_replica_group()
        if replica_group is None:
            return True
        leader = torch.distributed.get_global_rank(replica_group, 0)
        return torch.distributed.get_rank() == leader

    def _write_back(
        self,
        meta: "KVBatchMeta",
        fields: dict[str, torch.Tensor],
    ) -> None:
        """Leader-only ``kv_batch_put(meta.keys, fields=...)``.

        Tensors must be CPU and aligned to ``meta.keys`` order — the TQ
        adapter rejects GPU tensors / shape mismatches.

        Per-token fields are converted to jagged via
        :func:`maybe_pack_jagged` so they land with the same row lengths
        as the initial put. Without this, a worker logprob write-back
        (rectangular ``[N, S]``) would mismatch the jagged ``input_ids``
        on the next read.
        """
        if not self._is_replica_leader() or not fields:
            return
        from tensordict import TensorDict

        from nemo_rl.data_plane.codec import maybe_pack_jagged

        seq_lens = meta.sequence_lengths
        if seq_lens is not None:
            lengths = torch.tensor(seq_lens, dtype=torch.long)
            packed = {k: maybe_pack_jagged(v, lengths) for k, v in fields.items()}
        else:
            packed = {k: v.detach().contiguous() for k, v in fields.items()}

        td = TensorDict(packed, batch_size=[len(meta.keys)])
        self._require_dp_client().kv_batch_put(
            keys=meta.keys, partition_id=meta.partition_id, fields=td,
        )

    def _write_back_result_field(
        self,
        meta: "KVBatchMeta",
        result: Any,
        *,
        result_key: str,
        tq_field: str,
    ) -> None:
        """Write ``result[result_key]`` to TQ as column ``tq_field`` under
        ``meta.keys``. No-op if client unset, key missing, value not a
        tensor, or batch dim mismatched. Leader-only.

        This is the single chokepoint for ``*_presharded`` write-backs —
        keeps the per-method bodies declarative ("fetch, run, write back
        this column") instead of repeating the conditional plumbing.
        """
        if self._dp_client is None:
            return
        # ``BatchedDataDict`` is a ``UserDict``, not ``dict`` — test the
        # ``Mapping`` ABC so the result of ``self.get_logprobs(data)``
        # passes the type guard. ``isinstance(_, dict)`` would silently
        # skip and the worker write-back would never happen.
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
        if val.shape[0] != len(meta.keys):
            raise ValueError(
                f"_write_back_result_field: shape mismatch — "
                f"result[{result_key!r}] has batch dim {val.shape[0]} "
                f"but meta.keys has {len(meta.keys)}."
            )
        self._write_back(meta, {tq_field: val.detach().to("cpu")})

    @wrap_with_nvtx_name("policy_worker/get_logprobs_presharded")
    def get_logprobs_presharded(
        self,
        meta: KVBatchMeta,
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[Any]:
        """Per-rank logprob entrypoint. Fetch → packing prep → run → write back."""
        data = self._fetch(meta)
        data = self._attach_or_repack_pack_metadata(data, meta)
        result: BatchedDataDict[Any] = self.get_logprobs(  # type: ignore[attr-defined]
            data=data, micro_batch_size=micro_batch_size,
        )
        # Canonical TQ column name is "prev_logprobs" (matches DP_SEED_FIELDS
        # and what `train_presharded` fetches for the loss).
        self._write_back_result_field(
            meta, result, result_key="logprobs", tq_field="prev_logprobs",
        )
        return result

    @wrap_with_nvtx_name("policy_worker/get_reference_policy_logprobs_presharded")
    def get_reference_policy_logprobs_presharded(
        self,
        meta: KVBatchMeta,
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Per-rank reference-policy logprob entrypoint. Fetch → packing prep → run → write back."""
        data = self._fetch(meta)
        data = self._attach_or_repack_pack_metadata(data, meta)
        result: BatchedDataDict[ReferenceLogprobOutputSpec] = (
            self.get_reference_policy_logprobs(
                data=data, micro_batch_size=micro_batch_size,
            )
        )
        self._write_back_result_field(
            meta, result,
            result_key="reference_logprobs",
            tq_field="reference_policy_logprobs",
        )
        return result
