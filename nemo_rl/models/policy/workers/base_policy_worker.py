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
    # Data-plane (TransferQueue) integration — Stage 4 per-rank fetch.
    #
    # Pairs with ``@dp_dispatch(...)`` on the driver-side Policy methods.
    # The driver fans out per-rank ``KVBatchMeta``; each worker calls
    # ``self._fetch(meta, ...)`` to pull its slice from TQ, then runs
    # the existing legacy method body. No decorator is used here on
    # purpose — keeping the worker side as straight Python makes
    # debugging the fetch path obvious.
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

    def _fetch(
        self,
        meta: "KVBatchMeta",
        *,
        layout: str = "padded",
        fetch_policy: str = "independent",
        preprocess: Optional[Any] = None,
    ) -> BatchedDataDict[Any]:
        """Fetch this rank's slice from TQ and return a BatchedDataDict.

        Args:
            meta: per-DP-rank shard produced by the driver's
                :func:`shard_keys_by_seqlen`.
            layout: codec layout. Phase 1 always ``"padded"`` — the
                wire format is already padded. Stage 2 will introduce
                ``"jagged"``.
            fetch_policy: who calls ``kv_batch_get`` when this rank has
                TP/CP/PP siblings sharing the same ``meta``:
                  * ``"independent"`` — every sibling fetches (Phase 1
                    default; correct because Phase 1 is FSDP2 only with
                    TP=CP=PP=1, so there are no siblings).
                  * ``"leader_broadcast"`` — rank-zero of the replicated
                    axes fetches and broadcasts via NCCL inside the
                    sibling group. To be implemented when mcore TP/CP/PP
                    lands; see plan §Stage 4 TP/CP/PP subsection.
            preprocess: optional ``(worker, td) -> td`` callable applied
                between fetch+materialize and the user method. Useful for
                per-step transforms that need worker state (config,
                tokenizer). Default ``None`` (identity).
        """
        if fetch_policy not in {"independent", "leader_broadcast"}:
            raise ValueError(f"unknown fetch_policy: {fetch_policy!r}")
        if fetch_policy == "leader_broadcast":
            # Phase 2 / mcore. Defer until siblings actually exist.
            raise NotImplementedError(
                "fetch_policy='leader_broadcast' will land with mcore "
                "TP/CP/PP support — see plan §Stage 4. Phase 1 (FSDP2 "
                "with TP=CP=PP=1) uses 'independent', which is correct "
                "because there are no siblings to share work with."
            )

        # Lazy import — see setup_data_plane().
        from nemo_rl.data_plane import materialize

        client = self._require_dp_client()
        td = client.kv_batch_get(
            keys=meta.keys,
            partition_id=meta.partition_id,
            select_fields=list(meta.fields) if meta.fields else None,
        )
        data = materialize(td, layout=layout)
        if preprocess is not None:
            data = preprocess(self, data)
        return data

    def _apply_packing_prep(self, data: BatchedDataDict[Any]) -> BatchedDataDict[Any]:
        """Run the sequence-packing or dynamic-batching pre-pass on a
        per-rank ``BatchedDataDict``.

        The legacy DP path computes ``micro_batch_indices`` /
        ``micro_batch_lengths`` as a *side effect* of
        ``shard_by_batch_size(shards=dp, ..., sequence_packing_args=...)``.
        Our TQ presharded path does the DP-split via
        :func:`shard_keys_by_seqlen` (control-plane only), so the
        per-rank ``BatchedDataDict`` returned from ``_fetch`` arrives
        without those attrs set — and the worker's ``train`` body crashes
        on ``micro_batch_indices[0]`` (NoneType not subscriptable).

        Re-run ``shard_by_batch_size`` with ``shards=1`` on the local
        slice to compute the packing/batching metadata without further
        DP-splitting. Reads packing config from ``self.cfg`` (set in
        the worker's ``__init__``); no extra plumbing through the
        decorator.
        """
        cfg = getattr(self, "cfg", None)
        if not isinstance(cfg, dict):
            return data
        seqpack = cfg.get("sequence_packing", {}) or {}
        dynbatch = cfg.get("dynamic_batching", {}) or {}

        # Worker-local step counter for [DP_DEBUG] correlation across
        # ranks. Same-call-index across ranks should produce the same
        # packing layout under DP=1; divergence is the smoking gun for
        # the seqpack-TQ step-4 hang.
        if not hasattr(self, "_dp_debug_call_idx"):
            self._dp_debug_call_idx = 0
        self._dp_debug_call_idx += 1
        idx = self._dp_debug_call_idx

        def _dp_log(stage: str, **fields: Any) -> None:
            try:
                import torch.distributed as _dist
                rank = _dist.get_rank() if _dist.is_initialized() else -1
            except Exception:
                rank = -1
            kvs = " ".join(f"{k}={v}" for k, v in fields.items())
            print(f"[DP_DEBUG rank={rank} call={idx} stage={stage}] {kvs}", flush=True)

        # Pre-pack snapshot (after _fetch, before packing).
        try:
            il = data.get("input_lengths")
            il_summary = (
                il.tolist() if hasattr(il, "tolist") else list(il)
            )[:8]
            n_samples = (
                il.shape[0] if hasattr(il, "shape") else len(data["input_lengths"])
            )
        except Exception as e:
            il_summary = f"err:{e}"
            n_samples = -1
        _dp_log("pre_pack", n_samples=n_samples, input_lengths_first8=il_summary)

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
            packed0 = packed[0]
            mbi = getattr(packed0, "micro_batch_indices", None)
            mbl = getattr(packed0, "micro_batch_lengths", None)
            _dp_log(
                "post_seqpack",
                n_microbatches=(len(mbi[0]) if mbi else "None"),
                mbi_shape=(len(mbi) if mbi else "None"),
                mbl_first8=(mbl[0][:8] if mbl else "None"),
                spa_max_tokens=spa["max_tokens_per_microbatch"],
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
            sh0 = sharded[0]
            mbi = getattr(sh0, "micro_batch_indices", None)
            mbl = getattr(sh0, "micro_batch_lengths", None)
            _dp_log(
                "post_dynbatch",
                n_microbatches=(len(mbi[0]) if mbi else "None"),
                mbi_shape=(len(mbi) if mbi else "None"),
                mbl_first8=(mbl[0][:8] if mbl else "None"),
                dba_max_tokens=dba["max_tokens_per_microbatch"],
            )
            return sh0

        return data

    @wrap_with_nvtx_name("policy_worker/train_presharded")
    def train_presharded(
        self,
        meta: KVBatchMeta,
        loss_fn: Any,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Per-rank training entrypoint. Fetch → packing prep → delegate.

        When the driver pre-balanced packing across DP ranks (Option 1 fix
        for the seqpack/dynbatch step-4 NCCL hang), it ships per-shard
        ``micro_batch_indices``/``micro_batch_lengths`` in ``meta.extra_info``.
        Trust those instead of re-packing locally — local
        ``shard_by_batch_size(shards=1, ...)`` produces variable bin counts
        across DP groups and desyncs Megatron's per-microbatch collectives.
        """
        data = self._fetch(meta)
        extra = meta.extra_info or {}
        if (
            "micro_batch_indices" in extra
            and "micro_batch_lengths" in extra
        ):
            data.micro_batch_indices = extra["micro_batch_indices"]
            data.micro_batch_lengths = extra["micro_batch_lengths"]
            if "elem_counts_per_gb" in extra:
                data.elem_counts_per_gb = extra["elem_counts_per_gb"]
        else:
            data = self._apply_packing_prep(data)
        return self.train(  # type: ignore[attr-defined]
            data, loss_fn=loss_fn, eval_mode=eval_mode, gbs=gbs, mbs=mbs,
        )

    @wrap_with_nvtx_name("policy_worker/get_logprobs_presharded")
    def get_logprobs_presharded(
        self,
        meta: KVBatchMeta,
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[Any]:
        """Per-rank logprob entrypoint."""
        data = self._fetch(meta)
        return self.get_logprobs(  # type: ignore[attr-defined]
            data=data, micro_batch_size=micro_batch_size,
        )

    @wrap_with_nvtx_name("policy_worker/get_reference_policy_logprobs_presharded")
    def get_reference_policy_logprobs_presharded(
        self,
        meta: KVBatchMeta,
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Per-rank reference-policy logprob entrypoint."""
        data = self._fetch(meta)
        return self.get_reference_policy_logprobs(
            data=data, micro_batch_size=micro_batch_size,
        )
