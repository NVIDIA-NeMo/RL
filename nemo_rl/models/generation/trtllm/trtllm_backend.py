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

"""TRT-LLM WorkerExtension for NCCL / IPC weight synchronisation.

Injected into TRT-LLM's RayGPUWorker via ``ray_worker_extension_cls``.

- ``update_weights_from_collective`` — NCCL broadcast via
  ``packed_broadcast_consumer``, used in non-colocated mode.
- ``update_weights_via_ipc_zmq`` — CUDA IPC handles streamed over a
  per-GPU ZMQ socket, used in colocated mode (NCCL can't form a group
  when train and inference processes share the same physical GPU).
"""

import gc
import os
import traceback
from typing import Any

import torch
import zmq
from tensorrt_llm._ray_utils import control_action_decorator
from tensorrt_llm.llmapi.rlhf_utils import WorkerExtension

from nemo_rl.models.generation.trtllm.quantization import fp8 as fp8_quantization
from nemo_rl.models.policy.utils import (
    IPCProtocol,
    calculate_aligned_size,
    rebuild_cuda_tensor_from_ipc,
)
from nemo_rl.utils.packed_tensor import packed_broadcast_consumer

# Disable TRT-LLM weight loader's ThreadPoolExecutor: serial loading keeps
# all copies on the caller's stream (same as NCCL writes), so the existing
# stream-level sync in packed_broadcast_consumer covers them without us
# needing defensive cross-stream synchronize() calls. Also lower peak memory.
os.environ.setdefault("TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL", "True")


def _call_model_loader_hook_if_available(model_loader: Any, hook_name: str) -> bool:
    """Call a refit lifecycle hook when supported by the installed TRT-LLM."""
    hook = getattr(model_loader, hook_name, None)
    if hook is None:
        return False
    hook()
    return True


def _require_fp8_refit_hooks(model_loader: Any) -> None:
    """Require TRT-LLM hooks for transactional Qwen3.5 FP8 refits."""
    required_hooks = (
        "begin_update_weights",
        "finalize_update_weights",
        "abort_update_weights",
    )
    missing_hooks = [
        hook_name
        for hook_name in required_hooks
        if not callable(getattr(model_loader, hook_name, None))
    ]
    if not callable(getattr(WorkerExtension, "finalize_weight_update", None)):
        missing_hooks.append("WorkerExtension.finalize_weight_update")
    if missing_hooks:
        raise RuntimeError(
            "Qwen3.5 FP8 refit requires TRT-LLM weight-update hooks. "
            f"Missing APIs: {missing_hooks}."
        )


class NcclExtension(WorkerExtension):
    """NCCL-based weight update extension for TRT-LLM Ray workers.

    Attributes set by TRT-LLM's mixin injection (from ``RayGPUWorker``):
        self.engine    – ``PyExecutor`` instance
        self.device_id – int GPU ordinal
    """

    # ------------------------------------------------------------------ #
    #  Collective initialisation (called once during setup)
    # ------------------------------------------------------------------ #

    # Park the executor loop at a step boundary for the duration. Building the
    # refit group is a blocking ncclCommInitRank across train+inference ranks,
    # and the loop's own per-iteration object collectives are NCCL-backed since
    # tekit 97b62625. Running both on the same device deadlocks: the loop's
    # broadcast waits on peers whose main thread is inside ncclCommInitRank,
    # which in turn waits on every rank. Observed as PG5 stalling at work 85
    # after 84 clean iterations, then killed by the 600 s watchdog.
    @control_action_decorator
    def init_collective(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        assert torch.distributed.is_initialized(), (
            "TRT-LLM backend requires torch.distributed to be initialized before init_collective"
        )
        local_rank = torch.distributed.get_rank()
        rank = train_world_size + rank_prefix + local_rank

        pg = StatelessProcessGroup(
            master_address=ip,
            port=port,
            rank=rank,
            world_size=world_size,
        )
        pg.init_nccl_communicator(device=self.device_id)
        self.model_update_group = pg

    # ------------------------------------------------------------------ #
    #  GPU profiling (runs inside each nsys-wrapped GPU worker)
    # ------------------------------------------------------------------ #

    def start_gpu_profiling(self) -> None:
        """Start CUDA profiler on this GPU worker (nsys capture-range trigger)."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop CUDA profiler on this GPU worker."""
        torch.cuda.profiler.stop()

    # ------------------------------------------------------------------ #
    #  Refit metadata (weight name → (shape, dtype) mapping)
    # ------------------------------------------------------------------ #

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        self.state_dict_info = state_dict_info
        model = self.engine.model_engine.model
        self._local_expert_lookup = None
        if fp8_quantization.is_quantized_expert_refit(model.model_config.quant_config):
            fp8_quantization.validate_fused_expert_layout(state_dict_info)
            _require_fp8_refit_hooks(self.engine.model_engine.model_loader)
            # Every rank receives the full expert stacks but loads only its
            # own slots; convert just those (EP16: 32 of 512 experts).
            self._local_expert_lookup = fp8_quantization.build_local_expert_lookup(model)

    def _unwrap_compiled_model_for_refit(self) -> bool:
        """Unwrap torch.compile before weights are loaded.

        REQUIRED whenever torch.compile is enabled (which
        ``torch_compile_config.enable_piecewise_cuda_graph: true`` does
        implicitly). ``torch.compile`` wraps a submodule in an
        ``OptimizedModule`` whose child is ``_orig_mod``, so every parameter
        path under the compiled scope gains ``._orig_mod.`` -- e.g.
        ``llm.model._orig_mod.embed_tokens.weight``. TRT-LLM's
        ``load_weights`` matches checkpoint tensors by dotted path, and we load
        with ``allow_partial_loading=True``, so the compiled subtree is
        silently skipped and keeps its pre-refit weights. Nothing crashes: the
        refit reports success and training continues on stale weights, which
        shows up only as corrupted generations and a flat reward curve.

        Returns True when the hook exists (older TRT-LLM releases lack it).
        """
        model_engine = self.engine.model_engine
        # Renamed in TRT-LLM; the old name remains as an alias, so try both.
        unwrap = getattr(model_engine, "unwrap_compiled_model_for_refit", None) or getattr(
            model_engine, "release_piecewise_cuda_graphs_for_refit", None
        )
        if unwrap is None:
            return False
        unwrap()
        return True

    def _restore_compiled_model_after_refit(self) -> bool:
        """Re-wrap torch.compile after weights are loaded and finalized.

        Must run after all post-load processing, so the compiled callable is
        rebuilt over the finalized model. On current TRT-LLM this reuses the
        cached compiled artifact and leaves the piecewise captures intact
        (refit does not move any tensor), so it costs a few seconds against
        ~300 s of weight streaming.

        If this is skipped after a successful unwrap the engine still produces
        correct output -- it just runs eager, losing the torch.compile/PWCG
        speedup until the next refit.
        """
        model_engine = self.engine.model_engine
        restore = getattr(model_engine, "restore_compiled_model_after_refit", None) or getattr(
            model_engine, "recapture_piecewise_cuda_graphs_after_refit", None
        )
        if restore is None:
            return False
        restore(self.engine.resource_manager)
        return True

    def _finalize_weight_update(self) -> None:
        """Finalize refit using TRT-LLM's CUDA-graph-safe path when available."""
        # WorkerExtension gained this shared path after refit lifecycle hooks.
        # Retain the fallback while NeMo-RL supports older TRT-LLM releases.
        finalize_weight_update = getattr(
            WorkerExtension, "finalize_weight_update", None
        )
        if finalize_weight_update is not None:
            finalize_weight_update(self)
            return

        model_engine = self.engine.model_engine
        _call_model_loader_hook_if_available(
            model_engine.model_loader, "finalize_update_weights"
        )
        for module in model_engine.model.modules():
            if hasattr(module, "process_weights_after_loading") and not getattr(
                module, "_weights_removed", False
            ):
                module.process_weights_after_loading()
            if hasattr(module, "post_load_weights") and not getattr(
                module, "_weights_removed", False
            ):
                module.post_load_weights()

    def _ensure_refit_usable(self) -> None:
        failure = getattr(self, "_fp8_refit_failure", None)
        if failure is not None:
            raise RuntimeError(
                "This TRT-LLM worker is unusable after a failed partial FP8 "
                f"refit and must be restarted. Original failure: {failure}"
            )

    def _abort_weight_update_after_failure(
        self, model: Any, model_loader: Any, error: Exception
    ) -> None:
        fp8_refit_failed = fp8_quantization.is_quantized_expert_refit(
            model.model_config.quant_config
        )
        if fp8_refit_failed:
            # Record poisoning before abort: abort itself may fail, but this
            # worker must never serve with partially updated FP8 weights.
            self._fp8_refit_failure = repr(error)
        try:
            _call_model_loader_hook_if_available(model_loader, "abort_update_weights")
        finally:
            if fp8_refit_failed:
                raise RuntimeError(
                    "Partial Qwen3.5 FP8 refit failed after runtime weights may have "
                    "been modified. The TRT-LLM worker is poisoned and must be "
                    "restarted."
                ) from error

    # ------------------------------------------------------------------ #
    #  NCCL weight receive + reload
    # ------------------------------------------------------------------ #

    def update_weights_from_collective(
        self,
        *,
        drain: bool = True,
        recompute_kv: bool = False,
    ) -> bool:
        """Receive weights via NCCL broadcast and update model parameters.

        Args:
            drain: If True (default), wait for all in-flight requests to
                drain before applying weights — exclusive engine access.
                If False, the swap happens at a scheduler step boundary
                with in-flight requests still in the engine (in-flight
                weight update).
            recompute_kv: Only meaningful with ``drain=False``. If True,
                preempt in-flight requests so they re-prefill under the new weights.
                Otherwise, they keep decoding with their current KV cache. The
                reusable prefix cache is cleared after every weight update.
        """
        assert hasattr(self, "state_dict_info") and self.state_dict_info is not None, (
            "state_dict_info not set — call prepare_refit_info first"
        )
        model_engine = self.engine.model_engine
        model = model_engine.model
        self._ensure_refit_usable()

        load_model_weight_func = self._reload_bucket

        import time

        self._reset_refit_stats()
        phases: dict = {}
        t_start = time.perf_counter()
        with self.engine.control_action(drain=drain):
            try:
                # TRT-LLM uses the overlap scheduler by default: control_action
                # fires at a step boundary as soon as scheduling for the previous
                # iter is enqueued, but its GPU forward may still be in flight.
                # Block here so we don't overwrite weights mid-forward
                torch.cuda.synchronize()
                phases["drain"] = time.perf_counter() - t_start
                # Must precede any weight loading: while a torch.compile
                # wrapper is installed, parameter paths carry "_orig_mod" and
                # load_weights silently matches nothing.
                self._unwrap_compiled_model_for_refit()
                _call_model_loader_hook_if_available(
                    model_engine.model_loader, "begin_update_weights"
                )
                for module in model.modules():
                    if hasattr(module, "pre_reload_weights") and not getattr(
                        module, "_weights_removed", False
                    ):
                        module.pre_reload_weights()
                t = time.perf_counter()
                packed_broadcast_consumer(
                    iterator=iter(self.state_dict_info.items()),
                    group=self.model_update_group,
                    src=0,
                    post_unpack_func=load_model_weight_func,
                )
                phases["transfer_and_load"] = time.perf_counter() - t
                t = time.perf_counter()
                self._finalize_weight_update()
                torch.cuda.current_stream().synchronize()
                phases["finalize"] = time.perf_counter() - t

                t = time.perf_counter()
                self.engine.recompute_active_requests()
                phases["recompute"] = time.perf_counter() - t
                # After recompute_active_requests, not before: with the full
                # TRT-LLM lifecycle this replays warmup batches, and doing that
                # once the in-flight requests have released their KV keeps the
                # cache state clean.
                t = time.perf_counter()
                self._restore_compiled_model_after_refit()
                phases["restore"] = time.perf_counter() - t
                phases["total"] = time.perf_counter() - t_start
                self._log_refit_timing("collective", phases)
            except Exception as e:
                self._abort_weight_update_after_failure(
                    model, model_engine.model_loader, e
                )
                import traceback

                print(f"Error in NcclExtension.update_weights_from_collective: {e}")
                traceback.print_exc()
                return False

        return True

    # ------------------------------------------------------------------ #
    #  IPC weight receive + reload (colocated mode)
    # ------------------------------------------------------------------ #

    def get_zmq_address(self) -> str:
        # Trainer side binds the same path (per-GPU UUID) so workers sharing
        # the same physical GPU meet on one socket.
        return f"ipc:///tmp/{self.report_device_id()}.sock"

    def maybe_init_zmq(self) -> None:
        if hasattr(self, "zmq_socket"):
            return
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.setsockopt(zmq.SNDTIMEO, 120000)
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, 120000)
        self.zmq_socket.setsockopt(zmq.LINGER, 0)
        self.zmq_socket.connect(self.get_zmq_address())

    @control_action_decorator
    def update_weights_via_ipc_zmq(self) -> bool:
        """Receive weights via CUDA-IPC + ZMQ, reload model.

        Trainer sends ``(ipc_handle, list_keys, used_bytes)`` chunks; end of
        refit is signalled by ``IPCProtocol.COMPLETE``.
        """
        assert hasattr(self, "state_dict_info") and self.state_dict_info is not None, (
            "state_dict_info not set — call prepare_refit_info first"
        )
        model_engine = self.engine.model_engine
        model = model_engine.model
        self._ensure_refit_usable()

        buffer = None
        weights = None
        try:
            self.maybe_init_zmq()
            # See _unwrap_compiled_model_for_refit: must precede any loading.
            self._unwrap_compiled_model_for_refit()
            _call_model_loader_hook_if_available(
                model_engine.model_loader, "begin_update_weights"
            )
            for module in model.modules():
                if hasattr(module, "pre_reload_weights") and not getattr(
                    module, "_weights_removed", False
                ):
                    module.pre_reload_weights()

            while True:
                payload = self.zmq_socket.recv_pyobj()

                if payload == IPCProtocol.COMPLETE:
                    self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                    break

                ipc_handle, list_keys, used_bytes = payload
                buffer = rebuild_cuda_tensor_from_ipc(ipc_handle, self.device_id)

                weights = {}
                offset = 0
                for key in list_keys:
                    shape, dtype = self.state_dict_info[key]
                    if isinstance(shape, list):
                        shape = torch.Size(shape)
                    size_in_bytes = dtype.itemsize * shape.numel()
                    weights[key] = (
                        buffer[offset : offset + size_in_bytes]
                        .view(dtype=dtype)
                        .view(shape)
                    )
                    offset += calculate_aligned_size(size_in_bytes)

                assert offset == used_bytes, (
                    f"IPC payload offset mismatch: computed={offset}, sent={used_bytes}. "
                    "Likely stale state_dict_info (wrong shape/dtype for some key)."
                )

                if fp8_quantization.is_quantized_expert_refit(model.model_config.quant_config):
                    weights = fp8_quantization.load_weights(
                        weights.items(),
                        is_mx=fp8_quantization.is_mxfp8_model(
                            model.model_config.quant_config
                        ),
                        local_experts=getattr(self, "_local_expert_lookup", None),
                    )
                    # Qwen3.5's mapper may retain split QKVZ/BA tensors until a
                    # later IPC chunk completes the fusion group. Detach those
                    # views before ACK lets the trainer reuse its transport buffer.
                    weights = fp8_quantization.clone_mapper_staging_weights(weights)

                model_engine.model_loader.reload(
                    model,
                    weights,
                    allow_partial_loading=True,
                )
                torch.cuda.current_stream().synchronize()

                # Drop views before ACK — trainer reuses the buffer on the
                # next chunk, lingering views would read corrupted data.
                del weights, buffer
                weights = None
                buffer = None
                self.zmq_socket.send(IPCProtocol.ACK.value.encode())

            self._finalize_weight_update()
            torch.cuda.current_stream().synchronize()
            self.engine.reset_prefix_cache()
            gc.collect()
            torch.cuda.empty_cache()
            self._restore_compiled_model_after_refit()
            return True
        except Exception as e:
            self._abort_weight_update_after_failure(
                model, model_engine.model_loader, e
            )
            print(
                f"Error in NcclExtension.update_weights_via_ipc_zmq: {e}\n"
                f"{traceback.format_exc()}"
            )
            return False

    def cleanup_zmq(self) -> None:
        """Close ZMQ socket if open — called from worker shutdown."""
        if hasattr(self, "zmq_socket"):
            self.zmq_socket.close()
            del self.zmq_socket
        if hasattr(self, "zmq_context"):
            self.zmq_context.destroy()
            del self.zmq_context

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    def report_device_id(self) -> str:
        from tensorrt_llm._torch.utils import get_device_uuid

        return get_device_uuid(self.device_id)

    # ------------------------------------------------------------------ #
    #  Per-bucket reload shared by the collective and nccl_reshard paths
    # ------------------------------------------------------------------ #
    def _reload_bucket(self, weight_list) -> None:
        """Convert routed experts if the engine is FP8/MXFP8, then reload."""
        import time

        model_engine = self.engine.model_engine
        model = model_engine.model
        quant_config = model.model_config.quant_config
        stats = self._refit_stats
        t0 = time.perf_counter()
        if fp8_quantization.is_quantized_expert_refit(quant_config):
            weights = fp8_quantization.load_weights(
                weight_list,
                is_mx=fp8_quantization.is_mxfp8_model(quant_config),
                local_experts=getattr(self, "_local_expert_lookup", None),
            )
        else:
            weights = dict(weight_list)
        torch.cuda.current_stream().synchronize()
        t1 = time.perf_counter()
        model_engine.model_loader.reload(
            model,
            weights,
            allow_partial_loading=True,
        )
        torch.cuda.current_stream().synchronize()
        stats["convert_s"] += t1 - t0
        stats["reload_s"] += time.perf_counter() - t1
        stats["buckets"] += 1
        stats["tensors"] += len(weights)

    @property
    def _refit_stats(self) -> dict:
        stats = getattr(self, "_refit_stats_dict", None)
        if stats is None:
            stats = self._reset_refit_stats()
        return stats

    def _reset_refit_stats(self) -> dict:
        self._refit_stats_dict = {
            "convert_s": 0.0,
            "reload_s": 0.0,
            "buckets": 0,
            "tensors": 0,
        }
        return self._refit_stats_dict

    def _log_refit_timing(self, path: str, phases: dict) -> None:
        """One rank-0 line per refit: where this engine's wall time went."""
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        stats = self._refit_stats
        parts = " ".join(f"{k}={v:.2f}s" for k, v in phases.items())
        print(
            f"[refit-timing] path={path} {parts} "
            f"convert={stats['convert_s']:.2f}s reload={stats['reload_s']:.2f}s "
            f"buckets={stats['buckets']} tensors={stats['tensors']}",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    #  nccl_reshard (shard-to-shard) refit
    # ------------------------------------------------------------------ #
    # The trainer reshards the routed experts straight into each engine's
    # EP-local slots over one communicator per PP stage (that stage's train
    # ranks + this engine layout's ranks); everything else rides the misc
    # packed broadcast on model_update_group, loaded by _reload_bucket.
    @control_action_decorator
    def init_nccl_reshard_comm_group(
        self,
        rank_prefix: int,
        pp_ips: list[str],
        pp_ports: list[int],
        pp_size: int,
        train_ranks_per_stage: int,
        sub_world_size: int,
    ) -> None:
        """Join the bulk-path communicator of every PP stage.

        Parked at a step boundary for the same reason as ``init_collective``:
        ncclCommInitRank across train + inference ranks deadlocks against the
        executor loop's own NCCL object collectives.
        """
        from nemo_rl.distributed.refit_watchdog import RELEASE_GRACE_S, release_within
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        assert torch.distributed.is_initialized(), (
            "TRT-LLM backend requires torch.distributed to be initialized before "
            "init_nccl_reshard_comm_group"
        )
        gen_rank_in_group = (
            train_ranks_per_stage + rank_prefix + torch.distributed.get_rank()
        )
        stale_groups = list((getattr(self, "pp_comm_groups", None) or {}).values())
        self.pp_comm_groups = {}
        torch.cuda.empty_cache()
        for stage in range(pp_size):
            print(
                f"  refit: reshard rendezvous [gen] stage={stage} "
                f"addr={pp_ips[stage]}:{pp_ports[stage]} "
                f"rank={gen_rank_in_group} world_size={sub_world_size}",
                flush=True,
            )
            group = StatelessProcessGroup(
                master_address=pp_ips[stage],
                port=pp_ports[stage],
                rank=gen_rank_in_group,
                world_size=sub_world_size,
            )
            group.init_nccl_communicator(device=self.device_id)
            self.pp_comm_groups[stage] = group
        for previous in stale_groups:
            release_within(
                previous.abort, RELEASE_GRACE_S, "a previous reshard bulk communicator"
            )
        refit_info = getattr(self, "nccl_reshard_refit_info", None)
        if refit_info is not None:
            self.hf_to_local_param_map = self._build_expert_local_param_map(refit_info)

    def prepare_nccl_reshard_refit_info(self, refit_info: dict) -> None:
        """Keep the per-layer transfer plan and map its bulk params to local slots."""
        from nemo_rl.weight_sync.nccl_reshard_utils import (
            _STR_TO_DTYPE,
            restore_refit_info_placements,
        )

        self.nccl_reshard_refit_info = restore_refit_info_placements(refit_info)
        model = self.engine.model_engine.model
        quant_config = model.model_config.quant_config
        # The local-expert lookup doubles as the slot oracle for the bulk
        # specs, so build it for bf16 engines too.
        self._local_expert_lookup = fp8_quantization.build_local_expert_lookup(model)
        if self._local_expert_lookup is None:
            raise RuntimeError(
                "nccl_reshard refit needs the MoE modules' initial_local_expert_ids "
                "(none found, or an expert load balancer is active)"
            )
        if fp8_quantization.is_quantized_expert_refit(quant_config):
            _require_fp8_refit_hooks(self.engine.model_engine.model_loader)
        misc_meta = self.nccl_reshard_refit_info.get("misc_meta", {}) or {}
        misc_state_dict_info = {}
        for name, meta in misc_meta.items():
            if isinstance(meta, dict):
                shape, dtype = meta["shape"], meta["dtype"]
            else:
                shape, dtype = meta[0], meta[1]
            dtype = _STR_TO_DTYPE[str(dtype)] if not isinstance(dtype, torch.dtype) else dtype
            misc_state_dict_info[name] = (torch.Size(shape), dtype)
        self.misc_state_dict_info = misc_state_dict_info
        # The misc consumer reuses the collective path's per-bucket reload.
        self.state_dict_info = misc_state_dict_info
        if getattr(self, "pp_comm_groups", None):
            self.hf_to_local_param_map = self._build_expert_local_param_map(
                self.nccl_reshard_refit_info
            )

    @staticmethod
    def _local_shard_slices(param_info: dict, rank: int) -> tuple:
        """This rank's slices of the HF-global tensor under the dst placements."""
        from nemo_rl.weight_sync.xferdtensor_python import _compute_shard_slices

        dst_mesh = param_info["dst_mesh_info"]
        mesh_tensor = getattr(dst_mesh, "mesh", None)
        if mesh_tensor is None:
            mesh_tensor = getattr(dst_mesh, "_mesh", None)
        if mesh_tensor is None:
            raise ValueError("Destination mesh does not expose its ranks")
        coordinates = (mesh_tensor == rank).nonzero(as_tuple=False)
        if coordinates.numel() == 0:
            raise ValueError(f"Rank {rank} is absent from the destination mesh")
        return tuple(
            _compute_shard_slices(
                param_info["global_shape"],
                list(mesh_tensor.shape),
                coordinates[0].tolist(),
                param_info["dst_placements"],
            )
        )

    def _build_expert_local_param_map(self, refit_info: dict):
        """One LocalParamSpec per grouped routed-expert projection.

        ``pre`` allocates a canonical ``[E_local, out, in]`` bf16 staging
        buffer for this rank's EP slice, ``post`` turns it into per-expert HF
        entries (FP8/MXFP8 for quantized engines) and hands them to TRT-LLM's
        reload. The EP slice must be the slots TRT-LLM loads
        (``initial_local_expert_ids``); a mismatch is an error, never a
        silent partial update.
        """
        from torch.distributed._tensor import Shard

        from nemo_rl.weight_sync.nccl_reshard_utils import (
            _STR_TO_DTYPE,
            HFToLocalParamMap,
            LocalParamSpec,
            RefitCtx,
        )

        pp_comm_groups = getattr(self, "pp_comm_groups", None)
        if not pp_comm_groups:
            raise RuntimeError(
                "nccl_reshard refit mapping needs the per-PP-stage communicators"
            )
        lookup = getattr(self, "_local_expert_lookup", None)
        assert lookup is not None, "prepare_nccl_reshard_refit_info must run first"
        device = torch.device("cuda", self.device_id)
        specs = {}
        for layer_name in refit_info["layer_names"]:
            for param_info in refit_info["per_layer_params"][layer_name]:
                name = param_info["name"]
                projection = param_info.get("grouped_expert_proj")
                if projection is None:
                    raise NotImplementedError(
                        "TRT-LLM nccl_reshard refit moves grouped routed experts only; "
                        f"bulk param {name!r} would need this engine's TP layout"
                    )
                bad_dims = [
                    placement.dim
                    for placement in param_info["dst_placements"]
                    if isinstance(placement, Shard) and placement.dim != 0
                ]
                if bad_dims:
                    raise ValueError(
                        f"{name!r}: TRT-LLM experts shard on EP only (dst Shard dims "
                        f"{bad_dims} unsupported)"
                    )
                stage = param_info.get("pp_stage", 0)
                if stage not in pp_comm_groups:
                    raise RuntimeError(f"no reshard communicator for PP stage {stage}")
                rank = pp_comm_groups[stage].rank
                slices = self._local_shard_slices(param_info, rank)
                global_shape = tuple(param_info["global_shape"])
                local_shape = []
                for size, sl in zip(global_shape, slices):
                    start = 0 if sl.start is None else sl.start
                    stop = size if sl.stop is None else sl.stop
                    local_shape.append(stop - start)
                local_shape = tuple(local_shape)
                expert_start = 0 if slices[0].start is None else slices[0].start
                expert_ids = list(range(expert_start, expert_start + local_shape[0]))
                prefix = name.rsplit(f".{projection}.weight", 1)[0]
                expected = lookup(prefix)
                if expected is None:
                    raise RuntimeError(
                        f"{name!r}: no MoE module for expert prefix {prefix!r} on this rank"
                    )
                if list(expected) != expert_ids:
                    raise RuntimeError(
                        f"{name!r}: reshard EP slice {expert_ids[0]}..{expert_ids[-1]} "
                        f"differs from TRT-LLM's local slots "
                        f"{list(expected)[0]}..{list(expected)[-1]}"
                    )
                dtype_value = param_info.get("dtype")
                dtype = (
                    dtype_value
                    if isinstance(dtype_value, torch.dtype)
                    else _STR_TO_DTYPE.get(str(dtype_value))
                )
                if dtype is None:
                    raise ValueError(f"{name!r}: unsupported wire dtype {dtype_value!r}")

                def pre(_base, shape=local_shape, dtype=dtype):
                    return RefitCtx(buf=torch.empty(shape, dtype=dtype, device=device))

                def post(
                    ctx, prefix=prefix, projection=projection, expert_ids=expert_ids
                ):
                    self._load_received_experts(prefix, projection, expert_ids, ctx.buf)

                specs[name] = LocalParamSpec(base=None, pre=pre, post=post)
        return HFToLocalParamMap(specs=specs)

    def _load_received_experts(
        self, prefix: str, projection: str, expert_ids: list[int], stack: torch.Tensor
    ) -> None:
        """Load one received ``[E_local, out, in]`` projection stack."""
        model_engine = self.engine.model_engine
        model = model_engine.model
        quant_config = model.model_config.quant_config
        if fp8_quantization.is_quantized_expert_refit(quant_config):
            weights: dict = {}
            fp8_quantization.convert_expert_projection_stack(
                weights,
                prefix=prefix,
                projection=projection,
                tensor=stack,
                expert_ids=expert_ids,
                is_mx=fp8_quantization.is_mxfp8_model(quant_config),
            )
        else:
            weights = {
                f"{prefix}.{expert_id}.{projection}.weight": expert
                for expert_id, expert in zip(expert_ids, stack.unbind(0))
            }
        model_engine.model_loader.reload(model, weights, allow_partial_loading=True)

    def _recv_bulk_params(self) -> None:
        """Receive every bulk param of every PP stage into its local slots."""
        from collections import OrderedDict

        from nemo_rl.weight_sync.nccl_reshard_utils import RefitCtx
        from nemo_rl.weight_sync.xferdtensor import DTensorRef, xferdtensor

        refit_info = self.nccl_reshard_refit_info
        pp_comm_groups = self.pp_comm_groups
        param_map = self.hf_to_local_param_map
        stage_params = OrderedDict()
        for layer_name in refit_info["layer_names"]:
            for param_info in refit_info["per_layer_params"][layer_name]:
                stage_params.setdefault(param_info.get("pp_stage", 0), []).append(
                    param_info
                )
        num_streams = max(
            1, min(int(os.environ.get("NRL_REFIT_NUM_STREAMS", "2")), len(stage_params))
        )
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        events = {}
        for idx, (stage, params) in enumerate(stage_params.items()):
            if (idx - num_streams) in events:
                events[idx - num_streams].synchronize()
            stream = streams[idx % num_streams]
            with torch.cuda.stream(stream):
                group = pp_comm_groups[stage]
                for param_info in params:
                    spec = param_map.get(param_info["name"])
                    assert spec is not None, (
                        f"nccl_reshard_refit: {param_info['name']!r} has no local spec "
                        "(its weights would be discarded)"
                    )
                    ctx = spec.pre(spec.base) if spec.pre is not None else RefitCtx(buf=spec.base)
                    xferdtensor(
                        None,
                        param_info["src_mesh_info"],
                        param_info["src_placements"],
                        DTensorRef(ctx.buf, param_info["global_shape"]),
                        param_info["dst_mesh_info"],
                        param_info["dst_placements"],
                        group,
                        stream,
                    )
                    if spec.post is not None:
                        spec.post(ctx)
                    del ctx
                event = torch.cuda.Event()
                event.record()
                events[idx] = event
        torch.cuda.synchronize()

    def _receive_and_load_misc_params(self) -> None:
        """Misc params: the collective path's packed broadcast + per-bucket reload."""
        misc_state_dict_info = getattr(self, "misc_state_dict_info", None) or {}
        if not misc_state_dict_info:
            return
        packed_broadcast_consumer(
            iterator=iter(misc_state_dict_info.items()),
            group=self.model_update_group,
            src=0,
            post_unpack_func=self._reload_bucket,
        )

    def nccl_reshard_refit(
        self, refit_timeout_s=None, *, drain: bool = True, recompute_kv: bool = False
    ) -> bool:
        """Receive a refit shard-to-shard, then the misc broadcast, then finalize.

        Same engine bracket as ``update_weights_from_collective``: ``drain``
        picks exclusive access at a step boundary (True) or the in-flight
        weight update (False, ``recompute_kv`` as there); torch.compile is
        unwrapped around the loads, deferred weight processing and the
        KV-cache reset run in ``_finalize_weight_update`` /
        ``recompute_active_requests``.
        """
        refit_info = getattr(self, "nccl_reshard_refit_info", None)
        if refit_info is None:
            raise RuntimeError("prepare_nccl_reshard_refit_info must run before the refit")
        if not getattr(self, "pp_comm_groups", None):
            raise RuntimeError("init_nccl_reshard_comm_group must run before the refit")
        if getattr(self, "hf_to_local_param_map", None) is None:
            self.hf_to_local_param_map = self._build_expert_local_param_map(refit_info)
        model_engine = self.engine.model_engine
        model = model_engine.model
        self._ensure_refit_usable()
        import time

        self._reset_refit_stats()
        phases: dict = {}
        t_start = time.perf_counter()
        with self.engine.control_action(drain=drain):
            try:
                torch.cuda.synchronize()
                phases["drain"] = time.perf_counter() - t_start
                self._unwrap_compiled_model_for_refit()
                _call_model_loader_hook_if_available(
                    model_engine.model_loader, "begin_update_weights"
                )
                for module in model.modules():
                    if hasattr(module, "pre_reload_weights") and not getattr(
                        module, "_weights_removed", False
                    ):
                        module.pre_reload_weights()
                t = time.perf_counter()
                self._recv_bulk_params()
                torch.cuda.empty_cache()
                phases["bulk"] = time.perf_counter() - t
                t = time.perf_counter()
                self._receive_and_load_misc_params()
                phases["misc"] = time.perf_counter() - t
                t = time.perf_counter()
                self._finalize_weight_update()
                torch.cuda.current_stream().synchronize()
                phases["finalize"] = time.perf_counter() - t
                t = time.perf_counter()
                self.engine.recompute_active_requests()
                phases["recompute"] = time.perf_counter() - t
                t = time.perf_counter()
                self._restore_compiled_model_after_refit()
                phases["restore"] = time.perf_counter() - t
                phases["total"] = time.perf_counter() - t_start
                self._log_refit_timing("nccl_reshard", phases)
            except Exception as e:
                self._abort_weight_update_after_failure(
                    model, model_engine.model_loader, e
                )
                print(f"Error in NcclExtension.nccl_reshard_refit: {e}")
                traceback.print_exc()
                return False
        return True
