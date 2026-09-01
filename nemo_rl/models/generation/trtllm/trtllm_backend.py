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
        if fp8_quantization.is_quantized_expert_refit(model.model_config.quant_config):
            fp8_quantization.validate_fused_expert_layout(state_dict_info)
            _require_fp8_refit_hooks(self.engine.model_engine.model_loader)

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

        def load_model_weight_func(weight_list):
            if fp8_quantization.is_quantized_expert_refit(model.model_config.quant_config):
                weights = fp8_quantization.load_weights(
                    weight_list,
                    is_mx=fp8_quantization.is_mxfp8_model(
                        model.model_config.quant_config
                    ),
                )
            else:
                weights = dict(weight_list)
            model_engine.model_loader.reload(
                model,
                weights,
                allow_partial_loading=True,
            )

        with self.engine.control_action(drain=drain):
            try:
                # TRT-LLM uses the overlap scheduler by default: control_action
                # fires at a step boundary as soon as scheduling for the previous
                # iter is enqueued, but its GPU forward may still be in flight.
                # Block here so we don't overwrite weights mid-forward
                torch.cuda.synchronize()
                _call_model_loader_hook_if_available(
                    model_engine.model_loader, "begin_update_weights"
                )
                for module in model.modules():
                    if hasattr(module, "pre_reload_weights") and not getattr(
                        module, "_weights_removed", False
                    ):
                        module.pre_reload_weights()
                packed_broadcast_consumer(
                    iterator=iter(self.state_dict_info.items()),
                    group=self.model_update_group,
                    src=0,
                    post_unpack_func=load_model_weight_func,
                )
                self._finalize_weight_update()
                torch.cuda.current_stream().synchronize()

                self.engine.recompute_active_requests()
            except Exception as e:
                self._abort_weight_update_after_failure(
                    model, model_engine.model_loader, e
                )
                print(f"Error in NcclExtension.update_weights_from_collective: {e}")
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
