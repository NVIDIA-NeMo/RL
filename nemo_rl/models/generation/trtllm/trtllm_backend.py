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
    #  Refit metadata (weight name → (shape, dtype) mapping)
    # ------------------------------------------------------------------ #

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        self.state_dict_info = state_dict_info

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

        def load_model_weight_func(weight_list):
            model_engine.model_loader.reload(
                model,
                dict(weight_list),
                allow_partial_loading=True,
            )

        with self.engine.control_action(drain=drain):
            try:
                # TRT-LLM uses the overlap scheduler by default: control_action
                # fires at a step boundary as soon as scheduling for the previous
                # iter is enqueued, but its GPU forward may still be in flight.
                # Block here so we don't overwrite weights mid-forward
                torch.cuda.synchronize()
                model_engine.model_loader.begin_update_weights()
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
                model_engine.model_loader.finalize_update_weights()
                for module in model.modules():
                    if hasattr(module, "process_weights_after_loading") and not getattr(
                        module, "_weights_removed", False
                    ):
                        module.process_weights_after_loading()
                    if hasattr(module, "post_load_weights") and not getattr(
                        module, "_weights_removed", False
                    ):
                        module.post_load_weights()
                torch.cuda.current_stream().synchronize()

                self.engine.reset_prefix_cache()
            except Exception as e:
                model_engine.model_loader.abort_update_weights()
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

        buffer = None
        weights = None
        try:
            self.maybe_init_zmq()
            model_engine.model_loader.begin_update_weights()
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

            model_engine.model_loader.finalize_update_weights()
            for module in model.modules():
                if hasattr(module, "process_weights_after_loading") and not getattr(
                    module, "_weights_removed", False
                ):
                    module.process_weights_after_loading()
                if hasattr(module, "post_load_weights") and not getattr(
                    module, "_weights_removed", False
                ):
                    module.post_load_weights()
            torch.cuda.current_stream().synchronize()
            self.engine.reset_prefix_cache()
            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            model_engine.model_loader.abort_update_weights()
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
