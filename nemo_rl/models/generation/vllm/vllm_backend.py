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
import gc
from typing import Any

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

import zmq
from nemo_rl.utils.nsys import wrap_with_nvtx_name

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "This error can also happen if the venv creation was aborted or errored out in the middle. In that case, "
        "please run at least once with the environment variable NRL_FORCE_REBUILD_VENVS=true set to force the rebuild of the environment."
    )


class VllmInternalWorkerExtension:
    def init_collective(
        self, rank_prefix: int, ip: str, port: int, world_size: int
    ) -> None:
        """Initialize the collective communication."""
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        local_rank = torch.distributed.get_rank()
        rank = rank_prefix + local_rank + 1  # 1 is the head node of the train cluster

        pg = StatelessProcessGroup.create(
            host=ip, port=port, rank=rank, world_size=world_size
        )
        self.model_update_group = PyNcclCommunicator(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            pg, device=self.device
        )

    def report_device_id(self) -> str:
        from nemo_rl.utils.nvml import get_device_uuid

        return get_device_uuid(self.device.index)

    def get_zmq_address(self):
        """Get the ZMQ address for the current device."""
        return f"ipc:///{self.report_device_id()}.sock"

    def maybe_init_zmq(self):
        """Initialize the ZMQ socket if it doesn't exist."""
        if not hasattr(self, "zmq_socket"):
            self.zmq_context = zmq.Context()  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            self.zmq_socket = self.zmq_context.socket(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
                zmq.REP
            )  
            # Set receive timeout to 30 seconds to avoid hanging indefinitely
            self.zmq_socket.setsockopt(
                zmq.RCVTIMEO, 30000
            )  # 30 seconds in milliseconds
            self.zmq_socket.connect(self.get_zmq_address())

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Prepare the info for refit.

        Args:
            state_dict_info (dict): A dictionary containing the info for refit.
                e.g. {tensor_name: (shape, dtype)}
        """
        self.state_dict_info = state_dict_info  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored

    @wrap_with_nvtx_name("vllm_internal_worker_extension/update_weights_via_ipc_zmq")
    def update_weights_via_ipc_zmq(self) -> bool:
        """Update weights from local IPC handles via ZMQ socket.

        Args:
            None

        Returns:
            bool: True if weights were successfully updated.
        """
        buffer = None
        weights = None

        try:
            self.maybe_init_zmq()
            from nemo_rl.models.policy.utils import calculate_aligned_size

            while True:
                # Blocking receive with timeout (this is the main operation)
                payload = self.zmq_socket.recv_pyobj()

                if payload == "complete":
                    # means the update is done
                    self.zmq_socket.send(b"")
                    break

                packed_tensor_handle, list_keys, used_bytes = payload
                device_id = self.device.index
                func = rebuild_cuda_tensor
                args = packed_tensor_handle[0]
                list_args = list(args)
                list_args[6] = device_id
                buffer = func(*list_args)

                weights = []
                offset = 0
                for key in list_keys:
                    shape, dtype = self.state_dict_info[key]  # pyrefly
                    if isinstance(shape, list):
                        shape = torch.Size(shape)
                    size_in_bytes = dtype.itemsize * shape.numel()
                    weights.append(
                        (
                            key,
                            buffer[offset : offset + size_in_bytes]
                            .view(dtype=dtype)
                            .view(shape),
                        )
                    )
                    aligned_size = calculate_aligned_size(size_in_bytes)
                    offset += aligned_size
                assert offset == used_bytes, (
                    "Offset is not equal to used bytes, usually indicate key info inaccurate like dtype"
                )
                # Load weights into the model
                from nemo_rl.models.generation import fp8

                if fp8.is_fp8_model(self.model_runner.vllm_config):
                    # the fp8 load_weights additionally casts bf16 weights into fp8
                    fp8.load_weights(weights, self.model_runner)
                else:
                    self.model_runner.model.load_weights(weights=weights)

                torch.cuda.current_stream().synchronize()
                self.zmq_socket.send(b"")

            if buffer is not None:
                del buffer
            if weights is not None:
                del weights
            gc.collect()
            torch.cuda.empty_cache()
            return True

        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_ipc_handles: {e}"
            )
            return False

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_collective"
    )
    def update_weights_from_collective(self) -> bool:
        """Update the model weights from collective communication."""
        assert self.state_dict_info is not None, (
            "state_dict_info is not prepared. "
            "Please call prepare_refit_info when initializing the worker."
        )

        try:
            for name, (shape, dtype) in self.state_dict_info.items():
                weight = torch.empty(shape, dtype=dtype, device="cuda")
                self.model_update_group.broadcast(weight, src=0)

                from nemo_rl.models.generation import fp8

                if fp8.is_fp8_model(self.model_runner.vllm_config):
                    # the fp8 load_weights additionally casts bf16 weights into fp8
                    fp8.load_weights([(name, weight)], self.model_runner)
                else:
                    self.model_runner.model.load_weights(weights=[(name, weight)])
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_collective: {e}"
            )
            return False

        return True

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()
