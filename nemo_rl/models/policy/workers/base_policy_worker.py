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
import warnings
from collections.abc import Generator
from typing import Any, Optional

import ray
import torch
import zmq

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.interfaces import ReferenceLogprobOutputSpec
from nemo_rl.utils.nsys import wrap_with_nvtx_name


def maybe_preinit_nixl_checkpoint_engine(config: dict[str, Any]) -> Any:
    """Preinitialize NIXL when checkpoint-engine refit is configured."""
    generation_cfg = config.get("generation")
    if not generation_cfg:
        return None
    checkpoint_cfg = generation_cfg.get("checkpoint_engine")
    if not (
        checkpoint_cfg
        and checkpoint_cfg["enabled"]
        and checkpoint_cfg["backend"] == "nixl"
    ):
        return None

    from nemo_rl.utils.checkpoint_engines.nixl import (
        preinit_nixl_agent,
        resolve_nixl_backend_kwargs,
    )

    backend_name, backend_init_params = resolve_nixl_backend_kwargs(
        checkpoint_cfg["engine_kwargs"]["nixl"]
    )
    return preinit_nixl_agent(
        backend_name=backend_name, backend_init_params=backend_init_params
    )


class DTensorCheckpointEngineSendMixin:
    """Shared checkpoint-engine send hooks for DTensor/FSDP2 policy workers.

    With cpu_offload enabled the model sits in host memory between steps, so it
    must be onloaded to CUDA for the transfer and offloaded again afterwards.
    """

    def _prepare_checkpoint_engine_weight_send(self) -> None:
        if self.cpu_offload:
            warnings.warn(
                "cpu_offload adds an onload/offload cycle during non-colocated "
                "checkpoint-engine refit. Disable it unless GPU memory requires it.",
                stacklevel=2,
            )
            self.model = self.move_to_cuda(self.model)

    def _finalize_checkpoint_engine_weight_send(self) -> None:
        if self.cpu_offload:
            self.model = self.move_to_cpu(self.model)


class AbstractPolicyWorker:
    """Base class for policy workers with shared functionality."""

    def _checkpoint_engine_weight_iterator(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Yield policy weights for checkpoint-engine refit."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support checkpoint-engine refit."
        )

    def _prepare_checkpoint_engine_weight_send(self) -> None:
        """Prepare worker state before checkpoint-engine weight transfer."""
        pass

    def _finalize_checkpoint_engine_weight_send(self) -> None:
        """Restore worker state after checkpoint-engine weight transfer."""
        pass

    async def send_weights_via_checkpoint_engine(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        self._prepare_checkpoint_engine_weight_send()
        try:
            with torch.no_grad():
                await self.checkpoint_engine.send_weights(
                    self._checkpoint_engine_weight_iterator(kv_scales=kv_scales)
                )
        finally:
            self._finalize_checkpoint_engine_weight_send()

    async def checkpoint_engine_rpc(
        self, checkpoint_method: str, method_kwargs: Optional[dict[str, Any]] = None
    ) -> Any:
        kwargs = method_kwargs or {}
        if checkpoint_method == "init_checkpoint_engine":
            if getattr(self, "checkpoint_engine", None) is None:
                from nemo_rl.utils.checkpoint_engines.base import (
                    create_checkpoint_engine,
                )

                self.checkpoint_engine = create_checkpoint_engine(
                    kwargs["backend"],
                    bucket_size_bytes=kwargs["bucket_size_bytes"],
                    engine_kwargs=kwargs["engine_kwargs"],
                )
            return
        if checkpoint_method == "prepare_checkpoint_engine":
            metadata = self.checkpoint_engine.prepare()
            if isinstance(metadata, dict):
                return {**metadata, "rank": self.rank}
            return metadata
        if checkpoint_method == "init_checkpoint_engine_process_group":
            return self.checkpoint_engine.init_policy_process_group(
                worker_rank=self.rank, **kwargs
            )
        if checkpoint_method == "send_weights_via_checkpoint_engine":
            return await self.send_weights_via_checkpoint_engine(**kwargs)
        if checkpoint_method == "finalize_checkpoint_engine":
            checkpoint_engine = getattr(self, "checkpoint_engine", None)
            if checkpoint_engine is not None:
                checkpoint_engine.finalize()
            return
        return getattr(self.checkpoint_engine, checkpoint_method)(**kwargs)

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
        # Workers that manage their own LOCAL_RANK will have an empty `ray.get_gpu_ids()`.
        gpu_ids = ray.get_gpu_ids()
        gpu_id = gpu_ids[0] if gpu_ids else torch.cuda.current_device()
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
