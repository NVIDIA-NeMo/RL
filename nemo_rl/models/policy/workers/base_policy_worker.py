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
from typing import Any, Optional

import ray
import torch
import zmq

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
            train_world_size: Number of training workers participating in the
                cross-cluster process group.

        NOTE: under the RefitWorker architecture (RL-412 follow-up) the
        cross-cluster group is owned by a sibling ``RefitWorker`` actor
        on the same node as train rank 0 — train workers do NOT
        participate. ``Policy.init_collective`` skips dispatching to
        train workers in that mode, so this method's body only runs on
        the legacy DTensor path (or with ``NRL_USE_REFIT_WORKER=0``).
        """
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        # Idempotent re-init: tear down any prior group (e.g. world size
        # shrunk after a fault). Order matters:
        #   1) ``destroy()`` first — NCCL's abort marks pending kernels
        #      cancelled without waiting; sync'ing first would block on
        #      ops that NCCL has noticed cannot make progress (their
        #      peer is gone) and surface ``cudaErrorLaunchFailure`` into
        #      the CUDA context.
        #   2) ``cuda.synchronize()`` second, in try/except, to drain and
        #      swallow the queued async error.
        #   3) ``cuda.empty_cache()`` third, also try/except'd.
        old = getattr(self, "model_update_group", None)
        if old is not None:
            try:
                old.destroy()
            except Exception as e:  # noqa: BLE001
                print(
                    f"[policy_worker] warning: old model_update_group destroy raised {e}",
                    flush=True,
                )
            try:
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                print(
                    f"[policy_worker] post-destroy synchronize swallowed: {type(e).__name__}: {e}",
                    flush=True,
                )
            try:
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                print(
                    f"[policy_worker] post-destroy empty_cache swallowed: {type(e).__name__}: {e}",
                    flush=True,
                )

        self.model_update_group = StatelessProcessGroup(
            master_address=ip, port=port, rank=self.rank, world_size=world_size
        )
        device = torch.cuda.current_device()
        self.model_update_group.init_nccl_communicator(device=device)

    def abort_collective(self) -> None:
        """Abort the cross-cluster weight-sync NCCL group on this worker.

        Legacy path only — under the RefitWorker architecture the
        cross-cluster group lives in a sibling actor and is torn down
        via ``ray.kill`` from ``Policy.abort_collective``.
        """
        old = getattr(self, "model_update_group", None)
        if old is None:
            return
        cuda_poisoned: Optional[BaseException] = None
        try:
            old.destroy()
        except Exception as e:  # noqa: BLE001
            cuda_poisoned = e
            rank = getattr(self, "rank", "?")
            print(
                f"[policy_worker.abort_collective] rank={rank} group destroy "
                f"surfaced {type(e).__name__}: {e}",
                flush=True,
            )
        self.model_update_group = None  # type: ignore[assignment]
        if cuda_poisoned is not None:
            raise cuda_poisoned

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

    def get_node_id(self) -> str:
        """Return the Ray node id this worker is running on.

        Used by ``Policy._ensure_refit_worker`` to colocate the
        cross-cluster ``RefitWorker`` actor on the same node as
        train rank 0 — chunks then transit via Ray-internal shared
        memory (zero-copy) instead of over the network.
        """
        return ray.get_runtime_context().get_node_id()

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
