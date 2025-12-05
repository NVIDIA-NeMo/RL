import ray
import torch
import zmq
from typing import Any, Callable, Optional

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.interfaces import (
    LogprobOutputSpec,
    ReferenceLogprobOutputSpec,
)


class BasePolicyWorker:
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
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        pg = StatelessProcessGroup.create(
            host=ip, port=port, rank=self.rank, world_size=world_size
        )
        device = torch.cuda.current_device()
        self.model_update_group = PyNcclCommunicator(pg, device=device)

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

    def shutdown(self) -> None:
        """Shutdown the policy."""
        # Clean up extension resources like ZMQ sockets
        if hasattr(self, "zmq_socket"):
            self.zmq_socket.close()
            self.zmq_context.term()

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()

    def report_node_ip_and_gpu_id(self) -> list[tuple[str, int]]:
        """Report the node IP and GPU ID of the current worker."""
        ip = ray._private.services.get_node_ip_address()
        gpu_id = ray.get_gpu_ids()[0]
        return (ip, gpu_id)

    def get_reference_policy_logprobs(
        self, *, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Get the logprobs from thereference policy for a batch of data.

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
    
    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    
    def get_logprobs(
        self, *, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a batch of data.

        If micro_batch_size is provided, it will be used instead of the configured
        logprob_batch_size.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def use_reference_model(self):
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references
        On exit: Restores original references and re-flips cuda/cpu
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_topk_logits(
        self,
        *,
        data: BatchedDataDict[Any],
        k: int,
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[Any]:
        """Get the top-k logits for a batch of data.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def prepare_refit_info(self) -> None:
        """Prepare the refit info for the model."""
        raise NotImplementedError("Subclasses must implement this method.")

    def stream_weights_via_ipc_zmq(self, buffer_size_bytes: int = 0) -> None:
        """Stream the weights of the model via ZMQ IPC."""
        raise NotImplementedError("Subclasses must implement this method.")

    def broadcast_weights_for_collective(self) -> None:
        """Broadcast the weights of the model for collective communication."""
        raise NotImplementedError("Subclasses must implement this method.")

    def prepare_for_lp_inference(self) -> None:
        """Prepare the model for LP inference."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def prepare_for_training(self, *args, **kwargs) -> None:
        """Prepare the model for training."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def offload_before_refit(self) -> None:
        """Offload the model before refit."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def offload_after_refit(self) -> None:
        """Offload the model after refit."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    #########################################################
    # TODO: are these needed?
    def move_optimizer_to_device(self, device: str | torch.device) -> None:
        """Move the optimizer to the device."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def move_model_to_device(self, model: torch.nn.Module, device: str | torch.device) -> torch.nn.Module:
        """Move the model to the device."""
        raise NotImplementedError("Subclasses must implement this method.")
    #########################################################
    
    def save_checkpoint(self, weights_path: str, optimizer_path: Optional[str] = None, **kwargs) -> None:
        """Save the checkpoint of the model."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def load_checkpoint(self, weights_path: str, optimizer_path: Optional[str] = None) -> None:
        """Load the checkpoint of the model."""
        raise NotImplementedError("Subclasses must implement this method.")
    

    ##### NEW APIS INTRODUCED WITH THIS REFACTOR ########################################################
    def forward(
        ### TODO: figure this out
        self,
        model,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
        collection_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> BatchedDataDict[Any]:
        """Forward pass through the model."""
        raise NotImplementedError("Subclasses must implement this method.")