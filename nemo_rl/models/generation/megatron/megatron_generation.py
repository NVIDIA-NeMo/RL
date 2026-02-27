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

"""MegatronGeneration: A GenerationInterface implementation for non-colocated
Megatron-based inference.

This module wraps a Policy object (configured for inference only, without
optimizer or reference model) and exposes it through the GenerationInterface.
It enables non-colocated inference where training and generation run on
separate GPU clusters, with weights synchronized via NCCL collective
communication.

The init_collective and update_weights_from_collective methods are currently
placeholders that will be implemented in a future PR.
"""

from typing import Any, Optional

import ray
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.policy import PolicyConfig


class MegatronGeneration(GenerationInterface):
    """Generation interface backed by Megatron for non-colocated inference.

    This class creates a Policy instance configured for inference only
    (no optimizer, no reference model) on a dedicated inference cluster.
    It implements the GenerationInterface so it can be used as a drop-in
    replacement for VllmGeneration in the non-colocated inference flow.

    The weight synchronization methods (init_collective, update_weights_from_collective)
    are placeholders that will be implemented in a future PR.
    """

    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: PolicyConfig,
        tokenizer: PreTrainedTokenizerBase,
        name_prefix: str = "megatron_generation",
        processor: Optional[AutoProcessor] = None,
        weights_path: Optional[str] = None,
    ):
        """Initialize a MegatronGeneration instance.

        Args:
            cluster: The RayVirtualCluster to deploy inference workers on.
            config: PolicyConfig for the Megatron model.
            tokenizer: The tokenizer for the model.
            name_prefix: Prefix for naming the worker group.
            processor: Optional processor for VLMs.
            weights_path: Optional path to model weights for initialization.
        """
        # Import here to avoid circular imports
        from nemo_rl.models.policy.lm_policy import Policy

        self.cfg = config

        # Create a Policy object configured for inference only:
        # - No optimizer (not training on this cluster)
        # - No reference model (not needed for generation)
        self._policy = Policy(
            cluster=cluster,
            config=config,
            tokenizer=tokenizer,
            name_prefix=name_prefix,
            processor=processor,
            init_optimizer=False,
            init_reference_model=False,
            weights_path=weights_path,
        )

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        """Initialize the collective communication for weight synchronization.

        This sets up NCCL communication between training workers and these
        inference workers so that updated model weights can be broadcast
        from the training cluster to the inference cluster.

        Uses init_collective_as_inference on the workers, which offsets each
        worker's rank by train_world_size to avoid colliding with training
        workers' ranks (rank = train_world_size + worker_rank).

        Args:
            ip: IP address for the process group rendezvous.
            port: Port for the process group rendezvous.
            world_size: Total world size (train + inference workers).
            train_world_size: Number of training workers (used to offset ranks).

        Returns:
            List of Ray ObjectRefs for the collective init futures.
        """
        futures = self._policy.worker_group.run_all_workers_single_data(
            "init_collective_as_inference",
            ip=ip,
            port=port,
            world_size=world_size,
            train_world_size=train_world_size,
        )
        return futures

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        """Receive updated weights from the training cluster via collective communication.

        This method is called after the training side calls
        policy.broadcast_weights_for_collective(). It receives the broadcast
        weights and updates the local model parameters.

        TODO: This is a placeholder. The actual implementation will:
        1. Iterate over the model's state_dict info
        2. Use packed_broadcast_consumer to receive weights from the training side
        3. Update the local model parameters with the received weights

        Returns:
            List of Ray ObjectRefs for the weight update futures.
        """
        futures = self._policy.worker_group.run_all_workers_single_data(
            "update_weights_from_collective",
        )
        return futures

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using the Megatron generation backend.

        Delegates to the internal Policy's generate method.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths.
            greedy: Whether to use greedy decoding.

        Returns:
            BatchedDataDict conforming to GenerationOutputSpec.
        """
        return self._policy.generate(data, greedy=greedy)

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Prepare the inference workers for generation.

        For Megatron generation, this is a no-op since the workers
        are always ready for inference.
        """
        return self._policy.prepare_for_generation(*args, **kwargs)

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Clean up after generation.

        For Megatron generation, this is a no-op.
        """
        return self._policy.finish_generation(*args, **kwargs)

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Prepare state dict metadata for weight refitting.

        This stores the state dict info (tensor names, shapes, dtypes) on each
        inference worker so that update_weights_from_collective knows what
        tensors to expect during the weight broadcast.

        Note: This calls store_refit_info on workers (not prepare_refit_info),
        because prepare_refit_info on MegatronPolicyWorker calculates and
        returns metadata (training-side), while store_refit_info accepts and
        stores metadata (inference-side).

        Args:
            state_dict_info: Dictionary mapping tensor names to (shape, dtype) tuples,
                as returned by the training-side prepare_refit_info().
        """
        futures = self._policy.worker_group.run_all_workers_single_data(
            "store_refit_info",
            state_dict_info=state_dict_info,
        )
        ray.get(futures)

    def shutdown(self) -> bool:
        """Shut down all inference workers and clean up resources."""
        return self._policy.shutdown()

    def __del__(self) -> None:
        """Safety net to ensure workers are shut down."""
        if hasattr(self, "_policy"):
            self._policy.shutdown()
