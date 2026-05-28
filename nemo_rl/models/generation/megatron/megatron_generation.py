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

from typing import Any, Optional, AsyncGenerator

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
from nemo_rl.models.generation.megatron.config import MCoreGenerationConfig
from nemo_rl.models.policy import PolicyConfig


class MegatronGeneration(GenerationInterface):
    """Generation interface backed by Megatron for non-colocated inference."""

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

        # `self.cfg` exposes the `generation` that matches the `GenerationInterface` contract.
        # `self._policy_config` keeps a reference to the full PolicyConfig.
        self._policy_config = config
        self.cfg: MCoreGenerationConfig = config["generation"]
        # Populated after the first prepare_for_generation (which starts the HTTP server).
        self.dp_openai_server_base_urls: list[Optional[str]] = []

        # Need to update the megatron_cfg with the mcore_generation_config parameters.
        self._policy_config["megatron_cfg"].update(self.cfg["mcore_generation_config"])

        # Create a Policy object configured for inference only:
        # - No optimizer (not training on this cluster)
        # - No reference model (not needed for generation)
        self._policy = Policy(
            cluster=cluster,
            config=self._policy_config,
            tokenizer=tokenizer,
            name_prefix=name_prefix,
            processor=processor,
            init_optimizer=False,
            init_reference_model=False,
            weights_path=weights_path,
        )

        # Start the inference engine + HTTP server during construction.
        self.prepare_for_generation()

        url_futures = self._policy.worker_group.run_all_workers_single_data(
            "report_dp_openai_server_base_url"
        )
        self.dp_openai_server_base_urls = [
            url for url in ray.get(url_futures) if url is not None
        ]

    def init_collective(
        self,
        ip: str,
        port: int,
        world_size: int,
        *,
        train_world_size: int,
        refit_backend: str = "gloo",
    ) -> list[ray.ObjectRef]:
        """Initialize the refit collective for weight synchronization.

        Args:
            ip: IP address for the process group rendezvous.
            port: Port for the process group rendezvous.
            world_size: Total world size (train + inference workers).
            train_world_size: Number of training workers (used to offset ranks).
            refit_backend: Copy service backend ("gloo" or "nvshmem").

        Returns:
            List of Ray ObjectRefs for the collective init futures.
        """
        return self._policy.init_collective_mcore_generation(
            ip,
            port,
            world_size,
            rank_offset=train_world_size,
            refit_backend=refit_backend,
        )

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        """Receive updated weights from the training cluster via collective communication."""
        return self._policy.swap_weights_via_reshard(is_source=False)

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using the Megatron generation backend.

        mcore's data-parallel coordinator only accepts requests from DP rank 0 —
        the other workers' engine loops drain the coordinator queue but never
        receive a Python-side call. So we dispatch straight to worker 0.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths.
            greedy: Whether to use greedy decoding.

        Returns:
            BatchedDataDict conforming to GenerationOutputSpec.
        """
        future = self._policy.worker_group.run_single_worker_single_data(
            method_name="generate",
            worker_idx=0,
            data=data,
            greedy=greedy,
        )
        return ray.get(future)

    async def generate_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate asynchronously, yielding `(index, batch)` tuples as they complete."""
        worker = self._policy.worker_group.workers[0]
        futures = worker.generate_async.options(num_returns="streaming").remote(
            data=data, greedy=greedy
        )
        async for result_ref in futures:
            index, result_batch = await result_ref
            result_batch["gen_leader_worker_idx"] = [0]
            yield index, result_batch

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Initialize / re-enter inference mode on every worker.

        First call starts the persistent inference engine, coordinator, and the OpenAI HTTP server.
        Subsequent calls re-enter inference mode after a refit.
        """
        futures = self._policy.worker_group.run_all_workers_single_data(
            "prepare_for_generation", **kwargs
        )
        ray.get(futures)
        return True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Clean up after generation."""
        futures = self._policy.worker_group.run_all_workers_single_data(
            "finish_generation"
        )
        ray.get(futures)
        return True

    def preinit_nvshmem_collective(self) -> list[ray.ObjectRef]:
        """Pre-initialize NVShmem collectively after CUDA graph capture.

        Must be called simultaneously on both training and inference workers.
        """
        return self._policy.preinit_nvshmem()

    def suspend_for_refit(self) -> None:
        """Suspend the inference engine for safe weight updates."""
        ray.get(
            self._policy.worker_group.run_all_workers_single_data("suspend_for_refit")
        )

    def resume_after_refit(self) -> None:
        """Resume the inference engine after weight updates."""
        ray.get(
            self._policy.worker_group.run_all_workers_single_data("resume_after_refit")
        )

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Prepare state dict metadata for weight refitting.

        Args:
            state_dict_info: Dictionary mapping tensor names to (shape, dtype) tuples.
        """
        futures = self._policy.worker_group.run_all_workers_single_data(
            "prepare_refit_info",
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
