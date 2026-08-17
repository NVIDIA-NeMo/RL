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

from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

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

if TYPE_CHECKING:
    from nemo_rl.models.policy.lm_policy import Policy


class MegatronGeneration(GenerationInterface):
    """Generation interface backed by Megatron (colocated or non-colocated)."""

    @staticmethod
    def effective_megatron_cfg(config: PolicyConfig) -> dict[str, Any]:
        """The megatron_cfg the generation workers actually run with.

        Colocated generation shares the training model, so the training
        values apply; non-colocated builds a dedicated policy with
        mcore_generation_config merged on top. Always returns a fresh dict.
        """
        megatron_cfg = config["megatron_cfg"]
        if config["generation"]["colocated"]["enabled"]:
            return dict(megatron_cfg)
        return {
            **megatron_cfg,
            **config["generation"].get("mcore_generation_config", {}),
        }

    @classmethod
    def nvlink_domain_span(cls, config: PolicyConfig) -> int:
        """Largest GPU group requiring full NVLink connectivity."""
        megatron_cfg = cls.effective_megatron_cfg(config)
        return max(
            megatron_cfg["tensor_model_parallel_size"]
            * megatron_cfg["context_parallel_size"],
            megatron_cfg.get("expert_tensor_parallel_size", 1)
            * megatron_cfg.get("expert_model_parallel_size", 1),
        )

    @classmethod
    def init_cluster_placement_groups(
        cls,
        cluster: RayVirtualCluster,
        config: PolicyConfig,
    ) -> None:
        """Pre-initialize the inference cluster's placement groups.

        Args:
            cluster: The inference `RayVirtualCluster`.
            config: The full `PolicyConfig` (megatron parallelism + colocation).
        """
        colocated = config["generation"]["colocated"]["enabled"]
        cluster._init_placement_groups(
            strategy=None if colocated else "PACK",
            use_unified_pg=cls.nvlink_domain_span(config) > cluster.num_gpus_per_node,
        )

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: PreTrainedTokenizerBase,
        cluster: Optional[RayVirtualCluster] = None,
        policy: Optional["Policy"] = None,
        name_prefix: str = "megatron_generation",
        processor: Optional[AutoProcessor] = None,
        weights_path: Optional[str] = None,
        skip_weight_load: bool = False,
    ):
        """Initialize a MegatronGeneration instance.

        Exactly one of `cluster` or `policy` must be provided.

        Args:
            config: PolicyConfig for the Megatron model.
            tokenizer: The tokenizer for the model.
            cluster: Cluster to deploy a dedicated inference Policy on.
            policy: Existing training Policy to reuse for generation.
            name_prefix: Prefix for naming the worker group (non-colocated only).
            processor: Optional processor for VLMs (non-colocated only).
            weights_path: Optional path to model weights (non-colocated only).
            skip_weight_load: Do not load the weights from the checkpoint; refit will do it.
        """
        # Import here to avoid circular imports
        from nemo_rl.models.policy.lm_policy import Policy

        assert (cluster is None) != (policy is None), (
            "Provide exactly one of `cluster` or `policy`."
        )
        assert not (skip_weight_load and policy is not None), (
            "skip_weight_load only applies to the dedicated inference policy."
        )

        # `self.cfg` exposes the `generation` that matches the `GenerationInterface` contract.
        # `self._policy_config` keeps a reference to the full PolicyConfig.
        self._policy_config = config
        self.cfg: MCoreGenerationConfig = config["generation"]
        # Populated after the first prepare_for_generation (which starts the HTTP server).
        self.dp_openai_server_base_urls: list[Optional[str]] = []
        self.current_generate_dp_shard_idx = 0

        if policy is not None:
            # Reuse the existing training policy.
            self._policy = policy
            self._owns_policy = False
            if self.cfg["mcore_generation_config"]["expose_http_server"]:
                self._policy.offload_before_refit()
                self.prepare_for_generation()
            return

        # Stand up a dedicated inference-only policy.
        self._owns_policy = True
        self._policy_config = {
            **config,
            "megatron_cfg": self.effective_megatron_cfg(config),
        }
        # Activation checkpointing is not compatible or useful in inference.
        self._policy_config["megatron_cfg"]["activation_checkpointing"] = False
        # Reserve GPUs before Policy workers grab them, to prevent disjoint NVLS domains.
        self.init_cluster_placement_groups(cluster, self._policy_config)
        self._policy = Policy(
            cluster=cluster,
            config=self._policy_config,
            tokenizer=tokenizer,
            name_prefix=name_prefix,
            processor=processor,
            init_optimizer=False,
            init_reference_model=False,
            weights_path=weights_path,
            skip_weight_load=skip_weight_load,
        )

        # Start the persistent inference engine + HTTP server during construction.
        self.prepare_for_generation()

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
            refit_backend: Copy service backend ("gloo" or "nccl";
                "nvshmem" is currently broken and warns at setup).

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

    @property
    def sharding_annotations(self):
        """Parallelism layout for the underlying Megatron policy workers."""
        return self._policy.sharding_annotations

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using the Megatron generation backend.

        Prompts are sharded across DP coordinator leaders (mirroring the vLLM
        backend). Each leader submits its shard to its replica's mcore
        coordinator; non-leader TP/PP ranks are not invoked.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths.
            greedy: Whether to use greedy decoding.

        Returns:
            BatchedDataDict conforming to GenerationOutputSpec.
        """
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            "input_ids and input_lengths are required in data for Megatron generation"
        )

        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data = data.shard_by_batch_size(dp_size, allow_uneven_shards=True)
        future_bundle = self._policy.worker_group.run_all_workers_sharded_data(
            "generate",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=None,
            output_is_replicated=None,
            common_kwargs={"greedy": greedy},
        )
        results = self._policy.worker_group.get_all_worker_results(future_bundle)
        combined: BatchedDataDict[GenerationOutputSpec] = BatchedDataDict.from_batches(
            results,
            pad_value_dict={"output_ids": self.cfg["_pad_token_id"]},
        )

        required_keys = [
            "output_ids",
            "generation_lengths",
            "unpadded_sequence_lengths",
            "logprobs",
        ]
        missing_keys = [key for key in required_keys if key not in combined]
        if missing_keys:
            raise ValueError(
                f"Missing required keys for GenerationOutputSpec: {missing_keys}"
            )

        return combined

    async def generate_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate asynchronously, yielding `(index, batch)` tuples as they complete.

        Single-sample requests are round-robin dispatched across DP coordinator
        leaders, matching the vLLM async rollout path.
        """
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert data.size == 1, (
            "Megatron generate_async handles one sample per call; batch outside this method."
        )

        leader_worker_idx = self._policy.worker_group.get_dp_leader_worker_idx(
            self.current_generate_dp_shard_idx
        )
        worker = self._policy.worker_group.workers[leader_worker_idx]
        futures = worker.generate_async.options(num_returns="streaming").remote(
            data=data, greedy=greedy
        )
        self.current_generate_dp_shard_idx = (
            self.current_generate_dp_shard_idx + 1
        ) % self._policy.worker_group.dp_size

        async for result_ref in futures:
            index, result_batch = await result_ref
            result_batch["gen_leader_worker_idx"] = [leader_worker_idx]
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
        if (
            not self.dp_openai_server_base_urls
            and self.cfg["mcore_generation_config"]["expose_http_server"]
        ):
            url_futures = []
            for dp_idx in range(self._policy.worker_group.dp_size):
                worker_idx = self._policy.worker_group.get_dp_leader_worker_idx(dp_idx)
                url_futures.append(
                    self._policy.worker_group.run_single_worker_single_data(
                        "report_dp_openai_server_base_url",
                        worker_idx=worker_idx,
                    )
                )
            self.dp_openai_server_base_urls = [
                url for url in ray.get(url_futures) if url is not None
            ]
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

    def prepare_refit_info(self, state_dict_info: Optional[dict[str, Any]]) -> None:
        """Accept the cross-backend refit-prep contract; Megatron needs none of it."""
        pass

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling on the dedicated inference workers.

        No-op when colocated: the shared workers are already profiled through the training policy.
        """
        if self._owns_policy:
            self._policy.start_gpu_profiling()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling on the dedicated inference workers."""
        if self._owns_policy:
            self._policy.stop_gpu_profiling()

    def shutdown(self) -> bool:
        """Shut down all inference workers and clean up resources."""
        if not self._owns_policy:
            return True
        return self._policy.shutdown()

    def __del__(self) -> None:
        """Safety net to ensure workers are shut down."""
        if hasattr(self, "_policy") and self._owns_policy:
            self._policy.shutdown()
