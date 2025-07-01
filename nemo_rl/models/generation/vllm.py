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

from collections import defaultdict
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    NotRequired,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
import ray
from ray.util.placement_group import PlacementGroup

from nemo_rl.distributed.batched_data_dict import BatchedDataDict, SlicedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.virtual_cluster import (
    RayVirtualCluster,
)
from nemo_rl.distributed.worker_groups import (
    RayWorkerBuilder,
    RayWorkerGroup,
)
from nemo_rl.models.generation.interfaces import (
    GenerationConfig,
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)


class VllmSpecificArgs(TypedDict):
    tensor_parallel_size: int
    pipeline_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    # Additional arguments for vLLM inserted by nemo rl based on the context of when vllm is used
    skip_tokenizer_init: bool
    async_engine: bool
    load_format: NotRequired[str]
    precision: NotRequired[str]


class VllmConfig(GenerationConfig):
    vllm_cfg: VllmSpecificArgs
    vllm_kwargs: NotRequired[dict[str, Any]]


class VllmGeneration(GenerationInterface):
    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: VllmConfig,
        name_prefix: str = "vllm_policy",
        workers_per_node: Optional[Union[int, list[int]]] = None,
    ):
        """Initialize a vLLM policy with distributed workers."""
        # Store config
        self.cfg = config
        if self.cfg["vllm_cfg"]["pipeline_parallel_size"] > 1:
            assert self.cfg["vllm_cfg"]["async_engine"], (
                "When pipeline_parallel_size > 1, async_engine must be set to True in the vLLM configuration. "
                "You can enable it by adding `policy.generation.vllm_cfg.async_engine=true` to your command."
            )

        # Ensure all required VllmConfig fields are present
        missing_keys = [
            key for key in VllmConfig.__required_keys__ if key not in self.cfg
        ]
        assert not missing_keys, (
            f"VLLM Configuration Error: Missing required keys in VllmConfig.\n"
            f"Missing keys: {', '.join(missing_keys)}\n"
            f"Provided keys: {', '.join(self.cfg.keys())}\n"
            f"Please update your configuration to include all required VLLM parameters."
        )

        self.sharding_annotations = NamedSharding(
            layout=np.arange(cluster.world_size()).reshape(
                -1,  # DP
                config["vllm_cfg"]["pipeline_parallel_size"],  # PP
                config["vllm_cfg"]["tensor_parallel_size"],  # TP
            ),
            names=["data_parallel", "pipeline_parallel", "tensor_parallel"],
        )
        self.model_parallel_size = self.sharding_annotations.get_axis_size(
            "tensor_parallel"
        ) * self.sharding_annotations.get_axis_size("pipeline_parallel")

        # Determine if we need cross-node model parallelism
        needs_cross_node_parallelism = (
            self.model_parallel_size > cluster.num_gpus_per_node
        )

        # Initialize placement groups with the appropriate mode
        cluster._init_placement_groups(use_unified_pg=needs_cross_node_parallelism)

        # Create worker builder for VllmGenerationWorker
        if self.cfg["vllm_cfg"]["async_engine"]:
            worker_cls = (
                "nemo_rl.models.generation.vllm_worker_async.VllmAsyncGenerationWorker"
            )
        else:
            worker_cls = "nemo_rl.models.generation.vllm_worker.VllmGenerationWorker"
        worker_builder = RayWorkerBuilder(worker_cls, config)

        # Check if we need parallelism-aware worker group creation
        if self.model_parallel_size > 1:
            # For parallelism, create node-aware worker groups
            node_bundle_indices = self._get_tied_worker_bundle_indices(cluster)

            self.worker_group = RayWorkerGroup(
                cluster,
                worker_builder,
                name_prefix=name_prefix,
                bundle_indices_list=node_bundle_indices,
                sharding_annotations=self.sharding_annotations,
            )
        else:
            # Use standard worker group creation for non-parallel case
            self.worker_group = RayWorkerGroup(
                cluster,
                worker_builder,
                name_prefix=name_prefix,
                workers_per_node=workers_per_node,
                sharding_annotations=self.sharding_annotations,
            )

        # Number of data parallel groups is the number of tied worker groups
        self.dp_size = self.worker_group.dp_size

        # Used to track the round-robin selection of worker groups for generate_async
        self.current_generate_dp_shard_idx = 0

        # Save the device uuids for the workers
        self.device_uuids = self._report_device_id()

    def _get_tied_worker_bundle_indices(
        self, cluster: RayVirtualCluster
    ) -> List[Tuple[int, List[int]]]:
        """Calculate bundle indices for tensor and pipeline parallel workers.

        Handles both unified placement groups (for cross-node model parallelism) and
        per-node placement groups (for node-local model parallelism).
        """
        # Get the placement groups from the cluster
        placement_groups = cluster.get_placement_groups()

        if not placement_groups:
            raise ValueError("No placement groups available in the cluster")

        # Total parallel sizes
        tp_size = self.sharding_annotations.get_axis_size("tensor_parallel")
        pp_size = self.sharding_annotations.get_axis_size("pipeline_parallel")
        model_parallel_size = tp_size * pp_size

        if len(placement_groups) == 1:
            # Single unified placement group used when we need multiple nodes for model parallelism
            unified_pg = placement_groups[0]

            def get_node_bundles(
                pg: PlacementGroup,
            ) -> Dict[str, List[int]]:
                # Retrieve mapping from node ID to bundle indices from a placement group.
                try:
                    pg_table = ray.util.placement_group_table(pg)
                    bundle_to_node = pg_table["bundles_to_node_id"]
                except Exception as e:
                    raise RuntimeError(
                        "Failed to retrieve bundle/node mapping from placement group"
                    ) from e

                node_bundles: Dict[str, List[int]] = defaultdict(list)
                for bundle_idx, node_id in bundle_to_node.items():
                    node_bundles[node_id].append(bundle_idx)
                for bundles in node_bundles.values():
                    bundles.sort()
                return dict(node_bundles)

            def allocate_worker_groups(
                pg: PlacementGroup, tp_size: int, pp_size: int
            ) -> List[Tuple[int, List[int]]]:
                # Allocate worker groups for TP and PP training, assuming all nodes have identical bundle counts.

                # Retrieve both bundle mapping and per-node bundles
                pg_table = ray.util.placement_group_table(pg)
                bundle_to_node = pg_table["bundles_to_node_id"]
                node_bundles = get_node_bundles(pg)

                if not node_bundles:
                    raise ValueError("Placement group contains no bundles")

                # Ensure all nodes have the same number of bundles
                counts = [len(b) for b in node_bundles.values()]
                assert len(set(counts)) == 1, (
                    "All nodes must have identical bundle counts"
                )

                total = sum(counts)
                model_parallel_size = tp_size * pp_size
                num_groups = total // model_parallel_size
                if num_groups == 0:
                    raise ValueError(
                        "Unable to allocate any worker groups with the available resources."
                    )

                # Create reproducible node indices
                sorted_nodes = sorted(node_bundles)
                node_idx = {nid: idx for idx, nid in enumerate(sorted_nodes)}

                # Flatten bundles in node order
                flat: List[int] = []
                for nid in sorted_nodes:
                    flat.extend(node_bundles[nid])

                # Slice into groups and assign logical index
                groups: List[Tuple[int, List[int]]] = []
                for i in range(num_groups):
                    slice_ = flat[
                        i * model_parallel_size : (i + 1) * model_parallel_size
                    ]
                    first_node = bundle_to_node[slice_[0]]
                    groups.append((node_idx[first_node], slice_))

                return groups

            tied_groups = allocate_worker_groups(unified_pg, tp_size, pp_size)
        else:
            tied_groups = []
            # For per-node PGs, each PG represents a node
            for pg_idx, pg in enumerate(placement_groups):
                if pg.bundle_count == 0:
                    continue

                # Check if this PG has enough bundles for at least one group
                num_groups_in_pg = pg.bundle_count // model_parallel_size

                # Create groups within this PG
                for group_idx in range(num_groups_in_pg):
                    start_idx = group_idx * model_parallel_size
                    end_idx = start_idx + model_parallel_size
                    bundle_indices = list(range(start_idx, end_idx))
                    # Use pg_idx as the node identifier
                    tied_groups.append((pg_idx, bundle_indices))

        if not tied_groups:
            raise ValueError(
                "Unable to allocate any worker groups with the available resources."
            )

        return tied_groups

    def _report_device_id(self) -> list[list[str]]:
        """Report the device ID of vllm workers."""
        # Use run_all_workers_single_data for methods that don't need data
        futures = self.worker_group.run_all_workers_single_data(
            method_name="report_device_id",
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )
        # Wait for all futures to complete
        results = ray.get(futures)
        return results

    def init_collective(
        self, ip: str, port: int, world_size: int
    ) -> list[ray.ObjectRef]:
        """Initialize the collective communication."""
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")

        # Prepare rank
        total_workers = len(self.worker_group.workers)
        if self.dp_size == 0:
            raise RuntimeError(
                "Data parallel size is zero, cannot initialize collective."
            )
        workers_per_group = total_workers // self.dp_size
        rank_prefix_list = list(range(0, total_workers, workers_per_group))

        # Send world_size and rank for init collective to all workers
        futures = self.worker_group.run_all_workers_multiple_data(
            method_name="init_collective",
            data=rank_prefix_list,
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
            common_kwargs={"ip": ip, "port": port, "world_size": world_size},
        )

        # this function should co-work with lm_policy, so we should wait for all futures to complete outside
        return futures

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using vLLM."""
        assert not self.cfg["vllm_cfg"]["async_engine"], (
            "generate can only be used when async_engine is not enabled in VllmConfig."
        )
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            "input_ids and input_lengths are required in data for vLLM generation"
        )

        # Shard the data across the tied worker groups
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict] = data.shard_by_batch_size(
            dp_size, allow_uneven_shards=True
        )
        future_bundle = self.worker_group.run_all_workers_sharded_data(
            "generate",
            sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=None,  # just run on tp rank 0
            output_is_replicated=None,
            common_kwargs={"greedy": greedy},
        )

        # Get results from the workers, respecting tied worker groups (only one result per tied worker group)
        results = self.worker_group.get_all_worker_results(future_bundle)

        # Combine results from all tied worker groups
        combined: BatchedDataDict[GenerationOutputSpec] = BatchedDataDict.from_batches(
            results, pad_value_dict={"output_ids": self.cfg["pad_token_id"]}
        )

        # Verify the output has all required fields
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

    def generate_text(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate text responses using vLLM."""
        assert not self.cfg["vllm_cfg"]["async_engine"], (
            "generate can only be used when async_engine is not enabled in VllmConfig."
        )
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )

        # Shard the data across the tied worker groups
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict] = data.shard_by_batch_size(
            dp_size, allow_uneven_shards=True
        )
        future_bundle = self.worker_group.run_all_workers_sharded_data(
            "generate_text",
            sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=None,  # just run on tp rank 0
            output_is_replicated=None,
            common_kwargs={"greedy": greedy},
        )

        # Get results from the workers, respecting tied worker groups (only one result per tied worker group)
        results = self.worker_group.get_all_worker_results(future_bundle)

        # Combine results from all tied worker groups
        combined: BatchedDataDict[GenerationOutputSpec] = BatchedDataDict.from_batches(
            results, pad_value_dict={"output_ids": self.cfg["pad_token_id"]}
        )

        # Verify the output has all required fields
        required_keys = ["texts"]
        missing_keys = [key for key in required_keys if key not in combined]
        if missing_keys:
            raise ValueError(
                f"Missing required keys for GenerationOutputSpec: {missing_keys}"
            )

        return combined

    async def generate_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate responses asynchronously, yielding individual samples as they complete.

        This method provides per-sample streaming across all workers, yielding each
        sample result as soon as it's ready, regardless of which worker processed it.
        """
        assert self.cfg["vllm_cfg"]["async_engine"], (
            "generate_async can only be used when async_engine is enabled in VllmConfig."
        )
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            "input_ids and input_lengths are required in data for vLLM generation"
        )

        # Ensure generate_async only receives single samples (batch_size = 1)
        batch_size = data["input_ids"].shape[0]
        assert batch_size == 1, (
            f"generate_async is restricted to handle only single samples, "
            f"but received batch_size={batch_size}. Please handle batching outside this method."
        )

        # Handle empty input case
        if len(data["input_ids"]) == 0:
            return

        # Determine the leader worker for the current data parallel shard
        leader_worker_idx = self.worker_group.get_dp_leader_worker_idx(
            self.current_generate_dp_shard_idx
        )

        # Run the generate method on the selected leader worker. This returns an ObjectRefGenerator.
        worker_gen_proxy = self.worker_group.run_single_worker_single_data(
            method_name="generate",
            worker_idx=leader_worker_idx,
            data=data,
            greedy=greedy,
        )

        # Increment the round-robin worker group index
        self.current_generate_dp_shard_idx += 1
        self.current_generate_dp_shard_idx %= self.worker_group.dp_size

        result = await worker_gen_proxy
        return result

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Wake workers up for colocated inference."""
        # non-colocated no need to wake up
        if not self.cfg["colocated"]["enabled"]:
            return True

        try:
            # Use run_all_workers_single_data for methods that don't need data
            futures = self.worker_group.run_all_workers_single_data(
                method_name="wake_up",
                run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
                **kwargs,
            )
            # Wait for all futures to complete
            results = ray.get(futures)
            return all(result for result in results if result is not None)
        except Exception as e:
            print(f"Error during policy preparation: {e}")
            return False

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Sleep workers and reset prefix cache."""
        try:
            # non-colocated only needs reset prefix cache, no need to sleep.
            method_name = (
                "sleep" if self.cfg["colocated"]["enabled"] else "reset_prefix_cache"
            )
            # Use run_all_workers_single_data for methods that don't need data
            futures = self.worker_group.run_all_workers_single_data(
                method_name,
                run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
            )
            # Wait for all futures to complete
            results = ray.get(futures)
            return all(result for result in results if result is not None)
        except Exception as e:
            print(f"Error during policy preparation: {e}")
            return False

    def shutdown(self) -> bool:
        """Shut down all vLLM workers and clean up resources."""
        try:
            # Use the worker group's shutdown method with the worker's cleanup method
            return self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            print(f"Error during policy shutdown: {e}")
            return False

    def update_weights(self, ipc_handles: dict[str, Any]) -> bool:
        """Update weights of the policy using IPC handles, considering tensor parallelism.

        For tp > 1, only the leader in each tensor parallel tied worker group will update weights.

        Args:
            ipc_handles (dict): Dictionary mapping device UUIDs (str) to parameter IPC handles.

        Returns:
            bool: True if weights were successfully updated, False otherwise.
        """
        if not self.worker_group or not self.worker_group.workers:
            return False

        # Only send the ipc handles required by the current worker
        ipc_handles_list = []
        for worker_device_uuids in self.device_uuids:
            worker_ipc_handles = {
                device_uuid: ipc_handles[device_uuid]
                for device_uuid in worker_device_uuids
            }
            ipc_handles_list.append(worker_ipc_handles)

        try:
            # Directly pass ipc_handles to the method
            futures = self.worker_group.run_all_workers_multiple_data(
                method_name="update_weights_from_ipc_handles",
                data=ipc_handles_list,
                run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
            )
            # Wait for all futures to complete
            results = ray.get(futures)
            return all(result for result in results if result is not None)
        except Exception as e:
            print(f"Error during update weights: {e}")
            return False

    def update_weights_from_collective(
        self, info: dict[str, Any]
    ) -> list[ray.ObjectRef]:
        """Update weights of the policy using collective communication."""
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")

        # Use run_all_workers_single_data to send data to all workers
        futures = self.worker_group.run_all_workers_single_data(
            method_name="update_weights_from_collective",
            data=info,
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )

        # this function should co-work with lm_policy, so we should wait for all futures to complete outside
        return futures

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        futures = self.worker_group.run_all_workers_single_data("start_gpu_profiling")
        ray.get(futures)

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        futures = self.worker_group.run_all_workers_single_data("stop_gpu_profiling")
        ray.get(futures)

    def __del__(self) -> None:
        """Shuts down the worker groups when the object is deleted or is garbage collected.

        This is an extra safety net in case the user forgets to call shutdown() and the pointer to
        the object is lost due to leaving a function scope. It's always recommended that the
        user calls shutdown().
        """
        self.shutdown()
