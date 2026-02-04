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

import asyncio
import os
from collections import defaultdict
from typing import (
    Any,
    AsyncGenerator,
    Optional,
    Union,
)

import numpy as np
import ray
from ray.util.placement_group import PlacementGroup

from nemo_rl.distributed.batched_data_dict import BatchedDataDict, SlicedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.generation.vllm.config import VllmConfig

# Global thresholds for top_k and top_p validation.
# While top-k/p are not supported, these values allow for token filtering while the logprobs should be compatible.
# See https://github.com/NVIDIA-NeMo/RL/issues/69 and https://github.com/NVIDIA-NeMo/RL/issues/237 for more details.
TOP_K_THRESHOLD = 8000  # Allow top_k >= 8000 (effectively no filtering)
TOP_P_THRESHOLD = 0.99  # Allow top_p >= 0.99 (close to 1.0)


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
        self.tp_size = self.cfg["vllm_cfg"]["tensor_parallel_size"]
        self.pp_size = self.cfg["vllm_cfg"]["pipeline_parallel_size"]
        self.ep_size = self.cfg["vllm_cfg"]["expert_parallel_size"]
        self.model_parallel_size = self.tp_size * self.pp_size

        assert cluster.world_size() % self.model_parallel_size == 0, (
            "World size must be a multiple of model parallel size. "
            f"Got world size {cluster.world_size()} and model parallel size (TP * PP) {self.model_parallel_size}."
        )
        self.dp_size = cluster.world_size() // self.model_parallel_size
        self.vllm_dp_size = self.ep_size // self.tp_size

        if self.pp_size > 1:
            assert self.cfg["vllm_cfg"]["async_engine"], (
                "When pipeline_parallel_size > 1, async_engine must be set to True in the vLLM configuration. "
                "You can enable it by adding `policy.generation.vllm_cfg.async_engine=true` to your command."
            )

        if self.ep_size > 1:
            assert self.ep_size % self.tp_size == 0, (
                "When EP > 1, EP must be a multiple of TP since vLLM's EP = DP * TP. "
                "Please update your configuration to set expert_parallel_size to a multiple of tensor_parallel_size."
            )
            if self.ep_size != self.tp_size:
                # vLLM's EP = DP * TP, so here we need to use DP inside vLLM.
                assert not self.cfg["vllm_cfg"]["async_engine"], (
                    "vLLM async_engine has some issues when using DP inside vLLM. "
                    "Please update your configuration to set `policy.generation.vllm_cfg.async_engine=false`. "
                    "See https://github.com/NVIDIA-NeMo/RL/issues/1101 for more details."
                )

        # Validate sampling parameters early to avoid resource allocation with unsupported configs.
        # The vLLM sampler patch only supports temperature scaling and does not handle top_p/top_k correctly.
        # However, we allow values above certain thresholds for token filtering purposes.
        top_k = self.cfg["top_k"]
        if top_k is not None and top_k != -1 and top_k < TOP_K_THRESHOLD:
            raise ValueError(
                (
                    f"top_k sampling with values < {TOP_K_THRESHOLD} is not supported because the vLLM V1 engine "
                    "does not return logprobs after top_k filtering. Values >= {TOP_K_THRESHOLD} are allowed "
                    "for token filtering purposes. If you understand the implications and still want to use "
                    f"a lower top_k value, please manually comment out this check. Got top_k={top_k}. "
                    "See https://github.com/NVIDIA-NeMo/RL/issues/69 for more details."
                )
            )

        top_p: float = self.cfg.get("top_p", 1.0)
        if top_p < TOP_P_THRESHOLD:
            raise ValueError(
                (
                    f"top_p sampling with values < {TOP_P_THRESHOLD} is not supported because the vLLM V1 engine "
                    "does not return logprobs after top_p filtering. Values >= {TOP_P_THRESHOLD} are allowed "
                    "for token filtering purposes. If you understand the implications and still want to use "
                    f"a lower top_p value, please manually comment out this check. Got top_p={top_p}. "
                    "See https://github.com/NVIDIA-NeMo/RL/issues/69 for more details."
                )
            )

        # Ensure all required VllmConfig fields are present
        missing_keys = [
            key for key in VllmConfig.__required_keys__ if key not in self.cfg
        ]
        # Also check for model_name which is required by VllmGenerationWorker but marked as NotRequired in GenerationConfig because it's not expected to be set in the job yaml.
        if "model_name" not in self.cfg:
            missing_keys.append("model_name")

        assert not missing_keys, (
            f"VLLM Configuration Error: Missing required keys in VllmConfig.\n"
            f"Missing keys: {', '.join(missing_keys)}\n"
            f"Provided keys: {', '.join(self.cfg.keys())}\n"
            f"Please update your configuration to include all required VLLM parameters."
        )

        self.sharding_annotations = NamedSharding(
            layout=np.arange(cluster.world_size()).reshape(
                self.dp_size, self.pp_size, self.tp_size
            ),
            names=["data_parallel", "pipeline_parallel", "tensor_parallel"],
        )

        # non-colocated needs to use PACK strategy to avoid uneven node_bundles
        # e.g. assuming we use 3 nodes with 8GPUs, 2 nodes for train and 1 node for inference.
        # if we use SPREAD, then the node bundles will be something like 0: [0,3,6] 1: [1,4,7] 2: [2,5], which is not correct.
        strategy = None if self.cfg["colocated"]["enabled"] else "PACK"

        # Determine if we need cross-node model parallelism
        needs_cross_node_parallelism = (
            self.model_parallel_size > cluster.num_gpus_per_node
        )

        # Initialize placement groups with the appropriate mode
        cluster._init_placement_groups(
            strategy=strategy,
            use_unified_pg=needs_cross_node_parallelism,
        )

        # Create worker builder for VllmGenerationWorker
        if self.cfg["vllm_cfg"]["async_engine"]:
            worker_cls = "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"
        else:
            worker_cls = (
                "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
            )
        worker_builder = RayWorkerBuilder(worker_cls, config)

        # It's necessary to set env_vars here to ensure that vllm non-leader workers also have these env_vars
        env_vars = {}
        # Explicitly set NCCL_CUMEM_ENABLE to 1 to avoid the P2P initialization error for PyNCCLCommunicator.
        # See https://github.com/NVIDIA-NeMo/RL/issues/564 for more details.
        if not self.cfg["colocated"]["enabled"]:
            env_vars["NCCL_CUMEM_ENABLE"] = "1"

        if needs_cross_node_parallelism:
            # When using cross-node model parallelism with non-colocated inference,
            # we are disabling NCCL_NVLS_ENABLE to avoid the NCCL error.
            # See https://github.com/NVIDIA-NeMo/RL/issues/1352 for more details.
            env_vars["NCCL_NVLS_ENABLE"] = "0"
            print(
                "[INFO] NCCL_NVLS_ENABLE is set to 0 for non-colocated inference with cross-node model parallelism."
                "See https://github.com/NVIDIA-NeMo/RL/issues/1352 for more details."
            )
        # We should use vLLM DP if ep_size > tp_size since EP_SIZE = DP_SIZE * TP_SIZE in vLLM.
        # See details in https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/data_parallel.py
        if self.ep_size > self.tp_size:
            env_vars["VLLM_DP_SIZE"] = str(self.vllm_dp_size)

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
                env_vars=env_vars,
            )
        else:
            # Use standard worker group creation for non-parallel case
            self.worker_group = RayWorkerGroup(
                cluster,
                worker_builder,
                name_prefix=name_prefix,
                workers_per_node=workers_per_node,
                sharding_annotations=self.sharding_annotations,
                env_vars=env_vars,
            )

        # Call some collective rpc functions in VllmGenerationWorker when initializing the vLLM engine
        # This is necessary for async engine to work
        self._post_init()

        # dp_openai_server_base_urls is only returned by Async vLLM flow when http server is active
        self.dp_openai_server_base_urls = self._report_dp_openai_server_base_urls()

        # Number of data parallel groups is the number of tied worker groups
        assert self.dp_size == self.worker_group.dp_size, (
            f"Data parallel size mismatch. Expected {self.dp_size}, got {self.worker_group.dp_size}"
        )

        # Used to track the round-robin selection of worker groups for generate_async
        self.current_generate_dp_shard_idx = 0

        # Save the device uuids for the workers
        self.device_uuids = self._report_device_id()

    def _get_tied_worker_bundle_indices(
        self, cluster: RayVirtualCluster
    ) -> list[tuple[int, list[int]]]:
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
            ) -> dict[str, list[int]]:
                # Retrieve mapping from node ID to bundle indices from a placement group.
                try:
                    pg_table = ray.util.placement_group_table(pg)
                    bundle_to_node = pg_table["bundles_to_node_id"]
                except Exception as e:
                    raise RuntimeError(
                        "Failed to retrieve bundle/node mapping from placement group"
                    ) from e

                node_bundles: dict[str, list[int]] = defaultdict(list)
                for bundle_idx, node_id in bundle_to_node.items():
                    node_bundles[node_id].append(bundle_idx)
                for bundles in node_bundles.values():
                    bundles.sort()
                return dict(node_bundles)

            def allocate_worker_groups(
                pg: PlacementGroup, tp_size: int, pp_size: int
            ) -> list[tuple[int, list[int]]]:
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
                flat: list[int] = []
                for nid in sorted_nodes:
                    flat.extend(node_bundles[nid])

                # Slice into groups and assign logical index
                groups: list[tuple[int, list[int]]] = []
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
        # Choose the appropriate method based on async_engine setting
        method_name = (
            "report_device_id_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "report_device_id"
        )
        # Use run_all_workers_single_data for methods that don't need data
        futures = self.worker_group.run_all_workers_single_data(
            method_name, run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"]
        )
        # Wait for all futures to complete
        results = ray.get(futures)
        return results

    def _report_dp_openai_server_base_urls(self) -> list[Optional[str]]:
        """Report the data parallel OpenAI server base URLs of vLLM workers, only populated if it is async vLLM engine and the HTTP server is active."""
        if not self.cfg["vllm_cfg"]["async_engine"]:
            return [None]  # Not applicable since this is sync

        # Use run_all_workers_single_data for methods that don't need data
        futures = self.worker_group.run_all_workers_single_data(
            "report_dp_openai_server_base_url",
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )
        # Wait for all futures to complete
        results = ray.get(futures)
        return results

    def _post_init(self):
        # Choose the appropriate method based on async_engine setting
        method_name = (
            "post_init_async" if self.cfg["vllm_cfg"]["async_engine"] else "post_init"
        )
        # Use run_all_workers_single_data for methods that don't need data
        futures = self.worker_group.run_all_workers_single_data(
            method_name, run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"]
        )
        # Wait for all futures to complete
        results = ray.get(futures)
        return results

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        """Initialize the collective communication."""
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")

        # Choose the appropriate method based on async_engine setting
        method_name = (
            "init_collective_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "init_collective"
        )

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
            method_name,
            rank_prefix=rank_prefix_list,
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
            common_kwargs={
                "ip": ip,
                "port": port,
                "world_size": world_size,
                "train_world_size": train_world_size,
            },
        )

        # this function should co-work with lm_policy, so we should wait for all futures to complete outside
        return futures

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using vLLM."""
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
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=None,  # just run on tp rank 0
            output_is_replicated=None,
            common_kwargs={"greedy": greedy},
        )

        # Get results from the workers, respecting tied worker groups (only one result per tied worker group)
        results = self.worker_group.get_all_worker_results(future_bundle)

        # Combine results from all tied worker groups
        combined: BatchedDataDict[GenerationOutputSpec] = BatchedDataDict.from_batches(
            results, pad_value_dict={"output_ids": self.cfg["_pad_token_id"]}
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
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )

        # Check if async engine is enabled
        if self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "generate_text cannot be used with async_engine=True. Use generate_text_async instead."
            )

        # Shard the data across the tied worker groups
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict] = data.shard_by_batch_size(
            dp_size, allow_uneven_shards=True
        )
        future_bundle = self.worker_group.run_all_workers_sharded_data(
            "generate_text",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=None,  # just run on tp rank 0
            output_is_replicated=None,
            common_kwargs={"greedy": greedy},
        )

        # Get results from the workers, respecting tied worker groups (only one result per tied worker group)
        results = self.worker_group.get_all_worker_results(future_bundle)

        # Combine results from all tied worker groups
        combined: BatchedDataDict[GenerationOutputSpec] = BatchedDataDict.from_batches(
            results, pad_value_dict={"output_ids": self.cfg["_pad_token_id"]}
        )

        # Verify the output has all required fields
        required_keys = ["texts"]
        missing_keys = [key for key in required_keys if key not in combined]
        if missing_keys:
            raise ValueError(
                f"Missing required keys for GenerationOutputSpec: {missing_keys}"
            )

        return combined

    async def _async_generate_base(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        method_name: str,
        data_validation_fn,
        greedy: bool = False,
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Base async generation method that handles common worker management logic.

        Args:
            data: Input data for generation
            method_name: Name of the worker method to call ('generate_async' or 'generate_text_async')
            data_validation_fn: Function to validate input data
            greedy: Whether to use greedy decoding

        Yields:
            Tuple of (original_index, BatchedDataDict containing generation result)
        """
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                f"{method_name} can only be used when async_engine is enabled in vLLM config."
            )

        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )

        # Validate input data and handle empty case
        if not data_validation_fn(data):
            return

        # Determine the leader worker for the current data parallel shard
        leader_worker_idx = self.worker_group.get_dp_leader_worker_idx(
            self.current_generate_dp_shard_idx
        )

        # Run the async method on the selected leader worker
        worker_gen_proxy = self.worker_group.run_single_worker_single_data(
            method_name=method_name,
            worker_idx=leader_worker_idx,
            data=data,
            greedy=greedy,
        )

        # Increment the round-robin worker group index
        self.current_generate_dp_shard_idx += 1
        self.current_generate_dp_shard_idx %= self.worker_group.dp_size

        # Create a queue to collect sample results from the worker as they complete
        result_queue = asyncio.Queue()
        finished = False

        async def consume_worker_generator(worker_idx, worker_gen):
            """Consume a single worker generator and put sample results in the queue."""
            nonlocal finished
            worker_name = f"Worker-{worker_idx}"
            try:
                async for sample_result_ref in worker_gen:
                    sample_result = await sample_result_ref
                    # sample_result is a tuple: (original_idx, BatchedDataDict)
                    # Tag the result with worker index for downstream attribution
                    original_idx, result_batch = sample_result
                    # Use a length-one list so BatchedDataDict.from_batches can merge without shape errors
                    result_batch["gen_leader_worker_idx"] = [int(worker_idx)]
                    sample_result = (original_idx, result_batch)
                    await result_queue.put(("sample", sample_result))
            except Exception as e:
                # Log the error before putting it in the queue for better debugging
                import traceback

                print(f"Exception in worker {worker_name}")
                traceback.print_exc()
                await result_queue.put(("error", e))
            finally:
                finished = True
                await result_queue.put(("worker_done", None))

        # Start the task to consume the worker generator
        worker_task = asyncio.create_task(
            consume_worker_generator(leader_worker_idx, worker_gen_proxy)
        )

        # Yield sample results as they become available from the worker
        timeout_seconds = float(
            os.environ.get("NRL_VLLM_ASYNC_TIMEOUT_SECONDS", "600")
        )  # Default 10 minutes

        while not finished:
            try:
                msg_type, item = await asyncio.wait_for(
                    result_queue.get(), timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                print(
                    f"Timeout waiting for results after {timeout_seconds}s. Worker has not finished."
                )
                print(
                    f"For longer sequences, increase the timeout by setting: export NRL_VLLM_ASYNC_TIMEOUT_SECONDS={int(timeout_seconds * 2)}"
                )
                # Cancel the task
                if not worker_task.done():
                    worker_task.cancel()
                await asyncio.gather(worker_task, return_exceptions=True)
                raise RuntimeError(
                    f"Timeout waiting for worker results after {timeout_seconds}s. "
                    f"For longer sequences, increase timeout by setting: export NRL_VLLM_ASYNC_TIMEOUT_SECONDS={int(timeout_seconds * 2)}"
                )

            if msg_type == "sample":
                # Yield individual sample result immediately
                yield item
            elif msg_type == "error":
                # Cancel the task and propagate error
                if not worker_task.done():
                    worker_task.cancel()
                await asyncio.gather(worker_task, return_exceptions=True)
                raise item
            elif msg_type == "worker_done":
                # Worker finished, just continue the loop
                pass
            else:
                raise RuntimeError(f"Unexpected message type: {msg_type}")

        # Verify the task is actually done
        assert worker_task.done(), (
            f"Worker task {leader_worker_idx} should be done but isn't"
        )

    async def generate_text_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate text responses asynchronously, yielding results as they are ready.

        Args:
            data: BatchedDataDict containing prompts with text strings
            greedy: Whether to use greedy decoding instead of sampling

        Yields:
            Tuple of (original_index, BatchedDataDict containing single text response)
        """

        def validate_text_data(data):
            if len(data["prompts"]) == 0:
                return False  # Return False for empty case to trigger early return
            return True

        async for result in self._async_generate_base(
            data, "generate_text_async", validate_text_data, greedy
        ):
            yield result

    async def generate_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate responses asynchronously, yielding individual samples as they complete.

        This method provides per-sample streaming across all workers, yielding each
        sample result as soon as it's ready, regardless of which worker processed it.
        """

        def validate_generate_data(data):
            if "input_ids" not in data or "input_lengths" not in data:
                raise AssertionError(
                    "input_ids and input_lengths are required in data for vLLM generation"
                )
            if len(data["input_ids"]) == 0:
                return False  # Return False for empty case to trigger early return
            return True

        async for result in self._async_generate_base(
            data, "generate_async", validate_generate_data, greedy
        ):
            yield result

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Wake workers up for colocated inference."""
        # non-colocated no need to wake up
        if not self.cfg["colocated"]["enabled"]:
            return True

        try:
            # Choose the appropriate method based on async_engine setting
            method_name = (
                "wake_up_async" if self.cfg["vllm_cfg"]["async_engine"] else "wake_up"
            )
            # Use run_all_workers_single_data for methods that don't need data
            futures = self.worker_group.run_all_workers_single_data(
                method_name,
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
            # Choose the appropriate method based on setting
            # non-colocated only needs reset prefix cache, no need to sleep.
            if self.cfg["colocated"]["enabled"]:
                method_name = (
                    "sleep_async" if self.cfg["vllm_cfg"]["async_engine"] else "sleep"
                )
            else:
                method_name = (
                    "reset_prefix_cache_async"
                    if self.cfg["vllm_cfg"]["async_engine"]
                    else "reset_prefix_cache"
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

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Prepare the info for refit."""
        # Choose the appropriate method based on async_engine setting
        method_name = (
            "prepare_refit_info_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "prepare_refit_info"
        )

        # Use run_all_workers_single_data to send data to all workers
        futures = self.worker_group.run_all_workers_single_data(
            method_name,
            state_dict_info=state_dict_info,
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )

        # Wait for all futures to complete
        ray.get(futures)

    def update_weights_via_ipc_zmq(self) -> list[ray.ObjectRef]:
        """Update weights of the policy using IPC handles via ZMQ socket."""
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")

        # Choose the appropriate method based on async_engine setting
        method_name = (
            "update_weights_via_ipc_zmq_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "update_weights_via_ipc_zmq"
        )

        # Use run_all_workers_single_data since no data needs to be passed
        futures = self.worker_group.run_all_workers_single_data(
            method_name,
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )

        # this function should co-work with lm_policy, so we should wait for all futures to complete outside
        return futures

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        """Update weights of the policy using collective communication."""
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")

        # Choose the appropriate method based on async_engine setting
        method_name = (
            "update_weights_from_collective_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "update_weights_from_collective"
        )

        # Use run_all_workers_single_data for methods that don't need data
        futures = self.worker_group.run_all_workers_single_data(
            method_name,
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

    def get_vllm_logger_metrics(self) -> dict[str, Any]:
        """Collect vLLM logger metrics from vLLM workers (model-owner actors only)."""
        if not self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            return {}
        if not self.cfg["vllm_cfg"].get("async_engine", False):
            return {}

        futures: list[ray.ObjectRef] = []
        dp_indices: list[int] = []
        for dp_idx in range(self.worker_group.dp_size):
            worker_idx = self.worker_group.get_dp_leader_worker_idx(dp_idx)
            future = self.worker_group.run_single_worker_single_data(
                "get_vllm_logger_metrics",
                worker_idx=worker_idx,
            )
            futures.append(future)
            dp_indices.append(dp_idx)

        results = ray.get(futures)
        vllm_logger_metrics: dict[str, dict[int, list[Any]]] = {
            "inflight_batch_sizes": {},  # dp_idx -> list[int]
            "num_pending_samples": {},  # dp_idx -> list[int]
            "kv_cache_usage_perc": {},  # dp_idx -> list[float]
            "generation_tokens": {},  # dp_idx -> list[int]
        }

        for dp_idx, stats in zip(dp_indices, results):
            if not stats:
                continue
            inflight_batch_sizes = stats.get("inflight_batch_sizes")
            if inflight_batch_sizes:
                vllm_logger_metrics["inflight_batch_sizes"][dp_idx] = (
                    inflight_batch_sizes
                )
            num_pending_samples = stats.get("num_pending_samples")
            if num_pending_samples:
                vllm_logger_metrics["num_pending_samples"][dp_idx] = num_pending_samples
            kv_cache_usage_perc = stats.get("kv_cache_usage_perc")
            if kv_cache_usage_perc:
                vllm_logger_metrics["kv_cache_usage_perc"][dp_idx] = kv_cache_usage_perc
            generation_tokens = stats.get("generation_tokens")
            if generation_tokens:
                vllm_logger_metrics["generation_tokens"][dp_idx] = generation_tokens

        return vllm_logger_metrics

    def clear_vllm_logger_metrics(self) -> None:
        if not self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            return
        if not self.cfg["vllm_cfg"].get("async_engine", False):
            return
        futures = self.worker_group.run_all_workers_single_data(
            "clear_vllm_logger_metrics",
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )
        ray.get(futures)

    def clear_logger_metrics(self) -> None:
        """Clear logger metrics for performance reporting."""
        self.clear_vllm_logger_metrics()

    def get_logger_metrics(self) -> dict[str, Any]:
        """Get logger metrics for performance reporting."""
        return self.get_vllm_logger_metrics()

    def __del__(self) -> None:
        """Shuts down the worker groups when the object is deleted or is garbage collected.

        This is an extra safety net in case the user forgets to call shutdown() and the pointer to
        the object is lost due to leaving a function scope. It's always recommended that the
        user calls shutdown().
        """
        self.shutdown()

    def invalidate_kv_cache(self) -> bool:
        """Invalidate reusable caches in vLLM (e.g., prefix/KV cache) after weight updates.

        For async_engine, calls reset_prefix_cache_async on workers. For sync, calls reset_prefix_cache.
        Returns True if all workers report success.
        """
        try:
            method_name = (
                "reset_prefix_cache_async"
                if self.cfg["vllm_cfg"]["async_engine"]
                else "reset_prefix_cache"
            )
            futures = self.worker_group.run_all_workers_single_data(
                method_name,
                run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
            )
            results = ray.get(futures)
            return all(result for result in results if result is not None)
        except Exception as e:
            print(f"Error invalidating vLLM caches: {e}")
            return False

    @property
    def requires_kv_scale_sync(self) -> bool:
        """Check if KV cache scales should be synchronized during refit.

        Returns True if kv_cache_dtype is fp8/fp8_e4m3.
        """
        return "kv_cache_dtype" in self.cfg["vllm_cfg"] and self.cfg["vllm_cfg"][
            "kv_cache_dtype"
        ].startswith("fp8")
