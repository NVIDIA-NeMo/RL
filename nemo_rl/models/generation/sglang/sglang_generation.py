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

import logging
import os
from typing import (
    Any,
    Optional,
    Union,
)

import numpy as np
import ray

from nemo_rl.distributed.batched_data_dict import BatchedDataDict, SlicedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.generation.sglang.config import SGLangConfig

# Global thresholds for top_k and top_p validation.
# While top-k/p are not supported, these values allow for token filtering while the logprobs should be compatible.
# See https://github.com/NVIDIA-NeMo/RL/issues/69 and https://github.com/NVIDIA-NeMo/RL/issues/237 for more details.
TOP_K_THRESHOLD = 8000  # Allow top_k >= 8000 (effectively no filtering)
TOP_P_THRESHOLD = 0.99  # Allow top_p >= 0.99 (close to 1.0)

logger = logging.getLogger(__name__)


class SGLangGeneration(GenerationInterface):
    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: SGLangConfig,
        name_prefix: str = "sglang_policy",
        workers_per_node: Optional[Union[int, list[int]]] = None,
    ):
        """Initialize a SGLang policy with distributed workers.

        SGLang server manages TP/PP internally, but we still need to:
        1. Manage data parallel distribution across multiple servers
        2. Assign GPU bundles to each server

        Each server will see logical GPUs 0-N (via CUDA_VISIBLE_DEVICES set by Ray),
        so we just need to tell SGLang how many GPUs to use (tp_size).
        """
        # Store config
        self.cfg = config
        self.sglang_cfg = config["sglang_cfg"]

        gpus_per_server = self.sglang_cfg.get("gpus_per_server", None)
        if gpus_per_server is None:
            raise ValueError("gpus_per_server must be set in SGLangConfig.sglang_cfg.")

        # Calculate number of servers based on available resources
        total_gpus = cluster.world_size()
        num_servers = total_gpus // gpus_per_server

        if num_servers == 0:
            raise ValueError(
                f"Not enough GPUs. Need at least {gpus_per_server} GPUs per server, "
                f"but only have {total_gpus} GPUs total."
            )

        if total_gpus % gpus_per_server != 0:
            logger.warning(
                f"[WARNING] Total GPUs ({total_gpus}) is not divisible by GPUs per server ({gpus_per_server}). "
                f"Will use {num_servers} servers, leaving {total_gpus % gpus_per_server} GPUs unused."
            )

        self.dp_size = num_servers
        self.gpus_per_server = gpus_per_server

        # Create sharding annotations
        # Even though SGLang manages TP internally, we include it in the layout to support
        # RayWorkerGroup's worker management (which creates one worker per GPU bundle).
        # The TP dimension becomes a "free axis" in run_all_workers_sharded_data, ensuring
        # only the primary workers (TP rank 0) are called.
        total_workers = num_servers * gpus_per_server
        self.sharding_annotations = NamedSharding(
            layout=np.arange(total_workers).reshape(num_servers, gpus_per_server),
            names=["data_parallel", "tensor_parallel"],
        )

        # Initialize placement groups
        # For SGLang, we use PACK strategy to keep bundles together
        # colocated is always at top level, not in sglang_cfg
        strategy = None if self.cfg["colocated"]["enabled"] else "PACK"
        cluster._init_placement_groups(
            strategy=strategy,
            use_unified_pg=False,  # SGLang servers don't need cross-node model parallelism
        )

        # Create worker builder for SGLangGenerationWorker
        worker_cls = (
            "nemo_rl.models.generation.sglang.sglang_worker.SGLangGenerationWorker"
        )
        worker_builder = RayWorkerBuilder(worker_cls, config)

        env_vars = {}
        global_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if global_cvd:
            # Explicitly pass CUDA_VISIBLE_DEVICES to workers via env_vars
            # This ensures all workers see the same global value, even though
            env_vars["CUDA_VISIBLE_DEVICES"] = global_cvd

        # Allocate bundles for each server
        # Each server gets consecutive bundles
        bundle_indices_list = self._allocate_bundles_for_servers(
            cluster, num_servers, gpus_per_server
        )

        # Create worker group with explicit bundle allocation
        self.worker_group = RayWorkerGroup(
            cluster,
            worker_builder,
            name_prefix=name_prefix,
            bundle_indices_list=bundle_indices_list,
            sharding_annotations=self.sharding_annotations,
            env_vars=env_vars,
        )

        # Verify data parallel size matches
        assert self.dp_size == self.worker_group.dp_size, (
            f"Data parallel size mismatch. Expected {self.dp_size}, got {self.worker_group.dp_size}"
        )

        # Used to track the round-robin selection of worker groups for generate_async
        self.current_generate_dp_shard_idx = 0

    def _allocate_bundles_for_servers(
        self,
        cluster: RayVirtualCluster,
        num_servers: int,
        gpus_per_server: int,
    ) -> list[tuple[int, list[int]]]:
        """Allocate GPU bundles to each SGLang server.

        Each server gets consecutive bundles within the same placement group (node).
        Ray will automatically set CUDA_VISIBLE_DEVICES so each server sees logical GPUs 0, 1, 2, ..., gpus_per_server-1.

        Args:
            cluster: The Ray virtual cluster
            num_servers: Total number of SGLang servers to create
            gpus_per_server: Number of GPUs each server needs

        Returns:
            List of (node_idx, [bundle_indices]) tuples for each server
        """
        placement_groups = cluster.get_placement_groups()

        if not placement_groups:
            raise ValueError("No placement groups available in the cluster")

        bundle_indices_list = []

        # Each server's bundles must be within the same placement group (node)
        server_idx = 0
        for pg_idx, pg in enumerate(placement_groups):
            if pg.bundle_count == 0:
                continue

            # Calculate how many servers can fit in this placement group
            num_servers_in_pg = pg.bundle_count // gpus_per_server

            # Allocate servers within this placement group
            for local_server_idx in range(num_servers_in_pg):
                if server_idx >= num_servers:
                    break

                # Calculate which bundles this server gets (consecutive within the PG)
                start_bundle = local_server_idx * gpus_per_server
                server_bundles = list(
                    range(start_bundle, start_bundle + gpus_per_server)
                )

                # Each server gets a tuple of (node_idx, [local_bundle_indices])
                bundle_indices_list.append((pg_idx, server_bundles))
                server_idx += 1

            if server_idx >= num_servers:
                break

        if len(bundle_indices_list) < num_servers:
            total_available = sum(
                pg.bundle_count // gpus_per_server
                for pg in placement_groups
                if pg.bundle_count > 0
            )
            raise ValueError(
                f"Not enough bundles to allocate all {num_servers} servers. "
                f"Only {total_available} servers can be allocated "
                f"(each server needs {gpus_per_server} GPUs)."
            )

        return bundle_indices_list

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        """Initialize the collective communication.

        TODO:       if weight updates via NCCL are needed in the future.
        """
        return []

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using SGLang."""
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            "input_ids and input_lengths are required in data for SGLang generation"
        )

        # Shard the data across the data parallel servers
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict] = data.shard_by_batch_size(
            dp_size, allow_uneven_shards=True
        )
        future_bundle = self.worker_group.run_all_workers_sharded_data(
            "generate",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=None,
            output_is_replicated=None,
            common_kwargs={"greedy": greedy},
        )

        # Get results from the workers
        results = self.worker_group.get_all_worker_results(future_bundle)

        # Combine results from all servers
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

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        pass

    def update_weights_via_ipc_zmq(self) -> list[ray.ObjectRef]:
        return []

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        return []

    def get_sglang_server_urls(self) -> list[str]:
        """Get base URLs of all SGLang servers.

        Returns:
            List of base URLs (e.g., ["http://localhost:30000", "http://localhost:30001"])
        """
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")

        # Get base URLs from all workers (only primary workers, TP rank 0)
        # Use run_rank_0_only_axes to only get URLs from primary workers
        futures = self.worker_group.run_all_workers_single_data(
            "get_base_url",
            run_rank_0_only_axes=["tensor_parallel"],
        )
        urls = ray.get(futures)
        # Filter out None values and return unique URLs
        return list(set(url for url in urls if url is not None))

    def get_sglang_url_to_gpu_uuids(self) -> dict[str, list[str]]:
        """Get mapping from SGLang server URL to list of GPU UUIDs it uses.

        Returns:
            Dict mapping server URL to list of GPU UUIDs
            e.g., {"http://localhost:30000": ["GPU-aaa", "GPU-bbb"], ...}
        """
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")

        # Get base URLs and GPU UUIDs from all primary workers (TP rank 0)
        futures_url = self.worker_group.run_all_workers_single_data(
            "get_base_url",
            run_rank_0_only_axes=["tensor_parallel"],
        )
        futures_uuids = self.worker_group.run_all_workers_single_data(
            "get_gpu_uuids",
            run_rank_0_only_axes=["tensor_parallel"],
        )

        urls = ray.get(futures_url)
        uuids_list = ray.get(futures_uuids)

        # Create mapping
        url_to_uuids = {}
        for url, uuids in zip(urls, uuids_list):
            if url is not None and uuids is not None:
                url_to_uuids[url] = uuids

        return url_to_uuids

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Wake workers up for colocated inference."""
        pass

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Sleep workers and reset prefix cache."""
        pass

    def shutdown(self) -> bool:
        """Shut down all SGLang workers and clean up resources."""
        try:
            # Use the worker group's shutdown method with the worker's cleanup method
            return self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            logger.error(f"Error during SGLang policy shutdown: {e}")
            return False

    def __del__(self) -> None:
        """Shuts down the worker groups when the object is deleted or is garbage collected.

        This is an extra safety net in case the user forgets to call shutdown() and the pointer to
        the object is lost due to leaving a function scope. It's always recommended that the
        user calls shutdown().
        """
        self.shutdown()

    def invalidate_kv_cache(self) -> bool:
        """Invalidate KV cache before weight updates (Megatron-style).

        This flushes the cache before weight updates to clear stale cache.
        Only primary workers (TP rank 0, model owners) will flush their cache.

        Returns:
            bool: True if all caches were flushed successfully, False otherwise
        """
        try:
            futures = self.worker_group.run_all_workers_single_data(
                "invalidate_kv_cache",
                run_rank_0_only_axes=["tensor_parallel"],
            )
            results = ray.get(futures)
            results = [r for r in results if r is not None]
            success = all(result for result in results) if results else True
            if success:
                logger.info(
                    "[sglang refit] All SGLang server caches flushed successfully"
                )
            else:
                logger.warning(
                    "[sglang refit] WARNING - Some SGLang server caches failed to flush"
                )
            return success
        except Exception as e:
            logger.error(f"[sglang refit] Error flushing SGLang caches: {e}")
            return False
