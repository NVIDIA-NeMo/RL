# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import os
import warnings
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
from nemo_rl.distributed.virtual_cluster import NVLINK_DOMAIN_UNKNOWN, RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_rl.models.generation.fleet_health import (
    GenerationFleetHealth,
    HealthyShardSelector,
)
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.generation.vllm.utils import (
    aggregate_spec_decode_counters,
    compute_spec_decode_metrics,
    resolve_generation_worker_cls,
)
from nemo_rl.telemetry.instrumentation import trace_fn
from nemo_rl.telemetry.metrics import warn_once
from nemo_rl.telemetry.setup import get_telemetry_handle
from nemo_rl.telemetry.span_groups import RLSpanGroup
from nemo_rl.utils.multimodal_payload_metrics import (
    collect_multimodal_payload_metrics,
    collect_sharded_multimodal_payload_metrics,
    print_multimodal_payload_metrics,
)
from nemo_rl.weight_sync.interfaces import WeightSynchronizer
from nemo_rl.weight_sync.membership import RefitMembership

logger = logging.getLogger(__name__)


def _record_vllm_generation_metrics(
    model_name: str | None,
    data: BatchedDataDict,
    combined: BatchedDataDict,
) -> None:
    """Record vLLM token-usage metrics to nemo-lens (no-op unless exporting)."""
    telemetry = get_telemetry_handle()
    if telemetry is None or not telemetry.is_exporting:
        return
    from nemo.lens.instruments.inference import record_inference_metrics

    # Guards only the recording: this runs per generation call, so it must not
    # break generation, but a permanently dead metric should still be visible
    # once at default verbosity rather than only under debug.
    try:
        input_tokens = (
            int(data["input_lengths"].sum()) if "input_lengths" in data else None
        )
        output_tokens = (
            int(combined["generation_lengths"].sum())
            if "generation_lengths" in combined
            else None
        )
        record_inference_metrics(
            telemetry.meter,
            model=model_name or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_name="vllm",
        )
    except Exception:
        warn_once("vllm_inference_metrics", "nemo-lens: failed to record vLLM metrics")


class VllmGeneration(GenerationInterface):
    @staticmethod
    def init_cluster_placement_groups(
        cluster: RayVirtualCluster,
        config: VllmConfig,
    ) -> None:
        """Pre-initialize placement groups matching the strategy VllmGeneration expects.

        Call this *before* constructing ``VllmGeneration`` when other components
        compete for the same Ray resources and you need deterministic ordering —
        topology-constrained inference PGs should be created before unconstrained
        ones so they claim domain-aligned nodes first.

        ``VllmGeneration.__init__`` calls ``_init_placement_groups`` internally,
        but that call early-returns when PGs already exist, so calling this
        method first is safe.
        """
        tp = config["vllm_cfg"]["tensor_parallel_size"]
        pp = config["vllm_cfg"]["pipeline_parallel_size"]
        model_parallel_size = tp * pp
        colocated = config["colocated"]["enabled"]

        strategy = None if colocated else "PACK"
        needs_cross_node = model_parallel_size > cluster.num_gpus_per_node

        cluster._init_placement_groups(
            strategy=strategy,
            use_unified_pg=needs_cross_node,
        )

    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: VllmConfig,
        name_prefix: str = "vllm_policy",
        workers_per_node: Optional[Union[int, list[int]]] = None,
        defer_model_load: bool = False,
    ):
        """Initialize a vLLM policy with distributed workers.

        When defer_model_load=True, workers only reserve ports (seconds) and
        dp_openai_server_base_urls is populated immediately from reserved ports.
        Call load_and_start() later to perform heavy model loading. This enables
        overlapping vLLM model loading with NeMo Gym init.

        Args:
            cluster: Virtual cluster for worker placement
            config: VllmConfig dictionary
            name_prefix: Prefix for Ray actor names
            workers_per_node: Workers per node override
            defer_model_load: If True, defer model loading for overlapped init
        """
        # Store config
        self.cfg = config
        self._defer_model_load = defer_model_load
        self.weight_synchronizer: WeightSynchronizer | None = None
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
        top_k: int | None = self.cfg["top_k"]
        if top_k is not None and top_k != -1 and top_k < 1:
            raise ValueError(
                f"top_k valid values: i) None or -1: no filtering. ii) >= 1: top-k filtering. Got top_k={top_k}."
            )

        top_p: float = self.cfg["top_p"]
        if top_p <= 0 or top_p > 1.0:
            raise ValueError(
                f"top_p valid values: i) 1.0: no filtering. ii) (0, 1]: top-p filtering. Got top_p={top_p}."
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
        worker_cls = resolve_generation_worker_cls(worker_cls, self.cfg)
        if self.cfg["vllm_cfg"]["async_engine"]:
            worker_builder = RayWorkerBuilder(
                worker_cls, config, defer_model_load=defer_model_load
            )
        else:
            worker_builder = RayWorkerBuilder(worker_cls, config)

        # It's necessary to set env_vars here to ensure that vllm non-leader workers also have these env_vars
        env_vars = {}
        # User-supplied per-recipe env vars (e.g. vllm_cfg.env_vars in the yaml).
        # Scoped to this generation config so it does not impact other test cases.
        for k, v in self.cfg["vllm_cfg"].get("env_vars", {}).items():
            env_vars[str(k)] = str(v)
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

        # Number of data parallel groups is the number of tied worker groups
        assert self.dp_size == self.worker_group.dp_size, (
            f"Data parallel size mismatch. Expected {self.dp_size}, got {self.worker_group.dp_size}"
        )

        # Used to track the round-robin selection of worker groups for generate_async
        # when no fleet selector is attached.
        self.current_generate_dp_shard_idx = 0

        # Set by attach_fleet_health when async_rl.generation_fleet_health is enabled. While None,
        # shard selection stays health-blind, which is the historical behaviour.
        self.fleet_monitor: Optional[GenerationFleetHealth] = None
        self.fleet_selector: Optional[HealthyShardSelector] = None
        # Declared here rather than springing into existence in set_refit_membership.
        # None means "no shard has been lost", which is the state for the whole life of
        # any run that never loses one -- so absence is a real value, not a missing one,
        # and it should not be discovered with getattr at the read site.
        self._refit_membership: Optional["RefitMembership"] = None

        if defer_model_load:
            # Workers only reserved ports — collect URLs immediately and defer
            # the heavy model loading (and HTTP server start) to load_and_start().
            self.dp_openai_server_base_urls = self._collect_reserved_urls()
            self.device_uuids = None
        else:
            # Full init: call some collective rpc functions in the worker when
            # initializing the vLLM engine (necessary for async engine to work),
            # then report server URLs and device ids.
            self._post_init()
            # dp_openai_server_base_urls is only returned by the async vLLM flow
            # when the http server is active.
            self.dp_openai_server_base_urls = self._report_dp_openai_server_base_urls()
            self.device_uuids = self._report_device_id()

        self._step_metrics_snapshot: dict[str | tuple[str, int], float] | None = None

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
                pg: PlacementGroup,
                tp_size: int,
                pp_size: int,
                sorted_bundle_indices: list[int] | None = None,
                nvlink_domain_per_bundle_index: tuple[str, ...] | None = None,
            ) -> list[tuple[int, list[int]]]:
                """Partition a unified PG's bundles into model-parallel worker groups.

                Slices the flat bundle list into consecutive chunks of ``tp_size * pp_size``
                bundles. Each chunk becomes one DP replica (one vLLM engine instance).

                Args:
                    pg: The single unified placement group containing all inference bundles.
                    tp_size: Tensor-parallel degree.
                    pp_size: Pipeline-parallel degree.
                    sorted_bundle_indices: Topology-sorted bundle order from
                        ``RayVirtualCluster._sorted_bundle_indices``. When provided, bundles
                        are ordered by (NVLink domain, topo_rank, gpu_id) so consecutive
                        slices of TP*PP stay within the same NVLink domain (when the domain
                        GPU count is divisible by TP*PP). When None, bundles are sorted by
                        (node_id, bundle_idx) as a deterministic fallback.
                    nvlink_domain_per_bundle_index: Per-bundle NVLink domain from
                        ``RayVirtualCluster._nvlink_domain_per_bundle_index``. Used only
                        for logging a warning when a worker group straddles multiple
                        NVLink domains.

                Returns:
                    List of (node_idx, bundle_indices) tuples — one per DP replica.
                    ``node_idx`` is the index of the first bundle's physical node within the
                    PG's sorted unique node set.
                """
                pg_table = ray.util.placement_group_table(pg)
                bundle_to_node = pg_table["bundles_to_node_id"]

                model_parallel_size = tp_size * pp_size

                if sorted_bundle_indices is not None:
                    # Topology-aware: bundles sorted by (domain, topo_rank, gpu_id).
                    # Each model-parallel group is a consecutive slice of that list; it
                    # stays within one NVLink domain only when TP*PP divides the usable
                    # GPU count per domain in this ordering (see topology logs).
                    flat = list(sorted_bundle_indices)
                else:
                    # Fallback: sort by node ID for deterministic ordering.
                    node_bundles = get_node_bundles(pg)
                    if not node_bundles:
                        raise ValueError("Placement group contains no bundles")
                    counts = [len(b) for b in node_bundles.values()]
                    assert len(set(counts)) == 1, (
                        "All nodes must have identical bundle counts"
                    )
                    sorted_nodes = sorted(node_bundles)
                    flat = []
                    for nid in sorted_nodes:
                        flat.extend(node_bundles[nid])

                num_groups = len(flat) // model_parallel_size
                if num_groups == 0:
                    raise ValueError(
                        "Unable to allocate any worker groups with the available resources."
                    )

                unique_nodes = sorted(set(bundle_to_node.values()))
                node_idx = {nid: idx for idx, nid in enumerate(unique_nodes)}

                groups: list[tuple[int, list[int]]] = []
                for i in range(num_groups):
                    slice_ = flat[
                        i * model_parallel_size : (i + 1) * model_parallel_size
                    ]
                    if (
                        nvlink_domain_per_bundle_index is not None
                        and sorted_bundle_indices is not None
                    ):
                        domains: set[str] = set()
                        for bidx in slice_:
                            if 0 <= bidx < len(nvlink_domain_per_bundle_index):
                                d = nvlink_domain_per_bundle_index[bidx]
                                if d != NVLINK_DOMAIN_UNKNOWN:
                                    domains.add(d)
                        if len(domains) > 1:
                            logger.warning(
                                "[TOPOLOGY] Model-parallel group %s (TP*PP=%s) spans %s NVLink "
                                "domains %s; cross-domain collectives may use slower links (e.g. "
                                "IB). Prefer TP*PP that divides usable GPUs per domain, or adjust "
                                "segment/domain allocation.",
                                i,
                                model_parallel_size,
                                len(domains),
                                sorted(domains),
                            )
                    first_node = bundle_to_node[slice_[0]]
                    groups.append((node_idx[first_node], slice_))

                return groups

            tied_groups = allocate_worker_groups(
                unified_pg,
                tp_size,
                pp_size,
                sorted_bundle_indices=cluster._sorted_bundle_indices,
                nvlink_domain_per_bundle_index=cluster._nvlink_domain_per_bundle_index,
            )
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

    def _collect_reserved_urls(self) -> list[Optional[str]]:
        """Collect reserved URLs from DP leaders before model loading.

        Only called when defer_model_load=True. Workers have bound ports
        during __init__ and can report their reserved URLs immediately.
        """
        if not self.cfg["vllm_cfg"]["async_engine"]:
            return [None]

        futures = self.worker_group.run_all_workers_single_data(
            "get_reserved_url",
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )
        results = ray.get(futures)
        return results

    def load_and_start(self) -> None:
        """Load models on all workers and start HTTP servers.

        Called after a deferred init (defer_model_load=True) to perform the
        heavy model loading. Updates dp_openai_server_base_urls with the actual
        running server URLs and populates device_uuids.
        """
        # Call load_model() on all model-owner workers
        futures = self.worker_group.run_all_workers_single_data(
            "load_model",
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )
        ray.get(futures)

        # Post-init (collective rpc functions needed for async engine)
        self._post_init()

        # Refresh URLs from the actual running servers
        self.dp_openai_server_base_urls = self._report_dp_openai_server_base_urls()

        # Save device UUIDs
        self.device_uuids = self._report_device_id()

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

    def _get_raw_spec_counters(self) -> dict[str | tuple[str, int], float]:
        """Collect raw spec decode counters from workers."""
        futures = self.worker_group.run_all_workers_single_data(
            "_get_raw_spec_counters",
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )
        worker_metrics = ray.get(futures)

        # Aggregate across workers
        return aggregate_spec_decode_counters(worker_metrics)

    def snapshot_step_metrics(self) -> None:
        """Snapshot current spec decode counters to begin tracking a training step.

        Call this before generation to establish a baseline for metrics delta.

        Raises:
            RuntimeWarning: If called twice without get_step_metrics() in between.
        """
        if self._step_metrics_snapshot is not None:
            warnings.warn(
                "snapshot_step_metrics() called again without get_step_metrics(). "
                "Previous snapshot will be overwritten.",
                RuntimeWarning,
            )
        self._step_metrics_snapshot = self._get_raw_spec_counters()

    def get_step_metrics(self) -> dict[str, float]:
        """Get speculative decoding metrics delta since snapshot_step_metrics().

        Returns:
            Dictionary of delta metrics with 'vllm/' prefix.
            Returns empty dict if snapshot_step_metrics() was not called.

        Raises:
            RuntimeWarning: If called without snapshot_step_metrics() first.
        """
        if self._step_metrics_snapshot is None:
            warnings.warn(
                "get_step_metrics() called without snapshot_step_metrics(). "
                "Call snapshot_step_metrics() before generation to track metrics.",
                RuntimeWarning,
            )
            return {}

        counters_end = self._get_raw_spec_counters()
        step_metrics = compute_spec_decode_metrics(
            self._step_metrics_snapshot, counters_end
        )

        # Reset snapshot for next step
        self._step_metrics_snapshot = None

        return step_metrics

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

    def set_refit_membership(self, membership: "RefitMembership") -> None:
        """Record which shards take part in refits from now on.

        Rebuilding the communicator is not enough on its own. Every refit dispatch --
        ``update_weights_from_collective``, ``nccl_reshard_refit`` -- goes through
        ``run_all_workers_*``, which walks the whole worker group. Left alone they would
        keep calling the dead shard's Ray actor after the rebuild and fail the refit with
        RayActorError, so the run would still die, just differently.
        """
        self._refit_membership = membership

    def _refit_leader_workers(self) -> list[Any]:
        """DP leaders that should receive refit calls, in rank order.

        Falls back to every leader when no membership has been recorded, which is the
        state for the entire life of a run that never loses a shard.
        """
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")
        workers = self.worker_group.workers
        membership = self._refit_membership
        if membership is None:
            per_shard = len(workers) // self.dp_size
            return [workers[idx * per_shard] for idx in range(self.dp_size)]
        leaders = []
        for shard_idx in membership.shard_prefixes:
            leader_idx = shard_idx * membership.workers_per_shard
            if leader_idx >= len(workers):
                raise RuntimeError(
                    f"shard {shard_idx} maps to worker {leader_idx}, but the group has "
                    f"{len(workers)} workers"
                )
            leaders.append(workers[leader_idx])
        return leaders

    def rebuild_collective(
        self, membership: "RefitMembership", ip: str, port: int
    ) -> list[ray.ObjectRef]:
        """Re-init the collective over the surviving shards only.

        Deliberately not ``init_collective`` with a filter.
        ``run_all_workers_multiple_data`` walks every worker in the group, so it would
        dispatch to the shard we are rebuilding *because* it is gone -- and calling into
        a dead Ray actor is the hang this is meant to end. Here the surviving DP leaders
        are addressed directly.

        Only leaders are called: each one ``collective_rpc``s into its own TP/PP workers,
        so one Ray call per shard reaches every rank in it.
        """
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")

        method_name = (
            "init_collective_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "init_collective"
        )
        workers = self.worker_group.workers
        futures: list[ray.ObjectRef] = []
        for shard_idx, rank_prefix in membership.shard_prefixes.items():
            leader_idx = shard_idx * membership.workers_per_shard
            if leader_idx >= len(workers):
                raise RuntimeError(
                    f"shard {shard_idx} maps to worker {leader_idx}, but the group has "
                    f"{len(workers)} workers"
                )
            futures.append(
                getattr(workers[leader_idx], method_name).remote(
                    rank_prefix=rank_prefix,
                    ip=ip,
                    port=port,
                    world_size=membership.world_size,
                    train_world_size=membership.train_world_size,
                )
            )
        return futures

    @trace_fn(RLSpanGroup.GENERATION, "rl.vllm.generate")
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
        print_multimodal_payload_metrics(
            collect_sharded_multimodal_payload_metrics(
                sharded_data,
                "vllm_generation",
                enabled=bool(self.cfg.get("_debug_payload_metrics")),
            )
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

        _record_vllm_generation_metrics(self.cfg.get("model_name"), data, combined)
        return combined

    @trace_fn(RLSpanGroup.GENERATION, "rl.vllm.generate_text")
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
        print_multimodal_payload_metrics(
            collect_sharded_multimodal_payload_metrics(
                sharded_data,
                "vllm_text_generation",
                enabled=bool(self.cfg.get("_debug_payload_metrics")),
            )
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

        _record_vllm_generation_metrics(self.cfg.get("model_name"), data, combined)
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

        # VllmAsyncGenerationWorker.generate_async: one sample per call.
        assert data.size == 1, (
            f"{method_name} is restricted to handle only single samples, "
            f"but received batch_size={data.size}. Please handle batching "
            f"outside this method."
        )

        # Pick the data-parallel shard to serve this request. With a fleet selector
        # attached, a quarantined shard is skipped; without one this is the historical
        # health-blind round-robin, so an unconfigured run behaves exactly as before.
        dp_shard_idx = self._next_dp_shard_idx()
        leader_worker_idx = self.worker_group.get_dp_leader_worker_idx(dp_shard_idx)
        print_multimodal_payload_metrics(
            collect_multimodal_payload_metrics(
                data,
                "vllm_generation_async",
                enabled=bool(self.cfg.get("_debug_payload_metrics")),
            )
        )

        if self.fleet_selector is not None:
            self.fleet_selector.acquire(dp_shard_idx)
        try:
            async for result in self._generate_on_shard(
                data=data,
                method_name=method_name,
                greedy=greedy,
                dp_shard_idx=dp_shard_idx,
                leader_worker_idx=leader_worker_idx,
            ):
                yield result
        finally:
            if self.fleet_selector is not None:
                self.fleet_selector.release(dp_shard_idx)

    def attach_fleet_health(
        self,
        monitor: GenerationFleetHealth,
        selector: HealthyShardSelector,
    ) -> None:
        """Route generation through fleet health from now on.

        Args:
            monitor: Owns shard eligibility and receives observed failures.
            selector: Picks among the shards the monitor considers serving.
        """
        if monitor.shard_count != self.worker_group.dp_size:
            raise ValueError(
                f"fleet monitor tracks {monitor.shard_count} shards but the worker "
                f"group has {self.worker_group.dp_size} data-parallel shards"
            )
        self.fleet_monitor = monitor
        self.fleet_selector = selector

    def _next_dp_shard_idx(self) -> int:
        """Return the data-parallel shard that should serve the next request."""
        if self.fleet_selector is not None:
            return self.fleet_selector.next_shard()

        shard_idx = self.current_generate_dp_shard_idx
        self.current_generate_dp_shard_idx = (
            self.current_generate_dp_shard_idx + 1
        ) % self.worker_group.dp_size
        return shard_idx

    async def _generate_on_shard(
        self,
        *,
        data: BatchedDataDict[GenerationDatumSpec],
        method_name: str,
        greedy: bool,
        dp_shard_idx: int,
        leader_worker_idx: int,
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Run one generation on a chosen shard, reporting its failures to the fleet.

        A dead worker surfaces here as a Ray actor error. Reporting it is what lets the
        next request skip this shard instead of rediscovering the same corpse, and
        re-raising it as ``GenerationUnavailable`` is what tells the rollout retry policy
        the prompt is fine and worth re-dispatching.
        """
        # Local import: nemo_rl.experience pulls the rollout stack, which must not
        # become a load-time dependency of the generation backend.
        from nemo_rl.experience.failures import GenerationUnavailable

        timeout_seconds = float(
            os.environ.get("NRL_VLLM_ASYNC_TIMEOUT_SECONDS", "900")
        )  # Default 15 minutes

        # Propagate cancellation to the Ray worker and its vLLM request.
        try:
            worker_gen_proxy = self.worker_group.run_single_worker_single_data(
                method_name=method_name,
                worker_idx=leader_worker_idx,
                data=data,
                greedy=greedy,
            )

            try:
                sample_result_ref = await anext(worker_gen_proxy)
            except StopAsyncIteration:
                raise RuntimeError(
                    f"Worker produced no output for the given sample {data}."
                )

            # Materialize the result from Ray's object store. ``anext`` above
            # resolves when the worker yields, but the object bytes have not yet
            # crossed the network to the driver — this is where that happens, and
            # where a Ray deadlock / unreachable worker would manifest, hence the
            # timeout.
            try:
                sample_result = await asyncio.wait_for(
                    sample_result_ref, timeout=timeout_seconds
                )
            except asyncio.TimeoutError as error:
                ray.cancel(worker_gen_proxy)
                # Reported and typed, not a bare RuntimeError. This is the one failure
                # the fleet-health docstrings cite to justify reactive reporting -- an
                # engine that answers is_alive from a live worker and still never
                # returns a generation -- and it used to be the one case that never
                # reached the ledger, because raising inside the try meant the
                # RayError handler below could not see it. The shard then dropped back
                # to inflight=0 and became the *preferred* next pick, at 900s a visit.
                if self.fleet_monitor is not None:
                    self.fleet_monitor.report_failure(dp_shard_idx, error)
                raise GenerationUnavailable(
                    f"generation shard {dp_shard_idx} (worker {leader_worker_idx}) "
                    f"did not return within {timeout_seconds}s. For longer sequences, "
                    f"increase timeout by setting: "
                    f"export NRL_VLLM_ASYNC_TIMEOUT_SECONDS="
                    f"{int(timeout_seconds * 2)}"
                ) from error

            # sample_result is a tuple: (original_idx, BatchedDataDict).
            original_idx, result_batch = sample_result
            result_batch["gen_leader_worker_idx"] = [int(leader_worker_idx)]
            # Clears the reported-failure streak: it counts *consecutive* failures, so
            # without a success signal it is monotonic and any shard eventually reaches
            # unhealthy_threshold however healthy it is.
            if self.fleet_monitor is not None:
                self.fleet_monitor.report_success(dp_shard_idx)
            # Inside the try: main added the cancellation handler below precisely so a
            # consumer abandoning this generator mid-yield still cancels the Ray call.
            yield (original_idx, result_batch)
        except ray.exceptions.RayError as error:
            if self.fleet_monitor is not None:
                self.fleet_monitor.report_failure(dp_shard_idx, error)
            raise GenerationUnavailable(
                f"generation shard {dp_shard_idx} (worker {leader_worker_idx}) "
                f"is unavailable: {type(error).__name__}: {error}"
            ) from error
        except (asyncio.CancelledError, GeneratorExit):
            ray.cancel(worker_gen_proxy)
            raise

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
            if self.weight_synchronizer is not None:
                self.weight_synchronizer.shutdown()
            # Use the worker group's shutdown method with the worker's cleanup method
            return self.worker_group.shutdown(cleanup_method="shutdown")
        except ray.exceptions.RayActorError:
            # Workers already dead (e.g., shut down via another handle to the same actors).
            return True
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

    def update_weights_from_collective(
        self, refit_timeout_s: Optional[float] = None
    ) -> list[ray.ObjectRef]:
        """Update weights of the policy using collective communication."""
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")

        # Choose the appropriate method based on async_engine setting
        method_name = (
            "update_weights_from_collective_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "update_weights_from_collective"
        )

        # Addressed per surviving leader rather than via run_all_workers_single_data,
        # which walks the whole group: after a shard is lost that would call its dead
        # actor and fail the refit, undoing the rebuild that just happened.
        futures = [
            getattr(worker, method_name).remote(refit_timeout_s=refit_timeout_s)
            for worker in self._refit_leader_workers()
        ]

        # this function should co-work with lm_policy, so we should wait for all futures to complete outside
        return futures

    def init_nccl_reshard_comm_group(
        self,
        pp_ips: list[str],
        pp_ports: list[int],
        pp_size: int,
        train_ranks_per_stage: int,
        sub_world_size: int,
    ) -> list[ray.ObjectRef]:
        """Initialize the nccl_reshard bulk-path comm group(s) on all gen workers.

        One group per PP stage (non-PP = ``pp_size`` 1).
        """
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")

        method_name = (
            "init_nccl_reshard_comm_group_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "init_nccl_reshard_comm_group"
        )

        total_workers = len(self.worker_group.workers)
        workers_per_group = total_workers // self.dp_size
        rank_prefix_list = list(range(0, total_workers, workers_per_group))

        futures = self.worker_group.run_all_workers_multiple_data(
            method_name,
            rank_prefix=rank_prefix_list,
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
            common_kwargs={
                "pp_ips": pp_ips,
                "pp_ports": pp_ports,
                "pp_size": pp_size,
                "train_ranks_per_stage": train_ranks_per_stage,
                "sub_world_size": sub_world_size,
            },
        )
        # co-works with lm_policy; wait for all futures to complete outside
        return futures

    def prepare_nccl_reshard_refit_info(self, refit_info: dict) -> None:
        """Forward per-layer param metadata to vLLM workers for nccl_reshard refit."""
        method_name = (
            "prepare_nccl_reshard_refit_info_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "prepare_nccl_reshard_refit_info"
        )
        # Surviving leaders only; see update_weights_from_collective. This one matters
        # doubly: the plan being distributed is the *regenerated* one, sized for the
        # surviving fleet, and handing it to a shard that is not in that fleet is
        # meaningless even if its actor happened to answer.
        futures = [
            getattr(worker, method_name).remote(refit_info=refit_info)
            for worker in self._refit_leader_workers()
        ]
        ray.get(futures)

    def rebuild_nccl_reshard_comm_group(
        self,
        membership: "RefitMembership",
        pp_ips: list[str],
        pp_ports: list[int],
        pp_size: int,
        train_ranks_per_stage: int,
        sub_world_size: int,
    ) -> list[ray.ObjectRef]:
        """Re-init the bulk-path comm groups over the surviving shards only.

        The bulk groups are sized ``train_ranks_per_stage + inference_world_size``, so
        losing a shard changes their world size as well as the shared
        ``model_update_group``'s -- both families have to be rebuilt together or the two
        disagree about who is present.
        """
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")

        method_name = (
            "init_nccl_reshard_comm_group_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "init_nccl_reshard_comm_group"
        )
        workers = self.worker_group.workers
        futures: list[ray.ObjectRef] = []
        for shard_idx, rank_prefix in membership.shard_prefixes.items():
            leader = workers[shard_idx * membership.workers_per_shard]
            futures.append(
                getattr(leader, method_name).remote(
                    rank_prefix=rank_prefix,
                    pp_ips=pp_ips,
                    pp_ports=pp_ports,
                    pp_size=pp_size,
                    train_ranks_per_stage=train_ranks_per_stage,
                    sub_world_size=sub_world_size,
                )
            )
        return futures

    def nccl_reshard_refit(
        self, refit_timeout_s: Optional[float] = None
    ) -> list[ray.ObjectRef]:
        """Receive weights from training workers via nccl_reshard (xferdtensor)."""
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")

        method_name = (
            "nccl_reshard_refit_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "nccl_reshard_refit"
        )
        # Surviving leaders only; see update_weights_from_collective.
        return [
            getattr(worker, method_name).remote(refit_timeout_s=refit_timeout_s)
            for worker in self._refit_leader_workers()
        ]

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

    def pause_generation_for_refit(self, *, clear_cache: bool) -> bool:
        """Pause every async vLLM engine while preserving in-flight requests."""
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError("pause_generation_for_refit requires async_engine=True")
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")

        futures = self.worker_group.run_all_workers_single_data(
            "pause_generation_async",
            clear_cache=clear_cache,
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )
        if not all(ray.get(futures)):
            raise RuntimeError("Failed to pause every async vLLM engine")
        return True

    def resume_generation_after_refit(self) -> bool:
        """Resume every async vLLM engine paused for refit."""
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "resume_generation_after_refit requires async_engine=True"
            )
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group is not initialized")

        futures = self.worker_group.run_all_workers_single_data(
            "resume_generation_async",
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )
        if not all(ray.get(futures)):
            raise RuntimeError("Failed to resume every async vLLM engine")
        return True

    @property
    def requires_kv_scale_sync(self) -> bool:
        """Check if KV cache scales should be synchronized during refit.

        Returns True if kv_cache_dtype is fp8/fp8_e4m3.
        """
        return "kv_cache_dtype" in self.cfg["vllm_cfg"] and self.cfg["vllm_cfg"][
            "kv_cache_dtype"
        ].startswith("fp8")
