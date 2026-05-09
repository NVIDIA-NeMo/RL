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
import warnings
from collections import defaultdict
from typing import (
    Any,
    AsyncGenerator,
    Callable,
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
from nemo_rl.models.generation.vllm.utils import (
    aggregate_spec_decode_counters,
    compute_spec_decode_metrics,
)


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

        # Prepare rank — over surviving workers only (RL-412: shard may have
        # been killed by the router after a fault). Dead worker indices
        # were dropped from worker_group via mark_workers_dead.
        live_indices = [
            i for i in range(len(self.worker_group.workers))
            if i not in self.worker_group.dead_indices
        ]
        total_workers = len(live_indices)
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

    def reset_collective(self) -> list[ray.ObjectRef]:
        """Tear down the weight-sync NCCL group across all vLLM workers.

        Idempotent — workers that don't currently hold a group are no-ops.
        """
        if not self.worker_group or not self.worker_group.workers:
            return []
        method_name = (
            "reset_collective_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "reset_collective"
        )
        return self.worker_group.run_all_workers_single_data(
            method_name,
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )

    def add_dp_worker(
        self,
        dp_shard_idx: Optional[int] = None,
        pre_append_hook: Optional[Callable[[], None]] = None,
    ) -> tuple[list[Any], PlacementGroup, list[int], Optional[str]]:
        """Allocate a fresh placement group + actor(s) and append as a new DP shard.

        RL-412 scale-up path. Supports TP>=1, PP>=1 — for one DP shard with
        ``model_parallel_size = TP * PP``, allocates a single PACK placement
        group with ``model_parallel_size`` GPU bundles and spawns one Ray
        actor per bundle. The first actor (local_rank=0) is the leader: it
        owns the vLLM engine, drives ``prepare_refit_info`` /
        ``report_dp_openai_server_base_url``, and is the only entry recorded
        in ``dp_leader_worker_indices``. The followers reserve their bundles
        but skip vLLM model loading (matching the initial-cohort layout in
        ``_create_workers_from_bundle_indices``).

        The KubeRay autoscaler v2 (in-tree) sees the unsatisfiable PG demand
        and provisions a new worker pod into the gpu-shard worker group.
        Once the bundle is ready, we spawn the actors via
        ``RayWorkerGroup.spawn_workers_for_shard`` /
        ``append_spawned_shard_workers`` and refresh
        ``sharding_annotations`` + ``dp_openai_server_base_urls`` so
        subsequent ``init_collective`` / ``generate`` dispatches include the
        new shard.

        EP>TP (vLLM-internal DP) is not yet supported — that mode introduces
        an extra layer of internal parallelism inside one DP shard and would
        need its own bundle/env-var layout.

        Args:
            dp_shard_idx: Logical DP shard index for the new worker
                (informational — used to populate worker metadata).
                Defaults to the post-compaction DP size.

        Returns:
            ``(actor_handles, placement_group, worker_indices, base_url)``.
            - ``actor_handles``: every actor for this shard, leader first.
            - ``placement_group``: the PG; the router holds a reference for
              cleanup via ``remove_shard``.
            - ``worker_indices``: indices into ``worker_group._workers``,
              leader first. Used by the router for ``mark_workers_dead`` /
              ``ray.kill`` on remove.
            - ``base_url``: leader's OpenAI server URL (or ``None`` for sync
              engines).
        """
        if self.ep_size > self.tp_size:
            raise NotImplementedError(
                "add_dp_worker does not yet support EP>TP (vLLM-internal DP); "
                f"got ep={self.ep_size}, tp={self.tp_size}."
            )

        # Allocate a multi-GPU placement group for the new shard. STRICT_PACK
        # so all TP*PP bundles land on a single pod (one nvl72 clique on
        # GB200/GB300). Bundle shape mirrors the initial cohort
        # (cluster._create_placement_groups_internal): one GPU + N CPU per
        # bundle, where N = max_colocated_worker_groups.
        # Note: do NOT use lifetime="detached" — caused leaked PGs in earlier
        # iterations. The router holds a reference via ShardEntry.placement_group
        # so the PG outlives this call but is still GC'd on daemon exit.
        cluster = self.worker_group.cluster
        num_cpus_per_bundle = cluster.max_colocated_worker_groups
        num_gpus_per_bundle = 1 if cluster.use_gpus else 0
        bundles = [
            {"CPU": num_cpus_per_bundle, "GPU": num_gpus_per_bundle}
            for _ in range(self.model_parallel_size)
        ]
        pg = ray.util.placement_group(bundles=bundles, strategy="STRICT_PACK")
        # Block until ready (caller invokes us via run_in_executor so the
        # asyncio loop stays responsive while autoscaler v2 schedules the pod).
        ray.get(pg.ready())

        # Register-only gating: split the spawn vs append phases so vLLM
        # init (which can take 30-60s on a fresh pod) happens BEFORE the
        # router's ``_nccl_reinit_in_progress`` gate goes up. Without this
        # split the gate covers the entire vLLM init, which spikes
        # train-side step time by ~60-90s on every kill+recover cycle
        # while train waits on /refit_ready.
        #
        # Phase A (UNGATED, slow): spawn ALL TP*PP actor handles, then call
        # ``prepare_refit_info`` and ``report_dp_openai_server_base_url``
        # on the LEADER. Setup methods serialize behind the leader's
        # ``__init__`` (vLLM model load + CUDA graph capture), so the wait
        # happens here while train is still happily refitting at the
        # smaller post-remove world.
        #
        # Phase B (GATED, fast): pre_append_hook fires (router raises
        # gate), worker_group append (instant), sharding rebuild
        # (instant). Total ~5-10s. Then control returns to the router
        # which inserts the shard table entry and runs reset_collective
        # — also under the same brief gate.
        if dp_shard_idx is None:
            dp_shard_idx = self.worker_group.dp_size
        bundle_indices_per_worker = list(range(self.model_parallel_size))

        def _cleanup_partial_spawn(
            spawned_actors: Optional[list[Any]],
        ) -> None:
            """Tear down a partially-spawned shard on add failure.

            Phase A failure cleanup (TP>1 correctness): if any setup
            step between ``spawn_workers_for_shard`` and the final
            ``append_spawned_shard_workers`` raises, we MUST ray.kill
            every spawned actor (leader AND followers) before releasing
            the PG. Relying on Ray's GC to reclaim out-of-scope handles
            is racy at TP>1 — followers reserve bundles independently
            of the leader and stay alive on the PG until the PG is
            removed. Without explicit kills, a recovered shard's
            ``init_collective`` can dispatch onto the orphaned follower
            (still in some out-of-band registry) and surface as a stale
            ``ActorDiedError`` on the next refit. ray.kill is
            idempotent against already-dead actors, so this is safe to
            run on the leader-died-before-followers-spawned case too.
            """
            from ray.util.placement_group import remove_placement_group

            if spawned_actors:
                for actor in spawned_actors:
                    try:
                        ray.kill(actor, no_restart=True)
                    except Exception:  # noqa: BLE001
                        # Already dead / in-flight kill — fine.
                        pass
            try:
                remove_placement_group(pg)
            except Exception:  # noqa: BLE001
                pass

        try:
            spawned = self.worker_group.spawn_workers_for_shard(
                placement_group=pg,
                bundle_indices=(0, bundle_indices_per_worker),
                dp_shard_idx=dp_shard_idx,
            )
        except Exception:
            # Spawn failure (rare — usually only on transient ray
            # gcs/dashboard issues, since the actors' __init__ runs
            # async). No actors to kill yet; just release the PG so
            # the autoscaler can reschedule.
            _cleanup_partial_spawn(spawned_actors=None)
            raise

        # Leader is always first in the spawned list (local_rank=0).
        leader_actor = spawned[0][0]
        # Pre-collected for the cleanup path: every spawned actor
        # handle, leader-first. Followers are spawned by
        # ``spawn_workers_for_shard`` on local_rank>=1 in the same TP
        # group; they live alongside the leader on the same PG and must
        # be ray.kill'd alongside it on Phase A failure.
        all_spawned_actors = [tup[0] for tup in spawned]

        # Phase A — drive setup methods on the LEADER handle directly.
        # These block until the leader's ``__init__`` completes; the actors
        # are NOT yet in ``worker_group._workers``, so dispatcher methods
        # (``init_collective``, ``generate``) skip them and the gate stays
        # DOWN — train continues refitting at the smaller world.
        cached = getattr(self, "_cached_state_dict_info", None)
        if cached is not None:
            prepare_method_name = (
                "prepare_refit_info_async"
                if self.cfg["vllm_cfg"]["async_engine"]
                else "prepare_refit_info"
            )
            try:
                # ``getattr(leader, name).remote(...)`` is the bare-handle
                # equivalent of ``run_single_worker_single_data`` —
                # serializes behind the leader's __init__, so this waits
                # for vLLM to finish loading.
                prepare_future = getattr(leader_actor, prepare_method_name).remote(
                    cached
                )
                ray.get(prepare_future)
            except Exception:
                # __init__ failed (CUDA wait timeout, OOM, etc).
                # ray.kill all spawned actors (leader + followers) so
                # the followers don't end up orphaned on the PG, then
                # release the PG.
                _cleanup_partial_spawn(spawned_actors=all_spawned_actors)
                raise
        else:
            print(
                "[vllm_generation.add_dp_worker] WARNING: "
                "_cached_state_dict_info is None — new shard's "
                "Worker.state_dict_info will be unset; the next refit "
                "will raise AttributeError. Ensure prepare_refit_info "
                "ran on the initial cohort before add_dp_worker.",
                flush=True,
            )

        # Pre-warm NCCL library state in the leader's engine_core process.
        # The cross-cluster ``model_update_group`` is the FIRST NCCL
        # collective the engine_core ever sees. NCCL's per-process lazy init
        # (dlopen, IB device probe, cuMemMap) runs on that first call and
        # adds 15-20s to the first ``init_collective`` after this shard
        # joins — surfacing as a step-time spike. Calling
        # ``warmup_nccl_library`` here on the leader (still outside
        # ``worker_group``, so dispatcher methods don't see it yet) primes
        # the library state during Phase A, before Phase B raises the
        # refit gate. Cost: ~1-2s, fully hidden by the existing vLLM init.
        warmup_method_name = (
            "warmup_nccl_library_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "warmup_nccl_library"
        )
        try:
            warmup_future = getattr(leader_actor, warmup_method_name).remote()
            ray.get(warmup_future)
        except Exception as _e:  # noqa: BLE001 - non-fatal optimization
            print(
                f"[vllm_generation.add_dp_worker] warmup_nccl_library "
                f"raised {type(_e).__name__}: {_e} — continuing without "
                f"pre-warm; first refit will pay full NCCL init cost",
                flush=True,
            )

        if self.cfg["vllm_cfg"]["async_engine"]:
            try:
                base_url = ray.get(
                    leader_actor.report_dp_openai_server_base_url.remote()
                )
            except Exception:
                # Leader died before reporting its OpenAI URL (e.g.
                # vLLM bind failed, port conflict, late OOM). Followers
                # are still on the PG — kill them all so the next
                # add_shard gets a clean PG slot.
                _cleanup_partial_spawn(spawned_actors=all_spawned_actors)
                raise
        else:
            base_url = None

        # Phase B — gate UP, append all spawned actors, rebuild sharding.
        # ~5-10s total. Only the leader gets recorded in
        # ``dp_leader_worker_indices``; followers exist in ``_workers`` but
        # are skipped by the ``run_rank_0_only_axes`` filter on
        # ``init_collective`` / ``generate``.
        worker_indices = self.worker_group.append_spawned_shard_workers(
            spawned=spawned,
            pre_append_hook=pre_append_hook,
        )

        # Rebuild sharding_annotations to cover the new TP/PP layout. The
        # layout has size dp_size * pp * tp; positions correspond to the
        # ALIVE workers' global indices. Dead indices are zero (compacted
        # away inside ``spawn_workers_for_shard``), so this layout has no
        # gaps.
        new_dp_size = self.worker_group.dp_size
        total_workers = new_dp_size * self.pp_size * self.tp_size
        self.sharding_annotations = NamedSharding(
            layout=np.arange(total_workers).reshape(
                new_dp_size, self.pp_size, self.tp_size
            ),
            names=["data_parallel", "pipeline_parallel", "tensor_parallel"],
        )
        self.worker_group.sharding_annotations = self.sharding_annotations
        self.dp_size = new_dp_size

        if base_url is not None:
            self.dp_openai_server_base_urls = (
                self.dp_openai_server_base_urls or []
            ) + [base_url]

        actor_handles = [tup[0] for tup in spawned]
        return actor_handles, pg, worker_indices, base_url

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

        # RL-412 scale-up: cache so ``add_dp_worker`` can replay this on a
        # newly-spawned worker. Without this, the new shard joins NCCL
        # but its ``Worker`` extension never sees state_dict_info → next
        # refit's ``update_weights_from_collective`` raises AttributeError
        # on the new worker.
        self._cached_state_dict_info = state_dict_info

        # Pre-warm NCCL library state on the initial cohort. With TP=PP=EP=1
        # disagg there is no intra-shard NCCL, so the first
        # cross-cluster ``init_collective`` would otherwise pay
        # NCCL's per-process lazy-init cost (~15-20s). Doing it here
        # — once, off the hot path — means the very first refit lands
        # on warm NCCL state and finishes in the steady-state ~2-3s
        # window. See ``vllm_backend.warmup_nccl_library`` for details.
        warmup_method_name = (
            "warmup_nccl_library_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "warmup_nccl_library"
        )
        try:
            warmup_futures = self.worker_group.run_all_workers_single_data(
                warmup_method_name,
                run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
            )
            ray.get(warmup_futures)
        except Exception as _e:  # noqa: BLE001 - non-fatal optimization
            print(
                f"[vllm_generation.prepare_refit_info] warmup_nccl_library "
                f"on initial cohort raised {type(_e).__name__}: {_e} — "
                f"continuing; first refit will pay full NCCL init cost",
                flush=True,
            )

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
