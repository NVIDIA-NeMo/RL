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
import copy
import json
import os
import time
import uuid
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
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

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
from nemo_rl.models.generation.vllm.lfs.dispatcher import (
    CrossDpDispatcherActor,
)
from nemo_rl.models.generation.vllm.inflight_profiler import (
    inflight_interval_s,
    inflight_output_dir,
    inflight_profiling_enabled,
)
from nemo_rl.models.generation.vllm.utils import (
    aggregate_spec_decode_counters,
    compute_spec_decode_metrics,
)


def _dp_batch_stats_enabled() -> bool:
    """Whether to emit per-data-parallel-rank rollout batch/load stats.

    Controlled by the NRL_LOG_DP_BATCH_STATS env var (off by default so there
    is zero overhead in normal runs). Set to 1/true/yes to enable.
    """
    return os.environ.get("NRL_LOG_DP_BATCH_STATS", "0").strip().lower() in (
        "1",
        "true",
        "yes",
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
        self._cross_dp_dispatcher = None
        self._cross_dp_dp_selection_mode: str | None = None
        self._cross_dp_lfs_admission_fairness_interval: int | None = None
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

        # vLLM only populates DebugPerfStats when both the engine observability
        # option and this environment gate are enabled. The worker sets the
        # engine option and installs the read-only StatLogger.
        if self.cfg["vllm_cfg"].get("enable_vllm_step_trace", False):
            env_vars["VLLM_DEBUG_MFU_METRICS"] = "1"

        # Benchmark-only source-tree development can opt in to propagating the
        # driver's PYTHONPATH into the isolated NeMo worker. Ray otherwise
        # constructs the actor environment only from ``env_vars`` here, so a
        # worker may silently import the image-installed NeMo package instead
        # of the source tree used by the driver.
        if os.environ.get("NRL_VLLM_PROPAGATE_PYTHONPATH") == "1":
            pythonpath = os.environ.get("PYTHONPATH")
            if not pythonpath:
                raise RuntimeError(
                    "NRL_VLLM_PROPAGATE_PYTHONPATH=1 requires PYTHONPATH"
                )
            env_vars["PYTHONPATH"] = pythonpath
            env_vars["NRL_VLLM_PROPAGATE_PYTHONPATH"] = "1"
            if os.environ.get("NRL_FORCED_SEQUENCE_AUDIT") == "1":
                env_vars["NRL_FORCED_SEQUENCE_AUDIT"] = "1"

        # Propagate the in-flight rollout profiler toggle (and cadence) to the
        # worker actors so they start their scheduler samplers. The driver writes
        # the collected per-DP timeline to JSONL after each generate() call.
        self._inflight_profiling = inflight_profiling_enabled()
        if self._inflight_profiling:
            env_vars["NRL_PROFILE_INFLIGHT"] = "1"
            env_vars["NRL_PROFILE_INFLIGHT_INTERVAL"] = str(inflight_interval_s())

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

        # Optional middleware scheduler shared by every serialized copy of this
        # generation object.  Keeping the mutable queue in a Ray actor makes it
        # safe across repeated asyncio.run() calls and async-GRPO's OS threads.
        cross_dp_mode = os.environ.get("NRL_VLLM_CROSS_DP_SCHED", "").strip().lower()
        if cross_dp_mode in ("", "0", "off", "none"):
            cross_dp_mode = ""
        elif cross_dp_mode not in (
            "fcfs",
            "lfs",
            "predicted_lfs",
            "history_lfs",
            "oracle_probe_lfs",
            "exact_length_lpt",
        ):
            raise ValueError(
                "NRL_VLLM_CROSS_DP_SCHED must be one of "
                "off/fcfs/lfs/predicted_lfs/history_lfs/oracle_probe_lfs/"
                "exact_length_lpt, "
                f"got {cross_dp_mode!r}"
            )

        self._cross_dp_scheduler_mode = cross_dp_mode or None

        # Report devices before creating the optional dispatcher actor so a
        # constructor failure cannot orphan that actor.
        self.device_uuids = self._report_device_id()

        if self._cross_dp_scheduler_mode is not None:
            if not self.cfg["vllm_cfg"]["async_engine"]:
                raise ValueError(
                    "Cross-DP scheduling requires policy.generation.vllm_cfg."
                    "async_engine=true"
                )
            if os.environ.get("NRL_VLLM_LFS_SCHED") == "1":
                raise ValueError(
                    "NRL_VLLM_CROSS_DP_SCHED and the engine-local "
                    "NRL_VLLM_LFS_SCHED cannot be enabled together. Cross-DP "
                    "scheduling requires vanilla FCFS inside each vLLM engine."
                )

            configured_cap = self.cfg.get("vllm_kwargs", {}).get("max_num_seqs")
            max_num_seqs = os.environ.get("NRL_VLLM_MAX_NUM_SEQS", configured_cap)
            if max_num_seqs is None:
                raise ValueError(
                    "Cross-DP scheduling needs an explicit per-engine capacity. "
                    "Set NRL_VLLM_MAX_NUM_SEQS or "
                    "policy.generation.vllm_kwargs.max_num_seqs."
                )
            max_num_seqs = int(max_num_seqs)
            if max_num_seqs <= 0:
                raise ValueError(
                    f"max_num_seqs must be positive, got {max_num_seqs}"
                )
            cross_dp_lookahead = int(
                os.environ.get("NRL_VLLM_CROSS_DP_LOOKAHEAD", "0")
            )
            if cross_dp_lookahead < 0:
                raise ValueError(
                    "NRL_VLLM_CROSS_DP_LOOKAHEAD must be non-negative, "
                    f"got {cross_dp_lookahead}"
                )
            aggregate_admission_limit = self.worker_group.dp_size * (
                max_num_seqs + cross_dp_lookahead
            )
            cross_dp_global_admission_limit = int(
                os.environ.get(
                    "NRL_VLLM_CROSS_DP_GLOBAL_ADMISSION_LIMIT",
                    str(aggregate_admission_limit),
                )
            )
            if not (
                0
                < cross_dp_global_admission_limit
                <= aggregate_admission_limit
            ):
                raise ValueError(
                    "NRL_VLLM_CROSS_DP_GLOBAL_ADMISSION_LIMIT must be in "
                    f"[1, {aggregate_admission_limit}], got "
                    f"{cross_dp_global_admission_limit}"
                )
            cross_dp_dp_selection_mode = os.environ.get(
                "NRL_VLLM_CROSS_DP_DP_SELECTION_MODE",
                "inflight_count",
            ).strip().lower()
            if cross_dp_dp_selection_mode not in (
                "static_cost",
                "inflight_count",
            ):
                raise ValueError(
                    "NRL_VLLM_CROSS_DP_DP_SELECTION_MODE must be "
                    "'static_cost' or 'inflight_count', got "
                    f"{cross_dp_dp_selection_mode!r}"
                )
            self._cross_dp_dp_selection_mode = cross_dp_dp_selection_mode
            cross_dp_lfs_admission_fairness_interval = int(
                os.environ.get(
                    "NRL_VLLM_CROSS_DP_LFS_ADMISSION_FAIRNESS_INTERVAL",
                    "0",
                )
            )
            if cross_dp_lfs_admission_fairness_interval < 0:
                raise ValueError(
                    "NRL_VLLM_CROSS_DP_LFS_ADMISSION_FAIRNESS_INTERVAL "
                    "must be non-negative, got "
                    f"{cross_dp_lfs_admission_fairness_interval}"
                )
            if (
                cross_dp_lfs_admission_fairness_interval > 0
                and self._cross_dp_scheduler_mode
                not in ("lfs", "predicted_lfs", "oracle_probe_lfs")
            ):
                raise ValueError(
                    "NRL_VLLM_CROSS_DP_LFS_ADMISSION_FAIRNESS_INTERVAL "
                    "is only supported with lfs/predicted_lfs/"
                    "oracle_probe_lfs, got "
                    f"{self._cross_dp_scheduler_mode!r}"
                )
            self._cross_dp_lfs_admission_fairness_interval = (
                cross_dp_lfs_admission_fairness_interval
            )

            trace = os.environ.get("NRL_VLLM_CROSS_DP_TRACE", "0").lower() in (
                "1",
                "true",
                "yes",
            )
            initial_group_history = None
            history_path = os.environ.get("NRL_VLLM_GROUP_HISTORY_PATH")
            if (
                self._cross_dp_scheduler_mode == "history_lfs"
                and history_path
                and os.path.exists(history_path)
            ):
                with open(history_path, encoding="utf-8") as history_file:
                    history_payload = json.load(history_file)
                initial_group_history = history_payload.get(
                    "group_history", history_payload
                )
                print(
                    "[CROSS-DP-SCHED] restoring "
                    f"{len(initial_group_history)} prompt histories from "
                    f"{history_path}",
                    flush=True,
                )
            # Dispatcher and VllmGeneration exchange latency/audit timestamps
            # in the driver's monotonic clock domain. Pin the zero-CPU actor to
            # this node so a multi-node Ray cluster cannot place it elsewhere.
            self._cross_dp_dispatcher = CrossDpDispatcherActor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                )
            ).remote(
                self.worker_group.dp_size,
                max_num_seqs,
                self._cross_dp_scheduler_mode,
                trace,
                initial_group_history,
                cross_dp_lookahead,
                cross_dp_global_admission_limit,
                cross_dp_dp_selection_mode,
                cross_dp_lfs_admission_fairness_interval,
            )
            print(
                "[CROSS-DP-SCHED] enabled "
                f"mode={self._cross_dp_scheduler_mode} "
                f"dp_size={self.worker_group.dp_size} "
                f"max_num_seqs_per_dp={max_num_seqs} "
                f"lookahead_per_dp={cross_dp_lookahead} "
                "global_admission_limit="
                f"{cross_dp_global_admission_limit} "
                f"dp_selection_mode={cross_dp_dp_selection_mode} "
                "lfs_admission_fairness_interval="
                f"{cross_dp_lfs_admission_fairness_interval}",
                flush=True,
            )

        self._step_metrics_snapshot: dict[str | tuple[str, int], float] | None = None

        # Monotonic counter used to label each generate() call when
        # NRL_LOG_DP_BATCH_STATS is enabled (see _log_dp_batch_stats).
        self._dp_stats_call_idx = 0

        # Separate counter labelling each generate() call in the in-flight
        # timeline JSONL (see _dump_inflight_timeline).
        self._inflight_call_idx = 0

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

    def _log_dp_batch_stats(
        self,
        sharded_data: list[SlicedDataDict],
        results: list[BatchedDataDict[GenerationOutputSpec]],
        wall_time_s: float,
        phase: str = "generate",
    ) -> None:
        """Log per-data-parallel-rank batch sizes / token load and a global summary.

        Intended for rollout performance analysis: it surfaces how the work is
        distributed across the ``dp_size`` data-parallel vLLM replicas so DP load
        imbalance and stragglers are easy to spot. ``sharded_data[i]`` and
        ``results[i]`` both correspond to data-parallel rank ``i`` (the worker
        group preserves DP order), so the stats are attributed per DP rank.

        All lines are prefixed with ``[DP-BATCH-STATS]`` for easy grepping out of
        the driver log. Enabled via NRL_LOG_DP_BATCH_STATS (see
        :func:`_dp_batch_stats_enabled`); a no-op overhead-wise when disabled
        because it is only called from the guarded path.
        """
        call_idx = self._dp_stats_call_idx
        self._dp_stats_call_idx += 1

        dp_size = len(sharded_data)
        in_seqs = np.zeros(dp_size, dtype=np.int64)
        in_tokens = np.zeros(dp_size, dtype=np.int64)
        gen_tokens = np.zeros(dp_size, dtype=np.int64)
        gen_max = np.zeros(dp_size, dtype=np.int64)
        truncated = np.zeros(dp_size, dtype=np.int64)

        for i in range(dp_size):
            shard = sharded_data[i]
            in_seqs[i] = len(shard["input_ids"])
            if "input_lengths" in shard and len(shard["input_lengths"]) > 0:
                in_tokens[i] = int(shard["input_lengths"].sum().item())

            res = results[i] if i < len(results) else None
            if res is None:
                continue
            gen_len = res.get("generation_lengths")
            if gen_len is not None and len(gen_len) > 0:
                gen_tokens[i] = int(gen_len.sum().item())
                gen_max[i] = int(gen_len.max().item())
            trunc = res.get("truncated")
            if trunc is not None and len(trunc) > 0:
                truncated[i] = int(trunc.sum().item())

        def _summ(arr: np.ndarray) -> str:
            return (
                f"min={int(arr.min())} max={int(arr.max())} "
                f"mean={arr.mean():.1f} std={arr.std():.1f}"
            )

        mean_gen = gen_tokens.mean()
        # Imbalance > 1 means the busiest DP does more decode work than the
        # average; the step is gated by this straggler.
        imbalance = float(gen_tokens.max() / mean_gen) if mean_gen > 0 else float("nan")
        straggler_dp = int(gen_tokens.argmax())

        lines = [
            f"[DP-BATCH-STATS] call={call_idx} phase={phase} dp_size={dp_size} "
            f"wall={wall_time_s:.2f}s total_in_seqs={int(in_seqs.sum())} "
            f"total_gen_tokens={int(gen_tokens.sum())}",
            f"[DP-BATCH-STATS]   in_seqs   : {_summ(in_seqs)}",
            f"[DP-BATCH-STATS]   in_tokens : {_summ(in_tokens)}",
            f"[DP-BATCH-STATS]   gen_tokens: {_summ(gen_tokens)} "
            f"imbalance(max/mean)={imbalance:.2f} straggler_dp={straggler_dp}",
            f"[DP-BATCH-STATS]   gen_len   : {_summ(gen_max)} (max = decode steps gating each dp)",
            f"[DP-BATCH-STATS]   truncated : total={int(truncated.sum())} "
            f"max_per_dp={int(truncated.max())}",
        ]
        detail = " ".join(
            f"[dp{i}:seq={int(in_seqs[i])},in_tok={int(in_tokens[i])},"
            f"gen_tok={int(gen_tokens[i])},gen_max={int(gen_max[i])},"
            f"trunc={int(truncated[i])}]"
            for i in range(dp_size)
        )
        lines.append(f"[DP-BATCH-STATS]   dp_detail: {detail}")
        print("\n".join(lines), flush=True)

    def _dump_inflight_timeline(self) -> None:
        """Collect the per-DP in-flight batch timeline and append it to JSONL.

        Pulls each data-parallel leader's samples (captured live by its
        :class:`InflightProfiler` during the generate() that just finished) and
        writes one JSON line per sample, tagged with the generate-call index and
        DP rank, into ``NRL_PROFILE_INFLIGHT_DIR/inflight_timeline.jsonl``. Lines
        from the same ``call`` across all ``dp`` ranks share a wall-clock window,
        so grouping by ``call`` and overlaying ``dp`` gives the global view of how
        each worker's batch size / context lengths evolve over time. Best-effort:
        never raises into the generation path.
        """
        call_idx = self._inflight_call_idx
        self._inflight_call_idx += 1

        futures: list[ray.ObjectRef] = []
        dp_indices: list[int] = []
        for dp_idx in range(self.worker_group.dp_size):
            worker_idx = self.worker_group.get_dp_leader_worker_idx(dp_idx)
            futures.append(
                self.worker_group.run_single_worker_single_data(
                    "get_inflight_timeline", worker_idx=worker_idx
                )
            )
            dp_indices.append(dp_idx)

        try:
            results = ray.get(futures)
        except Exception as e:
            print(
                f"[INFLIGHT-PROFILER] call={call_idx}: failed to collect timelines: {e}",
                flush=True,
            )
            return

        out_dir = inflight_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "inflight_timeline.jsonl")

        n_samples = 0
        with open(path, "a") as f:
            for dp_idx, samples in zip(dp_indices, results):
                if not samples:
                    continue
                for sample in samples:
                    ctx_lens = sample.get("ctx_lens") or []
                    n = len(ctx_lens)
                    record = {
                        "call": call_idx,
                        "dp": dp_idx,
                        "t": sample.get("t"),
                        "batch_size": sample.get("batch_size"),
                        "waiting": sample.get("waiting"),
                        "ctx_min": min(ctx_lens) if n else 0,
                        "ctx_max": max(ctx_lens) if n else 0,
                        "ctx_mean": (sum(ctx_lens) / n) if n else 0.0,
                        "ctx_sum": sum(ctx_lens),
                        "ctx_lens": ctx_lens,
                        "prompt_lens": sample.get("prompt_lens"),
                        "gen_lens": sample.get("gen_lens"),
                    }
                    f.write(json.dumps(record) + "\n")
                    n_samples += 1

        print(
            f"[INFLIGHT-PROFILER] call={call_idx}: wrote {n_samples} samples "
            f"across {len(dp_indices)} dp ranks -> {path}",
            flush=True,
        )

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

        log_dp_stats = _dp_batch_stats_enabled()
        gen_start_time = time.perf_counter() if log_dp_stats else 0.0

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

        if log_dp_stats:
            self._log_dp_batch_stats(
                sharded_data,
                results,
                time.perf_counter() - gen_start_time,
                phase="generate",
            )

        if self._inflight_profiling:
            self._dump_inflight_timeline()

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

    @property
    def cross_dp_scheduler_enabled(self) -> bool:
        return self._cross_dp_dispatcher is not None

    @property
    def cross_dp_scheduler_mode(self) -> str | None:
        return self._cross_dp_scheduler_mode

    @property
    def cross_dp_dp_selection_mode(self) -> str | None:
        return self._cross_dp_dp_selection_mode

    @property
    def cross_dp_lfs_admission_fairness_interval(self) -> int | None:
        return self._cross_dp_lfs_admission_fairness_interval

    def _build_cross_dp_session(
        self,
        group_ids: list[str | int],
        participant_count: int,
        participant_indices: list[int] | None,
        request_costs: list[int] | None,
        predicted_group_costs: list[int] | None = None,
        designated_probe_flags: list[bool] | None = None,
        preferred_dp_indices: list[int] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if self._cross_dp_dispatcher is None:
            raise RuntimeError("Cross-DP scheduling is not enabled")
        if not group_ids:
            raise ValueError("A cross-DP session requires at least one request")
        if participant_count <= 0:
            raise ValueError("participant_count must be positive")
        if participant_indices is None:
            if participant_count != 1:
                raise ValueError(
                    "participant_indices are required when participant_count > 1"
                )
            participant_indices = [0] * len(group_ids)
        if len(participant_indices) != len(group_ids) or any(
            index < 0 or index >= participant_count for index in participant_indices
        ):
            raise ValueError(
                "participant_indices must align with group_ids and fall within "
                f"[0, {participant_count})"
            )
        if request_costs is not None:
            if len(request_costs) != len(group_ids):
                raise ValueError("request_costs must align with group_ids")
            if any(int(cost) <= 0 for cost in request_costs):
                raise ValueError("request_costs must all be positive")
            if self._cross_dp_scheduler_mode not in (
                "exact_length_lpt",
                "oracle_probe_lfs",
            ):
                raise ValueError(
                    "request_costs are benchmark-only exact-length metadata "
                    "and require NRL_VLLM_CROSS_DP_SCHED=exact_length_lpt "
                    "or oracle_probe_lfs"
                )
        elif self._cross_dp_scheduler_mode == "oracle_probe_lfs":
            raise ValueError(
                "oracle_probe_lfs requires explicit benchmark request_costs"
            )
        if predicted_group_costs is not None:
            if self._cross_dp_scheduler_mode != "predicted_lfs":
                raise ValueError(
                    "predicted_group_costs require "
                    "NRL_VLLM_CROSS_DP_SCHED=predicted_lfs"
                )
            if len(predicted_group_costs) != len(group_ids):
                raise ValueError(
                    "predicted_group_costs must align with group_ids"
                )
            if any(int(cost) <= 0 for cost in predicted_group_costs):
                raise ValueError(
                    "predicted_group_costs must all be positive"
                )
        elif self._cross_dp_scheduler_mode == "predicted_lfs":
            raise ValueError(
                "predicted_lfs requires explicit predicted_group_costs"
            )
        if preferred_dp_indices is not None:
            if self._cross_dp_scheduler_mode != "exact_length_lpt":
                raise ValueError(
                    "preferred_dp_indices require "
                    "NRL_VLLM_CROSS_DP_SCHED=exact_length_lpt"
                )
            if request_costs is None:
                raise ValueError(
                    "preferred_dp_indices require explicit exact-length "
                    "request_costs"
                )
            if len(preferred_dp_indices) != len(group_ids):
                raise ValueError(
                    "preferred_dp_indices must align with group_ids"
                )
            dp_size = int(self.dp_size)
            invalid_preferred_dp_indices = [
                value
                for value in preferred_dp_indices
                if type(value) is not int or not 0 <= value < dp_size
            ]
            if invalid_preferred_dp_indices:
                raise ValueError(
                    "preferred_dp_indices must contain only ints (not bool) "
                    f"in [0, {dp_size}); invalid values="
                    f"{invalid_preferred_dp_indices}"
                )
        if designated_probe_flags is not None:
            if len(designated_probe_flags) != len(group_ids):
                raise ValueError(
                    "designated_probe_flags must align with group_ids"
                )
            if any(type(flag) is not bool for flag in designated_probe_flags):
                raise ValueError(
                    "designated_probe_flags must contain only bool values"
                )
            designated_counts: dict[str, int] = {}
            for group_id, is_designated_probe in zip(
                group_ids, designated_probe_flags, strict=True
            ):
                normalized_group = str(group_id)
                designated_counts.setdefault(normalized_group, 0)
                designated_counts[normalized_group] += int(
                    is_designated_probe
                )
            invalid_groups = {
                group_id: count
                for group_id, count in designated_counts.items()
                if count != 1
            }
            if invalid_groups:
                raise ValueError(
                    "designated_probe_flags must select exactly one request "
                    f"per group; invalid_groups={invalid_groups}"
                )

        session_id = uuid.uuid4().hex
        request_ids = [f"{session_id}:{index}" for index in range(len(group_ids))]
        participant_ids = [
            f"{session_id}:participant:{index}"
            for index in range(participant_count)
        ]
        request_participant_ids = [
            participant_ids[index] for index in participant_indices
        ]
        normalized_groups = [str(group_id) for group_id in group_ids]
        fallback_costs = (
            [int(cost) for cost in request_costs]
            if (
                request_costs is not None
                and self._cross_dp_scheduler_mode == "exact_length_lpt"
            )
            else [max(1, int(self.cfg["max_new_tokens"]))] * len(group_ids)
        )
        request_catalog = []
        for index, (
            request_id,
            group_id,
            participant_id,
            fallback_cost,
        ) in enumerate(
            zip(
                request_ids,
                normalized_groups,
                request_participant_ids,
                fallback_costs,
                strict=True,
            )
        ):
            item = {
                "request_id": request_id,
                "group_id": group_id,
                "participant_id": participant_id,
                "fallback_cost": fallback_cost,
            }
            if self._cross_dp_scheduler_mode == "oracle_probe_lfs":
                assert request_costs is not None
                item["oracle_cost"] = int(request_costs[index])
            if self._cross_dp_scheduler_mode == "predicted_lfs":
                assert predicted_group_costs is not None
                item["predicted_cost"] = int(predicted_group_costs[index])
            if designated_probe_flags is not None:
                item["is_designated_probe"] = designated_probe_flags[index]
            if preferred_dp_indices is not None:
                item["preferred_dp_idx"] = preferred_dp_indices[index]
            request_catalog.append(item)
        session = {
            "session_id": session_id,
            "request_ids": request_ids,
            "group_ids": normalized_groups,
            "participant_ids": participant_ids,
            "request_participant_ids": request_participant_ids,
        }
        return session, request_catalog

    async def open_cross_dp_session(
        self,
        group_ids: list[str | int],
        participant_count: int = 1,
        participant_indices: list[int] | None = None,
        request_costs: list[int] | None = None,
        predicted_group_costs: list[int] | None = None,
        designated_probe_flags: list[bool] | None = None,
        preferred_dp_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Open a globally visible rollout session without blocking its event loop."""
        session, request_catalog = self._build_cross_dp_session(
            group_ids,
            participant_count,
            participant_indices,
            request_costs,
            predicted_group_costs,
            designated_probe_flags,
            preferred_dp_indices,
        )
        assert self._cross_dp_dispatcher is not None
        await self._cross_dp_dispatcher.open_session.remote(
            session["session_id"], request_catalog, session["participant_ids"]
        )
        return session

    def open_cross_dp_session_sync(
        self,
        group_ids: list[str | int],
        participant_count: int = 1,
        participant_indices: list[int] | None = None,
        request_costs: list[int] | None = None,
        predicted_group_costs: list[int] | None = None,
        designated_probe_flags: list[bool] | None = None,
        preferred_dp_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        """Synchronous session opener for async-GRPO's collector actor."""
        session, request_catalog = self._build_cross_dp_session(
            group_ids,
            participant_count,
            participant_indices,
            request_costs,
            predicted_group_costs,
            designated_probe_flags,
            preferred_dp_indices,
        )
        assert self._cross_dp_dispatcher is not None
        ray.get(
            self._cross_dp_dispatcher.open_session.remote(
                session["session_id"], request_catalog, session["participant_ids"]
            )
        )
        return session

    async def close_cross_dp_session_participant(
        self, session_id: str, participant_id: str
    ) -> None:
        if self._cross_dp_dispatcher is None:
            return
        await self._cross_dp_dispatcher.close_participant.remote(
            session_id, participant_id
        )

    def close_cross_dp_session_participant_sync(
        self, session_id: str, participant_id: str
    ) -> None:
        if self._cross_dp_dispatcher is None:
            return
        ray.get(
            self._cross_dp_dispatcher.close_participant.remote(
                session_id, participant_id
            )
        )

    def get_cross_dp_scheduler_snapshot(self) -> dict[str, Any] | None:
        if self._cross_dp_dispatcher is None:
            return None
        return ray.get(self._cross_dp_dispatcher.snapshot.remote())

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

        metadata_keys = (
            "_cross_dp_session_id",
            "_cross_dp_participant_id",
            "_cross_dp_request_id",
            "_cross_dp_group_id",
        )
        present_metadata = [key in data for key in metadata_keys]
        if any(present_metadata) and not all(present_metadata):
            raise ValueError(
                "Cross-DP request metadata must contain all of "
                f"{metadata_keys}, got keys {list(data.keys())}"
            )
        if all(present_metadata) and method_name != "generate_async":
            raise NotImplementedError(
                "Cross-DP scheduling currently supports token generation only; "
                "generate_text_async does not report generation_lengths needed "
                "to release and update the middleware lease"
            )

        def first_metadata_value(key: str) -> str:
            value = data[key]
            if len(value) != 1:
                raise ValueError(
                    "Cross-DP async dispatch requires singleton request metadata, "
                    f"but {key} has length {len(value)}"
                )
            item = value[0]
            if hasattr(item, "item"):
                item = item.item()
            return str(item)

        dispatch_request_id: str | None = None
        worker_submitted = False
        dispatch_completed = False
        remote_state_known_finished = False
        worker_task: asyncio.Task | None = None
        caught_error: BaseException | None = None
        lease_received_at_monotonic_s: float | None = None
        worker_proxy_created_at_monotonic_s: float | None = None
        cross_dp_assignment_sequence: int | None = None
        cross_dp_dp_assignment_ordinal: int | None = None
        cross_dp_session_dp_assignment_ordinal: int | None = None
        cross_dp_frontend_submission: dict[str, Any] | None = None

        try:
            if all(present_metadata):
                if self._cross_dp_dispatcher is None:
                    raise RuntimeError(
                        "Received cross-DP request metadata, but "
                        "NRL_VLLM_CROSS_DP_SCHED is disabled"
                    )
                session_id = first_metadata_value("_cross_dp_session_id")
                participant_id = first_metadata_value("_cross_dp_participant_id")
                dispatch_request_id = first_metadata_value("_cross_dp_request_id")
                group_id = first_metadata_value("_cross_dp_group_id")
                lease = await self._cross_dp_dispatcher.acquire.remote(
                    session_id,
                    participant_id,
                    dispatch_request_id,
                    group_id,
                    max(1, int(self.cfg["max_new_tokens"])),
                )
                lease_received_at_monotonic_s = time.monotonic()
                dp_shard_idx = int(lease["dp_idx"])
                cross_dp_assignment_sequence = int(
                    lease["assignment_sequence"]
                )
                cross_dp_dp_assignment_ordinal = int(
                    lease["dp_assignment_ordinal"]
                )
                cross_dp_session_dp_assignment_ordinal = int(
                    lease["session_dp_assignment_ordinal"]
                )
                cross_dp_frontend_submission = {
                    "dispatcher": self._cross_dp_dispatcher,
                    "session_id": session_id,
                    "request_id": dispatch_request_id,
                    "assignment_sequence": cross_dp_assignment_sequence,
                    "dp_assignment_ordinal": cross_dp_dp_assignment_ordinal,
                    "session_dp_assignment_ordinal": (
                        cross_dp_session_dp_assignment_ordinal
                    ),
                }
            else:
                # Preserve vanilla behavior for direct callers that did not
                # opt into a scheduling session.
                dp_shard_idx = self.current_generate_dp_shard_idx
                self.current_generate_dp_shard_idx += 1
                self.current_generate_dp_shard_idx %= self.worker_group.dp_size

            leader_worker_idx = self.worker_group.get_dp_leader_worker_idx(
                dp_shard_idx
            )
            worker_call_kwargs: dict[str, Any] = {
                "data": data,
                "greedy": greedy,
            }
            if cross_dp_frontend_submission is not None:
                worker_call_kwargs["cross_dp_frontend_submission"] = (
                    cross_dp_frontend_submission
                )
            worker_gen_proxy = self.worker_group.run_single_worker_single_data(
                method_name=method_name,
                worker_idx=leader_worker_idx,
                **worker_call_kwargs,
            )
            worker_proxy_created_at_monotonic_s = time.monotonic()
            worker_submitted = True

            result_queue = asyncio.Queue()

            async def consume_worker_generator(worker_idx, worker_gen):
                """Consume a worker generator and enqueue its single result."""
                worker_name = f"Worker-{worker_idx}"
                try:
                    async for sample_result_ref in worker_gen:
                        sample_result = await sample_result_ref
                        original_idx, result_batch = sample_result
                        result_batch["gen_leader_worker_idx"] = [int(worker_idx)]
                        result_batch["gen_dp_shard_idx"] = [int(dp_shard_idx)]
                        if lease_received_at_monotonic_s is not None:
                            result_batch[
                                "gen_cross_dp_lease_received_at_monotonic_s"
                            ] = [lease_received_at_monotonic_s]
                        if worker_proxy_created_at_monotonic_s is not None:
                            result_batch[
                                "gen_worker_proxy_created_at_monotonic_s"
                            ] = [worker_proxy_created_at_monotonic_s]
                        if cross_dp_assignment_sequence is not None:
                            result_batch[
                                "gen_cross_dp_assignment_sequence"
                            ] = [cross_dp_assignment_sequence]
                        if cross_dp_dp_assignment_ordinal is not None:
                            result_batch[
                                "gen_cross_dp_dp_assignment_ordinal"
                            ] = [cross_dp_dp_assignment_ordinal]
                        if cross_dp_session_dp_assignment_ordinal is not None:
                            result_batch[
                                "gen_cross_dp_session_dp_assignment_ordinal"
                            ] = [cross_dp_session_dp_assignment_ordinal]
                        await result_queue.put(
                            ("sample", (original_idx, result_batch))
                        )
                except Exception as error:
                    import traceback

                    print(f"Exception in worker {worker_name}")
                    traceback.print_exc()
                    await result_queue.put(("error", error))
                finally:
                    await result_queue.put(("worker_done", None))

            worker_task = asyncio.create_task(
                consume_worker_generator(leader_worker_idx, worker_gen_proxy)
            )
            timeout_seconds = float(
                os.environ.get("NRL_VLLM_ASYNC_TIMEOUT_SECONDS", "600")
            )

            while True:
                try:
                    msg_type, item = await asyncio.wait_for(
                        result_queue.get(), timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    print(
                        f"Timeout waiting for results after {timeout_seconds}s. "
                        "Worker has not finished."
                    )
                    print(
                        "For longer sequences, increase the timeout by setting: "
                        "export NRL_VLLM_ASYNC_TIMEOUT_SECONDS="
                        f"{int(timeout_seconds * 2)}"
                    )
                    if not worker_task.done():
                        worker_task.cancel()
                    await asyncio.gather(worker_task, return_exceptions=True)
                    raise RuntimeError(
                        f"Timeout waiting for worker results after {timeout_seconds}s. "
                        "For longer sequences, increase timeout by setting: "
                        "export NRL_VLLM_ASYNC_TIMEOUT_SECONDS="
                        f"{int(timeout_seconds * 2)}"
                    )

                if msg_type == "sample":
                    remote_state_known_finished = True
                    if dispatch_request_id is not None and not dispatch_completed:
                        _, result_batch = item
                        generation_length = int(
                            result_batch["generation_lengths"][0].item()
                        )
                        assert self._cross_dp_dispatcher is not None
                        # Release/refill before yielding.  Some consumers stop
                        # the async generator immediately after its first item.
                        client_reported_at_unix_s = time.time()
                        client_reported_at_monotonic_s = time.monotonic()
                        await self._cross_dp_dispatcher.complete.remote(
                            dispatch_request_id,
                            generation_length,
                            client_reported_at_unix_s,
                            client_reported_at_monotonic_s,
                            os.uname().nodename,
                        )
                        dispatch_completed = True
                    yield item
                elif msg_type == "error":
                    # A Ray async-generator exception does not prove that the
                    # remote EngineCore request stopped. It may represent an
                    # actor/transport failure, or an abort that itself failed.
                    # Keep the state unknown and fail the dispatcher globally
                    # rather than releasing and reusing a possibly live slot.
                    if not worker_task.done():
                        worker_task.cancel()
                    await asyncio.gather(worker_task, return_exceptions=True)
                    raise item
                elif msg_type == "worker_done":
                    if dispatch_request_id is not None and not dispatch_completed:
                        remote_state_known_finished = True
                        raise RuntimeError(
                            f"Worker {leader_worker_idx} returned no generation result"
                        )
                    break
                else:
                    raise RuntimeError(f"Unexpected message type: {msg_type}")

            assert worker_task.done(), (
                f"Worker task {leader_worker_idx} should be done but isn't"
            )
        except BaseException as error:
            caught_error = error
            raise
        finally:
            if worker_task is not None and not worker_task.done():
                worker_task.cancel()
                await asyncio.gather(worker_task, return_exceptions=True)
            if dispatch_request_id is not None and not dispatch_completed:
                assert self._cross_dp_dispatcher is not None
                try:
                    if worker_submitted:
                        failure_method = (
                            self._cross_dp_dispatcher.fail_terminated
                            if remote_state_known_finished
                            else self._cross_dp_dispatcher.fail_unknown
                        )
                        await failure_method.remote(
                            dispatch_request_id,
                            repr(caught_error)
                            if caught_error is not None
                            else "async generator closed before completion",
                        )
                    else:
                        await self._cross_dp_dispatcher.cancel_unsubmitted.remote(
                            dispatch_request_id
                        )
                except Exception as cleanup_error:
                    print(
                        "Failed to clean up cross-DP dispatcher lease "
                        f"{dispatch_request_id}: {cleanup_error}",
                        flush=True,
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
        result = False
        try:
            # Use the worker group's shutdown method with the worker's cleanup method
            result = self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            print(f"Error during policy shutdown: {e}")
        finally:
            dispatcher = getattr(self, "_cross_dp_dispatcher", None)
            self._cross_dp_dispatcher = None
            if dispatcher is not None:
                try:
                    ray.kill(dispatcher, no_restart=True)
                except Exception as e:
                    print(f"Error shutting down cross-DP dispatcher: {e}")
                    result = False
        return result

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
        method_name = (
            "start_gpu_profiling_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "start_gpu_profiling"
        )
        futures = self.worker_group.run_all_workers_single_data(method_name)
        ray.get(futures)

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        method_name = (
            "stop_gpu_profiling_async"
            if self.cfg["vllm_cfg"]["async_engine"]
            else "stop_gpu_profiling"
        )
        futures = self.worker_group.run_all_workers_single_data(method_name)
        ray.get(futures)

    def arm_model_step_gpu_profile(
        self,
        start_step: int,
        stop_step: int,
    ) -> dict[int, list[dict[str, Any]]]:
        """Arm an exact model-step range on every TP rank of every DP engine."""
        if not self.cfg["vllm_cfg"].get("async_engine", False):
            raise RuntimeError(
                "exact model-step GPU profiling requires async vLLM"
            )
        futures: list[ray.ObjectRef] = []
        dp_indices: list[int] = []
        for dp_idx in range(self.worker_group.dp_size):
            worker_idx = self.worker_group.get_dp_leader_worker_idx(dp_idx)
            futures.append(
                self.worker_group.run_single_worker_single_data(
                    "arm_model_step_gpu_profile_async",
                    worker_idx=worker_idx,
                    start_step=start_step,
                    stop_step=stop_step,
                )
            )
            dp_indices.append(dp_idx)
        return dict(zip(dp_indices, ray.get(futures), strict=True))

    def get_model_step_gpu_profile(
        self,
    ) -> dict[int, list[dict[str, Any]]]:
        """Require and collect exact-range proofs from all nested TP ranks."""
        if not self.cfg["vllm_cfg"].get("async_engine", False):
            raise RuntimeError(
                "exact model-step GPU profiling requires async vLLM"
            )
        futures: list[ray.ObjectRef] = []
        dp_indices: list[int] = []
        for dp_idx in range(self.worker_group.dp_size):
            worker_idx = self.worker_group.get_dp_leader_worker_idx(dp_idx)
            futures.append(
                self.worker_group.run_single_worker_single_data(
                    "get_model_step_gpu_profile_async",
                    worker_idx=worker_idx,
                )
            )
            dp_indices.append(dp_idx)
        return dict(zip(dp_indices, ray.get(futures), strict=True))

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
        vllm_logger_metrics: dict[str, dict[int, Any]] = {
            "inflight_batch_sizes": {},  # dp_idx -> list[int]
            "num_pending_samples": {},  # dp_idx -> list[int]
            "kv_cache_usage_perc": {},  # dp_idx -> list[float]
            "generation_tokens": {},  # dp_idx -> list[int]
            "num_preemptions": {},  # dp_idx -> list[int]
            "metric_samples": {},  # dp_idx -> list[dict[str, Any]]
            "metric_source_series": {},  # dp_idx -> dict[field, source]
            "metric_sampler_errors": {},  # dp_idx -> list[dict[str, Any]]
            "metric_sampler_interval_s": {},  # dp_idx -> float
            "generation_tokens_baseline": {},  # dp_idx -> int
            "num_preemptions_baseline": {},  # dp_idx -> int
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
            num_preemptions = stats.get("num_preemptions")
            if num_preemptions:
                vllm_logger_metrics["num_preemptions"][dp_idx] = num_preemptions
            metric_samples = stats.get("metric_samples")
            if metric_samples:
                vllm_logger_metrics["metric_samples"][dp_idx] = metric_samples
            vllm_logger_metrics["metric_source_series"][dp_idx] = copy.deepcopy(
                stats.get("metric_source_series")
            )
            vllm_logger_metrics["metric_sampler_errors"][dp_idx] = copy.deepcopy(
                stats.get("metric_sampler_errors", [])
            )
            vllm_logger_metrics["metric_sampler_interval_s"][dp_idx] = float(
                stats.get("metric_sampler_interval_s", 0.0)
            )
            vllm_logger_metrics["generation_tokens_baseline"][dp_idx] = int(
                stats.get("generation_tokens_baseline", 0)
            )
            vllm_logger_metrics["num_preemptions_baseline"][dp_idx] = int(
                stats.get("num_preemptions_baseline", 0)
            )

        return vllm_logger_metrics

    def get_kv_cache_shapes(self) -> list[dict[str, int]]:
        """Collect initialized KV cache capacity from every DP engine."""
        if not self.cfg["vllm_cfg"].get("async_engine", False):
            raise RuntimeError("KV cache shape collection requires async vLLM")
        futures: list[ray.ObjectRef] = []
        for dp_idx in range(self.worker_group.dp_size):
            worker_idx = self.worker_group.get_dp_leader_worker_idx(dp_idx)
            futures.append(
                self.worker_group.run_single_worker_single_data(
                    "get_kv_cache_shape",
                    worker_idx=worker_idx,
                )
            )
        return ray.get(futures)

    def clear_vllm_logger_metrics(self) -> dict[int, dict[str, Any]]:
        if not self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            return {}
        if not self.cfg["vllm_cfg"].get("async_engine", False):
            return {}

        # Record a driver-clock RPC interval for each remote anchor.  A worker's
        # local monotonic deltas can then be projected onto the driver timeline
        # without assuming CLOCK_MONOTONIC shares an epoch across physical
        # nodes.  The half round-trip is retained as alignment uncertainty.
        pending: dict[ray.ObjectRef, tuple[int, float, float]] = {}
        for dp_idx in range(self.worker_group.dp_size):
            worker_idx = self.worker_group.get_dp_leader_worker_idx(dp_idx)
            sent_monotonic_s = time.monotonic()
            sent_unix_s = time.time()
            future = self.worker_group.run_single_worker_single_data(
                "clear_vllm_logger_metrics",
                worker_idx=worker_idx,
            )
            pending[future] = (dp_idx, sent_monotonic_s, sent_unix_s)

        anchors_by_dp: dict[int, dict[str, Any]] = {}
        while pending:
            ready, _ = ray.wait(list(pending), num_returns=1)
            received_monotonic_s = time.monotonic()
            received_unix_s = time.time()
            future = ready[0]
            dp_idx, sent_monotonic_s, sent_unix_s = pending.pop(future)
            worker_anchor = ray.get(future)
            if not isinstance(worker_anchor, dict):
                raise RuntimeError(
                    f"DP {dp_idx} did not return a metric measurement anchor"
                )
            anchors_by_dp[dp_idx] = {
                "worker_anchor": worker_anchor,
                "driver_rpc_sent_monotonic_s": sent_monotonic_s,
                "driver_rpc_received_monotonic_s": received_monotonic_s,
                "driver_rpc_midpoint_monotonic_s": (
                    sent_monotonic_s + received_monotonic_s
                )
                / 2.0,
                "driver_rpc_sent_unix_s": sent_unix_s,
                "driver_rpc_received_unix_s": received_unix_s,
                "driver_rpc_round_trip_s": (
                    received_monotonic_s - sent_monotonic_s
                ),
                "driver_alignment_uncertainty_s": (
                    received_monotonic_s - sent_monotonic_s
                )
                / 2.0,
            }
        return anchors_by_dp

    def get_vllm_step_trace(self) -> dict[int, dict[str, Any]]:
        """Collect one exact step-trace snapshot from every DP engine."""
        if not self.cfg["vllm_cfg"].get("enable_vllm_step_trace", False):
            return {}
        if not self.cfg["vllm_cfg"].get("async_engine", False):
            raise RuntimeError("vLLM step tracing requires the async engine")

        futures: list[ray.ObjectRef] = []
        dp_indices: list[int] = []
        for dp_idx in range(self.worker_group.dp_size):
            worker_idx = self.worker_group.get_dp_leader_worker_idx(dp_idx)
            futures.append(
                self.worker_group.run_single_worker_single_data(
                    "get_vllm_step_trace",
                    worker_idx=worker_idx,
                )
            )
            dp_indices.append(dp_idx)
        return dict(zip(dp_indices, ray.get(futures), strict=True))

    def clear_vllm_step_trace(self) -> None:
        """Open a fresh exact step-trace window on every DP engine."""
        if not self.cfg["vllm_cfg"].get("enable_vllm_step_trace", False):
            return
        if not self.cfg["vllm_cfg"].get("async_engine", False):
            raise RuntimeError("vLLM step tracing requires the async engine")

        futures: list[ray.ObjectRef] = []
        for dp_idx in range(self.worker_group.dp_size):
            worker_idx = self.worker_group.get_dp_leader_worker_idx(dp_idx)
            futures.append(
                self.worker_group.run_single_worker_single_data(
                    "clear_vllm_step_trace",
                    worker_idx=worker_idx,
                )
            )
        ray.get(futures)

    def clear_logger_metrics_with_alignment_anchors(
        self,
    ) -> dict[int, dict[str, Any]]:
        """Clear metrics and return benchmark-only cross-host anchors."""
        metric_anchors = self.clear_vllm_logger_metrics()
        self.clear_vllm_step_trace()
        # Async engine has no blocking generate() to bracket the in-flight
        # profiler window, so we open a fresh window here (called per training
        # step before the rollout). The sync engine handles this inside generate().
        if self._inflight_profiling and self.cfg["vllm_cfg"]["async_engine"]:
            self._clear_inflight_timeline_workers()
        return metric_anchors

    def clear_logger_metrics(self) -> None:
        """Clear logger metrics for performance reporting."""
        self.clear_logger_metrics_with_alignment_anchors()

    def get_logger_metrics(self) -> dict[str, Any]:
        """Get logger metrics for performance reporting."""
        metrics = self.get_vllm_logger_metrics()
        # Async counterpart of the sync generate()-time dump: collect the window
        # opened by clear_logger_metrics() and append it to the timeline JSONL.
        if self._inflight_profiling and self.cfg["vllm_cfg"]["async_engine"]:
            self._dump_inflight_timeline()
        return metrics

    def _clear_inflight_timeline_workers(self) -> None:
        """Open a fresh in-flight sampling window on each data-parallel leader."""
        futures: list[ray.ObjectRef] = []
        for dp_idx in range(self.worker_group.dp_size):
            worker_idx = self.worker_group.get_dp_leader_worker_idx(dp_idx)
            futures.append(
                self.worker_group.run_single_worker_single_data(
                    "clear_inflight_timeline", worker_idx=worker_idx
                )
            )
        try:
            ray.get(futures)
        except Exception as e:
            print(
                f"[INFLIGHT-PROFILER] failed to clear timelines: {e}",
                flush=True,
            )

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
