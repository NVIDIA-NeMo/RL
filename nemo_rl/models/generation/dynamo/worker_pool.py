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

"""Dynamically scalable pool of DynamoVllmWorker Ray actors.

Workers are created individually via ``RayWorkerBuilder`` and auto-register
with etcd for service discovery.  No ``RayWorkerGroup`` or collective
communication is used — each worker is independent.
"""

import os
from copy import deepcopy
from typing import Any

import ray
from tqdm import tqdm

from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder
from nemo_rl.utils.venvs import create_local_venv_on_each_node

# Bundle identifier: (placement_group_index, (local_bundle_indices...))
Bundle = tuple[int, tuple[int, ...]]

_WORKER_CLS = "nemo_rl.models.generation.dynamo.dynamo_worker.DynamoVllmWorker"


class DynamoWorkerPool:
    """Manages a scalable set of DynamoVllmWorker Ray actors.

    Tracks GPU bundles (free vs. used) and provides ``add_workers`` /
    ``remove_workers`` for both initial startup and planner-driven scaling.
    """

    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: dict[str, Any],
        tp_size: int,
        base_env_vars: dict[str, str],
        name_prefix: str = "dynamo_vllm",
    ):
        self._cluster = cluster
        self._tp_size = tp_size
        self._name_prefix = name_prefix

        # Worker creation infrastructure.
        self._worker_builder = RayWorkerBuilder(_WORKER_CLS, config)
        self._num_gpus = (
            1 / cluster.max_colocated_worker_groups if cluster.use_gpus else 0
        )
        self._py_executable = self._resolve_py_executable()
        self._base_env_vars = self._build_base_env(base_env_vars)

        # Bundle tracking.
        self._all_bundles = self._compute_all_bundles(cluster)
        self._free_bundles: set[Bundle] = set(self._all_bundles)
        self._used_bundles: set[Bundle] = set()

        # Worker tracking (all three lists are kept in parallel).
        self._workers: list[ray.actor.ActorHandle] = []
        self._worker_initializers: list[ray.actor.ActorHandle] = []
        self._worker_bundles: list[Bundle] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_size(self) -> int:
        return len(self._all_bundles)

    @property
    def size(self) -> int:
        return len(self._workers)

    @property
    def free_slots(self) -> int:
        return len(self._free_bundles)

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def add_workers(self, count: int) -> int:
        """Add up to ``count`` workers.  Returns the number actually added."""
        available = sorted(self._free_bundles)
        to_add = min(count, len(available))
        if to_add == 0:
            return 0

        bundles = available[:to_add]
        self._create_workers(bundles)
        return to_add

    def remove_workers(self, count: int) -> int:
        """Remove up to ``count`` workers from the tail.  Returns the number actually removed."""
        to_remove = min(count, len(self._workers))
        if to_remove == 0:
            return 0

        for _ in range(to_remove):
            worker = self._workers.pop()
            self._worker_initializers.pop()
            bundle = self._worker_bundles.pop()

            try:
                ray.get(worker.shutdown.remote(), timeout=15)
            except Exception:
                ray.kill(worker)

            self._used_bundles.discard(bundle)
            self._free_bundles.add(bundle)

        print(
            f"  [Dynamo] Removed {to_remove} worker(s), "
            f"{len(self._workers)} remaining",
            flush=True,
        )
        return to_remove

    def scale_to(self, desired: int) -> None:
        """Scale the pool to exactly ``desired`` workers."""
        current = self.size
        if desired > current:
            added = self.add_workers(desired - current)
            if added < desired - current:
                print(
                    f"  [Dynamo] Warning: wanted {desired - current} more workers "
                    f"but only {added} bundles available",
                    flush=True,
                )
        elif desired < current:
            self.remove_workers(current - desired)

    def shutdown(self) -> None:
        """Shutdown all workers."""
        for worker in self._workers:
            try:
                ray.get(worker.shutdown.remote(), timeout=15)
            except Exception:
                ray.kill(worker)
        self._workers.clear()
        self._worker_initializers.clear()
        self._worker_bundles.clear()
        self._used_bundles.clear()
        self._free_bundles = set(self._all_bundles)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_workers(self, bundles: list[Bundle]) -> None:
        """Create workers on the given bundles (batch with progress bar)."""
        placement_groups = self._cluster.get_placement_groups()
        worker_futures: list[tuple[ray.ObjectRef, ray.actor.ActorHandle]] = []

        for bundle in bundles:
            pg_idx, local_indices = bundle
            pg = (
                placement_groups[0]
                if len(placement_groups) == 1
                else placement_groups[pg_idx]
            )
            first_bundle_idx = local_indices[0]

            runtime_env = {
                "env_vars": deepcopy(self._base_env_vars),
                "py_executable": self._py_executable,
            }
            runtime_env["env_vars"]["VIRTUAL_ENV"] = self._py_executable
            runtime_env["env_vars"]["UV_PROJECT_ENVIRONMENT"] = self._py_executable

            name = f"{self._name_prefix}-{pg_idx}-{first_bundle_idx}"

            future, initializer = self._worker_builder.create_worker_async(
                placement_group=pg,
                placement_group_bundle_index=first_bundle_idx,
                num_gpus=self._num_gpus,
                bundle_indices=(pg_idx, list(local_indices)),
                runtime_env=runtime_env,
                name=name,
            )
            worker_futures.append((future, initializer))

            self._free_bundles.discard(bundle)
            self._used_bundles.add(bundle)

        # Wait for all workers to initialize.
        refs = [f for f, _ in worker_futures]
        with tqdm(
            total=len(refs),
            desc=f"Initializing {self._name_prefix} workers",
            unit="worker",
        ) as pbar:
            remaining = list(refs)
            while remaining:
                ready, remaining = ray.wait(remaining, num_returns=1)
                pbar.update(len(ready))

        workers = ray.get(refs)
        for worker, (_, initializer) in zip(workers, worker_futures):
            worker._RAY_INITIALIZER_ACTOR_REF_TO_AVOID_GC = initializer
            self._workers.append(worker)
            self._worker_initializers.append(initializer)

        self._worker_bundles.extend(bundles)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_py_executable() -> str:
        actor_python_env = get_actor_python_env(_WORKER_CLS)
        if actor_python_env.startswith("uv"):
            return create_local_venv_on_each_node(
                py_executable=actor_python_env,
                venv_name=_WORKER_CLS,
            )
        return actor_python_env

    @staticmethod
    def _build_base_env(extra: dict[str, str]) -> dict[str, str]:
        env = dict(os.environ)
        env.update(extra)
        for key in [
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
            "RAY_CLIENT_MODE",
            "RAY_JOB_ID",
            "RAY_LD_PRELOAD",
            "RAY_RAYLET_PID",
            "RAY_USAGE_STATS_ENABLED",
        ]:
            env.pop(key, None)
        return env

    def _compute_all_bundles(self, cluster: RayVirtualCluster) -> list[Bundle]:
        if self._tp_size > 1:
            return [
                (pg_idx, tuple(indices))
                for pg_idx, indices in self._get_tied_bundle_indices(cluster)
            ]

        placement_groups = cluster.get_placement_groups()
        bundles: list[Bundle] = []
        for pg_idx, pg in enumerate(placement_groups):
            for bundle_idx in range(pg.bundle_count):
                bundles.append((pg_idx, (bundle_idx,)))
        return bundles

    def _get_tied_bundle_indices(
        self, cluster: RayVirtualCluster
    ) -> list[tuple[int, list[int]]]:
        from ray.util.placement_group import PlacementGroup

        placement_groups = cluster.get_placement_groups()
        if not placement_groups:
            raise ValueError("No placement groups available in the cluster")

        tp_size = self._tp_size

        if len(placement_groups) == 1:
            pg = placement_groups[0]

            def get_node_bundles(pg: PlacementGroup) -> dict[str, list[int]]:
                table = ray.util.placement_group_table(pg)
                node_to_bundles: dict[str, list[int]] = {}
                for idx, bundle in enumerate(table.get("bundles_to_node_id", [])):
                    node_to_bundles.setdefault(bundle, []).append(idx)
                return node_to_bundles

            node_bundles = get_node_bundles(pg)
            bundle_to_node = {}
            for node, bundles in node_bundles.items():
                for b in bundles:
                    bundle_to_node[b] = node

            sorted_nodes = sorted(node_bundles)
            node_idx = {nid: idx for idx, nid in enumerate(sorted_nodes)}
            flat: list[int] = []
            for nid in sorted_nodes:
                flat.extend(node_bundles[nid])

            num_groups = pg.bundle_count // tp_size
            groups: list[tuple[int, list[int]]] = []
            for i in range(num_groups):
                slice_ = flat[i * tp_size : (i + 1) * tp_size]
                first_node = bundle_to_node[slice_[0]]
                groups.append((node_idx[first_node], slice_))
            return groups
        else:
            tied_groups: list[tuple[int, list[int]]] = []
            for pg_idx, pg in enumerate(placement_groups):
                if pg.bundle_count == 0:
                    continue
                num_groups_in_pg = pg.bundle_count // tp_size
                for group_idx in range(num_groups_in_pg):
                    start_idx = group_idx * tp_size
                    bundle_indices = list(range(start_idx, start_idx + tp_size))
                    tied_groups.append((pg_idx, bundle_indices))
            if not tied_groups:
                raise ValueError(
                    "Unable to allocate any worker groups with the available resources."
                )
            return tied_groups
