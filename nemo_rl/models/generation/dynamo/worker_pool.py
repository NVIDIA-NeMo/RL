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

"""Fixed Ray-managed pool of Dynamo vLLM subprocess owners."""

import os
from typing import Any

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.dynamo.dynamo_worker import (
    DynamoGpuReservation,
    DynamoVllmWorker,
)
from nemo_rl.utils.venvs import create_local_venv_on_each_node

_WORKER_FQN = "nemo_rl.models.generation.dynamo.dynamo_worker.DynamoVllmWorker"


def _system_port_for_group(base: int, group_index: int) -> int:
    port = base + group_index
    if port > 65535:
        raise ValueError(f"Computed DYN_SYSTEM_PORT {port} exceeds 65535.")
    return port


class FixedDynamoWorkerPool:
    """Reserve inference GPUs and launch one worker per model-parallel group."""

    def __init__(
        self,
        *,
        cluster: RayVirtualCluster,
        config: dict[str, Any],
        namespace: str,
        engine_world_size: int,
        system_port_base: int,
        manager_env: dict[str, str],
        startup_timeout_s: float,
    ) -> None:
        self._cluster = cluster
        self._config = config
        self._namespace = namespace
        self._engine_world_size = engine_world_size
        self._system_port_base = system_port_base
        self._manager_env = manager_env
        self._startup_timeout_s = startup_timeout_s
        self._workers: list[ray.actor.ActorHandle] = []
        self._reservations: list[ray.actor.ActorHandle] = []
        self._cleanup_reservations: list[ray.actor.ActorHandle] = []
        self._reservation_metadata: list[dict[str, Any]] = []
        self._metadata: list[dict[str, Any]] = []

    @property
    def size(self) -> int:
        return len(self._workers)

    def start(self) -> None:
        if self._workers:
            raise RuntimeError("Managed Dynamo worker pool is already started.")
        placement_groups = self._cluster.get_placement_groups()
        python_env = get_actor_python_env(_WORKER_FQN)
        if python_env.startswith("uv"):
            python_env = create_local_venv_on_each_node(
                py_executable=python_env,
                venv_name=_WORKER_FQN,
            )
        actor_venv = os.path.dirname(os.path.dirname(python_env))
        runtime_env = {
            "py_executable": python_env,
            "env_vars": {
                **os.environ,
                "VIRTUAL_ENV": actor_venv,
                "UV_PROJECT_ENVIRONMENT": actor_venv,
            },
        }

        group_index = 0
        for pg_index, placement_group in enumerate(placement_groups):
            bundle_count = placement_group.bundle_count
            if bundle_count % self._engine_world_size != 0:
                raise ValueError(
                    f"Inference placement group {pg_index} has {bundle_count} GPU "
                    f"bundles, which is not divisible by engine_world_size="
                    f"{self._engine_world_size}."
                )
            for start in range(0, bundle_count, self._engine_world_size):
                bundle_indices = list(range(start, start + self._engine_world_size))
                reservation_handles = []
                for bundle_index in bundle_indices:
                    strategy = PlacementGroupSchedulingStrategy(
                        placement_group=placement_group,
                        placement_group_bundle_index=bundle_index,
                        placement_group_capture_child_tasks=True,
                    )
                    reservation_handles.append(
                        DynamoGpuReservation.options(
                            num_gpus=1,
                            runtime_env=runtime_env,
                            scheduling_strategy=strategy,
                        ).remote()
                    )
                self._reservations.extend(reservation_handles)
                self._cleanup_reservations.append(reservation_handles[0])
                reservation_metadata = ray.get(
                    [handle.metadata.remote() for handle in reservation_handles]
                )
                self._reservation_metadata.extend(reservation_metadata)
                node_ips = {item["node_ip"] for item in reservation_metadata}
                if len(node_ips) != 1:
                    raise RuntimeError(
                        "A managed Dynamo engine group spans multiple nodes. "
                        "Multi-node TP/PP is not supported in the fixed-fleet milestone."
                    )
                cuda_devices = [item["gpu_id"] for item in reservation_metadata]
                # A node may host several one-bundle placement groups. Use the
                # global engine index rather than the per-placement-group bundle
                # offset so TP1 workers never collide on DYN_SYSTEM_PORT.
                system_port = _system_port_for_group(
                    self._system_port_base, group_index
                )
                group_name = f"{self._namespace}-dynamo-vllm-{pg_index}-{group_index}"
                leader_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_bundle_index=bundle_indices[0],
                    placement_group_capture_child_tasks=True,
                )
                worker = DynamoVllmWorker.options(
                    num_gpus=0,
                    runtime_env={
                        **runtime_env,
                        "env_vars": {
                            **runtime_env["env_vars"],
                            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                        },
                    },
                    scheduling_strategy=leader_strategy,
                    name=group_name,
                ).remote(
                    self._config,
                    namespace=self._namespace,
                    group_name=group_name,
                    cuda_devices=cuda_devices,
                    system_port=system_port,
                    manager_env=self._manager_env,
                    startup_timeout_s=self._startup_timeout_s,
                    seed=pg_index * 1024 + group_index,
                )
                self._workers.append(worker)
                group_index += 1

        self._metadata = ray.get([worker.metadata.remote() for worker in self._workers])

    def refit_workers(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._metadata]

    def validate(self, expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._workers or not all(
            ray.get([w.is_alive.remote() for w in self._workers])
        ):
            raise RuntimeError("A Ray-managed Dynamo vLLM worker exited.")
        try:
            current_reservations = ray.get(
                [reservation.metadata.remote() for reservation in self._reservations]
            )
        except Exception as exc:
            raise RuntimeError(
                "A Ray-managed Dynamo GPU reservation actor exited."
            ) from exc
        if current_reservations != self._reservation_metadata:
            raise RuntimeError(
                "Ray-managed Dynamo GPU reservation membership changed: "
                f"expected={self._reservation_metadata}, current={current_reservations}."
            )
        current = ray.get([worker.metadata.remote() for worker in self._workers])
        if current != expected:
            raise RuntimeError(
                "Ray-managed Dynamo worker membership changed after NCCL collective "
                f"initialization: expected={expected}, current={current}."
            )
        return [dict(item) for item in current]

    def shutdown(self) -> None:
        needs_fallback_cleanup = False
        if self._workers:
            shutdown_refs = [worker.shutdown.remote() for worker in self._workers]
            try:
                ray.get(shutdown_refs, timeout=30)
            except Exception:
                needs_fallback_cleanup = True
        if needs_fallback_cleanup:
            cleanup_refs = [
                reservation.cleanup_process_group.remote(metadata["process_pid"])
                for reservation, metadata in zip(
                    self._cleanup_reservations, self._metadata, strict=False
                )
                if "process_pid" in metadata
            ]
            if cleanup_refs:
                try:
                    ray.get(cleanup_refs, timeout=15)
                except Exception:
                    pass
        for worker in self._workers:
            try:
                ray.kill(worker, no_restart=True)
            except Exception:
                pass
        for reservation in self._reservations:
            try:
                ray.kill(reservation, no_restart=True)
            except Exception:
                pass
        self._workers.clear()
        self._reservations.clear()
        self._cleanup_reservations.clear()
        self._reservation_metadata.clear()
        self._metadata.clear()
