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

"""Dynamo vLLM generation orchestrator for NeMo RL.

Manages the lifecycle of:
- An etcd subprocess for service discovery
- N ``python -m dynamo.vllm`` worker subprocesses (one per DP shard)
- A single ``python -m dynamo.frontend`` subprocess for HTTP routing

All generation requests go through the frontend.  Weight updates are
not supported — calling any weight-update method will ``assert False``.
"""

import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from typing import Any, Optional, Union

import numpy as np
import ray
import requests

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.virtual_cluster import (
    PY_EXECUTABLES,
    RayVirtualCluster,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_rl.utils.venvs import create_local_venv_on_each_node
from nemo_rl.models.generation.dynamo.config import DynamoVllmConfig
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)


class DynamoVllmGeneration(GenerationInterface):
    """Dynamo-backed generation for the NeMo-Gym HTTP path.

    This class is an infrastructure manager — it starts the dynamo stack
    and exposes a single frontend URL.  The ``generate()`` method is
    intentionally unsupported; NeMo-Gym sends HTTP requests to the
    frontend directly.
    """

    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: DynamoVllmConfig,
        name_prefix: str = "dynamo_vllm",
        workers_per_node: Optional[Union[int, list[int]]] = None,
    ):
        self.cfg = config
        vllm_cfg = config["vllm_cfg"]
        dynamo_cfg = config.get("dynamo_cfg", {})

        self.tp_size = vllm_cfg["tensor_parallel_size"]
        assert cluster.world_size() % self.tp_size == 0, (
            f"World size {cluster.world_size()} must be divisible by "
            f"tensor_parallel_size {self.tp_size}"
        )
        self.dp_size = cluster.world_size() // self.tp_size

        # Dynamo namespace — unique per run to avoid collisions on shared etcd.
        self._namespace = dynamo_cfg.get("namespace", "nemo_rl")
        self._router_mode = dynamo_cfg.get("router_mode", "round-robin")

        # Reserve ports.
        self._host = _get_node_ip_local()
        etcd_port = dynamo_cfg.get("etcd_port", 0)
        self._etcd_port = etcd_port if etcd_port else _get_free_port_local()
        frontend_port = dynamo_cfg.get("frontend_http_port", 0)
        self._frontend_port = frontend_port if frontend_port else _get_free_port_local()

        # Resolve the vllm venv python for subprocesses.
        # If PY_EXECUTABLES.VLLM starts with "uv", create a local venv first.
        # Otherwise (container builds), it's already a usable python path.
        vllm_exec = PY_EXECUTABLES.VLLM
        if vllm_exec.startswith("uv"):
            self._vllm_python = create_local_venv_on_each_node(
                py_executable=vllm_exec,
                venv_name="dynamo_vllm_subprocess",
            )
        else:
            self._vllm_python = vllm_exec

        # etcd peer port (cluster-internal, not used by clients but must not collide).
        etcd_peer_port = dynamo_cfg.get("etcd_peer_port", 0)
        self._etcd_peer_port = etcd_peer_port if etcd_peer_port else _get_free_port_local()

        # NATS port for the event plane.
        self._nats_port = _get_free_port_local()

        # Subprocess handles.
        self._etcd_process: Optional[subprocess.Popen] = None
        self._etcd_data_dir: Optional[str] = None
        self._nats_process: Optional[subprocess.Popen] = None
        self._frontend_process: Optional[subprocess.Popen] = None

        # ------------------------------------------------------------------
        # 1. Start etcd and NATS
        # ------------------------------------------------------------------
        self._start_etcd()
        self._start_nats()

        # ------------------------------------------------------------------
        # 2. Placement groups & worker group
        # ------------------------------------------------------------------
        needs_cross_node_parallelism = self.tp_size > cluster.num_gpus_per_node
        strategy = None if config["colocated"]["enabled"] else "PACK"

        cluster._init_placement_groups(
            strategy=strategy,
            use_unified_pg=needs_cross_node_parallelism,
        )

        self.sharding_annotations = NamedSharding(
            layout=np.arange(cluster.world_size()).reshape(self.dp_size, self.tp_size),
            names=["data_parallel", "tensor_parallel"],
        )

        worker_cls = "nemo_rl.models.generation.dynamo.dynamo_worker.DynamoVllmWorker"
        worker_builder = RayWorkerBuilder(worker_cls, config)

        env_vars = {
            "ETCD_ENDPOINTS": f"http://{self._host}:{self._etcd_port}",
            "NATS_SERVER": f"nats://{self._host}:{self._nats_port}",
            "DYN_DISCOVERY_BACKEND": "etcd",
            "DYN_NAMESPACE": self._namespace,
            "ALLOW_NONE_AUTHENTICATION": "yes",
            "DYNAMO_VLLM_PYTHON": self._vllm_python,
        }

        if self.tp_size > 1:
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
            self.worker_group = RayWorkerGroup(
                cluster,
                worker_builder,
                name_prefix=name_prefix,
                workers_per_node=workers_per_node,
                sharding_annotations=self.sharding_annotations,
                env_vars=env_vars,
            )

        # ------------------------------------------------------------------
        # 3. Start frontend (blocks until workers register)
        # ------------------------------------------------------------------
        self._start_frontend()
        self._healthcheck_frontend()

        # ------------------------------------------------------------------
        # 4. Expose URL for NeMo-Gym
        # ------------------------------------------------------------------
        self.dp_openai_server_base_urls: list[Optional[str]] = [
            f"http://{self._host}:{self._frontend_port}/v1"
        ]
        print(
            f"  [Dynamo] Ready at {self.dp_openai_server_base_urls[0]}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Placement group helpers (mirrors VllmGeneration)
    # ------------------------------------------------------------------

    def _get_tied_worker_bundle_indices(
        self, cluster: RayVirtualCluster
    ) -> list[tuple[int, list[int]]]:
        from ray.util.placement_group import PlacementGroup

        placement_groups = cluster.get_placement_groups()
        if not placement_groups:
            raise ValueError("No placement groups available in the cluster")

        tp_size = self.tp_size

        if len(placement_groups) == 1:
            pg = placement_groups[0]
            num_groups = pg.bundle_count // tp_size

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

    # ------------------------------------------------------------------
    # etcd lifecycle
    # ------------------------------------------------------------------

    def _start_etcd(self) -> None:
        if self._etcd_process is not None:
            return

        self._etcd_data_dir = tempfile.mkdtemp(prefix="nemorl_etcd_")

        cmd = [
            "etcd",
            "--listen-client-urls",
            f"http://0.0.0.0:{self._etcd_port}",
            "--advertise-client-urls",
            f"http://{self._host}:{self._etcd_port}",
            "--listen-peer-urls",
            f"http://0.0.0.0:{self._etcd_peer_port}",
            "--data-dir",
            self._etcd_data_dir,
            # Increase timeouts so etcd survives CPU contention during
            # vLLM model loading on the same node.  Keep election timeout
            # moderate so initial leader election completes quickly.
            "--heartbeat-interval",
            "500",
            "--election-timeout",
            "5000",
        ]

        env = os.environ.copy()
        env["ALLOW_NONE_AUTHENTICATION"] = "yes"

        self._etcd_process = subprocess.Popen(
            cmd,
            env=env,
        )

        # Wait for etcd to be fully ready (not just TCP-open, but serving).
        self._wait_for_etcd(timeout=30)
        print(
            f"  [Dynamo] etcd started on port {self._etcd_port} "
            f"(pid={self._etcd_process.pid})",
            flush=True,
        )

    def _stop_etcd(self) -> None:
        if self._etcd_process is not None:
            self._etcd_process.send_signal(signal.SIGTERM)
            try:
                self._etcd_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._etcd_process.kill()
            self._etcd_process = None
            print("  [Dynamo] etcd stopped", flush=True)

        if self._etcd_data_dir is not None:
            shutil.rmtree(self._etcd_data_dir, ignore_errors=True)
            self._etcd_data_dir = None

    # ------------------------------------------------------------------
    # NATS lifecycle
    # ------------------------------------------------------------------

    def _start_nats(self) -> None:
        if self._nats_process is not None:
            return

        cmd = [
            "nats-server",
            "-p",
            str(self._nats_port),
        ]

        self._nats_process = subprocess.Popen(
            cmd,
        )

        self._wait_for_port(self._nats_port, timeout=30, label="NATS")
        print(
            f"  [Dynamo] NATS started on port {self._nats_port} "
            f"(pid={self._nats_process.pid})",
            flush=True,
        )

    def _stop_nats(self) -> None:
        if self._nats_process is not None:
            self._nats_process.send_signal(signal.SIGTERM)
            try:
                self._nats_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._nats_process.kill()
            self._nats_process = None
            print("  [Dynamo] NATS stopped", flush=True)

    # ------------------------------------------------------------------
    # Frontend lifecycle
    # ------------------------------------------------------------------

    def _start_frontend(self) -> None:
        if self._frontend_process is not None:
            return

        cmd = [
            self._vllm_python,
            "-m",
            "dynamo.frontend",
            "--http-port",
            str(self._frontend_port),
            "--http-host",
            "0.0.0.0",
            "--router-mode",
            self._router_mode,
            "--discovery-backend",
            "etcd",
            "--namespace-prefix",
            self._namespace,
        ]

        env = os.environ.copy()
        env["ETCD_ENDPOINTS"] = f"http://{self._host}:{self._etcd_port}"
        env["NATS_SERVER"] = f"nats://{self._host}:{self._nats_port}"
        env["ALLOW_NONE_AUTHENTICATION"] = "yes"

        self._frontend_process = subprocess.Popen(
            cmd,
            env=env,
        )

        print(
            f"  [Dynamo] Frontend starting on http://{self._host}:{self._frontend_port} "
            f"(pid={self._frontend_process.pid}, router={self._router_mode})",
            flush=True,
        )

    def _stop_frontend(self) -> None:
        if self._frontend_process is not None:
            self._frontend_process.send_signal(signal.SIGTERM)
            try:
                self._frontend_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._frontend_process.kill()
            self._frontend_process = None
            print("  [Dynamo] Frontend stopped", flush=True)

    def _healthcheck_frontend(self, timeout: float = 300) -> None:
        """Poll ``/health`` until all workers have registered."""
        url = f"http://localhost:{self._frontend_port}/health"
        deadline = time.monotonic() + timeout
        last_error = None

        while time.monotonic() < deadline:
            if self._frontend_process is not None:
                retcode = self._frontend_process.poll()
                if retcode is not None:
                    raise RuntimeError(
                        f"Dynamo frontend exited with code {retcode}. "
                        f"Check console output above for details."
                    )
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    instances = data.get("instances", [])
                    if len(instances) >= self.dp_size:
                        print(
                            f"  [Dynamo] Frontend healthy — "
                            f"{len(instances)} worker(s) registered",
                            flush=True,
                        )
                        return
            except requests.RequestException as e:
                last_error = e
            time.sleep(2)

        raise RuntimeError(
            f"Dynamo frontend did not become healthy within {timeout}s. "
            f"Expected {self.dp_size} workers, last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _wait_for_port(port: int, timeout: float = 30, label: str = "service") -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("localhost", port), timeout=1):
                    return
            except OSError:
                time.sleep(0.5)
        raise RuntimeError(f"{label} did not start within {timeout}s on port {port}")

    def _wait_for_etcd(self, timeout: float = 30) -> None:
        """Wait until etcd is fully ready to serve requests (not just TCP-open)."""
        url = f"http://localhost:{self._etcd_port}/health"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._etcd_process is not None:
                retcode = self._etcd_process.poll()
                if retcode is not None:
                    raise RuntimeError(
                        f"etcd exited with code {retcode} before becoming healthy. "
                        f"Check console output above for details."
                    )
            try:
                resp = requests.get(url, timeout=2)
                if resp.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise RuntimeError(
            f"etcd did not become healthy within {timeout}s on port {self._etcd_port}"
        )

    # ------------------------------------------------------------------
    # GenerationInterface — supported methods
    # ------------------------------------------------------------------

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def shutdown(self) -> bool:
        try:
            self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            print(f"  [Dynamo] Warning: worker shutdown failed: {e}", flush=True)
        self._stop_frontend()
        self._stop_nats()
        self._stop_etcd()
        return True

    def __del__(self) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # GenerationInterface — unsupported methods
    # ------------------------------------------------------------------

    def generate(
        self, data: BatchedDataDict["GenerationDatumSpec"], greedy: bool = False
    ) -> BatchedDataDict["GenerationOutputSpec"]:
        assert False, (
            "DynamoVllmGeneration does not support direct generate(). "
            "Use the NeMo-Gym HTTP path instead."
        )

    def init_collective(
        self, ip: str, port: int, world_size: int, **kwargs: Any
    ) -> list[ray.ObjectRef]:
        assert False, (
            "DynamoVllmGeneration does not support weight updates via collective."
        )

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        assert False, "DynamoVllmGeneration does not support weight refit."

    def update_weights_via_ipc_zmq(self) -> list[ray.ObjectRef]:
        assert False, "DynamoVllmGeneration does not support weight updates."

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        assert False, "DynamoVllmGeneration does not support weight updates."
