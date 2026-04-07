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

import asyncio
import atexit
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
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

# (attr_name, display_name, stop_timeout_seconds)
# Order determines shutdown sequence.
_SUBPROCESS_REGISTRY: list[tuple[str, str, int]] = [
    ("_planner_process", "planner", 15),
    ("_frontend_process", "frontend", 15),
    ("_nats_process", "NATS", 10),
    ("_etcd_process", "etcd", 10),
]


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

        # Subprocess handles (see _SUBPROCESS_REGISTRY for ordering/timeouts).
        self._etcd_process: Optional[subprocess.Popen] = None
        self._etcd_data_dir: Optional[str] = None
        self._nats_process: Optional[subprocess.Popen] = None
        self._frontend_process: Optional[subprocess.Popen] = None
        self._planner_process: Optional[subprocess.Popen] = None
        self._planner_config_file: Optional[str] = None

        # VirtualConnectorClient state.
        self._vc_stop = threading.Event()
        self._vc_thread: Optional[threading.Thread] = None

        self._start_etcd()
        self._start_nats()

        if dynamo_cfg.get("enable_planner", False):
            self._start_planner()
            self._start_virtual_connector_client()

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
            **self._dynamo_env_vars(),
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

        self._start_frontend()
        self._healthcheck_frontend()

        self.dp_openai_server_base_urls: list[Optional[str]] = [
            f"http://{self._host}:{self._frontend_port}/v1"
        ]
        print(
            f"  [Dynamo] Ready at {self.dp_openai_server_base_urls[0]}",
            flush=True,
        )

        # Ensure subprocesses are cleaned up on normal exit.
        atexit.register(self.shutdown)

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
    # Utilities
    # ------------------------------------------------------------------

    def _dynamo_env_vars(self) -> dict[str, str]:
        """Env vars required by all dynamo components."""
        return {
            "ETCD_ENDPOINTS": f"http://{self._host}:{self._etcd_port}",
            "NATS_SERVER": f"nats://{self._host}:{self._nats_port}",
            "DYN_NAMESPACE": self._namespace,
            "DYN_DISCOVERY_BACKEND": "etcd",
        }

    @staticmethod
    def _stop_subprocess(proc: subprocess.Popen, name: str, timeout: int) -> None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(f"  [Dynamo] {name} stopped", flush=True)

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
    # Subprocess lifecycle
    # ------------------------------------------------------------------

    def _start_etcd(self) -> None:
        if self._etcd_process is not None:
            return

        self._etcd_data_dir = tempfile.mkdtemp(prefix="nemorl_etcd_")

        peer_url = f"http://{self._host}:{self._etcd_peer_port}"
        env = os.environ.copy()
        env["ALLOW_NONE_AUTHENTICATION"] = "yes"
        self._etcd_process = subprocess.Popen(
            [
                "etcd",
                "--listen-client-urls", f"http://0.0.0.0:{self._etcd_port}",
                "--advertise-client-urls", f"http://{self._host}:{self._etcd_port}",
                "--listen-peer-urls", f"http://0.0.0.0:{self._etcd_peer_port}",
                "--initial-advertise-peer-urls", peer_url,
                "--initial-cluster", f"default={peer_url}",
                "--data-dir", self._etcd_data_dir,
                # Increase timeouts so etcd survives CPU contention during
                # vLLM model loading on the same node.  Keep election timeout
                # moderate so initial leader election completes quickly.
                "--heartbeat-interval", "500",
                "--election-timeout", "5000",
            ],
            env=env,
        )
        self._wait_for_etcd(timeout=30)
        print(
            f"  [Dynamo] etcd started on port {self._etcd_port} "
            f"(pid={self._etcd_process.pid})",
            flush=True,
        )

    def _start_nats(self) -> None:
        if self._nats_process is not None:
            return

        self._nats_process = subprocess.Popen(
            ["nats-server", "-p", str(self._nats_port)],
        )
        self._wait_for_port(self._nats_port, timeout=30, label="NATS")
        print(
            f"  [Dynamo] NATS started on port {self._nats_port} "
            f"(pid={self._nats_process.pid})",
            flush=True,
        )

    def _start_planner(self) -> None:
        if self._planner_process is not None:
            return

        planner_config = {
            "environment": "virtual",
            "mode": "decode",
            "backend": "vllm",
            "namespace": self._namespace,
            "model_name": self.cfg.get("model", "unknown"),
            "enable_throughput_scaling": False,
            "enable_load_scaling": True,
            "pre_deployment_sweeping_mode": "none",
            "decode_engine_num_gpu": self.tp_size,
            "ttft": 500.0,
            "itl": 50.0,
            "max_gpu_budget": 2,
            "min_endpoint": 1,
            "load_adjustment_interval": 5,
            "load_scaling_down_sensitivity": 80,
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(planner_config, tf)
            self._planner_config_file = tf.name

        env = os.environ.copy()
        env.update(self._dynamo_env_vars())
        self._planner_process = subprocess.Popen(
            [self._vllm_python, "-m", "dynamo.planner", "--config", self._planner_config_file],
            env=env,
        )
        print(
            f"  [Dynamo] Planner starting (pid={self._planner_process.pid})",
            flush=True,
        )

    def _start_frontend(self) -> None:
        if self._frontend_process is not None:
            return

        env = os.environ.copy()
        env.update(self._dynamo_env_vars())
        env["ALLOW_NONE_AUTHENTICATION"] = "yes"
        self._frontend_process = subprocess.Popen(
            [
                self._vllm_python, "-m", "dynamo.frontend",
                "--http-port", str(self._frontend_port),
                "--http-host", "0.0.0.0",
                "--router-mode", self._router_mode,
                "--discovery-backend", "etcd",
                "--namespace-prefix", self._namespace,
            ],
            env=env,
        )
        print(
            f"  [Dynamo] Frontend starting on http://{self._host}:{self._frontend_port} "
            f"(pid={self._frontend_process.pid}, router={self._router_mode})",
            flush=True,
        )

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
                    num_instances = sum(
                        1 for i in data.get("instances", [])
                        if i.get("endpoint") == "generate"
                    )
                    if num_instances >= self.dp_size:
                        print(
                            f"  [Dynamo] Frontend healthy — "
                            f"{num_instances} worker(s) registered",
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
    # VirtualConnectorClient lifecycle
    # ------------------------------------------------------------------

    def _start_virtual_connector_client(self) -> None:
        # DistributedRuntime (Rust) reads these from the process environment.
        os.environ.update(self._dynamo_env_vars())

        self._vc_thread = threading.Thread(
            target=self._vc_listener_thread,
            daemon=True,
            name="dynamo-vc-listener",
        )
        self._vc_thread.start()

    def _stop_virtual_connector_client(self) -> None:
        self._vc_stop.set()
        if self._vc_thread is not None and self._vc_thread.is_alive():
            self._vc_thread.join(timeout=10)
            self._vc_thread = None

    def _vc_listener_thread(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._vc_listener_loop(loop))
        except Exception as e:
            if not self._vc_stop.is_set():
                print(f"  [Dynamo] VirtualConnectorClient listener error: {e}", flush=True)
        finally:
            loop.close()

    async def _vc_listener_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        from dynamo._core import DistributedRuntime, VirtualConnectorClient

        runtime = DistributedRuntime(loop, "etcd", "tcp", True)
        client = VirtualConnectorClient(runtime, self._namespace)
        print("  [Dynamo] VirtualConnectorClient listening for scaling events", flush=True)

        while not self._vc_stop.is_set():
            try:
                await client.wait()
            except Exception as e:
                if not self._vc_stop.is_set():
                    print(f"  [Dynamo] VirtualConnectorClient error: {e}", flush=True)
                break

            # Check planner health after an event is signaled, not during idle wait.
            if self._planner_process is not None:
                retcode = self._planner_process.poll()
                if retcode is not None:
                    raise RuntimeError(
                        f"Dynamo planner exited with code {retcode}. "
                        f"Check console output above for details."
                    )

            try:
                event = await client.get()
                if event.decision_id != -1:
                    self._handle_scaling_event(event)
                    await client.complete(event)
            except Exception as e:
                if not self._vc_stop.is_set():
                    print(f"  [Dynamo] VirtualConnectorClient error: {e}", flush=True)
                break

    def _handle_scaling_event(self, event: Any) -> None:
        """Mock handler — logs the scaling decision without actually scaling."""
        print(
            f"  [Dynamo] Scaling event (decision_id={event.decision_id}): "
            f"prefill={event.num_prefill_workers}, decode={event.num_decode_workers}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # GenerationInterface
    # ------------------------------------------------------------------

    def _check_subprocesses(self) -> None:
        """Raise if any managed subprocess has exited unexpectedly."""
        for attr, name, _ in _SUBPROCESS_REGISTRY:
            proc: Optional[subprocess.Popen] = getattr(self, attr)
            if proc is not None:
                retcode = proc.poll()
                if retcode is not None:
                    raise RuntimeError(
                        f"Dynamo {name} exited unexpectedly with code {retcode}."
                    )

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        self._check_subprocesses()
        return True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def shutdown(self) -> bool:
        self._stop_virtual_connector_client()

        for attr, name, timeout in _SUBPROCESS_REGISTRY:
            proc: Optional[subprocess.Popen] = getattr(self, attr)
            if proc is not None:
                self._stop_subprocess(proc, name, timeout)
                setattr(self, attr, None)

        if self._planner_config_file is not None:
            os.unlink(self._planner_config_file)
            self._planner_config_file = None
        if self._etcd_data_dir is not None:
            shutil.rmtree(self._etcd_data_dir, ignore_errors=True)
            self._etcd_data_dir = None

        try:
            self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            print(f"  [Dynamo] Warning: worker shutdown failed: {e}", flush=True)

        return True

    # ------------------------------------------------------------------
    # Unsupported weight-update methods
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
