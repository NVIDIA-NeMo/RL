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

import logging
import multiprocessing
import os
import time
from collections.abc import Callable

import ray
import requests

from nemo_rl.models.generation.sglang.utils.ip_port_utils import _format_v6_uri
from nemo_rl.models.generation.sglang.utils.patches import _apply_sglang_compat_patches
from nemo_rl.models.generation.sglang.utils.ray_utils import (
    get_current_node_ip,
    get_free_port,
)

logger = logging.getLogger(__name__)


@ray.remote  # pragma: no cover
class SGLangGenerationWorker:
    def __init__(
        self,
        gpus_per_node: int,
        sglang_cfg,
        rank: int,
        base_gpu_id: int | None = None,
        num_gpus_per_engine: int | None = None,
    ):
        _apply_sglang_compat_patches()
        self.gpus_per_node = gpus_per_node
        self.sglang_cfg = sglang_cfg
        self.rank = rank
        self.base_gpu_id = base_gpu_id
        self.num_gpus_per_engine = num_gpus_per_engine

    def init(
        self,
        dist_init_addr,
        port,
        nccl_port,
        host,
        router_ip,
        router_port,
    ):
        self.router_ip = router_ip
        self.router_port = router_port

        host = _format_v6_uri(host)
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        dist_init_addr = f"{_format_v6_uri(ip_part)}:{port_part}"

        server_args_dict = self._compute_server_args(
            dist_init_addr,
            nccl_port,
            host,
            port,
        )

        self.node_rank = server_args_dict["node_rank"]
        self.server_host = server_args_dict["host"]  # with [] if ipv6
        self.server_port = server_args_dict["port"]
        self.server_base_url = f"http://{self.server_host}:{self.server_port}"

        self._launch_server_process(server_args_dict)

    def _launch_server_process(self, server_args_dict):
        from sglang.srt.entrypoints.http_server import launch_server
        from sglang.srt.server_args import ServerArgs

        logger.info(
            f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}"
        )

        server_args = ServerArgs(**server_args_dict)
        multiprocessing.set_start_method("spawn", force=True)
        server_args.host = server_args.host.strip("[]")
        p = multiprocessing.Process(target=launch_server, args=(server_args,))
        p.start()

        if server_args.node_rank == 0:
            self._wait_server_healthy(
                base_url=server_args.url(),
                api_key=server_args.api_key,
                process_alive_fn=lambda: p.is_alive(),
            )

        self.process = p

        if self.node_rank == 0 and self.router_ip and self.router_port:
            payload = {
                "url": self.server_base_url,
                "worker_type": "regular",
            }
            response = requests.post(
                f"http://{self.router_ip}:{self.router_port}/workers",
                json=payload,
            )
            response.raise_for_status()
            self._wait_for_router_registration()

    def _wait_for_router_registration(
        self, timeout: float = 30.0, interval: float = 0.5
    ) -> None:
        """Wait until this worker is visible through the router's workers API."""
        workers_url = f"http://{self.router_ip}:{self.router_port}/workers"
        deadline = time.monotonic() + timeout
        last_error = None

        while time.monotonic() < deadline:
            try:
                response = requests.get(workers_url, timeout=5)
                response.raise_for_status()
                workers = response.json().get("workers", [])
                if any(worker.get("url") == self.server_base_url for worker in workers):
                    return
            except Exception as e:
                last_error = e

            time.sleep(interval)

        detail = f" Last error: {last_error}" if last_error is not None else ""
        raise RuntimeError(
            f"Timed out waiting for worker {self.server_base_url} to appear in "
            f"router {workers_url}.{detail}"
        )

    def _make_request(self, endpoint: str, payload: dict | None = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        if self.node_rank != 0:
            return

        url = f"{self.server_base_url}/{endpoint}"
        response = requests.post(url, json=payload or {})
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        return response.json()

    @staticmethod
    def _get_current_node_ip_and_free_port(start_port=10000, consecutive=1):
        return get_current_node_ip(), get_free_port(
            start_port=start_port, consecutive=consecutive
        )

    def health_generate(self, timeout: float = 5.0) -> bool:
        """Run /health_generate on the underlying SGLang HTTP server.

        Args:
            timeout: Timeout for the health request in seconds.

        Returns:
            True if the server responds with HTTP 200.

        Raises:
            requests.RequestException: If the request fails for any reason, including timeout.
        """
        if self.node_rank != 0:
            return True

        response = requests.get(
            f"{self.server_base_url}/health_generate",
            timeout=timeout,
        )
        response.raise_for_status()
        return True

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ):
        """Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_tensor",
            payload,
        )

    def shutdown(self):
        from sglang.srt.utils import kill_process_tree

        logger.info(f"Shutdown engine {self.server_host}:{self.server_port}...")
        if self.node_rank == 0:
            worker_url = self.server_base_url
            response = None
            try:
                all_workers = requests.get(
                    f"http://{self.router_ip}:{self.router_port}/workers"
                ).json()["workers"]
                for worker in all_workers:
                    if worker["url"] == worker_url:
                        worker_id = worker["id"]
                        response = requests.delete(
                            f"http://{self.router_ip}:{self.router_port}/workers/{worker_id}"
                        )
                        break
                else:
                    logger.warning(
                        f"Worker {worker_url} not found in router during shutdown."
                    )
            except Exception as e:
                logger.warning(f"Failed to fetch workers list or remove worker: {e}")

            if response is not None:
                response.raise_for_status()
        kill_process_tree(self.process.pid)

    def release_memory_occupation(self, tags: list[str] | None = None):
        """Release memory occupation. Available tags: weights, kv_cache."""
        from sglang.srt.constants import (
            GPU_MEMORY_TYPE_CUDA_GRAPH,
            GPU_MEMORY_TYPE_KV_CACHE,
            GPU_MEMORY_TYPE_WEIGHTS,
        )

        if tags is None:
            tags = ["weights", "kv_cache"]

        sglang_tags = []
        if "weights" in tags:
            sglang_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
        if "kv_cache" in tags:
            sglang_tags.extend([GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

        self.invalidate_kv_cache()
        return self._make_request(
            "release_memory_occupation",
            {"tags": sglang_tags},
        )

    def resume_memory_occupation(self, tags: list[str] | None = None):
        """Available tags for multi-stage resume: weights, kv_cache."""
        from sglang.srt.constants import (
            GPU_MEMORY_TYPE_CUDA_GRAPH,
            GPU_MEMORY_TYPE_KV_CACHE,
            GPU_MEMORY_TYPE_WEIGHTS,
        )

        if tags is None:
            tags = ["weights", "kv_cache"]

        sglang_tags = []
        if "weights" in tags:
            sglang_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
        if "kv_cache" in tags:
            sglang_tags.extend([GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

        return self._make_request(
            "resume_memory_occupation",
            {"tags": sglang_tags},
        )

    def check_weights(self, action: str):
        return self._make_request("weights_checker", {"action": action})

    def start_profile(
        self,
        # The output directory
        output_dir: str | None = None,
        # If set, it profile as many as this number of steps.
        # If it is set, profiling is automatically stopped after this step, and
        # the caller doesn't need to run stop_profile.
        start_step: int | None = None,
        num_steps: int | None = None,
        activities: list[str] | None = None,
        profile_by_stage: bool = False,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
    ):
        response = requests.post(
            f"{self.server_base_url}/start_profile",
            json={
                "output_dir": output_dir,
                "start_step": start_step,
                "num_steps": num_steps,
                "activities": activities,
                "profile_by_stage": profile_by_stage,
                "with_stack": with_stack,
                "record_shapes": record_shapes,
            },
        )
        response.raise_for_status()
        return response

    def stop_profile(self):
        response = requests.post(f"{self.server_base_url}/stop_profile", json={})
        response.raise_for_status()
        return response

    # ---------------------------------------------------------------------------
    # Compatible with parent class or old interfaces
    # ---------------------------------------------------------------------------
    def get_base_url(self) -> str | None:
        """Return the ``http://host:port`` base URL of this SGLang server.

        Only node-rank 0 owns the HTTP server; peer ranks return ``None`` so
        callers can filter them out when collecting per-engine URLs.
        """
        if self.node_rank != 0:
            return None
        return self.server_base_url

    def invalidate_kv_cache(self) -> None:
        """Flush this server's KV cache."""
        if self.node_rank != 0:
            return

        response = requests.get(f"{self.server_base_url}/flush_cache")
        if response.status_code != 200:
            response.raise_for_status()
            raise RuntimeError(
                f"flush_cache returned unexpected status {response.status_code}"
            )

    # ----------------------------------------------------------------------------
    # Compute Server args
    # ----------------------------------------------------------------------------
    def _compute_server_args(
        self,
        dist_init_addr,
        nccl_port,
        host,
        port,
    ):
        sglang_cfg_inner = self.sglang_cfg["sglang_cfg"]
        sglang_server_cfg = sglang_cfg_inner["sglang_server_config"]
        _gpus_per_engine = (
            self.num_gpus_per_engine or sglang_server_cfg["num_gpus_per_engine"]
        )
        tp_size = sglang_cfg_inner["tp_size"]
        pp_size = sglang_cfg_inner["pp_size"]
        assert tp_size == _gpus_per_engine // pp_size, (
            f"tp_size ({tp_size}) must equal num_gpus_per_engine ({_gpus_per_engine}) "
            f"// pp_size ({pp_size})"
        )
        nnodes = max(1, _gpus_per_engine // self.gpus_per_node)
        node_rank = self.rank % nnodes
        local_gpu_id = self._to_local_gpu_id(self.base_gpu_id)
        kwargs = {
            "model_path": sglang_cfg_inner["model_path"],
            "trust_remote_code": True,
            "random_seed": sglang_cfg_inner["random_seed"] + self.rank,
            # memory
            "enable_memory_saver": sglang_server_cfg["needs_offload"],
            "enable_weights_cpu_backup": sglang_server_cfg["cpu_weight_backup"],
            # distributed
            "host": host,
            "port": port,
            "nccl_port": nccl_port,
            "nnodes": nnodes,
            "node_rank": node_rank,
            "dist_init_addr": dist_init_addr,
            "gpu_id_step": 1,
            "base_gpu_id": local_gpu_id,
            # parallel
            "tp_size": tp_size,
            "dp_size": sglang_cfg_inner["dp_size"],
            "pp_size": pp_size,
            "ep_size": sglang_cfg_inner["ep_size"],
            # always skip warmup to prevent warmup timeout.
            "skip_server_warmup": sglang_cfg_inner["skip_server_warmup"],
            # always enable draft weights cpu backup so that we run training without mtp weights.
            "enable_draft_weights_cpu_backup": True,
        }

        for key in [
            "dtype",
            "kv_cache_dtype",
            "context_length",
            "max_running_requests",
            "chunked_prefill_size",
            "max_prefill_tokens",
            "schedule_policy",
            "schedule_conservativeness",
            "cpu_offload_gb",
            "log_level",
            "mem_fraction_static",
            "allow_auto_truncate",
            "disable_piecewise_cuda_graph",
            "disable_cuda_graph",
            "disable_cuda_graph_padding",
            "cuda_graph_max_bs",
            "cuda_graph_bs",
        ]:
            if key in sglang_cfg_inner:
                kwargs[key] = sglang_cfg_inner[key]

        return kwargs

    def _to_local_gpu_id(self, physical_gpu_id: int) -> int:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not cvd:
            return physical_gpu_id  # no remapping
        # CUDA_VISIBLE_DEVICES can be like "4,5,6,7"
        visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
        # In a remapped process, valid torch device indices are 0..len(visible)-1
        if physical_gpu_id in visible:
            return visible.index(physical_gpu_id)
        # If we're already getting local IDs, allow them
        if 0 <= physical_gpu_id < len(visible):
            return physical_gpu_id
        raise RuntimeError(
            f"GPU id {physical_gpu_id} is not valid under CUDA_VISIBLE_DEVICES={cvd}. "
            f"Expected one of {visible} (physical) or 0..{len(visible) - 1} (local)."
        )

    def _wait_server_healthy(
        self,
        base_url: str,
        api_key: str | None,
        process_alive_fn: Callable[[], bool],
    ) -> None:
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {api_key}",
        }

        with requests.Session() as session:
            while True:
                try:
                    response = session.get(
                        f"{base_url}/health_generate", headers=headers
                    )
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass

                if not process_alive_fn():
                    raise Exception("Server process terminated unexpectedly.")

                time.sleep(2)

            # use flush_cache to make sure the working queue is empty, so that we can do offload
            while True:
                try:
                    response = session.get(f"{base_url}/flush_cache", headers=headers)
                    if response.status_code == 200:
                        break

                except requests.RequestException:
                    pass

                if not process_alive_fn():
                    raise Exception("Server process terminated unexpectedly.")

                time.sleep(2)
