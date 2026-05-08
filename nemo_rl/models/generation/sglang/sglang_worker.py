import ipaddress
import logging
import multiprocessing
import os
import time
from collections.abc import Callable

import ray
import requests
from urllib3.exceptions import NewConnectionError

from nemo_rl.models.generation.sglang.utils.ray_utils import (
    get_current_node_ip,
    get_free_port,
    get_host_info,
)

logger = logging.getLogger(__name__)


def _get_sglang_file(relative_path: str) -> str:
    from importlib.util import find_spec

    spec = find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            f"sglang package not found while attempting to patch '{relative_path}'. "
        )

    base_dir = next(iter(spec.submodule_search_locations))
    file_path = os.path.join(base_dir, *relative_path.split("/"))
    if not os.path.exists(file_path):
        raise RuntimeError(
            f"Expected sglang file '{relative_path}' not found at '{file_path}'. "
            "The sglang version may have moved this file; compat patch cannot be applied."
        )
    return file_path


def _write_and_verify(file_path: str, content: str, sentinel: str) -> None:
    tmp_path = f"{file_path}.nemo_rl_compat.{os.getpid()}.tmp"
    with open(tmp_path, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, file_path)

    with open(file_path, "r") as f:
        verify = f.read()
    if sentinel not in verify:
        raise RuntimeError(
            f"Compat patch verification failed for {file_path}: "
            f"sentinel '{sentinel}' not present after write. "
            "The write may have been silently dropped by the filesystem."
        )


def _patch_sglang_safe_unpickler() -> None:
    file_to_patch = _get_sglang_file("srt/utils/common.py")

    with open(file_to_patch, "r") as f:
        content = f.read()

    sentinel = '"nemo_rl.models.policy.torch_reductions_utils."'
    if sentinel in content:
        return

    anchor = '        "torch.nn.parameter.",\n'
    insertion = anchor + '        "nemo_rl.models.policy.torch_reductions_utils.",\n'
    if anchor not in content:
        raise RuntimeError(
            f"SafeUnpickler allowlist anchor '{anchor.strip()}' not found in "
            f"{file_to_patch}."
        )

    content = content.replace(anchor, insertion, 1)
    _write_and_verify(file_to_patch, content, sentinel)
    logger.info("Patched SafeUnpickler allowlist in %s.", file_to_patch)


def _override_sglang_imbalance_check_env() -> None:
    """Force-disable sglang's per-GPU memory imbalance check.

    Pop the legacy names so the shim has nothing to copy, then set
    ``ENABLE=false`` directly. Inherited env reaches the subprocesses
    cleaned, so the shim no longer overwrites our ENABLE on re-import.
    """
    for legacy in (
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK",
        "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK",
    ):
        os.environ.pop(legacy, None)
    os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] = "false"


def _get_megatron_file(subpackage: str, relative_path: str) -> str | None:
    """Locate a file inside ``megatron.<subpackage>`` (e.g. ``core``, ``training``).

    Returns ``None`` if megatron isn't importable so callers can treat that
    as "nothing to patch". Raises if the package is present but the
    expected file is missing (signals a megatron version mismatch).
    """
    from importlib.util import find_spec

    full_pkg = f"megatron.{subpackage}"
    try:
        spec = find_spec(full_pkg)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None

    base_dir = next(iter(spec.submodule_search_locations))
    file_path = os.path.join(base_dir, *relative_path.split("/"))
    if not os.path.exists(file_path):
        raise RuntimeError(
            f"Expected megatron file '{full_pkg}/{relative_path}' not found at "
            f"'{file_path}'. The megatron version may have moved this file; "
            "compat patch cannot be applied."
        )
    return file_path


def _patch_megatron_hook_mode_in(file_path: str) -> None:
    """Comment out ``torch_memory_saver.hook_mode = "torch"`` in a megatron file.

    Megatron sets ``tms.hook_mode = "torch"`` at module import time on the
    global ``torch_memory_saver`` singleton. That mutation breaks sglang's
    pauseable CUDA graph path, which asserts ``_hook_mode == "preload"``
    inside ``TorchMemorySaver.cuda_graph(...)``. Commenting the line out
    leaves the singleton at its default ``"preload"`` mode that sglang
    expects.
    """
    with open(file_path, "r") as f:
        content = f.read()

    sentinel = '# torch_memory_saver.hook_mode = "torch"'
    if sentinel in content:
        return

    anchor = '    torch_memory_saver.hook_mode = "torch"\n'
    if anchor not in content:
        raise RuntimeError(
            f"Megatron hook_mode anchor '{anchor.strip()}' not found in "
            f"{file_path}; the megatron version may have moved or removed it."
        )

    replacement = (
        '    # torch_memory_saver.hook_mode = "torch"  '
        "# patched by nemo_rl: conflicts with sglang pauseable CUDA Graph\n"
    )
    content = content.replace(anchor, replacement, 1)
    _write_and_verify(file_path, content, sentinel)
    logger.info("Patched megatron tms.hook_mode mutation in %s.", file_path)


def _patch_megatron_dynamic_context_hook_mode() -> None:
    file_path = _get_megatron_file("core", "inference/contexts/dynamic_context.py")
    if file_path is None:
        return
    _patch_megatron_hook_mode_in(file_path)


def _patch_megatron_training_hook_mode() -> None:
    file_path = _get_megatron_file("training", "training.py")
    if file_path is None:
        return
    _patch_megatron_hook_mode_in(file_path)


def _apply_sglang_compat_patches() -> None:
    _patch_sglang_safe_unpickler()
    _override_sglang_imbalance_check_env()
    _patch_megatron_dynamic_context_hook_mode()
    _patch_megatron_training_hook_mode()


def get_base_gpu_id(gpus_per_node: int, sglang_cfg, rank):
    num_gpus = min(gpus_per_node, sglang_cfg["sglang_server"]["num_gpus_per_engine"])
    start_index = (rank * num_gpus) % gpus_per_node
    return start_index


def _to_local_gpu_id(physical_gpu_id: int) -> int:
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


def launch_server_process(server_args) -> multiprocessing.Process:
    from sglang.srt.entrypoints.http_server import launch_server

    multiprocessing.set_start_method("spawn", force=True)
    server_args.host = server_args.host.strip("[]")
    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()

    if server_args.node_rank != 0:
        return p

    _wait_server_healthy(
        base_url=server_args.url(),
        api_key=server_args.api_key,
        process_alive_fn=lambda: p.is_alive(),
    )

    return p


def _wait_server_healthy(
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
                response = session.get(f"{base_url}/health_generate", headers=headers)
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
        host=None,
        router_ip=None,
        router_port=None,
    ):

        self.router_ip = (
            router_ip
            if router_ip is not None
            else self.sglang_cfg["sglang_router"]["sglang_router_ip"]
        )
        self.router_port = (
            router_port
            if router_port is not None
            else self.sglang_cfg["sglang_router"]["sglang_router_port"]
        )

        host = host or get_host_info()[1]

        def _format_v6_uri(addr):
            if not addr or addr.startswith("["):
                return addr
            try:
                if ipaddress.ip_address(addr).version == 6:
                    return f"[{addr}]"
            except ValueError:
                pass
            return addr

        host = _format_v6_uri(host)
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        dist_init_addr = f"{_format_v6_uri(ip_part)}:{port_part}"

        server_args_dict = _compute_server_args(
            self.gpus_per_node,
            self.sglang_cfg,
            self.rank,
            dist_init_addr,
            nccl_port,
            host,
            port,
            base_gpu_id=self.base_gpu_id,
            num_gpus_per_engine=self.num_gpus_per_engine,
        )

        self.node_rank = server_args_dict["node_rank"]
        self.server_host = server_args_dict["host"]  # with [] if ipv6
        self.server_port = server_args_dict["port"]
        self.server_base_url = f"http://{self.server_host}:{self.server_port}"

        self._init_normal(server_args_dict)

    def _init_normal(self, server_args_dict):
        from sglang.srt.server_args import ServerArgs

        logger.info(
            f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}"
        )
        self.process = launch_server_process(ServerArgs(**server_args_dict))

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

    def flush_cache(self):
        """Flush the cache of the server."""
        if self.node_rank != 0:
            return
        # flush cache will not return status_code 200 when there are pending requests
        for _ in range(60):
            try:
                response = requests.get(f"{self.server_base_url}/flush_cache")
                if response.status_code == 200:
                    break
            except NewConnectionError as e:
                raise e
            except Exception as e:
                logger.info(f"Error flushing cache: {e}")
            # Pace retries on both non-200 and exception paths; otherwise the
            # 60 iterations fly by in milliseconds and timeout before sglang
            # has a chance to drain its queue.
            time.sleep(1)
        else:
            raise TimeoutError("Timeout while flushing cache.")

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

    def get_weight_version(self):
        if self.node_rank != 0:
            return
        # new sglang change api from /get_weight_version to /model_info
        for endpoint in ("/model_info", "/get_weight_version"):
            response = requests.get(f"{self.server_base_url}{endpoint}")
            if response.status_code == 200:
                return response.json()["weight_version"]
        response.raise_for_status()

    def release_memory_occupation(self, tags: list[str] | None = None):
        """Release memory occupation. Available tags: weights, kv_cache."""
        self.flush_cache()
        return self._make_request(
            "release_memory_occupation",
            {"tags": tags},
        )

    def resume_memory_occupation(self, tags: list[str] | None = None):
        """Available tags for multi-stage resume: weights, kv_cache."""
        return self._make_request(
            "resume_memory_occupation",
            {"tags": tags},
        )

    def release_memory_weights(self):
        from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

        return self.release_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    def release_memory_kv_cache_and_cuda_graph(self):
        from sglang.srt.constants import (
            GPU_MEMORY_TYPE_CUDA_GRAPH,
            GPU_MEMORY_TYPE_KV_CACHE,
        )

        return self.release_memory_occupation(
            tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]
        )

    def resume_memory_weights(self):
        from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

        return self.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    def resume_memory_kv_cache_and_cuda_graph(self):
        from sglang.srt.constants import (
            GPU_MEMORY_TYPE_CUDA_GRAPH,
            GPU_MEMORY_TYPE_KV_CACHE,
        )

        return self.resume_memory_occupation(
            tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]
        )

    def check_weights(self, action: str):
        return self._make_request("weights_checker", {"action": action})

    def init_weights_update_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend
    ):
        return self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def destroy_weights_update_group(self, group_name):
        try:
            return self._make_request(
                "destroy_weights_update_group",
                {
                    "group_name": group_name,
                },
            )
        except requests.exceptions.RequestException:
            # catch the case there the engine is just created and does not have the group.
            pass

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        flush_cache=False,
        weight_version: str | None = None,
    ):
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_distributed",
            payload,
        )

    def pause_generation(self, mode: str = "retract"):
        response = requests.post(
            f"{self.server_base_url}/pause_generation",
            json={"mode": mode},
        )
        response.raise_for_status()
        return response

    def continue_generation(self):
        response = requests.post(f"{self.server_base_url}/continue_generation", json={})
        response.raise_for_status()
        return response

    def post_process_weights(
        self,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = False,
    ):
        """Update model weights from tensor data.

        The HTTP server will only post meta data, and the real weights will be
        copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality
        to work properly. If you encounter issues, ensure your model is loaded
        on GPU devices rather than CPU.
        """
        return self._make_request(
            "post_process_weights",
            {
                "restore_weights_before_load": restore_weights_before_load,
                "post_process_quantization": post_process_quantization,
            },
        )

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

    def _simulate_crash(self):
        """Test-only: tear the engine down to simulate a crash.

        Underscore-prefixed to signal this is **not** part of the public
        worker API; production code should never call it.
        """
        logger.info(
            f"Simulating crash on engine {self.server_host}:{self.server_port}..."
        )
        self.shutdown()

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

    def invalidate_kv_cache(self) -> bool:
        """Flush the cache of the server.

        Returns:
            ``True`` on a successful flush. Peer (non-node-0) ranks return
            ``True`` since they do not own the HTTP server.

        Raises:
            NewConnectionError: if the engine HTTP server is unreachable
                (engine likely crashed); the caller cannot make progress
                with stale KV state, so we surface the failure rather than
                swallowing it.
            TimeoutError: if the server keeps replying non-200 for the full
                retry window — equivalent to a hang we shouldn't ignore.
        """
        if self.node_rank != 0:
            return True
        # flush cache will not return status_code 200 when there are pending requests
        for _ in range(60):
            try:
                response = requests.get(f"{self.server_base_url}/flush_cache")
                if response.status_code == 200:
                    return True
            except NewConnectionError:
                logger.exception("Connection error flushing cache")
                raise
            except Exception as e:
                logger.info(f"Error flushing cache: {e}")
            # Pace retries on both non-200 and exception paths; otherwise
            # the 60 iterations fly by in milliseconds.
            time.sleep(1)
        raise TimeoutError("Timeout while flushing cache.")


# ----------------------------------------------------------------------------
# Compute Server args
# ----------------------------------------------------------------------------
def _compute_server_args(
    gpus_per_node: int,
    sglang_cfg,
    rank,
    dist_init_addr,
    nccl_port,
    host,
    port,
    base_gpu_id: int | None = None,
    num_gpus_per_engine: int | None = None,
):
    _gpus_per_engine = (
        num_gpus_per_engine or sglang_cfg["sglang_server"]["num_gpus_per_engine"]
    )
    nnodes = max(1, _gpus_per_engine // gpus_per_node)
    node_rank = rank % nnodes
    base = (
        base_gpu_id
        if base_gpu_id is not None
        else get_base_gpu_id(gpus_per_node, sglang_cfg, rank)
    )
    base = _to_local_gpu_id(base)
    # ``_gpus_per_engine`` is the engine's total GPU count (TP × PP). When PP=1
    # (the historical default) this equals ``tp_size``; when ``pp_size > 1``
    # the engine spans ``tp_size * pp_size`` GPUs, so derive ``tp_size`` by
    # dividing out PP. Falsy/0 ``pp_size`` is treated as 1.
    _pp_size = sglang_cfg["sglang_cfg"].get("pp_size", 1) or 1
    kwargs = {
        "model_path": sglang_cfg["sglang_cfg"]["model_path"],
        "trust_remote_code": True,
        "random_seed": sglang_cfg["sglang_cfg"]["random_seed"] + rank,
        # memory
        "enable_memory_saver": sglang_cfg["sglang_server"]["needs_offload"],
        "enable_weights_cpu_backup": sglang_cfg["sglang_server"]["cpu_weight_backup"],
        # distributed
        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
        "gpu_id_step": 1,
        "base_gpu_id": base,
        # parallel
        "tp_size": _gpus_per_engine // _pp_size,
        "dp_size": sglang_cfg["sglang_cfg"]["dp_size"],
        "pp_size": _pp_size,
        "ep_size": sglang_cfg["sglang_cfg"]["ep_size"],
        # always skip warmup to prevent warmup timeout.
        "skip_server_warmup": sglang_cfg["sglang_cfg"]["skip_server_warmup"],
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
        # CUDA graph batch-size cap (Optional[int], default None).
        "cuda_graph_max_bs",
        # DP-attention switch (newer sglang forks): replicates attention along
        # ``dp_size`` while keeping MoE/MLP under TP.
        "enable_dp_attention",
        # MoE all-to-all backend: "none" | "deepep" | "mooncake" | "mori" |
        # "ascend_fuseep" | "flashinfer". Replaces the older
        # ``enable_ep_moe`` boolean knob.
        "moe_a2a_backend",
        # DeepEP routing mode (used when ``moe_a2a_backend == "deepep"``):
        # "auto" | "normal" | "low_latency".
        "deepep_mode",
    ]:
        if key in sglang_cfg["sglang_cfg"]:
            kwargs[key] = sglang_cfg["sglang_cfg"][key]

    return kwargs
