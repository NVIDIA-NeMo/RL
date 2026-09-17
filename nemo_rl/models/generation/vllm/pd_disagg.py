# SPDX-License-Identifier: Apache-2.0
"""Prefill/decode disaggregation for the async vLLM fleet.

EXPERIMENT (hemild, 2026-09-16), after Brian Yu's design (Slack C0AMULJ2L72,
2026-09-01): the fleet topology NeMo RL builds is unchanged, so refit needs no
work. Engines differ only in their NIXL KV-transfer role and batching limits,
and a ``vllm-router`` in front of them (prefill -> decode with
``kv_transfer_params``) is the single URL NeMo-Gym talks to.

Config block: ``policy.generation.vllm_cfg.pd_disagg``::

    enabled: true
    num_prefill_engines: 1          # engines [0, N) are prefill (kv_producer), the rest decode (kv_consumer)
    engines_per_node: 1             # TP4 on 4-GPU nodes -> one engine per node
    capture_on_prefill: false        # prefill splices the token-in prefix like decode but never registers/commits the call       # token capture stages on decode engines only (they see prompt + all output)
    kv_load_failure_policy: fail
    nixl_side_channel_port: {prefill: 5600, decode: 5700}   # + engine index on the node
    common_env: {...}               # exported in every engine process before vLLM starts
    prefill_env: {...}
    decode_env: {...}
    prefill_kwargs: {...}           # merged into vllm_kwargs for that role (dicts deep-merged)
    decode_kwargs: {...}
    router:
      bin: vllm-router
      prefill_policy: cache_aware
      decode_policy: cache_aware
      intra_node_data_parallel_size: 1
      request_timeout_s: 86400
      worker_startup_timeout_s: 3600
      port_range_low / port_range_high
      extra_args: []

Engine index: ``configure_worker`` exports ``NRL_VLLM_ENGINE_NODE_IDX`` and
``NRL_VLLM_ENGINE_IDX_ON_NODE`` per worker; the global index is
``node_idx * engines_per_node + idx_on_node``. The router partitions the DP-shard
ordered URL list with the same rule, which holds for one engine per node (the
tied worker groups are built in node order). Both sides log their view so a
mismatch is visible in the driver log.
"""

from __future__ import annotations

import atexit
import copy
import os
import shlex
import socket
import subprocess
import time
from typing import Any, Optional

ENGINE_NODE_IDX_ENV = "NRL_VLLM_ENGINE_NODE_IDX"
ENGINE_IDX_ON_NODE_ENV = "NRL_VLLM_ENGINE_IDX_ON_NODE"
_DEFAULT_SIDE_CHANNEL_PORTS = {"prefill": 5600, "decode": 5700}


def pd_config(generation_cfg: Any) -> Optional[dict[str, Any]]:
    """The ``pd_disagg`` block when enabled, else None. Accepts the generation config dict."""
    try:
        pd = (generation_cfg.get("vllm_cfg") or {}).get("pd_disagg") or None
    except AttributeError:
        return None
    if not pd or not pd.get("enabled"):
        return None
    if int(pd.get("num_prefill_engines", 0)) < 1:
        raise ValueError("pd_disagg.enabled requires num_prefill_engines >= 1")
    return dict(pd)


def engine_global_index(pd: dict[str, Any]) -> int:
    node_idx = int(os.environ.get(ENGINE_NODE_IDX_ENV, "0"))
    on_node = int(os.environ.get(ENGINE_IDX_ON_NODE_ENV, "0"))
    return node_idx * int(pd.get("engines_per_node", 1)) + on_node


def role_for_index(pd: dict[str, Any], index: int) -> str:
    return "prefill" if index < int(pd["num_prefill_engines"]) else "decode"


def capture_disabled_for_role(generation_cfg: Any, role: Optional[str]) -> bool:
    """True when this engine must not host token capture (prefill unless opted in)."""
    pd = pd_config(generation_cfg)
    if pd is None or role != "prefill":
        return False
    return not bool(pd.get("capture_on_prefill", False))


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)


def apply_pd_disagg(
    generation_cfg: Any, vllm_kwargs: dict[str, Any], node_ip: str
) -> Optional[str]:
    """Give this engine its P/D role. Mutates os.environ and ``vllm_kwargs``; returns the role."""
    pd = pd_config(generation_cfg)
    if pd is None:
        return None
    index = engine_global_index(pd)
    role = role_for_index(pd, index)

    exported: list[str] = []
    for key, value in (pd.get("common_env") or {}).items():
        os.environ[str(key)] = str(value); exported.append(str(key))
    for key, value in (pd.get(f"{role}_env") or {}).items():
        os.environ[str(key)] = str(value); exported.append(str(key))
    ports = dict(_DEFAULT_SIDE_CHANNEL_PORTS)
    ports.update(pd.get("nixl_side_channel_port") or {})
    side_port = int(ports[role]) + int(os.environ.get(ENGINE_IDX_ON_NODE_ENV, "0"))
    # NIXL's side channel must be reachable from the peer engines' nodes.
    os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = node_ip
    os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(side_port)
    exported += ["VLLM_NIXL_SIDE_CHANNEL_HOST", "VLLM_NIXL_SIDE_CHANNEL_PORT"]
    # With the Ray distributed executor the TP workers are Ray actors that only
    # receive the env vars vLLM copies explicitly. The KV cache is registered with
    # NIXL/UCX in those workers, so every transport variable must travel (RL's
    # checkpoint-engine refit guide; job 7193372 segfaulted in uct_ib_ops with
    # UCX defaults because these were only set in the engine process).
    _copy_env_to_ray_workers(exported)

    from vllm.config import KVTransferConfig

    vllm_kwargs["kv_transfer_config"] = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_producer" if role == "prefill" else "kv_consumer",
        kv_load_failure_policy=pd.get("kv_load_failure_policy", "fail"),
    )
    _deep_merge(vllm_kwargs, pd.get(f"{role}_kwargs") or {})
    print(
        f"[pd_disagg] engine index {index} -> {role} "
        f"(node_idx={os.environ.get(ENGINE_NODE_IDX_ENV, '?')}, "
        f"idx_on_node={os.environ.get(ENGINE_IDX_ON_NODE_ENV, '?')}, "
        f"nixl side channel {node_ip}:{side_port}, "
        f"max_num_batched_tokens={vllm_kwargs.get('max_num_batched_tokens')}, "
        f"max_num_seqs={vllm_kwargs.get('max_num_seqs')})",
        flush=True,
    )
    return role


def _copy_env_to_ray_workers(names: list[str]) -> None:
    """Add names to vLLM's additive VLLM_RAY_EXTRA_ENV_VARS_TO_COPY (read at worker creation)."""
    existing = os.environ.get("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", "")
    merged = {v.strip() for v in (*existing.split(","), *names) if v.strip()}
    os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] = ",".join(sorted(merged))


def _strip_v1(url: str) -> str:
    url = url.rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


def partition_urls(pd: dict[str, Any], base_urls: list[Optional[str]]) -> tuple[list[str], list[str]]:
    """Split the DP-shard ordered engine URLs into (prefill, decode) router targets."""
    urls = [u for u in base_urls if u]
    n_prefill = int(pd["num_prefill_engines"])
    if n_prefill >= len(urls):
        raise ValueError(
            f"pd_disagg.num_prefill_engines={n_prefill} leaves no decode engine "
            f"({len(urls)} engines reported URLs)"
        )
    return [_strip_v1(u) for u in urls[:n_prefill]], [_strip_v1(u) for u in urls[n_prefill:]]


def _free_port(host: str, low: int, high: int) -> int:
    for port in range(low, high):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free port for the P/D router in [{low}, {high})")


_ROUTER_PROCS: list[subprocess.Popen] = []


def _kill_routers() -> None:
    for proc in _ROUTER_PROCS:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def start_pd_router(
    pd: dict[str, Any], base_urls: list[Optional[str]], host: str, log_path: str
) -> str:
    """Launch ``vllm-router`` in P/D mode on this node; return the URL Gym should use."""
    router = pd.get("router") or {}
    prefill, decode = partition_urls(pd, base_urls)
    port = int(router.get("port", 0)) or _free_port(
        host, int(router.get("port_range_low", 1210)), int(router.get("port_range_high", 1290))
    )
    cmd = [
        str(router.get("bin", "vllm-router")),
        "--vllm-pd-disaggregation",
        "--host", host,
        "--port", str(port),
        "--prefill-policy", str(router.get("prefill_policy", "cache_aware")),
        "--decode-policy", str(router.get("decode_policy", "cache_aware")),
        "--intra-node-data-parallel-size", str(router.get("intra_node_data_parallel_size", 1)),
        "--request-timeout-secs", str(router.get("request_timeout_s", 86400)),
        "--worker-startup-timeout-secs", str(router.get("worker_startup_timeout_s", 3600)),
        # engines answer GET /health only once loaded; poll often so the router binds soon
        # after the last engine is up (Gym's readiness check on the router is time-bounded)
        "--worker-startup-check-interval", str(router.get("worker_startup_check_interval_s", 10)),
    ]
    for url in prefill:
        cmd += ["--prefill", url]
    for url in decode:
        cmd += ["--decode", url]
    cmd += [str(a) for a in (router.get("extra_args") or [])]

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log = open(log_path, "ab")  # noqa: SIM115 - handed to the child for its lifetime
    log.write((" ".join(shlex.quote(c) for c in cmd) + "\n").encode())
    log.flush()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    if not _ROUTER_PROCS:
        atexit.register(_kill_routers)
    _ROUTER_PROCS.append(proc)
    time.sleep(2.0)
    if proc.poll() is not None:
        raise RuntimeError(
            f"vllm-router exited immediately (rc={proc.returncode}); see {log_path}"
        )
    url = f"http://{host}:{port}/v1"
    print(
        f"[pd_disagg] vllm-router pid {proc.pid} at {url}: "
        f"{len(prefill)} prefill {prefill} -> {len(decode)} decode {decode}; log {log_path}",
        flush=True,
    )
    return url


def maybe_start_pd_router(master_config: Any, base_urls: list[Optional[str]]) -> Optional[str]:
    """setup.py hook: start the router when pd_disagg is enabled; return its URL or None."""
    generation_cfg = master_config.policy["generation"]
    pd = pd_config(generation_cfg)
    if pd is None:
        return None
    from nemo_rl.distributed.virtual_cluster import _get_node_ip_local

    logger_cfg = getattr(master_config, "logger", None) or {}
    log_dir = None
    if isinstance(logger_cfg, dict):
        log_dir = logger_cfg.get("log_dir")
    else:
        log_dir = getattr(logger_cfg, "log_dir", None)
    log_path = os.path.join(log_dir or os.getcwd(), "pd_router.log")
    return start_pd_router(pd, base_urls, host=_get_node_ip_local(), log_path=log_path)
