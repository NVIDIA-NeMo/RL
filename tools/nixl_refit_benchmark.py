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

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import os
import socket
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_rl.utils.checkpoint_engines.nixl import NIXLCheckpointEngine

_GIB = 1024**3
_SUPPORTED_TORCH_BASELINES = {"gloo", "nccl"}


@dataclass(frozen=True)
class Variant:
    name: str
    transfer_mode: str
    direct_stripe_count: int
    metadata_batch_size: int
    buffer_count: int


def _env_subset() -> dict[str, str]:
    prefixes = ("UCX_", "NIXL_", "MELLANOX_", "NVIDIA_", "FI_")
    names = {
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTHONPATH",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "UV_PROJECT_ENVIRONMENT",
        "VIRTUAL_ENV",
    }
    return {
        key: value
        for key, value in os.environ.items()
        if key in names or key.startswith(prefixes)
    }


def _alive_nodes() -> list[dict[str, Any]]:
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    return sorted(nodes, key=lambda node: node["NodeManagerAddress"])


def _node_label(node: dict[str, Any]) -> str:
    return f"{node['NodeManagerAddress']}:{node['NodeID'][:8]}"


def _wait_for_nodes(min_nodes: int, timeout_s: int) -> list[dict[str, Any]]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        nodes = _alive_nodes()
        if len(nodes) >= min_nodes:
            return nodes
        time.sleep(2)
    nodes = _alive_nodes()
    raise RuntimeError(
        f"Timed out waiting for {min_nodes} live Ray nodes; saw "
        f"{len(nodes)}: {[_node_label(node) for node in nodes]}"
    )


def _actor_options(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_cpus": 2,
        "num_gpus": 1,
        "runtime_env": {
            "env_vars": _env_subset(),
            "py_executable": sys.executable,
        },
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=node["NodeID"],
            soft=False,
        ),
    }


def _parse_devices(value: str) -> list[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one device")
    unsupported = sorted(set(devices) - {"cuda", "cpu"})
    if unsupported:
        raise ValueError(f"unsupported devices: {unsupported}")
    return devices


def _parse_variants(value: str) -> list[Variant]:
    variants = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        fields = item.split(":")
        if len(fields) not in (4, 5):
            raise ValueError(
                "variant entries must be name:transfer_mode:stripe:metadata_batch"
                "[:buffer_count]"
            )
        name, transfer_mode, stripe, metadata_batch = fields[:4]
        buffer_count = fields[4] if len(fields) == 5 else "3"
        variants.append(
            Variant(
                name=name,
                transfer_mode=transfer_mode,
                direct_stripe_count=int(stripe),
                metadata_batch_size=int(metadata_batch),
                buffer_count=int(buffer_count),
            )
        )
    if not variants:
        raise ValueError("--variants must contain at least one variant")
    return variants


def _parse_torch_baselines(value: str) -> list[str]:
    baselines = [item.strip() for item in value.split(",") if item.strip()]
    unsupported = sorted(set(baselines) - _SUPPORTED_TORCH_BASELINES)
    if unsupported:
        raise ValueError(f"unsupported torch baselines: {unsupported}")
    return baselines


def _parse_backend_init_params(values: list[str]) -> dict[str, str]:
    params = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"backend init parameter {value!r} must be formatted as key=value"
            )
        key, param_value = value.split("=", 1)
        if not key:
            raise ValueError(f"backend init parameter {value!r} has an empty key")
        params[key] = param_value
    return params


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _min(values: list[float]) -> float:
    return min(values) if values else 0.0


def _p50(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _emit(prefix: str, payload: dict[str, Any]) -> None:
    print(f"{prefix} {json.dumps(payload, sort_keys=True)}", flush=True)


def _torch_diagnostics() -> dict[str, Any]:
    torch_cuda = getattr(torch, "cuda", None)
    return {
        "torch_file": getattr(torch, "__file__", None),
        "torch_path": list(getattr(torch, "__path__", [])),
        "torch_version": getattr(torch, "__version__", None),
        "torch_has_device": hasattr(torch, "device"),
        "torch_has_cuda": torch_cuda is not None,
        "torch_cuda_available": bool(
            torch_cuda is not None and torch_cuda.is_available()
        ),
        "torch_cuda_device_count": (
            torch_cuda.device_count() if torch_cuda is not None else 0
        ),
    }


@ray.remote
class RefitBenchmarkEndpoint:
    def __init__(
        self,
        *,
        label: str,
        bucket_size_bytes: int,
        backend_name: str,
        backend_init_params: dict[str, str],
    ) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        self.label = label
        self.bucket_size_bytes = bucket_size_bytes
        self.backend_name = backend_name
        self.backend_init_params = backend_init_params
        self.hostname = os.uname().nodename
        self.node_ip = ray.util.get_node_ip_address().strip("[]")
        self.engine: NIXLCheckpointEngine | None = None
        self.device: Any = None
        self.load_device: Any = None
        self.source: Any = None
        self.destination: Any = None
        self.torch_pg_backend: str | None = None
        self.torch_pg_rank: int | None = None
        self.metadata_send_calls = 0
        self.metadata_bucket_records = 0

    def report(self) -> dict[str, Any]:
        torch_report = _torch_diagnostics()
        return {
            "label": self.label,
            "hostname": self.hostname,
            "node_ip": self.node_ip,
            "pid": os.getpid(),
            "python": os.sys.executable,
            "path": os.environ.get("PATH"),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "sys_path_head": sys.path[:12],
            "uv_project_environment": os.environ.get("UV_PROJECT_ENVIRONMENT"),
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ucx_net_devices": os.environ.get("UCX_NET_DEVICES"),
            "ucx_tls": os.environ.get("UCX_TLS"),
            "ucx_max_rndv_rails": os.environ.get("UCX_MAX_RNDV_RAILS"),
            **torch_report,
        }

    def reserve_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", 0))
            return int(sock.getsockname()[1])

    def configure(
        self,
        *,
        device: str,
        load_device: str,
        variant: dict[str, Any],
    ) -> dict[str, Any]:
        self.close()
        self.device = torch.device(device)
        self.load_device = torch.device(load_device)
        if self.device.type == "cuda" or self.load_device.type == "cuda":
            torch.cuda.set_device(0)
            torch.empty(1, device="cuda").fill_(1)
            torch.cuda.synchronize()

        self.engine = NIXLCheckpointEngine(
            bucket_size=self.bucket_size_bytes,
            device=self.device,
            backend_name=self.backend_name,
            backend_init_params=self.backend_init_params,
            cleanup_after_load=False,
            topology="paired",
            transfer_mode=variant["transfer_mode"],
            buffer_count=variant["buffer_count"],
            direct_min_bytes=0,
            background_progress=False,
            load_batch_bucket_count=1,
            direct_stripe_count=variant["direct_stripe_count"],
            metadata_batch_size=variant["metadata_batch_size"],
        )
        self._install_metadata_counter()
        return {
            "label": self.label,
            "device": str(self.device),
            "load_device": str(self.load_device),
            "variant": variant,
        }

    def configure_torch_process_group(
        self,
        *,
        backend: str,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        device: str,
    ) -> dict[str, Any]:
        self.close()
        self.close_torch_process_group()
        self.device = torch.device(device)
        self.load_device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(0)
            torch.empty(1, device="cuda").fill_(1)
            torch.cuda.synchronize()

        torch.distributed.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=300),
        )
        self.torch_pg_backend = backend
        self.torch_pg_rank = rank
        return {
            "label": self.label,
            "backend": backend,
            "rank": rank,
            "world_size": world_size,
            "device": str(self.device),
            "master_addr": master_addr,
            "master_port": master_port,
        }

    def _install_metadata_counter(self) -> None:
        if self.engine is None:
            raise RuntimeError("engine is not configured")
        original_send_messages = self.engine.agent.send_messages

        def counted_send_messages(
            agent_name: str, messages: list[dict[str, Any]]
        ) -> None:
            self.metadata_send_calls += 1
            self.metadata_bucket_records += len(messages)
            original_send_messages(agent_name, messages)

        self.engine.agent.send_messages = counted_send_messages

    def prepare(self) -> Any:
        if self.engine is None:
            raise RuntimeError("engine is not configured")
        return self.engine.prepare()

    def init_policy(self, *, metadata: list[Any]) -> None:
        if self.engine is None:
            raise RuntimeError("engine is not configured")
        self.engine.init_policy_process_group(
            worker_rank=0,
            train_world_size=1,
            rollout_world_size=1,
            metadata=metadata,
        )

    def init_rollout(self, *, metadata: list[Any]) -> None:
        if self.engine is None:
            raise RuntimeError("engine is not configured")
        self.engine.init_rollout_process_group(
            rollout_rank=0,
            train_world_size=1,
            rollout_world_size=1,
            metadata=metadata,
        )

    def allocate_source(self, *, size_bytes: int, fill_value: int) -> dict[str, Any]:
        pin_memory = self.device.type == "cpu" and torch.cuda.is_available()
        start = time.perf_counter()
        self.source = torch.empty(
            size_bytes,
            dtype=torch.uint8,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.source.fill_(fill_value)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {
            "label": self.label,
            "bytes": size_bytes,
            "device": str(self.source.device),
            "pin_memory": pin_memory,
            "fill_value": fill_value,
            "elapsed_s": time.perf_counter() - start,
        }

    def allocate_destination(
        self, *, size_bytes: int, fill_value: int
    ) -> dict[str, Any]:
        pin_memory = self.load_device.type == "cpu" and torch.cuda.is_available()
        start = time.perf_counter()
        self.destination = torch.empty(
            size_bytes,
            dtype=torch.uint8,
            device=self.load_device,
            pin_memory=pin_memory,
        )
        self.destination.fill_(fill_value)
        if self.load_device.type == "cuda":
            torch.cuda.synchronize(self.load_device)
        return {
            "label": self.label,
            "bytes": size_bytes,
            "device": str(self.destination.device),
            "pin_memory": pin_memory,
            "fill_value": fill_value,
            "elapsed_s": time.perf_counter() - start,
        }

    def reset_destination(self, *, fill_value: int) -> None:
        if self.destination is None:
            raise RuntimeError("destination is not allocated")
        self.destination.fill_(fill_value)
        if self.destination.device.type == "cuda":
            torch.cuda.synchronize(self.destination.device)

    def send_once(self, *, run_index: int) -> dict[str, Any]:
        if self.engine is None:
            raise RuntimeError("engine is not configured")
        if self.source is None:
            raise RuntimeError("source is not allocated")

        self.metadata_send_calls = 0
        self.metadata_bucket_records = 0

        async def run() -> None:
            await self.engine.send_weights(iter([("bench.weight", self.source)]))

        start = time.perf_counter()
        asyncio.run(run())
        elapsed = time.perf_counter() - start
        return {
            "label": self.label,
            "run_index": run_index,
            "elapsed_s": elapsed,
            "bytes": self.source.nbytes,
            "logical_gib_per_s": self.source.nbytes / elapsed / _GIB,
            "metadata_send_calls": self.metadata_send_calls,
            "metadata_bucket_records": self.metadata_bucket_records,
        }

    def receive_once(
        self,
        *,
        run_index: int,
        expected_bytes: int,
        expected_fill_value: int,
    ) -> dict[str, Any]:
        if self.engine is None:
            raise RuntimeError("engine is not configured")
        if self.destination is None:
            raise RuntimeError("destination is not allocated")

        async def run() -> dict[str, Any]:
            batch_count = 0
            weight_count = 0
            logical_bytes = 0
            load_enqueue_s = 0.0
            async for batch in self.engine.receive_weight_batches():
                batch_count += 1
                copy_start = time.perf_counter()
                for name, tensor in batch:
                    if name != "bench.weight":
                        raise RuntimeError(f"unexpected tensor {name!r}")
                    if tensor.nbytes != expected_bytes:
                        raise RuntimeError(
                            f"expected {expected_bytes} bytes, got {tensor.nbytes}"
                        )
                    self.destination.copy_(tensor, non_blocking=True)
                    weight_count += 1
                    logical_bytes += tensor.nbytes
                load_enqueue_s += time.perf_counter() - copy_start
                record_load_complete = getattr(
                    batch,
                    "record_cuda_load_complete",
                    None,
                )
                if callable(record_load_complete):
                    record_load_complete()
            return {
                "batch_count": batch_count,
                "weight_count": weight_count,
                "logical_bytes": logical_bytes,
                "load_enqueue_s": load_enqueue_s,
            }

        start = time.perf_counter()
        stats = asyncio.run(run())
        receive_elapsed_s = time.perf_counter() - start

        sync_start = time.perf_counter()
        if self.destination.device.type == "cuda":
            torch.cuda.synchronize(self.destination.device)
        load_sync_s = time.perf_counter() - sync_start

        first_value = int(self.destination[0].item())
        last_value = int(self.destination[-1].item())
        if first_value != expected_fill_value or last_value != expected_fill_value:
            raise RuntimeError(
                f"destination validation failed: first={first_value} "
                f"last={last_value} expected={expected_fill_value}"
            )

        logical_bytes = stats["logical_bytes"]
        return {
            "label": self.label,
            "run_index": run_index,
            "elapsed_s": receive_elapsed_s,
            "bytes": logical_bytes,
            "logical_gib_per_s": logical_bytes / receive_elapsed_s / _GIB,
            "load_enqueue_s": stats["load_enqueue_s"],
            "load_sync_s": load_sync_s,
            "batch_count": stats["batch_count"],
            "weight_count": stats["weight_count"],
            "first_value": first_value,
            "last_value": last_value,
        }

    def torch_broadcast_once(
        self,
        *,
        run_index: int,
        expected_fill_value: int,
    ) -> dict[str, Any]:
        if self.torch_pg_rank is None:
            raise RuntimeError("torch process group is not configured")
        tensor = self.source if self.torch_pg_rank == 0 else self.destination
        if tensor is None:
            raise RuntimeError("broadcast tensor is not allocated")

        start = time.perf_counter()
        torch.distributed.broadcast(tensor, src=0)
        if tensor.device.type == "cuda":
            torch.cuda.synchronize(tensor.device)
        elapsed = time.perf_counter() - start

        first_value = int(tensor[0].item())
        last_value = int(tensor[-1].item())
        if first_value != expected_fill_value or last_value != expected_fill_value:
            raise RuntimeError(
                f"torch broadcast validation failed on rank {self.torch_pg_rank}: "
                f"first={first_value} last={last_value} expected={expected_fill_value}"
            )
        return {
            "label": self.label,
            "backend": self.torch_pg_backend,
            "rank": self.torch_pg_rank,
            "run_index": run_index,
            "elapsed_s": elapsed,
            "bytes": tensor.nbytes,
            "logical_gib_per_s": tensor.nbytes / elapsed / _GIB,
            "first_value": first_value,
            "last_value": last_value,
        }

    def close_torch_process_group(self) -> None:
        torch_distributed = getattr(torch, "distributed", None)
        if (
            torch_distributed is not None
            and torch_distributed.is_available()
            and torch_distributed.is_initialized()
        ):
            torch_distributed.destroy_process_group()
        self.torch_pg_backend = None
        self.torch_pg_rank = None

    def close(self) -> None:
        if self.engine is not None:
            self.engine.close()
            self.engine = None
        self.close_torch_process_group()
        self.source = None
        self.destination = None


def _summarize_runs(
    *,
    device: str,
    variant: Variant,
    size_bytes: int,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    send_elapsed = [run["send"]["elapsed_s"] for run in runs]
    receive_elapsed = [run["receive"]["elapsed_s"] for run in runs]
    driver_wall = [run["driver_wall_s"] for run in runs]
    refit_elapsed = [
        max(run["send"]["elapsed_s"], run["receive"]["elapsed_s"]) for run in runs
    ]
    send_bw = [run["send"]["logical_gib_per_s"] for run in runs]
    receive_bw = [run["receive"]["logical_gib_per_s"] for run in runs]
    return {
        "transport": "nixl",
        "device": device,
        "variant": variant.name,
        "transfer_mode": variant.transfer_mode,
        "direct_stripe_count": variant.direct_stripe_count,
        "metadata_batch_size": variant.metadata_batch_size,
        "buffer_count": variant.buffer_count,
        "size_gib": size_bytes / _GIB,
        "runs": len(runs),
        "send_elapsed_mean_s": _mean(send_elapsed),
        "send_elapsed_min_s": _min(send_elapsed),
        "send_elapsed_p50_s": _p50(send_elapsed),
        "receive_elapsed_mean_s": _mean(receive_elapsed),
        "receive_elapsed_min_s": _min(receive_elapsed),
        "receive_elapsed_p50_s": _p50(receive_elapsed),
        "refit_elapsed_mean_s": _mean(refit_elapsed),
        "refit_elapsed_min_s": _min(refit_elapsed),
        "refit_elapsed_p50_s": _p50(refit_elapsed),
        "driver_wall_mean_s": _mean(driver_wall),
        "send_logical_gib_per_s_mean": _mean(send_bw),
        "receive_logical_gib_per_s_mean": _mean(receive_bw),
        "metadata_send_calls_mean": _mean(
            [run["send"]["metadata_send_calls"] for run in runs]
        ),
        "metadata_bucket_records_mean": _mean(
            [run["send"]["metadata_bucket_records"] for run in runs]
        ),
        "load_enqueue_mean_s": _mean(
            [run["receive"]["load_enqueue_s"] for run in runs]
        ),
        "load_sync_mean_s": _mean([run["receive"]["load_sync_s"] for run in runs]),
        "receive_batch_count_mean": _mean(
            [run["receive"]["batch_count"] for run in runs]
        ),
    }


def _summarize_torch_runs(
    *,
    backend: str,
    device: str,
    size_bytes: int,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    send_elapsed = [run["send"]["elapsed_s"] for run in runs]
    receive_elapsed = [run["receive"]["elapsed_s"] for run in runs]
    driver_wall = [run["driver_wall_s"] for run in runs]
    refit_elapsed = [
        max(run["send"]["elapsed_s"], run["receive"]["elapsed_s"]) for run in runs
    ]
    send_bw = [run["send"]["logical_gib_per_s"] for run in runs]
    receive_bw = [run["receive"]["logical_gib_per_s"] for run in runs]
    return {
        "transport": backend,
        "device": device,
        "variant": backend,
        "size_gib": size_bytes / _GIB,
        "runs": len(runs),
        "send_elapsed_mean_s": _mean(send_elapsed),
        "send_elapsed_min_s": _min(send_elapsed),
        "send_elapsed_p50_s": _p50(send_elapsed),
        "receive_elapsed_mean_s": _mean(receive_elapsed),
        "receive_elapsed_min_s": _min(receive_elapsed),
        "receive_elapsed_p50_s": _p50(receive_elapsed),
        "refit_elapsed_mean_s": _mean(refit_elapsed),
        "refit_elapsed_min_s": _min(refit_elapsed),
        "refit_elapsed_p50_s": _p50(refit_elapsed),
        "driver_wall_mean_s": _mean(driver_wall),
        "send_logical_gib_per_s_mean": _mean(send_bw),
        "receive_logical_gib_per_s_mean": _mean(receive_bw),
    }


def _variant_payload(variant: Variant) -> dict[str, Any]:
    return {
        "name": variant.name,
        "transfer_mode": variant.transfer_mode,
        "direct_stripe_count": variant.direct_stripe_count,
        "metadata_batch_size": variant.metadata_batch_size,
        "buffer_count": variant.buffer_count,
    }


def _load_device_for(engine_device: str, args: argparse.Namespace) -> str:
    if engine_device == "cpu":
        return args.cpu_load_device
    return args.gpu_load_device


def _size_bytes_for(engine_device: str, args: argparse.Namespace) -> int:
    size_gib = args.cpu_size_gib if engine_device == "cpu" else args.gpu_size_gib
    return int(size_gib * _GIB)


def _backend_init_params_for_device(
    engine_device: str,
    args: argparse.Namespace,
    base_params: dict[str, str],
) -> dict[str, str]:
    params = dict(base_params)
    if args.backend_name == "UCX":
        params.setdefault("ucx_error_handling_mode", args.ucx_error_handling_mode)
        if engine_device == "cuda":
            if args.ucx_engine_config:
                params.setdefault("engine_config", args.ucx_engine_config)
            if args.ucx_device_list:
                params.setdefault("device_list", args.ucx_device_list)
    return params


def _make_endpoint(
    *,
    label: str,
    node: dict[str, Any],
    bucket_size_bytes: int,
    backend_name: str,
    backend_init_params: dict[str, str],
) -> Any:
    return RefitBenchmarkEndpoint.options(**_actor_options(node)).remote(
        label=label,
        bucket_size_bytes=bucket_size_bytes,
        backend_name=backend_name,
        backend_init_params=backend_init_params,
    )


def _run_variant(
    *,
    policy: Any,
    rollout: Any,
    device: str,
    load_device: str,
    variant: Variant,
    size_bytes: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    variant_dict = _variant_payload(variant)
    ray.get(
        [
            policy.configure.remote(
                device=device,
                load_device=device,
                variant=variant_dict,
            ),
            rollout.configure.remote(
                device=device,
                load_device=load_device,
                variant=variant_dict,
            ),
        ]
    )

    metadata = ray.get([policy.prepare.remote(), rollout.prepare.remote()])
    ray.get(
        [
            policy.init_policy.remote(metadata=metadata),
            rollout.init_rollout.remote(metadata=metadata),
        ]
    )

    fill_value = args.fill_value
    reset_value = (fill_value + 1) % 256
    allocation = ray.get(
        [
            policy.allocate_source.remote(
                size_bytes=size_bytes,
                fill_value=fill_value,
            ),
            rollout.allocate_destination.remote(
                size_bytes=size_bytes,
                fill_value=reset_value,
            ),
        ]
    )
    _emit(
        "REFIT_BENCH_ALLOCATION",
        {
            "device": device,
            "load_device": load_device,
            "variant": variant.name,
            "allocation": allocation,
        },
    )

    all_runs = []
    total_runs = args.warmup + args.repeats
    for run_index in range(total_runs):
        timed = run_index >= args.warmup
        run_context = {
            "device": device,
            "load_device": load_device,
            "variant": variant.name,
            "run_index": run_index,
            "timed": timed,
        }
        _emit("REFIT_BENCH_RUN_START", run_context)
        ray.get(
            rollout.reset_destination.remote(fill_value=reset_value),
            timeout=args.transfer_timeout_s,
        )
        _emit("REFIT_BENCH_RESET_DONE", run_context)
        driver_start = time.perf_counter()
        receive_ref = rollout.receive_once.remote(
            run_index=run_index,
            expected_bytes=size_bytes,
            expected_fill_value=fill_value,
        )
        send_ref = policy.send_once.remote(run_index=run_index)
        _emit("REFIT_BENCH_TRANSFER_WAIT", run_context)
        send_result, receive_result = ray.get(
            [send_ref, receive_ref],
            timeout=args.transfer_timeout_s,
        )
        driver_wall_s = time.perf_counter() - driver_start
        run_result = {
            "device": device,
            "load_device": load_device,
            "variant": variant.name,
            "run_index": run_index,
            "timed": timed,
            "driver_wall_s": driver_wall_s,
            "send": send_result,
            "receive": receive_result,
        }
        _emit("REFIT_BENCH_RUN", run_result)
        if timed:
            all_runs.append(run_result)

    return _summarize_runs(
        device=device,
        variant=variant,
        size_bytes=size_bytes,
        runs=all_runs,
    )


def _run_torch_baseline(
    *,
    policy: Any,
    rollout: Any,
    backend: str,
    device: str,
    size_bytes: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    policy_report = ray.get(policy.report.remote())
    master_port = ray.get(policy.reserve_port.remote())
    ray.get(
        [
            policy.configure_torch_process_group.remote(
                backend=backend,
                rank=0,
                world_size=2,
                master_addr=policy_report["node_ip"],
                master_port=master_port,
                device=device,
            ),
            rollout.configure_torch_process_group.remote(
                backend=backend,
                rank=1,
                world_size=2,
                master_addr=policy_report["node_ip"],
                master_port=master_port,
                device=device,
            ),
        ],
        timeout=args.transfer_timeout_s,
    )

    fill_value = args.fill_value
    reset_value = (fill_value + 1) % 256
    allocation = ray.get(
        [
            policy.allocate_source.remote(
                size_bytes=size_bytes,
                fill_value=fill_value,
            ),
            rollout.allocate_destination.remote(
                size_bytes=size_bytes,
                fill_value=reset_value,
            ),
        ]
    )
    _emit(
        "REFIT_BENCH_BASELINE_ALLOCATION",
        {
            "backend": backend,
            "device": device,
            "allocation": allocation,
        },
    )

    all_runs = []
    total_runs = args.warmup + args.repeats
    for run_index in range(total_runs):
        timed = run_index >= args.warmup
        run_context = {
            "backend": backend,
            "device": device,
            "run_index": run_index,
            "timed": timed,
        }
        _emit("REFIT_BENCH_BASELINE_RUN_START", run_context)
        ray.get(
            rollout.reset_destination.remote(fill_value=reset_value),
            timeout=args.transfer_timeout_s,
        )
        driver_start = time.perf_counter()
        send_ref = policy.torch_broadcast_once.remote(
            run_index=run_index,
            expected_fill_value=fill_value,
        )
        receive_ref = rollout.torch_broadcast_once.remote(
            run_index=run_index,
            expected_fill_value=fill_value,
        )
        send_result, receive_result = ray.get(
            [send_ref, receive_ref],
            timeout=args.transfer_timeout_s,
        )
        driver_wall_s = time.perf_counter() - driver_start
        run_result = {
            "backend": backend,
            "device": device,
            "run_index": run_index,
            "timed": timed,
            "driver_wall_s": driver_wall_s,
            "send": send_result,
            "receive": receive_result,
        }
        _emit("REFIT_BENCH_BASELINE_RUN", run_result)
        if timed:
            all_runs.append(run_result)

    ray.get(
        [
            policy.close_torch_process_group.remote(),
            rollout.close_torch_process_group.remote(),
        ]
    )
    return _summarize_torch_runs(
        backend=backend,
        device=device,
        size_bytes=size_bytes,
        runs=all_runs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark NIXL checkpoint-engine refit transfer modes."
    )
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--min-nodes", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--transfer-timeout-s", type=int, default=900)
    parser.add_argument("--backend-name", default="UCX")
    parser.add_argument("--bucket-mb", type=int, default=1024)
    parser.add_argument("--devices", default="cuda")
    parser.add_argument("--gpu-load-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cpu-load-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu-size-gib", type=float, default=8.0)
    parser.add_argument("--cpu-size-gib", type=float, default=4.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--fill-value", type=int, default=17)
    parser.add_argument(
        "--variants",
        default=(
            "staged:staged:1:1:3,"
            "direct:direct:1:1:3,"
            "direct_s2_b2:direct:2:2:3,"
            "direct_s4_b4_buf5:direct:4:4:5"
        ),
    )
    parser.add_argument(
        "--backend-init-param",
        action="append",
        default=[],
        help="Backend init parameter as key=value. May be repeated.",
    )
    parser.add_argument("--ucx-error-handling-mode", default="none")
    parser.add_argument("--ucx-engine-config", default="")
    parser.add_argument("--ucx-device-list", default="")
    parser.add_argument(
        "--torch-baselines",
        default="",
        help=(
            "Comma-separated torch distributed baselines to run. "
            "'nccl' is valid for cuda, 'gloo' is valid for cpu."
        ),
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("--warmup must be >= 0 and --repeats must be >= 1")
    if args.fill_value < 0 or args.fill_value > 255:
        raise ValueError("--fill-value must be in [0, 255]")

    devices = _parse_devices(args.devices)
    variants = _parse_variants(args.variants)
    torch_baselines = _parse_torch_baselines(args.torch_baselines)
    base_backend_init_params = _parse_backend_init_params(args.backend_init_param)
    bucket_size_bytes = args.bucket_mb * 1024 * 1024

    if args.ray_address == "local":
        ray.init()
    else:
        ray.init(address=args.ray_address)

    nodes = _wait_for_nodes(args.min_nodes, args.timeout_s)
    selected_nodes = nodes[: args.min_nodes]
    policy_node = selected_nodes[0]
    rollout_node = selected_nodes[1] if len(selected_nodes) > 1 else selected_nodes[0]
    _emit(
        "REFIT_BENCH_NODES",
        {
            "selected": [_node_label(node) for node in selected_nodes],
            "policy": _node_label(policy_node),
            "rollout": _node_label(rollout_node),
        },
    )

    summaries = []
    policy = None
    rollout = None
    try:
        for device in devices:
            backend_init_params = _backend_init_params_for_device(
                device,
                args,
                base_backend_init_params,
            )
            policy = _make_endpoint(
                label=f"policy-{device}",
                node=policy_node,
                bucket_size_bytes=bucket_size_bytes,
                backend_name=args.backend_name,
                backend_init_params=backend_init_params,
            )
            rollout = _make_endpoint(
                label=f"rollout-{device}",
                node=rollout_node,
                bucket_size_bytes=bucket_size_bytes,
                backend_name=args.backend_name,
                backend_init_params=backend_init_params,
            )
            reports = ray.get([policy.report.remote(), rollout.report.remote()])
            _emit(
                "REFIT_BENCH_ENDPOINTS",
                {
                    "device": device,
                    "backend_name": args.backend_name,
                    "backend_init_params": backend_init_params,
                    "reports": reports,
                },
            )

            load_device = _load_device_for(device, args)
            size_bytes = _size_bytes_for(device, args)
            for variant in variants:
                _emit(
                    "REFIT_BENCH_VARIANT_START",
                    {
                        "device": device,
                        "load_device": load_device,
                        "size_gib": size_bytes / _GIB,
                        "bucket_mb": args.bucket_mb,
                        "variant": _variant_payload(variant),
                    },
                )
                summary = _run_variant(
                    policy=policy,
                    rollout=rollout,
                    device=device,
                    load_device=load_device,
                    variant=variant,
                    size_bytes=size_bytes,
                    args=args,
                )
                summaries.append(summary)
                _emit("REFIT_BENCH_SUMMARY", summary)

            for baseline in torch_baselines:
                if baseline == "nccl" and device != "cuda":
                    _emit(
                        "REFIT_BENCH_BASELINE_SKIPPED",
                        {
                            "backend": baseline,
                            "device": device,
                            "reason": "NCCL requires CUDA tensors",
                        },
                    )
                    continue
                if baseline == "gloo" and device != "cpu":
                    _emit(
                        "REFIT_BENCH_BASELINE_SKIPPED",
                        {
                            "backend": baseline,
                            "device": device,
                            "reason": "CPU Gloo baseline only",
                        },
                    )
                    continue
                _emit(
                    "REFIT_BENCH_BASELINE_START",
                    {
                        "backend": baseline,
                        "device": device,
                        "size_gib": size_bytes / _GIB,
                    },
                )
                summary = _run_torch_baseline(
                    policy=policy,
                    rollout=rollout,
                    backend=baseline,
                    device=device,
                    size_bytes=size_bytes,
                    args=args,
                )
                summaries.append(summary)
                _emit("REFIT_BENCH_SUMMARY", summary)

            ray.get([policy.close.remote(), rollout.close.remote()])
            ray.kill(policy, no_restart=True)
            ray.kill(rollout, no_restart=True)
            policy = None
            rollout = None
    finally:
        cleanup_refs = []
        if policy is not None:
            cleanup_refs.append(policy.close.remote())
        if rollout is not None:
            cleanup_refs.append(rollout.close.remote())
        if cleanup_refs:
            ray.get(cleanup_refs)
        if policy is not None:
            ray.kill(policy, no_restart=True)
        if rollout is not None:
            ray.kill(rollout, no_restart=True)
        ray.shutdown()

    _emit("REFIT_BENCH_COMPLETED", {"summaries": summaries})


if __name__ == "__main__":
    main()
