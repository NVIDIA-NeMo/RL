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
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_rl.utils.checkpoint_engine import create_checkpoint_engine


@dataclass(frozen=True)
class RolloutHandle:
    rank: int
    generation: int
    node_id: str
    node_ip: str
    actor: Any


def _env_subset() -> dict[str, str]:
    prefixes = ("UCX_", "NIXL_", "MELLANOX_", "NVIDIA_")
    names = {
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    }
    return {
        key: value
        for key, value in os.environ.items()
        if key in names or key.startswith(prefixes)
    }


def _node_label(node: dict[str, Any]) -> str:
    return f"{node['NodeManagerAddress']}:{node['NodeID'][:8]}"


def _alive_nodes() -> list[dict[str, Any]]:
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    return sorted(nodes, key=lambda node: node["NodeManagerAddress"])


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


def _actor_options(node: dict[str, Any], *, use_gpu: bool) -> dict[str, Any]:
    return {
        "num_cpus": 1,
        "num_gpus": 1 if use_gpu else 0,
        "runtime_env": {"env_vars": _env_subset()},
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=node["NodeID"],
            soft=False,
        ),
    }


def _parse_sequence(value: str) -> list[int]:
    sequence = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sequence:
        raise ValueError("rollout sequence must contain at least one count")
    if min(sequence) < 1:
        raise ValueError("rollout counts must be positive")
    return sequence


@ray.remote
class ElasticNixlEndpoint:
    def __init__(
        self,
        *,
        label: str,
        backend: str,
        bucket_size_bytes: int,
        engine_kwargs: dict[str, Any],
        default_device: str,
    ) -> None:
        self.label = label
        self.hostname = os.uname().nodename
        self.node_ip = ray.util.get_node_ip_address().strip("[]")
        self.device = torch.device(default_device)
        if self.device.type == "cuda":
            torch.cuda.set_device(0)
            torch.empty(1, device="cuda").fill_(1)
            torch.cuda.synchronize()

        self.engine = create_checkpoint_engine(
            backend,
            bucket_size_bytes=bucket_size_bytes,
            engine_kwargs=engine_kwargs,
            default_device=self.device,
        )

    def report(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "hostname": self.hostname,
            "node_ip": self.node_ip,
            "pid": os.getpid(),
            "device": str(self.device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ucx_net_devices": os.environ.get("UCX_NET_DEVICES"),
            "ucx_tls": os.environ.get("UCX_TLS"),
            "ucx_max_rndv_rails": os.environ.get("UCX_MAX_RNDV_RAILS"),
        }

    def prepare(self) -> Any:
        return self.engine.prepare()

    def init_policy(
        self,
        *,
        metadata: list[Any],
        train_world_size: int,
        rollout_world_size: int,
    ) -> None:
        self.engine.init_policy_process_group(
            worker_rank=0,
            train_world_size=train_world_size,
            rollout_world_size=rollout_world_size,
            metadata=metadata,
        )

    def init_rollout(
        self,
        *,
        rollout_rank: int,
        metadata: list[Any],
        train_world_size: int,
        rollout_world_size: int,
    ) -> None:
        self.engine.init_rollout_process_group(
            rollout_rank=rollout_rank,
            train_world_size=train_world_size,
            rollout_world_size=rollout_world_size,
            metadata=metadata,
        )

    def finalize(self) -> None:
        self.engine.finalize()

    def close(self) -> None:
        self.engine.close()

    def send_weights(self, *, phase: int, tensor_mb: int) -> dict[str, Any]:
        element_count = tensor_mb * 1024 * 1024 // torch.int32.itemsize
        weights = [
            (
                "dense.weight",
                torch.arange(
                    element_count,
                    dtype=torch.int32,
                    device=self.device,
                )
                + phase * 1000,
            ),
            (
                "router.weight",
                torch.full(
                    (4096,),
                    phase,
                    dtype=torch.int32,
                    device=self.device,
                ),
            ),
        ]
        expected = {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sum": int(tensor.detach().cpu().sum(dtype=torch.int64).item()),
            }
            for name, tensor in weights
        }

        async def run() -> None:
            await self.engine.send_weights(iter(weights))

        start = time.time()
        asyncio.run(run())
        return {
            "label": self.label,
            "phase": phase,
            "elapsed_s": time.time() - start,
            "expected": expected,
        }

    def receive_weights(self, *, phase: int) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            received: dict[str, Any] = {}
            async for batch in self.engine.receive_weight_batches():
                for name, tensor in batch:
                    cpu_tensor = tensor.detach().cpu()
                    received[name] = {
                        "shape": list(cpu_tensor.shape),
                        "dtype": str(cpu_tensor.dtype),
                        "sum": int(cpu_tensor.sum(dtype=torch.int64).item()),
                    }
            return received

        start = time.time()
        received = asyncio.run(run())
        return {
            "label": self.label,
            "phase": phase,
            "elapsed_s": time.time() - start,
            "received": received,
        }


def _new_endpoint(
    *,
    label: str,
    node: dict[str, Any],
    backend: str,
    bucket_size_bytes: int,
    engine_kwargs: dict[str, Any],
    default_device: str,
    use_gpu: bool,
) -> Any:
    return ElasticNixlEndpoint.options(**_actor_options(node, use_gpu=use_gpu)).remote(
        label=label,
        backend=backend,
        bucket_size_bytes=bucket_size_bytes,
        engine_kwargs=engine_kwargs,
        default_device=default_device,
    )


def _validate_results(
    *,
    expected: dict[str, Any],
    rollout_results: list[dict[str, Any]],
) -> None:
    for result in rollout_results:
        if result["received"] != expected:
            raise RuntimeError(
                "Rollout result mismatch:\n"
                f"expected={json.dumps(expected, sort_keys=True)}\n"
                f"actual={json.dumps(result, sort_keys=True)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate NIXL checkpoint-engine topology rebuilds while rollout "
            "actors are added and removed."
        )
    )
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--min-nodes", type=int, default=4)
    parser.add_argument("--rollout-sequence", default="1,3,2,4,1")
    parser.add_argument("--backend", default="nixl")
    parser.add_argument("--nixl-backend-name", default="UCX")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--bucket-mb", type=int, default=64)
    parser.add_argument("--tensor-mb", type=int, default=96)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument(
        "--ucx-error-handling-mode",
        default="none",
        help="NIXL UCX backend ucx_error_handling_mode.",
    )
    args = parser.parse_args()

    rollout_sequence = _parse_sequence(args.rollout_sequence)
    if max(rollout_sequence) > args.min_nodes:
        raise ValueError(
            "This demo maps rollout ranks across the reserved nodes; "
            f"max rollout count {max(rollout_sequence)} exceeds "
            f"--min-nodes={args.min_nodes}."
        )

    ray.init(address=args.ray_address)
    nodes = _wait_for_nodes(args.min_nodes, args.timeout_s)
    print(
        "ELASTIC_NIXL_DEMO_NODES "
        + json.dumps([_node_label(node) for node in nodes[: args.min_nodes]]),
        flush=True,
    )

    use_gpu = args.device == "cuda"
    bucket_size_bytes = args.bucket_mb * 1024 * 1024
    engine_kwargs = {
        "device": args.device,
        "backend_name": args.nixl_backend_name,
        "backend_init_params": {
            "ucx_error_handling_mode": args.ucx_error_handling_mode,
        },
        "cleanup_after_load": False,
    }

    policy = _new_endpoint(
        label="policy-0",
        node=nodes[0],
        backend=args.backend,
        bucket_size_bytes=bucket_size_bytes,
        engine_kwargs=engine_kwargs,
        default_device=args.device,
        use_gpu=use_gpu,
    )
    print(
        "ELASTIC_NIXL_POLICY " + json.dumps(ray.get(policy.report.remote())), flush=True
    )

    active_rollouts: dict[int, RolloutHandle] = {}
    rollout_generations: dict[int, int] = {}

    try:
        for phase, target_count in enumerate(rollout_sequence, start=1):
            current_ranks = set(active_rollouts)
            target_ranks = set(range(target_count))
            removed_ranks = sorted(current_ranks - target_ranks)
            added_ranks = sorted(target_ranks - current_ranks)

            for rank in removed_ranks:
                handle = active_rollouts.pop(rank)
                ray.get(handle.actor.close.remote())
                ray.kill(handle.actor, no_restart=True)

            for rank in added_ranks:
                generation = rollout_generations.get(rank, 0) + 1
                rollout_generations[rank] = generation
                node = nodes[rank % len(nodes)]
                actor = _new_endpoint(
                    label=f"rollout-{rank}-gen-{generation}",
                    node=node,
                    backend=args.backend,
                    bucket_size_bytes=bucket_size_bytes,
                    engine_kwargs=engine_kwargs,
                    default_device=args.device,
                    use_gpu=use_gpu,
                )
                active_rollouts[rank] = RolloutHandle(
                    rank=rank,
                    generation=generation,
                    node_id=node["NodeID"],
                    node_ip=node["NodeManagerAddress"],
                    actor=actor,
                )

            active = [active_rollouts[rank] for rank in sorted(active_rollouts)]
            reports = ray.get([handle.actor.report.remote() for handle in active])
            print(
                "ELASTIC_NIXL_PHASE "
                + json.dumps(
                    {
                        "phase": phase,
                        "target_rollouts": target_count,
                        "added": added_ranks,
                        "removed": removed_ranks,
                        "active": [handle.rank for handle in active],
                        "reports": reports,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

            policy_metadata = ray.get(policy.prepare.remote())
            rollout_metadata = ray.get(
                [handle.actor.prepare.remote() for handle in active]
            )
            metadata = [policy_metadata] + rollout_metadata
            ray.get(
                policy.init_policy.remote(
                    metadata=metadata,
                    train_world_size=1,
                    rollout_world_size=len(active),
                )
            )
            ray.get(
                [
                    handle.actor.init_rollout.remote(
                        rollout_rank=index,
                        metadata=metadata,
                        train_world_size=1,
                        rollout_world_size=len(active),
                    )
                    for index, handle in enumerate(active)
                ]
            )

            receive_refs = [
                handle.actor.receive_weights.remote(phase=phase) for handle in active
            ]
            send_ref = policy.send_weights.remote(
                phase=phase,
                tensor_mb=args.tensor_mb,
            )
            send_result = ray.get(send_ref)
            rollout_results = ray.get(receive_refs)
            _validate_results(
                expected=send_result["expected"],
                rollout_results=rollout_results,
            )

            finalize_refs = [policy.finalize.remote()] + [
                handle.actor.finalize.remote() for handle in active
            ]
            ray.get(finalize_refs)

            print(
                "ELASTIC_NIXL_RESULT "
                + json.dumps(
                    {
                        "phase": phase,
                        "rollout_count": len(active),
                        "added": added_ranks,
                        "removed": removed_ranks,
                        "send_elapsed_s": round(send_result["elapsed_s"], 6),
                        "receive_elapsed_s": [
                            round(result["elapsed_s"], 6) for result in rollout_results
                        ],
                        "status": "ok",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    finally:
        cleanup_refs = [policy.close.remote()]
        cleanup_refs.extend(
            handle.actor.close.remote() for handle in active_rollouts.values()
        )
        ray.get(cleanup_refs)
        for handle in active_rollouts.values():
            ray.kill(handle.actor, no_restart=True)
        ray.kill(policy, no_restart=True)
        ray.shutdown()

    print("ELASTIC_NIXL_DEMO_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
