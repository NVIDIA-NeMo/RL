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

"""Demonstrate NIXL checkpoint-engine refit after rollout actors change.

The current NIXL checkpoint engine pairs policy rank ``i`` with rollout rank
``i``. This demo creates one policy endpoint for every possible rollout rank,
then adds and removes rollout endpoints between phases. Each phase exchanges
fresh NIXL metadata, rebuilds the process groups, performs a full synthetic
weight refit, and verifies the active rollout endpoints before continuing.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class EndpointHandle:
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


def _alive_nodes() -> list[dict[str, Any]]:  # pragma: no cover
    import ray

    nodes = [node for node in ray.nodes() if node.get("Alive")]
    return sorted(nodes, key=lambda node: node["NodeManagerAddress"])


def _wait_for_nodes(
    min_nodes: int, timeout_s: int
) -> list[dict[str, Any]]:  # pragma: no cover
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
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

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


def _phase_delta(
    active_ranks: Iterable[int], target_count: int
) -> tuple[list[int], list[int]]:
    current = set(active_ranks)
    target = set(range(target_count))
    return sorted(target - current), sorted(current - target)


def _build_engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "device": args.device,
        "backend_name": args.nixl_backend_name,
        "backend_init_params": {
            "ucx_error_handling_mode": args.ucx_error_handling_mode,
        },
        "cleanup_after_load": False,
    }


def _tensor_summary(
    weights: Iterable[tuple[str, "torch.Tensor"]],
) -> dict[str, dict[str, Any]]:
    import torch

    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sum": int(tensor.detach().cpu().sum(dtype=torch.int64).item()),
        }
        for name, tensor in weights
    }


class ElasticNixlEndpoint:  # pragma: no cover
    def __init__(
        self,
        *,
        label: str,
        rank: int,
        backend: str,
        bucket_size_bytes: int,
        engine_kwargs: dict[str, Any],
        device: str,
    ) -> None:
        import ray
        import torch

        from nemo_rl.utils.checkpoint_engines.base import create_checkpoint_engine

        self.label = label
        self.rank = rank
        self.hostname = os.uname().nodename
        self.node_ip = ray.util.get_node_ip_address().strip("[]")
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(0)
            torch.empty(1, device="cuda").fill_(1)
            torch.cuda.synchronize()

        self.engine = create_checkpoint_engine(
            backend,
            bucket_size_bytes=bucket_size_bytes,
            engine_kwargs=engine_kwargs,
        )

    def report(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "rank": self.rank,
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
            worker_rank=self.rank,
            train_world_size=train_world_size,
            rollout_world_size=rollout_world_size,
            metadata=metadata,
        )

    def init_rollout(
        self,
        *,
        metadata: list[Any],
        train_world_size: int,
        rollout_world_size: int,
    ) -> None:
        self.engine.init_rollout_process_group(
            rollout_rank=self.rank,
            train_world_size=train_world_size,
            rollout_world_size=rollout_world_size,
            metadata=metadata,
        )

    def finalize(self) -> None:
        self.engine.finalize()

    def close(self) -> None:
        self.engine.finalize()
        disconnect = getattr(self.engine, "_disconnect_peers", None)
        if callable(disconnect):
            disconnect()

    def send_weights(self, *, phase: int, tensor_mb: int) -> dict[str, Any]:
        import torch

        element_count = tensor_mb * 1024 * 1024 // torch.int32.itemsize
        base_value = phase * 1000 + self.rank
        weights = [
            (
                "dense.weight",
                torch.arange(
                    element_count,
                    dtype=torch.int32,
                    device=self.device,
                )
                + base_value,
            ),
            (
                "router.weight",
                torch.full(
                    (4096,),
                    base_value,
                    dtype=torch.int32,
                    device=self.device,
                ),
            ),
        ]
        expected = _tensor_summary(weights)

        async def run() -> None:
            await self.engine.send_weights(iter(weights))

        start = time.time()
        asyncio.run(run())
        return {
            "label": self.label,
            "rank": self.rank,
            "phase": phase,
            "elapsed_s": time.time() - start,
            "expected": expected,
        }

    def receive_weights(self, *, phase: int) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            received: dict[str, Any] = {}
            async for batch in self.engine.receive_weight_batches():
                received.update(_tensor_summary(batch))
            return received

        start = time.time()
        received = asyncio.run(run())
        return {
            "label": self.label,
            "rank": self.rank,
            "phase": phase,
            "elapsed_s": time.time() - start,
            "received": received,
        }


_REMOTE_ENDPOINT_CLASS: Any | None = None


def _remote_endpoint_class() -> Any:  # pragma: no cover
    global _REMOTE_ENDPOINT_CLASS
    if _REMOTE_ENDPOINT_CLASS is None:
        import ray

        _REMOTE_ENDPOINT_CLASS = ray.remote(ElasticNixlEndpoint)
    return _REMOTE_ENDPOINT_CLASS


def _new_endpoint(
    *,
    label: str,
    rank: int,
    node: dict[str, Any],
    backend: str,
    bucket_size_bytes: int,
    engine_kwargs: dict[str, Any],
    device: str,
    use_gpu: bool,
) -> Any:  # pragma: no cover
    return (
        _remote_endpoint_class()
        .options(**_actor_options(node, use_gpu=use_gpu))
        .remote(
            label=label,
            rank=rank,
            backend=backend,
            bucket_size_bytes=bucket_size_bytes,
            engine_kwargs=engine_kwargs,
            device=device,
        )
    )


def _validate_results(
    *,
    expected_by_rank: dict[int, dict[str, Any]],
    rollout_results: list[dict[str, Any]],
) -> None:
    for result in rollout_results:
        rank = result["rank"]
        expected = expected_by_rank.get(rank)
        if expected is None:
            raise RuntimeError(f"Received result for unexpected rollout rank {rank}.")
        if result["received"] != expected:
            raise RuntimeError(
                "Rollout result mismatch:\n"
                f"rank={rank}\n"
                f"expected={json.dumps(expected, sort_keys=True)}\n"
                f"actual={json.dumps(result, sort_keys=True)}"
            )


def _parse_args() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate NIXL checkpoint-engine peer rebuilds while rollout "
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
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = _parse_args()
    import ray

    rollout_sequence = _parse_sequence(args.rollout_sequence)
    policy_count = max(rollout_sequence)
    if policy_count > args.min_nodes:
        raise ValueError(
            "This demo maps endpoint ranks across the reserved nodes; "
            f"max rollout count {policy_count} exceeds --min-nodes={args.min_nodes}."
        )

    ray.init(address=args.ray_address)
    nodes = _wait_for_nodes(args.min_nodes, args.timeout_s)[: args.min_nodes]
    print(
        "ELASTIC_NIXL_DEMO_NODES " + json.dumps([_node_label(node) for node in nodes]),
        flush=True,
    )

    use_gpu = args.device == "cuda"
    bucket_size_bytes = args.bucket_mb * 1024 * 1024
    engine_kwargs = _build_engine_kwargs(args)

    policy_handles: list[EndpointHandle] = []
    active_rollouts: dict[int, EndpointHandle] = {}
    rollout_generations: dict[int, int] = {}

    try:
        for rank in range(policy_count):
            node = nodes[rank % len(nodes)]
            actor = _new_endpoint(
                label=f"policy-{rank}",
                rank=rank,
                node=node,
                backend=args.backend,
                bucket_size_bytes=bucket_size_bytes,
                engine_kwargs=engine_kwargs,
                device=args.device,
                use_gpu=use_gpu,
            )
            policy_handles.append(
                EndpointHandle(
                    rank=rank,
                    generation=1,
                    node_id=node["NodeID"],
                    node_ip=node["NodeManagerAddress"],
                    actor=actor,
                )
            )
        print(
            "ELASTIC_NIXL_POLICIES "
            + json.dumps(
                ray.get([handle.actor.report.remote() for handle in policy_handles]),
                sort_keys=True,
            ),
            flush=True,
        )

        for phase, target_count in enumerate(rollout_sequence, start=1):
            added_ranks, removed_ranks = _phase_delta(active_rollouts, target_count)

            for rank in removed_ranks:
                handle = active_rollouts.pop(rank)
                ray.get(handle.actor.close.remote())
                ray.kill(handle.actor, no_restart=True)

            for rank in added_ranks:
                generation = rollout_generations.get(rank, 0) + 1
                rollout_generations[rank] = generation
                node = nodes[(rank + 1) % len(nodes)]
                actor = _new_endpoint(
                    label=f"rollout-{rank}-gen-{generation}",
                    rank=rank,
                    node=node,
                    backend=args.backend,
                    bucket_size_bytes=bucket_size_bytes,
                    engine_kwargs=engine_kwargs,
                    device=args.device,
                    use_gpu=use_gpu,
                )
                active_rollouts[rank] = EndpointHandle(
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

            policy_metadata = ray.get(
                [handle.actor.prepare.remote() for handle in policy_handles]
            )
            rollout_metadata = ray.get(
                [handle.actor.prepare.remote() for handle in active]
            )
            metadata = policy_metadata + rollout_metadata
            train_world_size = len(policy_handles)
            rollout_world_size = len(active)
            ray.get(
                [
                    handle.actor.init_policy.remote(
                        metadata=metadata,
                        train_world_size=train_world_size,
                        rollout_world_size=rollout_world_size,
                    )
                    for handle in policy_handles
                ]
            )
            ray.get(
                [
                    handle.actor.init_rollout.remote(
                        metadata=metadata,
                        train_world_size=train_world_size,
                        rollout_world_size=rollout_world_size,
                    )
                    for handle in active
                ]
            )

            receive_refs = [
                handle.actor.receive_weights.remote(phase=phase) for handle in active
            ]
            send_refs = [
                handle.actor.send_weights.remote(
                    phase=phase,
                    tensor_mb=args.tensor_mb,
                )
                for handle in policy_handles
            ]
            send_results = ray.get(send_refs)
            rollout_results = ray.get(receive_refs)
            expected_by_rank = {
                result["rank"]: result["expected"]
                for result in send_results
                if result["rank"] < rollout_world_size
            }
            _validate_results(
                expected_by_rank=expected_by_rank,
                rollout_results=rollout_results,
            )

            ray.get([handle.actor.finalize.remote() for handle in policy_handles])
            ray.get([handle.actor.finalize.remote() for handle in active])

            print(
                "ELASTIC_NIXL_RESULT "
                + json.dumps(
                    {
                        "phase": phase,
                        "rollout_count": len(active),
                        "added": added_ranks,
                        "removed": removed_ranks,
                        "send_elapsed_s": [
                            round(result["elapsed_s"], 6) for result in send_results
                        ],
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
        for handle in [*policy_handles, *active_rollouts.values()]:
            with contextlib.suppress(Exception):
                ray.get(handle.actor.close.remote())
            with contextlib.suppress(Exception):
                ray.kill(handle.actor, no_restart=True)
        ray.shutdown()

    print("ELASTIC_NIXL_DEMO_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
