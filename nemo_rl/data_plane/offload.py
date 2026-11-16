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
"""CPU Ray actors that do data-plane work for another process.

An ``OffloadClient`` is a CPU-only Ray actor with its own data-plane client.
A subclass overrides ``process(request)``: read, compute and write as it needs.
An ``OffloadPool`` starts N of them with one placement and hands out free ones.
Each user starts its own pool; pools are never shared.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_rl.data_plane.adapters.tq_mooncake_checkpoint import run_checkpoint_command
from nemo_rl.data_plane.interfaces import DataPlaneConfig
from nemo_rl.data_plane.observability import is_metrics_client

# Maps an actor's index in the pool to its Ray scheduling strategy.
Placement = Callable[[int], Any]


class OffloadClient:
    """Own a connect-only data-plane client and do one job on it.

    ``owns_storage`` decides whether this process lends memory to a Mooncake
    store. Rows written through a client that owns storage live in this
    process, so a data-plane checkpoint must reach it (``mooncake_checkpoint``).
    A client that owns none (``segment_size=0``) writes into other processes'
    storage and needs no checkpoint hook.
    """

    owns_storage: ClassVar[bool] = False

    def __init__(self, dp_config: DataPlaneConfig, *, label: str = "0") -> None:
        from nemo_rl.data_plane import build_data_plane_client

        self.label = label
        self.dp_client = build_data_plane_client(
            dp_config,
            bootstrap=False,
            segment_size=None if self.owns_storage else 0,
        )

    def process(self, request: Any) -> Any:
        """Do this client's job for one request."""
        raise NotImplementedError

    def __repr__(self) -> str:
        # Ray prefixes this actor's log lines with its repr.
        return f"{type(self).__name__}[{self.label}]"

    def mooncake_checkpoint(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """Save or restore the rows this process holds; metadata only."""
        return run_checkpoint_command(body) if self.owns_storage else None

    def get_data_plane_snapshot(self) -> dict[str, Any] | None:
        """This client's data-plane I/O counters, if metrics are on."""
        if not is_metrics_client(self.dp_client):
            return None
        return self.dp_client.snapshot(reset_step_window=True)


def on_this_node() -> Placement:
    """Pin every actor to the caller's node; fail rather than land elsewhere."""
    node_id = ray.get_runtime_context().get_node_id()
    return lambda _: NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)


def spread_off_head() -> Placement:
    """Round-robin over the nodes that are not a dedicated Ray head.

    The head under ray.sub's DEDICATED_RAY_HEAD=1 hosts the controller and is
    compute-free. CPU-only actors that ask for plain CPUs would otherwise be
    packed onto the driver's node, the head. Soft, because a hard pin waits
    forever on a node that cannot fit ``num_cpus=1``. Without a dedicated head
    there is no node to single out, so actors spread.
    """
    if ray.cluster_resources().get("ray_head", 0) < 1:
        return lambda _: "SPREAD"
    node_ids = sorted(
        node["NodeID"]
        for node in ray.nodes()
        if node.get("Alive") and not node.get("Resources", {}).get("ray_head")
    )
    if not node_ids:
        return lambda _: "SPREAD"
    return lambda index: NodeAffinitySchedulingStrategy(
        node_id=node_ids[index % len(node_ids)], soft=True
    )


class OffloadPool:
    """A fixed set of offload actors, and which of them are free.

    ``acquire`` waits for a free actor; ``release`` gives it back after its call
    finished. A caller does not release an actor whose call raised: its
    outcome is unknown, so it is never handed out again.
    """

    def __init__(self, actors: Sequence[Any]) -> None:
        self.actors = list(actors)
        self._free: asyncio.Queue[Any] = asyncio.Queue()
        for actor in self.actors:
            self._free.put_nowait(actor)

    @classmethod
    def start(
        cls,
        client_cls: Any,
        *args: Any,
        n: int,
        placement: Placement | None = None,
        **ray_options: Any,
    ) -> OffloadPool:
        """Create ``n`` CPU-only, non-restarting actors; do not wait for them.

        ``client_cls`` is an ``OffloadClient`` subclass, plain or already
        wrapped by ``@ray.remote``. Each actor gets its index as ``label``.
        """
        if n <= 0:
            raise ValueError(f"an offload pool needs at least one actor, got {n}")
        remote_cls = (
            client_cls if hasattr(client_cls, "options") else ray.remote(client_cls)
        )
        options = {
            "num_cpus": 1,
            "num_gpus": 0,
            "max_restarts": 0,
            "max_task_retries": 0,
            **ray_options,
        }
        actors = [
            remote_cls.options(
                **options,
                **(
                    {}
                    if placement is None
                    else {"scheduling_strategy": placement(index)}
                ),
            ).remote(*args, label=str(index))
            for index in range(n)
        ]
        return cls(actors)

    def wait_ready(
        self, method: str = "__ray_ready__", *, timeout_s: float | None = None
    ) -> list[Any]:
        """Block until every actor answers ``method``; kill all if one fails."""
        try:
            return ray.get(
                [getattr(actor, method).remote() for actor in self.actors],
                timeout=timeout_s,
            )
        except BaseException:
            self.kill()
            raise

    async def wait_ready_async(
        self, method: str = "__ray_ready__", *, timeout_s: float | None = None
    ) -> list[Any]:
        """``wait_ready`` for a caller that must not block its event loop."""
        try:
            return await asyncio.wait_for(
                asyncio.gather(
                    *(getattr(actor, method).remote() for actor in self.actors)
                ),
                timeout=timeout_s,
            )
        except BaseException:
            self.kill()
            raise

    @property
    def size(self) -> int:
        return len(self.actors)

    @property
    def free(self) -> int:
        return self._free.qsize()

    async def acquire(self) -> Any:
        """Wait until an actor is free and take it."""
        return await self._free.get()

    def release(self, actor: Any) -> None:
        """Give back an actor whose call finished, successfully or not at all."""
        self._free.put_nowait(actor)

    def kill(self) -> None:
        """Kill every actor; errors from already-dead actors are reported."""
        for actor in self.actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception as error:
                print(f"offload actor cleanup failed: {error}", flush=True)
