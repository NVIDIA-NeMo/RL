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

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypedDict

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class RoutingPolicyConfig(TypedDict, total=False):
    """Configuration for routing policy selection.

    Attributes:
        type: Policy type - "round_robin" (default) or "cache_aware".
        router_class: Fully qualified class name for the external router (cache_aware only).
        router_kwargs: Keyword arguments passed to the external router constructor.
    """

    type: str
    router_class: str
    router_kwargs: dict[str, Any]


@dataclass
class RoutingDecision:
    """Result of a routing policy decision.

    Attributes:
        dp_shard_idx: Index of the selected data parallel shard.
        worker_idx: Global worker index to route the request to.
    """

    dp_shard_idx: int
    worker_idx: int


class ExternalRouter(ABC):
    """Abstract base class that external routers must subclass for cache-aware routing.

    dp_leader_worker_indices (mapping dp_shard_idx -> leader worker global index)
    is provided at construction time since it is fixed for the lifetime of the
    worker group. Its length equals dp_size.
    """

    @abstractmethod
    def __init__(
        self,
        dp_leader_worker_indices: list[int],
        generation_cfg: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Initialize the external router.

        Args:
            dp_leader_worker_indices: Maps dp_shard_idx -> leader worker global index.
                Its length equals dp_size.
            generation_cfg: The full generation config dict (includes vllm_cfg, etc.).
            **kwargs: Additional keyword arguments from router_kwargs config.
        """

    @abstractmethod
    def route(self, data: BatchedDataDict, request_id: str) -> RoutingDecision:
        """Select a DP shard and worker for routing a generation batch.

        Args:
            data: The generation input batch.
            request_id: Unique ID for lifecycle tracking (same ID passed to
                prefill_complete and generation_complete).

        Returns:
            RoutingDecision with dp_shard_idx and worker_idx.
        """

    @abstractmethod
    def prefill_complete(self, request_id: str) -> None:
        """Called when prefill is complete for request_id."""

    @abstractmethod
    def generation_complete(self, request_id: str) -> None:
        """Called when generation is complete for request_id."""

    @abstractmethod
    def weights_updated(self) -> None:
        """Called when the model weights have been updated."""


class RoutingPolicy(ABC):
    """Abstract base class for routing policies that select a DP shard and worker.

    dp_leader_worker_indices is provided at construction time since it is fixed
    for the lifetime of the worker group.
    """

    @abstractmethod
    def select_worker(
        self,
        *,
        data: BatchedDataDict,
        request_id: str,
    ) -> RoutingDecision:
        """Select a DP shard and worker for routing a generation batch.

        Args:
            data: The generation input batch.
            request_id: Unique ID for lifecycle tracking.

        Returns:
            RoutingDecision with dp_shard_idx and worker_idx.
        """

    def on_prefill_complete(self, request_id: str) -> None:
        """Called when prefill is complete. Default no-op."""

    def on_generation_complete(self, request_id: str) -> None:
        """Called when generation is complete. Default no-op."""

    def on_weights_updated(self) -> None:
        """Called when model weights have been updated. Default no-op."""


class RoundRobinRoutingPolicy(RoutingPolicy):
    """Round-robin routing across data parallel shards.

    Cycles through DP shards in order, always selecting the leader worker
    for the chosen shard. This reproduces the original hardcoded behavior.
    """

    def __init__(self, dp_leader_worker_indices: list[int]) -> None:
        self._dp_leader_worker_indices = dp_leader_worker_indices
        self._counter: int = 0

    def select_worker(
        self,
        *,
        data: BatchedDataDict,
        request_id: str,
    ) -> RoutingDecision:
        dp_shard_idx = self._counter % len(self._dp_leader_worker_indices)
        self._counter += 1
        return RoutingDecision(
            dp_shard_idx=dp_shard_idx,
            worker_idx=self._dp_leader_worker_indices[dp_shard_idx],
        )


class CacheAwareRoutingPolicy(RoutingPolicy):
    """Cache-aware routing that delegates to an external router.

    The external router is expected to expose:
    - route(data) -> RoutingDecision
    - prefill_complete(request_id)
    - generation_complete(request_id)
    """

    def __init__(self, external_router: ExternalRouter, dp_leader_worker_indices: list[int]) -> None:
        self._external_router = external_router
        self._dp_leader_worker_indices = dp_leader_worker_indices

    def select_worker(
        self,
        *,
        data: BatchedDataDict,
        request_id: str,
    ) -> RoutingDecision:
        decision = self._external_router.route(data, request_id)

        dp_size = len(self._dp_leader_worker_indices)
        if not (0 <= decision.dp_shard_idx < dp_size):
            raise ValueError(
                f"External router returned dp_shard_idx={decision.dp_shard_idx}, "
                f"expected range [0, {dp_size})"
            )

        num_workers = max(self._dp_leader_worker_indices) + 1 if self._dp_leader_worker_indices else 0
        if not (0 <= decision.worker_idx < num_workers):
            raise ValueError(
                f"External router returned worker_idx={decision.worker_idx}, "
                f"expected range [0, {num_workers})"
            )

        return decision

    def on_prefill_complete(self, request_id: str) -> None:
        self._external_router.prefill_complete(request_id)

    def on_generation_complete(self, request_id: str) -> None:
        self._external_router.generation_complete(request_id)

    def on_weights_updated(self) -> None:
        self._external_router.weights_updated()


def create_routing_policy(
    config: RoutingPolicyConfig | None,
    dp_leader_worker_indices: list[int],
    generation_cfg: dict[str, Any] | None = None,
) -> RoutingPolicy:
    """Factory function to create a routing policy from config.

    Args:
        config: Routing policy configuration. None or missing 'type' defaults to round_robin.
        dp_leader_worker_indices: Maps dp_shard_idx -> leader worker global index.
            Its length equals dp_size.
        generation_cfg: The full generation config dict, passed to external routers.

    Returns:
        An instantiated RoutingPolicy.
    """
    if config is None:
        return RoundRobinRoutingPolicy(dp_leader_worker_indices)

    policy_type = config.get("type", "round_robin")

    if policy_type == "round_robin":
        return RoundRobinRoutingPolicy(dp_leader_worker_indices)
    elif policy_type == "cache_aware":
        router_class_path = config.get("router_class")
        if not router_class_path:
            raise ValueError("CacheAwareRoutingPolicy requires 'router_class' in config")

        module_path, class_name = router_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        router_cls = getattr(module, class_name)

        if not issubclass(router_cls, ExternalRouter):
            raise TypeError(
                f"router_class {router_class_path!r} must be a subclass of "
                f"nemo_rl.models.generation.vllm.routing_policy.ExternalRouter"
            )

        router_kwargs = config.get("router_kwargs", {})
        external_router = router_cls(
            dp_leader_worker_indices=dp_leader_worker_indices,
            generation_cfg=generation_cfg or {},
            **router_kwargs,
        )

        return CacheAwareRoutingPolicy(external_router, dp_leader_worker_indices)
    else:
        raise ValueError(f"Unknown routing policy type: {policy_type!r}")
