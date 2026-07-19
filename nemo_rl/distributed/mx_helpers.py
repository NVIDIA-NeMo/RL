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
"""Trainer-side helpers for rank-local ModelExpress publication."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import torch

logger = logging.getLogger("nemo_rl.distributed.mx_helpers")


@dataclass(frozen=True)
class ModelExpressPublisherOptions:
    """Internal settings supplied by a ModelExpress weight synchronizer.

    Args:
        mx_server_url: gRPC URL of the MX server.
        nic_pin: NIC pinning strategy passed to ``pin_local_nic``:
            ``"auto"`` | ``"off"`` | concrete ``"mlx5_<i>"``.
        megatron_role_overrides: Parameter-name substrings mapped to explicit
            ModelExpress Megatron roles.

    This is not a user-facing configuration schema. The future
    ``ModelExpressWeightSynchronizer`` will validate user configuration and
    construct these settings explicitly.
    """

    mx_server_url: str
    nic_pin: str
    megatron_role_overrides: Mapping[str, str]


class ModelExpressPublisher(Protocol):
    """Publisher operations used by NeMo RL trainer workers."""

    def initialize(self, *, model_name: str, dtype: str) -> None: ...

    def reset_tensors(self) -> None: ...

    def set_megatron_sidecar(self, sidecar: dict[str, Any]) -> None: ...

    def set_megatron_mesh_position(
        self, *, tp_rank: int, pp_rank: int, ep_rank: int
    ) -> None: ...

    def add_tensor(
        self,
        *,
        name: str,
        tensor: "torch.Tensor",
        is_expert: bool = False,
        expert_axis: int = 0,
        owned_expert_ids: tuple[int, ...] | set[int] | list[int] = (),
        megatron_role: str | None = None,
        megatron_extras: dict[str, str] | None = None,
        shard_spec: Any | None = None,
    ) -> None: ...

    def publish(self, *, version: int) -> str: ...

    def mark_ready(self) -> bool: ...


@dataclass(frozen=True)
class DTensorShardSpec:
    """ModelExpress-compatible metadata for one materialized DTensor shard."""

    global_shape: tuple[int, ...]
    shard_axis: int
    local_shard_range: tuple[int, int]


def get_dtensor_local_shard(
    tensor: Any,
) -> tuple["torch.Tensor", DTensorShardSpec | None]:
    """Materialize a DTensor's local buffer and describe its single shard axis.

    ModelExpress currently represents one shard axis per tensor. Replicated
    DTensors are supported, while partial or multi-axis sharding fails before
    publication rather than falling back to an all-gather.
    """
    from torch.distributed.tensor import Partial, Replicate, Shard

    local = tensor.to_local()
    placements = tuple(tensor.placements)
    if any(isinstance(placement, Partial) for placement in placements):
        raise NotImplementedError("ModelExpress does not support partial DTensors")

    sharded_mesh_dims = [
        (mesh_dim, placement)
        for mesh_dim, placement in enumerate(placements)
        if isinstance(placement, Shard)
    ]
    if not sharded_mesh_dims:
        if not all(isinstance(placement, Replicate) for placement in placements):
            raise NotImplementedError(f"unsupported DTensor placements: {placements!r}")
        return local, None
    if len(sharded_mesh_dims) != 1:
        raise NotImplementedError(
            "ModelExpress supports one DTensor shard axis per tensor; "
            f"got placements={placements!r}"
        )

    mesh_dim, placement = sharded_mesh_dims[0]
    coordinate = tensor.device_mesh.get_coordinate()
    if coordinate is None:
        raise RuntimeError("current rank is not part of the DTensor device mesh")

    axis = int(placement.dim)
    global_shape = tuple(int(size) for size in tensor.shape)
    world_size = int(tensor.device_mesh.size(mesh_dim))
    rank = int(coordinate[mesh_dim])
    chunk_size = (global_shape[axis] + world_size - 1) // world_size
    start = min(rank * chunk_size, global_shape[axis])
    local_extent = int(local.shape[axis])
    end = start + local_extent
    if end > global_shape[axis]:
        raise ValueError(
            f"local shard range ({start}, {end}) exceeds global axis "
            f"size {global_shape[axis]}"
        )

    return local, DTensorShardSpec(
        global_shape=global_shape,
        shard_axis=axis,
        local_shard_range=(start, end),
    )


def pin_local_nic(*, device_id: int, mode: str) -> None:
    """Configure the requested NIC policy before NIXL initializes.

    Automatic mode delegates topology selection to ModelExpress. A concrete
    device name sets the UCX interface explicitly.
    """
    if mode == "off":
        return

    # Imported only when MX is selected so normal NeMo RL imports do not
    # require the external modelexpress package.
    from modelexpress.ucx_utils import apply_nic_pin_for_device

    if mode == "auto":
        apply_nic_pin_for_device(device_id=device_id)
        logger.info("pinned NIC for device %d (auto)", device_id)
    else:
        os.environ["UCX_NET_DEVICES"] = mode
        os.environ["MX_RDMA_NIC_PIN"] = "off"
        logger.info("pinned NIC explicitly: %s", mode)


def build_v2_publisher(
    *,
    rank: int,
    device_id: int,
    fsdp_world_size: int,
    tp_world_size: int,
    pp_world_size: int,
    ep_world_size: int,
    publisher_options: ModelExpressPublisherOptions,
    agent_name: str | None = None,
) -> ModelExpressPublisher:
    """Construct a :class:`MxV2TrainingPublisher` and pin its NIC.

    Returns a :class:`modelexpress.MxV2TrainingPublisher`. Caller must invoke
    ``initialize(model_name=...)``, then ``add_tensor`` per tensor, then
    ``publish(version=...)``, then ``mark_ready()``.
    """
    # Imported only when MX is selected so normal NeMo RL imports do not
    # require the external modelexpress package.
    from modelexpress import MxV2TrainingPublisher, TrainerWorldLayout

    pin_local_nic(device_id=device_id, mode=publisher_options.nic_pin)

    return MxV2TrainingPublisher(
        agent_name=agent_name or f"nemo-rl-trainer-r{rank}",
        device_id=device_id,
        mx_server_url=publisher_options.mx_server_url,
        worker_rank=rank,
        world_layout=TrainerWorldLayout(
            fsdp_world_size=fsdp_world_size,
            tp_world_size=tp_world_size,
            pp_world_size=pp_world_size,
            ep_world_size=ep_world_size,
        ),
        heartbeat=True,
    )


def start_model_express_publication(publisher: ModelExpressPublisher) -> None:
    """Clear per-version tensor registrations before adding the next version."""
    publisher.reset_tensors()


def finish_model_express_publication(
    publisher: ModelExpressPublisher,
    *,
    version: int,
    worker_rank: int,
) -> str:
    """Publish one complete version and require a successful READY transition."""
    source_id = publisher.publish(version=version)
    if not publisher.mark_ready():
        raise RuntimeError(
            f"ModelExpress failed to mark trainer rank {worker_rank} ready"
        )
    return source_id


__all__ = [
    "ModelExpressPublisher",
    "ModelExpressPublisherOptions",
    "build_v2_publisher",
    "finish_model_express_publication",
    "get_dtensor_local_shard",
    "pin_local_nic",
    "start_model_express_publication",
]
