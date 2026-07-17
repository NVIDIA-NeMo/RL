# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Trainer-side helpers for rank-local ModelExpress publication."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger("nemo_rl.distributed.mx_helpers")


@dataclass
class MxConfig:
    """Configuration for trainer-side ModelExpress publication.

    Args:
        enabled: Master switch for ModelExpress publication.
        mx_server_url: gRPC URL of the MX server.
        same_rank_only: whether consumers are expected to use same-rank sources.
        nic_pin: NIC pinning strategy passed to ``pin_local_nic``:
            ``"auto"`` (default) | ``"off"`` | concrete ``"mlx5_<i>"``.
        megatron_role_overrides: Parameter-name substrings mapped to explicit
            ModelExpress Megatron roles.
    """

    enabled: bool = False
    mx_server_url: str = "modelexpress-server:8001"
    same_rank_only: bool = True
    nic_pin: str = "auto"
    megatron_role_overrides: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "MxConfig":
        if not d:
            return cls()
        return cls(
            enabled=bool(d.get("enabled", False)),
            mx_server_url=str(d.get("mx_server_url", "modelexpress-server:8001")),
            same_rank_only=bool(d.get("same_rank_only", True)),
            nic_pin=str(d.get("nic_pin", "auto")),
            megatron_role_overrides={
                str(name): str(role)
                for name, role in (d.get("megatron_role_overrides") or {}).items()
            },
        )


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


def pin_local_nic(*, device_id: int, mode: str = "auto") -> None:
    """Best-effort NUMA-local NIC pinning before NIXL initializes.

    Automatic mode delegates topology selection to ModelExpress. A concrete
    device name sets the UCX interface explicitly.
    """
    if mode == "off":
        return
    try:
        from modelexpress.ucx_utils import apply_nic_pin_for_device

        if mode == "auto":
            apply_nic_pin_for_device(device_id=device_id)
            logger.info("pinned NIC for device %d (auto)", device_id)
        else:
            os.environ["UCX_NET_DEVICES"] = mode
            os.environ["MX_RDMA_NIC_PIN"] = "off"  # explicit override
            logger.info("pinned NIC explicitly: %s", mode)
    except Exception as exc:  # noqa: BLE001
        logger.warning("NIC pin failed (mode=%s): %s", mode, exc)


def build_v2_publisher(
    *,
    rank: int,
    device_id: int,
    fsdp_world_size: int,
    tp_world_size: int,
    pp_world_size: int,
    ep_world_size: int,
    mx_config: MxConfig,
    agent_name: str | None = None,
) -> Any:
    """Construct a :class:`MxV2TrainingPublisher` and pin its NIC.

    Returns a :class:`modelexpress.MxV2TrainingPublisher`. Caller must invoke
    ``initialize(model_name=...)``, then ``add_tensor`` per tensor, then
    ``publish(version=...)``, then ``mark_ready()``.
    """
    from modelexpress import MxV2TrainingPublisher, TrainerWorldLayout

    pin_local_nic(device_id=device_id, mode=mx_config.nic_pin)

    return MxV2TrainingPublisher(
        agent_name=agent_name or f"nemo-rl-trainer-r{rank}",
        device_id=device_id,
        mx_server_url=mx_config.mx_server_url,
        worker_rank=rank,
        world_layout=TrainerWorldLayout(
            fsdp_world_size=fsdp_world_size,
            tp_world_size=tp_world_size,
            pp_world_size=pp_world_size,
            ep_world_size=ep_world_size,
        ),
        heartbeat=True,
    )


def reset_v2_publisher_tensors(publisher: Any) -> None:
    """Reset one publisher's per-version tensor registrations.

    Prefer the public lifecycle method when provided by ModelExpress. The
    private-field fallback supports the currently released v2 publisher and
    is isolated here so it can be removed when that release is retired.
    """
    reset_tensors = getattr(publisher, "reset_tensors", None)
    if callable(reset_tensors):
        reset_tensors()
        return

    registry = getattr(publisher, "_registry", None)
    registered_tensors = getattr(publisher, "_registered_tensors", None)
    if registry is None or registered_tensors is None:
        raise RuntimeError(
            "unsupported ModelExpress publisher: expected reset_tensors()"
        )
    registry.clear()
    registered_tensors.clear()


__all__ = [
    "MxConfig",
    "build_v2_publisher",
    "get_dtensor_local_shard",
    "pin_local_nic",
    "reset_v2_publisher_tensors",
]
