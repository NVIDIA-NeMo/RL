# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""ModelExpress + NIXL RDMA weight-sync helpers for NemoRL.

This module wires the v2 MX integration (`pensieve/RL/NemoRL/04_design_v2_moe_rank_to_rank.md`)
into NemoRL's existing ColocatablePolicyInterface / GenerationInterface.

Design pillars (see also the design doc):

  1. Rank-to-rank publish — each trainer rank publishes its assigned tensor
     buffers and same-rank inference peers receive directly via NIXL RDMA. This
     is the lesson from PrimeRL's live debug on GB200 (May 6, 2026), now applied
     as the default.

  2. Tree fan-out — receivers republish themselves as `inference_replica`
     sources after a successful refit. Subsequent receivers can pull from
     them rather than contending on the trainer's NIC. (TensorHub paper
     pipeline replication, arXiv 2604.09107v1 §4.3.3.)

  3. MoE expert filtering — when ``owned_experts_per_layer`` is set on the
     trainer, the receiver pulls only the experts its EP rank actually uses,
     skipping the rest. Compatible with Composer 2's router-replay strategy
     (Cursor, 2026).

  4. Explicit shape registry — the trainer publishes a JSON registry under
     ``SourceIdentity.extra_parameters["shape_registry"]`` so the receiver
     knows the exact placement of every tensor. Versioned per training
     step.

Heartbeats are wired up automatically via :class:`modelexpress.HeartbeatThread`
so MX state stays clean across orchestrator restarts.

Public surface (all symbols also accessible as ``nemo_rl.distributed.mx_helpers.*``):

  - :class:`MxConfig` — config dataclass plumbed through ``cfg.cluster.weight_sync``.
  - :func:`build_v2_publisher` — convenience constructor for trainer side.
  - :func:`build_v2_receiver` — convenience constructor for inference side.
  - :func:`pin_local_nic` — best-effort NUMA-local NIC pinning before NIXL init.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger("nemo_rl.distributed.mx_helpers")


@dataclass
class MxConfig:
    """Configuration for the MX path (matches ``cfg.cluster.weight_sync``).

    Args:
        enabled: master switch. When False, refit_policy_generation falls back
            to NCCL collective.
        mx_server_url: gRPC URL of the MX server.
        timeout_seconds: max wait for source discovery / RDMA receive.
        same_rank_only: if True (default, recommended for GB200/EFA), restrict
            transfers to (trainer rank N → inference rank N) pairs. Required
            for multi-subnet RDMA fabrics where cross-NIC writes are unrouted.
        tree_scale_out: if True, inference workers republish themselves as
            additional sources after receiving. Subsequent rank-N receivers
            (cold starts, restarted replicas) can pull from peers instead of
            re-funneling through the trainer.
        moe_expert_filter: if True, receivers request only the expert shards
            their EP rank owns. Requires the trainer to set per-layer
            ``owned_expert_ids``.
        nic_pin: NIC pinning strategy passed to ``pin_local_nic``:
            ``"auto"`` (default) | ``"off"`` | concrete ``"mlx5_<i>"``.
    """

    enabled: bool = False
    mx_server_url: str = "modelexpress-server:8001"
    timeout_seconds: float = 300.0
    same_rank_only: bool = True
    tree_scale_out: bool = True
    moe_expert_filter: bool = True
    nic_pin: str = "auto"

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "MxConfig":
        if not d:
            return cls()
        return cls(
            enabled=bool(d.get("enabled", False)),
            mx_server_url=str(d.get("mx_server_url", "modelexpress-server:8001")),
            timeout_seconds=float(d.get("timeout_seconds", 300.0)),
            same_rank_only=bool(d.get("same_rank_only", True)),
            tree_scale_out=bool(d.get("tree_scale_out", True)),
            moe_expert_filter=bool(d.get("moe_expert_filter", True)),
            nic_pin=str(d.get("nic_pin", "auto")),
        )


def pin_local_nic(*, device_id: int, mode: str = "auto") -> None:
    """Best-effort NUMA-local NIC pinning before NIXL initializes.

    On GCP GB200 the four ``mlx5_N`` NICs are each on their own L3 subnet, so
    cross-NIC (and therefore cross-rank) writes are unrouted. To work around
    this, we want each rank's NIXL agent to bind to the NIC NUMA-closest to
    its GPU. The MX client already implements this via
    ``modelexpress.ucx_utils.apply_nic_pin_for_device``. We just call it
    here with the same args NemoRL would use.
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


def build_v2_receiver(
    *,
    rank: int,
    device_id: int,
    mx_config: MxConfig,
    agent_name: str | None = None,
) -> Any:
    """Construct a :class:`MxV2RefitReceiver` and pin its NIC."""
    from modelexpress import MxV2RefitReceiver

    pin_local_nic(device_id=device_id, mode=mx_config.nic_pin)

    return MxV2RefitReceiver(
        agent_name=agent_name or f"nemo-rl-inference-r{rank}",
        device_id=device_id,
        mx_server_url=mx_config.mx_server_url,
        worker_rank=rank,
    )


def detect_moe_expert_layout(
    model: "torch.nn.Module",
    *,
    ep_world_size: int,
    rank: int,
) -> dict[str, tuple[int, set[int]]]:
    """Best-effort detection of MoE expert tensors and which experts each rank owns.

    Returns ``{tensor_name: (expert_axis, owned_expert_ids)}`` for tensors
    that look like expert weights (heuristic: name contains ``"experts"`` and
    the leading axis size is divisible by ``ep_world_size``).

    Customize via the ``NRL_MX_EXPERT_TENSOR_PATTERN`` env var if your model
    uses different naming.
    """
    if ep_world_size <= 1:
        return {}

    pattern = os.environ.get("NRL_MX_EXPERT_TENSOR_PATTERN", "experts")
    out: dict[str, tuple[int, set[int]]] = {}
    for name, tensor in model.state_dict().items():
        if pattern not in name:
            continue
        if tensor.ndim < 2:
            continue
        # The leading axis is conventionally the expert axis. Confirm
        # divisibility by ep_world_size.
        leading = tensor.shape[0]
        if leading % ep_world_size != 0:
            continue
        chunk = leading // ep_world_size
        owned = set(range(rank * chunk, (rank + 1) * chunk))
        out[name] = (0, owned)
    return out


__all__ = [
    "MxConfig",
    "build_v2_publisher",
    "build_v2_receiver",
    "detect_moe_expert_layout",
    "pin_local_nic",
]
