# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Lightweight ModelExpress reshard endpoint configuration helpers."""

from __future__ import annotations

from typing import Any

DEFAULT_PUBLISHER_LISTEN_PORT_BASE = 5555
DEFAULT_RECEIVER_LISTEN_PORT_BASE = 15555
LEGACY_RECEIVER_PORT_OFFSET = 10000
MAX_TCP_PORT = 65535


def _get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def maybe_preinit_mx_reshard_nixl(config: Any) -> Any:
    """Load NIXL/UCX before model construction when MX reshard is selected.

    On GB200, loading Megatron first can leave the later NIXL agent without
    UCX's CUDA and InfiniBand components. Keep this agent alive for the worker
    lifetime so the production publisher can create its listening agent after
    the model and Bridge mappings exist.
    """
    generation = _get(config, "generation")
    if generation is None or _get(generation, "refit_transport") != "mx_reshard":
        return None

    from nemo_rl.utils.checkpoint_engines.nixl import preinit_nixl_agent

    return preinit_nixl_agent()


def resolve_mx_reshard_listen_port_bases(config: Any) -> tuple[int, int]:
    """Resolve distinct publisher/receiver bases, including the legacy fallback."""
    legacy = _get(config, "listen_port_base")
    publisher = _get(config, "publisher_listen_port_base")
    receiver = _get(config, "receiver_listen_port_base")

    if publisher is None:
        publisher = legacy if legacy is not None else DEFAULT_PUBLISHER_LISTEN_PORT_BASE
    if receiver is None:
        receiver = (
            int(legacy) + LEGACY_RECEIVER_PORT_OFFSET
            if legacy is not None
            else DEFAULT_RECEIVER_LISTEN_PORT_BASE
        )
    return int(publisher), int(receiver)


def resolve_mx_reshard_publisher_listen_port_base(config: Any) -> int:
    """Resolve the base passed only to trainer publishers."""
    return resolve_mx_reshard_listen_port_bases(config)[0]


def resolve_mx_reshard_receiver_listen_port_base(config: Any) -> int:
    """Resolve the base passed only to inference receivers."""
    return resolve_mx_reshard_listen_port_bases(config)[1]


def validate_mx_reshard_listen_port_ranges(
    config: Any,
    *,
    train_world_size: int,
    inference_world_size: int,
) -> tuple[int, int]:
    """Require valid, disjoint physical-rank port ranges before actor startup."""
    publisher, receiver = resolve_mx_reshard_listen_port_bases(config)
    if train_world_size <= 0 or inference_world_size <= 0:
        raise ValueError(
            "ModelExpress train and inference world sizes must be positive"
        )

    publisher_range = (publisher, publisher + train_world_size - 1)
    receiver_range = (receiver, receiver + inference_world_size - 1)
    for role, port_range in (
        ("publisher", publisher_range),
        ("receiver", receiver_range),
    ):
        if port_range[0] <= 0 or port_range[1] > MAX_TCP_PORT:
            raise ValueError(
                f"ModelExpress {role} listen port range {port_range} is outside "
                f"1..{MAX_TCP_PORT}"
            )
    if max(publisher_range[0], receiver_range[0]) <= min(
        publisher_range[1], receiver_range[1]
    ):
        raise ValueError(
            "ModelExpress publisher and receiver listen port ranges overlap: "
            f"publisher={publisher_range}, receiver={receiver_range}"
        )
    return publisher, receiver


__all__ = [
    "DEFAULT_PUBLISHER_LISTEN_PORT_BASE",
    "DEFAULT_RECEIVER_LISTEN_PORT_BASE",
    "maybe_preinit_mx_reshard_nixl",
    "resolve_mx_reshard_listen_port_bases",
    "resolve_mx_reshard_publisher_listen_port_base",
    "resolve_mx_reshard_receiver_listen_port_base",
    "validate_mx_reshard_listen_port_ranges",
]
