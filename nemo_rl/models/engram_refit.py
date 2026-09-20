# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded row payloads for owner-sharded DeepSeek V4.1 Engram tables."""

from collections.abc import Iterator

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

_ROW_MARKER = ".__nrl_engram_rows_"
_CHUNK_BYTES = 64 * 1024 * 1024


def row_chunks(
    name: str, tensor: torch.Tensor, dtype: torch.dtype
) -> Iterator[tuple[str, int, int, int]]:
    """Yield wire name, owner rank, global start, and length without gathering."""
    if tensor.ndim != 2 or not name.endswith(".engram.embed.weight"):
        raise ValueError(f"Not an Engram row table: {name} {tensor.shape}")
    owners = 1
    if isinstance(tensor, DTensor):
        if tensor.device_mesh.ndim != 1 or tensor.placements != (Shard(0),):
            raise ValueError("Engram refit requires a one-dimensional Shard(0) mesh")
        owners = tensor.device_mesh.size()
    rows, columns = tensor.shape
    owner_rows = (rows + owners - 1) // owners
    chunk_rows = max(1, _CHUNK_BYTES // (columns * dtype.itemsize))
    for owner in range(owners):
        for start in range(
            owner * owner_rows, min((owner + 1) * owner_rows, rows), chunk_rows
        ):
            count = min(chunk_rows, min((owner + 1) * owner_rows, rows) - start)
            yield f"{name}{_ROW_MARKER}{start}_{count}", owner, start, count


def stream_rows(
    name: str, tensor: torch.Tensor, dtype: torch.dtype
) -> Iterator[tuple[str, torch.Tensor]]:
    """Replicate one bounded owner slice at a time for existing refit transports."""
    distributed = isinstance(tensor, DTensor)
    local = tensor.to_local() if distributed else tensor
    mesh = tensor.device_mesh if distributed else None
    rank = mesh.get_local_rank() if mesh is not None else 0
    owners = mesh.size() if mesh is not None else 1
    owner_rows = (tensor.shape[0] + owners - 1) // owners
    for wire_name, owner, start, count in row_chunks(name, tensor, dtype):
        if owner == rank:
            offset = start - owner * owner_rows
            payload = local[offset : offset + count].to(dtype).contiguous()
            if payload.shape[0] != count:
                raise ValueError(
                    f"Engram owner {owner} is missing rows for {wire_name}"
                )
        else:
            payload = torch.empty(
                (count, tensor.shape[1]), dtype=dtype, device=local.device
            )
        if mesh is not None:
            dist.broadcast(
                payload, src=int(mesh.mesh[owner].item()), group=mesh.get_group()
            )
        yield wire_name, payload


def parse_row_name(name: str) -> tuple[str, int, int] | None:
    """Decode an internal row payload, rejecting malformed ranges."""
    if _ROW_MARKER not in name:
        return None
    base, encoded = name.rsplit(_ROW_MARKER, 1)
    start, count = (int(value) for value in encoded.split("_"))
    if not base.endswith(".engram.embed.weight") or start < 0 or count <= 0:
        raise ValueError(f"Invalid Engram row payload name: {name}")
    return base, start, count
