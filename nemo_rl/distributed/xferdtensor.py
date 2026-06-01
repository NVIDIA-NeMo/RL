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

"""Reference XferDTensor implementation (broadcast-based golden).

This file contains only the transfer kernel and its private helpers.  The
lightweight wrapper types it expects on its inputs (``MeshInfo``,
``DTensorRef``) live in ``nccl_reshard_utils.py`` alongside the refit
metadata builders; ``xferdtensor_golden`` reads them only via duck typing
(``.mesh`` / ``._local_tensor``) so this module has no import dependency
on them.

The 7-argument signature of ``xferdtensor_golden`` matches the real
``nccl_reshard.XferDTensor`` API so it can be swapped in trivially later.
"""

import torch
from torch.distributed._tensor import Shard

# ===========================================================
#  NCCL-Xfer-reshard API call
# ===========================================================


def xferdtensor(
    src_tensor,
    src_mesh,
    src_placement,
    dst_tensor,
    dst_mesh,
    dst_placement,
    process_group,
):
    """Public XferDTensor entry point used by all external callers.

    For now this just forwards to the broadcast-based reference
    implementation (``xferdtensor_golden``).  Once the real low-level
    NCCL-XferReshard API is available, replace the body here to call into
    it instead — external callers will pick it up automatically without
    any further changes.
    """
    return xferdtensor_golden(
        src_tensor,
        src_mesh,
        src_placement,
        dst_tensor,
        dst_mesh,
        dst_placement,
        process_group,
    )


# ===========================================================
#  Functional reference implementation for NCCL-Xfer-reshard
# ===========================================================


def _flatten_mesh_ranks(mesh):
    mesh_tensor = getattr(mesh, "mesh", None)
    if mesh_tensor is None:
        mesh_tensor = getattr(mesh, "_mesh", None)
    if mesh_tensor is None:
        raise ValueError("DeviceMesh does not expose mesh ranks.")
    return [int(rank) for rank in mesh_tensor.flatten().tolist()]


def _get_tensor_meta(src_tensor, dst_tensor):
    if src_tensor is not None:
        return src_tensor.shape, src_tensor.device
    if dst_tensor is not None:
        return dst_tensor.shape, dst_tensor.device
    device = torch.device("cuda", torch.cuda.current_device())
    return None, device


def _get_mesh_coords(mesh, rank):
    mesh_tensor = getattr(mesh, "mesh", None)
    if mesh_tensor is None:
        mesh_tensor = getattr(mesh, "_mesh", None)
    if mesh_tensor is None:
        raise ValueError("DeviceMesh does not expose mesh ranks.")
    coords = (mesh_tensor == rank).nonzero(as_tuple=False)
    if coords.numel() == 0:
        return None
    return coords[0].tolist()


def _compute_shard_slices(global_shape, mesh_shape, mesh_coords, placements):
    slices = [slice(None) for _ in range(len(global_shape))]
    shard_map = {}
    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, Shard):
            shard_map.setdefault(placement.dim, []).append(
                (mesh_dim, mesh_shape[mesh_dim], mesh_coords[mesh_dim])
            )

    for tensor_dim, shard_info in shard_map.items():
        shard_info.sort(key=lambda item: item[0])
        num_chunks = 1
        for _, size, _ in shard_info:
            num_chunks *= size

        total_size = int(global_shape[tensor_dim])
        base = total_size // num_chunks
        remainder = total_size % num_chunks
        sizes = [base + 1 if i < remainder else base for i in range(num_chunks)]

        strides = []
        running = 1
        for _, size, _ in reversed(shard_info):
            strides.append(running)
            running *= size
        strides.reverse()

        linear_index = 0
        for (mesh_dim, size, coord), stride in zip(shard_info, strides):
            if coord >= size:
                raise ValueError(f"Invalid mesh coord {coord} for mesh dim {mesh_dim}.")
            linear_index += coord * stride

        start = sum(sizes[:linear_index])
        end = start + sizes[linear_index]
        slices[tensor_dim] = slice(start, end)

    return slices


def xferdtensor_golden(
    src_tensor,
    src_mesh,
    src_placement,
    dst_tensor,
    dst_mesh,
    dst_placement,
    process_group,
):
    """Broadcast-based reference implementation of XferDTensor.

    Reconstructs the full (global) tensor from TP-local shards on the source
    mesh, then each destination rank extracts its local shard based on
    ``dst_placement``.

    When all ``src_placement`` entries are ``Replicate``, a single broadcast
    from ``src_ranks[0]`` suffices.  Otherwise, one broadcast per unique shard
    region reconstructs the full tensor.

    This is the canonical 7-argument signature matching the real
    ``nccl_reshard.XferDTensor`` API.  Callers must wrap raw tensors in
    ``DTensorRef`` so that ``.shape`` reports the global shape and
    ``._local_tensor`` holds the local shard.
    """
    rank = process_group.rank
    src_ranks = _flatten_mesh_ranks(src_mesh)
    dst_ranks = _flatten_mesh_ranks(dst_mesh)

    global_shape, device = _get_tensor_meta(src_tensor, dst_tensor)
    if global_shape is None:
        raise ValueError("Unable to infer tensor shape/dtype from src or dst tensor.")
    dtype = src_tensor.dtype if src_tensor is not None else dst_tensor.dtype

    has_shard = any(isinstance(p, Shard) for p in src_placement)

    if not has_shard:
        # Fast path: all Replicate — single broadcast from src_ranks[0]
        if src_tensor is not None:
            full_tensor = src_tensor._local_tensor.clone()
        else:
            full_tensor = torch.empty(global_shape, device=device, dtype=dtype)
        process_group.broadcast(full_tensor.view(torch.uint8), src=src_ranks[0])
    else:
        # Reconstruct the full tensor from per-rank shards.
        # Deduplicate: DP-replicated ranks hold the same shard, so only one
        # representative rank broadcasts per unique shard region.
        full_tensor = torch.empty(global_shape, device=device, dtype=dtype)
        src_mesh_tensor = getattr(src_mesh, "mesh")
        mesh_shape = list(src_mesh_tensor.shape)

        seen_slices: dict[tuple, tuple] = {}
        for src_rank in src_ranks:
            coords = _get_mesh_coords(src_mesh, src_rank)
            shard_slices = _compute_shard_slices(
                global_shape, mesh_shape, coords, src_placement
            )
            slice_key = tuple((s.start, s.stop) for s in shard_slices)
            if slice_key not in seen_slices:
                seen_slices[slice_key] = (src_rank, shard_slices)

        for src_rank, shard_slices in seen_slices.values():
            shard_shape = tuple(
                (s.stop - s.start) if s.start is not None else global_shape[i]
                for i, s in enumerate(shard_slices)
            )
            shard_buf = torch.empty(shard_shape, device=device, dtype=dtype)
            if rank == src_rank and src_tensor is not None:
                shard_buf.copy_(src_tensor._local_tensor)
            process_group.broadcast(shard_buf.view(torch.uint8), src=src_rank)
            full_tensor[tuple(shard_slices)] = shard_buf

    # Destination side: extract local shard
    if rank in dst_ranks:
        mesh_coords = _get_mesh_coords(dst_mesh, rank)
        if mesh_coords is None:
            raise RuntimeError(f"Rank {rank} expected in dst_mesh but not found.")
        mesh_tensor = getattr(dst_mesh, "mesh")
        mesh_shape = list(mesh_tensor.shape)
        shard_slices = _compute_shard_slices(
            global_shape=full_tensor.shape,
            mesh_shape=mesh_shape,
            mesh_coords=mesh_coords,
            placements=dst_placement,
        )
        resharded_local = full_tensor[tuple(shard_slices)]
        if dst_tensor is not None:
            if hasattr(dst_tensor, "_local_tensor"):
                dst_tensor._local_tensor.copy_(resharded_local)
            else:
                dst_tensor_local = dst_tensor.to_local()
                dst_tensor_local.copy_(resharded_local)
    return
