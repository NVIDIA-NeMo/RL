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

"""XferDTensor: cross-mesh DTensor reshard for disaggregated refit."""

import logging
import os
from contextlib import nullcontext

import torch
from torch.distributed._tensor import Shard

try:
    from nccl.m2n import (  # pyrefly: ignore[import-error]
        reshard as _reshard,
    )
except ImportError:
    # Containers without the nccl4py M2N integration don't ship the real
    # reshard op; xferdtensor() then defaults to the Python exact-transfer
    # implementation (xferdtensor_python).
    _reshard = None
    logging.warning("nccl.m2n.reshard not found, defaulting to xferdtensor_python")

# Log the selected reshard path once per process (real op / python / golden).
_XFERDTENSOR_PATH_LOGGED = False


class DTensorRef:
    """DTensor-compatible reference for xferdtensor.

    Provides the interface xferdtensor reads via duck typing:
    - ``.shape``: global tensor shape (torch.Size)
    - ``._local_tensor``: local shard (src side) or dst buffer (dst side)
    - ``.dtype``, ``.device``: tensor metadata

    On the **src side** (train), ``local_tensor`` is the TP-local shard
    from Megatron parameters (no PP broadcast or TP gather needed).
    ``global_shape`` is the full unsharded shape.

    On the **dst side** (gen), ``local_tensor`` is either the vLLM local
    parameter (for direct params) or a temporary buffer (for merged/unmapped
    params). ``global_shape`` is always the full unsharded shape.
    """

    def __init__(
        self, local_tensor: torch.Tensor, global_shape, dtype=None, device=None
    ):
        self._local_tensor = local_tensor
        self.shape = (
            torch.Size(global_shape)
            if not isinstance(global_shape, torch.Size)
            else global_shape
        )
        self.dtype = dtype if dtype is not None else local_tensor.dtype
        self.device = device if device is not None else local_tensor.device


# ===========================================================
#  NCCL M2N reshard API call
# ===========================================================


def _use_golden_api() -> bool:
    """Whether ``NRL_XFERDTENSOR_GOLDEN`` forces the golden reshard path."""
    return os.environ.get("NRL_XFERDTENSOR_GOLDEN", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _use_python_api() -> bool:
    """Use Python only implementation in for the xferdtensor.

    ``xferdtensor_python_impl`` is the backup implementation when the
    nccl.m2n.reshard is not available.
    """
    return os.environ.get("NRL_XFERDTENSOR_PYTHON", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def xferdtensor(
    src_tensor,
    src_mesh,
    src_placement,
    dst_tensor,
    dst_mesh,
    dst_placement,
    process_group,
    stream=None,
):
    """Public XferDTensor entry point used by all external callers."""
    global _XFERDTENSOR_PATH_LOGGED
    use_golden = _use_golden_api()
    use_python = _use_python_api()
    if not _XFERDTENSOR_PATH_LOGGED:
        if use_golden:
            path = "golden (broadcast)"
        elif use_python or _reshard is None:
            path = "xferdtensor_python (exact-transfer)"
        else:
            path = "real nccl.m2n.reshard"
        print(
            f"[xferdtensor] reshard path: {path} "
            f"(real_op_available={_reshard is not None}, "
            f"force_golden={use_golden}, force_python={use_python})",
            flush=True,
        )
        _XFERDTENSOR_PATH_LOGGED = True

    if not use_golden and (use_python or _reshard is None):
        # Default when the real op is absent: Python exact-transfer reshard
        # (per-overlap P2P + cached split-comm replica broadcast); same
        # 7-arg contract.
        from nemo_rl.weight_sync.xferdtensor_python import (
            xferdtensor_python_impl,
        )

        return xferdtensor_python_impl(
            src_tensor,
            src_mesh,
            src_placement,
            dst_tensor,
            dst_mesh,
            dst_placement,
            process_group,
            stream,
        )

    if use_golden:
        stream_ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with stream_ctx:
            return xferdtensor_golden(
                src_tensor,
                src_mesh,
                src_placement,
                dst_tensor,
                dst_mesh,
                dst_placement,
                process_group,
            )

    src_local = src_tensor._local_tensor if src_tensor is not None else None
    dst_local = dst_tensor._local_tensor if dst_tensor is not None else None

    reshard_kwargs = {}
    if stream is not None:
        reshard_kwargs["stream"] = int(stream.cuda_stream)
    _reshard(  # pyrefly: ignore[not-callable]
        src_local,
        dst_local,
        process_group.nccl_communicator,
        src_mesh=src_mesh.mesh.tolist(),
        src_placements=src_placement,
        dst_mesh=dst_mesh.mesh.tolist(),
        dst_placements=dst_placement,
        **reshard_kwargs,
    )


# ===========================================================
#  Functional reference implementation for NCCL-Reshard
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
    """Return the slice of the global tensor this rank owns."""
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

        # linear_index = this rank's flat chunk number among num_chunks.
        # (example: tp coord 2 * stride 1 -> linear_index = 2.)
        linear_index = 0
        for (mesh_dim, size, coord), stride in zip(shard_info, strides):
            if coord >= size:
                raise ValueError(f"Invalid mesh coord {coord} for mesh dim {mesh_dim}.")
            linear_index += coord * stride

        # This rank owns chunk `linear_index`; its slice starts past every
        # earlier chunk. (example: start = 64+64 = 128, end = 192 -> [128:192].)
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
    ``nccl.m2n.reshard`` op.  Callers must wrap raw tensors in ``DTensorRef`` so
    that ``.shape`` reports the global shape and ``._local_tensor`` holds the
    local shard.
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
            full_tensor = src_tensor._local_tensor.contiguous().clone()
        else:
            full_tensor = torch.empty(global_shape, device=device, dtype=dtype)
        process_group.broadcast(full_tensor.view(torch.uint8), src=src_ranks[0])
    else:
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
            dst_tensor._local_tensor.copy_(resharded_local)
    return
