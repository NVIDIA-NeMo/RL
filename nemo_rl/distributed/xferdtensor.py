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

"""XferDTensor: cross-mesh DTensor reshard for disaggregated refit.

The public entry point ``xferdtensor()`` dispatches to the real ``nccl.xfer``
reshard op (``XdtensorRedistribute``) when it is available and not overridden
by ``NRL_XFERDTENSOR_GOLDEN``, otherwise to the broadcast-based
``xferdtensor_golden`` reference in this file.  Both share the canonical
7-argument signature, so callers are identical on either path.

This file also holds the golden kernel's private helpers and the
``DTensorRef`` wrapper callers pass as the src/dst tensor.  The ``MeshInfo``
mesh wrapper lives in ``nccl_xfer_utils.py`` alongside the refit metadata
builders; ``xferdtensor_golden`` reads it only via duck typing (``.mesh``), so
this module needs no import from there.
"""

import logging
import os

import torch
from torch.distributed._tensor import Shard

try:
    from nccl.xfer.api import (
        XdtensorRedistribute as _XdtensorRedistribute,
    )  # pyrefly: ignore[import-error]
except ImportError:
    # Golden-only containers (e.g. nemo_rl.v0.6.0 without the nccl4py xfer
    # integration) don't ship the real reshard op; xferdtensor() then falls
    # back to the broadcast-based golden implementation below.
    _XdtensorRedistribute = None
    logging.warning(
        "XdtensorRedistribute not found, falling back to golden implementation"
    )

# Log the selected reshard path once per process (real op vs golden fallback).
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
#  NCCL-Xfer-reshard API call
# ===========================================================


def _use_golden_api() -> bool:
    """Whether ``NRL_XFERDTENSOR_GOLDEN`` forces the golden reshard path.

    Set ``NRL_XFERDTENSOR_GOLDEN=1`` to force golden — e.g. to A/B against the
    real op, or for a config the real op doesn't yet support.  When the real op
    is simply absent (golden-only container) xferdtensor() also falls back to
    golden automatically.
    """
    return os.environ.get("NRL_XFERDTENSOR_GOLDEN", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _to_xfer_mesh(mesh):
    """Build an ``nccl.xfer.mesh.Mesh`` from a MeshInfo/DeviceMesh rank grid.

    ``Mesh.from_ranks`` derives the 2-D mesh dims from the grid shape (a 1-D
    grid maps to dims ``(N, 1)`` like PyTorch) and validates the row-major
    contiguous rank interval ``ncclXferMesh_t`` requires.  Passing the grid
    (not a flattened list) preserves the 2-D (e.g. DP x TP) structure the
    placements index into.
    """
    from nccl.xfer.mesh import Mesh  # pyrefly: ignore[import-error]

    rank_tensor = getattr(mesh, "mesh", None)
    if rank_tensor is None:
        rank_tensor = getattr(mesh, "_mesh", None)
    if rank_tensor is None:
        raise ValueError("mesh does not expose its rank tensor (.mesh/._mesh)")
    return Mesh.from_ranks(rank_tensor)


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

    Calls the real ``nccl.xfer`` reshard op (``XdtensorRedistribute``) when it
    is available and not overridden by ``NRL_XFERDTENSOR_GOLDEN``; otherwise
    forwards to the broadcast-based golden reference.  External callers are
    unchanged either way.

    Each rank holds only ONE side (train ranks own the src shard, gen ranks the
    dst shard), so ``src_tensor`` or ``dst_tensor`` is ``None`` on every rank;
    the real op accepts a one-sided ``None`` and derives the absent side's local
    shape from the present side's global shape + mesh/placements.  PyTorch
    Shard/Replicate placements are passed through directly.
    """
    global _XFERDTENSOR_PATH_LOGGED
    use_golden = _use_golden_api() or _XdtensorRedistribute is None
    if not _XFERDTENSOR_PATH_LOGGED:
        path = (
            "golden (broadcast)"
            if use_golden
            else "real nccl.xfer XdtensorRedistribute"
        )
        print(
            f"[xferdtensor] reshard path: {path} "
            f"(real_op_available={_XdtensorRedistribute is not None}, "
            f"force_golden={_use_golden_api()})",
            flush=True,
        )
        _XFERDTENSOR_PATH_LOGGED = True

    if use_golden:
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

    _XdtensorRedistribute(  # pyrefly: ignore[not-callable]
        src_local,
        dst_local,
        process_group.nccl_communicator,
        src_mesh=_to_xfer_mesh(src_mesh),  # pyrefly: ignore[bad-argument-type]
        src_placements=src_placement,
        dst_mesh=_to_xfer_mesh(dst_mesh),  # pyrefly: ignore[bad-argument-type]
        dst_placements=dst_placement,
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
    """Return, per tensor dim, the slice of the GLOBAL tensor this rank owns.

    Given a device mesh and DTensor ``placements`` (one per mesh dim), work out
    which contiguous block of the full tensor the rank at ``mesh_coords`` holds.
    Replicated tensor dims get ``slice(None)`` (the whole extent); sharded dims
    get this rank's chunk.

    Args:
        global_shape: full (unsharded) tensor shape, e.g. ``(256, 512)``.
        mesh_shape:   device-mesh shape, one size per mesh dim, e.g. ``[2, 4]``.
        mesh_coords:  this rank's coordinate on each mesh dim, e.g. ``[1, 2]``.
        placements:   one placement per mesh dim, e.g. ``[Replicate(), Shard(0)]``.

    Worked example — a 2-D ``[dp=2, tp=4]`` mesh, a weight of shape
    ``(256, 512)``, placements ``[Replicate(), Shard(0)]`` (DP replicates, TP
    shards tensor dim 0), this rank at ``mesh_coords=[1, 2]`` (dp=1, tp=2):
      * mesh dim 0 (dp) is Replicate -> ignored; tensor dim 1 stays slice(None).
      * mesh dim 1 (tp, size 4) shards tensor dim 0 -> 4 chunks of 256/4 = 64.
      * this rank's tp coord is 2 -> it owns chunk 2 -> rows [128:192].
      * result: ``[slice(128, 192), slice(None)]``.

    Generalization: two mesh dims may shard the SAME tensor dim (e.g. FSDP and
    TP both on dim 0). Then the chunk count is the product of their sizes and
    ``strides`` folds the per-axis coords into one chunk index, row-major (outer
    mesh dim = larger stride), exactly like indexing a multi-dim array.
    """
    # Start every tensor dim fully replicated (whole extent); below we narrow
    # only the dims that some mesh axis shards.
    slices = [slice(None) for _ in range(len(global_shape))]
    # Group the sharding mesh axes by which TENSOR dim they shard. A tensor dim
    # can be sharded by more than one mesh axis, hence a list per dim. Each entry
    # is (mesh_dim, that axis's size, this rank's coord on that axis).
    # (example: shard_map = {0: [(mesh_dim=1, size=4, coord=2)]} — tensor dim 0
    # sharded by mesh axis 1 (tp, size 4), this rank at coord 2 on it.)
    shard_map = {}
    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, Shard):
            shard_map.setdefault(placement.dim, []).append(
                (mesh_dim, mesh_shape[mesh_dim], mesh_coords[mesh_dim])
            )

    for tensor_dim, shard_info in shard_map.items():
        # Order the sharding axes outer->inner (ascending mesh dim) so the
        # row-major chunk numbering below matches the mesh layout.
        shard_info.sort(key=lambda item: item[0])
        # Total chunks this tensor dim is split into = product of the axis sizes.
        # (example: a single tp axis of size 4 -> 4 chunks.)
        num_chunks = 1
        for _, size, _ in shard_info:
            num_chunks *= size

        # Split total_size into num_chunks near-equal chunks; when it doesn't
        # divide evenly the first `remainder` chunks each take one extra element.
        # (example: 256 / 4 -> sizes = [64, 64, 64, 64].)
        total_size = int(global_shape[tensor_dim])
        base = total_size // num_chunks
        remainder = total_size % num_chunks
        sizes = [base + 1 if i < remainder else base for i in range(num_chunks)]

        # Row-major strides that fold this rank's per-axis coords into ONE chunk
        # index: innermost (last) axis gets stride 1, the next gets the inner
        # axis's size, and so on. (single axis -> strides = [1].)
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
    ``nccl.xfer.api.XdtensorRedistribute`` op.  Callers must wrap raw tensors in
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
            # .contiguous() guards the .view(torch.uint8) below: the source is a
            # Megatron param view that may be non-contiguous, which .view (unlike
            # .reshape) cannot reinterpret. .clone() keeps full_tensor an
            # independent buffer so a broadcast-in on a non-root src rank can't
            # mutate the live param.
            full_tensor = src_tensor._local_tensor.contiguous().clone()
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
            dst_tensor._local_tensor.copy_(resharded_local)
    return
