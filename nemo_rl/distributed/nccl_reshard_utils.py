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

"""Utilities for nccl_reshard-based refit between disaggregated train/gen meshes.

This module provides:
- xferdtensor_golden: broadcast-based reference implementation of XferDTensor
- MeshInfo / TensorWrapper: lightweight wrappers for cross-world communication
- Placement rules: mapping param names to TP/EP sharding strategies
- build_nccl_reshard_refit_info: compute per-layer param metadata for refit
"""

import re
from collections import OrderedDict
from typing import Any, Optional

import torch
from torch.distributed._tensor import Shard
from torch.distributed.tensor.placement_types import Replicate

# =========================================================================
# MeshInfo / TensorWrapper: lightweight wrappers for cross-world comms
# =========================================================================


class MeshInfo:
    """Lightweight mesh metadata compatible with xferdtensor_golden.

    Provides the same .mesh / ._mesh interface as DeviceMesh but without
    requiring torch.distributed process groups — allowing xferdtensor_golden
    to read mesh topology across separate torch.distributed worlds.
    """

    def __init__(self, rank_tensor: torch.Tensor):
        self.mesh = rank_tensor
        self._mesh = rank_tensor

    @property
    def ndim(self):
        return self.mesh.ndim


class TensorWrapper:
    """Wraps a plain GPU tensor with a DTensor-like interface.

    xferdtensor_golden writes into ``dst_tensor._local_tensor`` on the
    destination side.  This wrapper lets plain vLLM parameters satisfy that
    interface.
    """

    def __init__(self, tensor: torch.Tensor):
        self._local_tensor = tensor
        self.dtype = tensor.dtype
        self.shape = tensor.shape
        self.device = tensor.device


# =========================================================================
# Placement rules (from xferdtensor/src/placement_rules.py)
# =========================================================================

# Column-parallel suffixes: TP shards along dim 0 (output dimension)
COLUMN_PARALLEL_SUFFIXES = [
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    # DeepSeek MLA projections
    "q_a_proj.weight",
    "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_b_proj.weight",
]

# Row-parallel suffixes: TP shards along dim 1 (input dimension)
ROW_PARALLEL_SUFFIXES = [
    "o_proj.weight",
    "down_proj.weight",
]

# Vocabulary-parallel: TP shards along dim 0 (vocab dimension)
VOCAB_PARALLEL_NAMES = [
    "embed_tokens.weight",
    "lm_head.weight",
]


def get_tp_shard_dim(param_name: str) -> Optional[int]:
    """Return the tensor dimension to shard for TP, or None if replicated."""
    # MoE expert params use EP, not TP
    if ".experts." in param_name:
        return None
    # MoE router gate is always replicated
    if (
        param_name.endswith("mlp.gate.weight")
        or "e_score_correction_bias" in param_name
    ):
        return None

    for suffix in COLUMN_PARALLEL_SUFFIXES:
        if param_name.endswith(suffix):
            return 0
    for suffix in ROW_PARALLEL_SUFFIXES:
        if param_name.endswith(suffix):
            return 1
    for name in VOCAB_PARALLEL_NAMES:
        if name in param_name:
            return 0
    return None


def is_expert_param(param_name: str) -> bool:
    """Return True if the parameter is a MoE expert weight (sharded by EP)."""
    return ".experts." in param_name


def _get_expert_tp_shard_dim(param_name: str) -> Optional[int]:
    """Like get_tp_shard_dim but does NOT skip .experts. params."""
    for suffix in COLUMN_PARALLEL_SUFFIXES:
        if param_name.endswith(suffix):
            return 0
    for suffix in ROW_PARALLEL_SUFFIXES:
        if param_name.endswith(suffix):
            return 1
    return None


# =========================================================================
# xferdtensor_golden: broadcast-based reference implementation
# =========================================================================

_STR_TO_DTYPE = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _get_mesh_tensor(mesh) -> torch.Tensor:
    """Extract the rank tensor from a MeshInfo or DeviceMesh."""
    mesh_tensor = getattr(mesh, "mesh", None)
    if mesh_tensor is None:
        mesh_tensor = getattr(mesh, "_mesh", None)
    if mesh_tensor is None:
        raise ValueError("Mesh object does not expose rank tensor.")
    return mesh_tensor


def _flatten_mesh_ranks(mesh) -> list[int]:
    """Return a flat list of global ranks from a MeshInfo or DeviceMesh."""
    return _get_mesh_tensor(mesh).flatten().tolist()


def _get_mesh_coords(mesh, rank) -> Optional[list[int]]:
    """Return the N-D coordinates of *rank* in *mesh*, or None if absent."""
    mesh_tensor = _get_mesh_tensor(mesh)
    coords = (mesh_tensor == rank).nonzero(as_tuple=False)
    if coords.numel() == 0:
        return None
    return coords[0].tolist()


def _parse_placement(placement) -> tuple[str, Optional[int]]:
    """Normalise a placement to ``("shard", dim)`` or ``("replicate", None)``.

    Handles both native Shard/Replicate objects and the dict form produced by
    vLLM's msgspec serialization (``{}`` → Replicate, ``{"dim": N}`` → Shard(N)).
    """
    if isinstance(placement, Shard):
        return ("shard", placement.dim)
    if isinstance(placement, Replicate):
        return ("replicate", None)
    if isinstance(placement, dict):
        if "dim" in placement:
            return ("shard", placement["dim"])
        return ("replicate", None)
    return ("replicate", None)


def _compute_shard_slices(global_shape, mesh_shape, mesh_coords, placements):
    """Compute the local slice for each tensor dimension given mesh coordinates."""
    slices = [slice(None)] * len(global_shape)
    shard_map: dict[int, list] = {}
    for mesh_dim, placement in enumerate(placements):
        ptype, pdim = _parse_placement(placement)
        if ptype == "shard":
            shard_map.setdefault(pdim, []).append(
                (mesh_dim, mesh_shape[mesh_dim], mesh_coords[mesh_dim])
            )

    for tensor_dim, shard_info in shard_map.items():
        shard_info.sort(key=lambda item: item[0])
        num_chunks = 1
        for _, size, _ in shard_info:
            num_chunks *= size

        total_size = int(global_shape[tensor_dim])
        base, remainder = divmod(total_size, num_chunks)
        sizes = [base + (1 if i < remainder else 0) for i in range(num_chunks)]

        # Compute linear index from multi-dimensional coords
        strides = []
        running = 1
        for _, size, _ in reversed(shard_info):
            strides.append(running)
            running *= size
        strides.reverse()

        linear_index = sum(
            coord * stride for (_, _, coord), stride in zip(shard_info, strides)
        )
        slices[tensor_dim] = slice(
            sum(sizes[:linear_index]), sum(sizes[: linear_index + 1])
        )

    return slices


def xferdtensor_golden(
    src_tensor,
    src_mesh,
    src_placement,
    dst_tensor,
    dst_mesh,
    dst_placement,
    process_group,
    global_shape=None,
    dtype=None,
    param_name=None,
):
    """Broadcast-based reference implementation of XferDTensor.

    The source mesh's rank-0 broadcasts the full (global) tensor to every rank.
    Each destination rank then extracts its local shard based on ``dst_placement``.

    On the **train side**, ``src_tensor`` is a regular tensor (or DTensor) and
    ``dst_tensor`` is ``None``.  On the **gen side**, the opposite holds.

    ``global_shape`` and ``dtype`` are used to allocate the receive buffer when
    ``src_tensor`` is ``None`` (i.e. on gen-side ranks).
    """
    rank = process_group.rank
    src_ranks = _flatten_mesh_ranks(src_mesh)
    dst_ranks = _flatten_mesh_ranks(dst_mesh)
    device = torch.device("cuda", torch.cuda.current_device())

    # --- resolve broadcast buffer shape and dtype ---
    if global_shape is not None:
        buf_shape = (
            torch.Size(global_shape)
            if not isinstance(global_shape, torch.Size)
            else global_shape
        )
    elif src_tensor is not None:
        buf_shape = src_tensor.shape
    elif dst_tensor is not None:
        buf_shape = dst_tensor.shape
    else:
        raise ValueError("Cannot infer buffer shape: provide global_shape or a tensor.")

    if src_tensor is not None:
        buf_dtype = src_tensor.dtype
    elif dst_tensor is not None:
        buf_dtype = dst_tensor.dtype
    elif dtype is not None:
        buf_dtype = (
            _STR_TO_DTYPE.get(dtype, torch.bfloat16)
            if isinstance(dtype, str)
            else dtype
        )
    else:
        buf_dtype = torch.bfloat16

    # --- build or reuse the broadcast buffer ---
    if src_tensor is not None:
        if hasattr(src_tensor, "full_tensor"):
            full_tensor = src_tensor.full_tensor()
            # full_tensor() may run an all-gather on PyTorch's NCCL stream;
            # synchronize so the data is visible on the current stream.
            torch.cuda.synchronize()
        elif hasattr(src_tensor, "_local_tensor"):
            full_tensor = src_tensor._local_tensor
        else:
            full_tensor = src_tensor
    else:
        full_tensor = torch.empty(buf_shape, device=device, dtype=buf_dtype)

    # --- broadcast from src rank 0 to all ranks ---
    process_group.broadcast(full_tensor, src=src_ranks[0])

    # --- on dst ranks, extract local shard and write into dst_tensor ---
    if rank in dst_ranks and dst_tensor is not None:
        coords = _get_mesh_coords(dst_mesh, rank)
        if coords is None:
            raise RuntimeError(f"Rank {rank} not found in dst_mesh.")
        slices = _compute_shard_slices(
            full_tensor.shape, list(dst_mesh.mesh.shape), coords, dst_placement
        )
        local = full_tensor[tuple(slices)]
        if hasattr(dst_tensor, "_local_tensor"):
            dst_tensor._local_tensor.data.copy_(local)
        else:
            dst_tensor.to_local().data.copy_(local)


# =========================================================================
# Mesh and placement construction
# =========================================================================


def build_mesh_info(
    num_gpus: int,
    rank_offset: int,
    tp_size: int = 1,
    ep_size: int = 1,
    pp_size: int = 1,
) -> tuple:
    """Build a ``MeshInfo`` and *dim_map* from a parallelism config.

    Dimension ordering (inner→outer): tp, ep, dp, pp.
    Trivial (size-1) dimensions are dropped.

    Returns:
        ``(MeshInfo, dim_map)`` where *dim_map* maps ``"tp"``/``"ep"``/``"dp"``/``"pp"``
        to the corresponding mesh-tensor axis index.
    """
    dp_size = num_gpus // (tp_size * ep_size * pp_size)
    assert dp_size * tp_size * ep_size * pp_size == num_gpus, (
        f"Cannot divide {num_gpus} GPUs into TP={tp_size} EP={ep_size} PP={pp_size} DP={dp_size}"
    )

    dim_sizes = {"tp": tp_size, "ep": ep_size, "dp": dp_size, "pp": pp_size}
    active_dims = [
        (n, dim_sizes[n]) for n in ("tp", "ep", "dp", "pp") if dim_sizes[n] > 1
    ]

    if not active_dims:
        return MeshInfo(torch.arange(rank_offset, rank_offset + num_gpus)), {}

    # Reverse to outer→inner for the row-major rank tensor
    active_dims_rev = list(reversed(active_dims))
    mesh_shape = [s for _, s in active_dims_rev]
    dim_map = {name: i for i, (name, _) in enumerate(active_dims_rev)}
    ranks = torch.arange(rank_offset, rank_offset + num_gpus).reshape(mesh_shape)
    return MeshInfo(ranks), dim_map


def get_placements(param_name: str, dim_map: dict, ndim: int) -> list:
    """Determine DTensor placements for a parameter given a *dim_map*.

    1-D params (layernorm, bias) are always fully replicated.
    Expert params shard dim 0 on EP; their TP shard dims are shifted by +1.
    """
    num_mesh_dims = max(dim_map.values()) + 1 if dim_map else 1
    placements = [Replicate() for _ in range(num_mesh_dims)]

    if ndim < 2:
        return placements

    if is_expert_param(param_name):
        if "ep" in dim_map:
            placements[dim_map["ep"]] = Shard(0)
        tp_dim = _get_expert_tp_shard_dim(param_name)
        if tp_dim is not None and "tp" in dim_map:
            placements[dim_map["tp"]] = Shard(tp_dim + 1)
    else:
        tp_dim = get_tp_shard_dim(param_name)
        if tp_dim is not None and "tp" in dim_map:
            placements[dim_map["tp"]] = Shard(tp_dim)

    return placements


# =========================================================================
# Layer grouping and refit-info construction
# =========================================================================

_LAYER_RE = re.compile(r"(model\.layers\.\d+)\.")
_MODEL_PREFIX_RE = re.compile(r"(model\.\w+)\.")


def _extract_layer_name(param_name: str) -> str:
    """Extract the layer group name from a parameter name.

    Examples:
        ``model.layers.0.self_attn.q_proj.weight`` → ``model.layers.0``
        ``model.embed_tokens.weight`` → ``model.embed_tokens``
        ``lm_head.weight`` → ``lm_head``
    """
    m = _LAYER_RE.match(param_name)
    if m:
        return m.group(1)
    m = _MODEL_PREFIX_RE.match(param_name)
    if m:
        return m.group(1)
    return param_name.split(".")[0]


def build_nccl_reshard_refit_info(
    state_dict_metadata: dict[str, dict[str, Any]],
    train_parallelism: dict[str, int],
    gen_parallelism: dict[str, int],
    train_world_size: int,
    gen_world_size: int,
) -> dict[str, Any]:
    """Build per-layer parameter info for nccl_reshard-based refit.

    Args:
        state_dict_metadata: ``{param_name: {"shape": list, "dtype": str}}``
        train_parallelism / gen_parallelism: ``{"tp_size", "ep_size", "pp_size"}``
        train_world_size / gen_world_size: number of GPUs per side

    Returns:
        ``{"layer_names": [...], "per_layer_params": {layer: [param_info, ...]}}``
    """
    src_mesh, src_dim_map = build_mesh_info(
        train_world_size,
        rank_offset=0,
        **{k: train_parallelism.get(k, 1) for k in ("tp_size", "ep_size", "pp_size")},
    )
    dst_mesh, dst_dim_map = build_mesh_info(
        gen_world_size,
        rank_offset=train_world_size,
        **{k: gen_parallelism.get(k, 1) for k in ("tp_size", "ep_size", "pp_size")},
    )

    per_layer_params: dict[str, list] = OrderedDict()
    for name, meta in state_dict_metadata.items():
        layer = _extract_layer_name(name)
        ndim = len(meta["shape"])
        info = {
            "name": name,
            "global_shape": tuple(meta["shape"]),
            "dtype": meta["dtype"],
            "src_mesh_info": src_mesh,
            "src_placements": get_placements(name, src_dim_map, ndim),
            "dst_mesh_info": dst_mesh,
            "dst_placements": get_placements(name, dst_dim_map, ndim),
        }
        per_layer_params.setdefault(layer, []).append(info)

    return {
        "layer_names": list(per_layer_params.keys()),
        "per_layer_params": per_layer_params,
        "train_world_size": train_world_size,
        "gen_world_size": gen_world_size,
    }
