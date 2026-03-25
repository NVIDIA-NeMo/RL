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
# MeshInfo: lightweight mesh metadata for cross-world xferdtensor_golden
# =========================================================================


class MeshInfo:
    """Lightweight mesh metadata for xferdtensor_golden.

    Holds a rank tensor with the same interface as DeviceMesh (.mesh / ._mesh)
    but without requiring torch.distributed process groups. This allows
    xferdtensor_golden to read mesh topology (ranks, shape, coords) across
    separate torch.distributed worlds.
    """

    def __init__(self, rank_tensor: torch.Tensor):
        self.mesh = rank_tensor
        self._mesh = rank_tensor

    @property
    def ndim(self):
        return self.mesh.ndim


class TensorWrapper:
    """Wraps a plain GPU tensor to provide DTensor-like interface for xferdtensor_golden.

    xferdtensor_golden writes into dst_tensor._local_tensor on the destination side.
    This wrapper makes plain vLLM parameters compatible with that interface.
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
    """Return the tensor dimension to shard for TP, or None if replicated.

    Args:
        param_name: Full parameter name (e.g., "model.layers.0.self_attn.q_proj.weight")

    Returns:
        int or None: Dimension to shard, or None if the parameter should be replicated.
    """
    # Skip MoE expert params for TP (they use EP instead)
    if ".experts." in param_name:
        return None

    # Skip MoE router gate
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

    # Default: replicate
    return None


def is_expert_param(param_name: str) -> bool:
    """Return True if the parameter is a MoE expert weight (sharded by EP)."""
    return ".experts." in param_name


def _get_expert_tp_shard_dim(param_name: str) -> Optional[int]:
    """Get TP shard dim for an expert param by checking the weight suffix.

    Unlike get_tp_shard_dim, this does NOT skip .experts. params.
    """
    for suffix in COLUMN_PARALLEL_SUFFIXES:
        if param_name.endswith(suffix):
            return 0

    for suffix in ROW_PARALLEL_SUFFIXES:
        if param_name.endswith(suffix):
            return 1

    return None


# =========================================================================
# xferdtensor_golden: broadcast-based reference implementation
# (from xferdtensor/src/xferdtensor.py)
# =========================================================================


def _str_to_dtype(s: str) -> torch.dtype:
    """Convert a string dtype to torch.dtype."""
    mapping = {
        "torch.bfloat16": torch.bfloat16,
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(s, torch.bfloat16)


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


def _parse_placement(placement):
    """Parse a placement that may be a Shard/Replicate object or a dict from serialization.

    vLLM's collective_rpc uses msgspec serialization which converts Shard/Replicate
    to dicts: {} for Replicate(), {'dim': N} for Shard(N).
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
    slices = [slice(None) for _ in range(len(global_shape))]
    shard_map = {}
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
    global_shape=None,
    dtype=None,
    param_name=None,
):
    """Broadcast-based reference implementation of XferDTensor.

    Args:
        src_tensor: Source DTensor/plain tensor (on train side) or None (on gen side)
        src_mesh: MeshInfo with global ranks for the source mesh
        src_placement: List of Shard/Replicate placements for source
        dst_tensor: Destination DTensor/TensorWrapper (on gen side) or None (on train side)
        dst_mesh: MeshInfo with global ranks for the destination mesh
        dst_placement: List of Shard/Replicate placements for destination
        process_group: StatelessProcessGroup with .rank and .broadcast()
        global_shape: Optional fallback global shape (from refit_info)
        dtype: Optional fallback dtype (from refit_info)
        param_name: Optional parameter name for debug logging
    """
    rank = process_group.rank
    src_ranks = _flatten_mesh_ranks(src_mesh)
    dst_ranks = _flatten_mesh_ranks(dst_mesh)

    # Broadcast buffer must use the GLOBAL shape so all ranks have matching sizes.
    # dst_tensor may hold a TP-sharded local slice; using its shape would cause
    # an NCCL size mismatch (hang) when TP > 1.
    _, device = _get_tensor_meta(src_tensor, dst_tensor)
    if global_shape is not None:
        inferred_shape = (
            torch.Size(global_shape)
            if isinstance(global_shape, (list, tuple))
            else global_shape
        )
    else:
        inferred_shape, _ = _get_tensor_meta(src_tensor, dst_tensor)
    if device is None or device.type == "cpu":
        device = torch.device("cuda", torch.cuda.current_device())
    if inferred_shape is None:
        raise ValueError("Unable to infer tensor shape/dtype from src or dst tensor.")

    # Resolve dtype
    if src_tensor is not None:
        tensor_dtype = src_tensor.dtype
    elif dst_tensor is not None:
        tensor_dtype = dst_tensor.dtype
    elif dtype is not None:
        tensor_dtype = _str_to_dtype(dtype)
    else:
        tensor_dtype = torch.bfloat16

    if src_tensor is not None:
        if hasattr(src_tensor, "full_tensor"):
            full_tensor = src_tensor.full_tensor()
        elif hasattr(src_tensor, "_local_tensor"):
            full_tensor = src_tensor._local_tensor
        else:
            full_tensor = src_tensor
    else:
        full_tensor = torch.empty(inferred_shape, device=device, dtype=tensor_dtype)

    process_group.broadcast(full_tensor, src=src_ranks[0])

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
                dst_tensor._local_tensor.data.copy_(resharded_local)
            else:
                dst_tensor_local = dst_tensor.to_local()
                dst_tensor_local.data.copy_(resharded_local)


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
    """Build a MeshInfo and dim_map from parallelism config.

    Uses the same mapping convention as the xferdtensor tests:
    inner-to-outer: tp-ep-dp-pp. Trivial dimensions (size=1) are dropped.

    Args:
        num_gpus: Number of GPUs for this mesh
        rank_offset: Starting global rank (0 for train, train_world_size for gen)
        tp_size: Tensor parallel size
        ep_size: Expert parallel size
        pp_size: Pipeline parallel size

    Returns:
        (MeshInfo, dim_map): mesh info and {parallelism_type: mesh_dim_index}
    """
    dp_size = num_gpus // (tp_size * ep_size * pp_size)
    assert dp_size * tp_size * ep_size * pp_size == num_gpus, (
        f"Cannot divide {num_gpus} GPUs into TP={tp_size} EP={ep_size} "
        f"PP={pp_size} DP={dp_size}"
    )

    # Inner-to-outer ordering: tp, ep, dp, pp
    dim_names = ["tp", "ep", "dp", "pp"]
    dim_sizes = {"tp": tp_size, "ep": ep_size, "dp": dp_size, "pp": pp_size}

    # Keep only non-trivial dimensions
    active_dims = [(name, dim_sizes[name]) for name in dim_names if dim_sizes[name] > 1]

    if not active_dims:
        # Fully replicated
        mesh_tensor = torch.arange(rank_offset, rank_offset + num_gpus)
        return MeshInfo(mesh_tensor), {}

    # Reverse to get outer-to-inner for the mesh tensor (row-major)
    active_dims_reversed = list(reversed(active_dims))
    mesh_shape = [size for _, size in active_dims_reversed]
    dim_map = {name: i for i, (name, _) in enumerate(active_dims_reversed)}

    ranks = torch.arange(rank_offset, rank_offset + num_gpus).reshape(mesh_shape)
    return MeshInfo(ranks), dim_map


def get_placements(param_name: str, dim_map: dict, ndim: int) -> list:
    """Determine DTensor placements for a parameter given a dim_map.

    For expert params (combined 3D tensors after grouping), dim 0 is the expert dimension:
      - EP shards dim 0 (expert dimension)
      - TP shard dims are shifted by +1

    Args:
        param_name: Parameter name
        dim_map: {parallelism_type: mesh_dim_index}
        ndim: Number of tensor dimensions

    Returns:
        list of Shard/Replicate placements, one per mesh dimension
    """
    num_mesh_dims = max(dim_map.values()) + 1 if dim_map else 1
    placements = [Replicate() for _ in range(num_mesh_dims)]

    # 1-D params are always fully replicated (layernorm, bias)
    if ndim < 2:
        return placements

    if is_expert_param(param_name):
        # EP shards the expert dimension (dim 0)
        if "ep" in dim_map:
            placements[dim_map["ep"]] = Shard(0)

        # TP shards the weight dimension, shifted by +1 for the expert dim
        tp_shard_dim = _get_expert_tp_shard_dim(param_name)
        if tp_shard_dim is not None and "tp" in dim_map:
            placements[dim_map["tp"]] = Shard(tp_shard_dim + 1)
    else:
        tp_shard_dim = get_tp_shard_dim(param_name)
        if tp_shard_dim is not None and "tp" in dim_map:
            placements[dim_map["tp"]] = Shard(tp_shard_dim)

    return placements


# =========================================================================
# Layer grouping and refit info construction
# =========================================================================


def _extract_layer_name(param_name: str) -> str:
    """Extract the layer group name from a parameter name.

    Examples:
        "model.embed_tokens.weight" → "model.embed_tokens"
        "model.layers.0.self_attn.q_proj.weight" → "model.layers.0"
        "model.layers.31.mlp.experts.0.gate_proj.weight" → "model.layers.31"
        "model.norm.weight" → "model.norm"
        "lm_head.weight" → "lm_head"
    """
    # Match model.layers.{i}
    match = re.match(r"(model\.layers\.\d+)\.", param_name)
    if match:
        return match.group(1)

    # Match other model.* prefixes (embed_tokens, norm, etc.)
    match = re.match(r"(model\.\w+)\.", param_name)
    if match:
        return match.group(1)

    # Top-level (lm_head, etc.)
    parts = param_name.split(".")
    return parts[0]


def build_nccl_reshard_refit_info(
    state_dict_metadata: dict[str, dict[str, Any]],
    train_parallelism: dict[str, int],
    gen_parallelism: dict[str, int],
    train_world_size: int,
    gen_world_size: int,
) -> dict[str, Any]:
    """Build per-layer parameter info for nccl_reshard-based refit.

    Args:
        state_dict_metadata: {param_name: {"shape": list, "dtype": str}}
            from the policy worker's model state_dict
        train_parallelism: {"tp_size": int, "ep_size": int, "pp_size": int}
        gen_parallelism: {"tp_size": int, "ep_size": int, "pp_size": int}
        train_world_size: Number of training GPUs
        gen_world_size: Number of generation GPUs

    Returns:
        dict with "layer_names" and "per_layer_params"
    """
    # Build mesh info for train (ranks [0, train_ws)) and gen (ranks [train_ws, total))
    src_mesh_info, src_dim_map = build_mesh_info(
        num_gpus=train_world_size,
        rank_offset=0,
        tp_size=train_parallelism.get("tp_size", 1),
        ep_size=train_parallelism.get("ep_size", 1),
        pp_size=train_parallelism.get("pp_size", 1),
    )
    dst_mesh_info, dst_dim_map = build_mesh_info(
        num_gpus=gen_world_size,
        rank_offset=train_world_size,
        tp_size=gen_parallelism.get("tp_size", 1),
        ep_size=gen_parallelism.get("ep_size", 1),
        pp_size=gen_parallelism.get("pp_size", 1),
    )

    # Group parameters by layer
    per_layer_params: dict[str, list] = OrderedDict()

    for name, meta in state_dict_metadata.items():
        layer_name = _extract_layer_name(name)
        global_shape = meta["shape"]
        ndim = len(global_shape)

        src_placements = get_placements(name, src_dim_map, ndim)
        dst_placements = get_placements(name, dst_dim_map, ndim)

        param_info = {
            "name": name,
            "global_shape": tuple(global_shape),
            "dtype": meta["dtype"],
            "src_mesh_info": src_mesh_info,
            "src_placements": src_placements,
            "dst_mesh_info": dst_mesh_info,
            "dst_placements": dst_placements,
        }

        if layer_name not in per_layer_params:
            per_layer_params[layer_name] = []
        per_layer_params[layer_name].append(param_info)

    layer_names = list(per_layer_params.keys())

    return {
        "layer_names": layer_names,
        "per_layer_params": per_layer_params,
        "train_world_size": train_world_size,
        "gen_world_size": gen_world_size,
    }
