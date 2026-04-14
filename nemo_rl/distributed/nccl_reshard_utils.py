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
- xferdtensor_golden: canonical broadcast-based reference implementation of XferDTensor
- DTensorRef: lightweight DTensor-compatible wrapper for xferdtensor_golden
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
# MeshInfo / TensorWrapper / DTensorRef
# =========================================================================


class MeshInfo:
    """Lightweight mesh metadata compatible with xferdtensor_golden.

    Provides the same .mesh / ._mesh interface as DeviceMesh but without
    requiring torch.distributed process groups -- allowing xferdtensor_golden
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


class DTensorRef:
    """DTensor-compatible reference for xferdtensor_golden.

    Provides the interface expected by the canonical xferdtensor_golden:
    - ``.shape``: global tensor shape (torch.Size)
    - ``.full_tensor()``: returns the underlying tensor (used on src side)
    - ``._local_tensor``: local tensor for in-place copy (used on dst side)
    - ``.dtype``, ``.device``: tensor metadata

    On the **src side** (train), ``local_tensor`` is the full global tensor
    from Megatron export. ``.full_tensor()`` returns it directly.

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

    def full_tensor(self):
        """Return the underlying tensor (used on src side by xferdtensor_golden)."""
        return self._local_tensor


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
# xferdtensor_golden: canonical implementation from xferdtensor repo
# =========================================================================
# The functions below are copied verbatim from xferdtensor/src/xferdtensor.py.
# The 7-argument signature of xferdtensor_golden matches the real
# nccl_reshard.XferDTensor API for trivial future replacement.

_STR_TO_DTYPE = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


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

    The source mesh's rank-0 broadcasts the full (global) tensor to every rank.
    Each destination rank then extracts its local shard based on ``dst_placement``.

    This is the canonical 7-argument signature matching the real
    ``nccl_reshard.XferDTensor`` API.  Callers must wrap raw tensors in
    ``DTensorRef`` so that ``.shape`` reports the global shape and
    ``.full_tensor()`` / ``._local_tensor`` are available.
    """
    rank = process_group.rank
    src_ranks = _flatten_mesh_ranks(src_mesh)
    dst_ranks = _flatten_mesh_ranks(dst_mesh)

    global_shape, device = _get_tensor_meta(src_tensor, dst_tensor)
    if global_shape is None:
        raise ValueError("Unable to infer tensor shape/dtype from src or dst tensor.")

    if src_tensor is not None:
        full_tensor = src_tensor.full_tensor()
    else:
        full_tensor = torch.empty(global_shape, device=device, dtype=dst_tensor.dtype)

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
                dst_tensor._local_tensor.copy_(resharded_local)
            else:
                dst_tensor_local = dst_tensor.to_local()
                dst_tensor_local.copy_(resharded_local)
    return


# =========================================================================
# Placement normalization (for msgspec serialization compatibility)
# =========================================================================


def _normalize_placement(p):
    """Convert a single placement (possibly dict-serialized) to Shard/Replicate."""
    if isinstance(p, (Shard, Replicate)):
        return p
    if isinstance(p, dict):
        if "dim" in p:
            return Shard(p["dim"])
        return Replicate()
    return Replicate()


def normalize_refit_info_placements(refit_info: dict) -> dict:
    """Normalize all placements in refit_info to Shard/Replicate objects.

    vLLM's collective_rpc may serialize ``Shard(N)`` to ``{"dim": N}`` and
    ``Replicate()`` to ``{}``.  This function converts them back so that the
    canonical ``xferdtensor_golden`` (which uses ``isinstance(p, Shard)``)
    works correctly.

    Also reconstructs MeshInfo objects if they were serialized to dicts.
    """
    for layer_name in refit_info.get("layer_names", []):
        for param_info in refit_info.get("per_layer_params", {}).get(layer_name, []):
            param_info["src_placements"] = [
                _normalize_placement(p) for p in param_info["src_placements"]
            ]
            param_info["dst_placements"] = [
                _normalize_placement(p) for p in param_info["dst_placements"]
            ]
            # Reconstruct MeshInfo if serialized to dict
            for key in ("src_mesh_info", "dst_mesh_info"):
                mesh = param_info.get(key)
                if mesh is not None and not isinstance(mesh, MeshInfo):
                    if isinstance(mesh, dict) and "mesh" in mesh:
                        mesh_tensor = mesh["mesh"]
                        if not isinstance(mesh_tensor, torch.Tensor):
                            mesh_tensor = torch.tensor(mesh_tensor)
                        param_info[key] = MeshInfo(mesh_tensor)
    return refit_info


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

    Dimension ordering (inner->outer): tp, ep, dp, pp.
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

    # Reverse to outer->inner for the row-major rank tensor
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
        ``model.layers.0.self_attn.q_proj.weight`` -> ``model.layers.0``
        ``model.embed_tokens.weight`` -> ``model.embed_tokens``
        ``lm_head.weight`` -> ``lm_head``
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
