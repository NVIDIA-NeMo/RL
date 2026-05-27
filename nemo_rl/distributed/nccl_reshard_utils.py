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

"""Refit metadata builders and lightweight wrapper types for nccl_reshard.

This module provides:
- MeshInfo: lightweight DeviceMesh-compatible wrapper that doesn't require a
  shared torch.distributed process group (needed for cross-world transfers)
- DTensorRef: lightweight DTensor-compatible wrapper that xferdtensor
  reads via duck typing (``.shape``, ``._local_tensor``)
- Placement rules: mapping param names to TP/EP sharding strategies
- build_nccl_reshard_refit_info: compute per-layer param metadata for refit
- restore_refit_info_placements: undo msgspec dict-flattening of placements
  and meshes on the receiving side

The transfer kernel itself (``xferdtensor``) lives in
``nemo_rl/distributed/xferdtensor.py`` — import from there directly.
"""

import re
from collections import OrderedDict
from typing import Any, Optional

import torch
from torch.distributed._tensor import Shard
from torch.distributed.tensor.placement_types import Replicate

# =========================================================================
# MeshInfo / DTensorRef — lightweight wrapper types
# =========================================================================


class MeshInfo:
    """Lightweight mesh metadata compatible with xferdtensor.

    Provides the same .mesh / ._mesh interface as DeviceMesh but without
    requiring torch.distributed process groups -- allowing xferdtensor
    to read mesh topology across separate torch.distributed worlds.
    """

    def __init__(self, rank_tensor: torch.Tensor):
        self.mesh = rank_tensor
        self._mesh = rank_tensor

    @property
    def ndim(self):
        return self.mesh.ndim


class DTensorRef:
    """DTensor-compatible reference for xferdtensor.

    Provides the interface expected by the canonical xferdtensor:
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

    def full_tensor(self):
        """Return the underlying tensor (used on src side by xferdtensor)."""
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
    # DeepSeek MLA up-projections (column-parallel on output: num_heads * head_dim)
    "q_b_proj.weight",
    "kv_b_proj.weight",
    # Fused MoE expert params (vLLM naming: gate+up fused)
    "w13_weight",
]

# DeepSeek MLA down-projections.  vLLM creates these as ReplicatedLinear (or
# merges them into ``fused_qkv_a_proj`` with ``disable_tp=True``), so the full
# tensor lives on every TP rank.  Megatron with TP=1 also keeps them
# unsharded, so leaving them out of COLUMN_PARALLEL_SUFFIXES gives the right
# "all Replicate" placement on both sides.

# Row-parallel suffixes: TP shards along dim 1 (input dimension)
ROW_PARALLEL_SUFFIXES = [
    "o_proj.weight",
    "down_proj.weight",
    # Fused MoE expert param (vLLM naming: down proj)
    "w2_weight",
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
# Dtype lookup (used when receiving msgspec'd dtype names over the wire)
# =========================================================================

_STR_TO_DTYPE = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


# =========================================================================
# Placement restoration (undo msgspec dict-serialization on the wire)
# =========================================================================


def _restore_placement(p):
    """Reconstruct a single placement object after msgspec round-trip.

    vLLM's collective_rpc serializes ``Shard(N)`` to ``{"dim": N}`` and
    ``Replicate()`` to ``{}``; this restores them to real instances. No-op
    if the input is already a Shard/Replicate.
    """
    if isinstance(p, (Shard, Replicate)):
        return p
    if isinstance(p, dict):
        if "dim" in p:
            return Shard(p["dim"])
        return Replicate()
    return Replicate()


def restore_refit_info_placements(refit_info: dict) -> dict:
    """Restore placements and meshes in ``refit_info`` after msgspec transit.

    vLLM's collective_rpc encodes ``Shard``/``Replicate`` as plain dicts and
    ``MeshInfo`` as a dict with a nested mesh list.  This rebuilds the
    original Python objects in place so that the canonical
    ``xferdtensor`` (which relies on ``isinstance(p, Shard)``) works
    correctly.  Idempotent — safe to call on already-restored ``refit_info``.
    """
    for layer_name in refit_info.get("layer_names", []):
        for param_info in refit_info.get("per_layer_params", {}).get(layer_name, []):
            param_info["src_placements"] = [
                _restore_placement(p) for p in param_info["src_placements"]
            ]
            param_info["dst_placements"] = [
                _restore_placement(p) for p in param_info["dst_placements"]
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

    Dimension ordering (inner->outer): ep, tp, dp, pp.  EP is innermost so
    that ``global_rank % ep_size`` recovers the EP coord — matching
    Megatron-Core's rank layout for MoE configs.  TP is next innermost,
    matching the standard ``tp_rank = global_rank // ep_size % tp_size``
    layout that Megatron uses for non-expert params when EP=1.

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
    num_mesh_dims = len(dim_map) or 1
    placements = [Replicate() for _ in range(num_mesh_dims)]

    if ndim < 2:
        # For 1-D params, it assumes that it is always fully replicated.
        return placements

    if is_expert_param(param_name):
        if "ep" in dim_map:
            placements[dim_map["ep"]] = Shard(0)
        else:
            # We currently never shards both EP and TP for the expert params
            # so far. If MCore etp is enabled, the logic should be changed
            tp_dim = _get_expert_tp_shard_dim(param_name)
            if tp_dim is not None and "tp" in dim_map:
                placements[dim_map["tp"]] = Shard(tp_dim + 1)
    else:
        # for non-expert params
        tp_dim = get_tp_shard_dim(param_name)
        if tp_dim is not None and "tp" in dim_map:
            placements[dim_map["tp"]] = Shard(tp_dim)

    return placements


# =========================================================================
# MoE expert param fusion
# =========================================================================

# Matches individual expert params: model.layers.X.mlp.experts.Y.proj.weight
_INDIVIDUAL_EXPERT_RE = re.compile(
    r"(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
)


def fuse_expert_params_in_metadata(
    state_dict_metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Fuse individual MoE expert params into combined w13/w2 entries.

    Converts individual HF expert params (one per expert) into vLLM-style
    fused params:
    - gate_proj + up_proj → w13_weight: [num_experts_global, 2*intermediate, hidden]
    - down_proj → w2_weight: [num_experts_global, hidden, intermediate]

    The input ``state_dict_metadata`` is required to be EP-gathered, i.e. it
    must already enumerate every expert globally (not just this rank's local
    subset).  Both backends meet this invariant: Megatron's metadata pipeline
    routes through ``export_hf_weights`` which does the EP all-gather, and
    DTensor doesn't use EP at all.

    Non-expert params are passed through unchanged.
    """
    # Group individual expert params by (prefix, proj_type)
    expert_groups: dict[tuple[str, str], list[tuple[str, dict]]] = {}
    fused_metadata: dict[str, dict[str, Any]] = OrderedDict()

    for name, meta in state_dict_metadata.items():
        m = _INDIVIDUAL_EXPERT_RE.match(name)
        if m:
            prefix = m.group(1)  # e.g., "model.layers.0.mlp.experts"
            proj = m.group(3)  # "gate_proj", "up_proj", "down_proj"
            key = (prefix, proj)
            expert_groups.setdefault(key, []).append((name, meta))
        else:
            fused_metadata[name] = meta

    if not expert_groups:
        return state_dict_metadata

    # Build fused entries from expert groups
    # Group gate_proj + up_proj → w13_weight, down_proj → w2_weight
    w13_groups: dict[str, dict] = {}  # prefix → {gate: entries, up: entries}
    w2_groups: dict[str, list] = {}  # prefix → entries

    for (prefix, proj), entries in expert_groups.items():
        if proj in ("gate_proj", "up_proj"):
            w13_groups.setdefault(prefix, {})
            w13_groups[prefix][proj] = entries
        else:  # down_proj
            w2_groups[prefix] = entries

    # Create w13_weight entries (fused gate+up)
    for prefix, projs in w13_groups.items():
        gate_entries = projs.get("gate_proj", [])
        if not gate_entries:
            continue
        num_experts_global = len(gate_entries)
        gate_shape = gate_entries[0][1]["shape"]  # [intermediate, hidden]
        intermediate_size = gate_shape[0]
        hidden_size = gate_shape[1]
        fused_shape = [num_experts_global, 2 * intermediate_size, hidden_size]
        fused_name = f"{prefix}.w13_weight"
        fused_metadata[fused_name] = {
            "shape": fused_shape,
            "dtype": gate_entries[0][1]["dtype"],
            "fused_expert_param_type": "w13",
        }

    # Create w2_weight entries (down)
    for prefix, entries in w2_groups.items():
        num_experts_global = len(entries)
        down_shape = entries[0][1]["shape"]  # [hidden, intermediate]
        hidden_size = down_shape[0]
        intermediate_size = down_shape[1]
        fused_shape = [num_experts_global, hidden_size, intermediate_size]
        fused_name = f"{prefix}.w2_weight"
        fused_metadata[fused_name] = {
            "shape": fused_shape,
            "dtype": entries[0][1]["dtype"],
            "fused_expert_param_type": "w2",
        }

    return fused_metadata


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
    layer_to_pp_stage: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Build per-layer parameter info for nccl_reshard-based refit.

    The input ``state_dict_metadata`` must be EP-gathered (enumerate every
    expert globally, not per-rank).  Both backends meet this naturally:
    Megatron's metadata pipeline routes through ``export_hf_weights`` (which
    does the EP all-gather) and DTensor doesn't use EP at all.

    Args:
        state_dict_metadata: ``{param_name: {"shape": list, "dtype": str}}``
        train_parallelism / gen_parallelism: ``{"tp_size", "ep_size", "pp_size"}``
        train_world_size / gen_world_size: number of GPUs per side
        layer_to_pp_stage: optional mapping from layer name to PP stage index.
            When provided (PP>1), per-stage meshes are built so each PP stage's
            train ranks + all gen ranks form an independent sub-group.

    Returns:
        ``{"layer_names": [...], "per_layer_params": {layer: [param_info, ...]},
           "pp_size": int}``
    """
    state_dict_metadata = fuse_expert_params_in_metadata(state_dict_metadata)

    pp_size = train_parallelism.get("pp_size", 1)
    tp_size = train_parallelism.get("tp_size", 1)
    ep_size = train_parallelism.get("ep_size", 1)
    use_per_stage = layer_to_pp_stage is not None and pp_size > 1

    # Megatron-Core with ETP=1 gives non-expert and expert params *different*
    # rank-to-coord layouts on the same physical ranks: for non-expert params,
    # ranks are partitioned (tp, dp); for expert params, ranks are partitioned
    # (ep, edp).  TP and EP are not jointly indexable on a single rectangular
    # mesh (the coords aren't independent — e.g. EP4×TP2 has tp=r%2 AND
    # ep=r%4 on the same rank).  We therefore build two src meshes per stage
    # and pick per-param based on whether the param is a MoE expert weight.
    def _build_train_src_meshes(num_gpus: int, rank_offset: int, stage_pp: int):
        ne_mesh, ne_dim_map = build_mesh_info(
            num_gpus,
            rank_offset=rank_offset,
            tp_size=tp_size,
            ep_size=1,
            pp_size=stage_pp,
        )
        ex_mesh, ex_dim_map = build_mesh_info(
            num_gpus,
            rank_offset=rank_offset,
            tp_size=1,
            ep_size=ep_size,
            pp_size=stage_pp,
        )
        return (ne_mesh, ne_dim_map), (ex_mesh, ex_dim_map)

    if use_per_stage:
        # Per-PP-stage meshes: within each sub-group, train ranks are
        # 0..train_ranks_per_stage-1 and gen ranks follow immediately after.
        train_ranks_per_stage = train_world_size // pp_size
        per_stage_src_nonexpert = {}
        per_stage_src_expert = {}
        for s in range(pp_size):
            ne, ex = _build_train_src_meshes(
                train_ranks_per_stage, rank_offset=0, stage_pp=1
            )
            per_stage_src_nonexpert[s] = ne
            per_stage_src_expert[s] = ex

        # dst mesh: gen ranks start at train_ranks_per_stage within each sub-group
        dst_mesh, dst_dim_map = build_mesh_info(
            gen_world_size,
            rank_offset=train_ranks_per_stage,
            **{k: gen_parallelism.get(k, 1) for k in ("tp_size", "ep_size", "pp_size")},
        )
    else:
        # Single global mesh pair (PP=1 or no per-stage mapping)
        (src_mesh_ne, src_dim_map_ne), (src_mesh_ex, src_dim_map_ex) = (
            _build_train_src_meshes(train_world_size, rank_offset=0, stage_pp=pp_size)
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
        expert = is_expert_param(name)

        if use_per_stage:
            stage = layer_to_pp_stage.get(layer, 0)
            stage_src_mesh, stage_src_dim_map = (
                per_stage_src_expert[stage]
                if expert
                else per_stage_src_nonexpert[stage]
            )
            info = {
                "name": name,
                "global_shape": tuple(meta["shape"]),
                "dtype": meta["dtype"],
                "pp_stage": stage,
                "src_mesh_info": stage_src_mesh,
                "src_placements": get_placements(name, stage_src_dim_map, ndim),
                "dst_mesh_info": dst_mesh,
                "dst_placements": get_placements(name, dst_dim_map, ndim),
            }
        else:
            this_src_mesh, this_src_dim_map = (
                (src_mesh_ex, src_dim_map_ex)
                if expert
                else (src_mesh_ne, src_dim_map_ne)
            )
            info = {
                "name": name,
                "global_shape": tuple(meta["shape"]),
                "dtype": meta["dtype"],
                "src_mesh_info": this_src_mesh,
                "src_placements": get_placements(name, this_src_dim_map, ndim),
                "dst_mesh_info": dst_mesh,
                "dst_placements": get_placements(name, dst_dim_map, ndim),
            }

        # Propagate fused-expert type ("w13" or "w2") for train-side expert stacking
        if "fused_expert_param_type" in meta:
            info["fused_expert_param_type"] = meta["fused_expert_param_type"]

        per_layer_params.setdefault(layer, []).append(info)

    return {
        "layer_names": list(per_layer_params.keys()),
        "per_layer_params": per_layer_params,
        "train_world_size": train_world_size,
        "gen_world_size": gen_world_size,
        "pp_size": pp_size,
        "gen_tp_size": gen_parallelism.get("tp_size", 1),
    }
