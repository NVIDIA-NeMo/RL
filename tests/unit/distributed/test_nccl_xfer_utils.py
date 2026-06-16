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

"""Pure-CPU unit tests for the nccl_xfer refit metadata builders.

Covers the metadata/placement logic in ``nemo_rl/distributed/nccl_xfer_utils.py``
that runs on the train side to build the shared ``nccl_xfer_refit_info``:
mesh construction, placement rules, expert grouping, and the top-level
``build_nccl_xfer_refit_info``.  All pure functions — no GPU, no
torch.distributed, no model object — so this module runs on CPU with no extras.
"""

import pytest
import torch
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_rl.distributed.nccl_xfer_utils import (
    MeshInfo,
    build_mesh_info,
    build_nccl_xfer_refit_info,
    get_placements,
    get_tp_shard_dim,
    group_expert_params_in_metadata,
    is_expert_param,
)


# --------------------------------------------------------------------------
# MeshInfo
# --------------------------------------------------------------------------
def test_mesh_info_exposes_rank_grid():
    grid = torch.arange(8).reshape(2, 4)
    mesh = MeshInfo(grid)
    assert mesh.ndim == 2
    assert torch.equal(mesh.mesh, grid)
    assert torch.equal(mesh._mesh, grid)  # duck-typed alias xferdtensor reads


# --------------------------------------------------------------------------
# build_mesh_info
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "num_gpus,rank_offset,tp,ep,pp,exp_shape,exp_dim_map",
    [
        # TP4 + DP2 (non-expert mesh): emit (tp, dp) -> reversed (dp, tp).
        (8, 0, 4, 1, 1, (2, 4), {"dp": 0, "tp": 1}),
        # EP4 + DP2 (expert mesh).
        (8, 0, 1, 4, 1, (2, 4), {"dp": 0, "ep": 1}),
        # TP2 + EP4, DP1 (both >1, dp dropped): reversed (ep, tp).
        (8, 0, 2, 4, 1, (4, 2), {"ep": 0, "tp": 1}),
        # PP2 + DP2.
        (4, 0, 1, 1, 2, (2, 2), {"pp": 0, "dp": 1}),
        # TP8 only, 1-D mesh, with a rank offset (gen side starts after train).
        (8, 8, 8, 1, 1, (8,), {"tp": 0}),
    ],
)
def test_build_mesh_info_shape_and_dim_map(
    num_gpus, rank_offset, tp, ep, pp, exp_shape, exp_dim_map
):
    mesh, dim_map = build_mesh_info(
        num_gpus, rank_offset, tp_size=tp, ep_size=ep, pp_size=pp
    )
    assert tuple(mesh.mesh.shape) == exp_shape
    assert dim_map == exp_dim_map
    # Ranks are a contiguous row-major interval [offset, offset + num_gpus).
    assert mesh.mesh.flatten().tolist() == list(
        range(rank_offset, rank_offset + num_gpus)
    )


def test_build_mesh_info_dp_only():
    # tp=ep=pp=1 with >1 GPU -> all GPUs are DP; single "dp" mesh axis.
    mesh, dim_map = build_mesh_info(4, 0, tp_size=1, ep_size=1, pp_size=1)
    assert dim_map == {"dp": 0}
    assert tuple(mesh.mesh.shape) == (4,)
    assert mesh.mesh.flatten().tolist() == [0, 1, 2, 3]


def test_build_mesh_info_single_gpu_empty_dim_map():
    # The only case with no active dims (all sizes 1) is a 1-GPU mesh.
    mesh, dim_map = build_mesh_info(1, 0, tp_size=1, ep_size=1, pp_size=1)
    assert dim_map == {}
    assert mesh.mesh.flatten().tolist() == [0]


def test_build_mesh_info_rank_offset_disjoint_from_train():
    # Gen mesh placed after the train world (disaggregated layout).
    mesh, _ = build_mesh_info(8, rank_offset=16, tp_size=4)
    assert int(mesh.mesh.min()) == 16
    assert int(mesh.mesh.max()) == 23


def test_build_mesh_info_indivisible_raises():
    with pytest.raises(AssertionError):
        build_mesh_info(8, 0, tp_size=3)  # 8 not divisible by 3


# --------------------------------------------------------------------------
# is_expert_param / get_tp_shard_dim
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", True),
        ("model.layers.3.mlp.experts.17.down_proj.weight", True),
        ("model.layers.0.mlp.gate_proj.weight", False),
        ("model.layers.0.self_attn.q_proj.weight", False),
    ],
)
def test_is_expert_param(name, expected):
    assert is_expert_param(name) is expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.layers.0.self_attn.q_proj.weight", 0),  # column-parallel
        ("model.layers.0.self_attn.o_proj.weight", 1),  # row-parallel
        ("model.embed_tokens.weight", 0),  # vocab-parallel
        ("lm_head.weight", 0),
        ("model.layers.0.mlp.gate.weight", None),  # MoE router replicated
        ("model.layers.0.mlp.experts.0.gate_proj.weight", None),  # EP, not TP
        ("model.layers.0.input_layernorm.weight", None),  # no shard rule
    ],
)
def test_get_tp_shard_dim(name, expected):
    assert get_tp_shard_dim(name) == expected


# --------------------------------------------------------------------------
# get_placements
# --------------------------------------------------------------------------
def _shard_dim_at(placements, dim_map, axis):
    """Return the Shard.dim at the given dim_map axis, or None if Replicate."""
    p = placements[dim_map[axis]]
    return p.dim if isinstance(p, Shard) else None


def test_get_placements_tp_only_mesh():
    dm = {"tp": 0}
    # column-parallel -> Shard(0)
    assert _shard_dim_at(get_placements("a.q_proj.weight", dm, 2), dm, "tp") == 0
    # row-parallel -> Shard(1)
    assert _shard_dim_at(get_placements("a.o_proj.weight", dm, 2), dm, "tp") == 1
    # vocab -> Shard(0)
    assert (
        _shard_dim_at(get_placements("model.embed_tokens.weight", dm, 2), dm, "tp") == 0
    )
    # router gate -> Replicate
    assert all(
        isinstance(p, Replicate) for p in get_placements("a.mlp.gate.weight", dm, 2)
    )
    # 1-D param -> all Replicate regardless of rule
    assert all(
        isinstance(p, Replicate) for p in get_placements("a.q_proj.weight", dm, 1)
    )


def test_get_placements_2d_mesh_shards_only_tp_axis():
    dm = {"dp": 0, "tp": 1}
    placements = get_placements("a.q_proj.weight", dm, 2)
    assert len(placements) == 2
    assert isinstance(placements[dm["dp"]], Replicate)
    assert isinstance(placements[dm["tp"]], Shard) and placements[dm["tp"]].dim == 0


def test_get_placements_expert_ep_shards_expert_dim0():
    # Grouped expert param [E, inter, hidden]; EP shards the leading expert dim.
    dm = {"dp": 0, "ep": 1}
    placements = get_placements("a.mlp.experts.gate_proj.weight", dm, 3)
    assert _shard_dim_at(placements, dm, "ep") == 0


def test_get_placements_expert_tp_shifts_by_one():
    # No EP axis: experts fall back to TP with the shard dim shifted +1 for the
    # prepended expert dim. gate_proj (col, dim0) -> Shard(1); down_proj (row,
    # dim1) -> Shard(2).
    dm = {"tp": 0}
    assert (
        _shard_dim_at(get_placements("a.mlp.experts.gate_proj.weight", dm, 3), dm, "tp")
        == 1
    )
    assert (
        _shard_dim_at(get_placements("a.mlp.experts.down_proj.weight", dm, 3), dm, "tp")
        == 2
    )


# --------------------------------------------------------------------------
# group_expert_params_in_metadata
# --------------------------------------------------------------------------
def _moe_metadata(num_experts=2, inter=1536, hidden=4096):
    md = {}
    for e in range(num_experts):
        p = f"model.layers.0.mlp.experts.{e}"
        md[f"{p}.gate_proj.weight"] = {
            "shape": [inter, hidden],
            "dtype": "torch.bfloat16",
        }
        md[f"{p}.up_proj.weight"] = {
            "shape": [inter, hidden],
            "dtype": "torch.bfloat16",
        }
        md[f"{p}.down_proj.weight"] = {
            "shape": [hidden, inter],
            "dtype": "torch.bfloat16",
        }
    md["model.layers.0.self_attn.q_proj.weight"] = {
        "shape": [hidden, hidden],
        "dtype": "torch.bfloat16",
    }
    return md


def test_group_expert_params_collapses_to_grouped_hf_entries():
    md = _moe_metadata(num_experts=2, inter=1536, hidden=4096)
    grouped = group_expert_params_in_metadata(md)

    base = "model.layers.0.mlp.experts"
    assert grouped[f"{base}.gate_proj.weight"]["shape"] == [2, 1536, 4096]
    assert grouped[f"{base}.gate_proj.weight"]["grouped_expert_proj"] == "gate_proj"
    assert grouped[f"{base}.up_proj.weight"]["shape"] == [2, 1536, 4096]
    assert grouped[f"{base}.down_proj.weight"]["shape"] == [2, 4096, 1536]
    assert grouped[f"{base}.down_proj.weight"]["grouped_expert_proj"] == "down_proj"

    # Per-expert numeric-index entries are gone; non-expert passes through.
    assert not any(".experts.0." in k or ".experts.1." in k for k in grouped)
    assert "model.layers.0.self_attn.q_proj.weight" in grouped
    assert (
        "grouped_expert_proj" not in grouped["model.layers.0.self_attn.q_proj.weight"]
    )


def test_group_expert_params_no_experts_is_identity():
    md = {
        "model.layers.0.self_attn.q_proj.weight": {
            "shape": [8, 8],
            "dtype": "torch.bfloat16",
        }
    }
    assert group_expert_params_in_metadata(md) == md


# --------------------------------------------------------------------------
# build_nccl_xfer_refit_info
# --------------------------------------------------------------------------
def _dense_metadata(hidden=32, vocab=128, inter=64):
    return {
        "model.embed_tokens.weight": {
            "shape": [vocab, hidden],
            "dtype": "torch.bfloat16",
        },
        "model.layers.0.input_layernorm.weight": {
            "shape": [hidden],
            "dtype": "torch.bfloat16",
        },
        "model.layers.0.self_attn.q_proj.weight": {
            "shape": [hidden, hidden],
            "dtype": "torch.bfloat16",
        },
        "model.layers.0.self_attn.o_proj.weight": {
            "shape": [hidden, hidden],
            "dtype": "torch.bfloat16",
        },
        "model.layers.0.mlp.gate_proj.weight": {
            "shape": [inter, hidden],
            "dtype": "torch.bfloat16",
        },
        "model.layers.0.mlp.down_proj.weight": {
            "shape": [hidden, inter],
            "dtype": "torch.bfloat16",
        },
        "model.norm.weight": {"shape": [hidden], "dtype": "torch.bfloat16"},
        "lm_head.weight": {"shape": [vocab, hidden], "dtype": "torch.bfloat16"},
    }


def _find(info, name):
    for layer in info["layer_names"]:
        for p in info["per_layer_params"][layer]:
            if p["name"] == name:
                return p
    raise AssertionError(f"{name} not in refit_info")


def test_build_refit_info_top_level_and_param_fields():
    info = build_nccl_xfer_refit_info(
        _dense_metadata(),
        train_parallelism={"tp_size": 2, "ep_size": 1, "pp_size": 1},
        gen_parallelism={"tp_size": 4, "ep_size": 1, "pp_size": 1},
        train_world_size=2,
        gen_world_size=4,
    )
    assert set(info) >= {
        "layer_names",
        "per_layer_params",
        "train_world_size",
        "gen_world_size",
        "pp_size",
        "gen_tp_size",
    }
    assert info["train_world_size"] == 2
    assert info["gen_world_size"] == 4
    assert info["pp_size"] == 1
    assert info["gen_tp_size"] == 4

    # Every param carries the canonical fields; no pp_stage when pp_size == 1.
    for layer in info["layer_names"]:
        for p in info["per_layer_params"][layer]:
            assert {
                "name",
                "global_shape",
                "dtype",
                "src_mesh_info",
                "src_placements",
                "dst_mesh_info",
                "dst_placements",
            } <= set(p)
            assert "pp_stage" not in p
            assert isinstance(p["src_mesh_info"], MeshInfo)
            assert isinstance(p["dst_mesh_info"], MeshInfo)

    # q_proj is column-parallel -> Shard(0) on both train (TP2) and gen (TP4).
    q = _find(info, "model.layers.0.self_attn.q_proj.weight")
    assert q["global_shape"] == (32, 32)
    assert any(isinstance(p, Shard) and p.dim == 0 for p in q["src_placements"])
    assert any(isinstance(p, Shard) and p.dim == 0 for p in q["dst_placements"])
    # 1-D norm is fully replicated on both sides.
    ln = _find(info, "model.layers.0.input_layernorm.weight")
    assert all(isinstance(p, Replicate) for p in ln["src_placements"])
    assert all(isinstance(p, Replicate) for p in ln["dst_placements"])


def test_build_refit_info_sets_pp_stage_when_pp_gt_1():
    md = {
        "model.embed_tokens.weight": {"shape": [128, 32], "dtype": "torch.bfloat16"},
        "model.layers.0.self_attn.q_proj.weight": {
            "shape": [32, 32],
            "dtype": "torch.bfloat16",
        },
        "model.layers.1.self_attn.q_proj.weight": {
            "shape": [32, 32],
            "dtype": "torch.bfloat16",
        },
        "lm_head.weight": {"shape": [128, 32], "dtype": "torch.bfloat16"},
    }
    layer_to_pp_stage = {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 1,
        "lm_head": 1,
    }
    info = build_nccl_xfer_refit_info(
        md,
        train_parallelism={"tp_size": 1, "ep_size": 1, "pp_size": 2},
        gen_parallelism={"tp_size": 2, "ep_size": 1, "pp_size": 1},
        train_world_size=2,
        gen_world_size=2,
        layer_to_pp_stage=layer_to_pp_stage,
    )
    assert info["pp_size"] == 2
    for layer in info["layer_names"]:
        for p in info["per_layer_params"][layer]:
            assert p["pp_stage"] == layer_to_pp_stage[layer]


def test_build_refit_info_groups_experts_and_tags_them():
    info = build_nccl_xfer_refit_info(
        _moe_metadata(num_experts=2),
        train_parallelism={"tp_size": 1, "ep_size": 2, "pp_size": 1},
        gen_parallelism={"tp_size": 2, "ep_size": 1, "pp_size": 1},
        train_world_size=2,
        gen_world_size=2,
    )
    gate = _find(info, "model.layers.0.mlp.experts.gate_proj.weight")
    assert gate["grouped_expert_proj"] == "gate_proj"
    assert gate["global_shape"][0] == 2  # grouped leading expert dim E=2
    # Train shards the expert dim (EP=2) -> Shard(0) somewhere in src placements.
    assert any(isinstance(p, Shard) and p.dim == 0 for p in gate["src_placements"])
    # No raw per-expert numeric entries survived grouping.
    assert not any(
        ".experts.0." in p["name"] or ".experts.1." in p["name"]
        for layer in info["layer_names"]
        for p in info["per_layer_params"][layer]
    )
