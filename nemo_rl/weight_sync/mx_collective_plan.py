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

"""Translate NeMo RL refit metadata into a ModelExpress collective plan.

The bytes move the same way either transport is chosen: both end in
``nccl.m2n.reshard`` with the same meshes and placements. What ModelExpress
adds is the rendezvous -- admission against an expected participant set,
fencing of a stale worker generation, and a readiness state the trainer can
observe before it enters the collective. Where the native path bootstraps
through a ``TCPStore`` at an address the driver has to allocate per pipeline
stage, MX brokers the ``ncclUniqueId`` itself.

So this module is a translation layer and nothing more. It takes the metadata
``build_nccl_reshard_refit_info`` already produces and expresses it in MX's
plan vocabulary; it does not re-derive meshes, placements, or the bulk split.
"""

from typing import Any, Optional

import torch

from nemo_rl.weight_sync.nccl_reshard_utils import MeshInfo

_DTYPE_NAMES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float8_e4m3fn: "float8_e4m3fn",
    torch.float8_e5m2: "float8_e5m2",
}


class MxPlanTranslationError(ValueError):
    """The refit metadata cannot be expressed as a ModelExpress plan.

    Raised rather than approximated. Every case below would otherwise produce a
    plan that looks valid and describes a different transfer, and a wrong mesh
    does not fail the collective -- it moves the wrong bytes.
    """


def dtype_name(dtype: Any) -> str:
    """Canonical dtype string for the plan digest.

    Both sides hash this, so an unrecognized dtype has to raise rather than
    fall back to ``str(dtype)``: two workers disagreeing on the spelling would
    compute different digests and never form a group.
    """
    if isinstance(dtype, str):
        return dtype.removeprefix("torch.")
    name = _DTYPE_NAMES.get(dtype)
    if name is None:
        raise MxPlanTranslationError(f"no canonical name for dtype {dtype!r}")
    return name


def mesh_to_spec(mesh: MeshInfo) -> tuple[tuple[int, ...], int]:
    """Convert a ``MeshInfo`` rank grid into MX's (shape, rank_offset) form.

    MX describes a mesh as a shape plus the lane-local rank its grid starts at,
    which assumes the ranks are contiguous and ascending in row-major order.
    Every mesh ``build_mesh_info`` produces satisfies that, because it builds
    them from ``torch.arange(offset, offset + n).reshape(...)``. A mesh that
    does not is rejected here rather than silently flattened into a different
    topology.
    """
    tensor = getattr(mesh, "mesh", None)
    if tensor is None:
        tensor = getattr(mesh, "_mesh", None)
    if tensor is None:
        raise MxPlanTranslationError("mesh does not expose its rank grid")
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)

    shape = tuple(int(extent) for extent in tensor.shape)
    flat = [int(rank) for rank in tensor.flatten().tolist()]
    if not flat:
        raise MxPlanTranslationError("mesh has no ranks")

    offset = flat[0]
    if flat != list(range(offset, offset + len(flat))):
        raise MxPlanTranslationError(
            "ModelExpress requires a mesh whose ranks are contiguous and ascending "
            f"in row-major order; got {flat[:8]}"
        )
    return shape, offset


def placements_to_mx(placements: list[Any]) -> tuple[Any, ...]:
    """Convert DTensor placements into MX's torch-free records."""
    from modelexpress_rl.collective import Placement

    converted = []
    for placement in placements:
        if getattr(placement, "is_shard", None) is not None and placement.is_shard():
            converted.append(Placement.shard(int(placement.dim)))
        elif isinstance(placement, dict) and "dim" in placement:
            converted.append(Placement.shard(int(placement["dim"])))
        else:
            converted.append(Placement.replicate())
    return tuple(converted)


def build_mx_plan(
    refit_info: dict,
    misc_meta: Optional[dict] = None,
) -> Any:
    """Build a ModelExpress ``ReshardPlan`` from NeMo RL refit metadata.

    ``refit_info`` is what ``build_nccl_reshard_refit_info`` returns.
    ``misc_meta`` is the ordered mapping of parameters that ride the packed
    broadcast; its order is preserved because it is the broadcast payload
    layout and MX folds it into the plan digest.
    """
    from modelexpress_rl.collective import MiscParam, ParamPlan, ReshardPlan

    bulk = []
    for layer in refit_info.get("layer_names", []):
        for info in refit_info.get("per_layer_params", {}).get(layer, []):
            src_shape, src_offset = mesh_to_spec(info["src_mesh_info"])
            dst_shape, dst_offset = mesh_to_spec(info["dst_mesh_info"])
            bulk.append(
                ParamPlan(
                    name=info["name"],
                    global_shape=tuple(int(e) for e in info["global_shape"]),
                    dtype=dtype_name(info["dtype"]),
                    partition_id=int(info.get("pp_stage", 0)),
                    src_mesh=_mesh_spec(src_shape, src_offset),
                    src_placements=placements_to_mx(info["src_placements"]),
                    dst_mesh=_mesh_spec(dst_shape, dst_offset),
                    dst_placements=placements_to_mx(info["dst_placements"]),
                    group_key=info.get("grouped_expert_proj"),
                )
            )

    misc = [
        MiscParam(
            name=name,
            global_shape=tuple(int(e) for e in meta["shape"]),
            dtype=dtype_name(meta["dtype"]),
        )
        for name, meta in (misc_meta or {}).items()
    ]

    return ReshardPlan(
        bulk=bulk,
        misc=misc,
        source_partition_count=int(refit_info.get("pp_size", 1)),
    )


def _mesh_spec(shape: tuple[int, ...], offset: int) -> Any:
    from modelexpress_rl.collective import MeshSpec

    return MeshSpec(shape=shape, rank_offset=offset)


def slot_ids(
    train_world_size: int,
    gen_world_size: int,
) -> tuple[list[str], list[str]]:
    """Stable participant identities for one MX group.

    Slots are logical and survive a restart; MX admits a slot once and fences a
    second worker generation claiming it. Deriving them from global rank keeps
    them stable across the run without needing any new bookkeeping.
    """
    trainers = [f"train/{rank}" for rank in range(train_world_size)]
    generators = [f"gen/{rank}" for rank in range(gen_world_size)]
    return trainers, generators
