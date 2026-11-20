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
"""Local sharing and learner-world residual transfers for co-resident refit."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from nemo_rl.models.megatron.sm_partition import RefitSlices, RefitSource

if TYPE_CHECKING:
    from megatron.core.resharding.utils import ParameterMetadata, ReshardPlan

_FLOAT_BYTES = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}
_GROUP_FIELDS = (
    "tensor_parallel_group_ranks",
    "expert_parallel_group_ranks",
    "data_parallel_group_ranks",
    "pipeline_parallel_group_ranks",
)


@dataclass(frozen=True)
class LocalSourcePlan:
    """Engine-local names mapped to local views or residual destination metadata."""

    local_sources: dict[str, RefitSource]
    residual: tuple["ParameterMetadata", ...]


def _validate_metadata(metadata: "ParameterMetadata") -> str:
    resolved = metadata.resolved_name
    if not metadata.name or not resolved:
        raise ValueError("Refit metadata needs explicit local and resolved names")
    if type(metadata.owner_rank) is not int or metadata.owner_rank < 0:
        raise ValueError(f"Invalid owner rank for {metadata.name}")
    if (
        metadata.dtype not in _FLOAT_BYTES
        or metadata.element_size != _FLOAT_BYTES[metadata.dtype]
    ):
        raise ValueError(f"Unsupported dtype or element size for {metadata.name}")
    if metadata.is_gtp or metadata.gtp_pad_length or metadata.gtp_remat_group_ranks:
        raise ValueError("Co-resident residual refit does not support GTP weights")
    if any(type(extent) is not int or extent <= 0 for extent in metadata.shape):
        raise ValueError(f"Invalid shape for {metadata.name}: {metadata.shape}")
    if type(metadata.partition_stride) is not int or metadata.partition_stride < 1:
        raise ValueError(f"Invalid partition stride for {metadata.name}")
    if metadata.is_tp and not 0 <= metadata.partition_dim < len(metadata.shape):
        raise ValueError(f"Invalid TP partition dimension for {metadata.name}")
    if metadata.partition_sizes is not None:
        if not 0 <= metadata.partition_dim < len(metadata.shape) or any(
            type(size) is not int or size <= 0 for size in metadata.partition_sizes
        ):
            raise ValueError(f"Invalid packed partition sizes for {metadata.name}")
        if sum(metadata.partition_sizes) != metadata.shape[metadata.partition_dim]:
            raise ValueError(f"Packed partition sizes do not cover {metadata.name}")
    for field in _GROUP_FIELDS:
        ranks = getattr(metadata, field)
        if ranks is not None and (
            not ranks
            or any(type(rank) is not int or rank < 0 for rank in ranks)
            or len(set(ranks)) != len(ranks)
            or metadata.owner_rank not in ranks
        ):
            raise ValueError(f"Invalid {field} for {metadata.name}")
    if metadata.is_tp and metadata.tensor_parallel_group_ranks is None:
        raise ValueError(f"Missing TP group for {metadata.name}")
    if metadata.is_ep and (
        metadata.expert_parallel_group_ranks is None
        or metadata.global_expert_index is None
    ):
        raise ValueError(
            f"Expert refit needs canonical per-expert metadata: {metadata.name}"
        )

    return resolved


def remap_metadata(
    metadata: list["ParameterMetadata"],
    engine_to_learner: dict[int, int],
    *,
    learner_world_size: int | None = None,
) -> list["ParameterMetadata"]:
    """Copy engine metadata into learner-world ranks, retaining group order and names."""
    world = set(range(len(engine_to_learner)))
    learner_world = len(world) if learner_world_size is None else learner_world_size
    if (
        not world
        or type(learner_world) is not int
        or learner_world < len(world)
        or any(
            type(rank) is not int for pair in engine_to_learner.items() for rank in pair
        )
        or set(engine_to_learner) != world
        or len(set(engine_to_learner.values())) != len(world)
        or any(rank < 0 or rank >= learner_world for rank in engine_to_learner.values())
    ):
        raise ValueError(
            "Engine-to-learner mapping must be a complete equal-world bijection"
            if learner_world_size is None
            else "Engine-to-learner mapping must cover the engine world with distinct, bounded learner ranks"
        )
    remapped = deepcopy(metadata)
    for item in remapped:
        _validate_metadata(item)
        ranks = {item.owner_rank}
        for field in _GROUP_FIELDS:
            ranks.update(getattr(item, field) or ())
        if not ranks <= world:
            raise ValueError(f"Metadata rank is outside the placement map: {item.name}")
        item.owner_rank = engine_to_learner[item.owner_rank]
        for field in _GROUP_FIELDS:
            group = getattr(item, field)
            if group is not None:
                setattr(item, field, [engine_to_learner[rank] for rank in group])
    return remapped


def plan_local_sources(
    src: dict[str, torch.Tensor],
    all_src_metadata: list["ParameterMetadata"],
    dst_metadata: list["ParameterMetadata"],
    *,
    learner_rank: int,
) -> LocalSourcePlan:
    """Prove complete native coverage, then separate local views from residual copies.

    Source tensors are keyed by resolved name on the paired learner GPU.
    Destination metadata must already be remapped into learner-world ranks.
    A destination needing any remote fragment remains wholly private.
    """
    from megatron.core.resharding.planner import _find_source_metadata
    from megatron.core.resharding.shard_planner import plan_sharded_transfer
    from megatron.core.resharding.utils import select_src_metadata_balanced

    roster: dict[str, list[ParameterMetadata]] = defaultdict(list)
    local: dict[str, ParameterMetadata] = {}
    owner_names: set[tuple[int, str]] = set()
    owner_resolved: set[tuple[int, str]] = set()
    for item in all_src_metadata:
        resolved = _validate_metadata(item)
        name_key, resolved_key = (
            (item.owner_rank, item.name),
            (item.owner_rank, resolved),
        )
        if name_key in owner_names or resolved_key in owner_resolved:
            raise ValueError("Duplicate source metadata on one learner rank")
        owner_names.add(name_key)
        owner_resolved.add(resolved_key)
        roster[resolved].append(item)
        if item.owner_rank == learner_rank:
            local[resolved] = item
    if src.keys() != local.keys():
        raise ValueError("Local source tensors and owned metadata must match exactly")
    for name, tensor in src.items():
        item = local[name]
        if (
            type(tensor) not in (torch.Tensor, torch.nn.Parameter)
            or tuple(tensor.shape) != tuple(item.shape)
            or tensor.dtype != item.dtype
            or tensor.layout != torch.strided
        ):
            raise ValueError(
                f"Unsupported tensor storage or stale local metadata for {name}"
            )
    mapped: dict[str, RefitSource] = {}
    residual = []
    names, resolved_names = set(), set()
    for destination in dst_metadata:
        resolved = _validate_metadata(destination)
        if destination.owner_rank != learner_rank:
            raise ValueError(
                "Destination metadata must use its paired learner owner rank"
            )
        if destination.name in names or resolved in resolved_names:
            raise ValueError("Duplicate destination metadata")
        names.add(destination.name)
        resolved_names.add(resolved)
        candidates = _find_source_metadata(roster, resolved)
        if not candidates:
            raise ValueError(f"No learner owns {destination.resolved_name}")
        if any(item.dtype != destination.dtype for item in candidates):
            raise ValueError(
                f"Residual transport requires matching dtypes for {destination.resolved_name}"
            )
        selected = select_src_metadata_balanced(candidates, destination, learner_rank)
        operations = plan_sharded_transfer(resolved, candidates, selected, destination)
        if any(owner != learner_rank for owner, _, _ in operations):
            if (
                not selected.is_tp
                and not destination.is_tp
                and (
                    selected.data_parallel_group_ranks is None
                    or destination.data_parallel_group_ranks is None
                )
            ):
                raise ValueError(
                    f"Residual replica transfer needs DP groups for {resolved}"
                )
            residual.append(deepcopy(destination))
            continue
        source = src[_validate_metadata(selected)]
        parts = tuple(
            (source[source_slice], target_slice)
            for _, source_slice, target_slice in operations
        )
        if len(parts) == 1 and all(
            part.start == 0 and part.stop == extent
            for part, extent in zip(parts[0][1], destination.shape)
        ):
            mapped[destination.name] = parts[0][0]
        else:
            mapped[destination.name] = RefitSlices(tuple(destination.shape), parts)
    return LocalSourcePlan(mapped, tuple(residual))


def build_residual_plan(
    destinations: dict[int, dict[str, "ParameterMetadata"]],
    sources: dict[str, list["ParameterMetadata"]],
    rank: int,
    *,
    execution_batch_bytes: Optional[int] = None,
) -> "ReshardPlan":
    """Build the native point-to-point schedule for private inference tensors.

    Residual rosters retain the logical TP coordinates of locally bound shards.
    Only private destinations receive data, so a whole-mesh collective would
    write into weights that may alias the learner.
    """
    from megatron.core.resharding.planner import (
        _build_execution_batch_ids,
        _iter_global_transfer_ops,
    )
    from megatron.core.resharding.utils import ReshardPlan, TransferOp

    batches, num_batches = _build_execution_batch_ids(
        destinations, sources, max_batch_bytes=execution_batch_bytes
    )
    plan = ReshardPlan(
        [], [], num_batches=num_batches, execution_batch_bytes=execution_batch_bytes
    )
    total_tasks = 0
    for (
        task_id,
        dst,
        src,
        src_slice,
        dst_slice,
        sender,
        receiver,
    ) in _iter_global_transfer_ops(destinations, sources):
        total_tasks = task_id + 1
        batch_id = batches[receiver.resolved_name or receiver.name]
        if src == rank:
            plan.send_ops.append(
                TransferOp(
                    param_name=sender.name,
                    peer_rank=dst,
                    is_send=True,
                    my_slice=src_slice,
                    peer_slice=dst_slice,
                    task_id=task_id,
                    batch_id=batch_id,
                )
            )
        if dst == rank:
            plan.recv_ops.append(
                TransferOp(
                    param_name=receiver.name,
                    peer_rank=src,
                    is_send=False,
                    my_slice=dst_slice,
                    peer_slice=src_slice,
                    task_id=task_id,
                    batch_id=batch_id,
                )
            )
    plan.total_tasks = total_tasks
    return plan


def named_tensor_proxy(tensors: dict[str, torch.Tensor]) -> torch.nn.Module:
    """Expose exact destination tensors through native dotted-path buffer enumeration."""
    from megatron.core.resharding.utils import named_refit_tensors

    proxy = torch.nn.Module()
    identities = set()
    for name, tensor in tensors.items():
        if not name or any(not part for part in name.split(".")):
            raise ValueError(f"Invalid destination tensor name: {name!r}")
        if (
            type(tensor) not in (torch.Tensor, torch.nn.Parameter)
            or tensor.dtype not in _FLOAT_BYTES
            or tensor.layout != torch.strided
        ):
            raise ValueError(f"Unsupported destination tensor: {name}")
        if id(tensor) in identities:
            raise ValueError(
                "Distinct proxy names cannot reference the exact same Tensor object"
            )
        identities.add(id(tensor))
        parent = proxy
        parts = name.split(".")
        for part in parts[:-1]:
            if part not in parent._modules:
                if hasattr(parent, part):
                    raise ValueError(
                        f"Destination module/tensor path collision: {name}"
                    )
                parent.add_module(part, torch.nn.Module())
            parent = parent.get_submodule(part)
        if hasattr(parent, parts[-1]):
            raise ValueError(f"Destination tensor path collision: {name}")
        parent.register_buffer(parts[-1], tensor)
    actual = dict(named_refit_tensors(proxy))
    if actual.keys() != tensors.keys() or any(
        actual[name] is not tensor for name, tensor in tensors.items()
    ):
        raise ValueError(
            "Native proxy enumeration changed destination names or tensors"
        )
    return proxy
