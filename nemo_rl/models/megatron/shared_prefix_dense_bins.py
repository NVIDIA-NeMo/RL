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
"""Opt-in training packing: align dense row bins, then share exact prefixes."""

from collections.abc import Sequence
from typing import Any, cast

from nemo_rl.data.packing import SharedPrefixForestLayout, build_shared_prefix_layout
from nemo_rl.data.packing.algorithms import get_packer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.data import (
    _build_shared_prefix_rows,
    _normalize_shared_prefix_group_ids,
    _resolve_shared_prefix_execution_topology,
    _SharedPrefixExecutionUnit,
)
from nemo_rl.models.policy import PolicyConfig, SequencePackingConfig


def _row_costs(data: BatchedDataDict[Any], multiple: int) -> list[int]:
    return [
        (int(length) + multiple - 1) // multiple * multiple
        for length in data["input_lengths"].tolist()
    ]


def _validate_dense_partition(
    units: Sequence[_SharedPrefixExecutionUnit],
    costs: Sequence[int],
    capacity: int,
) -> None:
    if sorted(row for unit in units for row in unit.row_indices) != list(
        range(len(costs))
    ):
        raise ValueError("Dense training bins must cover every source row exactly once")
    for unit in units:
        expanded = sum(costs[row] for row in unit.row_indices)
        if (
            unit.shared_layout is not None
            or not unit.row_indices
            or not 0 < expanded <= capacity
            or unit.physical_length != expanded
        ):
            raise ValueError(
                "Expected conventional training bins within the token budget"
            )


def plan_dense_training_bins(
    data: BatchedDataDict[Any], *, cfg: PolicyConfig, bin_capacity: int
) -> tuple[_SharedPrefixExecutionUnit, ...]:
    """Pack full rows across groups; leave prefix reuse until after DP alignment."""
    *_, multiple = _resolve_shared_prefix_execution_topology(cfg)
    costs = _row_costs(data, multiple)
    if any(cost <= 0 or cost > bin_capacity for cost in costs):
        raise ValueError("Training rows must fit the expanded token budget")
    # Shared-prefix validation already requires sequence packing to be enabled.
    packing = cast(SequencePackingConfig, cfg["sequence_packing"])
    if packing.get("pair_grouping_key") is not None:
        raise ValueError("Dense-bin prefix sharing does not support pair_grouping_key")
    bins = get_packer(
        packing["algorithm"],
        bin_capacity=bin_capacity,
        max_sequences_per_bin=packing.get("max_sequences_per_bin"),
    ).pack(costs)
    units = tuple(
        _SharedPrefixExecutionUnit(
            row_indices=tuple(indices),
            shared_layout=None,
            physical_length=sum(costs[row] for row in indices),
        )
        for indices in bins
    )
    _validate_dense_partition(units, costs, bin_capacity)
    return units


def share_prefixes_in_dense_training_bins(
    data: BatchedDataDict[Any],
    units: Sequence[_SharedPrefixExecutionUnit],
    *,
    cfg: PolicyConfig,
    bin_capacity: int,
) -> tuple[_SharedPrefixExecutionUnit, ...]:
    """Replace each aligned dense bin with an exact forest when it saves work.

    Every bin retains one MTP auxiliary-loss normalization group, including
    independent singleton roots. Causal boundaries remain per source row.
    Ineligible bins remain conventional, with no dropped or synthetic rows.
    """
    *_, multiple = _resolve_shared_prefix_execution_topology(cfg)
    costs = _row_costs(data, multiple)
    _validate_dense_partition(units, costs, bin_capacity)
    rows = _build_shared_prefix_rows(_normalize_shared_prefix_group_ids(data))
    result = []
    for unit in units:
        selected = [rows[index] for index in unit.row_indices]
        if any(
            row.group_id is None or row.prompt_length == 0 or row.completion_length == 0
            for row in selected
        ):
            result.append(unit)
            continue
        groups = {}
        for row in selected:
            groups.setdefault((row.group_id, row.prompt_token_ids), []).append(row)
        if all(len(group) == 1 for group in groups.values()):
            result.append(unit)
            continue
        roots = tuple(
            build_shared_prefix_layout(
                group[start : start + 16],
                sequence_length_pad_multiple=multiple,
                allow_singleton=True,
            )
            for group in groups.values()
            for start in range(0, len(group), 16)
        )
        forest = SharedPrefixForestLayout(
            roots, mtp_loss_group_root_counts=(len(roots),)
        )
        expanded = sum(
            len(root.row_indices) * root.prompt_length
            + sum(root.physical_completion_lengths)
            for root in roots
        )
        padded_physical = (
            (forest.physical_total_length + multiple - 1) // multiple * multiple
        )
        if (
            expanded != unit.physical_length
            or padded_physical > expanded
            or sorted(forest.row_indices) != sorted(unit.row_indices)
        ):
            raise ValueError("Shared reconstruction changed the dense training bin")
        result.append(
            _SharedPrefixExecutionUnit(
                row_indices=forest.row_indices,
                shared_layout=forest,
                physical_length=forest.physical_total_length,
            )
        )
    return tuple(result)
