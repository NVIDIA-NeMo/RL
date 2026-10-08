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

from megatron.rl.shared_prefix_dense_bins import (
    plan_dense_training_bins as plan_dense_bins,
    share_prefixes_in_dense_training_bins as share_dense_bins,
)
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
    packer = get_packer(
        packing["algorithm"],
        bin_capacity=bin_capacity,
        max_sequences_per_bin=packing.get("max_sequences_per_bin"),
    )
    return plan_dense_bins(
        costs=costs, bin_capacity=bin_capacity, dense_packer=packer.pack
    )


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
    rows = _build_shared_prefix_rows(_normalize_shared_prefix_group_ids(data))
    return share_dense_bins(
        rows, units, costs=costs, padding_multiple=multiple, bin_capacity=bin_capacity
    )
