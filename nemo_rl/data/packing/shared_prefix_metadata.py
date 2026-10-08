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

"""Compatibility adapter for :mod:`megatron.rl.shared_prefix_metadata`.

The implementation is owned by Megatron. Imports stay lazy so ordinary dense
NeMo RL backends do not require the optional Megatron installation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

if TYPE_CHECKING:
    from megatron.rl.shared_prefix_metadata import (
        GroupCoherentShardPlan,
        FixedExecutionSlotPlan,
    )

__all__ = [
    "GroupCoherentShardPlan",
    "FixedExecutionSlotPlan",
    "plan_fixed_execution_slots",
    "plan_group_coherent_shards",
    "make_repeated_group_ids",
]


def __getattr__(name: str) -> Any:
    if name not in ("GroupCoherentShardPlan", "FixedExecutionSlotPlan"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module("megatron.rl.shared_prefix_metadata"), name)
    globals()[name] = value
    return value


def plan_fixed_execution_slots(
    *,
    group_ids: Sequence[str],
    sequence_lengths: Sequence[int],
    bin_capacity: int,
    batch_size: int | None = None,
    sequence_length_pad_multiple: int = 1,
    max_rows_per_slot: int = 16,
) -> FixedExecutionSlotPlan:
    """Delegate to :func:`megatron.rl.shared_prefix_metadata.plan_fixed_execution_slots`."""
    from megatron.rl.shared_prefix_metadata import (
        plan_fixed_execution_slots as implementation,
    )

    return implementation(
        group_ids=group_ids,
        sequence_lengths=sequence_lengths,
        bin_capacity=bin_capacity,
        batch_size=batch_size,
        sequence_length_pad_multiple=sequence_length_pad_multiple,
        max_rows_per_slot=max_rows_per_slot,
    )


def plan_group_coherent_shards(
    *,
    group_ids: Sequence[str],
    sequence_lengths: Sequence[int],
    num_shards: int,
    batch_size: int | None = None,
) -> GroupCoherentShardPlan:
    """Delegate to :func:`megatron.rl.shared_prefix_metadata.plan_group_coherent_shards`."""
    from megatron.rl.shared_prefix_metadata import (
        plan_group_coherent_shards as implementation,
    )

    return implementation(
        group_ids=group_ids,
        sequence_lengths=sequence_lengths,
        num_shards=num_shards,
        batch_size=batch_size,
    )


def make_repeated_group_ids(
    *,
    num_rows: int,
    group_size: int,
    namespace: str,
) -> list[str]:
    """Delegate to :func:`megatron.rl.shared_prefix_metadata.make_repeated_group_ids`."""
    from megatron.rl.shared_prefix_metadata import (
        make_repeated_group_ids as implementation,
    )

    return implementation(num_rows=num_rows, group_size=group_size, namespace=namespace)


SHARED_PREFIX_GROUP_ID = "shared_prefix_group_id"
SHARED_PREFIX_PROMPT_LENGTHS = "shared_prefix_prompt_lengths"
SHARED_PREFIX_EXECUTION_SLOT = "_shared_prefix_execution_slot"


def stamp_repeated_group_ids(
    batch: "BatchedDataDict[Any]",
    *,
    group_size: int,
    namespace: str,
) -> None:
    """Attach validated prompt-group IDs to a repeated batch in place."""
    if SHARED_PREFIX_GROUP_ID in batch:
        raise ValueError(
            f"batch already contains reserved field {SHARED_PREFIX_GROUP_ID!r}"
        )
    batch[SHARED_PREFIX_GROUP_ID] = make_repeated_group_ids(
        num_rows=batch.size,
        group_size=group_size,
        namespace=namespace,
    )


def parse_grouped_sample_id(sample_id: str) -> tuple[str, int]:
    """Parse a TQ sample ID following the ``{group_id}_g{index}`` contract."""
    if not isinstance(sample_id, str):
        raise TypeError(
            f"shared-prefix sample IDs must be strings, got {type(sample_id).__name__}"
        )
    group_id, separator, generation_index = sample_id.rpartition("_g")
    if (
        not separator
        or not group_id
        or not generation_index.isascii()
        or not generation_index.isdigit()
    ):
        raise ValueError(
            "shared-prefix sample IDs must use the form '{group_id}_g{index}', "
            f"got {sample_id!r}"
        )
    return group_id, int(generation_index)


def group_id_from_sample_id(sample_id: str) -> str:
    """Recover the TQ prompt-group prefix from ``{group_id}_g{index}``."""
    group_id, _ = parse_grouped_sample_id(sample_id)
    return group_id
