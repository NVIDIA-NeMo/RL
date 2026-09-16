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
"""Validate complete logical owners before the existing GRPO estimator runs."""

from dataclasses import dataclass

import torch

from nemo_rl.data_plane import KVBatchMeta

_OWNER_TAGS = frozenset(
    {
        "dispatch_group_id",
        "logical_rollout_id",
        "logical_slot",
        "logical_group_size",
        "segment_index",
        "segment_count",
        "is_execution_padding",
    }
)


def has_logical_owners(meta: KVBatchMeta) -> bool:
    """Detect even partially tagged CC batches so they cannot take ordinary GRPO."""
    return any(_OWNER_TAGS.intersection(tag) for tag in meta.tags or [])


@dataclass(frozen=True)
class LogicalOwnerBatch:
    """Indices into physical rows; padding has row_owner=-1 and no reward vote."""

    representative_rows: torch.Tensor
    group_ids: torch.Tensor
    row_owner: torch.Tensor
    valid_mask: torch.Tensor

    def fanout(self, values: torch.Tensor) -> torch.Tensor:
        """Broadcast one scalar per owner, with zero for ownerless padding."""
        return values[self.row_owner.clamp_min(0)] * (self.row_owner >= 0)


def build_logical_owner_batch(
    meta: KVBatchMeta,
    *,
    prompt_ids: torch.Tensor,
    rewards: torch.Tensor,
    sample_mask: torch.Tensor,
    expected_group_size: int,
) -> LogicalOwnerBatch:
    """Check complete groups, deduplicate owner rows, and combine their validity.

    Dispatch IDs determine the estimator's grouping, independently of prompt
    tokens. Each logical owner contributes once, regardless of segment count.
    """
    if (
        meta.tags is None
        or len(meta.tags) != meta.size
        or len(set(meta.sample_ids)) != meta.size
        or prompt_ids.ndim != 2
        or prompt_ids.shape[0] != meta.size
        or prompt_ids.dtype not in (torch.int32, torch.int64)
        or rewards.shape != (meta.size,)
        or sample_mask.shape != (meta.size,)
        or not torch.isfinite(rewards).all()
        or not ((sample_mask == 0) | (sample_mask == 1)).all()
    ):
        raise ValueError(
            "CC advantage inputs must be finite, binary-masked, row-aligned"
        )

    groups: dict[str, dict[int, list[int]]] = {}
    for row, tag in enumerate(meta.tags):
        if not _OWNER_TAGS.issubset(tag):
            raise ValueError("CC batches require complete owner tags on every row")
        group = tag["dispatch_group_id"]
        size = tag["logical_group_size"]
        if (
            not isinstance(group, str)
            or not group
            or type(size) is not int
            or size != expected_group_size
            or type(tag["is_execution_padding"]) is not bool
        ):
            raise ValueError("CC dispatch group identity or cardinality is invalid")
        slots = groups.setdefault(group, {})
        if tag["is_execution_padding"]:
            if (
                tag["logical_rollout_id"] is not None
                or tag["logical_slot"] is not None
                or tag["segment_index"] is not None
                or type(tag["segment_count"]) is not int
                or tag["segment_count"] != 0
                or sample_mask[row] != 0
            ):
                raise ValueError("CC execution padding must be ownerless and masked")
            continue
        slot, segment, count = (
            tag["logical_slot"],
            tag["segment_index"],
            tag["segment_count"],
        )
        if (
            type(slot) is not int
            or not 0 <= slot < size
            or type(segment) is not int
            or type(count) is not int
            or not 0 <= segment < count
            or tag["logical_rollout_id"] != f"{group}_g{slot}"
            or meta.sample_ids[row] != f"{group}_g{slot}_s{segment}"
        ):
            raise ValueError("CC owner/segment identity is invalid")
        slots.setdefault(slot, []).append(row)

    representatives, validity, group_ids = [], [], []
    row_owner = torch.full((meta.size,), -1, dtype=torch.long, device=rewards.device)
    for group_index, slots in enumerate(groups.values()):
        if set(slots) != set(range(expected_group_size)):
            raise ValueError("CC estimator requires every logical slot in each group")
        for rows in slots.values():
            first = rows[0]
            counts = {meta.tags[row]["segment_count"] for row in rows}
            indices = {meta.tags[row]["segment_index"] for row in rows}
            if counts != {len(rows)} or indices != set(range(len(rows))):
                raise ValueError("CC estimator requires every segment exactly once")
            if any(
                rewards[row] != rewards[first]
                or not torch.equal(prompt_ids[row], prompt_ids[first])
                for row in rows[1:]
            ):
                raise ValueError("CC owner rows disagree on reward or original prompt")
            row_owner[rows] = len(representatives)
            representatives.append(first)
            group_ids.append(group_index)
            validity.append(sample_mask[rows].amin())
    if not representatives:
        raise ValueError("CC estimator requires logical owners, not padding alone")
    return LogicalOwnerBatch(
        representative_rows=torch.tensor(
            representatives, dtype=torch.long, device=rewards.device
        ),
        group_ids=torch.tensor(group_ids, dtype=torch.long, device=rewards.device)[
            :, None
        ],
        row_owner=row_owner,
        valid_mask=torch.stack(validity),
    )
