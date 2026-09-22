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

"""Map policy parallel ranks to logical Energon loader copies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

from nemo_rl.distributed.named_sharding import REPLICATED_AXES, NamedSharding


@dataclass(frozen=True)
class LoaderCopy:
    """One logical data shard and the policy ranks that consume it."""

    logical_rank: int
    logical_world_size: int
    owner_rank: int
    delivery_ranks: tuple[int, ...]


@dataclass(frozen=True)
class DataLoaderPlacementPlan:
    """Stable mapping from logical data shards to policy ranks."""

    copies: tuple[LoaderCopy, ...]
    placement_hash: str

    @property
    def logical_world_size(self) -> int:
        return len(self.copies)

    def copy_for_logical_rank(self, logical_rank: int) -> LoaderCopy:
        """Return the copy for ``logical_rank``."""
        try:
            return self.copies[logical_rank]
        except IndexError as error:
            raise ValueError(
                f"Logical data rank {logical_rank} is outside [0, {len(self.copies)})."
            ) from error


class DataLoaderTopologyMapper(Protocol):
    """Build a logical loader placement for a policy sharding layout."""

    def map(self, sharding: NamedSharding) -> DataLoaderPlacementPlan: ...


class DefaultDataLoaderTopologyMapper:
    """Create one loader per policy DP replica on TP0/PP0/CP0."""

    def map(self, sharding: NamedSharding) -> DataLoaderPlacementPlan:
        """Map each DP coordinate to one owner and all replica consumers."""
        if "data_parallel" not in sharding.names:
            raise ValueError("Policy sharding must define a data_parallel axis.")

        dp_size = sharding.get_axis_size("data_parallel")
        copies: list[LoaderCopy] = []
        for dp_rank in range(dp_size):
            delivery_ranks = tuple(sharding.get_ranks_by_coord(data_parallel=dp_rank))
            owner_coords = {"data_parallel": dp_rank}
            owner_coords.update(
                {axis: 0 for axis in REPLICATED_AXES if axis in sharding.names}
            )
            owner_candidates = sharding.get_ranks_by_coord(**owner_coords)
            if len(owner_candidates) != 1:
                raise ValueError(
                    "Default loader placement requires exactly one TP0/PP0/CP0 "
                    f"owner for DP rank {dp_rank}; found {owner_candidates}."
                )
            copies.append(
                LoaderCopy(
                    logical_rank=dp_rank,
                    logical_world_size=dp_size,
                    owner_rank=owner_candidates[0],
                    delivery_ranks=delivery_ranks,
                )
            )

        payload = {
            "parallel_shape": sharding.shape,
            "parallel_axes": sharding.names,
            "copies": [
                {
                    "logical_rank": copy.logical_rank,
                    "logical_world_size": copy.logical_world_size,
                    "owner_rank": copy.owner_rank,
                    "delivery_ranks": list(copy.delivery_ranks),
                }
                for copy in copies
            ],
        }
        placement_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return DataLoaderPlacementPlan(
            copies=tuple(copies), placement_hash=placement_hash
        )


def resolve_topology_mapper(name: str) -> DataLoaderTopologyMapper:
    """Resolve the Stage 1 topology mapper."""
    if name != "default":
        raise ValueError(f"Unknown data-loader topology mapper {name!r}.")
    return DefaultDataLoaderTopologyMapper()
