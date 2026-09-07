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

from __future__ import annotations

import numpy as np
import pytest

from nemo_rl.data.energon.topology import (
    DefaultDataLoaderTopologyMapper,
    resolve_topology_mapper,
)
from nemo_rl.distributed.named_sharding import NamedSharding


def _sharding(*, dp: int, pp: int = 1, cp: int = 1, tp: int = 1) -> NamedSharding:
    world_size = dp * pp * cp * tp
    return NamedSharding(
        np.arange(world_size).reshape(pp, dp, cp, tp),
        [
            "pipeline_parallel",
            "data_parallel",
            "context_parallel",
            "tensor_parallel",
        ],
    )


@pytest.mark.parametrize("dp", [1, 2, 4])
def test_default_mapper_creates_one_copy_per_dp_replica(dp: int) -> None:
    plan = DefaultDataLoaderTopologyMapper().map(_sharding(dp=dp))

    assert plan.logical_world_size == dp
    assert [copy.logical_rank for copy in plan.copies] == list(range(dp))
    assert [copy.owner_rank for copy in plan.copies] == list(range(dp))


def test_default_mapper_delivers_to_all_tp_pp_cp_ranks() -> None:
    sharding = _sharding(dp=2, pp=2, cp=2, tp=2)
    plan = DefaultDataLoaderTopologyMapper().map(sharding)

    for copy in plan.copies:
        expected = tuple(sharding.get_ranks_by_coord(data_parallel=copy.logical_rank))
        assert copy.delivery_ranks == expected
        owner_coords = sharding.get_worker_coords(copy.owner_rank)
        assert owner_coords == {
            "pipeline_parallel": 0,
            "data_parallel": copy.logical_rank,
            "context_parallel": 0,
            "tensor_parallel": 0,
        }


def test_placement_hash_is_stable_and_layout_sensitive() -> None:
    mapper = DefaultDataLoaderTopologyMapper()
    first = mapper.map(_sharding(dp=2, cp=2))
    second = mapper.map(_sharding(dp=2, cp=2))
    different = mapper.map(_sharding(dp=2, tp=2))

    assert first.placement_hash == second.placement_hash
    assert first.placement_hash != different.placement_hash


def test_resolver_rejects_unknown_mapper() -> None:
    with pytest.raises(ValueError, match="Unknown data-loader topology mapper"):
        resolve_topology_mapper("mimo")
