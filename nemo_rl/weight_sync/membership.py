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

"""Rank layout for a refit communicator rebuilt over the surviving generation shards.

Kept as pure arithmetic, separate from the Ray dispatch that applies it, because this is
the part that has to be exactly right and the part that cannot be exercised on a
workstation: reproducing a shard loss needs at least three GPUs (one trainer and two
generation shards, so that losing one still leaves a fleet). The dispatch is a handful of
``.remote()`` calls; the rank layout is where an off-by-one silently corrupts a refit.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RefitMembership:
    """Where each surviving generation shard sits in the rebuilt communicator.

    Attributes:
        world_size: total ranks, training plus surviving generation.
        train_world_size: unchanged; trainers are never excluded, which is what keeps
            the broadcast root (rank 0) stable across a rebuild.
        shard_prefixes: surviving shard index -> its rank prefix within the generation
            block. Insertion-ordered by shard index.
        workers_per_shard: Ray workers backing one shard (tp x pp).
    """

    world_size: int
    train_world_size: int
    shard_prefixes: dict[int, int]
    workers_per_shard: int

    @property
    def surviving_shards(self) -> list[int]:
        return list(self.shard_prefixes)


class NoSurvivingShards(RuntimeError):
    """Every generation shard is gone, so there is nothing to rebuild onto."""


def plan_refit_membership(
    *,
    surviving_shards: Sequence[int],
    dp_size: int,
    total_gen_workers: int,
    train_world_size: int,
) -> RefitMembership:
    """Lay out a refit communicator containing only the surviving generation shards.

    Prefixes are reassigned so the surviving ranks are **contiguous** from the start of
    the generation block, rather than leaving a hole where the dead shard was. That is
    not cosmetic: the nccl_reshard transport builds its destination device mesh as
    ``torch.arange(rank_offset, rank_offset + num_gpus)``, so a gap would silently
    misalign every parameter's placements. It also matches what ``shrink`` does to a
    live communicator, which keeps the two paths describing the same world.

    Args:
        surviving_shards: shard indices still able to take part. Order is ignored; the
            result is sorted, so a caller that hands them over in discovery order still
            gets a deterministic layout.
        dp_size: generation data-parallel size, i.e. the shard count.
        total_gen_workers: Ray workers across the whole generation fleet.
        train_world_size: training ranks, all of which stay in the communicator.

    Raises:
        NoSurvivingShards: if nothing survives.
        ValueError: on an inconsistent topology or an unknown shard index.
    """
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    if total_gen_workers % dp_size != 0:
        raise ValueError(
            f"generation fleet is not evenly sharded: {total_gen_workers} workers "
            f"across dp_size={dp_size}"
        )

    unique = sorted(set(surviving_shards))
    if not unique:
        raise NoSurvivingShards(
            "no generation shard can take part in the refit; the fleet is gone"
        )
    out_of_range = [idx for idx in unique if not 0 <= idx < dp_size]
    if out_of_range:
        raise ValueError(
            f"shard indices {out_of_range} are outside the fleet (dp_size={dp_size})"
        )

    workers_per_shard = total_gen_workers // dp_size
    prefixes = {
        shard: position * workers_per_shard for position, shard in enumerate(unique)
    }
    return RefitMembership(
        world_size=train_world_size + len(unique) * workers_per_shard,
        train_world_size=train_world_size,
        shard_prefixes=prefixes,
        workers_per_shard=workers_per_shard,
    )
