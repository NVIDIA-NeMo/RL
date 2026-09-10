# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Cross-process store for the wire guard's row fingerprints.

The guard folds a digest per row on ``put_samples`` and re-folds it on
``get_samples``. Holding the wire-in reading in the putting process only ever
verifies a same-process round trip, and the transfers worth checking are not
those: the rollout actor writes what the policy workers read, so every one of
those rows counted as ``rows_unverified``.

One actor holds the readings instead. A writer records; a reader that has no
local reading fetches. Digests are 8 bytes per row per field, so the fetch is
~100 kB against a get that already moves tens of MB.
"""

from __future__ import annotations

import ray

WIRE_GUARD_ACTOR_NAME = "nemo_rl_wire_guard"


class WireGuardState:
    """``partition -> sample_id -> field -> wire-in digest``.

    Deliberately dumb: it stores and returns readings, and the comparison
    stays in the reading client so ``HashStats`` keeps counting in the process
    that owns the transfer. Two clients recording the same row is not a
    conflict -- a row written field by field arrives that way.
    """

    def __init__(self) -> None:
        self._by_partition: dict[str, dict[str, dict[str, int]]] = {}

    def record(
        self, partition_id: str, digests_by_uid: dict[str, dict[str, int]]
    ) -> None:
        partition = self._by_partition.setdefault(partition_id, {})
        for uid, per_field in digests_by_uid.items():
            partition.setdefault(uid, {}).update(per_field)

    def fetch(self, partition_id: str, uids: list[str]) -> dict[str, dict[str, int]]:
        partition = self._by_partition.get(partition_id, {})
        return {uid: partition[uid] for uid in uids if uid in partition}

    def release(self, partition_id: str, uids: list[str] | None) -> None:
        if uids is None:
            self._by_partition.pop(partition_id, None)
            return
        partition = self._by_partition.get(partition_id)
        if partition is None:
            return
        for uid in uids:
            partition.pop(uid, None)
        if not partition:
            del self._by_partition[partition_id]

    def n_rows(self) -> int:
        """Row count, for tests and for spotting a store that never drains."""
        return sum(len(p) for p in self._by_partition.values())


# The state is a plain class so its logic is testable without a Ray cluster.
WireGuardStore = ray.remote(num_cpus=0)(WireGuardState)


def get_wire_guard() -> ray.actor.ActorHandle | None:
    """The run's store, created on first call; ``None`` without Ray.

    ``get_if_exists`` rather than a create-then-lookup dance: every client
    builds itself independently and any of them may be first.

    Returning ``None`` off-Ray keeps the guard usable in a single process --
    unit tests and the local adapter -- where the putting client is also the
    reading one and its own store already answers every lookup.
    """
    if not ray.is_initialized():
        return None
    return WireGuardStore.options(  # type: ignore[attr-defined]
        name=WIRE_GUARD_ACTOR_NAME,
        get_if_exists=True,
    ).remote()
