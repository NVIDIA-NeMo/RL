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

"""Passive Ray actor for durable rollout checkpoint operations."""

from __future__ import annotations

import os

import ray

from nemo_rl.experience.rollout_checkpoint import (
    CompletedSiblingRecord,
    PersistAck,
    RolloutCheckpointStore,
    RolloutRecoveryManifest,
    RolloutWorkItem,
)

DEFAULT_ROLLOUT_CHECKPOINT_WRITER_CONCURRENCY = 8


@ray.remote(max_concurrency=DEFAULT_ROLLOUT_CHECKPOINT_WRITER_CONCURRENCY)
class RolloutCheckpointWriter:  # pragma: no cover
    """Threaded, passive wrapper around ``RolloutCheckpointStore``.

    Ray runs synchronous actor methods in a bounded thread pool when
    ``max_concurrency`` is greater than one. The store provides group-scoped
    ordering, so operations for different groups can use those threads while
    writes, fences, reads, and deletion for one group remain serialized.

    Exactly one writer actor may own a checkpoint root. Setup is responsible
    for enforcing that deployment invariant.
    """

    def __init__(self, root_dir: str | os.PathLike[str]) -> None:
        self._store = RolloutCheckpointStore(root_dir)

    def persist_completed(self, record: CompletedSiblingRecord) -> PersistAck:
        """Durably persist one completed sibling."""
        return self._store.persist_completed(record)

    def ensure_compatible_manifest(
        self,
        expected: RolloutRecoveryManifest,
    ) -> RolloutRecoveryManifest:
        """Create or validate the directory-scoped recovery manifest."""
        return self._store.ensure_compatible_manifest(expected)

    def load_completed(
        self, work: RolloutWorkItem
    ) -> dict[int, CompletedSiblingRecord]:
        """Load durable siblings compatible with a rollout work item."""
        return self._store.load_completed(work)

    def fence(self, run_id: str, group_id: str, min_attempt_id: int) -> int:
        """Durably advance a rollout group's minimum accepted attempt."""
        return self._store.fence(run_id, group_id, min_attempt_id)

    def get_fence(self, run_id: str, group_id: str) -> int:
        """Return a rollout group's durable minimum accepted attempt."""
        return self._store.get_fence(run_id, group_id)

    def validate_attempt(self, run_id: str, group_id: str, attempt_id: int) -> None:
        """Raise when an attempt is older than the durable group fence."""
        self._store.validate_attempt(run_id, group_id, attempt_id)

    def delete_group(self, run_id: str, group_id: str) -> None:
        """Idempotently delete every durable record for a rollout group."""
        self._store.delete_group(run_id, group_id)
