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

"""Ownership of Gym file/source captures across the rollout handoff."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from nemo_gym.token_id_capture.protocols import TokenSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureSnapshotRef:
    """Token-free identity of an accepted frozen capture, not a native TQ receipt."""

    rollout_id: str
    snapshot_id: str
    version: int


class GymCaptureReader:
    """Hold one actor's source and retire only snapshots it handed to a consumer."""

    def __init__(
        self, source: TokenSource, *, owns_source: bool, retain_consumed: bool = False
    ):
        self.source = source
        self._owns_source = owns_source
        self._retain_consumed = retain_consumed
        self._pending: set[CaptureSnapshotRef] = set()

    @classmethod
    def from_config(cls, config: dict) -> GymCaptureReader | None:
        """Resolve Gym's single configured source once; installed sources are borrowed."""
        # Gym is an optional dependency and only exists in Gym actor environments.
        from nemo_gym.token_id_capture import (
            TokenCaptureStore,
            TokenIdCaptureConfig,
            token_id_capture_dirs_from_config,
        )
        from nemo_gym.token_id_capture.protocols import installed_token_source

        capture_config = TokenIdCaptureConfig.model_validate(config)
        if not capture_config.enabled:
            return None
        retain_consumed = capture_config.token_id_capture.retain_consumed
        source = installed_token_source()
        if source is not None:
            return cls(source, owns_source=False, retain_consumed=retain_consumed)
        directories = token_id_capture_dirs_from_config(config)
        if len(directories) != 1:
            raise ValueError(
                "Training token capture requires one readable local capture "
                "directory or an installed TokenSource in the Gym actor."
            )
        directory = directories[0]
        if str(directory).startswith(("/lustre", "/gpfs", "/mnt/shared")):
            logger.warning(
                "Gym token capture directory %s is shared; the writer and reader "
                "are colocated, so prefer a node-local directory.",
                directory,
            )
        return cls(
            TokenCaptureStore(directory),
            owns_source=True,
            retain_consumed=retain_consumed,
        )

    def register(
        self, rollout_id: str, built: dict | None
    ) -> CaptureSnapshotRef | None:
        """Remember successful builds without holding their token or route arrays."""
        # Keep eligibility identical to Gym's delivery contract.
        from nemo_gym.token_id_capture.delivery import capture_build_can_retire

        if self._retain_consumed or not capture_build_can_retire(built):
            return None
        assert built is not None
        snapshot = built.get("_capture_snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError(
                "Successful Gym capture build lacks a frozen snapshot identity"
            )
        ref = CaptureSnapshotRef(
            rollout_id, str(snapshot["snapshot_id"]), int(snapshot["version"])
        )
        self._pending.add(ref)
        return ref

    async def acknowledge(self, refs: Sequence[CaptureSnapshotRef]) -> int:
        """Delete only known, unchanged snapshots after the consumer accepts them."""
        retired = 0
        for ref in set(refs):
            if ref not in self._pending:
                continue
            try:
                dropped = await self.source.drop(
                    ref.rollout_id, snapshot_id=ref.snapshot_id, version=ref.version
                )
            except Exception:
                logger.exception(
                    "Could not retire accepted Gym capture %s", ref.rollout_id
                )
                continue
            if dropped:
                self._pending.remove(ref)
                retired += 1
            else:
                logger.warning(
                    "Gym capture %s changed after freeze; retaining evidence",
                    ref.rollout_id,
                )
        return retired

    async def close(self) -> None:
        """Close owned resources without deleting unacknowledged capture evidence."""
        if self._owns_source:
            await self.source.close()
            self._owns_source = False


async def acknowledge_gym_captures(
    env: Any, refs: Sequence[CaptureSnapshotRef]
) -> None:
    """Best-effort cleanup after acceptance; never turn it into a rollout retry."""
    if not refs:
        return
    try:
        await asyncio.wait_for(
            env.acknowledge_token_captures.remote(tuple(refs)), timeout=30
        )
    except Exception:
        # The payload already belongs to the consumer. Keep it even if Gym died.
        logger.exception(
            "Gym capture cleanup failed after payload acceptance; retaining capture evidence"
        )


def acknowledge_gym_captures_sync(env: Any, refs: Sequence[CaptureSnapshotRef]) -> None:
    """Synchronous counterpart for the synchronous TQ ingestion boundary."""
    if not refs:
        return
    # Ray is needed only by framework callers, not the capture ownership types.
    import ray

    try:
        ray.get(env.acknowledge_token_captures.remote(tuple(refs)), timeout=30)
    except Exception:
        logger.exception(
            "Gym capture cleanup failed after payload acceptance; retaining capture evidence"
        )
