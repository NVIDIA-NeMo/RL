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

"""Checks that the TQ rows match what the controller says it owns.

Both saving and restoring a checkpoint run these, so they are plain functions
rather than members of ``RolloutCheckpointing``: turning rollout checkpointing
off must not turn off the checks a restore runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from nemo_rl.algorithms.async_utils.replay_buffer import (
    REPLAY_BUFFER_METADATA_FILENAME,
    DataPlaneMutationCut,
    TQReplayMetadataState,
)
from nemo_rl.data_plane.async_utils import call_data_plane
from nemo_rl.data_plane.schema import ROUTE_PLAN_TAG
from nemo_rl.experience.reward_penalties import CaptureRewardSettings
from nemo_rl.experience.rollouts import get_effort_config
from nemo_rl.experience.route_plan import decode_route_plan

if TYPE_CHECKING:
    from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
    from nemo_rl.experience.rollout_recovery import RolloutRecoveryLedger


def capture_reward_settings(
    master_config: MasterConfig,
) -> CaptureRewardSettings | None:
    """Save run-scoped settings once, alongside raw captured rewards."""
    if not master_config.token_capture.enabled:
        return None
    return CaptureRewardSettings.from_configs(
        master_config.reward_penalties,
        get_effort_config(master_config.env),
    )


async def validate_replay_inventory(
    replay_metadata: TQReplayMetadataState,
    *,
    dp_client: Any,
    partition_id: str,
) -> None:
    """Require the canonical TQ keys to match the SC replay index exactly.

    Live checkpoint callers must hold the exclusive data-plane barrier so
    commits and clears cannot race this inventory read. Restore calls are
    also safe before the rollout and train pumps start any live writers.
    """
    expected_sample_ids = {
        sample_id
        for group in replay_metadata["groups"]
        for sample_id in group["meta"].sample_ids
    }
    actual_sample_ids = set(
        await call_data_plane(
            dp_client,
            "list_sample_ids",
            offload_sync=True,
            partition_id=partition_id,
        )
    )
    missing_sample_ids = sorted(expected_sample_ids - actual_sample_ids)
    unexpected_sample_ids = sorted(actual_sample_ids - expected_sample_ids)
    if missing_sample_ids or unexpected_sample_ids:
        raise RuntimeError(
            "Native TQ checkpoint inventory does not match "
            f"{REPLAY_BUFFER_METADATA_FILENAME}: "
            f"missing={missing_sample_ids[:10]!r} "
            f"(total={len(missing_sample_ids)}), "
            f"unexpected={unexpected_sample_ids[:10]!r} "
            f"(total={len(unexpected_sample_ids)})"
        )
    print(
        f"📦 Native TQ replay inventory validated: samples={len(actual_sample_ids)}",
        flush=True,
    )


async def validate_rollout_recovery_inventory(
    cut: DataPlaneMutationCut,
    *,
    dp_client: Any,
    recovery_ledger: RolloutRecoveryLedger,
    staging_partition: str,
    replay_metadata: Optional[TQReplayMetadataState],
    clear_unreferenced: bool,
    additional_expected_staging_keys: Optional[set[str]] = None,
) -> int:
    """Validate staging ownership while the caller holds a stable cut."""
    cut.require_live()
    expected_staging_keys = recovery_ledger.expected_staging_keys()
    expected_staging_keys.update(additional_expected_staging_keys or ())
    if replay_metadata is not None:
        for group in replay_metadata["groups"]:
            for tag in group["meta"].tags or []:
                encoded_plan = tag.get(ROUTE_PLAN_TAG)
                if encoded_plan is not None:
                    expected_staging_keys.update(
                        decode_route_plan(encoded_plan).cleanup_staging_keys
                    )

    actual_staging_keys = set(
        await call_data_plane(
            dp_client,
            "list_sample_ids",
            offload_sync=True,
            partition_id=staging_partition,
        )
    )
    missing = sorted(expected_staging_keys - actual_staging_keys)
    if missing:
        raise RuntimeError(
            "rollout-recovery ownership references staging rows missing "
            f"from live TQ state: missing={missing[:10]!r} "
            f"(total={len(missing)})"
        )
    unreferenced = sorted(actual_staging_keys - expected_staging_keys)
    if clear_unreferenced and unreferenced:
        await call_data_plane(
            dp_client,
            "clear_samples",
            offload_sync=True,
            sample_ids=unreferenced,
            partition_id=staging_partition,
        )
        print(
            "rollout recovery cleared unreferenced staging rows: "
            f"count={len(unreferenced)}",
            flush=True,
        )
    print(
        "📦 Rollout-recovery staging inventory validated: "
        f"referenced={len(expected_staging_keys)}",
        flush=True,
    )
    return len(expected_staging_keys)
