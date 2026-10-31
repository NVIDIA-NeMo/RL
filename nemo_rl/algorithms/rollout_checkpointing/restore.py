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

"""Restore the rollout state a resumed checkpoint saved.

Runs once before the pumps start; it fills the shared runtime state and
the recovery ledger, and the rollout pump then redispatches what it loaded.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Optional,
    cast,
)

import torch

from nemo_rl.algorithms.async_utils.replay_buffer import (
    LEGACY_REPLAY_BUFFER_FILENAME,
    REPLACEMENT_RESERVE_FILENAME,
    REPLAY_BUFFER_METADATA_FILENAME,
    DataPlaneMutationCut,
    TQReplayGroupMetadata,
    validate_forest_checkpoint_metadata,
)
from nemo_rl.algorithms.rollout_checkpointing.inventory import (
    capture_reward_settings,
    validate_replay_inventory,
    validate_rollout_recovery_inventory,
)
from nemo_rl.algorithms.rollout_checkpointing.state import (
    ResumeSource,
    RolloutParts,
    _UsesRolloutParts,
)
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.checkpointing.interfaces import (
    RestoredEnvironmentRollouts,
)
from nemo_rl.experience.rollout_recovery import (
    ROLLOUT_RECOVERY_STATE_FILENAME,
    SUPPORTED_ROLLOUT_RECOVERY_SCHEMA_VERSIONS,
    ParsedRolloutRecoveryState,
    PromptGroupPhase,
    RecoveryTargetLevel,
    parse_rollout_recovery_state,
)
from nemo_rl.utils.path_utils import (
    load_checkpoint_object,
    path_exists,
)

if TYPE_CHECKING:
    from nemo_rl.algorithms.single_controller_utils.config import MasterConfig


async def read_recovery_sidecar(
    recovery_path: Path,
    metadata: Mapping[str, Any],
    *,
    master_config: MasterConfig,
) -> ParsedRolloutRecoveryState:
    """Read the recovery-ledger sidecar and check it against its TQ checkpoint.

    The native TQ metadata advertises the sidecar's checksum, ledger schema and
    group count; the sidecar must match all three, and its reward settings
    must match this run's when token capture is on.
    """
    expected_payload_sha256 = metadata["rollout_recovery_payload_sha256"]
    if not isinstance(expected_payload_sha256, str):
        raise TypeError(
            "rollout_recovery_payload_sha256 must be a string in native "
            "TQ checkpoint metadata"
        )
    expected_schema_version = metadata.get("rollout_recovery_schema_version")
    if (
        isinstance(expected_schema_version, bool)
        or not isinstance(expected_schema_version, int)
        or expected_schema_version not in SUPPORTED_ROLLOUT_RECOVERY_SCHEMA_VERSIONS
    ):
        raise ValueError(
            "native TQ checkpoint rollout recovery schema mismatch: "
            f"checkpoint={expected_schema_version!r}, "
            f"supported={sorted(SUPPORTED_ROLLOUT_RECOVERY_SCHEMA_VERSIONS)}"
        )
    expected_group_count = metadata.get("rollout_recovery_group_count")
    if (
        isinstance(expected_group_count, bool)
        or not isinstance(expected_group_count, int)
        or expected_group_count < 0
    ):
        raise TypeError(
            "rollout_recovery_group_count must be an integer in native "
            "TQ checkpoint metadata"
        )
    if not recovery_path.is_file():
        raise FileNotFoundError(
            "native TQ checkpoint advertises rollout recovery, but the "
            f"sidecar is missing at {recovery_path}"
        )

    payload = await asyncio.to_thread(recovery_path.read_bytes)
    actual_payload_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_payload_sha256 != expected_payload_sha256:
        raise ValueError(
            "rollout recovery sidecar checksum mismatch: "
            f"checkpoint={expected_payload_sha256}, "
            f"actual={actual_payload_sha256}"
        )
    state = await asyncio.to_thread(
        torch.load,
        io.BytesIO(payload),
        weights_only=True,
    )
    parsed_state = parse_rollout_recovery_state(state)
    current_reward_settings = capture_reward_settings(master_config)
    if current_reward_settings is not None:
        if parsed_state.reward_settings is None:
            raise ValueError(
                "rollout recovery sidecar was saved with token capture disabled; "
                "it cannot be resumed with token capture enabled"
            )
        parsed_state.reward_settings.require_compatible(current_reward_settings)
    if parsed_state.ledger_state["schema_version"] != expected_schema_version:
        raise ValueError(
            "rollout recovery sidecar schema does not match native TQ metadata"
        )
    if len(parsed_state.ledger_state["groups"]) != expected_group_count:
        raise ValueError(
            "rollout recovery sidecar group count does not match native "
            "TQ checkpoint metadata"
        )
    return parsed_state


class RolloutRestorer(_UsesRolloutParts):
    """Restore the rollout state a resumed checkpoint saved, then redispatch it."""

    def __init__(
        self,
        parts: RolloutParts,
        *,
        algo_cfg: Any,
        buffer_capacity: asyncio.Semaphore,
        resume: ResumeSource,
    ) -> None:
        """Store the parts restore reads and rebuilds.

        Args:
            parts: The controller's parts shared with saving.
            algo_cfg: The algorithm config; restore reads its prompt-group shape.
            buffer_capacity: Replay-capacity permits; restore takes one per
                restored group.
            resume: What setup restored from, which restore continues.
        """
        super().__init__(parts)
        self._algo_cfg = algo_cfg
        self._buffer_capacity = buffer_capacity
        self._resume = resume
        # Restore fills these: replay groups it discarded and queued for fresh
        # generation, and whether only ledger-recorded dispatch can launch them.
        self._groups_to_regenerate: list[TQReplayGroupMetadata] = []
        self._requires_recovery_dispatch = False

    async def restore(self) -> bool:
        """Restore the rollout state the resumed checkpoint saved.

        Call it once, after the weights are in sync and before the pumps start.
        Restores the replay buffer, the recovery ledger and the spare prompts
        into the shared runtime state, checks the sampler cursor against them
        and logs restore timings.

        Returns:
            Whether restore queued groups only ledger-recorded dispatch can
            launch, so the rollout pump must record prompt groups.
        """
        replay_restore_started = time.monotonic()
        restored_replay_groups = await self._restore_replay_buffer()
        replay_restore_seconds = time.monotonic() - replay_restore_started
        recovery_prepare_started = time.monotonic()
        await self._restore_recovery_ledger(
            restored_replay_groups=restored_replay_groups
        )
        await self._restore_spare_prompts()
        self._validate_sampler_cursor()
        recovery_prepare_seconds = time.monotonic() - recovery_prepare_started
        self._log_restore_metrics(
            replay_metadata_load_seconds=replay_restore_seconds,
            recovery_prepare_seconds=recovery_prepare_seconds,
            restored_replay_groups=restored_replay_groups,
        )
        return self._requires_recovery_dispatch

    def _log_restore_metrics(
        self,
        *,
        replay_metadata_load_seconds: float,
        recovery_prepare_seconds: float,
        restored_replay_groups: int,
    ) -> None:
        """Log rollout restore phases after the controller is ready to dispatch."""
        if self._resume.load_metrics is None:
            return
        load_metrics = dict(self._resume.load_metrics)
        load_metrics["replay_metadata_load_seconds"] = replay_metadata_load_seconds
        load_metrics["recovery_prepare_seconds"] = recovery_prepare_seconds
        # At this point, load_metrics contains restore-phase timers only. Sum by
        # the metric contract instead of maintaining a second hard-coded phase
        # inventory that can silently omit a newly added restore timer.
        load_metrics["total_load_seconds"] = sum(
            value
            for key, value in load_metrics.items()
            if key.endswith("_seconds") and key != "total_load_seconds"
        )
        load_metrics["groups_complete_restored"] = float(restored_replay_groups)
        self._log_metrics(
            load_metrics,
            step=self._read_progress().train_steps,
            prefix="timing/rollout_recovery",
        )

    # ── internal helpers ───────────────────────────────────────────────────

    def _validate_forest_checkpoint_metadata(self) -> None:
        """Validate checkpoint row semantics before replay or custody changes."""
        if self._resume.data_plane_metadata is not None:
            validate_forest_checkpoint_metadata(
                self._resume.data_plane_metadata,
                forest_training=self._master_config.token_capture.forest_training,
                execution_row_multiple=(
                    self._rollout_manager.execution_row_multiple
                    if self._master_config.token_capture.forest_training
                    else None
                ),
            )

    async def _restore_replay_buffer(self) -> int:
        """Restore the local replay index for the native TQ checkpoint.

        Recovery is authoritative only for samplers that explicitly support
        buffered-group restoration. The native snapshot and replay metadata file
        must both be present and agree on their manifest and group count.
        """
        if self._resume.path is None:
            return 0
        self._validate_forest_checkpoint_metadata()
        metadata_path = os.path.join(self._resume.path, REPLAY_BUFFER_METADATA_FILENAME)
        if (
            os.path.exists(metadata_path)
            and not self._sampler.supports_buffer_checkpoint
        ):
            raise RuntimeError(
                "The checkpoint contains native replay state, but the configured "
                f"sampler {self._async_cfg.sampler.name!r} does not support "
                "replay-buffer recovery"
            )
        if not self._sampler.supports_buffer_checkpoint:
            return 0
        if not os.path.exists(metadata_path):
            legacy_path = os.path.join(self._resume.path, LEGACY_REPLAY_BUFFER_FILENAME)
            if os.path.exists(legacy_path):
                raise RuntimeError(
                    "Checkpoint contains legacy replay_buffer.pt state, which "
                    "predates authoritative native TQ replay recovery. Resume it "
                    "with the older implementation or explicitly start without "
                    "restoring buffered rollouts."
                )
            print(
                f"⚠️ No native replay metadata found at {metadata_path}. "
                "Starting with an empty replay buffer.",
                flush=True,
            )
            return 0
        print(f"📦 Restoring replay buffer metadata: {metadata_path}")
        # weights_only=False: the replay metadata file contains pickled KVBatchMeta
        # objects but no rollout tensor payloads. It is a trusted same-job artifact.
        buffer_state = await asyncio.to_thread(
            torch.load, metadata_path, weights_only=False
        )
        if self._resume.data_plane_metadata is None:
            raise RuntimeError(
                "Found metadata-only replay checkpoint, but the matching "
                "native TQ checkpoint was not restored during setup"
            )
        expected_manifest_digest_value = self._resume.data_plane_metadata.get(
            "replay_manifest_digest"
        )
        if not isinstance(expected_manifest_digest_value, str):
            raise ValueError(
                "Restored TQ checkpoint metadata is missing a replay manifest digest"
            )
        expected_group_count = self._resume.data_plane_metadata.get(
            "replay_group_count"
        )
        groups = buffer_state.get("groups")
        if (
            not isinstance(expected_group_count, int)
            or not isinstance(groups, list)
            or len(groups) != expected_group_count
        ):
            raise ValueError(
                "Replay-buffer metadata group count does not match the "
                "loaded TQ checkpoint metadata"
            )
        restored = await self._buffer.load_state_dict(
            buffer_state,
            max_groups=self._async_cfg.max_buffered_rollouts,
            expected_partition_id=self._partition_id,
            expected_group_size=self._algo_cfg.num_generations_per_prompt,
            expected_manifest_digest=expected_manifest_digest_value,
        )
        await validate_replay_inventory(
            buffer_state, dp_client=self._dp_client, partition_id=self._partition_id
        )

        if self._master_config.checkpointing.get("load_replay_buffer") is False:
            # Validate and load the native snapshot before discarding it. This keeps
            # replay-free resume fail-closed: a mismatched/corrupt checkpoint must
            # not silently turn into a different training stream.
            self._groups_to_regenerate = list(groups)
            removed = await self._buffer.remove(
                list(range(restored)), remove_in_dp=True
            )
            if removed != restored:
                raise RuntimeError(
                    "replay-free resume did not discard every restored replay group: "
                    f"restored={restored}, removed={removed}"
                )
            print(
                "📦 Discarded "
                f"{restored} restored replay group(s); "
                "checkpointing.load_replay_buffer=false will regenerate their "
                "prompts on the current policy",
                flush=True,
            )
            return 0

        # Each buffered group holds one _buffer_capacity permit. Restore fails
        # above if the saved group count exceeds current capacity.
        assert restored <= self._async_cfg.max_buffered_rollouts
        for _ in range(restored):
            await self._buffer_capacity.acquire()
        return restored

    async def _restore_recovery_ledger(
        self,
        *,
        restored_replay_groups: int,
    ) -> None:
        """Restore unfinished ownership for prioritized rollout-pump redispatch."""
        if self._resume.path is None:
            return
        self._validate_forest_checkpoint_metadata()
        recovery_path = Path(
            self._resume.path,
            ROLLOUT_RECOVERY_STATE_FILENAME,
        )
        metadata = self._resume.data_plane_metadata or {}
        expected_payload_sha256 = metadata.get("rollout_recovery_payload_sha256")
        if expected_payload_sha256 is None:
            if recovery_path.is_file():
                raise RuntimeError(
                    f"{ROLLOUT_RECOVERY_STATE_FILENAME} exists, but the matching "
                    "native TQ checkpoint does not advertise rollout recovery"
                )
            if self._groups_to_regenerate:
                async with self._data_plane_checkpoint_barrier.mutation() as cut:
                    await self._queue_replay_groups_for_regeneration(cut)
            return
        parsed_state = await read_recovery_sidecar(
            recovery_path, metadata, master_config=self._master_config
        )

        recovery_ledger = self._rollout_manager.recovery_ledger
        async with self._data_plane_checkpoint_barrier.mutation(
            "recovery_restore"
        ) as cut:
            recovery_ledger.load_state_dict(cut, parsed_state.ledger_state)
            recovery_ledger.assert_checkpoint_safe()
            turn_groups = [
                group
                for group in recovery_ledger.groups()
                if group.restore_level is RecoveryTargetLevel.TURN
            ]
            restored = RestoredEnvironmentRollouts()
            # The environment stays restored until the rest of this restore
            # succeeds; on any failure it drops what it restored before it reopens.
            async with contextlib.AsyncExitStack() as environment_restore:
                if turn_groups and not self._resume.from_rollout_snapshot:
                    # A trainer checkpoint never holds environment state; with no
                    # later rollout snapshot, its unfinished turns start over below.
                    print(
                        "📦 Restoring a trainer checkpoint without environment "
                        f"state: {len(turn_groups)} unfinished turn-level group(s) "
                        "restart from their tasks",
                        flush=True,
                    )
                elif turn_groups:
                    restored = await environment_restore.enter_async_context(
                        self._environment_checkpoint.restored(Path(self._resume.path))
                    )

                self._rollout_runtime.batch_shortfall = parsed_state.batch_shortfall
                canonical_state = self._buffer.metadata_state_dict(
                    saved_capacity=self._async_cfg.max_buffered_rollouts
                )
                canonical_group_ids = {
                    group["group_id"] for group in canonical_state["groups"]
                }
                recovery_ledger.discard_canonical_groups(cut, canonical_group_ids)
                self._rollout_runtime.finalizer_metrics_by_group = {
                    group_id: metrics
                    for group_id, metrics in parsed_state.finalizer_metrics_by_group.items()
                    if group_id in canonical_group_ids
                }
                self._rollout_runtime.finalizer_rewards_by_group = {
                    group_id: rows
                    for group_id, rows in parsed_state.finalizer_rewards_by_group.items()
                    if group_id in canonical_group_ids
                }
                if self._master_config.token_capture.enabled:
                    # Validate saved custody before restart normalization can discard it.
                    await validate_rollout_recovery_inventory(
                        cut,
                        dp_client=self._dp_client,
                        recovery_ledger=self._rollout_recovery_ledger,
                        staging_partition=(
                            self._master_config.token_capture.staging_partition
                        ),
                        replay_metadata=canonical_state,
                        clear_unreferenced=False,
                        additional_expected_staging_keys=set(restored.staging_keys),
                    )
                recovery_ledger.prepare_for_restart(
                    cut,
                    continued_attempts=set(restored.continued_attempts),
                )
                if self._master_config.token_capture.enabled:
                    await validate_rollout_recovery_inventory(
                        cut,
                        dp_client=self._dp_client,
                        recovery_ledger=self._rollout_recovery_ledger,
                        staging_partition=(
                            self._master_config.token_capture.staging_partition
                        ),
                        replay_metadata=canonical_state,
                        clear_unreferenced=True,
                        additional_expected_staging_keys=set(restored.staging_keys),
                    )
                await self._rehydrate_prompts(cut)
                await self._queue_replay_groups_for_regeneration(cut)
        self._rollout_runtime.sampler_stamps_target_steps = (
            parsed_state.sampler_stamps_target_steps
            if parsed_state.sampler_stamps_target_steps is not None
            else any(
                group.target_step is not None for group in recovery_ledger.groups()
            )
            or any(
                group.get("target_step") is not None
                for group in canonical_state["groups"]
            )
        )

        groups_to_recover = recovery_ledger.groups()
        if groups_to_recover:
            print(
                f"📦 Loaded {len(groups_to_recover)} unfinished rollout "
                f"group(s) next to {restored_replay_groups} canonical group(s); "
                "the rollout pump will redispatch them before new dataloader work",
                flush=True,
            )

    async def _queue_replay_groups_for_regeneration(
        self,
        cut: DataPlaneMutationCut,
    ) -> None:
        """Convert discarded canonical replay groups into fresh prompt work."""
        groups = self._groups_to_regenerate
        if not groups:
            return

        for group in groups:
            tags = group["meta"].tags or []
            prompt_indices = {tag.get("prompt_idx") for tag in tags}
            if len(prompt_indices) != 1:
                raise ValueError(
                    "replay-free resume requires one stable prompt_idx per group: "
                    f"group={group['group_id']!r}, values={prompt_indices!r}"
                )
            prompt_index = next(iter(prompt_indices))
            if isinstance(prompt_index, bool) or not isinstance(prompt_index, int):
                raise TypeError(
                    "replay-free resume requires integer prompt_idx tags: "
                    f"group={group['group_id']!r}, value={prompt_index!r}"
                )
            prompt = await self._load_recovery_prompt(
                group_id=group["group_id"], sample_id=str(prompt_index)
            )
            restored_prompt_index = prompt.get("idx")
            if (
                isinstance(restored_prompt_index, bool)
                or not isinstance(restored_prompt_index, int)
                or restored_prompt_index != prompt_index
            ):
                raise ValueError(
                    "replay-free resume resolved a different prompt identity: "
                    f"group={group['group_id']!r}, expected={prompt_index!r}, "
                    f"actual={restored_prompt_index!r}"
                )
            self._rollout_manager.reserve_prompt_group(
                cut,
                prompt,
                target_step=group["target_step"],
                admitted=True,
            )

        print(
            "📦 Queued "
            f"{len(groups)} restored replay prompt group(s) for fresh generation",
            flush=True,
        )
        # Restored ownership must be dispatched even when future checkpoint
        # saving is disabled and the native snapshot predates recovery sidecars.
        self._requires_recovery_dispatch = True
        self._groups_to_regenerate = []

    async def _load_recovery_prompt(
        self,
        *,
        group_id: str,
        sample_id: str,
    ) -> DatumSpec:
        """Resolve and collate one stable dataset prompt reference."""
        try:
            sample_index = int(sample_id)
        except ValueError as error:
            raise ValueError(
                f"recovery group {group_id!r} has a non-integer "
                f"dataset sample_id={sample_id!r}"
            ) from error
        if sample_index < 0 or str(sample_index) != sample_id:
            raise ValueError(
                f"recovery group {group_id!r} has a non-canonical "
                f"dataset sample_id={sample_id!r}"
            )

        dataset = getattr(self._dataloader, "dataset", None)
        if dataset is None:
            raise RuntimeError(
                "cannot restore unfinished rollouts because the dataloader does "
                "not expose its source dataset"
            )
        try:
            dataset_prompt = await asyncio.to_thread(dataset.__getitem__, sample_index)
        except (IndexError, KeyError) as error:
            raise RuntimeError(
                f"cannot rehydrate recovery group {group_id!r}: "
                f"dataset sample_id={sample_id!r} is unavailable"
            ) from error
        if not isinstance(dataset_prompt, dict):
            raise TypeError(
                f"dataset sample_id={sample_id!r} resolved to "
                f"{type(dataset_prompt).__name__}, expected a DatumSpec dictionary"
            )

        collate_fn = getattr(self._dataloader, "collate_fn", None)
        if collate_fn is None:
            return cast(DatumSpec, dataset_prompt)
        prompt_batch = await asyncio.to_thread(collate_fn, [dataset_prompt])
        if isinstance(prompt_batch, BatchedDataDict):
            if prompt_batch.size != 1:
                raise ValueError(
                    "recovery collation must return exactly one prompt; "
                    f"sample_id={sample_id!r}, size={prompt_batch.size}"
                )
            return cast(
                DatumSpec,
                {key: value[0] for key, value in prompt_batch.items()},
            )
        if isinstance(prompt_batch, dict):
            return cast(DatumSpec, prompt_batch)
        raise TypeError(
            "recovery collation for "
            f"sample_id={sample_id!r} returned "
            f"{type(prompt_batch).__name__}, expected a mapping"
        )

    def _validate_sampler_cursor(self) -> None:
        """Require the sampler cursor to cover every restored target step."""
        restored_target_steps = [
            target_step
            for target_step in self._buffer.target_step_list
            if target_step is not None
        ]
        restored_target_steps.extend(
            group.target_step
            for group in self._rollout_manager.recovery_ledger.groups()
            if group.target_step is not None
        )
        if not restored_target_steps:
            return

        max_target_step = max(restored_target_steps)
        if self._sampler.dispatch_index < max_target_step:
            raise RuntimeError(
                "restored sampler cursor is older than restored rollout work: "
                f"dispatch_index={self._sampler.dispatch_index}, "
                f"max_target_step={max_target_step}. Refusing to continue because "
                "a new admission could collide with restored work and discard prompts."
            )

    async def _rehydrate_prompts(
        self,
        cut: DataPlaneMutationCut,
    ) -> None:
        """Resolve positional prompt references against a stable map-style dataset.

        This assumes the dataset exposes integer ``__getitem__`` and retains the
        same ordering across checkpoint and restart.
        """
        recovery_ledger = self._rollout_manager.recovery_ledger
        groups = recovery_ledger.groups()
        if not groups:
            return

        resolved_prompts: dict[str, DatumSpec] = {}
        for group in groups:
            sample_id = group.prompt_ref.sample_id
            prompt = resolved_prompts.get(sample_id)
            if prompt is None:
                prompt = await self._load_recovery_prompt(
                    group_id=group.group_id, sample_id=sample_id
                )
                resolved_prompts[sample_id] = prompt
            recovery_ledger.bind_runtime_prompt(
                cut,
                group.group_id,
                cast(DatumSpec, prompt),
            )

    async def redispatch_restored(
        self,
        launch: Callable[[DatumSpec, Optional[int], str], Awaitable[None]],
        admit: Callable[[list[str]], Awaitable[Any]],
    ) -> None:
        """Prioritize durable unfinished groups while the train pump drains TQ.

        Launching happens inside the ordinary rollout pump so restored groups use
        the same in-flight and replay-capacity semaphores as new work. The train
        pump runs concurrently and releases replay capacity as it consumes
        canonical or newly recovered groups; therefore recovery cannot deadlock
        merely because the checkpoint contained more unfinished ownership records
        than free replay slots.
        """
        recovery_ledger = self._rollout_manager.recovery_ledger
        groups_to_recover = recovery_ledger.groups()
        if not groups_to_recover:
            return
        recovery_started = time.monotonic()
        reused_siblings = 0
        redispatched_siblings = 0
        continued_siblings = 0

        recognized_phases = (
            PromptGroupPhase.ADMITTED,
            PromptGroupPhase.RESERVED,
        )
        unhandled_groups = [
            group for group in groups_to_recover if group.phase not in recognized_phases
        ]
        if unhandled_groups:
            details = ", ".join(
                f"{group.group_id}={group.phase!r}" for group in unhandled_groups
            )
            raise RuntimeError(f"unrecognized rollout recovery phase(s): {details}")

        # ADMITTED groups may be the only work capable of advancing the trainer and
        # opening the sampler gate. Launch them before waiting to re-admit RESERVED
        # groups, or restore can deadlock with the trainer waiting for recovered work
        # that this method has not launched yet.
        for group in groups_to_recover:
            if group.phase is PromptGroupPhase.ADMITTED:
                await launch(
                    group.prompt_payload,
                    group.target_step,
                    group.group_id,
                )
                reused = len(group.sealed_generation_indices)
                reused_siblings += reused
                redispatched_siblings += group.expected_generations - reused
                continued_siblings += len(group.continued_generation_indices)

        # A checkpoint may land after dataloader ownership is recorded but before
        # sampler admission commits. Re-admit each original dataloader batch once and
        # launch it immediately; do not wait for every reserved batch to pass its gate.
        reserved_admissions: dict[str, list[str]] = {}
        for group in groups_to_recover:
            if group.phase is PromptGroupPhase.RESERVED:
                reserved_admissions.setdefault(group.admission_id, []).append(
                    group.group_id
                )
        for group_ids in reserved_admissions.values():
            await admit(group_ids)
            for group_id in group_ids:
                group = recovery_ledger.get_group(group_id)
                await launch(
                    group.prompt_payload,
                    group.target_step,
                    group.group_id,
                )
                reused = len(group.sealed_generation_indices)
                reused_siblings += reused
                redispatched_siblings += group.expected_generations - reused
                continued_siblings += len(group.continued_generation_indices)

        self._rollout_manager.record_recovery_siblings(
            reused=reused_siblings,
            redispatched=redispatched_siblings,
        )
        self._log_metrics(
            {
                "redispatch_schedule_seconds": time.monotonic() - recovery_started,
                "groups_unfinished_found": float(len(groups_to_recover)),
                "siblings_reused": float(reused_siblings),
                "siblings_rerun": float(redispatched_siblings),
                # The re-run siblings split by where they resume: restored
                # environment state (turn level) or their prompt.
                "siblings_continued": float(continued_siblings),
                "siblings_restarted": float(redispatched_siblings - continued_siblings),
            },
            step=self._read_progress().train_steps,
            prefix="timing/rollout_recovery",
        )

        print(
            f"📦 Redispatched {len(groups_to_recover)} unfinished rollout "
            f"group(s) before new dataloader work: {reused_siblings} sibling(s) "
            f"reused, {continued_siblings} continued from environment state, "
            f"{redispatched_siblings - continued_siblings} restarted from their prompt",
            flush=True,
        )

    async def _restore_spare_prompts(self) -> None:
        """Restore spare prompts diverted before the previous run's checkpoint.

        These were pulled from the dataloader and never dispatched, so the restored
        dataloader resumes past them. Nothing else in the checkpoint holds them, and
        without this they are simply gone: one batch of the dataset per divert, plus
        the training step ``_clamp_max_num_steps`` had budgeted for it.

        No sampler-name guard, unlike the buffer restore. Spares carry no stamp -- they
        are prompts that never reached ``admit`` -- so nothing about them depends on
        which sampler wrote the checkpoint. They are restored even into a run that has
        since switched to ``on_dropped_prompt="shrink"``, where the pool is never drawn
        on but is still drained back into training at the end of the dataloader.
        """
        if self._resume.path is None:
            return
        reserve_path = os.path.join(self._resume.path, REPLACEMENT_RESERVE_FILENAME)
        # New checkpoints write even an empty reserve; legacy absence remains valid.
        if not path_exists(reserve_path):
            if (self._resume.data_plane_metadata or {}).get(
                "rollout_recovery_schema_version", 0
            ) >= 5:
                raise FileNotFoundError(f"checkpoint is missing {reserve_path}")
            return
        # weights_only=False: spares are pickled DatumSpecs, and the checkpoint is a
        # trusted same-job artifact (the replay buffer restore loads on the same terms).
        reserve_state = await asyncio.to_thread(
            load_checkpoint_object, reserve_path, weights_only=False
        )
        self._rollout_runtime.replacement_reserve.extend(reserve_state)
        print(
            f"📦 Restored {len(reserve_state)} pooled spare prompt(s) from checkpoint: "
            f"{reserve_path}",
            flush=True,
        )
