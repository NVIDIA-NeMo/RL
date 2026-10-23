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

"""Restore the rollout state a checkpoint saved, before the pumps start.

Every resume runs this, whether or not the new run saves rollout state: a
checkpoint can hold buffered rollouts, unfinished groups and spare prompts even
when the run that resumes it has rollout checkpointing turned off. So both
``RolloutCheckpointing`` implementations hand restore to one ``RolloutRestorer``.

The restorer never writes controller fields. What the controller must take back
comes out as a ``RestoreReport``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import torch

from nemo_rl.algorithms.async_utils.replay_buffer import (
    LEGACY_REPLAY_BUFFER_FILENAME,
    REPLACEMENT_RESERVE_FILENAME,
    REPLAY_BUFFER_METADATA_FILENAME,
    DataPlaneCheckpointBarrier,
    DataPlaneCheckpointMetadata,
    DataPlaneMutationCut,
    TQReplayGroupMetadata,
)
from nemo_rl.algorithms.rollout_checkpointing.inventory import (
    capture_reward_settings,
    validate_replay_inventory,
    validate_rollout_recovery_inventory,
)
from nemo_rl.algorithms.single_controller_utils.config import algo_config
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.environment_checkpoint import (
    EnvironmentCheckpoint,
    RestoredEnvironmentRollouts,
)
from nemo_rl.experience.rollout_recovery import (
    ROLLOUT_RECOVERY_STATE_FILENAME,
    SUPPORTED_ROLLOUT_RECOVERY_SCHEMA_VERSIONS,
    PromptGroupPhase,
    RecoveryTargetLevel,
    parse_rollout_recovery_state,
)
from nemo_rl.utils.path_utils import load_checkpoint_object, path_exists

if TYPE_CHECKING:
    from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
    from nemo_rl.algorithms.async_utils.staleness_sampler import PromptGroupSampler
    from nemo_rl.algorithms.rollout_checkpointing.interface import (
        RolloutCheckpointingHost,
    )
    from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
    from nemo_rl.experience.reward_penalties import FinalizedReward
    from nemo_rl.experience.rollout_manager import RolloutManager

LaunchFn = Callable[[DatumSpec, Optional[int], str], Awaitable[None]]
"""Dispatch one restored group: its prompt, target step and group ID."""

AdmitFn = Callable[[Sequence[Optional[str]]], Awaitable[Optional[int]]]
"""Commit one sampler admission for reserved groups; return their target step."""


@dataclass(frozen=True)
class CheckpointToRestore:
    """The checkpoint setup resumed from, as far as rollout restore needs it.

    The defaults describe a fresh run, which has nothing to restore.

    Attributes:
        path: The trainer checkpoint or rollout snapshot directory, or ``None``
            for a fresh run.
        from_rollout_snapshot: Whether ``path`` is a rollout snapshot. Only a
            snapshot holds environment state.
        data_plane_metadata: The native TQ checkpoint's metadata, or ``None``
            when setup restored no TQ snapshot.
        load_metrics: Restore timers measured during setup, or ``None`` when
            restore telemetry is off.
    """

    path: Optional[str] = None
    from_rollout_snapshot: bool = False
    data_plane_metadata: Optional[DataPlaneCheckpointMetadata] = None
    load_metrics: Optional[dict[str, float]] = None


@dataclass(frozen=True)
class RestoreReport:
    """Controller state a restore rebuilt, for the controller to take back.

    A field left as ``None`` means the checkpoint had nothing for it, so the
    controller keeps its own value.

    Attributes:
        restored_replay_groups: Finished groups now back in the replay buffer.
            Each one holds a replay-capacity permit.
        replacement_reserve: Spare prompts to put back in the pool.
        batch_shortfall: Groups each stamped step will never receive.
        finalizer_metrics_by_group: Pending finalizer metrics, for the
            restored finished groups only.
        finalizer_rewards_by_group: Pending finalizer rewards, for the
            restored finished groups only.
        sampler_stamps_target_steps: Whether the sampler had stamped a step.
        requires_recovery_dispatch: Whether restore queued groups that only
            ledger-tracked dispatch can launch.
    """

    restored_replay_groups: int = 0
    replacement_reserve: list[DatumSpec] = field(default_factory=list)
    batch_shortfall: Optional[dict[int, int]] = None
    finalizer_metrics_by_group: Optional[dict[str, dict[str, float]]] = None
    finalizer_rewards_by_group: Optional[dict[str, list[FinalizedReward]]] = None
    sampler_stamps_target_steps: Optional[bool] = None
    requires_recovery_dispatch: bool = False


class RolloutRestorer:
    """Restore buffered rollouts, unfinished groups and spare prompts.

    Single use: build it with the checkpoint to restore, call ``restore()``
    once before the pumps start, then ``redispatch_restored()`` once from the
    rollout pump.
    """

    def __init__(
        self,
        *,
        host: RolloutCheckpointingHost,
        master_config: MasterConfig,
        dp_client: Any,
        partition_id: str,
        buffer: TQReplayBuffer,
        sampler: PromptGroupSampler,
        dataloader: Any,
        rollout_manager: RolloutManager,
        data_plane_barrier: DataPlaneCheckpointBarrier,
        environment_checkpoint: EnvironmentCheckpoint,
        checkpoint: CheckpointToRestore,
    ) -> None:
        """Store the parts restore reads and rebuilds.

        Args:
            host: The controller, through the calls restore may make.
            master_config: The run's config.
            dp_client: The controller's data-plane client.
            partition_id: The partition that holds finished rollouts.
            buffer: The replay buffer, which owns the replay index.
            sampler: The sampler whose restored cursor restore checks.
            dataloader: The training dataloader. Its dataset resolves saved
                prompt references.
            rollout_manager: Owns the recovery ledger.
            data_plane_barrier: The barrier every TQ mutation goes through.
            environment_checkpoint: The environment backend's rollout state.
            checkpoint: What setup resumed from.
        """
        self._host = host
        self._master_config = master_config
        self._dp_client = dp_client
        self._partition_id = partition_id
        self._buffer = buffer
        self._sampler = sampler
        self._dataloader = dataloader
        self._rollout_manager = rollout_manager
        self._data_plane_checkpoint_barrier = data_plane_barrier
        self._environment_checkpoint = environment_checkpoint
        self.checkpoint = checkpoint
        self._last_checkpoint_path = checkpoint.path
        self._restored_from_rollout_snapshot = checkpoint.from_rollout_snapshot
        self._data_plane_checkpoint_metadata = checkpoint.data_plane_metadata
        self._rollout_checkpoint_load_metrics = checkpoint.load_metrics

        # Restore fills these in as it goes; restore() returns them.
        self._restored_replay_groups_to_regenerate: list[TQReplayGroupMetadata] = []
        self._batch_shortfall: Optional[dict[int, int]] = None
        self._finalizer_metrics_by_group: Optional[dict[str, dict[str, float]]] = None
        self._finalizer_rewards_by_group: Optional[dict[str, list[FinalizedReward]]] = (
            None
        )
        self._sampler_stamps_target_steps: Optional[bool] = None
        self._requires_recovery_dispatch = False

    async def restore(self) -> RestoreReport:
        """Restore the checkpoint's rollout state and report what changed.

        Runs before the rollout and train pumps start, so nothing else writes
        TQ or the ledger while it runs.
        """
        replay_restore_started = time.monotonic()
        restored_replay_groups = await self._maybe_restore_replay_buffer()
        replay_restore_seconds = time.monotonic() - replay_restore_started
        recovery_prepare_started = time.monotonic()
        await self._maybe_restore_rollout_recovery(
            restored_replay_groups=restored_replay_groups
        )
        replacement_reserve = await self._maybe_restore_replacement_reserve()
        self._validate_restored_sampler_cursor()
        recovery_prepare_seconds = time.monotonic() - recovery_prepare_started
        self._log_rollout_restore_metrics(
            replay_metadata_load_seconds=replay_restore_seconds,
            recovery_prepare_seconds=recovery_prepare_seconds,
            restored_replay_groups=restored_replay_groups,
        )
        return self._report(
            restored_replay_groups=restored_replay_groups,
            replacement_reserve=replacement_reserve,
        )

    def _report(
        self,
        *,
        restored_replay_groups: int,
        replacement_reserve: list[DatumSpec],
    ) -> RestoreReport:
        """Collect what the restore steps rebuilt for the controller."""
        return RestoreReport(
            restored_replay_groups=restored_replay_groups,
            replacement_reserve=replacement_reserve,
            batch_shortfall=self._batch_shortfall,
            finalizer_metrics_by_group=self._finalizer_metrics_by_group,
            finalizer_rewards_by_group=self._finalizer_rewards_by_group,
            sampler_stamps_target_steps=self._sampler_stamps_target_steps,
            requires_recovery_dispatch=self._requires_recovery_dispatch,
        )

    async def redispatch_restored(self, launch: LaunchFn, admit: AdmitFn) -> None:
        """Launch the unfinished groups restore loaded, before new work.

        Args:
            launch: The rollout pump's launch, so restored groups use the same
                in-flight and replay-capacity limits as new work.
            admit: The rollout pump's sampler admission for reserved groups.
        """
        await self._redispatch_restored_rollouts(launch, admit)

    async def _maybe_restore_replay_buffer(self) -> int:
        """Restore the local replay index for the native TQ checkpoint.

        Recovery is authoritative only for samplers that explicitly support
        buffered-group restoration. The native snapshot and replay metadata file
        must both be present and agree on their manifest and group count.
        """
        if self._last_checkpoint_path is None:
            return 0
        metadata_path = os.path.join(
            self._last_checkpoint_path, REPLAY_BUFFER_METADATA_FILENAME
        )
        if (
            os.path.exists(metadata_path)
            and not self._sampler.supports_buffer_checkpoint
        ):
            raise RuntimeError(
                "The checkpoint contains native replay state, but the configured "
                f"sampler {self._master_config.async_rl.sampler.name!r} does not support "
                "replay-buffer recovery"
            )
        if not self._sampler.supports_buffer_checkpoint:
            return 0
        if not os.path.exists(metadata_path):
            legacy_path = os.path.join(
                self._last_checkpoint_path, LEGACY_REPLAY_BUFFER_FILENAME
            )
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
        if self._data_plane_checkpoint_metadata is None:
            raise RuntimeError(
                "Found metadata-only replay checkpoint, but the matching "
                "native TQ checkpoint was not restored during setup"
            )
        expected_manifest_digest_value = self._data_plane_checkpoint_metadata.get(
            "replay_manifest_digest"
        )
        if not isinstance(expected_manifest_digest_value, str):
            raise ValueError(
                "Restored TQ checkpoint metadata is missing a replay manifest digest"
            )
        expected_group_count = self._data_plane_checkpoint_metadata.get(
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
            max_groups=self._master_config.async_rl.max_buffered_rollouts,
            expected_partition_id=self._partition_id,
            expected_group_size=algo_config(
                self._master_config
            ).num_generations_per_prompt,
            expected_manifest_digest=expected_manifest_digest_value,
        )
        await validate_replay_inventory(
            buffer_state, dp_client=self._dp_client, partition_id=self._partition_id
        )

        if self._master_config.checkpointing.get("load_replay_buffer") is False:
            # Validate and load the native snapshot before discarding it. This keeps
            # replay-free resume fail-closed: a mismatched/corrupt checkpoint must
            # not silently turn into a different training stream.
            self._restored_replay_groups_to_regenerate = list(groups)
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

        # Each buffered group holds one replay-capacity permit, which the
        # controller takes for every group counted here. Restore fails above if
        # the saved group count exceeds current capacity.
        assert restored <= self._master_config.async_rl.max_buffered_rollouts
        return restored

    async def _maybe_restore_rollout_recovery(
        self,
        *,
        restored_replay_groups: int,
    ) -> None:
        """Restore unfinished ownership for prioritized rollout-pump redispatch."""
        if self._last_checkpoint_path is None:
            return
        recovery_path = Path(
            self._last_checkpoint_path,
            ROLLOUT_RECOVERY_STATE_FILENAME,
        )
        metadata = self._data_plane_checkpoint_metadata or {}
        expected_payload_sha256 = metadata.get("rollout_recovery_payload_sha256")
        if expected_payload_sha256 is None:
            if recovery_path.is_file():
                raise RuntimeError(
                    f"{ROLLOUT_RECOVERY_STATE_FILENAME} exists, but the matching "
                    "native TQ checkpoint does not advertise rollout recovery"
                )
            if self._restored_replay_groups_to_regenerate:
                async with self._data_plane_checkpoint_barrier.mutation() as cut:
                    await self._queue_restored_replay_groups_for_regeneration(cut)
            return
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
                "native TQ checkpoint has an unsupported rollout recovery schema: "
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
        current_reward_settings = capture_reward_settings(self._master_config)
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

        recovery_ledger = self._rollout_manager.recovery_ledger
        async with self._data_plane_checkpoint_barrier.mutation(
            "recovery_restore"
        ) as cut:
            recovery_ledger.load_state_dict(cut, parsed_state.ledger_state)
            turn_groups = [
                group
                for group in recovery_ledger.groups()
                if group.restore_level is RecoveryTargetLevel.TURN
            ]
            restored = RestoredEnvironmentRollouts()
            # The environment stays restored until the rest of this restore
            # succeeds; on any failure it drops what it restored before it reopens.
            async with contextlib.AsyncExitStack() as environment_restore:
                if turn_groups and not self._restored_from_rollout_snapshot:
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
                        self._environment_checkpoint.restored(
                            Path(self._last_checkpoint_path)
                        )
                    )

                recovery_ledger.prepare_for_restart(
                    cut,
                    continued_attempts=set(restored.continued_attempts),
                )
                self._batch_shortfall = parsed_state.batch_shortfall
                canonical_state = self._buffer.metadata_state_dict(
                    saved_capacity=self._master_config.async_rl.max_buffered_rollouts
                )
                canonical_group_ids = {
                    group["group_id"] for group in canonical_state["groups"]
                }
                recovery_ledger.discard_canonical_groups(cut, canonical_group_ids)
                self._finalizer_metrics_by_group = {
                    group_id: metrics
                    for group_id, metrics in parsed_state.finalizer_metrics_by_group.items()
                    if group_id in canonical_group_ids
                }
                self._finalizer_rewards_by_group = {
                    group_id: rows
                    for group_id, rows in parsed_state.finalizer_rewards_by_group.items()
                    if group_id in canonical_group_ids
                }
                if self._master_config.token_capture.enabled:
                    await validate_rollout_recovery_inventory(
                        cut,
                        dp_client=self._dp_client,
                        recovery_ledger=recovery_ledger,
                        staging_partition=(
                            self._master_config.token_capture.staging_partition
                        ),
                        replay_metadata=canonical_state,
                        clear_unreferenced=True,
                        additional_expected_staging_keys=set(restored.staging_keys),
                    )
                await self._rehydrate_rollout_recovery_prompts(cut)
                await self._queue_restored_replay_groups_for_regeneration(cut)
        self._sampler_stamps_target_steps = (
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

    async def _queue_restored_replay_groups_for_regeneration(
        self,
        cut: DataPlaneMutationCut,
    ) -> None:
        """Convert discarded canonical replay groups into fresh prompt work."""
        groups = self._restored_replay_groups_to_regenerate
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
        self._restored_replay_groups_to_regenerate = []

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

    async def _rehydrate_rollout_recovery_prompts(
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

    async def _maybe_restore_replacement_reserve(self) -> list[DatumSpec]:
        """Return spare prompts diverted before the previous run's checkpoint.

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
        if self._last_checkpoint_path is None:
            return []
        reserve_path = os.path.join(
            self._last_checkpoint_path, REPLACEMENT_RESERVE_FILENAME
        )
        # Absent for every run that never diverted a batch, which is every run that
        # does not use "replace" -- so silence here rather than the buffer restore's
        # warning, since this is the ordinary case rather than a lost artifact.
        if not path_exists(reserve_path):
            return []
        # weights_only=False: spares are pickled DatumSpecs, and the checkpoint is a
        # trusted same-job artifact (the replay buffer restore loads on the same terms).
        reserve_state = await asyncio.to_thread(
            load_checkpoint_object, reserve_path, weights_only=False
        )
        print(
            f"📦 Restored {len(reserve_state)} pooled spare prompt(s) from checkpoint: "
            f"{reserve_path}",
            flush=True,
        )
        return list(reserve_state)

    def _validate_restored_sampler_cursor(self) -> None:
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

    def _log_rollout_restore_metrics(
        self,
        *,
        replay_metadata_load_seconds: float,
        recovery_prepare_seconds: float,
        restored_replay_groups: int,
    ) -> None:
        """Log rollout restore phases after the controller is ready to dispatch."""
        if self._rollout_checkpoint_load_metrics is None:
            return
        load_metrics = dict(self._rollout_checkpoint_load_metrics)
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
        self._host.log_telemetry_metrics(
            load_metrics,
            step=self._host.train_steps,
            prefix="timing/rollout_recovery",
        )

    async def _redispatch_restored_rollouts(
        self,
        launch: LaunchFn,
        admit: AdmitFn,
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
        self._host.log_telemetry_metrics(
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
            step=self._host.train_steps,
            prefix="timing/rollout_recovery",
        )

        print(
            f"📦 Redispatched {len(groups_to_recover)} unfinished rollout "
            f"group(s) before new dataloader work: {reused_siblings} sibling(s) "
            f"reused, {continued_siblings} continued from environment state, "
            f"{redispatched_siblings - continued_siblings} restarted from their prompt",
            flush=True,
        )
