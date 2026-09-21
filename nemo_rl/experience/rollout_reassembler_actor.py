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
"""CPU Ray actors for metadata-only token-capture finalization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import ray

from nemo_rl.data_plane import DataPlaneConfig, build_data_plane_client
from nemo_rl.data_plane.adapters.tq_mooncake_checkpoint import run_checkpoint_command
from nemo_rl.data_plane.schema import ROLLOUT_METRICS
from nemo_rl.experience.metric_utils import RolloutTelemetry, calculate_single_metric
from nemo_rl.experience.rollout_reassembler import FinalizedGroup, RolloutReassembler
from nemo_rl.utils.rpc_guard import assert_metadata_only
from nemo_rl.utils.venvs import make_actor_runtime_env


@dataclass(frozen=True)
class ReassemblyRequest:
    """Metadata-only input for one prompt group's finalization."""

    group_id: str
    rollout_ids: tuple[str, ...]
    canonical_sample_ids: tuple[str, ...]
    receipts: tuple[Optional[dict[str, Any]], ...]
    rewards: tuple[float, ...]
    fallback_weight_version: int
    # Stable dataset prompt index; pack_payload stamps it on every row's tag.
    prompt_idx: int
    # Per-rollout advantage-stage flag the receipt path already knows at
    # dispatch time. The finalizer publishes it with the rebuilt rows so the
    # train pump reads the same ``mask_sample`` field as the native path
    # (SingleController reads it unconditionally).
    mask_sample: tuple[bool, ...]
    # Dataset-level loss weight shared by every completion in this prompt group.
    loss_multiplier: float = 1.0
    rollout_environment: str = "unknown"
    telemetry: tuple[Optional[RolloutTelemetry], ...] = ()


@dataclass(frozen=True)
class RolloutReassemblerActorConfig:
    """Internal constructor values shared by every finalizer actor."""

    partition_id: str
    staging_partition: str
    pad_token_id: int
    router_replay_enabled: bool
    defer_routed_experts_to_policy: bool
    max_seq_len: int
    # Whether the staging partition carries media columns (VLM capture).
    capture_media: bool


@ray.remote(
    num_cpus=1,
    num_gpus=0,
    max_restarts=0,
    max_task_retries=0,
)
class RolloutReassemblerActor:  # pragma: no cover
    """Own a connect-only TQ client and lightweight finalizer in one process."""

    def __init__(
        self,
        dp_config: DataPlaneConfig,
        config: RolloutReassemblerActorConfig,
    ) -> None:
        self._max_seq_len = config.max_seq_len
        dp_client = build_data_plane_client(dp_config, bootstrap=False)
        self._finalizer = RolloutReassembler(
            dp_client,
            partition_id=config.partition_id,
            staging_partition=config.staging_partition,
            pad_token_id=config.pad_token_id,
            router_replay_enabled=config.router_replay_enabled,
            defer_routed_experts_to_policy=config.defer_routed_experts_to_policy,
            max_seq_len=config.max_seq_len,
            capture_media=config.capture_media,
        )

    def mooncake_checkpoint(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """Run an owner-local checkpoint command; return metadata, never payloads."""
        return run_checkpoint_command(body)

    def check_dependencies(self) -> None:
        """Import the finalization API before the controller starts rollouts."""
        # Deferred: only the actor's environment needs the optional Gym extra.
        from nemo_gym.token_id_capture.staging.rebuild import (  # noqa: F401
            RebuildError,
            ReceiptVerificationError,
            verify_and_linearize,
        )
        from nemo_gym.token_id_capture.staging.records import (
            RolloutReceipt,  # noqa: F401
        )

    def finalize(self, request: ReassemblyRequest) -> FinalizedGroup:
        """Finalize one request without allowing tensor payloads across Ray RPC."""
        assert_metadata_only(request)
        if request.telemetry and len(request.telemetry) != len(request.rollout_ids):
            raise ValueError("Finalizer telemetry must align with logical siblings")
        if any(
            snapshot is not None and snapshot.environment != request.rollout_environment
            for snapshot in request.telemetry
        ):
            raise ValueError("Finalizer telemetry environment mismatch")
        if not (
            len(request.rollout_ids)
            == len(request.canonical_sample_ids)
            == len(request.receipts)
            == len(request.rewards)
            == len(request.mask_sample)
        ):
            raise ValueError(
                "finalizer request rollout_ids, canonical_sample_ids, receipts, "
                "rewards, and mask_sample must be parallel"
            )
        result = self._finalizer.finalize_group(
            request.group_id,
            list(request.rollout_ids),
            list(request.receipts),
            list(request.rewards),
            mask_sample=list(request.mask_sample),
            fallback_weight_version=request.fallback_weight_version,
            prompt_idx=request.prompt_idx,
            loss_multiplier=request.loss_multiplier,
            rollout_environment=request.rollout_environment,
            canonical_sample_ids=list(request.canonical_sample_ids),
        )
        if result.meta is not None:
            if request.telemetry and all(s is not None for s in request.telemetry):
                selected_metrics = []
                lengths = result.meta.sequence_lengths
                assert lengths is not None, "Finalized canonical rows require lengths"
                for snapshot, length in zip(request.telemetry, lengths, strict=True):
                    assert snapshot is not None
                    metrics = snapshot.to_metrics()
                    # Receipts have no token payload: their producer uses a
                    # placeholder False for truncated. Match the finalizer's
                    # actual length-cap flag, never log that placeholder.
                    truncated = int(length == self._max_seq_len)
                    prefix = f"environment/{snapshot.environment}"
                    metrics.update(
                        calculate_single_metric([truncated], 1, f"{prefix}/truncated")
                    )
                    for scope in ("", f"{prefix}/"):
                        metrics[f"{scope}truncation_rate"] = float(truncated)
                        metrics[f"{scope}natural_termination_rate"] = float(
                            1 - truncated
                        )
                    selected_metrics.append(metrics)
                result.meta.extra_info[ROLLOUT_METRICS] = selected_metrics
            else:
                # Older recovery sidecars do not contain observations. Do not
                # present a partial sibling population as a complete distribution.
                logging.getLogger(__name__).warning(
                    "Group %s lacks complete rollout telemetry; omitting producer "
                    "distributions, retaining selected-row validity accounting",
                    request.group_id,
                )
        assert_metadata_only(result)
        return result


def create_rollout_reassembler_actors(
    dp_config: DataPlaneConfig,
    config: RolloutReassemblerActorConfig,
    *,
    num_workers: int,
) -> list[Any]:
    """Construct and validate the pool after TQ partitions are registered."""
    if num_workers <= 0:
        raise ValueError(f"num_reassembler_workers must be positive, got {num_workers}")
    runtime_env = make_actor_runtime_env(
        "nemo_rl.experience.rollout_reassembler_actor.RolloutReassemblerActor"
    )
    actors = [
        RolloutReassemblerActor.options(runtime_env=runtime_env).remote(
            dp_config, config
        )
        for _ in range(num_workers)
    ]
    try:
        ray.get([actor.check_dependencies.remote() for actor in actors])
    except ray.exceptions.RayError:
        # Cleanup errors must not hide the startup failure or skip other actors.
        for actor in actors:
            try:
                ray.kill(actor)
            except Exception as error:
                print(f"finalizer actor termination failed: {error}", flush=True)
        raise
    return actors
