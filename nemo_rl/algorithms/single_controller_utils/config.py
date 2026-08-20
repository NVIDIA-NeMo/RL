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

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, PositiveInt

from nemo_rl.algorithms.async_utils.staleness_sampler import (
    InOrderSamplerConfig,
    SamplerConfig,
    required_buffer_capacity_for_config,
)
from nemo_rl.algorithms.grpo import GRPOConfig, GRPOLoggerConfig
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.data import DataConfig
from nemo_rl.data_plane.interfaces import DataPlaneConfig
from nemo_rl.distributed.virtual_cluster import ClusterConfig
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.utils.checkpoint import CheckpointingConfig

# ── User-facing SingleController configs ────────────────────────────────────


class AsyncRLConfig(BaseModel, extra="allow"):
    # Staleness policy shared by the rollout and train pumps.
    sampler: SamplerConfig = Field(
        default_factory=InOrderSamplerConfig,
    )
    # Recompute generation KV caches after each weight update.
    recompute_kv_cache_after_weight_updates: bool = False
    # Min ready groups the streaming trainer waits for before dispatching a batch.
    min_groups_for_streaming_train: int = 32
    # Cap on in-flight generate_and_push calls in the rollout pump.
    max_inflight_prompts: int = 32
    # Cap on unconsumed rollout groups buffered in the DataPlane (backpressure).
    max_buffered_rollouts: int = 64
    # Enable per-rollout diagnostic prints (prompt content / completion previews).
    diagnostics: bool = False


class TokenCaptureConfig(BaseModel, extra="allow"):
    """Ledger-authoritative token capture (token-in/token-out via NeMo-Gym).

    Dormant by default: with ``enabled=False`` every legacy codepath behaves
    exactly as before — no staging partition is registered, no ledger is
    installed, and rollouts ride the token-echo path. See
    docs/design-docs/token-capture-ledger.md.
    """

    enabled: bool = False
    # TQ partition holding per-call staged token deltas (cleared by the
    # finalizer; distinct from the canonical rollout partition).
    staging_partition: str = "rollout_staging"
    # A failed worker-side stage poisons the rollout; "continue" serves the
    # completion and lets the finalizer emit a placeholder row, "abort" fails
    # the whole rollout in the ledger.
    on_capture_failure: Literal["continue", "abort"] = "continue"
    # "allow" trains groups whose calls span a refit (staleness accounted via
    # group_min_wv); "reject" placeholders them. Strict modes beyond the MVP
    # matrix raise NotImplementedError at setup.
    mixed_weight_version_policy: Literal["allow", "reject"] = "allow"
    # Drop the whole group when fewer than this fraction of its rollouts
    # produced valid rows (None keeps every group).
    min_valid_fraction_per_group: Optional[float] = None
    # Bearer token for Gym's token-capture control routes. None =
    # minted per run at setup; set explicitly only for multi-controller
    # setups that must share one ledger.
    control_auth_token: Optional[str] = None
    # Hard deadline per control-plane call (S5 finding: control-plane death must
    # surface as a failed dispatch, not a silent retry stall).
    control_timeout_s: float = 60.0
    # Root for Gym's per-rollout capture ledgers and base capture layer. None =
    # derived at setup
    # under the run's log dir.
    capture_dir: Optional[str] = None
    # Keep routed_experts out of canonical rows and assemble them on policy
    # workers from strict staged-fragment plans.
    defer_routed_experts_to_policy: bool = False
    # Fixed CPU finalizer pool size; actors are never automatically replaced.
    num_finalizer_workers: PositiveInt = 2


class MasterConfig(BaseModel, extra="allow"):
    policy: PolicyConfig
    loss_fn: ClippedPGLossConfig
    env: dict[str, Any]
    data: DataConfig
    grpo: GRPOConfig
    logger: GRPOLoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig
    data_plane: DataPlaneConfig
    async_rl: AsyncRLConfig
    token_capture: TokenCaptureConfig = Field(default_factory=TokenCaptureConfig)


def validate_sampler_buffer_capacity(
    async_config: AsyncRLConfig,
    *,
    required_capacity: Optional[int],
    sampler_name: str,
) -> None:
    """Validate that backpressure cannot deadlock the selected sampler."""
    if (
        required_capacity is not None
        and async_config.max_buffered_rollouts < required_capacity
    ):
        raise ValueError(
            f"max_buffered_rollouts ({async_config.max_buffered_rollouts}) is below "
            f"the {sampler_name} sampler's required capacity "
            f"({required_capacity}); the rollout pump would deadlock waiting for "
            f"buffer slots."
        )


def validate_single_controller_config(master_config: MasterConfig) -> None:
    """Validate cross-section SingleController constraints before setup."""
    async_config = master_config.async_rl
    num_prompts_per_step = master_config.grpo.num_prompts_per_step
    if num_prompts_per_step < async_config.min_groups_for_streaming_train:
        raise ValueError(
            f"grpo.num_prompts_per_step ({num_prompts_per_step}) "
            f"must be >= async_rl.min_groups_for_streaming_train "
            f"({async_config.min_groups_for_streaming_train})"
        )

    rl_step_samples = (
        num_prompts_per_step * master_config.grpo.num_generations_per_prompt
    )
    train_global_batch_size = master_config.policy["train_global_batch_size"]
    if rl_step_samples != train_global_batch_size:
        raise ValueError(
            "num_prompts_per_step * num_generations_per_prompt "
            f"({rl_step_samples}) must equal policy.train_global_batch_size "
            f"({train_global_batch_size}) so that one RL step maps to exactly one "
            "optimizer.step. Multi-mini-step inside a single RL step is not "
            "supported on the SC split path."
        )

    required_capacity = required_buffer_capacity_for_config(
        async_config.sampler,
        num_prompts_per_step,
    )
    validate_sampler_buffer_capacity(
        async_config,
        required_capacity=required_capacity,
        sampler_name=async_config.sampler.name,
    )

    token_capture_config = master_config.token_capture
    if token_capture_config.defer_routed_experts_to_policy and not (
        token_capture_config.enabled
    ):
        raise ValueError(
            "token_capture.defer_routed_experts_to_policy requires "
            "token_capture.enabled=true"
        )
    if (
        token_capture_config.enabled
        and token_capture_config.num_finalizer_workers
        > async_config.max_buffered_rollouts
    ):
        warnings.warn(
            "token_capture.num_finalizer_workers exceeds "
            "async_rl.max_buffered_rollouts; excess finalizer actors cannot be busy",
            stacklevel=2,
        )

    # A non-zero reference-policy KL penalty makes the loss read
    # ``reference_policy_logprobs``, but the SC train pump only computes them
    # when ``skip_reference_policy_logprobs_calculation`` is false (see
    # SingleControllerActor._reference_logprobs_required). Catch the
    # inconsistent pair at setup instead of a mid-training KeyError.
    reference_policy_kl_penalty = getattr(
        master_config.loss_fn, "reference_policy_kl_penalty", 0
    )
    if (
        reference_policy_kl_penalty
        and master_config.grpo.skip_reference_policy_logprobs_calculation
    ):
        raise ValueError(
            "loss_fn.reference_policy_kl_penalty="
            f"{reference_policy_kl_penalty} requires reference_policy_logprobs, "
            "but grpo.skip_reference_policy_logprobs_calculation=true skips "
            "computing them on the SingleController path. Set "
            "grpo.skip_reference_policy_logprobs_calculation=false, or set "
            "loss_fn.reference_policy_kl_penalty=0."
        )


# ── Internal SingleController configs ────────────────────────────────────


@dataclass
class AdvantageConfig:
    """Internal DataPlane field mapping for advantage calculation."""

    output_field: str = "advantages"
    prompt_ids_field: str = "prompt_ids_for_adv"
    reward_field: str = "total_reward"
    token_mask_field: str = "token_mask"
    sample_mask_field: str = "sample_mask"
    repeated_batch_fields: list[str] = field(default_factory=list)
    policy_logprobs_field: str = "prev_logprobs"
    reference_logprobs_field: str = "reference_policy_logprobs"
