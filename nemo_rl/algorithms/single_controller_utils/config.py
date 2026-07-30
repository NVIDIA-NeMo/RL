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

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

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


class RolloutFailureConfig(BaseModel, extra="allow"):
    """Retry budgets for a rollout that fails, split by failure class.

    Infrastructure failures re-dispatch the prompt onto a different generation shard;
    data failures are deterministic, so their budget is small and exhausting it is
    reported rather than absorbed. Nothing here ever discards a prompt silently.
    """

    # Attempts for infrastructure failures (timeout, dead shard, transport). Each retry
    # re-enters shard selection, so it lands elsewhere. Exhausting this means the fleet
    # is broken rather than the prompt, and the run fails.
    max_attempts_per_prompt: PositiveInt = 5
    # Attempts for deterministic, prompt-specific failures. One retry separates a
    # transient empty response from a genuinely bad prompt; a second identical failure
    # confirms the prompt is at fault.
    max_data_attempts_per_prompt: PositiveInt = 2
    # First infra-retry delay, doubled per attempt.
    backoff_base_s: PositiveFloat = 1.0
    # Ceiling on the exponential backoff, so a long outage retries at a steady rate.
    max_backoff_s: PositiveFloat = 30.0
    # After max_data_attempts_per_prompt: fail the run, or continue without the prompt.
    on_data_exhausted: Literal["fail_fast", "skip"] = "fail_fast"
    # Only with on_data_exhausted="skip": distinct prompts that may be skipped before
    # the run fails anyway.
    max_skipped_prompts: NonNegativeInt = 0
    # NeMo-Gym only. Attempts to re-dispatch just the rows that never arrived, before
    # falling back to retrying the whole prompt group. Gym's stream dies on its first
    # failing row, so one bad row takes every later row with it; recovering those
    # individually is much cheaper than redoing all num_generations_per_prompt of them.
    max_gym_row_attempts: PositiveInt = 3

    @model_validator(mode="after")
    def _check_consistent(self) -> "RolloutFailureConfig":
        if self.max_backoff_s < self.backoff_base_s:
            raise ValueError(
                f"async_rl.rollout_failure.max_backoff_s ({self.max_backoff_s}) must be "
                f">= backoff_base_s ({self.backoff_base_s})"
            )
        # Otherwise "skip" silently behaves exactly like "fail_fast".
        if self.on_data_exhausted == "skip" and self.max_skipped_prompts < 1:
            raise ValueError(
                "async_rl.rollout_failure.on_data_exhausted='skip' requires "
                "max_skipped_prompts >= 1; got "
                f"{self.max_skipped_prompts}, which would skip nothing and fail the run "
                "on the first exhausted prompt. Set max_skipped_prompts, or use "
                "on_data_exhausted='fail_fast'."
            )
        return self


class WatchdogConfig(BaseModel, extra="allow"):
    """Last-resort detection for stalls that no other layer catches."""

    # How often the watchdog task runs its checks.
    interval_s: PositiveFloat = 30.0
    # Rollouts in flight but none committed for this long counts as a stall.
    stall_timeout_s: PositiveFloat = 600.0
    # Whether a detected stall only reports, or ends the run.
    stall_action: Literal["warn", "abort"] = "warn"
    # Poll NeMo-Gym's own RunHelper for dead subprocess servers each tick.
    gym_subprocess_check: bool = True

    @model_validator(mode="after")
    def _check_consistent(self) -> "WatchdogConfig":
        if self.stall_timeout_s <= self.interval_s:
            raise ValueError(
                f"async_rl.watchdog.stall_timeout_s ({self.stall_timeout_s}) must be "
                f"> interval_s ({self.interval_s}); otherwise the watchdog reports a "
                "stall before it has had a chance to observe one."
            )
        return self


class AsyncRLConfig(BaseModel, extra="allow"):
    # Staleness policy shared by the rollout and train pumps.
    sampler: SamplerConfig = Field(
        default_factory=InOrderSamplerConfig,
    )
    # Deadline for one NeMo-Gym prompt-group rollout, covering the whole streaming
    # response. None disables.
    rollout_timeout_s: Optional[PositiveFloat] = None
    # Deadline for a single generate_async turn on the native GRPO path. None disables.
    generation_timeout_s: Optional[PositiveFloat] = None
    # Deadline for one environment step on the native GRPO path. None disables.
    env_timeout_s: Optional[PositiveFloat] = None
    # Retry budgets for failed rollouts.
    rollout_failure: RolloutFailureConfig = Field(
        default_factory=RolloutFailureConfig,
    )
    # Stall detection.
    watchdog: WatchdogConfig = Field(default_factory=WatchdogConfig)
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

    @model_validator(mode="after")
    def _check_watchdog_outlasts_rollouts(self) -> "AsyncRLConfig":
        # A rollout that is merely slow already has its own deadline; the watchdog must
        # give it a chance to fire first, or every long rollout reads as a stall.
        if (
            self.rollout_timeout_s is not None
            and self.watchdog.stall_timeout_s <= self.rollout_timeout_s
        ):
            raise ValueError(
                f"async_rl.watchdog.stall_timeout_s ({self.watchdog.stall_timeout_s}) "
                f"must be > async_rl.rollout_timeout_s ({self.rollout_timeout_s}); "
                "otherwise the watchdog reports a stall for rollouts that are merely "
                "slow and would have timed out on their own."
            )
        return self


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
    num_prompts_per_step = master_config.grpo["num_prompts_per_step"]
    if num_prompts_per_step < async_config.min_groups_for_streaming_train:
        raise ValueError(
            f"grpo.num_prompts_per_step ({num_prompts_per_step}) "
            f"must be >= async_rl.min_groups_for_streaming_train "
            f"({async_config.min_groups_for_streaming_train})"
        )

    rl_step_samples = (
        num_prompts_per_step * master_config.grpo["num_generations_per_prompt"]
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

    # A non-zero reference-policy KL penalty makes the loss read
    # ``reference_policy_logprobs``, but the SC train pump only computes them
    # when ``skip_reference_policy_logprobs_calculation`` is false (see
    # SingleControllerActor._reference_logprobs_required). Catch the
    # inconsistent pair at setup instead of a mid-training KeyError.
    reference_policy_kl_penalty = getattr(
        master_config.loss_fn, "reference_policy_kl_penalty", 0
    )
    if reference_policy_kl_penalty and master_config.grpo.get(
        "skip_reference_policy_logprobs_calculation"
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
