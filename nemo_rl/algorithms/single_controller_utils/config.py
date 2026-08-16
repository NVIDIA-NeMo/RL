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
from typing import Any, Optional

from pydantic import BaseModel, Field

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
    # Retries for a prompt whose rollout raised, before giving up on it. Gym
    # retries dropped connections itself but surfaces HTTP 5xx and truncated
    # bodies to us, and those are usually transient. 0 disables retry; 3 gives
    # the four attempts v1's AsyncTrajectoryCollector makes
    # (1 + _MAX_NEMO_GYM_STREAM_RETRIES), so both stacks absorb the same faults.
    rollout_retries: int = 3
    # Doubles per attempt, so the delays before attempts 2, 3 and 4 are 1s, 2s
    # and 4s — v1's schedule, and the only measured evidence of how fast this
    # fault class recovers. Retrying matters more here than in v1: a prompt that
    # exhausts its attempts leaves its batch a group short, which the in_order
    # sampler cannot close.
    rollout_retry_backoff_base_seconds: float = 1.0
    # Consecutive prompts that may exhaust their retries before the run aborts.
    # Isolated rollout failures are normal and get dropped; this many in a row
    # means the environment servers or generation backend are down, not that one
    # prompt is bad, and failing fast beats burning the allocation.
    max_consecutive_rollout_failures: int = 32
    # Enable per-rollout diagnostic prints (prompt content / completion previews).
    diagnostics: bool = False
    # Fail every Nth prompt index deliberately, to exercise the retry and drop
    # paths above without needing the connection pressure that produces real
    # rollout faults. 0, the default, is inert; only a validation run sets it.
    fault_inject_every_nth_prompt: int = 0


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

    # DELIBERATE DIVERGENCE FROM PR 3582 — do not re-add without reading this.
    #
    # Upstream pairs ReadyFirstSampler with a guard here rejecting the sampler
    # unless loss_fn.use_importance_sampling_correction is true and
    # loss_fn.force_on_policy_ratio is false. The guard is well motivated:
    # ready_first admits stale data by design, so it wants a genuine importance
    # -sampling correction, and under force_on_policy_ratio the ratio is 1 by
    # construction — there is no off-policy correction at all, whatever
    # use_importance_sampling_correction says.
    #
    # It is omitted on this branch because the staleness sweep exists to compare
    # ready_first against v1 and against the windowed arms, and every one of
    # those arms runs force_on_policy_ratio=true (inherited from
    # nemotron-3-ultra/student_rlvr1.yaml). Adopting the guard would abort each
    # arm at setup; satisfying it would flip _policy_logprobs_required in
    # SingleControllerActor and put the prev_logprobs forward pass back into the
    # step — on the ready_first arms only. That moves step time, which is the
    # metric being compared, so the guard would corrupt the experiment it was
    # meant to protect.
    #
    # This is acceptable *only* because every arm is in the same state. The
    # supporting evidence is that measured divergence is flat: gen_kl_error and
    # js_divergence_error vary under 3% across staleness 1-6. Note also the PR
    # review's separate finding that the guard keys on the sampler's config
    # class rather than on the staleness actually configured, so it fires on
    # ready_first at staleness 0 and stays silent on windowed at staleness 6.
    #
    # Restore the guard (and set force_on_policy_ratio=false) before ready_first
    # is used for anything but this comparison. It must not carry into
    # production as-is.

    # Top-k retention keys off checkpointing.metric_name, but SC has no
    # validation loop yet (see _save_checkpoint), so a "val:" metric would
    # never be collected and top-k would silently degrade to a no-op.
    metric_name = master_config.checkpointing["metric_name"]
    if (
        master_config.checkpointing["enabled"]
        and metric_name is not None
        and not metric_name.startswith("train:")
    ):
        raise ValueError(
            f"checkpointing.metric_name={metric_name!r} is not usable on the "
            "SingleController path: it has no validation loop yet, so only "
            "'train:<name>' metrics are collected. Use 'train:<name>' (e.g. "
            "'train:loss') or set checkpointing.metric_name=null."
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
    generation_logprobs_field: str = "generation_logprobs"
    reference_logprobs_field: str = "reference_policy_logprobs"
