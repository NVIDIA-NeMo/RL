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


class NativeRolloutFTConfig(BaseModel, extra="allow"):
    """Fault-tolerance knobs read only by ``AsyncRolloutImpl`` (the native GRPO path).

    Setting these on a NeMo-Gym run does nothing; ``validate_single_controller_config``
    rejects that rather than letting it pass silently.
    """

    # Deadline for a single generate_async turn. None disables.
    generation_timeout_s: Optional[PositiveFloat] = None
    # Deadline for one environment step. None disables.
    env_timeout_s: Optional[PositiveFloat] = None


class NemoGymRolloutFTConfig(BaseModel, extra="allow"):
    """Fault-tolerance knobs read only by ``AsyncNemoGymRolloutImpl``.

    Setting these on a native run does nothing; ``validate_single_controller_config``
    rejects that rather than letting it pass silently.
    """

    # Deadline for one whole prompt-group rollout, re-dispatches included. It spans the
    # entire stream rather than each await: Gym yields rows as they finish, so a
    # per-await budget would reset every time a fast row landed and never fire for the
    # slow row actually holding the group up. None disables.
    rollout_timeout_s: Optional[PositiveFloat] = None
    # Attempts to re-dispatch just the rows that never arrived, before falling back to
    # retrying the whole prompt group. Gym's stream dies on its first failing row, so one
    # bad row takes every later row with it; recovering those individually is much
    # cheaper than redoing all num_generations_per_prompt of them.
    max_row_attempts: PositiveInt = 3


class RolloutFailureConfig(BaseModel, extra="allow"):
    """Fault tolerance for a rollout that fails.

    The budgets at the top level are consumed by ``generate_and_push``, which sits above
    the native/NeMo-Gym split, so they govern **both** paths. Everything path-specific
    lives in the ``native`` and ``nemo_gym`` sub-blocks, so the structure itself says
    which knob applies where -- these used to dangle on ``async_rl`` among unrelated
    pump and buffer settings, where a native-path operator could set the most
    generic-sounding one (``rollout_timeout_s``) and silently get no deadline at all.

    Infrastructure failures re-dispatch the prompt onto a different generation shard;
    data failures are deterministic, so their budget is small and exhausting it is
    reported rather than absorbed. Nothing here ever discards a prompt silently.
    """

    # ── shared: consumed by generate_and_push, above the impl split ──
    # Attempts for infrastructure failures (timeout, dead shard, transport). Each retry
    # re-enters shard selection, so it lands elsewhere. Exhausting this means the fleet
    # is broken rather than the prompt, and the run fails.
    #
    # Named for the failure class it bounds, not "per prompt": this budget and the data
    # budget below are INDEPENDENT counters, not a total and a sub-total. Worst case for
    # one prompt is their sum minus one -- 6 attempts under the defaults below.
    max_infra_attempts_per_prompt: PositiveInt = 5
    # Attempts for deterministic, prompt-specific failures. One retry separates a
    # transient empty response from a genuinely bad prompt; a second identical failure
    # confirms the prompt is at fault.
    max_data_attempts_per_prompt: PositiveInt = 2
    # First infra-retry delay, doubled per attempt.
    backoff_base_s: PositiveFloat = 1.0
    # Ceiling on the exponential backoff, so a long outage retries at a steady rate.
    max_backoff_s: PositiveFloat = 30.0
    # Distinct prompts that may exhaust their data budget and be dropped before the run
    # fails anyway. 0 (the default) fails on the first one, propagating the original
    # error. One knob rather than two: an enum plus a count could express "skip, but
    # never actually skip anything", which meant a validator existed purely to reject
    # that one combination. At 0 the name reads as its own documentation.
    max_skipped_prompts: NonNegativeInt = 0
    # ── path-specific ──
    native: NativeRolloutFTConfig = Field(default_factory=NativeRolloutFTConfig)
    nemo_gym: NemoGymRolloutFTConfig = Field(default_factory=NemoGymRolloutFTConfig)

    @model_validator(mode="after")
    def _check_consistent(self) -> "RolloutFailureConfig":
        if self.max_backoff_s < self.backoff_base_s:
            raise ValueError(
                f"async_rl.rollout_failure.max_backoff_s ({self.max_backoff_s}) must be "
                f">= backoff_base_s ({self.backoff_base_s})"
            )
        return self

    @model_validator(mode="after")
    def _reject_renamed_keys(self) -> "RolloutFailureConfig":
        """Fail loudly on the previous key names rather than ignoring them.

        ``extra="allow"`` means an old key parses fine and then does nothing. For
        ``on_data_exhausted: skip`` that is a behaviour change -- prompts that used to be
        skipped now fail the run -- arriving with no diagnostic at all.
        """
        renamed = {
            "max_attempts_per_prompt": "max_infra_attempts_per_prompt",
            "on_data_exhausted": "max_skipped_prompts (0 = the old 'fail_fast')",
            "max_gym_row_attempts": "nemo_gym.max_row_attempts",
        }
        stale = [
            f"  async_rl.rollout_failure.{old} -> async_rl.rollout_failure.{new}"
            for old, new in renamed.items()
            if getattr(self, old, None) is not None
        ]
        if stale:
            raise ValueError(
                "async_rl.rollout_failure keys have been renamed:\n" + "\n".join(stale)
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
    # Fault tolerance for failed rollouts: shared budgets plus per-path deadlines.
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
        #
        # Checks EVERY deadline, not just the NeMo-Gym one. This previously compared
        # against rollout_timeout_s alone, so the invariant it advertises went unchecked
        # on the native path -- where generation_timeout_s and env_timeout_s are the
        # deadlines, and where a stall_timeout_s below either produces exactly the false
        # stall reports this guard exists to prevent.
        deadlines = (
            (
                "rollout_failure.nemo_gym.rollout_timeout_s",
                self.rollout_failure.nemo_gym.rollout_timeout_s,
            ),
            (
                "rollout_failure.native.generation_timeout_s",
                self.rollout_failure.native.generation_timeout_s,
            ),
            (
                "rollout_failure.native.env_timeout_s",
                self.rollout_failure.native.env_timeout_s,
            ),
        )
        for name, deadline in deadlines:
            if deadline is not None and self.watchdog.stall_timeout_s <= deadline:
                raise ValueError(
                    f"async_rl.watchdog.stall_timeout_s "
                    f"({self.watchdog.stall_timeout_s}) must be > async_rl.{name} "
                    f"({deadline}); otherwise the watchdog reports a stall for rollouts "
                    "that are merely slow and would have timed out on their own."
                )
        return self

    @model_validator(mode="after")
    def _reject_relocated_keys(self) -> "AsyncRLConfig":
        """Fail loudly on keys that moved, instead of ignoring them.

        These models are ``extra="allow"``, so a config written against the previous
        layout keeps parsing and its fault-tolerance settings simply stop taking effect.
        A silently ignored ``rollout_timeout_s: 900`` is precisely the failure mode the
        restructure was meant to remove, so the move must not create one on its way out.
        """
        moved = {
            "rollout_timeout_s": "rollout_failure.nemo_gym.rollout_timeout_s",
            "generation_timeout_s": "rollout_failure.native.generation_timeout_s",
            "env_timeout_s": "rollout_failure.native.env_timeout_s",
        }
        stale = [
            f"  async_rl.{old} -> async_rl.{new}"
            for old, new in moved.items()
            if getattr(self, old, None) is not None
        ]
        if stale:
            raise ValueError(
                "async_rl fault-tolerance keys have moved into rollout_failure:\n"
                + "\n".join(stale)
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

    # Nesting says which knob applies to which path, but nothing stops an operator
    # filling in the block for the path this run is not taking -- and a populated
    # wrong-path block is still a silent no-op, which is the failure this whole
    # restructure exists to remove. Only a check at setup actually closes it.
    #
    # ``env`` is a required field, so a run built the production way -- through
    # MasterConfig(**cfg), which validates -- always carries it. model_construct
    # skips validation and only fills fields that have defaults, so a config
    # assembled that way can genuinely lack the attribute, and without it the
    # rollout path is unknowable. This check only reads it to decide which half of
    # the block is inert, so skip rather than fail a construction over it.
    env_config = getattr(master_config, "env", None)
    if env_config is not None:
        use_nemo_gym = bool(env_config.get("should_use_nemo_gym"))
        unused_name = "native" if use_nemo_gym else "nemo_gym"
        unused_block = getattr(async_config.rollout_failure, unused_name)
        unused_defaults = type(unused_block)()
        populated = [
            f"  async_rl.rollout_failure.{unused_name}.{field}="
            f"{getattr(unused_block, field)!r}"
            for field in type(unused_block).model_fields
            if getattr(unused_block, field) != getattr(unused_defaults, field)
        ]
        if populated:
            active = "nemo_gym" if use_nemo_gym else "native"
            raise ValueError(
                f"this run uses the {active} rollout path, so these "
                f"{unused_name}-only settings would be silently ignored:\n"
                + "\n".join(populated)
                + f"\nMove them under async_rl.rollout_failure.{active}, or remove them."
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
