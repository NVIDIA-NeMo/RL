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

from pydantic import BaseModel, Field

from nemo_rl.algorithms.grpo import GRPOConfig, GRPOLoggerConfig
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.data import DataConfig
from nemo_rl.data_plane.interfaces import DataPlaneConfig
from nemo_rl.distributed.virtual_cluster import ClusterConfig
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.utils.checkpoint import CheckpointingConfig

# ── User-facing SingleController configs ────────────────────────────────────


class AsyncRLConfig(BaseModel, extra="allow"):
    batch_selection_strategy: Literal[
        "strict_on_policy",
        "staleness_window",
    ] = "strict_on_policy"
    # Sampler / on-policy enforcement.
    max_weight_staleness_versions: int = 1
    min_prompt_groups_per_batch: int = 2
    # Pump concurrency caps.
    max_inflight_prompts: int = 8
    max_buffered_rollouts: int = 8
    # True : over-generates and wastes rollouts that age past the staleness window;
    # False: enforces per-weight-version dispatch quota.
    over_sampling: bool = True
    # Tag rollouts with their dispatch-time target step and require an exact
    # match at sample time (legacy target_weight semantics). Requires
    # over_sampling=False.
    force_in_order: bool = False


class TokenCaptureConfig(BaseModel, extra="allow"):
    """Gate-authoritative token capture (token-in/token-out via NeMo-Gym).

    Dormant by default: with ``enabled=False`` every legacy codepath behaves
    exactly as before — no staging partition is registered, no gate is
    installed, and rollouts ride the token-echo path. See
    docs/design-docs/tq-gym-gate-authoritative.md.
    """

    enabled: bool = False
    # TQ partition holding per-call staged token deltas (cleared by the
    # finalizer; distinct from the canonical rollout partition).
    staging_partition: str = "rollout_staging"
    # A failed worker-side stage poisons the rollout; "continue" serves the
    # completion and lets the finalizer emit a placeholder row, "abort" fails
    # the whole rollout at the gate.
    on_capture_failure: Literal["continue", "abort"] = "continue"
    # "allow" trains groups whose calls span a refit (staleness accounted via
    # group_min_wv); "reject" placeholders them. Strict modes beyond the MVP
    # matrix raise NotImplementedError at setup.
    mixed_weight_version_policy: Literal["allow", "reject"] = "allow"
    # Drop the whole group when fewer than this fraction of its rollouts
    # produced valid rows (None keeps every group).
    min_valid_fraction_per_group: Optional[float] = None
    # Gate-side cleanup backstops.
    registration_ttl_s: float = 3600.0
    staging_ttl_s: float = 3600.0
    # Gym LineageIndex capacity (finding M: it holds each in-flight rollout's
    # full cumulative token sequence, and eviction of a live rollout silently
    # degrades token-in to fallbacks). None = derived at setup from the
    # training config: rollouts ≈ 2 × max in-flight; tokens ≈ rollouts × max
    # sequence length. Set explicitly for agentic workloads whose per-rollout
    # call trees hold more than one context of tokens.
    lineage_max_rollouts: Optional[int] = None
    lineage_max_tokens: Optional[int] = None
    # Bearer token for the gate's /ng-control/* routes (finding S). None =
    # minted per run at setup; set explicitly only for multi-controller
    # setups that must share one gate.
    control_auth_token: Optional[str] = None
    # Hard deadline per control-plane call (S5 finding: gate death must
    # surface as a failed dispatch, not a silent retry stall).
    control_timeout_s: float = 60.0
    # Directory for the Gym base capture layer the gate rides on (#2124-c1:
    # the capture middleware only engages with a capture dir configured; the
    # dir stays essentially empty on the gate path). None = derived at setup
    # under the run's log dir.
    capture_dir: Optional[str] = None


class MasterConfig(BaseModel, extra="allow"):
    policy: PolicyConfig
    loss_fn: ClippedPGLossConfig
    env: dict[str, Any] = Field(default_factory=dict)
    data: DataConfig
    grpo: GRPOConfig
    logger: GRPOLoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig
    data_plane: DataPlaneConfig
    async_rl: AsyncRLConfig = Field(default_factory=AsyncRLConfig)
    token_capture: TokenCaptureConfig = Field(default_factory=TokenCaptureConfig)


# ── Internal SingleController configs ────────────────────────────────────


@dataclass
class AdvantageConfig:
    output_field: str = "advantages"
    prompt_ids_field: str = "prompt_ids_for_adv"
    reward_field: str = "total_reward"
    token_mask_field: str = "token_mask"
    sample_mask_field: str = "sample_mask"
    repeated_batch_fields: list[str] = field(default_factory=list)
    policy_logprobs_field: Optional[str] = "prev_logprobs"
    reference_logprobs_field: Optional[str] = "reference_policy_logprobs"


@dataclass
class WeightSyncConfig:
    transport: str = "stub"
    nccl_addr: str = "127.0.0.1"
    nccl_port: Optional[int] = None
