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
    # Sampler / on-policy enforcement.
    max_weight_staleness_versions: int = 1
    min_prompt_groups_per_batch: int = 2
    target_prompt_groups_per_step: Optional[int] = None
    batch_selection_strategy: Literal[
        "strict_on_policy",
        "staleness_window",
    ] = "strict_on_policy"
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
