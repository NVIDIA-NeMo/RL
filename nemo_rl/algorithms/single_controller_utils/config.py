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

from pydantic import BaseModel

from nemo_rl.algorithms.grpo import GRPOConfig, GRPOLoggerConfig
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.data import DataConfig
from nemo_rl.data_plane.interfaces import DataPlaneConfig
from nemo_rl.distributed.virtual_cluster import ClusterConfig
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.utils.checkpoint import CheckpointingConfig

# ── User-facing SingleController configs ────────────────────────────────────


class AsyncRLConfig(BaseModel, extra="allow"):
    # Max weight-version gap between the trainer and a consumed rollout.
    max_weight_staleness_versions: int = 1
    # Min ready groups the streaming trainer waits for before dispatching a batch.
    min_groups_for_streaming_train: int = 32
    # Cap on in-flight generate_and_push calls in the rollout pump.
    max_inflight_prompts: int = 32
    # Cap on unconsumed rollout groups buffered in the DataPlane (backpressure);
    # over_sampling=False requires this == num_prompts_per_step * (max_weight_staleness_versions + 1).
    max_buffered_rollouts: int = 64
    # True : rollout pump keeps dispatching; samples aged past the staleness window are wasted.
    # False: pump gates each batch on trainer version — one dispatch batch per trainer step.
    over_sampling: bool = False
    # Require each sampled rollout's dispatch-time trainer_version to match the trainer exactly
    # (legacy target_weight semantics). Requires over_sampling=False.
    force_in_order: bool = True


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
