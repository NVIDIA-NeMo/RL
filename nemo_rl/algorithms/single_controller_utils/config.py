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

"""Configuration schema for the SingleController async-RL path.

Sibling to :class:`nemo_rl.algorithms.grpo.MasterConfig` — *not* a
subclass. Both top-level configs share the same set of cross-cutting
component sub-configs (policy, data, cluster, …) because the underlying
machinery is the same, but the SC entrypoint owns its own root so
SC-specific knobs aren't bolted onto the sync trainer's config and
vice versa.

Follows the v2 (BaseModel) convention from
``docs/design-docs/design-and-philosophy.md``: SC-specific sub-configs
declared here carry their defaults on the field, with ``extra="allow"``
so older configs that don't yet name every key continue to load.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from nemo_rl.algorithms.grpo import GRPOConfig, GRPOLoggerConfig
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.data import DataConfig
from nemo_rl.data_plane.interfaces import DataPlaneConfig
from nemo_rl.distributed.virtual_cluster import ClusterConfig
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.utils.checkpoint import CheckpointingConfig


# ── SC-specific component sub-configs ────────────────────────────────────


class StalenessConfig(BaseModel, extra="allow"):
    """Selection-window + on-policy enforcement knobs.

    The sampler reads these to decide which buffered prompt groups are
    eligible for the next train step. ``strict_on_policy`` forces
    ``max_weight_staleness_versions`` to 0 at SC start-up.
    """

    max_weight_staleness_versions: int = 1
    min_prompt_groups_per_batch: int = 2
    target_prompt_groups_per_step: Optional[int] = None
    generations_per_prompt: int = 4
    batch_selection_strategy: Literal[
        "strict_on_policy",
        "staleness_window",
    ] = "strict_on_policy"


class ConcurrencyConfig(BaseModel, extra="allow"):
    """Pump-level concurrency caps.

    ``max_inflight_prompts`` bounds rollouts in flight at once;
    ``max_buffered_rollouts`` sizes the backpressure semaphore so the
    rollout pump blocks once that many groups sit in DataPlane unread.
    """

    max_inflight_prompts: int = 8
    max_buffered_rollouts: int = 8


class TrainingConfig(BaseModel, extra="allow"):
    """Outer training loop limits.

    ``max_num_epochs=None`` lets the rollout pump cycle the dataloader
    until SC is cancelled.
    """

    max_train_steps: int = 10
    max_num_epochs: Optional[int] = None


class AdvantageConfig(BaseModel, extra="allow"):
    """SC's prompt-group-scoped advantage stage.

    SC owns this stage because the selected ``KVBatchMeta`` still
    contains whole prompt groups before the trainer's DP sharding. Field
    names address columns in DataPlane.
    """

    enabled: bool = False
    output_field: str = "advantages"
    prompt_ids_field: str = "prompt_ids_for_adv"
    reward_field: str = "total_reward"
    token_mask_field: str = "token_mask"
    sample_mask_field: str = "sample_mask"
    repeated_batch_fields: list[str] = Field(default_factory=list)
    policy_logprobs_field: Optional[str] = None
    reference_logprobs_field: Optional[str] = None


class WeightSyncConfig(BaseModel, extra="allow"):
    """Weight-transport backend selection.

    ``transport="stub"`` is the dry-run sentinel; ``"nccl"`` is the
    production collective path. NCCL coordinates rendezvous via
    ``nccl_addr`` + ``nccl_port`` (port=None lets the cluster pick).
    """

    transport: str = "stub"
    nccl_addr: str = "127.0.0.1"
    nccl_port: Optional[int] = None


# ── Top-level MasterConfig ───────────────────────────────────────────────


class MasterConfig(BaseModel, extra="allow"):
    """Top-level config for ``examples/run_grpo_single_controller.py``.

    Independent of :class:`nemo_rl.algorithms.grpo.MasterConfig` —
    they're peers, not parent/child. Cross-cutting components
    (``policy``, ``loss_fn``, ``data``, ``cluster``, ``checkpointing``,
    ``data_plane``) are reused via the same TypedDict schemas the sync
    trainer uses; SC-specific knobs (``staleness``, ``concurrency``,
    ``training``, ``advantage``, ``weight_sync``) hang off the root in
    per-component sub-configs so users can override individual sections
    via Hydra without touching the others.

    ``data_plane`` is required (no ``Optional``) — SC is built on the
    TransferQueue data plane.
    """

    # Cross-cutting components (same shape as grpo.MasterConfig).
    policy: PolicyConfig
    loss_fn: ClippedPGLossConfig
    env: dict[str, Any] = Field(default_factory=dict)
    data: DataConfig
    grpo: GRPOConfig
    logger: GRPOLoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig
    data_plane: DataPlaneConfig

    # SC-specific components.
    staleness: StalenessConfig = Field(default_factory=StalenessConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    advantage: AdvantageConfig = Field(default_factory=AdvantageConfig)
    weight_sync: WeightSyncConfig = Field(default_factory=WeightSyncConfig)
    partition_id: str = "rollout_data"
    diagnostics: bool = False
