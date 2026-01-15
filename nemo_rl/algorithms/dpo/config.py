# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""DPO configuration classes."""

from typing import NotRequired, TypedDict

from nemo_rl.data import DataConfig
from nemo_rl.distributed.virtual_cluster import ClusterConfig
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.utils.checkpoint import CheckpointingConfig
from nemo_rl.utils.logger import LoggerConfig


class DPOConfig(TypedDict):
    """Main DPO training configuration."""

    max_num_epochs: int
    max_num_steps: int
    val_period: int
    val_batches: int
    val_global_batch_size: int
    val_micro_batch_size: int
    val_at_start: bool
    seed: int
    reference_policy_kl_penalty: float
    preference_average_log_probs: bool
    sft_average_log_probs: bool
    preference_loss_weight: float
    sft_loss_weight: float


class DPOSaveState(TypedDict):
    """State saved during checkpointing."""

    epoch: int
    step: int
    total_steps: int
    consumed_samples: int
    total_valid_tokens: int


class DPOValMetrics(TypedDict):
    """Validation metrics for DPO."""

    loss: float
    sft_loss: float
    preference_loss: float
    accuracy: float
    rewards_chosen_mean: float
    rewards_rejected_mean: float
    num_valid_samples: float
    global_valid_seqs: float
    global_valid_toks: float


def default_dpo_save_state() -> DPOSaveState:
    """Create default DPO save state."""
    return {
        "epoch": 0,
        "step": 0,
        "total_steps": 0,
        "consumed_samples": 0,
        "total_valid_tokens": 0,
    }


class MasterConfig(TypedDict):
    """Complete DPO training configuration."""

    policy: PolicyConfig
    data: DataConfig
    dpo: DPOConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig
