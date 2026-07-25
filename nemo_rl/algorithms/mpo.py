# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Mixed Preference Optimization built on the maintained preference trainer."""

from dataclasses import dataclass
from typing import Any, cast

import torch
from pydantic import BaseModel
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.dpo import (
    DPOConfig,
    DPOSaveState,
    DPOValMetrics,
    dpo_train,
    setup as setup_preference_training,
)
from nemo_rl.algorithms.loss import MPOLossConfig, MPOLossFn
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig


class MPOConfig(DPOConfig):
    bco_loss_weight: float = 1.0
    quality_average_log_probs: bool = False
    reward_shift_momentum: float = 0.99
    reward_shift: float = 0.0


class MasterConfig(BaseModel, extra="ignore"):
    policy: PolicyConfig
    data: DataConfig
    mpo: MPOConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig

    @property
    def dpo(self) -> MPOConfig:
        """Compatibility view consumed by the shared preference trainer."""
        return self.mpo


@dataclass
class MPOSaveState(DPOSaveState):
    reward_shift: float = 0.0
    reward_shift_num_updates: int = 0


@dataclass
class MPOValMetrics(DPOValMetrics):
    bco_loss: float
    bco_rewards_chosen_mean: float
    bco_rewards_rejected_mean: float


def _initial_mpo_save_state(config: MPOConfig) -> MPOSaveState:
    return MPOSaveState(
        epoch=0,
        step=0,
        total_steps=0,
        consumed_samples=0,
        total_valid_tokens=0,
        reward_shift=config.reward_shift,
        reward_shift_num_updates=0,
    )


def _configure_pair_safe_packing(policy_config: PolicyConfig) -> None:
    if policy_config["dynamic_batching"]["enabled"]:
        raise ValueError("Dynamic batching is not supported with MPO.")
    if not policy_config["sequence_packing"]["enabled"]:
        return

    packing_config = cast(dict[str, Any], policy_config["sequence_packing"])
    packing_config["fuse_loss"] = True
    configured_grouping = packing_config.get("pair_grouping_key")
    if configured_grouping not in (None, "pair_index"):
        raise ValueError(
            "MPO requires policy.sequence_packing.pair_grouping_key='pair_index'; "
            f"got {configured_grouping!r}."
        )
    packing_config["pair_grouping_key"] = "pair_index"
    # CP shards the language model but not the vision encoder. One atomic pair
    # per packed microbatch preserves the unpacked vision-memory envelope.
    packing_config.setdefault("max_sequences_per_bin", 1)


def _make_loss_fn(
    config: MPOConfig,
    policy_config: PolicyConfig,
    save_state: MPOSaveState,
) -> MPOLossFn:
    loss_config = config.model_dump()
    loss_config["reward_shift"] = save_state.reward_shift
    megatron_config = policy_config.get("megatron_cfg", {"enabled": False})
    use_fused_linear_logprobs = bool(
        megatron_config["enabled"]
        and megatron_config.get("use_fused_linear_logprobs", False)
    )
    return MPOLossFn(
        MPOLossConfig(**loss_config),
        use_fused_linear_logprobs=use_fused_linear_logprobs,
    )


def _sum_metric(values: list[Any]) -> float:
    total = 0.0
    for value in values:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        total += float(value)
    return total


def _update_reward_shift(
    train_results: dict[str, Any],
    loss_fn: MPOLossFn,
    save_state: MPOSaveState,
) -> None:
    """Synchronize the BCO shift once per completed optimizer step."""
    metrics = train_results["all_mb_metrics"]
    reward_sum = _sum_metric(metrics.get("bco_reward_sum", []))
    reward_count = _sum_metric(metrics.get("bco_reward_count", []))
    if reward_count <= 0:
        return
    save_state.reward_shift = loss_fn.update_reward_shift(
        reward_sum=reward_sum,
        reward_count=reward_count,
    )
    save_state.reward_shift_num_updates += 1
    metrics["reward_shift"] = [save_state.reward_shift]


def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: dict[str, AllTaskProcessedDataset],
) -> tuple[
    Policy,
    RayVirtualCluster,
    StatefulDataLoader,
    dict[str, StatefulDataLoader],
    MPOLossFn,
    Logger,
    CheckpointManager,
    MPOSaveState,
    MasterConfig,
]:
    """Set up MPO without reintroducing the legacy Omni collapse/expand path."""
    _configure_pair_safe_packing(master_config.policy)
    result = setup_preference_training(
        master_config,  # type: ignore[arg-type]
        tokenizer,
        train_dataset,
        val_dataset,
        loss_fn_factory=_make_loss_fn,
        save_state_cls=MPOSaveState,
        initial_save_state_fn=lambda: _initial_mpo_save_state(master_config.mpo),
        allow_sequence_packing=True,
        cluster_name="mpo_cluster",
    )
    return result  # type: ignore[return-value]


def mpo_train(
    policy: Policy,
    train_dataloader: StatefulDataLoader,
    val_dataloader: dict[str, StatefulDataLoader],
    tokenizer: AutoTokenizer,
    loss_fn: MPOLossFn,
    master_config: MasterConfig,
    logger: Logger,
    checkpointer: CheckpointManager,
    mpo_save_state: MPOSaveState,
) -> None:
    """Run MPO with driver-owned, checkpointed reward-shift updates."""
    dpo_train(
        policy,
        train_dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        master_config,
        logger,
        checkpointer,
        mpo_save_state,
        metrics_cls=MPOValMetrics,
        post_train_step=_update_reward_shift,
    )
