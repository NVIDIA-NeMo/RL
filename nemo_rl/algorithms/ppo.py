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
import gc
import os
import time
import traceback
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

import numpy as np
import ray
import torch
from pydantic import BaseModel, Field, model_validator
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.advantage_estimator import (
    GeneralizedAdvantageEstimator,
    RawRewardAdvantageEstimator,
)
from nemo_rl.algorithms.grpo import (
    RewardPenaltyConfig,
    RewardScalingConfig,
    _raise_if_reward_penalties_enabled_without_nemo_gym,
    _should_use_async_rollouts,
    _should_use_nemo_gym,
    aggregate_rollout_metrics,
    compute_and_apply_seq_logprob_error_masking,
    extract_initial_prompt_messages,
    refit_policy_generation,
    scale_rewards,
)
from nemo_rl.algorithms.loss import (
    ClippedPGLossConfig,
    ClippedPGLossDataDict,
    ClippedPGLossFn,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.algorithms.loss.loss_functions import MseValueLossConfig, MseValueLossFn
from nemo_rl.algorithms.reward_functions import (
    RewardShapingConfig,
    apply_reward_shaping,
)
from nemo_rl.algorithms.utils import (
    print_efficiency_summary,
    print_performance_metrics,
    set_seed,
)
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.data.utils import extract_necessary_env_names, load_dataloader_state
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
    get_ray_cluster_topology,
    prepare_segment_topology,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.nemo_gym import spinup_nemo_gym_actor
from nemo_rl.experience.rollouts import (
    get_nemo_gym_thinking_tags,
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
    run_nemo_gym_rollout_sync,
)
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.generation.sglang.config import SGLangConfig
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.generation.vllm.config import (
    VLLM_SPARSE_REFIT_TRANSPORTS,
    normalize_vllm_refit_config,
)
from nemo_rl.models.policy import MegatronConfig, PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.value import Value, ValueConfig
from nemo_rl.models.value.interfaces import ValueInterface
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
    print_message_log_samples,
    should_log_nemo_gym_full_result_tables,
)
from nemo_rl.utils.memory_tracker import MemoryTracker
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer
from nemo_rl.utils.venvs import make_actor_runtime_env

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


class AsyncPPOConfig(BaseModel, extra="allow"):
    """Configuration for asynchronous PPO training."""

    # Enables the replay-buffer training loop.
    enabled: bool = False
    # Maximum generation-version age accepted for training.
    max_trajectory_age_steps: int = Field(default=1, ge=1)
    # Maximum generation-version age for rollouts banked by a frozen policy.
    # None keeps the normal trajectory-age limit throughout warmup.
    warmup_max_trajectory_age_steps: int | None = Field(default=None, ge=1)
    # Allows weight updates while rollout requests are still in flight.
    in_flight_weight_updates: bool = True
    # Invalidates and rebuilds the generation KV cache after an in-flight update.
    recompute_kv_cache_after_weight_updates: bool = False
    # Regenerates a partial replay-buffer frontier after resume.
    drop_incomplete_targets_on_restore: bool = True

    @model_validator(mode="after")
    def validate_settings(self) -> "AsyncPPOConfig":
        if (
            self.recompute_kv_cache_after_weight_updates
            and not self.in_flight_weight_updates
        ):
            raise ValueError(
                "recompute_kv_cache_after_weight_updates requires "
                "in_flight_weight_updates=true"
            )
        if (
            self.warmup_max_trajectory_age_steps is not None
            and self.warmup_max_trajectory_age_steps < self.max_trajectory_age_steps
        ):
            raise ValueError(
                "warmup_max_trajectory_age_steps must be greater than or equal "
                "to max_trajectory_age_steps"
            )
        return self

    @property
    def effective_warmup_max_trajectory_age_steps(self) -> int:
        """Return the configured warmup age or the normal training age."""
        if self.warmup_max_trajectory_age_steps is None:
            return self.max_trajectory_age_steps
        return self.warmup_max_trajectory_age_steps


class AdvEstimatorConfig(TypedDict):
    """Configuration for PPO advantage estimator (GAE or raw_reward)."""

    name: str  # "gae" or "raw_reward"
    # GAE-specific (only used when name="gae")
    gae_lambda: NotRequired[float]
    gae_gamma: NotRequired[float]
    normalize_advantages: NotRequired[bool]
    # VAPO decoupled GAE (None = standard GAE, no decoupling)
    gae_lambda_value: NotRequired[Optional[float]]
    gae_lambda_policy: NotRequired[Optional[float]]
    # Length-adaptive λ_policy = 1 - 1/(α·l). 0 = disabled.
    length_adaptive_alpha: NotRequired[float]
    # CompactionRL correction across independently packed trajectory segments.
    cross_trajectory: NotRequired[bool]


class PPOConfig(TypedDict):
    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_num_epochs: int
    max_num_steps: int
    max_rollout_turns: int
    val_period: int
    val_batch_size: int | None
    val_at_start: bool
    # Whether to run validation on the last training step. Setting this to True ensures the
    # final checkpoint has validation metrics, which is required for get_best_checkpoint_path().
    val_at_end: bool
    max_val_samples: int | None
    skip_reference_policy_logprobs_calculation: NotRequired[bool]
    seed: int
    overlong_filtering: bool
    # whether to enable dynamic sampling, i.e.
    # whether to discard prompts whose rewards have zero standard deviation
    use_dynamic_sampling: bool
    # When using dynamic sampling, the maximum number of batches to generate
    # before throwing an error
    dynamic_sampling_max_gen_batches: NotRequired[int]
    # When using dynamic sampling, generation prompt batch size will equal
    # num_prompts_per_step * batch_multiplier
    batch_multiplier: NotRequired[float]
    ppo_epochs: int
    # Number of critic optimizer updates per rollout batch. null preserves the
    # legacy behavior by following ppo_epochs.
    critic_train_epochs: NotRequired[int | None]
    reward_shaping: RewardShapingConfig
    reward_scaling: RewardScalingConfig
    # By default advantages are calculated on CPU. Setting this flag to true leverages GPU for their computation.
    calculate_advantages_on_gpu: NotRequired[bool]
    # Advantage estimator configuration (gae or raw_reward)
    adv_estimator: AdvEstimatorConfig
    # Number of PPO steps of critic-only warmup before policy training begins.
    # Value model trains from step 0; policy training is skipped for
    # total_steps < this value. Default 0 (train from start).
    policy_training_start_step: NotRequired[int]
    # Optional weight-only critic warm start for a fresh PPO run. Native resume
    # checkpoints always take precedence. Optimizer and run state remain fresh.
    initial_value_weights_path: NotRequired[str | None]
    initial_policy_weights_path: NotRequired[str | None]
    # Nullable sequence-level multiplicative probability-error threshold.
    # None logs metrics without masking; values above the threshold are excluded.
    seq_logprob_error_threshold: float | None
    # Asynchronous PPO uses a replay buffer with non-colocated generation.
    async_ppo: NotRequired[AsyncPPOConfig]


class PPOSaveState(TypedDict):
    consumed_samples: int
    current_step: int
    current_epoch: int
    total_steps: int
    total_valid_tokens: int  # Track total number of non-padding tokens during training
    val_reward: NotRequired[
        float
    ]  # Optional field - may not be present during training


def _resolve_critic_train_epochs(ppo_config: PPOConfig) -> int:
    """Resolve the explicit critic update count or the legacy coupled value."""
    configured_epochs = ppo_config.get("critic_train_epochs")
    return (
        ppo_config["ppo_epochs"]
        if configured_epochs is None
        else configured_epochs
    )


def _default_ppo_save_state() -> PPOSaveState:
    return {
        "consumed_samples": 0,
        "current_step": 0,
        "current_epoch": 0,
        "total_steps": 0,
        "total_valid_tokens": 0,
        "val_reward": -99999999.0,
    }


def _apply_ppo_seq_logprob_error_masking(
    train_data: BatchedDataDict,
    rewards: torch.Tensor,
    seq_logprob_error_threshold: float | None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Apply optional mismatch masking and return the GAE mask and metrics."""
    metrics = compute_and_apply_seq_logprob_error_masking(
        train_data=train_data,
        rewards=rewards,
        seq_logprob_error_threshold=seq_logprob_error_threshold,
    )
    metrics["num_masked_seqs_by_logprob_error"] = metrics.pop("num_masked_seqs")
    advantage_mask = train_data["token_mask"] * train_data["sample_mask"].unsqueeze(-1)
    if not advantage_mask.bool().any():
        raise RuntimeError(
            "PPO has no valid response tokens after filtering. Check overlong "
            "filtering and ppo.seq_logprob_error_threshold to avoid an optimizer "
            "step with an empty batch."
        )
    return advantage_mask, metrics


def _apply_ppo_mask_sample_filter(repeated_batch: BatchedDataDict) -> int:
    """Honor the per-sample loss mask emitted by Gym environments.

    Gym exports censored/invalid trajectories through ``mask_sample``.  Apply
    it before PPO constructs ``sample_mask`` so the same samples are excluded
    from GAE, actor loss, critic targets, and critic loss.  Explicit generation
    truncation remains controlled independently by ``overlong_filtering``.
    """
    if "mask_sample" not in repeated_batch:
        return 0

    loss_multiplier = repeated_batch["loss_multiplier"].clone()
    mask_sample = torch.as_tensor(
        repeated_batch["mask_sample"],
        dtype=torch.bool,
        device=loss_multiplier.device,
    )
    if mask_sample.shape != loss_multiplier.shape:
        raise ValueError(
            "mask_sample and loss_multiplier must have the same shape: "
            f"got {tuple(mask_sample.shape)} and "
            f"{tuple(loss_multiplier.shape)}"
        )

    loss_multiplier[mask_sample] = 0
    repeated_batch["loss_multiplier"] = loss_multiplier
    return int(mask_sample.sum().item())


def _resolve_initial_policy_weights(
    ppo_config: PPOConfig,
    last_checkpoint_path: Optional[os.PathLike],
    policy_weights_path: Optional[Path],
    policy_optimizer_path: Optional[Path],
) -> tuple[Optional[Path], Optional[Path]]:
    """Resolve an optional weight-only actor warm start for a fresh run."""
    configured_path = ppo_config.get("initial_policy_weights_path")
    if last_checkpoint_path is not None or configured_path is None:
        return policy_weights_path, policy_optimizer_path

    initial_policy_weights_path = Path(configured_path)
    if not initial_policy_weights_path.is_dir():
        raise FileNotFoundError(
            "ppo.initial_policy_weights_path is not a checkpoint directory: "
            f"{initial_policy_weights_path}"
        )
    print(
        "  ✓ Warm-starting actor weights only from "
        f"{initial_policy_weights_path} (fresh optimizer and training state)",
        flush=True,
    )
    return initial_policy_weights_path, None


def _resolve_initial_value_weights(
    ppo_config: PPOConfig,
    last_checkpoint_path: Optional[os.PathLike],
    value_weights_path: Optional[Path],
    value_optimizer_path: Optional[Path],
) -> tuple[Optional[Path], Optional[Path]]:
    """Resolve an optional weight-only critic warm start for a fresh run."""
    configured_path = ppo_config.get("initial_value_weights_path")
    if last_checkpoint_path is not None or configured_path is None:
        return value_weights_path, value_optimizer_path

    initial_value_weights_path = Path(configured_path)
    if not initial_value_weights_path.is_dir():
        raise FileNotFoundError(
            "ppo.initial_value_weights_path is not a checkpoint directory: "
            f"{initial_value_weights_path}"
        )
    print(
        "  ✓ Warm-starting critic weights only from "
        f"{initial_value_weights_path} (fresh optimizer and training state)",
        flush=True,
    )
    return initial_value_weights_path, None


class PPOLoggerConfig(LoggerConfig):
    num_val_samples_to_print: int  # number of val samples to print to stdout


class MasterConfig(BaseModel, extra="allow"):
    policy: PolicyConfig
    value: ValueConfig
    loss_fn: ClippedPGLossConfig
    value_loss_fn: MseValueLossConfig
    env: dict[str, Any]
    data: DataConfig
    ppo: PPOConfig
    logger: PPOLoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig
    reward_penalties: RewardPenaltyConfig = Field(default_factory=RewardPenaltyConfig)


# ===============================================================================
# Setup & Initialization
# ===============================================================================


def setup(
    master_config: MasterConfig,
    tokenizer: TokenizerType,
    dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
    processor: Optional[AutoProcessor] = None,
) -> tuple[
    ColocatablePolicyInterface,
    Optional[GenerationInterface],
    Any,
    ValueInterface,
    tuple[RayVirtualCluster, RayVirtualCluster],
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    ClippedPGLossFn,
    MseValueLossFn,
    Logger,
    CheckpointManager,
    PPOSaveState,
    MasterConfig,
]:
    """Main entry point for running PPO algorithm.

    Returns:
        tuple of (policy, policy_generation, nemo_gym_actor, value_model, clusters,
        dataloader, val_dataloader, loss_fn, value_loss_fn, logger,
        checkpointer, ppo_save_state, master_config). ``nemo_gym_actor`` is None
        unless NeMo-Gym is enabled.
    """
    # Start timing the entire setup process
    setup_start_time = time.perf_counter()

    # Extract individual configs for easier access
    policy_config = master_config.policy
    value_config = master_config.value
    generation_config = master_config.policy["generation"]
    env_configs = master_config.env
    loss_config: ClippedPGLossConfig = master_config.loss_fn
    ppo_config = master_config.ppo
    data_config = master_config.data
    logger_config = master_config.logger
    cluster_config = master_config.cluster

    assert generation_config is not None, (
        "A generation config in the PolicyConfig is required for PPO"
    )
    if generation_config["backend"] == "vllm":
        vllm_config = cast(VllmConfig, generation_config)
        normalize_vllm_refit_config(vllm_config)
        refit_transport = vllm_config.get("refit_transport")
        if refit_transport in VLLM_SPARSE_REFIT_TRANSPORTS:
            raise ValueError(
                "Remote sparse refit is currently supported only by GRPO; PPO "
                "support is tracked in "
                "https://github.com/NVIDIA-NeMo/RL/issues/3275."
            )
        if refit_transport is not None:
            raise ValueError(
                "Checkpoint-engine refit requires non-colocated generation, but "
                "PPO currently requires colocated generation. Non-colocated PPO "
                "support is tracked in "
                "https://github.com/NVIDIA-NeMo/RL/issues/3275."
            )

    if "megatron_cfg" in policy_config and policy_config["megatron_cfg"]["enabled"]:
        policy_megatron_config = cast(MegatronConfig, policy_config["megatron_cfg"])

        # Policy optimizer state first appears after critic warmup, so a cached
        # checkpoint layout cannot represent both the warmup and training states.
        assert not (
            ppo_config["policy_training_start_step"] > 0
            and master_config.checkpointing["enabled"]
            and master_config.checkpointing["save_optimizer"]
            and "checkpoint" in policy_megatron_config
            and policy_megatron_config["checkpoint"].get(
                "ckpt_assume_constant_structure"
            )
        ), (
            "policy.megatron_cfg.checkpoint.ckpt_assume_constant_structure=true "
            "is incompatible with PPO critic warmup when optimizer checkpointing "
            "is enabled. Set ckpt_assume_constant_structure=false, "
            "ppo.policy_training_start_step=0, or checkpointing.save_optimizer=false."
        )

    if value_config["megatron_cfg"]["enabled"]:
        # Context parallelism for the Megatron value model requires sequence packing,
        # matching Megatron-Core (CP shards are produced/reassembled per packed sequence).
        if value_config["megatron_cfg"]["context_parallel_size"] > 1:
            assert value_config["sequence_packing"]["enabled"], (
                "Context parallelism (CP>1) for the Megatron PPO value model requires "
                "value.sequence_packing.enabled=true."
            )
    else:
        # DTensor PPO value model currently doesn't support sequence packing and CP.
        assert value_config["dtensor_cfg"]["enabled"], (
            "Exactly one of value.megatron_cfg.enabled or value.dtensor_cfg.enabled "
            "must be true for the PPO value model."
        )
        assert value_config["sequence_packing"]["enabled"] is False, (
            "Sequence packing is currently not supported for the DTensor PPO value model. "
            "See https://github.com/NVIDIA-NeMo/RL/issues/2951."
        )
        assert value_config["dtensor_cfg"]["context_parallel_size"] == 1, (
            "Context parallelism (CP>1) is currently not supported for the DTensor PPO value model. "
            "See https://github.com/NVIDIA-NeMo/RL/issues/2951."
        )
        assert value_config["dynamic_batching"]["enabled"] is False, (
            "Dynamic batching currently has some issue for the DTensor PPO value model. "
            "See https://github.com/NVIDIA-NeMo/RL/issues/2953."
        )

    # Set seed for all random number generators
    set_seed(ppo_config["seed"])

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config.model_dump())

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config.checkpointing)
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    ppo_save_state: Optional[PPOSaveState] = cast(
        Optional[PPOSaveState], checkpointer.load_training_info(last_checkpoint_path)
    )
    if ppo_save_state is None:
        ppo_save_state = _default_ppo_save_state()

    # ==========================
    #           Data
    # ==========================
    # Validate batch_multiplier
    batch_multiplier = ppo_config["batch_multiplier"]
    dataloader_batch_size = ppo_config["num_prompts_per_step"]
    if not ppo_config["use_dynamic_sampling"]:
        assert batch_multiplier == 1, (
            "batch_multiplier>1 can only be used if use_dynamic_sampling=True"
        )
    else:
        dataloader_batch_size = int(dataloader_batch_size * batch_multiplier)

    dataloader = StatefulDataLoader(
        dataset,
        batch_size=dataloader_batch_size,
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=data_config["num_workers"],
    )
    if last_checkpoint_path is not None:
        load_dataloader_state(dataloader, last_checkpoint_path, data_config)

    print(f"  ✓ Training dataloader loaded with {len(dataset)} samples", flush=True)

    # Load validation dataset if provided
    val_dataloader: Optional[StatefulDataLoader] = None
    # If validation is enabled, load the validation dataloader
    if (
        ppo_config["val_period"] > 0
        or ppo_config["val_at_start"]
        or ppo_config["val_at_end"]
    ):
        assert val_dataset is not None, (
            "Validation dataset is required if validation is enabled"
        )
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=ppo_config["val_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
            num_workers=data_config["num_workers"],
        )
        print(
            f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples",
            flush=True,
        )

    # ==========================
    #        Loss Function
    # ==========================
    loss_fn = ClippedPGLossFn(loss_config)
    value_loss_fn = MseValueLossFn(master_config.value_loss_fn)

    if ppo_config["adv_estimator"].get("cross_trajectory"):
        assert ppo_config["adv_estimator"]["name"] == "gae", (
            "cross_trajectory requires the GAE advantage estimator"
        )
        assert ppo_config["async_ppo"].enabled, (
            "cross_trajectory currently supports async PPO only"
        )
        assert ppo_config["num_generations_per_prompt"] == 1, (
            "cross_trajectory currently requires num_generations_per_prompt=1"
        )
        assert master_config.env["should_use_nemo_gym"], (
            "cross_trajectory currently supports NeMo-Gym rollouts only"
        )
        assert loss_config.token_level_loss, (
            "CompactionRL requires token_level_loss=true"
        )

    # Validate force_on_policy_ratio
    if loss_config.force_on_policy_ratio:
        assert (
            ppo_config["num_prompts_per_step"]
            * ppo_config["num_generations_per_prompt"]
            == policy_config["train_global_batch_size"]
        ), (
            "force_on_policy_ratio requires train_global_batch_size == num_prompts_per_step * num_generations_per_prompt"
        )
        os.environ["NRL_IGNORE_TP_ACCURACY_CHECK"] = "1"
        print("  ✓ force_on_policy_ratio enabled")

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...", flush=True)
    colocated_inference = generation_config["colocated"]["enabled"]
    backend = generation_config["backend"]
    if not colocated_inference:
        assert backend == "vllm", (
            "Non-colocated PPO generation currently supports only vLLM; "
            f"got backend={backend!r}. SGLang does not yet implement the "
            "cross-cluster collective weight update path."
        )

    reward_model_enabled = "reward_model" in extract_necessary_env_names(data_config)
    segment_size = cluster_config.get("segment_size")

    total_nodes = cluster_config["num_nodes"]
    if reward_model_enabled:
        rm_resource = env_configs["reward_model"]["resources"]
        rm_nodes = rm_resource["num_nodes"]
        rm_gpus_per_node = rm_resource["gpus_per_node"]
    else:
        rm_nodes = 0
        rm_gpus_per_node = 0

    if total_nodes == 1:
        policy_nodes = total_nodes
    else:
        policy_nodes = total_nodes - rm_nodes
        assert policy_nodes > 0, (
            "policy_nodes must be > 0, but got "
            f"policy_nodes:{policy_nodes} + rm_nodes:{rm_nodes} = total_nodes:{total_nodes}"
        )

    if colocated_inference:
        if total_nodes == 1:
            policy_gpus_per_node = cluster_config["gpus_per_node"] - rm_gpus_per_node
            assert policy_gpus_per_node > 0, (
                "policy.generation.colocated.resources.gpus_per_node must be > 0 "
                "when cluster.num_nodes = 1, "
                f"but got {policy_gpus_per_node}."
            )
        else:
            policy_gpus_per_node = cluster_config["gpus_per_node"]

        cluster = RayVirtualCluster(
            name="ppo_policy_cluster",
            bundle_ct_per_node_list=[policy_gpus_per_node] * policy_nodes,
            use_gpus=True,
            num_gpus_per_node=policy_gpus_per_node,
            max_colocated_worker_groups=1 if backend == "megatron" else 3,
        )
        train_cluster = cluster
        inference_cluster = cluster
        print(
            f"  ✓ Ray cluster for policy initialized with {policy_nodes} nodes",
            flush=True,
        )
    else:
        train_gpus_per_node = cluster_config["gpus_per_node"]
        train_nodes = policy_nodes

        inference_resources = generation_config["colocated"]["resources"]
        inference_gpus_per_node = inference_resources["gpus_per_node"]
        inference_nodes = inference_resources["num_nodes"]
        shared_node_inference = policy_nodes == 1

        if shared_node_inference:
            assert (
                inference_gpus_per_node is not None and inference_gpus_per_node > 0
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be explicitly set to a value > 0 "
                "when policy_nodes = 1 and inference is non-colocated, "
                f"but got {inference_gpus_per_node}."
            )
            assert inference_nodes is None or inference_nodes == 1, (
                "policy.generation.colocated.resources.num_nodes must be 1 or set to null "
                "when policy_nodes = 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )

            inference_nodes = 1
            reward_gpus_to_subtract = rm_gpus_per_node if total_nodes == 1 else 0
            train_gpus_per_node -= inference_gpus_per_node + reward_gpus_to_subtract
            assert train_gpus_per_node > 0, (
                "Not enough GPUs for PPO training after reserving non-colocated "
                "generation resources: "
                f"train_gpus_per_node={train_gpus_per_node}, "
                f"cluster.gpus_per_node={cluster_config['gpus_per_node']}, "
                f"inference_gpus_per_node={inference_gpus_per_node}, "
                f"reward_gpus_per_node={reward_gpus_to_subtract}."
            )
        else:
            assert inference_nodes is not None and inference_nodes > 0, (
                "policy.generation.colocated.resources.num_nodes must be > 0 "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            assert (
                inference_gpus_per_node is not None
                and inference_gpus_per_node == cluster_config["gpus_per_node"]
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be explicitly set and equal to cluster.gpus_per_node "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got inference_gpus_per_node={inference_gpus_per_node}, "
                f"cluster.gpus_per_node={cluster_config['gpus_per_node']}."
            )
            train_nodes -= inference_nodes

        assert train_nodes > 0 and inference_nodes > 0, (
            "Non-colocated PPO requires both training and inference resources, "
            f"but got train_nodes={train_nodes}, inference_nodes={inference_nodes}."
        )
        assert inference_gpus_per_node is not None

        node_resource_constraints = None
        inference_node_resource_constraints = None
        inference_segment_size = None
        if segment_size is not None:
            topology = get_ray_cluster_topology()
            num_alive_nodes = len(topology)
            required_nodes = (
                train_nodes if shared_node_inference else train_nodes + inference_nodes
            )
            assert num_alive_nodes >= required_nodes, (
                "Not enough alive Ray nodes for all PPO roles: "
                f"need {required_nodes} "
                f"(train={train_nodes}, inference={inference_nodes}, "
                f"shared_node={shared_node_inference}), "
                f"but only {num_alive_nodes} alive nodes found"
            )
            node_resource_constraints, remaining_node_ids, topology = (
                prepare_segment_topology(
                    segment_size,
                    train_nodes,
                    topology=topology,
                    role="training",
                )
            )
            if node_resource_constraints is not None:
                vllm_cfg = cast(VllmConfig, generation_config)["vllm_cfg"]
                gpus_per_instance = (
                    vllm_cfg["tensor_parallel_size"]
                    * vllm_cfg["pipeline_parallel_size"]
                )
                nodes_per_instance = (
                    gpus_per_instance + inference_gpus_per_node - 1
                ) // inference_gpus_per_node
                if nodes_per_instance > 1 and inference_nodes % nodes_per_instance == 0:
                    remaining_topology = {
                        node_id: topology[node_id] for node_id in remaining_node_ids
                    }
                    (
                        inference_node_resource_constraints,
                        _,
                        _,
                    ) = prepare_segment_topology(
                        nodes_per_instance,
                        inference_nodes,
                        topology=remaining_topology,
                        role="inference",
                    )
                    inference_segment_size = nodes_per_instance
                elif nodes_per_instance > 1:
                    print(
                        f"  ⚠ inference_nodes={inference_nodes} is not divisible by "
                        f"nodes_per_instance={nodes_per_instance}; skipping inference "
                        "topology constraints",
                        flush=True,
                    )

        train_cluster = RayVirtualCluster(
            name="ppo_train_cluster",
            bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
            use_gpus=True,
            num_gpus_per_node=train_gpus_per_node,
            max_colocated_worker_groups=2,
            port_range_low=cluster_config.get("master_port_range_low"),
            port_range_high=cluster_config.get("master_port_range_high"),
            segment_size=segment_size,
            node_resource_constraints=node_resource_constraints,
        )
        if node_resource_constraints is not None:
            train_cluster.get_placement_groups()

        inference_cluster = RayVirtualCluster(
            name="ppo_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=1,
            port_range_low=cluster_config.get("master_port_range_low"),
            port_range_high=cluster_config.get("master_port_range_high"),
            segment_size=inference_segment_size,
            node_resource_constraints=inference_node_resource_constraints,
        )
        if inference_node_resource_constraints is not None:
            VllmGeneration.init_cluster_placement_groups(
                inference_cluster, generation_config
            )

        print(
            "  ✓ Separate PPO clusters initialized: "
            f"train={train_nodes}x{train_gpus_per_node} GPUs for policy/value, "
            f"inference={inference_nodes}x{inference_gpus_per_node} GPUs for vLLM",
            flush=True,
        )

    # ==========================
    #   Training and Inference
    # ==========================
    print("\n▶ Setting up model and training...", flush=True)

    generation_config["model_name"] = policy_config["model_name"]  # Needed for vLLM

    # Dictionary to store worker initialization timing stats for logging
    worker_init_timing_metrics = {}

    weights_path, optimizer_path = checkpointer.get_resume_paths(last_checkpoint_path)
    weights_path, optimizer_path = _resolve_initial_policy_weights(
        ppo_config,
        last_checkpoint_path,
        weights_path,
        optimizer_path,
    )
    value_weights_path, value_optimizer_path = checkpointer.get_resume_paths(
        last_checkpoint_path,
        model_component="value",
    )
    value_weights_path, value_optimizer_path = _resolve_initial_value_weights(
        ppo_config,
        last_checkpoint_path,
        value_weights_path,
        value_optimizer_path,
    )

    # Each Megatron worker advances its scheduler once per train() call. Actor
    # and critic can have different update frequencies, so give each model its
    # own scheduler-tick budget.
    ppo_epochs = ppo_config["ppo_epochs"]
    critic_train_epochs = _resolve_critic_train_epochs(ppo_config)
    if ppo_epochs < 1:
        raise ValueError("ppo.ppo_epochs must be at least 1")
    if critic_train_epochs < 1:
        raise ValueError("ppo.critic_train_epochs must be null or at least 1")
    async_config = ppo_config.get("async_ppo")
    if async_config is not None and async_config.enabled:
        outer_training_steps = ppo_config["max_num_steps"]
    else:
        outer_training_steps = min(
            ppo_config["max_num_steps"],
            ppo_config["max_num_epochs"] * len(dataloader),
        )
    policy_train_iters = outer_training_steps * ppo_epochs
    value_train_iters = outer_training_steps * critic_train_epochs

    if policy_config.get("megatron_cfg", {}).get("enabled", False):
        policy_config["megatron_cfg"]["train_iters"] = policy_train_iters

    if value_config.get("megatron_cfg", {}).get("enabled", False):
        value_config["megatron_cfg"]["train_iters"] = value_train_iters

    # Define initialization functions that will be used in all paths
    def init_policy():
        """Initialize policy training workers."""
        t0 = time.perf_counter()
        p = Policy(
            cluster=train_cluster,
            config=policy_config,
            tokenizer=tokenizer,
            processor=processor,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_optimizer=True,
        )
        return p, time.perf_counter() - t0

    def init_value():
        """Initialize value model training workers."""
        t0 = time.perf_counter()
        v = Value(
            cluster=train_cluster,
            config=value_config,
            tokenizer=tokenizer,
            name_prefix="lm_value",
            weights_path=value_weights_path,
            optimizer_path=value_optimizer_path,
            init_optimizer=True,
        )
        return v, time.perf_counter() - t0

    def init_vllm():
        """Initialize vLLM generation workers."""
        t0 = time.perf_counter()
        pg = VllmGeneration(cluster=inference_cluster, config=generation_config)
        pg.finish_generation()
        return pg, time.perf_counter() - t0

    def init_sglang():
        """Initialize SGLang generation workers."""
        t0 = time.perf_counter()
        pg = SGLangGeneration(
            cluster=inference_cluster,
            sglang_cfg=generation_config,
        )
        pg.finish_generation()
        return pg, time.perf_counter() - t0

    def initialize_generation_with_policy(
        init_generation_fn,
        generation_name: str,
        init_time_key: str,
        worker_init_timing_metrics: dict,
    ):
        """Generic function to initialize a generation engine (vLLM or SGLang) along with policy.

        Args:
            init_generation_fn: Function that initializes the generation engine (init_vllm or init_sglang)
            generation_name: Name of the generation engine ("vLLM" or "SGLang")
            init_time_key: Key name for storing initialization time in metrics ("vllm_init_time_s" or "sglang_init_time_s")
            worker_init_timing_metrics: Dictionary to store timing metrics

        Returns:
            Tuple of (policy_generation, policy, value_model)
        """
        mode = "colocated" if colocated_inference else "non-colocated"
        print(f"  ⚙️  Initializing workers ({mode} mode)", flush=True)

        # Policy and value initialize serially because they share training GPUs.
        policy_generation, generation_time = init_generation_fn()
        worker_init_timing_metrics[init_time_key] = generation_time

        policy, policy_time = init_policy()
        # Block until the policy worker's __init__ completes and offload to
        # CPU, freeing GPU for value model initialization. Policy will be
        # reloaded before the vLLM refit step below.
        policy.offload_to_cpu()
        worker_init_timing_metrics["policy_init_time_s"] = policy_time

        print("  ⚙️  Initializing value model for GAE...", flush=True)
        value_model, value_time = init_value()
        # Block until the value worker's __init__ completes and offload
        # model + optimizer to CPU. Without this, __init__ runs asynchronously
        # in the Ray actor and may overlap with vLLM generation, causing
        # GPU OOM.
        value_model.finish_training()
        worker_init_timing_metrics["value_init_time_s"] = value_time
        print(f"  ✓ Value model initialized in {value_time:.2f}s", flush=True)

        return policy_generation, policy, value_model

    assert backend in ("vllm", "sglang"), (
        f"PPO requires vllm or sglang generation backend; got {backend!r}. "
        "The megatron generation backend is not supported."
    )

    if backend == "vllm":
        # vLLM generation: setup config, then initialize with policy
        generation_config = cast(VllmConfig, generation_config)
        if generation_config["vllm_cfg"]["precision"] == "fp8":
            assert loss_config.use_importance_sampling_correction is True, (
                "Importance sampling must be enabled for vLLM FP8 generation for good convergence!"
            )
        if generation_config["vllm_cfg"]["kv_cache_dtype"].startswith("fp8"):
            # FP8 KV cache requires FP8 model precision
            assert generation_config["vllm_cfg"]["precision"] == "fp8", (
                f"kv_cache_dtype='{generation_config['vllm_cfg']['kv_cache_dtype']}' requires precision='fp8'. "
                "FP8 KV cache can only be used together with FP8 model weights."
            )
            # FP8 KV cache compatibility checks
            assert policy_config["dtensor_cfg"]["enabled"] == False, (
                "DTensor backend is not supported with kv cache fp8 enabled."
            )
            assert not _should_use_async_rollouts(master_config), (
                "Async rollouts is not supported with kv cache fp8 enabled."
            )
            assert policy_config["megatron_cfg"]["pipeline_model_parallel_size"] == 1, (
                "Currently when using FP8 KV cache in generation, then in megatron we only support pipeline_model_parallel_size=1. We will add more support in future."
            )

        ## make vllm hf overrides match the training policy
        generation_config["vllm_kwargs"]["hf_overrides"] = policy_config.get(
            "hf_config_overrides", {}
        )

        policy_generation, policy, value_model = initialize_generation_with_policy(
            init_generation_fn=init_vllm,
            generation_name="vLLM",
            init_time_key="vllm_init_time_s",
            worker_init_timing_metrics=worker_init_timing_metrics,
        )

        print(
            f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}",
            flush=True,
        )

    elif backend == "sglang":
        generation_config = cast(SGLangConfig, generation_config)

        # Set model_path if not already set
        if "model_path" not in generation_config["sglang_cfg"]:
            generation_config["sglang_cfg"]["model_path"] = policy_config["model_name"]

        policy_generation, policy, value_model = initialize_generation_with_policy(
            init_generation_fn=init_sglang,
            generation_name="SGLang",
            init_time_key="sglang_init_time_s",
            worker_init_timing_metrics=worker_init_timing_metrics,
        )

        print(
            f"  ✓ Using SGLang backend for generation with {policy_config['model_name']}",
            flush=True,
        )

    nemo_gym_actor = None
    enable_nemo_gym = _should_use_nemo_gym(master_config)
    _raise_if_reward_penalties_enabled_without_nemo_gym(
        master_config, enable_nemo_gym=enable_nemo_gym
    )
    if enable_nemo_gym:
        assert backend == "vllm", (
            f"NeMo-Gym requires the vLLM generation backend; got {backend!r}."
        )
        assert policy_generation is not None
        t0 = time.perf_counter()
        nemo_gym_actor = spinup_nemo_gym_actor(
            env_configs=env_configs,
            base_urls=policy_generation.dp_openai_server_base_urls,
            model_name=generation_config["model_name"],
            enable_router_replay=False,
            routed_experts_dtype="int16",
            use_fastokens=bool(policy_config["tokenizer"].get("use_fastokens")),
        )
        worker_init_timing_metrics["nemo_gym_init_time_s"] = (
            time.perf_counter() - t0
        )

    # Record when worker initialization completes (for calculating other setup time)
    worker_init_complete_time = time.perf_counter() - setup_start_time

    # print the node IP and GPU ID of the policy workers for debugging
    policy.print_node_ip_and_gpu_id()

    # Reload policy weights to GPU before refit (they may have been offloaded
    # during setup to free GPU for value model initialization).
    policy.prepare_for_training()

    if not colocated_inference:
        assert policy_generation is not None
        t0 = time.perf_counter()
        ip, port = train_cluster.get_master_address_and_port()
        print(
            f"Using ip: {ip}, port: {port} for collective communication",
            flush=True,
        )
        train_world_size = train_cluster.world_size()
        world_size = train_world_size + inference_cluster.world_size()

        futures_train = policy.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
        futures_inference = policy_generation.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )  # type: ignore[call-arg]
        ray.get(futures_train + futures_inference)
        worker_init_timing_metrics["collective_init_time_s"] = time.perf_counter() - t0

    # prepare refit info
    state_dict_info = policy.prepare_refit_info()
    if policy_generation is not None:
        policy_generation.prepare_refit_info(state_dict_info)

    # Calculate total setup time
    total_setup_time = time.perf_counter() - setup_start_time
    worker_init_timing_metrics["total_setup_time_s"] = total_setup_time

    # Log worker initialization timing metrics to logger
    if worker_init_timing_metrics:
        print("\n▶ Worker Initialization Timing:")

        vllm_time = worker_init_timing_metrics.get("vllm_init_time_s", 0)
        policy_time = worker_init_timing_metrics.get("policy_init_time_s", 0)
        total_setup = worker_init_timing_metrics.get("total_setup_time_s", 0)

        if vllm_time:
            print(f"  vLLM init: {vllm_time:.1f}s")

        if policy_time:
            print(f"  Policy init: {policy_time:.1f}s")

        # Calculate "other" time (time after worker init completes)
        other_time = total_setup - worker_init_complete_time
        worker_init_timing_metrics["other_setup_time_s"] = other_time
        print(f"  Other setup: {other_time:.1f}s")

        print(f"  Total setup: {total_setup:.1f}s")

        # Log all metrics to the logger for analysis
        logger.log_metrics(worker_init_timing_metrics, step=0, prefix="timing/setup")

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print(f"  Total setup time: {total_setup_time:.1f}s")
    print("=" * 60 + "\n", flush=True)

    return (
        policy,
        policy_generation,
        nemo_gym_actor,
        value_model,
        (train_cluster, inference_cluster),
        dataloader,
        val_dataloader,
        loss_fn,
        value_loss_fn,
        logger,
        checkpointer,
        ppo_save_state,
        master_config,
    )


def dynamic_sampling(
    repeated_batch: BatchedDataDict[DatumSpec],
    std: torch.Tensor,
    baseline: torch.Tensor,
    dynamic_sampling_num_gen_batches: int,
    master_config: MasterConfig,
    timer: Timer,
    batch_cache: BatchedDataDict[DatumSpec] = None,
) -> BatchedDataDict[DatumSpec]:
    """Implements the dynamic sampling algorithm to select prompts with non-zero standard deviation.

    This function filters the current batch to retain only those prompts that have a non-zero standard deviation.
    If the current batch has fewer number of prompts with non-zero standard deviation than the required batch size, defined as num_prompts_per_step * num_generations_per_prompt,
    we store it in the batch_cache to be used in later iterations.
    If the current batch has more number of prompts with non-zero standard deviation than the required batch size, defined as num_prompts_per_step * num_generations_per_prompt,
    the batch is sliced to ensure batch size is num_prompts_per_step * num_generations_per_prompt.
    is_batch_complete is set to False to indicate that the current batch is not enough to meet the required batch size. This is used as a signal in the training loop
    to continue sampling or proceed to training.
    This approach is based on the dynamic sampling algorithm from the DAPO paper:
    https://arxiv.org/pdf/2503.14476.

    Args:
        repeated_batch (BatchedDataDict[DatumSpec]): The current batch of data containing prompts, responses, rewards, baselines, and std.
        std (torch.Tensor): Tensor representing the standard deviation for each prompt group.
        baseline (torch.Tensor): Baseline values for each prompt group.
        dynamic_sampling_num_gen_batches (int): Number of generation batches processed at the current step.
        master_config (MasterConfig): Configuration containing PPO and policy settings.
        batch_cache (BatchedDataDict[DatumSpec], optional): Cache storing previously selected prompts with non-zero std.

    Returns:
        tuple: A tuple containing:
            - repeated_batch (BatchedDataDict[DatumSpec]): Updated batch with selected prompts.
            - is_batch_complete (bool): Indicates if the batch has enough samples with non-zero std for training.
            - batch_cache (BatchedDataDict[DatumSpec]): Updated cache for future iterations.
    """
    # is_batch_complete is used to indicate if the current batch was able to generate enough prompts with non-zero std.
    is_batch_complete = True

    # Required batch size for training
    train_prompts_size = (
        master_config.ppo["num_prompts_per_step"]
        * master_config.ppo["num_generations_per_prompt"]
    )
    # Store the baseline, std and total_reward for the current unfiltered batch.
    repeated_batch["baseline"] = baseline
    repeated_batch["std"] = std
    total_rewards = repeated_batch["total_reward"]
    dynamic_sampling_metrics = {}

    # Dynamic sampling algorithm (used in DAPO algorithm)
    # This block implements dynamic sampling by selecting prompt groups with non-zero std.
    # If sampled prompts (with non-zero std) are fewer than num_prompts_per_step * num_generations_per_prompt, continue sampling until dynamic_sampling_max_gen_batches is reached.
    if master_config.ppo["use_dynamic_sampling"]:
        with timer.time("dynamic_sampling"):
            # Get the prompt indices with non-zero std
            non_zero_std_mask = std != 0.0

            keep_prompt_indices = torch.arange(
                len(non_zero_std_mask), device=std.device
            )[non_zero_std_mask].tolist()

            # Only select the inputs that have non-zero std
            # total_reward is already a part of repeated_batch so we don't need to add it again
            filtered_repeated_batch = repeated_batch.select_indices(keep_prompt_indices)
            filtered_repeated_batch["std"] = std[keep_prompt_indices]
            filtered_repeated_batch["baseline"] = baseline[keep_prompt_indices]

            # Store filtered and total rewards to track them separately
            filtered_rewards = filtered_repeated_batch["total_reward"]
            filtered_repeated_batch["total_reward"] = total_rewards
            filtered_repeated_batch["filtered_reward"] = filtered_rewards

            # Store the total_reward for the current filtered batch.
            # If none of the prompts in current batch have non-zero std, filtered_repeated_batch.size will be 0.
            # In this case, the current batch will be ignored and the next batch will be processed and we generate responses for it.
            if filtered_repeated_batch.size > 0:
                # Concatenate the previous partially filled batch with the current batch. This serves as a cache to store and collect the prompts with non-zero std.
                # This is used in the next iteration when the current batch is not enough to fill the buffer.
                batch_cache = (
                    filtered_repeated_batch
                    if batch_cache is None
                    else BatchedDataDict.from_batches(
                        [batch_cache, filtered_repeated_batch]
                    )
                )
                filtered_repeated_batch = batch_cache

            filtered_prompts_size = filtered_repeated_batch.size
            print(
                f"Detected {filtered_prompts_size} prompts with non-zero std; "
                f"{train_prompts_size} are required and used for training."
            )

            # If the generation samples size is smaller than a fixed threshold (train_prompts_size), keep generating by processing the next batch
            if filtered_prompts_size < train_prompts_size:
                dynamic_sampling_max_gen_batches = master_config.ppo[
                    "dynamic_sampling_max_gen_batches"
                ]
                assert dynamic_sampling_max_gen_batches > 0, (
                    "When using ppo.use_dynamic_sampling, ppo.dynamic_sampling_max_gen_batches must be > 0"
                )
                if dynamic_sampling_num_gen_batches <= dynamic_sampling_max_gen_batches:
                    print(
                        f"Generation sample buffer size: {filtered_prompts_size} is smaller than train_prompts_size: {train_prompts_size}. Processed {dynamic_sampling_num_gen_batches} batches so far out of {dynamic_sampling_max_gen_batches}."
                    )
                    is_batch_complete = False
                else:
                    raise ValueError(
                        f"Dynamic sampling has reached the maximum allowed number of batches ({dynamic_sampling_max_gen_batches}). Consider evaluating the complexity of your data or adjusting the num_prompts_per_step or num_generations_per_prompt parameters to enhance the diversity of the samples."
                    )
            else:
                num_discarded_valid_samples = filtered_prompts_size - train_prompts_size
                dynamic_sampling_metrics[
                    "dynamic_sampling_num_discarded_valid_samples"
                ] = num_discarded_valid_samples

                #  Slice the batch, rewards, baselines and std to ensure batch size is train_prompts_size
                filtered_repeated_batch = filtered_repeated_batch.slice(
                    0, train_prompts_size
                )

    batch_to_return = (
        filtered_repeated_batch
        if master_config.ppo["use_dynamic_sampling"]
        else repeated_batch
    )
    return batch_to_return, is_batch_complete, batch_cache, dynamic_sampling_metrics


def _create_advantage_estimator(master_config: MasterConfig):
    """Create and return an advantage estimator based on configuration.

    PPO's training loop consumes a `(advantages, returns)` pair from a
    value-model-based estimator, so only `gae` and `raw_reward` are supported
    here. Group-relative estimators like GRPO / Reinforce++ are not compatible
    with PPO's loop and live in `grpo.py`.

    Args:
        master_config: The master configuration dictionary.

    Returns:
        A `GeneralizedAdvantageEstimator` or `RawRewardAdvantageEstimator` instance.

    Raises:
        ValueError: If the advantage estimator name is not recognized.
    """
    ppo_config = master_config.ppo
    loss_config = master_config.loss_fn

    adv_estimator_config = ppo_config["adv_estimator"]

    adv_estimator_name = adv_estimator_config["name"]
    if adv_estimator_name == "gae":
        adv_estimator = GeneralizedAdvantageEstimator(adv_estimator_config, loss_config)
        gae_lambda = adv_estimator_config["gae_lambda"]
        gae_gamma = adv_estimator_config["gae_gamma"]
        print(f"  ✓ Using GAE advantage estimator (λ={gae_lambda}, γ={gae_gamma})")
    elif adv_estimator_name == "raw_reward":
        adv_estimator = RawRewardAdvantageEstimator(adv_estimator_config, loss_config)
        print("  ✓ Using raw reward advantage estimator (no value model, no baselines)")
    else:
        raise ValueError(
            f"Invalid adv_estimator name for PPO: {adv_estimator_name!r}. "
            f"PPO only supports 'gae' or 'raw_reward'."
        )

    return adv_estimator


def _compute_critic_metrics(value_results: dict[str, Any]) -> dict[str, Any]:
    """Aggregate value-model metrics under the ``critic/`` namespace."""
    value_mb_metrics = value_results.get("all_mb_metrics", {})
    critic_metrics: dict[str, Any] = {
        "critic/grad_norm": value_results["grad_norm"].numpy(),
        "critic/loss": value_results["loss"].numpy(),
        "critic/vf_loss": value_results["loss"].numpy(),
    }
    for key, value in value_mb_metrics.items():
        metric_name = f"critic/{key}"
        if key in {"lr", "wd", "global_valid_seqs", "global_valid_toks", "grad_norm"}:
            critic_metrics[metric_name] = np.mean(value).item()
        elif key == "values_min":
            critic_metrics[metric_name] = np.min(value).item()
        elif key == "values_max":
            critic_metrics[metric_name] = np.max(value).item()
        elif isinstance(value, (np.ndarray, list)):
            critic_metrics[metric_name] = np.sum(value).item()
        else:
            raise ValueError(f"Unsupported value-model metric: {key}")
    returns_mean = critic_metrics.get("critic/returns_mean", 0)
    values_mean = critic_metrics.get("critic/values_mean", 0)
    returns_sq_mean = critic_metrics.get("critic/returns_sq_mean", 0)
    residual_sq_mean = critic_metrics.get("critic/residual_sq_mean", 0)
    returns_var = returns_sq_mean - returns_mean**2
    residual_var = residual_sq_mean - (returns_mean - values_mean) ** 2
    critic_metrics["critic/explained_var"] = 1.0 - residual_var / max(returns_var, 1e-8)
    return critic_metrics


def _compute_actor_metrics(
    train_results: dict[str, Any],
    reference_policy_kl_penalty: float,
) -> dict[str, Any]:
    """Aggregate policy-model optimizer and PPO metrics under ``actor/``."""
    actor_metrics: dict[str, Any] = {
        "actor/total_loss": train_results["loss"].numpy(),
        "actor/grad_norm": train_results["grad_norm"].numpy(),
        "actor/ref_kl_coef": reference_policy_kl_penalty,
    }
    metric_names = {
        "loss": "total_loss",
        "approx_entropy": "entropy",
        "kl_penalty": "ref_kl",
    }
    for key, value in train_results.get("all_mb_metrics", {}).items():
        metric_name = f"actor/{metric_names.get(key, key)}"
        if key == "loss":
            continue
        if key in {
            "lr",
            "wd",
            "global_valid_seqs",
            "global_valid_toks",
            "grad_norm",
        }:
            actor_metrics[metric_name] = np.mean(value).item()
        elif key in {"probs_ratio_min", "probs_ratio_clamped_min"}:
            valid_values = [x for x in value if not np.isinf(x)]
            actor_metrics[metric_name] = (
                np.min(valid_values).item() if valid_values else -1.0
            )
        elif key in {"probs_ratio_max", "probs_ratio_clamped_max"}:
            valid_values = [x for x in value if not np.isinf(x)]
            actor_metrics[metric_name] = (
                np.max(valid_values).item() if valid_values else -1.0
            )
        elif isinstance(value, (np.ndarray, list)):
            actor_metrics[metric_name] = np.sum(value).item()
        else:
            raise ValueError(f"Unsupported policy-model metric: {key}")
    return actor_metrics


def _masked_distribution_metrics(
    name: str,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Compute stable scalar summaries over valid response-token positions."""
    selected = torch.masked_select(values.detach().float(), mask.bool())
    if selected.numel() == 0:
        return {
            f"{name}/mean": 0.0,
            f"{name}/std": 0.0,
            f"{name}/min": 0.0,
            f"{name}/p05": 0.0,
            f"{name}/p50": 0.0,
            f"{name}/p95": 0.0,
            f"{name}/max": 0.0,
        }

    quantiles = torch.quantile(
        selected,
        torch.tensor([0.05, 0.5, 0.95], device=selected.device),
    )
    return {
        f"{name}/mean": selected.mean().item(),
        f"{name}/std": selected.std(unbiased=False).item(),
        f"{name}/min": selected.min().item(),
        f"{name}/p05": quantiles[0].item(),
        f"{name}/p50": quantiles[1].item(),
        f"{name}/p95": quantiles[2].item(),
        f"{name}/max": selected.max().item(),
    }


def _compute_rollout_critic_metrics(
    train_data: BatchedDataDict[ClippedPGLossDataDict],
    mask: torch.Tensor,
) -> dict[str, float]:
    """Summarize the critic predictions and GAE targets before PPO updates."""
    if "values" not in train_data or "returns" not in train_data:
        return {}

    values = train_data["values"].detach().float()
    returns = train_data["returns"].detach().float()
    raw_advantages = returns - values
    valid = mask.bool()

    metrics = {
        **_masked_distribution_metrics(
            "critic/pre_update_values", values, valid
        ),
        **_masked_distribution_metrics(
            "critic/return_targets", returns, valid
        ),
        **_masked_distribution_metrics(
            "critic/raw_advantages", raw_advantages, valid
        ),
    }

    selected_values = values[valid]
    selected_returns = returns[valid]
    residual = selected_values - selected_returns
    return_var = selected_returns.var(unbiased=False)
    residual_var = residual.var(unbiased=False)
    metrics.update(
        {
            "critic/pre_update_bias": residual.mean().item(),
            "critic/pre_update_mae": residual.abs().mean().item(),
            "critic/pre_update_rmse": residual.square().mean().sqrt().item(),
            "critic/pre_update_explained_var": (
                1.0 - residual_var / return_var.clamp_min(1e-8)
            ).item(),
        }
    )

    first_indices = valid.float().argmax(dim=1)
    last_indices = valid.shape[1] - 1 - valid.fliplr().float().argmax(dim=1)
    valid_samples = valid.any(dim=1)
    batch_indices = torch.arange(valid.shape[0], device=valid.device)[valid_samples]
    first_values = values[batch_indices, first_indices[valid_samples]]
    last_values = values[batch_indices, last_indices[valid_samples]]
    terminal_targets = returns[batch_indices, last_indices[valid_samples]]
    metrics.update(
        {
            "critic/first_response_token_value_mean": first_values.mean().item(),
            "critic/last_response_token_value_mean": last_values.mean().item(),
            "critic/last_response_token_mae": (
                last_values - terminal_targets
            ).abs().mean().item(),
        }
    )
    return metrics


def _pad_compaction_batch(
    data: BatchedDataDict,
    *,
    batch_multiple: int,
    mask_padding: bool,
) -> tuple[BatchedDataDict, int]:
    """Pad a variable segment batch to a model boundary's sharding multiple."""
    if data.size == 0:
        raise ValueError("Cannot pad an empty compaction batch")
    if batch_multiple <= 0:
        raise ValueError(f"batch_multiple must be positive, got {batch_multiple}")

    padding = (-data.size) % batch_multiple
    if padding == 0:
        return data, 0

    indices = list(range(data.size)) + [0] * padding
    padded = data.select_indices(indices)
    if mask_padding:
        if "sample_mask" not in padded:
            raise KeyError("sample_mask is required when masking compaction padding")
        padded["sample_mask"][-padding:] = 0
    return padded, padding


def _compaction_batch_multiple(dp_size: int, micro_batch_size: int) -> int:
    """Return the global row multiple required by a DP-sharded model call."""
    if dp_size <= 0 or micro_batch_size <= 0:
        raise ValueError(
            "Compaction batch alignment requires positive DP and micro-batch sizes, "
            f"got dp_size={dp_size}, micro_batch_size={micro_batch_size}"
        )
    return dp_size * micro_batch_size


# ===============================================================================
# Training & Validation
# ===============================================================================


def ppo_train(
    policy: ColocatablePolicyInterface,
    policy_generation: Optional[GenerationInterface],
    value_model: ValueInterface,
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: LossFunction,
    value_loss_fn: LossFunction,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    ppo_save_state: PPOSaveState,
    master_config: MasterConfig,
) -> None:
    """Run PPO training algorithm.

    Based on the grpo_train loop with PPO-specific modifications:
    - Value model inference and training (actor-critic)
    - GAE advantage estimation with value bootstrap
    - Multiple training steps per rollout (ppo_epochs)
    - Configurable policy training start epoch
    """
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config.checkpointing["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()
    memory_tracker = MemoryTracker()

    kv_scales_cache = None  # Cache reused for computed kv scales

    NEED_REFIT = True
    # If policy_generation is None, use the policy as the generation interface (megatron framework backend)
    if policy_generation is None:
        policy_generation = policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running
    assert policy_generation is not None  # for mypy type check

    if master_config.ppo.get("skip_reference_policy_logprobs_calculation"):
        assert master_config.loss_fn.reference_policy_kl_penalty == 0
        print(
            "Reference policy logprob calculation will be skipped since `ppo.skip_reference_policy_logprobs_calculation` is set to True and `loss_fn.reference_policy_kl_penalty` is 0."
        )

    # Check if we need to sync KV cache scales
    sync_kv_scales = getattr(policy_generation, "requires_kv_scale_sync", False)

    # common config/state
    current_step = ppo_save_state["current_step"]
    total_steps = ppo_save_state["total_steps"]
    max_num_steps = master_config.ppo["max_num_steps"]
    current_epoch = ppo_save_state["current_epoch"]
    max_num_epochs = master_config.ppo["max_num_epochs"]
    ppo_epochs = master_config.ppo["ppo_epochs"]
    critic_train_epochs = _resolve_critic_train_epochs(master_config.ppo)
    # Number of PPO steps to train only the critic before starting policy
    # training.  Despite the legacy name, this is compared against total_steps
    # (not current_epoch) to match veRL's critic_warmup semantics.
    policy_training_start_step = master_config.ppo["policy_training_start_step"]
    consumed_samples = ppo_save_state["consumed_samples"]
    total_valid_tokens = ppo_save_state.get("total_valid_tokens", 0)
    val_at_start = master_config.ppo["val_at_start"]
    val_at_end = master_config.ppo["val_at_end"]
    val_period = master_config.ppo["val_period"]
    colocated_inference = master_config.policy["generation"]["colocated"]["enabled"]

    # Initialize advantage estimator
    adv_estimator = _create_advantage_estimator(master_config)

    # Run validation at the start if configured
    if val_at_start and current_step == 0:
        print("\n🔍 Running initial validation...", flush=True)
        memory_tracker.snapshot_start_of_stage("Initial validation", dir())

        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(policy, policy_generation, colocated_inference)
            if not colocated_inference:
                # Colocated refit offloads policy inside
                # `refit_policy_generation`. Do it here so the value
                # model can reuse the training GPUs.
                policy.offload_to_cpu()
            POLICY_GENERATION_STALE = False
        else:
            policy_generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            policy_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=0,
            master_config=master_config,
            logger=logger,
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, current_step, prefix="validation")
        logger.log_metrics(validation_timings, current_step, prefix="timing/validation")

    ft_save_period = master_config.checkpointing.get("ft_save_period")

    while current_epoch < max_num_epochs and total_steps < max_num_steps:
        memory_tracker.snapshot_start_of_stage("Preparing batch", dir())
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")

        for batch in dataloader:
            metrics_logging_data = dict()
            metrics = dict()

            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(dataloader), max_num_steps)} {'=' * 25}",
                flush=True,
            )
            maybe_gpu_profile_step(policy, total_steps + 1)
            if policy != policy_generation:
                maybe_gpu_profile_step(policy_generation, total_steps + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                # Prepare batch
                print("▶ Preparing batch...", flush=True)
                with timer.time("data_processing"):
                    repeated_batch: BatchedDataDict[DatumSpec] = (
                        batch.repeat_interleave(
                            master_config.ppo["num_generations_per_prompt"]
                        )
                    )
                    batched_flat, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    )
                    input_ids = batched_flat["token_ids"]

                # Generate responses
                memory_tracker.snapshot_start_of_stage("Generation", dir())
                print(
                    f"▶ Generating responses for batch of size {repeated_batch.size}...",
                    flush=True,
                )
                with timer.time("prepare_for_generation/total"):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        # Ensure value is offloaded and policy params are on GPU before refit.
                        value_model.finish_training()
                        policy.prepare_for_lp_inference()

                        if sync_kv_scales and kv_scales_cache is None:
                            print("▶ Computing KV cache scales...", flush=True)
                            calib_flat, calib_input_lengths = (
                                batched_message_log_to_flat_message(
                                    repeated_batch["message_log"],
                                    pad_value_dict={
                                        "token_ids": tokenizer.pad_token_id
                                    },
                                    make_sequence_length_divisible_by=master_config.policy[
                                        "make_sequence_length_divisible_by"
                                    ],
                                )
                            )
                            calibration_data = BatchedDataDict[ClippedPGLossDataDict](
                                {
                                    "input_ids": calib_flat["token_ids"],
                                    "input_lengths": calib_input_lengths,
                                }
                            )
                            calibration_data.to("cpu")
                            kv_scales_cache = policy.calibrate_qkv_fp8_scales(
                                calibration_data, include_q=True
                            )["layers"]

                        refit_policy_generation(
                            policy,
                            policy_generation,
                            colocated_inference,
                            timer=timer,
                            kv_scales=kv_scales_cache if sync_kv_scales else None,
                        )
                        if not colocated_inference:
                            # Colocated refit offloads policy inside
                            # `refit_policy_generation`. Do it here so the value
                            # model can reuse the training GPUs.
                            with timer.time("policy_offload_after_refit"):
                                policy.offload_to_cpu()
                        POLICY_GENERATION_STALE = False
                    else:
                        if colocated_inference:
                            policy.offload_to_cpu()
                        policy_generation.prepare_for_generation()

                with timer.time("generation"):
                    if policy_generation is not None:
                        policy_generation.clear_logger_metrics()

                    if _should_use_nemo_gym(master_config):
                        generation_config = master_config.policy["generation"]
                        nemo_gym_rollout_result = run_nemo_gym_rollout_sync(
                            policy_generation=policy_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=None,
                            generation_config=generation_config,
                            log_full_result_tables=should_log_nemo_gym_full_result_tables(
                                wandb_enabled=master_config.logger["wandb_enabled"],
                                wandb_config=master_config.logger["wandb"],
                            ),
                            max_rollout_turns=None,
                            greedy=False,
                            reward_penalty_config=master_config.reward_penalties,
                            thinking_tags=get_nemo_gym_thinking_tags(master_config.env),
                        )
                        input_ids = nemo_gym_rollout_result.input_ids
                        repeated_batch = nemo_gym_rollout_result.final_batch
                        rollout_metrics = nemo_gym_rollout_result.rollout_metrics
                        del nemo_gym_rollout_result

                    elif _should_use_async_rollouts(master_config):
                        (
                            repeated_batch,
                            rollout_metrics,
                        ) = run_async_multi_turn_rollout(
                            policy_generation=policy_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config.policy[
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=master_config.ppo["max_rollout_turns"],
                            greedy=False,
                        )
                    else:
                        repeated_batch, rollout_metrics = run_multi_turn_rollout(
                            policy_generation=policy_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config.policy[
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=master_config.ppo["max_rollout_turns"],
                            greedy=False,
                        )
                    policy_generation.finish_generation()
                    generation_logger_metrics = policy_generation.get_logger_metrics()

                    metrics_logging_data["mean_gen_tokens_per_sample"] = (
                        rollout_metrics["mean_gen_tokens_per_sample"]
                    )
                    logger.log_metrics(rollout_metrics, total_steps + 1, prefix="train")

                repeated_batch = scale_rewards(
                    repeated_batch, master_config.ppo["reward_scaling"]
                )
                if master_config.ppo["reward_shaping"]["enabled"]:
                    repeated_batch = apply_reward_shaping(
                        repeated_batch, master_config.ppo["reward_shaping"]
                    )

                # Process rewards and build training data
                memory_tracker.snapshot_start_of_stage("Processing rewards", dir())
                print("▶ Processing rewards...", flush=True)
                with timer.time("reward_calculation"):
                    rewards = repeated_batch["total_reward"]

                with timer.time("data_processing"):
                    use_overlong_filtering = master_config.ppo["overlong_filtering"]
                    if use_overlong_filtering:
                        loss_multiplier = repeated_batch["loss_multiplier"].clone()
                        truncated = repeated_batch["truncated"]
                        if isinstance(truncated, list):
                            truncated = torch.tensor(truncated, dtype=torch.bool)
                        loss_multiplier[truncated] = 0
                        repeated_batch["loss_multiplier"] = loss_multiplier

                    metrics_logging_data["num_mask_sample_filtered"] = (
                        _apply_ppo_mask_sample_filter(repeated_batch)
                    )

                    for i, message_log in enumerate(repeated_batch["message_log"]):
                        for j, message in enumerate(message_log):
                            if message["role"] == "assistant":
                                message["token_loss_mask"] = torch.ones_like(
                                    message["token_ids"]
                                )
                            else:
                                message["token_loss_mask"] = torch.zeros_like(
                                    message["token_ids"]
                                )
                            if "generation_logprobs" not in message:
                                message["generation_logprobs"] = torch.zeros_like(
                                    message["token_ids"], dtype=torch.float32
                                )

                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config.policy[
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    train_data = BatchedDataDict[ClippedPGLossDataDict](
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "generation_logprobs": flat_messages["generation_logprobs"],
                            "rewards": repeated_batch["total_reward"],
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )
                    extra_multimodal_data = flat_messages.get_multimodal_dict(
                        as_tensors=False
                    )
                    train_data.update(extra_multimodal_data)
                    train_data.to("cpu")

                    metrics_logging_data["content"] = flat_messages["content"]

                memory_tracker.snapshot_start_of_stage("Value inference", dir())
                print("▶ Computing values...", flush=True)
                with timer.time("value_inference"):
                    value_model.prepare_for_inference()
                    values = value_model.get_values(train_data)
                    train_data["values"] = values["values"].squeeze(-1)
                    value_model.finish_inference()

                print(
                    f"  • Average batch reward: {rewards.mean().numpy():.4f}\n"
                    f"  • Average batch response length: {input_lengths.sum() / input_lengths.shape[0]:.4f}"
                )

                # Compute logprobs
                memory_tracker.snapshot_start_of_stage("Computing logprobs", dir())
                print("▶ Preparing for logprob inference...", flush=True)
                with timer.time("logprob_inference_prep"):
                    policy.prepare_for_lp_inference()

                print("▶ Computing logprobs...", flush=True)
                with timer.time("policy_and_reference_logprobs"):
                    logprob_data = BatchedDataDict[ClippedPGLossDataDict](
                        {
                            "input_ids": train_data["input_ids"],
                            "input_lengths": train_data["input_lengths"],
                            **extra_multimodal_data,
                        }
                    )
                    train_data["prev_logprobs"] = policy.get_logprobs(
                        logprob_data, timer=timer
                    )["logprobs"]

                    if not master_config.ppo.get(
                        "skip_reference_policy_logprobs_calculation"
                    ):
                        train_data["reference_policy_logprobs"] = (
                            policy.get_reference_policy_logprobs(
                                logprob_data,
                                timer=timer,
                            )["reference_logprobs"]
                        )

                    del logprob_data
                    del extra_multimodal_data

                    policy.finish_inference()

                (
                    advantage_mask,
                    seq_logprob_error_metrics,
                ) = _apply_ppo_seq_logprob_error_masking(
                    train_data=train_data,
                    rewards=rewards,
                    seq_logprob_error_threshold=master_config.ppo[
                        "seq_logprob_error_threshold"
                    ],
                )

                # Build prompt IDs for advantage estimation (groups responses from same prompt).
                # Use the token-length-based extractor so multi-turn prompts containing
                # assistant messages still resolve to the original prompt only.
                with timer.time("advantage_calculation"):
                    print("▶ Computing advantages...", flush=True)
                    initial_prompt_message_logs = extract_initial_prompt_messages(
                        repeated_batch["message_log"],
                        repeated_batch["length"],
                    )
                    prompt_batched_flat, _ = batched_message_log_to_flat_message(
                        initial_prompt_message_logs,
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    )
                    prompt_ids_for_adv = prompt_batched_flat["token_ids"]
                    del initial_prompt_message_logs
                    del prompt_batched_flat

                    adv_kwargs = dict(
                        prompt_ids=prompt_ids_for_adv,
                        rewards=train_data["rewards"],
                        mask=advantage_mask,
                        reference_logprobs=train_data.get("reference_policy_logprobs"),
                        logprobs=train_data["prev_logprobs"],
                    )
                    if "values" in train_data:
                        adv_kwargs["values"] = train_data["values"]
                    result = adv_estimator.compute_advantage(**adv_kwargs)
                    if isinstance(result, tuple):
                        advantages, returns = result
                    else:
                        advantages, returns = result, None
                    del prompt_ids_for_adv

                    train_data["advantages"] = advantages
                    if returns is not None:
                        train_data["returns"] = returns

                metrics.update(
                    _compute_rollout_critic_metrics(train_data, advantage_mask)
                )

                # Critic and actor update frequencies are independent. The
                # critic runs first so all actor updates use the same rollout
                # batch after the requested number of value updates.
                memory_tracker.snapshot_start_of_stage("Policy train", dir())
                value_results = None
                for critic_epoch in range(critic_train_epochs):
                    print(
                        f"▶ Critic update {critic_epoch + 1}/{critic_train_epochs}...",
                        flush=True,
                    )
                    with timer.time("value_training_prep"):
                        value_model.prepare_for_training()
                    with timer.time("value_training"):
                        print("▶ Training value...", flush=True)
                        value_results = value_model.train(
                            train_data,
                            value_loss_fn,
                            timer=timer,
                        )

                        value_model.finish_training()

                train_results = None
                if total_steps >= policy_training_start_step:
                    if (
                        total_steps == policy_training_start_step
                        and policy_training_start_step > 0
                    ):
                        print(
                            f"  ✓ Critic warmup complete ({policy_training_start_step} steps). "
                            f"Starting policy training.",
                            flush=True,
                        )
                    for actor_epoch in range(ppo_epochs):
                        print(
                            f"▶ Actor update {actor_epoch + 1}/{ppo_epochs}...",
                            flush=True,
                        )
                        print("▶ Preparing for training...", flush=True)
                        with timer.time("training_prep"):
                            policy.prepare_for_training()
                            POLICY_GENERATION_STALE = True

                        print("▶ Training policy...", flush=True)
                        with timer.time("policy_training"):
                            train_results = policy.train(
                                train_data,
                                loss_fn,
                                timer=timer,
                            )
                            if actor_epoch < ppo_epochs - 1:
                                policy.offload_to_cpu()

                if train_results is not None:
                    print(
                        f"    • Policy loss: {train_results['loss'].mean().item():.4f}"
                    )
                if value_results is not None:
                    print(
                        f"    • Value loss: {value_results['loss'].mean().item():.4f}"
                    )

                # Recompute KV scales after policy training if needed
                if sync_kv_scales:
                    with timer.time("recompute_kv_scales"):
                        print(
                            "▶ Recomputing KV cache scales after policy update...",
                            flush=True,
                        )
                        kv_scales_cache = policy.calibrate_qkv_fp8_scales(
                            train_data, include_q=True
                        )["layers"]
                        POLICY_GENERATION_STALE = True

                is_last_step = (total_steps + 1 >= max_num_steps) or (
                    (current_epoch + 1 == max_num_epochs)
                    and (current_step + 1 == len(dataloader))
                )

                # Validation
                if (val_period > 0 and (total_steps + 1) % val_period == 0) or (
                    val_at_end and is_last_step
                ):
                    memory_tracker.snapshot_start_of_stage("Validation", dir())
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            policy,
                            policy_generation,
                            colocated_inference,
                            kv_scales=kv_scales_cache if sync_kv_scales else None,
                        )
                        if not colocated_inference:
                            # Colocated refit offloads policy inside
                            # `refit_policy_generation`. Do it here so the value
                            # model can reuse the training GPUs.
                            with timer.time("policy_offload_after_refit"):
                                policy.offload_to_cpu()
                        POLICY_GENERATION_STALE = False
                    else:
                        if colocated_inference:
                            policy.offload_to_cpu()
                        policy_generation.prepare_for_generation()
                    val_metrics, validation_timings = validate(
                        policy_generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        step=total_steps + 1,
                        master_config=master_config,
                        logger=logger,
                    )
                    policy_generation.finish_generation()
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )

                # Metrics
                flat_advantages = train_data["advantages"]
                del flat_messages

                memory_tracker.snapshot_start_of_stage("Metrics", dir())
                if train_results is not None:
                    metrics.update(
                        _compute_actor_metrics(
                            train_results,
                            master_config.loss_fn.reference_policy_kl_penalty,
                        )
                    )
                    if "moe_metrics" in train_results:
                        metrics.update(
                            {
                                f"moe/{k}": v
                                for k, v in train_results["moe_metrics"].items()
                            }
                        )

                # Extract critic metrics from value training results
                if value_results is not None:
                    metrics.update(_compute_critic_metrics(value_results))
                metrics["actor/update_applied"] = float(train_results is not None)
                metrics["critic/update_applied"] = float(value_results is not None)
                metrics["actor/num_updates"] = (
                    float(ppo_epochs) if train_results is not None else 0.0
                )
                metrics["critic/num_updates"] = float(critic_train_epochs)
                metrics.update(
                    {
                        "reward": rewards.numpy(),
                        "mean_prompt_length": repeated_batch["length"].numpy(),
                        "total_num_tokens": input_lengths.numpy(),
                    }
                )
                metrics.update(
                    _masked_distribution_metrics(
                        "advantages", flat_advantages, advantage_mask
                    )
                )

                gen_step_metrics = {}
                if hasattr(policy_generation, "get_step_metrics"):
                    gen_step_metrics = policy_generation.get_step_metrics()
                metrics.update(gen_step_metrics)

                for k, v in metrics.items():
                    if k in {
                        "reward",
                        "global_valid_seqs",
                        "global_valid_toks",
                        "mean_prompt_length",
                    }:
                        metrics[k] = np.mean(v).item()
                    elif isinstance(v, (np.ndarray, list)):
                        metrics[k] = np.sum(v).item()

                metrics.update(rollout_metrics)
                metrics["generation_logger_metrics"] = generation_logger_metrics
                metrics.update(seq_logprob_error_metrics)
                if "global_valid_toks" in metrics:
                    total_valid_tokens += metrics["global_valid_toks"]

                ## Checkpointing
                consumed_samples += master_config.ppo["num_prompts_per_step"]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config.checkpointing["save_period"]
                    == 0
                    or (
                        ft_save_period is not None
                        and (total_steps + 1) % ft_save_period == 0
                    )
                )
                should_save_by_timeout = timeout.check_save()

                memory_tracker.snapshot_start_of_stage("Checkpointing", dir())
                if master_config.checkpointing["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    ppo_save_state["current_step"] = current_step + 1
                    ppo_save_state["total_steps"] = total_steps + 1
                    ppo_save_state["current_epoch"] = current_epoch
                    ppo_save_state["total_valid_tokens"] = total_valid_tokens
                    if val_metrics is not None:
                        ppo_save_state["val_reward"] = val_metrics["accuracy"]
                    elif "val_reward" in ppo_save_state:
                        del ppo_save_state["val_reward"]
                    ppo_save_state["consumed_samples"] = consumed_samples

                    full_metric_name = master_config.checkpointing["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:',\n"
                            f'followed by the corresponding name in the "val" or "train" metrics dictionary.'
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. "
                                "This checkpoint will not be saved as top-k.",
                                stacklevel=2,
                            )
                            if full_metric_name in ppo_save_state:
                                del ppo_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            ppo_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

                    with timer.time("checkpointing"):
                        print(
                            f"Saving checkpoint for step {total_steps + 1}...",
                            flush=True,
                        )
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, ppo_save_state, master_config
                        )

                        # Always save policy weights so every PPO checkpoint has
                        # the same component layout. Before the first real policy
                        # update, omit optimizer and scheduler state because their
                        # lazily initialized state is not yet safe to checkpoint.
                        policy.prepare_for_training()
                        policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=(
                                os.path.join(checkpoint_path, "policy", "optimizer")
                                if (
                                    checkpointer.save_optimizer
                                    and total_steps >= policy_training_start_step
                                )
                                else None
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config.checkpointing,
                        )
                        policy.offload_to_cpu()

                        value_model.prepare_for_training()
                        value_model.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "value", "weights"
                            ),
                            optimizer_path=(
                                os.path.join(checkpoint_path, "value", "optimizer")
                                if checkpointer.save_optimizer
                                else None
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "value", "tokenizer"
                            ),
                            checkpointing_cfg=master_config.checkpointing,
                        )
                        value_model.finish_training()

                        torch.save(
                            dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        # The value worker finalizes its own write synchronously
                        # (blocking=True) inside save_checkpoint, so only the
                        # policy's async write needs to be awaited before rename.
                        checkpointer.begin_finalization(
                            checkpoint_path,
                            wait_fn=policy.finalize_async_save,
                        )

            # Logging
            memory_tracker.snapshot_start_of_stage("Logging", dir())

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )  # type: ignore

            del train_data

            print("\n📊 Training Results:")
            if train_results is not None:
                print(f"  • Actor Total Loss: {metrics.get('actor/total_loss', 'N/A')}")
                print(f"  • Actor PG Loss: {metrics.get('actor/pg_loss', 'N/A')}")
                print(f"  • Actor Grad Norm: {metrics.get('actor/grad_norm', 'N/A')}")
                print(f"  • Actor LR: {metrics.get('actor/lr', 'N/A')}")
                print(f"  • Actor PPO KL: {metrics.get('actor/ppo_kl', 'N/A')}")
                print(
                    f"  • Actor PG Clip Frac: "
                    f"{metrics.get('actor/pg_clipfrac', 'N/A')}"
                )
            if value_results is not None:
                print(f"  • Critic VF Loss: {metrics.get('critic/vf_loss', 'N/A')}")
                print(f"  • Critic Grad Norm: {metrics.get('critic/grad_norm', 'N/A')}")
                if "critic/lr" in metrics:
                    print(f"  • Critic LR: {metrics['critic/lr']:.2e}")
                if "critic/vf_clipfrac" in metrics:
                    print(f"  • Critic Clip Frac: {metrics['critic/vf_clipfrac']:.4f}")
            print(f"  • Avg Reward: {np.mean(rewards.numpy()):.4f}")
            print(
                f"  • Mean Generation Length: {metrics_logging_data['mean_gen_tokens_per_sample']:.4f}",
                flush=True,
            )

            print("\n⏱️  Timing:", flush=True)
            total_time = timing_metrics.get("total_step_time", 0)

            number_of_samples_per_step = (
                master_config.ppo["num_prompts_per_step"]
                * master_config.ppo["num_generations_per_prompt"]
            )
            total_num_gpus = (
                master_config.cluster["num_nodes"]
                * master_config.cluster["gpus_per_node"]
            )

            print(f"  • Total step time: {total_time:.2f}s", flush=True)
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)", flush=True)

            if "global_valid_toks" in metrics:
                timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                    metrics["global_valid_toks"] / total_time / total_num_gpus
                    if total_time > 0
                    else 0
                )
            performance_metrics = print_performance_metrics(
                train_results if train_results is not None else (value_results or {}),
                metrics,
                timing_metrics,
                master_config,
            )

            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(
                performance_metrics, total_steps + 1, prefix="performance"
            )
            logger.log_metrics(
                timing_metrics,
                total_steps + 1,
                prefix="timing/train",
                step_finished=True,
            )

            # Clear mem
            memory_tracker.snapshot_start_of_stage("After CPU memory clear", dir())
            del repeated_batch
            del rewards
            del metrics
            del val_metrics

            timer.reset()
            current_step += 1
            total_steps += 1
            if should_save_by_timeout:
                checkpointer.shutdown()
                memory_tracker.snapshot_start_of_stage("", dir())
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if total_steps >= max_num_steps:
                checkpointer.shutdown()
                memory_tracker.snapshot_start_of_stage("", dir())
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

        current_epoch += 1
        current_step = 0

    # Flush the last checkpoint's background finalization on an epoch-bounded
    # exit. Reaching max_num_epochs falls through the while loop and bypasses
    # the inline shutdown() calls at the max_num_steps / timeout early returns,
    # so without this the daemon finalization thread could be killed before the
    # final tmp_step_N is renamed.
    checkpointer.shutdown()


def _async_ppo_generation_lead_steps(
    *,
    step: int,
    policy_training_start_step: int,
    max_trajectory_age_steps: int,
    warmup_max_trajectory_age_steps: int,
) -> int:
    """Return the collector lead without crossing the safe warmup frontier."""
    if step >= policy_training_start_step:
        return max_trajectory_age_steps

    max_warmup_target = policy_training_start_step + max_trajectory_age_steps
    remaining_to_frontier = max_warmup_target - step
    return max(
        max_trajectory_age_steps,
        min(warmup_max_trajectory_age_steps, remaining_to_frontier),
    )


def _async_ppo_buffer_max_age(
    *,
    step: int,
    policy_training_start_step: int,
    max_trajectory_age_steps: int,
    warmup_max_trajectory_age_steps: int,
) -> int:
    """Keep frozen-policy rollouts valid through their safe training frontier."""
    warmup_rollout_frontier = policy_training_start_step + max_trajectory_age_steps
    if policy_training_start_step > 0 and step <= warmup_rollout_frontier:
        return warmup_max_trajectory_age_steps
    return max_trajectory_age_steps


class _CyclingDataLoader:
    """Repeat a PPO dataloader until collection stops."""

    def __init__(self, dataloader: StatefulDataLoader) -> None:
        self.dataloader = dataloader

    def __iter__(self) -> Iterator[BatchedDataDict]:
        consecutive_empty_epochs = 0
        while True:
            produced_this_epoch = False
            for batch in self.dataloader:
                produced_this_epoch = True
                yield batch

            if produced_this_epoch:
                consecutive_empty_epochs = 0
            else:
                consecutive_empty_epochs += 1
                if consecutive_empty_epochs >= 2:
                    raise RuntimeError(
                        "Dataloader yielded no batches for two consecutive epochs"
                    )

    def state_dict(self) -> dict[str, Any]:
        return self.dataloader.state_dict()


def async_ppo_train(
    policy: ColocatablePolicyInterface,
    policy_generation: Optional[GenerationInterface],
    value_model: ValueInterface,
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: LossFunction,
    value_loss_fn: LossFunction,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    ppo_save_state: PPOSaveState,
    master_config: MasterConfig,
) -> None:
    """Run PPO while a background collector fills a replay buffer."""
    # ------------------------------------------------------------------
    # Entry guards (fail loud at startup, not deep in the loop)
    # ------------------------------------------------------------------
    generation_config = master_config.policy["generation"]
    backend = generation_config.get("backend", "") if generation_config else ""
    if backend != "vllm" or not _should_use_async_rollouts(master_config):
        raise ValueError(
            "Async PPO requires policy.generation.backend=vllm and "
            "policy.generation.vllm_cfg.async_engine=true"
        )
    if not master_config.loss_fn.use_importance_sampling_correction:
        raise ValueError(
            "Async PPO requires loss_fn.use_importance_sampling_correction=true"
        )
    if master_config.loss_fn.force_on_policy_ratio:
        raise ValueError("Async PPO does not support force_on_policy_ratio=true")
    colocated_inference = master_config.policy["generation"]["colocated"]["enabled"]
    if colocated_inference:
        raise ValueError("Async PPO requires non-colocated generation")

    async_config = master_config.ppo["async_ppo"]
    max_trajectory_age_steps = async_config.max_trajectory_age_steps
    warmup_max_trajectory_age_steps = (
        async_config.effective_warmup_max_trajectory_age_steps
    )
    policy_training_start_step = master_config.ppo["policy_training_start_step"]
    if master_config.ppo["ppo_epochs"] < 1:
        raise ValueError("ppo.ppo_epochs must be at least 1")
    # Keep launcher restrictions here as defensive checks for direct callers.
    if master_config.ppo["use_dynamic_sampling"]:
        raise NotImplementedError("Dynamic sampling is not supported for async PPO")
    if master_config.ppo["reward_scaling"]["enabled"]:
        raise NotImplementedError("Reward scaling is not supported for async PPO")
    if master_config.ppo["reward_shaping"]["enabled"]:
        raise NotImplementedError("Reward shaping is not supported for async PPO")
    if master_config.data["use_multiple_dataloader"]:
        raise NotImplementedError(
            "Multiple dataloaders are not supported for async PPO"
        )
    if max_trajectory_age_steps > 1:
        print(
            "⚠️ WARNING: max_trajectory_age_steps > 1 increases off-policy "
            "bias in GAE. The validated/recommended value is 1."
        )

    # Import async utilities only when needed (heavy Ray actors).
    from nemo_rl.algorithms.async_utils import AsyncTrajectoryCollector, ReplayBuffer

    timer = Timer(context={"worker": "driver"})
    training_wall_start = time.perf_counter()
    timeout = TimeoutChecker(
        timeout=master_config.checkpointing["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    # PPO async always uses non-colocated vLLM generation, so a refit is always
    # required and the generation engine is a real (non-None) actor.
    assert policy_generation is not None
    if getattr(policy_generation, "requires_kv_scale_sync", False):
        raise NotImplementedError("FP8 KV-scale synchronization is not supported yet")

    if master_config.ppo.get("skip_reference_policy_logprobs_calculation"):
        if master_config.loss_fn.reference_policy_kl_penalty != 0:
            raise ValueError(
                "Skipping reference logprobs requires "
                "loss_fn.reference_policy_kl_penalty=0"
            )

    # ------------------------------------------------------------------
    # Training state. `step` is the global monotonic training step; it is what
    # max_num_steps bounds and what the replay-buffer weight versioning tracks.
    # ------------------------------------------------------------------
    step = ppo_save_state["total_steps"]
    weight_version = step
    consumed_samples = ppo_save_state["consumed_samples"]
    total_valid_tokens = ppo_save_state.get("total_valid_tokens", 0)
    max_num_steps = master_config.ppo["max_num_steps"]
    ppo_epochs = master_config.ppo["ppo_epochs"]
    critic_train_epochs = _resolve_critic_train_epochs(master_config.ppo)
    val_period = master_config.ppo["val_period"]
    val_at_start = master_config.ppo["val_at_start"]
    val_at_end = master_config.ppo["val_at_end"]
    num_prompts_per_step = master_config.ppo["num_prompts_per_step"]
    ft_save_period = master_config.checkpointing.get("ft_save_period")
    max_training_steps = max_num_steps

    replay_buffer: Any = None
    trajectory_collector: Any = None

    def _shutdown_workers(*, propagate_checkpoint_error: bool) -> None:
        """Finalize pending saves and stop async PPO workers."""
        checkpoint_error = None
        try:
            checkpointer.shutdown()
        except Exception as error:
            checkpoint_error = error
            print(f"Error finalizing pending checkpoint: {error}")

        print("🛑 Stopping trajectory collection...")
        for actor, actor_name in (
            (trajectory_collector, "trajectory collector"),
            (replay_buffer, "replay buffer"),
        ):
            if actor is None:
                continue
            try:
                ray.kill(actor)
            except Exception as error:
                print(f"Error stopping {actor_name}: {error}")

        for env_dict in (task_to_env, val_task_to_env):
            if env_dict is None:
                continue
            for task_name, env in env_dict.items():
                print(f"🛑 Shutting down environment {task_name}...")
                try:
                    ray.get(env.shutdown.remote(), timeout=10)
                except Exception:
                    try:
                        ray.kill(env)
                    except Exception as error:
                        print(f"Error shutting down environment {task_name}: {error}")

        print("🛑 Shutting down generation workers...")
        try:
            policy_generation.shutdown()
        except Exception as error:
            print(f"Error shutting down generation workers: {error}")
        if policy is not policy_generation:
            print("🛑 Shutting down policy workers...")
            try:
                policy.shutdown()
            except Exception as error:
                print(f"Error shutting down policy workers: {error}")
        print("🛑 Shutting down value workers...")
        try:
            value_model.shutdown()
        except Exception as error:
            print(f"Error shutting down value workers: {error}")

        if checkpoint_error is not None and propagate_checkpoint_error:
            raise checkpoint_error

    if step >= max_training_steps:
        print(
            f"Training is already complete at step {step} "
            f"(configured limit: {max_training_steps})"
        )
        _shutdown_workers(propagate_checkpoint_error=True)
        return

    adv_estimator = _create_advantage_estimator(master_config)

    # ------------------------------------------------------------------
    # Spin up the replay buffer + trajectory collector Ray actors.
    # ------------------------------------------------------------------
    late_arrival_slack = 2
    buffer_age = max(
        max_trajectory_age_steps,
        warmup_max_trajectory_age_steps,
    )
    optimal_buffer_size = num_prompts_per_step * buffer_age * late_arrival_slack
    print("📊 Async PPO buffer requirements:")
    print(f"   - num_prompts_per_step: {num_prompts_per_step}")
    print(f"   - max_trajectory_age_steps: {max_trajectory_age_steps}")
    print(f"   - warmup_max_trajectory_age_steps: {warmup_max_trajectory_age_steps}")
    print(f"   - optimal_buffer_size: {optimal_buffer_size}")

    replay_buffer = ReplayBuffer.options(
        runtime_env=make_actor_runtime_env(
            "nemo_rl.algorithms.async_utils.ReplayBuffer"
        )
    ).remote(
        max_size=optimal_buffer_size,
        drop_incomplete_targets_on_restore=(
            async_config.drop_incomplete_targets_on_restore
        ),
    )

    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    if last_checkpoint_path is not None:
        replay_buffer_path = os.path.join(last_checkpoint_path, "replay_buffer.pt")
        if os.path.exists(replay_buffer_path):
            print(f"📦 Restoring replay buffer from checkpoint: {replay_buffer_path}")
            # weights_only=False: trajectories are pickled BatchedDataDict/dicts,
            # not plain tensors. The checkpoint is a trusted same-job artifact.
            replay_buffer_state = torch.load(replay_buffer_path, weights_only=False)
            restore_max_age = _async_ppo_buffer_max_age(
                step=step,
                policy_training_start_step=policy_training_start_step,
                max_trajectory_age_steps=max_trajectory_age_steps,
                warmup_max_trajectory_age_steps=warmup_max_trajectory_age_steps,
            )
            ray.get(
                replay_buffer.load_state_dict.remote(
                    replay_buffer_state,
                    num_prompts_per_step=num_prompts_per_step,
                    current_training_step=step,
                    max_age_steps=restore_max_age,
                )
            )
            print("✅ Replay buffer restored from checkpoint")
        else:
            print(
                f"⚠️ No replay buffer checkpoint found at {replay_buffer_path}. "
                "Starting with an empty replay buffer."
            )

    trajectory_collector = AsyncTrajectoryCollector.options(
        runtime_env=make_actor_runtime_env(
            "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector"
        )
    ).remote(
        policy_generation=policy_generation,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        master_config=master_config,
        replay_buffer=replay_buffer,
        start_step=step,
    )

    def _raise_if_collector_stopped(waiting_for: str) -> None:
        status = ray.get(trajectory_collector.get_status.remote())
        if status["errored"]:
            raise RuntimeError(
                f"Trajectory collector failed while {waiting_for}: "
                f"{status.get('error') or status}"
            )
        if (
            not status["running"]
            and status["inflight_workers"] == 0
            and status["data_exhausted"]
        ):
            raise RuntimeError(
                "Trajectory collector exhausted data before the configured "
                f"training limit while {waiting_for}: {status}"
            )

    try:
        # Refit first so resumed runs cannot generate with stale base weights.
        print("⏳ Preparing policy generation for training (initial refit)...")
        refit_policy_generation(policy, policy_generation, colocated_inference)
        policy.offload_to_cpu()

        if val_at_start and step == 0:
            print("\n🔍 Running initial validation...")
            val_metrics, validation_timings = validate(
                policy_generation,
                val_dataloader,
                tokenizer,
                val_task_to_env,
                step=0,
                master_config=master_config,
                logger=logger,
            )
            policy_generation.finish_generation()
            logger.log_metrics(val_metrics, step, prefix="validation")
            logger.log_metrics(validation_timings, step, prefix="timing/validation")

        policy_generation.clear_logger_metrics()

        initial_generation_lead = _async_ppo_generation_lead_steps(
            step=step,
            policy_training_start_step=policy_training_start_step,
            max_trajectory_age_steps=max_trajectory_age_steps,
            warmup_max_trajectory_age_steps=warmup_max_trajectory_age_steps,
        )
        initial_buffer_max_age = _async_ppo_buffer_max_age(
            step=step,
            policy_training_start_step=policy_training_start_step,
            max_trajectory_age_steps=max_trajectory_age_steps,
            warmup_max_trajectory_age_steps=warmup_max_trajectory_age_steps,
        )
        ray.get(
            trajectory_collector.set_generation_window.remote(
                weight_version=weight_version,
                generation_lead_steps=initial_generation_lead,
                max_trajectory_age_steps=initial_buffer_max_age,
            )
        )
        ray.get(
            trajectory_collector.start_collection.remote(_CyclingDataLoader(dataloader))
        )
        print("📦 Started continuous background trajectory collection")

        print(f"⏳ Waiting for replay buffer to be ready for step {step}...")
        timer.start("init/total")
        wait_iterations = 0
        while True:
            current_step_ready = ray.get(
                replay_buffer.has_complete_batch.remote(
                    step, num_prompts_per_step, initial_buffer_max_age
                )
            )
            if current_step_ready:
                # The initial collector is the only window that can generate
                # both `step` and `step + 1`. Fill both before the first refit.
                need_lookahead = step + 1 < max_training_steps
                if need_lookahead:
                    lookahead_step_ready = ray.get(
                        replay_buffer.has_complete_batch.remote(
                            step + 1,
                            num_prompts_per_step,
                            initial_buffer_max_age,
                        )
                    )
                    if not lookahead_step_ready:
                        if wait_iterations % 10 == 0:
                            print(
                                f"  Pipeline barrier: step {step} ready but "
                                f"step {step + 1} not yet — waiting for lookahead fill"
                            )
                        _raise_if_collector_stopped(
                            "waiting for the initial replay lookahead batch"
                        )
                        wait_iterations += 1
                        time.sleep(1.0)
                        continue
                break
            if wait_iterations % 10 == 0:
                buffer_size_current = ray.get(replay_buffer.size.remote())
                print(
                    f"  Wait iteration {wait_iterations}: "
                    f"buffer_size={buffer_size_current}, "
                    f"step {step} ready={current_step_ready}"
                )
            _raise_if_collector_stopped("waiting for the initial replay batch")
            wait_iterations += 1
            time.sleep(1.0)
        timer.stop("init/total")
        print(f"✅ Buffer ready for step {step}! Starting async PPO training loop...")
    except Exception:
        _shutdown_workers(propagate_checkpoint_error=False)
        raise

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    loop_failed = False
    try:
        while step < max_training_steps:
            print(f"\n{'=' * 25} Step {step + 1}/{max_training_steps} {'=' * 25}")
            maybe_gpu_profile_step(policy, step + 1)
            if policy != policy_generation:
                maybe_gpu_profile_step(policy_generation, step + 1)

            metrics: dict[str, Any] = {}
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                # ---- 1. Sample a fixed batch of trajectories from the buffer ----
                print("📦 Sampling from replay buffer...")
                with timer.time("exposed_generation"):
                    current_buffer_max_age = _async_ppo_buffer_max_age(
                        step=step,
                        policy_training_start_step=policy_training_start_step,
                        max_trajectory_age_steps=max_trajectory_age_steps,
                        warmup_max_trajectory_age_steps=(
                            warmup_max_trajectory_age_steps
                        ),
                    )
                    sample_result = ray.get(
                        replay_buffer.sample.remote(
                            num_prompt_groups=num_prompts_per_step,
                            current_weight_version=weight_version,
                            max_age_steps=current_buffer_max_age,
                        )
                    )
                    if (
                        sample_result is None
                        or len(sample_result["trajectories"]) != num_prompts_per_step
                    ):
                        print(
                            "⏳ Buffer empty or not enough groups for a full step, "
                            "waiting..."
                        )
                        _raise_if_collector_stopped(
                            f"waiting for the replay batch for step {step}"
                        )
                        with timer.time("idle/buffer_starvation"):
                            time.sleep(0.5)
                        continue

                    trajectories = sample_result["trajectories"]
                    avg_trajectory_age = sample_result["avg_trajectory_age"]
                    print(
                        f"✅ Sampled {len(trajectories)} trajectory groups "
                        f"(average age: {avg_trajectory_age:.2f} steps)"
                    )

                    per_prompt_batches = [t["batch"] for t in trajectories]
                    repeated_batch = BatchedDataDict.from_batches(per_prompt_batches)

                    per_group_metrics: dict[str, list] = {}
                    for t in trajectories:
                        for k, v in t["rollout_metrics"].items():
                            per_group_metrics.setdefault(k, []).append(v)
                    rollout_metrics = aggregate_rollout_metrics(per_group_metrics)

                expected_batch_size = (
                    master_config.ppo["num_prompts_per_step"]
                    * master_config.ppo["num_generations_per_prompt"]
                )
                cross_trajectory = master_config.ppo["adv_estimator"].get(
                    "cross_trajectory"
                )
                if cross_trajectory:
                    if "trajectory_id" not in repeated_batch:
                        raise RuntimeError(
                            "Cross-trajectory GAE requires trajectory_id metadata"
                        )
                    logical_trajectory_count = len(
                        set(repeated_batch["trajectory_id"])
                    )
                    if logical_trajectory_count != expected_batch_size:
                        raise RuntimeError(
                            "Unexpected logical trajectory count: got "
                            f"{logical_trajectory_count}, expected {expected_batch_size}"
                        )
                    metrics["compaction/logical_trajectories"] = (
                        logical_trajectory_count
                    )
                    metrics["compaction/training_segments"] = repeated_batch.size
                elif repeated_batch.size != expected_batch_size:
                    raise RuntimeError(
                        f"Unexpected training batch size: got {repeated_batch.size}, "
                        f"expected {expected_batch_size}"
                    )

                # ---- 2. Build PPO training data (rewards + inline loss mask) ----
                print("▶ Processing rewards...")
                with timer.time("data_processing"):
                    rewards = repeated_batch["total_reward"]

                    use_overlong_filtering = master_config.ppo["overlong_filtering"]
                    if use_overlong_filtering:
                        loss_multiplier = repeated_batch["loss_multiplier"].clone()
                        truncated = repeated_batch["truncated"]
                        if isinstance(truncated, list):
                            truncated = torch.tensor(truncated, dtype=torch.bool)
                        loss_multiplier[truncated] = 0
                        repeated_batch["loss_multiplier"] = loss_multiplier

                    metrics["num_mask_sample_filtered"] = (
                        _apply_ppo_mask_sample_filter(repeated_batch)
                    )

                    # PPO's inline loss-mask setup (unmask all assistant messages),
                    # matching sync ppo_train — deliberately NOT GRPO's helper,
                    # which only unmasks generated assistant messages.
                    for message_log in repeated_batch["message_log"]:
                        for message in message_log:
                            if message["role"] == "assistant":
                                message["token_loss_mask"] = torch.ones_like(
                                    message["token_ids"]
                                )
                            else:
                                message["token_loss_mask"] = torch.zeros_like(
                                    message["token_ids"]
                                )
                            if "generation_logprobs" not in message:
                                message["generation_logprobs"] = torch.zeros_like(
                                    message["token_ids"], dtype=torch.float32
                                )

                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config.policy[
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    train_data = BatchedDataDict[ClippedPGLossDataDict](
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "generation_logprobs": flat_messages["generation_logprobs"],
                            "rewards": repeated_batch["total_reward"],
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )
                    if cross_trajectory:
                        train_data.update(
                            {
                                "trajectory_id": repeated_batch["trajectory_id"],
                                "segment_index": repeated_batch["segment_index"],
                                "segment_type": repeated_batch["segment_type"],
                                "is_final_segment": repeated_batch["is_final_segment"],
                            }
                        )
                    extra_multimodal_data = flat_messages.get_multimodal_dict(
                        as_tensors=False
                    )
                    train_data.update(extra_multimodal_data)
                    train_data.to("cpu")

                # ---- 3. Value forward (critic on GPU, then offloaded) ----
                # GPU state entering here: policy OFF, value OFF (see refit/step
                # end below). Load value only.
                print("▶ Computing values...")
                with timer.time("value_inference"):
                    value_inference_data = train_data
                    value_inference_padding = 0
                    if cross_trajectory:
                        value_inference_data, value_inference_padding = (
                            _pad_compaction_batch(
                                train_data,
                                batch_multiple=_compaction_batch_multiple(
                                    value_model.sharding_annotations.get_axis_size(
                                        "data_parallel"
                                    ),
                                    master_config.value["train_micro_batch_size"],
                                ),
                                mask_padding=True,
                            )
                        )
                    value_model.prepare_for_inference()
                    values = value_model.get_values(value_inference_data)["values"]
                    value_model.finish_inference()
                    train_data["values"] = values[: train_data.size].squeeze(-1)
                    del values
                    if cross_trajectory:
                        metrics["compaction/value_inference_padding_rows"] = (
                            value_inference_padding
                        )
                        metrics["compaction/value_inference_physical_segments"] = (
                            value_inference_data.size
                        )
                        del value_inference_data

                # ---- 4. Policy / reference logprobs (policy on GPU, then off) ----
                print("▶ Computing logprobs...")
                with timer.time("logprob_inference_prep"):
                    policy.prepare_for_lp_inference()
                with timer.time("policy_and_reference_logprobs"):
                    logprob_data = BatchedDataDict[ClippedPGLossDataDict](
                        {
                            "input_ids": train_data["input_ids"],
                            "input_lengths": train_data["input_lengths"],
                            **extra_multimodal_data,
                        }
                    )
                    logprob_inference_data = logprob_data
                    logprob_inference_padding = 0
                    if cross_trajectory:
                        logprob_inference_data, logprob_inference_padding = (
                            _pad_compaction_batch(
                                logprob_data,
                                batch_multiple=_compaction_batch_multiple(
                                    policy.data_parallel_size,
                                    master_config.policy["logprob_batch_size"],
                                ),
                                mask_padding=False,
                            )
                        )
                    prev_logprobs = policy.get_logprobs(
                        logprob_inference_data, timer=timer
                    )["logprobs"]
                    train_data["prev_logprobs"] = prev_logprobs[: train_data.size]
                    del prev_logprobs
                    if not master_config.ppo.get(
                        "skip_reference_policy_logprobs_calculation"
                    ):
                        reference_logprobs = policy.get_reference_policy_logprobs(
                            logprob_inference_data,
                            timer=timer,
                        )["reference_logprobs"]
                        train_data["reference_policy_logprobs"] = (
                            reference_logprobs[: train_data.size]
                        )
                        del reference_logprobs
                    if cross_trajectory:
                        metrics["compaction/policy_logprob_padding_rows"] = (
                            logprob_inference_padding
                        )
                        metrics["compaction/policy_logprob_physical_segments"] = (
                            logprob_inference_data.size
                        )
                        del logprob_inference_data
                    del logprob_data
                    del extra_multimodal_data
                    policy.finish_inference()

                # ---- 5. Sequence-level train/inference mismatch diagnostics ----
                (
                    advantage_mask,
                    seq_logprob_error_metrics,
                ) = _apply_ppo_seq_logprob_error_masking(
                    train_data=train_data,
                    rewards=rewards,
                    seq_logprob_error_threshold=master_config.ppo[
                        "seq_logprob_error_threshold"
                    ],
                )

                # ---- 6. GAE advantages/returns (uses fresh values) ----
                with timer.time("advantage_calculation"):
                    print("▶ Computing advantages...")
                    initial_prompt_message_logs = extract_initial_prompt_messages(
                        repeated_batch["message_log"],
                        repeated_batch["length"],
                    )
                    prompt_batched_flat, _ = batched_message_log_to_flat_message(
                        initial_prompt_message_logs,
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    )
                    prompt_ids_for_adv = prompt_batched_flat["token_ids"]
                    del initial_prompt_message_logs
                    del prompt_batched_flat

                    adv_kwargs = dict(
                        prompt_ids=prompt_ids_for_adv,
                        rewards=train_data["rewards"],
                        mask=advantage_mask,
                        reference_logprobs=train_data.get("reference_policy_logprobs"),
                        logprobs=train_data["prev_logprobs"],
                    )
                    if "values" in train_data:
                        adv_kwargs["values"] = train_data["values"]
                    if cross_trajectory:
                        (
                            policy_segment_discounts,
                            value_segment_discounts,
                            policy_gae_lambdas,
                        ) = adv_estimator.build_cross_trajectory_discounts(
                            mask=advantage_mask,
                            trajectory_ids=train_data["trajectory_id"],
                            segment_indices=train_data["segment_index"],
                        )
                        adv_kwargs.update(
                            {
                                "policy_segment_discounts": policy_segment_discounts,
                                "value_segment_discounts": value_segment_discounts,
                                "policy_gae_lambdas": policy_gae_lambdas,
                            }
                        )
                        valid_segment_discounts = policy_segment_discounts[
                            train_data["sample_mask"].bool()
                        ]
                        if valid_segment_discounts.numel() > 0:
                            metrics.update(
                                {
                                    "compaction/cross_gae_discount_mean": (
                                        valid_segment_discounts.mean().item()
                                    ),
                                    "compaction/cross_gae_discount_min": (
                                        valid_segment_discounts.min().item()
                                    ),
                                    "compaction/trajectory_lambda_mean": (
                                        policy_gae_lambdas.mean().item()
                                    ),
                                }
                            )
                    result = adv_estimator.compute_advantage(**adv_kwargs)
                    if isinstance(result, tuple):
                        advantages, returns = result
                    else:
                        advantages, returns = result, None
                    del prompt_ids_for_adv
                    train_data["advantages"] = advantages
                    if returns is not None:
                        train_data["returns"] = returns

                metrics.update(
                    _compute_rollout_critic_metrics(train_data, advantage_mask)
                )

                # ---- 7. Independent critic and actor update loops ----
                # During warmup (step < policy_training_start_step) the policy is
                # frozen: it is never loaded/trained here, exactly as in sync
                # ppo_train, so train_results stays None for the step.
                is_policy_training_step = step >= policy_training_start_step
                actor_train_data = train_data
                critic_train_data = train_data
                actor_gbs = None
                critic_gbs = None
                if cross_trajectory:
                    critic_train_data, critic_padding = _pad_compaction_batch(
                        train_data,
                        batch_multiple=_compaction_batch_multiple(
                            value_model.sharding_annotations.get_axis_size(
                                "data_parallel"
                            ),
                            master_config.value["train_micro_batch_size"],
                        ),
                        mask_padding=True,
                    )
                    critic_gbs = critic_train_data.size
                    metrics["compaction/critic_training_padding_rows"] = (
                        critic_padding
                    )
                    metrics["compaction/critic_training_physical_segments"] = (
                        critic_gbs
                    )
                    if is_policy_training_step:
                        actor_train_data, actor_padding = _pad_compaction_batch(
                            train_data,
                            batch_multiple=_compaction_batch_multiple(
                                policy.data_parallel_size,
                                master_config.policy["train_micro_batch_size"],
                            ),
                            mask_padding=True,
                        )
                        actor_gbs = actor_train_data.size
                        metrics["compaction/actor_training_padding_rows"] = (
                            actor_padding
                        )
                        metrics[
                            "compaction/actor_training_physical_segments"
                        ] = actor_gbs
                train_results = None
                value_results = None
                for critic_epoch in range(critic_train_epochs):
                    print(
                        f"▶ Critic update {critic_epoch + 1}/{critic_train_epochs}..."
                    )
                    with timer.time("value_training_prep"):
                        value_model.prepare_for_training()
                    with timer.time("value_training"):
                        value_results = value_model.train(
                            critic_train_data,
                            value_loss_fn,
                            timer=timer,
                            gbs=critic_gbs,
                            scheduler_increment=(
                                master_config.value["train_global_batch_size"]
                                if cross_trajectory
                                else None
                            ),
                        )
                        value_model.finish_training()

                if is_policy_training_step:
                    if (
                        step == policy_training_start_step
                        and policy_training_start_step > 0
                    ):
                        print(
                            f"  ✓ Critic warmup complete ({policy_training_start_step} "
                            "steps). Starting policy training.",
                            flush=True,
                        )
                    for actor_epoch in range(ppo_epochs):
                        print(f"▶ Actor update {actor_epoch + 1}/{ppo_epochs}...")
                        with timer.time("training_prep"):
                            policy.prepare_for_training()
                        with timer.time("policy_training"):
                            train_results = policy.train(
                                actor_train_data,
                                loss_fn,
                                timer=timer,
                                gbs=actor_gbs,
                                scheduler_increment=(
                                    master_config.policy["train_global_batch_size"]
                                    if cross_trajectory
                                    else None
                                ),
                            )
                            if actor_epoch < ppo_epochs - 1:
                                policy.offload_to_cpu()

                if cross_trajectory:
                    del critic_train_data
                    if is_policy_training_step:
                        del actor_train_data

                # ---- 8. Refit once after all PPO epochs ----
                # Warmup still advances the replay-buffer version, but skips the
                # transfer because the policy has not changed.
                generation_logger_metrics = None
                print("🔄 Coordinating with trajectory collector before refit...")
                next_weight_version = weight_version + 1
                with timer.time("idle/refit_bubble"):
                    with timer.time("exposed_generation"):
                        ray.get(trajectory_collector.prepare_for_refit.remote())
                    generation_logger_metrics = policy_generation.get_logger_metrics()
                    with timer.time("weight_sync"):
                        if is_policy_training_step:
                            refit_policy_generation(
                                policy, policy_generation, colocated_inference
                            )
                        else:
                            print(
                                "▶ Critic warmup: skipping policy weight transfer "
                                "(policy frozen; generation already up to date)"
                            )
                        weight_version = next_weight_version
                        next_generation_lead = _async_ppo_generation_lead_steps(
                            step=weight_version,
                            policy_training_start_step=policy_training_start_step,
                            max_trajectory_age_steps=max_trajectory_age_steps,
                            warmup_max_trajectory_age_steps=(
                                warmup_max_trajectory_age_steps
                            ),
                        )
                        next_buffer_max_age = _async_ppo_buffer_max_age(
                            step=weight_version,
                            policy_training_start_step=policy_training_start_step,
                            max_trajectory_age_steps=max_trajectory_age_steps,
                            warmup_max_trajectory_age_steps=(
                                warmup_max_trajectory_age_steps
                            ),
                        )
                        ray.get(
                            trajectory_collector.set_generation_window.remote(
                                weight_version=weight_version,
                                generation_lead_steps=next_generation_lead,
                                max_trajectory_age_steps=next_buffer_max_age,
                            )
                        )
                        ray.get(trajectory_collector.resume_after_refit.remote())
                # Only the policy-training path leaves the policy resident on GPU;
                # during warmup it is already offloaded, so skip the redundant call.
                if is_policy_training_step:
                    policy.offload_to_cpu()

                policy_generation.clear_logger_metrics()

                # ---- Validation ----
                is_last_step = step + 1 == max_training_steps
                if (val_period > 0 and (step + 1) % val_period == 0) or (
                    val_at_end and is_last_step
                ):
                    with timer.time("idle/validation"):
                        ray.get(trajectory_collector.pause.remote())
                        ray.get(
                            trajectory_collector.wait_for_pending_generations.remote()
                        )
                        # Policy weights were synced by the refit above.
                        policy_generation.prepare_for_generation()
                        val_metrics, validation_timings = validate(
                            policy_generation,
                            val_dataloader,
                            tokenizer,
                            val_task_to_env,
                            step=step + 1,
                            master_config=master_config,
                            logger=logger,
                        )
                        policy_generation.finish_generation()
                        logger.log_metrics(
                            validation_timings, step + 1, prefix="timing/validation"
                        )
                        logger.log_metrics(val_metrics, step + 1, prefix="validation")
                        gc.collect()
                        torch.cuda.empty_cache()
                        ray.get(trajectory_collector.resume.remote())

                # ---- Metrics ----
                flat_advantages = train_data["advantages"]
                flat_messages_content = flat_messages.get("content", [])
                del flat_messages

                metrics.update(
                    {
                        "reward": rewards.numpy(),
                        "mean_prompt_length": repeated_batch["length"].numpy(),
                        "total_num_tokens": input_lengths.numpy(),
                    }
                )
                metrics.update(
                    _masked_distribution_metrics(
                        "advantages", flat_advantages, advantage_mask
                    )
                )
                # Policy metrics are absent during critic warmup (train_results is
                # None because the policy was not trained this step).
                if train_results is not None:
                    metrics.update(
                        _compute_actor_metrics(
                            train_results,
                            master_config.loss_fn.reference_policy_kl_penalty,
                        )
                    )
                    if "moe_metrics" in train_results:
                        metrics.update(
                            {
                                f"moe/{k}": v
                                for k, v in train_results["moe_metrics"].items()
                            }
                        )
                if value_results is not None:
                    metrics.update(_compute_critic_metrics(value_results))
                metrics["actor/update_applied"] = float(train_results is not None)
                metrics["critic/update_applied"] = float(value_results is not None)
                metrics["actor/num_updates"] = (
                    float(ppo_epochs) if train_results is not None else 0.0
                )
                metrics["critic/num_updates"] = float(critic_train_epochs)

                for k, v in metrics.items():
                    if k in {
                        "reward",
                        "global_valid_seqs",
                        "global_valid_toks",
                        "mean_prompt_length",
                    }:
                        metrics[k] = np.mean(v).item()
                    elif isinstance(v, (np.ndarray, list)):
                        metrics[k] = np.sum(v).item()

                metrics.update(rollout_metrics)
                if generation_logger_metrics is not None:
                    metrics["generation_logger_metrics"] = generation_logger_metrics
                if "global_valid_toks" in metrics:
                    total_valid_tokens += metrics["global_valid_toks"]
                # Always log seq-level error metrics (useful for tuning threshold).
                metrics.update(seq_logprob_error_metrics)

                # ---- Checkpointing ----
                consumed_samples += master_config.ppo["num_prompts_per_step"]
                timeout.mark_iteration()
                should_save_by_step = (
                    is_last_step
                    or (step + 1) % master_config.checkpointing["save_period"] == 0
                    or (ft_save_period is not None and (step + 1) % ft_save_period == 0)
                )
                should_save_by_timeout = timeout.check_save()
                if master_config.checkpointing["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    ppo_save_state["current_step"] = step + 1
                    ppo_save_state["total_steps"] = step + 1
                    ppo_save_state["total_valid_tokens"] = total_valid_tokens
                    if val_metrics is not None:
                        ppo_save_state["val_reward"] = val_metrics["accuracy"]
                    elif "val_reward" in ppo_save_state:
                        del ppo_save_state["val_reward"]
                    ppo_save_state["consumed_samples"] = consumed_samples

                    # Record the top-k ranking metric into the save state so
                    # get_best_checkpoint_path / top-k pruning work (parity with
                    # sync ppo_train and async_grpo_train).
                    full_metric_name = master_config.checkpointing["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:',\n"
                            f'followed by the corresponding name in the "val" or "train" metrics dictionary.'
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. "
                                "This checkpoint will not be saved as top-k.",
                                stacklevel=2,
                            )
                            if full_metric_name in ppo_save_state:
                                del ppo_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            ppo_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

                    with timer.time("checkpointing"):
                        print(f"Saving checkpoint for step {step + 1}...")
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            step + 1, ppo_save_state, master_config
                        )
                        policy.prepare_for_training()
                        policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=(
                                os.path.join(checkpoint_path, "policy", "optimizer")
                                if (
                                    checkpointer.save_optimizer
                                    and step >= policy_training_start_step
                                )
                                else None
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config.checkpointing,
                        )
                        policy.offload_to_cpu()

                        value_model.prepare_for_training()
                        value_model.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "value", "weights"
                            ),
                            optimizer_path=(
                                os.path.join(checkpoint_path, "value", "optimizer")
                                if checkpointer.save_optimizer
                                else None
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "value", "tokenizer"
                            ),
                            checkpointing_cfg=master_config.checkpointing,
                        )
                        value_model.finish_training()

                        dataloader_state = ray.get(
                            trajectory_collector.get_dataloader_state.remote()
                        )
                        torch.save(
                            dataloader_state,
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        print("📦 Saving replay buffer state...")
                        replay_buffer_state = ray.get(replay_buffer.state_dict.remote())
                        torch.save(
                            replay_buffer_state,
                            os.path.join(checkpoint_path, "replay_buffer.pt"),
                        )
                        checkpointer.begin_finalization(
                            checkpoint_path,
                            wait_fn=policy.finalize_async_save,
                        )

            # ---- Logging ----
            log_data = {
                "content": flat_messages_content,
                "rewards": rewards.tolist(),
                "input_lengths": input_lengths.tolist(),
                "token_ids": train_data["input_ids"].tolist(),
                "token_loss_mask": train_data["token_mask"].tolist(),
                "sample_loss_mask": train_data["sample_mask"].tolist(),
                "advantages": train_data["advantages"].tolist(),
                "generation_logprobs": train_data[
                    "generation_logprobs"
                ].tolist(),
                "prev_logprobs": train_data["prev_logprobs"].tolist(),
            }
            logger.log_batched_dict_as_jsonl(
                log_data, f"train_data_step{step + 1}.jsonl"
            )
            del log_data
            del flat_messages_content

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )  # type: ignore

            buffer_size_current = ray.get(replay_buffer.size.remote())
            metrics["buffer_size"] = buffer_size_current
            metrics["avg_trajectory_age"] = avg_trajectory_age

            # Track the worst-mismatch example plot (parity with sync PPO).
            if metrics.get("token_mult_prob_error", 0) > 1.05:
                logger.log_plot_token_mult_prob_error(
                    {
                        "prompt_lengths": repeated_batch["length"],
                        "full_lengths": input_lengths,
                        "generation_logprobs": train_data["generation_logprobs"],
                        "prev_logprobs": train_data["prev_logprobs"],
                        "token_mask": train_data["token_mask"],
                        "sample_mask": train_data["sample_mask"],
                    },
                    step + 1,
                    name="train/token_mult_prob_error_plot_sample",
                )
            del train_data

            print("\n📊 Training Results:")
            if "actor/total_loss" in metrics:
                print(f"  • Actor Total Loss: {metrics['actor/total_loss']:.4f}")
                print(
                    f"  • Generation KL Error: "
                    f"{metrics.get('actor/gen_kl_error', 'N/A')}"
                )
            else:
                print("  • (critic warmup: policy not trained this step)")
            if "critic/loss" in metrics:
                print(f"  • Critic Loss: {metrics['critic/loss']:.4f}")
            print(f"  • Avg Reward: {np.mean(rewards.numpy()):.4f}")
            print(f"  • Buffer Size: {buffer_size_current}")
            print(
                f"  • Avg Trajectory Age (gen-version): {avg_trajectory_age:.2f} steps"
            )

            total_time = timing_metrics.get("total_step_time", 0)
            total_num_gpus = (
                master_config.cluster["num_nodes"]
                * master_config.cluster["gpus_per_node"]
            )
            if total_time > 0 and "global_valid_toks" in metrics:
                timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                    metrics["global_valid_toks"] / total_time / total_num_gpus
                )
            performance_metrics = print_performance_metrics(
                train_results if train_results is not None else (value_results or {}),
                metrics,
                timing_metrics,
                master_config,
            )

            collector_efficiency = ray.get(
                trajectory_collector.get_efficiency_metrics.remote()
            )
            driver_efficiency = {
                cat: timer.reduce(cat, "sum")
                for cat in [
                    "init/total",
                    "idle/buffer_starvation",
                    "idle/refit_bubble",
                    "idle/validation",
                ]
                if cat in timer._timers
            }
            merged_efficiency = {**driver_efficiency}
            for cat, dur in collector_efficiency.items():
                merged_efficiency[cat] = merged_efficiency.get(cat, 0.0) + dur
            total_wall_time = time.perf_counter() - training_wall_start
            efficiency_loggable = print_efficiency_summary(
                merged_efficiency, total_wall_time, step + 1
            )

            logger.log_metrics(performance_metrics, step + 1, prefix="performance")
            logger.log_metrics(metrics, step + 1, prefix="train")
            logger.log_metrics(efficiency_loggable, step + 1, prefix="")
            logger.log_metrics(
                timing_metrics,
                step + 1,
                prefix="timing/train",
                step_finished=True,
            )

            timer.reset()
            step += 1
            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if step >= max_training_steps:
                print(
                    "Configured step/epoch limit has been reached, stopping training",
                    flush=True,
                )
                return

    except Exception as e:
        loop_failed = True
        print(f"❌ Error in async PPO loop: {e}")
        traceback.print_exc()
        raise

    finally:
        _shutdown_workers(propagate_checkpoint_error=not loop_failed)
        print("Async PPO training complete!")


def validate(
    policy_generation: GenerationInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    step: int,
    master_config: MasterConfig,
    logger: Optional[Logger] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        assert val_dataloader is not None or master_config.ppo["val_period"] == 0, (
            "val_dataloader is None, so ppo.val_period must be 0"
        )
        print("  ⚠️ No validation dataloader provided, skipping validation", flush=True)
        return {}, {}

    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...", flush=True)

        total_rewards = []
        total_lengths = []
        all_message_logs = []  # Collect all message logs

        max_batches = (
            master_config.ppo["max_val_samples"] // master_config.ppo["val_batch_size"]
        )
        for batch_idx, val_batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break

            additional_metrics_to_report = dict()

            # NeMo-Gym owns prompt construction and the complete agent rollout.
            # Its dataset message_log is only an empty compatibility placeholder,
            # so it must be dispatched before the generic async rollout path.
            if _should_use_nemo_gym(master_config):
                generation_config = master_config.policy["generation"]
                nemo_gym_rollout_result = run_nemo_gym_rollout_sync(
                    policy_generation=policy_generation,
                    input_batch=val_batch,
                    tokenizer=tokenizer,
                    task_to_env=val_task_to_env,
                    max_seq_len=master_config.policy["max_total_sequence_length"],
                    generation_config=generation_config,
                    log_full_result_tables=should_log_nemo_gym_full_result_tables(
                        wandb_enabled=master_config.logger["wandb_enabled"],
                        wandb_config=master_config.logger["wandb"],
                    ),
                    max_rollout_turns=None,
                    greedy=False,
                    reward_penalty_config=master_config.reward_penalties,
                    thinking_tags=get_nemo_gym_thinking_tags(master_config.env),
                )
                val_batch = nemo_gym_rollout_result.final_batch
                gen_metrics = nemo_gym_rollout_result.rollout_metrics
                additional_metrics_to_report = gen_metrics
                if "is_final_segment" in val_batch:
                    final_segment_indices = (
                        val_batch["is_final_segment"].nonzero(as_tuple=False).flatten()
                    )
                    val_batch = val_batch.select_indices(final_segment_indices)
            else:
                rollout_fn = (
                    run_async_multi_turn_rollout
                    if _should_use_async_rollouts(master_config)
                    else run_multi_turn_rollout
                )
                val_batch, gen_metrics = rollout_fn(
                    policy_generation=policy_generation,
                    input_batch=val_batch,
                    tokenizer=tokenizer,
                    task_to_env=val_task_to_env,
                    max_seq_len=master_config.policy["max_total_sequence_length"],
                    max_rollout_turns=master_config.ppo["max_rollout_turns"],
                    greedy=False,
                )

            total_rewards.extend(val_batch["total_reward"].tolist())
            total_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])

            # Collect message logs for later display
            to_env = [
                get_keys_from_message_log(
                    val_batch["message_log"][i], ["role", "content"]
                )
                for i in range(len(val_batch["message_log"]))
            ]

            all_message_logs.extend(to_env)

        # Calculate validation metrics
        num_samples = len(total_rewards)
        if num_samples > 0:
            rewards_t = torch.tensor(total_rewards, dtype=torch.float32)
            accuracy = rewards_t.mean().item()
        else:
            accuracy = 0.0

        avg_length = (
            sum(total_lengths) / len(total_lengths) if len(total_lengths) > 0 else 0.0
        )

        val_metrics = {
            "accuracy": accuracy,
            "avg_length": avg_length,
            **additional_metrics_to_report,
        }

        # Print sample conversations only once at the end of validation
        try:
            print_message_log_samples(
                all_message_logs,
                total_rewards,
                num_samples=min(
                    master_config.logger["num_val_samples_to_print"],
                    len(all_message_logs),
                ),
                step=step,
            )
        except Exception as e:
            print(f"\n  ⚠️ Error displaying message samples: {str(e)}")
            print("  ⚠️ Continuing validation without displaying samples...", flush=True)

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    # Print summary of validation results
    print("\n📊 Validation Results:")
    print(f"    • Accuracy: {accuracy:.4f}")
    print(f"    • Average response length: {avg_length:.1f} tokens")
    print(f"    • Samples processed: {len(total_rewards)}", flush=True)

    # Print timing information
    print("\n  ⏱️  Validation Timing:")
    print(f"    • Total validation time: {validation_time:.2f}s", flush=True)

    # Log validation data to JSONL file
    if logger is not None:
        val_log_data = {
            "content": all_message_logs,
            "rewards": total_rewards,
        }
        logger.log_batched_dict_as_jsonl(val_log_data, f"val_data_step{step}.jsonl")

    # Make sure to reset the timer after validation
    timer.reset()

    # Explicit GPU memory cleanup after validation
    gc.collect()
    torch.cuda.empty_cache()

    return val_metrics, timing_metrics
