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
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import (
    ClippedPGLossConfig,
    ClippedPGLossDataDict,
    ClippedPGLossFn,
)
from nemo_rl.algorithms.reward_functions import (
    RewardShapingConfig,
    apply_reward_shaping,
)
from nemo_rl.algorithms.utils import (
    calculate_baseline_and_std_per_prompt,
    log_generation_metrics_to_wandb,
    print_performance_metrics,
    set_seed,
)
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.dataloader import MultipleDataloaderWrapper
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_async_nemo_gym_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
    print_message_log_samples,
)
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer
from nemo_rl.utils.venvs import create_local_venv_on_each_node

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


class RewardScalingConfig(TypedDict):
    """Configure linear reward scaling with clamping.

    When `enabled` is True, each reward is clamped to the source interval
    [source_min, source_max] and linearly mapped to the target interval
    [target_min, target_max]. Refer to the scale_rewards function for the implementation.

    Defaults:
        source_min=0.0, source_max=1.0, target_min=0.0, target_max=1.0
    """

    enabled: bool
    source_min: NotRequired[float]
    source_max: NotRequired[float]
    target_min: NotRequired[float]
    target_max: NotRequired[float]


class AsyncGRPOConfig(TypedDict):
    enabled: bool
    # Maximum trajectory age in training steps for samples drawn from the
    # async replay buffer. Trajectories older than this are excluded during
    # sampling; buffer sizing also scales with this value.
    max_trajectory_age_steps: int
    # Does the weight synchronization as soon as the training is done
    # without waiting for the pending generations to finish.
    in_flight_weight_updates: NotRequired[bool]
    # Recomputes the KV cache after the in-flight weight updates.
    recompute_kv_cache_after_weight_updates: NotRequired[bool]


class GRPOConfig(TypedDict):
    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_num_epochs: int
    max_num_steps: int
    max_rollout_turns: int
    normalize_rewards: bool
    use_leave_one_out_baseline: bool
    val_period: int
    val_batch_size: int
    val_at_start: bool
    max_val_samples: int
    seed: int
    async_grpo: NotRequired[AsyncGRPOConfig]
    overlong_filtering: NotRequired[bool]
    # whether to enable dynamic sampling, i.e.
    # whether to discard prompts whose rewards have zero standard deviation
    use_dynamic_sampling: bool
    # When using dynamic sampling, the maximum number of batches to generate
    # before throwing an error
    dynamic_sampling_max_gen_batches: NotRequired[int]
    # When using dynamic sampling, generation prompt batch size will equal
    # num_prompts_per_step * batch_multiplier
    batch_multiplier: NotRequired[float]
    reward_shaping: RewardShapingConfig
    reward_scaling: RewardScalingConfig


class GRPOSaveState(TypedDict):
    consumed_samples: int
    current_step: int
    current_epoch: int
    total_steps: int
    total_valid_tokens: int  # Track total number of non-padding tokens during training
    val_reward: NotRequired[
        float
    ]  # Optional field - may not be present during training


def _default_grpo_save_state() -> GRPOSaveState:
    return {
        "consumed_samples": 0,
        "current_step": 0,
        "current_epoch": 0,
        "total_steps": 0,
        "total_valid_tokens": 0,
        "val_reward": -99999999.0,
    }


class GRPOLoggerConfig(LoggerConfig):
    num_val_samples_to_print: int  # number of val samples to print to stdout


class MasterConfig(TypedDict):
    policy: PolicyConfig
    loss_fn: ClippedPGLossConfig
    env: dict[str, Any]
    data: DataConfig
    grpo: GRPOConfig
    logger: GRPOLoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# ===============================================================================
# Setup & Initialization
# ===============================================================================


def setup(
    master_config: MasterConfig,
    tokenizer: TokenizerType,
    datasets: dict[str, AllTaskProcessedDataset],
    val_dataset: Optional[AllTaskProcessedDataset],
    processor: Optional[AutoProcessor] = None,
) -> tuple[
    ColocatablePolicyInterface,
    Optional[GenerationInterface],
    tuple[RayVirtualCluster, RayVirtualCluster],
    dict[str, StatefulDataLoader],
    Optional[StatefulDataLoader],
    ClippedPGLossFn,
    Logger,
    CheckpointManager,
    GRPOSaveState,
    MasterConfig,
]:
    """Main entry point for running GRPO algorithm.

    Returns:
        tuple of policy, cluster, dataloader, tokenizer, loss_fn, math_env, logger, master_config, val_dataloader
    """
    # Start timing the entire setup process
    setup_start_time = time.perf_counter()

    # Extract individual configs for easier access
    policy_config = master_config["policy"]
    generation_config = master_config["policy"]["generation"]
    env_configs = master_config["env"]
    loss_config = master_config["loss_fn"]
    grpo_config = master_config["grpo"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

    assert generation_config is not None, (
        "A generation config in the PolicyConfig is required for GRPO"
    )

    # Set seed for all random number generators
    set_seed(grpo_config["seed"])

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    grpo_save_state: Optional[GRPOSaveState] = cast(
        Optional[GRPOSaveState], checkpointer.load_training_info(last_checkpoint_path)
    )
    if grpo_save_state is None:
        grpo_save_state = _default_grpo_save_state()

    # ==========================
    #           Data
    # ==========================
    # Validate dataloader batch size
    dataloader_batch_size = data_config["num_prompts_per_dataloader"]
    if not data_config["use_multiple_dataloader"]:
        assert dataloader_batch_size == grpo_config["num_prompts_per_step"], (
            "data.num_prompts_per_dataloader must be equal to grpo.num_prompts_per_step if not using multiple dataloaders (data.use_multiple_dataloader=false)"
        )

    batch_multiplier = grpo_config["batch_multiplier"]
    if not grpo_config["use_dynamic_sampling"]:
        assert batch_multiplier == 1, (
            "batch_multiplier>1 can only be used if use_dynamic_sampling=True"
        )
    else:
        dataloader_batch_size = int(dataloader_batch_size * batch_multiplier)

    # Load train dataset
    dataloaders: dict[str, StatefulDataLoader] = {}
    for task_name, dataset in datasets.items():
        dataloaders[task_name] = StatefulDataLoader(
            dataset,
            batch_size=dataloader_batch_size,
            shuffle=data_config["shuffle"],
            collate_fn=rl_collate_fn,
            drop_last=True,
            num_workers=data_config["num_workers"],
        )
        if last_checkpoint_path is not None:
            dataloader_state_dict = torch.load(
                os.path.join(last_checkpoint_path, f"train_dataloader_{task_name}.pt")
            )
            dataloaders[task_name].load_state_dict(dataloader_state_dict)

    train_sample_count = sum(len(dataloader) for dataloader in dataloaders.values())
    print(
        f"  ✓ Training dataloader loaded with {train_sample_count} samples", flush=True
    )

    # Load validation dataset if provided
    val_dataloader: Optional[StatefulDataLoader] = None
    # If validation is enabled, load the validation dataloader
    if grpo_config["val_period"] > 0 or grpo_config["val_at_start"]:
        assert val_dataset is not None, (
            "Validation dataset is required if validation is enabled"
        )
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=grpo_config["val_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
            num_workers=data_config["num_workers"],
        )
        print(
            f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples",
            flush=True,
        )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...", flush=True)
    colocated_inference = generation_config["colocated"]["enabled"]
    reward_model_enabled = (
        "env_name" in data_config and data_config["env_name"] == "reward_model"
    )

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
            name="grpo_policy_cluster",
            bundle_ct_per_node_list=[policy_gpus_per_node] * policy_nodes,
            use_gpus=True,
            num_gpus_per_node=policy_gpus_per_node,
            max_colocated_worker_groups=1
            if generation_config["backend"] == "megatron"
            else 2,
        )
        train_cluster = cluster
        inference_cluster = cluster
        print(
            f"  ✓ Ray cluster for policy initialized with {policy_nodes} nodes",
            flush=True,
        )

    else:
        assert generation_config["backend"] != "megatron", (
            "Non-colocated inference is not supported for Megatron generation backends. "
            "Please use vLLM backend for generation."
        )

        # train resources will be updated through overall and inference resources below
        train_gpus_per_node = cluster_config["gpus_per_node"]
        train_nodes = policy_nodes

        inference_resources = generation_config["colocated"]["resources"]
        inference_gpus_per_node = inference_resources["gpus_per_node"]
        inference_nodes = inference_resources["num_nodes"]

        # validate and configure resources
        if policy_nodes == 1:
            # When policy_nodes == 1, train and inference are on the same node
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
            # If total_nodes == 1, reward model is also on the same node; otherwise it's on a different node
            reward_gpus_to_subtract = (
                rm_gpus_per_node if total_nodes == 1 and reward_model_enabled else 0
            )
            train_gpus_per_node -= inference_gpus_per_node + reward_gpus_to_subtract
            assert train_gpus_per_node > 0, (
                "No enough GPUs for training, "
                f"train_gpus_per_node:{train_gpus_per_node} = cluster_config['gpus_per_node']:{cluster_config['gpus_per_node']} - inference_gpus_per_node:{inference_gpus_per_node}"
                + (
                    f" - rm_gpus_per_node:{rm_gpus_per_node}"
                    if total_nodes == 1 and reward_model_enabled
                    else ""
                )
            )
        else:
            # train, inference, and reward model are all on different nodes
            assert inference_nodes > 0, (
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
                f"but got inference_gpus_per_node={inference_gpus_per_node}, cluster.gpus_per_node={cluster_config['gpus_per_node']}."
            )
            train_nodes -= inference_nodes

        # initialize train cluster
        train_cluster = RayVirtualCluster(
            name="grpo_train_cluster",
            bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
            use_gpus=True,
            num_gpus_per_node=train_gpus_per_node,
            max_colocated_worker_groups=1,
        )
        print(
            f"  ✓ Ray train cluster initialized with {train_nodes} nodes with {train_gpus_per_node} GPUs per node",
            flush=True,
        )

        # initialize inference cluster
        inference_cluster = RayVirtualCluster(
            name="grpo_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=1,
        )
        print(
            f"  ✓ Ray inference cluster initialized with {inference_nodes} nodes with {inference_gpus_per_node} GPUs per node",
            flush=True,
        )

    # ==========================
    #   Training and Inference
    # ==========================
    print("\n▶ Setting up model and training...", flush=True)

    # vllm model loading prefers clean environment, initialize policy_generation before policy in colocated mode
    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]  # Needed for vLLM

    # Dictionary to store worker initialization timing stats for logging
    worker_init_timing_metrics = {}

    # Prepare checkpoint paths
    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
    else:
        weights_path = None
        optimizer_path = None

    if policy_config.get("megatron_cfg", {}).get("enabled", False):
        ## NOTE: this is equal to the total number of scheduler steps
        total_train_iters = min(
            grpo_config["max_num_steps"],
            grpo_config["max_num_epochs"] * train_sample_count,
        )
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters

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

    def init_vllm():
        """Initialize vLLM generation workers."""
        t0 = time.perf_counter()
        pg = VllmGeneration(cluster=inference_cluster, config=generation_config)
        pg.finish_generation()
        return pg, time.perf_counter() - t0

    # Handle backend-specific setup
    if backend == "megatron":
        # Megatron backend: policy_generation is None, only initialize policy
        policy_generation = None
        print(
            f"  ✓ Using {backend} backend for generation with {policy_config['model_name']}",
            flush=True,
        )

        policy, policy_time = init_policy()
        worker_init_timing_metrics["policy_init_time_s"] = policy_time

    elif backend == "vllm":
        # vLLM backend: setup config, then decide parallel vs sequential init
        generation_config = cast(VllmConfig, generation_config)
        if generation_config["vllm_cfg"]["precision"] == "fp8":
            assert loss_config["use_importance_sampling_correction"] is True, (
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
        generation_config["vllm_cfg"]["hf_overrides"] = policy_config.get(
            "hf_config_overrides", {}
        )

        # Determine if parallel initialization is possible (non-colocated mode)
        use_parallel_init = not colocated_inference

        if use_parallel_init:
            # Parallel initialization: vLLM and Policy can initialize simultaneously
            print(
                "  ⚡ Using parallel worker initialization (non-colocated mode)",
                flush=True,
            )

            # Execute both initializations in parallel
            parallel_start_time = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as executor:
                vllm_future = executor.submit(init_vllm)
                policy_future = executor.submit(init_policy)
                policy_generation, vllm_time = vllm_future.result()
                policy, policy_time = policy_future.result()
            parallel_wall_time = time.perf_counter() - parallel_start_time

            # Store timing metrics
            worker_init_timing_metrics["vllm_init_time_s"] = vllm_time
            worker_init_timing_metrics["policy_init_time_s"] = policy_time
            worker_init_timing_metrics["parallel_wall_time_s"] = parallel_wall_time
            worker_init_timing_metrics["parallel_init_enabled"] = True

        else:
            # Sequential initialization: colocated mode (GPU memory requires vLLM first)
            print(
                "  ⚙️  Using sequential worker initialization (colocated mode)",
                flush=True,
            )

            # Initialize vLLM first (clean GPU memory), then policy
            policy_generation, vllm_time = init_vllm()
            worker_init_timing_metrics["vllm_init_time_s"] = vllm_time

            policy, policy_time = init_policy()
            worker_init_timing_metrics["policy_init_time_s"] = policy_time
            worker_init_timing_metrics["parallel_init_enabled"] = 0.0

        print(
            f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}",
            flush=True,
        )

    # Record when worker initialization completes (for calculating other setup time)
    worker_init_complete_time = time.perf_counter() - setup_start_time

    # print the node IP and GPU ID of the policy workers for debugging
    policy.print_node_ip_and_gpu_id()

    # if it is not colocated inference, initialize collective communication for update weights
    if not colocated_inference:
        t0 = time.perf_counter()
        ip, port = train_cluster.get_master_address_and_port()
        print(f"Using ip: {ip}, port: {port} for collective communication", flush=True)
        # world includes all training workers and all inference workers
        train_world_size = train_cluster.world_size()
        inference_world_size = inference_nodes * inference_gpus_per_node
        world_size = train_world_size + inference_world_size
        # init collective
        futures_train = policy.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
        futures_inference = policy_generation.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )  # type: ignore
        # wait for all futures to complete
        ray.get(futures_train + futures_inference)
        worker_init_timing_metrics["collective_init_time_s"] = time.perf_counter() - t0

    # prepare refit info
    state_dict_info = policy.prepare_refit_info()
    if policy_generation is not None:
        policy_generation.prepare_refit_info(state_dict_info)

    loss_fn = ClippedPGLossFn(loss_config)

    # Validate force_on_policy_ratio
    if loss_config.get("force_on_policy_ratio", False):
        assert (
            grpo_config["num_prompts_per_step"]
            * grpo_config["num_generations_per_prompt"]
            == policy_config["train_global_batch_size"]
        ), (
            "force_on_policy_ratio requires train_global_batch_size == num_prompts_per_step * num_generations_per_prompt"
        )
        print("  ✓ force_on_policy_ratio enabled")

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
        (train_cluster, inference_cluster),
        dataloaders,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_save_state,
        master_config,
    )


# ===============================================================================
# Core Algorithm Functions
# ===============================================================================


def normalize_advantages_with_epsilon(
    advantages: torch.Tensor,
    std: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Normalize advantages by standard deviation, skipping samples with zero std.

    When std is exactly zero (from leave-one-out baseline with identical rewards),
    normalization is skipped for those samples to prevent numerical instability.
    This makes normalize_rewards compatible with use_leave_one_out_baseline.

    Args:
        advantages: Tensor of shape (batch_size, 1) containing advantage values
        std: Tensor of shape (batch_size,) containing standard deviation values
        epsilon: Small value to avoid division by very small std, defaults to 1e-6

    Returns:
        Normalized advantages tensor of same shape as input advantages
    """
    # Only normalize where std > 0 to avoid division by near-zero
    non_zero_std_mask = std > 0
    advantages[non_zero_std_mask] = advantages[non_zero_std_mask] / (
        std.unsqueeze(-1)[non_zero_std_mask] + epsilon
    )
    return advantages


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
    is_batch_complete is set to False to indicate that the current batch is not enough to meet the required batch size. This is used as a signal in the GRPO training loop
    to continue sampling or proceed to training.
    This approach is based on the dynamic sampling algorithm from the DAPO paper:
    https://arxiv.org/pdf/2503.14476.

    Args:
        repeated_batch (BatchedDataDict[DatumSpec]): The current batch of data containing prompts, responses, rewards, baselines, and std.
        std (torch.Tensor): Tensor representing the standard deviation for each prompt group.
        baseline (torch.Tensor): Baseline values for each prompt group.
        dynamic_sampling_num_gen_batches (int): Number of generation batches processed at the current step.
        master_config (MasterConfig): Configuration containing GRPO and policy settings.
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
        master_config["grpo"]["num_prompts_per_step"]
        * master_config["grpo"]["num_generations_per_prompt"]
    )
    # Store the baseline, std and total_reward for the current unfiltered batch.
    repeated_batch["baseline"] = baseline
    repeated_batch["std"] = std
    total_rewards = repeated_batch["total_reward"]
    dynamic_sampling_metrics = {}

    # Dynamic sampling algorithm (used in DAPO algorithm)
    # This block implements dynamic sampling by selecting prompt groups with non-zero std.
    # If sampled prompts (with non-zero std) are fewer than num_prompts_per_step * num_generations_per_prompt, continue sampling until dynamic_sampling_max_gen_batches is reached.
    if master_config["grpo"]["use_dynamic_sampling"]:
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
                dynamic_sampling_max_gen_batches = master_config["grpo"][
                    "dynamic_sampling_max_gen_batches"
                ]
                assert dynamic_sampling_max_gen_batches > 0, (
                    "When using grpo.use_dynamic_sampling, grpo.dynamic_sampling_max_gen_batches must be > 0"
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
        if master_config["grpo"]["use_dynamic_sampling"]
        else repeated_batch
    )
    return batch_to_return, is_batch_complete, batch_cache, dynamic_sampling_metrics


def scale_rewards(
    repeated_batch: BatchedDataDict[DatumSpec], reward_scaling_cfg: RewardScalingConfig
) -> BatchedDataDict[DatumSpec]:
    """Linearly scales rewards from a source range to a target range.

    If `reward_scaling.enabled` is True, each reward in `repeated_batch["total_reward"]`
    is clamped to the configured source interval [source_min, source_max] and then
    rescaled to the target interval [target_min, target_max].

    Default configuration:
        source_min = 0.0
        source_max = 1.0
        target_min = 0.0
        target_max = 1.0
    """
    if reward_scaling_cfg["enabled"]:
        rewards = repeated_batch["total_reward"]
        source_min = float(reward_scaling_cfg["source_min"])
        source_max = float(reward_scaling_cfg["source_max"])
        target_min = float(reward_scaling_cfg["target_min"])
        target_max = float(reward_scaling_cfg["target_max"])

        # Detect out-of-range values
        out_of_range_mask = (rewards < source_min) | (rewards > source_max)
        if torch.any(out_of_range_mask):
            print(
                f"[reward_scaling] WARNING: {int(out_of_range_mask.sum())} rewards "
                f"are outside the configured source range [{source_min}, {source_max}]. "
                f"Values will be clipped before scaling."
            )

        # Clamp and scale
        rewards = torch.clamp(rewards, min=source_min, max=source_max)
        scaled_rewards = target_min + (rewards - source_min) / (
            source_max - source_min
        ) * (target_max - target_min)
        repeated_batch["total_reward"] = scaled_rewards

    return repeated_batch


def _should_use_async_rollouts(master_config: MasterConfig) -> bool:
    """Determine if async rollouts should be used based on the configuration.

    Returns True if vLLM backend is used with async_engine enabled.
    """
    generation_config = master_config["policy"]["generation"]
    if generation_config is None:
        return False

    backend = generation_config.get("backend", "")
    if backend != "vllm":
        return False

    vllm_cfg = generation_config.get("vllm_cfg", {})
    return vllm_cfg.get("async_engine", False)


def _should_use_nemo_gym(master_config: MasterConfig) -> bool:
    """Determine if NeMo-Gym should be used for rollouts and validation based on the configuration."""
    env_config = master_config.get("env") or dict()
    should_use_nemo_gym = bool(env_config.get("should_use_nemo_gym"))
    if not should_use_nemo_gym:
        return should_use_nemo_gym

    # Validate the setup for training with NeMo-Gym
    assert _should_use_async_rollouts(master_config), (
        "❌ Error: In order to use NeMo-Gym, you must use vllm generation backend with `async_engine: true`!"
    )

    generation_config = master_config["policy"]["generation"]

    # We piggyback off of `_should_use_async_rollouts` to guarantee the existence of these configs.
    should_expose_http_server = generation_config["vllm_cfg"].get("expose_http_server")
    assert should_expose_http_server, (
        "In order to use NeMo-Gym, you must expose the vllm server via `expose_http_server: true`!"
    )

    return should_use_nemo_gym


def refit_policy_generation(
    policy: ColocatablePolicyInterface,
    policy_generation: GenerationInterface,
    colocated_inference: bool,
    _refit_buffer_size_gb: Optional[int] = None,
    timer: Optional[Timer] = None,
    kv_scales: Optional[dict[str, float]] = None,
) -> None:
    """Refit the policy generation interface with the latest policy weights.

    Args:
        policy: The policy to provide weights to the inference engine.
        policy_generation: The inference engine to refit.
        _refit_buffer_size_gb: The size of the buffer to use for refitting.
            If it is None, the buffer size will be computed by the remaining memory.
            This parameter is primarily used for testing.
        timer: Optional Timer used to time the prepare/transfer/update phase
        kv_scales: Optional dictionary of KV cache scales for FP8 quantization.
    """
    if colocated_inference:
        policy.offload_before_refit()
        policy_generation.prepare_for_generation(tags=["weights"])

    # Create a context manager that does nothing when timer is None
    timer_context = (
        timer.time("prepare_for_generation/transfer_and_update_weights")
        if timer is not None
        else nullcontext()
    )
    with timer_context:
        # update weights
        update_success = False
        if colocated_inference:
            # get model param keys, which is grouped by size
            if _refit_buffer_size_gb is not None:
                buffer_size_bytes = _refit_buffer_size_gb * (1024**3)
            else:
                # Empirically sets ratio as 30% to maximize efficiency.
                # The remaining 70% is a necessary buffer reserved for the parameter all-gathering across the expert-parallelism dimension.
                memory_ratio = os.getenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0.3")
                buffer_size_bytes = int(
                    policy.get_free_memory_bytes() * float(memory_ratio)
                )

            futures_train = policy.stream_weights_via_ipc_zmq(
                buffer_size_bytes=buffer_size_bytes, kv_scales=kv_scales
            )
            futures_inference = policy_generation.update_weights_via_ipc_zmq()
            # wait for all futures to complete
            ray.get(futures_train)
            results = ray.get(futures_inference)
            update_success = all(result for result in results if result is not None)
        else:
            # update weights through nccl
            futures_train = policy.broadcast_weights_for_collective(kv_scales=kv_scales)
            futures_inference = policy_generation.update_weights_from_collective()
            # wait for all futures to complete
            ray.get(futures_train)
            results = ray.get(futures_inference)
            update_success = all(result for result in results if result is not None)

        # check if update is successful
        if not update_success:
            error_tag = "cuda-ipc" if colocated_inference else "nccl"
            error_message = (
                "❌ Error: Updating weights for the generation policy failed during refit.\n"
                f"This often indicates an issue with {error_tag} or "
                "a problem within the generation backend (e.g., vLLM worker).\n"
            )
            raise RuntimeError(error_message)

    if colocated_inference:
        policy.offload_after_refit()
        policy_generation.prepare_for_generation(tags=["kv_cache"])


# ===============================================================================
# Training & Validation
# ===============================================================================


def grpo_train(
    policy: ColocatablePolicyInterface,
    policy_generation: Optional[GenerationInterface],
    dataloaders: dict[str, StatefulDataLoader],
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: LossFunction,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    grpo_save_state: GRPOSaveState,
    master_config: MasterConfig,
    processor: Optional[AutoProcessor] = None,
) -> None:
    """Run GRPO training algorithm."""
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    kv_scales_cache = None  # Cache reused for computed kv scales

    NEED_REFIT = True
    # If policy_generation is None, use the policy as the generation interface (megatron framework backend)
    if policy_generation is None:
        policy_generation = policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running
    assert policy_generation is not None  # for mypy type check

    # Check if we need to sync KV cache scales
    # When fallback to policy as the policy_generation, we use getattr to check.
    sync_kv_scales = getattr(policy_generation, "requires_kv_scale_sync", False)

    # common config/state itmes
    current_step = grpo_save_state["current_step"]  # current step within an epoch
    total_steps = grpo_save_state["total_steps"]  # total steps across all epochs
    max_num_steps = master_config["grpo"][
        "max_num_steps"
    ]  # max number of steps to train for
    current_epoch = grpo_save_state["current_epoch"]  # current epoch
    max_num_epochs = master_config["grpo"][
        "max_num_epochs"
    ]  # max number of epochs to train for
    consumed_samples = grpo_save_state[
        "consumed_samples"
    ]  # total samples consumed across all epochs
    total_valid_tokens = grpo_save_state.get(
        "total_valid_tokens", 0
    )  # total valid tokens processed across all epochs; default to 0 for backward compatibility with older checkpoints
    val_at_start = master_config["grpo"]["val_at_start"]
    val_period = master_config["grpo"]["val_period"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]

    # Run validation at the start if configured
    # TODO: Add validation with kv scales if needed
    if val_at_start and current_step == 0:
        print("\n🔍 Running initial validation...", flush=True)
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(policy, policy_generation, colocated_inference)
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
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, current_step, prefix="validation")
        logger.log_metrics(validation_timings, current_step, prefix="timing/validation")

    # Set dataloader
    if master_config["data"]["use_multiple_dataloader"]:
        num_prompts_per_step = master_config["grpo"]["num_prompts_per_step"]
        batch_multiplier = master_config["grpo"]["batch_multiplier"]
        dataloader = MultipleDataloaderWrapper(
            expected_num_prompts=num_prompts_per_step * batch_multiplier,
            data_config=master_config["data"],
            dataloaders=dataloaders,
        )
    else:
        dataloader = dataloaders["all_tasks"]

    while current_epoch < max_num_epochs and total_steps < max_num_steps:
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")
        # batch cache is used for DAPO. We store prompts with non-zero standard deviation in this cache.
        batch_cache: BatchedDataDict[DatumSpec] = None
        # This is the number of batches we processed so far at each step to generate responses whose std is non-zero. Maximum threshold is set by dynamic_sampling_max_gen_batches. Used in the case of dynamic sampling.
        dynamic_sampling_num_gen_batches = 0

        # Run grpo/dapo training loop (single-turn)
        for batch in dataloader:
            if master_config["data"]["use_multiple_dataloader"]:
                print(
                    f"\n{'=' * 25} Step {current_step + 1}/{max_num_steps} {'=' * 25}",
                    flush=True,
                )
            else:
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
                    # Repeat batch items
                    repeated_batch: BatchedDataDict[DatumSpec] = (
                        batch.repeat_interleave(
                            master_config["grpo"]["num_generations_per_prompt"]
                        )
                    )
                    # Convert LLMMessageLogType to FlatMessagesType for generation
                    batched_flat, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    )
                    input_ids = batched_flat["token_ids"]

                # Generate responses - this updates the LLMMessageLogType in repeated_batch
                print(
                    f"▶ Generating responses for batch of size {repeated_batch.size}...",
                    flush=True,
                )
                with timer.time("prepare_for_generation/total"):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        # Compute KV scales if needed for FP8 quantization
                        if sync_kv_scales and kv_scales_cache is None:
                            print("▶ Computing KV cache scales...", flush=True)
                            policy.prepare_for_lp_inference()
                            # Align with training data processing to ensure parallel training compatibility
                            calib_flat, calib_input_lengths = (
                                batched_message_log_to_flat_message(
                                    repeated_batch["message_log"],
                                    pad_value_dict={
                                        "token_ids": tokenizer.pad_token_id
                                    },
                                    make_sequence_length_divisible_by=master_config[
                                        "policy"
                                    ]["make_sequence_length_divisible_by"],
                                )
                            )
                            # Create calibration data from flattened messages
                            calibration_data = BatchedDataDict[ClippedPGLossDataDict](
                                {
                                    "input_ids": calib_flat["token_ids"],
                                    "input_lengths": calib_input_lengths,
                                }
                            )
                            calibration_data.update(
                                calib_flat.get_multimodal_dict(as_tensors=False)
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
                        POLICY_GENERATION_STALE = False
                    else:
                        if colocated_inference:
                            policy.offload_after_refit()  # unload optimizer to make space for generation
                        policy_generation.prepare_for_generation()

                dynamic_sampling_num_gen_batches += 1
                with timer.time("generation"):
                    # Clear vLLM logger metrics for each generation step
                    if policy_generation is not None and hasattr(
                        policy_generation, "clear_vllm_logger_metrics"
                    ):
                        policy_generation.clear_vllm_logger_metrics()
                    # Use NeMo-Gym rollouts if enabled. We cascade NeMo-Gym first since NeMo-Gym requires async rollouts.
                    if _should_use_nemo_gym(master_config):
                        generation_config = master_config["policy"]["generation"]
                        nemo_gym_rollout_result = run_async_nemo_gym_rollout(
                            policy_generation=policy_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=None,
                            generation_config=generation_config,
                            max_rollout_turns=None,
                            greedy=False,
                        )
                        input_ids = nemo_gym_rollout_result.input_ids
                        repeated_batch = nemo_gym_rollout_result.final_batch
                        rollout_metrics = nemo_gym_rollout_result.rollout_metrics
                    # Use async rollouts if vLLM async engine is enabled
                    elif _should_use_async_rollouts(master_config):
                        (
                            repeated_batch,
                            rollout_metrics,
                        ) = run_async_multi_turn_rollout(
                            policy_generation=policy_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=master_config["grpo"][
                                "max_rollout_turns"
                            ],
                            greedy=False,
                        )
                    else:
                        repeated_batch, rollout_metrics = run_multi_turn_rollout(
                            policy_generation=policy_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=master_config["grpo"][
                                "max_rollout_turns"
                            ],
                            greedy=False,
                        )
                    policy_generation.finish_generation()
                    # Collect vLLM logger metrics for performance reporting after each generation step
                    # inflight batch sizes and num pending samples are collected from each vLLM worker
                    if policy_generation is not None and hasattr(
                        policy_generation, "get_vllm_logger_metrics"
                    ):
                        vllm_logger_metrics = (
                            policy_generation.get_vllm_logger_metrics()
                        )
                    else:
                        vllm_logger_metrics = {}

                repeated_batch = scale_rewards(
                    repeated_batch, master_config["grpo"]["reward_scaling"]
                )
                # Process rewards with custom reward function
                if master_config["grpo"]["reward_shaping"]["enabled"]:
                    repeated_batch = apply_reward_shaping(
                        repeated_batch, master_config["grpo"]["reward_shaping"]
                    )

                # Calculate rewards & advantages
                print("▶ Processing rewards...,", flush=True)
                with timer.time("reward_calculation"):
                    # Extract rewards from final_batch
                    rewards = repeated_batch["total_reward"]

                    print("▶ Computing advantages...", flush=True)
                    baseline, std = calculate_baseline_and_std_per_prompt(
                        input_ids,
                        rewards,
                        torch.ones_like(rewards),
                        leave_one_out_baseline=master_config["grpo"][
                            "use_leave_one_out_baseline"
                        ],
                    )
                    # Apply dynamic sampling to filter prompts with non-zero std (DAPO algorithm)
                    repeated_batch, is_batch_complete, batch_cache, ds_metrics = (
                        dynamic_sampling(
                            repeated_batch,
                            std,
                            baseline,
                            dynamic_sampling_num_gen_batches,
                            master_config,
                            timer,
                            batch_cache,
                        )
                    )
                    if ds_metrics:
                        ds_metrics["dynamic_sampling_num_gen_batches"] = (
                            dynamic_sampling_num_gen_batches
                        )
                    # Get the updated rewards and baselines. For DAPO, these rewards and baselines only correspond to the prompts with non-zero std.
                    rewards = (
                        repeated_batch["total_reward"]
                        if not master_config["grpo"]["use_dynamic_sampling"]
                        else repeated_batch["filtered_reward"]
                    )
                    baseline = repeated_batch["baseline"]
                    std = repeated_batch["std"]

                    # If the current batch is not enough to fill the buffer during dynamic sampling, we update the cache and process the next batch.
                    if not is_batch_complete:
                        continue
                    advantages = (rewards - baseline).unsqueeze(-1)

                    if master_config["grpo"]["normalize_rewards"]:
                        advantages = normalize_advantages_with_epsilon(
                            advantages=advantages,
                            std=std,
                        )

                with timer.time("data_processing"):
                    use_overlong_filtering = master_config["grpo"]["overlong_filtering"]
                    if use_overlong_filtering:
                        loss_multiplier = repeated_batch["loss_multiplier"].clone()
                        truncated = repeated_batch["truncated"]

                        if isinstance(truncated, list):
                            truncated = torch.tensor(truncated, dtype=torch.bool)

                        loss_multiplier[truncated] = 0
                        repeated_batch["loss_multiplier"] = loss_multiplier
                    # Add loss mask and advantages to each message in LLMMessageLogType
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
                            message["advantages"] = advantages[i].expand(
                                message["token_ids"].shape
                            )

                    # Convert updated LLMMessageLogType to FlatMessagesType for training
                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    # Create training data from flattened messages
                    train_data = BatchedDataDict[ClippedPGLossDataDict](
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "advantages": flat_messages["advantages"],
                            "generation_logprobs": flat_messages["generation_logprobs"],
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )
                    # this will be mini-batched inside the policy, so maintain the packed multimodal structure
                    train_data.update(
                        flat_messages.get_multimodal_dict(as_tensors=False)
                    )
                    train_data.to("cpu")

                print("▶ Preparing for logprob inference...", flush=True)
                with timer.time("logprob_inference_prep"):
                    policy.prepare_for_lp_inference()

                print("▶ Computing logprobs...", flush=True)
                with timer.time("policy_and_reference_logprobs"):
                    fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]
                    reference_logprobs = policy.get_reference_policy_logprobs(
                        train_data
                    )["reference_logprobs"]
                    train_data["prev_logprobs"] = fprop_logprobs
                    train_data["reference_policy_logprobs"] = reference_logprobs

                print("▶ Preparing for training...", flush=True)
                with timer.time("training_prep"):
                    policy.prepare_for_training()  # set model train and reload optim to GPU
                    POLICY_GENERATION_STALE = True

                print("▶ Training policy...", flush=True)
                with timer.time("policy_training"):
                    train_results = policy.train(train_data, loss_fn)

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
                        # Set generation as stale to force refit with new scales
                        POLICY_GENERATION_STALE = True

                is_last_step = total_steps + 1 >= max_num_steps
                if not master_config["data"]["use_multiple_dataloader"]:
                    is_last_step = is_last_step or (
                        (current_epoch + 1 == max_num_epochs)
                        and (current_step + 1 == len(dataloader))
                    )

                # Run validation if it's a validation step
                if val_period > 0 and (total_steps + 1) % val_period == 0:
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            policy,
                            policy_generation,
                            colocated_inference,
                            kv_scales=kv_scales_cache if sync_kv_scales else None,
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        if colocated_inference:
                            policy.offload_after_refit()  # unload optimizer to make space for generation
                        policy_generation.prepare_for_generation()
                    val_metrics, validation_timings = validate(
                        policy_generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        step=total_steps + 1,
                        master_config=master_config,
                    )
                    policy_generation.finish_generation()
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )

                # Get flat advantages and token mask for masked metrics computation
                flat_advantages = flat_messages["advantages"]
                flat_token_mask = flat_messages["token_loss_mask"]

                # Filter advantages using token mask (only valid response tokens)
                response_advantages = torch.masked_select(
                    flat_advantages, flat_token_mask.bool()
                )

                metrics = {
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                    "reward": rewards.numpy(),
                    "mean_prompt_length": repeated_batch["length"].numpy(),
                    "total_num_tokens": input_lengths.numpy(),
                    # Add masked advantages tracking metrics (only for valid response tokens)
                    "advantages/mean": torch.mean(response_advantages).detach().item()
                    if response_advantages.numel() > 0
                    else 0.0,
                    "advantages/max": torch.max(response_advantages).detach().item()
                    if response_advantages.numel() > 0
                    else 0.0,
                    "advantages/min": torch.min(response_advantages).detach().item()
                    if response_advantages.numel() > 0
                    else 0.0,
                    **ds_metrics,
                }
                if "moe_metrics" in train_results:
                    metrics.update(
                        {f"moe/{k}": v for k, v in train_results["moe_metrics"].items()}
                    )
                if master_config["grpo"]["use_dynamic_sampling"]:
                    metrics["filtered_reward"] = rewards.numpy()
                    metrics["reward"] = repeated_batch["total_reward"].numpy()

                metrics.update(train_results["all_mb_metrics"])
                for k, v in metrics.items():
                    if k in {"probs_ratio_min", "probs_ratio_clamped_min"}:
                        valid_values = [x for x in v if not np.isinf(x)]
                        metrics[k] = (
                            np.min(valid_values).item() if valid_values else -1.0
                        )
                    elif k in {"probs_ratio_max", "probs_ratio_clamped_max"}:
                        valid_values = [x for x in v if not np.isinf(x)]
                        metrics[k] = (
                            np.max(valid_values).item() if valid_values else -1.0
                        )
                    elif k in {
                        "lr",
                        "wd",
                        "reward",
                        "filtered_reward",
                        "global_valid_seqs",
                        "global_valid_toks",
                        "mean_prompt_length",
                    }:
                        metrics[k] = np.mean(v).item()
                    else:
                        metrics[k] = np.sum(v).item()

                metrics.update(rollout_metrics)
                metrics["vllm_logger_metrics"] = vllm_logger_metrics
                total_valid_tokens += metrics["global_valid_toks"]

                ## Checkpointing
                consumed_samples += master_config["grpo"]["num_prompts_per_step"]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                    == 0
                )
                # +1 because step is 0-indexed
                # Check if timeout-based checkpointing is enabled in config.
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    policy.prepare_for_training()

                    # +1 because step is 0-indexed
                    grpo_save_state["current_step"] = current_step + 1
                    grpo_save_state["total_steps"] = total_steps + 1
                    grpo_save_state["current_epoch"] = current_epoch
                    grpo_save_state["total_valid_tokens"] = total_valid_tokens
                    if val_metrics is not None:
                        grpo_save_state["val_reward"] = val_metrics["accuracy"]
                    elif "val_reward" in grpo_save_state:
                        del grpo_save_state["val_reward"]
                    grpo_save_state["consumed_samples"] = consumed_samples

                    full_metric_name = master_config["checkpointing"]["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:',\n"
                            f'followed by the corresponding name in the "val" or "train" metrics dictionary.'
                            f"  If you are using an old config, please updated checkpointing.metric_name to the new format, "
                            f" e.g. 'val_reward --> 'val:reward'"
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. "
                                "This checkpoint will not be saved as top-k.",
                                stacklevel=2,
                            )
                            if full_metric_name in grpo_save_state:
                                del grpo_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            grpo_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

                    with timer.time("checkpointing"):
                        print(
                            f"Saving checkpoint for step {total_steps + 1}...",
                            flush=True,
                        )
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, grpo_save_state, master_config
                        )
                        policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config["checkpointing"],
                        )
                        for task_name, task_dataloader in dataloaders.items():
                            torch.save(
                                task_dataloader.state_dict(),
                                os.path.join(
                                    checkpoint_path, f"train_dataloader_{task_name}.pt"
                                ),
                            )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # Logging
            # Log training data
            log_data = {"content": flat_messages["content"]}
            log_data["rewards"] = rewards.tolist()
            if master_config["grpo"]["use_dynamic_sampling"]:
                log_data["filtered_rewards"] = rewards.tolist()
                log_data["rewards"] = repeated_batch["total_reward"].tolist()

            log_data["generation_logprobs"] = train_data["generation_logprobs"].tolist()
            log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()
            log_data["input_lengths"] = input_lengths.tolist()
            logger.log_batched_dict_as_jsonl(
                log_data, f"train_data_step{total_steps + 1}.jsonl"
            )

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )  # type: ignore
            # track example with high token mult prob error above 1.05
            if metrics["token_mult_prob_error"] > 1.05:
                logger.log_plot_token_mult_prob_error(
                    {
                        "prompt_lengths": repeated_batch["length"],
                        "full_lengths": input_lengths,
                        "generation_logprobs": train_data["generation_logprobs"],
                        "prev_logprobs": train_data["prev_logprobs"],
                        "token_mask": train_data["token_mask"],
                        "sample_mask": train_data["sample_mask"],
                    },
                    total_steps + 1,
                    name="train/token_mult_prob_error_plot_sample",
                )
            if master_config["policy"]["generation"].get("vllm_cfg", {}).get(
                "enable_vllm_metrics_logger", False
            ) and master_config.get("logger", {}).get("wandb_enabled", False):
                log_generation_metrics_to_wandb(
                    vllm_logger_metrics,
                    total_steps + 1,
                    master_config["policy"]["generation"]["vllm_cfg"][
                        "vllm_metrics_logger_interval"
                    ],
                    logger,
                )

            # Plot ISL/OSL/ISL+OSL histograms to wandb
            if (
                master_config["policy"]["generation"]
                .get("vllm_cfg", {})
                .get("async_engine", False)
            ):
                for metric_name in metrics.keys():
                    if metric_name.startswith("histogram/"):
                        logger.log_histogram(
                            metrics[metric_name],
                            total_steps + 1,
                            f"generation_metrics/{metric_name}",
                        )

            print("\n📊 Training Results:")

            print(f"  • Loss: {metrics['loss']:.4f}")
            print(f"  • Generation KL Error: {metrics['gen_kl_error']:.4f}")
            if master_config["grpo"]["use_dynamic_sampling"]:
                print(f"  • Avg Filtered Reward: {np.mean(rewards.numpy()):.4f}")
                print(
                    f"  • Avg Total Reward: {np.mean(repeated_batch['total_reward'].numpy()):.4f}"
                )
            else:
                print(f"  • Avg Reward: {np.mean(rewards.numpy()):.4f}")
            print(
                f"  • Mean Generation Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}",
                flush=True,
            )

            print("\n⏱️  Timing:", flush=True)
            # Display total time first, separately
            total_time = timing_metrics.get("total_step_time", 0)

            number_of_samples_per_step = (
                master_config["grpo"]["num_prompts_per_step"]
                * master_config["grpo"]["num_generations_per_prompt"]
            )
            total_num_gpus = (
                master_config["cluster"]["num_nodes"]
                * master_config["cluster"]["gpus_per_node"]
            )

            print(f"  • Total step time: {total_time:.2f}s", flush=True)

            # Display all other timing metrics
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)", flush=True)

            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics["global_valid_toks"] / total_time / total_num_gpus
            )
            performance_metrics = print_performance_metrics(
                train_results, metrics, timing_metrics, master_config
            )

            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(
                performance_metrics, total_steps + 1, prefix="performance"
            )
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            # Reset the batch and set dynamic_sampling_num_gen_batches to 0
            batch_cache = None
            dynamic_sampling_num_gen_batches = 0

            timer.reset()
            current_step += 1
            total_steps += 1
            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if total_steps >= max_num_steps:
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch


def validate(
    policy_generation: GenerationInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    step: int,
    master_config: MasterConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        assert val_dataloader is not None or master_config["dpo"]["val_period"] == 0, (
            "val_dataloader is None, so dpo.val_period must be 0"
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
            master_config["grpo"]["max_val_samples"]
            // master_config["grpo"]["val_batch_size"]
        )
        for batch_idx, val_batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break

            additional_metrics_to_report = dict()
            # Generate responses (updates the LLMMessageLogType in batch_with_msg_logs)
            # Use async rollouts if vLLM async engine is enabled
            # We cascade NeMo-Gym first since NeMo-Gym also uses async rollouts.
            if _should_use_nemo_gym(master_config):
                generation_config = master_config["policy"]["generation"]
                nemo_gym_rollout_result = run_async_nemo_gym_rollout(
                    policy_generation=policy_generation,
                    input_batch=val_batch,
                    tokenizer=tokenizer,
                    task_to_env=val_task_to_env,
                    max_seq_len=None,
                    generation_config=generation_config,
                    max_rollout_turns=None,
                    greedy=False,
                )
                val_batch = nemo_gym_rollout_result.final_batch
                gen_metrics = nemo_gym_rollout_result.rollout_metrics
                additional_metrics_to_report = gen_metrics
            elif _should_use_async_rollouts(master_config):
                val_batch, gen_metrics = run_async_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                    greedy=False,
                )
            else:
                val_batch, gen_metrics = run_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
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
                    master_config["logger"]["num_val_samples_to_print"],
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
    validation_time = timing_metrics.get("total_validation_time", 0)
    print(f"    • Total validation time: {validation_time:.2f}s", flush=True)

    # Make sure to reset the timer after validation
    timer.reset()

    # Explicit GPU memory cleanup after validation
    gc.collect()
    torch.cuda.empty_cache()

    return val_metrics, timing_metrics


def async_grpo_train(
    policy: ColocatablePolicyInterface,
    policy_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: LossFunction,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    grpo_save_state: GRPOSaveState,
    master_config: MasterConfig,
    max_trajectory_age_steps: int = 1,
) -> None:
    """Run asynchronous GRPO training with replay buffer.

    Args:
        policy: Training policy
        policy_generation: Generation interface
        dataloader: Training data loader
        val_dataloader: Validation data loader
        tokenizer: Tokenizer
        loss_fn: Loss function
        task_to_env: Training environments
        val_task_to_env: Validation environments
        logger: Logger
        checkpointer: Checkpoint manager
        grpo_save_state: Training state
        master_config: Master configuration
        max_trajectory_age_steps: Maximum age (in training steps) for trajectories to be used in training
    """
    # Ensure we are running with a compatible async generation backend
    assert _should_use_async_rollouts(master_config), (
        "Async GRPO requires vLLM backend with vllm_cfg.async_engine=True. "
        "Set policy.generation.vllm_cfg.async_engine to true in your config."
    )
    assert master_config["loss_fn"]["use_importance_sampling_correction"] is True, (
        "Importance sampling correction must be enabled for async GRPO for good convergence due to off-policy samples!"
    )

    if master_config["grpo"]["async_grpo"]["max_trajectory_age_steps"] > 1:
        if not master_config["grpo"]["async_grpo"].get(
            "in_flight_weight_updates", False
        ):
            print(
                "⚠️ WARNING: In-flight weight updates must be enabled for async GRPO with max_trajectory_age_steps > 1. "
                "Without in-flight weight updates, having more max_trajectory_age_steps will not give any performance benefit."
            )

    # Import async utilities only when needed
    from nemo_rl.algorithms.async_utils import AsyncTrajectoryCollector, ReplayBuffer

    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()
    NEED_REFIT = True

    # Setup generation interface
    if policy_generation is None:
        policy_generation = policy
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True
    assert policy_generation is not None

    # Training state
    step = grpo_save_state["current_step"]
    weight_version = step  # Tracks refitted weight versions
    consumed_samples = grpo_save_state["consumed_samples"]
    total_valid_tokens = grpo_save_state.get(
        "total_valid_tokens", 0
    )  # Default to 0 for backward compatibility with older checkpoints
    val_period = master_config["grpo"]["val_period"]
    val_at_start = master_config["grpo"]["val_at_start"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]

    assert not colocated_inference, (
        "Colocated inference is not supported for async GRPO. Please use non-colocated inference."
    )

    # Calculate minimum buffer size from training requirements
    # In per-prompt buffer mode, one buffer entry is 1 prompt * num_generations_per_prompt
    num_prompts_per_step = master_config["grpo"]["num_prompts_per_step"]
    samples_per_prompt_group = master_config["grpo"]["num_generations_per_prompt"]
    train_gbs = master_config["policy"]["train_global_batch_size"]

    # Ensure the buffer has at least one step worth of prompt-groups before training
    min_trajectories_needed = num_prompts_per_step

    print("📊 Buffer requirements calculation:")
    print(f"   - num_prompts_per_step: {num_prompts_per_step}")
    print(f"   - num_generations_per_prompt: {samples_per_prompt_group}")
    print(f"   - samples_per_prompt_group: {samples_per_prompt_group}")
    print(f"   - train_global_batch_size: {train_gbs}")
    print(f"   - min_trajectories_needed: {min_trajectories_needed} (async mode)")

    _replay_py_exec = get_actor_python_env(
        "nemo_rl.algorithms.async_utils.ReplayBuffer"
    )
    if _replay_py_exec.startswith("uv"):
        # Lazily build a dedicated venv across all Ray nodes on-demand.
        _replay_py_exec = create_local_venv_on_each_node(
            _replay_py_exec,
            "nemo_rl.algorithms.async_utils.ReplayBuffer",
        )

    _replay_runtime_env = {
        "py_executable": _replay_py_exec,
        "env_vars": {
            **os.environ,
            "VIRTUAL_ENV": _replay_py_exec,
            "UV_PROJECT_ENVIRONMENT": _replay_py_exec,
        },
    }

    # Calculate optimal buffer size based on generation limits to prevent length bias
    # Each weight version generates exactly num_prompts_per_step trajectories
    # With max_age_steps, we keep trajectories from multiple weight versions
    num_prompts_per_step = master_config["grpo"]["num_prompts_per_step"]
    late_arrival_slack = 2
    optimal_buffer_size = (
        num_prompts_per_step * max_trajectory_age_steps * late_arrival_slack
    )

    replay_buffer = ReplayBuffer.options(runtime_env=_replay_runtime_env).remote(
        max_size=optimal_buffer_size
    )

    _tc_py_exec = get_actor_python_env(
        "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector"
    )
    if _tc_py_exec.startswith("uv"):
        _tc_py_exec = create_local_venv_on_each_node(
            _tc_py_exec,
            "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector",
        )

    _tc_runtime_env = {
        "py_executable": _tc_py_exec,
        "env_vars": {
            **os.environ,
            "VIRTUAL_ENV": _tc_py_exec,
            "UV_PROJECT_ENVIRONMENT": _tc_py_exec,
        },
    }

    # Initialize trajectory collector with synchronized collection
    trajectory_collector = AsyncTrajectoryCollector.options(
        runtime_env=_tc_runtime_env
    ).remote(
        policy_generation=policy_generation,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        master_config=master_config,
        replay_buffer=replay_buffer,
        start_step=step,
    )

    # Start trajectory collection in background
    collection_task = trajectory_collector.start_collection.remote(dataloader)

    # Ensure collector knows initial weight version
    trajectory_collector.set_weight_version.remote(weight_version)

    print("📦 Started continuous background trajectory collection")

    print(
        f"🚀 Starting async GRPO training with buffer_size={optimal_buffer_size}, max_age={max_trajectory_age_steps} steps"
    )

    print("⏳ Preparing policy generation for training...")
    if NEED_REFIT and POLICY_GENERATION_STALE:
        print("🔄 Refitting policy generation with actual model weights...")
        try:
            refit_policy_generation(policy, policy_generation, colocated_inference)
            print("✅ Policy generation refit completed successfully")
            POLICY_GENERATION_STALE = False
        except Exception as e:
            print(f"❌ Policy generation refit failed: {e}")
            import traceback

            traceback.print_exc()
            return
    else:
        print("🔄 Preparing policy generation for inference...")
        try:
            policy_generation.prepare_for_generation()
            print("✅ Policy generation preparation completed successfully")
        except Exception as e:
            print(f"❌ Policy generation preparation failed: {e}")
            import traceback

            traceback.print_exc()
            return

    print("✅ Policy generation setup complete, proceeding to validation...")

    # Run validation at start if configured
    if val_at_start and step == 0:
        print("\n🔍 Running initial validation...")
        # Pause trajectory collection during initial validation
        trajectory_collector.pause.remote()

        try:
            val_metrics, validation_timings = validate(
                policy_generation,
                val_dataloader,
                tokenizer,
                val_task_to_env,
                step=0,
                master_config=master_config,
            )
            policy_generation.finish_generation()
            logger.log_metrics(val_metrics, step, prefix="validation")
            logger.log_metrics(validation_timings, step, prefix="timing/validation")
            print("✅ Initial validation completed successfully")
        except Exception as e:
            print(f"❌ Initial validation failed: {e}")
            import traceback

            traceback.print_exc()
            # Continue anyway since validation is optional
        finally:
            # Resume trajectory collection after initial validation
            trajectory_collector.resume.remote()

    print("✅ All setup complete, starting buffer wait...")

    # Clear vLLM logger metrics after at start of training
    if policy_generation is not None and hasattr(
        policy_generation, "clear_vllm_logger_metrics"
    ):
        policy_generation.clear_vllm_logger_metrics()

    # Wait for initial buffer fill
    print(
        f"⏳ Waiting for replay buffer to have sufficient trajectories ({min_trajectories_needed} trajectories)..."
    )
    wait_iterations = 0
    while True:
        buffer_size_current = ray.get(replay_buffer.size.remote())

        print(
            f"  Wait iteration {wait_iterations}: buffer_filled_ratio={buffer_size_current}/{min_trajectories_needed}"
        )

        if buffer_size_current >= min_trajectories_needed:
            break

        time.sleep(1.0)

    print("✅ Buffer ready! Starting training loop...")

    # Main training loop
    try:
        while step < master_config["grpo"]["max_num_steps"]:
            print(
                f"\n{'=' * 25} Step {step + 1}/{master_config['grpo']['max_num_steps']} {'=' * 25}"
            )
            maybe_gpu_profile_step(policy, step + 1)
            if policy != policy_generation:
                maybe_gpu_profile_step(policy_generation, step + 1)

            with timer.time("total_step_time"):
                # Sample trajectories from replay buffer
                print("📦 Sampling from replay buffer...")
                with timer.time("exposed_generation"):
                    buffer_size_current = ray.get(replay_buffer.size.remote())
                    print(
                        f"📊 Step coordination: training_step={step}, max_age={max_trajectory_age_steps}, buffer_size={buffer_size_current}"
                    )

                    # Sample the required number of per-prompt groups.
                    num_prompt_groups_needed = master_config["grpo"][
                        "num_prompts_per_step"
                    ]
                    sample_result = ray.get(
                        replay_buffer.sample.remote(
                            num_prompt_groups=num_prompt_groups_needed,
                            current_weight_version=weight_version,
                            max_age_steps=max_trajectory_age_steps,
                        )
                    )

                    if (
                        sample_result is None
                        or len(sample_result["trajectories"])
                        != num_prompt_groups_needed
                    ):
                        print(
                            "⏳ Buffer empty or not enough groups to form a full step, waiting..."
                        )

                        # Get buffer debug info to help diagnose the issue
                        buffer_debug = ray.get(replay_buffer.get_debug_info.remote())
                        buffer_size = buffer_debug["total_trajectories"]

                        if buffer_size > 0:
                            print(
                                f"🔍 Debug: Buffer has {buffer_size} trajectories but sampling requires exactly {num_prompt_groups_needed}."
                            )
                            print(f"   Current weight version: {weight_version}")
                            print(f"   Max trajectory age: {max_trajectory_age_steps}")
                            print(
                                f"   Trajectory versions in buffer: {buffer_debug['trajectory_versions']}"
                            )

                        time.sleep(0.5)
                        continue

                    # Extract trajectories and metadata from sample result
                    trajectories = sample_result["trajectories"]
                    avg_trajectory_age = sample_result["avg_trajectory_age"]

                    print(
                        f"✅ Sampled {len(trajectories)} trajectory groups from buffer (avg age: {avg_trajectory_age:.2f} steps)"
                    )

                    # Concatenate per-prompt groups into a single training batch
                    per_prompt_batches = [t["batch"] for t in trajectories]
                    repeated_batch = BatchedDataDict.from_batches(per_prompt_batches)
                    # Aggregate rollout metrics across groups (simple mean where applicable)
                    rollout_metrics = {}
                    for t in trajectories:
                        for k, v in t["rollout_metrics"].items():
                            rollout_metrics.setdefault(k, []).append(v)
                    # TODO: this simple averaging might cause misleading information for such data as max_gen_tokens, etc.
                    rollout_metrics = {
                        k: (sum(v) / len(v) if isinstance(v[0], (int, float)) else v)
                        for k, v in rollout_metrics.items()
                    }

                # Enforce fixed training batch: num_prompts_per_step * num_generations_per_prompt
                expected_batch_size = (
                    master_config["grpo"]["num_prompts_per_step"]
                    * master_config["grpo"]["num_generations_per_prompt"]
                )
                if repeated_batch.size != expected_batch_size:
                    print(
                        f"❌ Unexpected training batch size: got {repeated_batch.size}, expected {expected_batch_size}. Skipping step and waiting for correct buffer content."
                    )
                    time.sleep(0.5)
                    continue

                # Optional sanity: ensure DP divisibility to avoid sharding issues
                dp_size = policy.sharding_annotations.get_axis_size("data_parallel")
                if expected_batch_size % dp_size != 0:
                    raise AssertionError(
                        f"Configuration error: (num_prompts_per_step * num_generations_per_prompt) = {expected_batch_size} must be divisible by data_parallel size {dp_size}."
                    )

                print(f"Got trajectory batch (size: {repeated_batch.size})")

                print("▶ Processing rewards...")
                with timer.time("reward_calculation"):
                    prompt_only_message_logs = []
                    for message_log in repeated_batch["message_log"]:
                        prompt_only_log = []
                        for message in message_log:
                            if message["role"] == "user" or message["role"] == "system":
                                prompt_only_log.append(message)
                        prompt_only_message_logs.append(prompt_only_log)

                    prompt_batched_flat, prompt_input_lengths = (
                        batched_message_log_to_flat_message(
                            prompt_only_message_logs,
                            pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        )
                    )
                    prompt_only_ids = prompt_batched_flat["token_ids"]

                    rewards = repeated_batch["total_reward"]

                    print("▶ Computing advantages...")

                    baseline, std = calculate_baseline_and_std_per_prompt(
                        prompt_only_ids,
                        rewards,
                        torch.ones_like(rewards),
                        leave_one_out_baseline=master_config["grpo"][
                            "use_leave_one_out_baseline"
                        ],
                    )
                    advantages = (rewards - baseline).unsqueeze(-1)

                    print(
                        f"  📊 Rewards stats: min={rewards.min():.4f}, max={rewards.max():.4f}, mean={rewards.mean():.4f}, std={rewards.std():.4f}"
                    )
                    print(
                        f"  📊 Baseline stats: min={baseline.min():.4f}, max={baseline.max():.4f}, mean={baseline.mean():.4f}"
                    )
                    print(
                        f"  📊 Advantages stats: min={advantages.min():.4f}, max={advantages.max():.4f}, mean={advantages.mean():.4f}, std={advantages.std():.4f}"
                    )

                    if master_config["grpo"]["normalize_rewards"]:
                        advantages = normalize_advantages_with_epsilon(
                            advantages=advantages,
                            std=std,
                        )

                        print(
                            f"  📊 Normalized advantages stats: min={advantages.min():.4f}, max={advantages.max():.4f}, mean={advantages.mean():.4f}, std={advantages.std():.4f}"
                        )

                # Prepare training data (same as sync version)
                with timer.time("data_processing"):
                    # Add loss mask and advantages to each message
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
                            message["advantages"] = advantages[i].expand(
                                message["token_ids"].shape
                            )

                    # Convert to flat format for training
                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    # Create training data
                    train_data = BatchedDataDict[ClippedPGLossDataDict](
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "advantages": flat_messages["advantages"],
                            "generation_logprobs": flat_messages["generation_logprobs"],
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )
                    train_data.to("cpu")

                # Training phase (same as sync version)
                print("▶ Preparing for logprob inference...")
                with timer.time("logprob_inference_prep"):
                    policy.prepare_for_lp_inference()

                print("▶ Computing logprobs...")
                with timer.time("policy_and_reference_logprobs"):
                    fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]
                    reference_logprobs = policy.get_reference_policy_logprobs(
                        train_data
                    )["reference_logprobs"]
                    train_data["prev_logprobs"] = fprop_logprobs
                    train_data["reference_policy_logprobs"] = reference_logprobs

                print("▶ Preparing for training...")
                with timer.time("training_prep"):
                    policy.prepare_for_training()
                    POLICY_GENERATION_STALE = True

                print("▶ Training policy...")
                with timer.time("policy_training"):
                    train_results = policy.train(train_data, loss_fn)

                print("🔄 Synchronizing policy weights to trajectory collector…")
                vllm_logger_metrics = None
                if NEED_REFIT:
                    # Measure pending-generation wait as exposed_generation time
                    print("🔄 Coordinating with trajectory collector before refit...")
                    with timer.time("exposed_generation"):
                        ray.get(trajectory_collector.prepare_for_refit.remote())

                    # Collect vLLM logger metrics for performance reporting
                    # inflight batch sizes and num pending samples are collected from each vLLM worker
                    if policy_generation is not None and hasattr(
                        policy_generation, "get_vllm_logger_metrics"
                    ):
                        vllm_logger_metrics = (
                            policy_generation.get_vllm_logger_metrics()
                        )
                    else:
                        vllm_logger_metrics = {}

                    # Only the actual refit/weight transfer should be counted as weight_sync
                    print("🔄 Performing policy generation refit...")
                    with timer.time("weight_sync"):
                        refit_policy_generation(
                            policy, policy_generation, colocated_inference
                        )
                        POLICY_GENERATION_STALE = False

                        # Update weight version before resuming trajectory collection so that all trajectories are updated with the new correct weight version
                        weight_version += 1
                        trajectory_collector.set_weight_version.remote(weight_version)
                        trajectory_collector.resume_after_refit.remote()

                # Clear vLLM logger metrics after each refit (weight sync), starting a new logging cycle
                if policy_generation is not None and hasattr(
                    policy_generation, "clear_vllm_logger_metrics"
                ):
                    policy_generation.clear_vllm_logger_metrics()

                # Validation
                val_metrics, validation_timings = None, None
                is_last_step = step + 1 == master_config["grpo"]["max_num_steps"]

                if val_period > 0 and (step + 1) % val_period == 0:
                    # Pause trajectory collection during validation to reduce memory pressure
                    trajectory_collector.pause.remote()

                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            policy, policy_generation, colocated_inference
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        policy_generation.prepare_for_generation()
                    val_metrics, validation_timings = validate(
                        policy_generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        step=step + 1,
                        master_config=master_config,
                    )
                    policy_generation.finish_generation()
                    logger.log_metrics(
                        validation_timings, step + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(val_metrics, step + 1, prefix="validation")

                    # Explicit GPU memory cleanup after validation in async mode
                    import gc

                    gc.collect()
                    torch.cuda.empty_cache()

                    # Resume trajectory collection after validation
                    trajectory_collector.resume.remote()
                # Get flat advantages and token mask for masked metrics computation
                flat_advantages = flat_messages["advantages"]
                flat_token_mask = flat_messages["token_loss_mask"]

                # Filter advantages using token mask (only valid response tokens)
                response_advantages = torch.masked_select(
                    flat_advantages, flat_token_mask.bool()
                )

                metrics = {
                    "loss": train_results["loss"].numpy(),
                    "reward": rewards.numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                    "mean_prompt_length": repeated_batch["length"].numpy(),
                    "total_num_tokens": input_lengths.numpy(),
                    # Add masked advantages tracking metrics (only for valid response tokens)
                    "advantages/mean": torch.mean(response_advantages).detach().item()
                    if response_advantages.numel() > 0
                    else 0.0,
                    "advantages/max": torch.max(response_advantages).detach().item()
                    if response_advantages.numel() > 0
                    else 0.0,
                    "advantages/min": torch.min(response_advantages).detach().item()
                    if response_advantages.numel() > 0
                    else 0.0,
                }
                if "moe_metrics" in train_results:
                    metrics.update(
                        {f"moe/{k}": v for k, v in train_results["moe_metrics"].items()}
                    )
                metrics.update(train_results["all_mb_metrics"])
                for k, v in metrics.items():
                    if k in {"probs_ratio_min", "probs_ratio_clamped_min"}:
                        valid_values = [x for x in v if not np.isinf(x)]
                        metrics[k] = (
                            np.min(valid_values).item() if valid_values else -1.0
                        )
                    elif k in {"probs_ratio_max", "probs_ratio_clamped_max"}:
                        valid_values = [x for x in v if not np.isinf(x)]
                        metrics[k] = (
                            np.max(valid_values).item() if valid_values else -1.0
                        )
                    elif k in {
                        "lr",
                        "wd",
                        "reward",
                        "global_valid_seqs",
                        "global_valid_toks",
                        "mean_prompt_length",
                    }:
                        metrics[k] = np.mean(v).item()
                    else:
                        metrics[k] = np.sum(v).item()
                metrics.update(rollout_metrics)
                if vllm_logger_metrics is not None:
                    metrics["vllm_logger_metrics"] = vllm_logger_metrics
                total_valid_tokens += metrics["global_valid_toks"]

                # Checkpointing (same as sync version)
                consumed_samples += master_config["grpo"]["num_prompts_per_step"]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (step + 1) % master_config["checkpointing"]["save_period"] == 0
                )
                # +1 because step is 0-indexed
                # Check if timeout-based checkpointing is enabled in config.
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    policy.prepare_for_training()

                    grpo_save_state["current_step"] = step + 1
                    grpo_save_state["total_valid_tokens"] = total_valid_tokens
                    if val_metrics is not None:
                        grpo_save_state["val_reward"] = val_metrics["accuracy"]
                    elif "val_reward" in grpo_save_state:
                        del grpo_save_state["val_reward"]
                    grpo_save_state["consumed_samples"] = consumed_samples

                    full_metric_name = master_config["checkpointing"]["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:',\n"
                            f'followed by the corresponding name in the "val" or "train" metrics dictionary.'
                            f"  If you are using an old config, please updated checkpointing.metric_name to the new format, "
                            f" e.g. 'val_reward --> 'val:accuracy'"
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. "
                                "This checkpoint will not be saved as top-k.",
                                stacklevel=2,
                            )
                            if full_metric_name in grpo_save_state:
                                del grpo_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            grpo_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

                    with timer.time("checkpointing"):
                        print(f"Saving checkpoint for step {step + 1}...")
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            step + 1, grpo_save_state, master_config
                        )
                        policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config["checkpointing"],
                        )
                        # Get dataloader state from trajectory collector
                        actual_dataloader_state = ray.get(
                            trajectory_collector.get_dataloader_state.remote()
                        )
                        torch.save(
                            actual_dataloader_state,
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)
                    policy.offload_after_refit()

            log_data = {"content": flat_messages["content"]}
            log_data["rewards"] = rewards.tolist()
            log_data["generation_logprobs"] = train_data["generation_logprobs"].tolist()
            log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()
            log_data["input_lengths"] = input_lengths.tolist()
            logger.log_batched_dict_as_jsonl(
                log_data, f"train_data_step{step + 1}.jsonl"
            )

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )

            # Add buffer stats
            buffer_size_current = ray.get(replay_buffer.size.remote())
            metrics["buffer_size"] = buffer_size_current
            metrics["avg_trajectory_age"] = avg_trajectory_age

            if master_config["policy"]["generation"].get("vllm_cfg", {}).get(
                "enable_vllm_metrics_logger", False
            ) and master_config.get("logger", {}).get("wandb_enabled", False):
                log_generation_metrics_to_wandb(
                    vllm_logger_metrics,
                    step + 1,
                    master_config["policy"]["generation"]["vllm_cfg"][
                        "vllm_metrics_logger_interval"
                    ],
                    logger,
                )

            # Plot ISL/OSL/ISL+OSL histograms to wandb
            if (
                master_config["policy"]["generation"]
                .get("vllm_cfg", {})
                .get("async_engine", False)
            ):
                for metric_name in metrics.keys():
                    if metric_name.startswith("histogram/"):
                        logger.log_histogram(
                            metrics[metric_name],
                            step + 1,
                            f"generation_metrics/{metric_name}",
                        )

            print("\n📊 Training Results:")
            print(f"  • Loss: {metrics['loss']:.4f}")
            print(f"  • Generation KL Error: {metrics['gen_kl_error']:.4f}")
            print(f"  • Avg Reward: {np.mean(rewards.numpy()):.4f}")
            print(f"  • Buffer Size: {buffer_size_current}")
            print(f"  • Avg Trajectory Age: {avg_trajectory_age:.2f} steps")

            print("\n⏱️  Timing:")
            total_time = timing_metrics.get("total_step_time", 0)
            print(f"  • Total step time: {total_time:.2f}s")
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

            total_num_gpus = (
                master_config["cluster"]["num_nodes"]
                * master_config["cluster"]["gpus_per_node"]
            )
            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics["global_valid_toks"] / total_time / total_num_gpus
            )
            performance_metrics = print_performance_metrics(
                train_results, metrics, timing_metrics, master_config
            )

            logger.log_metrics(performance_metrics, step + 1, prefix="performance")
            logger.log_metrics(metrics, step + 1, prefix="train")
            logger.log_metrics(timing_metrics, step + 1, prefix="timing/train")

            timer.reset()
            step += 1
            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if step >= master_config["grpo"]["max_num_steps"]:
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

    except Exception as e:
        print(f"❌ Error in async loop: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up
        print("🛑 Stopping trajectory collection...")
        try:
            ray.kill(trajectory_collector)
        except Exception as e:
            print(f"Error stopping trajectory collector: {e}")

        try:
            ray.kill(replay_buffer)
        except Exception as e:
            print(f"Error stopping replay buffer: {e}")

        print("Async GRPO training complete!")
