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

from nemo_rl.algorithms.advantage_estimator import (
    GeneralizedAdvantageEstimator,
    GRPOAdvantageEstimator,
    ReinforcePlusPlusAdvantageEstimator,
)
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
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.collectives import T
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_async_nemo_gym_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.automodel import train
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.generation.sglang import SGLangConfig, SGLangGeneration
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.value import Value, ValueConfig
from nemo_rl.models.value.interfaces import ValueInterface
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
    print_message_log_samples,
)
from nemo_rl.utils.memory_tracker import MemoryTracker
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


class AdvEstimatorConfig(TypedDict):
    """Configuration for advantage estimator (GRPO or Reinforce++)."""

    name: str  # "grpo" or "reinforce_plus_plus"
    # GRPO specific
    normalize_rewards: NotRequired[bool]
    use_leave_one_out_baseline: NotRequired[bool]
    # Reinforce++ specific
    minus_baseline: NotRequired[bool]


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
    # Whether to run validation on the last training step. Setting this to True ensures the
    # final checkpoint has validation metrics, which is required for get_best_checkpoint_path().
    val_at_end: bool
    max_val_samples: int
    skip_reference_policy_logprobs_calculation: NotRequired[bool]
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
    # By default advantages are calculated on CPU. Setting this flag to true leverages GPU for their computation.
    calculate_advantages_on_gpu: NotRequired[bool]
    # Advantage estimator configuration (grpo or reinforce_plus_plus)
    adv_estimator: NotRequired[AdvEstimatorConfig]


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
    value: NotRequired[ValueConfig]  # Value model configuration
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
    dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
    processor: Optional[AutoProcessor] = None,
) -> tuple[
    ColocatablePolicyInterface,
    Optional[GenerationInterface],
    Optional[ValueInterface],
    tuple[RayVirtualCluster, RayVirtualCluster],
    StatefulDataLoader,
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
    value_config = master_config.get("value", None)
    generation_config = master_config["policy"]["generation"]
    env_configs = master_config["env"]
    loss_config = master_config["loss_fn"]
    ppo_config = master_config["ppo"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

    assert generation_config is not None, (
        "A generation config in the PolicyConfig is required for GRPO"
    )

    # Set seed for all random number generators
    set_seed(ppo_config["seed"])

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
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        dataloader.load_state_dict(dataloader_state_dict)

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

    # Validate force_on_policy_ratio
    if loss_config.get("force_on_policy_ratio", False):
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
            else 3,
        )
        train_cluster = cluster
        inference_cluster = cluster
        value_cluster = cluster
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
            ppo_config["max_num_steps"],
            ppo_config["max_num_epochs"] * len(dataloader),
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

    def init_value():
        """Initialize value model training workers."""
        t0 = time.perf_counter()
        # Prepare checkpoint paths for value model
        if last_checkpoint_path:
            value_weights_path = Path(last_checkpoint_path) / "value" / "weights"
            value_optimizer_path = Path(last_checkpoint_path) / "value" / "optimizer"
        else:
            value_weights_path = None
            value_optimizer_path = None

        # TODO: Proper implementation of the value model
        # v = Value(
        #     cluster=train_cluster,
        #     config=value_config,
        #     tokenizer=tokenizer,
        #     name_prefix="lm_value",
        #     weights_path=value_weights_path,
        #     optimizer_path=value_optimizer_path,
        #     init_optimizer=True,
        # )
        v = None
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
        pg = SGLangGeneration(cluster=inference_cluster, config=generation_config)
        pg.finish_generation()
        return pg, time.perf_counter() - t0

    def initialize_generation_with_policy(
        init_generation_fn,
        generation_name: str,
        init_time_key: str,
        colocated_inference: bool,
        worker_init_timing_metrics: dict,
    ):
        """Generic function to initialize a generation engine (vLLM or SGLang) along with policy.

        Args:
            init_generation_fn: Function that initializes the generation engine (init_vllm or init_sglang)
            generation_name: Name of the generation engine ("vLLM" or "SGLang")
            init_time_key: Key name for storing initialization time in metrics ("vllm_init_time_s" or "sglang_init_time_s")
            colocated_inference: Whether inference is colocated with training
            worker_init_timing_metrics: Dictionary to store timing metrics

        Returns:
            Tuple of (policy_generation, policy)
        """
        # Determine if parallel initialization is possible (non-colocated mode)
        use_parallel_init = not colocated_inference

        if use_parallel_init:
            # Parallel initialization: Generation engine and Policy can initialize simultaneously
            print(
                "  ⚡ Using parallel worker initialization (non-colocated mode)",
                flush=True,
            )

            # Execute both initializations in parallel
            parallel_start_time = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as executor:
                generation_future = executor.submit(init_generation_fn)
                policy_future = executor.submit(init_policy)
                policy_generation, generation_time = generation_future.result()
                policy, policy_time = policy_future.result()
            parallel_wall_time = time.perf_counter() - parallel_start_time

            # Store timing metrics
            worker_init_timing_metrics[init_time_key] = generation_time
            worker_init_timing_metrics["policy_init_time_s"] = policy_time
            worker_init_timing_metrics["parallel_wall_time_s"] = parallel_wall_time
            worker_init_timing_metrics["parallel_init_enabled"] = True

            # Value model not supported in non-colocated mode yet
            value_model = None

        else:
            # Sequential initialization: colocated mode (GPU memory requires generation engine first)
            print(
                "  ⚙️  Using sequential worker initialization (colocated mode)",
                flush=True,
            )

            # Initialize generation engine first (clean GPU memory), then policy
            policy_generation, generation_time = init_generation_fn()
            worker_init_timing_metrics[init_time_key] = generation_time

            policy, policy_time = init_policy()
            worker_init_timing_metrics["policy_init_time_s"] = policy_time
            worker_init_timing_metrics["parallel_init_enabled"] = 0.0

            # Initialize value model if configured (for GAE in colocated mode)
            if value_config is not None:
                print("  ⚙️  Initializing value model for GAE...", flush=True)
                value_model, value_time = init_value()
                worker_init_timing_metrics["value_init_time_s"] = value_time
                print(f"  ✓ Value model initialized in {value_time:.2f}s", flush=True)
            else:
                value_model = None

        return policy_generation, policy, value_model

    # Handle generation-specific setup
    if backend == "megatron":
        # Megatron generation: policy_generation is None, only initialize policy
        policy_generation = None
        print(
            f"  ✓ Using {backend} backend for generation with {policy_config['model_name']}",
            flush=True,
        )

        policy, policy_time = init_policy()
        worker_init_timing_metrics["policy_init_time_s"] = policy_time

        # Value model not supported for megatron backend yet
        value_model = None

    elif backend == "vllm":
        # vLLM generation: setup config, then initialize with policy
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

        policy_generation, policy, value_model = initialize_generation_with_policy(
            init_generation_fn=init_vllm,
            generation_name="vLLM",
            init_time_key="vllm_init_time_s",
            colocated_inference=colocated_inference,
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
            colocated_inference=colocated_inference,
            worker_init_timing_metrics=worker_init_timing_metrics,
        )

        print(
            f"  ✓ Using SGLang backend for generation with {policy_config['model_name']}",
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
        value_model,
        (train_cluster, inference_cluster),
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_save_state,
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


def _create_advantage_estimator(master_config: MasterConfig):
    """Create and return an advantage estimator based on configuration.

    Args:
        master_config: The master configuration dictionary.

    Returns:
        An advantage estimator instance (GRPOAdvantageEstimator, ReinforcePlusPlusAdvantageEstimator, or GAEAdvantageEstimator).

    Raises:
        ValueError: If the advantage estimator name is not recognized.
    """
    ppo_config = master_config["ppo"]
    loss_config = master_config["loss_fn"]

    # Provide backward-compatible defaults when adv_estimator is not in config.
    # Fall back to top-level grpo.normalize_rewards / grpo.use_leave_one_out_baseline
    # which older configs still use.
    adv_estimator_config = ppo_config.get(
        "adv_estimator",
        {
            "name": "grpo",
            "normalize_rewards": ppo_config.get("normalize_rewards", True),
            "use_leave_one_out_baseline": ppo_config.get(
                "use_leave_one_out_baseline", False
            ),
            "minus_baseline": True,
        },
    )

    adv_estimator_name = adv_estimator_config["name"]
    if adv_estimator_name == "grpo":
        adv_estimator = GRPOAdvantageEstimator(adv_estimator_config, loss_config)
        print("  ✓ Using GRPO advantage estimator")
    elif adv_estimator_name == "reinforce_plus_plus":
        adv_estimator = ReinforcePlusPlusAdvantageEstimator(
            adv_estimator_config, loss_config
        )
        print("  ✓ Using Reinforce++ advantage estimator")
    elif adv_estimator_name == "gae":
        adv_estimator = GeneralizedAdvantageEstimator(adv_estimator_config, loss_config)
        gae_lambda = adv_estimator_config.get("gae_lambda", 0.95)
        gae_gamma = adv_estimator_config.get("gae_gamma", 0.99)
        print(f"  ✓ Using GAE advantage estimator (λ={gae_lambda}, γ={gae_gamma})")
    else:
        raise ValueError(f"Invalid adv_estimator name: {adv_estimator_name}")

    return adv_estimator


def _extract_prompt_only_messages(message_logs: list) -> list:
    """Extract only prompt messages (user/system) from message logs.

    This is used to get prompt IDs for advantage estimation, excluding
    any assistant responses.

    Args:
        message_logs: List of message logs, where each log is a list of messages.

    Returns:
        List of message logs containing only user and system messages.
    """
    prompt_only_message_logs = []
    for message_log in message_logs:
        prompt_only_log = []
        for message in message_log:
            if message["role"] == "user" or message["role"] == "system":
                prompt_only_log.append(message)
        prompt_only_message_logs.append(prompt_only_log)
    return prompt_only_message_logs


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

            if isinstance(policy_generation, SGLangGeneration):
                sglang_url_to_gpu_uuids = (
                    policy_generation.get_sglang_url_to_gpu_uuids()
                )
                # Stream weights via HTTP
                flush_success = policy_generation.invalidate_kv_cache()
                if not flush_success:
                    print("SGLang KV cache invalidation failed before weight update. ")
                futures_train = policy.stream_weights_via_http(
                    sglang_url_to_gpu_uuids=sglang_url_to_gpu_uuids,
                )
                # Wait for all workers to complete
                ray.get(futures_train)
                update_success = True
            else:
                # Original ZMQ IPC path for vLLM
                futures_train = policy.stream_weights_via_ipc_zmq(
                    buffer_size_bytes=buffer_size_bytes
                )
                futures_inference = policy_generation.update_weights_via_ipc_zmq()
                # wait for all futures to complete
                ray.get(futures_train)
                results = ray.get(futures_inference)
                update_success = all(result for result in results if result is not None)
        else:
            # update weights through nccl
            # SGLang haven't implemented non-colocated inference mode.
            if isinstance(policy_generation, SGLangGeneration):
                raise NotImplementedError(
                    "SGLang haven't implemented non-colocated inference mode. "
                )
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


def _log_mixed_rewards_and_advantages_information(
    logger: Logger,
    total_steps: int,
    metrics: dict[str, Any],
    baseline: torch.Tensor,
    advantages: torch.Tensor,
) -> None:
    # The histograms that are logged are logged with a prefix "train/" to the name, since that is what the remaining metrics will be logged with.
    logger.log_histogram(
        baseline.numpy(), total_steps + 1, "train/baseline_reward/histogram"
    )
    metrics["baseline_reward/pct_0"] = 100 * (baseline == 0).float().mean().item()
    metrics["baseline_reward/pct_1"] = 100 * (baseline == 1).float().mean().item()
    metrics["baseline_reward/pct_mixed"] = (
        100 - metrics["baseline_reward/pct_0"] - metrics["baseline_reward/pct_1"]
    )

    logger.log_histogram(
        advantages.numpy(), total_steps + 1, "train/advantages/histogram"
    )
    metrics["advantages/sum"] = advantages.float().sum().item()
    metrics["advantages/mean"] = advantages.float().mean().item()


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
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    grpo_save_state: GRPOSaveState,
    master_config: MasterConfig,
) -> None:
    """Run PPO training algorithm."""
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
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

    if master_config["ppo"].get("skip_reference_policy_logprobs_calculation"):
        assert master_config["loss_fn"]["reference_policy_kl_penalty"] == 0
        print(
            "Reference policy logprob calculation will be skipped since `grpo.skip_reference_policy_logprobs_calculation` is set to True and `loss_fn.reference_policy_kl_penalty` is 0."
        )

    # Check if we need to sync KV cache scales
    # When fallback to policy as the policy_generation, we use getattr to check.
    sync_kv_scales = getattr(policy_generation, "requires_kv_scale_sync", False)

    # common config/state times
    current_step = grpo_save_state["current_step"]  # current step within an epoch
    total_steps = grpo_save_state["total_steps"]  # total steps across all epochs
    current_epoch = grpo_save_state["current_epoch"]  # current epoch
    max_num_epochs = master_config["ppo"][
        "max_num_epochs"
    ]  # max number of epochs to train for
    steps_per_epoch = master_config["ppo"]["steps_per_epoch"]

    consumed_samples = grpo_save_state[
        "consumed_samples"
    ]  # total samples consumed across all epochs
    total_valid_tokens = grpo_save_state.get(
        "total_valid_tokens", 0
    )  # total valid tokens processed across all epochs; default to 0 for backward compatibility with older checkpoints
    val_at_start = master_config["ppo"]["val_at_start"]
    val_at_end = master_config["ppo"]["val_at_end"]
    val_period = master_config["ppo"]["val_period"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]

    # Initialize advantage estimator
    adv_estimator = _create_advantage_estimator(master_config)

    # Run validation at the start if configured
    # TODO: Add validation with kv scales if needed
    if val_at_start and current_step == 0:
        print("\n🔍 Running initial validation...", flush=True)
        memory_tracker.snapshot_start_of_stage("Initial validation", dir())

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
            logger=logger,
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, current_step, prefix="validation")
        logger.log_metrics(validation_timings, current_step, prefix="timing/validation")

    # Run PPO training loop
    train_loader_iter = iter(dataloader)
    for epoch in range(current_epoch, max_num_epochs):
        metrics_logging_data = dict()
        print(f"\n{'=' * 25} Epoch {epoch + 1}/{max_num_epochs} {'=' * 25}")

        with timer.time("total_epoch_time"):
            print("▶ Preparing batch...", flush=True)
            with timer.time("data_processing"):
                batch = next(train_loader_iter)
                repeated_batch: BatchedDataDict[DatumSpec] = batch.repeat_interleave(
                    master_config["ppo"]["num_generations_per_prompt"]
                )
                # Convert LLMMessageLogType to FlatMessagesType for generation
                batched_flat, input_lengths = batched_message_log_to_flat_message(
                    repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                )
                input_ids = batched_flat["token_ids"]
            print(
                f"▶ Generating responses for batch of size {repeated_batch.size}...",
                flush=True,
            )
            with timer.time("prepare_for_generation/total"):
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(
                        policy, policy_generation, colocated_inference
                    )
                    POLICY_GENERATION_STALE = False
                else:
                    policy_generation.prepare_for_generation()

            with timer.time("generation"):
                print("▶ Generating responses...", flush=True)
                # Clear logger metrics for each generation step
                if policy_generation is not None:
                    policy_generation.clear_logger_metrics()
                # Use NeMo-Gym rollouts if enabled. We cascade NeMo-Gym first since NeMo-Gym requires async rollouts.
                repeated_batch, rollout_metrics = run_multi_turn_rollout(
                    policy_generation=policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=tokenizer,
                    task_to_env=task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["ppo"]["max_rollout_turns"],
                    greedy=False,
                )

                policy_generation.finish_generation()
                # Collect generation logger metrics for performance reporting after each generation step
                # inflight batch sizes and num pending samples are collected from each worker
                if policy_generation is not None:
                    generation_logger_metrics = policy_generation.get_logger_metrics()
                metrics_logging_data["mean_gen_tokens_per_sample"] = rollout_metrics[
                    "mean_gen_tokens_per_sample"
                ]
                logger.log_metrics(rollout_metrics, total_steps + 1, prefix="train")

            repeated_batch = scale_rewards(
                repeated_batch, master_config["ppo"]["reward_scaling"]
            )

            with timer.time("reward_calculation"):
                print("▶ Calculating rewards and values...", flush=True)

                for message_log in repeated_batch["message_log"]:
                    for _, message in enumerate(message_log):
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

                messages, input_lengths = batched_message_log_to_flat_message(
                    repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    make_sequence_length_divisible_by=master_config["policy"][
                        "make_sequence_length_divisible_by"
                    ],
                )

                values = (
                    torch.arange(messages["token_ids"].shape[0])
                    / messages["token_ids"].shape[0]
                )[:, None].repeat(1, messages["token_ids"].shape[1])

                train_data = BatchedDataDict(
                    {
                        "input_ids": messages["token_ids"],
                        "input_lengths": input_lengths,
                        "generation_logprobs": messages["generation_logprobs"],
                        "values": values,
                        "rewards": repeated_batch["total_reward"],
                        "sample_mask": repeated_batch["loss_multiplier"],
                        "token_mask": messages["token_loss_mask"],
                    }
                )

            with timer.time("compute_advantages"):
                print("▶ Computing advantages...", flush=True)
                train_data["advantages"] = adv_estimator.compute_advantage(
                    prompt_ids=torch.arange(messages["token_ids"].shape[0]),
                    rewards=train_data["rewards"],
                    mask=torch.ones_like(messages["token_ids"]),
                    values=train_data["values"],
                    lengths=input_lengths,
                )

            with timer.time("logprob_inference_prep"):
                print("▶ Preparing for logprob inference...", flush=True)
                policy.prepare_for_lp_inference()

            with timer.time("policy_and_reference_logprobs"):
                print("▶ Computing policy and reference logprobs...", flush=True)
                train_data["prev_logprobs"] = policy.get_logprobs(
                    train_data, timer=timer
                )["logprobs"]

            for step in range(steps_per_epoch):
                print(f"▶ Step {step + 1}/{steps_per_epoch}...", flush=True)
                permutation = torch.randperm(train_data["advantages"].shape[0])
                train_data_permuted = BatchedDataDict(
                    {
                        "input_ids": train_data["input_ids"][permutation],
                        "input_lengths": train_data["input_lengths"][permutation],
                        "generation_logprobs": train_data["generation_logprobs"][
                            permutation
                        ],
                        "values": train_data["values"][permutation],
                        "rewards": train_data["rewards"][permutation],
                        "sample_mask": train_data["sample_mask"][permutation],
                        "token_mask": train_data["token_mask"][permutation],
                        "advantages": train_data["advantages"][permutation],
                        "prev_logprobs": train_data["prev_logprobs"][permutation],
                    }
                )

                with timer.time("policy_training_prep"):
                    policy.prepare_for_training()

                with timer.time("policy_training"):
                    print("▶ Training policy...", flush=True)
                    train_results = policy.train(
                        train_data_permuted,
                        loss_fn,
                        timer=timer,
                    )

                with timer.time("value_training"):
                    print("▶ Training value...", flush=True)
                    pass

            if current_epoch % val_period == 0 and current_step != 0:
                with timer.time("validation"):
                    print("▶ Validating...", flush=True)
                    val_metrics, validation_timings = validate(
                        policy,
                        val_dataloader,
                        tokenizer,
                        loss_fn,
                        step=total_steps + 1,
                        master_config=master_config,
                        logger=logger,
                    )


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
