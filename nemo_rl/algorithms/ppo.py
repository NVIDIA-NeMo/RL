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
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

import numpy as np
import torch
from pydantic import BaseModel
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.advantage_estimator import (
    GeneralizedAdvantageEstimator,
    RawRewardAdvantageEstimator,
)
from nemo_rl.algorithms.grpo import (
    RewardScalingConfig,
    _should_log_nemo_gym_responses,
    _should_use_async_rollouts,
    _should_use_nemo_gym,
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
from nemo_rl.algorithms.utils import print_performance_metrics, set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.data.utils import load_dataloader_state
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_async_nemo_gym_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.generation.sglang.config import SGLangConfig
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.value import Value, ValueConfig
from nemo_rl.models.value.interfaces import ValueInterface
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig, print_message_log_samples
from nemo_rl.utils.memory_tracker import MemoryTracker
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


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


class PPOConfig(TypedDict):
    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_num_epochs: int
    max_num_steps: int
    max_rollout_turns: int
    val_period: int
    val_batch_size: int
    val_at_start: bool
    # Whether to run validation on the last training step. Setting this to True ensures the
    # final checkpoint has validation metrics, which is required for get_best_checkpoint_path().
    val_at_end: bool
    max_val_samples: int
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


class PPOSaveState(TypedDict):
    consumed_samples: int
    current_step: int
    current_epoch: int
    total_steps: int
    total_valid_tokens: int  # Track total number of non-padding tokens during training
    val_reward: NotRequired[
        float
    ]  # Optional field - may not be present during training


def _default_ppo_save_state() -> PPOSaveState:
    return {
        "consumed_samples": 0,
        "current_step": 0,
        "current_epoch": 0,
        "total_steps": 0,
        "total_valid_tokens": 0,
        "val_reward": -99999999.0,
    }


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
        tuple of (policy, policy_generation, value_model, clusters,
        dataloader, val_dataloader, loss_fn, value_loss_fn, logger,
        checkpointer, ppo_save_state, master_config).
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
        assert value_config["dtensor_cfg"]["enabled"]
        assert value_config["sequence_packing"]["enabled"] is False, (
            "Sequence packing is currently not supported for the DTensor PPO value model. "
            "See https://github.com/NVIDIA-NeMo/RL/issues/2951."
        )
        assert value_config["dtensor_cfg"]["context_parallel_size"] == 1, (
            "Context parallelism (CP>1) is currently not supported for the DTensor PPO value model. "
            "See https://github.com/NVIDIA-NeMo/RL/issues/2951."
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
    assert colocated_inference, (
        "PPO currently requires colocated generation (vLLM / SGLang sharing GPUs "
        "with the policy worker). Set policy.generation.colocated.enabled=true. "
        "Non-colocated PPO is not yet supported."
    )
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
    print(
        f"  ✓ Ray cluster for policy initialized with {policy_nodes} nodes",
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

    # Prepare checkpoint paths.  During critic warmup the policy is not saved,
    # so the directory may not exist even when resuming from a valid checkpoint.
    if last_checkpoint_path:
        _policy_weights = Path(last_checkpoint_path) / "policy" / "weights"
        _policy_optim = Path(last_checkpoint_path) / "policy" / "optimizer"
        weights_path = _policy_weights if _policy_weights.exists() else None
        optimizer_path = _policy_optim if _policy_optim.exists() else None
        if weights_path is None:
            print(
                f"  ⚠ Policy weights not found in checkpoint {last_checkpoint_path} "
                f"(likely saved during critic warmup). Using base model weights.",
                flush=True,
            )
        else:
            print(f"  ✓ Resuming policy from checkpoint: {weights_path}", flush=True)
    else:
        weights_path = None
        optimizer_path = None

    # train_iters is the total scheduler-tick budget. Each Megatron worker
    # ticks once per train() call (matching upstream main's per-rollout
    # convention), and PPO calls each worker's train() `ppo_epochs` times
    # per outer step. So total ticks = (outer steps) * ppo_epochs.
    # Scale train_iters accordingly so the configured warmup/decay horizon
    # matches the actual scheduler-step count.
    ppo_epochs = ppo_config["ppo_epochs"]
    if policy_config.get("megatron_cfg", {}).get("enabled", False):
        total_train_iters = (
            min(
                ppo_config["max_num_steps"],
                ppo_config["max_num_epochs"] * len(dataloader),
            )
            * ppo_epochs
        )
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters

    if value_config.get("megatron_cfg", {}).get("enabled", False):
        total_train_iters = (
            min(
                ppo_config["max_num_steps"],
                ppo_config["max_num_epochs"] * len(dataloader),
            )
            * ppo_epochs
        )
        value_config["megatron_cfg"]["train_iters"] = total_train_iters

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
        # Prepare checkpoint paths for value model. Mirror the policy's
        # .exists() probe (see weights_path/optimizer_path resolution above):
        # the previous run may not have had a value model, so the value sub-
        # directory of last_checkpoint_path may be missing.
        if last_checkpoint_path:
            _value_weights = Path(last_checkpoint_path) / "value" / "weights"
            _value_optim = Path(last_checkpoint_path) / "value" / "optimizer"
            value_weights_path = _value_weights if _value_weights.exists() else None
            value_optimizer_path = _value_optim if _value_optim.exists() else None
            if value_weights_path is None:
                print(
                    f"  ⚠ Value weights not found in checkpoint {last_checkpoint_path} "
                    f"(likely the previous run didn't have a value model). "
                    f"Initializing value model from base weights.",
                    flush=True,
                )
            else:
                print(
                    f"  ✓ Resuming value from checkpoint: {value_weights_path}",
                    flush=True,
                )
        else:
            value_weights_path = None
            value_optimizer_path = None

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
            Tuple of (policy_generation, policy)
        """
        # Initialize generation engine first so it claims its GPU memory
        # before policy/value workers are constructed; then policy, then value.
        print("  ⚙️  Initializing workers (colocated mode)", flush=True)

        policy_generation, generation_time = init_generation_fn()
        worker_init_timing_metrics[init_time_key] = generation_time

        policy, policy_time = init_policy()
        # Block until the policy worker's __init__ completes and offload to
        # CPU, freeing GPU for value model initialization. Policy will be
        # reloaded before the vLLM refit step below.
        policy.offload_after_refit()
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

    # Record when worker initialization completes (for calculating other setup time)
    worker_init_complete_time = time.perf_counter() - setup_start_time

    # print the node IP and GPU ID of the policy workers for debugging
    policy.print_node_ip_and_gpu_id()

    # Reload policy weights to GPU before refit (they may have been offloaded
    # during setup to free GPU for value model initialization).
    policy.prepare_for_training()

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
                        POLICY_GENERATION_STALE = False
                    else:
                        if colocated_inference:
                            policy.offload_after_refit()
                        policy_generation.prepare_for_generation()

                with timer.time("generation"):
                    if policy_generation is not None:
                        policy_generation.clear_logger_metrics()

                    if _should_use_nemo_gym(master_config):
                        generation_config = master_config.policy["generation"]
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
                        del nemo_gym_rollout_result

                        if not _should_log_nemo_gym_responses(master_config):
                            for key in list(rollout_metrics):
                                if "full_result" in key:
                                    rollout_metrics.pop(key)

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
                        mask=train_data["token_mask"],
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

                # PPO: Multiple training steps per rollout
                memory_tracker.snapshot_start_of_stage("Policy train", dir())
                for step in range(ppo_epochs):
                    print(
                        f"▶ Step {step + 1}/{ppo_epochs}...",
                        flush=True,
                    )

                    # Train value model first (critic before actor, matching veRL).
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
                            if step < ppo_epochs - 1:
                                policy.offload_after_refit()

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
                        POLICY_GENERATION_STALE = False
                    else:
                        if colocated_inference:
                            policy.offload_after_refit()
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
                flat_token_mask = flat_messages["token_loss_mask"]
                del flat_messages

                response_advantages = torch.masked_select(
                    flat_advantages, flat_token_mask.bool()
                )

                memory_tracker.snapshot_start_of_stage("Metrics", dir())
                if train_results is not None:
                    metrics = {
                        **metrics,
                        "loss": train_results["loss"].numpy(),
                        "grad_norm": train_results["grad_norm"].numpy(),
                    }
                    metrics.update(train_results["all_mb_metrics"])
                    if "moe_metrics" in train_results:
                        metrics.update(
                            {
                                f"moe/{k}": v
                                for k, v in train_results["moe_metrics"].items()
                            }
                        )

                # Extract critic metrics from value training results
                if value_results is not None:
                    value_mb_metrics = value_results.get("all_mb_metrics", {})
                    critic_metrics = {
                        "critic/grad_norm": value_results["grad_norm"].numpy(),
                        "critic/loss": value_results["loss"].numpy(),
                    }

                    for k, v in value_mb_metrics.items():
                        if k in {
                            "lr",
                            "wd",
                            "global_valid_seqs",
                            "global_valid_toks",
                            "grad_norm",
                        }:
                            critic_metrics["critic/" + k] = np.mean(v).item()
                        elif k in {"values_min"}:
                            critic_metrics["critic/" + k] = np.min(v).item()
                        elif k in {"values_max"}:
                            critic_metrics["critic/" + k] = np.max(v).item()
                        elif isinstance(v, (np.ndarray, list)):
                            critic_metrics["critic/" + k] = np.sum(v).item()
                        else:
                            raise ValueError(
                                f"Unknown metric for value don't know how to handle: {k}"
                            )

                    # Compute explained variance from sufficient statistics:
                    # EV = 1 - Var(returns - values) / Var(returns)
                    r_mean = critic_metrics.get("critic/returns_mean", 0)
                    v_mean = critic_metrics.get("critic/values_mean", 0)
                    r_sq = critic_metrics.get("critic/returns_sq_mean", 0)
                    res_sq = critic_metrics.get("critic/residual_sq_mean", 0)
                    var_returns = r_sq - r_mean**2
                    var_residual = res_sq - (r_mean - v_mean) ** 2
                    critic_metrics["critic/explained_var"] = 1.0 - var_residual / max(
                        var_returns, 1e-8
                    )

                    metrics.update(critic_metrics)
                metrics.update(
                    {
                        "reward": rewards.numpy(),
                        "mean_prompt_length": repeated_batch["length"].numpy(),
                        "total_num_tokens": input_lengths.numpy(),
                        "advantages/mean": torch.mean(response_advantages)
                        .detach()
                        .item()
                        if response_advantages.numel() > 0
                        else 0.0,
                        "advantages/max": torch.max(response_advantages).detach().item()
                        if response_advantages.numel() > 0
                        else 0.0,
                        "advantages/min": torch.min(response_advantages).detach().item()
                        if response_advantages.numel() > 0
                        else 0.0,
                    }
                )

                gen_step_metrics = {}
                if hasattr(policy_generation, "get_step_metrics"):
                    gen_step_metrics = policy_generation.get_step_metrics()
                metrics.update(gen_step_metrics)

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
                    elif isinstance(v, (np.ndarray, list)):
                        metrics[k] = np.sum(v).item()

                metrics.update(rollout_metrics)
                metrics["generation_logger_metrics"] = generation_logger_metrics
                if "global_valid_toks" in metrics:
                    total_valid_tokens += metrics["global_valid_toks"]

                ## Checkpointing
                consumed_samples += master_config.ppo["num_prompts_per_step"]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config.checkpointing["save_period"]
                    == 0
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

                        # Save policy FIRST, then value.  This ordering
                        # matters: the resume path uses the presence of
                        # policy/weights to decide whether the policy was
                        # trained.  By saving policy first we guarantee that
                        # if any policy dir exists in a finalized checkpoint
                        # (tmp→step rename is atomic), its contents are
                        # complete.  During critic warmup the policy optimizer
                        # has no state yet (master_param not initialized), so
                        # we skip it entirely — the resume path will fall back
                        # to the base model weights.
                        if total_steps >= policy_training_start_step:
                            policy.prepare_for_training()
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
                                checkpointing_cfg=master_config.checkpointing,
                            )
                            policy.offload_after_refit()
                        else:
                            print(
                                f"Skipping policy checkpoint (critic warmup: "
                                f"step {total_steps} < {policy_training_start_step})",
                                flush=True,
                            )

                        value_model.prepare_for_training()
                        value_model.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "value", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "value", "optimizer"
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
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # Logging
            memory_tracker.snapshot_start_of_stage("Logging", dir())

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )  # type: ignore

            del train_data

            print("\n📊 Training Results:")
            if train_results is not None:
                print(f"  • Policy Loss: {metrics.get('loss', 'N/A')}")
            if value_results is not None:
                print(f"  • Critic Loss: {metrics.get('critic/loss', 'N/A')}")
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
                memory_tracker.snapshot_start_of_stage("", dir())
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if total_steps >= max_num_steps:
                memory_tracker.snapshot_start_of_stage("", dir())
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

        current_epoch += 1
        current_step = 0


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

            val_batch, gen_metrics = run_multi_turn_rollout(
                policy_generation,
                val_batch,
                tokenizer,
                val_task_to_env,
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
