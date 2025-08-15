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
import os
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
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
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt, set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, rl_collate_fn
from nemo_rl.data.interfaces import (
    DatumSpec,
)
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
)
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
)
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import (
    GenerationInterface,
)
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

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


class GRPOConfig(TypedDict):
    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_num_steps: int
    max_rollout_turns: int
    normalize_rewards: bool
    use_leave_one_out_baseline: bool
    val_period: int
    val_batch_size: int
    val_at_start: bool
    max_val_samples: int
    seed: int
    reward_shaping: RewardShapingConfig
    dapo_batch_multiplier: NotRequired[int]


class GRPOSaveState(TypedDict):
    step: int
    val_reward: NotRequired[
        float
    ]  # Optional field - may not be present during training
    consumed_samples: int


def _default_grpo_save_state() -> GRPOSaveState:
    return {
        "step": 0,
        "val_reward": -99999999.0,
        "consumed_samples": 0,
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
    dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> tuple[
    ColocatablePolicyInterface,
    Optional[GenerationInterface],
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
    # Extract individual configs for easier access
    policy_config = master_config["policy"]
    generation_config = master_config["policy"]["generation"]
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
    train_batch_size = (
        grpo_config["num_prompts_per_step"]
        * grpo_config.get("dapo_batch_multiplier", 1)
        if grpo_config["use_dynamic_sampling"]
        else grpo_config["num_prompts_per_step"]
    )
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
    )

    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        dataloader.load_state_dict(dataloader_state_dict)

    print(f"  ✓ Training dataloader loaded with {len(dataset)} samples")

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
        )
        print(f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples")

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    colocated_inference = generation_config["colocated"]["enabled"]

    if colocated_inference:
        cluster = RayVirtualCluster(
            name="grpo_policy_cluster",
            bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
            * cluster_config["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=cluster_config["gpus_per_node"],
            max_colocated_worker_groups=1
            if generation_config["backend"] == "megatron"
            else 2,
        )
        train_cluster = cluster
        inference_cluster = cluster
        print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    else:
        assert generation_config["backend"] != "megatron", (
            "Non-colocated inference is not supported for Megatron generation backends. "
            "Please use vLLM backend for generation."
        )

        # train resources will be updated through overall and inference resources below
        train_gpus_per_node = cluster_config["gpus_per_node"]
        train_nodes = cluster_config["num_nodes"]

        inference_resources = generation_config["colocated"]["resources"]
        inference_gpus_per_node = inference_resources["gpus_per_node"]
        inference_nodes = inference_resources["num_nodes"]

        # validate and configure resources
        if cluster_config["num_nodes"] == 1:
            assert inference_gpus_per_node > 0, (
                "policy.generation.colocated.resources.gpus_per_node must be > 0 "
                "when cluster.num_nodes = 1 and inference is non-colocated, "
                f"but got {inference_gpus_per_node}."
            )
            assert inference_nodes is None or inference_nodes == 1, (
                "policy.generation.colocated.resources.num_nodes must be 1 or set to null "
                "when cluster.num_nodes = 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            inference_nodes = 1
            train_gpus_per_node -= inference_gpus_per_node
        else:
            assert inference_nodes > 0, (
                "policy.generation.colocated.resources.num_nodes must be > 0 "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            assert (
                inference_gpus_per_node is None
                or inference_gpus_per_node == cluster_config["gpus_per_node"]
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be equal to cluster.gpus_per_node or set to null "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got {inference_gpus_per_node}."
            )
            inference_gpus_per_node = cluster_config["gpus_per_node"]
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
            f"  ✓ Ray train cluster initialized with {train_nodes} nodes with {train_gpus_per_node} GPUs per node"
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
            f"  ✓ Ray inference cluster initialized with {inference_nodes} nodes with {inference_gpus_per_node} GPUs per node"
        )

    # ==========================
    #   Training and Inference
    # ==========================
    print("\n▶ Setting up model and training...")

    # vllm model loading prefers clean environment, initialize policy_generation before policy (#52 will fix this)
    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]  # Needed for vLLM

    if backend == "megatron":
        policy_generation = None
        print(
            f"  ✓ Using {backend} backend for generation with {policy_config['model_name']}"
        )
    elif backend == "vllm":
        generation_config = cast(VllmConfig, generation_config)
        policy_generation = VllmGeneration(
            cluster=inference_cluster, config=generation_config
        )
        # Worker groups are not initialized until the first call to run something on workergroups.
        # vllm 0.8 fails in initialization if its called in the first training step since it has no clean view of the GPU memory (HF is sharing the same memory).
        policy_generation.finish_generation()
        print(
            f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}"
        )

    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
    else:
        weights_path = None
        optimizer_path = None

    policy = Policy(
        cluster=train_cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
    )

    # if it is not colocated inference, initialize collective communication for update weights
    if not colocated_inference:
        ip, port = train_cluster.get_master_address_and_port()
        print(f"Using ip: {ip}, port: {port} for collective communication")
        # inference cluster + head node of the train cluster
        world_size = inference_nodes * inference_gpus_per_node + 1
        # init collective
        futures_train = policy.init_collective(ip, port, world_size)
        futures_inference = policy_generation.init_collective(ip, port, world_size)  # type: ignore
        # wait for all futures to complete
        ray.get(futures_train + futures_inference)

    # prepare refit info
    state_dict_info = policy.prepare_refit_info()
    policy_generation.prepare_refit_info(state_dict_info)

    loss_fn = ClippedPGLossFn(loss_config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        policy,
        policy_generation,
        (train_cluster, inference_cluster),
        dataloader,
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


def refit_policy_generation(
    policy: ColocatablePolicyInterface,
    policy_generation: GenerationInterface,
    colocated_inference: bool,
    _refit_buffer_size_gb: Optional[int] = None,
    timer: Optional[Timer] = None,
) -> None:
    """Refit the policy generation interface with the latest policy weights.

    Args:
        policy: The policy to provide weights to the inference engine.
        policy_generation: The inference engine to refit.
        _refit_buffer_size_gb: The size of the buffer to use for refitting.
            If it is None, the buffer size will be computed by the remaining memory.
            This parameter is primarily used for testing.
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
            grouped_param_keys = policy.prepare_weights_for_ipc(
                _refit_buffer_size_gb=_refit_buffer_size_gb
            )
            total_num_keys = sum(len(k) for k in grouped_param_keys)
            print(
                f"[Refit] Split {total_num_keys} keys into {len(grouped_param_keys)} groups"
            )
            # do update
            for keys in grouped_param_keys:
                ipc_handles = policy.get_weights_ipc_handles(keys)
                update_success = policy_generation.update_weights_from_ipc_handles(
                    ipc_handles
                )
                if not update_success:
                    break
        else:
            # update weights through nccl
            futures_train = policy.broadcast_weights_for_collective()
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


def dynamic_sampling(
    repeated_batch: BatchedDataDict[DatumSpec],
    std: torch.Tensor,
    baseline: torch.Tensor,
    num_gen_batches: int,
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
        num_gen_batches (int): Number of generation batches processed at the current step.
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

    # Store the baseline, std and total_reward for the current unfiltered batch.
    repeated_batch["baseline"] = baseline
    repeated_batch["std"] = std
    total_rewards = repeated_batch["total_reward"]
    # Dynamic sampling algorithm (used in DAPO algorithm)
    # This block implements dynamic sampling by selecting prompt groups with non-zero std.
    # If sampled prompts (with non-zero std) are fewer than num_prompts_per_step * num_generations_per_prompt, continue sampling until max_num_gen_batches is reached.
    if master_config["grpo"]["use_dynamic_sampling"]:
        with timer.time("dynamic_sampling"):
            # split std into chunks of size num_generations_per_prompt (prompt groups)
            std_chunks_per_prompt = std.split(
                master_config["grpo"]["num_generations_per_prompt"]
            )
            keep_prompt_indices = []
            selected_std_chunks = []
            for chunk_idx, chunk in enumerate(std_chunks_per_prompt):
                chunk_length = chunk.shape[0]
                if torch.nonzero(chunk).shape[0] == chunk_length:
                    chunk_prompt_indices = [
                        chunk_idx * chunk_length + idx for idx in range(chunk_length)
                    ]
                    keep_prompt_indices.extend(chunk_prompt_indices)
                    selected_std_chunks.append(chunk)

            # Only select the inputs that have non-zero std
            # total_reward is already a part of repeated_batch so we don't need to add it again
            repeated_batch = repeated_batch.select_indices(keep_prompt_indices)
            repeated_batch["std"] = std[keep_prompt_indices]
            repeated_batch["baseline"] = baseline[keep_prompt_indices]

            # Store filtered and total rewards to track them separately
            filtered_rewards = repeated_batch["total_reward"]
            repeated_batch["total_reward"] = total_rewards
            repeated_batch["filtered_reward"] = filtered_rewards

            # Store the total_reward for the current filtered batch.
            # If none of the prompts in current batch have non-zero std, repeated_batch.size will be 0.
            # In this case, the current batch will be ignored and the next batch will be processed and we generate responses for it.
            if repeated_batch.size > 0:
                # Concatenate the previous partially filled batch with the current batch. This serves as a cache to store and collect the prompts with non-zero std.
                # This is used in the next iteration when the current batch is not enough to fill the buffer.
                batch_cache = (
                    repeated_batch
                    if batch_cache is None
                    else BatchedDataDict.from_batches([batch_cache, repeated_batch])
                )
                repeated_batch = batch_cache

            generation_sample_buffer_size = repeated_batch.size
            train_prompts_size = (
                master_config["grpo"]["num_prompts_per_step"]
                * master_config["grpo"]["num_generations_per_prompt"]
            )

            # If the generation samples size is smaller than a fixed threshold (train_prompts_size), keep generating by processing the next batch
            if generation_sample_buffer_size < train_prompts_size:
                max_num_gen_batches = master_config["grpo"].get(
                    "max_num_gen_batches", 0
                )
                if max_num_gen_batches <= 0 or num_gen_batches <= max_num_gen_batches:
                    print(
                        f"Generation sample buffer size: {generation_sample_buffer_size} is smaller than train_prompts_size: {train_prompts_size}. Processed {num_gen_batches} batches so far out of {max_num_gen_batches}."
                    )
                    is_batch_complete = False
                else:
                    raise ValueError(
                        f"Dynamic sampling has reached the maximum allowed number of batches ({max_num_gen_batches}). Consider evaluating the complexity of your data or adjusting the num_prompts_per_step or num_generations_per_prompt parameters to enhance the diversity of the samples."
                    )
            else:
                #  Slice the batch, rewards, baselines and std to ensure batch size is train_prompts_size
                repeated_batch = repeated_batch.slice(0, train_prompts_size)

    return repeated_batch, is_batch_complete, batch_cache


def grpo_train(
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
) -> None:
    """Run GRPO training algorithm."""
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    NEED_REFIT = True
    # If policy_generation is None, use the policy as the generation interface (megatron framework backend)
    if policy_generation is None:
        policy_generation = policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running
    assert policy_generation is not None  # for mypy type check

    # common config/state itmes
    step = grpo_save_state["step"]
    consumed_samples = grpo_save_state["consumed_samples"]
    val_period = master_config["grpo"]["val_period"]
    val_at_start = master_config["grpo"]["val_at_start"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]

    # Run validation at the start if configured
    if val_at_start and step == 0:
        print("\n🔍 Running initial validation...")
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
        logger.log_metrics(val_metrics, step, prefix="validation")
        logger.log_metrics(validation_timings, step, prefix="timing/validation")

    # batch cache is used for DAPO. We store prompts with non-zero standard deviation in this cache.
    batch_cache: BatchedDataDict[DatumSpec] = None
    # This is the number of batches we processed so far at each step to generate responses whose std is non-zero. Maximum threshold is set by max_num_gen_batches. Used in the case of dynamic sampling.
    num_gen_batches = 0
    # Run grpo/dapo training loop (single-turn)
    for batch in dataloader:
        print(
            f"\n{'=' * 25} Step {step + 1}/{min(len(dataloader), master_config['grpo']['max_num_steps'])} {'=' * 25}"
        )
        maybe_gpu_profile_step(policy, step + 1)
        if policy != policy_generation:
            maybe_gpu_profile_step(policy_generation, step + 1)
        val_metrics, validation_timings = None, None

        with timer.time("total_step_time"):
            # Prepare batch
            print("▶ Preparing batch...")
            with timer.time("data_processing"):
                # Repeat batch items
                repeated_batch: BatchedDataDict[DatumSpec] = batch.repeat_interleave(
                    master_config["grpo"]["num_generations_per_prompt"]
                )
                # Convert LLMMessageLogType to FlatMessagesType for generation
                batched_flat, input_lengths = batched_message_log_to_flat_message(
                    repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                )
                input_ids = batched_flat["token_ids"]

            # Generate responses - this updates the LLMMessageLogType in repeated_batch
            print(f"▶ Generating responses for batch of size {repeated_batch.size}...")
            with timer.time("prepare_for_generation"):
                if NEED_REFIT and POLICY_GENERATION_STALE:
                    refit_policy_generation(
                        policy, policy_generation, colocated_inference, timer=timer
                    )
                    POLICY_GENERATION_STALE = False
                else:
                    policy_generation.prepare_for_generation()

            num_gen_batches += 1
            with timer.time("generation"):
                # Use async rollouts if vLLM async engine is enabled
                if _should_use_async_rollouts(master_config):
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
                        max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
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
                        max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                        greedy=False,
                    )
                policy_generation.finish_generation()

            # Calculate rewards & advantages
            print("▶ Processing rewards...")
            with timer.time("reward_calculation"):
                # Extract rewards from final_batch
                rewards = repeated_batch["total_reward"]

                print("▶ Computing advantages...")
                baseline, std = calculate_baseline_and_std_per_prompt(
                    input_ids,
                    rewards,
                    torch.ones_like(rewards),
                    leave_one_out_baseline=master_config["grpo"][
                        "use_leave_one_out_baseline"
                    ],
                )

                # Apply dynamic sampling to filter prompts with non-zero std (DAPO algorithm)
                repeated_batch, is_batch_complete, batch_cache = dynamic_sampling(
                    repeated_batch,
                    std,
                    baseline,
                    num_gen_batches,
                    master_config,
                    timer,
                    batch_cache,
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

                # Process rewards with custom reward function
                if master_config["grpo"]["reward_shaping"]["enabled"]:
                    rewards = apply_reward_shaping(
                        repeated_batch, rewards, master_config["grpo"]["reward_shaping"]
                    )

                advantages = (rewards - baseline).unsqueeze(-1)

                if master_config["grpo"]["normalize_rewards"]:
                    # don't sharpen the ones with no variation
                    zero_std_mask = std > 0
                    advantages[zero_std_mask] = (
                        advantages[zero_std_mask] / std.unsqueeze(-1)[zero_std_mask]
                    )

            with timer.time("data_processing"):
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
                train_data.to("cpu")

            print("▶ Preparing for logprob inference...")
            with timer.time("logprob_inference_prep"):
                policy.prepare_for_lp_inference()

            print("▶ Computing logprobs...")
            with timer.time("policy_and_reference_logprobs"):
                fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]
                reference_logprobs = policy.get_reference_policy_logprobs(train_data)[
                    "reference_logprobs"
                ]
                train_data["prev_logprobs"] = fprop_logprobs
                train_data["reference_policy_logprobs"] = reference_logprobs

            print("▶ Preparing for training...")
            with timer.time("training_prep"):
                policy.prepare_for_training()  # set model train and reload optim to GPU
                POLICY_GENERATION_STALE = True

            print("▶ Training policy...")
            with timer.time("policy_training"):
                train_results = policy.train(train_data, loss_fn)

            is_last_step = step + 1 == min(
                master_config["grpo"]["max_num_steps"], len(dataloader)
            )

            # Run validation if it's a validation step
            if val_period > 0 and (step + 1) % val_period == 0:
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

            ## Checkpointing
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

                grpo_save_state["step"] = step + 1
                if val_metrics is not None:
                    grpo_save_state["val_reward"] = val_metrics["accuracy"]
                elif "val_reward" in grpo_save_state:
                    del grpo_save_state["val_reward"]
                grpo_save_state["consumed_samples"] = consumed_samples

                if master_config["checkpointing"]["metric_name"] is not None:
                    if (
                        master_config["checkpointing"]["metric_name"]
                        not in grpo_save_state
                    ):
                        warnings.warn(
                            f"You asked to save checkpoints based on {master_config['checkpointing']['metric_name']} but the metric is not found in the save state. "
                            "Saving most recent k checkpoints instead."
                        )
                        master_config["checkpointing"]["metric_name"] = None

                with timer.time("checkpointing"):
                    print(f"Saving checkpoint for step {step + 1}...")
                    checkpoint_path = checkpointer.init_tmp_checkpoint(
                        step + 1, grpo_save_state, master_config
                    )
                    policy.save_checkpoint(
                        weights_path=os.path.join(checkpoint_path, "policy", "weights"),
                        optimizer_path=os.path.join(
                            checkpoint_path, "policy", "optimizer"
                        ),
                        tokenizer_path=os.path.join(
                            checkpoint_path, "policy", "tokenizer"
                        ),
                    )
                    torch.save(
                        dataloader.state_dict(),
                        os.path.join(checkpoint_path, "train_dataloader.pt"),
                    )
                    checkpointer.finalize_checkpoint(checkpoint_path)

        # Logging
        # Log training data
        log_data = {"content": flat_messages["content"]}
        log_data["generation_logprobs"] = train_data["generation_logprobs"].tolist()
        log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()
        log_data["input_lengths"] = input_lengths.tolist()
        metrics = {
            "loss": train_results["loss"].numpy(),
            "grad_norm": train_results["grad_norm"].numpy(),
            "mean_prompt_length": repeated_batch["length"].numpy(),
            "total_num_tokens": input_lengths.numpy(),
        }
        # Store filtered and total rewards to track them separately in the case of dynamic sampling
        if master_config["grpo"]["use_dynamic_sampling"]:
            log_data["filtered_rewards"] = rewards.tolist()
            log_data["rewards"] = repeated_batch["total_reward"].tolist()
            metrics["reward"] = repeated_batch["total_reward"].numpy()
            metrics["filtered_reward"] = rewards.numpy()
        else:
            log_data["rewards"] = rewards.tolist()
            metrics["reward"] = rewards.numpy()

        logger.log_batched_dict_as_jsonl(log_data, f"train_data_step{step}.jsonl")

        metrics.update(train_results["all_mb_metrics"])
        for k, v in metrics.items():
            if k in {
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

        timing_metrics: dict[str, float] = timer.get_timing_metrics(reduction_op="sum")  # type: ignore
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
                step + 1,
                name="train/token_mult_prob_error_plot_sample",
            )

        print("\n📊 Training Results:")

        print(f"  • Loss: {metrics['loss']:.4f}")
        if master_config["grpo"]["use_dynamic_sampling"]:
            print(f"  • Avg Filtered Reward: {np.mean(rewards.numpy()):.4f}")
            print(
                f"  • Avg Total Reward: {np.mean(repeated_batch['total_reward'].numpy()):.4f}"
            )
        else:
            print(f"  • Avg Reward: {np.mean(rewards.numpy()):.4f}")
        print(
            f"  • Mean Generation Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}"
        )
        if "total_flops" in train_results:
            total_tflops = (
                train_results["total_flops"] / timing_metrics["policy_training"] / 1e12
            )
            num_ranks = train_results["num_ranks"]
            print(
                f"  • Training FLOPS: {total_tflops:.2f} TFLOPS ({total_tflops / num_ranks:.2f} TFLOPS per rank)"
            )
            if "theoretical_tflops" in train_results:
                theoretical_tflops = train_results["theoretical_tflops"]
                print(
                    f"  • Training Model Floating Point Utilization: {100 * total_tflops / theoretical_tflops:.2f}%"
                )
                metrics["train_fp_utilization"] = total_tflops / theoretical_tflops

        print("\n⏱️  Timing:")
        # Display total time first, separately
        total_time = timing_metrics.get("total_step_time", 0)

        total_num_gpus = (
            master_config["cluster"]["num_nodes"]
            * master_config["cluster"]["gpus_per_node"]
        )
        metrics.update(
            {
                "tokens_per_sec_per_gpu": metrics["total_num_tokens"]
                / total_time
                / total_num_gpus
            }
        )

        print(f"  • Total step time: {total_time:.2f}s")

        # Display all other timing metrics
        for k, v in sorted(
            timing_metrics.items(), key=lambda item: item[1], reverse=True
        ):
            if k != "total_step_time":
                percent = (v / total_time * 100) if total_time > 0 else 0
                print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

        logger.log_metrics(metrics, step + 1, prefix="train")
        logger.log_metrics(timing_metrics, step + 1, prefix="timing/train")

        # Reset the batch and set num_gen_batches to 0
        batch_cache = None
        num_gen_batches = 0

        timer.reset()
        step += 1
        if step >= master_config["grpo"]["max_num_steps"]:
            break


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
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return {}, {}

    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...")

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

            # Generate responses (updates the LLMMessageLogType in batch_with_msg_logs)
            # Use async rollouts if vLLM async engine is enabled
            if _should_use_async_rollouts(master_config):
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
            rewards = val_batch["total_reward"]

            total_rewards.extend(rewards.tolist())
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
        accuracy = sum(total_rewards) / len(total_rewards)
        avg_length = sum(total_lengths) / len(total_lengths)

        val_metrics = {
            "accuracy": accuracy,
            "avg_length": avg_length,
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
            print("  ⚠️ Continuing validation without displaying samples...")

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    # Print summary of validation results
    print("\n📊 Validation Results:")
    print(f"    • Accuracy: {accuracy:.4f}")
    print(f"    • Average response length: {avg_length:.1f} tokens")
    print(f"    • Samples processed: {len(total_rewards)}")

    # Print timing information
    print("\n  ⏱️  Validation Timing:")
    validation_time = timing_metrics.get("total_validation_time", 0)
    print(f"    • Total validation time: {validation_time:.2f}s")

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics
