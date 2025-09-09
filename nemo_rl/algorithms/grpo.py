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
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import (
    ClippedPGLossConfig,
    ClippedPGLossDataDict,
    ClippedPGLossFn,
)
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt, set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
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

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


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
    overlong_filtering: NotRequired[bool]


class GRPOSaveState(TypedDict):
    consumed_samples: int
    current_step: int
    current_epoch: int
    total_steps: int
    val_reward: NotRequired[
        float
    ]  # Optional field - may not be present during training


def _default_grpo_save_state() -> GRPOSaveState:
    return {
        "consumed_samples": 0,
        "current_step": 0,
        "current_epoch": 0,
        "total_steps": 0,
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
    dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
    processor: Optional[AutoProcessor] = None,
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
    dataloader = StatefulDataLoader(
        dataset,
        batch_size=grpo_config["num_prompts_per_step"],
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
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
        "reward_model" in env_configs and env_configs["reward_model"]["enabled"]
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
            assert inference_gpus_per_node > 0, (
                "policy.generation.colocated.resources.gpus_per_node must be > 0 "
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

    # vllm model loading prefers clean environment, initialize policy_generation before policy (#52 will fix this)
    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]  # Needed for vLLM

    if backend == "megatron":
        policy_generation = None
        print(
            f"  ✓ Using {backend} backend for generation with {policy_config['model_name']}",
            flush=True,
        )
    elif backend == "vllm":
        generation_config = cast(VllmConfig, generation_config)
        if generation_config["vllm_cfg"]["precision"] == "fp8":
            assert loss_config["use_importance_sampling_correction"] is True, (
                "Importance sampling must be enabled for vLLM FP8 generation for good convergence!"
            )

        policy_generation = VllmGeneration(
            cluster=inference_cluster, config=generation_config
        )
        # Worker groups are not initialized until the first call to run something on workergroups.
        # vllm 0.8 fails in initialization if its called in the first training step since it has no clean view of the GPU memory (HF is sharing the same memory).
        policy_generation.finish_generation()
        print(
            f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}",
            flush=True,
        )

    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
    else:
        weights_path = None
        optimizer_path = None

    if policy_config.get("megatron_cfg", {}).get("enabled", False):
        ## NOTE: this is equal to the total number of scheduler steps
        total_train_iters = min(grpo_config["max_num_steps"], len(dataloader))
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters

    policy = Policy(
        cluster=train_cluster,
        config=policy_config,
        tokenizer=tokenizer,
        processor=processor,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
    )

    # if it is not colocated inference, initialize collective communication for update weights
    if not colocated_inference:
        ip, port = train_cluster.get_master_address_and_port()
        print(f"Using ip: {ip}, port: {port} for collective communication", flush=True)
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
    print("=" * 60 + "\n", flush=True)

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
                f"[Refit] Split {total_num_keys} keys into {len(grouped_param_keys)} groups",
                flush=True,
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
    processor: Optional[AutoProcessor] = None,
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
    val_at_start = master_config["grpo"]["val_at_start"]
    val_period = master_config["grpo"]["val_period"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]

    # Run validation at the start if configured
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

    while current_epoch < max_num_epochs and total_steps < max_num_steps:
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")
        # Run grpo training (single-turn)
        batch: BatchedDataDict[DatumSpec]

        for batch in dataloader:
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
                        refit_policy_generation(
                            policy, policy_generation, colocated_inference, timer=timer
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        policy_generation.prepare_for_generation()

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
                    advantages = (rewards - baseline).unsqueeze(-1)

                    if master_config["grpo"]["normalize_rewards"]:
                        # don't sharpen the ones with no variation
                        zero_std_mask = std > 0
                        advantages[zero_std_mask] = (
                            advantages[zero_std_mask] / std.unsqueeze(-1)[zero_std_mask]
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

                is_last_step = (total_steps + 1 >= max_num_steps) or (
                    (current_epoch + 1 == max_num_epochs)
                    and (current_step + 1 == len(dataloader))
                )

                # Run validation if it's a validation step
                if val_period > 0 and (total_steps + 1) % val_period == 0:
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
                        torch.save(
                            dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # Logging
            # Log training data
            log_data = {"content": flat_messages["content"]}
            log_data["rewards"] = rewards.tolist()
            log_data["generation_logprobs"] = train_data["generation_logprobs"].tolist()
            log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()
            log_data["input_lengths"] = input_lengths.tolist()
            logger.log_batched_dict_as_jsonl(
                log_data, f"train_data_step{total_steps}.jsonl"
            )

            metrics = {
                "loss": train_results["loss"].numpy(),
                "reward": rewards.numpy(),
                "grad_norm": train_results["grad_norm"].numpy(),
                "mean_prompt_length": repeated_batch["length"].numpy(),
                "total_num_tokens": input_lengths.numpy(),
            }
            metrics.update(train_results["all_mb_metrics"])
            for k, v in metrics.items():
                if k in {
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

            print("\n📊 Training Results:")

            print(f"  • Loss: {metrics['loss']:.4f}")
            print(f"  • Avg Reward: {np.mean(rewards.numpy()):.4f}")
            print(
                f"  • Mean Generation Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}",
                flush=True,
            )
            if "total_flops" in train_results:
                total_tflops = (
                    train_results["total_flops"]
                    / timing_metrics["policy_training"]
                    / 1e12
                )
                num_ranks = train_results["num_ranks"]
                print(
                    f"  • Training FLOPS: {total_tflops:.2f} TFLOPS ({total_tflops / num_ranks:.2f} TFLOPS per rank)",
                    flush=True,
                )
                if "theoretical_tflops" in train_results:
                    theoretical_tflops = train_results["theoretical_tflops"]
                    print(
                        f"  • Training Model Floating Point Utilization: {100 * total_tflops / theoretical_tflops:.2f}%",
                        flush=True,
                    )
                    metrics["train_fp_utilization"] = total_tflops / theoretical_tflops

            print("\n⏱️  Timing:", flush=True)
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

            print(f"  • Total step time: {total_time:.2f}s", flush=True)

            # Display all other timing metrics
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)", flush=True)

            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1
            if should_save_by_timeout:
                break
            if total_steps >= max_num_steps:
                break

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

    return val_metrics, timing_metrics
