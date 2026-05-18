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
import functools
import copy
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
import ray.util.state
import torch
from ray.actor import ActorProxy
from ray.util.placement_group import (
    placement_group,
    remove_placement_group,
)
from ray.util.scheduling_strategies import (
    NodeAffinitySchedulingStrategy,
    PlacementGroupSchedulingStrategy,
)
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.advantage_estimator import (
    GRPOAdvantageEstimator,
    ReinforcePlusPlusAdvantageEstimator,
)
from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import (
    ClippedPGLossConfig,
    DWRLLossDataDict,
    DWRLLossFn,
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
from nemo_rl.data.collate_fn import rl_collate_fn, preference_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.data.utils import extract_necessary_env_names
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_PORT_RANGE_HIGH,
    DEFAULT_PORT_RANGE_LOW,
    ClusterConfig,
    RayClusterSetupHelper,
    RayVirtualCluster,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.nemo_gym import (
    NemoGym,
    NemoGymConfig,
    get_nemo_gym_uv_cache_dir,
    get_nemo_gym_venv_dir,
)
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_async_nemo_gym_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.generation.sglang import SGLangConfig, SGLangGeneration
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
from nemo_rl.utils.memory_tracker import MemoryTracker
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer
from nemo_rl.utils.venvs import create_local_venv_on_each_node
from nemo_rl.algorithms.grpo import (
    RewardScalingConfig,
    AsyncGRPOConfig,
    AdvEstimatorConfig,
    GRPOConfig,
    GRPOSaveState,
    _default_grpo_save_state,
    GRPOLoggerConfig,
    MasterConfig,
    extract_initial_prompt_messages,
    dynamic_sampling,
    scale_rewards,
    _should_use_async_rollouts,
    _should_use_nemo_gym,
    _should_log_nemo_gym_responses,
    _create_advantage_estimator,
    _extract_prompt_only_messages,
    refit_policy_generation,
    _log_mixed_rewards_and_advantages_information,
    compute_and_apply_seq_logprob_error_masking,
)

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)

# ===============================================================================
# Core Algorithm Functions
# ===============================================================================
def build_thought_and_answer_masks(
    seq_len: int,
    input_lengths: torch.Tensor,
    generation_lengths: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-token masks separating the thought and verdict parts.

    Both masks are in *unshifted* space (same indexing as input_ids / logprobs
    returned by the generation backend).  The loss function shifts them by 1
    internally to align with next-token logprobs.

    Mask conventions:
      • thought_mask[b, t] = 1  for generated tokens at positions
            input_lengths[b] ≤ t < (input_lengths[b] + generation_lengths[b] − 1)
        i.e. all generated tokens except the final verdict.
      • answer_mask[b, t]  = 1  at the verdict position only:
            t = input_lengths[b] + generation_lengths[b] − 1

    Args:
        seq_len:           Total (padded) sequence length.
        input_lengths:     [B] prompt lengths (number of input tokens).
        generation_lengths:[B] number of tokens generated (including verdict).
        device:            Target device.

    Returns:
        (thought_mask, answer_mask), each of shape [B, seq_len].
    """
    B = input_lengths.shape[0]
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)

    verdict_pos = generation_lengths - 1
    thought_mask = (positions < input_lengths.unsqueeze(-1)).float()

    answer_mask = (positions == verdict_pos.unsqueeze(-1)).float()

    return thought_mask, answer_mask

# ===============================================================================
# Training & Validation
# ===============================================================================


def dwrl_train_pairwise(
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
    memory_tracker = MemoryTracker()

    kv_scales_cache = None  # Cache reused for computed kv scales

    NEED_REFIT = True
    # If policy_generation is None, use the policy as the generation interface (megatron framework backend)
    if policy_generation is None:
        policy_generation = policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running
    assert policy_generation is not None  # for mypy type check

    if master_config["grpo"].get("skip_reference_policy_logprobs_calculation"):
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
    val_at_end = master_config["grpo"]["val_at_end"]
    val_period = master_config["grpo"]["val_period"]

    to_compute_kl = master_config["loss_fn"]["reference_policy_kl_penalty"] > 0
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
            policy,
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
        # batch cache is used for DAPO. We store prompts with non-zero standard deviation in this cache.
        batch_cache: BatchedDataDict[DatumSpec] = None
        # This is the number of batches we processed so far at each step to generate responses whose std is non-zero. Maximum threshold is set by dynamic_sampling_max_gen_batches. Used in the case of dynamic sampling.
        dynamic_sampling_num_gen_batches = 0

        # Run grpo/dapo training loop (single-turn)
        for batch in dataloader:
            # A central place to store logging data that won't be deleted until the loop ends
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
                    with timer.time("repeat_interleaving"):
                        # Repeat batch items
                        repeated_batch: BatchedDataDict[DatumSpec] = (
                            batch.repeat_interleave(
                                master_config["grpo"]["num_generations_per_prompt"]
                            )
                        )
                    with timer.time("batch_message"):
                        # Convert LLMMessageLogType to FlatMessagesType for generation
                        batched_flat, input_lengths = batched_message_log_to_flat_message(
                            repeated_batch["message_log"],
                            pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        )
                        input_ids = batched_flat["token_ids"]

                # Generate responses - this updates the LLMMessageLogType in repeated_batch
                memory_tracker.snapshot_start_of_stage("Generation", dir())
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
                            calibration_data = BatchedDataDict[DWRLLossDataDict](
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
                if dynamic_sampling_num_gen_batches == 1 and hasattr(
                    policy_generation, "snapshot_step_metrics"
                ):
                    policy_generation.snapshot_step_metrics()
                with timer.time("generation"):
                    # Clear logger metrics for each generation step
                    if policy_generation is not None:
                        policy_generation.clear_logger_metrics()
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
                            # GenRM compare config
                            use_genrm_compare=master_config["env"].get(
                                "use_genrm_compare", False
                            ),
                            num_generations_per_prompt=master_config["grpo"][
                                "num_generations_per_prompt"
                            ],
                            genrm_compare_server_name=master_config["env"].get(
                                "genrm_compare_server_name", "genrm_compare"
                            ),
                            genrm_agent_names=master_config["env"].get(
                                "genrm_agent_names", ["genrm_simple_agent"]
                            ),
                            master_config=master_config
                        )
                        input_ids = nemo_gym_rollout_result.input_ids
                        repeated_batch = nemo_gym_rollout_result.final_batch
                        rollout_metrics = nemo_gym_rollout_result.rollout_metrics
                        del nemo_gym_rollout_result

                        # NeMo Gym responses can be very large and expensive to log. Here we have logic to opt-in to logging.
                        if not _should_log_nemo_gym_responses(master_config):
                            for key in list(rollout_metrics):
                                if "full_result" in key:
                                    rollout_metrics.pop(key)

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
                    # Collect generation logger metrics for performance reporting after each generation step
                    # inflight batch sizes and num pending samples are collected from each worker
                    if policy_generation is not None:
                        generation_logger_metrics = (
                            policy_generation.get_logger_metrics()
                        )

                    metrics_logging_data["mean_gen_tokens_per_sample"] = (
                        rollout_metrics["mean_gen_tokens_per_sample"]
                    )
                    logger.log_metrics(rollout_metrics, total_steps + 1, prefix="train")

                repeated_batch = scale_rewards(
                    repeated_batch, master_config["grpo"]["reward_scaling"]
                )
                # Process rewards with custom reward function
                if master_config["grpo"]["reward_shaping"]["enabled"]:
                    repeated_batch = apply_reward_shaping(
                        repeated_batch, master_config["grpo"]["reward_shaping"]
                    )
                
                # Inject the second user turn for DWRL
                new_msg_log = []
                for outer in repeated_batch['message_log']:
                    p = []
                    for inner in outer:
                        if inner["role"] == "environment":
                            continue
                        #pp = {"role": inner["role"], "content": inner['content'], "token_ids": inner["token_ids"]}
                        #if "generation_logprobs" in inner:
                        #    pp["generation_logprobs"] = inner["generation_logprobs"]
                        pp = copy.deepcopy(inner)
                        p.append(pp)
                    px = {"role": "user", "content": master_config["grpo"]["dwrl"]["bt_prompt"]}
                    npx = tokenizer.apply_chat_template([px], tokenize=False, add_generation_prompt=True, add_special_tokens=False, enable_thinking=False)
                    px['content'] = npx.replace("<|im_start|>system\n<|im_end|>\n", "").replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", "")# + "\n\n</think>\n\n"
                    px["token_ids"] = tokenizer(px['content'], return_tensors="pt")["input_ids"][0]
                    p.append(px)
                    new_msg_log.append(p)

                new_fmt_list = []
                for idx, (msg_log, idx, extra_env_info, loss_mx) in enumerate(zip(new_msg_log, repeated_batch['idx'], repeated_batch['extra_env_info'], repeated_batch['loss_multiplier'])):
                    length = sum(len(m["token_ids"]) for m in msg_log)
                    output = {"message_log": msg_log, "length": length, "extra_env_info": extra_env_info, "loss_multiplier": loss_mx, "idx": idx, "task_name": "genrm_dwrl"}
                    new_fmt_list.append(output)
                
                # Collate the new repeated_batch
                repeated_batch_2 = rl_collate_fn(new_fmt_list)
                
                # if you need to do another round of generations to check the output
                '''
                if colocated_inference:
                    policy.offload_after_refit()  # unload optimizer to make space for generation
                policy_generation.prepare_for_generation()
                policy_generation.clear_logger_metrics()
                repeated_batch_2_chk, _ = run_multi_turn_rollout(
                    policy_generation=policy_generation,
                    input_batch=repeated_batch_2,
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
                print("*** REPEATED_BATCH_2_CHK_MSG_LOG_CHECK: ", [x['content'] for x in repeated_batch_2_chk['message_log'][0]], flush=True)
                raise RuntimeError("all stop")
                '''
                
                # Calculate rewards & advantages
                memory_tracker.snapshot_start_of_stage("Processing rewards", dir())
                print("▶ Processing rewards...,", flush=True)
                with timer.time("reward_calculation"):
                    # Extract rewards from final_batch
                    rewards = repeated_batch["total_reward"]

                    print("▶ Computing advantages...", flush=True)
                    if master_config["grpo"].get("calculate_advantages_on_gpu"):
                        print("Computing advantages on GPU!")
                        # Just fix the device id for now
                        device_id = 0
                        baseline, std = calculate_baseline_and_std_per_prompt(
                            input_ids.cuda(device_id),
                            rewards.cuda(device_id),
                            torch.ones_like(rewards).cuda(device_id),
                            leave_one_out_baseline=master_config["grpo"][
                                "use_leave_one_out_baseline"
                            ],
                        )
                        baseline = baseline.cpu()
                        std = std.cpu()
                    else:
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
                    gen_step_metrics = {}
                    if hasattr(policy_generation, "get_step_metrics"):
                        gen_step_metrics = policy_generation.get_step_metrics()
                    advantages = (rewards - baseline).unsqueeze(-1)

                    # Save baseline for logging (before deletion)
                    baseline_for_log = baseline.clone()

                    # Extract prompt-only messages for advantage estimation
                    prompt_only_message_logs = _extract_prompt_only_messages(
                        repeated_batch["message_log"]
                    )
                    prompt_batched_flat, _ = batched_message_log_to_flat_message(
                        prompt_only_message_logs,
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    )
                    prompt_ids_for_adv = prompt_batched_flat["token_ids"]
                    del prompt_only_message_logs
                    del prompt_batched_flat
                    del input_ids
                    del baseline
                    del std

                with timer.time("data_processing"):
                    with timer.time("overlong_filter"):
                        use_overlong_filtering = master_config["grpo"]["overlong_filtering"]
                        if use_overlong_filtering:
                            loss_multiplier = repeated_batch_2["loss_multiplier"].clone()
                            truncated = repeated_batch_2["truncated"]

                            if isinstance(truncated, list):
                                truncated = torch.tensor(truncated, dtype=torch.bool)

                            loss_multiplier[truncated] = 0
                            repeated_batch_2["loss_multiplier"] = loss_multiplier

                    with timer.time("add_loss_mask"):
                        for i, message_log in enumerate(repeated_batch_2["message_log"]):
                            for j, message in enumerate(message_log):
                                token_ids = message["token_ids"]
                                is_assistant = message["role"] == "assistant" and "generation_logprobs" in message
                                if message["role"] == "assistant" and "generation_logprobs" not in message:
                                    raise RuntimeError("assistant message with no generation_logprobs!")

                                if is_assistant and j == len(message_log) - 2:
                                    message["token_loss_mask"] = torch.ones_like(token_ids)
                                else:
                                    message["token_loss_mask"] = torch.zeros_like(token_ids)

                                if "generation_logprobs" not in message:
                                    message["generation_logprobs"] = torch.zeros_like(
                                        token_ids, dtype=torch.float32
                                    )

                    with timer.time("message_to_flat"):
                        # Convert updated LLMMessageLogType to FlatMessagesType for training
                        flat_messages, input_lengths = batched_message_log_to_flat_message(
                            repeated_batch_2["message_log"],
                            pad_value_dict={"token_ids": tokenizer.pad_token_id},
                            make_sequence_length_divisible_by=master_config["policy"][
                                "make_sequence_length_divisible_by"
                            ],
                        )
                    
                    yes_position = tokenizer.encode(master_config["grpo"]["dwrl"]["score_token"])[0]
                    no_position = tokenizer.encode(master_config["grpo"]["dwrl"]["opposite_token"])[0]
                    yes_tensor = torch.tensor([yes_position], device=flat_messages["token_ids"].device).long()
                    list_with_yes = [torch.cat([flat_messages["token_ids"][idx][:input_lengths[idx]], yes_tensor, flat_messages["token_ids"][idx][input_lengths[idx]:]], dim=-1) for idx in range(len(input_lengths))]
                    input_ids_with_yes = torch.stack(list_with_yes, dim=0)
                    flat_token_mask = torch.cat([flat_messages["token_loss_mask"], torch.zeros(flat_messages["token_loss_mask"].shape[0], 1)], dim=-1)

                    with timer.time("flatten"):
                        # Create training data from flattened messages
                        # Note: advantages will be computed and added after logprobs are available
                        train_data = BatchedDataDict[DWRLLossDataDict](
                            {
                                "input_ids": input_ids_with_yes,
                                "input_lengths": input_lengths + 1,
                                "generation_logprobs": torch.cat([flat_messages["generation_logprobs"], torch.ones(flat_messages["generation_logprobs"].shape[0], 1) * -0.69315], dim=-1),
                                "token_mask": flat_token_mask,
                                "sample_mask": repeated_batch_2["loss_multiplier"],
                                "metadata": repeated_batch["extra_env_info"],
                                "no_position": torch.ones_like(repeated_batch_2["loss_multiplier"]).long() * no_position,
                            }
                        )
                    # this will be mini-batched inside the policy, so maintain the packed multimodal structure
                    # This is also used to populate part of the downstream logprob calculation data
                    with timer.time("multimodal_dict"):
                        extra_multimodal_data = flat_messages.get_multimodal_dict(
                            as_tensors=False
                        )
                        train_data.update(extra_multimodal_data)
                        train_data.to("cpu")

                    metrics_logging_data["content"] = flat_messages["content"]

                memory_tracker.snapshot_start_of_stage("Computing logprobs", dir())
                # Skip prev_logprobs computation when force_on_policy_ratio=True
                # unless seq_logprob_error_threshold is set (which requires prev_logprobs)
                seq_logprob_error_threshold = master_config["grpo"].get(
                    "seq_logprob_error_threshold", None
                )
                #force_on_policy_ratio = master_config["loss_fn"].get("force_on_policy_ratio", False)
                skip_prev_logprobs = False
                print("▶ Preparing for logprob inference...", flush=True)
                with timer.time("logprob_inference_prep"):
                    policy.prepare_for_lp_inference()

                print("▶ Computing logprobs...", flush=True)
                with timer.time("policy_and_reference_logprobs"):
                    # Custom create this logprob_data so we avoid Ray comm overheads sending unused data to workers.
                    logprob_data = BatchedDataDict[DWRLLossDataDict](
                        {
                            "input_ids": train_data["input_ids"],
                            "input_lengths": train_data["input_lengths"],
                            **extra_multimodal_data,
                        }
                    )
                    if not skip_prev_logprobs:
                        train_data["prev_logprobs"] = policy.get_logprobs(
                            logprob_data, timer=timer
                        )["logprobs"]
                    else:
                        train_data["prev_logprobs"] = torch.zeros_like(train_data["generation_logprobs"])

                    if to_compute_kl and not master_config["grpo"].get(
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
                
                # Calculate rewards & advantages
                memory_tracker.snapshot_start_of_stage("Processing DWRL rewards", dir())
                print("▶ Processing DWRL rewards...,", flush=True)
                with timer.time("reward_calculation"):
                    # Extract rewards from final_batch
                    final_logprobs = train_data["prev_logprobs"].gather(-1, input_lengths.unsqueeze(-1)).squeeze(-1)
                    #bt_probs = final_logprobs.exp()
                    
                    _, input_lengths_rb_1 = batched_message_log_to_flat_message(
                        [[y for y in x if y['role'] != 'environment'] for x in repeated_batch["message_log"]],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"]["make_sequence_length_divisible_by"],
                    )

                    thought_mask, answer_mask = build_thought_and_answer_masks(
                        seq_len=input_ids_with_yes.shape[1],
                        input_lengths=input_lengths_rb_1,
                        generation_lengths=input_lengths + 1,
                        device=torch.device('cpu'),
                    )
                    
                    assert torch.equal(answer_mask.argmax(dim=-1), input_lengths), "answer_mask does not correspond to input_lengths position"
                    
                    ### calculate bt accuracy
                    sample_mask = repeated_batch_2["loss_multiplier"]
                    gt = torch.tensor([x['preference'] for x in train_data['metadata']], dtype=torch.int16, device=final_logprobs.device)
                    if sample_mask.sum() > 0:
                        bt_accuracy = (torch.where(final_logprobs.detach().exp() >= 0.5, 0, 1) == gt).sum().item() / sample_mask.sum().item()
                    else:
                        bt_accuracy = 0.0
                    
                    train_data["thought_mask"] = thought_mask
                    train_data["answer_mask"] = answer_mask

                # Seq-level logprob error metrics/masking require real prev_logprobs
                max_seq_mult_prob_error = 0.0
                mean_seq_mult_prob_error = 0.0
                min_seq_mult_prob_error = 0.0
                max_seq_mult_prob_error_after_mask = 0.0
                mean_seq_mult_prob_error_after_mask = 0.0
                min_seq_mult_prob_error_after_mask = 0.0
                num_masked_seqs = 0
                masked_correct_pct = 0.0
                '''
                seq_error_result = compute_and_apply_seq_logprob_error_masking(
                    train_data=train_data,
                    rewards=rewards,
                    seq_logprob_error_threshold=seq_logprob_error_threshold,
                )
                max_seq_mult_prob_error = seq_error_result["max_seq_mult_prob_error"]
                mean_seq_mult_prob_error = seq_error_result["mean_seq_mult_prob_error"]
                min_seq_mult_prob_error = seq_error_result["min_seq_mult_prob_error"]
                max_seq_mult_prob_error_after_mask = seq_error_result["max_seq_mult_prob_error_after_mask"]
                mean_seq_mult_prob_error_after_mask = seq_error_result["mean_seq_mult_prob_error_after_mask"]
                min_seq_mult_prob_error_after_mask = seq_error_result["min_seq_mult_prob_error_after_mask"]
                num_masked_seqs = seq_error_result["num_masked_seqs"]
                masked_correct_pct = seq_error_result["masked_correct_pct"]

                # Update sample_mask if masking was applied
                if seq_error_result["updated_sample_mask"] is not None:
                    train_data["sample_mask"] = seq_error_result["updated_sample_mask"]
                '''
                
                # Compute advantages with adv_estimator using correct mask and logprobs
                with timer.time("advantage_calculation"):
                    print("▶ Computing advantages...", flush=True)
                    # Get token-level mask: token_mask * sample_mask
                    token_mask = train_data["token_mask"]
                    sample_mask = train_data["sample_mask"]
                    mask = token_mask * sample_mask.unsqueeze(-1)

                    train_data["advantages"] = adv_estimator.compute_advantage(
                        prompt_ids=prompt_ids_for_adv,
                        rewards=rewards,
                        mask=mask,
                        logprobs_policy=train_data["prev_logprobs"],
                        logprobs_reference=train_data.get("reference_policy_logprobs"),
                    )
                    del prompt_ids_for_adv

                    # Log rewards and advantages information
                    _log_mixed_rewards_and_advantages_information(
                        logger=logger,
                        total_steps=total_steps,
                        metrics=metrics,
                        baseline=baseline_for_log,
                        advantages=train_data["advantages"],
                    )
                    del baseline_for_log

                    # Clip advantages to prevent extreme values from small std normalization
                    clip_low = master_config["grpo"].get("advantage_clip_low")
                    clip_high = master_config["grpo"].get("advantage_clip_high")
                    if clip_low is not None:
                        train_data["advantages"] = train_data["advantages"].clamp(min=clip_low)
                    if clip_high is not None:
                        train_data["advantages"] = train_data["advantages"].clamp(max=clip_high)

                    # Apply invalid tool call / malformed thinking penalization per-message.
                    # Only override the specific message's token positions within the
                    # flattened sequence.
                    penalize_invalid_tool_call = master_config["grpo"].get("penalize_invalid_tool_call", False)
                    penalize_malformed_thinking = master_config["grpo"].get("penalize_malformed_thinking", False)
                    if penalize_invalid_tool_call or penalize_malformed_thinking:
                        invalid_neg_adv = master_config["grpo"].get("invalid_tool_call_advantage", -5.0)
                        malformed_neg_adv = master_config["grpo"].get("malformed_thinking_advantage", -5.0)
                        for i, message_log in enumerate(repeated_batch["message_log"]):
                            token_offset = 0
                            for j, message in enumerate(message_log):
                                msg_len = len(message["token_ids"])
                                is_assistant = message["role"] == "assistant" and "generation_logprobs" in message
                                is_invalid = is_assistant and penalize_invalid_tool_call and message.get("is_invalid_tool_call", False)
                                is_malformed_thinking_msg = is_assistant and penalize_malformed_thinking and message.get("has_malformed_thinking", False)
                                if is_invalid:
                                    print(f"Setting negative advantage ({invalid_neg_adv}) for invalid tool call in assistant message {i} {j}", flush=True)
                                    train_data["advantages"][i, token_offset:token_offset + msg_len] = invalid_neg_adv
                                elif is_malformed_thinking_msg:
                                    print(f"Setting negative advantage ({malformed_neg_adv}) for malformed thinking in assistant message {i} {j}", flush=True)
                                    train_data["advantages"][i, token_offset:token_offset + msg_len] = malformed_neg_adv
                                token_offset += msg_len

                memory_tracker.snapshot_start_of_stage("Policy train", dir())
                print("▶ Preparing for training...", flush=True)
                with timer.time("training_prep"):
                    policy.prepare_for_training()  # set model train and reload optim to GPU
                    POLICY_GENERATION_STALE = True

                print("▶ Training policy...", flush=True)
                with timer.time("policy_training"):
                    train_results = policy.train(
                        train_data,
                        loss_fn,
                        timer=timer,
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
                        # Set generation as stale to force refit with new scales
                        POLICY_GENERATION_STALE = True

                is_last_step = (total_steps + 1 >= max_num_steps) or (
                    (current_epoch + 1 == max_num_epochs)
                    and (current_step + 1 == len(dataloader))
                )

                # Run validation if it's a validation step or last step with val_at_end
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
                            policy.offload_after_refit()  # unload optimizer to make space for generation
                        policy_generation.prepare_for_generation()
                    val_metrics, validation_timings = validate(
                        policy,
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

                # Get flat advantages and token mask for masked metrics computation
                flat_advantages = train_data["advantages"]
                #flat_token_mask = flat_messages["token_loss_mask"]

                # Filter advantages using token mask (only valid response tokens)
                response_advantages = torch.masked_select(
                    flat_advantages, flat_token_mask.bool()
                )

                memory_tracker.snapshot_start_of_stage("Metrics", dir())
                metrics = {
                    **metrics,
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                    "reward": rewards.numpy(),
                    "bt_accuracy": bt_accuracy,
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
                    #**ds_metrics,
                }
                if "moe_metrics" in train_results:
                    metrics.update(
                        {f"moe/{k}": v for k, v in train_results["moe_metrics"].items()}
                    )
                if "mtp_metrics" in train_results:
                    metrics.update(
                        {f"mtp/{k}": v for k, v in train_results["mtp_metrics"].items()}
                    )
                if master_config["grpo"]["use_dynamic_sampling"]:
                    metrics["filtered_reward"] = rewards.numpy()
                    metrics["reward"] = repeated_batch["total_reward"].numpy()

                metrics.update(train_results["all_mb_metrics"])
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
                        "filtered_reward",
                        "global_valid_seqs",
                        "global_valid_toks",
                        "mean_prompt_length",
                    }:
                        metrics[k] = np.mean(v).item()
                    elif isinstance(v, (np.ndarray, list)):
                        metrics[k] = np.sum(v).item()
                    else:
                        print(f"Skipping aggregation for {k} ({type(v)})")

                metrics.update(rollout_metrics)
                metrics["generation_logger_metrics"] = generation_logger_metrics
                total_valid_tokens += metrics["global_valid_toks"]

                # Always log sequence-level error metrics (useful for deciding threshold)
                metrics["max_seq_mult_prob_error"] = max_seq_mult_prob_error
                metrics["mean_seq_mult_prob_error"] = mean_seq_mult_prob_error
                metrics["min_seq_mult_prob_error"] = min_seq_mult_prob_error
                metrics["max_seq_mult_prob_error_after_mask"] = max_seq_mult_prob_error_after_mask
                metrics["mean_seq_mult_prob_error_after_mask"] = mean_seq_mult_prob_error_after_mask
                metrics["min_seq_mult_prob_error_after_mask"] = min_seq_mult_prob_error_after_mask
                metrics["num_masked_seqs_by_logprob_error"] = num_masked_seqs
                metrics["masked_correct_pct"] = masked_correct_pct

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

                memory_tracker.snapshot_start_of_stage("Checkpointing", dir())
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
                        torch.save(
                            dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # Logging
            # Log training data
            memory_tracker.snapshot_start_of_stage("Logging", dir())
            if not _should_log_nemo_gym_responses(master_config):
                log_data = {}
                if "agent_ref" in repeated_batch:
                    log_data["agent_ref"] = repeated_batch["agent_ref"]
                log_data["content"] = flat_messages["content"]
                log_data["rewards"] = rewards.tolist()
                if master_config["grpo"]["use_dynamic_sampling"]:
                    log_data["filtered_rewards"] = rewards.tolist()
                    log_data["rewards"] = repeated_batch["total_reward"].tolist()
                log_data["input_lengths"] = input_lengths.tolist()
                log_data["token_ids"] = train_data["input_ids"].tolist()
                log_data["token_loss_mask"] = train_data["token_mask"].tolist()
                log_data["sample_loss_mask"] = train_data["sample_mask"].tolist()
                log_data["advantages"] = train_data["advantages"].tolist()
                log_data["generation_logprobs"] = train_data[
                    "generation_logprobs"
                ].tolist()
                log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()

                logger.log_batched_dict_as_jsonl(
                    log_data, f"train_data_step{total_steps + 1}.jsonl"
                )
                del log_data
            del flat_messages

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
            del train_data
            if master_config["policy"]["generation"].get("vllm_cfg", {}).get(
                "enable_vllm_metrics_logger", False
            ) and master_config.get("logger", {}).get("wandb_enabled", False):
                log_generation_metrics_to_wandb(
                    generation_logger_metrics,
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
                f"  • Mean Generation Length: {metrics_logging_data['mean_gen_tokens_per_sample']:.4f}",
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
            # step_finished=True here since this is the final log of our current step.
            logger.log_metrics(
                timing_metrics,
                total_steps + 1,
                prefix="timing/train",
                step_finished=True,
            )

            # Reset the batch and set dynamic_sampling_num_gen_batches to 0
            batch_cache = None
            dynamic_sampling_num_gen_batches = 0

            # Clear mem
            memory_tracker.snapshot_start_of_stage("After CPU memory clear", dir())

            # processing rewards
            del repeated_batch, repeated_batch_2, thought_mask, answer_mask, bt_accuracy
            del rewards, new_msg_log, new_fmt_list, yes_position, no_position, yes_tensor, list_with_yes, input_ids_with_yes, flat_token_mask, input_lengths_rb_1
            # train_data already deleted after logging above
            # logging
            del metrics
            if "val_metrics" in dir():
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
        current_step = 0  # Reset step counter for new epoch


def validate(
    policy: ColocatablePolicyInterface,
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

        # Validate GenRM compare configuration
        use_genrm_compare = master_config["env"].get("use_genrm_compare", False)
        num_val_gens = master_config["grpo"].get("num_val_generations_per_prompt", 1)
        if use_genrm_compare and num_val_gens <= 1:
            raise ValueError(
                f"GenRM compare requires num_val_generations_per_prompt > 1 for pairwise comparison, "
                f"but got num_val_generations_per_prompt={num_val_gens}. "
                f"Set grpo.num_val_generations_per_prompt to at least 2."
            )

        total_rewards = []
        bt_probs = []
        total_lengths = []
        all_message_logs = []  # Collect all message logs
        results = []

        max_batches = (
            master_config["grpo"]["max_val_samples"]
            // master_config["grpo"]["val_batch_size"]
        )
        for batch_idx, val_batch in enumerate(val_dataloader):
            batch_size = len(val_batch["message_log"])
            active_indices = torch.arange(batch_size)
            additional_metrics_to_report = dict()
            if batch_idx >= max_batches:
                break

            # Duplicate prompts for multiple generations per prompt during validation
            # Similar to training, this allows evaluating model consistency and diversity
            if num_val_gens > 1:
                val_batch_for_rollout = val_batch.repeat_interleave(num_val_gens)
            else:
                val_batch_for_rollout = val_batch

            # Generate responses (updates the LLMMessageLogType in batch_with_msg_logs)
            # Use async rollouts if vLLM async engine is enabled
            # We cascade NeMo-Gym first since NeMo-Gym also uses async rollouts.
            if _should_use_nemo_gym(master_config):
                generation_config = master_config["policy"]["generation"]
                nemo_gym_rollout_result = run_async_nemo_gym_rollout(
                    policy_generation=policy_generation,
                    input_batch=val_batch_for_rollout,
                    tokenizer=tokenizer,
                    task_to_env=val_task_to_env,
                    max_seq_len=None,
                    generation_config=generation_config,
                    max_rollout_turns=None,
                    greedy=False,
                    # GenRM compare config
                    use_genrm_compare=use_genrm_compare,
                    num_generations_per_prompt=num_val_gens,
                    genrm_compare_server_name=master_config["env"].get(
                        "genrm_compare_server_name", "genrm_compare"
                    ),
                    genrm_agent_names=master_config["env"].get(
                        "genrm_agent_names", ["genrm_simple_agent"]
                    ),
                    master_config=master_config
                )
                val_batch = nemo_gym_rollout_result.final_batch
                gen_metrics = nemo_gym_rollout_result.rollout_metrics
                additional_metrics_to_report = gen_metrics
            elif _should_use_async_rollouts(master_config):
                val_batch, gen_metrics = run_async_multi_turn_rollout(
                    policy_generation,
                    val_batch_for_rollout,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                    greedy=False,
                )
            else:
                val_batch, gen_metrics = run_multi_turn_rollout(
                    policy_generation,
                    val_batch_for_rollout,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["grpo"]["max_rollout_turns"],
                    greedy=False,
                )
            
            # need to calculate reward in the DWRL way
            new_msg_log = []
            for outer in val_batch_for_rollout['message_log']:
                p = []
                for inner in outer:
                    if inner["role"] == "environment":
                        continue
                    #pp = {"role": inner["role"], "content": inner['content'], "token_ids": inner["token_ids"]}
                    #if "generation_logprobs" in inner:
                    #    pp["generation_logprobs"] = inner["generation_logprobs"]
                    pp = copy.deepcopy(inner)
                    p.append(pp)
                px = {"role": "user", "content": master_config["grpo"]["dwrl"]["bt_prompt"]}
                npx = tokenizer.apply_chat_template([px], tokenize=False, add_generation_prompt=True, add_special_tokens=False, enable_thinking=False)
                px['content'] = npx.replace("<|im_start|>system\n<|im_end|>\n", "").replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", "")# + "\n\n</think>\n\n"
                px["token_ids"] = tokenizer(px['content'], return_tensors="pt")["input_ids"][0]
                p.append(px)
                new_msg_log.append(p)

            new_fmt_list = []
            for idx, (msg_log, idx, extra_env_info, loss_mx) in enumerate(zip(new_msg_log, val_batch_for_rollout['idx'], val_batch_for_rollout['extra_env_info'], val_batch_for_rollout['loss_multiplier'])):
                length = sum(len(m["token_ids"]) for m in msg_log)
                output = {"message_log": msg_log, "length": length, "extra_env_info": extra_env_info, "loss_multiplier": loss_mx, "idx": idx, "task_name": "genrm_dwrl"}
                new_fmt_list.append(output)
            repeated_batch_2 = rl_collate_fn(new_fmt_list)
            
            flat_messages, input_lengths = (
                batched_message_log_to_flat_message(
                    repeated_batch_2["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                )
            )
            yes_position = tokenizer.encode(master_config["grpo"]["dwrl"]["score_token"])[0]
            yes_tensor = torch.tensor([yes_position], device=flat_messages["token_ids"].device).long()
            
            policy.prepare_for_lp_inference()
            list_with_yes = [torch.cat([flat_messages["token_ids"][idx][:input_lengths[idx]], yes_tensor, flat_messages["token_ids"][idx][input_lengths[idx]:]], dim=-1) for idx in range(len(input_lengths))]
            input_ids_with_yes = torch.stack(list_with_yes, dim=0)
            # Custom create this logprob_data so we avoid Ray comm overheads sending unused data to workers.
            logprob_data = BatchedDataDict[DWRLLossDataDict](
                {
                    "input_ids": input_ids_with_yes,
                    "input_lengths": input_lengths + 1,
                }
            )
            prev_logprobs_with_yes = policy.get_logprobs(logprob_data)["logprobs"]
            
            actual_rewards = prev_logprobs_with_yes.gather(-1, input_lengths.unsqueeze(-1)).squeeze(-1)
            del logprob_data, prev_logprobs_with_yes
            
            total_rewards.extend(val_batch["total_reward"].tolist())
            bt_probs.extend(actual_rewards.exp().tolist())
            total_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])

            # Collect message logs for later display
            to_env = [
                get_keys_from_message_log(
                    val_batch["message_log"][i], ["role", "content"]
                )
                for i in range(len(val_batch["message_log"]))
            ]

            all_message_logs.extend(to_env)
            
            #val_batch['bt_prob'] = actual_rewards.exp().cpu().tolist()
            
            for eei, pred in zip(val_batch["extra_env_info"], actual_rewards.exp().cpu().tolist()):
                gt = eei["preference"]
                
                results.append( int((pred >= 0.5 and gt == 0) or (pred < 0.5 and gt == 1)) )

        # Calculate validation metrics
        num_samples = len(total_rewards)
        if num_samples > 0:
            rewards_t = torch.tensor(total_rewards, dtype=torch.float32)
            rewards_mean = rewards_t.mean().item()
        else:
            rewards_mean = 0.0
        num_samples_env = len(bt_probs)
        if num_samples_env > 0:
            bt_probs_t = torch.tensor(bt_probs, dtype=torch.float32)
            bt_probs_mean = bt_probs_t.mean().item()
        else:
            bt_probs_mean = 0.0
        if len(results) > 0:
            results_t = torch.tensor(results, dtype=torch.float32)
            accuracy = results_t.mean().item()
        else:
            accuracy = 0.0

        avg_length = (
            sum(total_lengths) / len(total_lengths) if len(total_lengths) > 0 else 0.0
        )

        val_metrics = {
            "accuracy": accuracy,
            "rewards": rewards_mean,
            "bt_probs": bt_probs_mean,
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
    print(f"    • Rewards: {rewards_mean:.4f}")
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
            "accuracy": results,
            "rewards": total_rewards,
            "bt_probs": bt_probs,
        }
        logger.log_batched_dict_as_jsonl(val_log_data, f"val_data_step{step}.jsonl")

    # Make sure to reset the timer after validation
    timer.reset()

    # Explicit GPU memory cleanup after validation
    gc.collect()
    torch.cuda.empty_cache()

    return val_metrics, timing_metrics


def async_dwrl_train(
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
    val_at_end = master_config["grpo"]["val_at_end"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]

    # Initialize advantage estimator
    adv_estimator = _create_advantage_estimator(master_config)

    assert not colocated_inference, (
        "Colocated inference is not supported for async GRPO. Please use non-colocated inference."
    )

    # Calculate minimum buffer size from training requirements
    # In per-prompt buffer mode, one buffer entry is 1 prompt * num_generations_per_prompt
    num_prompts_per_step = master_config["grpo"]["num_prompts_per_step"]
    samples_per_prompt_group = master_config["grpo"]["num_generations_per_prompt"]
    train_gbs = master_config["policy"]["train_global_batch_size"]
    to_compute_kl = master_config["loss_fn"]["reference_policy_kl_penalty"] > 0

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

    # Restore replay buffer state from checkpoint if available
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    if last_checkpoint_path is not None:
        replay_buffer_path = os.path.join(last_checkpoint_path, "replay_buffer.pt")
        if os.path.exists(replay_buffer_path):
            print(f"📦 Restoring replay buffer from checkpoint: {replay_buffer_path}")
            try:
                replay_buffer_state = torch.load(
                    replay_buffer_path, weights_only=False
                )
                ray.get(
                    replay_buffer.load_state_dict.remote(
                        replay_buffer_state,
                        num_prompts_per_step=num_prompts_per_step,
                        current_training_step=step,
                    )
                )
                print(
                    f"✅ Replay buffer restored from checkpoint"
                )
            except Exception as e:
                print(f"⚠️ Failed to restore replay buffer state: {e}")
                raise e
        else:
            print(
                f"⚠️ No replay buffer checkpoint found at {replay_buffer_path}. "
                "Starting with empty buffer (this is expected for older checkpoints)."
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
                policy,
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
    # Clear logger metrics at start of training
    if policy_generation is not None:
        policy_generation.clear_logger_metrics()

    # Wait for initial buffer fill and current step to have complete batch
    print(
        f"⏳ Waiting for replay buffer to have sufficient trajectories for step {step}..."
    )
    wait_iterations = 0
    while True:
        buffer_size_current = ray.get(replay_buffer.size.remote())

        # Check if current training step has enough trajectories
        current_step_ready = ray.get(
            replay_buffer.has_complete_batch.remote(step, num_prompts_per_step)
        )

        print(
            f"  Wait iteration {wait_iterations}: buffer_size={buffer_size_current}, "
            f"step {step} ready={current_step_ready}"
        )

        if current_step_ready:
            break

        # Also break if we have minimum trajectories and it's a fresh start (no gap-filling needed)
        if buffer_size_current >= min_trajectories_needed and wait_iterations == 0:
            # Check how many trajectories needed for current step
            trajectories_needed = ray.get(
                replay_buffer.get_trajectories_needed.remote(step, num_prompts_per_step)
            )
            if trajectories_needed > 0:
                print(
                    f"  ⏳ Gap-filling in progress: need {trajectories_needed} more trajectories for step {step}"
                )

        wait_iterations += 1
        time.sleep(1.0)

    print(f"✅ Buffer ready for step {step}! Starting training loop...")

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
                    # Aggregate rollout metrics across groups with proper aggregation per metric type
                    rollout_metrics = {}
                    for t in trajectories:
                        for k, v in t["rollout_metrics"].items():
                            rollout_metrics.setdefault(k, []).append(v)
                    # Aggregate metrics properly based on their semantics
                    aggregated_rollout_metrics = {}
                    for k, v in rollout_metrics.items():
                        if not isinstance(v[0], (int, float)):
                            aggregated_rollout_metrics[k] = v
                        elif k.endswith("/min") or (k.startswith("min_") and not k.endswith("_rate")):
                            # For min metrics, take the actual minimum
                            # Handles both "key/min" format and "min_key" format (but not "min_*_rate" which are averages)
                            aggregated_rollout_metrics[k] = min(v)
                        elif k.endswith("/max") or (k.startswith("max_") and not k.endswith("_rate")):
                            # For max metrics, take the actual maximum
                            # Handles both "key/max" format and "max_key" format (but not "max_*_rate" which are averages)
                            aggregated_rollout_metrics[k] = max(v)
                        elif k == "total_turns":
                            # For total counts, sum them
                            aggregated_rollout_metrics[k] = sum(v)
                        else:
                            # For mean/rate metrics, take the average
                            aggregated_rollout_metrics[k] = sum(v) / len(v)
                    rollout_metrics = aggregated_rollout_metrics

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
                    # Extract original prompt messages using the length field
                    # This correctly handles multi-turn prompts that contain assistant messages
                    initial_prompt_message_logs = extract_initial_prompt_messages(
                        repeated_batch["message_log"],
                        repeated_batch["length"],
                    )

                    prompt_batched_flat, prompt_input_lengths = (
                        batched_message_log_to_flat_message(
                            initial_prompt_message_logs,
                            pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        )
                    )
                    prompt_ids_for_adv = prompt_batched_flat["token_ids"]
                    del initial_prompt_message_logs
                    del prompt_batched_flat

                    rewards = repeated_batch["total_reward"]

                    print(
                        f"  📊 Rewards stats: min={rewards.min():.4f}, max={rewards.max():.4f}, mean={rewards.mean():.4f}, std={rewards.std():.4f}"
                    )

                # Prepare training data (same as sync version)
                with timer.time("data_processing"):
                    # Apply overlong filtering - mask out truncated sequences from loss computation
                    with timer.time("overlong_filter"):
                        use_overlong_filtering = master_config["grpo"]["overlong_filtering"]
                        if use_overlong_filtering:
                            loss_multiplier = repeated_batch["loss_multiplier"].clone()
                            truncated = repeated_batch["truncated"]

                            if isinstance(truncated, list):
                                truncated = torch.tensor(truncated, dtype=torch.bool)

                            loss_multiplier[truncated] = 0
                            repeated_batch["loss_multiplier"] = loss_multiplier

                    with timer.time("add_loss_mask"):
                        # Add loss mask to each message
                        # Only unmask assistant messages that were actually generated (have generation_logprobs),
                        # not assistant messages that were part of the prompt history
                        for i, message_log in enumerate(repeated_batch["message_log"]):
                            for j, message in enumerate(message_log):
                                token_ids = message["token_ids"]
                                is_assistant = message["role"] == "assistant" and "generation_logprobs" in message

                                if is_assistant:
                                    message["token_loss_mask"] = torch.ones_like(token_ids)
                                else:
                                    message["token_loss_mask"] = torch.zeros_like(token_ids)

                                if "generation_logprobs" not in message:
                                    message["generation_logprobs"] = torch.zeros_like(token_ids, dtype=torch.float32)

                    # Convert to flat format for training
                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    # Create training data
                    # Note: advantages will be computed and added after logprobs are available
                    train_data = BatchedDataDict[DWRLLossDataDict](
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "generation_logprobs": flat_messages["generation_logprobs"],
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )
                    train_data.to("cpu")

                # Training phase (same as sync version)
                # Skip prev_logprobs computation when force_on_policy_ratio=True
                # unless seq_logprob_error_threshold is set (which requires prev_logprobs)
                seq_logprob_error_threshold = master_config["grpo"].get(
                    "seq_logprob_error_threshold", None
                )
                force_on_policy_ratio = master_config["loss_fn"].get("force_on_policy_ratio", False)
                skip_prev_logprobs = force_on_policy_ratio and seq_logprob_error_threshold is None

                # todo @jiaqi: is there a better way to skip prev_logprobs computation while still computing the seq-level error metrics?
                if force_on_policy_ratio and seq_logprob_error_threshold is not None:
                    warnings.warn(
                        "force_on_policy_ratio=True but seq_logprob_error_threshold is set. "
                        "Computing prev_logprobs anyway for seq-level error masking."
                    )

                if skip_prev_logprobs:
                    print("▶ Skipping prev_logprobs (force_on_policy_ratio=True)...", flush=True)
                    fprop_logprobs = torch.zeros_like(train_data["generation_logprobs"])
                else:
                    print("▶ Preparing for logprob inference...")
                    with timer.time("logprob_inference_prep"):
                        policy.prepare_for_lp_inference()

                print("▶ Computing logprobs...", flush=True)
                with timer.time("policy_and_reference_logprobs"):
                    if not skip_prev_logprobs:
                        fprop_logprobs = policy.get_logprobs(
                            train_data,
                            timer=timer,
                        )["logprobs"]

                    if to_compute_kl:
                        reference_logprobs = policy.get_reference_policy_logprobs(
                            train_data,
                            timer=timer,
                        )["reference_logprobs"]
                    else:
                        reference_logprobs = torch.zeros_like(train_data["generation_logprobs"])
                train_data["prev_logprobs"] = fprop_logprobs
                train_data["reference_policy_logprobs"] = reference_logprobs

                # Seq-level logprob error metrics/masking require real prev_logprobs
                if skip_prev_logprobs:
                    # Cannot compute seq-level metrics with placeholder prev_logprobs
                    max_seq_mult_prob_error = 0.0
                    mean_seq_mult_prob_error = 0.0
                    min_seq_mult_prob_error = 0.0
                    max_seq_mult_prob_error_after_mask = 0.0
                    mean_seq_mult_prob_error_after_mask = 0.0
                    min_seq_mult_prob_error_after_mask = 0.0
                    num_masked_seqs = 0
                    masked_correct_pct = 0.0
                else:
                    seq_error_result = compute_and_apply_seq_logprob_error_masking(
                        train_data=train_data,
                        rewards=rewards,
                        seq_logprob_error_threshold=seq_logprob_error_threshold,
                    )
                    max_seq_mult_prob_error = seq_error_result["max_seq_mult_prob_error"]
                    mean_seq_mult_prob_error = seq_error_result["mean_seq_mult_prob_error"]
                    min_seq_mult_prob_error = seq_error_result["min_seq_mult_prob_error"]
                    max_seq_mult_prob_error_after_mask = seq_error_result["max_seq_mult_prob_error_after_mask"]
                    mean_seq_mult_prob_error_after_mask = seq_error_result["mean_seq_mult_prob_error_after_mask"]
                    min_seq_mult_prob_error_after_mask = seq_error_result["min_seq_mult_prob_error_after_mask"]
                    num_masked_seqs = seq_error_result["num_masked_seqs"]
                    masked_correct_pct = seq_error_result["masked_correct_pct"]

                    # Update sample_mask if masking was applied
                    if seq_error_result["updated_sample_mask"] is not None:
                        train_data["sample_mask"] = seq_error_result["updated_sample_mask"]

                # Compute advantages with adv_estimator using correct mask and logprobs
                with timer.time("advantage_calculation"):
                    print("▶ Computing advantages...", flush=True)
                    # Get token-level mask: token_mask * sample_mask
                    token_mask = train_data["token_mask"]
                    sample_mask = train_data["sample_mask"]
                    mask = token_mask * sample_mask.unsqueeze(-1)

                    train_data["advantages"] = adv_estimator.compute_advantage(
                        prompt_ids=prompt_ids_for_adv,
                        rewards=rewards,
                        mask=mask,
                        logprobs_policy=train_data["prev_logprobs"],
                        logprobs_reference=train_data.get("reference_policy_logprobs"),
                    )
                    del prompt_ids_for_adv

                    # Log advantages stats
                    # Note: For GRPOAdvantageEstimator with normalize_rewards=True, these are
                    # already normalized advantages (equivalent to "Normalized advantages stats"
                    # in older versions). For ReinforcePlusPlusAdvantageEstimator, advantages
                    # are globally normalized across valid tokens.
                    advantages = train_data["advantages"]
                    print(
                        f"  📊 Advantages stats: min={advantages.min():.4f}, max={advantages.max():.4f}, mean={advantages.mean():.4f}, std={advantages.std():.4f}"
                    )

                    # Clip advantages to prevent extreme values from small std normalization
                    clip_low = master_config["grpo"].get("advantage_clip_low")
                    clip_high = master_config["grpo"].get("advantage_clip_high")
                    if clip_low is not None:
                        train_data["advantages"] = train_data["advantages"].clamp(min=clip_low)
                    if clip_high is not None:
                        train_data["advantages"] = train_data["advantages"].clamp(max=clip_high)

                    # Apply invalid tool call / malformed thinking penalization per-message.
                    # Only override the specific message's token positions within the
                    # flattened sequence.
                    penalize_invalid_tool_call = master_config["grpo"].get("penalize_invalid_tool_call", False)
                    penalize_malformed_thinking = master_config["grpo"].get("penalize_malformed_thinking", False)
                    if penalize_invalid_tool_call or penalize_malformed_thinking:
                        print(f"Penalize invalid tool call: {penalize_invalid_tool_call}", flush=True)
                        print(f"Penalize malformed thinking: {penalize_malformed_thinking}", flush=True)
                        invalid_neg_adv = master_config["grpo"].get("invalid_tool_call_advantage", -5.0)
                        malformed_neg_adv = master_config["grpo"].get("malformed_thinking_advantage", -5.0)
                        for i, message_log in enumerate(repeated_batch["message_log"]):
                            token_offset = 0
                            for j, message in enumerate(message_log):
                                msg_len = len(message["token_ids"])
                                is_assistant = message["role"] == "assistant" and "generation_logprobs" in message
                                is_invalid = is_assistant and penalize_invalid_tool_call and message.get("is_invalid_tool_call", False)
                                is_malformed_thinking = is_assistant and penalize_malformed_thinking and message.get("has_malformed_thinking", False)
                                if is_invalid:
                                    print(f"Setting negative advantage ({invalid_neg_adv}) for invalid tool call in assistant message {i} {j}", flush=True)
                                    train_data["advantages"][i, token_offset:token_offset + msg_len] = invalid_neg_adv
                                elif is_malformed_thinking:
                                    print(f"Setting negative advantage ({malformed_neg_adv}) for malformed thinking in assistant message {i} {j}", flush=True)
                                    train_data["advantages"][i, token_offset:token_offset + msg_len] = malformed_neg_adv
                                token_offset += msg_len

                print("▶ Preparing for training...")
                with timer.time("training_prep"):
                    policy.prepare_for_training()
                    POLICY_GENERATION_STALE = True

                print("▶ Training policy...")
                with timer.time("policy_training"):
                    train_results = policy.train(
                        train_data,
                        loss_fn,
                        timer=timer,
                    )

                print("🔄 Synchronizing policy weights to trajectory collector…")
                generation_logger_metrics = None
                if NEED_REFIT:
                    # Measure pending-generation wait as exposed_generation time
                    print("🔄 Coordinating with trajectory collector before refit...")
                    with timer.time("exposed_generation"):
                        ray.get(trajectory_collector.prepare_for_refit.remote())

                    # Collect generation logger metrics for performance reporting
                    # inflight batch sizes and num pending samples are collected from each worker
                    if policy_generation is not None:
                        generation_logger_metrics = (
                            policy_generation.get_logger_metrics()
                        )

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

                # Clear logger metrics after each refit (weight sync), starting a new logging cycle
                if policy_generation is not None:
                    policy_generation.clear_logger_metrics()

                # Validation
                val_metrics, validation_timings = None, None
                is_last_step = step + 1 == master_config["grpo"]["max_num_steps"]

                # Run validation if it's a validation step or last step with val_at_end
                if (val_period > 0 and (step + 1) % val_period == 0) or (
                    val_at_end and is_last_step
                ):
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
                        policy,
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

                    # Explicit GPU memory cleanup after validation in async mode
                    import gc

                    gc.collect()
                    torch.cuda.empty_cache()

                    # Resume trajectory collection after validation
                    trajectory_collector.resume.remote()
                # Get flat advantages and token mask for masked metrics computation
                flat_advantages = train_data["advantages"]
                flat_token_mask = flat_messages["token_loss_mask"]
                # Save content for logging before deleting flat_messages
                flat_messages_content = flat_messages.get("content", [])
                del flat_messages

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
                if "mtp_metrics" in train_results:
                    metrics.update(
                        {f"mtp/{k}": v for k, v in train_results["mtp_metrics"].items()}
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
                if generation_logger_metrics is not None:
                    metrics["generation_logger_metrics"] = generation_logger_metrics
                total_valid_tokens += metrics["global_valid_toks"]

                # Always log sequence-level error metrics (useful for deciding threshold)
                metrics["max_seq_mult_prob_error"] = max_seq_mult_prob_error
                metrics["mean_seq_mult_prob_error"] = mean_seq_mult_prob_error
                metrics["min_seq_mult_prob_error"] = min_seq_mult_prob_error
                metrics["max_seq_mult_prob_error_after_mask"] = max_seq_mult_prob_error_after_mask
                metrics["mean_seq_mult_prob_error_after_mask"] = mean_seq_mult_prob_error_after_mask
                metrics["min_seq_mult_prob_error_after_mask"] = min_seq_mult_prob_error_after_mask
                metrics["num_masked_seqs_by_logprob_error"] = num_masked_seqs
                metrics["masked_correct_pct"] = masked_correct_pct

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
                        # Save replay buffer state for resumption
                        print("📦 Saving replay buffer state...")
                        replay_buffer_state = ray.get(
                            replay_buffer.state_dict.remote()
                        )
                        torch.save(
                            replay_buffer_state,
                            os.path.join(checkpoint_path, "replay_buffer.pt"),
                        )
                        print(
                            f"✅ Saved replay buffer with {len(replay_buffer_state['trajectories'])} trajectories"
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)
                    policy.offload_after_refit()

            # Logging
            # Log training data (match sync GRPO logging payload for parity)
            log_data = {}
            if "agent_ref" in repeated_batch:
                log_data["agent_ref"] = repeated_batch["agent_ref"]
            log_data["content"] = flat_messages_content
            log_data["rewards"] = rewards.tolist()
            if master_config["grpo"]["use_dynamic_sampling"]:
                # In dynamic sampling, `rewards` corresponds to filtered rewards
                log_data["filtered_rewards"] = rewards.tolist()
                log_data["rewards"] = repeated_batch["total_reward"].tolist()
            log_data["input_lengths"] = input_lengths.tolist()
            log_data["token_ids"] = train_data["input_ids"].tolist()
            log_data["token_loss_mask"] = train_data["token_mask"].tolist()
            log_data["sample_loss_mask"] = train_data["sample_mask"].tolist()
            log_data["advantages"] = train_data["advantages"].tolist()
            log_data["generation_logprobs"] = train_data["generation_logprobs"].tolist()
            log_data["prev_logprobs"] = train_data["prev_logprobs"].tolist()
            logger.log_batched_dict_as_jsonl(
                log_data, f"train_data_step{step + 1}.jsonl"
            )
            del train_data
            del flat_messages_content

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
                    generation_logger_metrics,
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
