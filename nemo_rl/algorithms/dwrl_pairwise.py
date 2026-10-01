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
import functools
import copy
import gc
import json
import os
import time
import warnings
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, fields
from typing import Any, Callable, Optional, TypeVar, cast

import numpy as np
import ray
import torch
from pydantic import BaseModel, Field, model_validator
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms import opd as opd_module
from nemo_rl.algorithms import opd_diagnostics as opd_diag
from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GDPOAdvantageEstimator,
    GRPOAdvantageEstimator,
    OPDAdvantageEstimator,
    ReinforcePlusPlusAdvantageEstimator,
)
from nemo_rl.algorithms.logits_sampling_utils import (
    TrainingSamplingParams,
    need_top_k_or_top_p_filtering,
)
from nemo_rl.algorithms.loss import (
    ClippedPGLossConfig,
    DWRLLossDataDict,
    DWRLLossFn,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.algorithms.metric_utils import (
    SetupTimingMetrics,
    print_setup_timing_summary,
)
from nemo_rl.algorithms.opd import OnPolicyDistillationConfig
from nemo_rl.algorithms.reward_functions import (
    RewardShapingConfig,
    apply_reward_shaping,
)
from nemo_rl.algorithms.utils import (
    build_rollout_group_ids,
    calculate_baseline_and_std_per_prompt,
    get_gdpo_reward_component_keys,
    log_generation_metrics,
    print_efficiency_summary,
    print_performance_metrics,
    set_seed,
)
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn, preference_collate_fn
from nemo_rl.data.dataloader import MultipleDataloaderWrapper
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, VLMMessageLogType
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.data.utils import extract_necessary_env_names, load_dataloader_state
from nemo_rl.data_plane.interfaces import DataPlaneConfig
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import (
    TOPO_RANK_UNKNOWN,
    ClusterConfig,
    RayVirtualCluster,
    get_ray_cluster_topology,
    prepare_segment_topology,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.nemo_gym import should_use_nemo_gym, spinup_nemo_gym_actor
from nemo_rl.experience.interfaces import (
    FRONTIER_ORDINAL_KEY,
    NEMO_GYM_ATTEMPT_INDEX_KEY,
    NEMO_GYM_ROLLOUT_INDEX_KEY,
    NEMO_GYM_TASK_INDEX_KEY,
    NEMO_RL_EMPTY_RESPONSE_OUTPUT_KEY,
    NEXT_NEMO_GYM_TASK_INDEX_KEY,
    PENDING_PROMPTS_KEY,
    RESUME_BASE_ORDINAL_KEY,
    RETAINED_TASK_INDICES_KEY,
    TARGET_WEIGHT_VERSION_KEY,
    TRAINED_TASK_INDICES_KEY,
)
from nemo_rl.experience.metric_utils import is_histogram_metric
from nemo_rl.experience.rollouts import (
    EffortLevelsConfig,
    attach_initial_nemo_gym_image_payloads,
    backfill_missing_routed_experts,
    get_nemo_gym_thinking_tags,
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
    run_nemo_gym_rollout_sync,
    should_mask_flagged_samples,
)
from nemo_rl.models.generation.dynamo import DynamoConfig, DynamoGeneration
from nemo_rl.models.generation.interfaces import (
    GenerationConfig,
    GenerationInterface,
    GenerationSamplingParams,
    resolve_routed_experts_dtype_name_for_model,
    should_use_async_rollouts,
)
from nemo_rl.models.generation.megatron import MegatronGeneration
from nemo_rl.models.generation.sglang.config import SGLangConfig
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.generation.trtllm import TrtllmConfig, TrtllmGeneration
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.generation.vllm.config import (
    VLLM_SPARSE_REFIT_TRANSPORTS,
    normalize_vllm_refit_config,
)
from nemo_rl.models.megatron.router_replay import (
    configure_vllm_for_router_replay,
    router_replay_enabled,
    validate_router_replay_transport_path,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
    print_message_log_samples,
    should_log_nemo_gym_full_result_tables,
)
from nemo_rl.utils.memory_tracker import MemoryTracker
from nemo_rl.utils.multimodal_payload_metrics import (
    collect_multimodal_payload_metrics,
    drain_multimodal_payload_metrics,
    merge_multimodal_payload_metrics,
    print_multimodal_payload_metrics,
)
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.routed_experts_ref import retire_routed_experts_through
from nemo_rl.utils.timer import TimeoutChecker, Timer
from nemo_rl.utils.venvs import create_local_venv_on_each_node
from nemo_rl.weight_sync.checkpoint_engine_config import (
    checkpoint_engine_refit_config,
)
from nemo_rl.weight_sync.factory import create_weight_synchronizer
from nemo_rl.algorithms.grpo import (
    _maybe_restore_async_replay_buffer_checkpoint,
    _save_async_replay_buffer_checkpoint,
    RewardScalingConfig,
    AsyncGRPOConfig,
    RewardPenaltyTokenIdsConfig,
    RewardPenaltyConfig,
    _REWARD_PENALTY_FLAGS,
    GRPOConfig,
    GRPOSaveState,
    _initial_grpo_save_state,
    _get_grpo_save_state,
    GRPOLoggerConfig,
    MasterConfig,
    _validate_multimodal_dedup_capability,
    _needs_hf_refit_handshake,
    shutdown_environments,
    extract_initial_prompt_messages,
    dynamic_sampling,
    scale_rewards,
    _resolve_message_level_advantage_penalties,
    _raise_if_reward_penalties_enabled_without_nemo_gym,
    _batch_row_value,
    _async_sample_identity_fields,
    _emit_async_sample_event,
    _apply_message_level_advantage_penalties,
    _apply_configured_message_level_advantage_penalties,
    _preserve_router_replay_routed_experts,
    _policy_dtype,
    _build_async_grpo_train_data,
    _apply_mask_sample_filter,
    _should_log_nemo_gym_responses,
    _write_latest_checkpoint_status,
    _get_effort_config,
    _pad_teacher_logprobs,
    _create_advantage_estimator,
    _clip_grpo_advantages,
    refit_policy_generation,
    _initial_policy_generation_stale,
    _log_mixed_rewards_and_advantages_information,
    _placeholder_seq_logprob_error_metrics,
    _validate_use_kl_in_reward_compat,
    _resolve_logprob_skip_flags,
    compute_and_apply_seq_logprob_error_masking,
    _validation_stop_value,
    _validation_early_stop_message,
)

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)

# ===============================================================================
# Core Algorithm Functions
# ===============================================================================

def add_grpo_token_loss_masks_and_generation_logprobs(
    message_logs: list[LLMMessageLogType | VLMMessageLogType],
) -> None:
    """Add GRPO loss masks and ensure generation logprobs exist in message logs.

    Assistant messages can be part of the original multi-turn prompt history. Only
    generated assistant messages have generation_logprobs, so use that field as the
    trainable-token marker. This function mutates each message in-place by adding a
    token_loss_mask and, when missing, a zero-valued generation_logprobs tensor.
    Router-replay routes get the same treatment via
    :func:`backfill_missing_routed_experts`, so every per-token field is defined
    for every tokenized message before the batch is flattened.

    Args:
        message_logs: Batch of tokenized message logs. Each message must contain a
            ``role`` and ``token_ids`` field. Messages that already contain
            ``generation_logprobs`` are treated as rollout-generated messages.
    """
    backfill_missing_routed_experts(message_logs)
    for message_log in message_logs:
        for j, message in enumerate(message_log):
            role = cast(str, message["role"])
            token_ids = cast(torch.Tensor, message["token_ids"])
            if role == "assistant" and "generation_logprobs" not in message:
                raise RuntimeError("assistant message with no generation_logprobs!")

            if role == "assistant" and "generation_logprobs" in message and j == len(message_log) - 2:
                message["token_loss_mask"] = torch.ones_like(token_ids)
            else:
                message["token_loss_mask"] = torch.zeros_like(token_ids)

            if "generation_logprobs" not in message:
                message["generation_logprobs"] = torch.zeros_like(
                    token_ids, dtype=torch.float32
                )

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
    wrapped_dataloader: StatefulDataLoader | MultipleDataloaderWrapper,
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
    timer = Timer(context={"worker": "driver"})
    timeout = TimeoutChecker(
        timeout=master_config.checkpointing["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()
    memory_tracker = MemoryTracker()

    kv_scales_cache = None  # Cache reused for computed kv scales

    assert policy_generation is not None

    # Check if we need to sync KV cache scales
    # When fallback to policy as the policy_generation, we use getattr to check.
    sync_kv_scales = getattr(policy_generation, "requires_kv_scale_sync", False)

    # common config/state times
    current_step = grpo_save_state.current_step  # current step within an epoch
    total_steps = grpo_save_state.total_steps  # total steps across all epochs
    POLICY_GENERATION_STALE = _initial_policy_generation_stale(
        policy_generation, total_steps
    )
    max_num_steps = master_config.grpo.max_num_steps  # max number of steps to train for
    current_epoch = grpo_save_state.current_epoch  # current epoch
    max_num_epochs = (
        master_config.grpo.max_num_epochs
    )  # max number of epochs to train for
    consumed_samples = (
        grpo_save_state.consumed_samples
    )  # total samples consumed across all epochs
    total_valid_tokens = (
        grpo_save_state.total_valid_tokens
    )  # total valid tokens processed across all epochs
    val_at_start = master_config.grpo.val_at_start
    val_at_end = master_config.grpo.val_at_end
    val_period = master_config.grpo.val_period
    val_start_at = master_config.grpo.val_start_at
    colocated_inference = master_config.policy["generation"]["colocated"]["enabled"]
    refit_buffer_size_gb = master_config.policy.get("refit_buffer_size_gb")
    stop_at_validation_threshold = master_config.grpo.stop_at_validation_threshold
    stop_at_validation_metric = master_config.grpo.stop_at_validation_metric

    # Initialize advantage estimator
    adv_estimator = _create_advantage_estimator(master_config)

    # Run validation at the start if configured
    # TODO: Add validation with kv scales if needed
    if val_at_start and current_step == 0:
        print("\n🔍 Running initial validation...", flush=True)
        memory_tracker.snapshot_start_of_stage("Initial validation", dir())

        if POLICY_GENERATION_STALE:
            refit_policy_generation(
                policy,
                policy_generation,
                colocated_inference,
                _refit_buffer_size_gb=refit_buffer_size_gb,
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
            step=0,
            master_config=master_config,
            logger=logger,
            processor=processor,
        )
        policy_generation.finish_generation()
        logger.log_metrics(val_metrics, current_step, prefix="validation")
        logger.log_metrics(validation_timings, current_step, prefix="timing/validation")
        if master_config.grpo.debug_payload_metrics:
            validation_payload_metrics = drain_multimodal_payload_metrics()
            if validation_payload_metrics:
                logger.log_metrics(
                    validation_payload_metrics,
                    current_step,
                    prefix="validation",
                )
        stop_message = _validation_early_stop_message(
            val_metrics,
            stop_at_validation_threshold,
            stop_at_validation_metric,
            initial=True,
        )
        if stop_message is not None:
            print(stop_message, flush=True)
            # Flush pending checkpoint finalization, like the other early returns.
            checkpointer.shutdown()
            return

    if master_config.data["use_multiple_dataloader"]:
        warnings.warn(
            "When using multiple dataloaders, MultipleDataloaderWrapper operates as an infinite iterator. "
            "As a result, grpo.max_num_epochs will be ignored, and only grpo.max_num_steps will be used. "
            "See https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/grpo.md#multiple-dataloaders for more details."
        )

    ft_save_period = master_config.checkpointing.get("ft_save_period")

    while current_epoch < max_num_epochs and total_steps < max_num_steps:
        memory_tracker.snapshot_start_of_stage("Preparing batch", dir())
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")
        # batch cache is used for DAPO. We store prompts with non-zero standard deviation in this cache.
        batch_cache: BatchedDataDict[DatumSpec] = None
        # This is the number of batches we processed so far at each step to generate responses whose std is non-zero. Maximum threshold is set by dynamic_sampling_max_gen_batches. Used in the case of dynamic sampling.
        dynamic_sampling_num_gen_batches = 0
        next_rollout_group_id = 0

        # Run grpo/dapo training loop (single-turn)
        for batch in wrapped_dataloader:
            refit_metrics: dict[str, float] = {}
            # A central place to store logging data that won't be deleted until the loop ends
            metrics_logging_data = dict()
            metrics = dict()

            if master_config.data["use_multiple_dataloader"]:
                print(
                    f"\n{'=' * 25} Step {current_step + 1}/{max_num_steps} {'=' * 25}",
                    flush=True,
                )
            else:
                print(
                    f"\n{'=' * 25} Step {current_step + 1}/{min(len(wrapped_dataloader), max_num_steps)} {'=' * 25}",
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
                    if (
                        master_config.grpo.deduplicate_multimodal_data
                        and should_use_nemo_gym(master_config)
                    ):
                        attach_initial_nemo_gym_image_payloads(
                            batch,
                            processor,
                            env_config=master_config.env,
                        )
                    # Repeat batch items
                    repeated_batch: BatchedDataDict[DatumSpec] = (
                        batch.repeat_interleave(
                            master_config.grpo.num_generations_per_prompt,
                            share_immutable_media=(
                                master_config.grpo.deduplicate_multimodal_data
                            ),
                        )
                    )
                    print_multimodal_payload_metrics(
                        collect_multimodal_payload_metrics(
                            repeated_batch,
                            "prompt_repeat",
                            enabled=master_config.grpo.debug_payload_metrics,
                        )
                    )

                # Generate responses - this updates the LLMMessageLogType in repeated_batch
                memory_tracker.snapshot_start_of_stage("Generation", dir())
                print(
                    f"▶ Generating responses for batch of size {repeated_batch.size}...",
                    flush=True,
                )
                with timer.time("prepare_for_generation/total"):
                    if POLICY_GENERATION_STALE:
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
                                    make_sequence_length_divisible_by=master_config.policy[
                                        "make_sequence_length_divisible_by"
                                    ],
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
                                calib_flat.get_multimodal_dict(
                                    as_tensors=False,
                                    pixel_dtype=_policy_dtype(master_config.policy),
                                )
                            )
                            calibration_data.to("cpu")
                            kv_scales_cache = policy.calibrate_qkv_fp8_scales(
                                calibration_data, include_q=True
                            )["layers"]

                        refit_metrics = refit_policy_generation(
                            policy,
                            policy_generation,
                            colocated_inference,
                            _refit_buffer_size_gb=refit_buffer_size_gb,
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
                    if should_use_nemo_gym(master_config):
                        # configure_generation_config auto-fills stop_token_ids from the EOS
                        # token, but run_async_nemo_gym_rollout asserts these are unset because
                        # NeMo-Gym manages its own stop criteria. Clear them here so the
                        # assertion reflects user intent (null in YAML) rather than the auto-fill.
                        generation_config: GenerationConfig = {
                            **master_config.policy["generation"],
                            "stop_token_ids": None,
                            "stop_strings": None,
                        }
                        nemo_gym_rollout_result = run_nemo_gym_rollout_sync(
                            policy_generation=policy_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config.policy[
                                "max_total_sequence_length"
                            ],
                            generation_config=generation_config,
                            log_full_result_tables=should_log_nemo_gym_full_result_tables(
                                wandb_enabled=master_config.logger["wandb_enabled"],
                                wandb_config=master_config.logger["wandb"],
                            ),
                            max_rollout_turns=None,
                            greedy=False,
                            effort_config=_get_effort_config(master_config),
                            reward_penalty_config=master_config.reward_penalties,
                            length_penalty_config=master_config.grpo.model_dump(),
                            thinking_tags=get_nemo_gym_thinking_tags(master_config.env),
                            mask_env_flagged_samples=should_mask_flagged_samples(
                                master_config.env
                            ),
                            deduplicate_multimodal_data=(
                                master_config.grpo.deduplicate_multimodal_data
                            ),
                            debug_payload_metrics=(
                                master_config.grpo.debug_payload_metrics
                            ),
                        )
                        repeated_batch = nemo_gym_rollout_result.final_batch
                        rollout_metrics = nemo_gym_rollout_result.rollout_metrics
                        del nemo_gym_rollout_result

                    # Use async rollouts when enabled by config/backend defaults.
                    elif should_use_async_rollouts(master_config.policy["generation"]):
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
                            max_rollout_turns=master_config.grpo.max_rollout_turns,
                            greedy=False,
                            deduplicate_multimodal_data=(
                                master_config.grpo.deduplicate_multimodal_data
                            ),
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
                            max_rollout_turns=master_config.grpo.max_rollout_turns,
                            greedy=False,
                            deduplicate_multimodal_data=(
                                master_config.grpo.deduplicate_multimodal_data
                            ),
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
                    repeated_batch, master_config.grpo.reward_scaling
                )
                # Process rewards with custom reward function
                if master_config.grpo.reward_shaping.enabled:
                    repeated_batch = apply_reward_shaping(
                        repeated_batch, master_config.grpo.reward_shaping
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
                    px = {"role": "user", "content": master_config.grpo.dwrl["bt_prompt"]}
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
                print("â–¶ Processing rewards...,", flush=True)
                with timer.time("reward_calculation"):
                    # Extract rewards from final_batch
                    rewards = repeated_batch["total_reward"]

                    print("â–¶ Computing advantages...", flush=True)
                    # For DAPO with reward shaping, compute std on the raw
                    # pre-shaping reward so dynamic sampling filters prompt
                    # groups on the raw task metric (e.g. acc) instead of on
                    # length-dependent shaped reward variance. Baseline
                    # (which drives advantages) stays on the shaped reward.
                    std_rewards = (
                        repeated_batch["unshaped_total_reward"]
                        if master_config.grpo.use_dynamic_sampling
                        and "unshaped_total_reward" in repeated_batch
                        else None
                    )
                    reward_group_ids = build_rollout_group_ids(
                        repeated_batch.size,
                        master_config.grpo.num_generations_per_prompt,
                        start_group_id=next_rollout_group_id,
                    )
                    next_rollout_group_id += (
                        repeated_batch.size
                        // master_config.grpo.num_generations_per_prompt
                    )
                    # Dynamic sampling may cache and concatenate survivors from
                    # multiple generation batches. Carry the explicit identity
                    # through those transformations instead of rebuilding it.
                    repeated_batch["rollout_group_ids"] = reward_group_ids
                    if master_config.grpo.calculate_advantages_on_gpu:
                        print("Computing advantages on GPU!")
                        # Just fix the device id for now
                        device_id = 0
                        baseline, std = calculate_baseline_and_std_per_prompt(
                            reward_group_ids.cuda(device_id),
                            rewards.cuda(device_id),
                            torch.ones_like(rewards).cuda(device_id),
                            leave_one_out_baseline=master_config.grpo.use_leave_one_out_baseline,
                            std_rewards=(
                                std_rewards.cuda(device_id)
                                if std_rewards is not None
                                else None
                            ),
                        )
                        baseline = baseline.cpu()
                        std = std.cpu()
                    else:
                        baseline, std = calculate_baseline_and_std_per_prompt(
                            reward_group_ids,
                            rewards,
                            torch.ones_like(rewards),
                            leave_one_out_baseline=master_config.grpo.use_leave_one_out_baseline,
                            std_rewards=std_rewards,
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
                        if not master_config.grpo.use_dynamic_sampling
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

                    # Save baseline for logging (before deletion)
                    baseline_for_log = baseline.clone()

                    # Backfill before flattening the full rollout for training.
                    backfill_missing_routed_experts(repeated_batch["message_log"])

                    # Use the sampling group itself as the GRPO identity. Distinct
                    # media-conditioned prompts can have identical text tokens.
                    prompt_ids_for_adv = repeated_batch.pop("rollout_group_ids")
                    del baseline
                    del std

                with timer.time("data_processing"):
                    use_overlong_filtering = master_config.grpo.overlong_filtering
                    if use_overlong_filtering:
                        loss_multiplier = repeated_batch_2["loss_multiplier"].clone()
                        truncated = repeated_batch_2["truncated"]

                        if isinstance(truncated, list):
                            truncated = torch.tensor(truncated, dtype=torch.bool)

                        loss_multiplier[truncated] = 0
                        repeated_batch_2["loss_multiplier"] = loss_multiplier

                    num_mask_sample_filtered = _apply_mask_sample_filter(repeated_batch_2)
                    metrics["num_mask_sample_filtered"] = num_mask_sample_filtered

                    add_grpo_token_loss_masks_and_generation_logprobs(
                        repeated_batch_2["message_log"]
                    )

                    # Convert updated LLMMessageLogType to FlatMessagesType for training
                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch_2["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config.policy[
                            "make_sequence_length_divisible_by"
                        ],
                    )
                    yes_position = tokenizer.encode(master_config.grpo.dwrl["score_token"])[0]
                    no_position = tokenizer.encode(master_config.grpo.dwrl["opposite_token"])[0]
                    yes_tensor = torch.tensor([yes_position], device=flat_messages["token_ids"].device).long()
                    list_with_yes = [torch.cat([flat_messages["token_ids"][idx][:input_lengths[idx]], yes_tensor, flat_messages["token_ids"][idx][input_lengths[idx]:]], dim=-1) for idx in range(len(input_lengths))]
                    input_ids_with_yes = torch.stack(list_with_yes, dim=0)
                    flat_token_mask = torch.cat([flat_messages["token_loss_mask"], torch.zeros(flat_messages["token_loss_mask"].shape[0], 1)], dim=-1)

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
                    extra_multimodal_data = flat_messages.get_multimodal_dict(
                        as_tensors=False,
                        pixel_dtype=_policy_dtype(master_config.policy),
                    )
                    train_data.update(extra_multimodal_data)
                    print_multimodal_payload_metrics(
                        collect_multimodal_payload_metrics(
                            train_data,
                            "rollout_to_policy",
                            enabled=master_config.grpo.debug_payload_metrics,
                        )
                    )
                    # Router replay (R3) on the legacy data_plane.enabled=false
                    # driver path: routed_experts already rides flat_messages
                    # (attached to message_log during rollout, then batched into
                    # a [B, S, L, K] tensor by batched_message_log_to_flat_message),
                    # but the train_data whitelist above drops it. Copy it back so
                    # the Megatron worker's train-stage router-replay guard finds
                    # it. Mirrors the TQ producer (sync_rollout_actor.py).
                    _preserve_router_replay_routed_experts(
                        train_data, flat_messages, master_config.policy
                    )
                    train_data.to("cpu")

                    metrics_logging_data["content"] = flat_messages["content"]

                memory_tracker.snapshot_start_of_stage("Computing logprobs", dir())
                #skip_prev_logprobs, skip_reference_logprobs = (
                #    _resolve_logprob_skip_flags(master_config)
                #)
                skip_prev_logprobs = False
                skip_reference_logprobs = master_config.grpo.skip_reference_policy_logprobs_calculation
                seq_logprob_error_threshold = (
                    master_config.grpo.seq_logprob_error_threshold
                )

                if not (skip_prev_logprobs and skip_reference_logprobs):
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
                            "token_mask": train_data["token_mask"],
                            "sample_mask": train_data["sample_mask"],
                            **extra_multimodal_data,
                        }
                    )
                    # Router replay (R3): the prev-logprobs forward replays the
                    # recorded routed_experts, so logprob_data must carry the
                    # field too (it is a separate whitelist from train_data). The
                    # reference-policy logprobs call reuses logprob_data but
                    # intentionally ignores routed_experts (require_router_replay
                    # =False short-circuits before the field is read), so a
                    # present-but-unused field here is safe.
                    _preserve_router_replay_routed_experts(
                        logprob_data, flat_messages, master_config.policy
                    )

                    if not skip_prev_logprobs:
                        train_data["prev_logprobs"] = policy.get_logprobs(
                            logprob_data, timer=timer
                        )["logprobs"]
                    else:
                        print(
                            "▶ Skipping prev_logprobs (force_on_policy_ratio=True)...",
                            flush=True,
                        )
                        train_data["prev_logprobs"] = torch.zeros_like(
                            train_data["generation_logprobs"]
                        )

                    if not skip_reference_logprobs:
                        train_data["reference_policy_logprobs"] = (
                            policy.get_reference_policy_logprobs(
                                logprob_data,
                                timer=timer,
                            )["reference_logprobs"]
                        )
                    else:
                        print(
                            "▶ Skipping reference_logprobs (skip_reference_policy_logprobs_calculation=True)...",
                            flush=True,
                        )
                        train_data["reference_policy_logprobs"] = torch.zeros_like(
                            train_data["prev_logprobs"]
                        )

                    del logprob_data
                    del extra_multimodal_data
                
                # Calculate rewards & advantages
                memory_tracker.snapshot_start_of_stage("Processing rewards", dir())
                print("▶ Processing rewards...,", flush=True)
                with timer.time("reward_calculation"):
                    # Extract rewards from final_batch
                    final_logprobs = train_data["prev_logprobs"].gather(-1, input_lengths.unsqueeze(-1)).squeeze(-1)
                    #bt_probs = final_logprobs.exp()
                    
                    _, input_lengths_rb_1 = batched_message_log_to_flat_message(
                        [[y for y in x if y['role'] != 'environment'] for x in repeated_batch["message_log"]],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config.policy["make_sequence_length_divisible_by"],
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
                if skip_prev_logprobs:
                    # Cannot compute seq-level metrics with placeholder prev_logprobs
                    seq_logprob_error_metrics = _placeholder_seq_logprob_error_metrics()
                else:
                    seq_error_result = compute_and_apply_seq_logprob_error_masking(
                        train_data=train_data,
                        rewards=rewards,
                        seq_logprob_error_threshold=seq_logprob_error_threshold,
                    )
                    seq_logprob_error_metrics = seq_error_result
                    if "num_masked_seqs" in seq_logprob_error_metrics:
                        seq_logprob_error_metrics[
                            "num_masked_seqs_by_logprob_error"
                        ] = seq_logprob_error_metrics.pop("num_masked_seqs")

                # Compute advantages with adv_estimator using correct mask and logprobs
                with timer.time("advantage_calculation"):
                    print("▶ Computing advantages...", flush=True)
                    # Get token-level mask: token_mask * sample_mask
                    token_mask = train_data["token_mask"]
                    sample_mask = train_data["sample_mask"]
                    mask = token_mask * sample_mask.unsqueeze(-1)
                    if not master_config.grpo.dwrl.get("use_env_rewards", True):
                        final_probs = final_logprobs.detach().exp()
                        rewards = torch.where(gt.bool(), 1.0 - final_probs, final_probs)

                    
                    train_data["advantages"] = adv_estimator.compute_advantage(
                        prompt_ids=prompt_ids_for_adv,
                        rewards=rewards,
                        mask=mask,
                        repeated_batch=repeated_batch,
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

                    penalty_metrics = (
                        _apply_configured_message_level_advantage_penalties(
                            train_data, repeated_batch["message_log"], master_config
                        )
                    )

                    # Clip advantages to prevent extreme values from small std normalization
                    train_data["advantages"] = _clip_grpo_advantages(
                        train_data["advantages"], master_config.grpo
                    )

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

                is_last_step = total_steps + 1 >= max_num_steps
                if not master_config.data["use_multiple_dataloader"]:
                    is_last_step = is_last_step or (
                        (current_epoch + 1 == max_num_epochs)
                        and (current_step + 1 == len(wrapped_dataloader))
                    )

                early_stop_message: Optional[str] = None
                should_run_validation = (
                    val_period > 0
                    and (total_steps + 1) >= val_start_at
                    and (total_steps + 1) % val_period == 0
                ) or (val_at_end and is_last_step)

                # Keep training and validation traffic in separate metric intervals.
                payload_metrics: dict[str, int | float] = {}
                if master_config.grpo.debug_payload_metrics:
                    payload_metrics = drain_multimodal_payload_metrics()

                # Run validation if it's a validation step or last step with val_at_end
                if should_run_validation:
                    memory_tracker.snapshot_start_of_stage("Validation", dir())
                    if POLICY_GENERATION_STALE:
                        refit_metrics = refit_policy_generation(
                            policy,
                            policy_generation,
                            colocated_inference,
                            _refit_buffer_size_gb=refit_buffer_size_gb,
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
                        processor=processor,
                    )
                    policy_generation.finish_generation()
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )
                    if master_config.grpo.debug_payload_metrics:
                        validation_payload_metrics = drain_multimodal_payload_metrics()
                        if validation_payload_metrics:
                            logger.log_metrics(
                                validation_payload_metrics,
                                total_steps + 1,
                                prefix="validation",
                            )
                    early_stop_message = _validation_early_stop_message(
                        val_metrics,
                        stop_at_validation_threshold,
                        stop_at_validation_metric,
                    )
                    if early_stop_message is not None:
                        # Exit at the end of this step, after checkpointing.
                        print(early_stop_message, flush=True)

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
                if "draft_grad_norm" in train_results:
                    metrics["draft_grad_norm"] = train_results[
                        "draft_grad_norm"
                    ].numpy()
                if master_config.grpo.use_dynamic_sampling:
                    metrics["filtered_reward"] = rewards.numpy()
                    metrics["reward"] = repeated_batch["total_reward"].numpy()

                metrics.update(train_results["all_mb_metrics"])
                metrics.update(gen_step_metrics)
                metrics.update(penalty_metrics)
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
                metrics.update(seq_logprob_error_metrics)

                ## Checkpointing
                consumed_samples += master_config.grpo.num_prompts_per_step
                timeout.mark_iteration()

                # +1 because step is 0-indexed
                should_save_by_step = (
                    is_last_step
                    # Early stop saves the final state like a last step.
                    or early_stop_message is not None
                    or (total_steps + 1) % master_config.checkpointing["save_period"]
                    == 0
                    or (
                        ft_save_period is not None
                        and (total_steps + 1) % ft_save_period == 0
                    )
                )
                # Check if timeout-based checkpointing is enabled in config.
                should_save_by_timeout = timeout.check_save()

                memory_tracker.snapshot_start_of_stage("Checkpointing", dir())
                if master_config.checkpointing["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    policy.prepare_for_training()

                    # +1 because step is 0-indexed
                    grpo_save_state.current_step = current_step + 1
                    grpo_save_state.total_steps = total_steps + 1
                    grpo_save_state.current_epoch = current_epoch
                    grpo_save_state.total_valid_tokens = total_valid_tokens
                    if val_metrics is not None:
                        grpo_save_state.val_reward = val_metrics["accuracy"]
                    elif hasattr(grpo_save_state, "val_reward"):
                        delattr(grpo_save_state, "val_reward")
                    grpo_save_state.consumed_samples = consumed_samples

                    full_metric_name = master_config.checkpointing["metric_name"]
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
                            if hasattr(grpo_save_state, full_metric_name):
                                delattr(grpo_save_state, full_metric_name)
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            setattr(
                                grpo_save_state,
                                full_metric_name,
                                metrics_source[metric_name],
                            )

                    with timer.time("checkpointing"):
                        # Finalize the previous (possibly async) checkpoint before
                        # starting a new one. No-op with sync save / nothing pending.
                        checkpointer.finalize_pending()

                        print(
                            f"Saving checkpoint for step {total_steps + 1}...",
                            flush=True,
                        )
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, vars(grpo_save_state), master_config
                        )
                        policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            )
                            if checkpointer.save_optimizer
                            else None,
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config.checkpointing,
                        )
                        if master_config.data["use_multiple_dataloader"]:
                            for (
                                task_name,
                                task_dataloader,
                            ) in wrapped_dataloader.dataloaders.items():
                                torch.save(
                                    task_dataloader.state_dict(),
                                    os.path.join(
                                        checkpoint_path,
                                        f"train_dataloader_{task_name}.pt",
                                    ),
                                )
                        else:
                            torch.save(
                                wrapped_dataloader.state_dict(),
                                os.path.join(checkpoint_path, "train_dataloader.pt"),
                            )
                        # Finalize in the background. The directory rename is
                        # deferred until any async write completes (via wait_fn);
                        # with sync save it renames immediately. Finalization is
                        # flushed at the next save (finalize_pending) or on exit
                        # (shutdown).
                        checkpointer.begin_finalization(
                            checkpoint_path,
                            wait_fn=policy.finalize_async_save,
                        )

                        # Record last-successful-checkpoint time/step for external
                        # monitoring (parity with async_grpo_train; see
                        # _write_latest_checkpoint_status).
                        _write_latest_checkpoint_status(
                            checkpointer, last_checkpoint_step=total_steps + 1
                        )

            # Logging
            # Log training data
            memory_tracker.snapshot_start_of_stage("Logging", dir())
            if not _should_log_nemo_gym_responses(master_config):
                log_data = {}
                if "agent_ref" in repeated_batch:
                    log_data["agent_ref"] = repeated_batch["agent_ref"]
                log_data["content"] = flat_messages["content"]
                log_data["rewards"] = rewards.tolist()
                if master_config.grpo.use_dynamic_sampling:
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
            if (
                master_config.policy["generation"]
                .get("vllm_cfg", {})
                .get("enable_vllm_metrics_logger", False)
            ):
                log_generation_metrics(
                    generation_logger_metrics,
                    total_steps + 1,
                    master_config.policy["generation"]["vllm_cfg"][
                        "vllm_metrics_logger_interval"
                    ],
                    logger,
                )

            print("\n📊 Training Results:")

            print(f"  • Loss: {metrics['loss']:.4f}")
            if "draft_loss" in metrics:
                print(f"  • Draft Loss: {metrics['draft_loss']:.4f}")
            print(f"  • Generation KL Error: {metrics['gen_kl_error']:.4f}")
            if master_config.grpo.use_dynamic_sampling:
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
                master_config.grpo.num_prompts_per_step
                * master_config.grpo.num_generations_per_prompt
            )
            total_num_gpus = (
                master_config.cluster["num_nodes"]
                * master_config.cluster["gpus_per_node"]
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
                train_results,
                metrics,
                timing_metrics,
                master_config,
                num_prompts_per_step=master_config.grpo.num_prompts_per_step,
                num_generations_per_prompt=master_config.grpo.num_generations_per_prompt,
                is_async_rl=master_config.grpo.async_grpo.enabled,
            )

            if payload_metrics:
                logger.log_metrics(payload_metrics, total_steps + 1, prefix="")

            if refit_metrics:
                logger.log_metrics(refit_metrics, total_steps + 1, prefix="refit")
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
            next_rollout_group_id = 0

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
            if early_stop_message is not None:
                checkpointer.shutdown()
                memory_tracker.snapshot_start_of_stage("", dir())
                return
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
        current_step = 0  # Reset step counter for new epoch

    # Flush the last checkpoint's background finalization on an epoch-bounded
    # exit. Reaching max_num_epochs falls through the while loop and bypasses
    # the inline shutdown() calls at the max_num_steps / timeout early returns,
    # so without this the daemon finalization thread would be killed before the
    # final tmp_step_N is renamed.
    checkpointer.shutdown()


def validate(
    policy: ColocatablePolicyInterface,
    policy_generation: GenerationInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    step: int,
    master_config: MasterConfig,
    logger: Optional[Logger] = None,
    processor: Optional[AutoProcessor] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        assert val_dataloader is not None or master_config.grpo.val_period == 0, (
            "val_dataloader is None, so grpo.val_period must be 0"
        )
        print("  ⚠️ No validation dataloader provided, skipping validation", flush=True)
        return {}, {}

    timer = Timer(context={"worker": "validator"})
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...", flush=True)
        # >= 1 is validated in setup().
        val_num_generations_per_prompt = (
            master_config.grpo.val_num_generations_per_prompt
        )

        total_rewards = []
        bt_probs = []
        total_lengths = []
        all_message_logs = []  # Collect all message logs
        results = []

        max_batches = (
            master_config.grpo.max_val_samples // master_config.grpo.val_batch_size
        )
        for batch_idx, val_batch in enumerate(val_dataloader):
            batch_size = len(val_batch["message_log"])
            active_indices = torch.arange(batch_size)
            additional_metrics_to_report = dict()
            if batch_idx >= max_batches:
                break

            if val_num_generations_per_prompt > 1:
                val_batch = val_batch.repeat_interleave(val_num_generations_per_prompt)

            additional_metrics_to_report = dict()
            # Generate responses (updates the LLMMessageLogType in batch_with_msg_logs)
            # Use async rollouts when enabled by config/backend defaults.
            # We cascade NeMo-Gym first since NeMo-Gym also uses async rollouts.
            if should_use_nemo_gym(master_config):
                if master_config.grpo.deduplicate_multimodal_data:
                    attach_initial_nemo_gym_image_payloads(
                        val_batch,
                        processor,
                        env_config=master_config.env,
                    )
                generation_config = master_config.policy["generation"]
                # Validation-only sampling (e.g. near-greedy validation);
                # defaults to the train profile via the exemplar YAML
                # interpolations. Training rollouts keep policy.generation.
                val_sampling_params = GenerationSamplingParams(
                    temperature=generation_config["val_temperature"],
                    top_p=generation_config["val_top_p"],
                    top_k=generation_config["val_top_k"],
                )
                nemo_gym_rollout_result = run_nemo_gym_rollout_sync(
                    policy_generation=policy_generation,
                    input_batch=val_batch,
                    tokenizer=tokenizer,
                    task_to_env=val_task_to_env,
                    max_seq_len=master_config.policy["max_total_sequence_length"],
                    generation_config=generation_config,
                    sampling_params=val_sampling_params,
                    log_full_result_tables=should_log_nemo_gym_full_result_tables(
                        wandb_enabled=master_config.logger["wandb_enabled"],
                        wandb_config=master_config.logger["wandb"],
                    ),
                    max_rollout_turns=None,
                    greedy=False,
                    effort_config=_get_effort_config(master_config),
                    reward_penalty_config=master_config.reward_penalties,
                    # No length_penalty_config here: validation metrics
                    # (accuracy/pass_k) must reflect the raw env reward, and the
                    # adjustment code groups by the TRAINING stride
                    # (num_generations_per_prompt), which does not match
                    # val_num_generations_per_prompt.
                    thinking_tags=get_nemo_gym_thinking_tags(master_config.env),
                    mask_env_flagged_samples=should_mask_flagged_samples(
                        master_config.env
                    ),
                    deduplicate_multimodal_data=(
                        master_config.grpo.deduplicate_multimodal_data
                    ),
                    debug_payload_metrics=master_config.grpo.debug_payload_metrics,
                )
                val_batch = nemo_gym_rollout_result.final_batch
                gen_metrics = nemo_gym_rollout_result.rollout_metrics
                additional_metrics_to_report = gen_metrics
            elif should_use_async_rollouts(master_config.policy["generation"]):
                val_batch, gen_metrics = run_async_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=master_config.policy["max_total_sequence_length"],
                    max_rollout_turns=master_config.grpo.max_rollout_turns,
                    greedy=False,
                    deduplicate_multimodal_data=(
                        master_config.grpo.deduplicate_multimodal_data
                    ),
                )
            else:
                val_batch, gen_metrics = run_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=master_config.policy["max_total_sequence_length"],
                    max_rollout_turns=master_config.grpo.max_rollout_turns,
                    greedy=False,
                    deduplicate_multimodal_data=(
                        master_config.grpo.deduplicate_multimodal_data
                    ),
                )
            
            # need to calculate reward in the DWRL way
            new_msg_log = []
            for outer in val_batch['message_log']:
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
            for idx, (msg_log, idx, extra_env_info, loss_mx) in enumerate(zip(new_msg_log, val_batch['idx'], val_batch['extra_env_info'], val_batch['loss_multiplier'])):
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
            
            for eei, pred in zip(val_batch["extra_env_info"], actual_rewards.exp().cpu().tolist()):
                gt = eei["preference"]
                
                results.append( int((pred >= 0.5 and gt == 0) or (pred < 0.5 and gt == 1)) )

        # Calculate validation metrics. accuracy is the mean reward over all
        # rollouts; grouped validation (val_num_generations_per_prompt > 1)
        # additionally reports pass@k over each prompt's k rollouts as pass_k.
        num_samples = len(total_rewards)
        pass_k = None
        if num_samples > 0:
            rewards_t = torch.tensor(total_rewards, dtype=torch.float32)
            rewards_mean = rewards_t.mean().item()
            if val_num_generations_per_prompt > 1:
                assert num_samples % val_num_generations_per_prompt == 0, (
                    "Validation rewards must be divisible by "
                    "grpo.val_num_generations_per_prompt"
                )
                pass_k = (
                    (rewards_t.view(-1, val_num_generations_per_prompt) > 0)
                    .any(dim=1)
                    .float()
                    .mean()
                    .item()
                )
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
        if pass_k is not None:
            val_metrics["pass_k"] = pass_k

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


def aggregate_rollout_metrics(
    per_group_metrics: dict[str, list],
) -> dict[str, Any]:
    """Aggregate rollout metrics from multiple trajectory groups.

    Different metric types are aggregated according to their semantics:
    - Histogram observations: flattened into one step-level distribution
    - Metrics ending with "/min" or starting with "min_" (excluding "_rate" suffix): take the minimum
    - Metrics ending with "/max" or starting with "max_" (excluding "_rate" suffix): take the maximum
    - "total_turns": summed
    - Non-numeric values: passed through as-is
    - All other numeric metrics: averaged

    Args:
        per_group_metrics: A dict mapping metric names to lists of per-group values.

    Returns:
        A dict mapping metric names to their aggregated scalar values.
    """
    aggregated = {}
    for k, v in per_group_metrics.items():
        if is_histogram_metric(k):
            aggregated[k] = [observation for group in v for observation in group]
        elif not isinstance(v[0], (int, float)):
            aggregated[k] = v
        elif k.endswith("/min") or (k.startswith("min_") and not k.endswith("_rate")):
            aggregated[k] = min(v)
        elif k.endswith("/max") or (k.startswith("max_") and not k.endswith("_rate")):
            aggregated[k] = max(v)
        elif k == "total_turns":
            aggregated[k] = sum(v)
        elif k == "trajectory_duration_s":
            sorted_v = sorted(v)
            p95_idx = min(int(len(sorted_v) * 0.95), len(sorted_v) - 1)
            aggregated[k] = sum(v) / len(v)
            aggregated["trajectory_duration_s/max"] = max(v)
            aggregated["trajectory_duration_s/p95"] = (
                sorted_v[p95_idx] if sorted_v else 0
            )
        else:
            aggregated[k] = sum(v) / len(v)
    return aggregated


def _startup_pipeline_ready(
    replay_buffer: Any,
    collector_status: dict[str, Any],
    *,
    current_step_ready: bool,
    step: int,
    num_prompts_per_step: int,
    max_trajectory_age_steps: int,
    max_num_steps: int,
) -> bool:
    """Return whether async training can overlap safely with lookahead generation.

    A complete current step is sufficient only after the collector has either
    completed or claimed the next target. The claim preserves the same
    training/generation overlap used by steady-state async training without
    allowing an unrelated reservation to open the startup barrier.
    """
    if not current_step_ready:
        return False

    next_step = step + 1
    need_lookahead = max_trajectory_age_steps > 0 and next_step < max_num_steps
    if not need_lookahead:
        return True

    next_step_ready = ray.get(
        replay_buffer.has_complete_batch.remote(
            next_step, num_prompts_per_step, max_trajectory_age_steps
        )
    )
    return next_step_ready or next_step in collector_status.get(
        "generating_targets", ()
    )


def async_dwrl_train_pairwise(
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
    teacher_worker_groups: Optional[dict[str, Any]] = None,
    alias_to_group_alias: Optional[dict[str, str]] = None,
    processor: Optional[AutoProcessor] = None,
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
        processor: Optional multimodal processor used to attach compact policy
            media to NeMo Gym prompt rows.
    """
    # Ensure we are running with a compatible async generation backend.
    # Async GRPO supports vLLM, Megatron, TRT-LLM, and Dynamo;
    # SGLang async rollouts do not support the async GRPO replay path.
    generation_config = master_config.policy["generation"]
    backend = generation_config.get("backend", "") if generation_config else ""
    assert backend in ("vllm", "megatron", "trtllm", "dynamo"), (
        "Async GRPO supports the vLLM, Megatron, TRT-LLM, and Dynamo generation backends; "
        f"got policy.generation.backend={backend!r}."
    )
    assert should_use_async_rollouts(generation_config), (
        "Async GRPO requires Dynamo, Megatron, or an async vLLM or TRT-LLM "
        "generation engine. Set policy.generation.backend=dynamo, "
        "policy.generation.vllm_cfg.async_engine=true (vLLM), or "
        "policy.generation.trtllm_cfg.async_engine=true (TRT-LLM). "
        "Megatron Inference always uses its async engine."
    )
    assert master_config.loss_fn.use_importance_sampling_correction, (
        "Importance sampling correction must be enabled for async GRPO for good convergence due to off-policy samples!"
    )
    max_generation_failures = master_config.grpo.async_grpo.max_generation_failures

    if router_replay_enabled(master_config.policy) and (
        master_config.data_plane or {}
    ).get("enabled", False):
        raise NotImplementedError(
            "policy.router_replay.enabled=true with async GRPO on this "
            "entrypoint is supported only when data_plane.enabled=false. For "
            "async + TransferQueue, use the SingleController entrypoint: "
            "examples/run_grpo_single_controller.py with e.g. "
            "examples/configs/recipes/llm/"
            "grpo-qwen3-30ba3b-10n8g-megatron-cp2-r3-async-single-controller.yaml"
        )

    if master_config.grpo.async_grpo.max_trajectory_age_steps > 1:
        if not master_config.grpo.async_grpo.in_flight_weight_updates:
            print(
                "⚠️ WARNING: In-flight weight updates must be enabled for async GRPO with max_trajectory_age_steps > 1. "
                "Without in-flight weight updates, having more max_trajectory_age_steps will not give any performance benefit."
            )

    # Import async utilities only when needed
    from nemo_rl.algorithms.async_utils import AsyncTrajectoryCollector, ReplayBuffer

    timer = Timer(context={"worker": "driver"})
    training_wall_start = time.perf_counter()
    timeout = TimeoutChecker(
        timeout=master_config.checkpointing["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()
    assert policy_generation is not None

    # Training state
    step = grpo_save_state.current_step
    POLICY_GENERATION_STALE = _initial_policy_generation_stale(policy_generation, step)
    weight_version = step  # Tracks refitted weight versions
    consumed_samples = grpo_save_state.consumed_samples
    total_valid_tokens = grpo_save_state.total_valid_tokens
    val_period = master_config.grpo.val_period
    val_start_at = master_config.grpo.val_start_at
    val_at_start = master_config.grpo.val_at_start
    val_at_end = master_config.grpo.val_at_end
    colocated_inference = master_config.policy["generation"]["colocated"]["enabled"]
    stop_at_validation_threshold = master_config.grpo.stop_at_validation_threshold
    stop_at_validation_metric = master_config.grpo.stop_at_validation_metric

    assert (not colocated_inference) or (
        isinstance(policy_generation, MegatronGeneration)
    ), "Colocated async GRPO is only supported for the Megatron generation backend."

    # Initialize advantage estimator
    adv_estimator = _create_advantage_estimator(master_config)

    # Calculate minimum buffer size from training requirements
    # In per-prompt buffer mode, one buffer entry is 1 prompt * num_generations_per_prompt
    num_prompts_per_step = master_config.grpo.num_prompts_per_step
    samples_per_prompt_group = master_config.grpo.num_generations_per_prompt
    train_gbs = master_config.policy["train_global_batch_size"]

    # Ensure the buffer has at least one step worth of prompt-groups before training
    min_trajectories_needed = num_prompts_per_step

    print("📊 Buffer requirements calculation:", flush=True)
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

    _replay_py_venv = os.path.dirname(
        os.path.dirname(_replay_py_exec)
    )  # to remove the "bin/python" suffix

    _replay_runtime_env = {
        "py_executable": _replay_py_exec,
        "env_vars": {
            **os.environ,
            "VIRTUAL_ENV": _replay_py_venv,
            "UV_PROJECT_ENVIRONMENT": _replay_py_venv,
        },
    }

    # Calculate optimal buffer size based on generation limits to prevent length bias
    # Each weight version generates exactly num_prompts_per_step trajectories
    # With max_age_steps, we keep trajectories from multiple weight versions
    num_prompts_per_step = master_config.grpo.num_prompts_per_step
    late_arrival_slack = 2
    optimal_buffer_size = (
        num_prompts_per_step * max_trajectory_age_steps * late_arrival_slack
    )

    replay_buffer = ReplayBuffer.options(runtime_env=_replay_runtime_env).remote(
        max_size=optimal_buffer_size,
        drop_incomplete_targets_on_restore=False,
    )

    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    replay_buffer_restore_metadata: dict[str, Any] | None = None
    rollouts_state = None
    if last_checkpoint_path is not None:
        replay_buffer_restore_metadata = _maybe_restore_async_replay_buffer_checkpoint(
            replay_buffer,
            last_checkpoint_path,
            load_replay_buffer=master_config.checkpointing.get("load_replay_buffer"),
            num_prompts_per_step=num_prompts_per_step,
            current_training_step=step,
            max_age_steps=max_trajectory_age_steps,
        )

        rollouts_path = os.path.join(last_checkpoint_path, "rollouts.pt")
        if os.path.exists(rollouts_path):
            # weights_only=False: this is a trusted same-job checkpoint artifact.
            rollouts_state = torch.load(rollouts_path, weights_only=False)

    next_nemo_gym_task_index = max(
        int((rollouts_state or {}).get(NEXT_NEMO_GYM_TASK_INDEX_KEY, 0)),
        int(
            (replay_buffer_restore_metadata or {}).get(NEXT_NEMO_GYM_TASK_INDEX_KEY, 0)
        ),
    )

    # Frontier-aligned resume: the checkpoint saved the dataloader state at
    # the trained frontier rather than the live cursor, so the collector
    # re-yields the covered window and regenerates every prompt that is
    # neither trained nor retained in the restored buffer. Legacy checkpoints
    # (no frontier metadata) keep today's behavior.
    frontier_ordinal = (rollouts_state or {}).get(FRONTIER_ORDINAL_KEY)
    resume_base_ordinal = (rollouts_state or {}).get(RESUME_BASE_ORDINAL_KEY)
    frontier_restore = frontier_ordinal is not None and resume_base_ordinal is not None
    if frontier_restore:
        retained_task_indices = list(
            (replay_buffer_restore_metadata or {}).get(RETAINED_TASK_INDICES_KEY, [])
        )
        # Ordinals trained at/above the cut, covered like retained groups so
        # the re-yielded window regenerates only what was lost.
        trained_task_indices = [
            int(ordinal)
            for ordinal in (rollouts_state or {}).get(TRAINED_TASK_INDICES_KEY, [])
        ]
        covered_task_indices = sorted(
            set(retained_task_indices) | set(trained_task_indices)
        )
        collector_start_kwargs: dict[str, Any] = {
            "next_nemo_gym_task_index": int(resume_base_ordinal),
            "resume_frontier_ordinal": int(frontier_ordinal),
            "resume_covered_task_indices": covered_task_indices,
            # The rewound dataloader re-yields any carried-over remainder.
            "pending_batch": None,
            "ordinals_frontier_aligned": True,
        }
        print(
            "📦 Frontier-aligned resume: dataloader rewound to ordinal "
            f"{resume_base_ordinal}, trained frontier {frontier_ordinal}, "
            f"{len(retained_task_indices)} retained prompt groups, "
            f"{len(trained_task_indices)} trained above the cut"
        )
    else:
        collector_start_kwargs = {
            "next_nemo_gym_task_index": next_nemo_gym_task_index,
            "pending_batch": (rollouts_state or {}).get(PENDING_PROMPTS_KEY),
            # Ordinal == stream position only holds for runs that have used
            # frontier-aligned checkpoints from the start; a legacy resume
            # keeps live-cursor checkpoints.
            "ordinals_frontier_aligned": last_checkpoint_path is None,
        }
        if last_checkpoint_path is not None:
            print(
                "⚠️ Legacy checkpoint resume: frontier-aligned checkpointing "
                "is disabled for this run and every checkpoint descended from "
                "it. Checkpoints will save the live dataloader cursor, so a "
                "resume may skip prompts that were in flight at the save."
            )

    # High-water mark of trained group ordinals, exclusive — the checkpoint
    # frontier. consumed_samples cannot serve here: tolerated generation
    # failures leave stream holes it never sees, so it lags the true stream
    # position.
    trained_frontier_ordinal = (
        int(frontier_ordinal) if frontier_ordinal is not None else consumed_samples
    )
    # Trained ordinals at/above the last checkpoint cut, persisted so a
    # resume covers them instead of re-training them. Pruned at each save;
    # the cut never decreases, so pruning is safe.
    recent_trained_task_indices: set[int] = (
        set(trained_task_indices) if frontier_restore else set()
    )

    _tc_py_exec = get_actor_python_env(
        "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector"
    )
    if _tc_py_exec.startswith("uv"):
        _tc_py_exec = create_local_venv_on_each_node(
            _tc_py_exec,
            "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector",
        )

    _tc_py_venv = os.path.dirname(
        os.path.dirname(_tc_py_exec)
    )  # to remove the "bin/python" suffix

    _tc_runtime_env = {
        "py_executable": _tc_py_exec,
        "env_vars": {
            **os.environ,
            "VIRTUAL_ENV": _tc_py_venv,
            "UV_PROJECT_ENVIRONMENT": _tc_py_venv,
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
        teacher_worker_groups=teacher_worker_groups,
        alias_to_group_alias=alias_to_group_alias,
        on_policy_distillation_cfg=opd_module._opd_cfg(master_config),
        processor=processor,
        **collector_start_kwargs,
    )

    print(
        f"🚀 Starting async GRPO training with buffer_size={optimal_buffer_size}, "
        f"max_age={max_trajectory_age_steps} steps, "
        f"max_generation_failures={max_generation_failures}"
    )

    timer.start("init/total")
    print("⏳ Preparing policy generation for training...", flush=True)
    if POLICY_GENERATION_STALE:
        print("🔄 Refitting policy generation with actual model weights...", flush=True)
        try:
            refit_policy_generation(
                policy,
                policy_generation,
                colocated_inference,
            )
            print("✅ Policy generation refit completed successfully", flush=True)
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

    # Generation must hold the policy's real weights before any backend starts
    # collecting. In particular, vLLM and Dynamo start with dummy weights when
    # the first refit supplies model parameters.
    ray.get(trajectory_collector.set_weight_version.remote(weight_version))
    trajectory_collector.start_collection.remote(dataloader)
    print("📦 Started continuous background trajectory collection")

    print("✅ Policy generation setup complete, proceeding to validation...")

    # Run validation at start if configured
    if val_at_start and step == 0:
        print("\n🔍 Running initial validation...")
        # Pause trajectory collection during initial validation
        ray.get(trajectory_collector.pause.remote())

        initial_val_metrics: Optional[dict[str, Any]] = None
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
                processor=processor,
            )
            initial_val_metrics = val_metrics
            # A colocated engine keeps serving between phases (preserves its
            # KV/prefix cache); the backend makes that call, not the loop.
            policy_generation.finish_generation(release_gpu=False)
            logger.log_metrics(val_metrics, step, prefix="validation")
            logger.log_metrics(validation_timings, step, prefix="timing/validation")
            if master_config.grpo.debug_payload_metrics:
                validation_payload_metrics = drain_multimodal_payload_metrics()
                if validation_payload_metrics:
                    logger.log_metrics(
                        validation_payload_metrics,
                        step,
                        prefix="validation",
                    )
            print("✅ Initial validation completed successfully")
        except Exception as e:
            print(f"❌ Initial validation failed: {e}")
            import traceback

            traceback.print_exc()
            # Continue anyway since validation is optional
        finally:
            # Resume trajectory collection after initial validation
            trajectory_collector.resume.remote()

        stop_message = (
            _validation_early_stop_message(
                initial_val_metrics,
                stop_at_validation_threshold,
                stop_at_validation_metric,
                initial=True,
            )
            if initial_val_metrics is not None
            else None
        )
        if stop_message is not None:
            print(stop_message, flush=True)
            # Flush pending checkpoint finalization and stop rollout
            # generation; the remaining actors are reaped when the driver
            # exits right after this return.
            checkpointer.shutdown()
            try:
                ray.kill(trajectory_collector)
            except Exception as e:
                print(f"Error stopping trajectory collector: {e}")
            try:
                ray.kill(replay_buffer)
            except Exception as e:
                print(f"Error stopping replay buffer: {e}")
            return

    print("✅ All setup complete, starting buffer wait...")
    # Clear logger metrics at start of training
    if policy_generation is not None:
        policy_generation.clear_logger_metrics()

    # Wait for initial buffer fill for the current training step.
    print(
        f"⏳ Waiting for replay buffer to have sufficient trajectories for step {step}..."
    )
    # Initial rollout collection belongs to the first optimizer step.  Record
    # it as a first timing sample; the regular per-step contexts append their
    # samples below and get_timing_metrics(sum) combines both before logging.
    timer.stop("init/total")
    timer.start("total_step_time")
    timer.start("exposed_generation")
    wait_iterations = 0
    while True:
        buffer_size_current = ray.get(replay_buffer.size.remote())
        ray.get(trajectory_collector.check_health.remote())
        current_step_ready = ray.get(
            replay_buffer.has_complete_batch.remote(
                step, num_prompts_per_step, max_trajectory_age_steps
            )
        )

        print(
            f"  Wait iteration {wait_iterations}: buffer_size={buffer_size_current}, "
            f"step {step} ready={current_step_ready}",
            flush=True,
        )

        collector_status = ray.get(trajectory_collector.get_status.remote())
        pipeline_ready = _startup_pipeline_ready(
            replay_buffer,
            collector_status,
            current_step_ready=current_step_ready,
            step=step,
            num_prompts_per_step=num_prompts_per_step,
            max_trajectory_age_steps=max_trajectory_age_steps,
            max_num_steps=master_config.grpo.max_num_steps,
        )
        if current_step_ready and not pipeline_ready:
            print(
                f"  Pipeline barrier: step {step} ready but "
                f"step {step + 1} is not yet claimed — waiting for lookahead "
                f"to prevent resume deadlock"
            )

        if pipeline_ready:
            break

        trajectories_needed = ray.get(
            replay_buffer.get_trajectories_needed.remote(
                step, num_prompts_per_step, max_trajectory_age_steps
            )
        )
        if buffer_size_current >= min_trajectories_needed and trajectories_needed > 0:
            print(
                f"  ⏳ Gap-filling in progress: need {trajectories_needed} more "
                f"trajectories for step {step}"
            )

        if (
            (
                collector_status["data_exhausted"]
                or collector_status.get("errored", False)
            )
            and not collector_status["running"]
            and collector_status["inflight_workers"] == 0
        ):
            awaited_target = step + 1 if current_step_ready else step
            awaited_work = "lookahead claim" if current_step_ready else "buffer fill"
            stop_reason = (
                "dataloader exhausted"
                if collector_status["data_exhausted"]
                else "collector errored"
            )
            recovery_advice = (
                "Increase data.train.max_num_epochs or use a larger dataset."
                if collector_status["data_exhausted"]
                else "Inspect the preceding trajectory collector error."
            )
            raise RuntimeError(
                f"Trajectory collector stopped ({stop_reason}) while waiting for "
                f"{awaited_work} at target={awaited_target}. "
                f"Training cannot start without the required target. "
                f"Collector status: {collector_status}. "
                f"{recovery_advice}"
            )

        wait_iterations += 1
        time.sleep(1.0)

    timer.stop("exposed_generation")
    timer.stop("total_step_time")
    print(f"✅ Buffer ready for step {step}! Starting training loop...")

    ft_save_period = master_config.checkpointing.get("ft_save_period")

    # Main training loop
    try:
        while step < master_config.grpo.max_num_steps:
            ray.get(trajectory_collector.check_health.remote())
            refit_metrics: dict[str, float] = {}
            early_stop_message: Optional[str] = None
            print(
                f"\n{'=' * 25} Step {step + 1}/{master_config.grpo.max_num_steps} {'=' * 25}"
            )
            maybe_gpu_profile_step(policy, step + 1)
            if policy != policy_generation:
                maybe_gpu_profile_step(policy_generation, step + 1)

            with timer.time("total_step_time"):
                sample_mask_metrics: dict[str, int] = {}

                # Sample trajectories from replay buffer
                print("📦 Sampling from replay buffer...")
                with timer.time("exposed_generation"):
                    buffer_size_current = ray.get(replay_buffer.size.remote())
                    print(
                        f"📊 Step coordination: training_step={step}, max_age={max_trajectory_age_steps}, buffer_size={buffer_size_current}",
                        flush=True,
                    )

                    # Sample the required number of per-prompt groups.
                    num_prompt_groups_needed = master_config.grpo.num_prompts_per_step
                    sample_result = ray.get(
                        replay_buffer.sample.remote(
                            num_prompt_groups=num_prompt_groups_needed,
                            current_weight_version=weight_version,
                            max_age_steps=max_trajectory_age_steps,
                        )
                    )
                    if sample_result is not None:
                        print_multimodal_payload_metrics(
                            collect_multimodal_payload_metrics(
                                sample_result,
                                "replay_sample",
                                enabled=master_config.grpo.debug_payload_metrics,
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
                            diag = buffer_debug.get("starvation_diagnostics")
                            if diag:
                                print(
                                    "   📊 Buffer starvation diagnostics (long-tail root cause):"
                                )
                                print(
                                    f"      trajectory_duration_s: mean={diag['trajectory_duration_s']['mean']:.1f}s, "
                                    f"median={diag['trajectory_duration_s']['median']:.1f}s, "
                                    f"max={diag['trajectory_duration_s']['max']:.1f}s, "
                                    f"p95={diag['trajectory_duration_s']['p95']:.1f}s"
                                )
                                print(
                                    f"      max_gen_tokens_per_turn: mean={diag['max_gen_tokens_per_turn_in_buffer']['mean']:.0f}, "
                                    f"median={diag['max_gen_tokens_per_turn_in_buffer']['median']:.0f}, "
                                    f"max={diag['max_gen_tokens_per_turn_in_buffer']['max']:.0f}, "
                                    f"p95={diag['max_gen_tokens_per_turn_in_buffer']['p95']:.0f} "
                                    "(high = long single generations per turn)"
                                )
                                print(
                                    f"      turns_per_sample: mean={diag['turns_per_sample_in_buffer']['mean']:.1f}, "
                                    f"median={diag['turns_per_sample_in_buffer']['median']:.1f}, "
                                    f"max={diag['turns_per_sample_in_buffer']['max']:.0f}, "
                                    f"p95={diag['turns_per_sample_in_buffer']['p95']:.1f} "
                                    "(high = many turns per trajectory)"
                                )

                        collector_status = ray.get(
                            trajectory_collector.get_status.remote()
                        )
                        awaited_target = step
                        print(
                            f"   Awaiting target {awaited_target}; claimed by collector: "
                            f"{awaited_target in collector_status.get('generating_targets', ())}"
                        )
                        if (
                            (
                                collector_status["data_exhausted"]
                                or collector_status.get("errored", False)
                            )
                            and not collector_status["running"]
                            and collector_status["inflight_workers"] == 0
                        ):
                            raise RuntimeError(
                                f"Trajectory collector stopped: dataloader exhausted at training_step={step}. "
                                f"The dataset ran out of data before training could complete. "
                                f"Collector status: {collector_status}. "
                                f"Increase data.train.max_num_epochs or use a larger dataset."
                            )

                        with timer.time("idle/buffer_starvation"):
                            time.sleep(0.5)
                        continue

                    # Extract trajectories and metadata from sample result
                    trajectories = sample_result["trajectories"]
                    avg_trajectory_age = sample_result["avg_trajectory_age"]

                    # Advance the trained frontier from the sampled groups'
                    # own stream ordinals.
                    sampled_ordinals = [
                        trajectory.get(NEMO_GYM_TASK_INDEX_KEY)
                        for trajectory in trajectories
                        if isinstance(trajectory, dict)
                    ]
                    if sampled_ordinals and all(
                        ordinal is not None for ordinal in sampled_ordinals
                    ):
                        trained_frontier_ordinal = max(
                            trained_frontier_ordinal,
                            max(int(ordinal) for ordinal in sampled_ordinals) + 1,
                        )
                        recent_trained_task_indices.update(
                            int(ordinal) for ordinal in sampled_ordinals
                        )

                    print(
                        f"✅ Sampled {len(trajectories)} trajectory groups from buffer (avg age: {avg_trajectory_age:.2f} steps)"
                    )

                    # Concatenate per-prompt groups into a single training batch
                    per_prompt_batches = [t["batch"] for t in trajectories]
                    repeated_batch = BatchedDataDict.from_batches(
                        per_prompt_batches,
                        allow_missing_packed_tensors=(
                            master_config.grpo.deduplicate_multimodal_data
                        ),
                    )

                    # Teacher logprobs are stored in batch dict by collection-time
                    # computation and padded by from_batches. Extract here.
                    trajectory_teacher_logprobs = None
                    if opd_module.is_opd_enabled(master_config):
                        if "teacher_reference_logprobs" in repeated_batch:
                            trajectory_teacher_logprobs = repeated_batch[
                                "teacher_reference_logprobs"
                            ]

                    # Aggregate rollout metrics across groups with proper aggregation per metric type
                    per_group_metrics = {}
                    for t in trajectories:
                        for k, v in t["rollout_metrics"].items():
                            per_group_metrics.setdefault(k, []).append(v)
                    rollout_metrics = aggregate_rollout_metrics(per_group_metrics)

                # Enforce fixed training batch: num_prompts_per_step * num_generations_per_prompt
                expected_batch_size = (
                    master_config.grpo.num_prompts_per_step
                    * master_config.grpo.num_generations_per_prompt
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

                # Baseline spec-decode counters; the delta read at metrics time gives
                # MTP acceptance over this step's generation window (async generation
                # runs continuously in the background collector).
                if hasattr(policy_generation, "snapshot_step_metrics"):
                    policy_generation.snapshot_step_metrics()

                print("▶ Processing rewards...")
                with timer.time("reward_calculation"):
                    # Backfill before flattening the full rollout for training.
                    backfill_missing_routed_experts(repeated_batch["message_log"])
                    # Each replay entry is one complete prompt group and the
                    # concatenation above preserves group-contiguous ordering.
                    prompt_ids_for_adv = build_rollout_group_ids(
                        repeated_batch.size,
                        master_config.grpo.num_generations_per_prompt,
                    )

                    rewards = repeated_batch["total_reward"]

                    print(
                        f"  📊 Rewards stats: min={rewards.min():.4f}, max={rewards.max():.4f}, mean={rewards.mean():.4f}, std={rewards.std():.4f}"
                    )

                # Prepare training data (same as sync version)
                with timer.time("data_processing"):
                    with timer.time("async_sample_masking"):
                        loss_multiplier = repeated_batch["loss_multiplier"].clone()
                        if loss_multiplier.ndim != 1:
                            raise ValueError(
                                "loss_multiplier must be one-dimensional, got "
                                f"shape={tuple(loss_multiplier.shape)}"
                            )
                        batch_size = loss_multiplier.numel()
                        eligible = loss_multiplier != 0
                        sample_mask_metrics = {
                            "num_masked_seqs_by_loss_multiplier": int(
                                (~eligible).sum().item()
                            ),
                            "num_masked_seqs_by_empty_response_output": 0,
                            "num_masked_seqs_by_overlong_filtering": 0,
                            "num_masked_seqs_by_rollout": 0,
                            "num_masked_seqs_by_logprob_error": 0,
                        }

                        # Attribute overlapping masks in precedence order so each
                        # row contributes to exactly one reason metric.
                        for row_index in (
                            torch.nonzero(~eligible, as_tuple=False).flatten().tolist()
                        ):
                            _emit_async_sample_event(
                                "sample_masked",
                                repeated_batch,
                                row_index,
                                message=(
                                    "Async GRPO sample entered training with a zero "
                                    "loss multiplier"
                                ),
                                reason="loss_multiplier",
                                stage="batch_entry",
                            )

                        mask_reasons = [
                            (
                                NEMO_RL_EMPTY_RESPONSE_OUTPUT_KEY,
                                "empty_response_output",
                                "num_masked_seqs_by_empty_response_output",
                                "Masking async GRPO sample because its response output is empty",
                            )
                        ]
                        if master_config.grpo.overlong_filtering:
                            mask_reasons.append(
                                (
                                    "truncated",
                                    "overlong_filtering",
                                    "num_masked_seqs_by_overlong_filtering",
                                    "Masking async GRPO sample because of overlong filtering",
                                )
                            )
                        mask_reasons.append(
                            (
                                "mask_sample",
                                "rollout",
                                "num_masked_seqs_by_rollout",
                                "Masking async GRPO sample because the rollout marked it for masking",
                            )
                        )

                        for (
                            field_name,
                            reason,
                            metric_name,
                            diagnostic_message,
                        ) in mask_reasons:
                            if field_name not in repeated_batch:
                                continue
                            candidate_mask = repeated_batch[field_name]
                            if not isinstance(candidate_mask, torch.Tensor):
                                candidate_mask = torch.as_tensor(candidate_mask)
                            candidate_mask = candidate_mask.reshape(-1)
                            if candidate_mask.numel() != batch_size:
                                raise ValueError(
                                    f"{field_name} has {candidate_mask.numel()} rows; "
                                    f"expected {batch_size}"
                                )
                            candidate_mask = candidate_mask.to(
                                device=eligible.device, dtype=torch.bool
                            )
                            newly_masked = eligible & candidate_mask
                            sample_mask_metrics[metric_name] = int(
                                newly_masked.sum().item()
                            )
                            for row_index in (
                                torch.nonzero(newly_masked, as_tuple=False)
                                .flatten()
                                .tolist()
                            ):
                                _emit_async_sample_event(
                                    "sample_masked",
                                    repeated_batch,
                                    row_index,
                                    message=diagnostic_message,
                                    reason=reason,
                                    stage="pre_training",
                                )
                            loss_multiplier[newly_masked] = 0
                            eligible &= ~newly_masked

                        repeated_batch["loss_multiplier"] = loss_multiplier
                        sample_mask_metrics["num_masked_seqs_total"] = sum(
                            sample_mask_metrics.values()
                        )

                    # Add loss mask to each message
                    # Only unmask assistant messages that were actually generated (have generation_logprobs),
                    # not assistant messages that were part of the prompt history
                    add_grpo_token_loss_masks_and_generation_logprobs(
                        repeated_batch["message_log"]
                    )

                    # Convert to flat format for training
                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config.policy[
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    # Create training data. Advantages are added after logprobs.
                    train_data = _build_async_grpo_train_data(
                        flat_messages,
                        input_lengths,
                        repeated_batch,
                        master_config.policy,
                    )
                    print_multimodal_payload_metrics(
                        collect_multimodal_payload_metrics(
                            train_data,
                            "rollout_to_policy_async",
                            enabled=master_config.grpo.debug_payload_metrics,
                        )
                    )
                    train_data.to("cpu")

                generation_logger_metrics = None
                if policy_generation.blocks_training():
                    print("⏸️ Pausing colocated engine + collector for training...")
                    with timer.time("exposed_generation"):
                        ray.get(trajectory_collector.prepare_for_refit.remote())
                    generation_logger_metrics = policy_generation.get_logger_metrics()
                    policy_generation.finish_generation(release_gpu=True)

                # Training phase (same as sync version)
                skip_prev_logprobs, skip_reference_logprobs = (
                    _resolve_logprob_skip_flags(master_config)
                )
                opd_diagnostic_payloads: dict[str, Any] = {}
                fused_student_topk: Optional[dict[str, torch.Tensor]] = None
                should_fuse_student_topk = (
                    opd_diag._should_log_opd_topk_stats(master_config, step)
                    and opd_diag._get_opd_topk_stats_mode(master_config)
                    == opd_diag.OPD_TOPK_STATS_MODE_STUDENT_ONLINE_TEACHER_DEFERRED
                    and "agent_ref" in repeated_batch
                )
                fused_student_topk_k = (
                    opd_diag._get_opd_topk_stats_k(master_config)
                    if should_fuse_student_topk
                    else None
                )
                seq_logprob_error_threshold = (
                    master_config.grpo.seq_logprob_error_threshold
                )

                if not (skip_prev_logprobs and skip_reference_logprobs):
                    print("▶ Preparing for logprob inference...", flush=True)
                    with timer.time("logprob_inference_prep"):
                        policy.prepare_for_lp_inference()

                print("▶ Computing logprobs...", flush=True)
                with timer.time("policy_and_reference_logprobs"):
                    if not skip_prev_logprobs:
                        student_logprob_kwargs: dict[str, Any] = {"timer": timer}
                        if fused_student_topk_k is not None:
                            student_logprob_kwargs["topk"] = fused_student_topk_k
                        student_logprob_result = policy.get_logprobs(
                            train_data, **student_logprob_kwargs
                        )
                        train_data["prev_logprobs"] = student_logprob_result["logprobs"]
                        if should_fuse_student_topk:
                            fused_student_topk = {
                                "topk_logprobs": student_logprob_result[
                                    "topk_logprobs"
                                ],
                                "topk_indices": student_logprob_result["topk_indices"],
                            }
                    else:
                        train_data["prev_logprobs"] = torch.zeros_like(
                            train_data["generation_logprobs"]
                        )

                    if not skip_reference_logprobs:
                        train_data["reference_policy_logprobs"] = (
                            policy.get_reference_policy_logprobs(
                                train_data,
                                timer=timer,
                            )["reference_logprobs"]
                        )
                    else:
                        print(
                            "▶ Skipping reference_logprobs (skip_reference_policy_logprobs_calculation=True)...",
                            flush=True,
                        )
                        train_data["reference_policy_logprobs"] = torch.zeros_like(
                            train_data["prev_logprobs"]
                        )

                pre_seq_error_sample_loss_mask = train_data["sample_mask"].clone()
                # Seq-level logprob error metrics/masking require real prev_logprobs
                if skip_prev_logprobs:
                    # Cannot compute seq-level metrics with placeholder prev_logprobs
                    seq_logprob_error_metrics = _placeholder_seq_logprob_error_metrics()
                else:
                    seq_error_result = compute_and_apply_seq_logprob_error_masking(
                        train_data=train_data,
                        rewards=rewards,
                        seq_logprob_error_threshold=seq_logprob_error_threshold,
                        sample_metadata=repeated_batch,
                    )
                    seq_logprob_error_metrics = seq_error_result
                    if "num_masked_seqs" in seq_logprob_error_metrics:
                        seq_logprob_error_metrics[
                            "num_masked_seqs_by_logprob_error"
                        ] = seq_logprob_error_metrics.pop("num_masked_seqs")
                num_logprob_error_masks = int(
                    seq_logprob_error_metrics["num_masked_seqs_by_logprob_error"]
                )
                sample_mask_metrics["num_masked_seqs_by_logprob_error"] = (
                    num_logprob_error_masks
                )
                sample_mask_metrics["num_masked_seqs_total"] += num_logprob_error_masks

                # Pad teacher logprobs to match train_data sequence length.
                if trajectory_teacher_logprobs is not None:
                    trajectory_teacher_logprobs = _pad_teacher_logprobs(
                        trajectory_teacher_logprobs, train_data["input_ids"].shape[1]
                    )

                (
                    opd_diagnostic_payloads,
                    opd_diagnostic_metrics,
                ) = opd_diag._collect_opd_diagnostic_payloads(
                    master_config=master_config,
                    step=step,
                    tokenizer=tokenizer,
                    policy=policy,
                    trajectory_collector=trajectory_collector,
                    train_data=train_data,
                    repeated_batch=repeated_batch,
                    rewards=rewards,
                    input_lengths=input_lengths,
                    teacher_logprobs=trajectory_teacher_logprobs,
                    fused_student_topk=fused_student_topk,
                    pre_seq_error_sample_loss_mask=pre_seq_error_sample_loss_mask,
                    have_real_prev_logprobs=not skip_prev_logprobs,
                    timer=timer,
                )
                rollout_metrics.update(opd_diagnostic_metrics)

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
                        repeated_batch=repeated_batch,
                        logprobs_policy=train_data["prev_logprobs"],
                        logprobs_reference=train_data.get("reference_policy_logprobs"),
                        # OPD kwargs (ignored by non-OPD estimators via **kwargs)
                        teacher_logprobs=trajectory_teacher_logprobs.to(
                            train_data["prev_logprobs"].device
                        )
                        if trajectory_teacher_logprobs is not None
                        else None,
                        prev_logprobs=train_data["prev_logprobs"],
                        generation_logprobs=train_data["generation_logprobs"],
                        sample_mask=train_data["sample_mask"],
                    )
                    if (
                        hasattr(adv_estimator, "last_metrics")
                        and adv_estimator.last_metrics
                    ):
                        rollout_metrics.update(adv_estimator.last_metrics)
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

                    penalty_metrics = (
                        _apply_configured_message_level_advantage_penalties(
                            train_data,
                            repeated_batch["message_log"],
                            master_config,
                            log_config=True,
                            sample_metadata=repeated_batch,
                        )
                    )

                    # Clip advantages to prevent extreme values from small std normalization
                    train_data["advantages"] = _clip_grpo_advantages(
                        train_data["advantages"], master_config.grpo
                    )

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

                # weight_version is the target version just consumed. The
                # policy call has joined every worker, while all future
                # buffered/in-flight rollouts target strictly newer versions.
                with timer.time("router_replay_gc"):
                    routed_experts_gc = retire_routed_experts_through(
                        master_config.policy, weight_version
                    )
                if routed_experts_gc is not None:
                    print(
                        "🧹 Retired routed-experts Ray objects through target "
                        f"{weight_version}: {routed_experts_gc}",
                        flush=True,
                    )

                is_last_step = step + 1 == master_config.grpo.max_num_steps
                should_save_by_step = (
                    is_last_step
                    or (step + 1) % master_config.checkpointing["save_period"] == 0
                    or (ft_save_period is not None and (step + 1) % ft_save_period == 0)
                )
                # Checked pre-validation so the wake-deferral below can see it.
                # A crossing during refit/validation is caught by the lookahead in check_save.
                should_save_by_timeout = timeout.check_save()
                will_save_checkpoint = master_config.checkpointing["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                )
                # An early stop (known only after validation) also saves.
                saving_this_step = will_save_checkpoint
                # Save-bound colocated steps leave the engine asleep through save with no transfer.
                defer_wake_for_save = (
                    policy_generation.blocks_training()
                    and will_save_checkpoint
                    and policy_generation.wake_carries_weight_updates()
                )

                print("🔄 Synchronizing policy weights to trajectory collector…")
                if defer_wake_for_save:
                    # Wake-deferral (checkpoint scheduling, which the backend
                    # cannot see): the engine is about to be saved, so leave it
                    # asleep; just drop training-only buffers and version-stamp
                    # the weights. The post-save block wakes it and resumes
                    # collection.
                    print("⏸️ Keeping colocated engine asleep for checkpointing...")
                    # Seed the category with 0.0 (no refit wake happens on
                    # save-bound steps) so efficiency summaries, which skip
                    # missing keys, stay comparable across modes.
                    with timer.time("idle/refit_bubble"):
                        pass
                    with timer.time("offload_before_refit"):
                        policy.offload_before_refit()
                    POLICY_GENERATION_STALE = False
                    weight_version += 1
                    ray.get(
                        trajectory_collector.set_weight_version.remote(weight_version)
                    )
                else:
                    timer.start("idle/refit_bubble")

                    # Measure pending-generation wait as exposed_generation time
                    print("🔄 Coordinating with trajectory collector before refit...")
                    with timer.time("exposed_generation"):
                        ray.get(trajectory_collector.prepare_for_refit.remote())

                    # Collect generation logger metrics for performance reporting
                    # inflight batch sizes and num pending samples are collected from each worker
                    # (colocated collects them before the engine sleeps for training).
                    if generation_logger_metrics is None:
                        generation_logger_metrics = (
                            policy_generation.get_logger_metrics()
                        )

                    # Only the actual refit/weight transfer should be counted as weight_sync
                    print("🔄 Performing policy generation refit...")
                    with timer.time("weight_sync"):
                        refit_metrics = refit_policy_generation(
                            policy,
                            policy_generation,
                            colocated_inference,
                        )
                        POLICY_GENERATION_STALE = False

                        # Update weight version before resuming trajectory collection so that all trajectories are updated with the new correct weight version
                        weight_version += 1
                        ray.get(
                            trajectory_collector.set_weight_version.remote(
                                weight_version
                            )
                        )
                        ray.get(trajectory_collector.resume_after_refit.remote())

                    timer.stop("idle/refit_bubble")

                # Clear logger metrics after each refit (weight sync), starting a new logging cycle
                if policy_generation is not None:
                    policy_generation.clear_logger_metrics()

                # Validation
                val_metrics, validation_timings = None, None
                should_run_validation = (
                    val_period > 0
                    and (step + 1) >= val_start_at
                    and (step + 1) % val_period == 0
                ) or (val_at_end and is_last_step)

                payload_metrics: dict[str, int | float] = {}
                if should_run_validation:
                    # Stop new dispatch before separating the training and
                    # validation payload-metric intervals.
                    ray.get(trajectory_collector.pause.remote())
                    if master_config.grpo.debug_payload_metrics:
                        payload_metrics = merge_multimodal_payload_metrics(
                            [
                                drain_multimodal_payload_metrics(),
                                ray.get(
                                    trajectory_collector.drain_payload_metrics.remote()
                                ),
                            ]
                        )

                # Run validation if it's a validation step or last step with val_at_end
                if should_run_validation:
                    with timer.time("idle/validation"):
                        # No-op on an already-running engine;
                        # wakes the colocated engine when it stayed asleep for a save-bound step.
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
                            processor=processor,
                        )
                        # An early stop triggers a save; must note before engine wake/resume.
                        early_stop_message = _validation_early_stop_message(
                            val_metrics,
                            stop_at_validation_threshold,
                            stop_at_validation_metric,
                        )
                        saving_this_step = will_save_checkpoint or (
                            master_config.checkpointing["enabled"]
                            and early_stop_message is not None
                        )
                        # Save-bound steps need the GPUs for checkpointing,
                        # so the engine must stand down; otherwise a colocated
                        # engine keeps serving (backend's call).
                        policy_generation.finish_generation(
                            release_gpu=saving_this_step
                        )
                        logger.log_metrics(
                            validation_timings, step + 1, prefix="timing/validation"
                        )
                        logger.log_metrics(val_metrics, step + 1, prefix="validation")
                        if master_config.grpo.debug_payload_metrics:
                            validation_payload_metrics = (
                                drain_multimodal_payload_metrics()
                            )
                            if validation_payload_metrics:
                                logger.log_metrics(
                                    validation_payload_metrics,
                                    step + 1,
                                    prefix="validation",
                                )
                        if early_stop_message is not None:
                            # Exit at the end of this step, after checkpointing.
                            print(early_stop_message, flush=True)

                        # Explicit GPU memory cleanup after validation in async mode
                        gc.collect()
                        torch.cuda.empty_cache()

                        if early_stop_message is None:
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
                    **sample_mask_metrics,
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
                if "draft_grad_norm" in train_results:
                    metrics["draft_grad_norm"] = train_results[
                        "draft_grad_norm"
                    ].numpy()
                metrics.update(train_results["all_mb_metrics"])
                metrics.update(penalty_metrics)
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
                metrics.update(seq_logprob_error_metrics)

                # Speculative-decoding (MTP) acceptance metrics for this step.
                if hasattr(policy_generation, "get_step_metrics"):
                    metrics.update(policy_generation.get_step_metrics())

                # Checkpointing (same as sync version)
                consumed_samples += master_config.grpo.num_prompts_per_step
                timeout.mark_iteration()

                if saving_this_step:
                    grpo_save_state.current_step = step + 1
                    grpo_save_state.total_valid_tokens = total_valid_tokens
                    if val_metrics is not None:
                        grpo_save_state.val_reward = val_metrics["accuracy"]
                    elif hasattr(grpo_save_state, "val_reward"):
                        delattr(grpo_save_state, "val_reward")
                    grpo_save_state.consumed_samples = consumed_samples

                    full_metric_name = master_config.checkpointing["metric_name"]
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
                            if hasattr(grpo_save_state, full_metric_name):
                                delattr(grpo_save_state, full_metric_name)
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            setattr(
                                grpo_save_state,
                                full_metric_name,
                                metrics_source[metric_name],
                            )

                    with timer.time("checkpointing"):
                        # Finalize the previous (possibly async) checkpoint before
                        # starting a new one. No-op with sync save / nothing pending.
                        checkpointer.finalize_pending()

                        print(f"Saving checkpoint for step {step + 1}...")
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            step + 1, vars(grpo_save_state), master_config
                        )
                        policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            )
                            if checkpointer.save_optimizer
                            else None,
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config.checkpointing,
                        )
                        # Save the dataloader state at the checkpoint cut
                        # rather than the live cursor; a resume re-yields the
                        # covered window and regenerates what the restored
                        # buffer does not account for. One actor call returns
                        # the snapshot and rollout state as a consistent pair
                        # (separate reads would race the collection loop).
                        collector_checkpoint = ray.get(
                            trajectory_collector.get_checkpoint_state.remote(
                                trained_frontier_ordinal
                            )
                        )
                        dataloader_snapshot = collector_checkpoint["dataloader"]
                        rollouts_state = collector_checkpoint["rollouts"]
                        torch.save(
                            dataloader_snapshot["dataloader_state"],
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        _save_async_replay_buffer_checkpoint(
                            replay_buffer,
                            checkpoint_path,
                        )
                        if dataloader_snapshot["frontier_aligned"]:
                            # Persist the (possibly lowered) cut as the
                            # resume filter threshold, not the trained
                            # frontier.
                            cut_ordinal = int(dataloader_snapshot["frontier_ordinal"])
                            rollouts_state[FRONTIER_ORDINAL_KEY] = cut_ordinal
                            rollouts_state[RESUME_BASE_ORDINAL_KEY] = (
                                dataloader_snapshot["base_ordinal"]
                            )
                            # Ordinals trained at/above the cut: the resume
                            # must not regenerate these. Prune below the cut
                            # (it never decreases).
                            recent_trained_task_indices = {
                                ordinal
                                for ordinal in recent_trained_task_indices
                                if ordinal >= cut_ordinal
                            }
                            rollouts_state[TRAINED_TASK_INDICES_KEY] = sorted(
                                recent_trained_task_indices
                            )
                        torch.save(
                            rollouts_state,
                            os.path.join(checkpoint_path, "rollouts.pt"),
                        )

                        # Defer the directory rename until any async write
                        # completes; flushed at the next save or on training exit.
                        checkpointer.begin_finalization(
                            checkpoint_path,
                            wait_fn=policy.finalize_async_save,
                        )

                        # Record last-successful-checkpoint time/step for external
                        # monitoring (see _write_latest_checkpoint_status).
                        _write_latest_checkpoint_status(
                            checkpointer, last_checkpoint_step=step + 1
                        )

                    # On save-bound steps, engine stayed asleep after training;
                    # wake it unless the loop exits right below (last step, timeout, early stop),
                    # where a wake would only feed the teardown.
                    # The intervening logging runs with the collector paused either way.
                    if defer_wake_for_save and not (
                        is_last_step
                        or should_save_by_timeout
                        or early_stop_message is not None
                    ):
                        # The save onloaded model+optimizer;
                        # generation windows must start from the offloaded state.
                        policy.offload_after_refit()
                        policy_generation.prepare_for_generation()
                        ray.get(trajectory_collector.resume_after_refit.remote())

            # Logging
            # Log training data (match sync GRPO logging payload for parity).
            # NeMo Gym responses can be very large and expensive to log; when
            # env.should_log_nemo_gym_responses is true, skip this jsonl (see
            # _should_log_nemo_gym_responses).
            if not _should_log_nemo_gym_responses(master_config):
                log_data = {}
                if "agent_ref" in repeated_batch:
                    log_data["agent_ref"] = repeated_batch["agent_ref"]
                log_data["content"] = flat_messages_content
                log_data["rewards"] = rewards.tolist()
                if master_config.grpo.use_dynamic_sampling:
                    # In dynamic sampling, `rewards` corresponds to filtered rewards
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
                    log_data, f"train_data_step{step + 1}.jsonl"
                )
                del log_data
            if "sample" in opd_diagnostic_payloads:
                logger.log_batched_dict_as_jsonl(
                    opd_diagnostic_payloads["sample"],
                    f"opd_sample_stats_step{step + 1}.jsonl",
                )
            for payload_key, filename in (
                ("token", f"opd_token_stats_step{step + 1}.pt"),
                (
                    "topk_offline_inputs",
                    f"opd_topk_offline_inputs_step{step + 1}.pt",
                ),
                (
                    "topk_student_stats",
                    f"opd_topk_student_stats_step{step + 1}.pt",
                ),
                ("topk_stats", f"opd_topk_stats_step{step + 1}.pt"),
            ):
                if payload_key in opd_diagnostic_payloads:
                    torch.save(
                        opd_diagnostic_payloads[payload_key],
                        os.path.join(logger.base_log_dir, filename),
                    )
            del opd_diagnostic_payloads
            del train_data
            del flat_messages_content

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )

            # Add buffer stats
            buffer_size_current = ray.get(replay_buffer.size.remote())
            metrics["buffer_size"] = buffer_size_current
            metrics["avg_trajectory_age"] = avg_trajectory_age

            if (
                master_config.policy["generation"]
                .get("vllm_cfg", {})
                .get("enable_vllm_metrics_logger", False)
            ):
                log_generation_metrics(
                    generation_logger_metrics,
                    step + 1,
                    master_config.policy["generation"]["vllm_cfg"][
                        "vllm_metrics_logger_interval"
                    ],
                    logger,
                )

            print("\n📊 Training Results:")
            print(f"  • Loss: {metrics['loss']:.4f}")
            if "draft_loss" in metrics:
                print(f"  • Draft Loss: {metrics['draft_loss']:.4f}")
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
                master_config.cluster["num_nodes"]
                * master_config.cluster["gpus_per_node"]
            )
            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics["global_valid_toks"] / total_time / total_num_gpus
            )
            performance_metrics = print_performance_metrics(
                train_results,
                metrics,
                timing_metrics,
                master_config,
                num_prompts_per_step=master_config.grpo.num_prompts_per_step,
                num_generations_per_prompt=master_config.grpo.num_generations_per_prompt,
                is_async_rl=master_config.grpo.async_grpo.enabled,
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

            if master_config.grpo.debug_payload_metrics and not should_run_validation:
                payload_metrics = merge_multimodal_payload_metrics(
                    [
                        drain_multimodal_payload_metrics(),
                        ray.get(trajectory_collector.drain_payload_metrics.remote()),
                    ]
                )
            if payload_metrics:
                logger.log_metrics(payload_metrics, step + 1, prefix="")

            if refit_metrics:
                logger.log_metrics(refit_metrics, step + 1, prefix="refit")
            logger.log_metrics(performance_metrics, step + 1, prefix="performance")
            logger.log_metrics(metrics, step + 1, prefix="train")
            logger.log_metrics(efficiency_loggable, step + 1, prefix="")
            # step_finished=True here since this is the final log of our current step.
            logger.log_metrics(
                timing_metrics,
                step + 1,
                prefix="timing/train",
                step_finished=True,
            )

            timer.reset()
            step += 1
            if early_stop_message is not None:
                checkpointer.shutdown()
                return
            if should_save_by_timeout:
                checkpointer.shutdown()
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if step >= master_config.grpo.max_num_steps:
                checkpointer.shutdown()
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

    except Exception as e:
        print(f"❌ Error in async loop: {e}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        # Finalize any pending async checkpoint before tearing down workers.
        try:
            checkpointer.shutdown()
        except Exception as e:
            print(f"Error finalizing pending checkpoint: {e}")

        print("🛑 Stopping trajectory collection...")
        try:
            ray.kill(trajectory_collector)
        except Exception as e:
            print(f"Error stopping trajectory collector: {e}")

        try:
            ray.kill(replay_buffer)
        except Exception as e:
            print(f"Error stopping replay buffer: {e}")

        # Environments can have in-flight HTTP requests to generation workers.
        shutdown_environments(task_to_env, val_task_to_env)

        print("🛑 Shutting down generation workers...")
        try:
            policy_generation.shutdown()
        except Exception as e:
            print(f"Error shutting down generation workers: {e}")

        if policy is not policy_generation:
            print("🛑 Shutting down policy workers...")
            try:
                policy.shutdown()
            except Exception as e:
                print(f"Error shutting down policy workers: {e}")

        print("Async GRPO training complete!")
