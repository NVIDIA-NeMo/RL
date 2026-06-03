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

# Generate rollouts for arbitrary environments
# Supports multi-turn rollouts and many simultaneous environments (E.g. you can train on math, code, multi-turn games and more at once)

import asyncio
import copy
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional

import ray
import torch
from transformers import PreTrainedTokenizerBase
from wandb import Histogram

from nemo_rl.data.interfaces import (
    DatumSpec,
    FlatMessagesType,
    LLMMessageLogType,
)
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.models.generation.interfaces import (
    GenerationConfig,
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.utils.timer import Timer
import numpy as np

TokenizerType = PreTrainedTokenizerBase


def generate_responses(
    policy_generation: GenerationInterface,
    generation_input_data: BatchedDataDict[GenerationDatumSpec],
    batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    input_lengths: torch.Tensor,
    include_logprobs: bool = True,
    greedy: bool = False,
) -> tuple[BatchedDataDict[DatumSpec], list[torch.Tensor], dict[str, float | int]]:
    """Generate responses from policy using synchronous generation."""
    # Add stop_strings to generation_input_data if present in the batch
    if "stop_strings" in batch:
        generation_input_data["stop_strings"] = batch["stop_strings"]
    else:
        # Ensure the key exists even if it's None, matching GenerationDatumSpec
        generation_input_data["stop_strings"] = [None] * len(input_lengths)

    # Always use synchronous generation
    generation_outputs = policy_generation.generate(
        generation_input_data, greedy=greedy
    )

    # Extract everything we need from the generation outputs
    output_ids = generation_outputs["output_ids"]
    generation_lengths = generation_outputs["generation_lengths"]
    unpadded_sequence_lengths = generation_outputs["unpadded_sequence_lengths"]

    # Extract truncated info if available (response hit max_tokens without stop token)
    response_truncated = generation_outputs.get("truncated")

    # Extract generated parts
    generated_ids = []
    for i in range(len(input_lengths)):
        input_len = input_lengths[i].item()
        total_length = unpadded_sequence_lengths[i].item()
        full_output = output_ids[i]
        generated_part = full_output[input_len:total_length]
        generated_ids.append(generated_part)

    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Append to message log
    for i, (text, input_length, total_length) in enumerate(
        zip(generated_texts, input_lengths, unpadded_sequence_lengths)
    ):
        assistant_message = {
            "role": "assistant",
            "content": text,
            "token_ids": output_ids[i, input_length:total_length],
        }

        if include_logprobs and "logprobs" in generation_outputs:
            assistant_message["generation_logprobs"] = generation_outputs["logprobs"][
                i, input_length:total_length
            ]

        batch["message_log"][i].append(assistant_message)

    # Generation metrics
    gen_metrics = {
        "mean_generation_length": generation_lengths.float().mean().item(),
        "total_generated_tokens": generation_lengths.sum().item(),
    }

    # Add response_truncated to gen_metrics for use by caller
    if response_truncated is not None:
        gen_metrics["_response_truncated"] = response_truncated

    return batch, generated_ids, gen_metrics


async def generate_responses_async(
    policy_generation: GenerationInterface,
    generation_input_data: BatchedDataDict[GenerationDatumSpec],
    batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    input_lengths: torch.Tensor,
    include_logprobs: bool = True,
    greedy: bool = False,
) -> tuple[BatchedDataDict[DatumSpec], list[torch.Tensor], dict[str, float | int]]:
    """Async version of generate_responses that properly calls generate_async."""
    # Add stop_strings to generation_input_data if present in the batch
    if "stop_strings" in batch:
        generation_input_data["stop_strings"] = batch["stop_strings"]
    else:
        # Ensure the key exists even if it's None, matching GenerationDatumSpec
        generation_input_data["stop_strings"] = [None] * len(input_lengths)

    # Check if this is vLLM with async_engine enabled
    use_async_generation = (
        hasattr(policy_generation, "cfg")
        and "vllm_cfg" in policy_generation.cfg
        and policy_generation.cfg["vllm_cfg"]["async_engine"]
        and hasattr(policy_generation, "generate_async")
    )

    assert use_async_generation, (
        "Async generation is not enabled. Please enable async generation by setting async_engine=True in the vllm_cfg section of the policy config."
    )

    # Use async generation with per-sample streaming
    collected_indexed_outputs: list[
        tuple[int, BatchedDataDict[GenerationOutputSpec]]
    ] = []
    async for original_idx, single_item_output in policy_generation.generate_async(
        generation_input_data, greedy=greedy
    ):
        collected_indexed_outputs.append((original_idx, single_item_output))

    # Sort by original_idx to ensure order matches generation_input_data
    collected_indexed_outputs.sort(key=lambda x: x[0])

    # Extract in correct order
    ordered_batched_data_dicts = [item for _, item in collected_indexed_outputs]

    assert ordered_batched_data_dicts, (
        "Generation returned no outputs for a non-empty batch."
    )

    generation_outputs = BatchedDataDict.from_batches(
        ordered_batched_data_dicts,
        pad_value_dict={"output_ids": tokenizer.pad_token_id, "logprobs": 0.0},
    )

    # Extract everything we need from the generation outputs
    output_ids = generation_outputs["output_ids"]
    generation_lengths = generation_outputs["generation_lengths"]
    unpadded_sequence_lengths = generation_outputs["unpadded_sequence_lengths"]

    # Extract truncated info if available (response hit max_tokens without stop token)
    response_truncated = generation_outputs.get("truncated")

    # Extract generated parts
    generated_ids = []
    for i in range(len(input_lengths)):
        input_len = input_lengths[i].item()
        total_length = unpadded_sequence_lengths[i].item()
        full_output = output_ids[i]
        generated_part = full_output[input_len:total_length]
        generated_ids.append(generated_part)

    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Append to message log
    for i, (text, input_length, total_length) in enumerate(
        zip(generated_texts, input_lengths, unpadded_sequence_lengths)
    ):
        assistant_message = {
            "role": "assistant",
            "content": text,
            "token_ids": output_ids[i, input_length:total_length],
        }

        if include_logprobs and "logprobs" in generation_outputs:
            assistant_message["generation_logprobs"] = generation_outputs["logprobs"][
                i, input_length:total_length
            ]

        batch["message_log"][i].append(assistant_message)

    # Generation metrics
    gen_metrics = {
        "mean_generation_length": generation_lengths.float().mean().item(),
        "total_generated_tokens": generation_lengths.sum().item(),
    }
    # Attach worker metadata if present (async vLLM path)
    if "gen_leader_worker_idx" in generation_outputs:
        # generation_outputs carries this as a 1-length list per row; convert to int
        v = generation_outputs["gen_leader_worker_idx"][0]
        try:
            gen_metrics["gen_leader_worker_idx"] = (
                int(v[0]) if isinstance(v, list) else int(v)
            )
        except Exception as e:
            print(f"Error occurred while extracting gen_leader_worker_idx: {e}")

    # Add response_truncated to gen_metrics for use by caller
    if response_truncated is not None:
        gen_metrics["_response_truncated"] = response_truncated

    return batch, generated_ids, gen_metrics


def calculate_rewards(
    batch: BatchedDataDict[DatumSpec],
    task_to_env: dict[str, EnvironmentInterface],
) -> EnvironmentReturn:
    """Calculate rewards for generated responses and get environment feedback.

    Args:
        batch: Batch containing message_log (LLMMessageLogType) with generated responses
        task_to_env: Dictionary mapping task names to their corresponding environments

    Returns:
        EnvironmentReturn namedtuple containing:
            - observations: List of observations from the environment for the next turn.
            - metadata: List of extracted metadata from the environment.
            - next_stop_strings: List of stop strings for the next generation step.
            - rewards: Tensor of rewards for the last turn.
            - terminateds: Tensor of booleans indicating if an episode ended naturally.
    """
    # Extract message logs for environment (most recent interaction)
    to_env = [
        get_keys_from_message_log(batch["message_log"][i], ["role", "content"])
        for i in range(len(batch["message_log"]))
    ]
    task_names = batch["task_name"]

    # Group messages by task type
    task_groups: dict[str, list[tuple[int, LLMMessageLogType]]] = {}
    for i, task_name in enumerate(task_names):
        if task_name not in task_groups:
            task_groups[task_name] = []
        task_groups[task_name].append((i, to_env[i]))

    # Calculate rewards for each task group concurrently
    futures = []
    future_to_indices = {}  # Map future to its corresponding indices
    for task_name, group in task_groups.items():
        if task_name not in task_to_env:
            raise ValueError(f"No environment found for task type: {task_name}")

        # Extract indices and messages for this group
        indices = [idx for idx, _ in group]
        messages = [msg for _, msg in group]

        # Get corresponding environment info
        env_info = [batch["extra_env_info"][i] for i in indices]

        # Submit task to environment and store future
        future = task_to_env[task_name].step.remote(messages, env_info)  # type: ignore # ray actor call
        futures.append(future)
        future_to_indices[future] = indices

    results = ray.get(futures)
    all_rewards = []
    all_env_observations = []
    all_terminateds = []
    all_next_stop_strings = []
    all_metadata = []  # Store extracted metadata
    all_indices_order = []
    all_answers = []

    for future, result in zip(futures, results):
        indices = future_to_indices[future]
        # Environment step returns: EnvironmentReturn
        (
            env_observations,
            metadata,
            next_stop_strings,
            task_rewards,
            terminateds,
            answers,
        ) = result
        if next_stop_strings is None:
            next_stop_strings = [None] * len(task_rewards)
        if answers is None:
            answers = [None] * len(task_rewards)

        # Store results with their original indices
        for i, idx in enumerate(indices):
            all_indices_order.append(idx)
            all_rewards.append(task_rewards[i])
            all_env_observations.append(env_observations[i])
            all_terminateds.append(terminateds[i])
            all_next_stop_strings.append(next_stop_strings[i])
            all_metadata.append(metadata[i])
            all_answers.append(answers[i])

    # Sort results by original index to maintain order
    sorted_indices = sorted(
        range(len(all_indices_order)), key=lambda k: all_indices_order[k]
    )
    rewards = torch.tensor([all_rewards[i] for i in sorted_indices])
    env_observations = [all_env_observations[i] for i in sorted_indices]
    terminateds = torch.tensor([all_terminateds[i] for i in sorted_indices])
    next_stop_strings = [all_next_stop_strings[i] for i in sorted_indices]
    metadata = [all_metadata[i] for i in sorted_indices]  # Sort metadata
    answers = [all_answers[i] for i in sorted_indices]

    return EnvironmentReturn(
        observations=env_observations,
        metadata=metadata,
        next_stop_strings=next_stop_strings,
        rewards=rewards,
        terminateds=terminateds,
        answers=answers,
    )


def run_multi_turn_rollout(
    policy_generation: GenerationInterface,
    input_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    max_seq_len: int,
    max_rollout_turns: int = 999999,
    greedy: bool = False,
) -> tuple[BatchedDataDict[DatumSpec], dict[str, Any]]:
    """Runs a multi-turn rollout loop, interacting with the environment.

    Args:
        policy_generation: The generation interface (policy).
        input_batch: The starting batch containing initial message logs.
        tokenizer: The tokenizer.
        task_to_env: Dictionary mapping task names to environment instances.
        max_rollout_turns: Maximum number of agent-environment interaction turns.
        max_seq_len: Maximum sequence length allowed.
        greedy: Whether to use greedy decoding.

    Returns:
        Tuple containing:
            - BatchedDataDict with the full interaction history and accumulated rewards
            - Dictionary of rollout metrics
    """
    current_batch = input_batch.copy()  # Work on a copy
    batch_size = len(current_batch["message_log"])
    active_indices = torch.arange(batch_size)
    total_rewards = torch.zeros(batch_size, dtype=torch.float32)

    # Initialize stop_strings from the initial batch if present
    current_stop_strings = current_batch.get("stop_strings", [None] * batch_size)

    # Tracking metrics for each sample
    sample_turn_counts = torch.zeros(batch_size, dtype=torch.int32)
    sample_token_counts = torch.zeros(batch_size, dtype=torch.int32)
    sample_assistant_token_counts = torch.zeros(batch_size, dtype=torch.int32)
    sample_env_token_counts = torch.zeros(batch_size, dtype=torch.int32)
    sample_terminated = torch.zeros(batch_size, dtype=torch.bool)
    sample_truncated = torch.zeros(batch_size, dtype=torch.bool)
    sample_max_turns_reached = torch.zeros(batch_size, dtype=torch.bool)

    # Tracking per-turn metrics
    total_gen_tokens_per_turn = []
    active_samples_per_turn = []

    for turn in range(max_rollout_turns):
        if len(active_indices) == 0:
            break

        active_samples_per_turn.append(len(active_indices))

        # Convert LLMMessageLogType to FlatMessagesType for generation
        active_batch = current_batch.select_indices(active_indices)
        active_stop_strings = [current_stop_strings[i] for i in active_indices.tolist()]

        active_flat_messages: BatchedDataDict[FlatMessagesType]
        active_flat_messages, active_input_lengths = (
            batched_message_log_to_flat_message(
                active_batch["message_log"],
                pad_value_dict={"token_ids": tokenizer.pad_token_id},
            )
        )

        # Extract input_ids and lengths from the flat messages
        active_input_ids = active_flat_messages["token_ids"]

        # Prepare generation input data
        generation_input_data = BatchedDataDict[GenerationDatumSpec](
            {
                "input_ids": active_input_ids,
                "input_lengths": active_input_lengths,
                "stop_strings": active_stop_strings,
            }
        )
        # add the multimodal data to the generation input data
        multimodal_data = active_flat_messages.get_multimodal_dict(as_tensors=False)
        generation_input_data.update(multimodal_data)

        # keep message log for generation
        if "vllm_content" in active_batch:
            generation_input_data["vllm_content"] = active_batch["vllm_content"]
        if "vllm_images" in active_batch:
            generation_input_data["vllm_images"] = active_batch["vllm_images"]
        if "vllm_videos" in active_batch:
            generation_input_data["vllm_videos"] = active_batch["vllm_videos"]

        # generate_responses updates active_batch["message_log"] in-place
        active_batch, generated_ids, gen_metrics = generate_responses(
            policy_generation,
            generation_input_data,
            active_batch,
            tokenizer,
            input_lengths=active_input_lengths,
            greedy=greedy,
        )

        # Record response truncation (response hit max_tokens without stop token)
        response_truncated = gen_metrics.pop("_response_truncated", None)
        if response_truncated is not None:
            for i, global_idx in enumerate(active_indices.tolist()):
                if response_truncated[i]:
                    sample_truncated[global_idx] = True

        # Record token usage - assistant
        for i, global_idx in enumerate(active_indices.tolist()):
            sample_assistant_token_counts[global_idx] += len(generated_ids[i])
            sample_token_counts[global_idx] += len(generated_ids[i])

        # Track total generated tokens this turn
        total_gen_tokens_per_turn.append(sum(len(ids) for ids in generated_ids))

        # Calculate rewards and get environment feedback
        env_output: EnvironmentReturn = calculate_rewards(active_batch, task_to_env)

        total_rewards[active_indices] += env_output.rewards

        # Update message log for ALL active samples with env observation
        # This must happen BEFORE filtering based on done flags
        truncation_mask = torch.zeros_like(env_output.terminateds, dtype=torch.bool)
        for i, global_idx in enumerate(active_indices.tolist()):
            env_obs_content = env_output.observations[i]["content"]
            # Tokenize the raw content from the environment
            # TODO @sahilj: handle if we want these subsequent messages to have a chat template
            tokenized_obs = tokenizer(
                env_obs_content, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
            # tokenizer returns torch.float32 when env_obs_content is empty
            tokenized_obs = tokenized_obs.to(dtype=torch.int64)

            # check if new message overflows max_seq_len
            if (
                len(tokenized_obs) + len(generated_ids[i]) + active_input_lengths[i]
                >= max_seq_len
            ):
                tokens_left_for_obs = max_seq_len - (
                    len(generated_ids[i]) + active_input_lengths[i]
                )
                assert tokens_left_for_obs >= 0, (
                    f"tokens_left_for_obs={tokens_left_for_obs} should not be negative. This should not happen if the inference engine respects the max sequence length."
                )
                # truncate
                tokenized_obs = tokenized_obs[:tokens_left_for_obs]
                truncation_mask[i] = True
                # Record truncation
                sample_truncated[active_indices[i]] = True

            tokenized_env_obs_message = {
                "role": env_output.observations[i]["role"],
                "content": env_obs_content,
                "token_ids": tokenized_obs,
            }
            current_batch["message_log"][global_idx].append(tokenized_env_obs_message)

            # Record token usage - environment
            sample_env_token_counts[global_idx] += len(tokenized_obs)
            sample_token_counts[global_idx] += len(tokenized_obs)

            # Increment turn count
            sample_turn_counts[global_idx] += 1

        # Determine done samples and update active set
        terminateds = env_output.terminateds.bool()
        done = truncation_mask | terminateds
        sample_terminated[active_indices] |= done

        # Update active indices for the next iteration
        active_indices_local_next = torch.where(~done)[0]
        active_indices = active_indices[active_indices_local_next]
        continuing_indices_global = active_indices  # Indices relative to original batch
        # Get next stop strings and infos corresponding to the indices that are *continuing*
        continuing_next_stops = [
            env_output.next_stop_strings[i] for i in active_indices_local_next.tolist()
        ]
        # Get metadata corresponding to continuing indices, using the correct field name
        continuing_metadata = [
            env_output.metadata[i] for i in active_indices_local_next.tolist()
        ]

        for i, global_idx in enumerate(continuing_indices_global.tolist()):
            # Update stop strings for the next turn
            current_stop_strings[global_idx] = continuing_next_stops[i]
            # Update metadata (extra_env_info) using info from environment
            if continuing_metadata[i] is not None:
                current_batch["extra_env_info"][global_idx] = continuing_metadata[i]

    # Record samples that reached max turns
    sample_max_turns_reached[active_indices] = True

    # Add total rewards to the final batch
    current_batch["total_reward"] = total_rewards
    current_batch["truncated"] = sample_truncated

    # Calculate aggregate metrics
    rollout_metrics = {
        # Overall metrics
        "total_turns": int(sample_turn_counts.sum().item()),
        "avg_turns_per_sample": float(sample_turn_counts.float().mean().item()),
        "max_turns_per_sample": int(sample_turn_counts.max().item()),
        "natural_termination_rate": float(sample_terminated.float().mean().item()),
        "truncation_rate": float(sample_truncated.float().mean().item()),
        "max_turns_reached_rate": float(sample_max_turns_reached.float().mean().item()),
        # Token usage metrics
        "mean_total_tokens_per_sample": float(
            sample_token_counts.float().mean().item()
        ),
        "mean_gen_tokens_per_sample": float(
            sample_assistant_token_counts.float().mean().item()
        ),
        "max_gen_tokens_per_sample": float(
            sample_assistant_token_counts.float().max().item()
        ),
        "mean_env_tokens_per_sample": float(
            sample_env_token_counts.float().mean().item()
        ),
    }
    return current_batch, rollout_metrics


async def async_generate_response_for_sample_turn(
    policy_generation: GenerationInterface,
    sample_message_log: list[dict],
    sample_stop_strings: list[str] | None,
    tokenizer: TokenizerType,
    max_seq_len: int,
    greedy: bool = False,
) -> tuple[list[dict], torch.Tensor, torch.Tensor, dict[str, float]]:
    """Generate a response for a single sample's turn using async generation.

    Args:
        policy_generation: The generation interface to use
        sample_message_log: Message log for a single sample
        sample_stop_strings: Stop strings for this sample
        tokenizer: Tokenizer to use
        max_seq_len: Maximum sequence length
        greedy: Whether to use greedy decoding

    Returns:
        Tuple of (updated_message_log, generated_tokens, input_lengths, generation_metrics)
    """
    from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message

    # Convert single sample to batch format
    batch_message_logs = [sample_message_log]

    # Convert to flat format for generation
    flat_messages, input_lengths = batched_message_log_to_flat_message(
        batch_message_logs,
        pad_value_dict={"token_ids": tokenizer.pad_token_id},
    )

    # Create generation input
    generation_input_data = BatchedDataDict[GenerationDatumSpec](
        {
            "input_ids": flat_messages["token_ids"],
            "input_lengths": input_lengths,
            "stop_strings": [sample_stop_strings],
        }
    )

    # Create a dummy batch for generate_responses_async
    dummy_batch = BatchedDataDict[DatumSpec](
        {
            "message_log": batch_message_logs,
            "stop_strings": [sample_stop_strings],
        }
    )

    # Generate response using the async version
    updated_batch, generated_ids, gen_metrics = await generate_responses_async(
        policy_generation,
        generation_input_data,
        dummy_batch,
        tokenizer,
        input_lengths=input_lengths,
        include_logprobs=True,
        greedy=greedy,
    )

    # Extract results for the single sample
    updated_message_log = updated_batch["message_log"][0]
    generated_tokens = generated_ids[0] if generated_ids else torch.empty(0)

    return updated_message_log, generated_tokens, input_lengths, gen_metrics


async def run_sample_multi_turn_rollout(
    sample_idx: int,
    initial_sample_state: dict,
    policy_generation: GenerationInterface,
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    max_seq_len: int,
    max_rollout_turns: int = 999999,
    greedy: bool = False,
) -> tuple[dict, dict[str, Any]]:
    """Run a multi-turn rollout for a single sample.

    This function manages the complete lifecycle of one sample's interaction.
    Async generation is used internally when available.

    Args:
        sample_idx: Index of this sample in the original batch
        initial_sample_state: Initial state containing message_log, extra_env_info, etc.
        policy_generation: The generation interface
        tokenizer: Tokenizer to use
        task_to_env: Environment mapping
        max_seq_len: Maximum sequence length
        max_rollout_turns: Maximum number of turns
        greedy: Whether to use greedy decoding

    Returns:
        Tuple of (final_sample_state, sample_metrics)
    """
    # Initialize sample state
    current_message_log = copy.deepcopy(initial_sample_state["message_log"])
    current_extra_env_info = copy.deepcopy(initial_sample_state["extra_env_info"])
    current_stop_strings = initial_sample_state.get("stop_strings", None)
    task_name = initial_sample_state["task_name"]

    # Sample-level metrics
    total_reward = 0.0
    turn_count = 0
    token_count = 0
    assistant_token_count = 0
    env_token_count = 0
    terminated = False
    truncated = False
    max_turns_reached = False

    # Track per-turn metrics
    turn_gen_tokens = []
    turn_input_tokens = []
    turn_total_tokens = []
    # Track per-turn per-worker token accounting if available
    per_worker_token_counts = {}  # worker_idx -> token_count

    for turn in range(max_rollout_turns):
        if terminated or truncated:
            break

        turn_count += 1

        # Generate response for this sample using async generation
        try:
            (
                updated_message_log,
                generated_tokens,
                input_lengths,
                gen_metrics,
            ) = await async_generate_response_for_sample_turn(
                policy_generation,
                current_message_log,
                current_stop_strings,
                tokenizer,
                max_seq_len,
                greedy=greedy,
            )
            current_message_log = updated_message_log

            # Check if response was truncated (hit max_tokens without stop token)
            response_truncated = gen_metrics.pop("_response_truncated", None)
            if response_truncated is not None and response_truncated[0]:
                truncated = True

            # Update token counts
            gen_token_count = len(generated_tokens)
            assistant_token_count += gen_token_count
            token_count += gen_token_count
            turn_gen_tokens.append(gen_token_count)
            turn_input_tokens.append(int(input_lengths))
            turn_total_tokens.append(int(input_lengths) + gen_token_count)
            # Per-worker load accounting
            if "gen_leader_worker_idx" in gen_metrics:
                worker_idx = int(gen_metrics["gen_leader_worker_idx"])
                per_worker_token_counts[worker_idx] = (
                    per_worker_token_counts.get(worker_idx, 0) + gen_token_count
                )

        except Exception as e:
            print(f"Error generating response for sample {sample_idx}: {e}")
            break

        # Create single-sample batch for environment interaction
        sample_batch = BatchedDataDict[DatumSpec](
            {
                "message_log": [current_message_log],
                "extra_env_info": [current_extra_env_info],
                "task_name": [task_name],
            }
        )

        # Get environment feedback
        env_output = calculate_rewards(sample_batch, task_to_env)
        # Update total reward
        total_reward += float(env_output.rewards[0].item())
        # Check termination
        terminated = env_output.terminateds[0].item()
        env_obs_content = env_output.observations[0]["content"]
        # Tokenize environment response
        tokenized_obs = tokenizer(
            env_obs_content, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]

        # Check for sequence length overflow
        if input_lengths + gen_token_count + len(tokenized_obs) >= max_seq_len:
            # Truncate environment observation
            max_env_tokens = max_seq_len - input_lengths - gen_token_count
            if max_env_tokens > 0:
                tokenized_obs = tokenized_obs[:max_env_tokens]
            else:
                tokenized_obs = torch.empty(0, dtype=tokenized_obs.dtype)
            truncated = True

        env_message = {
            "role": env_output.observations[0]["role"],
            "content": env_obs_content,
            "token_ids": tokenized_obs,
        }
        current_message_log.append(env_message)

        # Update token counts
        env_token_count += len(tokenized_obs)
        token_count += len(tokenized_obs)

        # Update sample state for next turn
        if not terminated and not truncated:
            if env_output.next_stop_strings[0] is not None:
                current_stop_strings = env_output.next_stop_strings[0]
            if env_output.metadata[0] is not None:
                current_extra_env_info = env_output.metadata[0]

    # Check if max turns reached
    if turn_count >= max_rollout_turns:
        max_turns_reached = True

    # Prepare final sample state
    final_sample_state = {
        "message_log": current_message_log,
        "extra_env_info": current_extra_env_info,
        "task_name": task_name,
        "total_reward": torch.tensor(total_reward),
        "stop_strings": current_stop_strings,
        "idx": sample_idx,
    }

    # max_gen_tokens_per_turn: Diagnostic for long single generations
    max_gen_tokens_per_turn = max(turn_gen_tokens) if turn_gen_tokens else 0

    # Sample metrics
    sample_metrics = {
        "turn_count": turn_count,
        "total_tokens": token_count,
        "assistant_tokens": assistant_token_count,
        "env_tokens": env_token_count,
        "terminated": terminated,
        "truncated": truncated,
        "max_turns_reached": max_turns_reached,
        "total_reward": total_reward,
        "turn_gen_tokens": turn_gen_tokens,
        "turn_input_tokens": turn_input_tokens,
        "turn_total_tokens": turn_total_tokens,
        "max_gen_tokens_per_turn": max_gen_tokens_per_turn,
        # Pass-through per-worker per-turn accounting for aggregation at batch level
        "per_worker_token_counts": per_worker_token_counts,
    }

    return final_sample_state, sample_metrics


def run_async_multi_turn_rollout(
    policy_generation: GenerationInterface,
    input_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    max_seq_len: int,
    max_rollout_turns: int = 999999,
    greedy: bool = False,
) -> tuple[BatchedDataDict[DatumSpec], dict[str, Any]]:
    """Run multi-turn rollouts with sample-level processing.

    Each sample in the batch proceeds through its interaction independently.
    Async generation is used internally when available but the function is synchronous.

    Args:
        policy_generation: The generation interface (policy)
        input_batch: The starting batch containing initial message logs
        tokenizer: The tokenizer
        task_to_env: Dictionary mapping task names to environment instances
        max_seq_len: Maximum sequence length allowed
        max_rollout_turns: Maximum number of agent-environment interaction turns
        greedy: Whether to use greedy decoding

    Returns:
        Tuple containing:
            - BatchedDataDict with the full interaction history and accumulated rewards
            - Dictionary of rollout metrics
    """

    async def _async_rollout_implementation():
        """Internal async implementation."""
        batch_size = len(input_batch["message_log"])

        # Prepare initial states for each sample
        sample_initial_states = []
        for i in range(batch_size):
            sample_state = {
                "message_log": input_batch["message_log"][i],
                "extra_env_info": input_batch["extra_env_info"][i],
                "task_name": input_batch["task_name"][i],
                "stop_strings": input_batch.get("stop_strings", [None] * batch_size)[i],
                "idx": input_batch.get("idx", list(range(batch_size)))[i],
            }
            sample_initial_states.append(sample_state)

        # Run all samples concurrently
        async def run_single_sample_with_error_handling(i, sample_state):
            """Wrapper to handle errors for individual sample rollouts."""
            try:
                result = await run_sample_multi_turn_rollout(
                    sample_idx=i,
                    initial_sample_state=sample_state,
                    policy_generation=policy_generation,
                    tokenizer=tokenizer,
                    task_to_env=task_to_env,
                    max_seq_len=max_seq_len,
                    max_rollout_turns=max_rollout_turns,
                    greedy=greedy,
                )
                return result
            except Exception as e:
                raise RuntimeError(f"Error in sample {i} rollout: {e}") from e

        # Create tasks for all samples and run them concurrently
        sample_tasks = [
            run_single_sample_with_error_handling(i, sample_state)
            for i, sample_state in enumerate(sample_initial_states)
        ]

        # Execute all sample rollouts concurrently
        sample_results = await asyncio.gather(*sample_tasks, return_exceptions=False)

        # Process results
        final_sample_states = []
        all_sample_metrics = []

        for final_state, sample_metrics in sample_results:
            final_sample_states.append(final_state)
            all_sample_metrics.append(sample_metrics)

        # Reconstruct batch from sample results
        batch_size = len(final_sample_states)
        final_batch = BatchedDataDict[DatumSpec](
            {
                "message_log": [state["message_log"] for state in final_sample_states],
                "extra_env_info": [
                    state["extra_env_info"] for state in final_sample_states
                ],
                "task_name": [state["task_name"] for state in final_sample_states],
                "total_reward": torch.stack(
                    [state["total_reward"] for state in final_sample_states]
                ),
                "idx": [
                    state.get("idx", i) for i, state in enumerate(final_sample_states)
                ],
                "truncated": torch.tensor(
                    [metrics["truncated"] for metrics in all_sample_metrics],
                    dtype=torch.bool,
                ),
            }
        )

        # Preserve additional fields from the original input_batch
        for key in input_batch.keys():
            if key not in final_batch:
                final_batch[key] = input_batch[key]

        # Helper for percentile (buffer starvation diagnostics)
        def _pct(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            sorted_v = sorted(values)
            idx = min(int(len(sorted_v) * p / 100), len(sorted_v) - 1)
            return float(sorted_v[idx])

        turn_counts = [m["turn_count"] for m in all_sample_metrics]
        max_gen_tokens_per_turn_values = [
            m["max_gen_tokens_per_turn"] for m in all_sample_metrics
        ]

        # Aggregate metrics across all samples
        rollout_metrics = {
            # Overall metrics
            "total_turns": sum(turn_counts),
            "avg_turns_per_sample": sum(turn_counts) / batch_size,
            "max_turns_per_sample": max(turn_counts),
            "turns_per_sample/p95": _pct(turn_counts, 95),
            "turns_per_sample/p99": _pct(turn_counts, 99),
            "natural_termination_rate": sum(m["terminated"] for m in all_sample_metrics)
            / batch_size,
            "truncation_rate": sum(m["truncated"] for m in all_sample_metrics)
            / batch_size,
            "max_turns_reached_rate": sum(
                m["max_turns_reached"] for m in all_sample_metrics
            )
            / batch_size,
            # Token usage metrics
            "mean_total_tokens_per_sample": sum(
                m["total_tokens"] for m in all_sample_metrics
            )
            / batch_size,
            "mean_gen_tokens_per_sample": sum(
                m["assistant_tokens"] for m in all_sample_metrics
            )
            / batch_size,
            "max_gen_tokens_per_sample": max(
                m["assistant_tokens"] for m in all_sample_metrics
            ),
            "mean_env_tokens_per_sample": sum(
                m["env_tokens"] for m in all_sample_metrics
            )
            / batch_size,
            # max_gen_tokens_per_turn: Diagnostic for long single generations
            "max_gen_tokens_per_turn/max": max(max_gen_tokens_per_turn_values),
            "max_gen_tokens_per_turn/mean": sum(max_gen_tokens_per_turn_values)
            / batch_size,
            "max_gen_tokens_per_turn/p95": _pct(max_gen_tokens_per_turn_values, 95),
            # Reward metrics
            "mean_total_reward": sum(m["total_reward"] for m in all_sample_metrics)
            / batch_size,
            "max_total_reward": max(m["total_reward"] for m in all_sample_metrics),
            "min_total_reward": min(m["total_reward"] for m in all_sample_metrics),
        }

        # Calculate per-worker token counts
        if "per_worker_token_counts" in all_sample_metrics[0]:
            per_worker_token_counts = {}
            for m in all_sample_metrics:
                for k, v in m["per_worker_token_counts"].items():
                    per_worker_token_counts[k] = per_worker_token_counts.get(k, 0) + v
            rollout_metrics["per_worker_token_counts"] = per_worker_token_counts

        # Collect ISL, OSL, and ISL+OSL metrics for all samples
        rollout_metrics["histogram/gen_tokens_length"] = [
            t for m in all_sample_metrics for t in m["turn_gen_tokens"]
        ]
        rollout_metrics["histogram/input_tokens_length"] = [
            t for m in all_sample_metrics for t in m["turn_input_tokens"]
        ]
        rollout_metrics["histogram/total_tokens_length"] = [
            t for m in all_sample_metrics for t in m["turn_total_tokens"]
        ]

        return final_batch, rollout_metrics

    return asyncio.run(_async_rollout_implementation())


def _tensorize_by_key(message_logs: list, key: str):
    if not message_logs or key not in message_logs[0]:
        return

    for m in message_logs:
        m[key] = torch.tensor(m[key])


@dataclass
class AsyncNemoGymRolloutResult:
    input_ids: torch.Tensor
    final_batch: BatchedDataDict[DatumSpec]
    rollout_metrics: dict[str, Any]
    ng_task_index: Optional[int]


def _calculate_single_metric(
    values: list[float], batch_size: int, key_name: str
) -> dict:
    return {
        f"{key_name}/mean": sum(values) / batch_size,
        f"{key_name}/max": max(values),
        f"{key_name}/min": min(values),
        f"{key_name}/median": statistics.median(values),
        f"{key_name}/stddev": statistics.stdev(values) if len(values) > 1 else math.nan,
        f"{key_name}/histogram": Histogram(values),
    }


def apply_reward_penalties(results: list[dict], master_config: dict | None) -> dict[str, int]:
    """Apply reward penalties to results, setting reward to 0.0 when triggered.

    All penalties are gated by master_config flags. Returns a dict of penalty
    counts keyed by penalty name.

    NOTE: These penalties assume Gym-path message_log structure where roles
    strictly alternate "user" → "assistant". Tool responses are folded into
    user prompt tokens by _postprocess_nemo_gym_to_nemo_rl_result and never
    appear as separate message_log entries. Do not call from non-Gym rollout paths.

    Penalties:
      1. penalize_duplicated_reasoning (text-based)
         Checks response["output"] items. If a "reasoning" item's summary text
         exactly matches the next item's content text (after strip), the model
         is copying its thinking into the final answer verbatim.
         Data: full_result["response"]["output"] — reasoning has summary[0]["text"],
         message has content[0]["text"].

      2. penalize_empty_final_answer (text-based)
         Walks response["output"] in reverse to find the last message-type item.
         If no message item exists or its content text is empty, the model failed
         to produce a final answer. Skipped when the last output item is a
         function_call (model was mid-agentic-loop, not producing an empty answer).
         Data: full_result["response"]["output"] — message items have content[0]["text"].

      3. penalize_eos_token (token-based)
         The EOS token (default id 2, configurable via token_ids.eos) should never
         appear in any assistant generation. Checks message_log assistant entries.
         Data: message_log[i]["token_ids"] where role == "assistant".

      4. penalize_malformed_think_tag (token-based + string-based)
         Two complementary checks to catch malformed think tags:
         a) Token ID check: infers thinking mode from prompt token counts.
            If prompt has open==close: enable_thinking=False, expect 0 open
            and 0 close in generation. If prompt has open==close+1:
            enable_thinking=True, expect 0 open and 1 close in generation.
            Any other prompt pattern or mismatched generation counts is a violation.
            Token IDs configurable via token_ids.think_open / token_ids.think_close.
         b) String check: the model can spell out <think>/</think> with piecemeal
            regular tokens (e.g. "<", "/", "thi", "nk", ">") that bypass special
            token IDs. Checks generation_str (decoded generation text) per output
            item: "<think>" count must be 0 (always in prompt, never generated),
            "</think>" count must be 0 or 1.
         Data: message_log pairs for token IDs, full_result output items for strings.
    """
    counts = {
        "duplicated_reasoning": 0,
        "empty_final_answer": 0,
        "eos_token": 0,
        "malformed_think_tag": 0,
    }
    if not master_config or not results:
        return counts


    # Guard: penalties rely on Gym-path message_log (strictly alternating user/assistant roles).
    # Non-Gym paths may have "environment", "tool", or "system" roles which these checks don't handle.
    any_penalty_enabled = any(
        master_config.get(flag, False)
        for flag in ("penalize_duplicated_reasoning", "penalize_empty_final_answer",
                     "penalize_eos_token", "penalize_malformed_think_tag")
    )
    if any_penalty_enabled:
        for result in results:
            roles = {msg.get("role") for msg in result["message_log"]}
            assert roles <= {"user", "assistant"}, (
                f"apply_reward_penalties requires Gym-path message_log with only 'user' and 'assistant' roles, "
                f"but found roles: {roles}. These penalties are not supported for non-Gym rollout paths."
            )

    # --- Penalty 1: Duplicated reasoning / final answer ---
    if master_config.get("penalize_duplicated_reasoning", False):
        for result in results:
            output_items = result["full_result"].get("response", {}).get("output", [])
            is_duplicated = False
            for item1, item2 in zip(output_items, output_items[1:]):
                if item1.get("type") != "reasoning":
                    continue
                summary = item1.get("summary", [])
                if not summary or "text" not in summary[0]:
                    continue
                reasoning_text = summary[0]["text"].strip()
                content = item2.get("content", "")
                if isinstance(content, list) and content and "text" in content[0]:
                    chat_text = content[0]["text"].strip()
                elif isinstance(content, str):
                    chat_text = content.strip()
                else:
                    continue
                if reasoning_text and chat_text and reasoning_text == chat_text:
                    is_duplicated = True
                    break
            if is_duplicated:
                result["full_result"]["reward"] = 0.0

                counts["duplicated_reasoning"] += 1

    # --- Penalty 2: Empty final answer ---
    if master_config.get("penalize_empty_final_answer", False):
        for result in results:
            output_items = result["full_result"].get("response", {}).get("output", [])
            # Skip if the last output item is a function_call — it is legit for model to
            # produce reasoning and then a function_call as the last output item in PivotRL
            if output_items and output_items[-1].get("type") == "function_call":
                continue
            final_answer_text = None
            for item in reversed(output_items):
                # Skip items without content (function_call, function_call_output, etc.)
                if "content" not in item:
                    continue
                content = item["content"]
                if isinstance(content, list) and content and "text" in content[0]:
                    final_answer_text = content[0]["text"].strip()
                    break
                elif isinstance(content, str):
                    final_answer_text = content.strip()
                    break
            if final_answer_text is None or final_answer_text == "":
                result["full_result"]["reward"] = 0.0

                counts["empty_final_answer"] += 1

    # --- Penalty 3: EOS token in generation ---
    if master_config.get("penalize_eos_token", False):
        token_ids_cfg = master_config.get("token_ids", {})
        eos_token_id = token_ids_cfg.get("eos", 2)
        for result in results:
            has_eos = False
            for msg in result["message_log"]:
                if msg["role"] == "assistant" and eos_token_id in msg["token_ids"]:
                    has_eos = True
                    break
            if has_eos:
                result["full_result"]["reward"] = 0.0

                counts["eos_token"] += 1

    # --- Penalty 4: Malformed think tags (token ID + string) ---
    if master_config.get("penalize_malformed_think_tag", False):
        token_ids_cfg = master_config.get("token_ids", {})
        think_open_token_id = token_ids_cfg.get("think_open", 12)
        think_close_token_id = token_ids_cfg.get("think_close", 13)
        for result in results:
            has_violation = False
            # 4a) Token ID check per (user, assistant) turn pair
            # Infer thinking mode from prompt token counts:
            #   enable_thinking=True:  prompt has open=close+1 (trailing <think>), expect asst: 0 open, 1 close
            #   enable_thinking=False: prompt has open=close (balanced), expect asst: 0 open, 0 close
            msgs = result["message_log"]
            for i in range(len(msgs) - 1):
                if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                    user_ids = msgs[i]["token_ids"]
                    asst_ids = msgs[i + 1]["token_ids"]
                    prompt_open = (user_ids == think_open_token_id).sum().item()
                    prompt_close = (user_ids == think_close_token_id).sum().item()
                    asst_open = (asst_ids == think_open_token_id).sum().item()
                    asst_close = (asst_ids == think_close_token_id).sum().item()
                    if prompt_open == prompt_close:
                        # enable_thinking=False: both tags in prompt, none in generation
                        expected_open, expected_close = 0, 0
                    elif prompt_open == prompt_close + 1:
                        # enable_thinking=True: trailing <think> in prompt, expect </think> in generation
                        expected_open, expected_close = 0, 1
                    else:
                        # Unexpected prompt pattern — flag as violation
                        has_violation = True
                        break
                    if asst_open != expected_open or asst_close != expected_close:
                        has_violation = True
                        break
            # 4b) String check on generation_str per output item
            if not has_violation:
                output_items = result["full_result"].get("response", {}).get("output", [])
                for item in output_items:
                    gen_str = item.get("generation_str", "")
                    if not gen_str:
                        continue
                    if gen_str.count("<think>") > 0 or gen_str.count("</think>") > 1:
                        has_violation = True
                        break
            if has_violation:
                result["full_result"]["reward"] = 0.0

                counts["malformed_think_tag"] += 1

    return counts


async def run_async_nemo_gym_rollout(
    policy_generation: GenerationInterface,
    input_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    generation_config: GenerationConfig,
    num_generations: int,
    max_seq_len: Optional[int] = None,
    max_rollout_turns: Optional[int] = None,
    greedy: bool = False,
    master_config = None,
    group_num: Optional[int] = None,
    returns_entire_batch: bool = False,
) -> AsyncGenerator[AsyncNemoGymRolloutResult, None]:
    """Run multi-turn rollouts with NeMo-Gym. Please refer to the `run_async_multi_turn_rollout` docs for more information on the parameters."""
    # We leverage the same `extra_env_info` key as `run_async_multi_turn_rollout`.
    nemo_gym_rows = input_batch["extra_env_info"]

    # Handle generation parameters up front so we don't hide anything inside here to avoid being unintuitive to the user.
    # NeMo-Gym policy is "What you see is what you get".
    assert not greedy, "`greedy` is not supported in NeMo-Gym path!"
    assert max_rollout_turns is None, (
        "`max_rollout_turns` is not supported in NeMo-Gym path!"
    )
    assert max_seq_len is None, "`max_seq_len` is not supported in NeMo-Gym path!"
    # We don't use these stop criteria
    assert not generation_config["stop_strings"], (
        "Stop strings is not supported in the generation config in NeMo-Gym path!"
    )
    assert not generation_config["stop_token_ids"], (
        "Stop strings is not supported in the generation config in NeMo-Gym path!"
    )
    # Top k is not OpenAI compatible, so NeMo-Gym does not guarantee support over it.
    assert not generation_config["top_k"], (
        "Top k is not supported in the generation config in NeMo-Gym path!"
    )

    assert not returns_entire_batch or len(nemo_gym_rows) == num_generations

    timer = Timer(context={"worker": "rollout"})
    timer_prefix = "timing/rollout"
    timer.start(f"{timer_prefix}/total")

    groupidx_to_row = defaultdict(list)
    for rowidx, row in enumerate(nemo_gym_rows):
        # We may need better handling here. The max tokens set here would be the max new generated tokens, not the total max tokens.
        # Currently, we just rely on the underlying vLLM engine to do the truncation for us using the max model seq len set in the config.
        # row["max_tokens"] = max_seq_len

        responses_create_params = row["responses_create_params"]
        responses_create_params["temperature"] = generation_config["temperature"]
        responses_create_params["top_p"] = generation_config["top_p"]

        if group_num is not None:
            metadata = responses_create_params.get("metadata") or dict()
            metadata["group_num"] = str(group_num)
            responses_create_params["metadata"] = metadata

        # Max new tokens, just like max_seq_len above is ignored and we rely on the underlying vLLM engine for truncation.
        # generation_config["max_new_tokens"]

        row["_rowidx"] = rowidx
        # Assume the gym input samples are ordered
        groupidx_to_row[rowidx // num_generations].append(row)

    # `groupidx_to_rollouts` is ordered in completion order whereas `groupidx_to_row` is the input order
    groupidx_to_rollouts = defaultdict(list)
    groupidx_to_agent_refs = defaultdict(list)
    groupidx_to_rowidx = defaultdict(list)
    with timer.time(f"{timer_prefix}/run_rollouts"):
        nemo_gym_environment = task_to_env["nemo_gym"]
        for future in nemo_gym_environment.run_rollouts.remote(
            nemo_gym_rows, tokenizer, timer_prefix
        ):
            rowidx, result, timing_metrics = await future

            groupidx = rowidx // num_generations
            groupidx_to_rollouts[groupidx].append(result)
            groupidx_to_agent_refs[groupidx].append(nemo_gym_rows[rowidx]["agent_ref"]["name"])
            groupidx_to_rowidx[groupidx].append(rowidx)

            if len(groupidx_to_rollouts[groupidx]) == num_generations:
                agent_refs = groupidx_to_agent_refs.pop(groupidx)
                assert returns_entire_batch or len(set(agent_refs)) == 1, agent_refs

                rows = groupidx_to_row.pop(groupidx)
                rollouts = groupidx_to_rollouts.pop(groupidx)
                rowidxs = groupidx_to_rowidx.pop(groupidx)

                if returns_entire_batch:  # We expect the rollouts to be heterogenous
                    rollouts, _ = zip(*sorted(zip(rollouts, rowidxs), key=lambda p: p[1]))

                async_nemo_gym_rollout_result = _postprocess_single_group(
                    rows,
                    rollouts,
                    timer,
                    timer_prefix,
                    policy_generation,
                    input_batch.slice(groupidx * num_generations, (groupidx + 1) * num_generations),
                    tokenizer,
                    master_config,
                )
                if timing_metrics is not None:
                    timer.stop(f"{timer_prefix}/total")
                    async_nemo_gym_rollout_result.rollout_metrics |= timing_metrics | timer.get_timing_metrics("sum")

                yield async_nemo_gym_rollout_result


def _postprocess_single_group(
    nemo_gym_rows: list[dict],
    results: list[dict],
    timer: Timer,
    timer_prefix: str,
    policy_generation: GenerationInterface,
    input_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    master_config = None,
) -> AsyncNemoGymRolloutResult:
    # for effort level
    len_reward_low = []
    reward_low = []
    lengths3 = [[],[],[]]
    if master_config and 'effort_levels' in master_config:
        low_weight = 0.
        low_penlty = 1.
        low_ub     = 64000
        low_string = ""
        if 'low_weight' in master_config['effort_levels']:
            low_weight = master_config['effort_levels']['low_weight']
        if 'low_penalty' in master_config['effort_levels']:
            low_penlty = master_config['effort_levels']['low_penalty']
        if 'low_ub' in master_config['effort_levels']:
            low_ub = master_config['effort_levels']['low_ub']
        if 'low_string' in master_config['effort_levels']:
            low_string = master_config['effort_levels']['low_string']
        if low_weight>0 and low_string:
            lengths = [len(r["message_log"][-1]["token_ids"]) if r["message_log"][-1]["role"]=="assistant" else 0 for r in results]
            orig_rewards = [r["full_result"]["reward"] for r in results]
            for i in range(len(results)):
                prompt = ''
                for ii in reversed(nemo_gym_rows[i]['responses_create_params']['input']):
                    if 'role' in ii and ii['role'] == 'user' and 'content' in ii:
                        prompt = ii['content']
                        break
                if low_string in prompt:
                    len_reward = min(1.,low_weight * (1. - lengths[i]/low_ub))
                    new_r = orig_rewards[i] + orig_rewards[i] * max(len_reward,0.) + low_penlty * min(len_reward,0.)
                    results[i]["full_result"]["reward"] = new_r
                    len_reward_low.append(len_reward)
                    reward_low.append(new_r)
                    lengths3[0].append(lengths[i])
                else:
                    lengths3[2].append(lengths[i])

    penalty_counts = apply_reward_penalties(results, master_config)

    # Prepare for the rollout metrics calculation below. Not strictly necessary here, but good to have parity with `run_async_multi_turn_rollout`
    with timer.time(f"{timer_prefix}/prepare_for_metrics_calculation", should_log=False):
        batch_size = len(nemo_gym_rows)
        max_total_tokens_per_sample = policy_generation.cfg["vllm_cfg"]["max_model_len"]
        all_sample_metrics = [
            {
                "total_reward": r["full_result"]["reward"],
                "assistant_tokens": sum(
                    len(m["token_ids"])
                    for m in r["message_log"]
                    if m["role"] == "assistant"
                ),
                "total_tokens": sum(len(m["token_ids"]) for m in r["message_log"]),
                "turn_count": sum(1 for m in r["message_log"] if m["role"] == "user"),
                "hit_max_tokens": sum(len(m["token_ids"]) for m in r["message_log"])
                == max_total_tokens_per_sample,
                # max_gen_tokens_per_turn: Diagnostic for long single generations
                "max_gen_tokens_per_turn": max(
                    (
                        len(m["token_ids"])
                        for m in r["message_log"]
                        if m["role"] == "assistant"
                    ),
                    default=0,
                ),
            }
            for r in results
        ]

    # Aggregate metrics across all samples
    with timer.time(f"{timer_prefix}/aggregate_metrics", should_log=False):
        turn_counts = [m["turn_count"] for m in all_sample_metrics]
        max_gen_tokens_per_turn_values = [
            m["max_gen_tokens_per_turn"] for m in all_sample_metrics
        ]

        def _pct(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            sorted_v = sorted(values)
            idx = min(int(len(sorted_v) * p / 100), len(sorted_v) - 1)
            return float(sorted_v[idx])

        rollout_metrics = {
            **_calculate_single_metric(
                turn_counts,
                batch_size,
                "turns_per_sample",
            ),
            "turns_per_sample/p95": _pct(turn_counts, 95),
            "turns_per_sample/p99": _pct(turn_counts, 99),
            **_calculate_single_metric(
                [m["total_tokens"] for m in all_sample_metrics],
                batch_size,
                "total_tokens_per_sample",
            ),
            **_calculate_single_metric(
                [m["assistant_tokens"] for m in all_sample_metrics],
                batch_size,
                "gen_tokens_per_sample",
            ),
            **_calculate_single_metric(
                max_gen_tokens_per_turn_values,
                batch_size,
                "max_gen_tokens_per_turn",
            ),
            "max_gen_tokens_per_turn/p95": _pct(max_gen_tokens_per_turn_values, 95),
            **_calculate_single_metric(
                [m["total_reward"] for m in all_sample_metrics],
                batch_size,
                "total_reward",
            ),
            "natural_termination_rate": sum(
                not m["hit_max_tokens"] for m in all_sample_metrics
            )
            / batch_size,
            "truncation_rate": sum(m["hit_max_tokens"] for m in all_sample_metrics)
            / batch_size,
            # TODO enable this metric. We don't have a clear handle on which tokens are user or tool role.
            # We would probably need to re-tokenize the messages post-hoc to kind of figure this out.
            # "mean_env_tokens_per_sample": sum(
            #     m["env_tokens"] for m in all_sample_metrics
            # )
            # / batch_size,
        }

    # Per-agent misc metrics
    with timer.time(f"{timer_prefix}/per_agent_misc_metrics", should_log=False):
        agent_to_results: dict[str, list[dict]] = defaultdict(list)
        for nemo_gym_row, result in zip(nemo_gym_rows, results):
            agent_ref = nemo_gym_row["agent_ref"]
            agent_name = agent_ref["name"]
            agent_to_results[agent_name].append(result["full_result"])
            result["agent_ref"] = agent_ref

        per_agent_metrics = {}
        for agent_name, agent_results in agent_to_results.items():
            keys = agent_results[0].keys()
            for key in keys:
                values = [
                    float(r[key])
                    for r in agent_results
                    if isinstance(r.get(key), (bool, int, float))
                ]
                if values:
                    per_agent_metrics.update(
                        _calculate_single_metric(
                            values, len(agent_results), f"{agent_name}/{key}"
                        )
                    )

            # Full trajectory dumps are too large to log as a wandb Table (single
            # log call can be tens-to-hundreds of MB, which wandb routes through
            # its Artifact uploader and backs up the metric-commit pipeline,
            # causing the run to be flipped to CRASHED server-side). The raw
            # per-prompt data is already persisted by log_batched_dict_as_jsonl
            # and via the JsonlLogger backend for scalar metrics, so drop the
            # in-metric full_result Table here to keep wandb healthy.
            # Log the full result
            # to_log = [[json.dumps(r, separators=((",", ":")))] for r in agent_results]
            # per_agent_metrics[f"{agent_name}/full_result"] = Table(
            #     data=to_log, columns=["Full result"]
            # )

        rollout_metrics.update(per_agent_metrics)

    # Necessary for downstream nemo rl logging/printing.
    rollout_metrics["mean_gen_tokens_per_sample"] = rollout_metrics[
        "gen_tokens_per_sample/mean"
    ]

    # Convert LLMMessageLogType to FlatMessagesType for generation
    input_batch_for_input_ids = BatchedDataDict[DatumSpec](
        {
            "message_log": [r["input_message_log"] for r in results],
        }
    )
    batched_flat, _ = batched_message_log_to_flat_message(
        input_batch_for_input_ids["message_log"],
        pad_value_dict={"token_ids": tokenizer.pad_token_id},
    )
    input_ids = batched_flat["token_ids"]

    final_batch = BatchedDataDict[DatumSpec](
        {
            "agent_ref": [r["agent_ref"] for r in results],
            "message_log": [r["message_log"] for r in results],
            # length is used downstream for mean_prompt_length
            "length": torch.tensor(
                [len(r["input_message_log"][0]["token_ids"]) for r in results]
            ),
            "loss_multiplier": input_batch["loss_multiplier"],
            # Unnecessary parts of the DatumSpec unused by the GRPO algorithm
            # extra_env_info: dict[str, Any]
            # idx: int
            # task_name: NotRequired[str]
            # stop_strings: NotRequired[list[str]]  # Optional stop strings for generation
            # Extra information not in the DatumSpec used by the GRPO algorithm
            "total_reward": torch.tensor([r["full_result"]["reward"] for r in results]),
            # Add truncated field to match other rollout paths (reusing hit_max_tokens logic)
            "truncated": torch.tensor(
                [m["hit_max_tokens"] for m in all_sample_metrics], dtype=torch.bool
            ),
            # Agent/env-driven mask flag — True means this sample should be masked
            # from the GRPO gradient (kept for advantage computation).
            "mask_sample": torch.tensor(
                [
                    bool(
                        (r["full_result"].get("instance_config") or {}).get(
                            "mask_sample", False
                        )
                    )
                    for r in results
                ],
                dtype=torch.bool,
            ),
        }
    )

    # for effort level
    if len_reward_low:
        rollout_metrics['mean_length_reward_low'] = sum(len_reward_low)/len(len_reward_low)
    if reward_low:
        rollout_metrics['mean_reward_low'] = sum(reward_low)/len(reward_low)
    if lengths3[0]:
        rollout_metrics['mean_length_low'] = sum(lengths3[0])/len(lengths3[0])
        rollout_metrics['median_length_low'] = float(np.median(lengths3[0]))
    if lengths3[2]:
        rollout_metrics['mean_length_high'] = sum(lengths3[2])/len(lengths3[2])
        rollout_metrics['median_length_high'] = float(np.median(lengths3[2]))

    # Penalty metrics — map count keys to (config flag, metric name)
    _PENALTY_METRICS = {
        "duplicated_reasoning": ("penalize_duplicated_reasoning", "reasoning_equal_to_final_answer_rate"),
        "empty_final_answer": ("penalize_empty_final_answer", "empty_final_answer_rate"),
        "eos_token": ("penalize_eos_token", "eos_token_rate"),
        "malformed_think_tag": ("penalize_malformed_think_tag", "malformed_think_tag_rate"),
    }
    if master_config and results:
        for key, (flag, metric_name) in _PENALTY_METRICS.items():
            if master_config.get(flag, False):
                rollout_metrics[metric_name] = penalty_counts[key] / len(results)

    ng_task_index = None
    if nemo_gym_rows and "_ng_task_index" in nemo_gym_rows[0]:
        ng_task_index = int(nemo_gym_rows[0]["_ng_task_index"])
        assert all(
            "_ng_task_index" in row and int(row["_ng_task_index"]) == ng_task_index
            for row in nemo_gym_rows
        ), (
            "Expected a single prompt-group _ng_task_index, got "
            f"{[row.get('_ng_task_index') for row in nemo_gym_rows]}"
        )

    return AsyncNemoGymRolloutResult(
        input_ids=input_ids,
        final_batch=final_batch,
        rollout_metrics=rollout_metrics,
        ng_task_index=ng_task_index,
    )
