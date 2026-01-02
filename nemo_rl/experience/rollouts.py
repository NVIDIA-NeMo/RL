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
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import ray
import torch
from transformers import PreTrainedTokenizerBase
from wandb import Histogram, Table

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

        # Aggregate metrics across all samples
        rollout_metrics = {
            # Overall metrics
            "total_turns": sum(m["turn_count"] for m in all_sample_metrics),
            "avg_turns_per_sample": sum(m["turn_count"] for m in all_sample_metrics)
            / batch_size,
            "max_turns_per_sample": max(m["turn_count"] for m in all_sample_metrics),
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


@dataclass
class AsyncNemoGymRolloutResult:
    input_ids: torch.Tensor
    final_batch: BatchedDataDict[DatumSpec]
    rollout_metrics: dict[str, Any]


def _calculate_single_metric(
    values: list[float], batch_size: int, key_name: str
) -> dict:
    return {
        f"{key_name}/mean": sum(values) / batch_size,
        f"{key_name}/max": max(values),
        f"{key_name}/min": min(values),
        f"{key_name}/median": statistics.median(values),
        f"{key_name}/stddev": statistics.stdev(values),
        f"{key_name}/histogram": Histogram(values),
    }


def run_async_nemo_gym_rollout(
    policy_generation: GenerationInterface,
    input_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    generation_config: GenerationConfig,
    max_seq_len: Optional[int] = None,
    max_rollout_turns: Optional[int] = None,
    greedy: bool = False,
) -> AsyncNemoGymRolloutResult:
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

    timer = Timer()
    timer_prefix = "timing/rollout"
    timer.start(f"{timer_prefix}/total")

    for rowidx, row in enumerate(nemo_gym_rows):
        # We may need better handling here. The max tokens set here would be the max new generated tokens, not the total max tokens.
        # Currently, we just rely on the underlying vLLM engine to do the truncation for us using the max model seq len set in the config.
        # row["max_tokens"] = max_seq_len

        responses_create_params = row["responses_create_params"]
        responses_create_params["temperature"] = generation_config["temperature"]
        responses_create_params["top_p"] = generation_config["top_p"]

        # Max new tokens, just like max_seq_len above is ignored and we rely on the underlying vLLM engine for truncation.
        # generation_config["max_new_tokens"]

        row["_rowidx"] = rowidx

    with timer.time(f"{timer_prefix}/run_rollouts"):
        nemo_gym_environment = task_to_env["nemo_gym"]
        results, rollout_loop_timing_metrics = ray.get(
            nemo_gym_environment.run_rollouts.remote(
                nemo_gym_rows, tokenizer, timer_prefix
            )
        )

    # Prepare for the rollout metrics calculation below. Not strictly necessary here, but good to have parity with `run_async_multi_turn_rollout`
    with timer.time(f"{timer_prefix}/prepare_for_metrics_calculation"):
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
            }
            for r in results
        ]

    # Aggregate metrics across all samples
    with timer.time(f"{timer_prefix}/aggregate_metrics"):
        rollout_metrics = {
            **rollout_loop_timing_metrics,
            **_calculate_single_metric(
                [m["turn_count"] for m in all_sample_metrics],
                batch_size,
                "turns_per_sample",
            ),
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
    with timer.time(f"{timer_prefix}/per_agent_misc_metrics"):
        agent_to_results: dict[str, list[dict]] = defaultdict(list)
        for nemo_gym_row, result in zip(nemo_gym_rows, results):
            agent_name = nemo_gym_row["agent_ref"]["name"]
            agent_to_results[agent_name].append(result["full_result"])

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

            # Log the full result
            to_log = [[json.dumps(r, separators=((",", ":")))] for r in agent_results]
            per_agent_metrics[f"{agent_name}/full_result"] = Table(
                data=to_log, columns=["Full result"]
            )

        rollout_metrics.update(per_agent_metrics)

    # Necessary for downstream nemo rl logging/printing.
    rollout_metrics["mean_gen_tokens_per_sample"] = rollout_metrics[
        "gen_tokens_per_sample/mean"
    ]
    timer.stop(f"{timer_prefix}/total")
    rollout_metrics.update(timer.get_timing_metrics("sum"))

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
        }
    )

    return AsyncNemoGymRolloutResult(
        input_ids=input_ids,
        final_batch=final_batch,
        rollout_metrics=rollout_metrics,
    )
