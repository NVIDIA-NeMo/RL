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
from typing import Any

import ray
import torch
from transformers import PreTrainedTokenizerBase

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
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)

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

    pad_token_id = policy_generation.cfg.get("pad_token_id", tokenizer.pad_token_id)
    generation_outputs = BatchedDataDict.from_batches(
        ordered_batched_data_dicts,
        pad_value_dict={"output_ids": pad_token_id, "logprobs": 0.0},
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

    for future, result in zip(futures, results):
        indices = future_to_indices[future]
        # Environment step returns: EnvironmentReturn
        env_observations, metadata, next_stop_strings, task_rewards, terminateds = (
            result
        )
        if next_stop_strings is None:
            next_stop_strings = [None] * len(task_rewards)

        # Store results with their original indices
        for i, idx in enumerate(indices):
            all_indices_order.append(idx)
            all_rewards.append(task_rewards[i])
            all_env_observations.append(env_observations[i])
            all_terminateds.append(terminateds[i])
            all_next_stop_strings.append(next_stop_strings[i])
            all_metadata.append(metadata[i])

    # Sort results by original index to maintain order
    sorted_indices = sorted(
        range(len(all_indices_order)), key=lambda k: all_indices_order[k]
    )
    rewards = torch.tensor([all_rewards[i] for i in sorted_indices])
    env_observations = [all_env_observations[i] for i in sorted_indices]
    terminateds = torch.tensor([all_terminateds[i] for i in sorted_indices])
    next_stop_strings = [all_next_stop_strings[i] for i in sorted_indices]
    metadata = [all_metadata[i] for i in sorted_indices]  # Sort metadata

    return EnvironmentReturn(
        observations=env_observations,
        metadata=metadata,
        next_stop_strings=next_stop_strings,
        rewards=rewards,
        terminateds=terminateds,
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

        generation_input_data = BatchedDataDict[GenerationDatumSpec](
            {
                "input_ids": active_input_ids,
                "input_lengths": active_input_lengths,
                "stop_strings": active_stop_strings,
            }
        )

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
        total_reward += env_output.rewards[0].item()
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

        return final_batch, rollout_metrics

    return asyncio.run(_async_rollout_implementation())
