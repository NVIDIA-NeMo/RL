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
import warnings
from collections import defaultdict
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import ray
import torch
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase
from wandb import Table

from nemo_rl.algorithms.utils import get_gdpo_reward_component_keys
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
from nemo_rl.environments.nemo_gym import DEFAULT_THINKING_TAGS
from nemo_rl.experience.interfaces import NEMO_GYM_TASK_INDEX_KEY
from nemo_rl.experience.metric_utils import calculate_single_metric, pct
from nemo_rl.models.generation.interfaces import (
    GenerationConfig,
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.utils.timer import Timer

TokenizerType = PreTrainedTokenizerBase


def _add_r3_fallback_metrics(
    gen_metrics: dict[str, float | int],
    generation_outputs: BatchedDataDict,
) -> None:
    missing = generation_outputs.get("r3_routed_experts_missing_routes")
    if missing is None:
        return

    missing_cpu = missing.detach().cpu()
    expected = generation_outputs.get("r3_routed_experts_expected_routes")
    actual = generation_outputs.get("r3_routed_experts_actual_routes")
    expected_cpu = expected.detach().cpu() if expected is not None else None
    actual_cpu = actual.detach().cpu() if actual is not None else None

    missing_routes = int(missing_cpu.sum().item())
    fallback_samples = int((missing_cpu > 0).sum().item())
    expected_routes = int(expected_cpu.sum().item()) if expected_cpu is not None else 0
    actual_routes = int(actual_cpu.sum().item()) if actual_cpu is not None else 0
    gen_metrics["r3/routed_experts_fallback_samples"] = fallback_samples
    gen_metrics["r3/routed_experts_fallback_token_routes"] = missing_routes
    gen_metrics["r3/routed_experts_expected_token_routes"] = expected_routes
    gen_metrics["r3/routed_experts_actual_token_routes"] = actual_routes
    gen_metrics["r3/routed_experts_fallback_token_route_fraction"] = (
        float(missing_routes / expected_routes) if expected_routes > 0 else 0.0
    )


def _extract_mask_sample_flags(results: list[dict[str, Any]]) -> torch.Tensor:
    """Return True for samples the environment asks GRPO to mask from loss."""
    return torch.tensor(
        [
            bool(
                (result["full_result"].get("instance_config") or {}).get(
                    "mask_sample", False
                )
            )
            for result in results
        ],
        dtype=torch.bool,
    )


def _attach_routed_experts_to_message_log_prefix(
    message_log: list[dict],
    routed_experts: torch.Tensor,
) -> int:
    """Attach routed-expert slices to existing messages and return prefix length."""
    cursor = 0
    for msg in message_log:
        token_ids = msg.get("token_ids")
        if not isinstance(token_ids, torch.Tensor):
            continue
        msg_len = int(token_ids.shape[0])
        msg["routed_experts"] = routed_experts[cursor : cursor + msg_len]
        cursor += msg_len
    return cursor


def _find_routed_experts_template(message_log: list[dict]) -> Optional[torch.Tensor]:
    for msg in message_log:
        routed_experts = msg.get("routed_experts")
        if isinstance(routed_experts, torch.Tensor):
            return routed_experts
    return None


def _dummy_routed_experts_for_tokens(
    token_ids: torch.Tensor,
    template: torch.Tensor,
) -> torch.Tensor:
    if template.dim() != 3:
        raise ValueError(
            "routed_experts messages must have shape [tokens, layers, topk], "
            f"got {tuple(template.shape)}"
        )
    topk = template.shape[2]
    default_route = torch.arange(topk, dtype=template.dtype, device=template.device)
    return (
        default_route.view(1, 1, topk)
        .expand(int(token_ids.shape[0]), template.shape[1], topk)
        .clone()
    )


class EffortLevelsConfig(BaseModel, extra="allow"):
    """Controls length-based reward shaping for low-effort prompts.

    When a prompt contains ``low_string``, the final reward is adjusted by a
    length-reward term that penalises overly long responses.  The reward formula
    is::

        length_reward = min(1, low_weight * (1 - response_len / low_ub))
        new_reward    = orig_reward
                      + orig_reward * max(length_reward, 0)
                      + low_penalty * min(length_reward, 0)

    Setting ``low_weight = 0`` or leaving ``low_string`` empty disables the
    shaping entirely.
    """

    low_weight: float = 0.0
    """Weight applied to the length-reward term.  Set to 0 to disable."""
    low_penalty: float = 1.0
    """Coefficient for the negative length-reward penalty."""
    low_ub: int = 64000
    """Response-length upper bound (in tokens) used to normalise the term."""
    low_string: str = ""
    """Substring that must appear in the user prompt to trigger shaping."""


@dataclass
class _EffortShapingMetrics:
    length_rewards_low: list[float]
    rewards_low: list[float]
    low_lengths: list[int]
    high_lengths: list[int]


def _apply_effort_shaping(
    results: list[dict],
    nemo_gym_rows: list[dict],
    effort_config: Optional[EffortLevelsConfig],
) -> _EffortShapingMetrics:
    """Apply length-based reward shaping for low-effort prompts.

    Modifies ``results[i]["full_result"]["reward"]`` in place for samples whose
    last user-turn prompt contains ``effort_config.low_string``.  Returns per-sample
    tracking lists used to populate rollout metrics.

    No-ops (returns empty lists) when ``effort_config`` is ``None``,
    ``low_weight`` is zero, or ``low_string`` is empty.
    """
    length_rewards_low: list[float] = []
    rewards_low: list[float] = []
    low_lengths: list[int] = []
    high_lengths: list[int] = []

    if (
        effort_config is None
        or effort_config.low_weight <= 0
        or not effort_config.low_string
    ):
        return _EffortShapingMetrics(
            length_rewards_low, rewards_low, low_lengths, high_lengths
        )

    lengths = [
        len(r["message_log"][-1]["token_ids"])
        if r["message_log"][-1]["role"] == "assistant"
        else 0
        for r in results
    ]
    orig_rewards = [r["full_result"]["reward"] for r in results]
    for i, result in enumerate(results):
        prompt = next(
            (
                msg["content"]
                for msg in reversed(
                    nemo_gym_rows[i]["responses_create_params"]["input"]
                )
                if msg.get("role") == "user" and "content" in msg
            ),
            "",
        )
        if effort_config.low_string in prompt:
            length_reward = min(
                1.0,
                effort_config.low_weight * (1.0 - lengths[i] / effort_config.low_ub),
            )
            new_reward = (
                orig_rewards[i]
                + orig_rewards[i] * max(length_reward, 0.0)
                + effort_config.low_penalty * min(length_reward, 0.0)
            )
            result["full_result"]["reward"] = new_reward
            length_rewards_low.append(length_reward)
            rewards_low.append(new_reward)
            low_lengths.append(lengths[i])
        else:
            high_lengths.append(lengths[i])

    return _EffortShapingMetrics(
        length_rewards_low, rewards_low, low_lengths, high_lengths
    )


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
        if "routed_experts" in generation_outputs:
            routed_experts = generation_outputs["routed_experts"][i]
            prefix_length = _attach_routed_experts_to_message_log_prefix(
                batch["message_log"][i], routed_experts
            )
            if prefix_length != int(input_length.item()):
                raise RuntimeError(
                    "message_log token length does not match generation input_length "
                    f"({prefix_length} != {int(input_length.item())})."
                )
            assistant_message["routed_experts"] = routed_experts[
                input_length:total_length
            ]

        batch["message_log"][i].append(assistant_message)

    # Generation metrics
    gen_metrics = {
        "mean_generation_length": generation_lengths.float().mean().item(),
        "total_generated_tokens": generation_lengths.sum().item(),
    }
    _add_r3_fallback_metrics(gen_metrics, generation_outputs)

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

    # Check if this is a supported inference engine with async generation enabled.
    # SGLang exposes ``sglang_cfg`` and gates on ``use_async_rollouts``; vLLM and
    # Megatron expose ``cfg`` and gate on their respective ``async_engine`` flag.
    vllm_cfg = getattr(policy_generation, "cfg", None)
    sglang_cfg = getattr(policy_generation, "sglang_cfg", None)
    generation_config = vllm_cfg or sglang_cfg or {}
    backend = generation_config.get("backend", "")

    if backend == "sglang":
        use_async_generation = bool(generation_config.get("use_async_rollouts", False))
    elif backend == "vllm":
        use_async_generation = bool(
            generation_config.get("vllm_cfg", {}).get("async_engine", False)
        )
    elif backend == "trtllm":
        assert generation_config.get("trtllm_cfg", {}).get("async_engine", False), (
            "TRT-LLM backend requires trtllm_cfg.async_engine=true; the "
            "synchronous engine path (async_engine=false) is no longer supported."
        )
        use_async_generation = True
    elif backend == "megatron":
        use_async_generation = bool(
            generation_config.get("mcore_generation_config", {}).get(
                "async_engine", False
            )
        )
    else:
        use_async_generation = False

    assert use_async_generation and hasattr(policy_generation, "generate_async"), (
        "Async generation is not enabled. For SGLang, set "
        "policy.generation.use_async_rollouts=True. For vLLM, set "
        "policy.generation.vllm_cfg.async_engine=True. For Megatron, set "
        "policy.generation.mcore_generation_config.async_engine=True. The "
        "generation backend must also implement generate_async."
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
        if "routed_experts" in generation_outputs:
            routed_experts = generation_outputs["routed_experts"][i]
            prefix_length = _attach_routed_experts_to_message_log_prefix(
                batch["message_log"][i], routed_experts
            )
            if prefix_length != int(input_length.item()):
                raise RuntimeError(
                    "message_log token length does not match generation input_length "
                    f"({prefix_length} != {int(input_length.item())})."
                )
            assistant_message["routed_experts"] = routed_experts[
                input_length:total_length
            ]

        batch["message_log"][i].append(assistant_message)

    # Generation metrics
    gen_metrics = {
        "mean_generation_length": generation_lengths.float().mean().item(),
        "total_generated_tokens": generation_lengths.sum().item(),
    }
    _add_r3_fallback_metrics(gen_metrics, generation_outputs)
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
    all_rewards: list = []  # list of per-sample scalars/tensors (single-reward envs)
    all_dict_rewards: dict[str, list] | None = None  # for dict-based multi-reward envs
    is_dict_rewards = False
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

        is_dict_rewards = isinstance(task_rewards, dict)

        if next_stop_strings is None:
            next_stop_strings = [None] * len(terminateds)
        if answers is None:
            answers = [None] * len(terminateds)

        # Initialize dict-reward accumulator on first encounter (outside inner loop).
        if is_dict_rewards and all_dict_rewards is None:
            all_dict_rewards = {name: [] for name in task_rewards}

        # Store results with their original indices
        for i, idx in enumerate(indices):
            all_indices_order.append(idx)
            if is_dict_rewards:
                for name in task_rewards:
                    all_dict_rewards[name].append(task_rewards[name][i])  # type: ignore
            else:
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

    # Build rewards: dict-based for multi-reward envs, tensor for single-reward.
    if all_dict_rewards is not None:
        assert len(all_rewards) == 0, (
            "Mixing dict-based and scalar rewards across environments is not supported. "
            "All environments must return the same reward format (all dict or all scalar)."
        )
        rewards: torch.Tensor | dict[str, torch.Tensor] = {
            name: torch.stack([vals[i] for i in sorted_indices])
            for name, vals in all_dict_rewards.items()
        }
    elif len(all_rewards) > 0 and isinstance(all_rewards[0], torch.Tensor):
        rewards = torch.stack([all_rewards[i] for i in sorted_indices])
    else:
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

    # Multi-reward accumulator: dict of {name: Tensor[B]} for multi-reward envs (e.g. GDPO), None for single-reward.
    multi_rewards: dict[str, torch.Tensor] | None = None

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
        if "vllm_audios" in active_batch:
            generation_input_data["vllm_audios"] = active_batch["vllm_audios"]

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

        # Accumulate rewards: env returns dict[str, Tensor] for multi-reward, Tensor for single-reward.
        if isinstance(env_output.rewards, dict):
            # Initialize accumulators on first encounter
            if multi_rewards is None:
                multi_rewards = {
                    name: torch.zeros(batch_size, dtype=torch.float32)
                    for name in env_output.rewards
                }
            # this assert is to infer the type of multi_rewards for pyrefly
            assert multi_rewards is not None
            reward_dict: dict[str, torch.Tensor] = multi_rewards
            for name, r in env_output.rewards.items():
                reward_dict[name][active_indices] += r
            total_rewards[active_indices] += sum(env_output.rewards.values())
        else:
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

            tokenized_env_obs_message: dict[str, Any] = {
                "role": env_output.observations[i]["role"],
                "content": env_obs_content,
                "token_ids": tokenized_obs,
            }
            routed_template = _find_routed_experts_template(
                current_batch["message_log"][global_idx]
            )
            if routed_template is not None:
                tokenized_env_obs_message["routed_experts"] = (
                    _dummy_routed_experts_for_tokens(tokenized_obs, routed_template)
                )
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
    # Expose per-component rewards for multi-reward envs (e.g. GDPO advantage calculation).
    if multi_rewards is not None:
        for name, reward_tensor in multi_rewards.items():
            current_batch[name] = reward_tensor

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
    reward_acc_dict: dict[str, float] = {}  # per-component reward accumulators (named)
    multi_reward_seen = False
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

        # Get environment feedback.
        # calculate_rewards uses blocking ray.get internally. Running it
        # directly on the asyncio event loop (which this coroutine runs on)
        # blocks every other in-flight rollout coroutine for the entire env
        # step. In this case, need to wrap with asyncio.to_thread to make
        # this function yieldable.
        env_output = await asyncio.to_thread(
            calculate_rewards, sample_batch, task_to_env
        )
        # Update total reward and optional per-component reward signals.
        if isinstance(env_output.rewards, dict):
            multi_reward_seen = True
            for name, r in env_output.rewards.items():
                reward_acc_dict[name] = reward_acc_dict.get(name, 0.0) + float(
                    r[0].item()
                )
            total_reward += sum(float(r[0].item()) for r in env_output.rewards.values())
        else:
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

        env_message: dict[str, Any] = {
            "role": env_output.observations[0]["role"],
            "content": env_obs_content,
            "token_ids": tokenized_obs,
        }
        routed_template = _find_routed_experts_template(current_message_log)
        if routed_template is not None:
            env_message["routed_experts"] = _dummy_routed_experts_for_tokens(
                tokenized_obs, routed_template
            )
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
    if multi_reward_seen:
        for name, acc in reward_acc_dict.items():
            final_sample_state[name] = torch.tensor(acc)

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


@dataclass
class RolloutGroupResult:
    """One prompt group's rollout batch and metrics."""

    group_index: int
    final_batch: BatchedDataDict[DatumSpec]
    rollout_metrics: dict[str, Any]
    task_index: Optional[int] = None


def _aggregate_multi_turn_rollout_metrics(
    all_sample_metrics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate native rollout metrics over an arbitrary set of samples."""
    if not all_sample_metrics:
        raise ValueError("Cannot aggregate metrics for an empty rollout batch")

    batch_size = len(all_sample_metrics)
    turn_counts = [m["turn_count"] for m in all_sample_metrics]
    max_gen_tokens_per_turn_values = [
        m["max_gen_tokens_per_turn"] for m in all_sample_metrics
    ]

    rollout_metrics = {
        # Overall metrics
        "total_turns": sum(turn_counts),
        "avg_turns_per_sample": sum(turn_counts) / batch_size,
        "max_turns_per_sample": max(turn_counts),
        "turns_per_sample/p95": pct(turn_counts, 95),
        "turns_per_sample/p99": pct(turn_counts, 99),
        "natural_termination_rate": sum(m["terminated"] for m in all_sample_metrics)
        / batch_size,
        "truncation_rate": sum(m["truncated"] for m in all_sample_metrics) / batch_size,
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
        "mean_env_tokens_per_sample": sum(m["env_tokens"] for m in all_sample_metrics)
        / batch_size,
        # Diagnostics for long single generations.
        "max_gen_tokens_per_turn/max": max(max_gen_tokens_per_turn_values),
        "max_gen_tokens_per_turn/mean": sum(max_gen_tokens_per_turn_values)
        / batch_size,
        "max_gen_tokens_per_turn/p95": pct(max_gen_tokens_per_turn_values, 95),
        # Reward metrics
        "mean_total_reward": sum(m["total_reward"] for m in all_sample_metrics)
        / batch_size,
        "max_total_reward": max(m["total_reward"] for m in all_sample_metrics),
        "min_total_reward": min(m["total_reward"] for m in all_sample_metrics),
    }

    if "per_worker_token_counts" in all_sample_metrics[0]:
        per_worker_token_counts = {}
        for sample_metrics in all_sample_metrics:
            for worker, token_count in sample_metrics[
                "per_worker_token_counts"
            ].items():
                per_worker_token_counts[worker] = (
                    per_worker_token_counts.get(worker, 0) + token_count
                )
        rollout_metrics["per_worker_token_counts"] = per_worker_token_counts

    rollout_metrics["histogram/gen_tokens_length"] = [
        token_count
        for sample_metrics in all_sample_metrics
        for token_count in sample_metrics["turn_gen_tokens"]
    ]
    rollout_metrics["histogram/input_tokens_length"] = [
        token_count
        for sample_metrics in all_sample_metrics
        for token_count in sample_metrics["turn_input_tokens"]
    ]
    rollout_metrics["histogram/total_tokens_length"] = [
        token_count
        for sample_metrics in all_sample_metrics
        for token_count in sample_metrics["turn_total_tokens"]
    ]
    return rollout_metrics


async def _run_multi_turn_rollout_async(
    policy_generation: GenerationInterface,
    input_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    max_seq_len: int,
    max_rollout_turns: int = 999999,
    greedy: bool = False,
) -> tuple[BatchedDataDict[DatumSpec], list[dict[str, Any]]]:
    """Run one native rollout batch and retain metrics at sample granularity."""
    batch_size = len(input_batch["message_log"])

    sample_initial_states = []
    for i in range(batch_size):
        sample_initial_states.append(
            {
                "message_log": input_batch["message_log"][i],
                "extra_env_info": input_batch["extra_env_info"][i],
                "task_name": input_batch["task_name"][i],
                "stop_strings": input_batch.get("stop_strings", [None] * batch_size)[i],
                "idx": input_batch.get("idx", list(range(batch_size)))[i],
            }
        )

    async def run_single_sample_with_error_handling(i, sample_state):
        try:
            return await run_sample_multi_turn_rollout(
                sample_idx=i,
                initial_sample_state=sample_state,
                policy_generation=policy_generation,
                tokenizer=tokenizer,
                task_to_env=task_to_env,
                max_seq_len=max_seq_len,
                max_rollout_turns=max_rollout_turns,
                greedy=greedy,
            )
        except Exception as error:
            raise RuntimeError(f"Error in sample {i} rollout: {error}") from error

    sample_results = await asyncio.gather(
        *(
            run_single_sample_with_error_handling(i, sample_state)
            for i, sample_state in enumerate(sample_initial_states)
        ),
        return_exceptions=False,
    )
    final_sample_states = [result[0] for result in sample_results]
    all_sample_metrics = [result[1] for result in sample_results]

    # Reconstruct the batch in input order. asyncio.gather preserves the order
    # of the sample coroutines even when they finish out of order.
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
            "idx": [state.get("idx", i) for i, state in enumerate(final_sample_states)],
            "truncated": torch.tensor(
                [metrics["truncated"] for metrics in all_sample_metrics],
                dtype=torch.bool,
            ),
        }
    )

    # Preserve named per-component rewards for GDPO. Mixed environment batches
    # use zero for samples that do not expose a given reward component.
    reward_component_keys = sorted(
        set(
            key
            for state in final_sample_states
            for key in get_gdpo_reward_component_keys(state)
        )
    )
    for key in reward_component_keys:
        final_batch[key] = torch.stack(
            [
                state[key] if key in state else torch.tensor(0.0, dtype=torch.float32)
                for state in final_sample_states
            ]
        )

    for key in input_batch.keys():
        if key not in final_batch:
            final_batch[key] = input_batch[key]

    return final_batch, all_sample_metrics


def run_async_multi_turn_rollout(
    policy_generation: GenerationInterface,
    input_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    max_seq_len: int,
    max_rollout_turns: int = 999999,
    greedy: bool = False,
) -> tuple[BatchedDataDict[DatumSpec], dict[str, Any]]:
    """Run a complete native rollout batch from a synchronous call site.

    Each sample proceeds through its interaction independently. Generation is
    asynchronous internally, while this compatibility API returns only after
    the full batch and its aggregate metrics are ready.

    Args:
        policy_generation: Generation interface used to produce policy responses.
        input_batch: Batch containing the initial message logs and environment data.
        tokenizer: Tokenizer used to encode and decode rollout messages.
        task_to_env: Mapping from task names to their environment implementations.
        max_seq_len: Maximum total token length for each rollout sample.
        max_rollout_turns: Maximum number of agent-environment interaction turns.
        greedy: Whether policy generation should use greedy decoding.

    Returns:
        A tuple containing the completed rollout batch and metrics aggregated over
        every sample in that batch.

    Raises:
        RuntimeError: If an individual sample rollout fails.
    """
    final_batch, sample_metrics = asyncio.run(
        _run_multi_turn_rollout_async(
            policy_generation=policy_generation,
            input_batch=input_batch,
            tokenizer=tokenizer,
            task_to_env=task_to_env,
            max_seq_len=max_seq_len,
            max_rollout_turns=max_rollout_turns,
            greedy=greedy,
        )
    )
    return final_batch, _aggregate_multi_turn_rollout_metrics(sample_metrics)


async def run_async_multi_turn_rollout_groups(
    policy_generation: GenerationInterface,
    input_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    max_seq_len: int,
    num_generations: int,
    max_rollout_turns: int = 999999,
    greedy: bool = False,
) -> AsyncGenerator[RolloutGroupResult, None]:
    """Run one native batch, then yield prompt groups with group-local metrics.

    This intentionally retains the native path's full-batch completion barrier.
    The group iterator gives the collector a common interface with NeMo-Gym
    without changing native rollout scheduling semantics.

    Args:
        policy_generation: Generation interface used to produce policy responses.
        input_batch: Batch containing prompts repeated contiguously by group.
        tokenizer: Tokenizer used to encode and decode rollout messages.
        task_to_env: Mapping from task names to their environment implementations.
        max_seq_len: Maximum total token length for each rollout sample.
        num_generations: Number of contiguous rollout samples in each prompt group.
        max_rollout_turns: Maximum number of agent-environment interaction turns.
        greedy: Whether policy generation should use greedy decoding.

    Yields:
        Complete prompt groups in input order. Each ``RolloutGroupResult`` contains
        exactly ``num_generations`` samples and metrics aggregated only over those
        samples.

    Raises:
        ValueError: If ``num_generations`` is not positive or the batch size is not
            divisible by ``num_generations``.
        RuntimeError: If an individual sample rollout fails.
    """
    if num_generations <= 0:
        raise ValueError("num_generations must be greater than zero")
    if input_batch.size % num_generations != 0:
        raise ValueError(
            "Native rollout batch size must be divisible by num_generations"
        )

    final_batch, sample_metrics = await _run_multi_turn_rollout_async(
        policy_generation=policy_generation,
        input_batch=input_batch,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        max_seq_len=max_seq_len,
        max_rollout_turns=max_rollout_turns,
        greedy=greedy,
    )
    for group_index, start in enumerate(range(0, final_batch.size, num_generations)):
        end = start + num_generations
        yield RolloutGroupResult(
            group_index=group_index,
            final_batch=final_batch.slice(start, end),
            rollout_metrics=_aggregate_multi_turn_rollout_metrics(
                sample_metrics[start:end]
            ),
        )


def _tensorize_by_key(message_logs: list, key: str):
    if not message_logs or key not in message_logs[0]:
        return

    for m in message_logs:
        m[key] = torch.tensor(m[key])


@dataclass
class NemoGymRolloutResult:
    """Processed NeMo-Gym rollouts for one prompt group or synchronous batch."""

    input_ids: torch.Tensor
    final_batch: BatchedDataDict[DatumSpec]
    rollout_metrics: dict[str, Any]
    # Stable prompt identity used by the async collector; absent for sync callers.
    task_index: Optional[int]


@dataclass(frozen=True)
class _CompletedNemoGymGroup:
    """One complete Gym prompt group restored to input-row order."""

    group_index: int
    rows: list[dict]
    results: list[dict]


class _NemoGymStreamAccumulator:
    """Validate streamed Gym rows and assemble complete prompt groups.

    NeMo Gym returns rows in completion order. This accumulator owns all ordering
    and completeness rules so the rollout loop only needs to postprocess completed
    groups.
    """

    def __init__(
        self,
        rows: list[dict],
        num_generations: int,
        allow_mixed_agents: bool,
    ) -> None:
        self._rows = rows
        self._num_generations = num_generations
        self._allow_mixed_agents = allow_mixed_agents
        self._received_row_indices: set[int] = set()
        self._pending_results: dict[int, dict[int, dict]] = defaultdict(dict)

    @property
    def is_complete(self) -> bool:
        return len(self._received_row_indices) == len(self._rows)

    def add(self, row_index: int, result: dict) -> _CompletedNemoGymGroup | None:
        """Add one streamed row and return its group when that group is complete."""
        if not isinstance(row_index, int):
            raise TypeError(
                f"NeMo-Gym row index must be an int, got {type(row_index).__name__}"
            )
        if row_index < 0 or row_index >= len(self._rows):
            raise ValueError(
                f"NeMo-Gym returned row index {row_index} outside the expected "
                f"range [0, {len(self._rows)})"
            )
        if row_index in self._received_row_indices:
            raise ValueError(f"NeMo-Gym returned duplicate row index {row_index}")

        self._received_row_indices.add(row_index)
        group_index = row_index // self._num_generations
        group_results = self._pending_results[group_index]
        group_results[row_index] = result
        if len(group_results) < self._num_generations:
            return None

        start = group_index * self._num_generations
        end = start + self._num_generations
        expected_row_indices = range(start, end)
        missing_row_indices = [
            index for index in expected_row_indices if index not in group_results
        ]
        if missing_row_indices:
            raise RuntimeError(
                f"NeMo-Gym prompt group {group_index} completed with unexpected row "
                f"indices; missing {missing_row_indices}"
            )

        rows = self._rows[start:end]
        if not self._allow_mixed_agents:
            agent_names = [row["agent_ref"]["name"] for row in rows]
            if len(set(agent_names)) != 1:
                raise ValueError(
                    f"Expected one NeMo-Gym agent per prompt group, got {agent_names}"
                )

        ordered_results = [group_results[index] for index in expected_row_indices]
        del self._pending_results[group_index]
        return _CompletedNemoGymGroup(
            group_index=group_index,
            rows=rows,
            results=ordered_results,
        )

    def finish(self) -> None:
        """Raise when the stream ended before every expected row arrived."""
        if self.is_complete:
            return
        missing_row_indices = sorted(
            set(range(len(self._rows))) - self._received_row_indices
        )
        raise RuntimeError(
            "NeMo-Gym rollout stream ended before all rows arrived; missing row "
            f"indices {missing_row_indices}"
        )


def get_nemo_gym_thinking_tags(env_config: dict[str, Any]) -> list[str]:
    """Return thinking tags used by the Gym-side detector."""
    nemo_gym_config = env_config.get("nemo_gym")
    if isinstance(nemo_gym_config, dict) and nemo_gym_config.get("thinking_tags"):
        return list(nemo_gym_config["thinking_tags"])
    return list(DEFAULT_THINKING_TAGS)


def _get_reward_penalty_config_value(
    reward_penalty_config: dict[str, Any] | BaseModel | None,
    key: str,
) -> Any:
    if reward_penalty_config is None:
        return None
    if isinstance(reward_penalty_config, dict):
        return reward_penalty_config.get(key)

    return getattr(reward_penalty_config, key, None)


def _get_reward_penalty_token_id(
    reward_penalty_config: dict[str, Any] | BaseModel,
    key: str,
) -> int | None:
    token_ids = _get_reward_penalty_config_value(reward_penalty_config, "token_ids")
    value = _get_reward_penalty_config_value(token_ids, key)
    if value is None:
        return None
    return int(value)


def _get_required_reward_penalty_token_id(
    reward_penalty_config: dict[str, Any] | BaseModel,
    key: str,
) -> int:
    value = _get_reward_penalty_token_id(reward_penalty_config, key)
    if value is None:
        raise ValueError(f"reward_penalties.token_ids.{key} must be set")
    return value


def _get_reward_penalty_token_ids(
    reward_penalty_config: dict[str, Any] | BaseModel,
    key: str,
) -> list[int] | None:
    token_ids = _get_reward_penalty_config_value(reward_penalty_config, "token_ids")
    value = _get_reward_penalty_config_value(token_ids, key)
    if value is None:
        return None
    if isinstance(value, int):
        return [int(value)]
    return [int(token_id) for token_id in value]


def _get_required_reward_penalty_token_ids(
    reward_penalty_config: dict[str, Any] | BaseModel,
    key: str,
) -> list[int]:
    values = _get_reward_penalty_token_ids(reward_penalty_config, key)
    if not values:
        raise ValueError(f"reward_penalties.token_ids.{key} must be set")
    return values


def _infer_single_token_id(tokenizer: Any, text: str) -> int | None:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        return None
    return int(token_ids[0])


def resolve_reward_penalty_config(
    reward_penalty_config: dict[str, Any] | BaseModel | None,
    tokenizer: Any,
    thinking_tags: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any] | None:
    """Resolve tokenizer-derived reward penalty fields.

    User config must explicitly provide unwanted token IDs when
    penalize_unwanted_tokens is enabled.
    Think-tag IDs are inferred only when each configured tag is exactly one
    token.
    """
    if reward_penalty_config is None:
        return None

    resolved: dict[str, Any] = {}
    for flag in (
        "penalize_duplicated_reasoning",
        "penalize_empty_final_answer",
        "penalize_unwanted_tokens",
        "penalize_malformed_think_tag",
    ):
        value = _get_reward_penalty_config_value(reward_penalty_config, flag)
        if value is not None:
            resolved[flag] = value

    token_ids: dict[str, Any] = {}
    unwanted_token_ids = _get_reward_penalty_token_ids(
        reward_penalty_config, "unwanted"
    )
    if unwanted_token_ids is not None:
        token_ids["unwanted"] = unwanted_token_ids

    for key in ("think_open", "think_close"):
        value = _get_reward_penalty_token_id(reward_penalty_config, key)
        if value is not None:
            token_ids[key] = value

    if resolved.get("penalize_unwanted_tokens") and not token_ids.get("unwanted"):
        raise ValueError(
            "reward_penalties.token_ids.unwanted must be set when "
            "reward_penalties.penalize_unwanted_tokens is true"
        )

    if resolved.get("penalize_malformed_think_tag"):
        configured_thinking_tags = _get_reward_penalty_config_value(
            reward_penalty_config, "thinking_tags"
        )
        tags = tuple(thinking_tags or configured_thinking_tags or DEFAULT_THINKING_TAGS)
        resolved["thinking_tags"] = tags
        if len(tags) >= 2:
            explicit_open = "think_open" in token_ids
            explicit_close = "think_close" in token_ids
            inferred_open = None
            inferred_close = None
            if not explicit_open:
                inferred_open = _infer_single_token_id(tokenizer, tags[0])
            if not explicit_close:
                inferred_close = _infer_single_token_id(tokenizer, tags[1])

            if (
                explicit_open
                or explicit_close
                or (inferred_open is not None and inferred_close is not None)
            ):
                if inferred_open is not None:
                    token_ids["think_open"] = inferred_open
                if inferred_close is not None:
                    token_ids["think_close"] = inferred_close

    if token_ids:
        resolved["token_ids"] = token_ids

    return resolved


def apply_reward_penalties(
    results: list[dict], reward_penalty_config: dict[str, Any] | BaseModel | None
) -> dict[str, int]:
    """Apply reward penalties to results, setting reward to 0.0 when triggered.

    All penalties are gated by reward_penalty_config flags. Returns a dict of penalty
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

      3. penalize_unwanted_tokens (token-based)
         Currently checks that none of the explicitly configured unwanted
         token IDs appear anywhere in an assistant generation, including as the terminal
         token. A turn may contain multiple unwanted tokens, so the whole assistant
         token sequence is checked rather than excluding the trailing position.
         Data: message_log[i]["token_ids"] where role == "assistant".

      4. penalize_malformed_think_tag (message flag + token/string fallback)
         Three complementary checks to catch malformed think tags:
         a) Existing Gym flag: honors assistant message has_malformed_thinking.
         b) Token ID check: when think tag IDs are resolved from config override
            or single-token tokenizer encodings, infers thinking mode from
            prompt token counts. If prompt has open==close:
            enable_thinking=False, expect 0 open
            and 0 close in generation. If prompt has open==close+1:
            enable_thinking=True, expect 0 open and 1 close in generation.
            Any other prompt pattern or mismatched generation counts is a violation.
            This fallback is skipped when the tags do not resolve to one token
            each.
         c) String check: the model can spell out thinking tags with piecemeal
            regular tokens (e.g. "<", "/", "thi", "nk", ">") that bypass special
            token IDs. Checks generation_str (decoded generation text) per output
            item: open-tag count must be 0 (always in prompt, never generated),
            close-tag count must be 0 or 1.
         Data: message_log pairs for token IDs, full_result output items for strings.
    """
    counts = {
        "duplicated_reasoning": 0,
        "empty_final_answer": 0,
        "unwanted_token": 0,
        "malformed_think_tag": 0,
    }
    if not reward_penalty_config or not results:
        return counts

    # Guard: penalties rely on Gym-path message_log (strictly alternating user/assistant roles).
    # Non-Gym paths may have "environment", "tool", or "system" roles which these checks don't handle.
    any_penalty_enabled = any(
        _get_reward_penalty_config_value(reward_penalty_config, flag)
        for flag in (
            "penalize_duplicated_reasoning",
            "penalize_empty_final_answer",
            "penalize_unwanted_tokens",
            "penalize_malformed_think_tag",
        )
    )
    if any_penalty_enabled:
        for result in results:
            roles = {msg.get("role") for msg in result["message_log"]}
            assert roles <= {"user", "assistant"}, (
                f"apply_reward_penalties requires Gym-path message_log with only 'user' and 'assistant' roles, "
                f"but found roles: {roles}. These penalties are not supported for non-Gym rollout paths."
            )

    # --- Penalty 1: Duplicated reasoning / final answer ---
    if _get_reward_penalty_config_value(
        reward_penalty_config, "penalize_duplicated_reasoning"
    ):
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
    if _get_reward_penalty_config_value(
        reward_penalty_config, "penalize_empty_final_answer"
    ):
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

    # --- Penalty 3: unwanted token in generation ---
    if _get_reward_penalty_config_value(
        reward_penalty_config, "penalize_unwanted_tokens"
    ):
        unwanted_token_ids = _get_required_reward_penalty_token_ids(
            reward_penalty_config, "unwanted"
        )
        for result in results:
            has_unwanted_token = False
            for msg in result["message_log"]:
                if msg["role"] != "assistant":
                    continue
                # Penalize any configured unwanted token in the assistant generation,
                # including the terminal position.
                if any(token_id in msg["token_ids"] for token_id in unwanted_token_ids):
                    has_unwanted_token = True
                    break
            if has_unwanted_token:
                result["full_result"]["reward"] = 0.0

                counts["unwanted_token"] += 1

    # --- Penalty 4: Malformed think tags (existing flag + optional token ID + string) ---
    if _get_reward_penalty_config_value(
        reward_penalty_config, "penalize_malformed_think_tag"
    ):
        think_open_token_id = _get_reward_penalty_token_id(
            reward_penalty_config, "think_open"
        )
        think_close_token_id = _get_reward_penalty_token_id(
            reward_penalty_config, "think_close"
        )
        if (think_open_token_id is None) != (think_close_token_id is None):
            raise ValueError(
                "reward_penalties.token_ids.think_open and "
                "reward_penalties.token_ids.think_close must both be set"
            )
        for result in results:
            has_violation = any(
                msg.get("role") == "assistant"
                and msg.get("has_malformed_thinking", False)
                for msg in result["message_log"]
            )

            # 4a) Token ID check per (user, assistant) turn pair.
            # Infer thinking mode from prompt token counts:
            #   enable_thinking=True:  prompt has open=close+1 (trailing <think>), expect asst: 0 open, 1 close
            #   enable_thinking=False: prompt has open=close (balanced), expect asst: 0 open, 0 close
            msgs = result["message_log"]
            if (
                not has_violation
                and think_open_token_id is not None
                and think_close_token_id is not None
            ):
                for i in range(len(msgs) - 1):
                    if msgs[i]["role"] != "user" or msgs[i + 1]["role"] != "assistant":
                        continue
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
                        # Unexpected prompt pattern - flag as violation
                        has_violation = True
                        break
                    if asst_open != expected_open or asst_close != expected_close:
                        has_violation = True
                        break

            # 4b) String check on generation_str per output item.
            if not has_violation:
                thinking_tags = (
                    _get_reward_penalty_config_value(
                        reward_penalty_config, "thinking_tags"
                    )
                    or DEFAULT_THINKING_TAGS
                )
                if len(thinking_tags) < 2:
                    raise ValueError(
                        "reward_penalties.thinking_tags must contain open and close tags"
                    )
                think_open_text, think_close_text = thinking_tags[:2]
                output_items = (
                    result["full_result"].get("response", {}).get("output", [])
                )
                for item in output_items:
                    gen_str = item.get("generation_str", "")
                    if not gen_str:
                        continue
                    if (
                        gen_str.count(think_open_text) > 0
                        or gen_str.count(think_close_text) > 1
                    ):
                        has_violation = True
                        break
            if has_violation:
                result["full_result"]["reward"] = 0.0

                counts["malformed_think_tag"] += 1

    return counts


def _prepare_nemo_gym_rows(
    rows: list[dict], generation_config: GenerationConfig
) -> None:
    """Apply NeMo-RL sampling parameters and stable row indices in place."""
    for row_index, row in enumerate(rows):
        responses_create_params = row.get("responses_create_params")
        if not isinstance(responses_create_params, dict):
            raise TypeError(
                "Each NeMo-Gym row must contain a responses_create_params dict"
            )

        responses_create_params["temperature"] = generation_config["temperature"]
        responses_create_params["top_p"] = generation_config["top_p"]
        configured_max_tokens = generation_config["max_new_tokens"]
        row_max_tokens = responses_create_params.get("max_output_tokens")
        responses_create_params["max_output_tokens"] = (
            min(row_max_tokens, configured_max_tokens)
            if row_max_tokens is not None
            else configured_max_tokens
        )
        row["_rowidx"] = row_index


def _tensorize_nemo_gym_result(result: dict) -> None:
    """Convert token fields returned by the Gym actor back to tensors."""
    _tensorize_by_key(result["input_message_log"], "token_ids")
    _tensorize_by_key(result["message_log"], "token_ids")
    _tensorize_by_key(
        [
            message
            for message in result["message_log"]
            if message["role"] == "assistant"
        ],
        "generation_logprobs",
    )


async def run_async_nemo_gym_rollout(
    policy_generation: GenerationInterface,
    input_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    generation_config: GenerationConfig,
    num_generations: int,
    log_full_result_tables: bool,
    max_seq_len: Optional[int] = None,
    max_rollout_turns: Optional[int] = None,
    greedy: bool = False,
    effort_config: Optional[EffortLevelsConfig] = None,
    reward_penalty_config: dict[str, Any] | BaseModel | None = None,
    thinking_tags: list[str] | tuple[str, ...] | None = None,
    returns_entire_batch: bool = False,
) -> AsyncGenerator[NemoGymRolloutResult, None]:
    """Stream complete NeMo-Gym prompt groups in group-completion order.

    The actor streams individual rows in arbitrary completion order. Rows are
    validated and restored to input order within each ``num_generations`` group
    before the group is postprocessed and yielded. Synchronous call sites should
    use :func:`run_nemo_gym_rollout_sync`.

    Args:
        policy_generation: Generation interface whose configuration supplies the
            model's maximum sequence length.
        input_batch: Batch whose ``extra_env_info`` field contains NeMo-Gym rows.
        tokenizer: Tokenizer used by the NeMo-Gym actor and local postprocessing.
        task_to_env: Environment mapping containing the ``"nemo_gym"`` actor.
        generation_config: Sampling parameters forwarded to every NeMo-Gym row.
        num_generations: Number of contiguous rows belonging to each prompt group.
        log_full_result_tables: Whether to include complete per-agent result
            payloads as W&B Tables in the rollout metrics.
        max_seq_len: Policy sequence-length limit used for compatibility validation.
            NeMo-Gym still relies on the generation engine's configured limit.
        max_rollout_turns: Must be ``None`` because NeMo-Gym owns turn limits.
        greedy: Must be ``False`` because this path does not support greedy mode.
        effort_config: Optional configuration for effort-based reward shaping.
        reward_penalty_config: Optional reward-penalty configuration.
        thinking_tags: Optional opening and closing tags used by thinking penalties.
        returns_entire_batch: Whether to treat the input as one potentially
            heterogeneous group. This requires ``num_generations`` to equal the
            batch size and is used by synchronous callers.

    Yields:
        ``NemoGymRolloutResult`` objects in prompt-group completion order. Rows
        inside each result are restored to input order. The final result also
        carries actor-wide and rollout-wide timing metrics.

    Raises:
        AssertionError: If an unsupported generation option is requested.
        TypeError: If a row lacks a valid ``responses_create_params`` dictionary or
            the actor returns a non-integer row index.
        ValueError: If ``num_generations`` is not positive, the batch is empty or
            not divisible by ``num_generations``, ``returns_entire_batch`` has an
            incompatible size, a streamed row index is out of range or duplicated,
            a prompt group mixes agents, or its task indices disagree.
        RuntimeError: If the actor fails, returns NaN generation logprobs, ends the
            stream before all expected rows arrive, or produces no final group.
    """
    # We accept max_seq_len for API parity with the other rollout paths, but NeMo-Gym
    # still relies on the underlying model server's configured context/window limits.
    # We leverage the same `extra_env_info` key as `run_async_multi_turn_rollout`.
    nemo_gym_rows = input_batch["extra_env_info"]

    # Handle generation parameters up front so we don't hide anything inside here to avoid being unintuitive to the user.
    # NeMo-Gym policy is "What you see is what you get".
    assert not greedy, "`greedy` is not supported in NeMo-Gym path!"
    assert max_rollout_turns is None, (
        "`max_rollout_turns` is not supported in NeMo-Gym path!"
    )
    # Dispatch on the backend, not key-presence: an SGLang config can inherit a
    # vLLM exemplar's ``vllm_cfg`` via deep-merge, so a bare ``"vllm_cfg" in cfg``
    # check would shadow the SGLang branch. ``context_length`` is nullable, so fall
    # back to the policy sequence cap when it is unset.
    if policy_generation.cfg.get("backend") == "sglang":
        engine_max_model_len = (
            policy_generation.cfg["sglang_cfg"].get("context_length")
            or policy_generation.cfg["max_total_sequence_length"]
        )
    elif "vllm_cfg" in policy_generation.cfg:
        engine_max_model_len = policy_generation.cfg["vllm_cfg"]["max_model_len"]
    elif "mcore_generation_config" in policy_generation.cfg:
        engine_max_model_len = policy_generation.cfg["mcore_generation_config"][
            "max_model_len"
        ]
    else:
        engine_max_model_len = policy_generation.cfg["max_total_sequence_length"]
    if max_seq_len is not None and max_seq_len > engine_max_model_len:
        warnings.warn(
            f"policy max_total_sequence_length ({max_seq_len}) is greater than the "
            f"generation engine's max_model_len ({engine_max_model_len}). The engine "
            "will truncate sequences to its own limit, so the policy cap will not be "
            "honored. Lower max_total_sequence_length or raise the engine's max_model_len."
        )
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
    if num_generations <= 0:
        raise ValueError("num_generations must be greater than zero")
    if not nemo_gym_rows:
        raise ValueError("NeMo-Gym rollout batch must not be empty")
    if len(nemo_gym_rows) % num_generations != 0:
        raise ValueError(
            "NeMo-Gym rollout batch size must be divisible by num_generations"
        )
    if returns_entire_batch and len(nemo_gym_rows) != num_generations:
        raise ValueError(
            "returns_entire_batch requires num_generations to equal the batch size"
        )

    timer = Timer()
    timer_prefix = "timing/rollout"
    total_timer_label = f"{timer_prefix}/total"
    run_rollouts_timer_label = f"{timer_prefix}/run_rollouts"

    with timer.time(total_timer_label):
        _prepare_nemo_gym_rows(nemo_gym_rows, generation_config)
        accumulator = _NemoGymStreamAccumulator(
            rows=nemo_gym_rows,
            num_generations=num_generations,
            allow_mixed_agents=returns_entire_batch,
        )
        final_rollout_result: NemoGymRolloutResult | None = None
        actor_timing_metrics: dict[str, Any] = {}
        nemo_gym_environment = task_to_env["nemo_gym"]
        with timer.time(run_rollouts_timer_label):
            rollout_gen = nemo_gym_environment.run_rollouts.options(
                num_returns="streaming"
            ).remote(nemo_gym_rows, tokenizer, timer_prefix)
        rollout_iterator = rollout_gen.__aiter__()

    while True:
        stream_finished = False
        group_to_yield: NemoGymRolloutResult | None = None
        with timer.time(total_timer_label):
            with timer.time(run_rollouts_timer_label):
                try:
                    future = await anext(rollout_iterator)
                except StopAsyncIteration:
                    stream_finished = True
                else:
                    rowidx, result, timing_metrics = await future

            if not stream_finished:
                if timing_metrics is not None:
                    actor_timing_metrics = timing_metrics

                _tensorize_nemo_gym_result(result)
                completed_group = accumulator.add(rowidx, result)
                if completed_group is not None:
                    rollout_result = _postprocess_single_nemo_gym_group(
                        nemo_gym_rows=completed_group.rows,
                        results=completed_group.results,
                        timer=timer,
                        timer_prefix=timer_prefix,
                        policy_generation=policy_generation,
                        input_batch=input_batch.slice(
                            completed_group.group_index * num_generations,
                            (completed_group.group_index + 1) * num_generations,
                        ),
                        tokenizer=tokenizer,
                        log_full_result_tables=log_full_result_tables,
                        effort_config=effort_config,
                        reward_penalty_config=reward_penalty_config,
                        thinking_tags=thinking_tags,
                    )
                    if accumulator.is_complete:
                        final_rollout_result = rollout_result
                    else:
                        group_to_yield = rollout_result

        if stream_finished:
            break
        if group_to_yield is not None:
            yield group_to_yield

    with timer.time(total_timer_label):
        accumulator.finish()
        if final_rollout_result is None:
            raise RuntimeError(
                "NeMo-Gym completed without producing a final prompt group"
            )

    final_rollout_result.rollout_metrics.update(actor_timing_metrics)
    final_rollout_result.rollout_metrics.update(timer.get_timing_metrics("sum"))
    yield final_rollout_result


def run_nemo_gym_rollout_sync(
    policy_generation: GenerationInterface,
    input_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    task_to_env: dict[str, EnvironmentInterface],
    generation_config: GenerationConfig,
    log_full_result_tables: bool,
    max_seq_len: Optional[int] = None,
    max_rollout_turns: Optional[int] = None,
    greedy: bool = False,
    effort_config: Optional[EffortLevelsConfig] = None,
    reward_penalty_config: dict[str, Any] | BaseModel | None = None,
    thinking_tags: list[str] | tuple[str, ...] | None = None,
) -> NemoGymRolloutResult:
    """Run and return one complete NeMo-Gym batch synchronously.

    This compatibility API drains :func:`run_async_nemo_gym_rollout` with the
    whole input treated as one heterogeneous group, restoring input order and
    returning only after every row is complete.

    Args:
        policy_generation: Generation interface whose configuration supplies the
            model's maximum sequence length.
        input_batch: Batch whose ``extra_env_info`` field contains NeMo-Gym rows.
        tokenizer: Tokenizer used by the NeMo-Gym actor and local postprocessing.
        task_to_env: Environment mapping containing the ``"nemo_gym"`` actor.
        generation_config: Sampling parameters forwarded to every NeMo-Gym row.
        log_full_result_tables: Whether to include complete per-agent result
            payloads as W&B Tables in the rollout metrics.
        max_seq_len: Policy sequence-length limit used for compatibility validation.
        max_rollout_turns: Must be ``None`` because NeMo-Gym owns turn limits.
        greedy: Must be ``False`` because this path does not support greedy mode.
        effort_config: Optional configuration for effort-based reward shaping.
        reward_penalty_config: Optional reward-penalty configuration.
        thinking_tags: Optional opening and closing tags used by thinking penalties.

    Returns:
        The fully postprocessed NeMo-Gym rollout batch in input-row order.

    Raises:
        AssertionError: If an unsupported generation option is requested.
        TypeError: If a NeMo-Gym row or streamed row index has an invalid type.
        ValueError: If streamed rows violate the ordering, uniqueness, grouping, or
            task-index invariants documented by :func:`run_async_nemo_gym_rollout`.
        RuntimeError: If called from a running event loop, the actor or stream fails,
            or NeMo-Gym returns no complete rollout batch.
    """

    async def _consume_rollout() -> NemoGymRolloutResult:
        rollout_result = None
        async for rollout_result in run_async_nemo_gym_rollout(
            policy_generation=policy_generation,
            input_batch=input_batch,
            tokenizer=tokenizer,
            task_to_env=task_to_env,
            generation_config=generation_config,
            num_generations=input_batch.size,
            log_full_result_tables=log_full_result_tables,
            max_seq_len=max_seq_len,
            max_rollout_turns=max_rollout_turns,
            greedy=greedy,
            effort_config=effort_config,
            reward_penalty_config=reward_penalty_config,
            thinking_tags=thinking_tags,
            returns_entire_batch=True,
        ):
            pass
        if rollout_result is None:
            raise RuntimeError("NeMo-Gym did not return any rollouts")
        return rollout_result

    return asyncio.run(_consume_rollout())


def _postprocess_single_nemo_gym_group(
    nemo_gym_rows: list[dict],
    results: list[dict],
    timer: Timer,
    timer_prefix: str,
    policy_generation: GenerationInterface,
    input_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    log_full_result_tables: bool,
    effort_config: Optional[EffortLevelsConfig] = None,
    reward_penalty_config: dict[str, Any] | BaseModel | None = None,
    thinking_tags: list[str] | tuple[str, ...] | None = None,
) -> NemoGymRolloutResult:
    """Postprocess one complete prompt group from the NeMo-Gym stream."""
    # Length-based reward shaping for low-effort prompts
    shaping = _apply_effort_shaping(results, nemo_gym_rows, effort_config)
    length_rewards_low = shaping.length_rewards_low
    rewards_low = shaping.rewards_low
    low_lengths = shaping.low_lengths
    high_lengths = shaping.high_lengths

    resolved_reward_penalty_config = resolve_reward_penalty_config(
        reward_penalty_config, tokenizer, thinking_tags=thinking_tags
    )
    penalty_counts = apply_reward_penalties(results, resolved_reward_penalty_config)

    # Prepare for the rollout metrics calculation below. Not strictly necessary here, but good to have parity with `run_async_multi_turn_rollout`
    with timer.time(f"{timer_prefix}/prepare_for_metrics_calculation"):
        batch_size = len(nemo_gym_rows)
        if policy_generation.cfg.get("backend") == "sglang":
            max_total_tokens_per_sample = (
                policy_generation.cfg["sglang_cfg"].get("context_length")
                or policy_generation.cfg["max_total_sequence_length"]
            )
        elif "vllm_cfg" in policy_generation.cfg:
            max_total_tokens_per_sample = policy_generation.cfg["vllm_cfg"][
                "max_model_len"
            ]
        elif "trtllm_cfg" in policy_generation.cfg:
            max_total_tokens_per_sample = policy_generation.cfg["trtllm_cfg"][
                "max_model_len"
            ]
        elif "mcore_generation_config" in policy_generation.cfg:
            max_total_tokens_per_sample = policy_generation.cfg[
                "mcore_generation_config"
            ]["max_model_len"]
        else:
            max_total_tokens_per_sample = policy_generation.cfg[
                "max_total_sequence_length"
            ]
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
    with timer.time(f"{timer_prefix}/aggregate_metrics"):
        turn_counts = [m["turn_count"] for m in all_sample_metrics]
        max_gen_tokens_per_turn_values = [
            m["max_gen_tokens_per_turn"] for m in all_sample_metrics
        ]

        rollout_metrics = {
            **calculate_single_metric(
                turn_counts,
                batch_size,
                "turns_per_sample",
            ),
            "turns_per_sample/p95": pct(turn_counts, 95),
            "turns_per_sample/p99": pct(turn_counts, 99),
            **calculate_single_metric(
                [m["total_tokens"] for m in all_sample_metrics],
                batch_size,
                "total_tokens_per_sample",
            ),
            **calculate_single_metric(
                [m["assistant_tokens"] for m in all_sample_metrics],
                batch_size,
                "gen_tokens_per_sample",
            ),
            **calculate_single_metric(
                max_gen_tokens_per_turn_values,
                batch_size,
                "max_gen_tokens_per_turn",
            ),
            "max_gen_tokens_per_turn/p95": pct(max_gen_tokens_per_turn_values, 95),
            **calculate_single_metric(
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
                        calculate_single_metric(
                            values, len(agent_results), f"{agent_name}/{key}"
                        )
                    )

            if log_full_result_tables:
                to_log = [
                    [json.dumps(r, separators=((",", ":")))] for r in agent_results
                ]
                per_agent_metrics[f"{agent_name}/full_result"] = Table(
                    data=to_log, columns=["Full result"]
                )

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
            "mask_sample": _extract_mask_sample_flags(results),
        }
    )

    if length_rewards_low:
        rollout_metrics["mean_length_reward_low"] = sum(length_rewards_low) / len(
            length_rewards_low
        )
    if rewards_low:
        rollout_metrics["mean_reward_low"] = sum(rewards_low) / len(rewards_low)
    if low_lengths:
        rollout_metrics["mean_length_low"] = sum(low_lengths) / len(low_lengths)
        rollout_metrics["median_length_low"] = float(statistics.median(low_lengths))
    if high_lengths:
        rollout_metrics["mean_length_high"] = sum(high_lengths) / len(high_lengths)
        rollout_metrics["median_length_high"] = float(statistics.median(high_lengths))

    # Penalty metrics — map count keys to (config flag, metric name)
    _PENALTY_METRICS = {
        "duplicated_reasoning": (
            "penalize_duplicated_reasoning",
            "reasoning_equal_to_final_answer_rate",
        ),
        "empty_final_answer": (
            "penalize_empty_final_answer",
            "empty_final_answer_rate",
        ),
        "unwanted_token": ("penalize_unwanted_tokens", "unwanted_token_rate"),
        "malformed_think_tag": (
            "penalize_malformed_think_tag",
            "malformed_think_tag_rate",
        ),
    }
    if resolved_reward_penalty_config and results:
        for key, (flag, metric_name) in _PENALTY_METRICS.items():
            if _get_reward_penalty_config_value(resolved_reward_penalty_config, flag):
                rollout_metrics[metric_name] = penalty_counts[key] / len(results)

    # Expose per-component rewards as `reward/<name>` batch keys for multi-reward NeMo
    # Gym environments so GDPO can compute per-component advantages; single-reward envs
    # are unaffected. Mirrors the native rollout path's reward-component handling above.
    from nemo_rl.environments.nemo_gym import (
        build_reward_component_columns,
        extract_reward_components,
        validate_reward_components_match_scalar,
    )

    component_dicts = [extract_reward_components(r["full_result"]) for r in results]
    if any(c is not None for c in component_dicts):
        # Emit each component under a `reward/<name>` key (see
        # build_reward_component_columns): matches the native multi-reward path and what
        # get_gdpo_reward_component_keys() consumes.
        final_batch.update(build_reward_component_columns(component_dicts))
        # Leave total_reward as the verifier's scalar `reward` (set above); do not
        # silently overwrite it. When a verifier emits reward_components, the contract is
        # reward == sum(components), so overwriting would be a no-op in the correct case
        # and would only mask a misconfigured verifier when it isn't. Validate that
        # contract instead and fail fast on a real mismatch.
        validate_reward_components_match_scalar([r["full_result"] for r in results])

    group_task_index = None
    if nemo_gym_rows and NEMO_GYM_TASK_INDEX_KEY in nemo_gym_rows[0]:
        group_task_index = int(nemo_gym_rows[0][NEMO_GYM_TASK_INDEX_KEY])
        task_indices = [row.get(NEMO_GYM_TASK_INDEX_KEY) for row in nemo_gym_rows]
        if any(
            task_index is None or int(task_index) != group_task_index
            for task_index in task_indices
        ):
            raise ValueError(
                f"Expected one _ng_task_index per prompt group, got {task_indices}"
            )

    return NemoGymRolloutResult(
        input_ids=input_ids,
        final_batch=final_batch,
        rollout_metrics=rollout_metrics,
        task_index=group_task_index,
    )
