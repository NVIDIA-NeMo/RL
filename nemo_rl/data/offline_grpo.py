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

"""Grouped teacher-trajectory data support for offline GRPO."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from nemo_rl.data.datasets.utils import load_dataset_from_path
from nemo_rl.data.interfaces import LLMMessageLogType, TaskDataSpec
from nemo_rl.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
    get_formatted_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class OfflineGRPODatasetConfig(BaseModel, extra="allow"):
    """Location and column mapping for one offline GRPO dataset."""

    data_path: str
    subset: str | None = None
    split: str | None = "train"
    prompt_key: str = "prompt"
    responses_key: str = "responses"
    rewards_key: str = "rewards"
    prompt_file: str | None = None
    system_prompt_file: str | None = None


class OfflineGRPODataConfig(BaseModel, extra="allow"):
    """Data loading and tokenization configuration for offline GRPO."""

    max_input_seq_length: int
    add_bos: bool = True
    add_eos: bool = True
    shuffle: bool = True
    num_workers: int = 1
    train: OfflineGRPODatasetConfig
    validation: OfflineGRPODatasetConfig | None = None


@dataclass
class OfflineGRPOGroup:
    """All teacher trajectories and rewards associated with one prompt."""

    prompt_messages: list[dict[str, Any]]
    responses: list[Any]
    rewards: torch.Tensor
    dataset_index: int


@dataclass
class OfflineGRPOBatchMetrics:
    """Reward composition metrics for a selected batch of prompt groups."""

    mean_reward: float
    all_positive_group_fraction: float
    all_non_positive_group_fraction: float
    invalid_sequence_fraction: float
    num_prompt_groups: int


@dataclass
class PreparedOfflineGRPOBatch:
    """Flattened trajectory batch plus fields needed for advantage estimation."""

    data: BatchedDataDict[Any]
    rewards: torch.Tensor
    prompt_ids: torch.Tensor
    metrics: OfflineGRPOBatchMetrics


def _normalize_messages(
    value: Any, *, default_role: str, field_name: str
) -> list[dict]:
    """Normalize a string or chat-message sequence to OpenAI chat messages."""
    if isinstance(value, str):
        return [{"role": default_role, "content": value}]
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(
            f"{field_name} must be a string, message mapping, or message sequence; "
            f"got {type(value).__name__}"
        )

    messages = []
    for message in value:
        if not isinstance(message, Mapping):
            raise TypeError(
                f"Every entry in {field_name} must be a message mapping; "
                f"got {type(message).__name__}"
            )
        if "content" not in message:
            raise ValueError(f"Every message in {field_name} must contain 'content'")
        normalized = dict(message)
        normalized.setdefault("role", default_role)
        messages.append(normalized)
    if not messages:
        raise ValueError(f"{field_name} must contain at least one message")
    return messages


def _truncate_message_log(
    message_log: LLMMessageLogType, max_seq_length: int
) -> LLMMessageLogType:
    """Truncate tokenized messages to a total sequence-length budget."""
    remaining = max_seq_length
    truncated = []
    for message in message_log:
        new_message = dict(message)
        token_ids = new_message["token_ids"]
        assert isinstance(token_ids, torch.Tensor)
        new_message["token_ids"] = token_ids[:remaining]
        remaining -= len(new_message["token_ids"])
        truncated.append(new_message)
        if remaining == 0:
            break
    return truncated


class OfflineGRPODataset:
    """Load raw prompt groups and tokenize only selected trajectories per step."""

    def __init__(
        self,
        config: OfflineGRPODatasetConfig,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_seq_length: int,
        add_bos: bool,
        add_eos: bool,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.dataset = load_dataset_from_path(
            config.data_path, config.subset, config.split
        )
        self.task_spec = TaskDataSpec(
            task_name=Path(config.data_path).stem,
            prompt_file=config.prompt_file,
            system_prompt_file=config.system_prompt_file,
        )

        required_columns = {
            config.prompt_key,
            config.responses_key,
            config.rewards_key,
        }
        missing_columns = required_columns.difference(self.dataset.column_names)
        if missing_columns:
            raise ValueError(
                f"Offline GRPO dataset {config.data_path!r} is missing columns: "
                f"{sorted(missing_columns)}"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> OfflineGRPOGroup:
        row = self.dataset[idx]
        prompt_messages = _normalize_messages(
            row[self.config.prompt_key],
            default_role="user",
            field_name=self.config.prompt_key,
        )
        if self.task_spec.system_prompt and (
            not prompt_messages or prompt_messages[0].get("role") != "system"
        ):
            prompt_messages.insert(
                0, {"role": "system", "content": self.task_spec.system_prompt}
            )

        raw_responses = row[self.config.responses_key]
        raw_rewards = row[self.config.rewards_key]
        if not isinstance(raw_responses, Sequence) or isinstance(
            raw_responses, (str, bytes)
        ):
            raise TypeError(
                f"{self.config.responses_key} must be a sequence of trajectories"
            )
        if not isinstance(raw_rewards, Sequence) or isinstance(
            raw_rewards, (str, bytes)
        ):
            raise TypeError(f"{self.config.rewards_key} must be a sequence of rewards")
        if len(raw_responses) != len(raw_rewards):
            raise ValueError(
                f"Prompt {idx} has {len(raw_responses)} responses but "
                f"{len(raw_rewards)} rewards"
            )
        if not raw_responses:
            raise ValueError(f"Prompt {idx} has no teacher trajectories")

        rewards = []
        for response_idx, raw_reward in enumerate(raw_rewards):
            reward = float(raw_reward)
            if not math.isfinite(reward):
                raise ValueError(
                    f"Reward {response_idx} for prompt {idx} must be finite; got {reward}"
                )
            rewards.append(reward)

        return OfflineGRPOGroup(
            prompt_messages=prompt_messages,
            responses=list(raw_responses),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            dataset_index=idx,
        )


def offline_grpo_collate_fn(
    groups: list[OfflineGRPOGroup],
) -> list[OfflineGRPOGroup]:
    """Keep prompt groups intact until trajectory selection in the trainer."""
    return groups


def _select_response_indices(
    group: OfflineGRPOGroup,
    *,
    num_responses: int,
    selection: Literal["first", "random"],
    seed: int,
    step: int,
) -> torch.Tensor:
    """Choose trajectories deterministically for one prompt and training step."""
    available = len(group.responses)
    if available < num_responses:
        raise ValueError(
            f"Prompt at dataset index {group.dataset_index} has {available} teacher "
            f"trajectories, but offline_grpo.num_responses_per_prompt="
            f"{num_responses}"
        )
    if selection == "first":
        return torch.arange(num_responses)

    generator = torch.Generator()
    selection_seed = seed + step * 1_000_003 + group.dataset_index
    generator.manual_seed(selection_seed % torch.iinfo(torch.int64).max)
    return torch.randperm(available, generator=generator)[:num_responses]


def prepare_offline_grpo_batch(
    groups: list[OfflineGRPOGroup],
    dataset: OfflineGRPODataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    num_responses_per_prompt: int,
    response_selection: Literal["first", "random"],
    seed: int,
    step: int,
    positive_reward_threshold: float,
    make_sequence_length_divisible_by: int,
) -> PreparedOfflineGRPOBatch:
    """Select, flatten, mask, and pad grouped trajectories for policy training."""
    if not groups:
        raise ValueError("Cannot prepare an empty offline GRPO batch")

    message_logs = []
    reward_parts = []
    loss_multipliers = []
    prompt_id_parts = []
    all_positive_groups = 0
    all_non_positive_groups = 0

    for prompt_id, group in enumerate(groups):
        selected = _select_response_indices(
            group,
            num_responses=num_responses_per_prompt,
            selection=response_selection,
            seed=seed,
            step=step,
        )
        group_rewards = group.rewards[selected]
        if torch.all(group_rewards > positive_reward_threshold):
            all_positive_groups += 1
        if torch.all(group_rewards <= positive_reward_threshold):
            all_non_positive_groups += 1

        for response_idx in selected.tolist():
            response_messages = _normalize_messages(
                group.responses[response_idx],
                default_role="assistant",
                field_name=f"responses[{response_idx}]",
            )
            if response_messages[-1].get("role") != "assistant":
                raise ValueError(
                    f"Teacher trajectory {response_idx} for prompt "
                    f"{group.dataset_index} must end with an assistant message"
                )
            messages = [dict(message) for message in group.prompt_messages]
            messages.extend(response_messages)
            message_log = get_formatted_message_log(
                messages,
                tokenizer,
                dataset.task_spec,
                add_bos_token=dataset.add_bos,
                add_eos_token=dataset.add_eos,
                add_generation_prompt=False,
            )
            length = sum(len(message["token_ids"]) for message in message_log)
            is_valid = length <= dataset.max_seq_length
            if not is_valid:
                message_log = _truncate_message_log(message_log, dataset.max_seq_length)
            message_logs.append(message_log)
            loss_multipliers.append(float(is_valid))
        reward_parts.append(group_rewards)
        prompt_id_parts.append(
            torch.full((num_responses_per_prompt,), prompt_id, dtype=torch.long)
        )

    add_loss_mask_to_message_log(message_logs, only_unmask_final=True)
    flat_messages, input_lengths = batched_message_log_to_flat_message(
        message_logs,
        pad_value_dict={"token_ids": tokenizer.pad_token_id},
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
    )

    rewards = torch.cat(reward_parts)
    prompt_ids = torch.cat(prompt_id_parts)
    sample_mask = torch.tensor(loss_multipliers, dtype=torch.float32)
    data = BatchedDataDict[Any](
        {
            "input_ids": flat_messages["token_ids"],
            "input_lengths": input_lengths,
            "token_mask": flat_messages["token_loss_mask"],
            "sample_mask": sample_mask,
            "rewards": rewards,
            "prompt_ids": prompt_ids,
        }
    )

    num_groups = len(groups)
    metrics = OfflineGRPOBatchMetrics(
        mean_reward=rewards.mean().item(),
        all_positive_group_fraction=all_positive_groups / num_groups,
        all_non_positive_group_fraction=all_non_positive_groups / num_groups,
        invalid_sequence_fraction=1.0 - sample_mask.mean().item(),
        num_prompt_groups=num_groups,
    )
    return PreparedOfflineGRPOBatch(
        data=data,
        rewards=rewards,
        prompt_ids=prompt_ids,
        metrics=metrics,
    )


def setup_offline_grpo_data(
    tokenizer: PreTrainedTokenizerBase,
    config: OfflineGRPODataConfig,
) -> tuple[OfflineGRPODataset, OfflineGRPODataset | None]:
    """Load the configured offline GRPO train and validation datasets."""
    train_dataset = OfflineGRPODataset(
        config.train,
        tokenizer,
        max_seq_length=config.max_input_seq_length,
        add_bos=config.add_bos,
        add_eos=config.add_eos,
    )
    val_dataset = (
        OfflineGRPODataset(
            config.validation,
            tokenizer,
            max_seq_length=config.max_input_seq_length,
            add_bos=config.add_bos,
            add_eos=config.add_eos,
        )
        if config.validation is not None
        else None
    )
    return train_dataset, val_dataset
