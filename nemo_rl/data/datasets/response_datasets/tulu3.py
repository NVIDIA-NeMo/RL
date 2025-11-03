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

import json
from typing import Any

from datasets import load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


def to_preference_data_format(
    data: dict[str, Any],
) -> dict[
    str, list[dict[str, int | list[dict[str, str | Any]]]] | list[dict[str, str]]
]:
    chosen_conversation = data["chosen"]
    rejected_conversation = data["rejected"]

    context = chosen_conversation[:-1]

    # We assume that except last assistant response, all messages in
    # chosen and rejected conversations are similar. Validating this...
    assert json.dumps(context, ensure_ascii=False) == json.dumps(
        rejected_conversation[:-1], ensure_ascii=False
    ), (
        f"Context mismatch.\n\nchosen: {chosen_conversation}\n\n rejected: {rejected_conversation}"
    )

    # We assume that last response is always from the assistant. Validating this...
    assert chosen_conversation[-1]["role"] == "assistant", (
        f"The last chosen response ({chosen_conversation[-1]}) is not from assistant!"
    )
    assert rejected_conversation[-1]["role"] == "assistant", (
        f"The last rejected response ({rejected_conversation[-1]}) is not from assistant!"
    )

    chosen_response = chosen_conversation[-1]["content"]
    rejected_response = rejected_conversation[-1]["content"]

    return {
        "context": context,
        "completions": [
            {
                "rank": 0,
                "completion": [{"role": "assistant", "content": chosen_response}],
            },
            {
                "rank": 1,
                "completion": [{"role": "assistant", "content": rejected_response}],
            },
        ],
    }


class Tulu3PreferenceDataset:
    """Tulu3 preference dataset for DPO training."""

    def __init__(self) -> None:
        ds = load_dataset(
            path="allenai/llama-3.1-tulu-3-8b-preference-mixture",
            trust_remote_code=True,
        )
        self.formatted_ds = ds.map(to_preference_data_format)

        self.task_spec = TaskDataSpec(
            task_name="Tulu3Preference",
        )

def format_tulu3_sft_mixture(data: dict[str, Any]) -> dict[str, str | dict[str, str]]:
    """format for Tulu3 SFT data."""
    messages = data["messages"]
    
    # Ensure last message is from assistant
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError(f"Expected last message to be from assistant, got: {messages}")
    
    return {
        "messages": messages,
        "task_name": "tulu3_sft_mixture",
    }


class Tulu3SftMixtureDataset:
    """Tulu3 SFT mixture dataset."""

    def __init__(
        self, 
        seed: int = 42,
        test_size: float = 0.05,
        prompt_file: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        """Initialize the Tulu3 SFT mixture dataset.
        
        Args:
            seed: Random seed for train/validation split
            test_size: Proportion of data to use for validation (0.0-1.0)
            prompt_file: Optional prompt file path to be applied via TaskDataSpec
            max_samples: Optional maximum number of samples to use from the dataset
        """
        print(
            "WARNING: For reproducible experiments, preprocess the dataset once and define your own HfDataset subclass that directly uses the preprocessed datasets."
        )
        
        # Load the original dataset
        original_ds = load_dataset(
            path="allenai/tulu-3-sft-mixture",
            trust_remote_code=True,
        )["train"]  # This dataset only has a train split
        
        # Optionally limit the number of samples
        if max_samples is not None and max_samples > 0:
            original_ds = original_ds.shuffle(seed=seed).select(range(min(max_samples, len(original_ds))))
        
        # Split into train and validation sets
        split_ds = original_ds.train_test_split(test_size=test_size, seed=seed)
        
        # Format the examples without any reasoning processing
        train_formatted = split_ds["train"].map(
            format_tulu3_sft_mixture,
            remove_columns=split_ds["train"].column_names,
        )
        val_formatted = split_ds["test"].map(
            format_tulu3_sft_mixture,
            remove_columns=split_ds["test"].column_names,
        )

        self.formatted_ds = {
            "train": train_formatted,
            "validation": val_formatted,
        }

        self.task_spec = TaskDataSpec(
            task_name="Tulu3SftMixture",
            prompt_file=prompt_file,
        )