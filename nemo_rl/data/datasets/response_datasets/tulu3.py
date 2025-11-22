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

from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


def format_tulu3_sft_mixture(
    data: dict[str, Any], task_name: str = "tulu3_sft_mixture"
) -> dict[str, str | dict[str, str]]:
    """Format for Tulu3 SFT data."""
    messages = data["messages"]

    # Ensure last message is from assistant
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError(f"Expected last message to be from assistant, got: {messages}")

    return {
        "messages": messages,
        "task_name": task_name,
    }


class Tulu3SftMixtureDataset(RawDataset):
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

        self.task_name = "tulu3_sft_mixture"

        # Load the original dataset
        original_ds = load_dataset(
            path="allenai/tulu-3-sft-mixture",
            trust_remote_code=True,
        )["train"]  # This dataset only has a train split

        # Optionally limit the number of samples
        if max_samples is not None and max_samples > 0:
            original_ds = original_ds.shuffle(seed=seed).select(
                range(min(max_samples, len(original_ds)))
            )

        # Split into train and validation sets
        split_ds = original_ds.train_test_split(test_size=test_size, seed=seed)

        # Format the examples without any reasoning processing
        train_formatted = split_ds["train"].map(
            format_tulu3_sft_mixture,
            remove_columns=split_ds["train"].column_names,
            fn_kwargs={"task_name": self.task_name},
        )
        val_formatted = split_ds["test"].map(
            format_tulu3_sft_mixture,
            remove_columns=split_ds["test"].column_names,
            fn_kwargs={"task_name": self.task_name},
        )

        self.formatted_ds = {
            "train": train_formatted,
            "validation": val_formatted,
        }
