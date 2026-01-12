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

from datasets import Dataset, load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


def format_math(
    data: dict[str, str | float | int],
    output_key: str = "answer",
    task_name: str = "GSM8K",
) -> dict[str, list[Any] | str]:
    return {
        "messages": [
            {
                "role": "user",
                "content": data["question"],
            },
            {
                "role": "assistant",
                "content": data[output_key],
            },
        ],
        "task_name": task_name,
    }


def prepare_gsm8k_dataset(
    seed: int = 42,
    output_key: str = "answer",
    task_name: str = "GSM8K",
) -> dict[str, Dataset | None]:
    """Load and split the OpenMathInstruct-2 dataset into train and validation sets using HF's train_test_split."""
    print(
        "WARNING: For reproducible experiments, preprocess the dataset once and define your own HfDataset subclass that directly uses the preprocessed datasets."
    )

    # Load the original dataset
    original_ds = load_dataset("openai/gsm8k", "main")

    # Format the examples, removing original columns
    train_formatted = original_ds["train"].map(
        format_math,
        remove_columns=original_ds["train"].column_names,
        fn_kwargs={"output_key": output_key, "task_name": task_name},
    )
    val_formatted = original_ds["test"].map(
        format_math,
        remove_columns=original_ds["test"].column_names,
        fn_kwargs={"output_key": output_key, "task_name": task_name},
    )

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class GSM8KDataset(RawDataset):
    def __init__(
        self,
        seed: int = 42,
        output_key: str = "answer",
    ):
        """Initialize the OpenMathInstruct2 dataset with train/validation split.

        Args:
            seed: Random seed for reproducible splitting
        """
        self.task_name = "GSM8K"
        self.formatted_ds = prepare_gsm8k_dataset(
            seed=seed,
            output_key=output_key,
            task_name=self.task_name,
        )
