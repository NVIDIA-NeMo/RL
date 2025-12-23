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

from nemo_rl.data.interfaces import TaskDataSpec


def format_acereason_math(
    data: dict[str, str | float | int],
) -> dict[str, list[Any] | str]:
    """Format AceReason-Math data to the expected message format."""
    return {
        "messages": [
            {
                "role": "user",
                "content": data["problem"],
            },
            {
                "role": "assistant",
                "content": data["answer"],
            },
        ],
        # For v0.1 release, nemo rl datasets require a task_name key such that user can map a task processor per unique task.
        "task_name": "math",
    }


def extract_dataset(split_name: str, data_split: Any) -> Any:
    """Extract dataset split and add task_name field for GRPO compatibility."""
    if data_split is None:
        return None

    # Add task_name field to each sample for GRPO compatibility
    def add_task_name(example: dict) -> dict:
        example["task_name"] = "math"
        return example

    return data_split.map(add_task_name)


def prepare_acereason_math_dataset(seed: int = 42) -> dict[str, Dataset | None]:
    """Load and prepare the AceReason-Math dataset for GRPO training."""
    # Load the AceReason-Math dataset for training
    train_ds = load_dataset("nvidia/AceReason-Math", split="train")

    # Load AIME 2024 dataset for validation (following pattern of other math datasets)
    val_ds = load_dataset("HuggingFaceH4/aime_2024", split="train")

    # Shuffle the training dataset with the specified seed
    train_ds = train_ds.shuffle(seed=seed)

    # Format the examples, removing original columns
    train_formatted = train_ds.map(
        format_acereason_math, remove_columns=train_ds.column_names
    )
    val_formatted = val_ds.map(
        format_acereason_math, remove_columns=val_ds.column_names
    )

    formatted_ds_dict = {
        "train": extract_dataset("train", train_formatted),
        "validation": extract_dataset("validation", val_formatted),
    }

    return prepare_math_dataset(formatted_ds_dict)


def prepare_math_dataset(formatted_ds_dict: dict[str, Any]) -> dict[str, Any]:
    """Prepare math dataset with proper formatting for GRPO."""
    prepared_ds = {}
    for split, dataset in formatted_ds_dict.items():
        if dataset is not None:
            prepared_ds[split] = dataset
        else:
            prepared_ds[split] = None
    return prepared_ds


class AceReasonMathDataset:
    def __init__(self, seed: int = 42) -> None:
        """Initialize the AceReason-Math dataset with train/validation split.

        Args:
            seed: Random seed for reproducible splitting
        """
        self.formatted_ds = prepare_acereason_math_dataset(seed=seed)

        self.task_spec = TaskDataSpec(
            task_name="AceReason-Math",
        )
