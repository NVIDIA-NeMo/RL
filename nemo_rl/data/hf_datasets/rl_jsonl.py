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
import random

from nemo_rl.data.interfaces import TaskDataSpec
import json


def format_math(data: dict[str, str | float | int]) -> dict[str, list[Any] | str]:
    return {
        "messages": [
            {
                "role": "user",
                "content": data["question"],
            },
        ],
        # For v0.1 release, nemo rl datasets require a task_name key such that user can map a task processor per unique task.
        "task_name": "llm_judge",
        "metadata": {
            "question": data["question"],
        },
    }


def prepare_dataset(train_path: str, val_path: str, seed: int = 42) -> dict[str, Dataset | None]:
    """Load and split the DeepScaler dataset into train and test sets."""
    # Load the original dataset for training
    train_ds = []
    print(f"Loading jsonl dataset from {train_path}")
    with open(train_path, "r") as f:
        for line in f:
            data = json.loads(line)
            train_ds.append(data)

    val_ds = []
    print(f"Loading jsonl dataset from {val_path}")
    with open(val_path, "r") as f:
        for line in f:
            data = json.loads(line)
            val_ds.append(data)
    # Load hendrydong/aime24 dataset for validation
    val_ds = Dataset.from_list(val_ds)

    # Shuffle the training dataset with the specified seed
    random.seed(seed)
    random.shuffle(train_ds)

    train_repeated = []
    for _ in range(50):
        train_repeated.extend(train_ds)

    train_ds = Dataset.from_list(train_repeated)

    # Format the examples, removing original columns
    train_formatted = train_ds.map(format_math, remove_columns=train_ds.column_names)
    val_formatted = val_ds.map(format_math, remove_columns=val_ds.column_names)

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class JsonlDataset:
    def __init__(self, train_path: str, val_path: str, seed: int = 42) -> None:
        """Initialize the DeepScaler dataset with train/test split.

        Args:
            seed: Random seed for reproducible splitting
        """
        self.formatted_ds = prepare_dataset(train_path, val_path, seed=seed)

        self.task_spec = TaskDataSpec(
            task_name="DeepScaler",
        )

