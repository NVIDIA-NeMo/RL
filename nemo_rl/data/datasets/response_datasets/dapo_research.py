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


def format_dapo(data: dict[str, str | float | int]) -> dict[str, list[Any] | str]:
    return {
        "messages": [
            {
                "role": "user",
                "content": data["prompt"][0]["content"],
            },
            {
                "role": "assistant",
                "content": data["reward_model"]["ground_truth"],
            },
        ],
        # For v0.1 release, nemo rl datasets require a task_name key such that user can map a task processor per unique task.
        "task_name": "dapo",
    }


def prepare_dapo_dataset(seed: int = 42) -> dict[str, Dataset | None]:
    """Load and split the DAPO and AIME2024 datasets into train and test sets."""
    # Load the original dataset for training
    train_ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")

    # Load the 32 times repeated BytedTsinghua-SIA/AIME-2024 dataset for validation
    val_ds = load_dataset("BytedTsinghua-SIA/AIME-2024", split="train")

    ## NOTE: for now, turning off shuffling of train dataset to match verl
    #train_ds = train_ds.shuffle(seed=seed)

    # Format the examples, removing original columns
    train_formatted = train_ds.map(format_dapo, remove_columns=train_ds.column_names)
    val_formatted = val_ds.map(format_dapo, remove_columns=val_ds.column_names)

    # # Compute accuracy 16 times per sample (matching the DeepScaleR evaluation setting)
    # val_repeated = []
    # for _ in range(16):
    #     val_repeated.extend(val_formatted)
    # val_formatted = val_formatted.from_list(val_repeated)

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class DAPODataset:
    def __init__(self, seed: int = 42) -> None:
        """Initialize the DAPO dataset for train and AIME2024 dataset for test split.

        Args:
            seed: Random seed for reproducible splitting
        """
        self.formatted_ds = prepare_dapo_dataset(seed=seed)

        self.task_spec = TaskDataSpec(
            task_name="DAPO",
        )
