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


def format_dapo_math_17k(
    data: dict[str, str | float | int],
    task_name: str = "DAPOMath17K",
) -> dict[str, list[Any] | str]:
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
        "task_name": task_name,
    }


def prepare_dapo_math_17k_dataset(
    seed: int = 42, task_name: str = "DAPOMath17K"
) -> dict[str, Dataset | None]:
    """Load and split the DeepScaler dataset into train and test sets."""
    # Load the original dataset for training
    train_ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")

    # Load hendrydong/aime24 dataset for validation
    val_ds = load_dataset("BytedTsinghua-SIA/AIME-2024", split="train")

    # Shuffle the training dataset with the specified seed
    train_ds = train_ds.shuffle(seed=seed)

    # Format the examples, removing original columns
    train_formatted = train_ds.map(
        format_dapo_math_17k,
        remove_columns=train_ds.column_names,
        fn_kwargs={"task_name": task_name},
    )
    val_formatted = val_ds.map(
        format_dapo_math_17k,
        remove_columns=val_ds.column_names,
        fn_kwargs={"task_name": task_name},
    )

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class DAPOMath17KDataset(RawDataset):
    def __init__(self, seed: int = 42) -> None:
        """Initialize the DAPO Math 17K dataset with train split.

        Args:
            seed: Random seed for reproducible splitting
        """
        self.task_name = "DAPOMath17K"
        self.formatted_ds = prepare_dapo_math_17k_dataset(
            seed=seed, task_name=self.task_name
        )
