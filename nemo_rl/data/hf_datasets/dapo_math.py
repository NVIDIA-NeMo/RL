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

from nemo_rl.data.hf_datasets.deepscaler import format_math
from nemo_rl.data.interfaces import TaskDataSpec


def format_dapo_math_17k(
    data: dict[str, str | float | int],
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
        "task_name": "math",
    }


def prepare_dapo_math_17k_dataset(
    seed: int = 42, repititions: int = 32
) -> dict[str, Dataset | None]:
    """Load and split the DeepScaler dataset into train and test sets."""
    # Load the original dataset for training
    train_ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")

    # Load hendrydong/aime24 dataset for validation
    val_ds = load_dataset("HuggingFaceH4/aime_2024", split="train")

    # Shuffle the training dataset with the specified seed
    train_ds = train_ds.shuffle(seed=seed)

    # Format the examples, removing original columns
    train_formatted = train_ds.map(
        format_dapo_math_17k, remove_columns=train_ds.column_names
    )
    val_formatted = val_ds.map(format_math, remove_columns=val_ds.column_names)

    # DAPO paper repeats the AIME validation set 32 times and reports avg@32 score
    val_repeated = []
    for _ in range(repititions):
        val_repeated.extend(val_formatted)
    val_formatted = val_formatted.from_list(val_repeated)

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class DAPOMath17KDataset:
    def __init__(self, seed: int = 42) -> None:
        """Initialize the DAPO Math 17K dataset with train split.

        Args:
            seed: Random seed for reproducible splitting
        """
        self.formatted_ds = prepare_dapo_math_17k_dataset(seed=seed)

        self.task_spec = TaskDataSpec(
            task_name="DAPOMath17K",
        )
