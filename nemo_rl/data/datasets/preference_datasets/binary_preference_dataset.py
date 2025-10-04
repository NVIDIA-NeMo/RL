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
from typing import Any, Optional

from nemo_rl.data.datasets.utils import load_dataset_from_path
from nemo_rl.data.interfaces import TaskDataSpec


def to_preference_data_format(
    data: dict[str, Any], prompt_key: str, chosen_key: str, rejected_key: str
) -> dict[str, list[dict[str, Any]]]:
    return {
        "context": data[prompt_key]
        if isinstance(data[prompt_key], list)
        else [{"role": "user", "content": data[prompt_key]}],
        "completions": [
            {
                "rank": 0,
                "completion": [{"role": "assistant", "content": data[chosen_key]}],
            },
            {
                "rank": 1,
                "completion": [{"role": "assistant", "content": data[rejected_key]}],
            },
        ],
    }


class BinaryPreferenceDataset:
    """Dataset class for binary preference data which can be loaded from a JSON file.

    This class handles loading of preference data for DPO and RM training.
    It will be converted to the format of PreferenceDataset through the `to_preference_data_format` function.

    The input JSONL files should contain valid JSON objects formatted like this:
    {
        prompt_key: str,    # The input prompt/context
        chosen_key: str,    # The preferred/winning response
        rejected_key: str,  # The non-preferred/losing response
    }

    Args:
        train_data_path: Path to the JSON file containing training data
        val_data_path: Path to the JSON file containing validation data
        prompt_key: Key for the input prompt/context, default is "prompt"
        chosen_key: Key for the preferred/winning response, default is "chosen"
        rejected_key: Key for the non-preferred/losing response, default is "rejected"
        train_split: Split name for the training data, used for HuggingFace datasets, default is None
        val_split: Split name for the validation data, used for HuggingFace datasets, default is None
    """

    def __init__(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        train_split: Optional[str] = None,
        val_split: Optional[str] = None,
    ):
        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key

        # load from json file or huggingface
        train_ds = load_dataset_from_path(train_data_path, train_split)
        if val_data_path:
            val_ds = load_dataset_from_path(val_data_path, val_split)
        else:
            val_ds = None

        # format the dataset
        # convert to PreferenceDataset format
        train_ds = train_ds.map(
            to_preference_data_format,
            fn_kwargs={
                "prompt_key": prompt_key,
                "chosen_key": chosen_key,
                "rejected_key": rejected_key,
            },
        )
        if val_ds:
            val_ds = val_ds.map(
                to_preference_data_format,
                fn_kwargs={
                    "prompt_key": prompt_key,
                    "chosen_key": chosen_key,
                    "rejected_key": rejected_key,
                },
            )

        # store the formatted dataset
        self.formatted_ds = {
            "train": train_ds,
            "validation": val_ds,
        }

        self.task_spec = TaskDataSpec(task_name="BinaryPreferenceDataset")
