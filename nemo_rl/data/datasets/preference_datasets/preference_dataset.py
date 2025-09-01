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

from datasets import load_dataset

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


class PreferenceDataset:
    """Dataset class for preference data which can be loaded from a JSON file.

    This class handles loading of preference data for DPO and RM training.
    The input JSON files should contain examples with the either of the following structures:
    1. rank format:
    {
        "context": list of dicts, # The prompt message (including previous turns, if any)
        "completions": list of dicts, # The list of completions
            {
                "rank": int, # The rank of the completion (lower rank is preferred)
                "completion": list of dicts, # The completion message(s)
            }
    }
    2. chosen - rejected format:
    This format will be converted to the rank format.
    {
        prompt_key: str,    # The input prompt/context
        chosen_key: str,    # The preferred/winning response
        rejected_key: str,  # The non-preferred/losing response
    }

    Args:
        train_data_path: Path to the JSON file containing training data
        val_data_path: Path to the JSON file containing validation data
        prompt_key: Key for the input prompt/context, set to None and not used for rank format
        chosen_key: Key for the preferred/winning response, set to None and not used for rank format
        rejected_key: Key for the non-preferred/losing response, set to None and not used for rank format
        train_split: Split name for the training data, default is "train"
        val_split: Split name for the validation data, default is "train"
    """

    def __init__(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        prompt_key: str = None,
        chosen_key: str = None,
        rejected_key: str = None,
        train_split: str = "train",
        val_split: str = "train",
    ):
        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key

        is_rank_format = False
        if prompt_key is None or chosen_key is None or rejected_key is None:
            print(
                "One of prompt_key/chosen_key/rejected_key is None. Using the preference dataset in rank format."
            )
            is_rank_format = True
        else:
            print("Using the preference dataset in chosen - rejected format.")

        # load from json file
        train_ds = load_dataset("json", data_files=train_data_path)[train_split]
        if val_data_path:
            val_ds = load_dataset("json", data_files=val_data_path)[val_split]
        else:
            val_ds = None

        # format the dataset
        # convert chosen - rejected format to rank format
        if not is_rank_format:
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

        self.task_spec = TaskDataSpec(task_name="PreferenceDataset")
