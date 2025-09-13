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
from typing import Optional

from nemo_rl.data.datasets.utils import load_dataset_from_path
from nemo_rl.data.interfaces import TaskDataSpec


class PreferenceDataset:
    """Dataset class for preference data which can be loaded from a JSON file.

    This class handles loading of preference data for DPO and RM training.
    The input JSONL files should contain valid JSON objects formatted like this:
    {
        "context": list of dicts, # The prompt message (including previous turns, if any)
        "completions": list of dicts, # The list of completions
            {
                "rank": int, # The rank of the completion (lower rank is preferred)
                "completion": list of dicts, # The completion message(s)
            }
    }

    Args:
        train_data_path: Path to the JSON file containing training data
        val_data_path: Path to the JSON file containing validation data
        train_split: Split name for the training data, used for HuggingFace datasets, default is None
        val_split: Split name for the validation data, used for HuggingFace datasets, default is None
    """

    def __init__(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        train_split: Optional[str] = None,
        val_split: Optional[str] = None,
    ):
        # load from json file or huggingface
        train_ds = load_dataset_from_path(train_data_path, train_split)
        if val_data_path:
            val_ds = load_dataset_from_path(val_data_path, val_split)
        else:
            val_ds = None

        # store the formatted dataset
        self.formatted_ds = {
            "train": train_ds,
            "validation": val_ds,
        }

        self.task_spec = TaskDataSpec(task_name="PreferenceDataset")
