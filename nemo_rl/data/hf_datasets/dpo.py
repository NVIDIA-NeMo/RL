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
from datasets import load_dataset
from typing import Any

from nemo_rl.data.interfaces import TaskDataSpec

import warnings


def to_preference_data_format(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        "context": [{"role": "user", "content": data.pop("prompt")}],
        "completions": [
                {"rank": 0, "completion": [{"role": "assistant", "content": data.pop("chosen_response")}]},
                {"rank": 1, "completion": [{"role": "assistant", "content": data.pop("rejected_response")}]}
            ]
    }


class DPODataset:
    """Dataset class for Direct Preference Optimization (DPO) training.

    This class handles loading of preference data for DPO training.
    The input JSON files should contain examples with the following structure:
    {
        "prompt": str,           # The input prompt/context
        "chosen_response": str,  # The preferred/winning response
        "rejected_response": str # The non-preferred/losing response
    }

    Args:
        train_data_path (str): Path to the JSON file containing training data
        val_data_path (str): Path to the JSON file containing validation data

    """

    def __init__(self, train_data_path: str, val_data_path: str):
        warnings.warn(
            "DPODataset is deprecated and will be removed in a future version. Use PreferenceDataset instead.",
            category=DeprecationWarning,
            stacklevel=2
        )

        self.formatted_ds = {
            "train": load_dataset("json", data_files=train_data_path, split="train").map(to_preference_data_format),
            "validation": load_dataset("json", data_files=val_data_path, split="train").map(to_preference_data_format),
        }

        self.task_spec = TaskDataSpec(
            task_name="DPO",
        )
