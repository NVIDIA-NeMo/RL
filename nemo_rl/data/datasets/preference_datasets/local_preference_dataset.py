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

from datasets import load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


class LocalPreferenceDataset:
    """Dataset class for local preference data.

    This class handles loading of preference data for DPO and RM training.
    The input JSON files should contain examples with the following structure:
    {
        prompt_key: str,    # The input prompt/context
        chosen_key: str,    # The preferred/winning response
        rejected_key: str,  # The non-preferred/losing response
    }

    Args:
        train_data_path: Path to the JSON file containing training data
        val_data_path: Path to the JSON file containing validation data
        prompt_key: Key for the input prompt/context
        chosen_key: Key for the preferred/winning response
        rejected_key: Key for the non-preferred/losing response
    """

    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
    ):
        train_original_dataset = load_dataset("json", data_files=train_data_path)[
            "train"
        ]
        val_original_dataset = load_dataset("json", data_files=val_data_path)["train"]

        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key

        formatted_train_dataset = train_original_dataset.map(self._rekey)
        formatted_val_dataset = val_original_dataset.map(self._rekey)

        self.formatted_ds = {
            "train": formatted_train_dataset,
            "validation": formatted_val_dataset,
        }

        self.task_spec = TaskDataSpec(
            task_name="DPO",
        )

    def _rekey(self, data: dict[str, Any]):
        return {
            "prompt": data[self.prompt_key],
            "chosen_response": data[self.chosen_key],
            "rejected_response": data[self.rejected_key],
        }
