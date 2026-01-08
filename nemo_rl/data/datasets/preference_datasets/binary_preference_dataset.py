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

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import load_dataset_from_path


class BinaryPreferenceDataset(RawDataset):
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
        data_path: Path to the dataset JSON file
        prompt_key: Key for the input prompt/context, default is "prompt"
        chosen_key: Key for the preferred/winning response, default is "chosen"
        rejected_key: Key for the non-preferred/losing response, default is "rejected"
        split: Optional split name for the dataset, used for HuggingFace datasets
    """

    def __init__(
        self,
        data_path: str,
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        split: Optional[str] = None,
        **kwargs,
    ):
        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.task_name = data_path.split("/")[-1].split(".")[0]

        # load from local or huggingface
        self.dataset = load_dataset_from_path(data_path, split)

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(data[self.prompt_key], list):
            context = data[self.prompt_key]
        else:
            context = [{"role": "user", "content": data[self.prompt_key]}]

        return {
            "context": context,
            "completions": [
                {
                    "rank": 0,
                    "completion": [
                        {"role": "assistant", "content": data[self.chosen_key]}
                    ],
                },
                {
                    "rank": 1,
                    "completion": [
                        {"role": "assistant", "content": data[self.rejected_key]}
                    ],
                },
            ],
            "task_name": self.task_name,
        }
