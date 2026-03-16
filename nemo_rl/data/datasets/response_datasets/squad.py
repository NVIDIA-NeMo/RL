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

from nemo_rl.data.datasets.raw_dataset import RawDataset


class SquadDataset(RawDataset):
    """Simple wrapper around the squad dataset.

    Args:
        split: Split name for the dataset, default is "train"
    """

    def __init__(self, split: str = "train", **kwargs) -> None:
        self.task_name = "squad"

        # load from huggingface
        self.dataset = load_dataset("rajpurkar/squad")[split]

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {
                    "role": "system",
                    "content": data["context"],
                },
                {
                    "role": "user",
                    "content": data["question"],
                },
                {
                    "role": "assistant",
                    "content": data["answers"]["text"][0],
                },
            ],
            "task_name": self.task_name,
        }
