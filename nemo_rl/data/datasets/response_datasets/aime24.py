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


class AIME2024Dataset(RawDataset):
    def __init__(self, repeat: int = 16, **kwargs) -> None:
        """Initialize the AIME2024 dataset with train split."""
        self.task_name = "AIME2024"

        # load from huggingface
        self.dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

        # repeat the dataset
        self.dataset = self.dataset.repeat(repeat)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": data["problem"]},
                {"role": "assistant", "content": data["answer"]},
            ],
            "task_name": self.task_name,
        }
