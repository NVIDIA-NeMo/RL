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


class ResponseDataset(RawDataset):
    """Dataset class for response data which can be loaded from a JSON file.

    This class handles loading of response data for SFT and RL training.
    The input JSONL files should contain valid JSON objects formatted like this:
    {
        input_key: str,     # The input prompt/context
        output_key: str,    # The output response/answer
    }

    Args:
        data_path: Path to the JSON file containing training data
        input_key: Key for the input text
        output_key: Key for the output text
        split: Split name for the training data, used for HuggingFace datasets, default is None
        split_validation_size: Size of the validation data, default is 0
        seed: Seed for training/validation split when split_validation_size > 0, default is 42
    """

    def __init__(
        self,
        data_path: str,
        input_key: str = "input",
        output_key: str = "output",
        split: Optional[str] = None,
        split_validation_size: float = 0,
        seed: int = 42,
        **kwargs,
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.task_name = data_path.split("/")[-1].split(".")[0]

        # load from local or huggingface
        self.dataset = load_dataset_from_path(data_path, split)

        # format the dataset
        if "messages" not in self.dataset.column_names:
            self.dataset = self.dataset.map(
                self.format_data,
                remove_columns=self.dataset.column_names,
            )
        else:
            self.dataset = self.dataset.add_column(
                "task_name", [self.task_name] * len(self.dataset)
            )

        # `self.val_dataset` is used (not None) only when current dataset is used for both training and validation
        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": data[self.input_key]},
                {"role": "assistant", "content": data[self.output_key]},
            ],
            "task_name": self.task_name,
        }
