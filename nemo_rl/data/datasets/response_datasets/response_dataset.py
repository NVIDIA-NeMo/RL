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
        train_data_path: Path to the JSON file containing training data
        val_data_path: Path to the JSON file containing validation data
        input_key: Key for the input text
        output_key: Key for the output text
        train_split: Split name for the training data, used for HuggingFace datasets, default is None
        val_split: Split name for the validation data, used for HuggingFace datasets, default is None
    """

    def __init__(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        input_key: str = "input",
        output_key: str = "output",
        train_split: Optional[str] = None,
        val_split: Optional[str] = None,
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.task_name = "ResponseDataset"
        # load from json file or huggingface
        train_ds = load_dataset_from_path(train_data_path, train_split)
        if val_data_path:
            val_ds = load_dataset_from_path(val_data_path, val_split)
        else:
            val_ds = None

        # format the dataset
        train_ds = train_ds.map(
            self.add_messages_key, fn_kwargs={"task_name": self.task_name}
        )
        if val_ds:
            val_ds = val_ds.map(
                self.add_messages_key, fn_kwargs={"task_name": self.task_name}
            )

        # store the formatted dataset
        self.formatted_ds = {
            "train": train_ds,
            "validation": val_ds,
        }

    def add_messages_key(
        self, example: dict[str, Any], task_name: str = "ResponseDataset"
    ) -> dict[str, str | list[dict[str, Any]]]:
        return {
            "messages": [
                {"role": "user", "content": example[self.input_key]},
                {"role": "assistant", "content": example[self.output_key]},
            ],
            "task_name": task_name,
        }
