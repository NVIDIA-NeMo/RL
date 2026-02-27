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


class OpenMathInstruct2Dataset(RawDataset):
    """Simple wrapper around the OpenMathInstruct2 dataset.

    Args:
        output_key: Key for the output text, default is "expected_answer"
        split: Split name for the dataset, default is "train_1M"
        split_validation_size: Size of the validation data, default is 0.05
        seed: Seed for train/validation split when split_validation_size > 0, default is 42
    """

    def __init__(
        self,
        output_key: str = "expected_answer",
        split: str = "train_1M",
        split_validation_size: float = 0.05,
        seed: int = 42,
        **kwargs,
    ):
        # train, train_1M, train_2M, and train_5M are supported splits.
        if split not in ["train", "train_1M", "train_2M", "train_5M"]:
            raise ValueError(
                f"Invalid split: {split}. Please use 'train', 'train_1M', 'train_2M', or 'train_5M'."
            )

        self.input_key = "problem"
        self.output_key = output_key
        self.task_name = "OpenMathInstruct-2"

        # load from huggingface
        self.dataset = load_dataset("nvidia/OpenMathInstruct-2", split=split)

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
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
