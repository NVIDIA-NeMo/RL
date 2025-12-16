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
    def __init__(
        self,
        output_key: str = "expected_answer",
        split: str = "train_1M",
        split_validation_size: float = 0.05,
        seed: int = 42,
        **kwargs,
    ):
        """Initialize the OpenMathInstruct2 dataset with train/validation split.

        Args:
            seed: Random seed for reproducible splitting
            test_size: Proportion of data to use for validation (0.0-1.0)
        """
        # train, train_1M, train_2M, and train_5M are supported splits.
        if split not in ["train", "train_1M", "train_2M", "train_5M"]:
            raise ValueError(
                f"Invalid split: {split}. Please use 'train', 'train_1M', 'train_2M', or 'train_5M'."
            )

        self.input_key = "problem"
        self.output_key = output_key
        self.task_name = "OpenMathInstruct-2"

        # load from local or huggingface
        self.dataset = load_dataset("nvidia/OpenMathInstruct-2", split=split)

        # format the dataset
        self.dataset = self.dataset.map(
            self.add_messages_key,
            remove_columns=self.dataset.column_names,
        )

        # use only when current dataset is used for both training and validation
        self.val_dataset = None
        if split_validation_size > 0:
            split_dataset = self.dataset.train_test_split(
                test_size=split_validation_size, seed=seed
            )
            self.dataset = split_dataset["train"]
            self.val_dataset = split_dataset["test"]

    def add_messages_key(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": data[self.input_key]},
                {"role": "assistant", "content": data[self.output_key]},
            ],
            "task_name": self.task_name,
        }
