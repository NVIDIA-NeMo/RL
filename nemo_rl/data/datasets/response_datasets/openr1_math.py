# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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


class OpenR1Math220KDataset(RawDataset):
    """Simple wrapper around the OpenR1-Math-220k dataset.

    Args:
        subset: Hugging Face dataset config (subset) name, default is "default"
        split: Split name for the dataset, default is "train"
        split_validation_size: Size of the validation data, default is 0
        seed: Seed for train/validation split when split_validation_size > 0, default is 42
    """

    def __init__(
        self,
        subset: str = "default",
        split: str = "train",
        split_validation_size: float = 0.0,
        seed: int = 42,
        **kwargs,
    ) -> None:
        self.task_name = "OpenR1-Math-220k"

        # load from huggingface
        self.dataset = load_dataset("open-r1/OpenR1-Math-220k", subset, split=split)

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
                {"role": "user", "content": data["problem"]},
                {"role": "assistant", "content": data["answer"]},
            ],
            "task_name": self.task_name,
        }
