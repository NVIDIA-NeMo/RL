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


class Tulu3SftMixtureDataset(RawDataset):
    """Simple wrapper around the Tulu3 SFT mixture dataset with train split.

    Args:
        split_validation_size: Size of the validation data, default is 0.05
        seed: Seed for train/validation split when split_validation_size > 0, default is 42
        max_samples: Optional maximum number of samples to use from the dataset
    """

    def __init__(
        self,
        split_validation_size: float = 0.05,
        seed: int = 42,
        max_samples: int | None = None,
        **kwargs,
    ) -> None:
        print(
            "WARNING: For reproducible experiments, preprocess the dataset once and define your own HfDataset subclass that directly uses the preprocessed datasets."
        )

        self.task_name = "tulu3_sft_mixture"

        # load from huggingface
        self.dataset = load_dataset("allenai/tulu-3-sft-mixture")["train"]

        # Optionally limit the number of samples
        if max_samples is not None and max_samples > 0:
            self.dataset = self.dataset.shuffle(seed=seed).select(
                range(min(max_samples, len(self.dataset)))
            )

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=["id", "source"],
        )

        # `self.val_dataset` is used (not None) only when current dataset is used for both training and validation
        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        messages = data["messages"]

        # Ensure last message is from assistant
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError(
                f"Expected last message to be from assistant, got: {messages}"
            )

        return {"task_name": self.task_name}
