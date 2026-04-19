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

import glob
from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class ArrowTextDataset(RawDataset):
    """Load ``.arrow`` files with a raw-text column for SFT training.

    Each row is wrapped as a single assistant turn
    (``{"messages": [{"role": "assistant", "content": text}]}``) so training
    runs language-modeling style on the whole sequence.

    Args:
        arrow_files: Glob pattern or explicit list of ``.arrow`` paths.
        text_key: Column in the arrow files that holds the text (default ``"text"``).
        split_validation_size: Fraction of rows reserved for validation
            (default 0.05; 0 means no split).
        seed: Seed for the train/validation split, default is 42.
    """

    def __init__(
        self,
        arrow_files: str | list[str],
        text_key: str = "text",
        split_validation_size: float = 0.05,
        seed: int = 42,
        **kwargs,
    ) -> None:
        self.task_name = "arrow_text"
        self.text_key = text_key

        # resolve glob pattern to a concrete file list
        if isinstance(arrow_files, str):
            file_list = sorted(glob.glob(arrow_files))
            if not file_list:
                raise ValueError(
                    f"No arrow files found matching pattern: {arrow_files}"
                )
        else:
            file_list = list(arrow_files)

        # load from local arrow files
        self.dataset = load_dataset("arrow", data_files=file_list, split="train")

        if self.text_key not in self.dataset.column_names:
            raise ValueError(
                f"Column '{self.text_key}' not found in arrow files. "
                f"Available columns: {self.dataset.column_names}"
            )
        print(f"Loaded {len(self.dataset)} samples from {len(file_list)} arrow files.")

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

        # `self.val_dataset` is used only when current dataset is used for both training and validation
        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [{"role": "assistant", "content": data[self.text_key]}],
            "task_name": self.task_name,
        }
