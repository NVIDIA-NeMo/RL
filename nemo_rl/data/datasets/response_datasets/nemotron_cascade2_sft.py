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


class NemotronCascade2SFTMathDataset(RawDataset):
    """Simple wrapper around the Nemotron-Cascade-2-SFT-Data math split.

    Loads the ``math`` subset of ``nvidia/Nemotron-Cascade-2-SFT-Data`` from
    HuggingFace.  Each example already contains a ``messages`` field in
    OpenAI chat format (system / user / assistant turns), so no heavy
    reformatting is needed.

    Args:
        split: HuggingFace dataset split to load, default is "train"
        split_validation_size: Fraction of data held out for validation when
            no dedicated validation split exists, default is 0.05
        seed: Random seed used when creating the train/validation split,
            default is 42
        max_samples: If set, randomly sample this many examples from the
            dataset before any train/validation split, default is None (use all)
    """

    def __init__(
        self,
        split: str = "train",
        split_validation_size: float = 0.05,
        seed: int = 42,
        max_samples: int | None = None,
        **kwargs,
    ) -> None:
        self.task_name = "Nemotron-Cascade-2-SFT-Math"

        self.dataset = load_dataset(
            "nvidia/Nemotron-Cascade-2-SFT-Data",
            "math",
            split=split,
        )

        if max_samples is not None and max_samples > 0:
            self.dataset = self.dataset.shuffle(seed=seed).select(
                range(min(max_samples, len(self.dataset)))
            )

        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        messages = data["messages"]

        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError(
                f"Expected last message to be from assistant, got: {messages}"
            )

        return {
            "messages": messages,
            "task_name": self.task_name,
        }
