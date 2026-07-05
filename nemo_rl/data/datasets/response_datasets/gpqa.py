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

"""GPQA response dataset and its variants."""

import random
from typing import Any, Literal

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class GPQADataset(RawDataset):
    default_processor = "multichoice_qa_processor"

    def __init__(
        self,
        variant: Literal["diamond", "main"] = "diamond",
        seed: int | None = None,
        **kwargs,
    ):
        self.task_name = f"GPQA_{variant}"
        self._rng = random.Random(seed)

        dataset = load_dataset("Idavidrein/gpqa", f"gpqa_{variant}", split="train")
        self.dataset = dataset.map(
            self._rekey,
            remove_columns=dataset.column_names,
        )
        self.val_dataset = None

    def _rekey(self, data: dict[str, Any]) -> dict[str, Any]:
        choices = [
            data["Correct Answer"],
            data["Incorrect Answer 1"],
            data["Incorrect Answer 2"],
            data["Incorrect Answer 3"],
        ]
        permutation = self._rng.sample(range(4), 4)
        choices = [choices[i] for i in permutation]
        correct_index = choices.index(data["Correct Answer"])
        correct_answer = "ABCD"[correct_index]
        return {
            "question": data["Question"],
            "options": dict(
                A=choices[0],
                B=choices[1],
                C=choices[2],
                D=choices[3],
            ),
            "answer": correct_answer,
            "task_name": self.task_name,
        }
