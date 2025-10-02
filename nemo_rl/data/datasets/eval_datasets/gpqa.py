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

"""GPQA dataset and its variants."""

import random
from typing import Any, Literal, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class GPQADataset:
    def __init__(
        self,
        variant: Literal["diamond", "main"] = "diamond",
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        ds = load_dataset("Idavidrein/gpqa", f"gpqa_{variant}", split="train")
        self._rng = random.Random()
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name=f"GPQA_{variant}",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.multichoice_qa_processor

    def _rekey(self, data: dict[str, Any]):
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
        }
