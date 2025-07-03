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

"""MMLU dataset and its variants."""

from typing import Any, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class MMLUDataset:
    def __init__(
        self,
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        ds = load_dataset(
            "csv",
            data_files="https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv",
            split="train",
        )
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)

        self.task_spec = TaskDataSpec(
            task_name="MMLU",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.multichoice_qa_processor

    def _rekey(self, data: dict[str, Any]):
        return {
            "question": data["Question"],
            "options": dict(
                A=data["A"],
                B=data["B"],
                C=data["C"],
                D=data["D"],
            ),
            "answer": data["Answer"],
            "subject": data["Subject"],
        }
