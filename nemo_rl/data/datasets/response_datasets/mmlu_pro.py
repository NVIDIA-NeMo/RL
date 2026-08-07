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

"""MMLU-Pro response dataset."""

from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class MMLUProDataset(RawDataset):
    default_processor = "multichoice_qa_processor"

    def __init__(self, **kwargs):
        self.task_name = "MMLU-Pro"

        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        self.dataset = dataset.map(
            self._rekey,
            remove_columns=dataset.column_names,
        )
        self.val_dataset = None

    def _rekey(self, data: dict[str, Any]) -> dict[str, Any]:
        options = {
            chr(ord("A") + i): option for i, option in enumerate(data["options"])
        }
        return {
            "question": data["question"],
            "options": options,
            "answer": data["answer"],
            "subject": data["category"],
            "task_name": self.task_name,
        }
