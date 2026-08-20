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

"""MMLU response dataset and its language variants."""

from typing import Any, Literal

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset

MMLULanguage = Literal[
    "AR-XY",
    "BN-BD",
    "DE-DE",
    "EN-US",
    "ES-LA",
    "FR-FR",
    "HI-IN",
    "ID-ID",
    "IT-IT",
    "JA-JP",
    "KO-KR",
    "PT-BR",
    "ZH-CN",
    "SW-KE",
    "YO-NG",
]


class MMLUDataset(RawDataset):
    def __init__(self, language: MMLULanguage = "EN-US", **kwargs):
        self.task_name = f"MMLU_{language}"

        if language == "EN-US":
            data_files = (
                "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
            )
        else:
            data_files = f"https://openaipublic.blob.core.windows.net/simple-evals/mmlu_{language}.csv"

        dataset = load_dataset("csv", data_files=data_files, split="train")
        self.dataset = dataset.map(
            self._rekey,
            remove_columns=dataset.column_names,
        )
        self.val_dataset = None

    def _rekey(self, data: dict[str, Any]) -> dict[str, Any]:
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
            "task_name": self.task_name,
        }
