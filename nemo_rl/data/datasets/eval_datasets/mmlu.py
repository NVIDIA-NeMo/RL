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

from collections import defaultdict
from typing import Any, Literal, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec

ANSWER_INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


class MMLUDataset:
    def __init__(
        self,
        language: Literal[
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
        ] = "EN-US",
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
        num_few_shot: int = 0,
    ):
        if language != "EN-US":
            data_files = f"https://openaipublic.blob.core.windows.net/simple-evals/mmlu_{language}.csv"
        else:
            data_files = (
                "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
            )
        ds = load_dataset(
            "csv",
            data_files=data_files,
            split="train",
        )
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)

        if num_few_shot > 0:
            few_shot_prefixes = self._build_few_shot_prefixes(num_few_shot)
            self.rekeyed_ds = self.rekeyed_ds.map(
                lambda ex: {
                    "few_shot_prefix": few_shot_prefixes.get(ex["subject"], "")
                }
            )

        self.task_spec = TaskDataSpec(
            task_name=f"MMLU_{language}",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.multichoice_qa_processor

    @staticmethod
    def _build_few_shot_prefixes(num_few_shot: int) -> dict[str, str]:
        """Build per-subject few-shot prefixes from MMLU's dev (validation) split."""
        dev_ds = load_dataset("cais/mmlu", "all", split="validation")

        dev_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ex in dev_ds:
            dev_by_subject[ex["subject"]].append(ex)

        prefixes: dict[str, str] = {}
        for subject, examples in dev_by_subject.items():
            parts = []
            for fs_ex in examples[:num_few_shot]:
                choices = fs_ex["choices"]
                options_str = "\n".join(
                    f"{letter}) {choices[i]}"
                    for i, letter in enumerate(["A", "B", "C", "D"])
                )
                answer_letter = ANSWER_INDEX_TO_LETTER[fs_ex["answer"]]
                parts.append(
                    f"Question: {fs_ex['question']}\nOptions:\n{options_str}\n"
                    f"Answer: {answer_letter}"
                )
            prefixes[subject] = "\n\n".join(parts)

        return prefixes

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
