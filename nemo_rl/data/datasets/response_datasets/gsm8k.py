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

import os
from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


def _load_system_prompt(system_prompt_file: str | None) -> str:
    """Load system prompt from file. Returns empty string if path is None or missing."""
    if not system_prompt_file:
        return ""
    if os.path.exists(system_prompt_file):
        with open(system_prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    raise FileNotFoundError(f"System prompt file {system_prompt_file!r} not found.")


def _extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


class GSM8KDataset(RawDataset):
    """Simple wrapper around the GSM8K dataset.

    Args:
        split: Split name for the dataset, default is "train"
        extract_answer: Whether to extract the answer from the dataset, default is True
    """

    def __init__(self,
        split: str = "train",
        extract_answer: bool = True,
        system_prompt_file: str | None = None,
        **kwargs,
    ) -> None:
        self.task_name = "gsm8k"
        self.extract_answer = extract_answer
        self._system_prompt = _load_system_prompt(system_prompt_file)

        # load from huggingface
        self.dataset = load_dataset("openai/gsm8k", "main")[split]

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.extract_answer:
            answer = _extract_hash_answer(data["answer"])
        else:
            answer = data["answer"]

        return {
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": data["question"]},
                {"role": "assistant", "content": answer},
            ],
            "task_name": self.task_name,
        }
