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
    """Simple wrapper around the GSM8K dataset with train and validation splits.

    Args:
        seed: Random seed for shuffling the training set (default 42).
        system_prompt_file: Optional path to a text file containing the system prompt
            (e.g. examples/prompts/gsm8k.txt). If not provided, system prompt is empty.
    """

    def __init__(
        self,
        seed: int = 42,
        system_prompt_file: str | None = None,
        **kwargs,
    ) -> None:
        self.task_name = "gsm8k"
        self._system_prompt = _load_system_prompt(system_prompt_file)

        # Load from HuggingFace
        train_ds = load_dataset("openai/gsm8k", "main")["train"]
        val_ds = load_dataset("openai/gsm8k", "main")["test"]

        # Shuffle training with seed
        train_ds = train_ds.shuffle(seed=seed)

        # Format the datasets
        self.dataset = train_ds.map(
            self.format_data,
            remove_columns=train_ds.column_names,
        )
        self.val_dataset = val_ds.map(
            self.format_data,
            remove_columns=val_ds.column_names,
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": data["question"]},
                {"role": "assistant", "content": _extract_hash_answer(data["answer"])},
            ],
            "task_name": self.task_name,
        }
