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

import json
from typing import Any

import reasoning_gym
from datasets import Dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


def _entry_get(entry: Any, key: str) -> Any:
    """Access a reasoning_gym entry by key, tolerating dict-like or attribute access."""
    try:
        return entry[key]
    except (TypeError, KeyError):
        return getattr(entry, key)


class ReasoningGymDataset(RawDataset):
    """Generic wrapper around reasoning_gym tasks (e.g. countdown, sudoku, mini_sudoku).

    Args:
        task_name: reasoning_gym task name, also used as the NeMo-RL task name.
        size: Number of procedurally generated samples.
        seed: Seed for the reasoning_gym generator.
    """

    def __init__(
        self,
        task_name: str,
        size: int = 5000,
        seed: int = 42,
        system_prompt_file: str | None = None,
        prompt_style: str | None = None,
        **kwargs,
    ) -> None:
        self.task_name = task_name
        # "agrpo_countdown": build the user prompt as "Target: T\nNumbers: [...]"
        # (AGRPO format) instead of the reasoning_gym question text.
        self.prompt_style = prompt_style

        rg = reasoning_gym.create_dataset(task_name, size=size, seed=seed)
        # The rg_entry is stored as a JSON string column because HF Datasets do not
        # round-trip nested dicts cleanly (mirrors nemo_gym_data_processor).
        rows = []
        for e in rg:
            q = _entry_get(e, "question")
            rows.append(
                {
                    "question": q,
                    "rg_entry": json.dumps(
                        {
                            "question": q,
                            "answer": _entry_get(e, "answer"),
                            "metadata": dict(_entry_get(e, "metadata")),
                        }
                    ),
                }
            )

        self.dataset = Dataset.from_list(rows)
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.prompt_style == "agrpo_countdown":
            meta = json.loads(data["rg_entry"])["metadata"]
            content = f"Target: {meta['target']}\nNumbers: {meta['numbers']}"
        else:
            content = data["question"]
        return {
            "messages": [{"role": "user", "content": content}],
            "task_name": self.task_name,
            "rg_entry": data["rg_entry"],
        }
