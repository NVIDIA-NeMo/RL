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

from datasets import Dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class NemoGymDataset(RawDataset):
    """Simple wrapper around the Nemo Gym dataset.

    Args:
        data_path: Path to the dataset JSONL file
        repeat: Number of times to repeat the dataset, default is 1
    """

    def __init__(self, data_path: str, repeat: int = 1, **kwargs) -> None:
        self.task_name = "-".join(data_path.split("/")[-2:]).split(".")[0]
        if self.task_name[0] == "-":
            self.task_name = self.task_name[1:]

        # Keep raw lines because Dataset cannot reliably represent the nested Gym rows.
        # Cache the distinct agents while each source row is read once.
        # Repeating a dataset must not multiply this setup work.
        with open(data_path) as f:
            raw_rows = [raw_line for raw_line in f]
        self.agent_names = frozenset(
            agent_name
            for raw_row in raw_rows
            if (agent_name := _get_agent_name(json.loads(raw_row))) is not None
        )

        # format the dataset
        self.dataset = Dataset.from_dict(
            {
                "extra_env_info": raw_rows,
                "task_name": [self.task_name] * len(raw_rows),
            }
        )

        # repeat the dataset
        if repeat > 1:
            self.dataset = self.dataset.repeat(repeat)


def _get_agent_name(row: object) -> str | None:
    if not isinstance(row, dict):
        return None
    agent_ref = row.get("agent_ref")
    if not isinstance(agent_ref, dict) or "name" not in agent_ref:
        return None
    return str(agent_ref["name"])
