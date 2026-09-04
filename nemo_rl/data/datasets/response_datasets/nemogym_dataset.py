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

from datasets import Dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.interfaces import NemoGymSourceIdentity


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
        # Record a stable source identity without parsing rows on the unsharded path.
        source_path = os.path.realpath(data_path)
        source_stat = os.stat(source_path)
        source_identity = NemoGymSourceIdentity.from_stat(source_path, source_stat)
        with open(source_path) as f:
            raw_rows = [raw_line for raw_line in f]
        source_stat_after_read = os.stat(source_path)
        if source_identity.matches(source_stat_after_read):
            self.agent_name_sources = frozenset({source_identity})
        else:
            self.agent_name_sources = None

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
