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
from typing import Optional

from datasets import Dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class NemoGymDataset(RawDataset):
    """Simple wrapper around the Nemo Gym dataset."""

    def __init__(
        self, data_path: Optional[str] = None, repeat: int = 1, **kwargs
    ) -> None:
        self.task_name = data_path.split("/")[-1].split(".")[0]

        # load from jsonl
        with open(data_path) as f:
            self.dataset = list(map(json.loads, f))

        # format the dataset
        self.dataset = Dataset.from_dict(
            {
                "extra_env_info": self.dataset,
                "task_name": [self.task_name] * len(self.dataset),
            }
        )

        # repeat the dataset
        if repeat > 1:
            self.dataset = self.dataset.repeat(repeat)
