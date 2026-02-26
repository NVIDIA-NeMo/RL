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

from datasets import Dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class NemoGymDataset(RawDataset):
    """Simple wrapper around the Nemo Gym dataset.

    Args:
        data_path: Path to the dataset JSONL file
        repeat: Number of times to repeat the dataset, default is 1
    """

    def __init__(self, data_path: str, repeat: int = 1, **kwargs) -> None:
        # initialize common attributes (task name, prompt, system prompt, processor)
        default_task_name = self.extract_task_name_from_path(data_path)
        self.common_init(default_task_name=default_task_name, **kwargs)

        # load raw line from jsonl
        # will use `json.loads` to load to dict format at `nemo_gym_data_processor` later since `Dataset` cannot handle nested structure well
        with open(data_path) as f:
            self.dataset = [raw_line for raw_line in f]

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
