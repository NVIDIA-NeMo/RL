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
from typing import Optional

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import load_dataset_from_path


class PreferenceDataset(RawDataset):
    """Dataset class for preference data which can be loaded from a JSON file.

    This class handles loading of preference data for DPO and RM training.
    The input JSONL files should contain valid JSON objects formatted like this:
    {
        "context": list of dicts, # The prompt message (including previous turns, if any)
        "completions": list of dicts, # The list of completions
            {
                "rank": int, # The rank of the completion (lower rank is preferred)
                "completion": list of dicts, # The completion message(s)
            }
    }

    Args:
        data_path: Path to the dataset JSON file
        split: Optional split name for the dataset, used for HuggingFace datasets
    """

    def __init__(
        self,
        data_path: str,
        split: Optional[str] = None,
        **kwargs,
    ):
        self.task_name = data_path.split("/")[-1].split(".")[0]

        # load from local or huggingface
        self.dataset = load_dataset_from_path(data_path, split)

        # format the dataset
        self.dataset = self.dataset.add_column(
            "task_name", [self.task_name] * len(self.dataset)
        )
