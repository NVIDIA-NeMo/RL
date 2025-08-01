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
from typing import Any

from absl import logging
from collections import defaultdict
from datasets import Dataset, DatasetDict, load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


class PreferenceDataset:
    """Preference dataset. 

    This class handles loading of custom preference data.
    The input JSONL file should contain valid JSON objects formatted like this:
    {
        "context": list of dicts, # The prompt message (including previous turns, if any)
        "completions": list of dicts, # The list of completions
            {
                "rank": int, # The rank of the completion (lower rank is preferred)
                "completion": list of dicts, # The completion message(s)
            }
    }
    """

    def __init__(self, dataset_path: str, split: str) -> None:
        # Specifying split="train" returns Dataset instead of DatasetDict({"train": Dataset})
        self.formatted_ds = DatasetDict({split: load_dataset("json", data_files=dataset_path, split="train")})

        self.task_spec = TaskDataSpec(
            task_name="PreferenceData",
        )