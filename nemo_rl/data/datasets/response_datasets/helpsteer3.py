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
from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class HelpSteer3Dataset(RawDataset):
    """Simple wrapper around the HelpSteer3 dataset with preference subset.

    Args:
        split: Split name for the dataset, default is "train"
    """

    def __init__(self, split: str = "train", **kwargs):
        self.task_name = "HelpSteer3"

        # load from huggingface
        self.dataset = load_dataset("nvidia/HelpSteer3", "preference")[split]

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        response_1 = data["response1"]
        response_2 = data["response2"]
        overall_preference = data["overall_preference"]

        if overall_preference < 0:
            chosen = response_1
        elif overall_preference == 0:
            logging.log_every_n(
                logging.WARNING,
                "Preference is 0 for some examples! Setting chosen and rejected to response 1 since we don't know which response is better",
                1000,
            )
            chosen = response_1
        else:
            chosen = response_2

        if isinstance(data["context"], str):
            context = [{"role": "user", "content": data["context"]}]
        else:
            context = data["context"]

        return {
            "context": context,
            "response": [{"role": "assistant", "content": chosen}],
            "task_name": self.task_name,
        }
