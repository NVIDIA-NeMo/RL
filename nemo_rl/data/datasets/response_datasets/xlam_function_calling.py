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

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class XLAMFunctionCallingDataset(RawDataset):
    """Wrapper around the Salesforce/xlam-function-calling-60k dataset.

    Each sample contains a user query, tool definitions, and gold function-call answers.

    Args:
        split: Split name for the dataset, default is "train"
        split_validation_size: Size of the validation data, default is 0.05
        seed: Seed for train/validation split, default is 42
    """

    def __init__(
        self,
        split: str = "train",
        split_validation_size: float = 0.05,
        seed: int = 42,
        **kwargs,
    ):
        self.task_name = "xlam_function_calling"

        self.dataset = load_dataset("Salesforce/xlam-function-calling-60k", split=split)

        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        # tools and answers are JSON strings in this dataset
        tools = data.get("tools", "[]")
        if isinstance(tools, str):
            tools = json.loads(tools)

        answers = data.get("answers", "[]")
        if isinstance(answers, str):
            answers = json.loads(answers)

        return {
            "query": data["query"],
            "tools": json.dumps(tools),
            "gold_answers": json.dumps(answers),
            "task_name": self.task_name,
        }
