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

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class OpenThoughtsDataset(RawDataset):
    """Wrapper around OpenThoughts-114k with train split."""

    def __init__(self, **kwargs) -> None:
        self.task_name = "OpenThoughts"

        split = kwargs.get("split", "train")
        download_dir = kwargs.get("download_dir", None)

        self.dataset = load_dataset(
            "open-thoughts/OpenThoughts-114k",
            "metadata",
            split=split,
            
            cache_dir=download_dir,
        )



        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
            #load_from_cache_file = False,
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        #print(data.keys())
        assistant_answer = (
            data.get("ground_truth_solution")
            or data.get("deepseek_solution")
            or ""
        )

        #print(assistant_answer)

        return {
            "messages": [
                {"role": "user", "content": data.get("problem", "")},
                {"role": "assistant", "content": assistant_answer},
            ],
            "task_name": self.task_name,
            "problem": data.get("problem"),
            "ground_truth_solution": data.get("ground_truth_solution"),
            "deepseek_reasoning": data.get("deepseek_reasoning"),
            "deepseek_solution": data.get("deepseek_solution"),
            "domain": data.get("domain"),
            "source": data.get("source"),
            "test_cases": data.get("test_cases"),
            "starter_code": data.get("starter_code"),
        }
