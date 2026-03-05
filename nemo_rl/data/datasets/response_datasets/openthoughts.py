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
    """Wrapper around siyanzhao/Openthoughts_math_30k_opsd with train split."""

    def __init__(self, **kwargs) -> None:
        self.task_name = "OpenThoughts"

        split = kwargs.get("split", "train")
        download_dir = kwargs.get("download_dir", None)

        self.dataset = load_dataset(
            "siyanzhao/Openthoughts_math_30k_opsd",
            split=split,
            cache_dir=download_dir,
        )

        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        # Use 'problem' if available, fall back to 'Question'
        problem = data.get("problem") or data.get("Question", "")
        # Map new field names to internal keys used by the processor and self-distillation
        ground_truth_solution = data.get("Answer")
        deepseek_reasoning = data.get("COT_Reason")

        # inject teacher prompt creation here 

        

        assistant_answer = ground_truth_solution or ""

        return {
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": assistant_answer},
            ],
            "task_name": self.task_name,
            "problem": problem,
            "ground_truth_solution": ground_truth_solution,
            "deepseek_reasoning": deepseek_reasoning,
            "source": data.get("source"),
        }
