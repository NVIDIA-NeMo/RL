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

"""MATH response dataset and its variants."""

from typing import Any, Literal

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class MathDataset(RawDataset):
    def __init__(
        self,
        variant: Literal["math_test", "math_500_test"] = "math_test",
        **kwargs,
    ):
        self.task_name = variant

        dataset = load_dataset(
            "csv",
            data_files=f"https://openaipublic.blob.core.windows.net/simple-evals/{variant}.csv",
            split="train",
        )
        self.dataset = dataset.map(
            self._rekey,
            remove_columns=dataset.column_names,
        )
        self.val_dataset = None

    def _rekey(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "problem": data["Question"],
            "expected_answer": data["Answer"],
            "task_name": self.task_name,
        }
