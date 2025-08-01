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

"""AIME 2025 dataset."""

from typing import Any, Optional

from datasets import concatenate_datasets, load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class AIME2025Dataset:
    def __init__(
        self,
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        ds0 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
        ds1 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
        ds = concatenate_datasets([ds0, ds1])
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name="aime2025",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.math_data_processor

    def _rekey(self, data: dict[str, Any]):
        return {
            "problem": data["question"],
            "expected_answer": data["answer"],
        }
