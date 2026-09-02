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

"""HMMT dataset.

HMMT stands for the Harvard-MIT Mathematics Tournament. It is an annual
mathematics competition for high school students, organized and staffed
by students at Harvard University and the Massachusetts Institute of
Technology (MIT).
"""

from typing import Any, Literal, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec

VERSION_MAPPINGS = {
    "2025_feb": "MathArena/hmmt_feb_2025",
    "2025_nov": "MathArena/hmmt_nov_2025",
    "2026_feb": "MathArena/hmmt_feb_2026",
}


class HmmtDataset:
    def __init__(
        self,
        variant: Literal["2025_feb", "2025_nov", "2026_feb"] = "2025_feb",
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        ds = load_dataset(VERSION_MAPPINGS[variant], split="train")
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name=f"hmmt{variant}",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.math_data_processor

    def _rekey(self, data: dict[str, Any]):
        return {
            "problem": data["problem"],
            "expected_answer": data["answer"],
        }
