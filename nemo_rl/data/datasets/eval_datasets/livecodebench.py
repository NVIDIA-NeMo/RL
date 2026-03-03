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

"""LiveCodeBench dataset for code generation evaluation."""

import json
from typing import Any, Literal, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec

VARIANT_TO_FILE = {
    "release_v1": "test.jsonl",
    "release_v2": "test2.jsonl",
    "release_v3": "test3.jsonl",
    "release_v4": "test4.jsonl",
    "release_v5": "test5.jsonl",
    "release_v6": "test6.jsonl",
    "release_latest": "test6.jsonl",
}


class LiveCodeBenchDataset:
    def __init__(
        self,
        variant: Literal["release_v5", "release_v4", "release_v3", "release_latest"] = "release_v5",
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        data_file = VARIANT_TO_FILE.get(variant, f"test{variant.replace('release_v', '')}.jsonl")
        try:
            ds = load_dataset(
                "livecodebench/code_generation_lite",
                variant,
                split="test",
                trust_remote_code=True,
            )
        except (RuntimeError, ValueError):
            ds = load_dataset(
                "json",
                data_files=f"hf://datasets/livecodebench/code_generation_lite/{data_file}",
                split="train",
            )
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name="livecodebench",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.code_data_processor

    def _rekey(self, data: dict[str, Any]) -> dict[str, Any]:
        public_tests = data.get("public_test_cases", [])
        if isinstance(public_tests, str):
            try:
                public_tests = json.loads(public_tests)
            except (json.JSONDecodeError, TypeError):
                public_tests = []

        test_cases = []
        for tc in public_tests:
            test_cases.append({
                "input": tc.get("input", ""),
                "expected_output": tc.get("output", ""),
            })

        starter = data.get("starter_code", "")
        problem = data["question_content"]
        if starter:
            problem = f"{problem}\n\nStarter code:\n```python\n{starter}\n```"

        return {
            "problem": problem,
            "expected_answer": "",
            "test_cases": json.dumps(test_cases),
        }
