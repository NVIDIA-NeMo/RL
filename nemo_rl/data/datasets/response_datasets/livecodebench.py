# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""LiveCodeBench dataset for GRPO training with code generation.

Loads problems from the LiveCodeBench benchmark and formats them for
GRPO training with the CodeTestCaseEnvironment. Each sample includes
the problem statement and public test cases for reward computation.
"""

import json
from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset

VARIANT_TO_FILE = {
    "release_v1": "test.jsonl",
    "release_v2": "test2.jsonl",
    "release_v3": "test3.jsonl",
    "release_v4": "test4.jsonl",
    "release_v5": "test5.jsonl",
    "release_v6": "test6.jsonl",
    "release_latest": "test6.jsonl",
}


class LiveCodeBenchDataset(RawDataset):
    """LiveCodeBench dataset for code generation GRPO training.

    Loads coding problems from HuggingFace livecodebench/code_generation_lite
    and formats them with problem text and public test cases. Compatible with
    code_data_processor which reads 'problem' and 'test_cases' keys.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.task_name = "LiveCodeBench"
        self.val_dataset = None

        variant = kwargs.get("variant", "release_v5")
        data_file = VARIANT_TO_FILE.get(variant, "test5.jsonl")
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

        self.dataset = ds.map(
            self.format_data,
            remove_columns=ds.column_names,
        )

        split_validation_size = kwargs.get("split_validation_size", 0)
        seed = kwargs.get("seed", 42)
        self.split_train_validation(split_validation_size, seed)

    def _parse_test_cases(self, raw: Any) -> list[dict[str, str]]:
        """Parse public_test_cases into a normalized list of input/output dicts."""
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return []
        if not isinstance(raw, list):
            return []

        test_cases: list[dict[str, str]] = []
        for tc in raw:
            if isinstance(tc, dict):
                test_cases.append({
                    "input": str(tc.get("input", "")),
                    "expected_output": str(tc.get("output", "")),
                })
        return test_cases

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Format a LiveCodeBench example for code_data_processor consumption.

        Returns a dict with 'messages' (HF chat format for GRPO compatibility),
        'problem', 'expected_answer', and 'test_cases' keys.
        """
        test_cases = self._parse_test_cases(data.get("public_test_cases", []))

        starter = data.get("starter_code", "")
        problem = data["question_content"]
        if starter:
            problem = f"{problem}\n\nStarter code:\n```python\n{starter}\n```"

        return {
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": ""},
            ],
            "task_name": self.task_name,
            "problem": problem,
            "expected_answer": "",
            "test_cases": json.dumps(test_cases),
        }
