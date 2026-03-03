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

"""LiveCodeBench dataset for GRPO training with code generation."""

import json
from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class LiveCodeBenchDataset(RawDataset):
    """LiveCodeBench dataset for code generation GRPO training."""

    def __init__(self, **kwargs) -> None:
        self.task_name = "LiveCodeBench"
        self.val_dataset = None

        variant = kwargs.get("variant", "release_v5")
        variant_to_file = {
            "release_v1": "test.jsonl",
            "release_v2": "test2.jsonl",
            "release_v3": "test3.jsonl",
            "release_v4": "test4.jsonl",
            "release_v5": "test5.jsonl",
            "release_v6": "test6.jsonl",
            "release_latest": "test6.jsonl",
        }
        data_file = variant_to_file.get(variant, "test5.jsonl")
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

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
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
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": json.dumps(test_cases)},
            ],
            "task_name": self.task_name,
            "test_cases": json.dumps(test_cases),
        }
