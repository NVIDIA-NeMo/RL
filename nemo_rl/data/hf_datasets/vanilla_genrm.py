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

from typing import Any, Optional

from datasets import load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


class VanillaGenRMDataset:
    """Dataset class for vanilla GenRM evaluation data.

    This class handles loading of data for GenRM training where the model
    is trained to evaluate and score two responses to a question.

    The input JSONL files should contain valid JSON objects formatted like this:
    {
        "messages": [[{role, content}, ...]],  # List of message lists
        "metadata": {                           # Metadata with ground truth scores
            "score_1": int,
            "score_2": int,
            "ranking": int,
            "question_id": str,
            "question": str
        },
        "task_name": str
    }

    Args:
        train_data_path: Path to the JSON file containing training data
        val_data_path: Optional path to the JSON file containing validation data
        task_name: Name of the task (default: "vanilla_genrm")
    """

    def __init__(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        task_name: str = "vanilla_genrm",
    ):
        self.task_name = task_name

        # Load from json file
        train_ds = load_dataset("json", data_files=train_data_path)["train"]
        val_ds = None
        if val_data_path:
            val_ds = load_dataset("json", data_files=val_data_path)["train"]

        # The data is already in the correct format, just ensure task_name is set
        train_ds = train_ds.map(lambda x: self._ensure_format(x))
        if val_ds:
            val_ds = val_ds.map(lambda x: self._ensure_format(x))

        # Store the formatted dataset
        self.formatted_ds = {
            "train": train_ds,
            "validation": val_ds,
        }

        self.task_spec = TaskDataSpec(task_name=self.task_name)

    def _ensure_format(self, example: dict[str, Any]) -> dict[str, Any]:
        """Ensure the example has the required format and fields."""
        # Make sure task_name is set
        if "task_name" not in example:
            example["task_name"] = self.task_name

        # Ensure messages is a list of lists
        if "messages" in example:
            messages = example["messages"]
            # If messages is a single list of dicts, wrap it
            if messages and isinstance(messages[0], dict):
                example["messages"] = [messages]

        # Ensure metadata exists
        if "metadata" not in example:
            example["metadata"] = {}

        return example
