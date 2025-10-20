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


class GenRMRLHFDataset:
    """Dataset class for GenRM RLHF training data.

    This class handles loading of data for GenRM RLHF training where the model
    generates multiple responses per prompt and uses GenRM for pairwise comparisons.

    The input JSONL files should contain valid JSON objects formatted like this:
    {
        "messages": [[{role, content, metadata}, ...]],  # List of message lists (full conversations)
        "task_name": str,
        "dataset": str  # Optional dataset identifier
    }

    Each message list is a complete conversation with multiple user/assistant turns.
    The last user message MUST contain metadata with:
    - conversation_history: Previous conversation context (required, can be empty list [])

    Args:
        train_data_path: Path to the JSON file containing training data
        val_data_path: Optional path to the JSON file containing validation data
        task_name: Name of the task (default: "genrm_rlhf")
    """

    def __init__(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        task_name: str = "genrm_rlhf",
    ):
        self.task_name = task_name

        # Load from json file
        train_ds = load_dataset("json", data_files=train_data_path)["train"]
        val_ds = None
        if val_data_path:
            val_ds = load_dataset("json", data_files=val_data_path)["train"]

        # Process the data to extract conversation history from metadata
        train_ds = train_ds.map(lambda x: self._process_format(x))
        if val_ds:
            val_ds = val_ds.map(lambda x: self._process_format(x))

        # Store the formatted dataset
        self.formatted_ds = {
            "train": train_ds,
            "validation": val_ds,
        }

        self.task_spec = TaskDataSpec(task_name=self.task_name)

    def _process_format(self, example: dict[str, Any]) -> dict[str, Any]:
        """Ensure the example has the required format.

        Note: Metadata is expected to be in the last user message of each conversation.
        The data processor will extract it from there.
        """
        # Make sure task_name is set
        if "task_name" not in example:
            example["task_name"] = self.task_name

        # Ensure messages is a list of lists
        if "messages" in example:
            messages = example["messages"]
            # If messages is a single list of dicts, wrap it
            if messages and isinstance(messages[0], dict):
                example["messages"] = [messages]

        return example