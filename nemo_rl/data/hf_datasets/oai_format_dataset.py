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

from nemo_rl.data.interfaces import TaskDataSpec


class OpenAIFormatDataset:
    """This class is used to load an SFT dataset in the OpenAI format.

    The dataset should be in the following format:
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
    }
    system_key and system_prompt are optional. If provided, it will be added to the
    beginning of the dataset.
    chat_key should be the key of the messages list. Multi-turn conversations are
    supported.
    The last message in the conversation must be from the assistant.
    """

    def __init__(
        self,
        train_ds_path: str,
        val_ds_path: str,
        chat_key: str = "messages",
        system_key: str | None = None,
        system_prompt: str | None = None,
        tool_key: str | None = "tools",
    ):
        self.chat_key = chat_key
        self.system_key = system_key
        self.system_prompt = system_prompt
        self.tool_key = tool_key
        
        try:
            # Try the original approach first (faster and more standard)
            train_original_dataset = load_dataset("json", data_files=train_ds_path)["train"]
            val_original_dataset = load_dataset("json", data_files=val_ds_path)["train"]

            formatted_train_dataset = train_original_dataset.map(self.add_messages_key)
            formatted_val_dataset = val_original_dataset.map(self.add_messages_key)
            
            print(f"Loaded dataset using standard approach (train: {len(formatted_train_dataset)}, val: {len(formatted_val_dataset)})")
            
        except (TypeError, ValueError, Exception) as e:
            # Fallback to streaming approach if schema issues occur
            # This handles cases with heterogeneous tool arguments (e.g., varying is_input field)
            print(f"Standard loading failed with {type(e).__name__}, falling back to streaming approach...")
            print(f"This typically happens with heterogeneous tool argument schemas.")
            
            # Use streaming=True to bypass schema validation
            train_original_dataset = load_dataset("json", data_files=train_ds_path, streaming=True)["train"]
            val_original_dataset = load_dataset("json", data_files=val_ds_path, streaming=True)["train"]

            # Map the streaming datasets
            formatted_train_dataset = train_original_dataset.map(self.add_messages_key)
            formatted_val_dataset = val_original_dataset.map(self.add_messages_key)
            
            # Convert to regular datasets
            from datasets import Dataset
            formatted_train_dataset = Dataset.from_list(list(formatted_train_dataset))
            formatted_val_dataset = Dataset.from_list(list(formatted_val_dataset))
            
            print(f"Loaded dataset using streaming approach (train: {len(formatted_train_dataset)}, val: {len(formatted_val_dataset)})")

        self.formatted_ds = {
            "train": formatted_train_dataset,
            "validation": formatted_val_dataset,
        }

        self.task_spec = TaskDataSpec(
            "json_dataset",
        )

    def add_messages_key(
        self,
        example: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        messages = [message for message in example[self.chat_key]]
        if self.system_key is not None and self.system_key in example:
            messages = [
                {"role": "system", "content": example[self.system_key]}
            ] + messages
        elif self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        assert messages[-1]["role"] == "assistant"
        
        # Preserve tools if they exist in the data
        result = {"messages": messages}
        if self.tool_key and self.tool_key in example:
            result["tools"] = example[self.tool_key]
        
        return result
