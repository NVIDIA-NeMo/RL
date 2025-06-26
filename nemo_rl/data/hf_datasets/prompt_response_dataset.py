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


class PromptResponseDataset:
    def __init__(
        self,
        train_ds_path: str,
        val_ds_path: str,
        input_key: str = "input",
        output_key: str = "output",
    ):
        train_original_dataset = load_dataset("json", data_files=train_ds_path)["train"]
        val_original_dataset = load_dataset("json", data_files=val_ds_path)["train"]

        self.input_key = input_key
        self.output_key = output_key

        formatted_train_dataset = train_original_dataset.map(self.add_messages_key)
        formatted_val_dataset = val_original_dataset.map(self.add_messages_key)

        self.formatted_ds = {
            "train": formatted_train_dataset,
            "validation": formatted_val_dataset,
        }

        self.task_spec = TaskDataSpec(
            "json_dataset",
        )

    def add_messages_key(
        self, example: dict[str, Any]
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            "messages": [
                {"role": "user", "content": example[self.input_key]},
                {"role": "assistant", "content": example[self.output_key]},
            ]
        }
