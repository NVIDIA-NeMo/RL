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

import json
from typing import Any, Union

from datasets import load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


def format_dpo_chatml(data: dict[str, Any]) -> dict[str, str | dict[str, str]]:
    """Format DPO ChatML data similar to Tulu3 preference formatting.
    
    Expects input data with 'chosen' and 'rejected' conversation lists,
    where each conversation follows ChatML format with role/content structure.
    """
    chosen_conversation = data["chosen"]
    rejected_conversation = data["rejected"]

    context = chosen_conversation[:-1]

    # We assume that except last assistant response, all messages in
    # chosen and rejected conversations are similar. Validating this...
    assert json.dumps(context, ensure_ascii=False) == json.dumps(
        rejected_conversation[:-1], ensure_ascii=False
    ), (
        f"Context mismatch.\n\nchosen: {chosen_conversation}\n\n rejected: {rejected_conversation}"
    )

    # We assume that last response is always from the assistant. Validating this...
    assert chosen_conversation[-1]["role"] == "assistant", (
        f"The last chosen response ({chosen_conversation[-1]}) is not from assistant!"
    )
    assert rejected_conversation[-1]["role"] == "assistant", (
        f"The last rejected response ({rejected_conversation[-1]}) is not from assistant!"
    )

    chosen_response = chosen_conversation[-1]["content"]
    rejected_response = rejected_conversation[-1]["content"]

    return {
        "prompt": context,
        "chosen_response": chosen_response,
        "rejected_response": rejected_response,
    }


class DPOChatMLDataset:
    """Dataset class for Direct Preference Optimization (DPO) training with ChatML format.

    This class handles loading of preference data for DPO training from custom file paths.
    The input JSON files should contain examples with ChatML conversation format:
    {
        "chosen": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "rejected": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }

    Args:
        train_data_path (Union[str, list[str]]): Path(s) to the JSON file(s) containing training data
        val_data_path (Union[str, list[str]]): Path(s) to the JSON file(s) containing validation data

    """

    def __init__(self, train_data_path: Union[str, list[str]], val_data_path: Union[str, list[str]]):
        # Load datasets from custom paths
        train_ds = load_dataset("json", data_files=train_data_path, split="train")
        val_ds = load_dataset("json", data_files=val_data_path, split="train")
        
        # Apply ChatML formatting to both datasets
        self.formatted_ds = {
            "train": train_ds.map(format_dpo_chatml),
            "validation": val_ds.map(format_dpo_chatml),
        }

        self.task_spec = TaskDataSpec(
            task_name="DPOChatML",
        )
