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
from typing import Any

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class Tulu3PreferenceDataset(RawDataset):
    """Tulu3 preference dataset for DPO training."""

    def __init__(self, **kwargs):
        self.task_name = "Tulu3Preference"

        # load from huggingface
        self.dataset = load_dataset("allenai/llama-3.1-tulu-3-8b-preference-mixture")[
            "train"
        ]

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
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

        chosen = chosen_conversation[-1]["content"]
        rejected = rejected_conversation[-1]["content"]

        return {
            "context": context,
            "completions": [
                {"rank": 0, "completion": [{"role": "assistant", "content": chosen}]},
                {"rank": 1, "completion": [{"role": "assistant", "content": rejected}]},
            ],
            "task_name": self.task_name,
        }
