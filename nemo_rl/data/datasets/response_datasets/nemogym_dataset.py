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

import torch

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import load_dataset_from_path


class NemoGymDataset(RawDataset):
    """Simple wrapper around the Nemo Gym dataset."""

    def __init__(self, data_path: Optional[str] = None, **kwargs) -> None:
        self.task_name = "NemoGymDataset"

        # load from jsonl
        if data_path is None:
            # Allow optional at type level for config validation; enforce at runtime for clarity
            raise ValueError(
                "NemoGymDataset requires `data_path` in data_config to load examples."
            )
        self.dataset = load_dataset_from_path(data_path)

        # format the dataset
        # HuggingFace Dataset 在 map/写入 Arrow 时不会持久化 torch.Tensor，会把它序列化成 Python 列表。因此下游在取样时读到的是 []（list），触发断言
        self.dataset = self.dataset.map(
            self.format_data,
            with_indices=True,
        )
        if "repeat" in kwargs:
            self.dataset = self.dataset.repeat(kwargs["repeat"])

    def format_data(self, data: dict[str, Any], idx: int) -> dict[str, Any]:
        return {
            "message_log": [
                {"role": "user", "content": "", "token_ids": torch.tensor([])}
            ],
            "task_name": self.task_name,
            "length": 0,
            "extra_env_info": data,
            "loss_multiplier": 1.0,  # Fix to 1.0 to backprop on all examples
            "idx": idx,
            "stop_strings": None,
            # Extra vars
            "token_ids": [],  # Just need this empty key to be compatible with the current NeMo RL GRPO impl
        }
