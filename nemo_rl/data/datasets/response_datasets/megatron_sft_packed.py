# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Any, Optional, cast

from nemo_rl.data import ResponseDatasetConfig
from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import load_dataset_from_path
from nemo_rl.data.megatron_sft_packed import (
    megatron_sft_packed_preprocessor,
    resolve_megatron_sft_prompt_config,
)


class MegatronSFTPackedDataset(RawDataset):
    """Load Megatron-LM offline-packed SFT JSONL records."""

    def __init__(
        self,
        data_path: str,
        chat_key: str,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        split_validation_size: float = 0,
        seed: int = 42,
        context_parallel_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        if context_parallel_size is None or context_parallel_size < 1:
            raise ValueError(
                "Megatron SFT packed data requires policy context_parallel_size >= 1"
            )
        self.context_parallel_size = context_parallel_size
        self.chat_key = chat_key
        self.task_name = "megatron_sft_packed"
        self.dataset = load_dataset_from_path(data_path, subset, split)
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )
        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        messages = list(data[self.chat_key])
        if not messages or messages[0]["role"] != "system":
            raise ValueError(
                "Megatron SFT packed records must start with a system message"
            )
        if messages[-1]["role"] != "assistant":
            raise ValueError(
                "Megatron SFT packed records must end with an assistant message"
            )
        if any(not isinstance(message["content"], str) for message in messages):
            raise ValueError(
                "Megatron SFT packed records require string content; "
                "multimodal content is not supported on this path"
            )
        return {"packed_messages": messages, "task_name": self.task_name}

    def set_processor(self) -> None:
        data_config = cast(ResponseDatasetConfig, self.data_config)
        packed_config = data_config["megatron_sft"]
        unknown = set(packed_config) - {"prompt_format", "override_pad_token"}
        if unknown:
            raise ValueError(f"Unknown megatron_sft settings: {sorted(unknown)}")
        prompt_format = packed_config["prompt_format"]
        resolve_megatron_sft_prompt_config(
            prompt_format, packed_config.get("override_pad_token")
        )
        self.processor = partial(
            megatron_sft_packed_preprocessor,
            prompt_format=prompt_format,
            pad_token=packed_config.get("override_pad_token"),
            context_parallel_size=self.context_parallel_size,
        )
