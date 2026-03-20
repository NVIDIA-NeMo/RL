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

from nemo_rl.data.datasets.raw_dataset import RawDataset

VALID_SPLITS = ("high_part00", "high_part01", "high_part02", "medium", "low")


class NemotronMathV2Dataset(RawDataset):
    """Dataset class for nvidia/Nemotron-Math-v2.

    The dataset contains math reasoning traces generated at three reasoning depths
    (high/medium/low). Each example's ``messages`` field contains user and assistant
    turns where the assistant turn carries both a ``reasoning_content`` field
    (chain-of-thought) and a ``content`` field (final answer).

    This class reformats each assistant message so that the reasoning is wrapped in
    ``<think>...</think>`` tags and prepended to the final answer, producing a
    standard ``[{"role": ..., "content": ...}]`` message list that is compatible
    with ``apply_chat_template``.

    Args:
        split: Split name. Valid splits: ``high_part00``, ``high_part01``,
            ``high_part02``, ``medium``, ``low``. Defaults to ``"high_part02"``.
        split_validation_size: Fraction of data to use for validation. Defaults to 0.
        seed: Random seed for train/validation split. Defaults to 42.
    """

    def __init__(
        self,
        split: str = "high_part02",
        split_validation_size: float = 0,
        seed: int = 42,
        **kwargs,
    ):
        self.task_name = "Nemotron-Math-v2"

        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Valid splits are: {VALID_SPLITS}"
            )

        # self.dataset = load_dataset("nvidia/Nemotron-Math-v2", split=split)
        self.dataset = load_dataset(
            "parquet",
            data_files=f"hf://datasets/nvidia/Nemotron-Math-v2/data/{split}.parquet",
            split="train",
        )

        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        messages = []
        for msg in data["messages"]:
            role = msg["role"]
            content = msg.get("content") or ""
            reasoning_content: Optional[str] = msg.get("reasoning_content")

            if role == "assistant" and reasoning_content:
                content = f"<think>\n{reasoning_content}\n</think>\n{content}"

            messages.append({"role": role, "content": content})

        return {
            "messages": messages,
            "task_name": self.task_name,
        }
