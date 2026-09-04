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

from datasets import Dataset, load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class DAPOMath17KDataset(RawDataset):
    """Simple wrapper around the DAPO Math 17K dataset with train split."""

    def __init__(
        self,
        deduplicate_prompts: bool = False,
        drop_conflicting_rewards: bool = True,
        expected_num_prompts: int | None = None,
        **kwargs,
    ) -> None:
        self.task_name = "DAPOMath17K"

        # load from huggingface
        self.dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")

        if deduplicate_prompts:
            self.dataset = _select_unique_dapo_prompts(
                self.dataset,
                drop_conflicting_rewards=drop_conflicting_rewards,
            )
        if (
            expected_num_prompts is not None
            and len(self.dataset) != expected_num_prompts
        ):
            raise ValueError(
                "Unexpected DAPO prompt count after preprocessing: "
                f"expected {expected_num_prompts}, got {len(self.dataset)}"
            )

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": data["prompt"][0]["content"],
                },
                {
                    "role": "assistant",
                    "content": data["reward_model"]["ground_truth"],
                },
            ],
            "task_name": self.task_name,
        }


class DAPOMathAIME2024Dataset(DAPOMath17KDataset):
    def __init__(self, **kwargs) -> None:
        """Initialize the DAPO Math AIME 2024 dataset with train split."""
        self.task_name = "DAPOMathAIME2024"

        # load from huggingface
        self.dataset = load_dataset("BytedTsinghua-SIA/AIME-2024", split="train")

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )


def _select_unique_dapo_prompts(
    dataset: Dataset,
    *,
    drop_conflicting_rewards: bool,
) -> Dataset:
    """Return the first row for each prompt, optionally dropping conflicts.

    The hosted DAPO-Math-17k parquet contains about 100 accidental repetitions.
    A small number of identical prompts are also paired with different reward
    models. Iterating Arrow record batches avoids materializing the 1.79M-row
    dataset as a pandas DataFrame while preserving deterministic source order.
    """

    first_index_by_prompt: dict[str, int] = {}
    reward_key_by_prompt: dict[str, str] = {}
    conflicting_prompts: set[str] = set()
    row_index = 0

    for batch in dataset.iter(batch_size=10_000):
        prompts = batch["prompt"]
        reward_models = batch["reward_model"]
        for prompt_messages, reward_model in zip(
            prompts, reward_models, strict=True
        ):
            prompt = prompt_messages[0]["content"]
            reward_key = json.dumps(
                reward_model,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if prompt not in first_index_by_prompt:
                first_index_by_prompt[prompt] = row_index
                reward_key_by_prompt[prompt] = reward_key
            elif reward_key_by_prompt[prompt] != reward_key:
                conflicting_prompts.add(prompt)
            row_index += 1

    selected_indices = [
        index
        for prompt, index in first_index_by_prompt.items()
        if not drop_conflicting_rewards or prompt not in conflicting_prompts
    ]
    print(
        "  ✓ DAPO prompt deduplication: "
        f"{len(dataset)} rows -> {len(first_index_by_prompt)} unique prompts"
        + (
            f" -> {len(selected_indices)} after dropping "
            f"{len(conflicting_prompts)} reward conflicts"
            if drop_conflicting_rewards
            else ""
        ),
        flush=True,
    )
    return dataset.select(selected_indices)
