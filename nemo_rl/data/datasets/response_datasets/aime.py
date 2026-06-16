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

"""AIME response dataset (2024/2025/2026 variants)."""

from typing import Any, Literal

from datasets import concatenate_datasets, load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset

AIMEVariant = Literal["2024", "2025", "2026"]


class AIMEDataset(RawDataset):
    """Wrapper around an AIME competition dataset with a train split.

    Mirrors nemo_rl.data.datasets.eval_datasets.aime.AIMEDataset so the same
    year can be referenced both as a validation dataset during training and
    as an eval dataset.

    Args:
        variant: Which AIME edition to load: "2024" (default), "2025", or "2026".
        repeat: Number of times to repeat the dataset, default is 16. This
            matches the variance-reduction recipe used by the original
            AIME2024 entry.
    """

    def __init__(
        self,
        variant: AIMEVariant = "2024",
        repeat: int = 16,
        **kwargs,
    ) -> None:
        self.task_name = f"AIME{variant}"

        # load from huggingface
        if variant == "2024":
            ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
            self.input_key = "problem"
        elif variant == "2025":
            ds0 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
            ds1 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
            ds = concatenate_datasets([ds0, ds1])
            self.input_key = "question"
        elif variant == "2026":
            ds = load_dataset("MathArena/aime_2026", split="train")
            self.input_key = "problem"
        else:
            raise ValueError(f"Invalid AIME variant: {variant!r}")

        # format the dataset
        self.dataset = ds.map(
            self.format_data,
            remove_columns=ds.column_names,
        )

        # repeat the dataset
        self.dataset = self.dataset.repeat(repeat)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        # MathArena/aime_2026 stores `answer` as int64; cast to keep the schema uniform.
        return {
            "messages": [
                {"role": "user", "content": data[self.input_key]},
                {"role": "assistant", "content": str(data["answer"])},
            ],
            "task_name": self.task_name,
        }
