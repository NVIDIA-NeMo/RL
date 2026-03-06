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

from nemo_rl.data.datasets.raw_dataset import RawDataset


class AishellDataset(RawDataset):
    """Wrapper around the yuekai/aishell ASR dataset.

    Formats audio samples into OpenAI-style messages for speech recognition
    fine-tuning with Qwen2-Audio.

    Args:
        split: Split name for the dataset. Maps to HuggingFace subset name.
               Supported: "train", "dev", "test".
    """

    task_name = "aishell"

    def __init__(self, split: str = "train", **kwargs):
        VALID_SPLITS = ("train", "dev", "test")
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split: {split}. Please use one of {VALID_SPLITS}."
            )

        self.dataset = load_dataset("yuekai/aishell", split, split="test")

        self.dataset = self.dataset.add_column(
            "task_name", [self.task_name] * len(self.dataset)
        )

        self.preprocessor = self.format_data
        self.val_dataset = None

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        audio_array = data["audio"]["array"]
        # Remove spaces from Chinese text
        text = data["text"].replace(" ", "")

        user_content = [
            {"type": "audio", "audio": audio_array},
            {
                "type": "text",
                "text": "Detect the language and recognize the speech: <|zh|>",
            },
        ]
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": text},
            ],
            "task_name": self.task_name,
        }
