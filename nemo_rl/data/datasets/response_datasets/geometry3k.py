## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_rl.data.datasets.utils import pil_to_base64


def format_geometry3k_dataset(
    example: dict[str, Any], return_pil: bool = False
) -> dict[str, Any]:
    """Format the Geometry3K dataset into an OpenAI-API-like message log."""
    # isolate single image
    if isinstance(example["images"], list):
        example["image"] = example["images"][0]

    user_content = [
        {
            "type": "image",
            "image": pil_to_base64(example["image"])
            if not return_pil
            else example["image"],
        },
        {
            "type": "text",
            "text": str(example["problem"]).replace("<image>", ""),
        },
    ]

    assistant_content = str(example["answer"])

    ret = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "task_name": example["task_name"],
    }
    return ret


class Geometry3KDataset(RawDataset):
    def __init__(self, split: str = "train", **kwargs):
        """Simple wrapper around the Geometry3K dataset.

        Args:
            split: The split of the dataset to use.
        """
        # train, validation, and test are supported splits.
        assert split in ["train", "validation", "test"], (
            f"Invalid split: {split}. Please use 'train' or 'validation' or 'test'."
        )

        self.task_name = "geometry3k"

        # this dataset will process the image during training using `format_geometry3k_dataset`
        self.dataset = load_dataset("hiyouga/geometry3k")[split]

        # format - disable features to avoid schema conflicts
        self.dataset = self.dataset.add_column(
            "task_name", [self.task_name] * len(self.dataset)
        )
