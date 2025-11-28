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

from typing import Any, Optional

from datasets import load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import pil_to_base64


def format_answer_fromtags(answer: str) -> str:
    """Extract content between <answer> tags and strip whitespace."""
    import re

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, answer)
    ret = match.group(1).strip() if match else answer.strip()
    return ret


def format_clevr_cogent_dataset(
    example: dict[str, Any], return_pil: bool = False
) -> dict[str, Any]:
    """Format the CLEVR-CoGenT dataset into an OpenAI-API-like message log."""
    user_content = [
        {
            "type": "image",
            "image": pil_to_base64(example["image"])
            if not return_pil
            else example["image"],
        },
        {
            "type": "text",
            "text": str(example["problem"]),
        },
    ]

    assistant_content = format_answer_fromtags(str(example["solution"]))

    ret = {
        "messages": [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": assistant_content,
            },
        ],
        "task_name": "clevr-cogent",
    }
    return ret


# contain different variants of the CLEVR dataset
def prepare_clevr_cogent_dataset(
    split: str = "trainA", task_name: Optional[str] = None
):
    if task_name is None:
        task_name = "clevr-cogent"

    if split == "trainA":
        tr_dataset = load_dataset("MMInstruction/Clevr_CoGenT_TrainA_70K_Complex")[
            "train"
        ]
        val_dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValA")["train"]
    elif split == "trainB":
        tr_dataset = load_dataset("MMInstruction/Clevr_CoGenT_TrainA_70K_Complex")[
            "train"
        ]
        val_dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB")["train"]
    elif split == "valA":
        tr_dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValA")["train"]
        val_dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValA")["train"]
    elif split == "valB":
        tr_dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB")["train"]
        val_dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB")["train"]

    # format - disable features to avoid schema conflicts
    tr_dataset = tr_dataset.add_column("task_name", [task_name] * len(tr_dataset))
    val_dataset = val_dataset.add_column("task_name", [task_name] * len(val_dataset))

    return {
        "train": tr_dataset,
        "validation": val_dataset,
    }


class CLEVRCoGenTDataset(RawDataset):
    def __init__(
        self,
        split: str = "trainA",
        prompt_file: Optional[str] = None,
    ):
        """Simple wrapper around the CLEVR-CoGenT dataset.

        Args:
            split: The split of the dataset to use.
            prompt_file: The file containing the prompt for the dataset.
        """
        if split not in ["trainA", "trainB", "valA", "valB"]:
            raise ValueError(
                f"Invalid split: {split}. Please use 'trainA', 'trainB', 'valA', or 'valB'."
            )
        self.task_name = "clevr-cogent"

        self.formatted_ds = prepare_clevr_cogent_dataset(
            split=split, task_name=self.task_name
        )
