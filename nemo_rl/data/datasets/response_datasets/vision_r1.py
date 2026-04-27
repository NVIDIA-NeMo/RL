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

import base64
import io
from typing import Any, Optional

from datasets import load_dataset
from PIL import Image

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.interfaces import TaskDataSpec


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Converts a PIL Image object to a base64 encoded string.

    Args:
        image: The PIL Image object to convert.
        format: The image format (e.g., "PNG", "JPEG"). Defaults to "PNG".

    Returns:
        A base64 encoded string representation of the image.
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def format_answer_fromtags(answer: str) -> str:
    """Extract content between <answer> tags and strip whitespace."""
    import re

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, answer)
    ret = match.group(1).strip() if match else answer.strip()
    return ret


def format_vision_r1_dataset(
    example: dict[str, Any], return_pil: bool = False
) -> dict[str, Any]:
    """Format the Vision-R1 dataset into an OpenAI-API-like message log."""
    user_content = [
        {
            "type": "image",
            "image": pil_to_base64(example["images"][0])
            if not return_pil
            else example["images"][0],
        },
        {
            "type": "text",
            "text": str(example["problem"]).replace("<image>", ""),
        },
    ]

    assistant_content = format_answer_fromtags(str(example["answer"]))

    ret = {
        "messages": [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": assistant_content,
            },
        ],
        "task_name": "vision_r1",
    }
    return ret


def prepare_vision_r1_dataset(
    split: str = "train", task_name: Optional[str] = None
):
    if task_name is None:
        task_name = "vision_r1"

    # Load the full dataset to get available splits
    full_dataset = load_dataset("Osilly/Vision-R1-rl")

    # Get the train dataset (use the requested split if available, otherwise use the first available split)
    if split in full_dataset:
        train_dataset = full_dataset[split]
    elif "train" in full_dataset:
        train_dataset = full_dataset["train"]
    else:
        # Use the first available split as train
        available_splits = list(full_dataset.keys())
        train_dataset = full_dataset[available_splits[0]]
        print(f"Warning: Requested split '{split}' not found. Using '{available_splits[0]}' instead.")

    # For validation, try to use a validation split if available, otherwise use a subset of train
    if split in ["validation", "val", "test"] and split in full_dataset:
        val_dataset = full_dataset[split]
    else:
        # Create a small validation set from train data
        train_size = len(train_dataset)
        val_size = min(500, train_size // 10)  # Use 10% or 500 samples, whichever is smaller
        val_dataset = train_dataset.select(range(val_size))
        train_dataset = train_dataset.select(range(val_size, train_size))

    # Format - disable features to avoid schema conflicts
    train_dataset = train_dataset.add_column("task_name", [task_name] * len(train_dataset))
    val_dataset = val_dataset.add_column("task_name", [task_name] * len(val_dataset))

    return {
        "train": train_dataset,
        "validation": val_dataset,
    }


class VisionR1Dataset(RawDataset):
    def __init__(
        self,
        split: str = "train",
        prompt_file: Optional[str] = None,
    ):
        """Simple wrapper around the Vision-R1 dataset.

        Args:
            split: The split of the dataset to use.
            prompt_file: The file containing the prompt for the dataset.
        """
        if split not in ["train", "test"]:
            raise ValueError(
                f"Invalid split: {split}. Please use 'train' or 'test'."
            )
        self.task_name = "vision_r1"

        self.formatted_ds = prepare_vision_r1_dataset(
            split=split, task_name=self.task_name
        )
        self.task_spec = TaskDataSpec(
            task_name="Vision-R1",
            prompt_file=prompt_file,
        )
