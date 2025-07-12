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
import json

from datasets import Dataset, load_dataset

from nemo_rl.data.interfaces import TaskDataSpec
from PIL import Image
import io
import base64

def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Converts a PIL Image object to a base64 encoded string.

    Args:
        image: The PIL Image object to convert.
        format: The image format (e.g., "PNG", "JPEG"). Defaults to "PNG".

    Returns:
        A base64 encoded string representation of the image.
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # append data:image/png;base64,
    return f"data:image/png;base64,{img_str}"

def format_answer_fromtags(answer: str) -> str:
    """
    Format the answer from tags.
    """
    # Extract content between <answer> tags and strip whitespace
    import re
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, answer)
    ret = match.group(1).strip() if match else answer.strip()
    return ret

# ⚠️ this does not work because datasets.Dataset.from_pandas(pd.DataFrame(data)) does not take mix of list and non-list strings
def format_clevr_cogent_dataset(example: dict[str, Any]) -> dict[str, Any]:
    """
    Format the CLEVR-CoGenT dataset.
    """
    # Ensure consistent data types
    user_content = [
        {
            "type": "image",
            "image": pil_to_base64(example['image']),
        },
        {
            "type": "text", 
            "text": str(example["problem"]),
        }
    ]
    
    assistant_content = format_answer_fromtags(str(example["solution"]))
    
    ret = {
        "messages": [
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant", 
                "content": assistant_content,
            },
        ],
        "task_name": "clevr-cogent",
    }
    return ret
    

# contain different variants of the CLEVR dataset 
def prepare_clevr_cogent_dataset(split: str = "trainA", seed: int = 42, task_name: Optional[str] = None):
    if task_name is None:
        task_name = "clevr-cogent"

    if split == "trainA":
        tr_dataset = load_dataset("MMInstruction/Clevr_CoGenT_TrainA_70K_Complex")['train']
        val_dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValA")['train']
    elif split == "trainB":
        tr_dataset = load_dataset("MMInstruction/Clevr_CoGenT_TrainB_70K_Complex")['train']
        val_dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB")['train']
    elif split == "valA":
        tr_dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValA")['train']
        val_dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValA")['train']
    elif split == "valB":
        tr_dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB")['train']
        val_dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB")['train']
    
    # format - disable features to avoid schema conflicts
    tr_dataset = tr_dataset.add_column("task_name", [task_name] * len(tr_dataset))
    val_dataset = val_dataset.add_column("task_name", [task_name] * len(val_dataset))

    return {
        'train': tr_dataset,
        'validation': val_dataset,
    }
        

class CLEVRCoGenTDataset:
    def __init__(self, split: str = "trainA", seed: int = 42, prompt_file: Optional[str] = None, task_name: str = "clevr-cogent"):
        """
        Simple wrapper around the CLEVR-CoGenT dataset.

        Args:
            split: The split of the dataset to use.
            seed: The seed for the dataset.
            prompt_file: The file containing the prompt for the dataset.
            task_name: The name of the task.
        """
        if split not in ['trainA', 'trainB', 'valA', 'valB']:
            raise ValueError(f"Invalid split: {split}. Please use 'trainA', 'trainB', 'valA', or 'valB'.")
        
        self.formatted_ds = prepare_clevr_cogent_dataset(split=split, seed=seed, task_name=task_name)
        self.task_spec = TaskDataSpec(
            task_name="CLEVR",
            prompt_file=prompt_file,
        )
        
