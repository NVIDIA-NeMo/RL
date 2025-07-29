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

from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.hf_datasets.clevr import pil_to_base64

def format_geometry3k_dataset(example: dict[str, Any], return_pil: bool = False) -> dict[str, Any]:
    """
    Format the Geometry3K dataset into an OpenAI-API-like message log.
    """
    # isolate single image
    example['image'] = example['images'][0] if isinstance(example['images'], list) else example['images']

    user_content = [
        {
            "type": "image",
            "image": pil_to_base64(example['image']) if not return_pil else example['image'],
        },
        {
            "type": "text", 
            "text": str(example["problem"]).replace("<image>", ""),
        }
    ]

    assistant_content = str(example["answer"])
    
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
    

def prepare_geometry3k_dataset(split: str = "train", task_name: str = "geometry3k"):
    if split == "train":
        tr_dataset = load_dataset("hiyouga/geometry3k")['train']
        val_dataset = load_dataset("hiyouga/geometry3k")['validation']
    else:
        tr_dataset = load_dataset("hiyouga/geometry3k")[split]
        val_dataset = load_dataset("hiyouga/geometry3k")[split]
    
    # format - disable features to avoid schema conflicts
    tr_dataset = tr_dataset.add_column("task_name", [task_name] * len(tr_dataset))
    val_dataset = val_dataset.add_column("task_name", [task_name] * len(val_dataset))
    return {
        'train': tr_dataset,
        'validation': val_dataset,
    }
        

class Geometry3KDataset:
    def __init__(self, split: str = "train", seed: int = 42, prompt_file: Optional[str] = None, task_name: str = "geometry3k"):
        """
        Simple wrapper around the Geometry3K dataset.

        Args:
            split: The split of the dataset to use.
            seed: The seed for the dataset.
            prompt_file: The file containing the prompt for the dataset.
            task_name: The name of the task.
        """
        assert split in ['train', 'validation', 'test'], f"Invalid split: {split}. Please use 'train' or 'validation' or 'test'."
        
        self.formatted_ds = prepare_geometry3k_dataset(split=split, task_name=task_name)
        self.task_spec = TaskDataSpec(
            task_name="Geometry3K",
            prompt_file=prompt_file,
        )
        
