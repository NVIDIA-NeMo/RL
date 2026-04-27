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

import os
import shutil
import zipfile
from typing import Any, Optional

import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.interfaces import TaskDataSpec


def format_mmpr_tiny_dataset(example: dict[str, Any]) -> dict[str, Any]:
    """Format the MMPR-Tiny dataset into an OpenAI-API-like message log."""
    user_content = [
        {
            "type": "image",
            "image": example["images"][0],
        },
        {
            "type": "text",
            "text": str(example["question"]).replace("<image>", ""),
        },
    ]

    assistant_content = str(example["answer"])

    ret = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "task_name": "mmpr_tiny",
    }
    return ret


def _ensure_mmpr_cached(cache_dir: str) -> None:
    """Download and extract MMPR-Tiny images if not already cached.
    
    Thread-safe: Uses atomic marker file to prevent race conditions in distributed settings.
    """
    images_dir = os.path.join(cache_dir, "MMPR-Tiny", "images")
    parquet_path = os.path.join(cache_dir, "mmpr_tiny.parquet")
    ready_marker = os.path.join(cache_dir, ".mmpr_ready")
    
    # Check if already cached and ready
    if os.path.exists(ready_marker):
        return
    
    # Also check if data exists (from previous version without marker)
    if os.path.exists(images_dir) and os.path.exists(parquet_path):
        # Create marker for future
        with open(ready_marker, 'w') as f:
            f.write('ready\n')
        return
    
    # Use lock file to prevent multiple processes from downloading
    lock_file = os.path.join(cache_dir, ".mmpr_download.lock")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Try to acquire lock (atomic on most filesystems including Lustre)
    import fcntl
    try:
        with open(lock_file, 'w') as lock:
            # Non-blocking lock - first process wins, others wait
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            
            # Double-check after acquiring lock (another process might have finished)
            if os.path.exists(ready_marker):
                return
            
            # We have the lock - download and extract
            print(f"Downloading MMPR-Tiny to {cache_dir}...")
            
            # Download and extract images
            zip_path = hf_hub_download('OpenGVLab/MMPR-Tiny', 'images.zip', repo_type='dataset')
            with zipfile.ZipFile(zip_path, 'r') as zf:
                temp = os.path.join(cache_dir, '_temp')
                zf.extractall(temp)
                shutil.move(os.path.join(temp, 'images'), images_dir)
                os.rmdir(temp)
            
            # Download parquet
            pq = hf_hub_download('OpenGVLab/MMPR-Tiny', 'mmpr_tiny.parquet', repo_type='dataset')
            shutil.copy(pq, parquet_path)
            
            # Create ready marker atomically
            with open(ready_marker, 'w') as f:
                f.write('ready\n')
            print(f"MMPR-Tiny cached successfully at {cache_dir}")
            
    finally:
        # Lock automatically released when file closes
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except:
                pass


def prepare_mmpr_tiny_dataset(
    split: str = "train", 
    task_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    val_size: int = 500,
):
    """Load and prepare MMPR-Tiny dataset.
    
    Args:
        val_size: Number of samples for validation. Set to 0 to use all data for training (default 500).
    """
    if task_name is None:
        task_name = "mmpr_tiny"
    
    if cache_dir is None:
        raise ValueError("cache_dir is missing for MMPR-Tiny")
    
    # Ensure data is cached (only first worker will download, others wait)
    _ensure_mmpr_cached(cache_dir)
    
    # Load parquet
    df = pd.read_parquet(os.path.join(cache_dir, "mmpr_tiny.parquet"))
    
    # Extract fields - store image PATHS as list (matching vision_r1 structure)
    df['images'] = df['images'].str[0].apply(lambda x: [os.path.join(cache_dir, x['path'])])  # Wrap in list!
    df['question'] = df['prompt'].apply(lambda p: next((m['content'] for m in p if m.get('role') == 'user'), ''))
    df['answer'] = df['reward_model'].apply(lambda r: r.get('ground_truth', ''))
    df = df[['images', 'question', 'answer']]
    
    # Add task_name column
    df = df.assign(task_name=task_name)
    
    from datasets import Features, Value, Sequence
    features = Features({
        'images': Sequence(Value('string')),
        'question': Value('string'),
        'answer': Value('string'),
        'task_name': Value('string'),
    })
    
    full_dataset = Dataset.from_pandas(df, preserve_index=False, features=features)
    
    # Split train/val (following vision_r1 pattern)
    total_size = len(full_dataset)
    if val_size > 0:
        # Use specified number of samples for validation
        val_size = min(val_size, total_size // 10)  # Cap at 10% of data
        val_dataset = full_dataset.select(range(val_size))
        train_dataset = full_dataset.select(range(val_size, total_size))
    else:
        # Use all data for training (no validation)
        train_dataset = full_dataset
        val_dataset = None
    
    return {
        "train": train_dataset,
        "validation": val_dataset,
    }


class MMPRTinyDataset(RawDataset):
    def __init__(
        self,
        split: str = "train",
        prompt_file: Optional[str] = None,
        cache_dir: Optional[str] = None,
        val_size: int = 500,
    ):
        """Simple wrapper around the MMPR-Tiny dataset.

        Args:
            split: The split of the dataset to use.
            prompt_file: The file containing the prompt for the dataset.
            cache_dir: Directory to cache the extracted images.
            val_size: Number of samples for validation (default 500). Set to 0 to use all data for training.
        """
        if split not in ["train", "test"]:
            raise ValueError(
                f"Invalid split: {split}. Please use 'train' or 'test'."
            )
        self.task_name = "mmpr_tiny"

        self.formatted_ds = prepare_mmpr_tiny_dataset(
            split=split, task_name=self.task_name, cache_dir=cache_dir, val_size=val_size
        )
        self.task_spec = TaskDataSpec(
            task_name="MMPR-Tiny",
            prompt_file=prompt_file,
        )

