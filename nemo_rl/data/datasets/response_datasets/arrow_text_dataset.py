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

"""Arrow Text Dataset for loading arrow files with 'text' column."""

import glob
from typing import Any, Optional

from datasets import Dataset, load_dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset


class ArrowTextDataset(RawDataset):
    """Dataset class for loading arrow files containing raw text.
    
    This class loads arrow files with a 'text' column and converts them to
    the messages format expected by SFT training.
    
    The text is wrapped as an assistant message:
        {"messages": [{"role": "assistant", "content": <text>}]}
    
    This format allows training on all tokens (language modeling style).
    
    Args:
        arrow_files: Path pattern (glob) or list of arrow file paths
        val_split: Fraction of data to use for validation (default: 0.05)
        seed: Random seed for train/val split
        text_key: Key for text column in arrow files (default: "text")
    
    Example config:
        data:
          dataset_name: "arrow_text"
          arrow_files: "/path/to/data/*.arrow"
          val_split: 0.05
          max_input_seq_length: 4096
    """

    def __init__(
        self,
        arrow_files: str | list[str],
        val_split: float = 0.05,
        seed: int = 42,
        text_key: str = "text",
    ):
        # Don't call super().__init__() since RawDataset raises NotImplementedError
        self.seed = seed
        self.text_key = text_key
        self.task_name = "arrow_text_dataset"
        
        # Resolve glob pattern if string
        if isinstance(arrow_files, str):
            file_list = glob.glob(arrow_files)
            if not file_list:
                raise ValueError(f"No arrow files found matching pattern: {arrow_files}")
        else:
            file_list = arrow_files
        
        print(f"Loading {len(file_list)} arrow files...")
        dataset = load_dataset("arrow", data_files=file_list, split="train")
        print(f"  ✓ Loaded {len(dataset)} total samples")
        
        # Verify text column exists
        if self.text_key not in dataset.column_names:
            raise ValueError(
                f"Column '{self.text_key}' not found in arrow files. "
                f"Available columns: {dataset.column_names}"
            )
        
        # Convert text to messages format
        def text_to_messages(example: dict[str, Any]) -> dict[str, Any]:
            """Convert raw text to messages format for SFT training."""
            text = example[self.text_key]
            return {
                "messages": [{"role": "assistant", "content": text}]
            }
        
        formatted_dataset = dataset.map(text_to_messages, remove_columns=dataset.column_names)
        
        # Split into train/validation
        if val_split > 0:
            split = formatted_dataset.train_test_split(test_size=val_split, seed=seed)
            train_dataset = split["train"]
            val_dataset = split["test"]
        else:
            train_dataset = formatted_dataset
            # Create a small validation set from the end
            val_dataset = formatted_dataset.select(range(min(100, len(formatted_dataset))))
        
        print(f"  ✓ Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
        
        self.formatted_ds = {
            "train": train_dataset,
            "validation": val_dataset,
        }
