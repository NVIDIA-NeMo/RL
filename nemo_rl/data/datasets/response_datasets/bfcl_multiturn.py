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


from typing import Any, Optional
import jsonlines
from nemo_rl.data.interfaces import TaskDataSpec
from huggingface_hub import hf_hub_download


def format_bfcl(data: dict[str, Any]) -> dict[str, Any]:
    """Format BFCL data to the expected structure.
    
    The input data should already contain:
    - messages: list of conversation messages
    - task_name: task identifier  
    - dataset: dataset identifier
    And potentially other fields like metadata
    """
    # The data should already be in the correct format from the JSONL files
    formatted_data = dict(data)  # Make a copy
    
    # Ensure task_name is set for compatibility
    if "task_name" not in formatted_data:
        formatted_data["task_name"] = "bfcl_multiturn"
    return formatted_data

class BFCLMultiturnDataset:
    """BFCL Multiturn dataset that loads JSONL files directly without Arrow conversion."""
    
    def __init__(
        self,
        seed: int = 42,
        prompt_file: Optional[str] = None,
    ):
        """Initialize the BFCL Multiturn dataset.

        Args:
            split: Split to load ("train" or "validation") 
            seed: Random seed for reproducibility
            prompt_file: Optional prompt file path
        """
        self.seed = seed
        
        # Load the data directly as lists
        self.train_data, self.val_data = self._load_data()
        
        self.task_spec = TaskDataSpec(
            task_name="bfcl_multiturn",
            prompt_file=prompt_file,
        )

    def _load_data(self):
        """Load data directly from JSONL files"""
        print("Loading BFCL multiturn dataset directly from JSONL files")
        
        # Load training data
        train_data = []
        try:
            print("Downloading training file...")
            train_file = hf_hub_download(
                repo_id="slikhite/BFCLv3",
                filename="BFCL_v3_multi_turn_train_final.jsonl",
                repo_type="dataset"
            )
            
            print(f"Loading training data from {train_file}...")
            with jsonlines.open(train_file, 'r') as reader:
                for item in reader:
                    formatted_item = format_bfcl(item)
                    train_data.append(formatted_item)
            print(f"✅ Loaded {len(train_data)} training samples")
            
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            raise
        
        # Load validation data
        val_data = []
        try:
            print("Downloading validation file...")
            val_file = hf_hub_download(
                repo_id="slikhite/BFCLv3",
                filename="BFCL_v3_multi_turn_val_final.jsonl",
                repo_type="dataset"
            )
            
            print(f"Loading validation data from {val_file}...")
            with jsonlines.open(val_file, 'r') as reader:
                for item in reader:
                    formatted_item = format_bfcl(item)
                    val_data.append(formatted_item)
            print(f"✅ Loaded {len(val_data)} validation samples")
            
        except Exception as e:
            print(f"Could not load validation data: {e}")
            print("Continuing with training data only...")
        return train_data, val_data

    @property 
    def formatted_ds(self):
        """Return formatted datasets for compatibility."""
        return {
            "train": self.train_data,
            "validation": self.val_data,
        }

