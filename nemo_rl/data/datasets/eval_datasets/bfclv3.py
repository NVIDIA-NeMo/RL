from typing import Any, Optional
import random

import jsonlines

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class BFCLDataset:
    def __init__(
        self,
        data_path: str,
    ):
        self.data_path = data_path
        
        # Load and process data immediately
        raw_data = self._load_data()
        # Apply the rekeying transformation to create the processed dataset
        self.rekeyed_ds = [self._rekey(item) for item in raw_data]
        
        # Set up task specification and processor
        self.task_spec = TaskDataSpec(
            task_name="bfcl",
        )
        self.processor = processors.bfcl_processor

    def _load_data(self):
        """Load JSONL data into memory."""
        with jsonlines.open(self.data_path, "r") as reader:
            data = [line for line in reader]
        return data

    def __len__(self):
        return len(self.rekeyed_ds)
    
    def __getitem__(self, idx: int):
        """Return a processed data item."""
        return self.rekeyed_ds[idx]

    def _rekey(self, data: dict[str, Any]):
        single_message = data["messages"][0]
        system_content = ""
        user_content = ""
        metadata = {}

        for m in single_message:
            if m["role"] == "system":
                system_content = m["content"]
            if m["role"] == "user":
                user_content = m["content"]
                if "metadata" in m:
                    metadata = m["metadata"]
        return {
            "system_content": system_content,
            "user_content": user_content,
            "metadata": metadata,
            "task_name": data.get("task_name", "bfcl"),
            "dataset": data.get("dataset", "bfcl"),
        }