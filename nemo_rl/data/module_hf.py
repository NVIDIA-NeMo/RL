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
"""HuggingFace datasets integration for DataModule.

This module provides DataModule implementations that integrate with
HuggingFace datasets for easy dataset loading and processing.

Example:
    >>> from nemo_rl.data import HuggingFaceDataModule
    >>> 
    >>> # Load from HuggingFace Hub
    >>> datamodule = HuggingFaceDataModule("nvidia/OpenMathInstruct-2")
    >>> datamodule.setup()
    >>> 
    >>> for batch in datamodule.train_dataloader():
    ...     print(batch)
    
    >>> # Or with auto-detection
    >>> trainer.fit(dataset="nvidia/OpenMathInstruct-2")  # Auto-loads and maps columns
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping

from nemo_rl.data.module import MapDataModule

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset
    from torch.utils.data import DataLoader


# =========================================================================
# Known Dataset Column Mappings
# =========================================================================
# Auto-detect column mappings for common HuggingFace datasets
# Maps dataset names/patterns to their column mapping configurations
KNOWN_DATASET_MAPPINGS: dict[str, dict[str, str]] = {
    # Math instruction datasets
    "nvidia/OpenMathInstruct-2": {
        "problem": "prompt",
        "generated_solution": "response",
        "expected_answer": "answer",
    },
    "nvidia/OpenMathInstruct-1": {
        "question": "prompt",
        "generated_solution": "response",
        "expected_answer": "answer",
    },
    # Common instruction datasets
    "HuggingFaceH4/ultrachat_200k": {
        "messages": "messages",
    },
    "Open-Orca/OpenOrca": {
        "question": "prompt",
        "response": "response",
    },
    "tatsu-lab/alpaca": {
        "instruction": "prompt",
        "output": "response",
        "input": "context",
    },
    "databricks/dolly-15k": {
        "instruction": "prompt",
        "response": "response",
        "context": "context",
    },
    # Code datasets
    "codeparrot/github-code": {
        "code": "text",
    },
    "bigcode/the-stack": {
        "content": "text",
    },
    # Preference datasets (for DPO/RLHF)
    "Anthropic/hh-rlhf": {
        "chosen": "chosen",
        "rejected": "rejected",
    },
    "argilla/ultrafeedback-binarized-preferences": {
        "chosen": "chosen",
        "rejected": "rejected",
        "prompt": "prompt",
    },
}

# Column name patterns for auto-detection (fallback if not in known datasets)
COLUMN_NAME_PATTERNS: dict[str, list[str]] = {
    "prompt": ["prompt", "question", "instruction", "problem", "input", "query", "text"],
    "response": ["response", "answer", "output", "completion", "generated_solution", "solution"],
    "context": ["context", "input_context", "system"],
    "chosen": ["chosen", "chosen_response", "preferred"],
    "rejected": ["rejected", "rejected_response", "dispreferred"],
    "messages": ["messages", "conversation", "chat"],
}


def detect_dataset_format(columns: list[str]) -> str:
    """Detect the dataset format based on column names.
    
    Args:
        columns: List of column names in the dataset.
        
    Returns:
        One of: 'chat', 'completion', 'preference', 'text', 'unknown'
    """
    col_set = set(columns)
    
    # Check for chat/conversation format
    if any(c in col_set for c in COLUMN_NAME_PATTERNS["messages"]):
        return "chat"
    
    # Check for preference format (DPO)
    has_chosen = any(c in col_set for c in COLUMN_NAME_PATTERNS["chosen"])
    has_rejected = any(c in col_set for c in COLUMN_NAME_PATTERNS["rejected"])
    if has_chosen and has_rejected:
        return "preference"
    
    # Check for instruction/completion format
    has_prompt = any(c in col_set for c in COLUMN_NAME_PATTERNS["prompt"])
    has_response = any(c in col_set for c in COLUMN_NAME_PATTERNS["response"])
    if has_prompt and has_response:
        return "completion"
    
    # Check for raw text format
    if "text" in col_set or "content" in col_set:
        return "text"
    
    return "unknown"


def auto_detect_column_mapping(
    dataset_name: str,
    columns: list[str],
) -> dict[str, str]:
    """Auto-detect column mapping for a dataset.
    
    First checks known dataset mappings, then falls back to pattern matching.
    
    Args:
        dataset_name: Name of the HuggingFace dataset.
        columns: List of column names in the dataset.
        
    Returns:
        Dictionary mapping dataset columns to expected format.
    """
    # Check known mappings first
    if dataset_name in KNOWN_DATASET_MAPPINGS:
        known_mapping = KNOWN_DATASET_MAPPINGS[dataset_name]
        # Only return mappings for columns that exist
        return {k: v for k, v in known_mapping.items() if k in columns}
    
    # Fall back to pattern matching
    mapping: dict[str, str] = {}
    col_set = set(columns)
    
    for target_name, patterns in COLUMN_NAME_PATTERNS.items():
        for pattern in patterns:
            if pattern in col_set:
                # Don't remap if column already has the target name
                if pattern != target_name:
                    mapping[pattern] = target_name
                break
    
    return mapping


class HuggingFaceDataModule(MapDataModule):
    """DataModule for loading HuggingFace datasets.

    Provides automatic loading and processing of HuggingFace datasets
    with support for automatic column mapping and data transformations.
    
    When column_mapping is not provided, the module automatically detects
    the dataset format and maps columns to the expected names (prompt, 
    response, etc.).

    Attributes:
        dataset_name: Name of the HuggingFace dataset or path to local files.
        split: Dataset split to use (e.g., 'train', 'validation').
        val_dataset_name: Optional validation dataset name or split.
        column_mapping: Mapping from dataset columns to expected format.
            If None, auto-detection is used.
        transform: Optional transform function to apply to each example.
        auto_map_columns: Whether to auto-detect column mappings (default: True).
        dataset_format: Detected format ('chat', 'completion', 'preference', etc.).

    Example:
        >>> # Load from Hub with auto-column-mapping
        >>> dm = HuggingFaceDataModule("nvidia/OpenMathInstruct-2")
        >>> dm.setup()
        >>> print(dm.dataset_format)  # 'completion'
        >>> 
        >>> # Or with explicit column mapping
        >>> dm = HuggingFaceDataModule(
        ...     "my_dataset",
        ...     column_mapping={"input": "prompt", "output": "response"},
        ...     auto_map_columns=False,
        ... )
        
        >>> # Directly from trainer
        >>> trainer.fit(dataset="nvidia/OpenMathInstruct-2")  # Zero config!
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        val_dataset_name: str | None = None,
        val_split: str | None = None,
        column_mapping: Mapping[str, str] | None = None,
        transform: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        shuffle_train: bool = True,
        streaming: bool = False,
        trust_remote_code: bool = False,
        cache_dir: str | None = None,
        auto_map_columns: bool = True,
        **load_kwargs: Any,
    ):
        """Initialize the HuggingFaceDataModule.

        Args:
            dataset_name: HuggingFace dataset name or path to local files.
            split: Dataset split for training data.
            val_dataset_name: Validation dataset name (if different from dataset_name).
            val_split: Validation split (defaults to 'validation' or 'test').
            column_mapping: Mapping from dataset columns to expected format.
                If None and auto_map_columns=True, mapping is auto-detected.
            transform: Function to transform each example.
            batch_size: Batch size for data loaders.
            num_workers: Number of data loading workers.
            pin_memory: Whether to pin memory for faster GPU transfer.
            shuffle_train: Whether to shuffle training data.
            streaming: Whether to use streaming mode for large datasets.
            trust_remote_code: Whether to trust remote code in datasets.
            cache_dir: Directory to cache downloaded datasets.
            auto_map_columns: Whether to auto-detect column mappings when
                column_mapping is not provided. Default True.
            **load_kwargs: Additional arguments passed to datasets.load_dataset.
        """
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle_train=shuffle_train,
        )
        self.dataset_name = dataset_name
        self.split = split
        self.val_dataset_name = val_dataset_name
        self.val_split = val_split
        self._explicit_column_mapping = column_mapping
        self.column_mapping: dict[str, str] = dict(column_mapping) if column_mapping else {}
        self.transform = transform
        self.streaming = streaming
        self.trust_remote_code = trust_remote_code
        self.cache_dir = cache_dir
        self.auto_map_columns = auto_map_columns
        self.load_kwargs = load_kwargs

        # Datasets will be set in setup()
        self._train_dataset: "HFDataset | None" = None
        self._val_dataset: "HFDataset | None" = None
        
        # Detected format will be set in setup()
        self.dataset_format: str = "unknown"

    def prepare_data(self) -> None:
        """Download data (called only on main process).

        This downloads the dataset to cache if not already present.
        """
        try:
            from datasets import load_dataset

            # Just trigger download, don't store
            load_dataset(
                self.dataset_name,
                split=self.split,
                streaming=self.streaming,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir,
                **self.load_kwargs,
            )
        except Exception:
            # Download errors will be handled in setup()
            pass

    def setup(self, stage: str | None = None) -> None:
        """Load and prepare datasets.

        This method:
        1. Loads the dataset from HuggingFace Hub or local path
        2. Auto-detects column mappings if not explicitly provided
        3. Detects the dataset format (chat, completion, preference, etc.)
        4. Applies column mapping and transforms
        5. Loads validation split if available

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        from datasets import load_dataset

        if stage in (None, "fit"):
            # Load training dataset
            self._train_dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                streaming=self.streaming,
                trust_remote_code=self.trust_remote_code,
                cache_dir=self.cache_dir,
                **self.load_kwargs,
            )

            # Auto-detect column mapping if not explicitly provided
            if self.auto_map_columns and not self._explicit_column_mapping:
                columns = list(self._train_dataset.column_names)
                self.column_mapping = auto_detect_column_mapping(
                    self.dataset_name, columns
                )
                self.dataset_format = detect_dataset_format(columns)

            # Apply column mapping and transform
            if self.column_mapping or self.transform:
                self._train_dataset = self._apply_transforms(self._train_dataset)

            # Load validation dataset
            val_name = self.val_dataset_name or self.dataset_name
            val_split = self.val_split

            # Try to find validation split if not specified
            if val_split is None:
                for try_split in ["validation", "valid", "val", "test"]:
                    try:
                        self._val_dataset = load_dataset(
                            val_name,
                            split=try_split,
                            streaming=self.streaming,
                            trust_remote_code=self.trust_remote_code,
                            cache_dir=self.cache_dir,
                            **self.load_kwargs,
                        )
                        if self.column_mapping or self.transform:
                            self._val_dataset = self._apply_transforms(
                                self._val_dataset
                            )
                        break
                    except (ValueError, KeyError):
                        continue
            else:
                try:
                    self._val_dataset = load_dataset(
                        val_name,
                        split=val_split,
                        streaming=self.streaming,
                        trust_remote_code=self.trust_remote_code,
                        cache_dir=self.cache_dir,
                        **self.load_kwargs,
                    )
                    if self.column_mapping or self.transform:
                        self._val_dataset = self._apply_transforms(self._val_dataset)
                except (ValueError, KeyError):
                    # No validation split available
                    pass

        if stage in (None, "validate") and self._val_dataset is None:
            # Try to load validation data if not already loaded
            val_name = self.val_dataset_name or self.dataset_name
            val_split = self.val_split or "validation"
            try:
                self._val_dataset = load_dataset(
                    val_name,
                    split=val_split,
                    streaming=self.streaming,
                    trust_remote_code=self.trust_remote_code,
                    cache_dir=self.cache_dir,
                    **self.load_kwargs,
                )
                if self.column_mapping or self.transform:
                    self._val_dataset = self._apply_transforms(self._val_dataset)
            except (ValueError, KeyError):
                pass

        self._setup_done = True

    def _apply_transforms(self, dataset: "HFDataset") -> "HFDataset":
        """Apply column mapping and transforms to dataset.

        Args:
            dataset: HuggingFace dataset.

        Returns:
            Transformed dataset.
        """

        def _transform_example(example: Mapping[str, Any]) -> dict[str, Any]:
            # Apply column mapping
            result = dict(example)
            for old_name, new_name in self.column_mapping.items():
                if old_name in result:
                    result[new_name] = result.pop(old_name)

            # Apply custom transform
            if self.transform is not None:
                result = dict(self.transform(result))

            return result

        return dataset.map(_transform_example)

    def train_dataloader(self) -> "DataLoader[Any]":
        """Return training data loader.

        Returns:
            DataLoader for training data.

        Raises:
            RuntimeError: If setup() has not been called.
        """
        from torch.utils.data import DataLoader

        if self._train_dataset is None:
            raise RuntimeError("Training dataset not loaded. Call setup() first.")

        return DataLoader(
            self._train_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=self.shuffle_train and not self.streaming,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> "DataLoader[Any] | None":
        """Return validation data loader.

        Returns:
            DataLoader for validation data, or None if no validation data.
        """
        from torch.utils.data import DataLoader

        if self._val_dataset is None:
            return None

        return DataLoader(
            self._val_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def __repr__(self) -> str:
        return (
            f"HuggingFaceDataModule("
            f"dataset_name='{self.dataset_name}', "
            f"split='{self.split}', "
            f"batch_size={self.batch_size})"
        )
