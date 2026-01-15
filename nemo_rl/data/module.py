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
"""Unified DataModule interface for providing data to all algorithms.

This module provides a single, consistent way to supply data to NeMo RL trainers.
It supports both iterable and map-style datasets, and integrates with
HuggingFace datasets for zero-configuration data loading.

Example usage:
    >>> from nemo_rl.data import DataModule
    >>> 
    >>> # Create custom data module
    >>> class MyDataModule(DataModule):
    ...     def setup(self, stage=None):
    ...         self.train_data = MyDataset(...)
    ...     
    ...     def train_dataloader(self):
    ...         return DataLoader(self.train_data, batch_size=32)
    >>> 
    >>> # Pass to trainer
    >>> trainer.fit(datamodule=MyDataModule())

    >>> # Or use pre-built data modules
    >>> from nemo_rl.data import HuggingFaceDataModule
    >>> datamodule = HuggingFaceDataModule("nvidia/OpenMathInstruct-2")
    >>> trainer.fit(datamodule=datamodule)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Mapping,
    Sequence,
    TypeVar,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader, Dataset, IterableDataset


T = TypeVar("T")


class DataModule(ABC):
    """Base interface for providing data to NeMo RL trainers.

    DataModule provides a unified interface for all data operations required
    by training algorithms. It handles:
    - Data preparation and setup
    - Training data loading
    - Validation data loading
    - Test data loading (optional)

    Subclasses should implement the abstract methods to provide data in the
    format expected by the trainer.

    Attributes:
        batch_size: Default batch size for data loaders.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.
        shuffle_train: Whether to shuffle training data.

    Example:
        >>> class MyDataModule(DataModule):
        ...     def __init__(self, data_path: str):
        ...         super().__init__()
        ...         self.data_path = data_path
        ...
        ...     def setup(self, stage=None):
        ...         self.train_data = load_data(self.data_path)
        ...
        ...     def train_dataloader(self):
        ...         return DataLoader(self.train_data, batch_size=self.batch_size)
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        shuffle_train: bool = True,
    ):
        """Initialize the DataModule.

        Args:
            batch_size: Default batch size for data loaders.
            num_workers: Number of worker processes for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            shuffle_train: Whether to shuffle training data.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self._setup_done = False

    def prepare_data(self) -> None:
        """Download or prepare data (called only on main process).

        This method is called only once and on a single process.
        Use it for downloading data or any preparation that should
        not be done in parallel.

        Override this method if you need to download or prepare data
        before setup is called.
        """
        pass

    @abstractmethod
    def setup(self, stage: str | None = None) -> None:
        """Set up data for training/validation/testing.

        This method is called on every process when using distributed
        training. Use it to load datasets, apply transforms, and prepare
        data for the training stage.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
                   If None, setup for all stages.
        """
        pass

    @abstractmethod
    def train_dataloader(self) -> "DataLoader[T]":
        """Return the training data loader.

        Returns:
            DataLoader for training data.

        Raises:
            RuntimeError: If setup() has not been called.
        """
        pass

    def val_dataloader(self) -> "DataLoader[T] | None":
        """Return the validation data loader.

        Returns:
            DataLoader for validation data, or None if no validation data.
        """
        return None

    def test_dataloader(self) -> "DataLoader[T] | None":
        """Return the test data loader.

        Returns:
            DataLoader for test data, or None if no test data.
        """
        return None

    def predict_dataloader(self) -> "DataLoader[T] | None":
        """Return the prediction data loader.

        Returns:
            DataLoader for prediction data, or None if no prediction data.
        """
        return None

    def teardown(self, stage: str | None = None) -> None:
        """Clean up after training/validation/testing.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        pass

    def state_dict(self) -> dict[str, Any]:
        """Return state dict for checkpointing.

        Override this method to save any state that should be
        restored when resuming training.

        Returns:
            Dictionary containing state to checkpoint.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from checkpoint.

        Args:
            state_dict: State dictionary from checkpoint.
        """
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"batch_size={self.batch_size}, "
            f"num_workers={self.num_workers})"
        )


class IterableDataModule(DataModule):
    """DataModule for iterable datasets.

    Use this when your dataset is an IterableDataset that doesn't
    support random access indexing.

    Example:
        >>> class StreamingDataModule(IterableDataModule):
        ...     def setup(self, stage=None):
        ...         self.train_data = StreamingDataset(url="...")
        ...
        ...     def train_dataloader(self):
        ...         return DataLoader(self.train_data, batch_size=32)
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        """Initialize the IterableDataModule.

        Note: shuffle_train is not supported for iterable datasets
        as they don't support random access.

        Args:
            batch_size: Default batch size for data loaders.
            num_workers: Number of worker processes for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
        """
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle_train=False,  # Iterable datasets don't support shuffle
        )


class MapDataModule(DataModule):
    """DataModule for map-style datasets.

    Use this when your dataset supports random access indexing
    via __getitem__ and __len__.

    Example:
        >>> class JSONDataModule(MapDataModule):
        ...     def setup(self, stage=None):
        ...         self.train_data = JSONDataset("train.json")
        ...         self.val_data = JSONDataset("val.json")
        ...
        ...     def train_dataloader(self):
        ...         return DataLoader(
        ...             self.train_data,
        ...             batch_size=32,
        ...             shuffle=self.shuffle_train
        ...         )
        ...
        ...     def val_dataloader(self):
        ...         return DataLoader(self.val_data, batch_size=32)
    """

    pass


class InMemoryDataModule(MapDataModule):
    """DataModule that holds data in memory.

    Convenient for small datasets that can be loaded entirely into memory.

    Example:
        >>> datamodule = InMemoryDataModule(
        ...     train_data=[{"prompt": "Hello", "response": "Hi"}],
        ...     val_data=[{"prompt": "Bye", "response": "Goodbye"}],
        ... )
    """

    def __init__(
        self,
        train_data: Sequence[Mapping[str, Any]] | None = None,
        val_data: Sequence[Mapping[str, Any]] | None = None,
        test_data: Sequence[Mapping[str, Any]] | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        shuffle_train: bool = True,
        collate_fn: Callable[[list[Any]], Any] | None = None,
    ):
        """Initialize the InMemoryDataModule.

        Args:
            train_data: Training data as a sequence of dictionaries.
            val_data: Validation data as a sequence of dictionaries.
            test_data: Test data as a sequence of dictionaries.
            batch_size: Default batch size for data loaders.
            num_workers: Number of worker processes for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            shuffle_train: Whether to shuffle training data.
            collate_fn: Custom collate function for batching.
        """
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle_train=shuffle_train,
        )
        self._train_data = train_data
        self._val_data = val_data
        self._test_data = test_data
        self._collate_fn = collate_fn

        # Datasets will be created in setup()
        self._train_dataset: "Dataset[Any] | None" = None
        self._val_dataset: "Dataset[Any] | None" = None
        self._test_dataset: "Dataset[Any] | None" = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets from in-memory data.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        from torch.utils.data import Dataset

        class ListDataset(Dataset):
            """Simple dataset wrapping a list."""

            def __init__(self, data: Sequence[Mapping[str, Any]]):
                self.data = list(data)

            def __len__(self) -> int:
                return len(self.data)

            def __getitem__(self, idx: int) -> Mapping[str, Any]:
                return self.data[idx]

        if stage in (None, "fit"):
            if self._train_data is not None:
                self._train_dataset = ListDataset(self._train_data)
            if self._val_data is not None:
                self._val_dataset = ListDataset(self._val_data)

        if stage in (None, "validate"):
            if self._val_data is not None:
                self._val_dataset = ListDataset(self._val_data)

        if stage in (None, "test"):
            if self._test_data is not None:
                self._test_dataset = ListDataset(self._test_data)

        self._setup_done = True

    def train_dataloader(self) -> "DataLoader[Any]":
        """Return training data loader.

        Returns:
            DataLoader for training data.

        Raises:
            RuntimeError: If setup() has not been called or no training data.
        """
        from torch.utils.data import DataLoader

        if self._train_dataset is None:
            raise RuntimeError(
                "No training dataset. Call setup() first or provide train_data."
            )

        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
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
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> "DataLoader[Any] | None":
        """Return test data loader.

        Returns:
            DataLoader for test data, or None if no test data.
        """
        from torch.utils.data import DataLoader

        if self._test_dataset is None:
            return None

        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
        )


class CombinedDataModule(DataModule):
    """DataModule that combines multiple DataModules.

    Useful for multi-task training or combining different data sources.

    Example:
        >>> combined = CombinedDataModule([
        ...     MathDataModule(),
        ...     CodeDataModule(),
        ... ])
        >>> trainer.fit(datamodule=combined)
    """

    def __init__(
        self,
        datamodules: Sequence[DataModule],
        weights: Sequence[float] | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        """Initialize the CombinedDataModule.

        Args:
            datamodules: List of DataModules to combine.
            weights: Optional sampling weights for each DataModule.
            batch_size: Default batch size for data loaders.
            num_workers: Number of worker processes for data loading.
        """
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self._datamodules = list(datamodules)
        self._weights = weights

        if weights is not None and len(weights) != len(datamodules):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of datamodules ({len(datamodules)})"
            )

    @property
    def datamodules(self) -> list[DataModule]:
        """Return the list of combined DataModules."""
        return self._datamodules

    def prepare_data(self) -> None:
        """Prepare data for all sub-modules."""
        for dm in self._datamodules:
            dm.prepare_data()

    def setup(self, stage: str | None = None) -> None:
        """Set up all sub-modules.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        for dm in self._datamodules:
            dm.setup(stage)
        self._setup_done = True

    def train_dataloader(self) -> "DataLoader[Any]":
        """Return combined training data loader.

        Returns:
            DataLoader that samples from all training data.

        Raises:
            RuntimeError: If setup() has not been called.
        """
        from torch.utils.data import DataLoader

        # Get all train dataloaders
        train_loaders = [dm.train_dataloader() for dm in self._datamodules]

        # Combine using a wrapper
        combined_dataset = _CombinedIterableDataset(
            [loader.dataset for loader in train_loaders],
            weights=self._weights,
        )

        return DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> "DataLoader[Any] | None":
        """Return validation data loaders.

        Returns:
            List of validation DataLoaders, or None if no validation data.
        """
        val_loaders = [dm.val_dataloader() for dm in self._datamodules]
        val_loaders = [vl for vl in val_loaders if vl is not None]

        if not val_loaders:
            return None

        # Return first validation loader (common case)
        # For multi-task validation, users should iterate over datamodules
        return val_loaders[0]

    def teardown(self, stage: str | None = None) -> None:
        """Tear down all sub-modules.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        for dm in self._datamodules:
            dm.teardown(stage)

    def state_dict(self) -> dict[str, Any]:
        """Return combined state dict.

        Returns:
            Dictionary containing state from all sub-modules.
        """
        return {
            f"datamodule_{i}": dm.state_dict()
            for i, dm in enumerate(self._datamodules)
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state for all sub-modules.

        Args:
            state_dict: Combined state dictionary.
        """
        for i, dm in enumerate(self._datamodules):
            key = f"datamodule_{i}"
            if key in state_dict:
                dm.load_state_dict(state_dict[key])


class _CombinedIterableDataset:
    """Internal class for combining multiple datasets."""

    def __init__(
        self,
        datasets: Sequence[Any],
        weights: Sequence[float] | None = None,
    ):
        self.datasets = list(datasets)
        self.weights = weights or [1.0] * len(datasets)

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def __iter__(self) -> Iterator[Any]:
        import random

        iterators = [iter(ds) for ds in self.datasets]
        exhausted = [False] * len(self.datasets)

        while not all(exhausted):
            # Sample dataset according to weights
            idx = random.choices(range(len(self.datasets)), weights=self.weights)[0]

            if exhausted[idx]:
                # Find next non-exhausted dataset
                for i, ex in enumerate(exhausted):
                    if not ex:
                        idx = i
                        break
                else:
                    break

            try:
                yield next(iterators[idx])
            except StopIteration:
                exhausted[idx] = True


def create_datamodule(
    train_data: (
        str
        | Sequence[Mapping[str, Any]]
        | "Dataset[Any]"
        | "DataLoader[Any]"
        | None
    ) = None,
    val_data: (
        str
        | Sequence[Mapping[str, Any]]
        | "Dataset[Any]"
        | "DataLoader[Any]"
        | None
    ) = None,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
    **kwargs: Any,
) -> DataModule:
    """Factory function to create a DataModule from various input types.

    This is a convenience function that automatically creates the appropriate
    DataModule based on the input type.

    Args:
        train_data: Training data. Can be:
            - str: HuggingFace dataset name or local path
            - Sequence[dict]: List of data dictionaries
            - Dataset: PyTorch Dataset object
            - DataLoader: PyTorch DataLoader object
        val_data: Validation data (same types as train_data).
        batch_size: Batch size for data loaders.
        num_workers: Number of data loading workers.
        shuffle_train: Whether to shuffle training data.
        **kwargs: Additional arguments passed to the DataModule.

    Returns:
        Appropriate DataModule instance.

    Example:
        >>> # From list
        >>> dm = create_datamodule([{"text": "hello"}])
        >>> 
        >>> # From HuggingFace dataset name
        >>> dm = create_datamodule("nvidia/OpenMathInstruct-2")
    """
    # Handle list/sequence input
    if isinstance(train_data, (list, tuple)):
        return InMemoryDataModule(
            train_data=train_data,
            val_data=val_data if isinstance(val_data, (list, tuple)) else None,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle_train=shuffle_train,
            **kwargs,
        )

    # Handle string input (HuggingFace dataset name or path)
    if isinstance(train_data, str):
        # Lazy import to avoid dependency on datasets
        try:
            from nemo_rl.data.module_hf import HuggingFaceDataModule

            return HuggingFaceDataModule(
                dataset_name=train_data,
                val_dataset_name=val_data if isinstance(val_data, str) else None,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle_train=shuffle_train,
                **kwargs,
            )
        except ImportError:
            raise ImportError(
                "HuggingFace datasets support requires 'datasets' package. "
                "Install with: pip install datasets"
            )

    # Handle Dataset input
    try:
        from torch.utils.data import DataLoader, Dataset

        if isinstance(train_data, Dataset):

            class _DatasetDataModule(MapDataModule):
                def __init__(
                    self,
                    train_ds: "Dataset[Any]",
                    val_ds: "Dataset[Any] | None",
                    batch_size: int,
                    num_workers: int,
                    shuffle_train: bool,
                ):
                    super().__init__(
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle_train=shuffle_train,
                    )
                    self._train_ds = train_ds
                    self._val_ds = val_ds

                def setup(self, stage: str | None = None) -> None:
                    self._setup_done = True

                def train_dataloader(self) -> "DataLoader[Any]":
                    return DataLoader(
                        self._train_ds,
                        batch_size=self.batch_size,
                        shuffle=self.shuffle_train,
                        num_workers=self.num_workers,
                    )

                def val_dataloader(self) -> "DataLoader[Any] | None":
                    if self._val_ds is None:
                        return None
                    return DataLoader(
                        self._val_ds,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=self.num_workers,
                    )

            return _DatasetDataModule(
                train_ds=train_data,
                val_ds=val_data if isinstance(val_data, Dataset) else None,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle_train=shuffle_train,
            )

        # Handle DataLoader input
        if isinstance(train_data, DataLoader):

            class _DataLoaderDataModule(DataModule):
                def __init__(
                    self,
                    train_loader: "DataLoader[Any]",
                    val_loader: "DataLoader[Any] | None",
                ):
                    super().__init__(batch_size=train_loader.batch_size or 1)
                    self._train_loader = train_loader
                    self._val_loader = val_loader

                def setup(self, stage: str | None = None) -> None:
                    self._setup_done = True

                def train_dataloader(self) -> "DataLoader[Any]":
                    return self._train_loader

                def val_dataloader(self) -> "DataLoader[Any] | None":
                    return self._val_loader

            return _DataLoaderDataModule(
                train_loader=train_data,
                val_loader=val_data if isinstance(val_data, DataLoader) else None,
            )

    except ImportError:
        pass

    raise TypeError(
        f"Cannot create DataModule from type {type(train_data)}. "
        "Supported types: str, list, Dataset, DataLoader"
    )
