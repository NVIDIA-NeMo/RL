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
"""Unit tests for DataModule interface."""

import pytest

from nemo_rl.data.module import (
    CombinedDataModule,
    DataModule,
    InMemoryDataModule,
    IterableDataModule,
    MapDataModule,
    create_datamodule,
)


class TestDataModule:
    """Tests for the base DataModule class."""

    def test_datamodule_is_abstract(self):
        """Test that DataModule cannot be instantiated directly."""
        # DataModule is abstract, should not be instantiated
        # But it can be if you don't call abstract methods
        dm = DataModule.__new__(DataModule)
        dm.__init__(batch_size=32)
        assert dm.batch_size == 32

    def test_datamodule_default_values(self):
        """Test default parameter values."""

        class ConcreteDataModule(DataModule):
            def setup(self, stage=None):
                pass

            def train_dataloader(self):
                return None

        dm = ConcreteDataModule()
        assert dm.batch_size == 32
        assert dm.num_workers == 0
        assert dm.pin_memory is True
        assert dm.shuffle_train is True

    def test_datamodule_custom_values(self):
        """Test custom parameter values."""

        class ConcreteDataModule(DataModule):
            def setup(self, stage=None):
                pass

            def train_dataloader(self):
                return None

        dm = ConcreteDataModule(
            batch_size=64,
            num_workers=4,
            pin_memory=False,
            shuffle_train=False,
        )
        assert dm.batch_size == 64
        assert dm.num_workers == 4
        assert dm.pin_memory is False
        assert dm.shuffle_train is False

    def test_datamodule_optional_methods_return_none(self):
        """Test that optional methods return None by default."""

        class ConcreteDataModule(DataModule):
            def setup(self, stage=None):
                pass

            def train_dataloader(self):
                return None

        dm = ConcreteDataModule()
        assert dm.val_dataloader() is None
        assert dm.test_dataloader() is None
        assert dm.predict_dataloader() is None

    def test_datamodule_state_dict(self):
        """Test state_dict returns empty dict by default."""

        class ConcreteDataModule(DataModule):
            def setup(self, stage=None):
                pass

            def train_dataloader(self):
                return None

        dm = ConcreteDataModule()
        assert dm.state_dict() == {}

    def test_datamodule_repr(self):
        """Test string representation."""

        class ConcreteDataModule(DataModule):
            def setup(self, stage=None):
                pass

            def train_dataloader(self):
                return None

        dm = ConcreteDataModule(batch_size=64, num_workers=4)
        repr_str = repr(dm)
        assert "ConcreteDataModule" in repr_str
        assert "batch_size=64" in repr_str
        assert "num_workers=4" in repr_str


class TestIterableDataModule:
    """Tests for IterableDataModule."""

    def test_shuffle_disabled(self):
        """Test that shuffle is disabled for iterable datasets."""

        class ConcreteIterableModule(IterableDataModule):
            def setup(self, stage=None):
                pass

            def train_dataloader(self):
                return None

        dm = ConcreteIterableModule()
        assert dm.shuffle_train is False


class TestMapDataModule:
    """Tests for MapDataModule."""

    def test_shuffle_enabled(self):
        """Test that shuffle is enabled by default for map datasets."""

        class ConcreteMapModule(MapDataModule):
            def setup(self, stage=None):
                pass

            def train_dataloader(self):
                return None

        dm = ConcreteMapModule()
        assert dm.shuffle_train is True


class TestInMemoryDataModule:
    """Tests for InMemoryDataModule."""

    @pytest.fixture
    def sample_train_data(self):
        """Sample training data."""
        return [
            {"prompt": "What is 2+2?", "response": "4"},
            {"prompt": "What is 3+3?", "response": "6"},
            {"prompt": "What is 4+4?", "response": "8"},
        ]

    @pytest.fixture
    def sample_val_data(self):
        """Sample validation data."""
        return [
            {"prompt": "What is 5+5?", "response": "10"},
            {"prompt": "What is 6+6?", "response": "12"},
        ]

    def test_init_with_train_data(self, sample_train_data):
        """Test initialization with training data."""
        dm = InMemoryDataModule(train_data=sample_train_data)
        assert dm._train_data == sample_train_data
        assert dm._val_data is None
        assert dm._test_data is None

    def test_init_with_all_data(self, sample_train_data, sample_val_data):
        """Test initialization with train, val, and test data."""
        test_data = [{"prompt": "test", "response": "test"}]
        dm = InMemoryDataModule(
            train_data=sample_train_data,
            val_data=sample_val_data,
            test_data=test_data,
        )
        assert dm._train_data == sample_train_data
        assert dm._val_data == sample_val_data
        assert dm._test_data == test_data

    def test_setup_creates_datasets(self, sample_train_data, sample_val_data):
        """Test that setup creates datasets from data."""
        dm = InMemoryDataModule(
            train_data=sample_train_data,
            val_data=sample_val_data,
        )
        dm.setup()

        assert dm._train_dataset is not None
        assert dm._val_dataset is not None
        assert len(dm._train_dataset) == 3
        assert len(dm._val_dataset) == 2

    def test_setup_with_stage_fit(self, sample_train_data, sample_val_data):
        """Test setup with stage='fit'."""
        dm = InMemoryDataModule(
            train_data=sample_train_data,
            val_data=sample_val_data,
        )
        dm.setup(stage="fit")

        assert dm._train_dataset is not None
        assert dm._val_dataset is not None

    def test_setup_with_stage_test(self, sample_train_data):
        """Test setup with stage='test'."""
        test_data = [{"prompt": "test", "response": "test"}]
        dm = InMemoryDataModule(
            train_data=sample_train_data,
            test_data=test_data,
        )
        dm.setup(stage="test")

        assert dm._test_dataset is not None
        assert dm._train_dataset is None  # Not set for stage='test'

    def test_train_dataloader(self, sample_train_data):
        """Test train_dataloader returns DataLoader."""
        dm = InMemoryDataModule(train_data=sample_train_data, batch_size=2)
        dm.setup()

        loader = dm.train_dataloader()
        assert loader is not None
        assert loader.batch_size == 2

        # Check we can iterate
        batches = list(loader)
        assert len(batches) == 2  # 3 items / batch_size 2 = 2 batches

    def test_train_dataloader_raises_without_setup(self):
        """Test train_dataloader raises error if setup not called."""
        dm = InMemoryDataModule(train_data=None)

        with pytest.raises(RuntimeError, match="No training dataset"):
            dm.train_dataloader()

    def test_val_dataloader(self, sample_train_data, sample_val_data):
        """Test val_dataloader returns DataLoader."""
        dm = InMemoryDataModule(
            train_data=sample_train_data,
            val_data=sample_val_data,
            batch_size=2,
        )
        dm.setup()

        loader = dm.val_dataloader()
        assert loader is not None
        assert loader.batch_size == 2

    def test_val_dataloader_none_without_data(self, sample_train_data):
        """Test val_dataloader returns None if no val data."""
        dm = InMemoryDataModule(train_data=sample_train_data)
        dm.setup()

        assert dm.val_dataloader() is None

    def test_test_dataloader(self, sample_train_data):
        """Test test_dataloader returns DataLoader."""
        test_data = [{"prompt": "test", "response": "test"}]
        dm = InMemoryDataModule(
            train_data=sample_train_data,
            test_data=test_data,
        )
        dm.setup()

        loader = dm.test_dataloader()
        assert loader is not None

    def test_custom_collate_fn(self, sample_train_data):
        """Test custom collate function is used."""
        call_count = {"count": 0}

        def custom_collate(batch):
            call_count["count"] += 1
            return batch

        dm = InMemoryDataModule(
            train_data=sample_train_data,
            batch_size=2,
            collate_fn=custom_collate,
        )
        dm.setup()

        loader = dm.train_dataloader()
        list(loader)  # Consume all batches

        assert call_count["count"] > 0


class TestCombinedDataModule:
    """Tests for CombinedDataModule."""

    @pytest.fixture
    def datamodule1(self):
        """First data module."""
        return InMemoryDataModule(
            train_data=[{"text": "a"}, {"text": "b"}],
            val_data=[{"text": "val1"}],
        )

    @pytest.fixture
    def datamodule2(self):
        """Second data module."""
        return InMemoryDataModule(
            train_data=[{"text": "c"}, {"text": "d"}],
            val_data=[{"text": "val2"}],
        )

    def test_init_with_datamodules(self, datamodule1, datamodule2):
        """Test initialization with multiple datamodules."""
        combined = CombinedDataModule([datamodule1, datamodule2])
        assert len(combined.datamodules) == 2

    def test_init_with_weights(self, datamodule1, datamodule2):
        """Test initialization with sampling weights."""
        combined = CombinedDataModule(
            [datamodule1, datamodule2],
            weights=[0.7, 0.3],
        )
        assert combined._weights == [0.7, 0.3]

    def test_init_weights_mismatch_raises(self, datamodule1, datamodule2):
        """Test that mismatched weights raise error."""
        with pytest.raises(ValueError, match="Number of weights"):
            CombinedDataModule(
                [datamodule1, datamodule2],
                weights=[0.5],  # Only one weight for two datamodules
            )

    def test_setup_calls_all_submodules(self, datamodule1, datamodule2):
        """Test that setup calls setup on all sub-modules."""
        combined = CombinedDataModule([datamodule1, datamodule2])
        combined.setup()

        assert datamodule1._setup_done
        assert datamodule2._setup_done

    def test_prepare_data_calls_all_submodules(self, datamodule1, datamodule2):
        """Test that prepare_data calls prepare_data on all sub-modules."""
        # Track calls
        calls = {"dm1": False, "dm2": False}

        def mock_prepare1():
            calls["dm1"] = True

        def mock_prepare2():
            calls["dm2"] = True

        datamodule1.prepare_data = mock_prepare1
        datamodule2.prepare_data = mock_prepare2

        combined = CombinedDataModule([datamodule1, datamodule2])
        combined.prepare_data()

        assert calls["dm1"]
        assert calls["dm2"]

    def test_teardown_calls_all_submodules(self, datamodule1, datamodule2):
        """Test that teardown calls teardown on all sub-modules."""
        calls = {"dm1": False, "dm2": False}

        def mock_teardown1(stage=None):
            calls["dm1"] = True

        def mock_teardown2(stage=None):
            calls["dm2"] = True

        datamodule1.teardown = mock_teardown1
        datamodule2.teardown = mock_teardown2

        combined = CombinedDataModule([datamodule1, datamodule2])
        combined.teardown()

        assert calls["dm1"]
        assert calls["dm2"]

    def test_state_dict(self, datamodule1, datamodule2):
        """Test state_dict combines all sub-module states."""
        datamodule1.state_dict = lambda: {"key1": "value1"}
        datamodule2.state_dict = lambda: {"key2": "value2"}

        combined = CombinedDataModule([datamodule1, datamodule2])
        state = combined.state_dict()

        assert "datamodule_0" in state
        assert "datamodule_1" in state
        assert state["datamodule_0"] == {"key1": "value1"}
        assert state["datamodule_1"] == {"key2": "value2"}


class TestCreateDatamodule:
    """Tests for create_datamodule factory function."""

    def test_create_from_list(self):
        """Test creating datamodule from list."""
        data = [{"text": "hello"}, {"text": "world"}]
        dm = create_datamodule(train_data=data)

        assert isinstance(dm, InMemoryDataModule)
        dm.setup()
        assert dm.train_dataloader() is not None

    def test_create_from_list_with_val(self):
        """Test creating datamodule from lists with validation."""
        train_data = [{"text": "train1"}, {"text": "train2"}]
        val_data = [{"text": "val1"}]

        dm = create_datamodule(train_data=train_data, val_data=val_data)

        assert isinstance(dm, InMemoryDataModule)
        dm.setup()
        assert dm.val_dataloader() is not None

    def test_create_with_custom_batch_size(self):
        """Test creating datamodule with custom batch size."""
        data = [{"text": "a"}, {"text": "b"}, {"text": "c"}, {"text": "d"}]
        dm = create_datamodule(train_data=data, batch_size=2)

        dm.setup()
        loader = dm.train_dataloader()
        assert loader.batch_size == 2

    def test_create_from_tuple(self):
        """Test creating datamodule from tuple."""
        data = ({"text": "hello"}, {"text": "world"})
        dm = create_datamodule(train_data=data)

        assert isinstance(dm, InMemoryDataModule)

    def test_create_from_invalid_type_raises(self):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Cannot create DataModule"):
            create_datamodule(train_data=12345)  # type: ignore


class TestDataModuleWithPyTorch:
    """Tests that require PyTorch Dataset/DataLoader."""

    def test_create_from_dataset(self):
        """Test creating datamodule from PyTorch Dataset."""
        from torch.utils.data import Dataset

        class SimpleDataset(Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {"value": idx}

        ds = SimpleDataset()
        dm = create_datamodule(train_data=ds)

        dm.setup()
        loader = dm.train_dataloader()
        assert loader is not None

        # Verify we get data
        batch = next(iter(loader))
        assert "value" in batch

    def test_create_from_dataloader(self):
        """Test creating datamodule from PyTorch DataLoader."""
        from torch.utils.data import DataLoader, Dataset

        class SimpleDataset(Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {"value": idx}

        loader = DataLoader(SimpleDataset(), batch_size=5)
        dm = create_datamodule(train_data=loader)

        dm.setup()
        train_loader = dm.train_dataloader()
        assert train_loader is loader


class TestHuggingFaceColumnMapping:
    """Tests for HuggingFace auto-detection and column mapping."""

    def test_detect_dataset_format_chat(self):
        """Test detecting chat/conversation format."""
        from nemo_rl.data.module_hf import detect_dataset_format

        columns = ["messages", "id", "source"]
        assert detect_dataset_format(columns) == "chat"

    def test_detect_dataset_format_completion(self):
        """Test detecting instruction/completion format."""
        from nemo_rl.data.module_hf import detect_dataset_format

        columns = ["prompt", "response", "id"]
        assert detect_dataset_format(columns) == "completion"
        
        columns = ["question", "answer", "id"]
        assert detect_dataset_format(columns) == "completion"
        
        columns = ["instruction", "output", "id"]
        assert detect_dataset_format(columns) == "completion"

    def test_detect_dataset_format_preference(self):
        """Test detecting preference (DPO) format."""
        from nemo_rl.data.module_hf import detect_dataset_format

        columns = ["prompt", "chosen", "rejected"]
        assert detect_dataset_format(columns) == "preference"

    def test_detect_dataset_format_text(self):
        """Test detecting raw text format."""
        from nemo_rl.data.module_hf import detect_dataset_format

        columns = ["text", "id"]
        assert detect_dataset_format(columns) == "text"

    def test_detect_dataset_format_unknown(self):
        """Test detecting unknown format."""
        from nemo_rl.data.module_hf import detect_dataset_format

        columns = ["foo", "bar", "baz"]
        assert detect_dataset_format(columns) == "unknown"

    def test_auto_detect_column_mapping_known_dataset(self):
        """Test auto-detection for known datasets."""
        from nemo_rl.data.module_hf import auto_detect_column_mapping

        columns = ["problem", "generated_solution", "expected_answer", "id"]
        mapping = auto_detect_column_mapping(
            "nvidia/OpenMathInstruct-2", columns
        )
        
        assert mapping.get("problem") == "prompt"
        assert mapping.get("generated_solution") == "response"
        assert mapping.get("expected_answer") == "answer"

    def test_auto_detect_column_mapping_pattern_fallback(self):
        """Test auto-detection falls back to pattern matching."""
        from nemo_rl.data.module_hf import auto_detect_column_mapping

        columns = ["question", "answer", "id"]
        mapping = auto_detect_column_mapping("unknown/dataset", columns)
        
        assert mapping.get("question") == "prompt"
        assert mapping.get("answer") == "response"

    def test_auto_detect_column_mapping_no_remap_if_target_exists(self):
        """Test that columns already with target name are not remapped."""
        from nemo_rl.data.module_hf import auto_detect_column_mapping

        # "prompt" column already has the target name, shouldn't be remapped
        columns = ["prompt", "output", "id"]
        mapping = auto_detect_column_mapping("unknown/dataset", columns)
        
        assert "prompt" not in mapping  # Should not remap prompt -> prompt
        assert mapping.get("output") == "response"

    def test_known_dataset_mappings_exist(self):
        """Test that known dataset mappings are defined."""
        from nemo_rl.data.module_hf import KNOWN_DATASET_MAPPINGS

        assert "nvidia/OpenMathInstruct-2" in KNOWN_DATASET_MAPPINGS
        assert "tatsu-lab/alpaca" in KNOWN_DATASET_MAPPINGS
        assert "Anthropic/hh-rlhf" in KNOWN_DATASET_MAPPINGS


class TestHuggingFaceDataModule:
    """Tests for HuggingFaceDataModule class."""

    def test_huggingface_datamodule_init(self):
        """Test HuggingFaceDataModule initialization."""
        from nemo_rl.data.module_hf import HuggingFaceDataModule

        dm = HuggingFaceDataModule(
            dataset_name="test/dataset",
            split="train",
            batch_size=16,
        )
        
        assert dm.dataset_name == "test/dataset"
        assert dm.split == "train"
        assert dm.batch_size == 16
        assert dm.auto_map_columns is True

    def test_huggingface_datamodule_with_explicit_mapping(self):
        """Test HuggingFaceDataModule with explicit column mapping."""
        from nemo_rl.data.module_hf import HuggingFaceDataModule

        mapping = {"input": "prompt", "output": "response"}
        dm = HuggingFaceDataModule(
            dataset_name="test/dataset",
            column_mapping=mapping,
        )
        
        assert dm._explicit_column_mapping == mapping
        assert dm.column_mapping == mapping

    def test_huggingface_datamodule_auto_map_disabled(self):
        """Test HuggingFaceDataModule with auto_map_columns disabled."""
        from nemo_rl.data.module_hf import HuggingFaceDataModule

        dm = HuggingFaceDataModule(
            dataset_name="test/dataset",
            auto_map_columns=False,
        )
        
        assert dm.auto_map_columns is False
        assert dm.column_mapping == {}

    def test_huggingface_datamodule_repr(self):
        """Test HuggingFaceDataModule string representation."""
        from nemo_rl.data.module_hf import HuggingFaceDataModule

        dm = HuggingFaceDataModule(
            dataset_name="nvidia/OpenMathInstruct-2",
            batch_size=32,
        )
        
        repr_str = repr(dm)
        assert "HuggingFaceDataModule" in repr_str
        assert "nvidia/OpenMathInstruct-2" in repr_str
        assert "batch_size=32" in repr_str

    def test_create_datamodule_from_string(self):
        """Test create_datamodule with string creates HuggingFaceDataModule."""
        # This tests the integration with create_datamodule factory
        # Note: actual HF loading would require network access
        from nemo_rl.data.module import create_datamodule
        
        try:
            dm = create_datamodule(train_data="test/dataset")
            from nemo_rl.data.module_hf import HuggingFaceDataModule
            assert isinstance(dm, HuggingFaceDataModule)
        except ImportError:
            pytest.skip("HuggingFace datasets package not installed")


class TestTrainerDatasetIntegration:
    """Tests for trainer.fit(dataset=...) integration."""

    def test_trainer_fit_accepts_dataset_string(self):
        """Test that BaseTrainer.fit accepts dataset parameter."""
        from nemo_rl.trainers.base import BaseTrainer
        import inspect
        
        # Check that fit method has dataset parameter
        sig = inspect.signature(BaseTrainer.fit)
        params = list(sig.parameters.keys())
        
        assert "dataset" in params
        assert "train_data" in params
        assert "datamodule" in params

    def test_trainer_fit_dataset_doc(self):
        """Test that fit method documents the dataset parameter."""
        from nemo_rl.trainers.base import BaseTrainer
        
        doc = BaseTrainer.fit.__doc__
        assert "dataset" in doc
        assert "HuggingFace" in doc
