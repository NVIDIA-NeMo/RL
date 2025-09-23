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
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest
import torch

# Skip entire module if nemo_automodel is not available
pytest_plugins = []
try:
    import nemo_automodel  # noqa: F401
except ImportError:
    pytest.skip("nemo_automodel not available", allow_module_level=True)

from nemo_rl.utils.automodel_checkpoint import (
    detect_checkpoint_format,
    load_checkpoint,
    save_checkpoint,
)


class TestModel(torch.nn.Module):
    """Simple test model with a forward method."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(4, 4),
                torch.nn.LayerNorm(4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 1),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.fixture
def mock_model():
    """Create a simple mock model for testing."""
    return TestModel()


@pytest.fixture
def mock_optimizer():
    """Create a simple mock optimizer for testing."""
    model = torch.nn.Linear(4, 1)
    return torch.optim.Adam(model.parameters())


@pytest.mark.automodel
class TestDetectCheckpointFormat:
    """Test the detect_checkpoint_format function."""

    def test_directory_with_safetensors(self):
        """Test detection for directories containing safetensors files."""
        with TemporaryDirectory() as tmp_dir:
            # Create directory with safetensors files
            os.makedirs(os.path.join(tmp_dir, "weights", "model"))
            weights_path = os.path.join(tmp_dir, "weights", "model")

            # Create safetensors shard files
            with open(
                os.path.join(
                    weights_path, "shard-00001-model-00001-of-00001.safetensors"
                ),
                "w",
            ) as f:
                f.write("dummy content")
            with open(
                os.path.join(
                    weights_path, "shard-00002-model-00001-of-00001.safetensors"
                ),
                "w",
            ) as f:
                f.write("dummy content")

            format_type, is_peft = detect_checkpoint_format(weights_path)
            assert format_type == "safetensors"
            assert is_peft == False

    def test_directory_with_dcp_format(self):
        """Test detection for directories with DCP (Distributed Checkpoint) format."""
        with TemporaryDirectory() as tmp_dir:
            # Create directory structure like: step_3/policy/optimizer/optim
            optim_path = os.path.join(tmp_dir, "step_3", "policy", "optimizer", "optim")
            os.makedirs(optim_path)

            # Create DCP files (.distcp + .metadata)
            with open(os.path.join(optim_path, "__0_0.distcp"), "w") as f:
                f.write("dummy dcp content")
            with open(os.path.join(optim_path, "__1_0.distcp"), "w") as f:
                f.write("dummy dcp content")
            with open(os.path.join(optim_path, ".metadata"), "w") as f:
                f.write("dummy metadata")

            format_type, is_peft = detect_checkpoint_format(optim_path)
            assert format_type == "torch_save"  # DCP uses torch_save format
            assert is_peft == False

    def test_directory_with_torch_files(self):
        """Test detection for directories containing torch save files."""
        with TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model")
            os.makedirs(model_path)

            # Create torch save files
            with open(os.path.join(model_path, "pytorch_model.bin"), "w") as f:
                f.write("dummy content")

            format_type, is_peft = detect_checkpoint_format(model_path)
            assert format_type == "torch_save"
            assert is_peft == False

    def test_peft_detection_in_filenames(self):
        """Test PEFT detection from filenames within directories."""
        with TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "regular_model")
            os.makedirs(model_path)

            # Create file with adapter pattern in name
            with open(os.path.join(model_path, "adapter_model.safetensors"), "w") as f:
                f.write("dummy content")

            format_type, is_peft = detect_checkpoint_format(model_path)
            assert format_type == "safetensors"
            assert is_peft == True  # Should detect adapter in filename

    def test_default_fallback(self):
        """Test default behavior for non-existent directories."""
        # Non-existent directory should default to safetensors, no PEFT
        format_type, is_peft = detect_checkpoint_format("/non/existent/directory")
        assert format_type == "safetensors"
        assert is_peft == False

    def test_expected_structure(self):
        """Test with the expected folder structure from the user."""
        with TemporaryDirectory() as tmp_dir:
            # Create the expected structure: step_3/policy/weights/model
            weights_path = os.path.join(tmp_dir, "step_3", "policy", "weights", "model")
            os.makedirs(weights_path)

            # Create safetensors shard files as in the example
            with open(
                os.path.join(
                    weights_path, "shard-00001-model-00001-of-00001.safetensors"
                ),
                "w",
            ) as f:
                f.write("dummy content")
            with open(
                os.path.join(
                    weights_path, "shard-00002-model-00001-of-00001.safetensors"
                ),
                "w",
            ) as f:
                f.write("dummy content")

            format_type, is_peft = detect_checkpoint_format(weights_path)
            assert format_type == "safetensors"
            assert is_peft == False

    """Test the save_checkpoint function."""

    @pytest.mark.automodel
    @patch("nemo_rl.utils.automodel_checkpoint.save_model")
    @patch("nemo_rl.utils.automodel_checkpoint.save_optimizer")
    def test_save_model_only(self, mock_save_optimizer, mock_save_model, mock_model):
        """Test saving model weights only."""
        with TemporaryDirectory() as tmp_dir:
            weights_path = os.path.join(tmp_dir, "weights")
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)

            # Save checkpoint
            save_checkpoint(
                model=mock_model,
                weights_path=weights_path,
                model_save_format="safetensors",
                is_peft=False,
            )

            # Verify save_model was called correctly
            mock_save_model.assert_called_once()
            call_args = mock_save_model.call_args
            assert call_args[1]["model"] is mock_model
            assert call_args[1]["weights_path"] == weights_path
            assert (
                call_args[1]["checkpoint_config"].model_save_format.value
                == "safetensors"
            )
            assert call_args[1]["checkpoint_config"].is_peft == False

            # Verify optimizer saving was not called
            mock_save_optimizer.assert_not_called()

    @pytest.mark.automodel
    @patch("nemo_rl.utils.automodel_checkpoint.save_model")
    @patch("nemo_rl.utils.automodel_checkpoint.save_optimizer")
    def test_save_with_optimizer(
        self, mock_save_optimizer, mock_save_model, mock_model, mock_optimizer
    ):
        """Test saving model and optimizer weights."""
        with TemporaryDirectory() as tmp_dir:
            weights_path = os.path.join(tmp_dir, "model", "weights")
            optimizer_path = os.path.join(tmp_dir, "optimizer", "optim")
            os.makedirs(os.path.dirname(weights_path))
            os.makedirs(os.path.dirname(optimizer_path))

            # Save checkpoint with optimizer
            save_checkpoint(
                model=mock_model,
                weights_path=weights_path,
                optimizer=mock_optimizer,
                optimizer_path=optimizer_path,
                model_save_format="torch_save",
                is_peft=True,
            )

            # Verify both model and optimizer saving were called
            mock_save_model.assert_called_once()
            mock_save_optimizer.assert_called_once()

            # Check optimizer call args
            opt_call_args = mock_save_optimizer.call_args
            assert opt_call_args[1]["optimizer"] is mock_optimizer
            assert opt_call_args[1]["model"] is mock_model
            assert opt_call_args[1]["weights_path"] == optimizer_path

    @pytest.mark.automodel
    @patch("nemo_rl.utils.automodel_checkpoint.save_model")
    def test_save_with_tokenizer(self, mock_save_model, mock_model):
        """Test saving with tokenizer."""
        with TemporaryDirectory() as tmp_dir:
            weights_path = os.path.join(tmp_dir, "model", "weights")
            tokenizer_path = os.path.join(tmp_dir, "tokenizer")
            os.makedirs(os.path.dirname(weights_path))
            os.makedirs(tokenizer_path)

            # Create mock tokenizer
            mock_tokenizer = MagicMock()

            # Save checkpoint with tokenizer
            save_checkpoint(
                model=mock_model,
                weights_path=weights_path,
                tokenizer=mock_tokenizer,
                tokenizer_path=tokenizer_path,
            )

            # Verify tokenizer.save_pretrained was called
            mock_tokenizer.save_pretrained.assert_called_once_with(tokenizer_path)


@pytest.fixture
def mock_experiment():
    """Create a real model, optimizer, and scheduler for integration testing."""
    model = TestModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    return model, optimizer, scheduler


def check_dict_equality(dict1, dict2):
    """Recursively check equality of two dictionaries"""
    for k in dict1.keys():
        if isinstance(dict1[k], dict):
            check_dict_equality(dict1[k], dict2[k])
        elif isinstance(dict1[k], torch.Tensor):
            assert torch.allclose(dict1[k], dict2[k])
        else:
            assert dict1[k] == dict2[k]


@pytest.mark.automodel
class TestSaveLoadIntegration:
    """Integration tests that actually save and load checkpoints."""

    def test_save_and_load_model_only_safetensors(self, mock_experiment):
        """Test saving and loading model weights only with safetensors format."""
        test_model, _, _ = mock_experiment
        original_state_dict = test_model.state_dict()

        with TemporaryDirectory() as tmp_dir:
            weights_path = os.path.join(tmp_dir, "test_model")

            # Save checkpoint
            save_checkpoint(
                model=test_model,
                weights_path=weights_path,
                model_save_format="safetensors",
            )

            # Verify files are created
            assert os.path.exists(weights_path)
            files = os.listdir(os.path.join(weights_path, "model"))
            assert any(f.endswith(".safetensors") for f in files)

            # Create a new model with different weights
            new_model = TestModel()
            # Initialize with different values
            for param in new_model.parameters():
                param.data.fill_(999.0)

            # Load the checkpoint
            load_checkpoint(model=new_model, weights_path=weights_path)

            # Verify the weights match the original
            check_dict_equality(new_model.state_dict(), original_state_dict)

    def test_save_and_load_model_only_torch_save(self, mock_experiment):
        """Test saving and loading model weights only with torch_save format."""
        test_model, _, _ = mock_experiment
        original_state_dict = test_model.state_dict()

        with TemporaryDirectory() as tmp_dir:
            weights_path = os.path.join(tmp_dir, "test_model")

            # Save checkpoint
            save_checkpoint(
                model=test_model,
                weights_path=weights_path,
                model_save_format="torch_save",
            )

            # Verify files are created
            assert os.path.exists(weights_path)
            files = os.listdir(os.path.join(weights_path, "model"))
            assert any(f.endswith(".distcp") for f in files)

            # Create a new model with different weights
            new_model = TestModel()
            # Initialize with different values
            for param in new_model.parameters():
                param.data.fill_(999.0)

            # Load the checkpoint
            load_checkpoint(model=new_model, weights_path=weights_path)

            # Verify the weights match the original
            check_dict_equality(new_model.state_dict(), original_state_dict)

    def test_save_and_load_model_and_optimizer(self, mock_experiment):
        """Test saving and loading both model and optimizer."""
        test_model, optimizer, scheduler = mock_experiment

        # Take some optimization steps to change optimizer state
        for _ in range(5):
            loss = torch.nn.functional.mse_loss(
                test_model(torch.randn(2, 4)), torch.randn(2, 1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        original_model_state = test_model.state_dict()
        original_optimizer_state = optimizer.state_dict()
        original_scheduler_state = scheduler.state_dict()

        with TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model_and_optimizer", "model_path")
            optimizer_path = os.path.join(tmp_dir, "model_and_optimizer", "optimizer")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)

            # Save checkpoint
            save_checkpoint(
                model=test_model,
                weights_path=model_path,
                optimizer=optimizer,
                scheduler=scheduler,
                optimizer_path=optimizer_path,
            )

            # Verify files are created
            assert os.path.exists(model_path)
            assert os.path.exists(optimizer_path)

            # Create new model, optimizer, and scheduler with different state
            new_model = TestModel()
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
            new_scheduler = torch.optim.lr_scheduler.StepLR(
                new_optimizer, step_size=4, gamma=0.2
            )

            # Initialize with different values
            for param in new_model.parameters():
                param.data.fill_(999.0)

            # Load the checkpoint
            load_checkpoint(
                model=new_model,
                weights_path=model_path,
                optimizer=new_optimizer,
                scheduler=new_scheduler,
                optimizer_path=optimizer_path,
            )

            # Verify all states match the original
            check_dict_equality(new_model.state_dict(), original_model_state)
            check_dict_equality(new_optimizer.state_dict(), original_optimizer_state)
            assert new_scheduler.state_dict() == original_scheduler_state
