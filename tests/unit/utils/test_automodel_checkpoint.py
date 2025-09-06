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

import pytest
import torch

from nemo_rl.utils.automodel_checkpoint import (
    _get_actual_model_path,
    detect_checkpoint_format,
    load_checkpoint,
)


@pytest.fixture
def mock_model():
    """Create a simple mock model for testing."""
    model = torch.nn.ModuleList(
        [
            torch.nn.Linear(4, 4),
            torch.nn.LayerNorm(4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
        ]
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def test_detect_checkpoint_format_file_extensions():
    """Test format detection based on file extensions."""
    # Test safetensors format
    assert detect_checkpoint_format("/path/to/model.safetensors") == (
        "safetensors",
        False,
    )

    # Test torch save formats
    assert detect_checkpoint_format("/path/to/model.bin") == ("torch_save", False)
    assert detect_checkpoint_format("/path/to/model.pt") == ("torch_save", False)
    assert detect_checkpoint_format("/path/to/model.pth") == ("torch_save", False)

    # Test PEFT detection in paths
    assert detect_checkpoint_format("/path/to/adapter_model.safetensors") == (
        "safetensors",
        True,
    )
    assert detect_checkpoint_format("/path/to/lora_model.bin") == ("torch_save", True)
    assert detect_checkpoint_format("/path/to/peft_checkpoint.pt") == (
        "torch_save",
        True,
    )


def test_detect_checkpoint_format_dcp_structure():
    """Test format detection for DCP checkpoint structure."""
    with TemporaryDirectory() as tmp_dir:
        # Test DCP format (distcp + metadata)
        weights_dir = os.path.join(tmp_dir, "weights")
        model_dir = os.path.join(weights_dir, "model")
        os.makedirs(model_dir)

        # Create DCP files
        with open(os.path.join(model_dir, "__0_0.distcp"), "w") as f:
            f.write("dummy dcp content")
        with open(os.path.join(model_dir, ".metadata"), "w") as f:
            f.write("dummy metadata")

        format_type, is_peft = detect_checkpoint_format(weights_dir)
        assert format_type == "torch_save"  # DCP uses torch_save format
        assert is_peft == False


def test_detect_checkpoint_format_safetensors_structure():
    """Test format detection for safetensors checkpoint structure."""
    with TemporaryDirectory() as tmp_dir:
        # Test safetensors format
        weights_dir = os.path.join(tmp_dir, "weights")
        model_dir = os.path.join(weights_dir, "model")
        os.makedirs(model_dir)

        # Create safetensors shard files
        with open(
            os.path.join(model_dir, "shard-00001-model-00001-of-00001.safetensors"), "w"
        ) as f:
            f.write("dummy safetensors content")

        format_type, is_peft = detect_checkpoint_format(weights_dir)
        assert format_type == "safetensors"
        assert is_peft == False


def test_detect_checkpoint_format_peft_detection():
    """Test PEFT detection in various scenarios."""
    with TemporaryDirectory() as tmp_dir:
        # Test PEFT detection from directory name
        lora_dir = os.path.join(tmp_dir, "lora_weights", "model")
        os.makedirs(lora_dir)
        with open(os.path.join(lora_dir, "model.safetensors"), "w") as f:
            f.write("dummy")

        format_type, is_peft = detect_checkpoint_format(os.path.dirname(lora_dir))
        assert format_type == "safetensors"
        assert is_peft == True  # Should detect "lora" in path

        # Test PEFT detection from file names
        adapter_dir = os.path.join(tmp_dir, "regular_checkpoint", "model")
        os.makedirs(adapter_dir)
        with open(os.path.join(adapter_dir, "adapter_model.safetensors"), "w") as f:
            f.write("dummy")

        format_type, is_peft = detect_checkpoint_format(os.path.dirname(adapter_dir))
        assert format_type == "safetensors"
        assert is_peft == True  # Should detect "adapter" in file name


def test_get_actual_model_path():
    """Test the _get_actual_model_path helper function."""
    with TemporaryDirectory() as tmp_dir:
        # Test 1: Direct file path should return as-is
        file_path = os.path.join(tmp_dir, "model.safetensors")
        with open(file_path, "w") as f:
            f.write("dummy")
        assert _get_actual_model_path(file_path) == file_path

        # Test 2: Directory without nested model/ should return as-is
        simple_dir = os.path.join(tmp_dir, "simple_model")
        os.makedirs(simple_dir)
        with open(os.path.join(simple_dir, "model.safetensors"), "w") as f:
            f.write("dummy")
        assert _get_actual_model_path(simple_dir) == simple_dir

        # Test 3: Directory with nested model/ containing model files should return nested path
        weights_dir = os.path.join(tmp_dir, "weights")
        model_dir = os.path.join(weights_dir, "model")
        os.makedirs(model_dir)

        # Create typical shard files
        with open(
            os.path.join(model_dir, "shard-00001-model-00001-of-00001.safetensors"), "w"
        ) as f:
            f.write("dummy shard")

        actual_path = _get_actual_model_path(weights_dir)
        assert actual_path == model_dir

        # Test 4: Directory with DCP files in model/ subdirectory
        dcp_weights_dir = os.path.join(tmp_dir, "dcp_weights")
        dcp_model_dir = os.path.join(dcp_weights_dir, "model")
        os.makedirs(dcp_model_dir)

        with open(os.path.join(dcp_model_dir, "__0_0.distcp"), "w") as f:
            f.write("dummy dcp")
        with open(os.path.join(dcp_model_dir, ".metadata"), "w") as f:
            f.write("metadata")

        actual_path = _get_actual_model_path(dcp_weights_dir)
        assert actual_path == dcp_model_dir

        # Test 5: Non-existent path should return as-is
        fake_path = os.path.join(tmp_dir, "non_existent")
        assert _get_actual_model_path(fake_path) == fake_path


def test_load_checkpoint_auto_detection_dcp(mock_model, monkeypatch):
    """Test that load_checkpoint properly auto-detects DCP format."""
    captured_configs = []

    def mock_load_model(model, weights_path, checkpoint_config):
        captured_configs.append(checkpoint_config)

    monkeypatch.setattr(
        "nemo_rl.utils.automodel_checkpoint.load_model", mock_load_model
    )

    with TemporaryDirectory() as tmp_dir:
        # Create DCP checkpoint structure
        weights_dir = os.path.join(tmp_dir, "step_2", "policy", "weights")
        model_dir = os.path.join(weights_dir, "model")
        os.makedirs(model_dir)

        # Create DCP files
        with open(os.path.join(model_dir, "__0_0.distcp"), "w") as f:
            f.write("dummy dcp content")
        with open(os.path.join(model_dir, ".metadata"), "w") as f:
            f.write("dummy metadata")

        # Load checkpoint - should auto-detect DCP format
        load_checkpoint(mock_model, weights_dir)

        assert len(captured_configs) == 1
        config = captured_configs[0]
        assert config.model_save_format == "torch_save"  # DCP format
        assert config.is_peft == False
        assert config.checkpoint_dir == model_dir


def test_load_checkpoint_auto_detection_safetensors(mock_model, monkeypatch):
    """Test that load_checkpoint properly auto-detects safetensors format."""
    captured_configs = []

    def mock_load_model(model, weights_path, checkpoint_config):
        captured_configs.append(checkpoint_config)

    monkeypatch.setattr(
        "nemo_rl.utils.automodel_checkpoint.load_model", mock_load_model
    )

    with TemporaryDirectory() as tmp_dir:
        # Create safetensors checkpoint structure
        weights_dir = os.path.join(tmp_dir, "step_1", "policy", "weights")
        model_dir = os.path.join(weights_dir, "model")
        os.makedirs(model_dir)

        # Create safetensors shard files
        with open(
            os.path.join(model_dir, "shard-00001-model-00001-of-00001.safetensors"), "w"
        ) as f:
            f.write("dummy safetensors content")

        # Load checkpoint - should auto-detect safetensors format
        load_checkpoint(mock_model, weights_dir)

        assert len(captured_configs) == 1
        config = captured_configs[0]
        assert config.model_save_format == "safetensors"  # Safetensors format
        assert config.is_peft == False
        assert config.checkpoint_dir == model_dir


def test_load_checkpoint_explicit_parameters_override(mock_model, monkeypatch):
    """Test that explicit parameters override auto-detection."""
    captured_configs = []

    def mock_load_model(model, weights_path, checkpoint_config):
        captured_configs.append(checkpoint_config)

    monkeypatch.setattr(
        "nemo_rl.utils.automodel_checkpoint.load_model", mock_load_model
    )

    with TemporaryDirectory() as tmp_dir:
        # Create safetensors checkpoint
        weights_dir = os.path.join(tmp_dir, "weights")
        model_dir = os.path.join(weights_dir, "model")
        os.makedirs(model_dir)

        with open(os.path.join(model_dir, "model.safetensors"), "w") as f:
            f.write("dummy safetensors")

        # Load with explicit parameters - should override auto-detection
        load_checkpoint(
            mock_model,
            weights_dir,
            model_save_format="torch_save",  # Override detection
            is_peft=True,  # Override detection
        )

        assert len(captured_configs) == 1
        config = captured_configs[0]
        assert config.model_save_format == "torch_save"  # Should use explicit value
        assert config.is_peft == True  # Should use explicit value


def test_load_checkpoint_mixed_auto_detection(mock_model, monkeypatch):
    """Test auto-detection when only one parameter is provided."""
    captured_configs = []

    def mock_load_model(model, weights_path, checkpoint_config):
        captured_configs.append(checkpoint_config)

    monkeypatch.setattr(
        "nemo_rl.utils.automodel_checkpoint.load_model", mock_load_model
    )

    with TemporaryDirectory() as tmp_dir:
        # Create lora checkpoint with safetensors
        lora_dir = os.path.join(tmp_dir, "lora_weights", "model")
        os.makedirs(lora_dir)

        with open(os.path.join(lora_dir, "adapter_model.safetensors"), "w") as f:
            f.write("dummy lora")

        # Provide explicit format, auto-detect PEFT
        load_checkpoint(
            mock_model,
            os.path.dirname(lora_dir),
            model_save_format="torch_save",  # Explicit
            # is_peft will be auto-detected
        )

        config = captured_configs[0]
        assert config.model_save_format == "torch_save"  # Explicit
        assert (
            config.is_peft == True
        )  # Auto-detected from "lora" in path and "adapter" in filename

        # Provide explicit PEFT, auto-detect format
        captured_configs.clear()
        load_checkpoint(
            mock_model,
            os.path.dirname(lora_dir),
            is_peft=False,  # Explicit
            # model_save_format will be auto-detected
        )

        config = captured_configs[0]
        assert (
            config.model_save_format == "safetensors"
        )  # Auto-detected from .safetensors files
        assert config.is_peft == False  # Explicit


def test_detect_checkpoint_format_path_patterns():
    """Test PEFT detection based on path patterns."""
    test_cases = [
        ("/models/lora-adapter/weights", True),
        ("/checkpoints/adapter_weights/model", True),
        ("/output/peft_model/checkpoint", True),
        ("/base/regular_model/weights", False),
        ("/checkpoints/full_model/safetensors", False),
        ("/models/LoRA_fine_tuned/weights", True),  # Case insensitive
        ("/adapters/PEFT_checkpoint/model", True),  # Case insensitive
    ]

    for path, expected_peft in test_cases:
        _, is_peft = detect_checkpoint_format(path)
        assert is_peft == expected_peft, (
            f"Failed for path: {path}. Expected {expected_peft}, got {is_peft}"
        )
