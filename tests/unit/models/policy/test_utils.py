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
import unittest
from unittest.mock import MagicMock, patch

from nemo_rl.models.policy.utils import configure_expandable_segments


class TestConfigureExpandableSegments(unittest.TestCase):
    """Test cases for configure_expandable_segments function."""

    def setUp(self):
        """Set up test environment."""
        # Store original environment variable
        self.original_pytorch_cuda_alloc_conf = os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF"
        )

    def tearDown(self):
        """Clean up after tests."""
        # Restore original environment variable
        if self.original_pytorch_cuda_alloc_conf is not None:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                self.original_pytorch_cuda_alloc_conf
            )
        elif "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
            del os.environ["PYTORCH_CUDA_ALLOC_CONF"]

    @patch("torch.cuda.get_device_properties")
    def test_hopper_gpu_no_existing_config(self, mock_get_device_properties):
        """Test Hopper+ GPU (compute capability >= 9) with no existing PYTORCH_CUDA_ALLOC_CONF."""
        # Mock GPU properties for Hopper+ architecture
        mock_device_properties = MagicMock()
        mock_device_properties.major = 9
        mock_get_device_properties.return_value = mock_device_properties

        # Ensure no existing config
        if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
            del os.environ["PYTORCH_CUDA_ALLOC_CONF"]

        # Call the function
        configure_expandable_segments()

        # Verify the environment variable was set correctly
        self.assertEqual(
            os.environ["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True"
        )

    @patch("torch.cuda.get_device_properties")
    def test_hopper_gpu_with_existing_config(self, mock_get_device_properties):
        """Test Hopper+ GPU with existing PYTORCH_CUDA_ALLOC_CONF."""
        # Mock GPU properties for Hopper+ architecture
        mock_device_properties = MagicMock()
        mock_device_properties.major = 9
        mock_get_device_properties.return_value = mock_device_properties

        # Set existing config
        existing_config = "max_split_size_mb:128"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = existing_config

        # Call the function
        configure_expandable_segments()

        # Verify the environment variable was updated correctly
        expected_config = f"{existing_config},expandable_segments:True"
        self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"], expected_config)

    @patch("torch.cuda.get_device_properties")
    def test_hopper_gpu_already_configured(self, mock_get_device_properties):
        """Test Hopper+ GPU with existing config that already has expandable_segments."""
        # Mock GPU properties for Hopper+ architecture
        mock_device_properties = MagicMock()
        mock_device_properties.major = 9
        mock_get_device_properties.return_value = mock_device_properties

        # Set existing config with expandable_segments already present
        existing_config = "max_split_size_mb:128,expandable_segments:False"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = existing_config

        # Call the function
        configure_expandable_segments()

        # Verify the environment variable was not changed
        self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"], existing_config)

    @patch("torch.cuda.get_device_properties")
    def test_ampere_gpu_no_config_change(self, mock_get_device_properties):
        """Test Ampere GPU (compute capability < 9) should not modify config."""
        # Mock GPU properties for Ampere architecture
        mock_device_properties = MagicMock()
        mock_device_properties.major = 8  # Ampere
        mock_get_device_properties.return_value = mock_device_properties

        # Set existing config
        existing_config = "max_split_size_mb:128"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = existing_config

        # Call the function
        configure_expandable_segments()

        # Verify the environment variable was not changed
        self.assertEqual(os.environ["PYTORCH_CUDA_ALLOC_CONF"], existing_config)

    @patch("torch.cuda.get_device_properties")
    def test_ampere_gpu_no_existing_config(self, mock_get_device_properties):
        """Test Ampere GPU with no existing config should not set anything."""
        # Mock GPU properties for Ampere architecture
        mock_device_properties = MagicMock()
        mock_device_properties.major = 8  # Ampere
        mock_get_device_properties.return_value = mock_device_properties

        # Ensure no existing config
        if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
            del os.environ["PYTORCH_CUDA_ALLOC_CONF"]

        # Call the function
        configure_expandable_segments()

        # Verify the environment variable was not set
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", os.environ)
