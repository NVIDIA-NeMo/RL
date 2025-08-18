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
import unittest.mock
from unittest.mock import MagicMock, patch

import torch

from nemo_rl.models.policy.utils import (
    apply_top_k_top_p,
    configure_expandable_segments,
    get_megatron_checkpoint_dir,
)


class TestSamplingUtils(unittest.TestCase):
    """Test cases for sampling utility functions."""

    def setUp(self):
        """Set up deterministic test data"""
        # Create a deterministic logits tensor of shape [2, 2, 4]
        self.logits = torch.tensor(
            [
                [
                    [10.0, 5.0, 2.0, 1.0],
                    [8.0, 7.0, 3.0, 0.5],
                ],
                [
                    [6.0, 9.0, 1.5, 4.0],
                    [2.5, 1.0, 8.5, 7.5],
                ],
            ],
            dtype=torch.float32,
        )

    def test_top_k_only(self):
        """Test top-k only sampling"""
        k = 2
        result = apply_top_k_top_p(self.logits.clone(), top_k=k)

        # Expected result: keep top 2 values, mask others
        expected = torch.tensor(
            [
                [
                    [10.0, 5.0, -float("inf"), -float("inf")],
                    [8.0, 7.0, -float("inf"), -float("inf")],
                ],
                [
                    [6.0, 9.0, -float("inf"), -float("inf")],
                    [-float("inf"), -float("inf"), 8.5, 7.5],
                ],
            ],
            dtype=torch.float32,
        )

        self.assertTrue(torch.allclose(result, expected, equal_nan=True))

    def test_top_p_only(self):
        """Test top-p only sampling with specific probability threshold"""
        p = 0.8
        result = apply_top_k_top_p(self.logits.clone(), top_p=p)

        expected = torch.tensor(
            [
                [
                    [10.0, -float("inf"), -float("inf"), -float("inf")],
                    [8.0, 7.0, -float("inf"), -float("inf")],
                ],
                [
                    [-float("inf"), 9.0, -float("inf"), -float("inf")],
                    [-float("inf"), -float("inf"), 8.5, 7.5],
                ],
            ],
            dtype=torch.float32,
        )

        self.assertTrue(torch.allclose(result, expected, equal_nan=True))

    def test_top_k_and_top_p_combined(self):
        """Test both top-k and top-p sampling together"""
        k = 3
        p = 0.8
        result = apply_top_k_top_p(self.logits.clone(), top_k=k, top_p=p)

        # Expected: apply top-k=3 first, then top-p=0.8
        # For our data, top-k=3 keeps top 3 values, then top-p further filters
        expected = torch.tensor(
            [
                [
                    [10.0, -float("inf"), -float("inf"), -float("inf")],
                    [8.0, 7.0, -float("inf"), -float("inf")],
                ],
                [
                    [-float("inf"), 9.0, -float("inf"), -float("inf")],
                    [-float("inf"), -float("inf"), 8.5, 7.5],
                ],
            ],
            dtype=torch.float32,
        )

        self.assertTrue(torch.allclose(result, expected, equal_nan=True))

    def test_no_sampling(self):
        """Test no sampling (should return original logits)"""
        result = apply_top_k_top_p(self.logits.clone())
        self.assertTrue(torch.allclose(result, self.logits, equal_nan=True))

    def test_edge_cases(self):
        """Test edge cases for sampling parameters"""
        # k >= vocab size (should return original logits)
        result = apply_top_k_top_p(self.logits.clone(), top_k=4)
        self.assertTrue(torch.allclose(result, self.logits))

        result = apply_top_k_top_p(self.logits.clone(), top_k=10)
        self.assertTrue(torch.allclose(result, self.logits))

        # p = 1.0 (should return original logits)
        result = apply_top_k_top_p(self.logits.clone(), top_p=1.0)
        self.assertTrue(torch.allclose(result, self.logits))

    def test_gradient_flow(self):
        """Test that gradients flow through the sampling operations"""
        logits_with_grad = self.logits.clone().requires_grad_(True)
        result = apply_top_k_top_p(logits_with_grad, top_k=2, top_p=0.7)
        loss = result.sum()
        loss.backward()

        # Check that gradients exist and are non-zero for at least some elements
        self.assertIsNotNone(logits_with_grad.grad)
        self.assertEqual(logits_with_grad.grad.shape, logits_with_grad.shape)
        self.assertTrue(torch.any(logits_with_grad.grad != 0))


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

    @patch("torch.cuda.get_device_properties")
    def test_ampere_gpu_with_expandable_segments_true_raises_error(
        self, mock_get_device_properties
    ):
        """Test Ampere GPU with expandable_segments:True in config raises RuntimeError."""
        # Mock GPU properties for Ampere architecture
        mock_device_properties = MagicMock()
        mock_device_properties.major = 8  # Ampere
        mock_get_device_properties.return_value = mock_device_properties

        # Set config with expandable_segments:True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Call the function and expect RuntimeError
        with self.assertRaises(RuntimeError) as context:
            configure_expandable_segments()

        # Verify the error message
        self.assertIn("expandable_segments is enabled", str(context.exception))
        self.assertIn(
            "not supported on architectures older than Hopper", str(context.exception)
        )


class TestGetMegatronCheckpointDir:
    """Test cases for the get_megatron_checkpoint_dir function."""

    def test_nrl_megatron_checkpoint_dir_takes_precedence(self):
        """Test that NRL_MEGATRON_CHECKPOINT_DIR environment variable takes highest precedence."""
        expected_dir = "/custom/nrl/checkpoint/path"

        with unittest.mock.patch.dict(
            os.environ,
            {
                "NRL_MEGATRON_CHECKPOINT_DIR": expected_dir,
                "HF_HOME": "/some/hf/home",
                "HOME": "/some/home",
            },
        ):
            result = get_megatron_checkpoint_dir()
            assert result == expected_dir

    def test_hf_home_fallback_when_nrl_not_set(self):
        """Test that HF_HOME/nemo_rl is used when NRL_MEGATRON_CHECKPOINT_DIR is not set."""
        hf_home = "/path/to/hf/home"
        expected_dir = os.path.join(hf_home, "nemo_rl")

        env_vars = {"HF_HOME": hf_home, "HOME": "/some/home"}
        # Remove NRL_MEGATRON_CHECKPOINT_DIR if it exists
        env_vars.pop("NRL_MEGATRON_CHECKPOINT_DIR", None)

        with unittest.mock.patch.dict(os.environ, env_vars, clear=True):
            result = get_megatron_checkpoint_dir()
            assert result == expected_dir

    def test_default_fallback_when_no_env_vars_set(self):
        """Test that ~/.cache/huggingface/nemo_rl is used when no environment variables are set."""
        home_dir = "/home/testuser"
        expected_dir = os.path.join(home_dir, ".cache", "huggingface", "nemo_rl")

        with unittest.mock.patch.dict(os.environ, {"HOME": home_dir}, clear=True):
            with unittest.mock.patch("os.path.expanduser") as mock_expanduser:
                mock_expanduser.return_value = home_dir
                result = get_megatron_checkpoint_dir()
                assert result == expected_dir
                mock_expanduser.assert_called_once_with("~")

    def test_nrl_checkpoint_dir_empty_string_treated_as_unset(self):
        """Test that an empty NRL_MEGATRON_CHECKPOINT_DIR is treated as unset."""
        hf_home = "/path/to/hf/home"
        expected_dir = os.path.join(hf_home, "nemo_rl")

        with unittest.mock.patch.dict(
            os.environ,
            {
                "NRL_MEGATRON_CHECKPOINT_DIR": "",
                "HF_HOME": hf_home,
                "HOME": "/some/home",
            },
        ):
            result = get_megatron_checkpoint_dir()
            assert result == expected_dir

    def test_hf_home_empty_string_treated_as_unset(self):
        """Test that an empty HF_HOME is treated as unset."""
        home_dir = "/home/testuser"
        expected_dir = os.path.join(home_dir, ".cache", "huggingface", "nemo_rl")

        with unittest.mock.patch.dict(
            os.environ, {"HF_HOME": "", "HOME": home_dir}, clear=True
        ):
            with unittest.mock.patch("os.path.expanduser") as mock_expanduser:
                mock_expanduser.return_value = home_dir
                result = get_megatron_checkpoint_dir()
                assert result == expected_dir

    def test_function_prints_selected_directory(self, capsys):
        """Test that the function prints the selected directory."""
        expected_dir = "/custom/checkpoint/dir"

        with unittest.mock.patch.dict(
            os.environ, {"NRL_MEGATRON_CHECKPOINT_DIR": expected_dir}
        ):
            result = get_megatron_checkpoint_dir()

            captured = capsys.readouterr()
            assert (
                f"Using default megatron checkpoint dir: {expected_dir}" in captured.out
            )
            assert result == expected_dir
