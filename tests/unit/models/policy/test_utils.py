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
import torch.nn.functional as F

from nemo_rl.models.policy.utils import (
    apply_top_k_top_p,
    configure_expandable_segments,
    get_megatron_checkpoint_dir,
)


def manual_top_k(logits, k):
    """Manual reference implementation for top-k"""
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_values[..., -1:].expand_as(logits)
    mask = logits >= threshold
    return torch.where(
        mask,
        logits,
        torch.tensor(-float("inf"), device=logits.device, dtype=logits.dtype),
    )


def manual_top_p(logits, p):
    """Manual reference implementation for top-p"""
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs <= p
    # Always keep at least one
    mask[..., 0] = True
    keep_indices = torch.zeros_like(logits, dtype=torch.bool)
    for b in range(logits.shape[0]):
        for s in range(logits.shape[1]):
            keep = sorted_indices[b, s][mask[b, s]]
            keep_indices[b, s, keep] = True
    return torch.where(
        keep_indices,
        logits,
        torch.tensor(-float("inf"), device=logits.device, dtype=logits.dtype),
    )


class TestSamplingUtils(unittest.TestCase):
    """Test cases for sampling utility functions."""

    def test_top_k_only(self):
        """Test top-k only sampling"""
        torch.manual_seed(0)
        logits = torch.randn(2, 3, 10)
        k = 4
        result = apply_top_k_top_p(logits.clone(), top_k=k)
        expected = manual_top_k(logits, k)
        self.assertTrue(torch.allclose(result, expected))
        # Check only k values per row are not -inf
        non_inf = (result != -float("inf")).sum(dim=-1)
        self.assertTrue(torch.all(non_inf == k))

    def test_top_p_only(self):
        """Test top-p only sampling"""
        torch.manual_seed(1)
        logits = torch.randn(2, 3, 10)
        p = 0.7
        result = apply_top_k_top_p(logits.clone(), top_p=p)

        # Check that we have at least one non-inf value per position
        non_inf = (result != -float("inf")).sum(dim=-1)
        self.assertTrue(torch.all(non_inf >= 1))

        # Check that probability sums to 1.0
        result_probs = F.softmax(result, dim=-1)
        result_sum = result_probs.sum(dim=-1)
        self.assertTrue(torch.allclose(result_sum, torch.ones_like(result_sum)))

    def test_top_k_and_top_p(self):
        """Test both top-k and top-p sampling"""
        torch.manual_seed(2)
        logits = torch.randn(2, 3, 10)
        k = 5
        p = 0.8
        result = apply_top_k_top_p(logits.clone(), top_k=k, top_p=p)

        # Check basic properties
        non_inf = (result != -float("inf")).sum(dim=-1)
        self.assertTrue(torch.all(non_inf >= 1))
        self.assertTrue(torch.all(non_inf <= k))

        # Check that probability sums to 1.0
        result_probs = F.softmax(result, dim=-1)
        result_sum = result_probs.sum(dim=-1)
        self.assertTrue(torch.allclose(result_sum, torch.ones_like(result_sum)))

    def test_no_sampling(self):
        """Test no sampling (should return original logits)"""
        torch.manual_seed(3)
        logits = torch.randn(2, 3, 10)
        result = apply_top_k_top_p(logits.clone())
        self.assertTrue(torch.allclose(result, logits))

    def test_edge_cases(self):
        """Test edge cases for sampling parameters"""
        torch.manual_seed(4)
        logits = torch.randn(2, 3, 10)

        # k = vocab size (should return original logits)
        result = apply_top_k_top_p(logits.clone(), top_k=10)
        self.assertTrue(torch.allclose(result, logits))

        # k = 1 (should keep only one token)
        result = apply_top_k_top_p(logits.clone(), top_k=1)
        non_inf = (result != -float("inf")).sum(dim=-1)
        self.assertTrue(torch.all(non_inf == 1))

        # p = 1.0 (should return original logits)
        result = apply_top_k_top_p(logits.clone(), top_p=1.0)
        self.assertTrue(torch.allclose(result, logits))

        # p = 0.0 (should keep only the highest probability token)
        result = apply_top_k_top_p(logits.clone(), top_p=0.0)
        non_inf = (result != -float("inf")).sum(dim=-1)
        self.assertTrue(torch.all(non_inf == 1))

        # k > vocab size (should return original logits)
        result = apply_top_k_top_p(logits.clone(), top_k=20)
        self.assertTrue(torch.allclose(result, logits))

    def test_gradient_flow(self):
        """Test that gradients flow through the sampling operations"""
        torch.manual_seed(5)
        logits = torch.randn(2, 3, 10, requires_grad=True)
        result = apply_top_k_top_p(logits, top_k=3, top_p=0.7)
        loss = result.sum()
        loss.backward()

        # Check that gradients exist and are non-zero
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.shape, logits.shape)
        self.assertTrue(torch.any(logits.grad != 0))

    def test_large_vocab(self):
        """Test with larger vocabulary size"""
        torch.manual_seed(6)
        logits = torch.randn(1, 2, 1000)  # Large vocab
        k = 50
        p = 0.9

        result = apply_top_k_top_p(logits.clone(), top_k=k, top_p=p)

        # Check basic properties
        non_inf = (result != -float("inf")).sum(dim=-1)
        self.assertTrue(torch.all(non_inf >= 1))
        self.assertTrue(torch.all(non_inf <= k))

        # Check that probability sums to 1.0
        result_probs = F.softmax(result, dim=-1)
        result_sum = result_probs.sum(dim=-1)
        self.assertTrue(torch.allclose(result_sum, torch.ones_like(result_sum)))


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
