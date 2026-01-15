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
"""Tests for cluster configuration."""

import os
from unittest import mock

import pytest

from nemo_rl.config.base import ConfigValidationError
from nemo_rl.config.cluster import ClusterConfig


class TestClusterConfig:
    """Tests for ClusterConfig."""

    def test_default_values(self):
        """Test default values."""
        config = ClusterConfig()
        assert config.gpus_per_node == 8
        assert config.num_nodes == 1

    def test_custom_values(self):
        """Test custom values."""
        config = ClusterConfig(num_nodes=4, gpus_per_node=4)
        assert config.num_nodes == 4
        assert config.gpus_per_node == 4

    def test_invalid_gpus_per_node(self):
        """Test invalid GPUs per node."""
        with pytest.raises(ConfigValidationError):
            ClusterConfig(gpus_per_node=0)

    def test_invalid_num_nodes(self):
        """Test invalid number of nodes."""
        with pytest.raises(ConfigValidationError):
            ClusterConfig(num_nodes=-1)

    def test_total_gpus_property(self):
        """Test total_gpus property."""
        config = ClusterConfig(num_nodes=2, gpus_per_node=8)
        assert config.total_gpus == 16

    def test_world_size_property(self):
        """Test world_size property."""
        config = ClusterConfig(num_nodes=4, gpus_per_node=8)
        assert config.world_size == 32

    def test_auto_detect_default(self):
        """Test auto_detect with default environment."""
        with mock.patch.dict(os.environ, {}, clear=True):
            # Clear CUDA_VISIBLE_DEVICES and SLURM vars
            env_vars = {k: v for k, v in os.environ.items() 
                       if not k.startswith("CUDA") and not k.startswith("SLURM")}
            with mock.patch.dict(os.environ, env_vars, clear=True):
                # Mock torch.cuda.device_count to return a known value
                with mock.patch("torch.cuda.device_count", return_value=1):
                    config = ClusterConfig.auto_detect()
                    assert config.gpus_per_node >= 1
                    assert config.num_nodes == 1

    def test_auto_detect_with_cuda_visible_devices(self):
        """Test auto_detect with CUDA_VISIBLE_DEVICES set."""
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}):
            config = ClusterConfig.auto_detect()
            assert config.gpus_per_node == 4

    def test_auto_detect_with_slurm(self):
        """Test auto_detect with SLURM environment."""
        with mock.patch.dict(os.environ, {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "SLURM_NNODES": "4",
        }):
            config = ClusterConfig.auto_detect()
            assert config.gpus_per_node == 2
            assert config.num_nodes == 4

    def test_validate_parallelism_valid(self):
        """Test validate_parallelism with valid settings."""
        config = ClusterConfig(num_nodes=1, gpus_per_node=8)
        assert config.validate_parallelism(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=4,
        )

    def test_validate_parallelism_invalid_exceeds_resources(self):
        """Test validate_parallelism when exceeding available resources."""
        config = ClusterConfig(num_nodes=1, gpus_per_node=4)
        with pytest.raises(ValueError) as exc_info:
            config.validate_parallelism(
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
                data_parallel_size=4,
            )
        assert "16 GPUs" in str(exc_info.value)
        assert "4 available" in str(exc_info.value)

    def test_validate_parallelism_inferred_dp(self):
        """Test validate_parallelism with inferred data parallelism."""
        config = ClusterConfig(num_nodes=1, gpus_per_node=8)
        # Should infer data_parallel_size = 8 / (2 * 2) = 2
        assert config.validate_parallelism(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
        )

    def test_validate_parallelism_invalid_not_divisible(self):
        """Test validate_parallelism when not divisible."""
        config = ClusterConfig(num_nodes=1, gpus_per_node=8)
        with pytest.raises(ValueError) as exc_info:
            config.validate_parallelism(
                tensor_parallel_size=3,  # 8 not divisible by 3
                pipeline_parallel_size=1,
            )
        assert "divisible" in str(exc_info.value)
