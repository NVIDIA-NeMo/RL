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

from unittest.mock import Mock, patch

import torch

from nemo_rl.models.megatron.converters.common import (
    get_global_expert_num,
    get_global_key_from_local_key,
    get_global_layer_num,
    get_local_expert_num,
    get_local_layer_num,
    split_fc1_etp,
    split_fc1_tp,
    split_qkv_bias_gpu,
    split_qkv_gpu,
    update_transforms_for_nemorl,
)


class TestLayerNumberFunctions:
    """Test functions related to layer number extraction and conversion."""

    def test_get_local_layer_num_valid(self):
        """Test get_local_layer_num with valid layer keys."""
        assert get_local_layer_num("layers.5.attention.weight") == 5
        assert get_local_layer_num("decoder.layers.10.mlp.weight") == 10
        assert get_local_layer_num("model.layers.0.self_attn.weight") == 0

    def test_get_local_layer_num_invalid(self):
        """Test get_local_layer_num with invalid layer keys."""
        assert get_local_layer_num("attention.weight") is None
        assert get_local_layer_num("layers.abc.weight") is None
        assert get_local_layer_num("layers.") is None

    def test_get_global_layer_num_pp(self):
        """Test get_global_layer_num with simple pipeline configuration."""
        mock_cfg = Mock()
        mock_cfg.num_layers = 10
        mock_cfg.num_layers_in_first_pipeline_stage = 4
        mock_cfg.num_layers_in_last_pipeline_stage = 3

        with patch(
            "nemo_rl.models.megatron.converters.common.parallel_state"
        ) as mock_ps:
            mock_ps.get_pipeline_model_parallel_rank.return_value = 1
            mock_ps.get_pipeline_model_parallel_world_size.return_value = 3

            result = get_global_layer_num("layers.2.weight", mock_cfg)
            assert result == 6


class TestExpertNumberFunctions:
    """Test functions related to expert number extraction and conversion."""

    def test_get_local_expert_num_valid(self):
        """Test get_local_expert_num with valid expert keys."""
        assert get_local_expert_num("layers.0.mlp.experts.weight2") == 2
        assert get_local_expert_num("decoder.layers.1.experts.weight5") == 5
        assert get_local_expert_num("model.layers.0.experts.weight0") == 0

    def test_get_local_expert_num_invalid(self):
        """Test get_local_expert_num with invalid expert keys."""
        assert get_local_expert_num("layers.0.mlp.weight") is None
        assert get_local_expert_num("layers.0.mlp.experts.2._extra_state") is None

    def test_get_global_expert_num(self):
        """Test get_global_expert_num with expert parallel configuration."""
        mock_cfg = Mock()
        mock_cfg.num_moe_experts = 8

        with patch(
            "nemo_rl.models.megatron.converters.common.parallel_state"
        ) as mock_ps:
            mock_ps.get_expert_model_parallel_rank.return_value = 1
            mock_ps.get_expert_model_parallel_world_size.return_value = 2

            result = get_global_expert_num("layers.0.mlp.experts.weight2", mock_cfg)
            assert result == 6  # 8 // 2 + 2


class TestKeyConversionFunctions:
    """Test functions related to key conversion between local and global."""

    def test_get_global_key_from_local_key_layer_only(self):
        """Test key conversion with only layer numbers."""
        mock_cfg = Mock()
        mock_cfg.num_layers = 12
        mock_cfg.num_layers_in_first_pipeline_stage = None
        mock_cfg.num_layers_in_last_pipeline_stage = None

        with patch(
            "nemo_rl.models.megatron.converters.common.parallel_state"
        ) as mock_ps:
            mock_ps.get_pipeline_model_parallel_rank.return_value = 1
            mock_ps.get_pipeline_model_parallel_world_size.return_value = 2

            result = get_global_key_from_local_key(
                "layers.3.attention.weight", mock_cfg
            )
            assert result == "layers.9.attention.weight"

    def test_get_global_key_from_local_key_expert_and_layer(self):
        """Test key conversion with only expert numbers."""
        mock_cfg = Mock()
        mock_cfg.num_moe_experts = 8
        mock_cfg.num_layers = 12
        mock_cfg.num_layers_in_first_pipeline_stage = None
        mock_cfg.num_layers_in_last_pipeline_stage = None

        with patch(
            "nemo_rl.models.megatron.converters.common.parallel_state"
        ) as mock_ps:
            mock_ps.get_expert_model_parallel_rank.return_value = 1
            mock_ps.get_expert_model_parallel_world_size.return_value = 2

            mock_ps.get_pipeline_model_parallel_rank.return_value = 1
            mock_ps.get_pipeline_model_parallel_world_size.return_value = 3

            result = get_global_key_from_local_key(
                "layers.0.mlp.experts.weight2", mock_cfg
            )
            assert result == "layers.4.mlp.experts.weight6"


class TestTensorSplittingFunctions:
    """Test functions related to tensor splitting operations."""

    def test_split_fc1_tp(self):
        """Test split_fc1_tp function."""
        mock_ctx = Mock()
        mock_ctx.source.config.tensor_model_parallel_size = 2

        # Create a tensor with shape (4, 10) representing 2 TP ranks with 2 components each
        linear_fc1 = torch.randn(4, 10)

        gate_proj, up_proj = split_fc1_tp(mock_ctx, linear_fc1)

        assert gate_proj.shape == (2, 10)
        assert up_proj.shape == (2, 10)
        assert torch.allclose(gate_proj, linear_fc1[::2])
        assert torch.allclose(up_proj, linear_fc1[1::2])

    def test_split_fc1_etp(self):
        """Test split_fc1_etp function."""
        mock_ctx = Mock()
        mock_ctx.source.config.expert_tensor_parallel_size = 2

        # Create a tensor with shape (4, 10) representing 2 ETP ranks with 2 components each
        linear_fc1 = torch.randn(4, 10)

        gate_proj, up_proj = split_fc1_etp(mock_ctx, linear_fc1)

        assert gate_proj.shape == (2, 10)
        assert up_proj.shape == (2, 10)
        assert torch.allclose(gate_proj, linear_fc1[::2])
        assert torch.allclose(up_proj, linear_fc1[1::2])

    def test_split_qkv_gpu(self):
        """Test split_qkv_gpu function."""
        mock_ctx = Mock()
        mock_ctx.source.config.num_attention_heads = 8
        mock_ctx.source.config.num_query_groups = 2
        mock_ctx.source.config.kv_channels = 16

        # Create QKV tensor: (heads + 2*groups) * head_size * hidden_size
        qkv_total_dim = 8 + 2 * 2  # 12
        linear_qkv = torch.randn(qkv_total_dim, 16, 64)

        q_proj, k_proj, v_proj = split_qkv_gpu(mock_ctx, linear_qkv)

        # Q should have 8 heads * 16 channels = 128
        assert q_proj.shape == (128, 64)
        # K and V should have 2 groups * 16 channels = 32 each
        assert k_proj.shape == (32, 64)
        assert v_proj.shape == (32, 64)

    def test_split_qkv_bias_gpu(self):
        """Test split_qkv_bias_gpu function."""
        mock_ctx = Mock()
        mock_ctx.source.config.num_attention_heads = 8
        mock_ctx.source.config.num_query_groups = 2
        mock_ctx.source.config.kv_channels = 16

        # Create QKV bias tensor: (heads + 2*groups) * head_size
        qkv_total_dim = 8 + 2 * 2  # 12
        qkv_bias = torch.randn(qkv_total_dim, 16)

        q_bias, k_bias, v_bias = split_qkv_bias_gpu(mock_ctx, qkv_bias)

        # Q should have 8 heads * 16 channels = 128
        assert q_bias.shape == (128,)
        # K and V should have 2 groups * 16 channels = 32 each
        assert k_bias.shape == (32,)
        assert v_bias.shape == (32,)


class TestTransformUpdateFunctions:
    """Test functions related to transform updates."""

    def test_update_transforms_for_nemorl(self):
        """Test update_transforms_for_nemorl function."""
        # Create mock transforms
        mock_transform1 = Mock()
        mock_transform1.transform.__name__ = "split_fc1"
        mock_transform1.source_key = "layers.0.mlp.experts.0.linear_fc1.weight"

        mock_transform2 = Mock()
        mock_transform2.transform.__name__ = "split_fc1"
        mock_transform2.source_key = "layers.0.mlp.shared_experts.linear_fc1.weight"

        mock_transform3 = Mock()
        mock_transform3.transform.__name__ = "split_qkv"

        mock_transform4 = Mock()
        mock_transform4.transform.__name__ = "split_qkv_bias"

        transforms = [
            mock_transform1,
            mock_transform2,
            mock_transform3,
            mock_transform4,
        ]

        updated_transforms = update_transforms_for_nemorl(transforms)

        # Check that expert transforms use split_fc1_etp
        assert updated_transforms[0].transform == split_fc1_etp
        # Check that non-expert transforms use split_fc1_tp
        assert updated_transforms[1].transform == split_fc1_tp
        # Check that qkv transforms are updated
        assert updated_transforms[2].transform == split_qkv_gpu
        assert updated_transforms[3].transform == split_qkv_bias_gpu
