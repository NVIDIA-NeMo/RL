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
from contextlib import contextmanager
from tempfile import TemporaryDirectory

import pytest
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM


def dummy_qwen3_megatron_moe_config():
    from nemo.collections.llm.gpt.model.qwen3 import Qwen3MoEConfig

    return Qwen3MoEConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        ffn_hidden_size=128,
        moe_ffn_hidden_size=32,
        num_moe_experts=2,
        share_embeddings_and_output_weights=True,
        kv_channels=16,
    )


def dummy_qwen3_megatron_dense_config():
    from nemo.collections.llm.gpt.model.qwen3 import Qwen3Config

    return Qwen3Config(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        ffn_hidden_size=128,
        share_embeddings_and_output_weights=False,
        kv_channels=16,
    )


def create_dummy_hf_moe_config():
    """Create a dummy HF MoE config and save it to a temporary directory."""
    # Create a minimal HF config that matches the megatron config
    hf_config = AutoConfig.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)

    # Update config to match our dummy megatron config
    hf_config.num_hidden_layers = 2
    hf_config.hidden_size = 64
    hf_config.num_attention_heads = 4
    hf_config.num_key_value_heads = 2
    hf_config.intermediate_size = 128
    hf_config.moe_intermediate_size = 32
    hf_config.num_experts = 2
    hf_config.tie_word_embeddings = True
    hf_config.head_dim = 16

    return hf_config


def create_dummy_hf_dense_config():
    """Create a dummy HF dense config and save it to a temporary directory."""
    # Create a minimal HF config that matches the megatron config
    hf_config = AutoConfig.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

    # Update config to match our dummy megatron config
    hf_config.num_hidden_layers = 2
    hf_config.hidden_size = 64
    hf_config.num_attention_heads = 4
    hf_config.num_key_value_heads = 2
    hf_config.intermediate_size = 128
    hf_config.tie_word_embeddings = False
    hf_config.head_dim = 16

    return hf_config


@contextmanager
def setup_distributed_environment(port="6000"):
    """Set up distributed environment for testing."""
    # Set up environment variables for distributed training
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port

    # Initialize torch.distributed first
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=0,
            world_size=1,
        )

    # Initialize megatron parallel
    import megatron.core.parallel_state as parallel_state
    import megatron.core.tensor_parallel.random as tensor_parallel_random

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )

    # Set up CUDA RNG states for model parallel
    tensor_parallel_random.model_parallel_cuda_manual_seed(42)

    try:
        yield
    finally:
        # Clean up megatron parallel
        parallel_state.destroy_model_parallel()
        # Clean up distributed
        if dist.is_initialized():
            dist.destroy_process_group()


def create_model_and_converter(megatron_config, hf_config, model_name):
    """Create megatron model and converter for testing."""

    from nemo.collections.llm.gpt.model.qwen3 import Qwen3Model

    from nemo_rl.models.megatron.converters.common import MegatronToHFConverter

    # Create megatron model
    model = Qwen3Model(megatron_config)
    model.configure_model()

    # Create dummy HF config and save to temporary directory
    with TemporaryDirectory() as tmp_dir:
        hf_dir = os.path.join(tmp_dir, model_name)
        hf_config.save_pretrained(hf_dir)

        # Create a dummy HF model to get the model class
        dummy_model = AutoModelForCausalLM.from_config(
            hf_config, trust_remote_code=True
        )
        dummy_model.save_pretrained(hf_dir)

        original_state_dict = model.module.state_dict()

        converter = MegatronToHFConverter(
            hf_model_name=hf_dir,
            megatron_model=model.module,
        )

        converted_state_dict = converter.convert(original_state_dict, model.config)

        # Filter out _extra_state keys
        original_state_dict = {
            k: v for k, v in original_state_dict.items() if "_extra_state" not in k
        }

        return original_state_dict, converted_state_dict, hf_config, model


def calculate_chunk_sizes(hf_config):
    """Calculate chunk sizes for QKV tensor splitting."""
    q_chunk_size = hf_config.head_dim * (
        hf_config.num_attention_heads // hf_config.num_key_value_heads
    )
    kv_chunk_size = hf_config.head_dim * 2
    return q_chunk_size, kv_chunk_size


def assert_attention_tensors_match(
    original_state_dict, converted_state_dict, q_chunk_size, kv_chunk_size
):
    """Assert that attention tensors match between original and converted state dicts."""
    # Check q_layernorm
    torch.testing.assert_close(
        original_state_dict["decoder.layers.0.self_attention.q_layernorm.weight"],
        converted_state_dict["model.layers.0.self_attn.q_norm.weight"],
    )

    # Check first layer q_proj
    torch.testing.assert_close(
        original_state_dict["decoder.layers.0.self_attention.linear_qkv.weight"][
            :q_chunk_size
        ],
        converted_state_dict["model.layers.0.self_attn.q_proj.weight"][:q_chunk_size],
    )

    # Check second layer q_proj
    torch.testing.assert_close(
        original_state_dict["decoder.layers.1.self_attention.linear_qkv.weight"][
            (q_chunk_size + kv_chunk_size) : (2 * q_chunk_size + kv_chunk_size)
        ],
        converted_state_dict["model.layers.1.self_attn.q_proj.weight"][
            q_chunk_size : (2 * q_chunk_size)
        ],
    )


@pytest.mark.mcore
def test_conversion_to_hf_moe():
    """Test conversion of Qwen3 MoE model to HF format."""
    with setup_distributed_environment("6000"):
        mcore_config = dummy_qwen3_megatron_moe_config()
        hf_config = create_dummy_hf_moe_config()

        original_state_dict, converted_state_dict, hf_config, model = (
            create_model_and_converter(mcore_config, hf_config, "Qwen3-tiny-test-moe")
        )

        # Check that the number of keys in the original state dict is equal to the number of keys in the converted state dict minus the number of extra state keys
        # taking into account the qkv merging and the merging of the up and gate projections
        assert len(original_state_dict) == len(converted_state_dict) - (
            2 * hf_config.num_hidden_layers
            + (hf_config.num_hidden_layers * hf_config.num_experts)
        )

        q_chunk_size, kv_chunk_size = calculate_chunk_sizes(hf_config)

        # Check attention tensors
        assert_attention_tensors_match(
            original_state_dict, converted_state_dict, q_chunk_size, kv_chunk_size
        )

        # Check MoE MLP tensors
        torch.testing.assert_close(
            original_state_dict["decoder.layers.1.mlp.experts.linear_fc1.weight0"][
                mcore_config.moe_ffn_hidden_size :
            ],
            converted_state_dict["model.layers.1.mlp.experts.0.up_proj.weight"],
        )
        torch.testing.assert_close(
            original_state_dict["decoder.layers.1.mlp.experts.linear_fc1.weight0"][
                : mcore_config.moe_ffn_hidden_size
            ],
            converted_state_dict["model.layers.1.mlp.experts.0.gate_proj.weight"],
        )
        torch.testing.assert_close(
            original_state_dict["decoder.layers.0.mlp.experts.linear_fc2.weight1"],
            converted_state_dict["model.layers.0.mlp.experts.1.down_proj.weight"],
        )


@pytest.mark.mcore
def test_conversion_to_hf_dense():
    """Test conversion of Qwen3 dense model to HF format."""
    with setup_distributed_environment("6001"):
        mcore_config = dummy_qwen3_megatron_dense_config()
        hf_config = create_dummy_hf_dense_config()

        original_state_dict, converted_state_dict, hf_config, model = (
            create_model_and_converter(mcore_config, hf_config, "Qwen3-tiny-test-dense")
        )

        # Check that the number of keys in the original state dict is equal to the number of keys in the converted state dict minus the number of extra state keys
        # taking into account the qkv merging and the merging of the up and gate projections
        assert len(original_state_dict) == len(converted_state_dict) - (
            3 * hf_config.num_hidden_layers
        )

        q_chunk_size, kv_chunk_size = calculate_chunk_sizes(hf_config)

        # Check attention tensors
        assert_attention_tensors_match(
            original_state_dict, converted_state_dict, q_chunk_size, kv_chunk_size
        )

        # Check dense MLP tensors
        torch.testing.assert_close(
            original_state_dict["decoder.layers.1.mlp.linear_fc1.weight"][
                mcore_config.ffn_hidden_size :
            ],
            converted_state_dict["model.layers.1.mlp.up_proj.weight"],
        )
        torch.testing.assert_close(
            original_state_dict["decoder.layers.1.mlp.linear_fc1.weight"][
                : mcore_config.ffn_hidden_size
            ],
            converted_state_dict["model.layers.1.mlp.gate_proj.weight"],
        )
        torch.testing.assert_close(
            original_state_dict["decoder.layers.0.mlp.linear_fc2.weight"],
            converted_state_dict["model.layers.0.mlp.down_proj.weight"],
        )
