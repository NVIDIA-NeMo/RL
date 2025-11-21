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

"""
Unit tests for Policy class validation logic.

This module tests the early validation checks in the Policy class, particularly
the world_size compatibility validation that prevents confusing reshape errors
when the cluster size is insufficient for the specified parallelism configuration.
"""

from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.lm_policy import Policy


def create_mock_cluster(world_size: int):
    """Create a mock cluster with the specified world size."""
    cluster = MagicMock()
    cluster.world_size.return_value = world_size

    # Mock get_master_address_and_port method to return valid address and port
    cluster.get_master_address_and_port.return_value = ("127.0.0.1", 29500)

    # Mock get_placement_groups method to return a list of mock placement groups
    mock_pg = MagicMock()
    mock_pg.bundle_count = world_size  # Each placement group has world_size bundles
    cluster.get_placement_groups.return_value = [mock_pg]

    # Mock get_available_address_and_port method
    cluster.get_available_address_and_port.return_value = ("127.0.0.1", 29501)

    return cluster


def create_mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    return tokenizer


def create_dtensor_config(
    model_name: str, tp: int, pp: int = 1, cp: int = 1
) -> PolicyConfig:
    """Create a DTensor configuration for testing."""
    return {
        "model_name": model_name,
        "tokenizer": {"name": model_name},
        "generation_batch_size": 1,
        "train_global_batch_size": 4,
        "train_micro_batch_size": 1,
        "learning_rate": 5e-6,
        "logprob_batch_size": 1,
        "precision": "float32",
        "offload_optimizer_for_logprob": False,
        "generation": {
            "backend": "hf",
            "temperature": 1.0,
            "max_new_tokens": 16,
            "top_p": 1.0,
            "top_k": None,
            "stop_token_ids": None,
            "stop_strings": None,
            "colocated": {
                "enabled": True,
                "resources": {
                    "gpus_per_node": None,
                    "num_nodes": None,
                },
            },
        },
        "dtensor_cfg": {
            "enabled": True,
            "cpu_offload": False,
            "sequence_parallel": False,
            "activation_checkpointing": False,
            "tensor_parallel_size": tp,
            "context_parallel_size": cp,
        },
        "dynamic_batching": {
            "enabled": True,
            "train_mb_tokens": 128,
            "logprob_mb_tokens": 128,
            "sequence_length_round": 4,
        },
        "sequence_packing": {
            "enabled": False,
        },
        "optimizer": {
            "name": "torch.optim.AdamW",
            "lr": 5e-6,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
        },
    }


def create_megatron_config(
    model_name: str, tp: int, pp: int = 1, cp: int = 1
) -> PolicyConfig:
    """Create a Megatron configuration for testing."""
    return {
        "model_name": model_name,
        "tokenizer": {"name": model_name},
        "generation_batch_size": 1,
        "train_global_batch_size": 4,
        "train_micro_batch_size": 1,
        "learning_rate": 5e-6,
        "logprob_batch_size": 1,
        "precision": "float32",
        "offload_optimizer_for_logprob": False,
        "generation": {
            "backend": "hf",
            "temperature": 1.0,
            "max_new_tokens": 16,
            "top_p": 1.0,
            "top_k": None,
            "stop_token_ids": None,
            "stop_strings": None,
            "colocated": {
                "enabled": True,
                "resources": {
                    "gpus_per_node": None,
                    "num_nodes": None,
                },
            },
        },
        "megatron_cfg": {
            "enabled": True,
            "tensor_model_parallel_size": tp,
            "pipeline_model_parallel_size": pp,
            "context_parallel_size": cp,
        },
        "dynamic_batching": {
            "enabled": pp == 1,  # Only enable for single pipeline parallel stage
            "train_mb_tokens": 128,
            "logprob_mb_tokens": 128,
            "sequence_length_round": 4,
        },
        "sequence_packing": {
            "enabled": False,
        },
        "optimizer": {
            "name": "torch.optim.AdamW",
            "lr": 5e-6,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
        },
    }


@pytest.mark.parametrize(
    "world_size,tp,cp,should_pass,expected_error_type,description",
    [
        # Valid cases - DTensor backend (PP is always 1 for DTensor)
        (8, 8, 1, True, None, "Valid: DP=1, TP=8, PP=1, CP=1"),
        (16, 8, 1, True, None, "Valid: DP=2, TP=8, PP=1, CP=1"),
        (8, 4, 2, True, None, "Valid: DP=1, TP=4, PP=1, CP=2"),
        (16, 4, 2, True, None, "Valid: DP=2, TP=4, PP=1, CP=2"),
        (1, 1, 1, True, None, "Valid: Minimal config DP=1, TP=1, PP=1, CP=1"),
        # Invalid cases - insufficient world_size (DP < 1)
        (4, 8, 1, False, "insufficient", "Invalid: DP=0.5, TP=8, PP=1, CP=1"),
        (2, 8, 1, False, "insufficient", "Invalid: DP=0.25, TP=8, PP=1, CP=1"),
        (4, 4, 2, False, "insufficient", "Invalid: DP=0.5, TP=4, PP=1, CP=2"),
        # Invalid cases - not divisible (DP not integer)
        (10, 4, 2, False, "divisible", "Invalid: DP=1.25, TP=4, PP=1, CP=2"),
        (9, 8, 1, False, "divisible", "Invalid: DP=1.125, TP=8, PP=1, CP=1"),
        (6, 4, 1, False, "divisible", "Invalid: DP=1.5, TP=4, PP=1, CP=1"),
    ],
)
@patch("nemo_rl.models.policy.lm_policy.RayWorkerGroup")
def test_world_size_validation_dtensor(
    mock_ray_worker_group,
    tiny_llama_model_path,
    world_size,
    tp,
    cp,
    should_pass,
    expected_error_type,
    description,
):
    """Test world_size validation with DTensor backend.

    Note: DTensor backend always uses PP=1 (no pipeline parallelism support).
    Tests the constraint: world_size = DP * PP * CP * TP where DP >= 1 and DP must be integer.
    """
    cluster = create_mock_cluster(world_size)
    tokenizer = create_mock_tokenizer()
    config = create_dtensor_config(
        tiny_llama_model_path, tp, pp=1, cp=cp
    )  # DTensor always has PP=1

    # Mock RayWorkerGroup to prevent actual worker creation
    mock_worker_group_instance = MagicMock()
    mock_ray_worker_group.return_value = mock_worker_group_instance

    if should_pass:
        # Should succeed without raising an exception
        try:
            policy = Policy(cluster=cluster, config=config, tokenizer=tokenizer)
            # Verify the calculated DP makes sense
            expected_dp = world_size // (1 * cp * tp)  # PP=1 for DTensor
            assert expected_dp >= 1, f"Expected DP should be >= 1, got {expected_dp}"
            # Verify that worker group was created (validation passed)
            mock_ray_worker_group.assert_called_once()
        except Exception as e:
            pytest.fail(f"Expected success for {description}, but got error: {e}")
    else:
        # Should raise ValueError with specific error type
        with pytest.raises(ValueError) as exc_info:
            Policy(cluster=cluster, config=config, tokenizer=tokenizer)

        error_msg = str(exc_info.value)
        if expected_error_type == "insufficient":
            assert "insufficient" in error_msg, (
                f"Expected 'insufficient' error for {description}"
            )
            assert "DP must be ≥ 1" in error_msg, (
                f"Expected DP constraint message for {description}"
            )
        elif expected_error_type == "divisible":
            assert "must be divisible" in error_msg, (
                f"Expected 'divisible' error for {description}"
            )
            assert "not an integer" in error_msg, (
                f"Expected integer constraint message for {description}"
            )
        # For failing cases, worker group should not be created
        mock_ray_worker_group.assert_not_called()


@pytest.mark.parametrize(
    "world_size,tp,pp,cp,should_pass,expected_error_type,description",
    [
        # Valid cases - Megatron backend (supports PP > 1)
        (
            32,
            8,
            4,
            1,
            True,
            None,
            "Valid: DP=1, TP=8, PP=4, CP=1 (original error case fixed)",
        ),
        (64, 8, 4, 1, True, None, "Valid: DP=2, TP=8, PP=4, CP=1"),
        (16, 4, 2, 2, True, None, "Valid: DP=1, TP=4, PP=2, CP=2"),
        # Invalid cases - insufficient world_size (DP < 1)
        (
            8,
            8,
            4,
            1,
            False,
            "insufficient",
            "Invalid: DP=0.25, TP=8, PP=4, CP=1 (original error)",
        ),
        (16, 8, 4, 1, False, "insufficient", "Invalid: DP=0.5, TP=8, PP=4, CP=1"),
        # Invalid cases - not divisible (DP not integer)
        (33, 8, 4, 1, False, "divisible", "Invalid: DP=1.03, TP=8, PP=4, CP=1"),
        (18, 4, 2, 2, False, "divisible", "Invalid: DP=1.125, TP=4, PP=2, CP=2"),
    ],
)
@patch("nemo_rl.models.policy.lm_policy.RayWorkerGroup")
def test_world_size_validation_megatron(
    mock_ray_worker_group,
    tiny_llama_model_path,
    world_size,
    tp,
    pp,
    cp,
    should_pass,
    expected_error_type,
    description,
):
    """Test world_size validation with Megatron backend.

    Megatron backend supports pipeline parallelism (PP > 1) unlike DTensor.
    Tests the constraint: world_size = DP * PP * CP * TP where DP >= 1 and DP must be integer.
    Note: Expert Parallelism (EP) is handled internally by Megatron-Core, not at the worker level.
    """
    cluster = create_mock_cluster(world_size)
    tokenizer = create_mock_tokenizer()
    config = create_megatron_config(tiny_llama_model_path, tp, pp, cp)

    # Mock RayWorkerGroup to prevent actual worker creation
    mock_worker_group_instance = MagicMock()
    mock_ray_worker_group.return_value = mock_worker_group_instance

    if should_pass:
        # Should succeed without raising an exception
        try:
            policy = Policy(cluster=cluster, config=config, tokenizer=tokenizer)
            # Verify the calculated DP makes sense
            expected_dp = world_size // (pp * cp * tp)
            assert expected_dp >= 1, f"Expected DP should be >= 1, got {expected_dp}"
            # Verify that worker group was created (validation passed)
            mock_ray_worker_group.assert_called_once()
        except Exception as e:
            pytest.fail(f"Expected success for {description}, but got error: {e}")
    else:
        # Should raise ValueError with specific error type
        with pytest.raises(ValueError) as exc_info:
            Policy(cluster=cluster, config=config, tokenizer=tokenizer)

        error_msg = str(exc_info.value)
        if expected_error_type == "insufficient":
            assert "insufficient" in error_msg, (
                f"Expected 'insufficient' error for {description}"
            )
            assert "DP must be ≥ 1" in error_msg, (
                f"Expected DP constraint message for {description}"
            )
        elif expected_error_type == "divisible":
            assert "must be divisible" in error_msg, (
                f"Expected 'divisible' error for {description}"
            )
            assert "not an integer" in error_msg, (
                f"Expected integer constraint message for {description}"
            )
        # For failing cases, worker group should not be created
        mock_ray_worker_group.assert_not_called()
