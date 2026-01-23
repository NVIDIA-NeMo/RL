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
Unit tests for Megatron setup utilities.

This module tests the configuration validation and setup functions in
nemo_rl.models.megatron.setup, focusing on:
- Configuration validation functions
- Parallelism configuration application
- Precision and dtype configuration
- Checkpoint configuration creation
- Model path validation
"""

from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.mark.mcore
class TestValidateModelPaths:
    """Tests for validate_model_paths function."""

    def test_model_name_is_hf_model(self, tmp_path):
        """Test with a HuggingFace model name (not a local path)."""
        from nemo_rl.models.megatron.setup import validate_model_paths

        config = {"model_name": "meta-llama/Llama-3.2-1B"}

        with patch(
            "nemo_rl.models.megatron.setup.get_megatron_checkpoint_dir",
            return_value=str(tmp_path),
        ):
            hf_model_name, pretrained_path, pt_checkpoint_exists = validate_model_paths(
                config
            )

        assert hf_model_name == "meta-llama/Llama-3.2-1B"
        assert pretrained_path == f"{tmp_path}/meta-llama/Llama-3.2-1B"
        assert pt_checkpoint_exists is False

    def test_model_name_is_local_path(self, tmp_path):
        """Test with a local path as model name."""
        from nemo_rl.models.megatron.setup import validate_model_paths

        local_model_path = tmp_path / "local_model"
        local_model_path.mkdir()

        config = {"model_name": str(local_model_path)}

        with patch(
            "nemo_rl.models.megatron.setup.get_megatron_checkpoint_dir",
            return_value=str(tmp_path / "checkpoints"),
        ):
            hf_model_name, pretrained_path, pt_checkpoint_exists = validate_model_paths(
                config
            )

        assert hf_model_name == str(local_model_path)
        # Local path should be converted to model_<path> format
        assert "model_" in pretrained_path
        assert pt_checkpoint_exists is False

    def test_checkpoint_exists(self, tmp_path):
        """Test when a Megatron checkpoint already exists."""
        from nemo_rl.models.megatron.setup import validate_model_paths

        # Create the checkpoint directory structure
        checkpoint_dir = tmp_path / "checkpoints" / "test-model"
        iter_dir = checkpoint_dir / "iter_0000000"
        iter_dir.mkdir(parents=True)

        config = {"model_name": "test-model"}

        with patch(
            "nemo_rl.models.megatron.setup.get_megatron_checkpoint_dir",
            return_value=str(tmp_path / "checkpoints"),
        ):
            hf_model_name, pretrained_path, pt_checkpoint_exists = validate_model_paths(
                config
            )

        assert hf_model_name == "test-model"
        assert pt_checkpoint_exists is True


@pytest.mark.mcore
class TestApplyParallelismConfig:
    """Tests for _apply_parallelism_config function."""

    def test_basic_parallelism_config(self):
        """Test applying basic parallelism configuration."""
        from nemo_rl.models.megatron.setup import _apply_parallelism_config

        model_cfg = MagicMock()
        config = {
            "megatron_cfg": {
                "tensor_model_parallel_size": 4,
                "pipeline_model_parallel_size": 2,
                "num_layers_in_first_pipeline_stage": None,
                "num_layers_in_last_pipeline_stage": None,
                "sequence_parallel": True,
                "context_parallel_size": 1,
            },
            "sequence_packing": {"enabled": False},
        }

        _apply_parallelism_config(model_cfg, config)

        assert model_cfg.tensor_model_parallel_size == 4
        assert model_cfg.pipeline_model_parallel_size == 2
        assert model_cfg.sequence_parallel is True
        assert model_cfg.context_parallel_size == 1

    def test_context_parallel_requires_sequence_packing(self):
        """Test that context parallelism > 1 requires sequence packing."""
        from nemo_rl.models.megatron.setup import _apply_parallelism_config

        model_cfg = MagicMock()
        config = {
            "megatron_cfg": {
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "num_layers_in_first_pipeline_stage": None,
                "num_layers_in_last_pipeline_stage": None,
                "sequence_parallel": False,
                "context_parallel_size": 2,
            },
            "sequence_packing": {"enabled": False},
        }

        with pytest.raises(AssertionError) as exc_info:
            _apply_parallelism_config(model_cfg, config)

        assert "Sequence Packing must be enabled" in str(exc_info.value)

    def test_context_parallel_with_sequence_packing(self):
        """Test context parallelism with sequence packing enabled."""
        from nemo_rl.models.megatron.setup import _apply_parallelism_config

        model_cfg = MagicMock()
        config = {
            "megatron_cfg": {
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "num_layers_in_first_pipeline_stage": None,
                "num_layers_in_last_pipeline_stage": None,
                "sequence_parallel": False,
                "context_parallel_size": 4,
            },
            "sequence_packing": {"enabled": True},
        }

        _apply_parallelism_config(model_cfg, config)

        assert model_cfg.context_parallel_size == 4


@pytest.mark.mcore
class TestApplyMoeConfig:
    """Tests for _apply_moe_config function."""

    def test_moe_configuration(self):
        """Test applying MoE configuration."""
        from nemo_rl.models.megatron.setup import _apply_moe_config

        model_cfg = MagicMock()
        config = {
            "megatron_cfg": {
                "expert_tensor_parallel_size": 2,
                "expert_model_parallel_size": 4,
                "moe_router_dtype": "float32",
                "moe_router_load_balancing_type": "none",
                "moe_router_bias_update_rate": 0.0,
                "moe_permute_fusion": True,
                "moe_enable_deepep": False,
                "moe_token_dispatcher_type": "allgather",
                "moe_shared_expert_overlap": True,
            }
        }

        _apply_moe_config(model_cfg, config)

        assert model_cfg.expert_tensor_parallel_size == 2
        assert model_cfg.expert_model_parallel_size == 4
        assert model_cfg.moe_router_dtype == "float32"
        assert model_cfg.moe_router_load_balancing_type == "none"
        assert model_cfg.moe_router_bias_update_rate == 0.0
        assert model_cfg.moe_permute_fusion is True
        assert model_cfg.moe_enable_deepep is False
        assert model_cfg.moe_token_dispatcher_type == "allgather"
        assert model_cfg.moe_shared_expert_overlap is True


@pytest.mark.mcore
class TestApplyPrecisionConfig:
    """Tests for _apply_precision_config function."""

    @pytest.mark.parametrize(
        "dtype,expected_bf16,expected_fp16,expected_params_dtype",
        [
            (torch.bfloat16, True, False, torch.bfloat16),
            (torch.float16, False, True, torch.float16),
            (torch.float32, False, False, torch.float32),
        ],
        ids=["bfloat16", "float16", "float32"],
    )
    def test_precision_configurations(
        self, dtype, expected_bf16, expected_fp16, expected_params_dtype
    ):
        """Test precision configuration for different dtypes."""
        from nemo_rl.models.megatron.setup import _apply_precision_config

        model_cfg = MagicMock()
        model_cfg.bf16 = False
        model_cfg.fp16 = False
        config = {
            "megatron_cfg": {
                "pipeline_dtype": "bfloat16",
            }
        }

        _apply_precision_config(model_cfg, config, dtype)

        assert model_cfg.bf16 == expected_bf16
        assert model_cfg.fp16 == expected_fp16
        assert model_cfg.params_dtype == expected_params_dtype

    def test_pipeline_dtype_mapping(self):
        """Test that pipeline dtype is correctly mapped."""
        from nemo_rl.models.megatron.setup import _apply_precision_config

        model_cfg = MagicMock()
        model_cfg.bf16 = False
        model_cfg.fp16 = False

        for dtype_str, expected_dtype in [
            ("float32", torch.float32),
            ("bfloat16", torch.bfloat16),
            ("float16", torch.float16),
        ]:
            config = {
                "megatron_cfg": {
                    "pipeline_dtype": dtype_str,
                }
            }
            _apply_precision_config(model_cfg, config, torch.float32)
            assert model_cfg.pipeline_dtype == expected_dtype


@pytest.mark.mcore
class TestApplyPerformanceConfig:
    """Tests for _apply_performance_config function."""

    def test_basic_performance_config(self):
        """Test applying basic performance configuration."""
        from nemo_rl.models.megatron.setup import _apply_performance_config

        model_cfg = MagicMock()
        model_cfg.gated_linear_unit = True
        config = {
            "megatron_cfg": {
                "activation_checkpointing": False,
                "apply_rope_fusion": True,
                "bias_activation_fusion": True,
            }
        }

        _apply_performance_config(model_cfg, config)

        assert model_cfg.parallel_output is True
        assert model_cfg.apply_rope_fusion is True
        assert model_cfg.bias_activation_fusion is True

    def test_activation_checkpointing_enabled(self):
        """Test activation checkpointing configuration."""
        from nemo_rl.models.megatron.setup import _apply_performance_config

        model_cfg = MagicMock()
        model_cfg.gated_linear_unit = True
        config = {
            "megatron_cfg": {
                "activation_checkpointing": True,
                "apply_rope_fusion": False,
                "bias_activation_fusion": False,
            }
        }

        _apply_performance_config(model_cfg, config)

        assert model_cfg.recompute_granularity == "full"
        assert model_cfg.recompute_method == "uniform"
        assert model_cfg.recompute_num_layers == 1

    def test_activation_func_required_when_not_gated(self):
        """Test that activation_func is required when not using gated_linear_unit."""
        from nemo_rl.models.megatron.setup import _apply_performance_config

        model_cfg = MagicMock()
        model_cfg.gated_linear_unit = False
        model_cfg.activation_func = None
        config = {
            "megatron_cfg": {
                "activation_checkpointing": False,
                "apply_rope_fusion": False,
                "bias_activation_fusion": False,
            }
        }

        with pytest.raises(AssertionError) as exc_info:
            _apply_performance_config(model_cfg, config)

        assert "activation_func must be set" in str(exc_info.value)

    def test_fp8_configuration(self):
        """Test FP8 configuration."""
        from nemo_rl.models.megatron.setup import _apply_performance_config

        model_cfg = MagicMock()
        model_cfg.gated_linear_unit = True
        config = {
            "megatron_cfg": {
                "activation_checkpointing": False,
                "apply_rope_fusion": False,
                "bias_activation_fusion": False,
                "fp8_cfg": {
                    "enabled": True,
                    "fp8": "e4m3",
                    "fp8_recipe": "default",
                    "fp8_param": False,
                },
            }
        }

        _apply_performance_config(model_cfg, config)

        assert model_cfg.fp8 == "e4m3"
        assert model_cfg.fp8_recipe == "default"
        assert model_cfg.fp8_param is False

    def test_fp8_param_warning(self):
        """Test that fp8_param=True generates a warning."""
        from nemo_rl.models.megatron.setup import _apply_performance_config

        model_cfg = MagicMock()
        model_cfg.gated_linear_unit = True
        config = {
            "megatron_cfg": {
                "activation_checkpointing": False,
                "apply_rope_fusion": False,
                "bias_activation_fusion": False,
                "fp8_cfg": {
                    "enabled": True,
                    "fp8": "e4m3",
                    "fp8_recipe": "default",
                    "fp8_param": True,
                },
            }
        }

        with pytest.warns(UserWarning, match="fp8_param=True sometimes causes NaN"):
            _apply_performance_config(model_cfg, config)


@pytest.mark.mcore
class TestValidateOptimizerConfig:
    """Tests for _validate_optimizer_config function."""

    def test_cpu_offload_requires_full_fraction(self):
        """Test that CPU offload requires offload_fraction=1.0."""
        from nemo_rl.models.megatron.setup import _validate_optimizer_config

        config = {
            "megatron_cfg": {
                "optimizer": {
                    "optimizer_cpu_offload": True,
                    "optimizer_offload_fraction": 0.5,
                }
            }
        }

        with pytest.raises(AssertionError) as exc_info:
            _validate_optimizer_config(config)

        assert "optimizer_offload_fraction=1.0" in str(exc_info.value)

    def test_cpu_offload_with_full_fraction(self):
        """Test that CPU offload works with full fraction."""
        from nemo_rl.models.megatron.setup import _validate_optimizer_config

        config = {
            "megatron_cfg": {
                "optimizer": {
                    "optimizer_cpu_offload": True,
                    "optimizer_offload_fraction": 1.0,
                }
            }
        }

        # Should not raise
        _validate_optimizer_config(config)

    def test_no_cpu_offload(self):
        """Test configuration without CPU offload."""
        from nemo_rl.models.megatron.setup import _validate_optimizer_config

        config = {
            "megatron_cfg": {
                "optimizer": {
                    "optimizer_cpu_offload": False,
                    "optimizer_offload_fraction": 0.5,  # Should be ignored
                }
            }
        }

        # Should not raise
        _validate_optimizer_config(config)


@pytest.mark.mcore
class TestValidateChunkingConfig:
    """Tests for _validate_chunking_config function."""

    def test_logprob_chunk_requires_defer_fp32_logits(self):
        """Test that logprob chunking requires defer_fp32_logits=True."""
        from nemo_rl.models.megatron.setup import _validate_chunking_config

        config = {
            "logprob_chunk_size": 1024,
            "megatron_cfg": {
                "defer_fp32_logits": False,
            },
        }

        with pytest.raises(AssertionError) as exc_info:
            _validate_chunking_config(config)

        assert "defer_fp32_logits must be True" in str(exc_info.value)

    def test_logprob_chunk_with_defer_fp32_logits(self):
        """Test that logprob chunking works with defer_fp32_logits=True."""
        from nemo_rl.models.megatron.setup import _validate_chunking_config

        config = {
            "logprob_chunk_size": 1024,
            "megatron_cfg": {
                "defer_fp32_logits": True,
            },
        }

        # Should not raise
        _validate_chunking_config(config)

    @pytest.mark.parametrize(
        "logprob_chunk_size",
        [None, 0, -1],
        ids=["none", "zero", "negative"],
    )
    def test_no_chunking_skips_validation(self, logprob_chunk_size):
        """Test that validation is skipped when chunking is disabled."""
        from nemo_rl.models.megatron.setup import _validate_chunking_config

        config = {
            "logprob_chunk_size": logprob_chunk_size,
            "megatron_cfg": {
                "defer_fp32_logits": False,  # Doesn't matter when chunking is disabled
            },
        }

        # Should not raise
        _validate_chunking_config(config)

    def test_missing_logprob_chunk_size(self):
        """Test that missing logprob_chunk_size is handled."""
        from nemo_rl.models.megatron.setup import _validate_chunking_config

        config = {
            "megatron_cfg": {
                "defer_fp32_logits": False,
            },
        }

        # Should not raise
        _validate_chunking_config(config)


@pytest.mark.mcore
class TestCreateCheckpointConfig:
    """Tests for _create_checkpoint_config function."""

    def test_basic_checkpoint_config(self, tmp_path):
        """Test creating basic checkpoint configuration."""
        from nemo_rl.models.megatron.setup import _create_checkpoint_config

        pretrained_path = str(tmp_path / "pretrained")
        weights_path = str(tmp_path / "weights")

        checkpoint_config = _create_checkpoint_config(pretrained_path, weights_path)

        assert checkpoint_config.save == weights_path
        assert checkpoint_config.load == weights_path
        assert checkpoint_config.pretrained_checkpoint == pretrained_path
        assert checkpoint_config.async_save is False
        assert checkpoint_config.fully_parallel_save is True
        assert checkpoint_config.fully_parallel_load is True
        assert checkpoint_config.load_rng is False


@pytest.mark.mcore
class TestValidateTrainingConfig:
    """Tests for _validate_training_config function."""

    def test_train_iters_required(self):
        """Test that train_iters must be set."""
        from nemo_rl.models.megatron.setup import _validate_training_config

        model_cfg = MagicMock()
        model_cfg.moe_router_load_balancing_type = "none"
        model_cfg.moe_aux_loss_coeff = 0
        config = {
            "megatron_cfg": {},
        }

        with pytest.raises(AssertionError) as exc_info:
            _validate_training_config(config, model_cfg)

        assert "train_iters must be set" in str(exc_info.value)

    def test_training_config_sets_required_flags(self):
        """Test that training config sets required model flags."""
        from nemo_rl.models.megatron.setup import _validate_training_config

        model_cfg = MagicMock()
        model_cfg.moe_router_load_balancing_type = "none"
        model_cfg.moe_aux_loss_coeff = 0
        config = {
            "megatron_cfg": {
                "train_iters": 1000,
            },
        }

        _validate_training_config(config, model_cfg)

        assert model_cfg.calculate_per_token_loss is True
        assert model_cfg.perform_initialization is True

    def test_moe_aux_loss_not_supported(self):
        """Test that MoE aux loss is not supported."""
        from nemo_rl.models.megatron.setup import _validate_training_config

        model_cfg = MagicMock()
        model_cfg.moe_router_load_balancing_type = "aux_loss"
        model_cfg.moe_aux_loss_coeff = 0.1  # Non-zero
        config = {
            "megatron_cfg": {
                "train_iters": 1000,
            },
        }

        with pytest.raises(AssertionError) as exc_info:
            _validate_training_config(config, model_cfg)

        assert "MoE aux loss is currently not supported" in str(exc_info.value)

    def test_moe_aux_loss_with_zero_coeff_is_ok(self):
        """Test that MoE aux loss with zero coefficient is allowed."""
        from nemo_rl.models.megatron.setup import _validate_training_config

        model_cfg = MagicMock()
        model_cfg.moe_router_load_balancing_type = "aux_loss"
        model_cfg.moe_aux_loss_coeff = 0  # Zero is OK
        config = {
            "megatron_cfg": {
                "train_iters": 1000,
            },
        }

        # Should not raise
        _validate_training_config(config, model_cfg)


@pytest.mark.mcore
class TestValidateDtypeConfig:
    """Tests for _validate_dtype_config function."""

    def test_bfloat16_validation(self):
        """Test bfloat16 dtype validation."""
        from nemo_rl.models.megatron.setup import _validate_dtype_config

        model_cfg = MagicMock()
        model_cfg.bf16 = True
        model_cfg.fp16 = False

        optimizer_cfg = MagicMock()
        optimizer_cfg.use_precision_aware_optimizer = False
        optimizer_cfg.bf16 = False
        optimizer_cfg.fp16 = False

        # Should not raise
        _validate_dtype_config(torch.bfloat16, model_cfg, optimizer_cfg)

    def test_bfloat16_model_flag_mismatch(self):
        """Test bfloat16 validation fails when model.bf16=False."""
        from nemo_rl.models.megatron.setup import _validate_dtype_config

        model_cfg = MagicMock()
        model_cfg.bf16 = False  # Mismatch!
        model_cfg.fp16 = False

        optimizer_cfg = MagicMock()
        optimizer_cfg.use_precision_aware_optimizer = False

        with pytest.raises(AssertionError) as exc_info:
            _validate_dtype_config(torch.bfloat16, model_cfg, optimizer_cfg)

        assert "bf16=True must be set" in str(exc_info.value)

    def test_bfloat16_with_precision_aware_optimizer(self):
        """Test bfloat16 with precision aware optimizer requires optimizer.bf16=True."""
        from nemo_rl.models.megatron.setup import _validate_dtype_config

        model_cfg = MagicMock()
        model_cfg.bf16 = True
        model_cfg.fp16 = False

        optimizer_cfg = MagicMock()
        optimizer_cfg.use_precision_aware_optimizer = True
        optimizer_cfg.bf16 = False  # Mismatch!

        with pytest.raises(AssertionError) as exc_info:
            _validate_dtype_config(torch.bfloat16, model_cfg, optimizer_cfg)

        assert "optimizer.bf16=True must be set" in str(exc_info.value)

    def test_float16_validation(self):
        """Test float16 dtype validation."""
        from nemo_rl.models.megatron.setup import _validate_dtype_config

        model_cfg = MagicMock()
        model_cfg.bf16 = False
        model_cfg.fp16 = True

        optimizer_cfg = MagicMock()
        optimizer_cfg.use_precision_aware_optimizer = False

        # Should not raise
        _validate_dtype_config(torch.float16, model_cfg, optimizer_cfg)

    def test_float16_model_flag_mismatch(self):
        """Test float16 validation fails when model.fp16=False."""
        from nemo_rl.models.megatron.setup import _validate_dtype_config

        model_cfg = MagicMock()
        model_cfg.bf16 = False
        model_cfg.fp16 = False  # Mismatch!

        optimizer_cfg = MagicMock()
        optimizer_cfg.use_precision_aware_optimizer = False

        with pytest.raises(AssertionError) as exc_info:
            _validate_dtype_config(torch.float16, model_cfg, optimizer_cfg)

        assert "fp16=True must be set" in str(exc_info.value)

    def test_float32_validation(self):
        """Test float32 dtype validation."""
        from nemo_rl.models.megatron.setup import _validate_dtype_config

        model_cfg = MagicMock()
        model_cfg.bf16 = False
        model_cfg.fp16 = False

        optimizer_cfg = MagicMock()
        optimizer_cfg.bf16 = False
        optimizer_cfg.fp16 = False

        # Should not raise
        _validate_dtype_config(torch.float32, model_cfg, optimizer_cfg)

    def test_float32_with_bf16_model_flag(self):
        """Test float32 validation fails when model has bf16=True."""
        from nemo_rl.models.megatron.setup import _validate_dtype_config

        model_cfg = MagicMock()
        model_cfg.bf16 = True  # Mismatch!
        model_cfg.fp16 = False

        optimizer_cfg = MagicMock()
        optimizer_cfg.bf16 = False
        optimizer_cfg.fp16 = False

        with pytest.raises(AssertionError) as exc_info:
            _validate_dtype_config(torch.float32, model_cfg, optimizer_cfg)

        assert "bf16=False" in str(exc_info.value)

    def test_float32_with_fp16_optimizer_flag(self):
        """Test float32 validation fails when optimizer has fp16=True."""
        from nemo_rl.models.megatron.setup import _validate_dtype_config

        model_cfg = MagicMock()
        model_cfg.bf16 = False
        model_cfg.fp16 = False

        optimizer_cfg = MagicMock()
        optimizer_cfg.bf16 = False
        optimizer_cfg.fp16 = True  # Mismatch!

        with pytest.raises(AssertionError) as exc_info:
            _validate_dtype_config(torch.float32, model_cfg, optimizer_cfg)

        assert "optimizer" in str(exc_info.value).lower()


@pytest.mark.mcore
class TestValidateAndSetConfig:
    """Tests for validate_and_set_config function."""

    def test_reward_model_not_supported(self):
        """Test that reward models are not supported."""
        from nemo_rl.models.megatron.setup import validate_and_set_config

        config = {
            "reward_model_cfg": {"enabled": True},
            "precision": "bfloat16",
            "megatron_cfg": {
                "optimizer": {
                    "optimizer_cpu_offload": False,
                },
            },
            "offload_optimizer_for_logprob": False,
        }

        with pytest.raises(NotImplementedError) as exc_info:
            validate_and_set_config(
                config=config,
                rank=0,
                hf_model_name="test-model",
                pretrained_path="/path/to/model",
                weights_path=None,
                tokenizer=MagicMock(),
            )

        assert "Reward models are not yet supported" in str(exc_info.value)

    def test_generation_colocation_detection(self):
        """Test that generation colocation is properly detected."""
        # This test would require more mocking to fully test
        # For now, we just verify the config parsing works
        from nemo_rl.models.megatron.setup import validate_and_set_config

        config = {
            "generation": {
                "colocated": {"enabled": True},
            },
            "precision": "bfloat16",
            "megatron_cfg": {
                "optimizer": {
                    "optimizer_cpu_offload": False,
                },
                "tensor_model_parallel_size": 2,
            },
            "offload_optimizer_for_logprob": False,
        }

        # The function would fail on setup_model_config, but we test the initial parsing
        with patch(
            "nemo_rl.models.megatron.setup.setup_model_config"
        ) as mock_setup_model_config:
            mock_megatron_cfg = MagicMock()
            mock_megatron_cfg.model.vocab_size = 32000
            mock_setup_model_config.return_value = (mock_megatron_cfg, MagicMock())

            with patch(
                "nemo_rl.models.megatron.setup.calculate_padded_vocab_size",
                return_value=32000,
            ):
                runtime_config = validate_and_set_config(
                    config=config,
                    rank=0,
                    hf_model_name="test-model",
                    pretrained_path="/path/to/model",
                    weights_path=None,
                    tokenizer=MagicMock(),
                )

                assert runtime_config.is_generation_colocated is True


@pytest.mark.mcore
class TestRuntimeConfigNamedTuple:
    """Tests for RuntimeConfig named tuple."""

    def test_runtime_config_fields(self):
        """Test that RuntimeConfig has all expected fields."""
        from nemo_rl.models.megatron.config import RuntimeConfig

        runtime_config = RuntimeConfig(
            megatron_cfg=MagicMock(),
            model_cfg=MagicMock(),
            dtype=torch.bfloat16,
            optimizer_cpu_offload=False,
            offload_optimizer_for_logprob=True,
            is_generation_colocated=True,
            final_padded_vocab_size=32000,
        )

        assert runtime_config.dtype == torch.bfloat16
        assert runtime_config.optimizer_cpu_offload is False
        assert runtime_config.offload_optimizer_for_logprob is True
        assert runtime_config.is_generation_colocated is True
        assert runtime_config.final_padded_vocab_size == 32000


@pytest.mark.mcore
class TestModelAndOptimizerStateNamedTuple:
    """Tests for ModelAndOptimizerState named tuple."""

    def test_model_and_optimizer_state_fields(self):
        """Test that ModelAndOptimizerState has all expected fields."""
        from nemo_rl.models.megatron.config import ModelAndOptimizerState

        state = ModelAndOptimizerState(
            state=MagicMock(),
            model=MagicMock(),
            optimizer=MagicMock(),
            scheduler=MagicMock(),
            checkpointing_context={"test": "context"},
            param_sync_func=lambda: None,
        )

        assert state.checkpointing_context == {"test": "context"}
        assert callable(state.param_sync_func)


@pytest.mark.mcore
class TestHandleModelImport:
    """Tests for handle_model_import function."""

    def test_skip_import_when_checkpoint_exists(self, tmp_path, capsys):
        """Test that import is skipped when checkpoint exists."""
        from nemo_rl.models.megatron.setup import handle_model_import

        pretrained_path = str(tmp_path / "model")
        config = {"model_name": "test-model", "megatron_cfg": {}}

        handle_model_import(
            config, "test-model", pretrained_path, pt_checkpoint_exists=True
        )

        captured = capsys.readouterr()
        assert "Checkpoint already exists" in captured.out

    @patch("nemo_rl.models.megatron.setup.import_model_from_hf_name")
    @patch("nemo_rl.models.megatron.setup.parallel_state")
    def test_import_when_checkpoint_missing(self, mock_ps, mock_import, tmp_path):
        """Test that model is imported when checkpoint doesn't exist."""
        from nemo_rl.models.megatron.setup import handle_model_import

        mock_ps.model_parallel_is_initialized.return_value = False

        pretrained_path = str(tmp_path / "model")
        config = {
            "model_name": "test-model",
            "megatron_cfg": {"some_config": "value"},
            "hf_config_overrides": None,
        }

        handle_model_import(
            config, "test-model", pretrained_path, pt_checkpoint_exists=False
        )

        mock_import.assert_called_once_with(
            "test-model",
            pretrained_path,
            {"some_config": "value"},
        )

    @patch("nemo_rl.models.megatron.setup.import_model_from_hf_name")
    @patch("nemo_rl.models.megatron.setup.parallel_state")
    def test_reinitialize_parallel_state_after_import(
        self, mock_ps, mock_import, tmp_path, capsys
    ):
        """Test that parallel state is destroyed after model import."""
        from nemo_rl.models.megatron.setup import handle_model_import

        mock_ps.model_parallel_is_initialized.return_value = True

        pretrained_path = str(tmp_path / "model")
        config = {
            "model_name": "test-model",
            "megatron_cfg": {},
            "hf_config_overrides": {},
        }

        handle_model_import(
            config, "test-model", pretrained_path, pt_checkpoint_exists=False
        )

        mock_ps.destroy_model_parallel.assert_called_once()

        captured = capsys.readouterr()
        assert "Reinitializing model parallel" in captured.out


@pytest.mark.mcore
class TestSetupModelAndOptimizer:
    """Tests for setup_model_and_optimizer function."""

    @patch("nemo_rl.models.megatron.setup.GlobalState")
    @patch("nemo_rl.models.megatron.setup.initialize_megatron")
    @patch("nemo_rl.models.megatron.setup.set_jit_fusion_options")
    @patch("nemo_rl.models.megatron.setup.init_checkpointing_context")
    @patch("nemo_rl.models.megatron.setup.build_tokenizer")
    @patch("nemo_rl.models.megatron.setup.get_model")
    @patch("nemo_rl.models.megatron.setup.setup_optimizer")
    @patch("nemo_rl.models.megatron.setup.checkpoint_exists")
    @patch("nemo_rl.models.megatron.setup.MoEFloat16Module")
    @patch("torch.distributed.all_reduce")
    @patch("torch.distributed.barrier")
    @patch("torch.tensor")
    def test_setup_with_param_sync_and_frozen_moe_router(
        self,
        mock_tensor,
        mock_barrier,
        mock_all_reduce,
        mock_custom_float16,
        mock_checkpoint_exists,
        mock_setup_optimizer,
        mock_get_model,
        mock_build_tokenizer,
        mock_init_ckpt_context,
        mock_set_jit,
        mock_init_megatron,
        mock_global_state,
    ):
        """Test setup_model_and_optimizer with MoE router freezing."""
        from nemo_rl.models.megatron.setup import setup_model_and_optimizer

        # Setup mocks
        mock_state = MagicMock()
        mock_state.start_time = 0.0
        mock_global_state.return_value = mock_state

        mock_megatron_cfg = MagicMock()
        mock_megatron_cfg.ft = None
        mock_megatron_cfg.model.vocab_size = 32000
        mock_megatron_cfg.model.make_vocab_size_divisible_by = 128
        mock_megatron_cfg.model.tensor_model_parallel_size = 1
        # Enable param gather overlap
        mock_megatron_cfg.ddp.overlap_param_gather = True
        mock_megatron_cfg.ddp.align_param_gather = True
        mock_megatron_cfg.checkpoint.load = None
        mock_megatron_cfg.checkpoint.pretrained_checkpoint = None

        mock_model_chunk = MagicMock()
        mock_model_chunk.start_param_sync = MagicMock()
        mock_model = [mock_model_chunk]
        mock_get_model.return_value = mock_model

        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()
        mock_setup_optimizer.return_value = (mock_optimizer, mock_scheduler)

        mock_tensor_instance = MagicMock()
        mock_tensor_instance.item.return_value = 0.0
        mock_tensor.return_value = mock_tensor_instance

        mock_checkpoint_exists.return_value = False

        policy_cfg = {
            "megatron_cfg": {
                "freeze_moe_router": True,  # Enable MoE router freezing
            }
        }

        result = setup_model_and_optimizer(
            policy_cfg=policy_cfg,
            megatron_cfg=mock_megatron_cfg,
            load_optimizer=True,
        )

        # Verify get_model was called (the mixed_precision_wrapper should be CustomFloat16Module)
        mock_get_model.assert_called_once()
        call_kwargs = mock_get_model.call_args[1]
        # Check that pre_wrap_hook is not empty when freeze_moe_router is True
        assert len(call_kwargs.get("pre_wrap_hook", [])) > 0

        assert result.param_sync_func == mock_model_chunk.start_param_sync


@pytest.mark.mcore
class TestSetupReferenceModelState:
    """Tests for setup_reference_model_state function."""

    @patch("nemo_rl.models.megatron.setup.init_checkpointing_context")
    @patch("nemo_rl.models.megatron.setup.GlobalState")
    @patch("nemo_rl.models.megatron.setup.get_model")
    @patch("nemo_rl.models.megatron.setup.checkpoint_exists")
    @patch("nemo_rl.models.megatron.setup.load_checkpoint")
    @patch("nemo_rl.models.megatron.setup.HAVE_FSDP2", False)
    def test_setup_reference_model(
        self,
        mock_load_checkpoint,
        mock_checkpoint_exists,
        mock_get_model,
        mock_global_state,
        mock_init_ckpt_context,
        capsys,
    ):
        """Test setup_reference_model_state when checkpoint exists."""
        from nemo_rl.models.megatron.setup import setup_reference_model_state

        # Setup mocks
        mock_state = MagicMock()
        mock_global_state.return_value = mock_state

        mock_megatron_cfg = MagicMock()
        mock_megatron_cfg.dist.use_torch_fsdp2 = False

        # Create mock model with state dict
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {
            "layer1.weight": torch.tensor([1.0, 2.0]),
            "layer1.bias": torch.tensor([0.1]),
        }
        mock_get_model.return_value = [mock_model]

        mock_checkpoint_exists.return_value = True

        config = {
            "megatron_cfg": {
                "freeze_moe_router": False,
            }
        }

        result = setup_reference_model_state(
            config=config,
            megatron_cfg=mock_megatron_cfg,
            pretrained_path="/path/to/pretrained",
        )

        # Verify checkpoint was loaded
        mock_load_checkpoint.assert_called_once()

        # Verify model was set to eval mode
        mock_model.eval.assert_called_once()

        # Verify state dict is returned
        assert isinstance(result, dict)
        assert "layer1.weight" in result
        assert "layer1.bias" in result

        # Verify tensors are on CPU
        assert result["layer1.weight"].device.type == "cpu"

        captured = capsys.readouterr()
        assert "Reference model loaded" in captured.out


@pytest.mark.mcore
class TestFinalizeMegatronSetup:
    """Tests for finalize_megatron_setup function."""

    @patch("nemo_rl.models.megatron.setup._update_model_config_funcs")
    @patch("nemo_rl.models.megatron.setup.build_tokenizer")
    @patch("nemo_rl.models.megatron.setup.AutoBridge")
    def test_basic_finalize_setup(
        self,
        mock_auto_bridge,
        mock_build_tokenizer,
        mock_update_model_config,
    ):
        """Test basic finalize_megatron_setup."""
        from nemo_rl.models.megatron.setup import finalize_megatron_setup

        # Setup mocks
        mock_megatron_cfg = MagicMock()
        mock_megatron_cfg.model.make_vocab_size_divisible_by = 128

        mock_model = MagicMock()
        mock_optimizer = MagicMock()

        mock_worker_sharding = MagicMock()
        mock_worker_sharding.get_axis_size.return_value = 4  # dp_size = 4

        mock_tokenizer = MagicMock()
        mock_build_tokenizer.return_value = mock_tokenizer

        mock_bridge = MagicMock()
        mock_auto_bridge.from_hf_pretrained.return_value = mock_bridge

        config = {
            "megatron_cfg": {
                "tensor_model_parallel_size": 2,
                "optimizer": {
                    "use_distributed_optimizer": False,
                },
                "distributed_data_parallel_config": {
                    "overlap_param_gather": False,
                },
            }
        }

        result = finalize_megatron_setup(
            config=config,
            megatron_cfg=mock_megatron_cfg,
            hf_model_name="test-model",
            worker_sharding_annotations=mock_worker_sharding,
            model=mock_model,
            optimizer=mock_optimizer,
        )

        # Verify return values
        megatron_tokenizer, megatron_bridge, should_disable_hook, dp_size = result
        assert megatron_tokenizer == mock_tokenizer
        assert megatron_bridge == mock_bridge
        assert should_disable_hook is False
        assert dp_size == 4

        # Verify function calls
        mock_update_model_config.assert_called_once()
        mock_build_tokenizer.assert_called_once()
        mock_auto_bridge.from_hf_pretrained.assert_called_once_with(
            "test-model", trust_remote_code=True
        )
