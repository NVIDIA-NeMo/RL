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
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from nemo_rl.models.automodel.setup import (
    DistributedState,
    ModelAndOptimizerState,
    setup_distributed,
    setup_model_and_optimizer,
    validate_and_set_config,
)
from nemo_rl.models.automodel.types import RuntimeConfig


@pytest.fixture
def mock_config():
    return {
        "model_name": "gpt2",
        "precision": "bfloat16",
        "max_grad_norm": 1.0,
        "offload_optimizer_for_logprob": False,
        "sequence_packing": {"enabled": False},
        "dtensor_cfg": {
            "cpu_offload": False,
            "context_parallel_size": 1,
            "tensor_parallel_size": 1,
            "expert_parallel_size": 1,
            "data_parallel_size": None,
            "sequence_parallel": False,
            "use_hf_tp_plan": False,
            "activation_checkpointing": False,
        },
        "generation": None,
        "hf_config_overrides": {},
        "optimizer": {
            "name": "torch.optim.AdamW",
            "kwargs": {"lr": 1e-4},
        },
    }


@pytest.fixture
def mock_autoconfig():
    config = MagicMock()
    config.architectures = ["GPT2LMHeadModel"]
    config.model_type = "gpt2"
    config.num_labels = 2
    config.torch_dtype = "float32"
    return config


@pytest.mark.automodel
class TestValidateAndSetConfig:
    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch("nemo_rl.models.automodel.setup.sliding_window_overwrite")
    def test_basic_validation(
        self,
        mock_sliding_window,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        mock_sliding_window.return_value = {}
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock

        result = validate_and_set_config(
            config=mock_config,
            processor=None,
            rank=0,
        )

        # Verify result is a RuntimeConfig dataclass
        assert isinstance(result, RuntimeConfig)
        assert result.is_vlm is False
        assert result.dtype == torch.bfloat16
        assert result.cpu_offload is False
        assert result.offload_optimizer_for_logprob is False
        assert result.max_grad_norm == 1.0
        assert result.enable_seq_packing is False
        assert result.is_reward_model is False

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch("nemo_rl.models.automodel.setup.sliding_window_overwrite")
    def test_vlm_detection(
        self,
        mock_sliding_window,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        mock_sliding_window.return_value = {}
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock

        processor = MagicMock()
        result = validate_and_set_config(
            config=mock_config,
            processor=processor,
            rank=0,
        )

        assert result.is_vlm is True

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch("nemo_rl.models.automodel.setup.sliding_window_overwrite")
    def test_precision_validation_invalid(
        self,
        mock_sliding_window,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
    ):
        mock_config["precision"] = "invalid_precision"

        with pytest.raises(ValueError, match="Unknown precision"):
            validate_and_set_config(
                config=mock_config,
                processor=None,
                rank=0,
            )

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch("nemo_rl.models.automodel.setup.sliding_window_overwrite")
    def test_sequence_packing_with_vlm_raises_error(
        self,
        mock_sliding_window,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
    ):
        mock_config["sequence_packing"]["enabled"] = True
        processor = MagicMock()

        with pytest.raises(
            ValueError, match="Sequence packing is not supported for VLM"
        ):
            validate_and_set_config(
                config=mock_config,
                processor=processor,
                rank=0,
            )

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch("nemo_rl.models.automodel.setup.sliding_window_overwrite")
    @patch("nemo_automodel.NeMoAutoModelForSequenceClassification")
    def test_reward_model_bradley_terry(
        self,
        mock_rm_class,
        mock_sliding_window,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        mock_sliding_window.return_value = {}
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig

        mock_config["reward_model_cfg"] = {
            "enabled": True,
            "reward_model_type": "bradley_terry",
        }

        result = validate_and_set_config(
            config=mock_config,
            processor=None,
            rank=0,
        )

        assert result.is_reward_model is True
        # Verify num_labels was set to 1
        assert mock_autoconfig.num_labels == 1

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch("nemo_rl.models.automodel.setup.sliding_window_overwrite")
    def test_context_parallel_with_sequence_packing_raises_error(
        self,
        mock_sliding_window,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
    ):
        mock_config["sequence_packing"]["enabled"] = True
        mock_config["dtensor_cfg"]["context_parallel_size"] = 2

        with pytest.raises(
            ValueError, match="Context parallel is not supported for sequence packing"
        ):
            validate_and_set_config(
                config=mock_config,
                processor=None,
                rank=0,
            )

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch("nemo_rl.models.automodel.setup.sliding_window_overwrite")
    def test_sequence_parallel_with_large_tp_raises_error(
        self,
        mock_sliding_window,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
    ):
        mock_config["dtensor_cfg"]["sequence_parallel"] = True
        mock_config["dtensor_cfg"]["tensor_parallel_size"] = 2

        with pytest.raises(RuntimeError, match="Sequence parallel \\+ tp_size >1"):
            validate_and_set_config(
                config=mock_config,
                processor=None,
                rank=0,
            )

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch("nemo_rl.models.automodel.setup.sliding_window_overwrite")
    def test_attention_implementation_selection(
        self,
        mock_sliding_window,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        mock_sliding_window.return_value = {}
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock

        # Test FA2 for sequence packing with cp=1
        mock_config["sequence_packing"]["enabled"] = True
        mock_config["dtensor_cfg"]["context_parallel_size"] = 1
        result = validate_and_set_config(mock_config, None, 0)
        assert result.attn_impl == "flash_attention_2"

        # Test SDPA for cp > 1
        mock_config["sequence_packing"]["enabled"] = False
        mock_config["dtensor_cfg"]["context_parallel_size"] = 2
        result = validate_and_set_config(mock_config, None, 0)
        assert result.attn_impl == "sdpa"

        # Test None for cp=1 without sequence packing
        mock_config["dtensor_cfg"]["context_parallel_size"] = 1
        result = validate_and_set_config(mock_config, None, 0)
        assert result.attn_impl is None

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch("nemo_rl.models.automodel.setup.sliding_window_overwrite")
    def test_precision_types(
        self,
        mock_sliding_window,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        mock_sliding_window.return_value = {}
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock

        # Test float32
        mock_config["precision"] = "float32"
        result = validate_and_set_config(mock_config, None, 0)
        assert result.dtype == torch.float32

        # Test float16
        mock_config["precision"] = "float16"
        result = validate_and_set_config(mock_config, None, 0)
        assert result.dtype == torch.float16

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch("nemo_rl.models.automodel.setup.sliding_window_overwrite")
    @patch("os.environ", {})
    def test_generation_colocated(
        self,
        mock_sliding_window,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        mock_sliding_window.return_value = {}
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock

        # Test generation colocated = True (should not set NCCL env var)
        mock_config["generation"] = {"colocated": {"enabled": True}}
        result = validate_and_set_config(mock_config, None, 0)
        assert result.is_generation_colocated is True
        assert "NCCL_CUMEM_ENABLE" not in os.environ

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch("nemo_rl.models.automodel.setup.sliding_window_overwrite")
    def test_sequence_packing_enabled_print(
        self,
        mock_sliding_window,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
        capsys,
    ):
        mock_sliding_window.return_value = {}
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock

        mock_config["sequence_packing"]["enabled"] = True
        mock_config["dtensor_cfg"]["context_parallel_size"] = 1
        result = validate_and_set_config(mock_config, None, 0)

        captured = capsys.readouterr()
        assert "Sequence packing is enabled" in captured.out
        assert "Using FlashAttention2" in captured.out

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch("nemo_rl.models.automodel.setup.sliding_window_overwrite")
    def test_hf_config_overrides_none(
        self,
        mock_sliding_window,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        mock_sliding_window.return_value = {}
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock

        mock_config["hf_config_overrides"] = None
        result = validate_and_set_config(mock_config, None, 0)
        assert result.hf_config_overrides == {}

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch("nemo_rl.models.automodel.setup.sliding_window_overwrite")
    @patch("nemo_automodel.NeMoAutoModelForSequenceClassification")
    def test_reward_model_with_num_labels_equals_one(
        self,
        mock_rm_class,
        mock_sliding_window,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        mock_sliding_window.return_value = {}
        mock_autoconfig.num_labels = 1  # Already 1
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig

        mock_config["reward_model_cfg"] = {
            "enabled": True,
            "reward_model_type": "bradley_terry",
        }

        result = validate_and_set_config(mock_config, None, 0)
        assert result.is_reward_model is True
        # num_labels should remain 1
        assert mock_autoconfig.num_labels == 1

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch("nemo_rl.models.automodel.setup.sliding_window_overwrite")
    def test_reward_model_with_sequence_packing_error(
        self,
        mock_sliding_window,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        mock_sliding_window.return_value = {}
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig

        mock_config["sequence_packing"]["enabled"] = True
        mock_config["dtensor_cfg"]["context_parallel_size"] = 1
        mock_config["reward_model_cfg"] = {
            "enabled": True,
            "reward_model_type": "bradley_terry",
        }

        with pytest.raises(
            NotImplementedError,
            match="Sequence packing is not supported for reward models",
        ):
            validate_and_set_config(mock_config, None, 0)

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch("nemo_rl.models.automodel.setup.sliding_window_overwrite")
    def test_reward_model_with_unknown_type(
        self,
        mock_sliding_window,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
    ):
        mock_sliding_window.return_value = {}
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig

        mock_config["reward_model_cfg"] = {
            "enabled": True,
            "reward_model_type": "unknown_type",
        }

        with pytest.raises(ValueError, match="Unknown reward model type"):
            validate_and_set_config(mock_config, None, 0)

    @patch("nemo_rl.models.automodel.setup.AutoConfig")
    @patch("nemo_rl.models.automodel.setup.resolve_model_class")
    @patch("nemo_rl.models.automodel.setup.configure_dynamo_cache")
    @patch("nemo_rl.models.automodel.setup.sliding_window_overwrite")
    def test_sequence_parallel_with_tp_size_one_warning(
        self,
        mock_sliding_window,
        mock_dynamo,
        mock_resolve_class,
        mock_autoconfig_class,
        mock_config,
        mock_autoconfig,
        capsys,
    ):
        mock_sliding_window.return_value = {}
        mock_autoconfig_class.from_pretrained.return_value = mock_autoconfig
        mock_resolve_class.return_value = Mock

        mock_config["dtensor_cfg"]["sequence_parallel"] = True
        mock_config["dtensor_cfg"]["tensor_parallel_size"] = 1

        result = validate_and_set_config(mock_config, None, 0)

        captured = capsys.readouterr()
        assert "[WARNING]" in captured.out
        assert "sequence_parallel=True, but tp_size=1" in captured.out


@pytest.mark.automodel
class TestSetupDistributed:
    @patch("nemo_rl.models.automodel.setup.torch.distributed.init_process_group")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_world_size")
    @patch("nemo_rl.models.automodel.setup.FSDP2Manager")
    def test_basic_distributed_setup(
        self,
        mock_manager_class,
        mock_world_size,
        mock_get_rank,
        mock_init_pg,
        mock_config,
    ):
        # Setup mocks
        mock_get_rank.return_value = 0
        mock_world_size.return_value = 4

        # Create mock manager
        mock_manager = MagicMock()
        mock_device_mesh = {
            "dp_cp": MagicMock(),
            "dp": MagicMock(),
            "tp": MagicMock(),
            "cp": MagicMock(),
        }
        mock_manager.device_mesh = mock_device_mesh
        mock_manager.dp_size = 2
        mock_manager.tp_size = 1
        mock_manager.cp_size = 1
        mock_manager.moe_mesh = None
        mock_manager_class.return_value = mock_manager

        # Create runtime config
        runtime_config = RuntimeConfig(
            is_vlm=False,
            is_generation_colocated=None,
            dtype=torch.bfloat16,
            cpu_offload=False,
            offload_optimizer_for_logprob=False,
            max_grad_norm=1.0,
            enable_seq_packing=False,
            attn_impl=None,
            model_config=MagicMock(),
            allow_flash_attn_args=True,
            is_reward_model=False,
            model_class=Mock,
            hf_config_overrides={},
            is_hf_model=False,
            is_moe_model=False,
        )

        result = setup_distributed(
            config=mock_config,
            runtime_config=runtime_config,
        )

        # Verify result
        assert isinstance(result, DistributedState)
        assert result.rank == 0
        assert result.world_size == 4
        assert result.dp_size == 2
        assert result.tp_size == 1
        assert result.cp_size == 1
        assert result.ep_size == 1
        assert result.sequence_parallel_enabled is False
        assert result.device_mesh == mock_device_mesh
        assert result.manager == mock_manager

        # Verify init_process_group was called
        mock_init_pg.assert_called_once_with(backend="nccl")

    @patch("nemo_rl.models.automodel.setup.torch.distributed.init_process_group")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.setup.torch.distributed.get_world_size")
    @patch("nemo_rl.models.automodel.setup.FSDP2Manager")
    def test_cpu_offload_enabled(
        self,
        mock_manager_class,
        mock_world_size,
        mock_get_rank,
        mock_init_pg,
        mock_config,
    ):
        mock_get_rank.return_value = 0
        mock_world_size.return_value = 2

        mock_manager = MagicMock()
        mock_manager.device_mesh = {
            "dp_cp": MagicMock(),
            "dp": MagicMock(),
            "tp": MagicMock(),
            "cp": MagicMock(),
        }
        mock_manager.dp_size = 2
        mock_manager.tp_size = 1
        mock_manager.cp_size = 1
        mock_manager_class.return_value = mock_manager

        runtime_config = RuntimeConfig(
            is_vlm=False,
            is_generation_colocated=None,
            dtype=torch.float32,
            cpu_offload=True,  # Enable CPU offload
            offload_optimizer_for_logprob=False,
            max_grad_norm=1.0,
            enable_seq_packing=False,
            attn_impl=None,
            model_config=MagicMock(),
            allow_flash_attn_args=True,
            is_reward_model=False,
            model_class=Mock,
            hf_config_overrides={},
            is_hf_model=False,
            is_moe_model=False,
        )

        result = setup_distributed(mock_config, runtime_config)

        # Verify FSDP2Manager was called with CPU offload policy
        call_kwargs = mock_manager_class.call_args[1]
        assert call_kwargs["offload_policy"] is not None


@pytest.mark.automodel
class TestSetupModelAndOptimizer:
    @patch("torch.optim.lr_scheduler.LambdaLR")
    @patch("nemo_rl.models.automodel.setup.init_empty_weights")
    @patch("nemo_rl.models.policy.utils.import_class_from_path")
    @patch("nemo_automodel.components.distributed.tensor_utils.get_cpu_state_dict")
    def test_basic_model_setup(
        self,
        mock_get_cpu_state,
        mock_import_class,
        mock_init_empty,
        mock_lambda_lr,
        mock_config,
    ):
        # Create mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 50256

        mock_model_class = MagicMock()
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"layer.weight": torch.zeros(10, 10)}
        mock_model.config.pad_token_id = None
        mock_model.named_parameters.return_value = []
        mock_model_class.from_config.return_value = mock_model

        mock_optimizer_class = MagicMock()
        mock_optimizer = MagicMock()
        mock_optimizer_class.return_value = mock_optimizer
        mock_import_class.return_value = mock_optimizer_class

        mock_scheduler = MagicMock()
        mock_lambda_lr.return_value = mock_scheduler

        mock_get_cpu_state.return_value = {"layer.weight": torch.zeros(10, 10)}

        # Create worker instance mock
        mock_worker = MagicMock()
        mock_worker._ensure_checkpointer = MagicMock()
        mock_worker.checkpointer = MagicMock()
        mock_worker.checkpointer.config = MagicMock()
        mock_worker.checkpointer.load_base_model = MagicMock()
        mock_worker.move_to_device = MagicMock(side_effect=lambda m, d: m)

        # Create runtime config
        runtime_config = RuntimeConfig(
            is_vlm=False,
            is_generation_colocated=None,
            dtype=torch.bfloat16,
            cpu_offload=False,
            offload_optimizer_for_logprob=False,
            max_grad_norm=1.0,
            enable_seq_packing=False,
            attn_impl=None,
            model_config=MagicMock(),
            allow_flash_attn_args=True,
            is_reward_model=False,
            model_class=mock_model_class,
            hf_config_overrides={},
            is_hf_model=False,
            is_moe_model=False,
        )

        # Create distributed state
        mock_manager = MagicMock()
        mock_device_mesh = MagicMock()
        mock_device_mesh.mesh_dim_names = ["dp"]
        mock_manager.parallelize = MagicMock(return_value=mock_model)

        distributed_state = DistributedState(
            rank=0,
            world_size=1,
            device_mesh=mock_device_mesh,
            dp_cp_mesh=MagicMock(),
            dp_mesh=MagicMock(),
            tp_mesh=MagicMock(),
            cp_mesh=MagicMock(),
            moe_mesh=None,
            dp_size=4,
            tp_size=1,
            cp_size=1,
            ep_size=1,
            sequence_parallel_enabled=False,
            manager=MagicMock(),
        )

        result = setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            worker_instance=mock_worker,
            init_optimizer=True,
            init_reference_model=True,
        )

        # Verify result
        assert isinstance(result, ModelAndOptimizerState)
        assert result.model is not None
        assert result.optimizer == mock_optimizer
        assert result.scheduler == mock_scheduler
        assert result.reference_model_state_dict is not None
        assert len(result.model_state_dict_keys) > 0
        assert isinstance(result.is_hf_model, bool)
        assert isinstance(result.is_moe_model, bool)

    @patch("nemo_rl.models.automodel.setup.init_empty_weights")
    @patch("nemo_rl.models.policy.utils.import_class_from_path")
    def test_model_setup_without_optimizer(
        self,
        mock_import_class,
        mock_init_empty,
        mock_config,
    ):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 50256

        mock_model_class = MagicMock()
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"layer.weight": torch.zeros(10, 10)}
        mock_model.config.pad_token_id = None
        mock_model.named_parameters.return_value = []
        mock_model_class.from_config.return_value = mock_model

        mock_worker = MagicMock()
        mock_worker._ensure_checkpointer = MagicMock()
        mock_worker.checkpointer = MagicMock()
        mock_worker.checkpointer.config = MagicMock()
        mock_worker.checkpointer.load_base_model = MagicMock()
        mock_worker.move_to_device = MagicMock(side_effect=lambda m, d: m)

        runtime_config = RuntimeConfig(
            is_vlm=False,
            is_generation_colocated=None,
            dtype=torch.bfloat16,
            cpu_offload=False,
            offload_optimizer_for_logprob=False,
            max_grad_norm=1.0,
            enable_seq_packing=False,
            attn_impl=None,
            model_config=MagicMock(),
            allow_flash_attn_args=True,
            is_reward_model=False,
            model_class=mock_model_class,
            hf_config_overrides={},
            is_hf_model=False,
            is_moe_model=False,
        )

        mock_manager = MagicMock()
        mock_device_mesh = MagicMock()
        mock_device_mesh.mesh_dim_names = ["dp"]
        mock_manager.parallelize = MagicMock(return_value=mock_model)

        distributed_state = DistributedState(
            rank=0,
            world_size=1,
            device_mesh=mock_device_mesh,
            dp_cp_mesh=MagicMock(),
            dp_mesh=MagicMock(),
            tp_mesh=MagicMock(),
            cp_mesh=MagicMock(),
            moe_mesh=None,
            dp_size=4,
            tp_size=1,
            cp_size=1,
            ep_size=1,
            sequence_parallel_enabled=False,
            manager=MagicMock(),
        )

        result = setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            worker_instance=mock_worker,
            init_optimizer=False,  # Don't initialize optimizer
            init_reference_model=False,
        )

        # Verify optimizer and reference model are None
        assert result.optimizer is None
        assert result.scheduler is None
        assert result.reference_model_state_dict is None
        assert isinstance(result.is_hf_model, bool)
        assert isinstance(result.is_moe_model, bool)

    @patch("nemo_rl.models.automodel.setup.init_empty_weights")
    def test_context_parallel_with_gemma3_raises_error(
        self,
        mock_init_empty,
        mock_config,
    ):
        # Import the real Gemma3ForCausalLM to use as the class type
        from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 50256

        # Create a Gemma3 model mock with the correct class
        mock_model = MagicMock()
        mock_model.__class__ = Gemma3ForCausalLM
        mock_model.state_dict.return_value = {"layer.weight": torch.zeros(10, 10)}
        mock_model.config.pad_token_id = None

        mock_model_class = MagicMock()
        mock_model_class.from_config.return_value = mock_model

        mock_worker = MagicMock()

        runtime_config = RuntimeConfig(
            is_vlm=False,
            is_generation_colocated=None,
            dtype=torch.bfloat16,
            cpu_offload=False,
            offload_optimizer_for_logprob=False,
            max_grad_norm=1.0,
            enable_seq_packing=False,
            attn_impl=None,
            model_config=MagicMock(),
            allow_flash_attn_args=True,
            is_reward_model=False,
            model_class=mock_model_class,
            hf_config_overrides={},
            is_hf_model=False,
            is_moe_model=False,
        )

        distributed_state = DistributedState(
            rank=0,
            world_size=1,
            device_mesh=MagicMock(),
            dp_cp_mesh=MagicMock(),
            dp_mesh=MagicMock(),
            tp_mesh=MagicMock(),
            cp_mesh=MagicMock(),
            moe_mesh=None,
            dp_size=4,
            tp_size=1,
            cp_size=2,  # cp_size > 1 to trigger the error
            ep_size=1,
            sequence_parallel_enabled=False,
            manager=MagicMock(),
        )

        with pytest.raises(
            ValueError, match="Context parallel is not supported for Gemma3ForCausalLM"
        ):
            result = setup_model_and_optimizer(
                config=mock_config,
                tokenizer=mock_tokenizer,
                runtime_config=runtime_config,
                distributed_state=distributed_state,
                worker_instance=mock_worker,
            )

    @patch("torch.optim.lr_scheduler.SequentialLR")
    @patch("nemo_rl.models.automodel.setup.init_empty_weights")
    @patch("nemo_rl.models.policy.utils.import_class_from_path")
    def test_scheduler_as_list(
        self,
        mock_import_class,
        mock_init_empty,
        mock_sequential_lr,
        mock_config,
    ):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 50256

        mock_model_class = MagicMock()
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"layer.weight": torch.zeros(10, 10)}
        mock_model.config.pad_token_id = None
        mock_model.named_parameters.return_value = []
        mock_model_class.from_config.return_value = mock_model

        mock_optimizer = MagicMock()
        mock_scheduler_class = MagicMock()
        mock_scheduler1 = MagicMock()
        mock_scheduler2 = MagicMock()
        mock_scheduler_class.side_effect = [mock_scheduler1, mock_scheduler2]

        mock_final_scheduler = MagicMock()
        mock_sequential_lr.return_value = mock_final_scheduler

        def import_side_effect(path):
            if "optimizer" in path.lower() or "AdamW" in path:
                return lambda *args, **kwargs: mock_optimizer
            else:
                return mock_scheduler_class

        mock_import_class.side_effect = import_side_effect

        mock_worker = MagicMock()
        mock_worker._ensure_checkpointer = MagicMock()
        mock_worker.checkpointer = MagicMock()
        mock_worker.checkpointer.config = MagicMock()
        mock_worker.checkpointer.load_base_model = MagicMock()
        mock_worker.move_to_device = MagicMock(side_effect=lambda m, d: m)

        runtime_config = RuntimeConfig(
            is_vlm=False,
            is_generation_colocated=None,
            dtype=torch.bfloat16,
            cpu_offload=False,
            offload_optimizer_for_logprob=False,
            max_grad_norm=1.0,
            enable_seq_packing=False,
            attn_impl=None,
            model_config=MagicMock(),
            allow_flash_attn_args=True,
            is_reward_model=False,
            model_class=mock_model_class,
            hf_config_overrides={},
            is_hf_model=False,
            is_moe_model=False,
        )

        mock_manager = MagicMock()
        mock_device_mesh = MagicMock()
        mock_device_mesh.mesh_dim_names = ["dp"]
        mock_manager.parallelize = MagicMock(return_value=mock_model)

        distributed_state = DistributedState(
            rank=0,
            world_size=1,
            device_mesh=mock_device_mesh,
            dp_cp_mesh=MagicMock(),
            dp_mesh=MagicMock(),
            tp_mesh=MagicMock(),
            cp_mesh=MagicMock(),
            moe_mesh=None,
            dp_size=4,
            tp_size=1,
            cp_size=1,
            ep_size=1,
            sequence_parallel_enabled=False,
            manager=MagicMock(),
        )

        # Configure with list scheduler
        mock_config["scheduler"] = [
            {"name": "torch.optim.lr_scheduler.LinearLR", "kwargs": {}},
            {
                "name": "torch.optim.lr_scheduler.CosineAnnealingLR",
                "kwargs": {"T_max": 100},
            },
            {"milestones": [1000]},
        ]

        result = setup_model_and_optimizer(
            config=mock_config,
            tokenizer=mock_tokenizer,
            runtime_config=runtime_config,
            distributed_state=distributed_state,
            worker_instance=mock_worker,
            init_optimizer=True,
            init_reference_model=False,
        )

        assert result.scheduler == mock_final_scheduler
        assert isinstance(result.is_hf_model, bool)
        assert isinstance(result.is_moe_model, bool)

    @patch("nemo_rl.models.automodel.setup.init_empty_weights")
    @patch("nemo_rl.models.policy.utils.import_class_from_path")
    def test_context_parallel_with_tp_and_sp_error(
        self,
        mock_import_class,
        mock_init_empty,
        mock_config,
    ):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 50256

        mock_model_class = MagicMock()
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"layer.weight": torch.zeros(10, 10)}
        mock_model.config.pad_token_id = None
        mock_model_class.from_config.return_value = mock_model

        mock_worker = MagicMock()

        runtime_config = RuntimeConfig(
            is_vlm=False,
            is_generation_colocated=None,
            dtype=torch.bfloat16,
            cpu_offload=False,
            offload_optimizer_for_logprob=False,
            max_grad_norm=1.0,
            enable_seq_packing=False,
            attn_impl=None,
            model_config=MagicMock(),
            allow_flash_attn_args=True,
            is_reward_model=False,
            model_class=mock_model_class,
            hf_config_overrides={},
            is_hf_model=False,
            is_moe_model=False,
        )

        distributed_state = DistributedState(
            rank=0,
            world_size=1,
            device_mesh=MagicMock(),
            dp_cp_mesh=MagicMock(),
            dp_mesh=MagicMock(),
            tp_mesh=MagicMock(),
            cp_mesh=MagicMock(),
            moe_mesh=None,
            dp_size=4,
            tp_size=2,  # tp_size > 1 to trigger the error
            cp_size=2,  # cp_size > 1 to enter the validation block
            ep_size=1,
            sequence_parallel_enabled=True,  # Enable sequence parallel to trigger the error
            manager=MagicMock(),
        )

        with pytest.raises(
            ValueError,
            match="context parallel can't be used together with sequence parallel",
        ):
            result = setup_model_and_optimizer(
                config=mock_config,
                tokenizer=mock_tokenizer,
                runtime_config=runtime_config,
                distributed_state=distributed_state,
                worker_instance=mock_worker,
            )

    @patch("nemo_rl.models.automodel.setup.init_empty_weights")
    def test_context_parallel_with_vlm_error(
        self,
        mock_init_empty,
        mock_config,
    ):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 50256

        mock_model_class = MagicMock()
        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"layer.weight": torch.zeros(10, 10)}
        mock_model.config.pad_token_id = None
        mock_model_class.from_config.return_value = mock_model

        mock_worker = MagicMock()

        runtime_config = RuntimeConfig(
            is_vlm=True,  # VLM model
            is_generation_colocated=None,
            dtype=torch.bfloat16,
            cpu_offload=False,
            offload_optimizer_for_logprob=False,
            max_grad_norm=1.0,
            enable_seq_packing=False,
            attn_impl=None,
            model_config=MagicMock(),
            allow_flash_attn_args=True,
            is_reward_model=False,
            model_class=mock_model_class,
            hf_config_overrides={},
            is_hf_model=False,
            is_moe_model=False,
        )

        distributed_state = DistributedState(
            rank=0,
            world_size=1,
            device_mesh=MagicMock(),
            dp_cp_mesh=MagicMock(),
            dp_mesh=MagicMock(),
            tp_mesh=MagicMock(),
            cp_mesh=MagicMock(),
            moe_mesh=None,
            dp_size=4,
            tp_size=1,
            cp_size=2,  # cp_size > 1 to trigger the VLM validation
            ep_size=1,
            sequence_parallel_enabled=False,
            manager=MagicMock(),
        )

        with pytest.raises(
            ValueError, match="Context parallel is yet not supported for VLM models"
        ):
            result = setup_model_and_optimizer(
                config=mock_config,
                tokenizer=mock_tokenizer,
                runtime_config=runtime_config,
                distributed_state=distributed_state,
                worker_instance=mock_worker,
            )
