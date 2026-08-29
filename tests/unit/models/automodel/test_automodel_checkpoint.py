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
"""Unit tests for automodel checkpoint utilities."""

import os
import socket
from datetime import timedelta
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.multiprocessing as mp

# Skip entire module if nemo_automodel is not available
try:
    import nemo_automodel  # noqa: F401
except ImportError:
    pytest.skip("nemo_automodel not available", allow_module_level=True)

from nemo_automodel.components._peft.lora import (
    PeftConfig,
    apply_lora_to_linear_modules,
)
from nemo_automodel.components.checkpoint.config import SaveConsolidatedMode

from nemo_rl.models.automodel.checkpoint import (
    AutomodelCheckpointManager,
    _infer_checkpoint_root,
    detect_checkpoint_format,
)


class TestModel(torch.nn.Module):
    """Simple test model with a forward method."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(4, 4),
                torch.nn.LayerNorm(4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 1),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def apply_lora(self, lora_config: PeftConfig):
        apply_lora_to_linear_modules(self, lora_config)


@pytest.fixture
def mock_model():
    """Create a simple mock model for testing."""
    return TestModel()


@pytest.fixture
def mock_optimizer():
    """Create a simple mock optimizer for testing."""
    model = torch.nn.Linear(4, 1)
    return torch.optim.Adam(model.parameters())


@pytest.fixture
def mock_lora_config():
    """Create a simple mock LORA configuration for testing."""
    return PeftConfig(
        target_modules=[],
        match_all_linear=True,
        dim=2,
        alpha=2,
        dropout=0.1,
        dropout_position="post",
        lora_A_init="xavier",
        use_triton=False,
    )


def _cleanup_dcp_planner_cache():
    """Clean up DCP SavePlanner class-level caches.

    The SavePlanner class maintains class-level caches for plan caching optimization.
    When tests run with different checkpoint formats (safetensors vs torch_save),
    these caches contain stale data that causes errors. This function clears all
    planner caches to ensure test isolation.
    """
    try:
        from torch.distributed.checkpoint.planner import SavePlanner

        # Clear all class-level planner caches
        if hasattr(SavePlanner, "_cached_save_plan"):
            SavePlanner._cached_save_plan.clear()
        if hasattr(SavePlanner, "_cached_all_plans"):
            SavePlanner._cached_all_plans.clear()
        if hasattr(SavePlanner, "_cached_global_plan"):
            SavePlanner._cached_global_plan.clear()
        if hasattr(SavePlanner, "_cached_metadata"):
            SavePlanner._cached_metadata.clear()
        if hasattr(SavePlanner, "_cached_final_save_plan"):
            SavePlanner._cached_final_save_plan.clear()
    except Exception:
        pass


def _cleanup_device_mesh_cache():
    """Clean up device mesh cache."""
    import gc

    # Clear device mesh cache
    if hasattr(torch.distributed, "device_mesh") and hasattr(
        torch.distributed.device_mesh, "_mesh_resources"
    ):
        try:
            torch.distributed.device_mesh._mesh_resources.mesh_stack.clear()
            torch.distributed.device_mesh._mesh_resources.child_to_root_mapping.clear()
        except Exception:
            pass

    gc.collect()


@pytest.fixture
def init_distributed():
    """Initialize a single-process distributed environment for testing.

    Each test gets proper cleanup of DCP planner caches to ensure test isolation.
    The planner caches are class-level and shared across all DefaultSavePlanner
    instances, so they must be cleared between tests that use different
    checkpoint formats.
    """
    # Clean up any stale planner caches from previous tests
    _cleanup_dcp_planner_cache()
    _cleanup_device_mesh_cache()

    # Only initialize if not already initialized
    if not torch.distributed.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        # Use gloo backend for CPU-only tests
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)

    yield

    # Cleanup planner caches after test (critical for test isolation)
    _cleanup_dcp_planner_cache()
    _cleanup_device_mesh_cache()


@pytest.fixture
def mock_experiment():
    """Create a real model, optimizer, and scheduler for integration testing."""
    model = TestModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    return model, optimizer, scheduler


def check_dict_equality(dict1, dict2):
    """Recursively check equality of two dictionaries"""
    for k in dict1.keys():
        if isinstance(dict1[k], dict):
            check_dict_equality(dict1[k], dict2[k])
        elif isinstance(dict1[k], torch.Tensor):
            assert torch.allclose(dict1[k], dict2[k])
        else:
            assert dict1[k] == dict2[k]


class _WorldProcessMesh:
    """Minimal mesh interface backed by the default process group."""

    @staticmethod
    def get_group():
        return torch.distributed.group.WORLD


def _run_consolidated_group_rebuild(rank: int, world_size: int, port: int) -> None:
    """Verify every rank rebuilds with a dedicated consolidation group."""
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
    )
    manager = AutomodelCheckpointManager(
        dp_mesh=_WorldProcessMesh(),
        tp_mesh=_WorldProcessMesh(),
    )
    try:
        manager.init_checkpointer(config_updates={"save_consolidated": False})
        original_checkpointer = manager.checkpointer

        manager.update_checkpointer_config(
            config_updates={
                "save_consolidated": True,
                "consolidation_timeout_minutes": 1,
            }
        )

        assert manager.checkpointer is not original_checkpointer
        assert manager.checkpointer is not None
        consolidation_group = manager.checkpointer._consolidation_process_group
        assert consolidation_group is not None
        gathered_ranks = [None] * world_size
        torch.distributed.all_gather_object(
            gathered_ranks,
            rank,
            group=consolidation_group,
        )
        assert gathered_ranks == list(range(world_size))
    finally:
        if manager.checkpointer is not None:
            manager.checkpointer.close()
        torch.distributed.destroy_process_group()


@pytest.mark.automodel
class TestDetectCheckpointFormat:
    """Tests for detect_checkpoint_format function."""

    def test_detect_safetensors_format(self, tmp_path):
        """Test detection of safetensors format."""
        # Create a checkpoint directory with safetensors files
        model_dir = tmp_path / "weights" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.safetensors").touch()

        model_save_format, is_peft = detect_checkpoint_format(str(tmp_path / "weights"))

        assert model_save_format == "safetensors"
        assert is_peft is False

    def test_detect_torch_save_format_distcp(self, tmp_path):
        """Test detection of torch_save format with .distcp files."""
        # Create a checkpoint directory with .distcp files
        model_dir = tmp_path / "weights" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "checkpoint.distcp").touch()

        model_save_format, is_peft = detect_checkpoint_format(str(tmp_path / "weights"))

        assert model_save_format == "torch_save"
        assert is_peft is False

    def test_detect_torch_save_format_bin(self, tmp_path):
        """Test detection of torch_save format with .bin files."""
        # Create a checkpoint directory with .bin files
        model_dir = tmp_path / "weights" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "pytorch_model.bin").touch()

        model_save_format, is_peft = detect_checkpoint_format(str(tmp_path / "weights"))

        assert model_save_format == "torch_save"
        assert is_peft is False

    def test_detect_torch_save_format_pt(self, tmp_path):
        """Test detection of torch_save format with .pt files."""
        # Create a checkpoint directory with .pt files
        model_dir = tmp_path / "weights" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.pt").touch()

        model_save_format, is_peft = detect_checkpoint_format(str(tmp_path / "weights"))

        assert model_save_format == "torch_save"
        assert is_peft is False

    def test_detect_peft_adapter(self, tmp_path):
        """Test detection of PEFT adapter files."""
        # Create a checkpoint directory with adapter files
        model_dir = tmp_path / "weights" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "adapter_model.safetensors").touch()

        model_save_format, is_peft = detect_checkpoint_format(str(tmp_path / "weights"))

        assert model_save_format == "safetensors"
        assert is_peft is True

    def test_detect_peft_adapter_config(self, tmp_path):
        """Test detection of PEFT adapter config files."""
        # Create a checkpoint directory with adapter config
        model_dir = tmp_path / "weights" / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "adapter_config.json").touch()
        (model_dir / "model.safetensors").touch()

        model_save_format, is_peft = detect_checkpoint_format(str(tmp_path / "weights"))

        assert model_save_format == "safetensors"
        assert is_peft is True

    def test_detect_empty_directory(self, tmp_path):
        """Test detection with empty directory."""
        # Create an empty checkpoint directory
        model_dir = tmp_path / "weights"
        model_dir.mkdir(parents=True)

        model_save_format, is_peft = detect_checkpoint_format(str(model_dir))

        # Default to safetensors when no files found
        assert model_save_format == "safetensors"
        assert is_peft is False

    def test_detect_nonexistent_directory(self, tmp_path):
        """Test detection with non-existent directory."""
        nonexistent_path = str(tmp_path / "nonexistent")

        model_save_format, is_peft = detect_checkpoint_format(nonexistent_path)

        # Default to safetensors when directory doesn't exist
        assert model_save_format == "safetensors"
        assert is_peft is False

    def test_detect_nested_safetensors(self, tmp_path):
        """Test detection of safetensors in nested directories."""
        # Create nested structure
        nested_dir = tmp_path / "weights" / "model" / "rank_0"
        nested_dir.mkdir(parents=True)
        (nested_dir / "model-00001-of-00002.safetensors").touch()

        model_save_format, is_peft = detect_checkpoint_format(str(tmp_path / "weights"))

        assert model_save_format == "safetensors"
        assert is_peft is False


@pytest.mark.automodel
class TestInferCheckpointRoot:
    """Tests for _infer_checkpoint_root function."""

    def test_infer_root_from_weights_model_path(self):
        """Test inferring root from path ending with weights/model."""
        weights_path = "/path/to/checkpoint/weights/model"

        result = _infer_checkpoint_root(weights_path)

        assert result == "/path/to/checkpoint"

    def test_infer_root_from_weights_path(self):
        """Test inferring root from path ending with weights."""
        weights_path = "/path/to/checkpoint/weights"

        result = _infer_checkpoint_root(weights_path)

        assert result == "/path/to/checkpoint"

    def test_infer_root_from_other_path(self):
        """Test inferring root from path not ending with weights."""
        weights_path = "/path/to/checkpoint/custom_dir"

        result = _infer_checkpoint_root(weights_path)

        # Should return parent directory
        assert result == "/path/to/checkpoint"

    def test_infer_root_with_trailing_slash(self):
        """Test inferring root with trailing slash in path."""
        weights_path = "/path/to/checkpoint/weights/"

        result = _infer_checkpoint_root(weights_path)

        # dirname of "/path/to/checkpoint/weights/" is "/path/to/checkpoint/weights"
        # which ends with "weights", so parent is returned
        assert result == "/path/to/checkpoint"

    def test_infer_root_relative_path(self):
        """Test inferring root from relative path."""
        weights_path = "checkpoint/weights/model"

        result = _infer_checkpoint_root(weights_path)

        assert result == "checkpoint"

    def test_infer_root_single_level(self):
        """Test inferring root from single level path."""
        weights_path = "weights/model"

        result = _infer_checkpoint_root(weights_path)

        assert result == ""


@pytest.mark.automodel
class TestAutomodelCheckpointManager:
    """Tests for AutomodelCheckpointManager class.

    Note: Full integration tests require distributed environment setup.
    These tests focus on the helper methods and configuration.
    """

    @pytest.fixture
    def mock_meshes(self):
        """Create mock device meshes for testing."""
        mock_dp_mesh = MagicMock()
        mock_dp_mesh.get_group.return_value = MagicMock()

        mock_tp_mesh = MagicMock()
        mock_tp_mesh.get_group.return_value = MagicMock()

        return mock_dp_mesh, mock_tp_mesh

    @patch("torch.distributed.get_rank")
    def test_manager_initialization(self, mock_get_rank, mock_meshes):
        """Test AutomodelCheckpointManager initialization."""
        from nemo_rl.models.automodel.checkpoint import AutomodelCheckpointManager

        mock_get_rank.return_value = 0
        mock_dp_mesh, mock_tp_mesh = mock_meshes

        manager = AutomodelCheckpointManager(
            dp_mesh=mock_dp_mesh,
            tp_mesh=mock_tp_mesh,
        )

        assert manager.checkpointer is None
        assert manager.checkpoint_config is None
        assert manager.dp_mesh is mock_dp_mesh
        assert manager.tp_mesh is mock_tp_mesh

    @patch("torch.distributed.get_rank")
    def test_finalize_async_save_waits_on_checkpointer(
        self, mock_get_rank, mock_meshes
    ):
        """finalize_async_save must block on staging *and* upload completion.

        The manager is initialized with is_async=True by DTensorPolicyWorkerV2,
        so dcp.async_save writes from a separate process. Skipping either wait
        lets the caller rename tmp_step_N to step_N mid-write, producing a
        checkpoint with no optimizer shards and no .metadata.
        """
        from nemo_rl.models.automodel.checkpoint import AutomodelCheckpointManager

        mock_get_rank.return_value = 0
        mock_dp_mesh, mock_tp_mesh = mock_meshes

        manager = AutomodelCheckpointManager(
            dp_mesh=mock_dp_mesh,
            tp_mesh=mock_tp_mesh,
        )
        manager.checkpointer = MagicMock()

        manager.finalize_async_save()

        manager.checkpointer.maybe_wait_for_staging.assert_called_once_with()
        manager.checkpointer.async_wait.assert_called_once_with()

    @patch("torch.distributed.get_rank")
    def test_finalize_async_save_without_checkpointer_is_noop(
        self, mock_get_rank, mock_meshes
    ):
        """Finalizing before any save was issued must not raise."""
        from nemo_rl.models.automodel.checkpoint import AutomodelCheckpointManager

        mock_get_rank.return_value = 0
        mock_dp_mesh, mock_tp_mesh = mock_meshes

        manager = AutomodelCheckpointManager(
            dp_mesh=mock_dp_mesh,
            tp_mesh=mock_tp_mesh,
        )
        assert manager.checkpointer is None

        manager.finalize_async_save()

    def test_dtensor_worker_overrides_finalize_async_save(self):
        """The worker must not inherit the base class no-op.

        DTensorPolicyWorkerV2 passes is_async=True, so grpo.py's
        wait_fn=policy.finalize_async_save has to resolve to a real wait.
        """
        from nemo_rl.models.policy.workers.base_policy_worker import (
            AbstractPolicyWorker,
        )
        from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
            DTensorPolicyWorkerV2Impl,
        )

        assert (
            DTensorPolicyWorkerV2Impl.finalize_async_save
            is not AbstractPolicyWorker.finalize_async_save
        )

        worker = object.__new__(DTensorPolicyWorkerV2Impl)
        worker.checkpoint_manager = MagicMock()
        worker.finalize_async_save()
        worker.checkpoint_manager.finalize_async_save.assert_called_once_with()

    @patch("torch.distributed.get_rank")
    def test_save_checkpoint_without_checkpointer_raises(
        self, mock_get_rank, mock_meshes
    ):
        """Test that save_checkpoint raises error without initialized checkpointer."""
        from nemo_rl.models.automodel.checkpoint import AutomodelCheckpointManager

        mock_get_rank.return_value = 0
        mock_dp_mesh, mock_tp_mesh = mock_meshes

        manager = AutomodelCheckpointManager(
            dp_mesh=mock_dp_mesh,
            tp_mesh=mock_tp_mesh,
        )

        mock_model = MagicMock()

        with pytest.raises(AssertionError, match="Checkpointer must be initialized"):
            manager.save_checkpoint(
                model=mock_model,
                weights_path="/path/to/weights",
                checkpointing_cfg={"enabled": True},
            )

    @patch("torch.distributed.get_rank")
    def test_save_checkpoint_without_config_raises(self, mock_get_rank, mock_meshes):
        """Test that save_checkpoint raises error without checkpointing config."""
        from nemo_rl.models.automodel.checkpoint import AutomodelCheckpointManager

        mock_get_rank.return_value = 0
        mock_dp_mesh, mock_tp_mesh = mock_meshes

        manager = AutomodelCheckpointManager(
            dp_mesh=mock_dp_mesh,
            tp_mesh=mock_tp_mesh,
        )

        # Set up a mock checkpointer so we get past the first assertion
        manager.checkpointer = MagicMock()

        mock_model = MagicMock()

        with pytest.raises(ValueError, match="checkpointing_cfg must be provided"):
            manager.save_checkpoint(
                model=mock_model,
                weights_path="/path/to/weights",
                checkpointing_cfg=None,
            )

    @patch("torch.distributed.get_rank")
    def test_load_checkpoint_without_checkpointer_raises(
        self, mock_get_rank, mock_meshes
    ):
        """Test that load_checkpoint raises error without initialized checkpointer."""
        from nemo_rl.models.automodel.checkpoint import AutomodelCheckpointManager

        mock_get_rank.return_value = 0
        mock_dp_mesh, mock_tp_mesh = mock_meshes

        manager = AutomodelCheckpointManager(
            dp_mesh=mock_dp_mesh,
            tp_mesh=mock_tp_mesh,
        )

        mock_model = MagicMock()

        with pytest.raises(AssertionError, match="Checkpointer must be initialized"):
            manager.load_checkpoint(
                model=mock_model,
                weights_path="/path/to/weights",
            )

    @patch("torch.distributed.get_rank")
    def test_init_checkpointer_creates_checkpointer(self, mock_get_rank, mock_meshes):
        """Test that init_checkpointer creates a new checkpointer."""
        from nemo_rl.models.automodel.checkpoint import AutomodelCheckpointManager

        mock_get_rank.return_value = 0
        mock_dp_mesh, mock_tp_mesh = mock_meshes

        manager = AutomodelCheckpointManager(
            dp_mesh=mock_dp_mesh,
            tp_mesh=mock_tp_mesh,
        )

        assert manager.checkpointer is None

        # Mock the Checkpointer class
        with patch(
            "nemo_rl.models.automodel.checkpoint.Checkpointer"
        ) as mock_checkpointer_cls:
            mock_checkpointer = MagicMock()
            mock_checkpointer_cls.return_value = mock_checkpointer

            manager.init_checkpointer(
                config_updates={"model_repo_id": "test-model"},
                checkpoint_root="/path/to/checkpoints",
            )

            assert manager.checkpointer is mock_checkpointer
            mock_checkpointer_cls.assert_called_once()

    @patch("torch.distributed.get_rank")
    def test_init_checkpointer_does_nothing_if_exists(self, mock_get_rank, mock_meshes):
        """Test that init_checkpointer does nothing if checkpointer already exists."""
        from nemo_rl.models.automodel.checkpoint import AutomodelCheckpointManager

        mock_get_rank.return_value = 0
        mock_dp_mesh, mock_tp_mesh = mock_meshes

        manager = AutomodelCheckpointManager(
            dp_mesh=mock_dp_mesh,
            tp_mesh=mock_tp_mesh,
        )

        # Create a mock existing checkpointer
        existing_checkpointer = MagicMock()
        manager.checkpointer = existing_checkpointer

        # Try to init again
        with patch(
            "nemo_rl.models.automodel.checkpoint.Checkpointer"
        ) as mock_checkpointer_cls:
            manager.init_checkpointer(
                config_updates={"model_repo_id": "test-model"},
            )

            # Should not have created a new checkpointer
            mock_checkpointer_cls.assert_not_called()
            assert manager.checkpointer is existing_checkpointer

    @patch("torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.checkpoint.Checkpointer")
    def test_update_checkpointer_config_updates_mutable_config_in_place(
        self, mock_checkpointer_cls, mock_get_rank, mock_meshes
    ):
        """Mutable settings preserve the Checkpointer and its config references."""
        from nemo_rl.models.automodel.checkpoint import AutomodelCheckpointManager

        mock_get_rank.return_value = 0
        mock_dp_mesh, mock_tp_mesh = mock_meshes

        manager = AutomodelCheckpointManager(
            dp_mesh=mock_dp_mesh,
            tp_mesh=mock_tp_mesh,
        )

        mock_checkpointer = MagicMock()

        def build_checkpointer(*, config, **_kwargs):
            mock_checkpointer.config = config
            mock_checkpointer.lifecycle.config = config
            return mock_checkpointer

        mock_checkpointer_cls.side_effect = build_checkpointer
        manager.init_checkpointer()
        original_config = mock_checkpointer.config

        manager.update_checkpointer_config(
            config_updates={"model_repo_id": "updated-model"},
            checkpoint_root="/new/path",
        )

        assert manager.checkpointer is mock_checkpointer
        assert mock_checkpointer.config is original_config
        assert mock_checkpointer.lifecycle.config is original_config
        assert mock_checkpointer.config.checkpoint_dir == "/new/path"
        assert mock_checkpointer.lifecycle.config.checkpoint_dir == "/new/path"
        assert mock_checkpointer.config.model_repo_id == "updated-model"
        mock_checkpointer.close.assert_not_called()
        mock_checkpointer_cls.assert_called_once()

    @patch("torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.checkpoint.Checkpointer")
    def test_update_rebuilds_checkpointer_for_consolidation_resources(
        self, mock_checkpointer_cls, mock_get_rank, mock_meshes
    ):
        """Construction-time consolidation settings require a new Checkpointer."""
        mock_get_rank.return_value = 0
        mock_dp_mesh, mock_tp_mesh = mock_meshes
        built_checkpointers = []

        def build_checkpointer(*, config, **_kwargs):
            checkpointer = MagicMock()
            checkpointer.config = config
            built_checkpointers.append(checkpointer)
            return checkpointer

        mock_checkpointer_cls.side_effect = build_checkpointer
        manager = AutomodelCheckpointManager(
            dp_mesh=mock_dp_mesh,
            tp_mesh=mock_tp_mesh,
        )
        manager.init_checkpointer(
            config_updates={"is_async": True, "save_consolidated": False}
        )
        original_checkpointer = manager.checkpointer

        manager.update_checkpointer_config(
            config_updates={
                "save_consolidated": True,
                "consolidation_timeout_minutes": 5,
            }
        )

        assert len(built_checkpointers) == 2
        assert original_checkpointer is not None
        original_checkpointer.close.assert_called_once_with()
        assert manager.checkpointer is built_checkpointers[1]
        assert (
            manager.checkpointer.config.save_consolidated == SaveConsolidatedMode.EVERY
        )
        assert manager.checkpointer.config.is_async is True
        assert manager.checkpointer.config.consolidation_timeout_minutes == 5

    @patch("torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.checkpoint.Checkpointer")
    def test_invalid_resource_update_preserves_existing_checkpointer(
        self, mock_checkpointer_cls, mock_get_rank, mock_meshes
    ):
        """Validate replacement config before closing the working Checkpointer."""
        mock_get_rank.return_value = 0
        mock_dp_mesh, mock_tp_mesh = mock_meshes
        original_checkpointer = MagicMock()

        def build_checkpointer(*, config, **_kwargs):
            original_checkpointer.config = config
            return original_checkpointer

        mock_checkpointer_cls.side_effect = build_checkpointer
        manager = AutomodelCheckpointManager(
            dp_mesh=mock_dp_mesh,
            tp_mesh=mock_tp_mesh,
        )
        manager.init_checkpointer()

        with pytest.raises(
            ValueError,
            match="consolidation_timeout_minutes must be greater than 0",
        ):
            manager.update_checkpointer_config(
                config_updates={"consolidation_timeout_minutes": 0}
            )

        assert manager.checkpointer is original_checkpointer
        original_checkpointer.close.assert_not_called()

    @patch("torch.distributed.get_rank")
    def test_update_checkpointer_config_does_nothing_if_no_checkpointer(
        self, mock_get_rank, mock_meshes
    ):
        """Test that update_checkpointer_config does nothing without checkpointer."""
        from nemo_rl.models.automodel.checkpoint import AutomodelCheckpointManager

        mock_get_rank.return_value = 0
        mock_dp_mesh, mock_tp_mesh = mock_meshes

        manager = AutomodelCheckpointManager(
            dp_mesh=mock_dp_mesh,
            tp_mesh=mock_tp_mesh,
        )

        # Should not raise any error
        manager.update_checkpointer_config(
            config_updates={"is_peft": True},
            checkpoint_root="/new/path",
        )

        assert manager.checkpointer is None


@pytest.mark.automodel
def test_save_consolidated_rebuilds_group_on_every_rank():
    """Late consolidated-save enablement creates a usable Gloo group on all ranks."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    with patch.dict(os.environ, {"GLOO_SOCKET_IFNAME": "lo"}):
        mp.spawn(
            _run_consolidated_group_rebuild,
            args=(2, port),
            nprocs=2,
            join=True,
        )


@pytest.mark.automodel
class TestSaveCheckpointFunctional:
    """Functional tests for save_checkpoint method with mocked internals."""

    @pytest.fixture
    def mock_meshes(self):
        """Create mock device meshes for testing."""
        mock_dp_mesh = MagicMock()
        mock_dp_mesh.get_group.return_value = MagicMock()

        mock_tp_mesh = MagicMock()
        mock_tp_mesh.get_group.return_value = MagicMock()

        return mock_dp_mesh, mock_tp_mesh

    @patch("torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.checkpoint.Checkpointer")
    def test_save_passes_peft_config_only_to_save_model(
        self, mock_checkpointer_cls, mock_get_rank, mock_meshes, mock_model
    ):
        """PEFT config is a save argument, not an Automodel config field."""
        mock_get_rank.return_value = 0
        mock_dp_mesh, mock_tp_mesh = mock_meshes

        mock_checkpointer = MagicMock()
        mock_checkpointer_cls.return_value = mock_checkpointer
        peft_config = MagicMock()

        manager = AutomodelCheckpointManager(
            dp_mesh=mock_dp_mesh,
            tp_mesh=mock_tp_mesh,
        )
        manager.init_checkpointer()

        with TemporaryDirectory() as tmp_dir:
            weights_path = os.path.join(tmp_dir, "weights")
            manager.save_checkpoint(
                model=mock_model,
                weights_path=weights_path,
                checkpointing_cfg={
                    "enabled": True,
                    "peft_config": peft_config,
                },
            )

        mock_checkpointer.save_model.assert_called_once_with(
            model=mock_model,
            weights_path=weights_path,
            peft_config=peft_config,
            tokenizer=None,
        )

    @patch("torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.checkpoint.Checkpointer")
    def test_save_model_only(
        self, mock_checkpointer_cls, mock_get_rank, mock_meshes, mock_model
    ):
        """Test saving model weights only."""
        mock_get_rank.return_value = 0
        mock_dp_mesh, mock_tp_mesh = mock_meshes

        mock_checkpointer = MagicMock()
        mock_checkpointer_cls.return_value = mock_checkpointer

        manager = AutomodelCheckpointManager(
            dp_mesh=mock_dp_mesh,
            tp_mesh=mock_tp_mesh,
        )
        manager.init_checkpointer()

        with TemporaryDirectory() as tmp_dir:
            weights_path = os.path.join(tmp_dir, "weights")

            # Save checkpoint
            manager.save_checkpoint(
                model=mock_model,
                weights_path=weights_path,
                checkpointing_cfg={
                    "enabled": True,
                    "model_save_format": "safetensors",
                    "is_peft": False,
                },
            )

            # Verify save_model was called
            mock_checkpointer.save_model.assert_called_once()

            # Verify save_optimizer was not called
            mock_checkpointer.save_optimizer.assert_not_called()

    @patch("torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.checkpoint.Checkpointer")
    def test_save_forwards_consolidation_resource_config(
        self, mock_checkpointer_cls, mock_get_rank, mock_meshes, mock_model
    ):
        """Save-time Automodel resource settings reach the rebuilt Checkpointer."""
        mock_get_rank.return_value = 0
        mock_dp_mesh, mock_tp_mesh = mock_meshes
        built_configs = []

        def build_checkpointer(*, config, **_kwargs):
            checkpointer = MagicMock()
            checkpointer.config = config
            built_configs.append(config)
            return checkpointer

        mock_checkpointer_cls.side_effect = build_checkpointer
        manager = AutomodelCheckpointManager(
            dp_mesh=mock_dp_mesh,
            tp_mesh=mock_tp_mesh,
        )
        manager.init_checkpointer(config_updates={"is_async": True})

        with TemporaryDirectory() as tmp_dir:
            manager.save_checkpoint(
                model=mock_model,
                weights_path=os.path.join(tmp_dir, "weights"),
                checkpointing_cfg={
                    "enabled": True,
                    "save_consolidated": True,
                    "single_rank_consolidation": True,
                    "consolidation_timeout_minutes": 7,
                },
            )

        assert len(built_configs) == 2
        assert built_configs[-1].save_consolidated == SaveConsolidatedMode.EVERY
        assert built_configs[-1].single_rank_consolidation is True
        assert built_configs[-1].consolidation_timeout_minutes == 7

    @patch("torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.checkpoint.Checkpointer")
    def test_save_with_optimizer(
        self,
        mock_checkpointer_cls,
        mock_get_rank,
        mock_meshes,
        mock_model,
        mock_optimizer,
    ):
        """Test saving model and optimizer weights."""
        mock_get_rank.return_value = 0
        mock_dp_mesh, mock_tp_mesh = mock_meshes

        mock_checkpointer = MagicMock()
        mock_checkpointer_cls.return_value = mock_checkpointer

        manager = AutomodelCheckpointManager(
            dp_mesh=mock_dp_mesh,
            tp_mesh=mock_tp_mesh,
        )
        manager.init_checkpointer()

        with TemporaryDirectory() as tmp_dir:
            weights_path = os.path.join(tmp_dir, "model", "weights")
            optimizer_path = os.path.join(tmp_dir, "optimizer", "optim")

            # Save checkpoint with optimizer
            manager.save_checkpoint(
                model=mock_model,
                weights_path=weights_path,
                optimizer=mock_optimizer,
                optimizer_path=optimizer_path,
                checkpointing_cfg={
                    "enabled": True,
                    "model_save_format": "safetensors",
                    "is_peft": True,
                },
            )

            # Verify both model and optimizer saving were called
            mock_checkpointer.save_model.assert_called_once()
            mock_checkpointer.save_optimizer.assert_called_once()

    @patch("torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.checkpoint.Checkpointer")
    def test_save_with_tokenizer(
        self, mock_checkpointer_cls, mock_get_rank, mock_meshes, mock_model
    ):
        """Test saving with tokenizer."""
        mock_get_rank.return_value = 0
        mock_dp_mesh, mock_tp_mesh = mock_meshes

        mock_checkpointer = MagicMock()
        mock_checkpointer_cls.return_value = mock_checkpointer

        manager = AutomodelCheckpointManager(
            dp_mesh=mock_dp_mesh,
            tp_mesh=mock_tp_mesh,
        )
        manager.init_checkpointer()

        with TemporaryDirectory() as tmp_dir:
            weights_path = os.path.join(tmp_dir, "model", "weights")
            tokenizer_path = os.path.join(tmp_dir, "tokenizer")
            os.makedirs(tokenizer_path)

            # Create mock tokenizer
            mock_tokenizer = MagicMock()

            # Save checkpoint with tokenizer
            manager.save_checkpoint(
                model=mock_model,
                weights_path=weights_path,
                tokenizer=mock_tokenizer,
                tokenizer_path=tokenizer_path,
                checkpointing_cfg={"enabled": True},
            )

            # Verify tokenizer.save_pretrained was called
            mock_tokenizer.save_pretrained.assert_called_once_with(tokenizer_path)

    @patch("torch.distributed.barrier")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("nemo_rl.models.automodel.checkpoint.Checkpointer")
    def test_save_with_tokenizer_skipped_on_non_zero_rank(
        self,
        mock_checkpointer_cls,
        mock_get_rank,
        mock_is_initialized,
        mock_barrier,
        mock_meshes,
        mock_model,
    ):
        """Only rank 0 may write the tokenizer.

        The tokenizer is replicated across ranks and save_pretrained() writes
        rank-independent filenames, so 32 ranks writing the same path race on the
        same inode and can hang the job on a hard-mounted NFS share. The model
        save stays collective and must still run on every rank.
        """
        mock_is_initialized.return_value = True
        mock_get_rank.return_value = 5
        mock_dp_mesh, mock_tp_mesh = mock_meshes

        mock_checkpointer = MagicMock()
        mock_checkpointer_cls.return_value = mock_checkpointer

        manager = AutomodelCheckpointManager(
            dp_mesh=mock_dp_mesh,
            tp_mesh=mock_tp_mesh,
        )
        manager.init_checkpointer()

        with TemporaryDirectory() as tmp_dir:
            weights_path = os.path.join(tmp_dir, "model", "weights")
            tokenizer_path = os.path.join(tmp_dir, "tokenizer")
            os.makedirs(tokenizer_path)

            mock_tokenizer = MagicMock()

            manager.save_checkpoint(
                model=mock_model,
                weights_path=weights_path,
                tokenizer=mock_tokenizer,
                tokenizer_path=tokenizer_path,
                checkpointing_cfg={"enabled": True},
            )

            # Collective model save still happens on this rank.
            mock_checkpointer.save_model.assert_called_once()
            # Replicated tokenizer write is skipped off rank 0...
            mock_tokenizer.save_pretrained.assert_not_called()
            # ...but the rank still joins the barrier so rank 0 does not hang.
            mock_barrier.assert_called_once()


@pytest.mark.automodel
class TestSaveLoadIntegration:
    """Integration tests that actually save and load checkpoints."""

    def test_save_and_load_model_only_safetensors(
        self, init_distributed, mock_experiment
    ):
        """Test saving and loading model weights only with safetensors format."""
        test_model, _, _ = mock_experiment
        original_state_dict = {k: v.clone() for k, v in test_model.state_dict().items()}

        with TemporaryDirectory() as tmp_dir:
            weights_path = os.path.join(tmp_dir, "test_model")

            # Create device mesh and checkpoint manager
            mesh = torch.distributed.device_mesh.init_device_mesh(
                "cpu", (1,), mesh_dim_names=("dp",)
            )
            checkpoint_manager = AutomodelCheckpointManager(
                dp_mesh=mesh,
                tp_mesh=mesh,
            )
            checkpoint_manager.init_checkpointer(
                config_updates={"model_save_format": "safetensors"}
            )

            # Save checkpoint
            checkpoint_manager.save_checkpoint(
                model=test_model,
                weights_path=weights_path,
                checkpointing_cfg={
                    "enabled": True,
                    "model_save_format": "safetensors",
                },
            )

            # Verify files are created
            assert os.path.exists(weights_path)
            files = os.listdir(os.path.join(weights_path, "model"))
            assert any(f.endswith(".safetensors") for f in files)

            # Create a new model with different weights
            new_model = TestModel()
            # Initialize with different values
            for param in new_model.parameters():
                param.data.fill_(999.0)

            # Load the checkpoint
            checkpoint_manager.load_checkpoint(
                model=new_model, weights_path=weights_path
            )

            # Verify the weights match the original
            check_dict_equality(new_model.state_dict(), original_state_dict)

    def test_save_and_load_model_only_torch_save(
        self, init_distributed, mock_experiment
    ):
        """Test saving and loading model weights only with torch_save format."""
        test_model, _, _ = mock_experiment
        original_state_dict = {k: v.clone() for k, v in test_model.state_dict().items()}

        with TemporaryDirectory() as tmp_dir:
            weights_path = os.path.join(tmp_dir, "test_model")

            # Create device mesh and checkpoint manager
            mesh = torch.distributed.device_mesh.init_device_mesh(
                "cpu", (1,), mesh_dim_names=("dp",)
            )
            checkpoint_manager = AutomodelCheckpointManager(
                dp_mesh=mesh,
                tp_mesh=mesh,
            )
            checkpoint_manager.init_checkpointer(
                config_updates={"model_save_format": "torch_save"}
            )

            # Save checkpoint
            checkpoint_manager.save_checkpoint(
                model=test_model,
                weights_path=weights_path,
                checkpointing_cfg={
                    "enabled": True,
                    "model_save_format": "torch_save",
                },
            )

            # Verify files are created
            assert os.path.exists(weights_path)
            files = os.listdir(os.path.join(weights_path, "model"))
            assert any(f.endswith(".distcp") for f in files)

            # Create a new model with different weights
            new_model = TestModel()
            # Initialize with different values
            for param in new_model.parameters():
                param.data.fill_(999.0)

            # Load the checkpoint
            checkpoint_manager.load_checkpoint(
                model=new_model, weights_path=weights_path
            )

            # Verify the weights match the original
            check_dict_equality(new_model.state_dict(), original_state_dict)

    def test_save_and_load_model_and_optimizer(self, init_distributed, mock_experiment):
        """Test saving and loading both model and optimizer."""
        test_model, optimizer, scheduler = mock_experiment

        # Take some optimization steps to change optimizer state
        for _ in range(5):
            loss = torch.nn.functional.mse_loss(
                test_model(torch.randn(2, 4)), torch.randn(2, 1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        original_model_state = {
            k: v.clone() for k, v in test_model.state_dict().items()
        }
        original_optimizer_state = optimizer.state_dict()
        original_scheduler_state = scheduler.state_dict()

        with TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model_and_optimizer", "model_path")
            optimizer_path = os.path.join(tmp_dir, "model_and_optimizer", "optimizer")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)

            # Create device mesh and checkpoint manager
            mesh = torch.distributed.device_mesh.init_device_mesh(
                "cpu", (1,), mesh_dim_names=("dp",)
            )
            checkpoint_manager = AutomodelCheckpointManager(
                dp_mesh=mesh,
                tp_mesh=mesh,
            )
            checkpoint_manager.init_checkpointer(
                config_updates={"model_save_format": "safetensors"}
            )

            # Save checkpoint
            checkpoint_manager.save_checkpoint(
                model=test_model,
                weights_path=model_path,
                optimizer=optimizer,
                scheduler=scheduler,
                optimizer_path=optimizer_path,
                checkpointing_cfg={"enabled": True},
            )

            # Verify files are created
            assert os.path.exists(model_path)
            assert os.path.exists(optimizer_path)

            # Create new model, optimizer, and scheduler with different state
            new_model = TestModel()
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
            new_scheduler = torch.optim.lr_scheduler.StepLR(
                new_optimizer, step_size=4, gamma=0.2
            )

            # Initialize with different values
            for param in new_model.parameters():
                param.data.fill_(999.0)

            # Load the checkpoint
            checkpoint_manager.load_checkpoint(
                model=new_model,
                weights_path=model_path,
                optimizer=new_optimizer,
                scheduler=new_scheduler,
                optimizer_path=optimizer_path,
            )

            # Verify all states match the original
            check_dict_equality(new_model.state_dict(), original_model_state)
            check_dict_equality(new_optimizer.state_dict(), original_optimizer_state)
            assert new_scheduler.state_dict() == original_scheduler_state

    def test_save_and_load_model_with_lora(
        self, init_distributed, mock_experiment, mock_lora_config
    ):
        """Test saving and loading both model and optimizer with LORA."""
        test_model, _, _ = mock_experiment
        lora_config = mock_lora_config

        test_model.apply_lora(lora_config)
        lora_state_dict = test_model.state_dict()

        # Assert LoRA weights exist for layers.0 (Linear 4->4)
        assert "layers.0.lora_A.weight" in lora_state_dict, (
            "layers.0.lora_A.weight not found"
        )
        assert "layers.0.lora_B.weight" in lora_state_dict, (
            "layers.0.lora_B.weight not found"
        )

        # Assert LoRA weights exist for layers.3 (Linear 4->1)
        assert "layers.3.lora_A.weight" in lora_state_dict, (
            "layers.3.lora_A.weight not found"
        )
        assert "layers.3.lora_B.weight" in lora_state_dict, (
            "layers.3.lora_B.weight not found"
        )

        assert lora_state_dict["layers.0.lora_A.weight"].shape == (2, 4), (
            f"Expected layers.0.lora_A.weight shape (2, 4), got {lora_state_dict['layers.0.lora_A.weight'].shape}"
        )
        assert lora_state_dict["layers.0.lora_B.weight"].shape == (4, 2), (
            f"Expected layers.0.lora_B.weight shape (4, 2), got {lora_state_dict['layers.0.lora_B.weight'].shape}"
        )

        # For layers.3: Linear(4, 1) with dim=2
        # lora_A: (dim, in_features) = (2, 4)
        # lora_B: (out_features, dim) = (1, 2)
        assert lora_state_dict["layers.3.lora_A.weight"].shape == (2, 4), (
            f"Expected layers.3.lora_A.weight shape (2, 4), got {lora_state_dict['layers.3.lora_A.weight'].shape}"
        )
        assert lora_state_dict["layers.3.lora_B.weight"].shape == (1, 2), (
            f"Expected layers.3.lora_B.weight shape (1, 2), got {lora_state_dict['layers.3.lora_B.weight'].shape}"
        )

        with TemporaryDirectory() as tmp_dir:
            weights_path = os.path.join(tmp_dir, "test_model")

            # Create device mesh and checkpoint manager with PEFT enabled
            mesh = torch.distributed.device_mesh.init_device_mesh(
                "cpu", (1,), mesh_dim_names=("dp",)
            )
            checkpoint_manager = AutomodelCheckpointManager(
                dp_mesh=mesh,
                tp_mesh=mesh,
            )
            checkpoint_manager.init_checkpointer(
                config_updates={"model_save_format": "safetensors", "is_peft": True}
            )

            checkpoint_manager.save_checkpoint(
                model=test_model,
                weights_path=weights_path,
                checkpointing_cfg={
                    "enabled": True,
                    "model_save_format": "safetensors",
                    "is_peft": True,
                },
                lora_enabled=True,
                peft_config=lora_config,
            )

            # Verify files are created
            assert os.path.exists(weights_path)
            files = os.listdir(os.path.join(weights_path, "model"))
            assert any(f.endswith(".safetensors") for f in files)

            # Create a new model with different weights
            new_model = TestModel()
            new_model.apply_lora(lora_config)
            # Initialize with different values
            for param in new_model.parameters():
                param.data.fill_(999.0)

            # Load the checkpoint
            checkpoint_manager.load_checkpoint(
                model=new_model, weights_path=weights_path
            )
            # peft only save lora weights, so we need to filter out the non-lora weights
            lora_params_original = {
                k: v for k, v in lora_state_dict.items() if "lora" in k
            }
            lora_params_loaded = {
                k: v for k, v in new_model.state_dict().items() if "lora" in k
            }
            # Verify the weights match the original
            check_dict_equality(lora_params_loaded, lora_params_original)


@pytest.mark.automodel
def test_qwen_vl_vision_key_mapping_workaround_still_needed():
    """Tripwire for the temporary Qwen-VL vision-tower key-mapping workaround.

    ``_patch_qwen_vl_vision_key_mapping`` (checkpoint.py) re-injects the
    ``^visual.`` -> ``model.visual.`` rename that transformers v5.5.0 (#44627) dropped, so
    the vision tower loads instead of being left randomly initialized. transformers #45358
    (v5.6) fixed the upstream mapping; the workaround is needed only while the automodel pin
    uses the broken transformers 5.5.x.

    This calls the *unwrapped* real ``get_combined_key_mapping`` and asserts it still lacks the
    ``model.visual`` target (bug present). It runs under ``@pytest.mark.automodel`` (the
    automodel-extra venv, transformers 5.5.x, where the patch matters). It FAILS — telling us
    to remove ``_patch_qwen_vl_vision_key_mapping`` (and this test) — once that transformers
    pin includes #45358 (>=5.6) and the real mapping targets ``model.visual``.
    """
    import nemo_automodel.components.checkpoint.checkpointing as ckpt

    import nemo_rl.models.automodel.checkpoint  # noqa: F401  (import applies the patch)

    fn = ckpt.get_combined_key_mapping
    # Unwrap to the real automodel mapping fn (Change 3 exposes it as _nrl_orig); if the patch
    # was never applied, fn is already the original.
    orig = getattr(fn, "_nrl_orig", fn)

    for model_type in ("qwen2_5_vl", "qwen2_vl"):
        mapping = orig(model_type) or {}
        assert not any(
            str(target).startswith("model.visual") for target in mapping.values()
        ), (
            f"Automodel/transformers now maps visual->model.visual for {model_type} "
            "(likely transformers #45358 / >=5.6); the _patch_qwen_vl_vision_key_mapping "
            "workaround in nemo_rl/models/automodel/checkpoint.py is obsolete - remove it "
            "and this test."
        )
